"""
PGExplainer: Parameterized Explainer for Graph Neural Networks
===============================================================

Reference: "Parameterized Explainer for Graph Neural Network"
           (Luo et al., NeurIPS 2020)

Key differences from GNNExplainer:
- Learns a global explanation model (parameterized)
- Can explain new instances without retraining
- More efficient for multiple explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


class PGExplainer(nn.Module):
    """
    PGExplainer: Parameterized explainer for GNNs.
    
    Learns a neural network that predicts edge masks for explanations,
    enabling efficient explanation of new instances.
    """
    
    def __init__(self, model: nn.Module, in_channels: int, hidden_channels: int = 64,
                 num_hops: int = 2, num_epochs: int = 20, lr: float = 0.005,
                 temp: Tuple[float, float] = (5.0, 2.0)):
        """
        Args:
            model: The GNN model to explain
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension for explanation network
            num_hops: Number of hops in GNN
            num_epochs: Number of training epochs
            lr: Learning rate
            temp: Temperature range for concrete distribution (start, end)
        """
        super().__init__()
        self.model = model
        self.num_hops = num_hops
        self.num_epochs = num_epochs
        self.lr = lr
        self.temp_start, self.temp_end = temp
        
        # Explanation network
        # Takes node embeddings and edge info, predicts edge importance
        self.explainer_net = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def train_explainer(self, edge_indices: List[torch.Tensor],
                        xs: List[torch.Tensor], labels: List[torch.Tensor],
                        **kwargs):
        """
        Train the explainer on multiple graphs.
        
        Args:
            edge_indices: List of edge index tensors
            xs: List of node feature tensors
            labels: List of labels
            **kwargs: Additional arguments for model
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            # Anneal temperature
            temp = self.temp_start - (self.temp_start - self.temp_end) * epoch / self.num_epochs
            
            for edge_index, x, label in zip(edge_indices, xs, labels):
                optimizer.zero_grad()
                
                # Get node embeddings from model
                with torch.no_grad():
                    embeddings = self._get_embeddings(x, edge_index, **kwargs)
                
                # Predict edge masks
                edge_mask = self._predict_edge_mask(embeddings, edge_index, temp)
                
                # Compute loss: prediction consistency with sparse mask
                loss = self._compute_loss(x, edge_index, edge_mask, label, **kwargs)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(edge_indices):.4f}")
    
    def explain(self, edge_index: torch.Tensor, x: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Explain a graph using the trained explainer.
        
        Args:
            edge_index: Edge indices [2, E]
            x: Node features [N, F]
            **kwargs: Additional arguments
        
        Returns:
            Explanation dictionary
        """
        with torch.no_grad():
            embeddings = self._get_embeddings(x, edge_index, **kwargs)
            edge_mask = self._predict_edge_mask(embeddings, edge_index, self.temp_end)
        
        # Threshold to get hard mask
        threshold = edge_mask.mean()
        hard_mask = (edge_mask > threshold).float()
        
        return {
            'edge_mask': edge_mask,
            'hard_edge_mask': hard_mask,
        }
    
    def explain_node(self, node_idx: int, edge_index: torch.Tensor,
                     x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Explain prediction for a single node.
        
        Args:
            node_idx: Index of node to explain
            edge_index: Edge indices [2, E]
            x: Node features [N, F]
            **kwargs: Additional arguments
        
        Returns:
            Explanation dictionary
        """
        # Extract computation graph
        edge_mask_sub, subset, mapping = self._get_computation_graph(
            node_idx, edge_index, self.num_hops
        )
        
        x_sub = x[subset]
        edge_index_sub = edge_index[:, edge_mask_sub]
        
        # Remap indices
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset)}
        edge_index_sub = torch.stack([
            torch.tensor([node_map[idx.item()] for idx in edge_index_sub[0]]),
            torch.tensor([node_map[idx.item()] for idx in edge_index_sub[1]])
        ], dim=0).to(edge_index.device)
        
        # Explain subgraph
        explanation = self.explain(edge_index_sub, x_sub, **kwargs)
        explanation['subset'] = subset
        explanation['mapping'] = mapping
        
        return explanation
    
    def _predict_edge_mask(self, embeddings: torch.Tensor,
                           edge_index: torch.Tensor,
                           temperature: float) -> torch.Tensor:
        """
        Predict edge mask using the explanation network.
        
        Args:
            embeddings: Node embeddings [N, F]
            edge_index: Edge indices [2, E]
            temperature: Temperature for concrete distribution
        
        Returns:
            Edge mask [E]
        """
        row, col = edge_index
        
        # Edge features: concatenated node embeddings
        edge_feats = torch.cat([embeddings[row], embeddings[col]], dim=-1)
        
        # Predict logits
        logits = self.explainer_net(edge_feats).squeeze(-1)
        
        # Concrete distribution (reparameterization trick)
        if self.training:
            # Sample from concrete distribution during training
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
            z = torch.sigmoid((logits + gumbel_noise) / temperature)
        else:
            # Use probability during inference
            z = torch.sigmoid(logits)
        
        return z
    
    def _compute_loss(self, x: torch.Tensor, edge_index: torch.Tensor,
                      edge_mask: torch.Tensor, label: torch.Tensor,
                      **kwargs) -> torch.Tensor:
        """
        Compute loss for training explainer.
        
        Loss consists of:
        1. Prediction fidelity (explain masked graph should give same prediction)
        2. Size regularization (prefer small explanations)
        3. Entropy regularization (encourage discrete masks)
        """
        # Original prediction
        with torch.no_grad():
            out_orig = self.model(x=x, edge_index=edge_index, **kwargs)
            pred_orig = out_orig.argmax(dim=-1)
        
        # Masked prediction
        out_masked = self._forward_with_mask(x, edge_index, edge_mask, **kwargs)
        
        # Fidelity loss
        if out_masked.dim() == 1:
            fidelity_loss = F.cross_entropy(out_masked.unsqueeze(0), pred_orig)
        else:
            fidelity_loss = F.cross_entropy(out_masked, pred_orig)
        
        # Size regularization
        size_loss = edge_mask.mean()
        
        # Entropy regularization (encourage discreteness)
        entropy = -edge_mask * torch.log(edge_mask + 1e-8) - \
                  (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)
        entropy_loss = entropy.mean()
        
        # Combined loss
        loss = fidelity_loss + 0.005 * size_loss + 0.001 * entropy_loss
        
        return loss
    
    def _get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor,
                        **kwargs) -> torch.Tensor:
        """
        Get node embeddings from the model.
        
        This assumes the model has a method to extract intermediate features.
        For models without this, we use the node features directly.
        """
        # Try to get embeddings from model
        if hasattr(self.model, 'get_node_embeddings'):
            return self.model.get_node_embeddings(x, edge_index, **kwargs)
        
        # Fallback: use input features
        return x
    
    def _forward_with_mask(self, x: torch.Tensor, edge_index: torch.Tensor,
                           edge_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with masked edges."""
        # Apply mask to edge attributes if they exist
        if 'edge_attr' in kwargs and kwargs['edge_attr'] is not None:
            kwargs['edge_attr'] = kwargs['edge_attr'] * edge_mask.unsqueeze(-1)
        
        return self.model(x=x, edge_index=edge_index, **kwargs)
    
    def _get_computation_graph(self, node_idx: int, edge_index: torch.Tensor,
                               num_hops: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Extract k-hop computation graph."""
        node_mask = torch.zeros(edge_index.max().item() + 1, dtype=torch.bool)
        node_mask[node_idx] = True
        
        for _ in range(num_hops):
            edge_mask = node_mask[edge_index[0]] | node_mask[edge_index[1]]
            node_mask[edge_index[0][edge_mask]] = True
            node_mask[edge_index[1][edge_mask]] = True
        
        subset = torch.nonzero(node_mask, as_tuple=False).squeeze()
        mapping = (subset == node_idx).nonzero(as_tuple=True)[0].item()
        
        return edge_mask, subset, mapping
    
    def get_feature_importance(self, edge_index: torch.Tensor, x: torch.Tensor,
                               feature_names: Optional[List[str]] = None,
                               **kwargs) -> Dict[str, torch.Tensor]:
        """
        Get feature importance scores.
        
        Args:
            edge_index: Edge indices
            x: Node features
            feature_names: Names of features (optional)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with feature importance scores
        """
        num_features = x.shape[1]
        importances = torch.zeros(num_features)
        
        # Baseline prediction
        with torch.no_grad():
            baseline_out = self.model(x=x, edge_index=edge_index, **kwargs)
            baseline_pred = baseline_out.argmax(dim=-1)
        
        # Perturb each feature
        for i in range(num_features):
            x_perturbed = x.clone()
            x_perturbed[:, i] = 0  # Zero out feature
            
            with torch.no_grad():
                perturbed_out = self.model(x=x_perturbed, edge_index=edge_index, **kwargs)
            
            # Importance = change in prediction confidence
            importance = torch.abs(perturbed_out - baseline_out).mean()
            importances[i] = importance
        
        # Normalize
        importances = importances / importances.sum()
        
        result = {'feature_importance': importances}
        
        if feature_names is not None:
            result['feature_names'] = feature_names
        
        return result
    
    def visualize_explanation(self, explanation: Dict[str, torch.Tensor],
                              pos: Optional[torch.Tensor] = None,
                              edge_index: Optional[torch.Tensor] = None,
                              save_path: Optional[str] = None):
        """
        Visualize explanation.
        
        Args:
            explanation: Explanation dictionary
            pos: Node positions
            edge_index: Edge indices
            save_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib required for visualization")
            return
        
        edge_mask = explanation['edge_mask'].cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Edge importance
        axes[0].hist(edge_mask, bins=50, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Edge Importance')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Edge Importance Distribution')
        axes[0].axvline(edge_mask.mean(), color='red', linestyle='--', label='Mean')
        axes[0].legend()
        
        # 3D structure
        if pos is not None and edge_index is not None:
            ax = fig.add_subplot(122, projection='3d')
            
            pos_np = pos.cpu().numpy()
            edge_index_np = edge_index.cpu().numpy()
            
            # Plot all edges
            for i in range(edge_index_np.shape[1]):
                src, dst = edge_index_np[0, i], edge_index_np[1, i]
                importance = edge_mask[i] if i < len(edge_mask) else 0
                
                color = plt.cm.Reds(importance)
                ax.plot([pos_np[src, 0], pos_np[dst, 0]],
                       [pos_np[src, 1], pos_np[dst, 1]],
                       [pos_np[src, 2], pos_np[dst, 2]],
                       color=color, alpha=0.6, linewidth=importance * 3)
            
            # Plot nodes
            ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2],
                      c='blue', s=50, alpha=0.8)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Explanation Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
    
    def save(self, path: str):
        """Save trained explainer."""
        torch.save({
            'state_dict': self.state_dict(),
            'explainer_net': self.explainer_net,
            'num_hops': self.num_hops,
        }, path)
    
    def load(self, path: str):
        """Load trained explainer."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.num_hops = checkpoint['num_hops']
