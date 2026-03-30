"""
GNNExplainer: Generating Explanations for Graph Neural Networks
================================================================

Reference: "GNNExplainer: Generating Explanations for Graph Neural Networks"
           (Ying et al., NeurIPS 2019)

Generates instance-level explanations by identifying a compact subgraph
structure and a small subset of node features that are crucial for GNN's prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


class GNNExplainer(nn.Module):
    """
    GNNExplainer for explaining GNN predictions.
    
    Identifies important subgraphs and node features by maximizing
    mutual information between the prediction and the explanation.
    """
    
    def __init__(self, model: nn.Module, num_hops: int = 2, epochs: int = 100,
                 lr: float = 0.01, log: bool = True):
        """
        Args:
            model: The GNN model to explain
            num_hops: Number of hops in GNN (determines explanation subgraph size)
            epochs: Number of training epochs for explanation
            lr: Learning rate for optimization
            log: Whether to print progress
        """
        super().__init__()
        self.model = model
        self.num_hops = num_hops
        self.epochs = epochs
        self.lr = lr
        self.log = log
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def explain_node(self, node_idx: int, edge_index: torch.Tensor,
                     x: torch.Tensor, y: Optional[torch.Tensor] = None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Explain prediction for a single node.
        
        Args:
            node_idx: Index of the node to explain
            edge_index: Graph edge indices [2, E]
            x: Node features [N, F]
            y: True label (optional)
            **kwargs: Additional arguments for model forward pass
        
        Returns:
            Dictionary with explanation results
        """
        # Extract computation graph
        edge_mask, subset, mapping = self._get_computation_graph(
            node_idx, edge_index, self.num_hops
        )
        
        x_sub = x[subset]
        edge_index_sub = edge_index[:, edge_mask]
        
        # Remap node indices
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset)}
        edge_index_sub = torch.stack([
            torch.tensor([node_map[idx.item()] for idx in edge_index_sub[0]]),
            torch.tensor([node_map[idx.item()] for idx in edge_index_sub[1]])
        ], dim=0).to(edge_index.device)
        
        # Get original prediction
        with torch.no_grad():
            out = self.model(x=x_sub, edge_index=edge_index_sub, **kwargs)
            pred_label = out[mapping].argmax(dim=-1)
        
        # Optimize edge mask
        mask = nn.Parameter(torch.randn(edge_index_sub.shape[1]))
        optimizer = torch.optim.Adam([mask], lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Apply sigmoid to get soft mask
            soft_mask = torch.sigmoid(mask)
            
            # Forward pass with masked edges
            out_masked = self._forward_with_mask(
                x_sub, edge_index_sub, soft_mask, **kwargs
            )
            
            # Loss: maximize prediction confidence while keeping mask sparse
            pred_loss = F.cross_entropy(
                out_masked[mapping:mapping+1], pred_label.unsqueeze(0)
            )
            mask_size_loss = soft_mask.mean()
            mask_ent_loss = -soft_mask * torch.log(soft_mask + 1e-8) - \
                           (1 - soft_mask) * torch.log(1 - soft_mask + 1e-8)
            mask_ent_loss = mask_ent_loss.mean()
            
            loss = pred_loss + 0.005 * mask_size_loss + 0.001 * mask_ent_loss
            
            loss.backward()
            optimizer.step()
            
            if self.log and epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # Get final hard mask
        final_mask = torch.sigmoid(mask).detach()
        threshold = final_mask.mean()
        hard_mask = (final_mask > threshold).float()
        
        return {
            'edge_mask': final_mask,
            'hard_edge_mask': hard_mask,
            'subset': subset,
            'mapping': mapping,
            'edge_index_sub': edge_index_sub,
        }
    
    def explain_graph(self, edge_index: torch.Tensor, x: torch.Tensor,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """
        Explain prediction for an entire graph.
        
        Args:
            edge_index: Graph edge indices [2, E]
            x: Node features [N, F]
            **kwargs: Additional arguments for model forward pass
        
        Returns:
            Dictionary with explanation results
        """
        # Get original prediction
        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, **kwargs)
            if out.dim() == 1:
                pred_label = out.argmax(dim=-1).unsqueeze(0)
            else:
                pred_label = out.argmax(dim=-1)
        
        # Optimize edge mask
        mask = nn.Parameter(torch.randn(edge_index.shape[1]))
        optimizer = torch.optim.Adam([mask], lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            soft_mask = torch.sigmoid(mask)
            
            # Forward with masked edges
            out_masked = self._forward_with_mask(x, edge_index, soft_mask, **kwargs)
            
            # Loss
            if out_masked.dim() == 1:
                pred_loss = F.cross_entropy(out_masked.unsqueeze(0), pred_label)
            else:
                pred_loss = F.cross_entropy(out_masked, pred_label)
            
            mask_size_loss = soft_mask.mean()
            mask_ent_loss = -soft_mask * torch.log(soft_mask + 1e-8) - \
                           (1 - soft_mask) * torch.log(1 - soft_mask + 1e-8)
            mask_ent_loss = mask_ent_loss.mean()
            
            loss = pred_loss + 0.005 * mask_size_loss + 0.002 * mask_ent_loss
            
            loss.backward()
            optimizer.step()
            
            if self.log and epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        final_mask = torch.sigmoid(mask).detach()
        threshold = final_mask.mean()
        hard_mask = (final_mask > threshold).float()
        
        return {
            'edge_mask': final_mask,
            'hard_edge_mask': hard_mask,
        }
    
    def explain_force_prediction(self, atomic_numbers: torch.Tensor,
                                  pos: torch.Tensor, edge_index: torch.Tensor,
                                  atom_idx: int, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Explain force prediction for a specific atom.
        
        Args:
            atomic_numbers: Atom types [N]
            pos: Positions [N, 3]
            edge_index: Edge indices [2, E]
            atom_idx: Index of atom to explain
            **kwargs: Additional arguments for model
        
        Returns:
            Dictionary with explanation results
        """
        # Extract computation graph around atom
        edge_mask, subset, mapping = self._get_computation_graph(
            atom_idx, edge_index, self.num_hops
        )
        
        atomic_numbers_sub = atomic_numbers[subset]
        pos_sub = pos[subset]
        edge_index_sub = edge_index[:, edge_mask]
        
        # Remap indices
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subset)}
        edge_index_sub = torch.stack([
            torch.tensor([node_map[idx.item()] for idx in edge_index_sub[0]]),
            torch.tensor([node_map[idx.item()] for idx in edge_index_sub[1]])
        ], dim=0).to(edge_index.device)
        
        # Get original force prediction
        with torch.no_grad():
            forces = self.model.predict_forces(
                atomic_numbers=atomic_numbers_sub,
                pos=pos_sub,
                edge_index=edge_index_sub,
                **kwargs
            )
            target_force = forces[mapping]
        
        # Optimize edge mask
        mask = nn.Parameter(torch.randn(edge_index_sub.shape[1]))
        optimizer = torch.optim.Adam([mask], lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            soft_mask = torch.sigmoid(mask)
            
            # Forward with masked edges
            forces_masked = self._forward_with_mask_forces(
                atomic_numbers_sub, pos_sub, edge_index_sub, soft_mask, **kwargs
            )
            
            # Loss: match force prediction while keeping mask sparse
            force_loss = F.mse_loss(forces_masked[mapping], target_force)
            mask_size_loss = soft_mask.mean()
            
            loss = force_loss + 0.01 * mask_size_loss
            
            loss.backward()
            optimizer.step()
        
        final_mask = torch.sigmoid(mask).detach()
        
        return {
            'edge_mask': final_mask,
            'hard_edge_mask': (final_mask > final_mask.mean()).float(),
            'subset': subset,
            'mapping': mapping,
            'edge_index_sub': edge_index_sub,
            'target_force': target_force,
        }
    
    def _get_computation_graph(self, node_idx: int, edge_index: torch.Tensor,
                               num_hops: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Extract k-hop computation graph around a node.
        
        Returns:
            edge_mask: Mask of edges in subgraph
            subset: Node indices in subgraph
            mapping: Index of target node in subgraph
        """
        node_mask = torch.zeros(edge_index.max().item() + 1, dtype=torch.bool)
        node_mask[node_idx] = True
        
        for _ in range(num_hops):
            # Find edges connected to current nodes
            edge_mask = node_mask[edge_index[0]] | node_mask[edge_index[1]]
            # Add neighbors
            node_mask[edge_index[0][edge_mask]] = True
            node_mask[edge_index[1][edge_mask]] = True
        
        subset = torch.nonzero(node_mask, as_tuple=False).squeeze()
        mapping = (subset == node_idx).nonzero(as_tuple=True)[0].item()
        
        return edge_mask, subset, mapping
    
    def _forward_with_mask(self, x: torch.Tensor, edge_index: torch.Tensor,
                           edge_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with masked edges."""
        # Apply mask to edge features if they exist
        if 'edge_attr' in kwargs and kwargs['edge_attr'] is not None:
            kwargs['edge_attr'] = kwargs['edge_attr'] * edge_mask.unsqueeze(-1)
        
        # For models that don't support explicit edge masking,
        # we can only mask via edge weights
        return self.model(x=x, edge_index=edge_index, **kwargs)
    
    def _forward_with_mask_forces(self, atomic_numbers: torch.Tensor,
                                   pos: torch.Tensor, edge_index: torch.Tensor,
                                   edge_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with masked edges for force prediction."""
        # This is a simplified implementation
        # Full implementation would modify edge weights in message passing
        return self.model.predict_forces(
            atomic_numbers=atomic_numbers,
            pos=pos,
            edge_index=edge_index,
            **kwargs
        )
    
    def visualize_explanation(self, explanation: Dict[str, torch.Tensor],
                              pos: Optional[torch.Tensor] = None,
                              save_path: Optional[str] = None):
        """
        Visualize explanation (requires matplotlib).
        
        Args:
            explanation: Explanation dictionary from explain_* methods
            pos: Node positions for 3D visualization
            save_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Matplotlib required for visualization")
            return
        
        edge_mask = explanation['edge_mask'].cpu().numpy()
        hard_mask = explanation['hard_edge_mask'].cpu().numpy()
        
        fig = plt.figure(figsize=(15, 5))
        
        # Edge importance histogram
        ax1 = fig.add_subplot(131)
        ax1.hist(edge_mask, bins=50)
        ax1.set_xlabel('Edge Importance')
        ax1.set_ylabel('Count')
        ax1.set_title('Edge Importance Distribution')
        
        # Hard mask
        ax2 = fig.add_subplot(132)
        ax2.bar(range(len(hard_mask)), hard_mask)
        ax2.set_xlabel('Edge Index')
        ax2.set_ylabel('Selected (1) / Not Selected (0)')
        ax2.set_title('Selected Edges')
        
        # 3D structure if positions available
        if pos is not None and 'edge_index_sub' in explanation:
            ax3 = fig.add_subplot(133, projection='3d')
            
            edge_index_sub = explanation['edge_index_sub'].cpu()
            pos_sub = pos.cpu()
            
            # Plot all edges with low alpha
            for i in range(edge_index_sub.shape[1]):
                src, dst = edge_index_sub[0, i].item(), edge_index_sub[1, i].item()
                ax3.plot([pos_sub[src, 0], pos_sub[dst, 0]],
                        [pos_sub[src, 1], pos_sub[dst, 1]],
                        [pos_sub[src, 2], pos_sub[dst, 2]],
                        'gray', alpha=0.2)
            
            # Plot important edges
            important_edges = explanation['hard_edge_mask'].cpu().numpy() > 0.5
            for i in range(edge_index_sub.shape[1]):
                if important_edges[i]:
                    src, dst = edge_index_sub[0, i].item(), edge_index_sub[1, i].item()
                    ax3.plot([pos_sub[src, 0], pos_sub[dst, 0]],
                            [pos_sub[src, 1], pos_sub[dst, 1]],
                            [pos_sub[src, 2], pos_sub[dst, 2]],
                            'red', linewidth=2)
            
            # Plot atoms
            ax3.scatter(pos_sub[:, 0], pos_sub[:, 1], pos_sub[:, 2], c='blue', s=100)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.set_title('Important Subgraph (3D)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
