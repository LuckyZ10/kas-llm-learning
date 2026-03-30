"""
Physics-Informed Graph Neural Networks

Implementation of GNNs with physics constraints for molecular dynamics
and particle systems. Includes equivariant message passing and 
conservation law preservation.

Reference:
    - Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021
    - Batzner et al., "E(3)-equivariant graph neural networks for 
      data-efficient and accurate interatomic potentials", Nature Communications 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np


class EGNNLayer(nn.Module):
    """
    E(n) Equivariant Graph Neural Network Layer.
    
    Preserves Euclidean equivariance - rotating/translating the input
    rotates/translates the output.
    
    Reference:
        Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 64,
        act_fn: str = 'swish'
    ):
        """
        Initialize EGNN layer.
        
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension for message passing
            act_fn: Activation function
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Edge model: computes messages
        edge_input_dim = node_dim * 2 + 1 + edge_dim  # 1 for distance
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            self._get_activation(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Coordinate model: computes coordinate updates
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self._get_activation(act_fn),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Node model: computes node updates
        node_input_dim = node_dim + hidden_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            self._get_activation(act_fn),
            nn.Linear(hidden_dim, node_dim),
        )
        
    def _get_activation(self, act_fn: str):
        """Get activation function."""
        if act_fn == 'swish':
            return nn.SiLU()
        elif act_fn == 'relu':
            return nn.ReLU()
        elif act_fn == 'tanh':
            return nn.Tanh()
        else:
            return nn.SiLU()
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            h: Node features [n_nodes, node_dim]
            x: Node coordinates [n_nodes, 3]
            edge_index: Edge indices [2, n_edges]
            edge_attr: Edge features [n_edges, edge_dim]
            
        Returns:
            Updated (h, x)
        """
        row, col = edge_index
        
        # Compute edge features
        radial = torch.sum((x[row] - x[col]) ** 2, dim=1, keepdim=True)
        
        # Edge input: [h_i, h_j, ||x_i - x_j||^2, edge_attr]
        edge_input = [h[row], h[col], radial]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_input = torch.cat(edge_input, dim=-1)
        
        # Message (edge features)
        m = self.edge_mlp(edge_input)  # [n_edges, hidden_dim]
        
        # Coordinate update
        coord_diff = x[row] - x[col]  # [n_edges, 3]
        trans = coord_diff * self.coord_mlp(m)  # [n_edges, 3]
        
        # Aggregate coordinate updates
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, trans)
        
        # Update coordinates (equivariant)
        x = x + agg
        
        # Aggregate messages
        agg_h = torch.zeros(h.shape[0], m.shape[1], device=h.device)
        agg_h.index_add_(0, row, m)
        
        # Node update
        node_input = torch.cat([h, agg_h], dim=-1)
        h = h + self.node_mlp(node_input)  # Residual connection
        
        return h, x


class PhysicsInformedGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network.
    
    Incorporates physical constraints into message passing:
    - Energy conservation
    - Momentum conservation
    - Force/energy consistency
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        edge_dim: int = 0,
        output_type: str = 'energy',
        cutoff: float = 5.0,
        max_neighbors: int = 32
    ):
        """
        Initialize Physics-Informed GNN.
        
        Args:
            node_dim: Input node feature dimension
            hidden_dim: Hidden dimension
            n_layers: Number of message passing layers
            edge_dim: Edge feature dimension
            output_type: 'energy', 'force', or 'both'
            cutoff: Neighborhood cutoff distance
            max_neighbors: Maximum number of neighbors
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_type = output_type
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        
        # Node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Message passing layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Readout layers
        if output_type in ['energy', 'both']:
            self.energy_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        if output_type in ['force', 'both']:
            # Force prediction from energy gradient
            self.force_direct = False
        
    def build_graph(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Build radius graph based on cutoff.
        
        Args:
            pos: Positions [n_nodes, 3]
            batch: Batch indices [n_nodes]
            
        Returns:
            Edge index [2, n_edges]
        """
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        
        edge_index = []
        
        # Compute edges per batch
        for batch_idx in torch.unique(batch):
            mask = batch == batch_idx
            batch_pos = pos[mask]
            batch_indices = torch.where(mask)[0]
            
            # Compute distances
            dist = torch.cdist(batch_pos, batch_pos)
            
            # Apply cutoff
            adj = (dist < self.cutoff) & (dist > 0)
            
            # Get edge indices
            src, dst = torch.where(adj)
            
            # Map back to global indices
            src = batch_indices[src]
            dst = batch_indices[dst]
            
            edge_index.append(torch.stack([src, dst]))
        
        return torch.cat(edge_index, dim=1)
    
    def forward(
        self,
        node_attr: torch.Tensor,
        pos: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            node_attr: Node features [n_nodes, node_dim]
            pos: Positions [n_nodes, 3]
            edge_index: Precomputed edge index [2, n_edges]
            batch: Batch indices [n_nodes]
            
        Returns:
            Dictionary with predictions
        """
        # Build graph if not provided
        if edge_index is None:
            edge_index = self.build_graph(pos, batch)
        
        # Initial embedding
        h = self.node_embedding(node_attr)
        x = pos.clone()
        
        # Message passing
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        
        outputs = {}
        
        # Energy prediction
        if self.output_type in ['energy', 'both']:
            node_energies = self.energy_mlp(h).squeeze(-1)
            
            if batch is not None:
                # Aggregate per molecule
                energy = torch.zeros(batch.max().item() + 1, device=h.device)
                energy.index_add_(0, batch, node_energies)
            else:
                energy = node_energies.sum()
            
            outputs['energy'] = energy
        
        # Force prediction (from energy gradient)
        if self.output_type in ['force', 'both']:
            if 'energy' not in outputs:
                raise ValueError("Force prediction requires energy")
            
            pos_grad = torch.autograd.grad(
                outputs=['energy'],
                inputs=pos,
                grad_outputs=torch.ones_like(outputs['energy']),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # F = -dE/dx
            outputs['forces'] = -pos_grad
        
        return outputs


class MomentumConservingGNN(nn.Module):
    """
    Graph Neural Network with explicit momentum conservation.
    
    Based on:
    "A physics-informed graph neural network conserving linear and angular 
    momentum for dynamical systems", Nature Communications 2025
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_body_order: int = 2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_body_order = n_body_order
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # [dx, dy, dz, |r|]
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing
        self.message_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(n_layers)
        ])
        
        # Force decoder (momentum-preserving)
        self.force_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        masses: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with momentum conservation.
        
        Args:
            pos: Positions [n_nodes, 3]
            vel: Velocities [n_nodes, 3]
            masses: Masses [n_nodes]
            edge_index: Edge indices [2, n_edges]
            
        Returns:
            Dictionary with forces and other predictions
        """
        row, col = edge_index
        
        # Edge features (invariant under translation/rotation)
        r_ij = pos[row] - pos[col]  # Relative positions
        d_ij = torch.norm(r_ij, dim=-1, keepdim=True)
        
        edge_attr = torch.cat([r_ij, d_ij], dim=-1)
        edge_emb = self.edge_encoder(edge_attr)
        
        # Node features (velocity magnitude - invariant)
        v_norm = torch.norm(vel, dim=-1, keepdim=True)
        node_feat = v_norm
        
        # Message passing
        for message_net in self.message_mlp:
            # Aggregate messages
            messages = torch.zeros(
                pos.shape[0], self.hidden_dim,
                device=pos.device
            )
            
            message_input = torch.cat([edge_emb, node_feat[row]], dim=-1)
            message = message_net(message_input)
            messages.index_add_(0, col, message)
            
            node_feat = messages
        
        # Decode forces in direction of r_ij
        force_magnitude = self.force_decoder(node_feat[row])
        
        # Force is along r_ij (Newton's 3rd law preserved)
        force_direction = r_ij / (d_ij + 1e-8)
        pairwise_forces = force_magnitude * force_direction
        
        # Aggregate forces (antisymmetric: f_ij = -f_ji)
        forces = torch.zeros_like(pos)
        forces.index_add_(0, row, pairwise_forces)
        forces.index_add_(0, col, -pairwise_forces)
        
        return {'forces': forces}


class HamiltonianGNN(nn.Module):
    """
    Hamiltonian Graph Neural Network.
    
    Learns a scalar Hamiltonian and derives dynamics from it,
    guaranteeing energy conservation.
    
    Reference:
        Sanchez-Gonzalez et al., "Hamiltonian Graph Networks with ODE Integrators"
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4
    ):
        super().__init__()
        
        # Node embedding
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Hamiltonian network
        self.hamiltonian_net = nn.ModuleList([
            EGNNLayer(hidden_dim, 0, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Scalar energy output
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def compute_hamiltonian(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hamiltonian H(q, p).
        
        Args:
            q: Generalized positions [n_nodes, 3]
            p: Generalized momenta [n_nodes, 3]
            node_attr: Node attributes
            edge_index: Edge indices
            
        Returns:
            Hamiltonian (scalar per system)
        """
        h = self.node_embedding(node_attr)
        
        # Use positions as coordinates
        x = q
        
        for layer in self.hamiltonian_net:
            h, x = layer(h, x, edge_index)
        
        # Node energies
        node_energies = self.energy_head(h).squeeze(-1)
        
        # Kinetic energy from momenta
        kinetic = 0.5 * (p ** 2).sum(dim=-1)
        
        # Total Hamiltonian
        H = node_energies.sum() + kinetic.sum()
        
        return H
    
    def get_derivatives(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Hamilton's equations:
        dq/dt = dH/dp
        dp/dt = -dH/dq
        """
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        
        H = self.compute_hamiltonian(q, p, node_attr, edge_index)
        
        # dq/dt = dH/dp
        dqdt = torch.autograd.grad(
            H, p, create_graph=True, retain_graph=True
        )[0]
        
        # dp/dt = -dH/dq
        dpdt = -torch.autograd.grad(
            H, q, create_graph=True, retain_graph=True
        )[0]
        
        return dqdt, dpdt


class EquivariantTransformer(nn.Module):
    """
    Equivariant Transformer for point clouds.
    
    Combines transformer attention with equivariance constraints.
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EquivariantTransformerLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Output heads
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.force_scale = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        node_attr: torch.Tensor,
        pos: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        h = self.node_embedding(node_attr)
        
        # Relative positions for attention
        if edge_index is not None:
            row, col = edge_index
            rel_pos = pos[row] - pos[col]
        else:
            # Fully connected
            rel_pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        
        # Transformer layers
        for layer in self.layers:
            h, pos_update = layer(h, rel_pos)
            pos = pos + pos_update
        
        # Energy prediction
        energy = self.energy_head(h).sum()
        
        # Force prediction (equivariant)
        force_magnitude = self.force_scale(h)
        
        return {
            'energy': energy,
            'positions': pos
        }


class EquivariantTransformerLayer(nn.Module):
    """Single equivariant transformer layer."""
    
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Coordinate update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        h: torch.Tensor,
        rel_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Multi-head attention
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        
        # Reshape for multi-head
        q = q.view(-1, self.n_heads, self.head_dim)
        k = k.view(-1, self.n_heads, self.head_dim)
        v = v.view(-1, self.n_heads, self.head_dim)
        
        # Attention scores (invariant)
        scores = torch.einsum('nhd,mhd->nmh', q, k) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.einsum('nmh,mhd->nhd', attn, v)
        out = out.view(-1, self.n_heads * self.head_dim)
        out = self.o_proj(out)
        
        # Update features
        h = self.norm(h + out)
        
        # Update coordinates (equivariant)
        coord_weights = self.coord_mlp(h)
        pos_update = coord_weights * rel_pos.mean(dim=1)
        
        return h, pos_update
