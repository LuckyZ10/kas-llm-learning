"""
Allegro: Learning Local Equivariant Representations
=====================================================

Reference: "Learning Local Equivariant Representations for 
            Large-Scale Atomistic Dynamics" (Musaelian et al., Nature Comm 2023)

Key features:
- Strictly spatially local (no message passing)
- Predicts energy as function of edge embeddings
- Highly parallelizable
- Excellent generalization to out-of-distribution data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class RadialBasis(nn.Module):
    """Chebyshev radial basis functions."""
    
    def __init__(self, num_basis: int = 8, cutoff: float = 5.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
        # Chebyshev nodes
        k = torch.arange(1, num_basis + 1)
        self.register_buffer('centers', torch.cos((2 * k - 1) * math.pi / (2 * num_basis)))
        self.register_buffer('widths', torch.ones(num_basis) * (cutoff / num_basis))
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: Pairwise distances [E, 1]
        Returns:
            Radial basis [E, num_basis]
        """
        # Cutoff function
        x = distances / self.cutoff
        cutoff = 0.5 * (torch.cos(x * math.pi) + 1.0)
        cutoff = cutoff * (distances < self.cutoff).float()
        
        # Chebyshev basis (simplified to Gaussian for stability)
        diff = distances - self.centers * self.cutoff
        basis = torch.exp(-0.5 * (diff / self.widths) ** 2)
        
        return basis * cutoff


class SphericalHarmonics(nn.Module):
    """Spherical harmonics for edge vectors."""
    
    def __init__(self, lmax: int = 2):
        super().__init__()
        self.lmax = lmax
        self.num_sh = (lmax + 1) ** 2
    
    def forward(self, edge_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_vec: Edge vectors [E, 3]
        Returns:
            Spherical harmonics [E, num_sh]
        """
        r = torch.norm(edge_vec, dim=-1, keepdim=True).clamp(min=1e-8)
        x, y, z = edge_vec[:, 0] / r.squeeze(), edge_vec[:, 1] / r.squeeze(), edge_vec[:, 2] / r.squeeze()
        
        sh = [torch.ones_like(x) * 0.28209479177387814]  # l=0
        
        if self.lmax >= 1:
            sh.extend([
                -0.4886025119029199 * y,
                0.4886025119029199 * z,
                -0.4886025119029199 * x,
            ])
        
        if self.lmax >= 2:
            x2, y2, z2 = x**2, y**2, z**2
            sh.extend([
                1.0925484305920792 * x * y,
                -1.0925484305920792 * y * z,
                0.94617469575755997 * z2 - 0.31539156525251999 * (x2 + y2),
                -1.0925484305920792 * x * z,
                0.54627421529603959 * (x2 - y2),
            ])
        
        return torch.stack(sh, dim=-1)


class AllegroLayer(nn.Module):
    """
    Single Allegro layer for edge feature updates.
    
    Unlike message passing GNNs, Allegro operates on edge features directly.
    """
    
    def __init__(self, edge_dim: int, node_dim: int, num_basis: int,
                 hidden_dim: int = 64, lmax: int = 2):
        super().__init__()
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.lmax = lmax
        
        # Edge feature update network
        # Takes: edge features, node features of endpoints, radial basis
        mlp_input = edge_dim + 2 * node_dim + num_basis
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_input, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        
        self.layer_norm = nn.LayerNorm(edge_dim)
    
    def forward(self, edge_feats: torch.Tensor, node_feats: torch.Tensor,
                edge_index: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_feats: Edge features [E, edge_dim]
            node_feats: Node features [N, node_dim]
            edge_index: Edge indices [2, E]
            edge_rbf: Radial basis [E, num_basis]
        
        Returns:
            Updated edge features [E, edge_dim]
        """
        row, col = edge_index
        
        # Gather node features
        node_i = node_feats[row]  # Source
        node_j = node_feats[col]  # Target
        
        # Concatenate all inputs
        mlp_input = torch.cat([edge_feats, node_i, node_j, edge_rbf], dim=-1)
        
        # Update
        updated = self.edge_mlp(mlp_input)
        
        return self.layer_norm(edge_feats + updated)


class TensorProductLayer(nn.Module):
    """
    Tensor product layer for constructing equivariant edge features.
    
    Key innovation: pair-wise Clebsch-Gordan transforms for high-order features.
    """
    
    def __init__(self, edge_dim: int, num_sh: int, hidden_dim: int = 64):
        super().__init__()
        self.edge_dim = edge_dim
        self.num_sh = num_sh
        
        # Linear projection for tensor product
        self.tp_linear = nn.Linear(edge_dim * num_sh, edge_dim)
        
        # Weight network
        self.weight_net = nn.Sequential(
            nn.Linear(num_sh, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
    
    def forward(self, edge_feats: torch.Tensor, edge_sh: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_feats: Edge features [E, edge_dim]
            edge_sh: Spherical harmonics [E, num_sh]
        
        Returns:
            Updated edge features [E, edge_dim]
        """
        # Simplified tensor product: weight by spherical harmonics
        weights = self.weight_net(edge_sh)  # [E, edge_dim]
        
        # Element-wise product
        weighted = edge_feats * weights
        
        return weighted


class Allegro(nn.Module):
    """
    Allegro: Local Equivariant Representation Learning.
    
    Key differences from message-passing GNNs:
    - No iterative message passing
- Edge-centric architecture
    - Energy as sum of pairwise contributions
    - Highly parallelizable
    
    Args:
        num_atoms: Number of atom types
        hidden_dim: Hidden dimension
        edge_dim: Edge feature dimension
        num_layers: Number of edge update layers
        num_basis: Number of radial basis functions
        cutoff: Cutoff radius
        lmax: Maximum angular momentum
        output_dim: Output dimension
        num_outputs: Number of output properties
    """
    
    def __init__(self, num_atoms: int = 100, hidden_dim: int = 64,
                 edge_dim: int = 128, num_layers: int = 3, num_basis: int = 8,
                 cutoff: float = 5.0, lmax: int = 2, output_dim: int = 1,
                 num_outputs: int = 1):
        super().__init__()
        
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.lmax = lmax
        
        # Embeddings
        self.node_embedding = nn.Embedding(num_atoms, hidden_dim)
        
        # Edge initialization
        num_sh = (lmax + 1) ** 2
        self.edge_init = nn.Linear(hidden_dim * 2 + num_basis + num_sh, edge_dim)
        
        # Radial basis and spherical harmonics
        self.rbf = RadialBasis(num_basis, cutoff)
        self.sh = SphericalHarmonics(lmax)
        
        # Edge update layers
        self.edge_layers = nn.ModuleList([
            AllegroLayer(edge_dim, hidden_dim, num_basis, hidden_dim, lmax)
            for _ in range(num_layers)
        ])
        
        # Tensor product layers
        self.tp_layers = nn.ModuleList([
            TensorProductLayer(edge_dim, num_sh, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Energy prediction from edge features
        # Energy is sum of pairwise contributions
        self.energy_head = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_outputs),
        )
        
        # Atomic reference energies
        self.atomic_energies = nn.Embedding(num_atoms, 1)
    
    def forward(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None,
                compute_forces: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            atomic_numbers: Atom types [N]
            pos: Positions [N, 3]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N] (optional)
            compute_forces: Whether to compute forces
        
        Returns:
            Dictionary with 'energy' and optionally 'forces'
        """
        if compute_forces:
            pos.requires_grad_(True)
        
        # Edge features
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_length = torch.norm(edge_vec, dim=-1, keepdim=True)
        
        edge_rbf = self.rbf(edge_length)
        edge_sh = self.sh(edge_vec)
        
        # Node features
        node_feats = self.node_embedding(atomic_numbers)
        
        # Initialize edge features
        node_i = node_feats[row]
        node_j = node_feats[col]
        edge_feats = torch.cat([node_i, node_j, edge_rbf, edge_sh], dim=-1)
        edge_feats = self.edge_init(edge_feats)
        
        # Edge update layers
        for edge_layer, tp_layer in zip(self.edge_layers, self.tp_layers):
            edge_feats = edge_layer(edge_feats, node_feats, edge_index, edge_rbf)
            edge_feats = tp_layer(edge_feats, edge_sh)
        
        # Predict pairwise energies
        pair_energies = self.energy_head(edge_feats)  # [E, num_outputs]
        
        # Sum to get total energy
        if batch is None:
            total_energy = pair_energies.sum(dim=0, keepdim=True)  # [1, num_outputs]
        else:
            # Assign edges to graphs
            edge_batch = batch[row]
            num_graphs = batch.max().item() + 1
            total_energy = torch.zeros(num_graphs, pair_energies.shape[1], device=pair_energies.device)
            for i in range(num_graphs):
                mask = edge_batch == i
                total_energy[i] = pair_energies[mask].sum(dim=0)
        
        # Add atomic reference energies
        ref_energy = self.atomic_energies(atomic_numbers)
        if batch is None:
            total_energy = total_energy + ref_energy.sum()
        else:
            for i in range(batch.max().item() + 1):
                mask = batch == i
                total_energy[i] = total_energy[i] + ref_energy[mask].sum()
        
        result = {'energy': total_energy}
        
        if compute_forces:
            forces = -torch.autograd.grad(
                total_energy.sum(), pos, create_graph=self.training
            )[0]
            result['forces'] = forces
        
        return result
    
    def predict_forces(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                       edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict forces."""
        result = self.forward(atomic_numbers, pos, edge_index, batch, compute_forces=True)
        return result['forces']
    
    def get_edge_importance(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get edge importance scores for explainability.
        
        Returns:
            Edge importance scores [E]
        """
        with torch.no_grad():
            result = self.forward(atomic_numbers, pos, edge_index, compute_forces=False)
            
            # Compute gradient of energy w.r.t. edge features
            row, col = edge_index
            edge_vec = pos[col] - pos[row]
            edge_length = torch.norm(edge_vec, dim=-1, keepdim=True)
            
            node_feats = self.node_embedding(atomic_numbers)
            edge_rbf = self.rbf(edge_length)
            edge_sh = self.sh(edge_vec)
            
            node_i = node_feats[row]
            node_j = node_feats[col]
            edge_feats = torch.cat([node_i, node_j, edge_rbf, edge_sh], dim=-1)
            edge_feats = self.edge_init(edge_feats)
            
            for edge_layer, tp_layer in zip(self.edge_layers, self.tp_layers):
                edge_feats = edge_layer(edge_feats, node_feats, edge_index, edge_rbf)
                edge_feats = tp_layer(edge_feats, edge_sh)
            
            pair_energies = self.energy_head(edge_feats)
            
            # Importance = absolute contribution to energy
            importance = torch.abs(pair_energies.squeeze())
            
        return importance
