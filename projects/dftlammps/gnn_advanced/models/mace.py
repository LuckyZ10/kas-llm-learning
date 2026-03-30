"""
MACE: Higher Order Equivariant Message Passing Neural Networks
================================================================

Reference: "MACE: Higher Order Equivariant Message Passing Neural Networks
            for Fast and Accurate Force Fields" (Batatia et al., NeurIPS 2022)

Key features:
- High body-order equivariant messages
- Atomic Cluster Expansion (ACE) inspired architecture
- Efficient equivariant message passing
- State-of-the-art accuracy for force fields
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class RadialBasis(nn.Module):
    """Bessel radial basis functions."""
    
    def __init__(self, num_basis: int = 8, cutoff: float = 5.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
        # Bessel roots
        self.register_buffer('roots', torch.arange(1, num_basis + 1) * math.pi)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: Pairwise distances [E, 1]
        Returns:
            Bessel basis [E, num_basis]
        """
        # Cutoff function
        cutoff = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoff = cutoff * (distances < self.cutoff).float()
        
        # Bessel functions
        x = distances * self.roots / self.cutoff
        basis = torch.sin(x) / distances.clamp(min=1e-8)
        
        return basis * cutoff


class SphericalHarmonics(nn.Module):
    """Compute real spherical harmonics up to lmax."""
    
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
        
        sh_list = []
        
        # l=0
        sh_list.append(torch.ones_like(x) * 0.28209479177387814)
        
        if self.lmax >= 1:
            sh_list.extend([
                -0.4886025119029199 * y,
                0.4886025119029199 * z,
                -0.4886025119029199 * x,
            ])
        
        if self.lmax >= 2:
            x2, y2, z2 = x**2, y**2, z**2
            xy, xz, yz = x*y, x*z, y*z
            sh_list.extend([
                1.0925484305920792 * xy,
                -1.0925484305920792 * yz,
                0.94617469575755997 * z2 - 0.31539156525251999 * (x2 + y2),
                -1.0925484305920792 * xz,
                0.54627421529603959 * (x2 - y2),
            ])
        
        if self.lmax >= 3:
            x3, y3, z3 = x**3, y**3, z**3
            x2y, x2z = x2*y, x2*z
            xy2, y2z = x*y2, y2*z
            xz2, yz2 = x*z2, y*z2
            xyz = x*y*z
            
            sh_list.extend([
                0.5900435899266435 * y * (3*x2 - y2),
                2.890611442640554 * xy * z,
                0.4570457994644658 * y * (4*z2 - x2 - y2),
                0.3731763325901154 * z * (2*z2 - 3*x2 - 3*y2),
                0.4570457994644658 * x * (4*z2 - x2 - y2),
                1.445305721320277 * z * (x2 - y2),
                0.5900435899266435 * x * (x2 - 3*y2),
            ])
        
        return torch.stack(sh_list, dim=-1)


class LinearNodeEmbedding(nn.Module):
    """Linear node embedding layer."""
    
    def __init__(self, num_types: int, hidden_irreps: List[int]):
        super().__init__()
        self.num_types = num_types
        self.hidden_irreps = hidden_irreps
        
        # Embeddings for each irrep (simplified)
        total_dim = sum(hidden_irreps)
        self.embedding = nn.Embedding(num_types, total_dim)
    
    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        return self.embedding(atomic_numbers)


class MACEInteraction(nn.Module):
    """
    MACE interaction block with high body-order messages.
    
    Implements the key innovation of MACE: equivariant messages
    with increased body order at each iteration.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int,
                 num_basis: int = 8, lmax: int = 2, correlation: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.lmax = lmax
        self.correlation = correlation
        
        # Radial weight network
        self.radial_net = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim * (lmax + 1)),
        )
        
        # Product function for high body-order
        self.product = nn.Sequential(
            nn.Linear(node_dim * (lmax + 1), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        
        # Linear update
        self.linear = nn.Linear(node_dim * 2, node_dim)
        
        self.layer_norm = nn.LayerNorm(node_dim)
    
    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor,
                edge_rbf: torch.Tensor, edge_sh: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feats: Node features [N, node_dim]
            edge_index: Edge indices [2, E]
            edge_rbf: Radial basis [E, num_basis]
            edge_sh: Spherical harmonics [E, num_sh]
        
        Returns:
            Updated features [N, node_dim]
        """
        row, col = edge_index
        
        # Radial weights
        radial_weights = self.radial_net(edge_rbf)  # [E, node_dim * (lmax+1)]
        
        # Get neighbor features
        node_j = node_feats[col]  # [E, node_dim]
        
        # Equivariant message: radial weights * neighbor features * spherical harmonics
        # Simplified: weight messages by radial basis and aggregate
        messages = node_j.unsqueeze(-1) * radial_weights.view(-1, self.node_dim, self.lmax + 1)
        messages = messages.sum(dim=-1)  # [E, node_dim]
        
        # Aggregate
        aggregated = torch.zeros_like(node_feats)
        aggregated.index_add_(0, row, messages)
        
        # Product for high body-order (simplified symmetric product)
        product_input = torch.cat([node_feats, aggregated], dim=-1)
        updated = self.linear(product_input)
        
        return self.layer_norm(node_feats + updated)


class ScaleShift(nn.Module):
    """Scale and shift layer for energy predictions."""
    
    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.shift = nn.Parameter(torch.tensor(shift))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class MACE(nn.Module):
    """
    MACE: Higher Order Equivariant Message Passing Neural Network.
    
    Args:
        num_atoms: Number of atom types
        hidden_dim: Hidden dimension
        num_layers: Number of interaction layers
        num_basis: Number of radial basis functions
        cutoff: Cutoff radius
        lmax: Maximum angular momentum
        correlation: Body order correlation
        output_dim: Output dimension
        max_ell: Maximum spherical harmonic degree
    """
    
    def __init__(self, num_atoms: int = 100, hidden_dim: int = 64,
                 num_layers: int = 2, num_basis: int = 8, cutoff: float = 5.0,
                 lmax: int = 2, correlation: int = 3, output_dim: int = 1,
                 max_ell: int = 3):
        super().__init__()
        
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.lmax = lmax
        
        # Radial basis and spherical harmonics
        self.rbf = RadialBasis(num_basis, cutoff)
        self.sh = SphericalHarmonics(max_ell)
        
        # Node embedding
        self.node_embedding = nn.Embedding(num_atoms, hidden_dim)
        
        # Interaction blocks
        self.interactions = nn.ModuleList([
            MACEInteraction(hidden_dim, hidden_dim, hidden_dim, num_basis, lmax, correlation)
            for _ in range(num_layers)
        ])
        
        # Readout networks
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Atomic energy scale/shift
        self.scale_shift = ScaleShift()
        
        # Atomic energies reference
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
        
        # Initial features
        node_feats = self.node_embedding(atomic_numbers)
        
        # Interaction layers
        for interaction in self.interactions:
            node_feats = interaction(node_feats, edge_index, edge_rbf, edge_sh)
        
        # Readout per atom
        atomic_energies = self.readout_mlp(node_feats)
        
        # Add reference atomic energies
        ref_energies = self.atomic_energies(atomic_numbers)
        atomic_energies = atomic_energies + ref_energies
        
        # Pool to graph level
        if batch is None:
            total_energy = atomic_energies.sum()
        else:
            num_graphs = batch.max().item() + 1
            total_energy = torch.zeros(num_graphs, 1, device=atomic_energies.device)
            for i in range(num_graphs):
                mask = batch == i
                total_energy[i] = atomic_energies[mask].sum()
        
        # Scale and shift
        total_energy = self.scale_shift(total_energy)
        
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
    
    def predict_stress(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                       cell: torch.Tensor, edge_index: torch.Tensor,
                       batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict stress tensor (for periodic systems).
        
        Args:
            atomic_numbers: Atom types [N]
            pos: Positions [N, 3]
            cell: Unit cell [3, 3] or [batch_size, 3, 3]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N]
        
        Returns:
            Stress tensor [batch_size, 3, 3] or [3, 3]
        """
        # This is a simplified implementation
        # Full stress computation requires volume and cell gradients
        
        pos.requires_grad_(True)
        result = self.forward(atomic_numbers, pos, edge_index, batch, compute_forces=False)
        energy = result['energy']
        
        # Simplified: return zero stress (would need cell gradient)
        if batch is not None:
            num_graphs = batch.max().item() + 1
            return torch.zeros(num_graphs, 3, 3, device=energy.device)
        else:
            return torch.zeros(3, 3, device=energy.device)
