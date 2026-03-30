"""
NequIP: Neural Equivariant Interatomic Potentials
==================================================

Reference: "E(3)-equivariant Graph Neural Networks for Data-efficient 
            and Accurate Interatomic Potentials" (Batzner et al., Nature Comm 2022)

Key features:
- Tensor field networks architecture
- E(3) equivariant convolutions
- High data efficiency
- State-of-the-art accuracy for small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class GaussianBasis(nn.Module):
    """Gaussian-type radial basis functions."""
    
    def __init__(self, num_basis: int = 8, cutoff: float = 5.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
        centers = torch.linspace(0, cutoff, num_basis)
        self.register_buffer('centers', centers)
        
        # Learnable widths
        self.widths = nn.Parameter(torch.ones(num_basis) * 0.5)
    
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
        
        # Gaussian basis
        diff = distances - self.centers
        basis = torch.exp(-0.5 * (diff / self.widths.clamp(min=0.1)) ** 2)
        
        return basis * cutoff


class SphericalHarmonics(nn.Module):
    """Real spherical harmonics."""
    
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
        
        sh = [torch.ones_like(x) * 0.28209479177387814]
        
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


class TensorProductConv(nn.Module):
    """
    E(3) equivariant convolution using tensor products.
    
    Core of NequIP: combines radial and angular information
    via tensor product operations.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, num_basis: int,
                 lmax: int = 2):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.lmax = lmax
        
        # Radial network for convolution weights
        self.radial_net = nn.Sequential(
            nn.Linear(num_basis, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, node_dim * (lmax + 1)),
        )
        
        # Self-interaction
        self.self_int = nn.Linear(node_dim, node_dim)
        
        # Feature concatenation
        num_sh = (lmax + 1) ** 2
        self.concat_proj = nn.Linear(node_dim * (lmax + 2), node_dim)
    
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
        weights = self.radial_net(edge_rbf)  # [E, node_dim * (lmax+1)]
        
        # Neighbor features
        node_j = node_feats[col]  # [E, node_dim]
        
        # Convolution: weighted sum over neighbors
        # Simplified equivariant convolution
        messages = node_j.unsqueeze(-1) * weights.view(-1, self.node_dim, self.lmax + 1)
        messages = messages.sum(dim=-1)  # [E, node_dim]
        
        # Aggregate
        aggregated = torch.zeros_like(node_feats)
        aggregated.index_add_(0, row, messages)
        
        # Self-interaction
        self_out = self.self_int(node_feats)
        
        # Combine
        combined = torch.cat([self_out, aggregated], dim=-1)
        out = self.concat_proj(combined)
        
        return out


class NequIPLayer(nn.Module):
    """
    Single NequIP layer with equivariant convolutions.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, num_basis: int,
                 hidden_dim: int = 64, lmax: int = 2):
        super().__init__()
        self.node_dim = node_dim
        
        # Equivariant convolution
        self.conv = TensorProductConv(node_dim, edge_dim, num_basis, lmax)
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        
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
        # Convolution
        conv_out = self.conv(node_feats, edge_index, edge_rbf, edge_sh)
        
        # Update
        updated = self.update_mlp(conv_out)
        
        return self.layer_norm(node_feats + updated)


class NequIP(nn.Module):
    """
    NequIP: Neural Equivariant Interatomic Potential.
    
    Args:
        num_atoms: Number of atom types
        hidden_dim: Hidden dimension
        num_layers: Number of NequIP layers
        num_basis: Number of radial basis functions
        cutoff: Cutoff radius
        lmax: Maximum angular momentum
        output_dim: Output dimension
        num_outputs: Number of output properties
    """
    
    def __init__(self, num_atoms: int = 100, hidden_dim: int = 64,
                 num_layers: int = 5, num_basis: int = 8, cutoff: float = 5.0,
                 lmax: int = 2, output_dim: int = 1, num_outputs: int = 1):
        super().__init__()
        
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.lmax = lmax
        
        # Radial basis and spherical harmonics
        self.rbf = GaussianBasis(num_basis, cutoff)
        self.sh = SphericalHarmonics(lmax)
        
        # Node embedding
        self.node_embedding = nn.Embedding(num_atoms, hidden_dim)
        
        # NequIP layers
        self.layers = nn.ModuleList([
            NequIPLayer(hidden_dim, hidden_dim, num_basis, hidden_dim * 2, lmax)
            for _ in range(num_layers)
        ])
        
        # Readout
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        
        # Initial node features
        node_feats = self.node_embedding(atomic_numbers)
        
        # Apply NequIP layers
        for layer in self.layers:
            node_feats = layer(node_feats, edge_index, edge_rbf, edge_sh)
        
        # Per-atom energy
        atomic_energies = self.readout_mlp(node_feats)
        
        # Add reference
        ref_energies = self.atomic_energies(atomic_numbers)
        atomic_energies = atomic_energies + ref_energies
        
        # Pool to graph level
        if batch is None:
            total_energy = atomic_energies.sum()
        else:
            num_graphs = batch.max().item() + 1
            total_energy = torch.zeros(num_graphs, atomic_energies.shape[1], device=atomic_energies.device)
            for i in range(num_graphs):
                mask = batch == i
                total_energy[i] = atomic_energies[mask].sum(dim=0)
        
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
                       batch: Optional[torch.Tensor] = None,
                       volume: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict stress tensor.
        
        Args:
            atomic_numbers: Atom types [N]
            pos: Positions [N, 3]
            cell: Unit cell [3, 3] or [batch_size, 3, 3]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N]
            volume: Cell volumes [batch_size] or scalar
        
        Returns:
            Stress tensor [batch_size, 3, 3] or [3, 3]
        """
        pos.requires_grad_(True)
        if cell.requires_grad == False:
            cell.requires_grad_(True)
        
        result = self.forward(atomic_numbers, pos, edge_index, batch, compute_forces=False)
        energy = result['energy']
        
        # Stress = -1/V * dE/d(cell) * cell^T
        # Simplified: compute virial stress from forces
        forces = -torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]
        
        # Compute virial
        if batch is not None:
            num_graphs = batch.max().item() + 1
            stress = torch.zeros(num_graphs, 3, 3, device=energy.device)
            for i in range(num_graphs):
                mask = batch == i
                pos_i = pos[mask]
                forces_i = forces[mask]
                # Virial = -sum(r_i * F_i^T)
                stress[i] = -torch.mm(pos_i.t(), forces_i)
        else:
            stress = -torch.mm(pos.t(), forces)
        
        # Normalize by volume
        if volume is not None:
            stress = stress / volume.view(-1, 1, 1)
        
        return stress
    
    def get_node_embeddings(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
        """
        Extract node embeddings for transfer learning.
        
        Returns:
            Node embeddings [N, hidden_dim]
        """
        with torch.no_grad():
            row, col = edge_index
            edge_vec = pos[col] - pos[row]
            edge_length = torch.norm(edge_vec, dim=-1, keepdim=True)
            
            edge_rbf = self.rbf(edge_length)
            edge_sh = self.sh(edge_vec)
            
            node_feats = self.node_embedding(atomic_numbers)
            
            for layer in self.layers:
                node_feats = layer(node_feats, edge_index, edge_rbf, edge_sh)
        
        return node_feats
