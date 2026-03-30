"""
Utility functions for GNN operations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict
import math


def compute_distance(pos_i: torch.Tensor, pos_j: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance between two sets of positions."""
    return torch.norm(pos_i - pos_j, dim=-1, keepdim=True)


def compute_edge_vectors(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Compute edge vectors from positions and edge indices.
    
    Args:
        pos: Node positions [N, 3]
        edge_index: Edge indices [2, E]
    
    Returns:
        Edge vectors [E, 3]
    """
    row, col = edge_index
    return pos[col] - pos[row]


def compute_spherical_harmonics(edge_vec: torch.Tensor, lmax: int = 2) -> torch.Tensor:
    """
    Compute real spherical harmonics up to order lmax.
    
    Args:
        edge_vec: Edge vectors [E, 3]
        lmax: Maximum angular momentum (default 2)
    
    Returns:
        Spherical harmonics [E, (lmax+1)^2]
    """
    # Normalize edge vectors
    r = torch.norm(edge_vec, dim=-1, keepdim=True).clamp(min=1e-8)
    x, y, z = edge_vec[:, 0] / r.squeeze(), edge_vec[:, 1] / r.squeeze(), edge_vec[:, 2] / r.squeeze()
    
    sh_list = []
    
    # l = 0
    sh_list.append(torch.ones_like(x) * 0.28209479177387814)  # Y_00
    
    if lmax >= 1:
        # l = 1 (p orbitals)
        sh_list.extend([
            -0.4886025119029199 * y,  # Y_1-1
            0.4886025119029199 * z,   # Y_10
            -0.4886025119029199 * x,  # Y_11
        ])
    
    if lmax >= 2:
        # l = 2 (d orbitals)
        x2, y2, z2 = x**2, y**2, z**2
        xy, xz, yz = x*y, x*z, y*z
        sh_list.extend([
            1.0925484305920792 * xy,                                    # Y_2-2
            -1.0925484305920792 * yz,                                   # Y_2-1
            0.94617469575755997 * z2 - 0.31539156525251999 * (x2 + y2), # Y_20
            -1.0925484305920792 * xz,                                   # Y_21
            0.54627421529603959 * (x2 - y2),                            # Y_22
        ])
    
    if lmax >= 3:
        # l = 3 (f orbitals)
        sh_list.extend([
            0.5900435899266435 * y * (3*x2 - y2),  # Y_3-3
            2.890611442640554 * xy * z,             # Y_3-2
            0.4570457994644658 * y * (4*z2 - x2 - y2),  # Y_3-1
            0.3731763325901154 * z * (2*z2 - 3*x2 - 3*y2),  # Y_30
            0.4570457994644658 * x * (4*z2 - x2 - y2),  # Y_31
            1.445305721320277 * z * (x2 - y2),      # Y_32
            0.5900435899266435 * x * (x2 - 3*y2),   # Y_33
        ])
    
    return torch.stack(sh_list, dim=-1)


def radius_graph(pos: torch.Tensor, r_max: float, batch: Optional[torch.Tensor] = None,
                 max_num_neighbors: int = 32) -> torch.Tensor:
    """
    Build radius graph from positions.
    
    Args:
        pos: Node positions [N, 3]
        r_max: Maximum cutoff radius
        batch: Batch indices [N] (optional)
        max_num_neighbors: Maximum number of neighbors per node
    
    Returns:
        Edge indices [2, E]
    """
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    
    # Compute pairwise distances within each batch
    edge_list = []
    for batch_idx in torch.unique(batch):
        mask = batch == batch_idx
        pos_batch = pos[mask]
        n_nodes = pos_batch.shape[0]
        
        # Compute distance matrix
        dist = torch.cdist(pos_batch, pos_batch)
        
        # Apply cutoff
        mask_dist = (dist < r_max) & (dist > 0)
        
        # Get edge indices
        edge_idx = torch.nonzero(mask_dist, as_tuple=False)
        
        # Map back to original indices
        node_idx = torch.nonzero(mask, as_tuple=False).squeeze()
        edge_idx[:, 0] = node_idx[edge_idx[:, 0]]
        edge_idx[:, 1] = node_idx[edge_idx[:, 1]]
        
        edge_list.append(edge_idx.T)
    
    if len(edge_list) > 0:
        return torch.cat(edge_list, dim=1)
    else:
        return torch.zeros((2, 0), dtype=torch.long, device=pos.device)


class GaussianBasis(nn.Module):
    """Gaussian radial basis functions."""
    
    def __init__(self, n_rbf: int = 20, cutoff: float = 5.0, start: float = 0.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.start = start
        
        # Centers and widths
        centers = torch.linspace(start, cutoff, n_rbf)
        widths = (cutoff - start) / n_rbf * torch.ones(n_rbf)
        
        self.register_buffer('centers', centers)
        self.register_buffer('widths', widths)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: Pairwise distances [E, 1] or [E]
        Returns:
            Radial basis functions [E, n_rbf]
        """
        if distances.dim() == 1:
            distances = distances.unsqueeze(-1)
        
        # Compute Gaussian RBF
        return torch.exp(-((distances - self.centers) ** 2) / self.widths ** 2)


class CosineCutoff(nn.Module):
    """Cosine cutoff function."""
    
    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: Pairwise distances [E, 1] or [E]
        Returns:
            Cutoff values [E, 1] or [E]
        """
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class SphericalBasis(nn.Module):
    """Spherical basis functions for angular information."""
    
    def __init__(self, n_spherical: int = 7, l_spherical: int = 6):
        super().__init__()
        self.n_spherical = n_spherical
        self.l_spherical = l_spherical
    
    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angles: Angles in radians [E]
        Returns:
            Spherical basis functions [E, n_spherical * l_spherical]
        """
        # Simplified implementation - use Fourier basis
        basis = []
        for n in range(1, self.n_spherical + 1):
            for l in range(self.l_spherical + 1):
                basis.append(torch.cos(n * angles * (l + 1)))
                basis.append(torch.sin(n * angles * (l + 1)))
        
        return torch.stack(basis, dim=-1)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """
    Scatter add operation.
    
    Args:
        src: Source tensor
        index: Indices to scatter to
        dim: Dimension to scatter along
        dim_size: Output size in scatter dimension
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    
    shape = list(src.shape)
    shape[dim] = dim_size
    
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index.unsqueeze(dim).expand_as(src), src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    """
    Scatter mean operation.
    
    Args:
        src: Source tensor
        index: Indices to scatter to
        dim: Dimension to scatter along
        dim_size: Output size in scatter dimension
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    
    out = scatter_add(src, index, dim, dim_size)
    
    # Count occurrences
    ones = torch.ones_like(src)
    count = scatter_add(ones, index, dim, dim_size)
    count = count.clamp(min=1)
    
    return out / count


class LayerNorm(nn.Module):
    """Layer normalization with optional elementwise affine."""
    
    def __init__(self, hidden_dim: int, affine: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        if affine:
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., hidden_dim]
        Returns:
            Normalized tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + 1e-8)
        
        if self.weight is not None:
            x = x * self.weight + self.bias
        
        return x


class Irrep:
    """Irreducible representation for equivariant operations."""
    
    def __init__(self, l: int, p: int = 1):
        self.l = l  # Angular momentum
        self.p = p  # Parity (1 for even, -1 for odd)
    
    def __repr__(self):
        return f"Irrep({self.l}, {'e' if self.p == 1 else 'o'})"
    
    def dim(self) -> int:
        """Dimension of representation."""
        return 2 * self.l + 1


def tp_size(irreps_in1: List[Irrep], irreps_in2: List[Irrep]) -> List[Irrep]:
    """
    Compute output irreps from tensor product.
    
    Args:
        irreps_in1: First set of irreps
        irreps_in2: Second set of irreps
    
    Returns:
        Output irreps
    """
    result = []
    for ir1 in irreps_in1:
        for ir2 in irreps_in2:
            # Tensor product decomposition
            for l_out in range(abs(ir1.l - ir2.l), ir1.l + ir2.l + 1):
                p_out = ir1.p * ir2.p
                result.append(Irrep(l_out, p_out))
    return result
