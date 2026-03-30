"""
Equiformer: E(3) Equivariant Graph Attention Transformer
===========================================================

Reference: "Equiformer: Equivariant Graph Attention Transformer 
            for 3D Atomistic Graphs" (Liao & Smidt, ICLR 2023)

Key features:
- E(3) equivariant attention mechanism
- Irreducible representation (irrep) features
- Tensor product operations
- Simplified from SO(3) to focus on scalar and vector features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


class RadialBasisEmbedding(nn.Module):
    """Gaussian radial basis functions with learnable centers."""
    
    def __init__(self, num_basis: int = 32, cutoff: float = 5.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_basis))
        self.width = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: Pairwise distances [E, 1]
        Returns:
            Radial basis [E, num_basis]
        """
        # Cutoff function
        cutoff_val = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoff_val = cutoff_val * (distances < self.cutoff).float()
        
        # Gaussian basis
        diff = distances - self.centers
        basis = torch.exp(-(diff ** 2) / (self.width ** 2))
        
        return basis * cutoff_val


class SphericalHarmonics(nn.Module):
    """Compute spherical harmonics for edge vectors."""
    
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
        # Normalize
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
        
        return torch.stack(sh_list, dim=-1)


class TensorProductConv(nn.Module):
    """
    Tensor product convolution layer for equivariant features.
    
    Simplified to handle scalar (l=0) and vector (l=1) features.
    """
    
    def __init__(self, hidden_dim: int, num_basis: int = 32, lmax: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        self.lmax = lmax
        
        # For scalar features (l=0)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # For vector features (l=1)
        if lmax >= 1:
            self.vector_mlp = nn.Sequential(
                nn.Linear(num_basis, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
    
    def forward(self, scalar: torch.Tensor, vector: Optional[torch.Tensor],
                edge_sh: torch.Tensor, edge_length: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            scalar: Scalar features [N, hidden_dim]
            vector: Vector features [N, 3, hidden_dim] or None
            edge_sh: Edge spherical harmonics [E, num_sh]
            edge_length: Edge lengths [E, 1]
            edge_index: Edge indices [2, E]
        
        Returns:
            Updated scalar and vector features
        """
        row, col = edge_index
        
        # Scalar message (from l=0 x l=0 = l=0)
        scalar_weights = self.scalar_mlp(edge_length)  # [E, hidden_dim]
        scalar_msg = scalar[col] * scalar_weights  # [E, hidden_dim]
        
        # Aggregate scalar messages
        out_scalar = torch.zeros_like(scalar)
        out_scalar.index_add_(0, row, scalar_msg)
        
        out_vector = None
        if vector is not None and self.lmax >= 1:
            # Vector message from tensor product of scalars and edge vectors
            # Simplified: scalar * vector direction
            vector_weights = self.vector_mlp(edge_length)  # [E, hidden_dim]
            
            # Get vector features of neighbors
            vector_j = vector[col]  # [E, 3, hidden_dim]
            
            # Weight by radial function
            vector_msg = vector_j * vector_weights.unsqueeze(1)  # [E, 3, hidden_dim]
            
            # Aggregate
            out_vector = torch.zeros_like(vector)
            out_vector.index_add_(0, row, vector_msg)
        
        return out_scalar, out_vector


class EquivariantAttention(nn.Module):
    """E(3) equivariant graph attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, cutoff: float = 5.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cutoff = cutoff
        
        assert hidden_dim % num_heads == 0
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge embedding for attention weights
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_vec: torch.Tensor, edge_length: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edge indices [2, E]
            edge_vec: Edge vectors [E, 3]
            edge_length: Edge lengths [E, 1]
        
        Returns:
            Updated features [N, hidden_dim]
        """
        row, col = edge_index
        N = x.shape[0]
        
        # Compute Q, K, V
        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        
        # Get edge Q, K
        q_i = q[row]  # [E, num_heads, head_dim]
        k_j = k[col]  # [E, num_heads, head_dim]
        
        # Attention scores
        attn = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, num_heads]
        
        # Add edge bias
        edge_bias = self.edge_mlp(edge_length)  # [E, num_heads]
        attn = attn + edge_bias
        
        # Apply cutoff and softmax per source node
        cutoff = 0.5 * (torch.cos(edge_length * math.pi / self.cutoff) + 1.0)
        cutoff = cutoff * (edge_length < self.cutoff).float()
        attn = attn * cutoff
        
        # Softmax over neighbors
        attn_exp = torch.exp(attn - attn.max(dim=0, keepdim=True)[0])
        attn_sum = torch.zeros(N, self.num_heads, device=x.device)
        attn_sum.index_add_(0, row, attn_exp)
        attn = attn_exp / (attn_sum[row] + 1e-8)
        
        # Get values
        v_j = v[col]  # [E, num_heads, head_dim]
        
        # Weighted sum
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, row, attn.unsqueeze(-1) * v_j)
        
        # Reshape and project
        out = out.view(N, self.hidden_dim)
        out = self.out_proj(out)
        out = self.layer_norm(x + out)
        
        return out


class EquiformerLayer(nn.Module):
    """Single Equiformer layer with equivariant attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, num_basis: int = 32,
                 cutoff: float = 5.0, lmax: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lmax = lmax
        
        # Equivariant attention
        self.attention = EquivariantAttention(hidden_dim, num_heads, cutoff)
        
        # Tensor product convolution
        self.tp_conv = TensorProductConv(hidden_dim, num_basis, lmax)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Vector FFN if using vectors
        if lmax >= 1:
            self.vector_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
    
    def forward(self, scalar: torch.Tensor, vector: Optional[torch.Tensor],
                edge_index: torch.Tensor, edge_vec: torch.Tensor,
                edge_length: torch.Tensor, edge_sh: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            scalar: Scalar features [N, hidden_dim]
            vector: Vector features [N, 3, hidden_dim] or None
            edge_index: Edge indices [2, E]
            edge_vec: Edge vectors [E, 3]
            edge_length: Edge lengths [E, 1]
            edge_sh: Spherical harmonics [E, num_sh]
        
        Returns:
            Updated scalar and vector features
        """
        # Attention on scalar features
        h_attn = self.attention(scalar, edge_index, edge_vec, edge_length)
        
        # Tensor product convolution
        h_tp, v_tp = self.tp_conv(h_attn, vector, edge_sh, edge_length, edge_index)
        
        # Residual connection for scalars
        scalar_out = scalar + h_tp
        
        # FFN for scalars
        scalar_out = scalar_out + self.ffn(self.ffn_norm(scalar_out))
        
        # Vector updates
        vector_out = vector
        if vector is not None and self.lmax >= 1:
            if v_tp is not None:
                vector_out = vector + v_tp
            vector_out = self.vector_ffn(vector_out.transpose(1, 2)).transpose(1, 2)
        
        return scalar_out, vector_out


class Equiformer(nn.Module):
    """
    Equiformer: E(3) Equivariant Graph Attention Transformer.
    
    Args:
        num_atoms: Number of atom types
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        num_basis: Number of radial basis functions
        cutoff: Cutoff radius
        lmax: Maximum angular momentum (1 for vectors, 2 for tensors)
        output_dim: Output dimension
        dropout: Dropout rate
    """
    
    def __init__(self, num_atoms: int = 100, hidden_dim: int = 128,
                 num_layers: int = 6, num_heads: int = 8, num_basis: int = 32,
                 cutoff: float = 5.0, lmax: int = 1, output_dim: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.lmax = lmax
        
        # Embeddings
        self.atom_embedding = nn.Embedding(num_atoms, hidden_dim)
        
        # Radial basis and spherical harmonics
        self.rbf = RadialBasisEmbedding(num_basis, cutoff)
        self.sh = SphericalHarmonics(lmax)
        
        # Equiformer layers
        self.layers = nn.ModuleList([
            EquiformerLayer(hidden_dim, num_heads, num_basis, cutoff, lmax, dropout)
            for _ in range(num_layers)
        ])
        
        # Initialize vector features if lmax >= 1
        if lmax >= 1:
            self.vector_init = nn.Linear(hidden_dim, hidden_dim)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            atomic_numbers: Atom types [N]
            pos: Positions [N, 3]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N] (optional)
        
        Returns:
            Energy predictions [batch_size, output_dim]
        """
        # Compute edge features
        row, col = edge_index
        edge_vec = pos[col] - pos[row]
        edge_length = torch.norm(edge_vec, dim=-1, keepdim=True)
        
        # Radial basis and spherical harmonics
        edge_rbf = self.rbf(edge_length)
        edge_sh = self.sh(edge_vec)
        
        # Initial embeddings
        scalar = self.atom_embedding(atomic_numbers)
        
        # Initialize vector features
        vector = None
        if self.lmax >= 1:
            # Initialize vectors from scalars (simplified)
            vector = torch.zeros(scalar.shape[0], 3, self.hidden_dim, device=scalar.device)
        
        # Apply layers
        for layer in self.layers:
            scalar, vector = layer(scalar, vector, edge_index, edge_vec, edge_length, edge_sh)
        
        # Pooling
        if batch is None:
            h_graph = scalar.mean(dim=0, keepdim=True)
        else:
            num_graphs = batch.max().item() + 1
            h_graph = torch.zeros(num_graphs, self.hidden_dim, device=scalar.device)
            for i in range(num_graphs):
                mask = batch == i
                h_graph[i] = scalar[mask].mean(dim=0)
        
        # Output
        return self.output_head(h_graph)
    
    def predict_forces(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                       edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict forces as negative gradient of energy."""
        pos.requires_grad_(True)
        energy = self.forward(atomic_numbers, pos, edge_index, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return forces
