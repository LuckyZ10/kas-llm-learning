"""
GPS (Graph GPS) Model
=====================

GPS: General, Powerful, Scalable Graph Transformer
Reference: "Recipe for a General, Powerful, Scalable Graph Transformer" 
           (Rampasek et al., NeurIPS 2022)

This implementation combines:
- Message Passing Neural Networks (MPNN)
- Global Attention (Transformer)
- Positional/Structural encodings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional edge features."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [N, hidden_dim]
            attn_mask: Attention mask [N, N] (optional)
            edge_attr: Edge features for attention bias [E, hidden_dim] (optional)
        
        Returns:
            Updated features [N, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape if x.dim() == 3 else (1, x.shape[0], x.shape[1])
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        if batch_size == 1:
            out = out.squeeze(0)
        
        return out


class MPNNLayer(nn.Module):
    """Message Passing Neural Network layer."""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Message function
        in_dim = 2 * hidden_dim + edge_dim if edge_dim > 0 else 2 * hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
        
        Returns:
            Updated features [N, hidden_dim]
        """
        row, col = edge_index
        
        # Compute messages
        x_i = x[row]  # [E, hidden_dim]
        x_j = x[col]  # [E, hidden_dim]
        
        if edge_attr is not None:
            message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            message_input = torch.cat([x_i, x_j], dim=-1)
        
        messages = self.message_mlp(message_input)  # [E, hidden_dim]
        
        # Aggregate messages
        aggr = torch.zeros_like(x)
        aggr.index_add_(0, row, messages)
        
        # Update
        update_input = torch.cat([x, aggr], dim=-1)
        out = self.update_mlp(update_input)
        out = self.layer_norm(x + out)  # Residual + LayerNorm
        
        return out


class GPSLayer(nn.Module):
    """
    GPS Layer: Hybrid MPNN + Transformer layer.
    
    Combines local message passing with global attention.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, edge_dim: int = 0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # MPNN sub-layer
        self.mpnn = MPNNLayer(hidden_dim, edge_dim, dropout)
        
        # Global Attention sub-layer
        self.attn = MultiHeadAttention(hidden_dim, num_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch indices [N] (optional)
        
        Returns:
            Updated features [N, hidden_dim]
        """
        # MPNN branch
        h_mpnn = self.mpnn(x, edge_index, edge_attr)
        
        # Global attention branch
        # Create attention mask based on batch
        if batch is not None:
            # Nodes in the same graph attend to each other
            attn_mask = (batch.unsqueeze(0) == batch.unsqueeze(1)).float()
        else:
            attn_mask = None
        
        h_attn = self.attn(x, attn_mask)
        h_attn = self.attn_norm(x + self.dropout(h_attn))
        
        # Combine MPNN and Attention (sum by default)
        h = h_mpnn + h_attn
        
        # FFN
        h_out = self.ffn(h)
        h_out = self.ffn_norm(h + self.dropout(h_out))
        
        return h_out


class RWSEEncoder(nn.Module):
    """Random Walk Structural Encoding."""
    
    def __init__(self, dim_rwse: int = 20, hidden_dim: int = 64):
        super().__init__()
        self.dim_rwse = dim_rwse
        self.encoder = nn.Sequential(
            nn.Linear(dim_rwse, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, rwse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rwse: Random walk features [N, dim_rwse]
        Returns:
            Encoded features [N, hidden_dim]
        """
        return self.encoder(rwse)


class GPSModel(nn.Module):
    """
    GPS: General, Powerful, Scalable Graph Transformer.
    
    A hybrid architecture combining MPNN and Transformer layers
    for graph representation learning.
    
    Args:
        num_atoms: Number of atom types
        hidden_dim: Hidden dimension size
        num_layers: Number of GPS layers
        num_heads: Number of attention heads
        edge_dim: Edge feature dimension (0 for no edge features)
        output_dim: Output dimension
        dropout: Dropout rate
        use_rwse: Whether to use Random Walk Structural Encoding
        dim_rwse: Dimension of RWSE
    """
    
    def __init__(self, num_atoms: int = 100, hidden_dim: int = 128,
                 num_layers: int = 4, num_heads: int = 8, edge_dim: int = 0,
                 output_dim: int = 1, dropout: float = 0.1,
                 use_rwse: bool = False, dim_rwse: int = 20):
        super().__init__()
        
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_rwse = use_rwse
        
        # Atom embedding
        self.atom_embedding = nn.Embedding(num_atoms, hidden_dim)
        
        # Optional RWSE encoding
        if use_rwse:
            self.rwse_encoder = RWSEEncoder(dim_rwse, hidden_dim)
            self.rwse_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # GPS layers
        self.layers = nn.ModuleList([
            GPSLayer(hidden_dim, num_heads, edge_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                rwse: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            atomic_numbers: Atom types [N]
            pos: Atom positions [N, 3]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch indices [N] (optional)
            rwse: Random walk features [N, dim_rwse] (optional)
        
        Returns:
            Graph-level predictions [batch_size, output_dim]
        """
        # Node embeddings
        h = self.atom_embedding(atomic_numbers)
        
        # Add RWSE if provided
        if self.use_rwse and rwse is not None:
            h_rwse = self.rwse_encoder(rwse)
            h = self.rwse_proj(torch.cat([h, h_rwse], dim=-1))
        
        # Apply GPS layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, batch)
        
        # Global pooling
        if batch is None:
            h_graph = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            # Pool per graph
            num_graphs = batch.max().item() + 1
            h_graph = torch.zeros(num_graphs, self.hidden_dim, device=h.device)
            for i in range(num_graphs):
                mask = batch == i
                h_graph[i] = h[mask].mean(dim=0)
        
        # Output
        out = self.output_head(h_graph)
        
        return out
    
    def predict_forces(self, atomic_numbers: torch.Tensor, pos: torch.Tensor,
                       edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                       batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict forces via gradient of energy.
        
        Args:
            atomic_numbers: Atom types [N]
            pos: Atom positions [N, 3] (requires_grad=True)
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch indices [N] (optional)
        
        Returns:
            Forces [N, 3]
        """
        pos.requires_grad_(True)
        energy = self.forward(atomic_numbers, pos, edge_index, edge_attr, batch)
        
        # Forces are negative gradient of energy
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        
        return forces
