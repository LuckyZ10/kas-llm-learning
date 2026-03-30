"""
Riemannian Flow Matching for Crystal Generation
===============================================

Implementation of Flow Matching on manifolds for crystal structure generation.
Based on FlowMM (Miller et al., ICML 2024) and CrystalFlow (Nature Commun 2025).

Key features:
- Riemannian flow matching for periodic boundary conditions
- Conditional vector fields for property-guided generation
- Straight-line probability paths on manifolds

References:
- Miller et al., "FlowMM: Generating Materials with Riemannian Flow Matching", ICML 2024
- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class FlowMatchingConfig:
    """Configuration for Flow Matching model."""
    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_atoms: int = 100
    
    # Atom types
    num_elements: int = 100
    
    # Flow parameters
    sigma_min: float = 1e-4
    time_eps: float = 1e-5
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    # Conditioning
    use_conditioning: bool = True
    num_properties: int = 5
    
    # ODE solver
    ode_method: str = "euler"  # "euler", "rk4", "dopri5"
    num_steps: int = 50  # Number of integration steps


class FourierFeatures(nn.Module):
    """Fourier features for time conditioning."""
    
    def __init__(self, n_frequencies: int = 128):
        super().__init__()
        self.n_frequencies = n_frequencies
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time in [0, 1] (B,)
        Returns:
            Fourier features (B, 2*n_frequencies)
        """
        # Ensure t is in [0, 1]
        t = t.clamp(0, 1)
        
        # Frequencies: 1, 2, 4, 8, ..., 2^(n-1)
        freqs = 2 ** torch.arange(self.n_frequencies, device=t.device).float()
        
        # (B, 1) * (n_frequencies,) = (B, n_frequencies)
        args = t.unsqueeze(1) * freqs.unsqueeze(0) * 2 * math.pi
        
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EGNNLayer(nn.Module):
    """
    Equivariant Graph Neural Network layer for crystals.
    Handles periodic boundary conditions.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        # Edge network
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Node network
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim)
        )
        
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        lattice: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node features (B, N, in_dim)
            x: Fractional coordinates (B, N, 3)
            edge_index: Edge indices (2, E) - if None, fully connected
            lattice: Lattice parameters (B, 6)
            
        Returns:
            Updated h and x
        """
        B, N, _ = h.shape
        
        # Compute pairwise distances with periodic boundary conditions
        if edge_index is None:
            # Fully connected
            x_i = x.unsqueeze(2)  # (B, N, 1, 3)
            x_j = x.unsqueeze(1)  # (B, 1, N, 3)
            
            # Fractional coordinate differences
            dx = x_j - x_i  # (B, N, N, 3)
            
            # Apply minimum image convention if lattice provided
            if lattice is not None:
                dx = dx - torch.round(dx)  # Minimum image
            
            # Euclidean distance in fractional coords
            dist = torch.sqrt((dx ** 2).sum(dim=-1, keepdim=True) + 1e-8)  # (B, N, N, 1)
            
            # Edge features
            h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, in_dim)
            h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, in_dim)
            
            edge_input = torch.cat([h_i, h_j, dist], dim=-1)  # (B, N, N, 2*in_dim+1)
            
            # Edge messages
            edge_features = self.edge_mlp(edge_input)  # (B, N, N, hidden_dim)
            
            # Aggregate
            messages = edge_features.sum(dim=2)  # (B, N, hidden_dim)
        else:
            # Use provided edge index
            row, col = edge_index
            messages = torch.zeros_like(h)
            # Implementation for sparse edges would go here
        
        # Update nodes
        node_input = torch.cat([h, messages], dim=-1)
        h_out = h + self.node_mlp(node_input)
        
        # For coordinates, we don't update in flow matching (equivariance handled differently)
        x_out = x
        
        return h_out, x_out


class RiemannianFlowMatcher(nn.Module):
    """
    Riemannian Flow Matching for crystal structure generation.
    
    Learns vector fields on the manifold of crystal structures.
    """
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.atom_embed = nn.Embedding(config.num_elements + 1, config.hidden_dim)
        self.coord_embed = nn.Linear(3, config.hidden_dim)
        self.lattice_embed = nn.Linear(6, config.hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            FourierFeatures(64),
            nn.Linear(256, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Property conditioning
        if config.use_conditioning:
            self.prop_embed = nn.Sequential(
                nn.Linear(config.num_properties, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                config.hidden_dim,
                config.hidden_dim,
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # Output vector fields
        self.atom_vf = nn.Linear(config.hidden_dim, config.num_elements)
        self.coord_vf = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3)
        )
        self.lattice_vf = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 6)
        )
        
    def forward(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        t: torch.Tensor,
        properties: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict vector field at time t.
        
        Args:
            atom_types: (B, N) - can be soft (logits) or hard indices
            frac_coords: (B, N, 3)
            lattice: (B, 6)
            t: (B,) Time in [0, 1]
            properties: (B, num_properties)
            
        Returns:
            Vector fields for each component
        """
        B, N = atom_types.shape
        
        # Handle soft atom types (logits) vs hard indices
        if atom_types.dtype == torch.long:
            h = self.atom_embed(atom_types)  # (B, N, hidden_dim)
        else:
            # Soft: weighted embedding
            h = torch.matmul(atom_types, self.atom_embed.weight)  # (B, N, hidden_dim)
        
        h = h + self.coord_embed(frac_coords)
        
        # Add time embedding
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        h = h + t_emb.unsqueeze(1)
        
        # Add property conditioning
        if self.config.use_conditioning and properties is not None:
            p_emb = self.prop_embed(properties)
            h = h + p_emb.unsqueeze(1)
        
        # EGNN layers
        x = frac_coords
        for layer in self.egnn_layers:
            h, x = layer(h, x, lattice=lattice)
        
        # Predict vector fields
        atom_vf = self.atom_vf(h)  # (B, N, num_elements) - logits flow
        coord_vf = self.coord_vf(h)  # (B, N, 3)
        
        # Lattice vector field (aggregate over atoms)
        lattice_feat = h.mean(dim=1)  # (B, hidden_dim)
        lattice_vf = self.lattice_vf(lattice_feat)  # (B, 6)
        
        return {
            "atom_types": atom_vf,  # d/dt of logits
            "frac_coords": coord_vf,
            "lattice": lattice_vf
        }
    
    def sample_conditional_path(
        self,
        x0: Dict[str, torch.Tensor],
        x1: Dict[str, torch.Tensor],
        t: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Sample from conditional probability path.
        
        Args:
            x0: Source distribution samples (noise)
            x1: Target distribution samples (data)
            t: Time in [0, 1]
            
        Returns:
            xt: Sample at time t
            ut: Conditional vector field
        """
        B = t.shape[0]
        t_expanded = t.view(B, 1, 1)
        
        # Linear interpolation for continuous variables
        frac_coords_t = (1 - t_expanded) * x0["frac_coords"] + t_expanded * x1["frac_coords"]
        lattice_t = (1 - t.view(B, 1)) * x0["lattice"] + t.view(B, 1) * x1["lattice"]
        
        # Conditional vector fields
        ut_coords = x1["frac_coords"] - x0["frac_coords"]
        ut_lattice = x1["lattice"] - x0["lattice"]
        
        # For atom types (discrete), use soft interpolation
        atom_t = (1 - t_expanded) * x0["atom_types"] + t_expanded * x1["atom_types"]
        ut_atom = x1["atom_types"] - x0["atom_types"]
        
        xt = {
            "atom_types": atom_t,
            "frac_coords": frac_coords_t,
            "lattice": lattice_t
        }
        
        ut = {
            "atom_types": ut_atom,
            "frac_coords": ut_coords,
            "lattice": ut_lattice
        }
        
        return xt, ut
    
    def compute_loss(
        self,
        x1: Dict[str, torch.Tensor],
        properties: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            x1: Target samples (data)
            properties: Optional conditioning
            
        Returns:
            Loss value
        """
        B = x1["atom_types"].shape[0]
        
        # Sample time
        t = torch.rand(B, device=x1["atom_types"].device)
        
        # Sample noise (source distribution)
        N = x1["atom_types"].shape[1]
        x0 = {
            "atom_types": torch.randn(B, N, self.config.num_elements, device=x1["atom_types"].device),
            "frac_coords": torch.rand(B, N, 3, device=x1["frac_coords"].device),
            "lattice": torch.randn(B, 6, device=x1["lattice"].device)
        }
        
        # Sample conditional path
        xt, ut = self.sample_conditional_path(x0, x1, t)
        
        # Predict vector field
        vt = self.forward(
            xt["atom_types"],
            xt["frac_coords"],
            xt["lattice"],
            t,
            properties
        )
        
        # Compute loss
        loss_coords = F.mse_loss(vt["frac_coords"], ut["frac_coords"])
        loss_lattice = F.mse_loss(vt["lattice"], ut["lattice"])
        loss_atom = F.mse_loss(vt["atom_types"], ut["atom_types"])
        
        return loss_coords + loss_lattice + loss_atom
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        num_atoms: int,
        properties: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples using ODE solver.
        
        Args:
            batch_size: Number of samples
            num_atoms: Number of atoms per sample
            properties: Target properties (optional)
            num_steps: Number of integration steps
            
        Returns:
            Generated structures
        """
        if num_steps is None:
            num_steps = self.config.num_steps
        
        device = next(self.parameters()).device
        
        # Initialize from noise
        x = {
            "atom_types": torch.randn(batch_size, num_atoms, self.config.num_elements, device=device),
            "frac_coords": torch.rand(batch_size, num_atoms, 3, device=device),
            "lattice": torch.randn(batch_size, 6, device=device)
        }
        
        # Time steps
        dt = 1.0 / num_steps
        
        # Euler integration
        for i in range(num_steps):
            t = torch.ones(batch_size, device=device) * (i * dt)
            
            # Predict vector field
            vt = self.forward(
                x["atom_types"],
                x["frac_coords"],
                x["lattice"],
                t,
                properties
            )
            
            # Update
            x["frac_coords"] = x["frac_coords"] + dt * vt["frac_coords"]
            x["lattice"] = x["lattice"] + dt * vt["lattice"]
            x["atom_types"] = x["atom_types"] + dt * vt["atom_types"]
        
        # Convert atom logits to indices
        atom_types = torch.argmax(x["atom_types"], dim=-1)
        
        # Normalize fractional coordinates to [0, 1]
        x["frac_coords"] = x["frac_coords"] % 1.0
        
        return {
            "atom_types": atom_types,
            "frac_coords": x["frac_coords"],
            "lattice": x["lattice"]
        }


class CrystalFlow(nn.Module):
    """
    CrystalFlow: Flow-based generative model for crystalline materials.
    Simplified version for dftlammps integration.
    """
    
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        self.flow_matcher = RiemannianFlowMatcher(config)
        
    def forward(self, *args, **kwargs):
        return self.flow_matcher(*args, **kwargs)
    
    def compute_loss(self, *args, **kwargs):
        return self.flow_matcher.compute_loss(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.flow_matcher.generate(*args, **kwargs)
