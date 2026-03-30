"""
Consistency Models for Fast Crystal Generation
===============================================

Implementation of Consistency Models for one-step/few-step generation.
Based on Song et al., "Consistency Models", ICML 2023.

Key features:
- Single-step generation capability
- Progressive distillation from diffusion/flow models
- Flexible number of sampling steps (1-50)

References:
- Song et al., "Consistency Models", ICML 2023
- Dou et al., "Theory of Consistency Diffusion Models", ICML 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class ConsistencyConfig:
    """Configuration for Consistency Model."""
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_atoms: int = 100
    num_elements: int = 100
    
    # Consistency-specific
    sigma_min: float = 1e-4
    sigma_max: float = 10.0
    sigma_data: float = 1.0
    
    # Training
    mu: float = 0.999  # EMA decay rate
    learning_rate: float = 1e-4
    
    # Sampling
    num_discretization_steps: int = 50


class ConsistencyCrystalModel(nn.Module):
    """
    Consistency Model for crystal structure generation.
    
    Maps any point on the probability flow ODE trajectory to the origin (clean data).
    """
    
    def __init__(self, config: ConsistencyConfig):
        super().__init__()
        self.config = config
        
        # Student network
        self.student = ConsistencyBackbone(config)
        
        # Teacher network (EMA of student)
        self.teacher = ConsistencyBackbone(config)
        self.teacher.requires_grad_(False)
        
        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())
        
    def forward(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        sigma: torch.Tensor,
        properties: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict consistency function f(x, sigma).
        
        Args:
            atom_types: (B, N) or (B, N, num_elements)
            frac_coords: (B, N, 3)
            lattice: (B, 6)
            sigma: (B,) Noise level
            properties: (B, num_props)
            
        Returns:
            Predicted clean data
        """
        return self.student(atom_types, frac_coords, lattice, sigma, properties)
    
    def update_teacher(self):
        """Update teacher network with EMA of student."""
        mu = self.config.mu
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.student.parameters(),
                self.teacher.parameters()
            ):
                teacher_param.data.mul_(mu).add_(student_param.data, alpha=1 - mu)
    
    def compute_loss(
        self,
        x: Dict[str, torch.Tensor],
        properties: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute consistency distillation loss.
        
        Args:
            x: Clean data
            properties: Optional conditioning
            
        Returns:
            Loss value
        """
        B = x["atom_types"].shape[0]
        device = x["atom_types"].device
        
        # Sample noise level
        sigma = torch.exp(
            torch.rand(B, device=device) * 
            (math.log(self.config.sigma_max) - math.log(self.config.sigma_min)) +
            math.log(self.config.sigma_min)
        )
        
        # Add noise to data
        noise_coords = torch.randn_like(x["frac_coords"])
        noise_lattice = torch.randn_like(x["lattice"])
        
        x_noisy = {
            "atom_types": x["atom_types"],  # Keep discrete for now
            "frac_coords": x["frac_coords"] + sigma.view(B, 1, 1) * noise_coords,
            "lattice": x["lattice"] + sigma.view(B, 1) * noise_lattice
        }
        
        # Student prediction
        f_student = self.student(
            x_noisy["atom_types"],
            x_noisy["frac_coords"],
            x_noisy["lattice"],
            sigma,
            properties
        )
        
        # Teacher prediction (with larger noise level)
        sigma_next = sigma * 2  # Simplified schedule
        
        with torch.no_grad():
            x_noisy_next = {
                "atom_types": x["atom_types"],
                "frac_coords": x["frac_coords"] + sigma_next.view(B, 1, 1) * noise_coords,
                "lattice": x["lattice"] + sigma_next.view(B, 1) * noise_lattice
            }
            
            f_teacher = self.teacher(
                x_noisy_next["atom_types"],
                x_noisy_next["frac_coords"],
                x_noisy_next["lattice"],
                sigma_next,
                properties
            )
        
        # Consistency loss
        loss_coords = F.mse_loss(f_student["frac_coords"], f_teacher["frac_coords"])
        loss_lattice = F.mse_loss(f_student["lattice"], f_teacher["lattice"])
        
        return loss_coords + loss_lattice
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        num_atoms: int,
        properties: Optional[torch.Tensor] = None,
        num_steps: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples. With consistency models, can use 1-50 steps.
        
        Args:
            batch_size: Number of samples
            num_atoms: Number of atoms per sample
            properties: Target properties
            num_steps: Number of refinement steps (1 for single-step)
            
        Returns:
            Generated structures
        """
        device = next(self.parameters()).device
        
        # Sample from prior (noise)
        x = {
            "atom_types": torch.randint(0, self.config.num_elements, (batch_size, num_atoms), device=device),
            "frac_coords": torch.randn(batch_size, num_atoms, 3, device=device) * self.config.sigma_max,
            "lattice": torch.randn(batch_size, 6, device=device) * self.config.sigma_max
        }
        
        if num_steps == 1:
            # Single-step generation
            sigma = torch.ones(batch_size, device=device) * self.config.sigma_max
            return self.student(
                x["atom_types"],
                x["frac_coords"],
                x["lattice"],
                sigma,
                properties
            )
        else:
            # Multi-step generation (like improved consistency)
            sigmas = torch.exp(torch.linspace(
                math.log(self.config.sigma_max),
                math.log(self.config.sigma_min),
                num_steps + 1
            )).to(device)
            
            for i in range(num_steps):
                sigma = torch.full((batch_size,), sigmas[i], device=device)
                x = self.student(
                    x["atom_types"],
                    x["frac_coords"],
                    x["lattice"],
                    sigma,
                    properties
                )
                x["atom_types"] = x["atom_types"].argmax(dim=-1) if x["atom_types"].dim() == 3 else x["atom_types"]
            
            return x


class ConsistencyBackbone(nn.Module):
    """Backbone network for consistency model."""
    
    def __init__(self, config: ConsistencyConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.atom_embed = nn.Embedding(config.num_elements + 1, config.hidden_dim)
        self.coord_embed = nn.Linear(3, config.hidden_dim)
        self.lattice_embed = nn.Linear(6, config.hidden_dim)
        
        # Sigma (noise level) embedding
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Property conditioning
        if hasattr(config, 'num_properties') and config.num_properties > 0:
            self.prop_embed = nn.Sequential(
                nn.Linear(config.num_properties, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
        else:
            self.prop_embed = None
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Skip connections
        self.skip_scale = nn.Parameter(torch.ones(config.num_layers))
        
        # Output heads
        self.atom_head = nn.Linear(config.hidden_dim, config.num_elements)
        self.coord_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3)
        )
        self.lattice_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 6)
        )
        
    def forward(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        sigma: torch.Tensor,
        properties: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict clean data from noisy input.
        """
        B, N = atom_types.shape if atom_types.dim() == 2 else atom_types.shape[:2]
        
        # Embeddings
        if atom_types.dim() == 2:
            h = self.atom_embed(atom_types)
        else:
            h = torch.matmul(atom_types, self.atom_embed.weight)
        
        h = h + self.coord_embed(frac_coords)
        
        # Add sigma embedding
        sigma_emb = self.sigma_embed(sigma.view(B, 1))
        h = h + sigma_emb.unsqueeze(1)
        
        # Add property conditioning
        if self.prop_embed is not None and properties is not None:
            prop_emb = self.prop_embed(properties)
            h = h + prop_emb.unsqueeze(1)
        
        # Transformer with skip connections
        h_input = h
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = h + self.skip_scale[i] * h_input
        
        # Predictions
        atom_logits = self.atom_head(h)
        coord_pred = self.coord_head(h)
        
        # Lattice from aggregated features
        h_agg = h.mean(dim=1)
        lattice_pred = self.lattice_head(h_agg)
        
        # c_skip connection (as in EDM)
        c_skip = sigma.view(B, 1, 1) ** 2 / (self.config.sigma_data ** 2 + sigma.view(B, 1, 1) ** 2)
        
        # Denormalize
        frac_coords_out = c_skip * frac_coords + (1 - c_skip) * coord_pred
        
        c_skip_lattice = sigma.view(B, 1) ** 2 / (self.config.sigma_data ** 2 + sigma.view(B, 1) ** 2)
        lattice_out = c_skip_lattice * lattice + (1 - c_skip_lattice) * lattice_pred
        
        return {
            "atom_types": atom_logits,
            "frac_coords": frac_coords_out,
            "lattice": lattice_out
        }
