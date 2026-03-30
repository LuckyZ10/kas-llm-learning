"""
Crystal Diffusion Transformer (CrystalDiT)
==========================================

Implementation of Diffusion Transformer for crystal structure generation.
Based on DiT (Peebles & Xie 2023) and ADiT (ICML 2025) architectures.

Key features:
- Unified latent representation for molecules and crystals
- Transformer-based denoising with minimal inductive bias
- Scalable architecture (tested up to 500M parameters)
- SE(3) equivariance through data augmentation

References:
- Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
- Joshi et al., "All-atom Diffusion Transformers", ICML 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class CrystalDiTConfig:
    """Configuration for CrystalDiT model."""
    # Model architecture
    latent_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_atoms: int = 100
    
    # Atom types
    num_elements: int = 100  # Up to Fm (fermium)
    
    # Lattice parameters (6 independent for symmetric matrix)
    lattice_dim: int = 6
    
    # Diffusion parameters
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Conditioning
    use_conditioning: bool = True
    property_dim: int = 32
    num_properties: int = 5  # band_gap, formation_energy, etc.


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps (B,)
        Returns:
            Embeddings (B, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PropertyEmbedding(nn.Module):
    """Property conditioning embeddings."""
    
    def __init__(self, num_properties: int, property_dim: int, output_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, property_dim),
                nn.SiLU(),
                nn.Linear(property_dim, output_dim)
            ) for _ in range(num_properties)
        ])
        self.num_properties = num_properties
        
    def forward(self, properties: torch.Tensor) -> torch.Tensor:
        """
        Args:
            properties: (B, num_properties)
        Returns:
            Embeddings (B, output_dim)
        """
        # properties: [B, num_properties]
        # Handle NaN values (unconditioned)
        properties = torch.where(
            torch.isnan(properties),
            torch.zeros_like(properties),
            properties
        )
        
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            prop_i = properties[:, i:i+1]
            emb = emb_layer(prop_i)
            embeddings.append(emb)
        
        # Sum embeddings
        return sum(embeddings)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with adaptive layer norm.
    Similar to DiT but adapted for atomic systems.
    """
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, N, D)
            c: Conditioning (B, D) - timestep + property embeddings
        Returns:
            Output features (B, N, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Self-attention with adaptive norm
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with adaptive norm
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class CrystalDiT(nn.Module):
    """
    Crystal Diffusion Transformer for generating crystal structures.
    
    Generates:
    - Atomic types (discrete, categorical)
    - Fractional coordinates (continuous, periodic)
    - Lattice parameters (continuous)
    """
    
    def __init__(self, config: CrystalDiTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.atom_type_embed = nn.Embedding(config.num_elements + 1, config.latent_dim)
        self.coord_embed = nn.Linear(3, config.latent_dim)
        self.lattice_embed = nn.Linear(config.lattice_dim, config.latent_dim)
        
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_atoms + 1, config.latent_dim) * 0.02)
        
        # Timestep embedding
        self.t_embed = TimestepEmbedding(config.latent_dim)
        
        # Property conditioning
        if config.use_conditioning:
            self.p_embed = PropertyEmbedding(
                config.num_properties,
                config.property_dim,
                config.latent_dim
            )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                config.latent_dim,
                config.num_heads,
                config.mlp_ratio,
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.latent_dim, elementwise_affine=False, eps=1e-6)
        
        # Output heads
        self.atom_type_head = nn.Linear(config.latent_dim, config.num_elements)
        self.coord_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, 3)
        )
        self.lattice_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.lattice_dim)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers for stability
        nn.init.constant_(self.atom_type_head.weight, 0)
        nn.init.constant_(self.atom_type_head.bias, 0)
        nn.init.constant_(self.coord_head[-1].weight, 0)
        nn.init.constant_(self.coord_head[-1].bias, 0)
        
    def forward(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        t: torch.Tensor,
        properties: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass predicting noise/parameters.
        
        Args:
            atom_types: (B, N) Atomic type indices (noised)
            frac_coords: (B, N, 3) Fractional coordinates (noised)
            lattice: (B, 6) Lattice parameters (noised)
            t: (B,) Timesteps
            properties: (B, num_properties) Target properties (optional)
            mask: (B, N) Atom mask (True for valid atoms)
            
        Returns:
            Dictionary with predicted noise/parameters
        """
        B, N = atom_types.shape
        
        # Embeddings
        atom_emb = self.atom_type_embed(atom_types)  # (B, N, D)
        coord_emb = self.coord_embed(frac_coords)    # (B, N, D)
        
        # Lattice embedding (broadcast to all atoms)
        lattice_emb = self.lattice_embed(lattice).unsqueeze(1)  # (B, 1, D)
        
        # Combine embeddings
        x = atom_emb + coord_emb  # (B, N, D)
        
        # Add lattice as a global token
        x = torch.cat([lattice_emb, x], dim=1)  # (B, N+1, D)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :N+1, :]
        
        # Conditioning
        t_emb = self.t_embed(t)  # (B, D)
        
        if self.config.use_conditioning and properties is not None:
            p_emb = self.p_embed(properties)  # (B, D)
            c = t_emb + p_emb
        else:
            c = t_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        x = self.final_norm(x)
        
        # Split global and atom features
        lattice_feat = x[:, 0, :]    # (B, D)
        atom_feats = x[:, 1:, :]     # (B, N, D)
        
        # Apply mask if provided
        if mask is not None:
            atom_feats = atom_feats * mask.unsqueeze(-1).float()
        
        # Predictions
        atom_type_pred = self.atom_type_head(atom_feats)  # (B, N, num_elements)
        coord_pred = self.coord_head(atom_feats)          # (B, N, 3)
        lattice_pred = self.lattice_head(lattice_feat)    # (B, 6)
        
        return {
            "atom_types": atom_type_pred,
            "frac_coords": coord_pred,
            "lattice": lattice_pred,
        }
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ADiT(nn.Module):
    """
    All-atom Diffusion Transformer (ADiT)
    Unified latent diffusion for molecules and materials.
    
    Key innovation: Shared latent space for periodic and non-periodic systems.
    """
    
    def __init__(self, config: CrystalDiTConfig):
        super().__init__()
        self.config = config
        
        # VAE encoder for unified latent representation
        self.encoder = nn.ModuleDict({
            'atom': nn.Sequential(
                nn.Embedding(config.num_elements + 1, config.latent_dim),
                nn.Linear(config.latent_dim, config.latent_dim),
                nn.SiLU(),
                nn.Linear(config.latent_dim, config.latent_dim)
            ),
            'coord': nn.Sequential(
                nn.Linear(3, config.latent_dim),
                nn.SiLU(),
                nn.Linear(config.latent_dim, config.latent_dim)
            ),
            'lattice': nn.Sequential(
                nn.Linear(config.lattice_dim, config.latent_dim),
                nn.SiLU(),
                nn.Linear(config.latent_dim, config.latent_dim)
            )
        })
        
        # Latent diffusion model (DiT in latent space)
        self.latent_dim = config.latent_dim
        self.t_embed = TimestepEmbedding(config.latent_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                config.latent_dim,
                config.num_heads,
                config.mlp_ratio,
                config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # VAE decoder
        self.decoder = nn.ModuleDict({
            'atom': nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim),
                nn.SiLU(),
                nn.Linear(config.latent_dim, config.num_elements)
            ),
            'coord': nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim),
                nn.SiLU(),
                nn.Linear(config.latent_dim, 3)
            ),
            'lattice': nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim),
                nn.SiLU(),
                nn.Linear(config.latent_dim, config.lattice_dim)
            )
        })
        
    def encode(self, atom_types, frac_coords, lattice):
        """Encode to latent space."""
        atom_latent = self.encoder['atom'](atom_types).mean(dim=1)  # (B, D)
        coord_latent = self.encoder['coord'](frac_coords).mean(dim=1)  # (B, D)
        lattice_latent = self.encoder['lattice'](lattice)  # (B, D)
        
        # Combine into unified latent
        latent = atom_latent + coord_latent + lattice_latent
        return latent
    
    def decode(self, latent, num_atoms):
        """Decode from latent space."""
        # Expand latent for each atom
        latent_expanded = latent.unsqueeze(1).expand(-1, num_atoms, -1)
        
        atom_pred = self.decoder['atom'](latent_expanded)
        coord_pred = self.decoder['coord'](latent_expanded)
        lattice_pred = self.decoder['lattice'](latent)
        
        return {
            "atom_types": atom_pred,
            "frac_coords": coord_pred,
            "lattice": lattice_pred
        }
    
    def forward(self, z, t, properties=None):
        """
        Latent diffusion forward pass.
        
        Args:
            z: Latent representation (B, D)
            t: Timestep (B,)
            properties: Conditioning properties (B, num_properties)
        """
        t_emb = self.t_embed(t)
        
        if properties is not None:
            # Simple property conditioning
            p_emb = nn.Linear(properties.shape[1], self.latent_dim).to(z.device)(properties)
            c = t_emb + p_emb
        else:
            c = t_emb
        
        # Expand for transformer
        z = z.unsqueeze(1)  # (B, 1, D)
        
        for block in self.blocks:
            z = block(z, c)
        
        return z.squeeze(1)


class DiffusionScheduler:
    """Noise scheduling for diffusion models."""
    
    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine"):
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        
        if schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        elif schedule == "linear":
            self.betas = self._linear_beta_schedule()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
    def _cosine_beta_schedule(self, s: float = 0.008):
        """Cosine schedule as in Improved DDPM."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _linear_beta_schedule(self, beta_start: float = 1e-4, beta_end: float = 0.02):
        """Linear schedule."""
        return torch.linspace(beta_start, beta_end, self.num_timesteps)
    
    def add_noise(self, x, t, noise=None):
        """Add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x)
        
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[t]).reshape(-1, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alphas_cumprod[t]).reshape(-1, 1, 1)
        
        return sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise
