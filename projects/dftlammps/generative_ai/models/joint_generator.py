"""
Joint Molecular-Crystal Generator
==================================

Unified generation of molecular and crystal structures.
Based on ADiT (All-atom Diffusion Transformer) architecture.

Key features:
- Single model for both molecules and crystals
- Shared latent space representation
- Seamless handling of periodic and non-periodic systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple
from dataclasses import dataclass
import math


@dataclass
class JointGeneratorConfig:
    """Configuration for joint molecular-crystal generator."""
    # Model architecture
    latent_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1
    
    # Atom handling
    num_elements: int = 100
    max_atoms: int = 200  # Larger for flexibility
    
    # Diffusion
    num_timesteps: int = 1000
    
    # Conditioning
    use_periodicity_indicator: bool = True
    num_properties: int = 5


class JointMolecularCrystalGenerator(nn.Module):
    """
    Unified generator for both molecules (non-periodic) and crystals (periodic).
    
    Uses a shared latent representation and handles periodicity as a condition.
    """
    
    def __init__(self, config: JointGeneratorConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.encoder = UnifiedEncoder(config)
        
        # Latent diffusion model
        self.latent_denoiser = LatentDenoiser(config)
        
        # Shared decoder
        self.decoder = UnifiedDecoder(config)
        
        # Periodicity classifier
        if config.use_periodicity_indicator:
            self.periodicity_predictor = nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim // 2),
                nn.SiLU(),
                nn.Linear(config.latent_dim // 2, 1),
                nn.Sigmoid()
            )
    
    def encode(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        lattice: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode structure to latent space.
        
        Args:
            atom_types: (B, N) Atomic type indices
            coords: (B, N, 3) Coordinates (Cartesian for molecules, fractional for crystals)
            lattice: (B, 6) Lattice parameters (None for molecules)
            
        Returns:
            Latent representation (B, latent_dim)
        """
        is_periodic = lattice is not None
        return self.encoder(atom_types, coords, lattice, is_periodic)
    
    def decode(
        self,
        latent: torch.Tensor,
        num_atoms: int,
        is_periodic: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Decode from latent space.
        
        Args:
            latent: (B, latent_dim)
            num_atoms: Number of atoms to generate
            is_periodic: Whether to generate crystal (periodic) or molecule
            
        Returns:
            Decoded structure
        """
        return self.decoder(latent, num_atoms, is_periodic)
    
    def forward(
        self,
        latent_noisy: torch.Tensor,
        t: torch.Tensor,
        num_atoms: int,
        properties: Optional[torch.Tensor] = None,
        force_periodic: Optional[bool] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Denoise latent representation.
        
        Args:
            latent_noisy: (B, latent_dim) Noised latent
            t: (B,) Timestep
            num_atoms: Number of atoms
            properties: (B, num_properties) Optional conditioning
            force_periodic: Force periodic/non-periodic generation
            
        Returns:
            Denoised latent and periodicity prediction
        """
        # Denoise
        latent_denoised = self.latent_denoiser(latent_noisy, t, properties)
        
        # Predict periodicity
        periodicity_pred = None
        if self.config.use_periodicity_indicator and force_periodic is None:
            periodicity_pred = self.periodicity_predictor(latent_denoised)
        
        return latent_denoised, periodicity_pred
    
    def generate(
        self,
        batch_size: int,
        num_atoms: int,
        properties: Optional[torch.Tensor] = None,
        force_periodic: Optional[bool] = None,
        num_steps: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Generate structures.
        
        Args:
            batch_size: Number of samples
            num_atoms: Number of atoms per sample
            properties: Target properties
            force_periodic: Force crystal (True) or molecule (False)
            num_steps: Number of diffusion steps
            
        Returns:
            Generated structures
        """
        device = next(self.parameters()).device
        
        # Initialize from noise
        latent = torch.randn(batch_size, self.config.latent_dim, device=device)
        
        # DDIM sampling
        for i in reversed(range(num_steps)):
            t = torch.ones(batch_size, device=device) * (i / num_steps)
            
            # Predict noise
            latent_denoised, _ = self.forward(
                latent, t, num_atoms, properties, force_periodic
            )
            
            # Update
            alpha = (i / num_steps) ** 2
            latent = torch.sqrt(alpha) * latent_denoised + torch.sqrt(1 - alpha) * torch.randn_like(latent)
        
        # Decode
        is_periodic = force_periodic if force_periodic is not None else True
        output = self.decode(latent, num_atoms, is_periodic)
        output["is_periodic"] = torch.tensor([is_periodic] * batch_size, device=device)
        
        return output
    
    def convert_molecule_to_crystal(
        self,
        molecule: Dict[str, torch.Tensor],
        target_lattice: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a molecule to a crystal by placing it in a unit cell.
        
        Args:
            molecule: Molecular structure
            target_lattice: Target lattice parameters
            
        Returns:
            Crystal structure
        """
        # Encode molecule
        latent = self.encode(
            molecule["atom_types"],
            molecule["coords"],
            lattice=None
        )
        
        # Decode as crystal with target lattice
        crystal = self.decoder.decode_as_crystal(latent, molecule["atom_types"].shape[1], target_lattice)
        
        return crystal
    
    def convert_crystal_to_molecule(
        self,
        crystal: Dict[str, torch.Tensor],
        extraction_box: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract a molecule from a crystal structure.
        
        Args:
            crystal: Crystal structure
            extraction_box: Box for extraction (if None, extract one unit cell)
            
        Returns:
            Molecular structure
        """
        # Convert fractional to Cartesian
        # Extract cluster
        # Return as molecule
        
        coords_cartesian = frac_to_cartesian(
            crystal["frac_coords"],
            crystal["lattice"]
        )
        
        return {
            "atom_types": crystal["atom_types"],
            "coords": coords_cartesian,
            "is_periodic": torch.tensor([False] * crystal["atom_types"].shape[0])
        }


class UnifiedEncoder(nn.Module):
    """Unified encoder for molecules and crystals."""
    
    def __init__(self, config: JointGeneratorConfig):
        super().__init__()
        self.config = config
        
        self.atom_embed = nn.Embedding(config.num_elements + 1, config.latent_dim)
        self.coord_embed = nn.Linear(3, config.latent_dim)
        self.lattice_embed = nn.Linear(6, config.latent_dim)
        
        self.periodicity_embed = nn.Embedding(2, config.latent_dim)  # 0: molecule, 1: crystal
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.latent_dim * 4),
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.num_layers // 2
        )
        
        self.to_latent = nn.Linear(config.latent_dim, config.latent_dim)
    
    def forward(
        self,
        atom_types: torch.Tensor,
        coords: torch.Tensor,
        lattice: Optional[torch.Tensor],
        is_periodic: bool
    ) -> torch.Tensor:
        """Encode to latent."""
        B, N = atom_types.shape
        
        h = self.atom_embed(atom_types) + self.coord_embed(coords)
        
        # Add periodicity indicator
        periodicity_idx = torch.tensor([1 if is_periodic else 0] * B, device=atom_types.device)
        h = h + self.periodicity_embed(periodicity_idx).unsqueeze(1)
        
        # Add lattice if crystal
        if lattice is not None:
            h = h + self.lattice_embed(lattice).unsqueeze(1)
        
        # Transform
        h = self.transformer(h)
        
        # Pool to latent
        h = h.mean(dim=1)
        latent = self.to_latent(h)
        
        return latent


class LatentDenoiser(nn.Module):
    """Denoising model in latent space."""
    
    def __init__(self, config: JointGeneratorConfig):
        super().__init__()
        self.config = config
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
        # Property embedding
        if config.num_properties > 0:
            self.prop_embed = nn.Sequential(
                nn.Linear(config.num_properties, config.latent_dim),
                nn.SiLU(),
                nn.Linear(config.latent_dim, config.latent_dim)
            )
        
        # Denoising MLP
        layers = []
        for i in range(4):
            layers.extend([
                nn.Linear(config.latent_dim, config.latent_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout)
            ])
        layers.append(nn.Linear(config.latent_dim, config.latent_dim))
        
        self.denoiser = nn.Sequential(*layers)
    
    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        properties: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Denoise latent."""
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        h = latent + t_emb
        
        if properties is not None and hasattr(self, 'prop_embed'):
            p_emb = self.prop_embed(properties)
            h = h + p_emb
        
        return self.denoiser(h)


class UnifiedDecoder(nn.Module):
    """Unified decoder for molecules and crystals."""
    
    def __init__(self, config: JointGeneratorConfig):
        super().__init__()
        self.config = config
        
        self.from_latent = nn.Linear(config.latent_dim, config.latent_dim)
        
        self.atom_decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.num_elements)
        )
        
        self.coord_decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, 3)
        )
        
        self.lattice_decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, 6)
        )
    
    def forward(
        self,
        latent: torch.Tensor,
        num_atoms: int,
        is_periodic: bool
    ) -> Dict[str, torch.Tensor]:
        """Decode from latent."""
        B = latent.shape[0]
        
        # Expand latent
        h = self.from_latent(latent).unsqueeze(1).expand(-1, num_atoms, -1)
        
        # Decode
        atom_logits = self.atom_decoder(h)
        coords = self.coord_decoder(h)
        
        # Lattice only for crystals
        if is_periodic:
            lattice = self.lattice_decoder(latent)
            return {
                "atom_types": torch.argmax(atom_logits, dim=-1),
                "atom_logits": atom_logits,
                "frac_coords": torch.sigmoid(coords),  # Fractional coords in [0, 1]
                "lattice": lattice,
                "is_periodic": True
            }
        else:
            return {
                "atom_types": torch.argmax(atom_logits, dim=-1),
                "atom_logits": atom_logits,
                "coords": coords,  # Cartesian coords
                "is_periodic": False
            }
    
    def decode_as_crystal(
        self,
        latent: torch.Tensor,
        num_atoms: int,
        target_lattice: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Decode as crystal with specified lattice."""
        output = self.forward(latent, num_atoms, is_periodic=True)
        output["lattice"] = target_lattice
        return output


def frac_to_cartesian(frac_coords: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Convert fractional to Cartesian coordinates."""
    # lattice: (B, 6) - a, b, c, alpha, beta, gamma
    # Convert to lattice vectors
    # This is simplified - actual implementation would use full lattice matrix
    a = lattice[:, 0:1]
    b = lattice[:, 1:2]
    c = lattice[:, 2:3]
    
    # Simple orthogonal case
    cartesian = frac_coords * torch.stack([a, b, c], dim=-1)
    return cartesian
