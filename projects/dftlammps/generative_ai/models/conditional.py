"""
Conditional Diffusion for Property-Guided Generation
====================================================

Implementation of conditional diffusion/flow models for inverse design.
Supports:
- Classifier-free guidance (CFG)
- Property conditioning
- Multi-objective optimization

References:
- Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS 2021 Workshop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import math


@dataclass
class ConditionalConfig:
    """Configuration for conditional diffusion."""
    base_model: str = "crystal_dit"  # or "flow_matching"
    
    # Conditioning
    num_properties: int = 5
    property_names: List[str] = None
    
    # Classifier-free guidance
    use_cfg: bool = True
    cfg_dropout: float = 0.1
    guidance_scale: float = 2.0
    
    # Multi-objective
    use_mo: bool = False
    mo_weights: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.property_names is None:
            self.property_names = ["band_gap", "formation_energy", "bulk_modulus", "shear_modulus", "energy_per_atom"]


class ConditionalDiffusion(nn.Module):
    """
    Conditional Diffusion Model with Classifier-Free Guidance.
    
    Wraps a base diffusion model to enable property-guided generation.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: ConditionalConfig
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Property encoder
        if hasattr(base_model, 'config'):
            latent_dim = base_model.config.latent_dim
        else:
            latent_dim = 256
        
        self.property_encoder = nn.Sequential(
            nn.Linear(config.num_properties, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Property prediction head (for auxiliary loss)
        if hasattr(base_model, 'config'):
            hidden_dim = base_model.config.latent_dim
        else:
            hidden_dim = 256
        
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.num_properties)
        )
        
    def forward(
        self,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        t: torch.Tensor,
        properties: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cfg: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional classifier-free guidance.
        
        Args:
            atom_types: (B, N)
            frac_coords: (B, N, 3)
            lattice: (B, 6)
            t: (B,)
            properties: (B, num_properties)
            mask: (B, N)
            use_cfg: Whether to use classifier-free guidance
            
        Returns:
            Predictions with CFG applied if enabled
        """
        if not use_cfg or not self.config.use_cfg or properties is None:
            # Standard conditional forward
            return self.base_model(
                atom_types, frac_coords, lattice, t,
                properties=properties, mask=mask
            )
        
        # Classifier-free guidance
        # Conditional prediction
        cond_pred = self.base_model(
            atom_types, frac_coords, lattice, t,
            properties=properties, mask=mask
        )
        
        # Unconditional prediction (replace properties with zeros/NaN)
        uncond_properties = torch.full_like(properties, float('nan'))
        uncond_pred = self.base_model(
            atom_types, frac_coords, lattice, t,
            properties=uncond_properties, mask=mask
        )
        
        # Apply guidance
        scale = self.config.guidance_scale
        guided_pred = {
            key: uncond_pred[key] + scale * (cond_pred[key] - uncond_pred[key])
            for key in cond_pred.keys()
        }
        
        return guided_pred
    
    def compute_loss(
        self,
        x1: Dict[str, torch.Tensor],
        properties: torch.Tensor,
        noise: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute conditional training loss.
        
        Args:
            x1: Target data
            properties: Target properties
            noise: Optional pre-generated noise
            
        Returns:
            Dictionary of losses
        """
        B = x1["atom_types"].shape[0]
        device = x1["atom_types"].device
        
        # Sample timestep
        t = torch.randint(0, 1000, (B,), device=device).float() / 1000
        
        # Generate noise if not provided
        if noise is None:
            noise = {
                "atom_types": torch.randn_like(x1["atom_types"].float()),
                "frac_coords": torch.randn_like(x1["frac_coords"]),
                "lattice": torch.randn_like(x1["lattice"])
            }
        
        # Noisy input
        x0 = {
            "atom_types": x1["atom_types"].float() + noise["atom_types"],
            "frac_coords": x1["frac_coords"] + noise["frac_coords"],
            "lattice": x1["lattice"] + noise["lattice"]
        }
        
        # Classifier-free guidance training: randomly drop conditioning
        if self.training and self.config.use_cfg:
            mask = torch.rand(B, device=device) > self.config.cfg_dropout
            properties_masked = properties.clone()
            properties_masked[~mask] = float('nan')
        else:
            properties_masked = properties
        
        # Forward
        pred = self.forward(
            x0["atom_types"],
            x0["frac_coords"],
            x0["lattice"],
            t,
            properties=properties_masked
        )
        
        # Diffusion loss
        loss_diffusion = F.mse_loss(pred["frac_coords"], x1["frac_coords"])
        
        # Property prediction loss (auxiliary)
        # Aggregate features and predict properties
        if hasattr(self.base_model, 'final_norm'):
            # Use base model's features
            with torch.no_grad():
                feat = self._extract_features(x0, t, properties_masked)
        else:
            feat = None
        
        if feat is not None:
            pred_props = self.property_predictor(feat)
            loss_property = F.mse_loss(pred_props, properties)
        else:
            loss_property = torch.tensor(0.0, device=device)
        
        return {
            "loss": loss_diffusion + 0.1 * loss_property,
            "loss_diffusion": loss_diffusion,
            "loss_property": loss_property
        }
    
    def _extract_features(self, x, t, properties):
        """Extract features from base model."""
        # This is a placeholder - actual implementation depends on base model
        return torch.randn(x["atom_types"].shape[0], 256, device=x["atom_types"].device)
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        num_atoms: int,
        properties: torch.Tensor,
        num_steps: int = 100,
        guidance_scale: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples conditioned on properties.
        
        Args:
            batch_size: Number of samples
            num_atoms: Number of atoms
            properties: Target properties (B, num_properties)
            num_steps: Number of diffusion steps
            guidance_scale: CFG scale (overrides config)
            
        Returns:
            Generated structures
        """
        device = next(self.parameters()).device
        
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        
        # Initialize from noise
        x = {
            "atom_types": torch.randn(batch_size, num_atoms, device=device),
            "frac_coords": torch.randn(batch_size, num_atoms, 3, device=device),
            "lattice": torch.randn(batch_size, 6, device=device)
        }
        
        # DDPM sampling with CFG
        for i in reversed(range(num_steps)):
            t = torch.ones(batch_size, device=device) * (i / num_steps)
            
            # Conditional prediction
            pred = self.forward(
                x["atom_types"],
                x["frac_coords"],
                x["lattice"],
                t,
                properties=properties,
                use_cfg=True
            )
            
            # Update (simplified DDPM step)
            alpha = 1 - (i / num_steps) * 0.02
            x["frac_coords"] = (x["frac_coords"] - pred["frac_coords"]) / math.sqrt(alpha)
            x["lattice"] = (x["lattice"] - pred["lattice"]) / math.sqrt(alpha)
        
        # Convert to final format
        x["atom_types"] = torch.argmax(x["atom_types"], dim=-1) if x["atom_types"].dim() == 3 else x["atom_types"]
        x["frac_coords"] = x["frac_coords"] % 1.0
        
        return x
    
    def optimize_structure(
        self,
        initial_structure: Dict[str, torch.Tensor],
        target_properties: torch.Tensor,
        num_steps: int = 50,
        lr: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Optimize a structure towards target properties using gradient descent.
        
        Args:
            initial_structure: Starting structure
            target_properties: Target properties
            num_steps: Optimization steps
            lr: Learning rate
            
        Returns:
            Optimized structure
        """
        # Make parameters differentiable
        frac_coords = initial_structure["frac_coords"].clone().requires_grad_(True)
        lattice = initial_structure["lattice"].clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([frac_coords, lattice], lr=lr)
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            
            # Predict properties from current structure
            t = torch.zeros(frac_coords.shape[0], device=frac_coords.device)
            pred = self.forward(
                initial_structure["atom_types"],
                frac_coords,
                lattice,
                t,
                properties=target_properties
            )
            
            # Extract features and predict properties
            # Simplified - actual implementation would use a proper property predictor
            pred_props = torch.randn_like(target_properties)  # Placeholder
            
            loss = F.mse_loss(pred_props, target_properties)
            loss.backward()
            optimizer.step()
            
            # Project to valid domain
            with torch.no_grad():
                frac_coords.data = frac_coords.data % 1.0
        
        return {
            "atom_types": initial_structure["atom_types"],
            "frac_coords": frac_coords.detach(),
            "lattice": lattice.detach()
        }


class MultiObjectiveDiffusion(ConditionalDiffusion):
    """
    Multi-objective conditional diffusion for Pareto-optimal generation.
    """
    
    def __init__(self, base_model: nn.Module, config: ConditionalConfig):
        super().__init__(base_model, config)
        self.config = config
        
        if config.mo_weights is None:
            self.register_buffer('mo_weights', torch.ones(config.num_properties))
        else:
            self.register_buffer('mo_weights', torch.tensor(config.mo_weights))
    
    def generate_pareto_frontier(
        self,
        batch_size: int,
        num_atoms: int,
        property_ranges: Dict[str, Tuple[float, float]],
        num_samples: int = 100
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate Pareto frontier for multi-objective optimization.
        
        Args:
            batch_size: Batch size per generation
            num_atoms: Number of atoms
            property_ranges: Ranges for each property
            num_samples: Total number of samples
            
        Returns:
            List of structures with their properties
        """
        all_samples = []
        
        # Sample different weight combinations
        num_batches = num_samples // batch_size
        
        for _ in range(num_batches):
            # Random weight vector
            weights = torch.rand(self.config.num_properties)
            weights = weights / weights.sum()
            
            # Target properties (midpoint of ranges)
            target = torch.tensor([
                (property_ranges[p][0] + property_ranges[p][1]) / 2
                for p in self.config.property_names
            ])
            
            # Generate
            samples = self.generate(batch_size, num_atoms, target.unsqueeze(0))
            all_samples.append(samples)
        
        return all_samples
