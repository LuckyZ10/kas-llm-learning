"""
Sampling Utilities
==================

Sampling algorithms for generative models:
- DDPM/DDIM sampling for diffusion models
- ODE solvers for flow matching
- Multi-step sampling for consistency models
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Callable, List
import numpy as np
from tqdm import tqdm


class DiffusionSampler:
    """
    Sampler for diffusion models.
    
    Supports DDPM, DDIM, and classifier-free guidance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 1000,
        scheduler: str = "ddpm",
        guidance_scale: float = 1.0
    ):
        self.model = model
        self.num_steps = num_steps
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
        
        # Precompute noise schedule
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
    def _get_beta_schedule(self) -> torch.Tensor:
        """Get beta schedule."""
        if self.scheduler == "linear":
            return torch.linspace(1e-4, 0.02, self.num_steps)
        elif self.scheduler == "cosine":
            s = 0.008
            steps = self.num_steps + 1
            x = torch.linspace(0, self.num_steps, steps)
            alphas_cumprod = torch.cos(((x / self.num_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_atoms: int,
        properties: Optional[torch.Tensor] = None,
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from the diffusion model.
        
        Args:
            batch_size: Number of samples
            num_atoms: Number of atoms per sample
            properties: Conditioning properties
            device: Device to use
            
        Returns:
            Generated structures
        """
        # Initialize from noise
        shape_coords = (batch_size, num_atoms, 3)
        shape_lattice = (batch_size, 6)
        
        x = {
            "frac_coords": torch.randn(shape_coords, device=device),
            "lattice": torch.randn(shape_lattice, device=device),
            "atom_types": torch.randn(batch_size, num_atoms, 100, device=device)  # Soft
        }
        
        # Reverse diffusion
        for i in tqdm(reversed(range(self.num_steps)), desc="Sampling"):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, properties)
        
        # Convert to discrete
        x["atom_types"] = torch.argmax(x["atom_types"], dim=-1)
        x["frac_coords"] = x["frac_coords"] % 1.0
        
        return x
    
    def p_sample(
        self,
        x: Dict[str, torch.Tensor],
        t: torch.Tensor,
        properties: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Single denoising step."""
        B = t.shape[0]
        
        # Predict noise
        model_output = self.model(
            x["atom_types"],
            x["frac_coords"],
            x["lattice"],
            t.float() / self.num_steps,
            properties=properties
        )
        
        # DDPM sampling step
        alpha_t = self.alphas[t].view(B, 1, 1).to(x["frac_coords"].device)
        alpha_cumprod_t = self.alphas_cumprod[t].view(B, 1, 1).to(x["frac_coords"].device)
        beta_t = self.betas[t].view(B, 1, 1).to(x["frac_coords"].device)
        
        # Predict x_0
        pred_coords = (x["frac_coords"] - torch.sqrt(1 - alpha_cumprod_t) * model_output["frac_coords"]) / torch.sqrt(alpha_cumprod_t)
        pred_lattice = (x["lattice"] - torch.sqrt(1 - alpha_cumprod_t.squeeze(-1)) * model_output["lattice"]) / torch.sqrt(alpha_cumprod_t.squeeze(-1))
        
        # Compute x_{t-1}
        alpha_cumprod_prev = self.alphas_cumprod_prev[t].view(B, 1, 1).to(x["frac_coords"].device)
        
        coef1 = torch.sqrt(alpha_cumprod_prev) * beta_t / (1 - alpha_cumprod_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
        
        mean_coords = coef1 * pred_coords + coef2 * x["frac_coords"]
        mean_lattice = coef1.squeeze(-1) * pred_lattice + coef2.squeeze(-1) * x["lattice"]
        
        if t[0] > 0:
            noise_coords = torch.randn_like(x["frac_coords"])
            noise_lattice = torch.randn_like(x["lattice"])
            
            variance = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * beta_t
            std = torch.sqrt(variance)
            
            x["frac_coords"] = mean_coords + std * noise_coords
            x["lattice"] = mean_lattice + std.squeeze(-1) * noise_lattice
        else:
            x["frac_coords"] = mean_coords
            x["lattice"] = mean_lattice
        
        return x


class FlowSampler:
    """
    Sampler for flow matching models.
    
    Uses ODE solvers to integrate the flow.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50,
        method: str = "euler"
    ):
        self.model = model
        self.num_steps = num_steps
        self.method = method
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_atoms: int,
        properties: Optional[torch.Tensor] = None,
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """
        Sample using ODE integration.
        """
        # Initialize from noise
        x = {
            "atom_types": torch.randn(batch_size, num_atoms, 100, device=device),
            "frac_coords": torch.randn(batch_size, num_atoms, 3, device=device),
            "lattice": torch.randn(batch_size, 6, device=device)
        }
        
        if self.method == "euler":
            x = self._euler_integrate(x, properties)
        elif self.method == "rk4":
            x = self._rk4_integrate(x, properties)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Convert to discrete
        x["atom_types"] = torch.argmax(x["atom_types"], dim=-1)
        x["frac_coords"] = x["frac_coords"] % 1.0
        
        return x
    
    def _euler_integrate(
        self,
        x: Dict[str, torch.Tensor],
        properties: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Euler integration."""
        dt = 1.0 / self.num_steps
        B = x["frac_coords"].shape[0]
        
        for i in range(self.num_steps):
            t = torch.ones(B, device=x["frac_coords"].device) * (i * dt)
            
            # Get vector field
            vf = self.model(
                x["atom_types"],
                x["frac_coords"],
                x["lattice"],
                t,
                properties
            )
            
            # Update
            x["frac_coords"] = x["frac_coords"] + dt * vf["frac_coords"]
            x["lattice"] = x["lattice"] + dt * vf["lattice"]
            x["atom_types"] = x["atom_types"] + dt * vf["atom_types"]
        
        return x
    
    def _rk4_integrate(
        self,
        x: Dict[str, torch.Tensor],
        properties: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Runge-Kutta 4 integration."""
        dt = 1.0 / self.num_steps
        B = x["frac_coords"].shape[0]
        
        for i in range(self.num_steps):
            t = torch.ones(B, device=x["frac_coords"].device) * (i * dt)
            
            # k1
            vf1 = self.model(
                x["atom_types"],
                x["frac_coords"],
                x["lattice"],
                t,
                properties
            )
            
            # k2
            x2 = {
                "atom_types": x["atom_types"] + 0.5 * dt * vf1["atom_types"],
                "frac_coords": x["frac_coords"] + 0.5 * dt * vf1["frac_coords"],
                "lattice": x["lattice"] + 0.5 * dt * vf1["lattice"]
            }
            vf2 = self.model(
                x2["atom_types"],
                x2["frac_coords"],
                x2["lattice"],
                t + 0.5 * dt,
                properties
            )
            
            # k3
            x3 = {
                "atom_types": x["atom_types"] + 0.5 * dt * vf2["atom_types"],
                "frac_coords": x["frac_coords"] + 0.5 * dt * vf2["frac_coords"],
                "lattice": x["lattice"] + 0.5 * dt * vf2["lattice"]
            }
            vf3 = self.model(
                x3["atom_types"],
                x3["frac_coords"],
                x3["lattice"],
                t + 0.5 * dt,
                properties
            )
            
            # k4
            x4 = {
                "atom_types": x["atom_types"] + dt * vf3["atom_types"],
                "frac_coords": x["frac_coords"] + dt * vf3["frac_coords"],
                "lattice": x["lattice"] + dt * vf3["lattice"]
            }
            vf4 = self.model(
                x4["atom_types"],
                x4["frac_coords"],
                x4["lattice"],
                t + dt,
                properties
            )
            
            # Update
            x["atom_types"] = x["atom_types"] + (dt / 6) * (vf1["atom_types"] + 2*vf2["atom_types"] + 2*vf3["atom_types"] + vf4["atom_types"])
            x["frac_coords"] = x["frac_coords"] + (dt / 6) * (vf1["frac_coords"] + 2*vf2["frac_coords"] + 2*vf3["frac_coords"] + vf4["frac_coords"])
            x["lattice"] = x["lattice"] + (dt / 6) * (vf1["lattice"] + 2*vf2["lattice"] + 2*vf3["lattice"] + vf4["lattice"])
        
        return x


class ConsistencySampler:
    """
    Sampler for consistency models.
    
    Supports single-step and multi-step sampling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 1,
        sigma_max: float = 10.0
    ):
        self.model = model
        self.num_steps = num_steps
        self.sigma_max = sigma_max
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_atoms: int,
        properties: Optional[torch.Tensor] = None,
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """Sample from consistency model."""
        # Initialize from noise
        x = {
            "atom_types": torch.randint(0, 100, (batch_size, num_atoms), device=device),
            "frac_coords": torch.randn(batch_size, num_atoms, 3, device=device) * self.sigma_max,
            "lattice": torch.randn(batch_size, 6, device=device) * self.sigma_max
        }
        
        if self.num_steps == 1:
            # Single-step generation
            sigma = torch.ones(batch_size, device=device) * self.sigma_max
            
            if hasattr(self.model, 'student'):
                return self.model.student(
                    x["atom_types"],
                    x["frac_coords"],
                    x["lattice"],
                    sigma,
                    properties
                )
            else:
                return self.model(
                    x["atom_types"],
                    x["frac_coords"],
                    x["lattice"],
                    sigma,
                    properties
                )
        else:
            # Multi-step generation
            sigmas = torch.exp(torch.linspace(
                np.log(self.sigma_max),
                np.log(1e-4),
                self.num_steps + 1
            )).to(device)
            
            for i in range(self.num_steps):
                sigma = torch.full((batch_size,), sigmas[i], device=device)
                
                if hasattr(self.model, 'student'):
                    x = self.model.student(
                        x["atom_types"],
                        x["frac_coords"],
                        x["lattice"],
                        sigma,
                        properties
                    )
                else:
                    x = self.model(
                        x["atom_types"],
                        x["frac_coords"],
                        x["lattice"],
                        sigma,
                        properties
                    )
                
                x["atom_types"] = x["atom_types"].argmax(dim=-1) if x["atom_types"].dim() == 3 else x["atom_types"]
            
            return x
