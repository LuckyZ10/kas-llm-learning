"""
Consistency Model Trainer
=========================

Training infrastructure for consistency models.
Implements consistency distillation from a pretrained diffusion model
or consistency training from scratch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
from tqdm import tqdm
import copy


class ConsistencyTrainer:
    """
    Trainer for consistency models.
    
    Two training modes:
    1. Consistency Distillation: From pretrained diffusion model
    2. Consistency Training: From scratch
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        teacher_model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.teacher_model = teacher_model
        
        if teacher_model is not None:
            self.teacher_model = teacher_model.to(device)
            self.teacher_model.eval()
        
        self.config = {
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "max_epochs": 100,
            "save_every": 10,
            "eval_every": 5,
            "log_every": 100,
            "mu": 0.999,  # EMA decay
            "sigma_min": 1e-4,
            "sigma_max": 10.0,
            "sigma_data": 1.0,
            "mode": "distillation",  # or "training"
            "checkpoint_dir": "./checkpoints_consistency",
            **(config or {})
        }
        
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
        else:
            self.optimizer = optimizer
        
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def get_schedule_params(self, sigma: torch.Tensor):
        """Get c_skip, c_out, c_in, c_noise for preconditioning."""
        sigma_data = self.config["sigma_data"]
        
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_noise = 0.25 * torch.log(sigma)
        
        return c_skip, c_out, c_in, c_noise
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample noise levels (timesteps)."""
        # Uniform sampling in log-space
        rnd_uniform = torch.rand(batch_size, device=self.device)
        sigma = (
            self.config["sigma_max"] ** (1 - rnd_uniform) *
            self.config["sigma_min"] ** rnd_uniform
        )
        return sigma
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute consistency loss."""
        B = batch["atom_types"].shape[0]
        
        x = {
            "atom_types": batch["atom_types"],
            "frac_coords": batch["frac_coords"],
            "lattice": batch["lattice"]
        }
        
        properties = batch.get("properties")
        
        # Sample noise levels
        sigma = self.sample_timesteps(B)
        sigma_next = sigma * 2  # Next discretization step
        
        # Add noise
        noise_coords = torch.randn_like(x["frac_coords"])
        noise_lattice = torch.randn_like(x["lattice"])
        
        x_noisy = {
            "atom_types": x["atom_types"],
            "frac_coords": x["frac_coords"] + sigma.view(B, 1, 1) * noise_coords,
            "lattice": x["lattice"] + sigma.view(B, 1) * noise_lattice
        }
        
        # Student prediction
        if hasattr(self.model, 'student'):
            f_theta = self.model.student(
                x_noisy["atom_types"],
                x_noisy["frac_coords"],
                x_noisy["lattice"],
                sigma,
                properties
            )
        else:
            f_theta = self.model(
                x_noisy["atom_types"],
                x_noisy["frac_coords"],
                x_noisy["lattice"],
                sigma,
                properties
            )
        
        # Teacher prediction
        with torch.no_grad():
            x_noisy_next = {
                "atom_types": x["atom_types"],
                "frac_coords": x["frac_coords"] + sigma_next.view(B, 1, 1) * noise_coords,
                "lattice": x["lattice"] + sigma_next.view(B, 1) * noise_lattice
            }
            
            if self.teacher_model is not None:
                # Use pretrained teacher
                f_theta_minus = self.teacher_model(
                    x_noisy_next["atom_types"],
                    x_noisy_next["frac_coords"],
                    x_noisy_next["lattice"],
                    sigma_next,
                    properties
                )
            elif hasattr(self.model, 'teacher'):
                # Use EMA teacher
                f_theta_minus = self.model.teacher(
                    x_noisy_next["atom_types"],
                    x_noisy_next["frac_coords"],
                    x_noisy_next["lattice"],
                    sigma_next,
                    properties
                )
            else:
                # Self-consistency (predicting the same point)
                f_theta_minus = f_theta
        
        # Consistency loss
        loss_coords = nn.functional.mse_loss(
            f_theta["frac_coords"],
            f_theta_minus["frac_coords"]
        )
        loss_lattice = nn.functional.mse_loss(
            f_theta["lattice"],
            f_theta_minus["lattice"]
        )
        
        loss = loss_coords + loss_lattice
        
        return loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        if self.teacher_model is not None:
            self.teacher_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            loss = self.compute_loss(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update teacher (EMA)
            if hasattr(self.model, 'update_teacher'):
                self.model.update_teacher()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if batch_idx % self.config["log_every"] == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return {"train_loss": total_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            loss = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        
        return {"val_loss": total_loss / num_batches}
    
    @torch.no_grad()
    def evaluate_generation(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate generation quality."""
        self.model.eval()
        
        if hasattr(self.model, 'generate'):
            samples = self.model.generate(
                batch_size=num_samples,
                num_atoms=20,
                num_steps=1  # Single-step for consistency models
            )
            
            # Compute basic statistics
            metrics = {
                "generated_samples": num_samples,
                "avg_lattice_norm": samples["lattice"].norm(dim=1).mean().item()
            }
            return metrics
        
        return {}
    
    def save_checkpoint(self, filename: Optional[str] = None):
        """Save checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch}.pt"
        
        path = os.path.join(self.config["checkpoint_dir"], filename)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"Checkpoint loaded from {path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting consistency model training ({self.config['mode']} mode)")
        print(f"Device: {self.device}")
        
        for epoch in range(self.current_epoch, self.config["max_epochs"]):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            print(f"Epoch {epoch}: {train_metrics}")
            
            if self.val_loader is not None and epoch % self.config["eval_every"] == 0:
                val_metrics = self.validate()
                print(f"Validation: {val_metrics}")
                
                val_loss = val_metrics.get("val_loss", float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
            
            if epoch % self.config["save_every"] == 0:
                self.save_checkpoint()
                
                # Evaluate generation
                gen_metrics = self.evaluate_generation(num_samples=10)
                print(f"Generation metrics: {gen_metrics}")
        
        self.save_checkpoint("final_model.pt")
        print("Training complete!")
