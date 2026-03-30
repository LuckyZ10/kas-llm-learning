"""
Flow Matching Trainer
=====================

Training infrastructure for flow matching models.
Optimizes the conditional flow matching objective.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os
from tqdm import tqdm


class FlowMatchingTrainer:
    """Trainer for flow matching models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.config = {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 100,
            "save_every": 10,
            "eval_every": 5,
            "log_every": 100,
            "gradient_clip": 1.0,
            "checkpoint_dir": "./checkpoints_flow",
            "sigma_min": 1e-4,
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
        
        self.scheduler = scheduler
        
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def sample_conditional_path(
        self,
        x0: Dict[str, torch.Tensor],
        x1: Dict[str, torch.Tensor],
        t: torch.Tensor
    ):
        """Sample from conditional probability path."""
        B = t.shape[0]
        t_expanded = t.view(B, 1, 1)
        
        # Linear interpolation
        frac_coords_t = (1 - t_expanded) * x0["frac_coords"] + t_expanded * x1["frac_coords"]
        lattice_t = (1 - t.view(B, 1)) * x0["lattice"] + t.view(B, 1) * x1["lattice"]
        
        # Target vector field
        ut_coords = x1["frac_coords"] - x0["frac_coords"]
        ut_lattice = x1["lattice"] - x0["lattice"]
        
        # For atom types (continuous approximation)
        if x0["atom_types"].dim() == 3:  # Soft/logits
            atom_t = (1 - t_expanded) * x0["atom_types"] + t_expanded * x1["atom_types"]
            ut_atom = x1["atom_types"] - x0["atom_types"]
        else:
            atom_t = x1["atom_types"]  # Keep discrete
            ut_atom = None
        
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
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute flow matching loss."""
        B = batch["atom_types"].shape[0]
        
        # Target data
        x1 = {
            "atom_types": batch["atom_types"],
            "frac_coords": batch["frac_coords"],
            "lattice": batch["lattice"]
        }
        
        properties = batch.get("properties")
        
        # Sample time
        t = torch.rand(B, device=self.device)
        
        # Sample source (noise)
        N = x1["atom_types"].shape[1]
        x0 = {
            "atom_types": torch.randn(B, N, 100, device=self.device),  # Soft atom types
            "frac_coords": torch.randn(B, N, 3, device=self.device),
            "lattice": torch.randn(B, 6, device=self.device)
        }
        
        # Sample conditional path
        xt, ut = self.sample_conditional_path(x0, x1, t)
        
        # Predict vector field
        if hasattr(self.model, 'forward'):
            vt = self.model(
                xt["atom_types"],
                xt["frac_coords"],
                xt["lattice"],
                t,
                properties
            )
        else:
            vt = {"frac_coords": torch.zeros_like(ut["frac_coords"]),
                  "lattice": torch.zeros_like(ut["lattice"])}
        
        # Flow matching loss
        loss_coords = nn.functional.mse_loss(vt["frac_coords"], ut["frac_coords"])
        loss_lattice = nn.functional.mse_loss(vt["lattice"], ut["lattice"])
        
        loss = loss_coords + loss_lattice
        
        return loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            loss = self.compute_loss(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["gradient_clip"]
            )
            self.optimizer.step()
            
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
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
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
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Checkpoint loaded from {path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting flow matching training")
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
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if epoch % self.config["save_every"] == 0:
                self.save_checkpoint()
        
        self.save_checkpoint("final_model.pt")
        print("Training complete!")
