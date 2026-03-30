"""
Diffusion Model Trainer
=======================

Training infrastructure for diffusion-based generative models.
Supports:
- Standard diffusion training
- Conditional training with classifier-free guidance
- EMA of model weights
- Checkpointing and resumption
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import os
import json
from pathlib import Path
import time
from tqdm import tqdm


class EMAModel:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in model.parameters()]
        
    def update(self):
        """Update EMA parameters."""
        for shadow_param, param in zip(self.shadow_params, self.model.parameters()):
            shadow_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for shadow_param, param in zip(self.shadow_params, self.model.parameters()):
            param.data.copy_(shadow_param)
    
    def store(self, path: str):
        """Store EMA parameters."""
        torch.save(self.shadow_params, path)
    
    def load(self, path: str):
        """Load EMA parameters."""
        self.shadow_params = torch.load(path)


class DiffusionTrainer:
    """
    Trainer for diffusion models.
    """
    
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
        
        # Default config
        self.config = {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "max_epochs": 100,
            "save_every": 10,
            "eval_every": 5,
            "log_every": 100,
            "gradient_clip": 1.0,
            "use_ema": True,
            "ema_decay": 0.9999,
            "mixed_precision": True,
            "checkpoint_dir": "./checkpoints",
            **(config or {})
        }
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # EMA
        self.ema = None
        if self.config["use_ema"]:
            self.ema = EMAModel(model, decay=self.config["ema_decay"])
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config["mixed_precision"] else None
        
        # Checkpoint directory
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config["mixed_precision"]):
                loss = self.compute_loss(batch)
            
            # Backward
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["gradient_clip"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["gradient_clip"]
                )
                self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if batch_idx % self.config["log_every"] == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        
        return {"train_loss": avg_loss}
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for a batch.
        Override this for custom loss computation.
        """
        # Default: assume model has compute_loss method
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(batch)
        
        # Standard diffusion loss
        x1 = {
            "atom_types": batch["atom_types"],
            "frac_coords": batch["frac_coords"],
            "lattice": batch["lattice"]
        }
        
        properties = batch.get("properties")
        
        # Sample timestep
        B = x1["atom_types"].shape[0]
        t = torch.rand(B, device=self.device)
        
        # Sample noise
        noise = {
            "frac_coords": torch.randn_like(x1["frac_coords"]),
            "lattice": torch.randn_like(x1["lattice"])
        }
        
        # Add noise
        x0 = {
            "atom_types": x1["atom_types"],
            "frac_coords": (1 - t.view(B, 1, 1)) * torch.randn_like(x1["frac_coords"]) + t.view(B, 1, 1) * x1["frac_coords"],
            "lattice": (1 - t.view(B, 1)) * torch.randn_like(x1["lattice"]) + t.view(B, 1) * x1["lattice"]
        }
        
        # Predict
        pred = self.model(
            x0["atom_types"],
            x0["frac_coords"],
            x0["lattice"],
            t,
            properties=properties
        )
        
        # Loss
        loss = nn.functional.mse_loss(pred["frac_coords"], x1["frac_coords"])
        
        return loss
    
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
        
        avg_loss = total_loss / num_batches
        
        return {"val_loss": avg_loss}
    
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
        
        if self.ema is not None:
            ema_path = path.replace(".pt", "_ema.pt")
            self.ema.store(ema_path)
            checkpoint["ema_path"] = ema_path
        
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
        
        if self.ema is not None and "ema_path" in checkpoint:
            self.ema.load(checkpoint["ema_path"])
        
        print(f"Checkpoint loaded from {path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['max_epochs']} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.config["max_epochs"]):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log
            print(f"Epoch {epoch}: {train_metrics}")
            
            # Validate
            if self.val_loader is not None and epoch % self.config["eval_every"] == 0:
                val_metrics = self.validate()
                print(f"Validation: {val_metrics}")
                
                # Save best model
                val_loss = val_metrics.get("val_loss", float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config["save_every"] == 0:
                self.save_checkpoint()
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        print("Training complete!")
