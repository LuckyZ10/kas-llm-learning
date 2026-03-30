#!/usr/bin/env python3
"""
Orb Potential Training
======================

Training interface for Orb (Object-Relation-Behavior) graph neural
network potentials by Orbital Materials.

References:
- Orbital Materials - Orb framework
- Graph neural networks for materials
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import logging
import subprocess
from collections import defaultdict

# ASE
from ase import Atoms
from ase.io import read, write

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OrbDataConfig:
    """Configuration for Orb training data.
    
    Attributes:
        train_file: Training data path (extxyz)
        valid_file: Validation data path
        test_file: Test data path (optional)
        cutoff: Graph cutoff distance (Å)
        energy_unit: Unit of energy ('eV', 'kcal/mol')
        length_unit: Unit of length ('angstrom', 'bohr')
    """
    train_file: str
    valid_file: str
    test_file: Optional[str] = None
    cutoff: float = 5.0
    energy_unit: str = "eV"
    length_unit: str = "angstrom"


@dataclass
class OrbArchitectureConfig:
    """Orb model architecture configuration.
    
    Attributes:
        num_layers: Number of message passing layers
        hidden_dim: Hidden feature dimension
        num_radial: Number of radial basis functions
        num_spherical: Number of spherical harmonics
        interaction_block: Type of interaction block
    """
    num_layers: int = 4
    hidden_dim: int = 128
    num_radial: int = 8
    num_spherical: int = 7
    interaction_block: str = "SchNet"
    max_neighbors: int = 50


@dataclass
class OrbTrainingConfig:
    """Orb training hyperparameters.
    
    Attributes:
        batch_size: Training batch size
        epochs: Maximum training epochs
        lr: Initial learning rate
        optimizer: Optimizer type ('Adam', 'AdamW')
        scheduler: LR scheduler type
        warmup_steps: Number of warmup steps
        weight_decay: L2 regularization
        gradient_clip: Gradient clipping value
    """
    batch_size: int = 8
    epochs: int = 1000
    lr: float = 0.001
    optimizer: str = "AdamW"
    scheduler: str = "reduce_on_plateau"
    warmup_steps: int = 1000
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    
    # Loss weights
    energy_weight: float = 1.0
    force_weight: float = 10.0
    stress_weight: float = 0.1
    
    # Output
    save_dir: str = "./orb_model"
    model_name: str = "orb_model"
    device: str = "cuda"
    checkpoint_every: int = 10
    eval_every: int = 100


@dataclass
class OrbRunConfig:
    """Complete Orb run configuration."""
    data: OrbDataConfig
    architecture: OrbArchitectureConfig
    training: OrbTrainingConfig


class OrbDatasetPreparer:
    """Prepare datasets for Orb training."""
    
    def __init__(self, config: OrbDataConfig):
        self.config = config
    
    def load_and_validate(self) -> Dict[str, List[Atoms]]:
        """Load and validate training data."""
        logger.info(f"Loading training data from {self.config.train_file}")
        
        train_atoms = list(read(self.config.train_file, index=':'))
        valid_atoms = list(read(self.config.valid_file, index=':'))
        
        logger.info(f"Loaded {len(train_atoms)} training, {len(valid_atoms)} validation structures")
        
        # Validate
        for atoms in train_atoms[:5]:
            if 'energy' not in atoms.info:
                logger.warning("Missing energy in training data")
            if 'forces' not in atoms.arrays:
                logger.warning("Missing forces in training data")
        
        result = {'train': train_atoms, 'valid': valid_atoms}
        
        if self.config.test_file and Path(self.config.test_file).exists():
            test_atoms = list(read(self.config.test_file, index=':'))
            result['test'] = test_atoms
        
        return result
    
    def compute_statistics(self, atoms_list: List[Atoms]) -> Dict:
        """Compute dataset statistics."""
        energies = []
        forces = []
        n_atoms = []
        
        for atoms in atoms_list:
            n_atoms.append(len(atoms))
            
            if 'energy' in atoms.info:
                energies.append(atoms.info['energy'])
            
            if 'forces' in atoms.arrays:
                forces.extend(atoms.arrays['forces'].flatten())
        
        stats = {
            'n_structures': len(atoms_list),
            'mean_n_atoms': np.mean(n_atoms),
            'max_n_atoms': np.max(n_atoms),
            'min_n_atoms': np.min(n_atoms)
        }
        
        if energies:
            stats['mean_energy'] = np.mean(energies)
            stats['std_energy'] = np.std(energies)
            stats['energy_per_atom_mean'] = np.mean(energies) / np.mean(n_atoms)
        
        if forces:
            stats['force_mean'] = np.mean(np.abs(forces))
            stats['force_std'] = np.std(forces)
        
        return stats
    
    def prepare_data_splits(self, all_data: str,
                           output_dir: str,
                           train_ratio: float = 0.8,
                           val_ratio: float = 0.1) -> Dict[str, str]:
        """Prepare train/val/test splits from combined data."""
        logger.info(f"Loading data from {all_data}")
        atoms_list = list(read(all_data, index=':'))
        
        # Shuffle
        np.random.seed(42)
        n_total = len(atoms_list)
        indices = np.random.permutation(n_total)
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write splits
        train_file = output_path / "train.extxyz"
        val_file = output_path / "valid.extxyz"
        test_file = output_path / "test.extxyz"
        
        write(train_file, [atoms_list[i] for i in train_indices])
        write(val_file, [atoms_list[i] for i in val_indices])
        write(test_file, [atoms_list[i] for i in test_indices])
        
        logger.info(f"Split {n_total} structures: {len(train_indices)} train, "
                   f"{len(val_indices)} val, {len(test_indices)} test")
        
        return {
            'train': str(train_file),
            'valid': str(val_file),
            'test': str(test_file)
        }


class OrbTrainer:
    """Orb model training manager."""
    
    def __init__(self, config: OrbRunConfig):
        self.config = config
        self.save_dir = Path(config.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.trainer = None
    
    def create_model(self) -> Any:
        """Create Orb model."""
        try:
            # Try to import Orb
            from orb_models.forcefield import pretrained
            
            # Load pretrained Orb model as base
            self.model = pretrained.ORB_PRETRAINED_MODELS[
                'orb-v1-20240827'  # or other available model
            ]()
            
            logger.info("Created Orb model from pretrained")
            return self.model
        except ImportError:
            logger.warning("Orb not installed. Creating mock model.")
            self.model = None
            return None
    
    def generate_training_script(self) -> str:
        """Generate Python training script."""
        script = f'''#!/usr/bin/env python3
"""
Orb Training Script (Auto-generated)
"""

import torch
import numpy as np
from pathlib import Path
from ase.io import read

# Orb imports (if available)
try:
    from orb_models.forcefield import pretrained, GraphRegressor
    from orb_models.forcefield.data import OrbDataset, collate_fn
    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False
    print("Warning: Orb not installed")

# Configuration
train_file = "{self.config.data.train_file}"
val_file = "{self.config.data.valid_file}"
save_dir = Path("{self.config.training.save_dir}")

# Load data
print("Loading data...")
train_atoms = list(read(train_file, index=':'))
val_atoms = list(read(val_file, index=':'))

print(f"Training: {{len(train_atoms)}} structures")
print(f"Validation: {{len(val_atoms)}} structures")

if ORB_AVAILABLE:
    # Create datasets
    train_dataset = OrbDataset(train_atoms, 
                               energy_key='energy',
                               forces_key='forces')
    val_dataset = OrbDataset(val_atoms,
                             energy_key='energy',
                             forces_key='forces')
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size={self.config.training.batch_size},
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size={self.config.training.batch_size},
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Load pretrained model
    print("Loading pretrained Orb model...")
    model = pretrained.ORB_PRETRAINED_MODELS['orb-v1-20240827']()
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr={self.config.training.lr},
        weight_decay={self.config.training.weight_decay}
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    
    # Loss function
    def loss_fn(pred, target):
        energy_loss = torch.nn.functional.mse_loss(pred['energy'], target['energy'])
        force_loss = torch.nn.functional.mse_loss(pred['forces'], target['forces'])
        return ({self.config.training.energy_weight} * energy_loss + 
                {self.config.training.force_weight} * force_loss)
    
    # Training loop
    device = torch.device("{self.config.training.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range({self.config.training.epochs}):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch.to(device))
            loss = loss_fn(pred, batch.y)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), {self.config.training.gradient_clip})
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch.to(device))
                loss = loss_fn(pred, batch.y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {{epoch+1}}: Train={{train_loss:.6f}}, Val={{val_loss:.6f}}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "{self.config.training.model_name}_best.pt")
    
    print("Training complete!")
else:
    print("Orb training requires Orb package installation")
'''
        
        script_file = self.save_dir / "train_orb.py"
        with open(script_file, 'w') as f:
            f.write(script)
        
        os.chmod(script_file, 0o755)
        logger.info(f"Training script saved to {script_file}")
        return str(script_file)
    
    def run_training(self) -> Dict:
        """Run Orb training."""
        logger.info("Starting Orb training")
        
        train_script = self.generate_training_script()
        
        try:
            result = subprocess.run(
                [sys.executable, train_script],
                capture_output=True,
                text=True,
                timeout=604800,
                cwd=self.save_dir
            )
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'stdout': result.stdout[-5000:],  # Last 5000 chars
                'stderr': result.stderr[-2000:],
                'save_dir': str(self.save_dir)
            }
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}


class OrbEvaluator:
    """Evaluate Orb models."""
    
    def __init__(self, model_path: Optional[str] = None,
                use_pretrained: bool = True):
        self.model_path = model_path
        self.use_pretrained = use_pretrained
        self.model = None
        self.calculator = None
        self._load_model()
    
    def _load_model(self):
        """Load Orb model."""
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import OrbCalc
            
            if self.model_path:
                # Load custom model
                self.model = torch.load(self.model_path)
            elif self.use_pretrained:
                # Use pretrained
                self.model = pretrained.ORB_PRETRAINED_MODELS['orb-v1-20240827']()
            
            self.calculator = OrbCalc(self.model)
            logger.info("Loaded Orb model")
            
        except ImportError:
            logger.error("Orb not installed")
            self.model = None
            self.calculator = None
    
    def predict(self, atoms: Atoms) -> Dict:
        """Make prediction on structure."""
        if self.calculator is None:
            return {'error': 'Orb not available'}
        
        atoms.calc = self.calculator
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        result = {
            'energy': energy,
            'forces': forces
        }
        
        try:
            stress = atoms.get_stress()
            result['stress'] = stress
        except:
            pass
        
        return result
    
    def get_calculator(self):
        """Get ASE calculator."""
        return self.calculator


# Export public API
__all__ = [
    'OrbDataConfig',
    'OrbArchitectureConfig',
    'OrbTrainingConfig',
    'OrbRunConfig',
    'OrbDatasetPreparer',
    'OrbTrainer',
    'OrbEvaluator'
]
