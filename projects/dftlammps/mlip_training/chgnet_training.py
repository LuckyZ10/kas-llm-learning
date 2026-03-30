#!/usr/bin/env python3
"""
CHGNet Potential Training
=========================

Training interface for CHGNet (Crystal Hamiltonian Graph Neural Network),
a universal GNN potential for materials.

References:
- Deng et al. (2023) - CHGNet
- Crystal Hamiltonian Graph Neural Network
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
class CHGNetDataConfig:
    """Configuration for CHGNet training data.
    
    Attributes:
        train_file: Training data path (json or extxyz)
        valid_file: Validation data path
        test_file: Test data path
        cutoff: Graph construction cutoff
        include_stress: Include stress in training
        include_magmom: Include magnetic moments
    """
    train_file: str
    valid_file: str
    test_file: Optional[str] = None
    cutoff: float = 6.0
    include_stress: bool = False
    include_magmom: bool = False


@dataclass
class CHGNetArchitectureConfig:
    """CHGNet model architecture configuration.
    
    Attributes:
        n_atom_emb: Atom embedding dimension
        n_conv: Number of graph convolution layers
        n_filters: Number of filters in convolution
        n_spherical: Number of spherical harmonics
        n_radial: Number of radial basis functions
        bond_fea_len: Bond feature length
        angle_fea_len: Angle feature length
    """
    n_atom_emb: int = 64
    n_conv: int = 4
    n_filters: int = 64
    n_spherical: int = 8
    n_radial: int = 9
    bond_fea_len: int = 64
    angle_fea_len: int = 64


@dataclass
class CHGNetTrainingConfig:
    """CHGNet training hyperparameters.
    
    Attributes:
        batch_size: Training batch size
        epochs: Training epochs
        lr: Initial learning rate
        weight_decay: L2 regularization
        scheduler: LR scheduler type
        patience: Early stopping patience
    """
    batch_size: int = 32
    epochs: int = 200
    lr: float = 0.001
    weight_decay: float = 1e-5
    scheduler: str = "CosineAnnealingLR"
    warmup_epochs: int = 10
    patience: int = 50
    min_lr: float = 1e-6
    
    # Loss weights
    energy_weight: float = 1.0
    force_weight: float = 1.0
    stress_weight: float = 0.1
    magmom_weight: float = 0.1
    
    # Training settings
    save_dir: str = "./chgnet_model"
    model_name: str = "chgnet_model"
    device: str = "cuda"
    checkpoint_every: int = 10
    print_every: int = 100


@dataclass
class CHGNetRunConfig:
    """Complete CHGNet run configuration."""
    data: CHGNetDataConfig
    architecture: CHGNetArchitectureConfig
    training: CHGNetTrainingConfig


class CHGNetDatasetPreparer:
    """Prepare datasets for CHGNet training."""
    
    def __init__(self, config: CHGNetDataConfig):
        self.config = config
    
    def convert_to_chgnet_format(self, atoms_list: List[Atoms]) -> List[Dict]:
        """Convert ASE Atoms to CHGNet format."""
        data = []
        
        for atoms in atoms_list:
            structure_dict = {
                'atomic_numbers': atoms.get_atomic_numbers().tolist(),
                'positions': atoms.positions.tolist(),
                'cell': atoms.cell.tolist() if atoms.pbc.any() else None,
                'pbc': atoms.pbc.tolist()
            }
            
            # Add properties
            if 'energy' in atoms.info:
                structure_dict['energy'] = atoms.info['energy']
            
            if 'forces' in atoms.arrays:
                structure_dict['forces'] = atoms.arrays['forces'].tolist()
            
            if self.config.include_stress and 'stress' in atoms.info:
                structure_dict['stress'] = atoms.info['stress']
            
            if self.config.include_magmom and 'magmom' in atoms.arrays:
                structure_dict['magmom'] = atoms.arrays['magmom'].tolist()
            
            data.append(structure_dict)
        
        return data
    
    def prepare_dataset(self, atoms_list: List[Atoms],
                       output_file: str) -> str:
        """Prepare and save dataset in CHGNet format."""
        data = self.convert_to_chgnet_format(atoms_list)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Dataset saved to {output_path}")
        return str(output_path)
    
    def prepare_from_extxyz(self, extxyz_file: str,
                           output_file: str,
                           train_ratio: float = 0.8,
                           val_ratio: float = 0.1) -> Dict[str, str]:
        """Prepare dataset from extxyz file."""
        logger.info(f"Loading data from {extxyz_file}")
        atoms_list = list(read(extxyz_file, index=':'))
        
        n_total = len(atoms_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Shuffle
        np.random.seed(42)
        indices = np.random.permutation(n_total)
        
        train_atoms = [atoms_list[i] for i in indices[:n_train]]
        val_atoms = [atoms_list[i] for i in indices[n_train:n_train + n_val]]
        test_atoms = [atoms_list[i] for i in indices[n_train + n_val:]]
        
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_file = self.prepare_dataset(train_atoms, str(output_dir / "train.json"))
        val_file = self.prepare_dataset(val_atoms, str(output_dir / "val.json"))
        
        result = {'train': train_file, 'val': val_file}
        
        if test_atoms:
            test_file = self.prepare_dataset(test_atoms, str(output_dir / "test.json"))
            result['test'] = test_file
        
        return result


class CHGNetTrainer:
    """CHGNet model training manager."""
    
    def __init__(self, config: CHGNetRunConfig):
        self.config = config
        self.save_dir = Path(config.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.trainer = None
    
    def create_model(self) -> Any:
        """Create CHGNet model."""
        try:
            from chgnet.model import CHGNet
            
            # Try to load pretrained or create new
            if hasattr(self.config.architecture, 'use_pretrained') and \
               self.config.architecture.use_pretrained:
                self.model = CHGNet.load()
                logger.info("Loaded pretrained CHGNet")
            else:
                # Create new model with custom architecture
                # CHGNet uses fixed architecture, but we can fine-tune
                self.model = CHGNet.load()
                logger.info("Created CHGNet model from pretrained base")
            
            return self.model
        except ImportError:
            logger.error("CHGNet not installed. Install with: pip install chgnet")
            raise
    
    def generate_training_script(self) -> str:
        """Generate Python training script."""
        script = f'''#!/usr/bin/env python3
"""
CHGNet Training Script (Auto-generated)
"""

import json
import numpy as np
import torch
from pathlib import Path
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader

# Configuration
train_file = "{self.config.data.train_file}"
val_file = "{self.config.data.valid_file}"
save_dir = "{self.config.training.save_dir}"

# Load data
print("Loading datasets...")
with open(train_file, 'r') as f:
    train_data = json.load(f)
with open(val_file, 'r') as f:
    val_data = json.load(f)

# Create datasets
train_dataset = StructureData(train_data)
val_dataset = StructureData(val_data)

# Create dataloaders
train_loader, val_loader = get_train_val_test_loader(
    train_dataset, val_dataset, None,
    batch_size={self.config.training.batch_size},
    num_workers=4
)

# Load or create model
print("Creating model...")
chgnet = CHGNet.load()

# Create trainer
trainer = Trainer(
    model=chgnet,
    targets=["energy", "forces"] + {["stress"] if self.config.data.include_stress else []},
    criterion="MSE",
    optimizer="Adam",
    scheduler={self.config.training.scheduler},
    learning_rate={self.config.training.lr},
    epochs={self.config.training.epochs},
    use_device="{self.config.training.device}"
)

# Train
print("Starting training...")
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    save_dir=save_dir,
    save_name="{self.config.training.model_name}"
)

print("Training complete!")
'''
        
        script_file = self.save_dir / "train_chgnet.py"
        with open(script_file, 'w') as f:
            f.write(script)
        
        os.chmod(script_file, 0o755)
        logger.info(f"Training script saved to {script_file}")
        return str(script_file)
    
    def run_training(self) -> Dict:
        """Run CHGNet training."""
        logger.info("Starting CHGNet training")
        
        # Generate training script
        train_script = self.generate_training_script()
        
        try:
            # Run training
            result = subprocess.run(
                [sys.executable, train_script],
                capture_output=True,
                text=True,
                timeout=604800,  # 1 week
                cwd=self.save_dir
            )
            
            success = result.returncode == 0
            
            return {{
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'save_dir': str(self.save_dir)
            }}
        except Exception as e:
            logger.error(f"Training failed: {{e}}")
            return {{'success': False, 'error': str(e)}}
    
    def fine_tune(self, pretrained_model: str,
                 freeze_graph: bool = True,
                 lr_multiplier: float = 0.1) -> Dict:
        """Fine-tune pretrained CHGNet."""
        logger.info(f"Fine-tuning from {pretrained_model}")
        
        script = f'''#!/usr/bin/env python3
"""
CHGNet Fine-tuning Script
"""

from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader
import json

# Load pretrained model
print("Loading pretrained model...")
chgnet = CHGNet.load({pretrained_model!r})

# Optionally freeze graph layers
if {freeze_graph}:
    for param in chgnet.atom_convnet.parameters():
        param.requires_grad = False
    print("Froze graph convolution layers")

# Load data
with open("{self.config.data.train_file}", 'r') as f:
    train_data = json.load(f)
with open("{self.config.data.valid_file}", 'r') as f:
    val_data = json.load(f)

train_dataset = StructureData(train_data)
val_dataset = StructureData(val_data)

train_loader, val_loader = get_train_val_test_loader(
    train_dataset, val_dataset, None,
    batch_size={self.config.training.batch_size}
)

# Fine-tune with lower learning rate
trainer = Trainer(
    model=chgnet,
    targets=["energy", "forces"],
    learning_rate={self.config.training.lr * lr_multiplier},
    epochs={self.config.training.epochs},
    use_device="{self.config.training.device}"
)

trainer.train(train_loader, val_loader, 
              save_dir="{self.config.training.save_dir}",
              save_name="{self.config.training.model_name}_finetuned")

print("Fine-tuning complete!")
'''
        
        script_file = self.save_dir / "finetune_chgnet.py"
        with open(script_file, 'w') as f:
            f.write(script)
        
        try:
            result = subprocess.run(
                [sys.executable, script_file],
                capture_output=True,
                text=True,
                timeout=604800,
                cwd=self.save_dir
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return {'success': False, 'error': str(e)}


class CHGNetEvaluator:
    """Evaluate CHGNet models."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load CHGNet model."""
        try:
            from chgnet.model import CHGNet
            
            if self.model_path:
                self.model = CHGNet.load(self.model_path)
            else:
                self.model = CHGNet.load()
            
            logger.info(f"Loaded CHGNet model from {self.model_path or 'pretrained'}")
        except ImportError:
            logger.error("CHGNet not installed")
            raise
    
    def predict(self, atoms: Atoms) -> Dict:
        """Make prediction on structure."""
        from chgnet.model import Struct
        
        # Convert to CHGNet structure
        structure = Struct(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc
        )
        
        # Predict
        prediction = self.model.predict_structure(structure)
        
        result = {
            'energy': prediction['e'],
            'forces': prediction['f'],
            'stress': prediction.get('s', None)
        }
        
        if 'm' in prediction:
            result['magmom'] = prediction['m']
        
        return result
    
    def evaluate_dataset(self, atoms_list: List[Atoms],
                        ref_energies: Optional[List[float]] = None,
                        ref_forces: Optional[List[np.ndarray]] = None) -> pd.DataFrame:
        """Evaluate on dataset and compute errors."""
        results = []
        
        for i, atoms in enumerate(atoms_list):
            pred = self.predict(atoms)
            
            row = {
                'index': i,
                'pred_energy': pred['energy'],
                'pred_forces': pred['forces']
            }
            
            if ref_energies:
                row['ref_energy'] = ref_energies[i]
                row['energy_error'] = pred['energy'] - ref_energies[i]
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def get_calculator(self):
        """Get ASE calculator."""
        from chgnet.model import CHGNetCalculator
        return CHGNetCalculator(self.model)


# Export public API
__all__ = [
    'CHGNetDataConfig',
    'CHGNetArchitectureConfig',
    'CHGNetTrainingConfig',
    'CHGNetRunConfig',
    'CHGNetDatasetPreparer',
    'CHGNetTrainer',
    'CHGNetEvaluator'
]
