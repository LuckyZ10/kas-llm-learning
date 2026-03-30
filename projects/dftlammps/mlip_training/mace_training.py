#!/usr/bin/env python3
"""
MACE Potential Training
=======================

Training interface for MACE (Multi-Atomic Cluster Expansion) 
message-passing neural networks with E(3) equivariance.

References:
- Batatia et al. (2022) - MACE framework
- Design principles for equivariant neural networks
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import logging
import subprocess
import tempfile
from collections import defaultdict

# ASE
from ase import Atoms
from ase.io import read, write
from ase.io.extxyz import write_extxyz, read_extxyz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MACEDataConfig:
    """Configuration for MACE training data.
    
    Attributes:
        train_file: Path to training data (extxyz)
        valid_file: Path to validation data
        test_file: Path to test data (optional)
        cutoff: Maximum atomic interaction distance (Å)
        energy_key: Key for energy in Atoms.info
        forces_key: Key for forces in Atoms.arrays
        stress_key: Key for stress in Atoms.info
    """
    train_file: str
    valid_file: str
    test_file: Optional[str] = None
    cutoff: float = 5.0
    energy_key: str = "energy"
    forces_key: str = "forces"
    stress_key: str = "stress"
    virials_key: Optional[str] = None
    charges_key: Optional[str] = None
    
    def __post_init__(self):
        for f in [self.train_file, self.valid_file]:
            if not Path(f).exists():
                raise FileNotFoundError(f"Data file not found: {f}")


@dataclass
class MACEArchitectureConfig:
    """MACE model architecture configuration.
    
    Attributes:
        hidden_irreps: Irreps for hidden features (e.g., "128x0e + 128x1o")
        num_interactions: Number of message passing layers
        num_radial_basis: Number of radial basis functions
        max_ell: Maximum spherical harmonic degree
        correlation_order: Body order for equivariant message passing
        rbf_type: Radial basis function type ('bessel' or 'gaussian')
        interaction: Interaction type ('RealAgnosticResidualInteractionBlock')
    """
    hidden_irreps: str = "128x0e + 128x1o"
    num_interactions: int = 2
    num_radial_basis: int = 8
    max_ell: int = 3
    correlation_order: int = 3
    rbf_type: str = "bessel"
    interaction: str = "RealAgnosticResidualInteractionBlock"
    gate: str = "silu"
    MLP_irreps: str = "16x0e"
    radial_MLP: str = "64,64,64"
    avg_num_neighbors: Optional[float] = None


@dataclass
class MACETrainingConfig:
    """MACE training hyperparameters.
    
    Attributes:
        batch_size: Batch size for training
        max_num_epochs: Maximum training epochs
        lr: Initial learning rate
        scheduler_patience: Patience for learning rate scheduler
        scheduler_factor: Factor for LR reduction
        weight_decay: L2 regularization
        ema_decay: Exponential moving average decay
        patience: Early stopping patience
        error_table: Error metrics to track (e.g., 'PerAtomRMSE')
    """
    batch_size: int = 5
    max_num_epochs: int = 1000
    lr: float = 0.01
    scheduler_patience: int = 50
    scheduler_factor: float = 0.8
    weight_decay: float = 5e-7
    ema_decay: float = 0.99
    patience: int = 100
    min_lr: float = 1e-7
    max_grad_norm: float = 10.0
    
    # Loss weights
    energy_weight: float = 1.0
    forces_weight: float = 10.0
    stress_weight: float = 1.0
    virials_weight: float = 1.0
    
    # Output
    model_dir: str = "./mace_model"
    model_name: str = "mace_model"
    log_dir: str = "./mace_logs"
    device: str = "cuda"
    seed: int = 123


@dataclass
class MACERunConfig:
    """Complete MACE run configuration."""
    data: MACEDataConfig
    architecture: MACEArchitectureConfig
    training: MACETrainingConfig


class MACEDatasetPreparer:
    """Prepare datasets for MACE training."""
    
    def __init__(self, data_config: MACEDataConfig):
        self.config = data_config
        self.train_atoms: List[Atoms] = []
        self.valid_atoms: List[Atoms] = []
        self.test_atoms: List[Atoms] = []
    
    def load_data(self) -> Dict[str, List[Atoms]]:
        """Load training data from extxyz files."""
        logger.info(f"Loading training data from {self.config.train_file}")
        self.train_atoms = list(read(self.config.train_file, index=':'))
        logger.info(f"Loaded {len(self.train_atoms)} training structures")
        
        logger.info(f"Loading validation data from {self.config.valid_file}")
        self.valid_atoms = list(read(self.config.valid_file, index=':'))
        logger.info(f"Loaded {len(self.valid_atoms)} validation structures")
        
        if self.config.test_file and Path(self.config.test_file).exists():
            self.test_atoms = list(read(self.config.test_file, index=':'))
            logger.info(f"Loaded {len(self.test_atoms)} test structures")
        
        return {
            'train': self.train_atoms,
            'valid': self.valid_atoms,
            'test': self.test_atoms
        }
    
    def validate_data(self) -> Dict[str, Any]:
        """Validate data format and statistics."""
        stats = {
            'train': self._compute_stats(self.train_atoms),
            'valid': self._compute_stats(self.valid_atoms),
            'test': self._compute_stats(self.test_atoms) if self.test_atoms else None
        }
        
        # Check for required keys
        required_info = [self.config.energy_key]
        required_arrays = [self.config.forces_key]
        
        missing = []
        for atoms in self.train_atoms[:10]:
            for key in required_info:
                if key not in atoms.info:
                    missing.append(key)
            for key in required_arrays:
                if key not in atoms.arrays:
                    missing.append(key)
        
        if missing:
            logger.warning(f"Missing keys found: {set(missing)}")
        
        return stats
    
    def _compute_stats(self, atoms_list: List[Atoms]) -> Dict:
        """Compute statistics for dataset."""
        if not atoms_list:
            return {}
        
        n_atoms = [len(atoms) for atoms in atoms_list]
        
        energies = []
        for atoms in atoms_list:
            if self.config.energy_key in atoms.info:
                energies.append(atoms.info[self.config.energy_key])
        
        stats = {
            'n_structures': len(atoms_list),
            'mean_n_atoms': np.mean(n_atoms),
            'min_n_atoms': np.min(n_atoms),
            'max_n_atoms': np.max(n_atoms)
        }
        
        if energies:
            stats['mean_energy'] = np.mean(energies)
            stats['std_energy'] = np.std(energies)
            stats['min_energy'] = np.min(energies)
            stats['max_energy'] = np.max(energies)
        
        return stats
    
    def preprocess_data(self, output_dir: str = "./mace_data") -> Dict[str, str]:
        """Preprocess and save data in MACE format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_out = output_path / "train.extxyz"
        valid_out = output_path / "valid.extxyz"
        
        # Write extxyz files
        self._write_extxyz(self.train_atoms, train_out)
        self._write_extxyz(self.valid_atoms, valid_out)
        
        result = {
            'train': str(train_out),
            'valid': str(valid_out)
        }
        
        if self.test_atoms:
            test_out = output_path / "test.extxyz"
            self._write_extxyz(self.test_atoms, test_out)
            result['test'] = str(test_out)
        
        return result
    
    def _write_extxyz(self, atoms_list: List[Atoms], filename: Path):
        """Write atoms to extxyz file."""
        with open(filename, 'w') as f:
            for atoms in atoms_list:
                write_extxyz(f, atoms)
        logger.info(f"Wrote {len(atoms_list)} structures to {filename}")
    
    def compute_average_num_neighbors(self) -> float:
        """Compute average number of neighbors for normalization."""
        neighbor_counts = []
        
        for atoms in self.train_atoms[:100]:  # Sample
            # Compute neighbor list
            from ase.neighborlist import NeighborList, natural_cutoffs
            
            cutoffs = natural_cutoffs(atoms, mult=1.0)
            nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
            nl.update(atoms)
            
            for i in range(len(atoms)):
                indices, _ = nl.get_neighbors(i)
                neighbor_counts.append(len(indices))
        
        avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 10.0
        logger.info(f"Average number of neighbors: {avg_neighbors:.2f}")
        
        return avg_neighbors


class MACETrainer:
    """MACE model training manager."""
    
    def __init__(self, config: MACERunConfig):
        self.config = config
        self.data_preparer = MACEDatasetPreparer(config.data)
        self.model_dir = Path(config.training.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_config_file(self) -> str:
        """Generate MACE YAML configuration file."""
        config_dict = {
            'model': 'MACE',
            'name': self.config.training.model_name,
            'seed': self.config.training.seed,
            
            # Data
            'train_file': self.config.data.train_file,
            'valid_file': self.config.data.valid_file,
            'test_file': self.config.data.test_file,
            'energy_key': self.config.data.energy_key,
            'forces_key': self.config.data.forces_key,
            'stress_key': self.config.data.stress_key,
            
            # Architecture
            'r_max': self.config.data.cutoff,
            'num_radial_basis': self.config.architecture.num_radial_basis,
            'num_interactions': self.config.architecture.num_interactions,
            'hidden_irreps': self.config.architecture.hidden_irreps,
            'max_ell': self.config.architecture.max_ell,
            'correlation_order': self.config.architecture.correlation_order,
            'rbf_type': self.config.architecture.rbf_type,
            'gate': self.config.architecture.gate,
            'interaction': self.config.architecture.interaction,
            'MLP_irreps': self.config.architecture.MLP_irreps,
            'radial_MLP': self.config.architecture.radial_MLP,
            'avg_num_neighbors': self.config.architecture.avg_num_neighbors,
            
            # Training
            'batch_size': self.config.training.batch_size,
            'max_num_epochs': self.config.training.max_num_epochs,
            'lr': self.config.training.lr,
            'scheduler_patience': self.config.training.scheduler_patience,
            'scheduler_factor': self.config.training.scheduler_factor,
            'weight_decay': self.config.training.weight_decay,
            'ema_decay': self.config.training.ema_decay,
            'patience': self.config.training.patience,
            'max_grad_norm': self.config.training.max_grad_norm,
            
            # Loss weights
            'energy_weight': self.config.training.energy_weight,
            'forces_weight': self.config.training.forces_weight,
            'stress_weight': self.config.training.stress_weight,
            
            # Output
            'model_dir': str(self.model_dir),
            'log_dir': self.config.training.log_dir,
            'device': self.config.training.device,
            'default_dtype': 'float64'
        }
        
        config_file = self.model_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Config saved to {config_file}")
        return str(config_file)
    
    def generate_train_script(self) -> str:
        """Generate training command/script."""
        config_file = self.prepare_config_file()
        
        script = f"""#!/bin/bash
# MACE Training Script

# Load environment
source ~/mace_env/bin/activate  # Adjust as needed

# Run training
python -m mace.cli.run_train \\
    --config={config_file}

# Save best model
cp {self.model_dir}/{self.config.training.model_name}* \
   {self.model_dir}/final_model.model
"""
        
        script_file = self.model_dir / "train.sh"
        with open(script_file, 'w') as f:
            f.write(script)
        os.chmod(script_file, 0o755)
        
        logger.info(f"Training script saved to {script_file}")
        return str(script_file)
    
    def run_training(self, use_slurm: bool = False,
                    slurm_config: Optional[Dict] = None) -> Dict:
        """Run MACE training."""
        logger.info("Starting MACE training")
        
        # Prepare data
        self.data_preparer.load_data()
        self.data_preparer.validate_data()
        
        # Compute average neighbors if not provided
        if self.config.architecture.avg_num_neighbors is None:
            self.config.architecture.avg_num_neighbors = \
                self.data_preparer.compute_average_num_neighbors()
        
        # Generate training script
        train_script = self.generate_train_script()
        
        if use_slurm and slurm_config:
            # Generate SLURM script
            return self._submit_slurm_job(train_script, slurm_config)
        else:
            # Run locally
            try:
                result = subprocess.run(
                    ['bash', train_script],
                    capture_output=True,
                    text=True,
                    timeout=604800,  # 1 week
                    cwd=self.model_dir
                )
                
                success = result.returncode == 0
                
                return {
                    'success': success,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'model_dir': str(self.model_dir)
                }
            except Exception as e:
                logger.error(f"Training failed: {e}")
                return {'success': False, 'error': str(e)}
    
    def _submit_slurm_job(self, train_script: str, 
                         slurm_config: Dict) -> Dict:
        """Submit training as SLURM job."""
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=mace_train
#SBATCH --output={self.model_dir}/slurm_%j.out
#SBATCH --error={self.model_dir}/slurm_%j.err
#SBATCH --nodes={slurm_config.get('nodes', 1)}
#SBATCH --ntasks-per-node={slurm_config.get('tasks_per_node', 1)}
#SBATCH --gres=gpu:{slurm_config.get('gpus', 1)}
#SBATCH --time={slurm_config.get('time', '24:00:00')}
#SBATCH --partition={slurm_config.get('partition', 'gpu')}

module load cuda
module load python

bash {train_script}
"""
        
        slurm_file = self.model_dir / "submit.slurm"
        with open(slurm_file, 'w') as f:
            f.write(slurm_script)
        
        try:
            result = subprocess.run(
                ['sbatch', str(slurm_file)],
                capture_output=True,
                text=True,
                cwd=self.model_dir
            )
            
            # Parse job ID
            job_id = None
            if result.returncode == 0:
                output = result.stdout.strip()
                if 'Submitted batch job' in output:
                    job_id = output.split()[-1]
            
            return {
                'success': result.returncode == 0,
                'job_id': job_id,
                'slurm_script': str(slurm_file)
            }
        except Exception as e:
            logger.error(f"SLURM submission failed: {e}")
            return {'success': False, 'error': str(e)}


class MACEEvaluator:
    """Evaluate trained MACE models."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load trained MACE model."""
        try:
            from mace.calculators import MACECalculator
            
            self.model = MACECalculator(
                model_paths=self.model_path,
                device=self.device
            )
            logger.info(f"Loaded MACE model from {self.model_path}")
        except ImportError:
            logger.error("MACE not installed")
            raise
    
    def evaluate(self, atoms: Atoms) -> Dict[str, float]:
        """Evaluate model on a structure."""
        atoms.calc = self.model
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        result = {
            'energy': energy,
            'forces': forces,
            'max_force': np.max(np.linalg.norm(forces, axis=1))
        }
        
        try:
            stress = atoms.get_stress()
            result['stress'] = stress
        except:
            pass
        
        return result
    
    def evaluate_dataset(self, atoms_list: List[Atoms]) -> pd.DataFrame:
        """Evaluate model on dataset."""
        results = []
        
        for i, atoms in enumerate(atoms_list):
            result = self.evaluate(atoms)
            result['index'] = i
            results.append(result)
        
        return pd.DataFrame(results)
    
    def compute_errors(self, atoms_list: List[Atoms],
                      energy_key: str = 'energy',
                      forces_key: str = 'forces') -> Dict:
        """Compute prediction errors."""
        energy_errors = []
        force_errors = []
        
        for atoms in atoms_list:
            result = self.evaluate(atoms)
            
            if energy_key in atoms.info:
                ref_energy = atoms.info[energy_key]
                pred_energy = result['energy']
                energy_errors.append(abs(pred_energy - ref_energy))
            
            if forces_key in atoms.arrays:
                ref_forces = atoms.arrays[forces_key]
                pred_forces = result['forces']
                force_errors.append(np.abs(pred_forces - ref_forces).flatten())
        
        errors = {
            'energy_rmse': np.sqrt(np.mean(np.array(energy_errors) ** 2)),
            'energy_mae': np.mean(energy_errors),
            'force_rmse': np.sqrt(np.mean(np.concatenate(force_errors) ** 2)),
            'force_mae': np.mean(np.concatenate(force_errors))
        }
        
        return errors


# Export public API
__all__ = [
    'MACEDataConfig',
    'MACEArchitectureConfig',
    'MACETrainingConfig',
    'MACERunConfig',
    'MACEDatasetPreparer',
    'MACETrainer',
    'MACEEvaluator'
]
