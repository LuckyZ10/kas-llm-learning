#!/usr/bin/env python3
"""
Unified MLIP Interface
======================

Unified interface for training and using various ML interatomic potentials:
- MACE (Message-passing, E(3)-equivariant)
- CHGNet (Universal GNN potential)
- Orb (Orbital Materials graph potential)
- DeepMD (Deep Potential)
- NEP (Neuroevolution Potential)

Provides a consistent API regardless of underlying implementation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import logging

# ASE
from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator, all_changes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLIPType(Enum):
    """Supported MLIP types."""
    MACE = "mace"
    CHGNET = "chgnet"
    ORB = "orb"
    DEEPMD = "deepmd"
    NEP = "nep"


@dataclass
class UnifiedTrainingConfig:
    """Unified configuration for MLIP training.
    
    Attributes:
        mlip_type: Type of MLIP to train
        train_data: Path to training data (extxyz)
        valid_data: Path to validation data
        test_data: Path to test data
        cutoff: Interaction cutoff distance
        max_epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        device: Training device ('cuda' or 'cpu')
        output_dir: Output directory for models
    """
    mlip_type: MLIPType
    train_data: str
    valid_data: str
    test_data: Optional[str] = None
    cutoff: float = 5.0
    max_epochs: int = 1000
    batch_size: int = 8
    learning_rate: float = 0.001
    device: str = "cuda"
    output_dir: str = "./mlip_models"
    
    # MACE-specific
    hidden_irreps: str = "128x0e + 128x1o"
    num_interactions: int = 2
    
    # CHGNet-specific
    use_pretrained_chgnet: bool = True
    
    # Orb-specific
    orb_model_size: str = "medium"


@dataclass
class PredictionResult:
    """Standardized prediction result."""
    energy: float
    forces: np.ndarray
    stress: Optional[np.ndarray] = None
    magmom: Optional[np.ndarray] = None
    uncertainty: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'energy': float(self.energy),
            'forces': self.forces.tolist() if isinstance(self.forces, np.ndarray) else self.forces,
            'stress': self.stress.tolist() if self.stress is not None else None,
            'magmom': self.magmom.tolist() if self.magmom is not None else None,
            'uncertainty': self.uncertainty
        }


class BaseMLIP(ABC):
    """Abstract base class for MLIPs."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the MLIP model."""
        pass
    
    @abstractmethod
    def predict(self, atoms: Atoms) -> PredictionResult:
        """Make prediction on atoms."""
        pass
    
    @abstractmethod
    def get_calculator(self) -> Calculator:
        """Get ASE calculator."""
        pass
    
    def predict_batch(self, atoms_list: List[Atoms]) -> List[PredictionResult]:
        """Make predictions on batch of structures."""
        return [self.predict(atoms) for atoms in atoms_list]
    
    def compute_errors(self, atoms_list: List[Atoms],
                      ref_energies: Optional[List[float]] = None,
                      ref_forces: Optional[List[np.ndarray]] = None) -> Dict:
        """Compute prediction errors."""
        errors = {
            'energy_errors': [],
            'force_errors': []
        }
        
        for i, atoms in enumerate(atoms_list):
            pred = self.predict(atoms)
            
            if ref_energies and i < len(ref_energies):
                errors['energy_errors'].append(abs(pred.energy - ref_energies[i]))
            
            if ref_forces and i < len(ref_forces):
                force_diff = np.abs(pred.forces - ref_forces[i])
                errors['force_errors'].extend(force_diff.flatten())
        
        result = {}
        if errors['energy_errors']:
            result['energy_mae'] = np.mean(errors['energy_errors'])
            result['energy_rmse'] = np.sqrt(np.mean(np.array(errors['energy_errors']) ** 2))
        
        if errors['force_errors']:
            result['force_mae'] = np.mean(errors['force_errors'])
            result['force_rmse'] = np.sqrt(np.mean(np.array(errors['force_errors']) ** 2))
        
        return result


class MACEMLIP(BaseMLIP):
    """MACE MLIP wrapper."""
    
    def _load_model(self):
        """Load MACE model."""
        try:
            from mace.calculators import MACECalculator
            
            if self.model_path:
                self.model = MACECalculator(
                    model_paths=self.model_path,
                    device='cuda'
                )
            else:
                logger.warning("No model path provided for MACE")
        except ImportError:
            logger.error("MACE not installed")
            self.model = None
    
    def predict(self, atoms: Atoms) -> PredictionResult:
        """Make prediction."""
        if self.model is None:
            raise RuntimeError("MACE model not loaded")
        
        atoms.calc = self.model
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        try:
            stress = atoms.get_stress(voigt=True)
        except:
            stress = None
        
        return PredictionResult(
            energy=energy,
            forces=forces,
            stress=stress
        )
    
    def get_calculator(self) -> Calculator:
        """Get ASE calculator."""
        return self.model


class CHGNetMLIP(BaseMLIP):
    """CHGNet MLIP wrapper."""
    
    def _load_model(self):
        """Load CHGNet model."""
        try:
            from chgnet.model import CHGNet
            
            if self.model_path:
                self.model = CHGNet.load(self.model_path)
            else:
                self.model = CHGNet.load()
            
            self.calculator = self.model
        except ImportError:
            logger.error("CHGNet not installed")
            self.model = None
            self.calculator = None
    
    def predict(self, atoms: Atoms) -> PredictionResult:
        """Make prediction."""
        if self.model is None:
            raise RuntimeError("CHGNet model not loaded")
        
        from chgnet.model import Struct
        
        structure = Struct(
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc
        )
        
        pred = self.model.predict_structure(structure)
        
        return PredictionResult(
            energy=pred['e'],
            forces=pred['f'],
            magmom=pred.get('m')
        )
    
    def get_calculator(self) -> Calculator:
        """Get ASE calculator."""
        try:
            from chgnet.model import CHGNetCalculator
            return CHGNetCalculator(self.model)
        except:
            return None


class OrbMLIP(BaseMLIP):
    """Orb MLIP wrapper."""
    
    def _load_model(self):
        """Load Orb model."""
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import OrbCalc
            
            if self.model_path:
                import torch
                self.model = torch.load(self.model_path)
            else:
                self.model = pretrained.ORB_PRETRAINED_MODELS['orb-v1-20240827']()
            
            self.calculator = OrbCalc(self.model)
        except ImportError:
            logger.error("Orb not installed")
            self.model = None
            self.calculator = None
    
    def predict(self, atoms: Atoms) -> PredictionResult:
        """Make prediction."""
        if self.calculator is None:
            raise RuntimeError("Orb calculator not loaded")
        
        atoms.calc = self.calculator
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        return PredictionResult(
            energy=energy,
            forces=forces
        )
    
    def get_calculator(self) -> Calculator:
        """Get ASE calculator."""
        return self.calculator


class UnifiedMLIPTrainer:
    """Unified trainer for all MLIP types."""
    
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict:
        """Run training based on MLIP type."""
        logger.info(f"Training {self.config.mlip_type.value} model")
        
        if self.config.mlip_type == MLIPType.MACE:
            return self._train_mace()
        elif self.config.mlip_type == MLIPType.CHGNET:
            return self._train_chgnet()
        elif self.config.mlip_type == MLIPType.ORB:
            return self._train_orb()
        else:
            raise ValueError(f"Unsupported MLIP type: {self.config.mlip_type}")
    
    def _train_mace(self) -> Dict:
        """Train MACE model."""
        from .mace_training import (
            MACETrainer, MACERunConfig,
            MACEDataConfig, MACEArchitectureConfig, MACETrainingConfig
        )
        
        data_config = MACEDataConfig(
            train_file=self.config.train_data,
            valid_file=self.config.valid_data,
            test_file=self.config.test_data,
            cutoff=self.config.cutoff
        )
        
        arch_config = MACEArchitectureConfig(
            hidden_irreps=self.config.hidden_irreps,
            num_interactions=self.config.num_interactions
        )
        
        train_config = MACETrainingConfig(
            max_num_epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            lr=self.config.learning_rate,
            model_dir=str(self.output_dir),
            device=self.config.device
        )
        
        run_config = MACERunConfig(
            data=data_config,
            architecture=arch_config,
            training=train_config
        )
        
        trainer = MACETrainer(run_config)
        return trainer.run_training()
    
    def _train_chgnet(self) -> Dict:
        """Train CHGNet model."""
        from .chgnet_training import (
            CHGNetTrainer, CHGNetRunConfig,
            CHGNetDataConfig, CHGNetArchitectureConfig, CHGNetTrainingConfig
        )
        
        data_config = CHGNetDataConfig(
            train_file=self.config.train_data,
            valid_file=self.config.valid_data,
            test_file=self.config.test_data,
            cutoff=self.config.cutoff
        )
        
        arch_config = CHGNetArchitectureConfig()
        
        train_config = CHGNetTrainingConfig(
            epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            lr=self.config.learning_rate,
            save_dir=str(self.output_dir),
            device=self.config.device
        )
        
        run_config = CHGNetRunConfig(
            data=data_config,
            architecture=arch_config,
            training=train_config
        )
        
        trainer = CHGNetTrainer(run_config)
        return trainer.run_training()
    
    def _train_orb(self) -> Dict:
        """Train Orb model."""
        from .orb_training import (
            OrbTrainer, OrbRunConfig,
            OrbDataConfig, OrbArchitectureConfig, OrbTrainingConfig
        )
        
        data_config = OrbDataConfig(
            train_file=self.config.train_data,
            valid_file=self.config.valid_data,
            test_file=self.config.test_data,
            cutoff=self.config.cutoff
        )
        
        arch_config = OrbArchitectureConfig()
        
        train_config = OrbTrainingConfig(
            epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            lr=self.config.learning_rate,
            save_dir=str(self.output_dir),
            device=self.config.device
        )
        
        run_config = OrbRunConfig(
            data=data_config,
            architecture=arch_config,
            training=train_config
        )
        
        trainer = OrbTrainer(run_config)
        return trainer.run_training()


class UnifiedMLIPCalculator(Calculator):
    """Unified ASE calculator for MLIPs."""
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, mlip_type: MLIPType,
                 model_path: Optional[str] = None,
                 **kwargs):
        """Initialize calculator.
        
        Args:
            mlip_type: Type of MLIP
            model_path: Path to trained model
        """
        super().__init__(**kwargs)
        
        self.mlip_type = mlip_type
        self.model_path = model_path
        self.mlip = self._create_mlip()
    
    def _create_mlip(self) -> BaseMLIP:
        """Create MLIP instance."""
        if self.mlip_type == MLIPType.MACE:
            return MACEMLIP(self.model_path)
        elif self.mlip_type == MLIPType.CHGNET:
            return CHGNetMLIP(self.model_path)
        elif self.mlip_type == MLIPType.ORB:
            return OrbMLIP(self.model_path)
        else:
            raise ValueError(f"Unsupported MLIP type: {self.mlip_type}")
    
    def calculate(self, atoms: Atoms = None,
                 properties: List[str] = None,
                 system_changes: List[str] = None):
        """Calculate properties."""
        if properties is None:
            properties = ['energy', 'forces']
        
        Calculator.calculate(self, atoms, properties, system_changes)
        
        result = self.mlip.predict(atoms)
        
        self.results['energy'] = result.energy
        self.results['forces'] = result.forces
        
        if result.stress is not None:
            self.results['stress'] = result.stress


def quick_train(train_file: str,
               valid_file: str,
               mlip_type: str = "mace",
               output_dir: str = "./quick_train",
               epochs: int = 100) -> Dict:
    """Quick training function.
    
    Args:
        train_file: Training data in extxyz format
        valid_file: Validation data
        mlip_type: 'mace', 'chgnet', or 'orb'
        output_dir: Output directory
        epochs: Training epochs
    
    Returns:
        Training results dictionary
    """
    config = UnifiedTrainingConfig(
        mlip_type=MLIPType(mlip_type),
        train_data=train_file,
        valid_data=valid_file,
        max_epochs=epochs,
        output_dir=output_dir
    )
    
    trainer = UnifiedMLIPTrainer(config)
    return trainer.train()


def load_model(mlip_type: str,
              model_path: Optional[str] = None) -> BaseMLIP:
    """Load trained MLIP model.
    
    Args:
        mlip_type: 'mace', 'chgnet', or 'orb'
        model_path: Path to model file
    
    Returns:
        Loaded MLIP instance
    """
    mlip_enum = MLIPType(mlip_type)
    
    if mlip_enum == MLIPType.MACE:
        return MACEMLIP(model_path)
    elif mlip_enum == MLIPType.CHGNET:
        return CHGNetMLIP(model_path)
    elif mlip_enum == MLIPType.ORB:
        return OrbMLIP(model_path)
    else:
        raise ValueError(f"Unknown MLIP type: {mlip_type}")


# Export public API
__all__ = [
    'MLIPType',
    'UnifiedTrainingConfig',
    'PredictionResult',
    'BaseMLIP',
    'UnifiedMLIPTrainer',
    'UnifiedMLIPCalculator',
    'quick_train',
    'load_model'
]
