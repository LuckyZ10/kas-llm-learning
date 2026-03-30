"""
Physics-AI Module for DFT-LAMMPS Pipeline
=========================================

This module integrates physics constraints deeply into AI models for 
molecular dynamics simulations and potential energy surface fitting.

Key Components:
- Physics-Informed Neural Networks (PINNs)
- Neural Operators (DeepONet, FNO)
- Physics-Informed Graph Neural Networks
- Conservation Law Constraints
- Symbolic Regression Engine
- Physics Law Validators
- MD Potential Energy Surface Integration

Author: Phase 64 Research
"""

__version__ = "1.0.0"
__author__ = "Phase 64 Research Team"

from .core.physics_layer import PhysicsConstraintLayer
from .core.conservation import ConservationLaw
from .models.pinns import PhysicsInformedNN
from .models.deeponet import DeepONet
from .models.fno import FourierNeuralOperator
from .models.physics_gnn import PhysicsInformedGNN
from .symbolic.regression_engine import SymbolicRegressionEngine
from .validators.physics_validator import PhysicsLawValidator
from .integration.md_potential import MDPotentialFitter

__all__ = [
    'PhysicsConstraintLayer',
    'ConservationLaw',
    'PhysicsInformedNN',
    'DeepONet',
    'FourierNeuralOperator',
    'PhysicsInformedGNN',
    'SymbolicRegressionEngine',
    'PhysicsLawValidator',
    'MDPotentialFitter',
]
