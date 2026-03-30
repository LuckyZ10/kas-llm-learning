"""
DFTLAMMPS Multiscale Coupling Module
=====================================

This module provides tools for multiscale modeling from quantum to continuum scales:
- QM/MM coupling (VASP + LAMMPS)
- Machine learning coarse-graining
- Graph neural networks for cross-scale modeling
- Validation and analysis tools

Author: DFTLAMMPS Team
"""

__version__ = "0.1.0"
__author__ = "DFTLAMMPS Team"

from .qmmm import QMMMInterface, VASPLAMMPSCoupling
from .ml_cg import CoarseGrainer, MLCGMapping
from .gnn_models import CGGNN, MultiscaleGNN
from .validation import CrossScaleValidator, EnergyConsistencyCheck

__all__ = [
    'QMMMInterface',
    'VASPLAMMPSCoupling', 
    'CoarseGrainer',
    'MLCGMapping',
    'CGGNN',
    'MultiscaleGNN',
    'CrossScaleValidator',
    'EnergyConsistencyCheck'
]
