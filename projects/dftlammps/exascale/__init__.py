#!/usr/bin/env python3
"""
dftlammps/exascale/__init__.py - Exascale computing module

This module provides extreme-scale capabilities for DFT and MD simulations:
- Million-atom linear-scaling DFT
- Extreme conditions MD (shock, high P/T)
- GPU-accelerated linear algebra
- Domain decomposition parallelism

Author: DFT-LAMMPS Team
Version: 1.0.0
"""

from .exascale_dft import (
    ExascaleDFT,
    ExascaleDFTConfig,
    LinearScalingMethod,
    DomainDecomposition,
    GPULinearAlgebra,
    LocalizedOrbital,
    LinearScalingHamiltonian
)

from .extreme_md import (
    ExtremeMD,
    ExtremeMDConfig,
    ShockWaveSimulator,
    PhaseTransitionDetector,
    ShockMethod,
    PhaseTransitionMethod
)

from .applications import (
    PlanetaryCoreSimulator,
    NuclearMaterialSimulator,
    AsteroidImpactSimulator,
    PlanetaryCoreConfig,
    NuclearMaterialConfig,
    AsteroidImpactConfig
)

__version__ = "1.0.0"

__all__ = [
    # Exascale DFT
    'ExascaleDFT',
    'ExascaleDFTConfig',
    'LinearScalingMethod',
    'DomainDecomposition',
    'GPULinearAlgebra',
    'LocalizedOrbital',
    'LinearScalingHamiltonian',
    
    # Extreme MD
    'ExtremeMD',
    'ExtremeMDConfig',
    'ShockWaveSimulator',
    'PhaseTransitionDetector',
    'ShockMethod',
    'PhaseTransitionMethod',
    
    # Applications
    'PlanetaryCoreSimulator',
    'NuclearMaterialSimulator',
    'AsteroidImpactSimulator',
    'PlanetaryCoreConfig',
    'NuclearMaterialConfig',
    'AsteroidImpactConfig'
]


def get_module_info():
    """Get module information"""
    return {
        'name': 'dftlammps.exascale',
        'version': __version__,
        'description': 'Exascale computing for DFT and MD simulations',
        'capabilities': [
            'Million-atom linear-scaling DFT',
            'GPU-accelerated linear algebra',
            'Domain decomposition (MPI)',
            'Shock wave simulation',
            'Phase transition detection',
            'Extreme conditions (high P/T)'
        ]
    }
