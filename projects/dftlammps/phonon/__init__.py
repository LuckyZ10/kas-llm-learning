"""
DFTLammps Phonon Module
========================

Comprehensive phonon calculation and analysis module.

This module provides tools for:
- Phonon dispersion and DOS calculations
- Thermodynamic property calculations
- Lattice thermal conductivity
- Quasi-harmonic approximation

Submodules:
-----------
phonopy_interface : Main interface for phonon calculations
    - Force constant calculations from DFT
    - Band structure and DOS
    - Visualization tools

thermal_properties : Thermodynamic properties
    - Heat capacity, entropy, free energy
    - Thermal expansion (QHA)
    - Debye temperature

lattice_thermal_conductivity : Thermal transport
    - Third-order force constants (Phono3py)
    - RTA and LBTE methods
    - Phonon lifetimes and mean free paths

Example Usage:
--------------
    from dftlammps.phonon import PhonopyInterface, PhononConfig
    from dftlammps.phonon import ThermalPropertyCalculator
    from dftlammps.phonon import LatticeThermalConductivity

    # Phonon calculation
    config = PhononConfig(structure_path='POSCAR', output_dir='./phonon')
    phonon = PhonopyInterface(config)
    phonon.run_full_phonon_calculation('POSCAR')
    
    # Thermal properties
    thermal_calc = ThermalPropertyCalculator()
    thermal_results = thermal_calc.calculate_from_phonopy(phonon.phonopy)
    
    # Thermal conductivity
    kappa_calc = LatticeThermalConductivity()
    kappa_results = kappa_calc.run_thermal_conductivity_rta()
"""

__version__ = "1.0.0"
__author__ = "DFTLammps Phonon Team"

# Import main classes
from .phonopy_interface import (
    PhonopyInterface,
    PhononConfig,
    PhononResults,
    DFTCode,
    IBRIONMode,
    create_phonopy_from_vasp,
    create_phonopy_from_ase
)

from .thermal_properties import (
    ThermalPropertyCalculator,
    ThermalConfig,
    ThermalResults,
    QHACalculator,
    calculate_thermal_properties_from_phonopy
)

from .lattice_thermal_conductivity import (
    LatticeThermalConductivity,
    ThermalConductivityConfig,
    ThermalConductivityResults,
    ConductivityMethod,
    calculate_thermal_conductivity_workflow
)

# Module exports
__all__ = [
    # Phonopy interface
    'PhonopyInterface',
    'PhononConfig',
    'PhononResults',
    'DFTCode',
    'IBRIONMode',
    'create_phonopy_from_vasp',
    'create_phonopy_from_ase',
    
    # Thermal properties
    'ThermalPropertyCalculator',
    'ThermalConfig',
    'ThermalResults',
    'QHACalculator',
    'calculate_thermal_properties_from_phonopy',
    
    # Thermal conductivity
    'LatticeThermalConductivity',
    'ThermalConductivityConfig',
    'ThermalConductivityResults',
    'ConductivityMethod',
    'calculate_thermal_conductivity_workflow'
]

# Check dependencies
def _check_dependencies():
    """Check for required dependencies."""
    missing = []
    
    try:
        import phonopy
    except ImportError:
        missing.append("phonopy")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    if missing:
        import warnings
        warnings.warn(
            f"Missing optional dependencies: {', '.join(missing)}. "
            "Some features may not work correctly."
        )

_check_dependencies()
