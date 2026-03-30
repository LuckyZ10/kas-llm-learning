"""
DFT-LAMMPS Correlated Systems Module

This module provides tools for studying strongly correlated electron systems,
including Dynamical Mean-Field Theory (DMFT), Hubbard U calculations, and
interfaces to specialized codes like TRIQS.

Submodules:
-----------
dmft_interface : DMFT implementation with VASP+Wannier90 integration
hubbard_u : Hubbard U calculation methods (linear response, cRPA, etc.)
triqs_interface : Interface to TRIQS for advanced many-body calculations

Example Usage:
--------------
>>> from dftlammps.correlated import DMFTEngine, DMFTConfig
>>> from dftlammps.correlated import LinearResponseU, HubbardUConfig
>>>
>>> # Setup DMFT calculation
>>> config = DMFTConfig(temperature=300, u_value=4.0, j_value=0.6)
>>> dmft = DMFTEngine(config)
>>> dmft.initialize(solver_type="triqs")
>>>
>>> # Run self-consistent DMFT loop
>>> results = dmft.run_scf_loop(H_k, k_weights)
>>>
>>> # Calculate spectral function
>>> omega, A_w = dmft.calculate_spectral_function(H_k, k_points, k_weights)
>>>
>>> # Calculate Hubbard U using linear response
>>> u_calc = LinearResponseU()
>>> u_results = u_calc.calculate_linear_response_u("POSCAR", [0, 1, 2, 3, 4])
"""

__version__ = "0.1.0"
__author__ = "DFT-LAMMPS Team"

# Import main classes from dmft_interface
from .dmft_interface import (
    DMFTConfig,
    WannierProjectorConfig,
    DMFTEngine,
    CTQMCSolver,
    WannierProjector,
    VASPDMFTInterface,
    ImpuritySolver,
    calculate_double_occupancy,
    calculate_kinetic_energy,
    estimate_nev_order_parameter,
    check_fermi_liquid,
    calculate_spectral_weight
)

# Import main classes from hubbard_u
from .hubbard_u import (
    HubbardUConfig,
    LinearResponseU,
    ConstrainedRPA,
    SelfConsistentU,
    DFTPlusUOptimizer,
    UDatabase,
    calculate_u_eff,
    estimate_u_from_ionicity,
    check_u_reasonableness
)

# Import main classes from triqs_interface (if TRIQS available)
try:
    from .triqs_interface import (
        TRIQSConfig,
        MultiOrbitalHubbard,
        TwoParticleGF,
        SuperconductingSusceptibility,
        MagneticSusceptibility,
        triqs_to_numpy,
        numpy_to_triqs,
        calculate_spectral_moment,
        check_sum_rules,
        estimate_bath_parameters
    )
    TRIQS_AVAILABLE = True
except ImportError:
    TRIQS_AVAILABLE = False

# Define what gets imported with "from dftlammps.correlated import *"
__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # DMFT interface
    'DMFTConfig',
    'WannierProjectorConfig',
    'DMFTEngine',
    'CTQMCSolver',
    'WannierProjector',
    'VASPDMFTInterface',
    'ImpuritySolver',
    
    # Hubbard U
    'HubbardUConfig',
    'LinearResponseU',
    'ConstrainedRPA',
    'SelfConsistentU',
    'DFTPlusUOptimizer',
    'UDatabase',
    'calculate_u_eff',
    'estimate_u_from_ionicity',
    'check_u_reasonableness',
    
    # Utility functions
    'calculate_double_occupancy',
    'calculate_kinetic_energy',
    'estimate_nev_order_parameter',
    'check_fermi_liquid',
    'calculate_spectral_weight',
]

# Add TRIQS classes if available
if TRIQS_AVAILABLE:
    __all__.extend([
        'TRIQSConfig',
        'MultiOrbitalHubbard',
        'TwoParticleGF',
        'SuperconductingSusceptibility',
        'MagneticSusceptibility',
        'triqs_to_numpy',
        'numpy_to_triqs',
        'calculate_spectral_moment',
        'check_sum_rules',
        'estimate_bath_parameters',
    ])

# Convenience function to check available features
def get_capabilities():
    """Return dictionary of available capabilities"""
    return {
        'dmft': True,
        'hubbard_u': True,
        'triqs': TRIQS_AVAILABLE,
        'vasp_interface': True,
        'wannier90': True,
        'ctqmc_solvers': ['triqs', 'alps', 'ipet', 'comctqmc'],
        'u_methods': ['linear_response', 'crpa', 'self_consistent', 'optimization'],
        'triqs_features': [
            'multi_orbital_hubbard',
            'two_particle_gf',
            'superconducting_susceptibility',
            'magnetic_susceptibility'
        ] if TRIQS_AVAILABLE else []
    }

# Quick start helper function
def setup_dmft_calculation(temperature: float = 300.0,
                           u_value: float = 4.0,
                           j_value: float = 0.6,
                           n_orbitals: int = 5,
                           solver_type: str = "triqs") -> DMFTEngine:
    """
    Quick setup for DMFT calculation
    
    Parameters:
    -----------
    temperature : float
        Temperature in Kelvin
    u_value : float
        Hubbard U in eV
    j_value : float
        Hund's coupling J in eV
    n_orbitals : int
        Number of correlated orbitals
    solver_type : str
        CT-QMC solver type
        
    Returns:
    --------
    dmft : DMFTEngine
        Configured DMFT engine
    """
    config = DMFTConfig(
        temperature=temperature,
        u_value=u_value,
        j_value=j_value,
        n_orbitals=n_orbitals
    )
    
    dmft = DMFTEngine(config)
    dmft.initialize(solver_type=solver_type)
    
    return dmft