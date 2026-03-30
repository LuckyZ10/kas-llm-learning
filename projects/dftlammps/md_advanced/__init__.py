"""
DFT+LAMMPS Advanced MD Module
=============================

Advanced molecular dynamics simulation capabilities including:
- Enhanced sampling methods (Umbrella Sampling, Metadynamics, REMD, TAD)
- Free energy calculations (FEP, TI, BAR, WHAM)
- Rare event analysis (NEB, String Method, Dimer Method, TST)
- Reaction analysis (path search, rate constants, KMC)

Modules
-------
enhanced_sampling.py
    Umbrella Sampling, Metadynamics (PLUMED), REMD, TAD

free_energy.py
    FEP, TI, BAR, WHAM, MBAR

rare_events.py
    NEB (ASE/LAMMPS/VASP), String Method, Dimer Method, TST

reaction_analysis.py
    Automatic path search, reaction coordinates, rate constants, KMC

Example Usage
-------------
>>> from dftlammps.md_advanced import UmbrellaSampling, UmbrellaSamplingConfig
>>> config = UmbrellaSamplingConfig(
...     collective_variables=[{'name': 'dist', 'type': 'distance', 'group1': [0], 'group2': [1]}],
...     reaction_path=[[2.0], [3.0]],
...     n_windows=10
... )
>>> us = UmbrellaSampling(config)
>>> results = us.run_all_windows(atoms)

>>> from dftlammps.md_advanced import NEB, NEBConfig
>>> config = NEBConfig(n_images=10, climb=True)
>>> neb = NEB(config)
>>> result = neb.run(initial_atoms, final_atoms)
"""

from .enhanced_sampling import (
    UmbrellaSamplingConfig,
    MetadynamicsConfig,
    REMDConfig,
    TADConfig,
    CollectiveVariable,
    DistanceCV,
    AngleCV,
    DihedralCV,
    CoordinationNumberCV,
    UmbrellaSampling,
    Metadynamics,
    REMD,
    TemperatureAcceleratedDynamics,
    estimate_free_energy_barrier,
    compute_error_pmf
)

from .free_energy import (
    FEPConfig,
    TIConfig,
    BARConfig,
    WHAMConfig,
    FreeEnergyPerturbation,
    ThermodynamicIntegration,
    BennettAcceptanceRatio,
    WHAM,
    MBAR,
    FreeEnergyAnalyzer,
    compute_solvation_free_energy,
    compute_binding_free_energy
)

from .rare_events import (
    NEBConfig,
    StringMethodConfig,
    DimerConfig,
    TSTConfig,
    NEB,
    StringMethod,
    DimerMethod,
    TransitionStateTheory,
    find_mep_from_ts,
    compute_activation_entropy
)

from .reaction_analysis import (
    ReactionPathSearchConfig,
    ReactionCoordinateConfig,
    RateConstantConfig,
    KMCConfig,
    ReactionPathSearcher,
    ReactionCoordinate,
    RateConstantCalculator,
    KMCPreprocessor,
    ReactionNetwork
)

__all__ = [
    # Enhanced Sampling
    'UmbrellaSamplingConfig',
    'MetadynamicsConfig',
    'REMDConfig',
    'TADConfig',
    'CollectiveVariable',
    'DistanceCV',
    'AngleCV',
    'DihedralCV',
    'CoordinationNumberCV',
    'UmbrellaSampling',
    'Metadynamics',
    'REMD',
    'TemperatureAcceleratedDynamics',
    'estimate_free_energy_barrier',
    'compute_error_pmf',
    
    # Free Energy
    'FEPConfig',
    'TIConfig',
    'BARConfig',
    'WHAMConfig',
    'FreeEnergyPerturbation',
    'ThermodynamicIntegration',
    'BennettAcceptanceRatio',
    'WHAM',
    'MBAR',
    'FreeEnergyAnalyzer',
    'compute_solvation_free_energy',
    'compute_binding_free_energy',
    
    # Rare Events
    'NEBConfig',
    'StringMethodConfig',
    'DimerConfig',
    'TSTConfig',
    'NEB',
    'StringMethod',
    'DimerMethod',
    'TransitionStateTheory',
    'find_mep_from_ts',
    'compute_activation_entropy',
    
    # Reaction Analysis
    'ReactionPathSearchConfig',
    'ReactionCoordinateConfig',
    'RateConstantConfig',
    'KMCConfig',
    'ReactionPathSearcher',
    'ReactionCoordinate',
    'RateConstantCalculator',
    'KMCPreprocessor',
    'ReactionNetwork'
]
