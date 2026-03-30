"""
Accelerated Dynamics Module
===========================

This module provides methods for accelerating molecular dynamics simulations
to access longer timescales and rare events that are not accessible by
direct MD.

Methods:
--------
1. Hyperdynamics: Adds bias potential to accelerate barrier crossing
   - Bond-boost: Accelerates bond-breaking reactions
   - Coordinate-boost: Boosts along reaction coordinates
   - SIS: Self-learning hyperdynamics

2. Kinetic Monte Carlo (KMC): State-to-state dynamics with rates from MD
   - Gillespie algorithm for event selection
   - Rate catalog management
   - Defect evolution tracking
   - Parallel KMC (bKLMC)

Submodules:
-----------
- hyperdynamics: Hyperdynamics bias potential methods
- kmc_interface: Kinetic Monte Carlo simulation interface

Example Usage:
--------------
    # Hyperdynamics
    >>> from dftlammps.accelerated_dynamics import (
    ...     HyperdynamicsConfig, BondBoostPotential, HyperdynamicsSimulation
    ... )
    >>> config = HyperdynamicsConfig(
    ...     boost_method='bond_boost',
    ...     q_cutoff=0.2,
    ...     delta_v_max=1.0,
    ...     temperature=300.0
    ... )
    >>> sim = HyperdynamicsSimulation(config)
    >>> results = sim.run(atoms, n_steps=100000, timestep=1.0)
    >>> print(f"Boost factor: {results.boost_factor:.1f}x")
    
    # KMC
    >>> from dftlammps.accelerated_dynamics import (
    ...     KMCConfig, RateCatalog, RateProcess, KMCSimulator
    ... )
    >>> # Build rate catalog from MD data
    >>> catalog = RateCatalog()
    >>> catalog.add_process(RateProcess(
    ...     name='vacancy_hop',
    ...     initial_state='state_0',
    ...     final_state='state_1',
    ...     rate=1e10,
    ...     activation_energy=0.5
    ... ))
    >>> # Run KMC
    >>> config = KMCConfig(temperature=300.0, n_steps=100000)
    >>> kmc = KMCSimulator(config, catalog)
    >>> results = kmc.run(initial_state)
    >>> print(f"Simulated time: {results.total_time:.2e} s")

References:
-----------
- Voter (1997). J. Chem. Phys. 106, 4665 (Hyperdynamics)
- Miron & Fichthorn (2003). J. Chem. Phys. 119, 6210 (Bond-boost)
- Gillespie (1976). J. Comput. Phys. 22, 403 (KMC)
- Chatterjee & Vlachos (2007). J. Comput. Phys. 2, 179 (KMC review)
"""

# Hyperdynamics imports
from .hyperdynamics import (
    # Enums
    BoostMethod,
    TransitionDetectionMethod,
    
    # Configuration and Results
    HyperdynamicsConfig,
    BoostResults,
    
    # Bias Potentials
    BiasPotential,
    BondBoostPotential,
    CoordinateBoostPotential,
    SISHyperdynamics,
    
    # Analysis
    BoostFactorAnalyzer,
    
    # Main Simulation
    HyperdynamicsSimulation,
    
    # Functions
    estimate_boost_factor,
    calculate_accelerated_time,
    construct_bias_potential,
)

# KMC imports
from .kmc_interface import (
    # Enums
    ProcessType,
    KMCAlgorithm,
    
    # Configuration and Results
    KMCConfig,
    RateProcess,
    State,
    KMCSimulationState,
    KMCResults,
    
    # Main Classes
    RateCatalog,
    KMCSimulator,
    RateExtractor,
    DefectTracker,
    
    # Functions
    extract_rates_from_md,
    run_kmc,
    analyze_defect_evolution,
    calculate_mc_time,
)

__all__ = [
    # Enums
    'BoostMethod',
    'TransitionDetectionMethod',
    'ProcessType',
    'KMCAlgorithm',
    
    # Configuration
    'HyperdynamicsConfig',
    'KMCConfig',
    
    # Results
    'BoostResults',
    'RateProcess',
    'State',
    'KMCSimulationState',
    'KMCResults',
    
    # Hyperdynamics Classes
    'BiasPotential',
    'BondBoostPotential',
    'CoordinateBoostPotential',
    'SISHyperdynamics',
    'BoostFactorAnalyzer',
    'HyperdynamicsSimulation',
    
    # KMC Classes
    'RateCatalog',
    'KMCSimulator',
    'RateExtractor',
    'DefectTracker',
    
    # Functions
    'estimate_boost_factor',
    'calculate_accelerated_time',
    'construct_bias_potential',
    'extract_rates_from_md',
    'run_kmc',
    'analyze_defect_evolution',
    'calculate_mc_time',
]

# Version
__version__ = '1.0.0'
