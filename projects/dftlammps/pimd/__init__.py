"""
Path Integral Molecular Dynamics (PIMD) Module
==============================================

This module provides path integral molecular dynamics capabilities
for simulating nuclear quantum effects in materials.

The PIMD approach uses the isomorphism between quantum statistics
and classical ring polymers to exactly sample the quantum canonical
distribution at finite temperature.

Key Capabilities:
-----------------
- Path Integral Molecular Dynamics (PIMD)
- Ring Polymer Molecular Dynamics (RPMD)
- Centroid Molecular Dynamics (CMD)
- Thermostatted RPMD (TRPMD)
- Zero-point energy calculations
- Quantum diffusion coefficients
- Isotope effect calculations
- Multiple kinetic energy estimators

Submodules:
-----------
- ipi_interface: Interface to i-PI code
- pimd_properties: Quantum property calculations

Example Usage:
--------------
    >>> from dftlammps.pimd import IPIConfig, PIMDSimulation, RPMDSimulation
    >>> from ase import Atoms
    >>> 
    >>> # Setup configuration
    >>> config = IPIConfig(
    ...     n_beads=32,
    ...     temperature=300.0,
    ...     timestep=0.5,
    ...     n_steps=10000
    ... )
    >>> 
    >>> # Create simulation
    >>> sim = PIMDSimulation(config)
    >>> 
    >>> # Run with initial structure
    >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> results = sim.run(atoms, driver_cmd=['lmp', '-in', 'input.lmp'])
    >>> 
    >>> # Analyze results
    >>> from dftlammps.pimd import QuantumPropertyCalculator
    >>> calc = QuantumPropertyCalculator(results)
    >>> zpe = calc.calculate_zpe()
    >>> diffusion = calc.calculate_diffusion()

References:
-----------
- Tuckerman (2010). Statistical Mechanics: Theory and Molecular Simulation
- Ceriotti et al. (2010). Efficient calculation of free energy surfaces
- Marx & Parrinello (1996). Ab initio path integral molecular dynamics
- Craig & Manolopoulos (2004). Ring polymer molecular dynamics
"""

# i-PI Interface imports
from .ipi_interface import (
    # Configuration
    IPIConfig,
    IPIMode,
    ThermostatType,
    IntegratorType,
    
    # Results
    PIMDResults,
    
    # Main classes
    IPISocket,
    IPIInterface,
    PIMDSimulation,
    RPMDSimulation,
    TRPMDSimulation,
    PIConvergenceChecker,
    
    # Functions
    generate_input_xml,
    run_pimd,
    run_rpmd,
    estimate_required_beads,
)

# Property calculation imports
from .pimd_properties import (
    # Configuration
    EstimatorType,
    
    # Results
    ZPEResults,
    DiffusionResults,
    IsotopeResults,
    KineticEnergyResults,
    
    # Calculators
    KineticEnergyEstimator,
    ZeroPointEnergyCalculator,
    QuantumDiffusionCalculator,
    IsotopeEffectCalculator,
    QuantumPropertyCalculator,
    
    # Functions
    calculate_zpe,
    calculate_quantum_diffusion,
    calculate_isotope_fractionation,
    get_virial_estimator,
    get_centroid_virial_estimator,
)

__all__ = [
    # Enums
    'IPIMode',
    'ThermostatType',
    'IntegratorType',
    'EstimatorType',
    
    # Configuration
    'IPIConfig',
    
    # Results
    'PIMDResults',
    'ZPEResults',
    'DiffusionResults',
    'IsotopeResults',
    'KineticEnergyResults',
    
    # Main Classes
    'IPISocket',
    'IPIInterface',
    'PIMDSimulation',
    'RPMDSimulation',
    'TRPMDSimulation',
    'PIConvergenceChecker',
    
    # Property Calculators
    'KineticEnergyEstimator',
    'ZeroPointEnergyCalculator',
    'QuantumDiffusionCalculator',
    'IsotopeEffectCalculator',
    'QuantumPropertyCalculator',
    
    # Functions
    'generate_input_xml',
    'run_pimd',
    'run_rpmd',
    'estimate_required_beads',
    'calculate_zpe',
    'calculate_quantum_diffusion',
    'calculate_isotope_fractionation',
    'get_virial_estimator',
    'get_centroid_virial_estimator',
]

# Version
__version__ = '1.0.0'
