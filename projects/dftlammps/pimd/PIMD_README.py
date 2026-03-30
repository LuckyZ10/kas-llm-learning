"""
PIMD and Accelerated Dynamics Module Documentation
==================================================

This package provides comprehensive tools for path integral molecular dynamics
and accelerated sampling methods for rare events in materials simulations.

Modules
-------

1. dftlammps.pimd - Path Integral Molecular Dynamics
   
   Core functionality for simulating nuclear quantum effects:
   
   a) ipi_interface.py
      - IPIConfig: Configuration for i-PI simulations
      - PIMDSimulation: Path Integral MD
      - RPMDSimulation: Ring Polymer MD
      - TRPMDSimulation: Thermostatted RPMD
      - PIConvergenceChecker: Bead convergence analysis
      - run_pimd(), run_rpmd(): Convenience functions
   
   b) pimd_properties.py
      - QuantumPropertyCalculator: Main property calculator
      - ZeroPointEnergyCalculator: ZPE calculations
      - QuantumDiffusionCalculator: Quantum diffusion
      - IsotopeEffectCalculator: Isotope effects
      - KineticEnergyEstimator: Multiple KE estimators
      - calculate_zpe(), calculate_quantum_diffusion(): Functions
   
   c) proton_transfer_example.py
      - Proton transfer with NQE
      - Grotthuss mechanism analysis
      - Kinetic isotope effects

2. dftlammps.accelerated_dynamics - Accelerated Sampling Methods
   
   Methods for accessing long timescales and rare events:
   
   a) hyperdynamics.py
      - HyperdynamicsConfig: Configuration
      - BondBoostPotential: Bond-boost method
      - CoordinateBoostPotential: CV-based boost
      - SISHyperdynamics: Self-learning method
      - BoostFactorAnalyzer: Boost analysis
      - HyperdynamicsSimulation: Main simulation
   
   b) kmc_interface.py
      - KMCConfig: Configuration
      - RateCatalog: Rate process management
      - KMCSimulator: Main KMC engine
      - RateExtractor: Extract rates from MD
      - DefectTracker: Track defect evolution
      - run_kmc(): Convenience function
   
   c) vacancy_diffusion_example.py
      - Vacancy diffusion in metals
      - Hyperdynamics vs KMC comparison
      - Long timescale simulations
   
   d) catalytic_reaction_example.py
      - Surface catalytic reactions
      - CO oxidation example
      - Microkinetic modeling

Key Features
------------

Path Integral Molecular Dynamics:
- PIMD for exact quantum statistics
- RPMD for approximate quantum dynamics
- Multiple kinetic energy estimators (primitive, virial, centroid-virial)
- Zero-point energy calculations
- Quantum diffusion coefficients
- Isotope effect calculations
- Convergence checking with bead number

Hyperdynamics:
- Bond-boost for bond-breaking reactions
- Coordinate-boost for CV-based acceleration
- SIS (self-learning) hyperdynamics
- Automatic boost factor calculation
- Transition detection
- Time acceleration analysis

Kinetic Monte Carlo:
- Gillespie algorithm
- Rate catalog management
- State-to-state dynamics
- Defect evolution tracking
- Rate extraction from MD
- Parallel KMC support

Applications
------------

1. Proton Transfer
   - Nuclear quantum effects in proton conduction
   - Grotthuss mechanism
   - Kinetic isotope effects (H/D)
   
   Example:
   >>> from dftlammps.pimd import PIMDSimulation, IPIConfig
   >>> config = IPIConfig(n_beads=32, temperature=300)
   >>> sim = PIMDSimulation(config)
   >>> results = sim.run(atoms)

2. Vacancy Diffusion
   - Long timescale diffusion in metals
   - Activation energy calculations
   - Correlation effects
   
   Example:
   >>> from dftlammps.accelerated_dynamics import HyperdynamicsSimulation
   >>> sim = HyperdynamicsSimulation(config)
   >>> results = sim.run(atoms, n_steps=1000000)

3. Catalytic Reactions
   - Surface reaction mechanisms
   - Turnover frequencies
   - Selectivity analysis
   
   Example:
   >>> from dftlammps.accelerated_dynamics import KMCSimulator
   >>> kmc = KMCSimulator(config, rate_catalog)
   >>> results = kmc.run(initial_state)

Theory Background
-----------------

Path Integral Formalism:
The quantum canonical partition function can be written as:

    Z = Tr[exp(-βH)] = ∫ dr⟨r|exp(-βH)|r⟩

Using the Trotter decomposition:
    exp(-βH) ≈ [exp(-βH/P)]^P

This maps a quantum particle onto a classical ring polymer with P beads.

Kinetic Energy Estimators:
1. Primitive: K_P = (3N/2)Pk_BT - (mP/2ℏ²β²)Σ(r_i - r_{i+1})²
2. Virial: K_V = (3N/2β) + (1/2P)Σ r_i · F_i
3. Centroid Virial: K_CV = (3N/2β) + (1/2P)Σ(r_i - r_c) · F_i

Hyperdynamics:
The boost factor is given by:
    t_acc = t_sim × ⟨exp(V_bias/k_BT)⟩

Kinetic Monte Carlo:
The Gillespie algorithm selects events with probability proportional
to their rate constant and advances time by:
    Δt = -ln(u)/Σk_i

References
----------

PIMD/RPMD:
- Tuckerman (2010). Statistical Mechanics: Theory and Molecular Simulation
- Marx & Parrinello (1996). J. Chem. Phys. 104, 4077
- Craig & Manolopoulos (2004). J. Chem. Phys. 121, 3368

Hyperdynamics:
- Voter (1997). J. Chem. Phys. 106, 4665
- Miron & Fichthorn (2003). J. Chem. Phys. 119, 6210
- Hamelberg et al. (2004). J. Chem. Phys. 120, 11919

KMC:
- Gillespie (1976). J. Comput. Phys. 22, 403
- Chatterjee & Vlachos (2007). J. Comput. Phys. 2, 179
- Shin et al. (2021). Phys. Rev. Materials 5, L040801

Isotope Effects:
- Wolfsberg et al. (2010). Isotope Effects
- Ceriotti & Markland (2013). J. Chem. Phys. 138, 014112

Quick Start
-----------

1. Run PIMD simulation:
   
   from dftlammps.pimd import IPIConfig, PIMDSimulation
   
   config = IPIConfig(
       n_beads=32,
       temperature=300.0,
       timestep=0.5,
       n_steps=10000
   )
   
   sim = PIMDSimulation(config)
   results = sim.run(atoms, driver_cmd=['lmp', '-in', 'input.lmp'])

2. Calculate quantum properties:
   
   from dftlammps.pimd import QuantumPropertyCalculator
   
   calc = QuantumPropertyCalculator(results)
   zpe = calc.calculate_zpe()
   diffusion = calc.calculate_diffusion()

3. Run hyperdynamics:
   
   from dftlammps.accelerated_dynamics import (
       HyperdynamicsConfig, HyperdynamicsSimulation
   )
   
   config = HyperdynamicsConfig(
       boost_method='bond_boost',
       q_cutoff=0.2,
       delta_v_max=1.0
   )
   
   sim = HyperdynamicsSimulation(config)
   results = sim.run(atoms, n_steps=100000)

4. Run KMC:
   
   from dftlammps.accelerated_dynamics import (
       KMCConfig, KMCSimulator, RateCatalog, RateProcess
   )
   
   catalog = RateCatalog()
   catalog.add_process(RateProcess(
       name='diffusion',
       initial_state='A',
       final_state='B',
       rate=1e10,
       activation_energy=0.5
   ))
   
   config = KMCConfig(temperature=300, n_steps=100000)
   sim = KMCSimulator(config, catalog)
   results = sim.run(initial_state)

See Also
--------

- dftlammps.pimd.proton_transfer_example
- dftlammps.accelerated_dynamics.vacancy_diffusion_example
- dftlammps.accelerated_dynamics.catalytic_reaction_example
"""

__version__ = '1.0.0'
__author__ = 'DFT+LAMMPS Integration Team'
