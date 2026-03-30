"""
Advanced Quantum Monte Carlo Module
====================================

Advanced QMC methods including:
- Auxiliary-Field QMC (AFQMC)
- Path Integral Monte Carlo (PIMC)
- Quantum dynamics

Submodules:
-----------
- afqmc_interface : Ab-initio AFQMC calculations
- qmc_dynamics : PIMC and quantum dynamics

Example Usage:
--------------
>>> from dftlammps.qmc_advanced import AFQMCInterface, PathIntegralMonteCarlo
>>> 
>>> # Run AFQMC
>>> afqmc = AFQMCInterface(h1e, h2e, n_elec, n_basis, trial_wf)
>>> results = afqmc.run()
>>> 
>>> # Run PIMC
>>> pimc = PathIntegralMonteCarlo(n_atoms, masses, T, pes)
>>> results = pimc.run()
"""

__version__ = "1.0.0"

from .afqmc_interface import (
    AFQMCInterface,
    AFQMCWalker,
    AFQMCResults,
    VASPWaveFunctionInterface,
    QEinspressoInterface,
    import_pyscf_for_afqmc
)

from .qmc_dynamics import (
    PathIntegralMonteCarlo,
    QuantumDynamics,
    PotentialEnergySurface,
    PIMCBead,
    PIMCResults,
    calculate_thermodynamic_properties
)

__all__ = [
    # AFQMC
    'AFQMCInterface',
    'AFQMCWalker',
    'AFQMCResults',
    'VASPWaveFunctionInterface',
    'QEinspressoInterface',
    'import_pyscf_for_afqmc',
    
    # PIMC and Dynamics
    'PathIntegralMonteCarlo',
    'QuantumDynamics',
    'PotentialEnergySurface',
    'PIMCBead',
    'PIMCResults',
    'calculate_thermodynamic_properties'
]
