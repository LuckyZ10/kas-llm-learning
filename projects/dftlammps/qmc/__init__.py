"""
QMC Module - Initialization
===========================

Quantum Monte Carlo module for DFT-LAMMPS integration.

Provides:
- VMC (Variational Monte Carlo)
- DMC (Diffusion Monte Carlo)
- AFQMC (Auxiliary-Field Quantum Monte Carlo)
- PySCF interface for wave function preparation
- Statistical analysis tools

Example:
    from dftlammps.qmc import VMCCalculator, DMCCalculator
    from dftlammps.qmc.pyscf_qmc_interface import PySCFQMCInterface
"""

__version__ = "1.0.0"
__author__ = "QMC Expert Module"

# Core calculators
from .vmc_calculator import (
    VMCCalculator,
    SlaterJastrow,
    NeuralNetworkWaveFunction,
    WaveFunction,
    VMCResults,
    VMCSample,
    create_slater_jastrow_from_pyscf
)

from .dmc_calculator import (
    DMCCalculator,
    TrialWaveFunction,
    DMCResults,
    DMCWalker,
    create_trial_wf_from_vmc,
    extrapolate_dmc_energy,
    analyze_time_step_error
)

from .afqmc_calculator import (
    AFQMCCalculator,
    Hamiltonian,
    AFQMCResults,
    AFQMCWalker,
    create_hamiltonian_from_pyscf
)

# Interface
from .pyscf_qmc_interface import (
    PySCFQMCInterface,
    WaveFunctionData,
    create_molecule_from_xyz
)

# Analysis tools
from .qmc_analysis import (
    blocking_analysis,
    reblocking_analysis,
    analyze_monte_carlo_data,
    convergence_test,
    extrapolate_to_zero_time_step,
    StatisticalSummary
)

__all__ = [
    # VMC
    'VMCCalculator',
    'SlaterJastrow',
    'NeuralNetworkWaveFunction',
    'WaveFunction',
    'VMCResults',
    'VMCSample',
    'create_slater_jastrow_from_pyscf',
    # DMC
    'DMCCalculator',
    'TrialWaveFunction',
    'DMCResults',
    'DMCWalker',
    'create_trial_wf_from_vmc',
    'extrapolate_dmc_energy',
    'analyze_time_step_error',
    # AFQMC
    'AFQMCCalculator',
    'Hamiltonian',
    'AFQMCResults',
    'AFQMCWalker',
    'create_hamiltonian_from_pyscf',
    # Interface
    'PySCFQMCInterface',
    'WaveFunctionData',
    'create_molecule_from_xyz',
    # Analysis
    'blocking_analysis',
    'reblocking_analysis',
    'analyze_monte_carlo_data',
    'convergence_test',
    'extrapolate_to_zero_time_step',
    'StatisticalSummary'
]
