"""
Non-Adiabatic Dynamics Module
=============================

This module provides tools for simulating excited state processes
using non-adiabatic molecular dynamics methods.

Submodules:
- pyxaid_interface: Interface to PYXAID for TD-DFT based NAMD
- sharc_interface: Interface to SHARC for multi-reference NAMD
- excited_state_dynamics: High-level excited state dynamics workflows

Author: dftlammps development team
"""

from .pyxaid_interface import (
    PYXAIDConfig,
    PYXAIDWorkflow,
    SurfaceHoppingDynamics,
    CarrierLifetimeAnalyzer,
    NonAdiabaticCouplingCalculator,
    ElectronicState,
    DynamicsTrajectory,
    VASPTDDFTInterface,
)

from .sharc_interface import (
    SHARCConfig,
    SHARCSurfaceHopping,
    SHARCTrajectory,
    MultiReferenceInterface,
    SpinOrbitMatrix,
    SpinState,
    SpinOrbitState,
    MultiReferenceMethod,
    SpinOrbitMethod,
)

from .excited_state_dynamics import (
    ExcitedStateDynamicsWorkflow,
    ExcitonDynamics,
    CarrierDynamics,
    EnergyTransferNetwork,
    ExcitonState,
    CarrierState,
    EnergyTransferPathway,
    ChargeSeparationState,
    ExcitedProcess,
)

__all__ = [
    # PYXAID
    'PYXAIDConfig',
    'PYXAIDWorkflow',
    'SurfaceHoppingDynamics',
    'CarrierLifetimeAnalyzer',
    'NonAdiabaticCouplingCalculator',
    'ElectronicState',
    'DynamicsTrajectory',
    'VASPTDDFTInterface',
    
    # SHARC
    'SHARCConfig',
    'SHARCSurfaceHopping',
    'SHARCTrajectory',
    'MultiReferenceInterface',
    'SpinOrbitMatrix',
    'SpinState',
    'SpinOrbitState',
    'MultiReferenceMethod',
    'SpinOrbitMethod',
    
    # Excited state dynamics
    'ExcitedStateDynamicsWorkflow',
    'ExcitonDynamics',
    'CarrierDynamics',
    'EnergyTransferNetwork',
    'ExcitonState',
    'CarrierState',
    'EnergyTransferPathway',
    'ChargeSeparationState',
    'ExcitedProcess',
]

__version__ = '0.1.0'
