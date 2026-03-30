"""
High-Temperature Superconductor Analysis Module

This module provides DFT+DMFT workflows for studying high-temperature superconductors:
- Cuprates (hole-doped, d-wave pairing)
- Iron-based superconductors (electron/hole-doped, s± pairing)

Submodules:
-----------
cuprate_dft_dmft : Cuprate-specific workflows
iron_based_dmft : Iron-based superconductor analysis

Example Usage:
--------------
>>> from dftlammps.applications.case_high_tc_superconductor import CuprateDFTDMFT, CuprateConfig
>>> from dftlammps.applications.case_high_tc_superconductor import IronPnictideAnalyzer
>>>
>>> # Setup cuprate calculation
>>> config = CuprateConfig(material="La2CuO4", U_cu=8.0, hole_doping=0.15)
>>> cuprate = CuprateDFTDMFT(config)
>>> cuprate.generate_dft_input()
>>>
>>> # Analyze iron-based superconductor
>>> fe_as = IronPnictideAnalyzer("BaFe2As2")
>>> fs = fe_as.analyze_fermi_surface(k_grid)
>>> chi = fe_as.calculate_spin_susceptibility(q_grid)
"""

__version__ = "0.1.0"
__author__ = "DFT-LAMMPS Team"

from .cuprate_dft_dmft import (
    CuprateConfig,
    IronBasedConfig,
    CuprateDFTDMFT,
    IronBasedDFTDMFT,
    estimate_cuprate_tc,
    estimate_iron_based_tc,
    calculate_superfluid_stiffness
)

from .iron_based_dmft import (
    FeAsStructure,
    IronPnictideAnalyzer,
    FeSeAnalyzer,
    calculate_hund_coupling_effect,
    estimate_tc_from_spin_fluctuations
)

__all__ = [
    '__version__',
    '__author__',
    # Cuprates
    'CuprateConfig',
    'CuprateDFTDMFT',
    'estimate_cuprate_tc',
    'calculate_superfluid_stiffness',
    # Iron-based
    'IronBasedConfig',
    'IronBasedDFTDMFT',
    'FeAsStructure',
    'IronPnictideAnalyzer',
    'FeSeAnalyzer',
    'calculate_hund_coupling_effect',
    'estimate_tc_from_spin_fluctuations',
    'estimate_iron_based_tc',
]