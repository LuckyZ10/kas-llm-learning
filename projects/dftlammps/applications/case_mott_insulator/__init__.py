"""
Mott Insulator Analysis Case Studies

This module provides comprehensive workflows for studying
classic Mott insulators:
- NiO: Charge-transfer insulator with Type-II AFM
- CoO: Orbital-ordered insulator with spin-orbit effects

Submodules:
-----------
nio_coo_analysis : Detailed analyzers for NiO and CoO

Example Usage:
--------------
>>> from dftlammps.applications.case_mott_insulator import NiOAnalyzer
>>> from dftlammps.applications.case_mott_insulator import MottInsulatorWorkflow
>>>
>>> # Analyze NiO
>>> nio = NiOAnalyzer()
>>> electronic = nio.calculate_electronic_structure()
>>> print(f"Gap type: {electronic['gap_type']}")
>>> print(f"Gap size: {electronic['gap_size']:.2f} eV")
>>>
>>> exchange = nio.calculate_superexchange()
>>> print(f"J_super: {exchange['J_superexchange_180']*1000:.2f} meV")
>>>
>>> # Run complete workflow
>>> workflow = MottInsulatorWorkflow("NiO")
>>> results = workflow.run_complete_analysis()
>>> workflow.generate_dft_input()
"""

__version__ = "0.1.0"
__author__ = "DFT-LAMMPS Team"

from .nio_coo_analysis import (
    MottInsulatorConfig,
    NiOAnalyzer,
    CoOAnalyzer,
    MottInsulatorWorkflow,
    classify_insulator_type,
    estimate_neel_temperature,
    calculate_spin_wave_spectrum
)

__all__ = [
    '__version__',
    '__author__',
    'MottInsulatorConfig',
    'NiOAnalyzer',
    'CoOAnalyzer',
    'MottInsulatorWorkflow',
    'classify_insulator_type',
    'estimate_neel_temperature',
    'calculate_spin_wave_spectrum'
]