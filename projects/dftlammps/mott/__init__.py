"""
DFT-LAMMPS Mott Insulator Analysis Module

This module provides tools for analyzing Mott insulating behavior and
transitions in strongly correlated systems.

Submodules:
-----------
mott_analysis : Core analysis tools for Mott insulators

Features:
---------
- Electronic gap analysis and gap closure detection
- Metal-insulator transition identification
- Charge and spin order parameter analysis
- Phase diagram construction
- Critical behavior analysis

Example Usage:
--------------
>>> from dftlammps.mott import GapAnalyzer, MottAnalysisConfig
>>> from dftlammps.mott import MetalInsulatorTransition
>>>
>>> # Analyze electronic gap
>>> config = MottAnalysisConfig(gap_threshold=0.05)
>>> analyzer = GapAnalyzer(config)
>>> gap_info = analyzer.calculate_gap(eigenvalues, k_points)
>>> print(f"Gap: {gap_info['gap_indirect']:.3f} eV")
>>> print(f"Is insulator: {gap_info['is_insulator']}")
>>>
>>> # Track gap closure
>>> closure_info = analyzer.track_gap_closure(gaps, U_values)
>>> print(f"Critical U: {closure_info['critical_point']:.3f} eV")
>>>
>>> # Detect metal-insulator transition
>>> mit = MetalInsulatorTransition(config)>>> mit_results = mit.detect_mit_gap_criterion(gaps, U_values)
>>> print(f"MIT at U = {mit_results['transition_points']}")
"""

__version__ = "0.1.0"
__author__ = "DFT-LAMMPS Team"

# Import main classes
from .mott_analysis import (
    MottAnalysisConfig,
    GapAnalyzer,
    MetalInsulatorTransition,
    OrderParameterAnalyzer,
    estimate_mott_gap,
    calculate_u_critical_2d,
    calculate_u_critical_3d,
    analyze_quasiparticle_residue,
    estimate_brinkman_rice_z,
    check_luttinger_theorem
)

__all__ = [
    '__version__',
    '__author__',
    'MottAnalysisConfig',
    'GapAnalyzer',
    'MetalInsulatorTransition',
    'OrderParameterAnalyzer',
    'estimate_mott_gap',
    'calculate_u_critical_2d',
    'calculate_u_critical_3d',
    'analyze_quasiparticle_residue',
    'estimate_brinkman_rice_z',
    'check_luttinger_theorem'
]