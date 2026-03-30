"""
Correlated Transition Metal Oxide Catalysts

This module provides DFT+U/DMFT workflows for studying catalytic properties
of strongly correlated transition metal oxides.

Submodules:
-----------
tmo_catalyst : Core catalyst analysis tools

Supported Systems:
-----------------
- Co3O4: CO oxidation, OER/ORR
- Fe2O3: Photocatalysis, water splitting
- MnO2: Oxygen reduction
- NiO: Electrocatalysis

Example Usage:
--------------
>>> from dftlammps.applications.case_correlated_catalyst import Co3O4Catalyst
>>> from dftlammps.applications.case_correlated_catalyst import CatalyticActivityPredictor
>>>
>>> # Setup catalyst
>>> catalyst = Co3O4Catalyst()
>>> surface = catalyst.setup_surface_structure(n_layers=4)
>>>
>>> # Analyze redox chemistry
>>> redox = catalyst.analyze_cobalt_redox_chemistry()
>>> print(f"Active site: {redox['active_site']}")
>>>
>>> # Calculate OER mechanism
>>> oer = catalyst.calculate_oer_mechanism()
>>> print(f"Overpotential: {oer['overpotential_V']:.2f} V")
"""

__version__ = "0.1.0"
__author__ = "DFT-LAMMPS Team"

from .tmo_catalyst import (
    CatalysisConfig,
    TMOxideCatalyst,
    Co3O4Catalyst,
    Fe2O3Catalyst,
    CatalyticActivityPredictor,
    calculate_scaling_relations,
    estimate_turnover_frequency,
    sabatier_principle_analysis
)

__all__ = [
    '__version__',
    '__author__',
    'CatalysisConfig',
    'TMOxideCatalyst',
    'Co3O4Catalyst',
    'Fe2O3Catalyst',
    'CatalyticActivityPredictor',
    'calculate_scaling_relations',
    'estimate_turnover_frequency',
    'sabatier_principle_analysis'
]