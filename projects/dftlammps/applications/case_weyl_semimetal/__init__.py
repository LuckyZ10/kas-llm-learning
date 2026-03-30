"""
Weyl Semimetal Case Studies
============================

This module provides case studies for Weyl semimetal materials:
- TaAs: First experimentally confirmed Weyl semimetal

Example:
    from dftlammps.applications.case_weyl_semimetal import (
        TaAsWorkflow, analyze_taas, analyze_tap
    )
    
    # Run complete analysis
    results = analyze_taas("./TaAs_calc")
"""

from .taas_analysis import (
    TaAsConfig,
    TaAsStructure,
    TaAsWorkflow,
    analyze_taas,
    analyze_tap,
    generate_reference_data,
)

__all__ = [
    "TaAsConfig",
    "TaAsStructure",
    "TaAsWorkflow",
    "analyze_taas",
    "analyze_tap",
    "generate_reference_data",
]
