"""
Topological Insulator Case Studies
===================================

This module provides case studies for topological insulator materials:
- Bi2Se3 / Bi2Te3: Prototypical 3D topological insulators

Example:
    from dftlammps.applications.case_topological_insulator import (
        Bi2Se3Workflow, analyze_bi2se3, analyze_bi2te3
    )
    
    # Run complete analysis
    results = analyze_bi2se3("./Bi2Se3_calc")
"""

from .bi2se3_analysis import (
    TopologicalInsulatorConfig,
    Bi2Se3Structure,
    Bi2Se3Workflow,
    analyze_bi2se3,
    analyze_bi2te3,
    generate_reference_data,
)

__all__ = [
    "TopologicalInsulatorConfig",
    "Bi2Se3Structure",
    "Bi2Se3Workflow",
    "analyze_bi2se3",
    "analyze_bi2te3",
    "generate_reference_data",
]
