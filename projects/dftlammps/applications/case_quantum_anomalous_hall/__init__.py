"""
Quantum Anomalous Hall Effect Case Studies
===========================================

This module provides case studies for QAHE materials:
- Cr/V-doped (Bi,Sb)2Te3: Magnetically-doped topological insulators

Example:
    from dftlammps.applications.case_quantum_anomalous_hall import (
        QAHEWorkflow, analyze_cr_doped_bi2te3, analyze_v_doped_bi2te3
    )
    
    # Run complete analysis
    results = analyze_cr_doped_bi2te3(concentration=0.08)
"""

from .qahe_analysis import (
    QAHEConfig,
    DopedTIStructure,
    QAHEWorkflow,
    analyze_cr_doped_bi2te3,
    analyze_v_doped_bi2te3,
    generate_reference_data,
)

__all__ = [
    "QAHEConfig",
    "DopedTIStructure",
    "QAHEWorkflow",
    "analyze_cr_doped_bi2te3",
    "analyze_v_doped_bi2te3",
    "generate_reference_data",
]
