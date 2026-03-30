"""
DFT+LAMMPS Advanced Analysis Module
===================================

Advanced analysis tools for molecular dynamics simulations:
- Dynamic heterogeneity (glass transition, supercooled liquids)
- Structural analysis (ring statistics, Voronoi analysis, bond order)
- Time correlation functions (VACF, SACF, dielectric)

Modules
-------
dynamic_heterogeneity.py
    Non-Gaussian parameter, dynamic susceptibility, mobile particle clustering

structural_analysis.py
    Ring statistics, Voronoi indices, Steinhardt parameters, CNA

correlation_analysis.py
    VACF, stress autocorrelation, viscosity, diffusion coefficients

Example Usage
-------------
>>> from dftlammps.md_analysis_advanced import DynamicHeterogeneityAnalyzer
>>> analyzer = DynamicHeterogeneityAnalyzer(config)
>>> results = analyzer.analyze_dynamics(trajectory)

>>> from dftlammps.md_analysis_advanced import StructuralAnalyzer
>>> analyzer = StructuralAnalyzer()
>>> results = analyzer.full_analysis(positions)
"""

from .dynamic_heterogeneity import (
    DynamicHeterogeneityConfig,
    DynamicHeterogeneityAnalyzer
)

from .structural_analysis import (
    StructuralAnalysisConfig,
    RingStatistics,
    VoronoiAnalysis,
    BondOrientationalOrder,
    CommonNeighborAnalysis,
    StructuralAnalyzer
)

from .correlation_analysis import (
    CorrelationConfig,
    TimeCorrelationAnalyzer,
    VelocityAutocorrelation,
    StressAutocorrelation,
    DipoleAutocorrelation,
    IntermediateScattering,
    CorrelationAnalysisSuite
)

__all__ = [
    # Dynamic Heterogeneity
    'DynamicHeterogeneityConfig',
    'DynamicHeterogeneityAnalyzer',
    
    # Structural Analysis
    'StructuralAnalysisConfig',
    'RingStatistics',
    'VoronoiAnalysis',
    'BondOrientationalOrder',
    'CommonNeighborAnalysis',
    'StructuralAnalyzer',
    
    # Correlation Analysis
    'CorrelationConfig',
    'TimeCorrelationAnalyzer',
    'VelocityAutocorrelation',
    'StressAutocorrelation',
    'DipoleAutocorrelation',
    'IntermediateScattering',
    'CorrelationAnalysisSuite'
]
