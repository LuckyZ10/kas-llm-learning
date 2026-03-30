"""
DFT-LAMMPS 表征数据分析模块

提供XRD、SEM、TEM等表征数据的解析和对比功能
"""

from .xrd_analysis import (
    XRDPattern,
    XRDPeak,
    PhaseIdentification,
    XRDParser,
    PeakDetector,
    PhaseIdentifier,
    XRDAnalyzer,
    RietveldRefinement,
    analyze_xrd,
    compare_xrd,
    identify_phases
)

from .sem_analysis import (
    SEMImage,
    Particle,
    MorphologyMetrics,
    EDSData,
    SEMImageLoader,
    ParticleAnalyzer,
    EDSAnalyzer,
    SEMAnalyzer,
    analyze_sem,
    analyze_particles
)

from .tem_analysis import (
    HRTEMImage,
    SAEDPattern,
    LatticeFringe,
    DiffractionSpot,
    CrystalInfo,
    TEMImageLoader,
    LatticeAnalyzer,
    DiffractionAnalyzer,
    TEMAnalyzer,
    analyze_tem_hrtem,
    analyze_tem_saed
)

from .comparison import (
    ComparisonMetrics,
    StructureComparison,
    PropertyComparison,
    DataComparator,
    SpectrumComparator,
    XRDComparator,
    StructureComparator,
    PropertyComparator,
    ImageComparator,
    FeedbackLoop,
    ComparisonManager,
    compare_calculation_experiment,
    validate_calculation,
    generate_validation_report
)

__all__ = [
    # XRD
    'XRDPattern',
    'XRDPeak',
    'PhaseIdentification',
    'XRDParser',
    'PeakDetector',
    'PhaseIdentifier',
    'XRDAnalyzer',
    'RietveldRefinement',
    'analyze_xrd',
    'compare_xrd',
    'identify_phases',
    
    # SEM
    'SEMImage',
    'Particle',
    'MorphologyMetrics',
    'EDSData',
    'SEMImageLoader',
    'ParticleAnalyzer',
    'EDSAnalyzer',
    'SEMAnalyzer',
    'analyze_sem',
    'analyze_particles',
    
    # TEM
    'HRTEMImage',
    'SAEDPattern',
    'LatticeFringe',
    'DiffractionSpot',
    'CrystalInfo',
    'TEMImageLoader',
    'LatticeAnalyzer',
    'DiffractionAnalyzer',
    'TEMAnalyzer',
    'analyze_tem_hrtem',
    'analyze_tem_saed',
    
    # Comparison
    'ComparisonMetrics',
    'StructureComparison',
    'PropertyComparison',
    'DataComparator',
    'SpectrumComparator',
    'XRDComparator',
    'StructureComparator',
    'PropertyComparator',
    'ImageComparator',
    'FeedbackLoop',
    'ComparisonManager',
    'compare_calculation_experiment',
    'validate_calculation',
    'generate_validation_report'
]
