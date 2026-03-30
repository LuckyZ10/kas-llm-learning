"""
Topology Module for DFTLammps
=============================

This module provides comprehensive tools for topological materials calculations,
including Z2 invariants, Berry phases, surface states, and Weyl points.

Submodules:
- z2pack_interface: Z2Pack interface for topological invariant calculations
- wannier_tools_interface: WannierTools interface for surface states and Weyl points
- berry_phase: Berry phase and anomalous Hall conductivity calculations

Example Usage:
--------------
    from dftlammps.topology import (
        Z2VASPInterface, Z2PackConfig,
        WannierToolsCalculator, WannierToolsConfig,
        PolarizationCalculator, BerryCurvatureCalculator,
    )
    
    # Calculate Z2 invariant
    config = Z2PackConfig(num_bands=20, surface="ky-surface")
    interface = Z2VASPInterface("./Bi2Se3", config)
    result = interface.calculate_z2_invariant()
    print(f"Z2 index: {result.z2_index}")
    
    # Calculate surface states
    wt_calc = WannierToolsCalculator("wannier90_hr.dat")
    surface_result = wt_calc.calculate_surface_states()
    
    # Calculate Berry curvature
    berry_calc = BerryCurvatureCalculator("./calculation")
    curvature = berry_calc.calculate_berry_curvature()
"""

from .z2pack_interface import (
    # Configuration
    Z2PackConfig,
    
    # Enums
    TopologicalPhase,
    SymmetryType,
    
    # Results
    WilsonLoopResult,
    Z2InvariantResult,
    ChernNumberResult,
    
    # Main classes
    VASPWavefunctionExtractor,
    Z2VASPInterface,
    ChernNumberCalculator,
    TopologicalClassifier,
    
    # Convenience functions
    calculate_z2_index,
    calculate_chern_number,
    classify_topological_material,
)

from .wannier_tools_interface import (
    # Configuration
    WannierToolsConfig,
    
    # Enums
    SurfaceType,
    WeylChirality,
    
    # Results
    SurfaceStateResult,
    WeylPoint,
    WeylSearchResult,
    BandInversionResult,
    
    # Main classes
    Wannier90HamiltonianBuilder,
    WannierToolsCalculator,
    BandInversionAnalyzer,
    
    # Convenience functions
    calculate_surface_states,
    search_weyl_points,
    analyze_band_inversion,
)

from .berry_phase import (
    # Configuration
    BerryPhaseConfig,
    
    # Enums
    PolarizationDirection,
    BerryCurvatureMethod,
    
    # Results
    PolarizationResult,
    BerryCurvatureResult,
    AnomalousHallConductivityResult,
    
    # Main classes
    PolarizationCalculator,
    BerryCurvatureCalculator,
    AnomalousHallConductivityCalculator,
    
    # Convenience functions
    calculate_polarization,
    calculate_berry_curvature,
    calculate_anomalous_hall_conductivity,
)

__all__ = [
    # Z2Pack interface
    "Z2PackConfig",
    "TopologicalPhase",
    "SymmetryType",
    "WilsonLoopResult",
    "Z2InvariantResult",
    "ChernNumberResult",
    "VASPWavefunctionExtractor",
    "Z2VASPInterface",
    "ChernNumberCalculator",
    "TopologicalClassifier",
    "calculate_z2_index",
    "calculate_chern_number",
    "classify_topological_material",
    
    # WannierTools interface
    "WannierToolsConfig",
    "SurfaceType",
    "WeylChirality",
    "SurfaceStateResult",
    "WeylPoint",
    "WeylSearchResult",
    "BandInversionResult",
    "Wannier90HamiltonianBuilder",
    "WannierToolsCalculator",
    "BandInversionAnalyzer",
    "calculate_surface_states",
    "search_weyl_points",
    "analyze_band_inversion",
    
    # Berry phase
    "BerryPhaseConfig",
    "PolarizationDirection",
    "BerryCurvatureMethod",
    "PolarizationResult",
    "BerryCurvatureResult",
    "AnomalousHallConductivityResult",
    "PolarizationCalculator",
    "BerryCurvatureCalculator",
    "AnomalousHallConductivityCalculator",
    "calculate_polarization",
    "calculate_berry_curvature",
    "calculate_anomalous_hall_conductivity",
]

__version__ = "1.0.0"
