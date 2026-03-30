"""
Weyl Semimetal Module for DFTLammps
====================================

This module provides tools for studying Weyl semimetals including:
- Weyl point location and classification (Type I / Type II)
- Chirality calculation using Berry curvature
- Fermi arc surface state calculations
- Magnetotransport and chiral anomaly analysis

Example Usage:
--------------
    from dftlammps.weyl import (
        WeylPointLocator, WeylSemimetalConfig,
        FermiArcCalculator, ChiralityCalculator,
        MagnetotransportCalculator,
        locate_weyl_points, analyze_weyl_semimetal,
    )
    
    # Locate Weyl points
    config = WeylSemimetalConfig(k_mesh_fine=(50, 50, 50))
    locator = WeylPointLocator("./TaAs", config)
    weyl_points = locator.search_weyl_points()
    
    print(f"Found {len(weyl_points)} Weyl points")
    for wp in weyl_points:
        print(f"  k={wp.k_point}, C={wp.chirality}, E={wp.energy:.3f} eV")
    
    # Calculate Fermi arcs
    arc_calc = FermiArcCalculator("./TaAs")
    fermi_arcs = arc_calc.calculate_fermi_arcs(weyl_points)
    
    # Analyze magnetotransport
    transport = MagnetotransportCalculator(weyl_points)
    sigma_chiral = transport.calculate_chiral_anomaly(np.array([0, 0, 1]))
"""

from .weyl_semimetal import (
    # Configuration
    WeylSemimetalConfig,
    
    # Enums
    WeylType,
    FermiArcType,
    
    # Data classes
    WeylPointData,
    ChiralityResult,
    FermiArcData,
    MagnetotransportResult,
    
    # Main calculators
    WeylPointLocator,
    ChiralityCalculator,
    FermiArcCalculator,
    MagnetotransportCalculator,
    
    # Convenience functions
    locate_weyl_points,
    calculate_fermi_arcs,
    analyze_weyl_semimetal,
)

__all__ = [
    # Configuration
    "WeylSemimetalConfig",
    
    # Enums
    "WeylType",
    "FermiArcType",
    
    # Data classes
    "WeylPointData",
    "ChiralityResult",
    "FermiArcData",
    "MagnetotransportResult",
    
    # Calculators
    "WeylPointLocator",
    "ChiralityCalculator",
    "FermiArcCalculator",
    "MagnetotransportCalculator",
    
    # Convenience functions
    "locate_weyl_points",
    "calculate_fermi_arcs",
    "analyze_weyl_semimetal",
]

__version__ = "1.0.0"
