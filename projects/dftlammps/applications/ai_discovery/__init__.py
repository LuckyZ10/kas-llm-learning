"""
AI-Driven Materials Discovery Applications
============================================

This module provides complete application workflows for AI-driven materials
discovery, integrating with the DFT+LAMMPS platform.

Available Case Studies:
- Solid Electrolyte Discovery: Find novel superionic conductors
- High-Entropy Alloy Catalysts: Design multi-component catalysts

Example Usage:
    from dftlammps.applications.ai_discovery import (
        SolidElectrolyteDiscovery,
        HighEntropyAlloyDiscovery,
        run_solid_electrolyte_discovery,
        run_hea_catalyst_discovery
    )
    
    # Run solid electrolyte discovery
    results = run_solid_electrolyte_discovery(
        target_ion="Li",
        num_iterations=30
    )
    
    # Run HEA catalyst discovery
    results = run_hea_catalyst_discovery(
        target_reaction="ORR",
        num_iterations=30
    )

Author: DFT+LAMMPS AI Team
"""

from .case_studies import (
    SolidElectrolyteDiscovery,
    HighEntropyAlloyDiscovery,
    DiscoveryConfig,
    run_solid_electrolyte_discovery,
    run_hea_catalyst_discovery,
    compare_discovery_methods,
    generate_discovery_report,
)

__all__ = [
    # Classes
    "SolidElectrolyteDiscovery",
    "HighEntropyAlloyDiscovery",
    "DiscoveryConfig",
    
    # Functions
    "run_solid_electrolyte_discovery",
    "run_hea_catalyst_discovery",
    "compare_discovery_methods",
    "generate_discovery_report",
]
