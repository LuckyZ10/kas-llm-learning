"""
dftlammps.solvation
===================
Solvation effects for DFT calculations

Modules:
- vaspsol_interface: VASPsol implicit solvation model
- cp2k_solvation: CP2K explicit and implicit solvation
"""

from .vaspsol_interface import (
    VASPsolWorkflow,
    VASPsolCalculator,
    ElectrochemicalInterface,
    VASPsolConfig,
    SolvationResults,
    ElectrochemicalConfig,
)

from .cp2k_solvation import (
    CP2KSolvationWorkflow,
    CP2KElectrochemicalInterface,
    ExplicitSolventSetup,
    CP2KSolvationConfig,
    CP2KElectrolyteConfig,
    CP2KInputGenerator,
)

__all__ = [
    # VASPsol
    'VASPsolWorkflow',
    'VASPsolCalculator',
    'ElectrochemicalInterface',
    'VASPsolConfig',
    'SolvationResults',
    'ElectrochemicalConfig',
    
    # CP2K
    'CP2KSolvationWorkflow',
    'CP2KElectrochemicalInterface',
    'ExplicitSolventSetup',
    'CP2KSolvationConfig',
    'CP2KElectrolyteConfig',
    'CP2KInputGenerator',
]
