"""
DFTLammps Optical Advanced Module
==================================
Advanced optical property calculations:
- Second Harmonic Generation (SHG)
- Circular Dichroism (CD)
- High Harmonic Generation (HHG)

Submodules:
- shg_cd_hhg: SHG, CD, and HHG calculators

Example usage:
    from dftlammps.optical_advanced import SHGCalculator, SHGParameters
    from dftlammps.optical_advanced import CDCalculator, CDParameters
    from dftlammps.optical_advanced import HHGCalculator, HHGParameters
"""

from .shg_cd_hhg import (
    SHGCalculator,
    CDCalculator,
    HHGCalculator,
    SHGParameters,
    CDParameters,
    HHGParameters,
)

__all__ = [
    'SHGCalculator',
    'CDCalculator',
    'HHGCalculator',
    'SHGParameters',
    'CDParameters',
    'HHGParameters',
]

__version__ = '1.0.0'
