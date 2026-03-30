"""
DFTLammps Defect Excited States Module
=======================================
Defect excited state calculations:
- Color centers (NV, SiV, GeV, etc.)
- Point defect luminescence
- Spin-photon coupling
- Quantum bit properties

Submodules:
- color_center: Color center physics and quantum computing

Example usage:
    from dftlammps.defect_excited import NVHamiltonian, get_color_center
    from dftlammps.defect_excited import ColorCenterSpectrum
    from dftlammps.defect_excited import QuantumBitOperations
"""

from .color_center import (
    NVHamiltonian,
    ColorCenterSpectrum,
    QuantumBitOperations,
    ColorCenterParameters,
    PhononSideband,
    QuantumBit,
    get_color_center,
    COLOR_CENTER_DATABASE,
)

__all__ = [
    'NVHamiltonian',
    'ColorCenterSpectrum',
    'QuantumBitOperations',
    'ColorCenterParameters',
    'PhononSideband',
    'QuantumBit',
    'get_color_center',
    'COLOR_CENTER_DATABASE',
]

__version__ = '1.0.0'
