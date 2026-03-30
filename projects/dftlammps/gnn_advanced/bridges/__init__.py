"""
Bridge modules for integrating GNNs with DFT and MD simulations
=================================================================

Provides interfaces between:
- GNN models and DFT calculations
- GNN models and LAMMPS molecular dynamics
"""

from .dft_bridge import DFTGNNBridge
from .lammps_bridge import LAMMPSGNNBridge

__all__ = ["DFTGNNBridge", "LAMMPSGNNBridge"]
