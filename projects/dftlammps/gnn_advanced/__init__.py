"""
Advanced Graph Neural Networks for Materials Science
========================================================

A comprehensive implementation of state-of-the-art GNN architectures for 
molecular dynamics and materials property prediction.

Supported Architectures:
------------------------
- GPS (Graph GPS): Hybrid MPNN + Transformer architecture
- Equiformer: E(3) equivariant graph attention transformer
- MACE: Higher-order equivariant message passing neural networks
- Allegro: Strictly local equivariant representations
- NequIP: E(3)-equivariant neural network potentials

Features:
---------
- Full E(3) equivariance for 3D atomic systems
- Integration with LAMMPS for molecular dynamics
- Explainability tools (GNNExplainer, PGExplainer)
- Pre-trained model zoo
- DFT data fusion capabilities

Author: DFT-LAMMPS Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Team"

from .models import (
    GPSModel,
    Equiformer,
    MACE,
    Allegro,
    NequIP,
)

from .explainability import (
    GNNExplainer,
    PGExplainer,
)

from .bridges import (
    DFTGNNBridge,
    LAMMPSGNNBridge,
)

__all__ = [
    "GPSModel",
    "Equiformer", 
    "MACE",
    "Allegro",
    "NequIP",
    "GNNExplainer",
    "PGExplainer",
    "DFTGNNBridge",
    "LAMMPSGNNBridge",
]
