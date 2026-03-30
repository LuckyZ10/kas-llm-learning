"""
Utilities Module
================

Utility functions for generative models:

1. **DiffusionSampler** - DDPM/DDIM sampling
2. **FlowSampler** - ODE solvers for flow matching
3. **ConsistencySampler** - Fast sampling for consistency models
4. **CrystalMetrics** - Evaluation metrics
5. **SpaceGroupConstraint** - Symmetry handling
6. **WyckoffPositionEncoder** - Wyckoff position encoding
"""

from .sampling import (
    DiffusionSampler,
    FlowSampler,
    ConsistencySampler
)

from .evaluation import (
    CrystalMetrics,
    compute_frechet_distance
)

from .symmetry import (
    SpaceGroupConstraint,
    WyckoffPositionEncoder,
    SymmetryPreservingGenerator,
    SymmetryLoss,
    get_symmetry_operations,
    apply_symmetry_operation,
    compute_symmetry_score
)

__all__ = [
    # Sampling
    "DiffusionSampler",
    "FlowSampler",
    "ConsistencySampler",
    # Evaluation
    "CrystalMetrics",
    "compute_frechet_distance",
    # Symmetry
    "SpaceGroupConstraint",
    "WyckoffPositionEncoder",
    "SymmetryPreservingGenerator",
    "SymmetryLoss",
    "get_symmetry_operations",
    "apply_symmetry_operation",
    "compute_symmetry_score",
]
