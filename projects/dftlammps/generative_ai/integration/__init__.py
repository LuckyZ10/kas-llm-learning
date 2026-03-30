"""
Integration Module
==================

Integration with screening and design workflows:

1. **GenerativeScreening** - AI-guided structure generation for screening
2. **ActiveLearningGenerator** - Active learning with DFT validation
3. **InverseDesignPipeline** - Property-targeted inverse design
4. **MultiObjectiveOptimizer** - Pareto frontier optimization
"""

from .screening_integration import (
    GenerativeScreening,
    ActiveLearningGenerator
)

from .inverse_design import (
    InverseDesignPipeline,
    LatentSpaceOptimizer,
    BayesianOptimizationDesigner
)

__all__ = [
    "GenerativeScreening",
    "ActiveLearningGenerator",
    "InverseDesignPipeline",
    "LatentSpaceOptimizer",
    "BayesianOptimizationDesigner",
]
