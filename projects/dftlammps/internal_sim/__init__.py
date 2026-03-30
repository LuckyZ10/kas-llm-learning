"""
Internal Simulation Module
==========================

Fast physics simulation and abstract reasoning for mental models.

Components:
- FastPhysicsSimulator: High-speed learned physics simulator
- AbstractRepresentationLearner: Hierarchical knowledge compression
- DreamGenerator: Creative hypothetical scenario generation
- MentalSimulationEngine: Internal reasoning and planning
"""

from .simulator import (
    FastPhysicsSimulator,
    MultiScaleSimulator,
    SimulationConfig,
    SimulationGranularity,
    TransformerBlock,
    ResidualBlock
)

from .representation import (
    AbstractRepresentationLearner,
    VectorQuantizer,
    HierarchicalEncoder,
    HierarchicalDecoder,
    ConceptLibrary,
    RepresentationConfig,
    RepresentationLevel
)

from .dreams import (
    DreamGenerator,
    MentalSimulationEngine,
    LatentTransitionModel,
    DreamConfig,
    DreamType
)

__all__ = [
    # Simulator
    'FastPhysicsSimulator',
    'MultiScaleSimulator',
    'SimulationConfig',
    'SimulationGranularity',
    'TransformerBlock',
    'ResidualBlock',
    
    # Representation
    'AbstractRepresentationLearner',
    'VectorQuantizer',
    'HierarchicalEncoder',
    'HierarchicalDecoder',
    'ConceptLibrary',
    'RepresentationConfig',
    'RepresentationLevel',
    
    # Dreams
    'DreamGenerator',
    'MentalSimulationEngine',
    'LatentTransitionModel',
    'DreamConfig',
    'DreamType'
]

__version__ = '1.0.0'
