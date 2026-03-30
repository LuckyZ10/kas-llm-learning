"""
Generative AI Models Module
============================

This module contains various generative models for materials:

1. **CrystalDiT** - Diffusion Transformer for crystal generation
2. **ADiT** - All-atom Diffusion Transformer (unified molecules & crystals)
3. **RiemannianFlowMatcher** - Flow matching on manifolds
4. **CrystalFlow** - Flow-based generative model
5. **ConsistencyCrystalModel** - Fast sampling via consistency models
6. **ConditionalDiffusion** - Property-guided generation with CFG
7. **JointMolecularCrystalGenerator** - Unified molecular/crystal generation
"""

from .crystal_dit import (
    CrystalDiT,
    ADiT,
    CrystalDiTConfig,
    DiffusionScheduler,
    DiTBlock,
    TimestepEmbedding
)

from .flow_matching import (
    RiemannianFlowMatcher,
    CrystalFlow,
    FlowMatchingConfig,
    EGNNLayer,
    FourierFeatures
)

from .consistency import (
    ConsistencyCrystalModel,
    ConsistencyConfig,
    ConsistencyBackbone
)

from .conditional import (
    ConditionalDiffusion,
    MultiObjectiveDiffusion,
    ConditionalConfig
)

from .joint_generator import (
    JointMolecularCrystalGenerator,
    JointGeneratorConfig,
    UnifiedEncoder,
    UnifiedDecoder,
    LatentDenoiser
)

__all__ = [
    # CrystalDiT
    "CrystalDiT",
    "ADiT",
    "CrystalDiTConfig",
    "DiffusionScheduler",
    "DiTBlock",
    "TimestepEmbedding",
    # Flow Matching
    "RiemannianFlowMatcher",
    "CrystalFlow",
    "FlowMatchingConfig",
    "EGNNLayer",
    "FourierFeatures",
    # Consistency
    "ConsistencyCrystalModel",
    "ConsistencyConfig",
    "ConsistencyBackbone",
    # Conditional
    "ConditionalDiffusion",
    "MultiObjectiveDiffusion",
    "ConditionalConfig",
    # Joint
    "JointMolecularCrystalGenerator",
    "JointGeneratorConfig",
    "UnifiedEncoder",
    "UnifiedDecoder",
    "LatentDenoiser",
]
