"""
Lifelong Learning: Continuous Learning Without Forgetting

This package implements lifelong learning capabilities:
- Catastrophic forgetting prevention (EWC, Progressive Networks)
- Knowledge accumulation over time
- Skill composition and transfer
"""

__version__ = "1.0.0"

from .lifelong_learning import (
    LifelongLearningSystem,
    ExperienceReplayBuffer,
    ElasticWeightConsolidation,
    ProgressiveNeuralNetwork,
    KnowledgeGraph,
    SkillComposer,
    Experience,
    Skill
)

__all__ = [
    'LifelongLearningSystem',
    'ExperienceReplayBuffer',
    'ElasticWeightConsolidation',
    'ProgressiveNeuralNetwork',
    'KnowledgeGraph',
    'SkillComposer',
    'Experience',
    'Skill'
]
