"""
AGI Core: Universal Materials Intelligence & Self-Improvement System

This package contains the core AGI components for materials science:
- Meta-Learning V2: Learning to learn across tasks
- Self-Improvement: Automatic code and algorithm optimization
- Knowledge Creation: Automatic pattern discovery and theory generation
"""

__version__ = "1.0.0"
__author__ = "AGI Materials Intelligence System"

from .meta_learning_v2 import (
    MetaLearningPipeline,
    MetaLearnerV2,
    TaskConfig,
    TaskDistribution,
    CrossDomainTransfer,
    FewShotLearner
)

from .self_improvement import (
    SelfImprovementManager,
    CodeOptimizer,
    AlgorithmDiscoveryEngine,
    SelfReflectionEngine,
    PerformanceMetrics
)

from .knowledge_creation import (
    KnowledgeCreationPipeline,
    PatternDiscoveryEngine,
    TheoryGenerator,
    HypothesisValidator,
    SymbolicDiscovery,
    DiscoveredPattern,
    GeneratedTheory
)

__all__ = [
    # Meta-learning
    'MetaLearningPipeline',
    'MetaLearnerV2',
    'TaskConfig',
    'TaskDistribution',
    'CrossDomainTransfer',
    'FewShotLearner',
    
    # Self-improvement
    'SelfImprovementManager',
    'CodeOptimizer',
    'AlgorithmDiscoveryEngine',
    'SelfReflectionEngine',
    'PerformanceMetrics',
    
    # Knowledge creation
    'KnowledgeCreationPipeline',
    'PatternDiscoveryEngine',
    'TheoryGenerator',
    'HypothesisValidator',
    'SymbolicDiscovery',
    'DiscoveredPattern',
    'GeneratedTheory'
]
