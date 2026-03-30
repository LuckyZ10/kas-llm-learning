"""
神经符号融合与因果发现引擎 - Neuro-Symbolic Fusion and Causal Discovery Engine

本模块结合深度学习的模式识别能力与符号推理的可解释性，实现自动化因果发现。

主要组件:
- neural_perception: 神经感知层
- symbolic_reasoning: 符号推理引擎
- causal_discovery: 因果发现算法
- neural_symbolic_bridge: 神经-符号桥接
- explainable_ai: 可解释AI模块
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Team"

from .neural_perception import (
    NeuralPerceptionSystem,
    FeatureExtractor,
    AttentionPerception,
    GraphPerception,
    PatternDetector,
    FeatureConfig,
    PatternConfig
)

from .symbolic_reasoning import (
    SymbolicReasoner,
    KnowledgeBase,
    KnowledgeGraph,
    SLDResolution,
    RuleMiner,
    Literal,
    Term,
    Rule
)

from .causal_discovery import (
    CausalDiscovery,
    PCAlgorithm,
    GESAlgorithm,
    NOTEARS,
    CausalGraph,
    IndependenceTest
)

from .neural_symbolic_bridge import (
    NeuralSymbolicBridge,
    NeuralToSymbolic,
    SymbolicToNeural,
    AttentionAlignment,
    BilingualConceptSpace,
    BridgeConfig
)

from .explainable_ai import (
    ExplainableAI,
    SHAPExplainer,
    LIMEExplainer,
    ConceptActivationVector,
    IntegratedGradients,
    ExplanationAggregator,
    FeatureImportance
)

__all__ = [
    # Neural Perception
    'NeuralPerceptionSystem',
    'FeatureExtractor',
    'AttentionPerception',
    'GraphPerception',
    'PatternDetector',
    'FeatureConfig',
    'PatternConfig',
    
    # Symbolic Reasoning
    'SymbolicReasoner',
    'KnowledgeBase',
    'KnowledgeGraph',
    'SLDResolution',
    'RuleMiner',
    'Literal',
    'Term',
    'Rule',
    
    # Causal Discovery
    'CausalDiscovery',
    'PCAlgorithm',
    'GESAlgorithm',
    'NOTEARS',
    'CausalGraph',
    'IndependenceTest',
    
    # Neural-Symbolic Bridge
    'NeuralSymbolicBridge',
    'NeuralToSymbolic',
    'SymbolicToNeural',
    'AttentionAlignment',
    'BilingualConceptSpace',
    'BridgeConfig',
    
    # Explainable AI
    'ExplainableAI',
    'SHAPExplainer',
    'LIMEExplainer',
    'ConceptActivationVector',
    'IntegratedGradients',
    'ExplanationAggregator',
    'FeatureImportance',
]
