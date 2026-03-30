"""
因果AI模块 - Causal AI for Materials Discovery

本模块提供材料科学中的因果推断和可解释AI工具：
- 因果发现 (Causal Discovery)
- 可解释机器学习 (Explainable ML)
- 机理模型 (Mechanistic Models)

主要组件:
    - causal_discovery: 因果发现算法 (PC, GES, NOTEARS)
    - explainable_ml: 可解释方法 (SHAP, LIME, Attention, CAV)
    - mechanistic_model: 物理约束模型、符号回归、方程发现

作者: Causal AI Team
"""

from .causal_discovery import (
    # 主要类
    CausalGraph,
    CausalEdge,
    Intervention,
    CounterfactualQuery,
    IndependenceTester,
    
    # 算法
    PCAlgorithm,
    GESAlgorithm,
    NOTEARSAlgorithm,
    
    # 效应估计
    InterventionEffectEstimator,
    CounterfactualInference,
    
    # 管道
    CausalDiscoveryPipeline,
    
    # 枚举
    IndependenceTest,
    CausalAlgorithm
)

from .explainable_ml import (
    # 解释器
    SHAPExplainer,
    LIMEExplainer,
    AttentionVisualizer,
    ConceptActivationVectors,
    IntegratedGradients,
    PermutationImportance,
    
    # 数据结构
    Explanation,
    FeatureImportance,
    
    # 管道
    ExplainableMLPipeline
)

from .mechanistic_model import (
    # 物理约束神经网络
    PhysicsInformedNN,
    PINNTrainer,
    PhysicalConstraint,
    
    # 符号回归
    SymbolicRegression,
    ExpressionNode,
    
    # 方程发现
    EquationDiscovery,
    DiscoveredEquation,
    
    # 管道
    MechanisticModelPipeline
)

__version__ = "1.0.0"
__all__ = [
    # Causal Discovery
    "CausalGraph",
    "CausalEdge",
    "Intervention",
    "CounterfactualQuery",
    "IndependenceTester",
    "PCAlgorithm",
    "GESAlgorithm",
    "NOTEARSAlgorithm",
    "InterventionEffectEstimator",
    "CounterfactualInference",
    "CausalDiscoveryPipeline",
    "IndependenceTest",
    "CausalAlgorithm",
    
    # Explainable ML
    "SHAPExplainer",
    "LIMEExplainer",
    "AttentionVisualizer",
    "ConceptActivationVectors",
    "IntegratedGradients",
    "PermutationImportance",
    "Explanation",
    "FeatureImportance",
    "ExplainableMLPipeline",
    
    # Mechanistic Models
    "PhysicsInformedNN",
    "PINNTrainer",
    "PhysicalConstraint",
    "SymbolicRegression",
    "ExpressionNode",
    "EquationDiscovery",
    "DiscoveredEquation",
    "MechanisticModelPipeline"
]
