"""
不确定性量化与可靠性工程模块 - Uncertainty Quantification and Reliability Engineering

Phase 70: 计算不确定性量化体系

本模块提供完整的不确定性量化工具链：
1. 概率机器学习（贝叶斯神经网络、深度集成）
2. 概率数值方法（概率PDE求解）
3. 模型误差传播分析

主要组件:
    - bayesian_potential: 贝叶斯神经网络势函数
    - mc_propagation: 蒙特卡洛误差传播
    - sensitivity_analysis: 敏感性分析工具
    - probabilistic_numerics: 概率数值方法
    - workflow_reliability: 工作流可靠性评估

交付标准：可量化的不确定性、有案例验证
代码量：~3500行

作者: DFT-LAMMPS Team
"""

from .bayesian_potential import (
    # 贝叶斯势函数核心
    BayesianPotential,
    BayesianNeuralPotential,
    MCDropoutPotential,
    VariationalPotential,
    EnsemblePotential,
    
    # 势能预测
    EnergyPrediction,
    ForcePrediction,
    StressPrediction,
    PotentialUncertainty,
    
    # 训练与校准
    PotentialTrainer,
    BayesianCalibration,
    UncertaintyCalibrator,
    
    # 原子环境描述
    ACSFDescriptor,
    SOAPDescriptor,
    MBTRDescriptor,
)

from .mc_propagation import (
    # 蒙特卡洛传播
    MCErrorPropagation,
    UncertaintyPropagator,
    DirectSampling,
    LatinHypercubeSampling,
    QuasiMonteCarlo,
    
    # 传播结果
    PropagationResult,
    ErrorBudget,
    ConfidenceInterval,
    
    # 高级方法
    PolynomialChaosExpansion,
    StochasticCollocation,
    MarkovChainMonteCarlo,
)

from .sensitivity_analysis import (
    # 敏感性分析
    SensitivityAnalyzer,
    MorrisMethod,
    SobolSensitivity,
    FASTAnalysis,
    
    # 局部分析
    LocalSensitivity,
    GradientBasedAnalysis,
    FiniteDifferenceSensitivity,
    
    # 结果与可视化
    SensitivityIndices,
    SensitivityReport,
    ParameterImportance,
    
    # 筛选方法
    ScreeningAnalysis,
    ElementaryEffects,
)

from .probabilistic_numerics import (
    # 概率PDE求解
    ProbabilisticPDESolver,
    ProbabilisticFEM,
    BayesianQuadrature,
    
    # 概率线性代数
    ProbabilisticLinearSolver,
    BayesianCG,
    
    # 概率ODE/积分
    ProbabilisticODE,
    BayesianQuadrature,
    
    # 核心组件
    ProbabilityDistribution,
    GaussianProcessPrior,
    LinearOperatorUncertainty,
)

from .workflow_reliability import (
    # 可靠性评估
    ReliabilityEngine,
    WorkflowReliability,
    ReliabilityAssessment,
    
    # 失效分析
    FailureProbability,
    ReliabilityIndex,
    FORMAnalysis,
    SORMAnalysis,
    MonteCarloReliability,
    
    # 系统可靠性
    SystemReliability,
    FaultTreeAnalysis,
    EventTreeAnalysis,
    
    # 监控与预警
    ReliabilityMonitor,
    UncertaintyBudget,
    QualityAssurance,
)

__version__ = "1.0.0"
__all__ = [
    # Bayesian Potential
    "BayesianPotential",
    "BayesianNeuralPotential",
    "MCDropoutPotential",
    "VariationalPotential",
    "EnsemblePotential",
    "EnergyPrediction",
    "ForcePrediction",
    "StressPrediction",
    "PotentialUncertainty",
    "PotentialTrainer",
    "BayesianCalibration",
    "UncertaintyCalibrator",
    "ACSFDescriptor",
    "SOAPDescriptor",
    "MBTRDescriptor",
    
    # MC Propagation
    "MCErrorPropagation",
    "UncertaintyPropagator",
    "DirectSampling",
    "LatinHypercubeSampling",
    "QuasiMonteCarlo",
    "PropagationResult",
    "ErrorBudget",
    "ConfidenceInterval",
    "PolynomialChaosExpansion",
    "StochasticCollocation",
    "MarkovChainMonteCarlo",
    
    # Sensitivity Analysis
    "SensitivityAnalyzer",
    "MorrisMethod",
    "SobolSensitivity",
    "FASTAnalysis",
    "LocalSensitivity",
    "GradientBasedAnalysis",
    "FiniteDifferenceSensitivity",
    "SensitivityIndices",
    "SensitivityReport",
    "ParameterImportance",
    "ScreeningAnalysis",
    "ElementaryEffects",
    
    # Probabilistic Numerics
    "ProbabilisticPDESolver",
    "ProbabilisticFEM",
    "BayesianQuadrature",
    "ProbabilisticLinearSolver",
    "BayesianCG",
    "ProbabilisticODE",
    "ProbabilityDistribution",
    "GaussianProcessPrior",
    "LinearOperatorUncertainty",
    
    # Workflow Reliability
    "ReliabilityEngine",
    "WorkflowReliability",
    "ReliabilityAssessment",
    "FailureProbability",
    "ReliabilityIndex",
    "FORMAnalysis",
    "SORMAnalysis",
    "MonteCarloReliability",
    "SystemReliability",
    "FaultTreeAnalysis",
    "EventTreeAnalysis",
    "ReliabilityMonitor",
    "UncertaintyBudget",
    "QualityAssurance",
]
