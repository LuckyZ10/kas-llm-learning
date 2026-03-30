"""
不确定性量化模块 - Uncertainty Quantification for Materials

本模块提供材料科学中的不确定性量化工具：
- 贝叶斯神经网络
- 集合方法
- 共形预测

主要组件:
    - bayesian_nn: 贝叶斯神经网络、蒙特卡洛Dropout、贝叶斯优化
    - ensemble_methods: 深度集合、快照集成、加权集合
    - conformal_prediction: 标准/自适应/分位数共形预测

作者: Causal AI Team
"""

from .bayesian_nn import (
    # 贝叶斯神经网络
    BayesianNeuralNetwork,
    BNNTrainer,
    
    # MC Dropout
    MCDropoutNetwork,
    
    # 高斯过程
    GaussianProcessApproximation,
    
    # 贝叶斯优化
    BayesianOptimization,
    
    # 数据结构
    UncertaintyEstimate
)

from .ensemble_methods import (
    # 集合方法
    DeepEnsemble,
    SnapshotEnsemble,
    WeightedEnsemble,
    BootstrapAggregator,
    EnsembleCalibration,
    
    # 数据结构
    EnsemblePrediction
)

from .conformal_prediction import (
    # 共形预测器
    StandardConformalPredictor,
    AdaptiveConformalPredictor,
    ConformalizedQuantileRegression,
    MultiLabelConformalPredictor,
    TimeSeriesConformalPredictor,
    ConformalPredictionPipeline,
    
    # 数据结构
    ConformalPrediction
)

__version__ = "1.0.0"
__all__ = [
    # Bayesian NN
    "BayesianNeuralNetwork",
    "BNNTrainer",
    "MCDropoutNetwork",
    "GaussianProcessApproximation",
    "BayesianOptimization",
    "UncertaintyEstimate",
    
    # Ensemble Methods
    "DeepEnsemble",
    "SnapshotEnsemble",
    "WeightedEnsemble",
    "BootstrapAggregator",
    "EnsembleCalibration",
    "EnsemblePrediction",
    
    # Conformal Prediction
    "StandardConformalPredictor",
    "AdaptiveConformalPredictor",
    "ConformalizedQuantileRegression",
    "MultiLabelConformalPredictor",
    "TimeSeriesConformalPredictor",
    "ConformalPredictionPipeline",
    "ConformalPrediction"
]
