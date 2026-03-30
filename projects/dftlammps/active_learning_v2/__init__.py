#!/usr/bin/env python3
"""
主动学习V2模块 - 先进的主动学习策略库
Advanced Active Learning Strategies for ML Potentials

本模块实现了2024年最先进的主动学习策略，用于减少DFT计算成本：
1. 贝叶斯优化与主动学习结合 (Bayesian Optimization + Active Learning)
2. DPP多样性感知批量采样 (DPP-based Diversity-aware Batch Sampling)
3. 多保真度主动学习 (Multi-Fidelity Active Learning)
4. 证据学习不确定性量化 (Evidential Learning UQ)
5. 自适应混合策略 (Adaptive Hybrid Strategy)

作者: DFT-ML Research Team
日期: 2025-03-10
版本: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "DFT-ML Research Team"

from .strategies import (
    BayesianOptimizationStrategy,
    DPPDiversityStrategy,
    MultiFidelityStrategy,
    EvidentialLearningStrategy,
    AdaptiveHybridStrategy,
    StrategyConfig,
    SelectionResult,
    ActiveLearningStrategy,
)

from .uncertainty import (
    EnsembleUncertainty,
    MCDropoutUncertainty,
    EvidentialUncertainty,
    BayesianGPUncertainty,
    UncertaintyResult,
    UncertaintyQuantifier,
)

from .adaptive import (
    AdaptiveSampler,
    StrategySelector,
    PerformanceMonitor,
    SamplingPhase,
    PerformanceMetrics,
    StrategyRecommendation,
)

__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    # 策略基类
    'ActiveLearningStrategy',
    'StrategyConfig',
    'SelectionResult',
    # 策略
    'BayesianOptimizationStrategy',
    'DPPDiversityStrategy', 
    'MultiFidelityStrategy',
    'EvidentialLearningStrategy',
    'AdaptiveHybridStrategy',
    # 不确定性量化基类
    'UncertaintyQuantifier',
    'UncertaintyResult',
    # 不确定性量化
    'EnsembleUncertainty',
    'MCDropoutUncertainty',
    'EvidentialUncertainty',
    'BayesianGPUncertainty',
    # 自适应
    'AdaptiveSampler',
    'StrategySelector',
    'PerformanceMonitor',
    'SamplingPhase',
    'PerformanceMetrics',
]
