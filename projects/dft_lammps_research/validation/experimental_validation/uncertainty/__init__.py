"""
Uncertainty Quantification
==========================
不确定性量化模块

提供误差传播、置信区间估计和敏感性分析功能
"""

from .error_propagation import (
    ErrorPropagator,
    ConfidenceIntervalEstimator,
    SensitivityAnalyzer
)

__all__ = [
    'ErrorPropagator',
    'ConfidenceIntervalEstimator',
    'SensitivityAnalyzer',
]
