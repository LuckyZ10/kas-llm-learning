"""
ML-DFT: Machine Learning Enhanced Density Functional Theory
机器学习增强密度泛函理论模块

本模块提供神经网络XC泛函和深度学习DFT方法，包括:
- neural_xc: 通用神经XC泛函框架
- deepks_interface: DeePKS深度KS方法接口
- dm21_interface: DM21神经XC泛函接口

主要功能:
1. 可训练的神经网络XC泛函
2. 密度特征提取
3. 物理约束实施
4. 与主流DFT代码集成

作者: DFT-LAMMPS Team
日期: 2026-03-09
"""

from .neural_xc import (
    NeuralXCConfig,
    XCConstraints,
    DensityFeatureExtractor,
    NeuralXCNetwork,
    NeuralXCLoss,
    NeuralXCTrainer,
    NeuralXCFunctional,
    create_pretrained_neural_xc,
    PBEExchangeCorrelation,
)

from .deepks_interface import (
    DeepKSConfig,
    DescriptorGenerator,
    DeepKSEnergyCorrector,
    DeepKSInterface,
    create_deepks_from_pyscf,
)

from .dm21_interface import (
    DM21Config,
    DM21FeatureExtractor,
    DM21NeuralNetwork,
    DM21ExchangeCorrelation,
    DM21Interface,
    create_dm21_functional,
    compare_dm21_vs_pbe,
)

__version__ = '1.0.0'
__author__ = 'DFT-LAMMPS Team'

__all__ = [
    # Neural XC
    'NeuralXCConfig',
    'XCConstraints',
    'DensityFeatureExtractor',
    'NeuralXCNetwork',
    'NeuralXCLoss',
    'NeuralXCTrainer',
    'NeuralXCFunctional',
    'create_pretrained_neural_xc',
    'PBEExchangeCorrelation',
    
    # DeepKS
    'DeepKSConfig',
    'DescriptorGenerator',
    'DeepKSEnergyCorrector',
    'DeepKSInterface',
    'create_deepks_from_pyscf',
    
    # DM21
    'DM21Config',
    'DM21FeatureExtractor',
    'DM21NeuralNetwork',
    'DM21ExchangeCorrelation',
    'DM21Interface',
    'create_dm21_functional',
    'compare_dm21_vs_pbe',
]
