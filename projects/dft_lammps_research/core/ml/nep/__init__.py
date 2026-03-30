#!/usr/bin/env python3
"""
NEP Training Enhanced Module
============================
强化版NEP (Neural Equivariant Potential) 训练模块

特性:
1. 高级训练策略: 学习率调度、早停、模型集成
2. 多精度支持: FP32/FP16/BF16混合训练
3. 分布式训练: 多GPU并行 (DDP)
4. 主动学习: 自动识别关键训练样本
5. 预训练模型库与迁移学习
6. 实时训练监控和可视化

作者: DFT-ML Research Team
版本: 2.0.0
日期: 2026-03-11
"""

__version__ = "2.0.0"
__author__ = "DFT-ML Research Team"

# 核心组件
from .core import (
    NEPDataConfig,
    NEPModelConfig,
    NEPTrainingConfig,
    NEPCheckpoint,
    TrainingState,
    PrecisionMode,
    DistributedConfig,
)

# 训练器
from .trainer import (
    NEPTrainerV2,
    DistributedNEPTrainer,
    MixedPrecisionTrainer,
    EnsembleTrainer,
)

# 数据准备
from .data import (
    NEPDataPreparer,
    NEPDataLoader,
    NEPDataset,
    DataAugmenter,
)

# 高级训练策略
from .strategies import (
    LRScheduler,
    CosineAnnealingScheduler,
    ExponentialDecayScheduler,
    WarmupScheduler,
    EarlyStopping,
    ModelEnsemble,
    EnsembleConfig,
)

# 主动学习
from .active_learning import (
    NEPActiveLearning,
    UncertaintySampler,
    DiversitySampler,
    QueryStrategy,
    ALConfig,
)

# 模型库
from .model_library import (
    NEPModelLibrary,
    PretrainedModel,
    ModelVersion,
    TransferLearning,
    BenchmarkSuite,
)

# 监控与可视化
from .monitoring import (
    TrainingMonitor,
    MetricsTracker,
    TensorBoardLogger,
    WandbLogger,
    WebSocketLogger,
    TrainingDashboard,
    RealTimePlotter,
)

# 性能优化
from .optimization import (
    GPUMemoryOptimizer,
    DataLoaderOptimizer,
    TrainingSpeedOptimizer,
    InferenceOptimizer,
)

# 平台集成
from .integration import (
    NEPWorkflowModule,
    NEPNodeExecutor,
    NEPWorkflowBuilder,
    NEPWorkflowConfig,
    train_nep,
)

__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    
    # 核心配置
    'NEPDataConfig',
    'NEPModelConfig', 
    'NEPTrainingConfig',
    'NEPCheckpoint',
    'TrainingState',
    'PrecisionMode',
    'DistributedConfig',
    
    # 训练器
    'NEPTrainerV2',
    'DistributedNEPTrainer',
    'MixedPrecisionTrainer',
    'EnsembleTrainer',
    
    # 数据
    'NEPDataPreparer',
    'NEPDataLoader',
    'NEPDataset',
    'DataAugmenter',
    
    # 策略
    'LRScheduler',
    'CosineAnnealingScheduler',
    'ExponentialDecayScheduler',
    'WarmupScheduler',
    'EarlyStopping',
    'ModelEnsemble',
    'EnsembleConfig',
    
    # 主动学习
    'NEPActiveLearning',
    'UncertaintySampler',
    'DiversitySampler',
    'QueryStrategy',
    'ALConfig',
    
    # 模型库
    'NEPModelLibrary',
    'PretrainedModel',
    'ModelVersion',
    'TransferLearning',
    'BenchmarkSuite',
    
    # 监控
    'TrainingMonitor',
    'MetricsTracker',
    'TensorBoardLogger',
    'WandbLogger',
    'WebSocketLogger',
    'TrainingDashboard',
    'RealTimePlotter',
    
    # 优化
    'GPUMemoryOptimizer',
    'DataLoaderOptimizer',
    'TrainingSpeedOptimizer',
    'InferenceOptimizer',
    
    # 集成
    'NEPWorkflowModule',
    'NEPNodeExecutor',
    'NEPWorkflowBuilder',
    'NEPWorkflowConfig',
    'train_nep',
]
