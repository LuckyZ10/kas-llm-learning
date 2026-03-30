"""
材料基础大模型模块
提供大规模预训练、多任务学习和少样本学习功能
"""

from .material_foundation import (
    MaterialFoundationModel,
    FoundationModelConfig,
    MultiTaskTrainer,
    PrototypicalNetwork,
    ZeroShotPredictor,
    PretrainingTasks
)

from .applications import (
    ZeroShotMaterialDiscovery,
    CrossDomainTransfer,
    LargeScaleScreening,
    ApplicationConfig
)

__all__ = [
    # 基础模型
    'MaterialFoundationModel',
    'FoundationModelConfig',
    'MultiTaskTrainer',
    'PrototypicalNetwork',
    'ZeroShotPredictor',
    'PretrainingTasks',
    
    # 应用案例
    'ZeroShotMaterialDiscovery',
    'CrossDomainTransfer',
    'LargeScaleScreening',
    'ApplicationConfig',
]
