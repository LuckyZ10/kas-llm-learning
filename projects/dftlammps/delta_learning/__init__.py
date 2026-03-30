"""
Delta Learning (Δ-Learning) Module
Δ-学习模块

实现从低精度DFT到高精度方法的机器学习修正。

主要功能:
- 低精度DFT → ML → 高精度 (CCSD(T)/QMC)
- 能量/力/应力的多目标修正
- 转移学习策略
- SOAP/ACE描述符支持

示例:
    from dftlammps.delta_learning import DeltaLearningInterface
    
    # 创建Δ-学习模型
    delta_model = DeltaLearningInterface()
    
    # 训练
    delta_model.fit(training_structures)
    
    # 预测修正
    correction = delta_model.predict(structure)
    high_accuracy_energy = low_accuracy_energy + correction['energy_delta']

作者: DFT-LAMMPS Team
日期: 2026-03-09
"""

from .delta_learning import (
    DeltaLearningConfig,
    SOAPDescriptor,
    ACEDescriptor,
    DeltaLearningModel,
    DeltaLearningLoss,
    DeltaLearningDataset,
    DeltaLearningTrainer,
    DeltaLearningInterface,
    create_delta_learning_pipeline,
    transfer_learning_delta_model,
)

__version__ = '1.0.0'

__all__ = [
    'DeltaLearningConfig',
    'SOAPDescriptor',
    'ACEDescriptor',
    'DeltaLearningModel',
    'DeltaLearningLoss',
    'DeltaLearningDataset',
    'DeltaLearningTrainer',
    'DeltaLearningInterface',
    'create_delta_learning_pipeline',
    'transfer_learning_delta_model',
]
