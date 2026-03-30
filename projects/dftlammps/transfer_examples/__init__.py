"""
DFT-LAMMPS 迁移学习示例模块

展示迁移学习在材料发现中的应用
"""

from .few_shot_material_discovery import CrossDomainFewShotLearner, FewShotConfig
from .zero_shot_prediction import ZeroShotMaterialPredictor, AttributeBasedZeroShot, ZeroShotConfig
from .continual_learning_demo import ContinualMaterialLearner, ContinualLearningConfig

__all__ = [
    "CrossDomainFewShotLearner",
    "FewShotConfig",
    "ZeroShotMaterialPredictor",
    "AttributeBasedZeroShot",
    "ZeroShotConfig",
    "ContinualMaterialLearner",
    "ContinualLearningConfig",
]
