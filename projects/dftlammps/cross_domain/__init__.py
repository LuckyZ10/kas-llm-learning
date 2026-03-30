"""
DFT-LAMMPS 跨领域应用模块

实现不同材料领域间的知识迁移应用
"""

from .battery_to_catalyst import BatteryToCatalystTransfer, BatteryCatalystConfig
from .semiconductor_to_photovoltaic import SemiconductorToPVTransfer, SemiPVConfig
from .metal_to_ceramic import MetalToCeramicTransfer, MetalCeramicConfig
from .high_entropy_transfer import HighEntropyMaterialFramework, HighEntropyConfig

__all__ = [
    "BatteryToCatalystTransfer",
    "BatteryCatalystConfig",
    "SemiconductorToPVTransfer",
    "SemiPVConfig",
    "MetalToCeramicTransfer",
    "MetalCeramicConfig",
    "HighEntropyMaterialFramework",
    "HighEntropyConfig",
]
