"""
Phase Field Coupling
====================
多尺度耦合模块

实现相场模型与DFT/MD计算的双向耦合。
"""

from .dft_coupling import DFTCoupling, DFTCouplingConfig
from .md_coupling import MDCoupling, MDCouplingConfig
from .parameter_transfer import ParameterTransfer, TransferConfig

__all__ = [
    'DFTCoupling',
    'DFTCouplingConfig',
    'MDCoupling',
    'MDCouplingConfig',
    'ParameterTransfer',
    'TransferConfig',
]
