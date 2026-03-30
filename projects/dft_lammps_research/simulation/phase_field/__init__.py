"""
Phase Field - DFT Multi-scale Coupling Module
===============================================
相场-DFT多尺度耦合模块

填补微观到介观的尺度鸿沟，实现从DFT/MD到相场的参数传递
和从相场到DFT的结构反馈。

核心功能:
1. 物理模型: Cahn-Hilliard, Allen-Cahn, 电化学相场, 力学-化学耦合
2. DFT/MD耦合: 热力学参数提取、动力学参数传递、多尺度自动化
3. 数值实现: 有限差分/有限元、GPU加速、并行计算、自适应网格
4. 应用场景: SEI生长、沉淀相演化、晶界迁移、催化剂重构

作者: Phase Field Development Team
日期: 2026-03-11
"""

__version__ = "1.0.0"
__author__ = "DFT-MD Coupling Team"

# 核心物理模型
from .core.cahn_hilliard import CahnHilliardSolver, CahnHilliardConfig
from .core.allen_cahn import AllenCahnSolver, AllenCahnConfig
from .core.electrochemical import ElectrochemicalPhaseField, ElectrochemicalConfig
from .core.mechanochemistry import MechanoChemicalSolver, MechanoChemicalConfig

# 求解器
from .solvers.finite_difference import FiniteDifferenceSolver, FDSConfig
from .solvers.finite_element import FiniteElementSolver, FEConfig
from .solvers.gpu_solver import GPUSolver, GPUConfig
from .solvers.parallel_solver import ParallelSolver, ParallelConfig
from .solvers.adaptive_mesh import AdaptiveMesh, AMRConfig

# 耦合接口
from .coupling.dft_coupling import DFTCoupling, DFTCouplingConfig
from .coupling.md_coupling import MDCoupling, MDCouplingConfig
from .coupling.parameter_transfer import ParameterTransfer, TransferConfig

# 应用模块
from .applications.sei_growth import SEIGrowthSimulator, SEIConfig
from .applications.precipitation import PrecipitationSimulator, PrecipConfig
from .applications.grain_boundary import GrainBoundarySimulator, GBConfig
from .applications.catalyst_reconstruction import CatalystReconstructor, CatalystConfig

# 工作流
from .workflow import PhaseFieldWorkflow, WorkflowConfig

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # Cahn-Hilliard
    "CahnHilliardSolver",
    "CahnHilliardConfig",
    
    # Allen-Cahn
    "AllenCahnSolver", 
    "AllenCahnConfig",
    
    # 电化学相场
    "ElectrochemicalPhaseField",
    "ElectrochemicalConfig",
    
    # 力学-化学耦合
    "MechanoChemicalSolver",
    "MechanoChemicalConfig",
    
    # 求解器
    "FiniteDifferenceSolver",
    "FDSConfig",
    "FiniteElementSolver",
    "FEConfig",
    "GPUSolver",
    "GPUConfig",
    "ParallelSolver",
    "ParallelConfig",
    "AdaptiveMesh",
    "AMRConfig",
    
    # 耦合
    "DFTCoupling",
    "DFTCouplingConfig",
    "MDCoupling",
    "MDCouplingConfig",
    "ParameterTransfer",
    "TransferConfig",
    
    # 应用
    "SEIGrowthSimulator",
    "SEIConfig",
    "PrecipitationSimulator",
    "PrecipConfig",
    "GrainBoundarySimulator",
    "GBConfig",
    "CatalystReconstructor",
    "CatalystConfig",
    
    # 工作流
    "PhaseFieldWorkflow",
    "WorkflowConfig",
]
