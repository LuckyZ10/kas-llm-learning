"""
Inverse Design Module for Differentiable DFT
============================================

可微分DFT的逆向设计模块，用于目标导向的材料设计。

主要子模块:
- core: 核心逆向设计框架
- band_gap_design: 带隙逆向设计
- ion_conductor_design: 离子导体逆向设计

使用示例:
    from differentiable_dft.inverse_design import (
        InverseDesignOptimizer,
        BandGapCalculator,
        IonConductorTarget
    )
    
    # 创建设计目标
    target = IonConductorTarget(
        ion_type='Li',
        target_conductivity=1e-3
    )
    
    # 运行优化
    optimizer = InverseDesignOptimizer(objective)
    result = optimizer.optimize(initial_structure)
"""

# 从core模块导入核心类
from .core import (
    DesignTarget,
    DesignSpace,
    ParameterizedStructure,
    FractionalCoordinateStructure,
    WyckoffPositionStructure,
    ObjectiveFunction,
    InverseDesignOptimizer,
    MultiObjectiveOptimizer,
)

# 从band_gap_design导入带隙设计类
try:
    from .band_gap_design import (
        BandGapTarget,
        BandGapCalculator,
        BandGapObjective,
        SolarCellOptimizer,
        LEDMaterialDesigner,
        TransparentConductorOptimizer,
    )
except ImportError:
    pass

# 从ion_conductor_design导入离子导体设计类
try:
    from .ion_conductor_design import (
        IonConductorTarget,
        IonMigrationAnalyzer,
        SolidElectrolyteDesigner,
        NASICONDesigner,
        SulfideElectrolyteDesigner,
    )
except ImportError:
    pass

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "DesignTarget",
    "DesignSpace", 
    "ParameterizedStructure",
    "FractionalCoordinateStructure",
    "WyckoffPositionStructure",
    "ObjectiveFunction",
    "InverseDesignOptimizer",
    "MultiObjectiveOptimizer",
    
    # Band gap design
    "BandGapTarget",
    "BandGapCalculator",
    "BandGapObjective",
    "SolarCellOptimizer",
    "LEDMaterialDesigner",
    "TransparentConductorOptimizer",
    
    # Ion conductor design
    "IonConductorTarget",
    "IonMigrationAnalyzer",
    "SolidElectrolyteDesigner",
    "NASICONDesigner",
    "SulfideElectrolyteDesigner",
]
