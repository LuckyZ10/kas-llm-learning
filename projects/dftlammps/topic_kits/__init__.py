"""
DFT-LAMMPS Topic Kits
=====================

课题套件

提供针对特定研究领域的预配置工作流套件。

Modules:
    battery_research_kit: 电池研究套件
    catalyst_kit: 催化剂套件
    photovoltaic_kit: 光伏套件
    alloy_design_kit: 合金设计套件

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from .battery_research_kit import (
    BatteryResearchKit,
    BatteryMaterialSpec,
    IonConductivityResult,
    InterfaceStabilityResult,
    CycleLifePrediction,
    quick_battery_analysis,
    screen_battery_materials
)

from .catalyst_kit import (
    CatalystKit,
    CatalystSpec,
    AdsorptionResult,
    quick_catalyst_analysis
)

from .photovoltaic_kit import (
    PhotovoltaicKit,
    PVSpec,
    ElectronicProperties,
    OpticalProperties,
    TransportProperties,
    quick_pv_analysis,
    screen_solar_cell_materials
)

from .alloy_design_kit import (
    AlloyDesignKit,
    AlloyComposition,
    PhaseDiagramData,
    MechanicalProperties,
    CorrosionProperties,
    quick_alloy_design,
    screen_hea_compositions
)

__all__ = [
    # Battery Research
    'BatteryResearchKit',
    'BatteryMaterialSpec',
    'IonConductivityResult',
    'InterfaceStabilityResult',
    'CycleLifePrediction',
    'quick_battery_analysis',
    'screen_battery_materials',
    
    # Catalyst
    'CatalystKit',
    'CatalystSpec',
    'AdsorptionResult',
    'quick_catalyst_analysis',
    
    # Photovoltaic
    'PhotovoltaicKit',
    'PVSpec',
    'ElectronicProperties',
    'OpticalProperties',
    'TransportProperties',
    'quick_pv_analysis',
    'screen_solar_cell_materials',
    
    # Alloy Design
    'AlloyDesignKit',
    'AlloyComposition',
    'PhaseDiagramData',
    'MechanicalProperties',
    'CorrosionProperties',
    'quick_alloy_design',
    'screen_hea_compositions'
]