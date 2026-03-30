"""
DFT-LAMMPS 应用案例模块

提供自驱动实验室应用案例：
- 电池材料发现
- 机器人催化实验
- 自动合金设计
"""

from .autonomous_battery_discovery import (
    BatteryMaterial,
    OptimizationConfig,
    BatteryMaterialGenerator,
    BatteryPropertyPredictor,
    AutonomousBatteryDiscovery,
    run_battery_discovery
)

from .robotic_catalysis import (
    ReactionTypeCatalysis,
    Catalyst,
    ReactionCondition,
    CatalyticResult,
    CatalystLibrary,
    ReactionSetupPlanner,
    CatalyticActivityPredictor,
    RoboticCatalysisPlatform,
    run_catalysis_optimization
)

from .autonomous_alloy_design import (
    AlloySystem,
    AlloyComposition,
    PhasePrediction,
    MechanicalProperties,
    AlloyCandidate,
    AlloyDatabase,
    AlloyDesigner,
    RoboticAlloySynthesizer,
    AlloyTester,
    AutonomousAlloyDevelopment,
    develop_alloy
)

__all__ = [
    # Battery Materials
    'BatteryMaterial',
    'OptimizationConfig',
    'BatteryMaterialGenerator',
    'BatteryPropertyPredictor',
    'AutonomousBatteryDiscovery',
    'run_battery_discovery',
    
    # Catalysis
    'ReactionTypeCatalysis',
    'Catalyst',
    'ReactionCondition',
    'CatalyticResult',
    'CatalystLibrary',
    'ReactionSetupPlanner',
    'CatalyticActivityPredictor',
    'RoboticCatalysisPlatform',
    'run_catalysis_optimization',
    
    # Alloy Design
    'AlloySystem',
    'AlloyComposition',
    'PhasePrediction',
    'MechanicalProperties',
    'AlloyCandidate',
    'AlloyDatabase',
    'AlloyDesigner',
    'RoboticAlloySynthesizer',
    'AlloyTester',
    'AutonomousAlloyDevelopment',
    'develop_alloy'
]
