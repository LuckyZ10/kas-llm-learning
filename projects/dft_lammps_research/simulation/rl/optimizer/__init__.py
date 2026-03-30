#!/usr/bin/env python3
"""
强化学习材料优化引擎 (RL Optimizer for Materials Design)

一个基于强化学习的材料设计和优化引擎，用于：
1. 晶体结构优化
2. 化学组成设计
3. 材料性质预测与优化
4. 与DFT/MD计算耦合
5. 人机协作材料发现

主要组件:
- 环境设计: 晶体结构操作空间、化学组成调整空间
- RL算法: PPO、SAC、DQN、多目标RL、离线RL
- 材料场景: 电池、催化剂、合金、拓扑材料
- DFT/MD耦合: 计算奖励函数
- 可解释性: 注意力可视化、轨迹分析

作者: DFT-ML Research Team
日期: 2025-03-11
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DFT-ML Research Team"

# 环境模块
from .environment import (
    CrystalStructureEnv,
    CompositionEnv,
    MaterialOptEnv,
    StateRepresentation,
    ActionSpace,
)

# 算法模块
from .algorithms import (
    PPOAgent,
    SACAgent,
    DQNAgent,
    MultiObjectiveRL,
    NSGA3Agent,
    MOEADAgent,
    CQLAgent,
    DecisionTransformerAgent,
)

# 表示学习
from .representations import (
    CrystalGraphEncoder,
    CompositionEncoder,
    StateEncoder,
)

# 奖励函数
from .rewards import (
    EnergyReward,
    StabilityReward,
    PropertyReward,
    MultiObjectiveReward,
    RewardComposer,
)

# 材料场景
from .scenarios import (
    BatteryOptimizer,
    CatalystOptimizer,
    AlloyOptimizer,
    TopologicalOptimizer,
)

# DFT/MD耦合
from .coupling import (
    DFTCoupling,
    MLCoupling,
    ActiveLearningCoupling,
    HumanInTheLoop,
)

# 可解释性
from .explainability import (
    AttentionVisualizer,
    TrajectoryAnalyzer,
    ChemicalIntuitionExtractor,
    CounterfactualExplainer,
)

# 可视化
from .visualization import (
    OptimizationPlotter,
    StructureVisualizer,
    RewardVisualizer,
)

__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    
    # 环境
    'CrystalStructureEnv',
    'CompositionEnv',
    'MaterialOptEnv',
    'StateRepresentation',
    'ActionSpace',
    
    # 算法
    'PPOAgent',
    'SACAgent',
    'DQNAgent',
    'MultiObjectiveRL',
    'NSGA3Agent',
    'MOEADAgent',
    'CQLAgent',
    'DecisionTransformerAgent',
    
    # 表示
    'CrystalGraphEncoder',
    'CompositionEncoder',
    'StateEncoder',
    
    # 奖励
    'EnergyReward',
    'StabilityReward',
    'PropertyReward',
    'MultiObjectiveReward',
    'RewardComposer',
    
    # 场景
    'BatteryOptimizer',
    'CatalystOptimizer',
    'AlloyOptimizer',
    'TopologicalOptimizer',
    
    # 耦合
    'DFTCoupling',
    'MLCoupling',
    'ActiveLearningCoupling',
    'HumanInTheLoop',
    
    # 可解释性
    'AttentionVisualizer',
    'TrajectoryAnalyzer',
    'ChemicalIntuitionExtractor',
    'CounterfactualExplainer',
    
    # 可视化
    'OptimizationPlotter',
    'StructureVisualizer',
    'RewardVisualizer',
]
