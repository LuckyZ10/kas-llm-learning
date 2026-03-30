"""
DFT-LAMMPS RL Optimization Module
====================================

强化学习在材料发现和工艺优化中的应用

主要功能:
- GFlowNet分子生成器
- 离线强化学习(Offline RL)用于材料设计
- 工艺参数优化 (贝叶斯优化 vs RL)
- 奖励函数设计工具
- 高通量筛选工作流集成

Author: DFT-LAMMPS RL Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS RL Team"

# GFlowNet models
from .models.gflownet import (
    GFlowNet,
    GFlowNetGenerator,
    TrajectoryBalance,
    FlowMatchingGFlowNet,
    MoleculeGFlowNet,
    CrystalGFlowNet,
)

from .models.policy import (
    PolicyNetwork,
    StochasticPolicy,
    DeterministicPolicy,
    CategoricalPolicy,
)

# Offline RL models
from .models.offline_rl import (
    CQL,
    IQL,
    DecisionTransformer,
    TrajectoryTransformer,
    OfflineRLAgent,
)

# Environments
from .environments.molecule_env import (
    MoleculeEnvironment,
    MolecularGraphEnv,
    SMILESEnv,
)

from .environments.material_env import (
    MaterialDesignEnv,
    CompositionEnv,
    StructureEnv,
)

from .environments.process_env import (
    ProcessOptimizationEnv,
    SynthesisEnv,
    ParameterEnv,
)

# Reward functions
from .rewards.reward_design import (
    RewardFunction,
    CompositeReward,
    MultiObjectiveReward,
    PropertyReward,
    DiversityReward,
    ValidityReward,
)

from .rewards.reward_learning import (
    RewardModel,
    PreferenceLearning,
    InverseRL,
    RewardShaping,
)

# Training
from .training.gflownet_trainer import GFlowNetTrainer
from .training.offline_trainer import OfflineRLTrainer
from .training.process_trainer import ProcessOptimizationTrainer

# Integration
from .integration.screening_rl import ScreeningRLIntegration
from .integration.active_rl import ActiveLearningRL
from .integration.multi_objective import MultiObjectiveOptimizer

__all__ = [
    # GFlowNet
    "GFlowNet",
    "GFlowNetGenerator",
    "TrajectoryBalance",
    "FlowMatchingGFlowNet",
    "MoleculeGFlowNet",
    "CrystalGFlowNet",
    # Policy
    "PolicyNetwork",
    "StochasticPolicy",
    "DeterministicPolicy",
    "CategoricalPolicy",
    # Offline RL
    "CQL",
    "IQL",
    "DecisionTransformer",
    "TrajectoryTransformer",
    "OfflineRLAgent",
    # Environments
    "MoleculeEnvironment",
    "MolecularGraphEnv",
    "SMILESEnv",
    "MaterialDesignEnv",
    "CompositionEnv",
    "StructureEnv",
    "ProcessOptimizationEnv",
    "SynthesisEnv",
    "ParameterEnv",
    # Rewards
    "RewardFunction",
    "CompositeReward",
    "MultiObjectiveReward",
    "PropertyReward",
    "DiversityReward",
    "ValidityReward",
    "RewardModel",
    "PreferenceLearning",
    "InverseRL",
    "RewardShaping",
    # Trainers
    "GFlowNetTrainer",
    "OfflineRLTrainer",
    "ProcessOptimizationTrainer",
    # Integration
    "ScreeningRLIntegration",
    "ActiveLearningRL",
    "MultiObjectiveOptimizer",
]


def get_model(model_type: str, **kwargs):
    """Factory function to get RL model by type."""
    models = {
        # GFlowNet
        "gflownet": GFlowNet,
        "molecule_gfn": MoleculeGFlowNet,
        "crystal_gfn": CrystalGFlowNet,
        "flow_matching": FlowMatchingGFlowNet,
        # Offline RL
        "cql": CQL,
        "iql": IQL,
        "decision_transformer": DecisionTransformer,
        "trajectory_transformer": TrajectoryTransformer,
    }
    
    if model_type.lower() not in models:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type.lower()](**kwargs)


def get_environment(env_type: str, **kwargs):
    """Factory function to get environment by type."""
    envs = {
        "molecule": MoleculeEnvironment,
        "graph": MolecularGraphEnv,
        "smiles": SMILESEnv,
        "material": MaterialDesignEnv,
        "composition": CompositionEnv,
        "structure": StructureEnv,
        "process": ProcessOptimizationEnv,
        "synthesis": SynthesisEnv,
        "parameter": ParameterEnv,
    }
    
    if env_type.lower() not in envs:
        raise ValueError(f"Unknown environment: {env_type}. Available: {list(envs.keys())}")
    
    return envs[env_type.lower()](**kwargs)
