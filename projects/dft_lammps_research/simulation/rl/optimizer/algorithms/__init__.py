#!/usr/bin/env python3
"""
RL算法模块 - 材料优化的强化学习算法实现

包含:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- DQN及其变体
- 多目标RL (NSGA-III、MOEA/D)
- 离线RL (CQL、Decision Transformer)
"""

from .ppo import PPOAgent, PPOConfig
from .sac import SACAgent, SACConfig
from .dqn import DQNAgent, DQNConfig, DuelingDQNAgent, RainbowDQNAgent
from .multi_objective import (
    MultiObjectiveRL,
    NSGA3Agent,
    MOEADAgent,
    ParetoFront,
)
from .offline_rl import (
    CQLAgent,
    CQLConfig,
    DecisionTransformerAgent,
    DTConfig,
)

__all__ = [
    # PPO
    'PPOAgent',
    'PPOConfig',
    
    # SAC
    'SACAgent',
    'SACConfig',
    
    # DQN
    'DQNAgent',
    'DQNConfig',
    'DuelingDQNAgent',
    'RainbowDQNAgent',
    
    # 多目标
    'MultiObjectiveRL',
    'NSGA3Agent',
    'MOEADAgent',
    'ParetoFront',
    
    # 离线RL
    'CQLAgent',
    'CQLConfig',
    'DecisionTransformerAgent',
    'DTConfig',
]
