"""
KAS Deep Reinforcement Learning Enhancement Module

This module provides deep reinforcement learning capabilities for the KAS Agent system.

Main components:
    - core: State space, action space, and reward function definitions
    - algorithms: PPO, DDPG, SAC implementations
    - meta_learning: MAML and Reptile for quick adaptation
    - training: Training framework and environment
    - integration: LLMClient integration and deployment tools
"""

__version__ = "0.1.0"
__author__ = "KAS Team"

# Core components
from core.state_space import (
    StateSpace,
    StateEncoder,
    AgentState,
    TaskFeatures,
    UserFeedback,
    TelemetryStateTracker
)

from core.action_space import (
    ActionSpace,
    ActionNetwork,
    PromptAction,
    TemplateAction,
    ParameterAction
)

from core.reward import (
    RewardFunction,
    RewardConfig,
    InteractionOutcome,
    CurriculumReward,
    MultiObjectiveReward
)

# Algorithms
from algorithms.ppo import PPOAgent, PPOConfig
from algorithms.ddpg import DDPGAgent, DDPGConfig
from algorithms.sac import SACAgent, SACConfig

# Meta-learning
from meta_learning.maml import MAML, MAMLAgentAdapter
from meta_learning.reptile import Reptile, ReptileAgentPolicy
from meta_learning.encoders import (
    ProjectFeatureEncoder,
    TelemetryLSTMEncoder,
    MultiModalProjectEncoder
)

# Training
from training.environment import KASAgentEnv, CurriculumEnv
from training.trainer import Trainer, TrainingConfig, MetaTrainer
from training.online_learning import OnlineLearner, ContinualLearningManager

# Integration
from integration.llm_client import LLMClientAdapter, CompatibleLLMClient
from integration.fallback import FallbackManager, DRLFallbackWrapper
from integration.deployment import (
    CanaryDeployment,
    ABTestDeployment,
    ModelRegistry
)

__all__ = [
    # Core
    'StateSpace',
    'StateEncoder',
    'AgentState',
    'TaskFeatures',
    'UserFeedback',
    'ActionSpace',
    'RewardFunction',
    'RewardConfig',
    
    # Algorithms
    'PPOAgent',
    'PPOConfig',
    'DDPGAgent',
    'DDPGConfig',
    'SACAgent',
    'SACConfig',
    
    # Meta-learning
    'MAML',
    'Reptile',
    'ProjectFeatureEncoder',
    'TelemetryLSTMEncoder',
    
    # Training
    'KASAgentEnv',
    'CurriculumEnv',
    'Trainer',
    'TrainingConfig',
    'OnlineLearner',
    
    # Integration
    'LLMClientAdapter',
    'FallbackManager',
    'CanaryDeployment',
]
