#!/usr/bin/env python3
"""
RL环境模块 - 材料优化的强化学习环境实现

包含:
- CrystalStructureEnv: 晶体结构操作环境
- CompositionEnv: 化学组成优化环境
- MaterialOptEnv: 通用材料优化环境基类
"""

from .base_env import (
    MaterialOptEnv,
    StateRepresentation,
    ActionSpace,
    StepResult,
    EnvConfig,
)

from .crystal_env import (
    CrystalStructureEnv,
    CrystalAction,
    StructureModifier,
)

from .composition_env import (
    CompositionEnv,
    CompositionAction,
    ElementSelector,
)

__all__ = [
    # 基础类
    'MaterialOptEnv',
    'StateRepresentation',
    'ActionSpace',
    'StepResult',
    'EnvConfig',
    
    # 晶体环境
    'CrystalStructureEnv',
    'CrystalAction',
    'StructureModifier',
    
    # 组成环境
    'CompositionEnv',
    'CompositionAction',
    'ElementSelector',
]
