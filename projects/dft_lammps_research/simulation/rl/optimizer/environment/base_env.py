#!/usr/bin/env python3
"""
基础环境类 - 材料优化的强化学习环境基类

定义了通用的材料优化环境接口和共享功能。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """动作类型枚举"""
    ADD_ATOM = "add_atom"
    REMOVE_ATOM = "remove_atom"
    MOVE_ATOM = "move_atom"
    REPLACE_ATOM = "replace_atom"
    CHANGE_LATTICE = "change_lattice"
    SWAP_ATOMS = "swap_atoms"
    COMPOSITION_CHANGE = "composition_change"
    TERMINATE = "terminate"


@dataclass
class EnvConfig:
    """环境配置"""
    max_steps: int = 100
    max_atoms: int = 100
    min_atoms: int = 2
    lattice_bounds: Tuple[float, float] = (2.0, 20.0)  # 晶格常数范围 (Å)
    element_set: List[str] = field(default_factory=lambda: [
        'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb'
    ])
    allowed_species: Optional[List[str]] = None
    reward_scale: float = 1.0
    penalty_scale: float = 0.1
    verbose: bool = False


@dataclass
class StepResult:
    """单步结果"""
    state: np.ndarray
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_tuple(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """转换为Gym风格的元组"""
        return (self.state, self.reward, self.done, self.truncated, self.info)


class StateRepresentation:
    """状态表示类 - 将材料结构转换为向量表示"""
    
    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.feature_dim = 0  # 将在子类中定义
    
    def encode(self, structure: Any) -> np.ndarray:
        """将结构编码为向量"""
        raise NotImplementedError
    
    def decode(self, state: np.ndarray) -> Any:
        """将向量解码为结构"""
        raise NotImplementedError
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.feature_dim


class ActionSpace:
    """动作空间定义"""
    
    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.action_types = list(ActionType)
        self.n_actions = len(self.action_types)
    
    def sample(self) -> Tuple[ActionType, Dict]:
        """随机采样动作"""
        action_type = np.random.choice(self.action_types)
        params = self._sample_params(action_type)
        return action_type, params
    
    def _sample_params(self, action_type: ActionType) -> Dict:
        """为动作类型采样参数"""
        params = {}
        
        if action_type == ActionType.ADD_ATOM:
            params['element'] = np.random.choice(self.config.element_set)
            params['position'] = np.random.uniform(0, 1, 3)  # 分数坐标
            
        elif action_type == ActionType.REMOVE_ATOM:
            params['atom_index'] = None  # 将在执行时确定
            
        elif action_type == ActionType.MOVE_ATOM:
            params['atom_index'] = None
            params['displacement'] = np.random.uniform(-0.5, 0.5, 3)
            
        elif action_type == ActionType.REPLACE_ATOM:
            params['atom_index'] = None
            params['new_element'] = np.random.choice(self.config.element_set)
            
        elif action_type == ActionType.CHANGE_LATTICE:
            params['scale'] = np.random.uniform(0.9, 1.1)
            
        elif action_type == ActionType.SWAP_ATOMS:
            params['index1'] = None
            params['index2'] = None
            
        elif action_type == ActionType.COMPOSITION_CHANGE:
            params['element'] = np.random.choice(self.config.element_set)
            params['delta'] = np.random.choice([-1, 1])
            
        return params
    
    def get_action_dim(self) -> int:
        """获取动作空间维度"""
        # 动作类型 + 参数
        return self.n_actions + 10  # 简化表示


class MaterialOptEnv(ABC):
    """
    材料优化环境基类
    
    遵循OpenAI Gym接口风格的强化学习环境。
    支持晶体结构操作和化学组成调整。
    """
    
    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.action_space = ActionSpace(self.config)
        self.state_rep = None  # 在子类中初始化
        
        self.current_step = 0
        self.current_structure = None
        self.history = []
        self.episode_reward = 0.0
        
        # 约束检查
        self.constraints = []
    
    @abstractmethod
    def reset(self, initial_structure: Optional[Any] = None) -> np.ndarray:
        """
        重置环境
        
        Args:
            initial_structure: 可选的初始结构
            
        Returns:
            初始状态向量
        """
        pass
    
    @abstractmethod
    def step(self, action: Tuple[ActionType, Dict]) -> StepResult:
        """
        执行一步
        
        Args:
            action: (动作类型, 参数字典)
            
        Returns:
            StepResult
        """
        pass
    
    @abstractmethod
    def compute_reward(self, structure: Any, action_info: Dict) -> float:
        """
        计算奖励
        
        Args:
            structure: 当前结构
            action_info: 动作相关信息
            
        Returns:
            奖励值
        """
        pass
    
    def add_constraint(self, constraint_fn: callable):
        """添加约束函数"""
        self.constraints.append(constraint_fn)
    
    def check_constraints(self, structure: Any) -> Tuple[bool, List[str]]:
        """
        检查结构是否满足所有约束
        
        Returns:
            (是否满足, 违反的约束列表)
        """
        violations = []
        for constraint in self.constraints:
            is_valid, message = constraint(structure)
            if not is_valid:
                violations.append(message)
        
        return len(violations) == 0, violations
    
    def is_terminal(self, structure: Any) -> bool:
        """检查是否达到终止条件"""
        # 基础终止条件
        if self.current_step >= self.config.max_steps:
            return True
        
        # 检查约束
        is_valid, violations = self.check_constraints(structure)
        if not is_valid and len(violations) > 0:
            return True
        
        return False
    
    def get_observation(self) -> np.ndarray:
        """获取当前观测"""
        if self.current_structure is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return self.state_rep.encode(self.current_structure)
    
    def render(self, mode: str = 'human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Structure: {self.current_structure}")
            print(f"Episode Reward: {self.episode_reward}")
        elif mode == 'ascii':
            # ASCII艺术渲染
            print(f"[{self.current_step}] {'='*20}")
    
    def close(self):
        """关闭环境"""
        pass
    
    def seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
    
    def get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'step': self.current_step,
            'n_atoms': self._get_atom_count(),
            'history_length': len(self.history),
        }
    
    def _get_atom_count(self) -> int:
        """获取当前原子数"""
        if self.current_structure is None:
            return 0
        return len(getattr(self.current_structure, 'positions', []))
    
    def get_history(self) -> List[Dict]:
        """获取优化历史"""
        return self.history.copy()


# 常用约束函数

def max_atoms_constraint(max_atoms: int):
    """最大原子数约束"""
    def constraint(structure):
        n_atoms = len(getattr(structure, 'positions', []))
        if n_atoms > max_atoms:
            return False, f"Atom count {n_atoms} exceeds maximum {max_atoms}"
        return True, ""
    return constraint


def min_atoms_constraint(min_atoms: int):
    """最小原子数约束"""
    def constraint(structure):
        n_atoms = len(getattr(structure, 'positions', []))
        if n_atoms < min_atoms:
            return False, f"Atom count {n_atoms} below minimum {min_atoms}"
        return True, ""
    return constraint


def charge_neutrality_constraint():
    """电中性约束"""
    def constraint(structure):
        total_charge = sum(getattr(structure, 'charges', []))
        if abs(total_charge) > 0.1:
            return False, f"Non-neutral structure: charge = {total_charge}"
        return True, ""
    return constraint


def stoichiometry_constraint(allowed_ratios: Dict[str, Tuple[float, float]]):
    """化学计量比约束"""
    def constraint(structure):
        composition = getattr(structure, 'composition', {})
        for element, (min_ratio, max_ratio) in allowed_ratios.items():
            ratio = composition.get(element, 0)
            total = sum(composition.values())
            if total > 0:
                actual_ratio = ratio / total
                if not (min_ratio <= actual_ratio <= max_ratio):
                    return False, f"Element {element} ratio {actual_ratio} outside [{min_ratio}, {max_ratio}]"
        return True, ""
    return constraint
