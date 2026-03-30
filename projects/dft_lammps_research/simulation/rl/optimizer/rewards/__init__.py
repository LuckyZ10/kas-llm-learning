#!/usr/bin/env python3
"""
奖励函数模块 - 材料优化的奖励函数设计

包含:
- 能量奖励
- 稳定性奖励
- 性质奖励 (离子电导率、催化活性等)
- 多目标奖励组合
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """奖励函数配置"""
    scale: float = 1.0
    clip_range: Optional[Tuple[float, float]] = None
    normalize: bool = False
    bonus_for_improvement: float = 0.0


class RewardFunction(ABC):
    """奖励函数基类"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.history = []
        self.best_value = float('-inf')
    
    @abstractmethod
    def compute(
        self,
        structure: Any,
        info: Optional[Dict] = None
    ) -> float:
        """
        计算奖励
        
        Args:
            structure: 材料结构
            info: 额外信息
            
        Returns:
            奖励值
        """
        pass
    
    def __call__(self, structure: Any, info: Optional[Dict] = None) -> float:
        """调用接口"""
        reward = self.compute(structure, info)
        
        # 应用缩放
        reward *= self.config.scale
        
        # 裁剪
        if self.config.clip_range is not None:
            reward = np.clip(reward, self.config.clip_range[0], self.config.clip_range[1])
        
        # 记录历史
        self.history.append(reward)
        
        # 改进奖励
        if reward > self.best_value:
            reward += self.config.bonus_for_improvement
            self.best_value = reward
        
        return reward
    
    def get_stats(self) -> Dict[str, float]:
        """获取奖励统计"""
        if not self.history:
            return {}
        
        return {
            'mean': np.mean(self.history),
            'std': np.std(self.history),
            'min': np.min(self.history),
            'max': np.max(self.history),
            'best': self.best_value
        }


class EnergyReward(RewardFunction):
    """
    能量奖励
    
    基于DFT计算的能量奖励，用于优化稳定结构。
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        dft_calculator: Optional[Callable] = None,
        reference_energy: float = 0.0,
        per_atom: bool = True
    ):
        super().__init__(config)
        self.dft_calculator = dft_calculator
        self.reference_energy = reference_energy
        self.per_atom = per_atom
    
    def compute(self, structure: Any, info: Optional[Dict] = None) -> float:
        """计算能量奖励"""
        try:
            # 获取能量
            if self.dft_calculator is not None:
                energy = self.dft_calculator(structure)
            elif hasattr(structure, 'energy'):
                energy = structure.energy
            else:
                # 简化估算
                energy = self._estimate_energy(structure)
            
            # 归一化 (每原子能量)
            if self.per_atom and hasattr(structure, 'positions'):
                n_atoms = len(structure.positions)
                if n_atoms > 0:
                    energy /= n_atoms
            
            # 奖励 = - (能量 - 参考能量)
            # 能量越低，奖励越高
            reward = -(energy - self.reference_energy)
            
            return reward
            
        except Exception as e:
            logger.warning(f"Energy calculation failed: {e}")
            return -1.0
    
    def _estimate_energy(self, structure: Any) -> float:
        """估算能量 (简化版)"""
        # 基于原子间距的简化估算
        if hasattr(structure, 'positions') and len(structure.positions) >= 2:
            from scipy.spatial.distance import pdist
            positions_cart = structure.positions @ structure.lattice
            distances = pdist(positions_cart)
            
            # Lennard-Jones-like势 (简化)
            sigma = 2.5  # Å
            epsilon = 0.1  # eV
            
            energy = 0
            for r in distances:
                if r > 0:
                    sr6 = (sigma / r) ** 6
                    energy += 4 * epsilon * (sr6 ** 2 - sr6)
            
            return energy
        
        return 0.0


class StabilityReward(RewardFunction):
    """
    稳定性奖励
    
    基于凸包分析的稳定性奖励。
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        phase_diagram: Optional[Any] = None,
        hull_distance_threshold: float = 0.1  # eV/atom
    ):
        super().__init__(config)
        self.phase_diagram = phase_diagram
        self.hull_distance_threshold = hull_distance_threshold
    
    def compute(self, structure: Any, info: Optional[Dict] = None) -> float:
        """计算稳定性奖励"""
        try:
            # 获取形成能
            if self.phase_diagram is not None:
                formation_energy = self._get_formation_energy(structure)
                hull_distance = self.phase_diagram.get_hull_distance(structure.composition)
            elif hasattr(structure, 'formation_energy'):
                formation_energy = structure.formation_energy
                hull_distance = formation_energy  # 简化
            else:
                # 简化估算
                hull_distance = 0.0
            
            # 奖励: 越接近凸包 (越稳定)，奖励越高
            if hull_distance <= 0:
                # 在凸包上 (稳定相)
                reward = 1.0
            elif hull_distance < self.hull_distance_threshold:
                # 接近凸包
                reward = 1.0 - (hull_distance / self.hull_distance_threshold)
            else:
                # 远离凸包 (不稳定)
                reward = -hull_distance
            
            return reward
            
        except Exception as e:
            logger.warning(f"Stability calculation failed: {e}")
            return -1.0
    
    def _get_formation_energy(self, structure: Any) -> float:
        """获取形成能"""
        # 简化实现
        return getattr(structure, 'formation_energy', 0.0)


class PropertyReward(RewardFunction):
    """
    性质奖励
    
    用于优化特定材料性质。
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        property_name: str = 'band_gap',
        target_value: Optional[float] = None,
        target_range: Optional[Tuple[float, float]] = None,
        property_calculator: Optional[Callable] = None,
        maximize: bool = True
    ):
        super().__init__(config)
        self.property_name = property_name
        self.target_value = target_value
        self.target_range = target_range
        self.property_calculator = property_calculator
        self.maximize = maximize
    
    def compute(self, structure: Any, info: Optional[Dict] = None) -> float:
        """计算性质奖励"""
        try:
            # 获取性质值
            if self.property_calculator is not None:
                value = self.property_calculator(structure)
            elif hasattr(structure, self.property_name):
                value = getattr(structure, self.property_name)
            elif info and self.property_name in info:
                value = info[self.property_name]
            else:
                # 尝试从ML势预测
                value = self._predict_property(structure)
            
            # 计算奖励
            if self.target_value is not None:
                # 目标值优化
                distance = abs(value - self.target_value)
                reward = -distance
            elif self.target_range is not None:
                # 目标范围优化
                min_val, max_val = self.target_range
                if min_val <= value <= max_val:
                    # 在目标范围内
                    reward = 1.0
                else:
                    # 在范围外
                    distance = min(abs(value - min_val), abs(value - max_val))
                    reward = -distance
            elif self.maximize:
                # 最大化
                reward = value
            else:
                # 最小化
                reward = -value
            
            return reward
            
        except Exception as e:
            logger.warning(f"Property calculation failed: {e}")
            return 0.0
    
    def _predict_property(self, structure: Any) -> float:
        """使用ML势预测性质"""
        # 简化实现 - 实际应调用ML势模型
        return 0.0


class MultiObjectiveReward(RewardFunction):
    """
    多目标奖励
    
    组合多个奖励函数。
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        rewards: Optional[List[Tuple[RewardFunction, float]]] = None,
        scalarization: str = 'linear'  # 'linear', 'chebyshev', 'product'
    ):
        super().__init__(config)
        self.rewards = rewards or []
        self.scalarization = scalarization
    
    def add_reward(self, reward: RewardFunction, weight: float = 1.0):
        """添加奖励函数"""
        self.rewards.append((reward, weight))
    
    def compute(self, structure: Any, info: Optional[Dict] = None) -> float:
        """计算多目标奖励"""
        if not self.rewards:
            return 0.0
        
        values = []
        weights = []
        
        for reward_fn, weight in self.rewards:
            value = reward_fn.compute(structure, info)
            values.append(value)
            weights.append(weight)
        
        values = np.array(values)
        weights = np.array(weights)
        
        # 标量化
        if self.scalarization == 'linear':
            # 线性加权
            reward = np.sum(values * weights)
        elif self.scalarization == 'chebyshev':
            # Tchebycheff标量化
            reward = -np.max(weights * np.abs(values))
        elif self.scalarization == 'product':
            # 乘积标量化
            # 确保值为正
            values_pos = values - np.min(values) + 1e-6
            reward = np.prod(values_pos ** weights)
        else:
            reward = np.sum(values * weights)
        
        return reward
    
    def get_individual_rewards(self, structure: Any, info: Optional[Dict] = None) -> Dict[str, float]:
        """获取各个奖励的值"""
        result = {}
        for i, (reward_fn, weight) in enumerate(self.rewards):
            value = reward_fn.compute(structure, info)
            result[f'reward_{i}'] = value
        return result


class RewardComposer:
    """
    奖励组合器
    
    用于动态组合和调整奖励函数。
    """
    
    def __init__(self):
        self.reward_functions: Dict[str, RewardFunction] = {}
        self.weights: Dict[str, float] = {}
        self.active_rewards: List[str] = []
    
    def register(
        self,
        name: str,
        reward_fn: RewardFunction,
        weight: float = 1.0,
        active: bool = True
    ):
        """注册奖励函数"""
        self.reward_functions[name] = reward_fn
        self.weights[name] = weight
        
        if active:
            self.active_rewards.append(name)
    
    def set_weight(self, name: str, weight: float):
        """设置权重"""
        if name in self.weights:
            self.weights[name] = weight
    
    def activate(self, name: str):
        """激活奖励函数"""
        if name in self.reward_functions and name not in self.active_rewards:
            self.active_rewards.append(name)
    
    def deactivate(self, name: str):
        """停用奖励函数"""
        if name in self.active_rewards:
            self.active_rewards.remove(name)
    
    def compute(self, structure: Any, info: Optional[Dict] = None) -> Tuple[float, Dict[str, float]]:
        """
        计算组合奖励
        
        Returns:
            (总奖励, 各奖励明细)
        """
        total_reward = 0.0
        details = {}
        
        for name in self.active_rewards:
            reward_fn = self.reward_functions[name]
            weight = self.weights[name]
            
            value = reward_fn(structure, info)
            weighted_value = value * weight
            
            total_reward += weighted_value
            details[name] = {
                'raw': value,
                'weighted': weighted_value,
                'weight': weight
            }
        
        return total_reward, details
    
    def get_stats(self) -> Dict[str, Dict]:
        """获取所有奖励函数的统计"""
        return {
            name: fn.get_stats()
            for name, fn in self.reward_functions.items()
        }


# 预定义奖励函数

def create_battery_reward(
    ionic_conductivity_weight: float = 1.0,
    voltage_weight: float = 1.0,
    stability_weight: float = 0.5,
    dft_calculator: Optional[Callable] = None
) -> RewardComposer:
    """
    创建电池材料优化奖励
    
    目标:
    - 高离子电导率
    - 合适的工作电压
    - 结构稳定性
    """
    composer = RewardComposer()
    
    # 离子电导率奖励 (最大化)
    composer.register(
        'ionic_conductivity',
        PropertyReward(
            property_name='ionic_conductivity',
            maximize=True
        ),
        weight=ionic_conductivity_weight,
        active=True
    )
    
    # 电压窗口奖励 (目标范围)
    composer.register(
        'voltage',
        PropertyReward(
            property_name='voltage',
            target_range=(2.0, 5.0)  # V
        ),
        weight=voltage_weight,
        active=True
    )
    
    # 稳定性奖励
    composer.register(
        'stability',
        StabilityReward(),
        weight=stability_weight,
        active=True
    )
    
    return composer


def create_catalyst_reward(
    activity_weight: float = 1.0,
    selectivity_weight: float = 1.0,
    stability_weight: float = 0.5
) -> RewardComposer:
    """
    创建催化剂优化奖励
    
    目标:
    - 高催化活性
    - 高选择性
    - 结构稳定性
    """
    composer = RewardComposer()
    
    # 活性奖励 (最大化)
    composer.register(
        'activity',
        PropertyReward(property_name='activity', maximize=True),
        weight=activity_weight
    )
    
    # 选择性奖励 (最大化)
    composer.register(
        'selectivity',
        PropertyReward(property_name='selectivity', maximize=True),
        weight=selectivity_weight
    )
    
    # 稳定性奖励
    composer.register(
        'stability',
        StabilityReward(),
        weight=stability_weight
    )
    
    return composer


def create_alloy_reward(
    strength_weight: float = 1.0,
    ductility_weight: float = 1.0,
    density_weight: float = 0.3
) -> RewardComposer:
    """
    创建合金优化奖励
    
    目标:
    - 高强度
    - 高延展性
    - 低密度 (轻量化)
    """
    composer = RewardComposer()
    
    # 强度奖励 (最大化)
    composer.register(
        'strength',
        PropertyReward(property_name='strength', maximize=True),
        weight=strength_weight
    )
    
    # 延展性奖励 (最大化)
    composer.register(
        'ductility',
        PropertyReward(property_name='ductility', maximize=True),
        weight=ductility_weight
    )
    
    # 密度奖励 (最小化)
    composer.register(
        'density',
        PropertyReward(property_name='density', maximize=False),
        weight=density_weight
    )
    
    return composer


def create_topological_reward(
    band_inversion_weight: float = 1.0,
    gap_weight: float = 1.0,
    stability_weight: float = 0.5
) -> RewardComposer:
    """
    创建拓扑材料优化奖励
    
    目标:
    - 能带反转 (拓扑指标)
    - 非平庸带隙
    - 稳定性
    """
    composer = RewardComposer()
    
    # 拓扑指标奖励
    composer.register(
        'topological_index',
        PropertyReward(property_name='z2_index', maximize=True),
        weight=band_inversion_weight
    )
    
    # 带隙奖励 (目标范围)
    composer.register(
        'band_gap',
        PropertyReward(property_name='band_gap', target_range=(0.1, 1.0)),
        weight=gap_weight
    )
    
    # 稳定性奖励
    composer.register(
        'stability',
        StabilityReward(),
        weight=stability_weight
    )
    
    return composer
