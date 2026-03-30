#!/usr/bin/env python3
"""
电池材料优化场景

优化目标:
- 高离子电导率
- 合适的工作电压窗口
- 良好的结构稳定性
- 低成本元素组成
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging

from ..environment import CrystalStructureEnv, CrystalState, EnvConfig, CrystalAction
from ..algorithms import PPOAgent, SACAgent, PPOConfig, SACConfig
from ..rewards import create_battery_reward, RewardComposer

logger = logging.getLogger(__name__)


@dataclass
class BatteryConfig:
    """电池优化配置"""
    # 目标性质
    target_ionic_conductivity: float = 1e-3  # S/cm
    target_voltage_min: float = 2.0  # V
    target_voltage_max: float = 5.0  # V
    
    # 约束
    allowed_cations: List[str] = None
    allowed_anions: List[str] = None
    max_cost_per_kwh: float = 100.0  # USD/kWh
    
    # 优化参数
    n_episodes: int = 1000
    max_steps_per_episode: int = 50
    
    def __post_init__(self):
        if self.allowed_cations is None:
            self.allowed_cations = ['Li', 'Na', 'K', 'Mg', 'Ca']
        if self.allowed_anions is None:
            self.allowed_anions = ['O', 'S', 'Se', 'F', 'Cl']


class BatteryOptimizer:
    """
    电池材料优化器
    
    使用RL优化固态电解质和电极材料的组成和结构。
    """
    
    def __init__(
        self,
        config: Optional[BatteryConfig] = None,
        env_config: Optional[EnvConfig] = None,
        agent_type: str = 'sac'
    ):
        self.config = config or BatteryConfig()
        self.env_config = env_config or EnvConfig()
        
        # 设置元素集
        self.env_config.element_set = (
            self.config.allowed_cations +
            self.config.allowed_anions +
            ['P', 'Si', 'Ge', 'Al', 'B', 'C']  # 额外的框架元素
        )
        
        # 创建奖励函数
        self.reward_composer = create_battery_reward(
            ionic_conductivity_weight=1.0,
            voltage_weight=1.0,
            stability_weight=0.5
        )
        
        # 创建环境
        self.env = CrystalStructureEnv(
            config=self.env_config,
            reward_calculator=self._compute_reward
        )
        
        # 创建智能体
        if agent_type == 'sac':
            agent_config = SACConfig(
                state_dim=self.env.state_rep.get_feature_dim(),
                action_dim=self.env_config.action_dim
            )
            self.agent = SACAgent(agent_config)
        elif agent_type == 'ppo':
            agent_config = PPOConfig(
                state_dim=self.env.state_rep.get_feature_dim(),
                action_dim=self.env_config.action_dim
            )
            self.agent = PPOAgent(agent_config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def _compute_reward(self, structure: CrystalState, info: Dict) -> float:
        """计算电池材料奖励"""
        reward, details = self.reward_composer.compute(structure, info)
        
        # 额外惩罚：非目标元素
        composition = structure.get_composition()
        for elem in composition:
            if elem not in self.env_config.element_set:
                reward -= 0.5
        
        return reward
    
    def train(self, n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """训练优化器"""
        n_episodes = n_episodes or self.config.n_episodes
        
        logger.info(f"Starting battery optimization training for {n_episodes} episodes")
        
        best_structures = []
        
        for episode in range(n_episodes):
            # 训练episode
            result = self.agent.train_episode(
                self.env,
                max_steps=self.config.max_steps_per_episode
            )
            
            # 记录最佳结构
            history = self.env.get_history()
            if history:
                best_step = max(history, key=lambda x: x['reward'])
                best_structures.append({
                    'episode': episode,
                    'structure': best_step['structure'],
                    'reward': best_step['reward'],
                    'formula': best_step['structure'].get_composition()
                })
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: reward={result.get('episode_reward', 0):.4f}, "
                          f"mean={result.get('mean_reward', 0):.4f}")
        
        # 排序并返回最佳结构
        best_structures.sort(key=lambda x: x['reward'], reverse=True)
        
        return {
            'best_structures': best_structures[:10],
            'final_stats': self.reward_composer.get_stats()
        }
    
    def optimize(self, initial_structure: Optional[CrystalState] = None) -> CrystalState:
        """执行单次优化"""
        state = self.env.reset(initial_structure)
        
        for step in range(self.config.max_steps_per_episode):
            action = self.agent.select_action(state, deterministic=True)
            result = self.env.step(action)
            
            if result.done:
                break
            
            state = result.state
        
        return self.env.get_structure()
    
    def evaluate(self, structure: CrystalState) -> Dict[str, float]:
        """评估结构性能"""
        # 使用DFT或ML势评估
        properties = {}
        
        # 离子电导率 (简化估算)
        properties['ionic_conductivity'] = self._estimate_ionic_conductivity(structure)
        
        # 电压窗口
        properties['voltage'] = self._estimate_voltage(structure)
        
        # 稳定性
        properties['stability'] = self._estimate_stability(structure)
        
        # 成本估算
        properties['cost'] = self._estimate_cost(structure)
        
        return properties
    
    def _estimate_ionic_conductivity(self, structure: CrystalState) -> float:
        """估算离子电导率 (简化版)"""
        # 基于通道尺寸和离子浓度的简化估算
        composition = structure.get_composition()
        
        # 检查是否有移动离子
        mobile_ions = sum(composition.get(elem, 0) for elem in self.config.allowed_cations)
        
        if mobile_ions == 0:
            return 1e-10
        
        # 估算扩散路径
        volume = structure.get_volume()
        n_atoms = len(structure.elements)
        
        if n_atoms > 0 and volume > 0:
            # 简化: 体积越大，通道越多
            porosity = 1.0 - (n_atoms * 10) / volume  # 粗略估算
            conductivity = mobile_ions * max(0, porosity) * 1e-4
        else:
            conductivity = 1e-10
        
        return max(1e-10, conductivity)
    
    def _estimate_voltage(self, structure: CrystalState) -> float:
        """估算电压"""
        # 简化估算：基于元素电负性差异
        composition = structure.get_composition()
        
        electronegativity = {
            'Li': 0.98, 'Na': 0.93, 'K': 0.82, 'Mg': 1.31, 'Ca': 1.00,
            'O': 3.44, 'S': 2.58, 'Se': 2.55, 'F': 3.98, 'Cl': 3.16,
            'P': 2.19, 'Si': 1.90, 'Ge': 2.01, 'Al': 1.61, 'B': 2.04
        }
        
        voltage = 0.0
        total = sum(composition.values())
        
        for elem, count in composition.items():
            if elem in electronegativity:
                voltage += electronegativity[elem] * count / total
        
        # 归一化到合理范围
        return voltage * 2
    
    def _estimate_stability(self, structure: CrystalState) -> float:
        """估算稳定性"""
        # 简化：检查配位合理性
        return 0.5  # 占位符
    
    def _estimate_cost(self, structure: CrystalState) -> float:
        """估算成本 (USD/kWh)"""
        # 简化成本估算
        composition = structure.get_composition()
        
        element_cost = {
            'Li': 20.0, 'Na': 3.0, 'K': 13.0, 'Mg': 2.3, 'Ca': 2.8,
            'O': 0.0, 'S': 0.1, 'Se': 30.0, 'F': 2.0, 'Cl': 0.2,
            'P': 3.0, 'Si': 1.7, 'Ge': 1000.0, 'Al': 1.8, 'B': 3.0
        }
        
        total_cost = 0.0
        total_mass = 0.0
        
        atomic_mass = {
            'Li': 6.94, 'Na': 22.99, 'K': 39.10, 'Mg': 24.31, 'Ca': 40.08,
            'O': 16.00, 'S': 32.07, 'Se': 78.96, 'F': 19.00, 'Cl': 35.45,
            'P': 30.97, 'Si': 28.09, 'Ge': 72.63, 'Al': 26.98, 'B': 10.81
        }
        
        for elem, count in composition.items():
            mass = atomic_mass.get(elem, 50.0)
            cost = element_cost.get(elem, 10.0)
            total_cost += cost * count
            total_mass += mass * count
        
        if total_mass > 0:
            return total_cost / total_mass * 1000  # 转换为 USD/kWh 近似
        return 1000.0
