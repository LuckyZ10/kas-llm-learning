#!/usr/bin/env python3
"""
催化剂优化场景

优化目标:
- 高催化活性
- 高选择性
- 稳定性
- 原子利用效率
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ..environment import CrystalStructureEnv, CrystalState, EnvConfig
from ..algorithms import SACAgent, SACConfig
from ..rewards import create_catalyst_reward

logger = logging.getLogger(__name__)


@dataclass
class CatalystConfig:
    """催化剂优化配置"""
    # 反应类型
    reaction_type: str = 'ORR'  # ORR, HER, OER, CO2RR, NRR
    
    # 目标性质
    target_activity: float = 1.0  # 标准化活性
    target_selectivity: float = 0.9  # 90%选择性
    
    # 材料约束
    max_precious_metal_content: float = 0.1  # 最大贵金属含量
    allowed_metals: List[str] = None
    
    # 优化参数
    n_episodes: int = 1000
    max_steps_per_episode: int = 50
    
    def __post_init__(self):
        if self.allowed_metals is None:
            self.allowed_metals = [
                'Pt', 'Pd', 'Au', 'Ag',  # 贵金属
                'Fe', 'Co', 'Ni', 'Cu', 'Mn',  # 过渡金属
                'Mo', 'W', 'V', 'Cr'  # 其他过渡金属
            ]


class CatalystOptimizer:
    """
    催化剂优化器
    
    使用RL优化催化剂的结构、组成和活性位点。
    """
    
    def __init__(
        self,
        config: Optional[CatalystConfig] = None,
        env_config: Optional[EnvConfig] = None
    ):
        self.config = config or CatalystConfig()
        self.env_config = env_config or EnvConfig()
        
        # 设置元素集
        self.env_config.element_set = (
            self.config.allowed_metals +
            ['C', 'N', 'O', 'S', 'P', 'B']  # 掺杂元素
        )
        
        # 创建奖励函数
        self.reward_composer = create_catalyst_reward(
            activity_weight=1.0,
            selectivity_weight=1.0,
            stability_weight=0.5
        )
        
        # 创建环境
        self.env = CrystalStructureEnv(
            config=self.env_config,
            reward_calculator=self._compute_reward
        )
        
        # 创建智能体
        agent_config = SACConfig(
            state_dim=self.env.state_rep.get_feature_dim(),
            action_dim=self.env_config.action_dim
        )
        self.agent = SACAgent(agent_config)
    
    def _compute_reward(self, structure: CrystalState, info: Dict) -> float:
        """计算催化剂奖励"""
        reward, details = self.reward_composer.compute(structure, info)
        
        # 额外惩罚：高贵金属含量
        composition = structure.get_composition()
        precious_metals = ['Pt', 'Pd', 'Au', 'Ag', 'Ru', 'Rh', 'Ir', 'Os']
        precious_content = sum(composition.get(m, 0) for m in precious_metals)
        total = sum(composition.values())
        
        if total > 0:
            precious_fraction = precious_content / total
            if precious_fraction > self.config.max_precious_metal_content:
                reward -= (precious_fraction - self.config.max_precious_metal_content) * 2
        
        # 奖励：表面原子比例 (催化剂需要高表面积)
        # 简化：小团簇得分更高
        n_atoms = len(structure.elements)
        if n_atoms > 0 and n_atoms < 100:
            reward += 0.1 * (100 - n_atoms) / 100
        
        return reward
    
    def train(self, n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """训练优化器"""
        n_episodes = n_episodes or self.config.n_episodes
        
        logger.info(f"Starting catalyst optimization training for {n_episodes} episodes")
        logger.info(f"Target reaction: {self.config.reaction_type}")
        
        best_structures = []
        
        for episode in range(n_episodes):
            result = self.agent.train_episode(
                self.env,
                max_steps=self.config.max_steps_per_episode
            )
            
            history = self.env.get_history()
            if history:
                best_step = max(history, key=lambda x: x['reward'])
                best_structures.append({
                    'episode': episode,
                    'structure': best_step['structure'],
                    'reward': best_step['reward']
                })
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: reward={result.get('episode_reward', 0):.4f}")
        
        best_structures.sort(key=lambda x: x['reward'], reverse=True)
        
        return {
            'best_structures': best_structures[:10],
            'reaction_type': self.config.reaction_type
        }
    
    def evaluate_active_sites(self, structure: CrystalState) -> List[Dict]:
        """识别和评估活性位点"""
        sites = []
        
        # 简化：识别低配位原子作为潜在活性位点
        positions = structure.positions @ structure.lattice
        elements = structure.elements
        
        for i, (pos, elem) in enumerate(zip(positions, elements)):
            # 计算配位数 (简化：计算距离范围内的邻居数)
            distances = np.linalg.norm(positions - pos, axis=1)
            distances[i] = np.inf  # 排除自身
            coordination = np.sum(distances < 3.0)  # 3Å截止
            
            if coordination <= 6:  # 低配位
                sites.append({
                    'index': i,
                    'element': elem,
                    'coordination': coordination,
                    'position': pos,
                    'estimated_activity': 1.0 / (coordination + 1)
                })
        
        # 按估计活性排序
        sites.sort(key=lambda x: x['estimated_activity'], reverse=True)
        
        return sites
