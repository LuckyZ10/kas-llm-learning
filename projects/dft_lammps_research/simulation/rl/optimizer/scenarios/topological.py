#!/usr/bin/env python3
"""
拓扑材料发现场景

优化目标:
- 发现具有非平庸拓扑性质的材料
- 具有可观测的带隙
- 稳定性
- 易于实验合成
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ..environment import CrystalStructureEnv, CrystalState, EnvConfig
from ..algorithms import PPOAgent, PPOConfig
from ..rewards import create_topological_reward

logger = logging.getLogger(__name__)


@dataclass
class TopologicalConfig:
    """拓扑材料优化配置"""
    # 目标拓扑性质
    target_invariant: str = 'Z2'  # Z2, Chern, Z
    target_gap_range: Tuple[float, float] = (0.1, 1.0)  # eV
    
    # 材料类型偏好
    prefer_2d: bool = False
    prefer_heavy_elements: bool = True  # 重元素通常有更强的自旋轨道耦合
    
    # 元素约束
    required_elements: List[str] = None  # 必须包含的元素 (如Bi, Sb等)
    
    # 优化参数
    n_episodes: int = 1000
    max_steps_per_episode: int = 50
    
    def __post_init__(self):
        if self.required_elements is None:
            if self.target_invariant == 'Z2':
                self.required_elements = ['Bi', 'Sb']  # 常见Z2拓扑绝缘体元素
            elif self.target_invariant == 'Chern':
                self.required_elements = ['Cr', 'Mn']  # 磁性元素


class TopologicalOptimizer:
    """
    拓扑材料发现器
    
    使用RL探索可能的拓扑材料组成和结构，
    优化拓扑不变量和带隙。
    """
    
    def __init__(
        self,
        config: Optional[TopologicalConfig] = None,
        env_config: Optional[EnvConfig] = None
    ):
        self.config = config or TopologicalConfig()
        self.env_config = env_config or EnvConfig()
        
        # 设置元素集 (重元素为主)
        heavy_elements = [
            'Bi', 'Sb', 'Pb', 'Sn', 'Tl',  # 主族重元素
            'W', 'Mo', 'Ta', 'Re', 'Os',  # 过渡金属
            'Pt', 'Au', 'Hg',  # 贵金属
            'Te', 'Se', 'S'  # 硫族元素
        ]
        
        if self.config.required_elements:
            heavy_elements = list(set(heavy_elements + self.config.required_elements))
        
        self.env_config.element_set = heavy_elements
        
        # 创建奖励函数
        self.reward_composer = create_topological_reward()
        
        # 创建环境
        self.env = CrystalStructureEnv(
            config=self.env_config,
            reward_calculator=self._compute_reward
        )
        
        # 创建智能体
        agent_config = PPOConfig(
            state_dim=self.env.state_rep.get_feature_dim(),
            action_dim=self.env_config.action_dim
        )
        self.agent = PPOAgent(agent_config)
        
        # 发现的拓扑材料
        self.discovered_materials = []
    
    def _compute_reward(self, structure: CrystalState, info: Dict) -> float:
        """计算拓扑材料奖励"""
        reward, details = self.reward_composer.compute(structure, info)
        
        composition = structure.get_composition()
        
        # 额外奖励：包含必需元素
        for elem in self.config.required_elements:
            if elem in composition:
                reward += 0.2
        
        # 额外奖励：重元素比例高 (强SOC)
        heavy_elements = ['Bi', 'Sb', 'Pb', 'Tl', 'W', 'Re', 'Os', 'Pt', 'Au', 'Hg']
        heavy_content = sum(composition.get(elem, 0) for elem in heavy_elements)
        total = sum(composition.values())
        
        if total > 0 and self.config.prefer_heavy_elements:
            heavy_fraction = heavy_content / total
            reward += heavy_fraction * 0.3
        
        # 惩罚：过大的带隙 (可能无法观察到拓扑表面态)
        # 这部分在奖励函数中处理
        
        return reward
    
    def train(self, n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """训练优化器"""
        n_episodes = n_episodes or self.config.n_episodes
        
        logger.info(f"Starting topological material discovery for {n_episodes} episodes")
        logger.info(f"Target invariant: {self.config.target_invariant}")
        
        for episode in range(n_episodes):
            result = self.agent.train_episode(
                self.env,
                max_steps=self.config.max_steps_per_episode
            )
            
            # 记录发现的材料
            history = self.env.get_history()
            if history:
                best_step = max(history, key=lambda x: x['reward'])
                
                if best_step['reward'] > 0.5:  # 阈值
                    material = {
                        'episode': episode,
                        'structure': best_step['structure'],
                        'reward': best_step['reward'],
                        'formula': best_step['structure'].get_composition()
                    }
                    self.discovered_materials.append(material)
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: reward={result.get('episode_reward', 0):.4f}, "
                          f"discovered={len(self.discovered_materials)}")
        
        return {
            'discovered_materials': self.discovered_materials,
            'n_discovered': len(self.discovered_materials)
        }
    
    def validate_topology(self, structure: CrystalState) -> Dict[str, Any]:
        """
        验证材料的拓扑性质
        
        这需要调用DFT计算，计算Z2不变量或Chern数。
        简化版本使用启发式规则。
        """
        result = {
            'is_topological': False,
            'invariant_type': self.config.target_invariant,
            'invariant_value': None,
            'confidence': 0.0
        }
        
        composition = structure.get_composition()
        
        # 启发式检查
        # 1. 检查必需元素
        has_required = any(elem in composition for elem in self.config.required_elements)
        
        # 2. 检查空间群 (简化)
        # 实际应使用spglib等库分析空间群
        
        # 3. 估算带隙 (基于元素电负性差异)
        en_diff = self._estimate_band_gap(structure)
        
        if has_required and 0.1 < en_diff < 1.0:
            result['is_topological'] = True
            result['confidence'] = 0.6
            result['estimated_gap'] = en_diff
        
        return result
    
    def _estimate_band_gap(self, structure: CrystalState) -> float:
        """估算带隙"""
        composition = structure.get_composition()
        
        # 简化估算：基于元素电负性差异
        electronegativity = {
            'Bi': 2.02, 'Sb': 2.05, 'Pb': 2.33, 'Sn': 1.96, 'Tl': 1.62,
            'W': 2.36, 'Mo': 2.16, 'Ta': 1.5, 'Re': 1.9, 'Os': 2.2,
            'Pt': 2.28, 'Au': 2.54, 'Hg': 2.0,
            'Te': 2.1, 'Se': 2.55, 'S': 2.58
        }
        
        ens = [electronegativity.get(elem, 2.0) for elem in composition.keys()]
        if len(ens) >= 2:
            en_diff = max(ens) - min(ens)
            return en_diff
        
        return 0.5
    
    def generate_dft_input(self, structure: CrystalState, filename: str):
        """生成DFT计算输入文件"""
        # 生成VASP POSCAR格式
        composition = structure.get_composition()
        formula = ''.join(f"{elem}{int(count) if count == int(count) else count}"
                         for elem, count in sorted(composition.items()))
        
        lines = [formula, '1.0']
        
        # 晶格
        for row in structure.lattice:
            lines.append(f"{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}")
        
        # 元素和数量
        elements = list(composition.keys())
        counts = [str(int(composition[elem])) for elem in elements]
        lines.append(' '.join(elements))
        lines.append(' '.join(counts))
        
        # 分数坐标
        lines.append('Direct')
        for pos in structure.positions:
            lines.append(f"{pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")
        
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Generated DFT input: {filename}")
