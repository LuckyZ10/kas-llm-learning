"""
Material Design Environments
============================

实现材料设计环境:
- 材料组成环境
- 晶体结构环境
- 成分-结构联合优化环境
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MaterialEnvConfig:
    """材料环境配置"""
    element_set: List[str] = None
    max_elements: int = 5
    max_atoms: int = 50
    min_lattice: float = 2.0
    max_lattice: float = 15.0
    num_lattice_bins: int = 20
    position_resolution: float = 0.1
    
    def __post_init__(self):
        if self.element_set is None:
            # 常见材料元素
            self.element_set = [
                'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg',
                'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V',
                'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
                'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs',
                'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Hf', 'Ta', 'W', 'Re', 'Os',
                'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'
            ]


class MaterialDesignEnv(ABC):
    """材料设计环境基类"""
    
    def __init__(self, config: Optional[MaterialEnvConfig] = None):
        self.config = config or MaterialEnvConfig()
        self.state = None
        self.step_count = 0
        self.composition = {}  # 元素 -> 数量
        self.structure = None  # 晶体结构
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """重置环境"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        pass
    
    @abstractmethod
    def get_valid_actions(self) -> List[int]:
        """获取有效动作"""
        pass
    
    @abstractmethod
    def compute_reward(self) -> float:
        """计算奖励"""
        pass
    
    def get_sample(self) -> Dict[str, Any]:
        """获取当前样本"""
        return {
            'composition': self.composition.copy(),
            'structure': self.structure,
            'step_count': self.step_count,
        }


class CompositionEnv(MaterialDesignEnv):
    """
    材料成分优化环境
    
    通过逐步添加/调整元素组成来优化材料
    """
    
    def __init__(self, config: Optional[MaterialEnvConfig] = None):
        super().__init__(config)
        
        self.num_elements = len(self.config.element_set)
        self.max_atoms = self.config.max_atoms
        
        # 动作空间
        # 0 ~ num_elements-1: 添加元素
        # num_elements: 减少最后一个添加的元素
        # num_elements + 1: 终止
        self.action_dim = self.num_elements + 2
        self.terminate_action = self.num_elements + 1
        
        # 状态向量大小
        self.state_dim = self.num_elements + 10
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.composition = {elem: 0 for elem in self.config.element_set}
        self.step_count = 0
        self.structure = None
        return self._get_state_vector()
    
    def _get_state_vector(self) -> np.ndarray:
        """获取状态向量"""
        # 成分向量
        comp_vec = np.array([
            self.composition.get(elem, 0) / self.max_atoms
            for elem in self.config.element_set
        ])
        
        # 统计特征
        stats = np.array([
            sum(self.composition.values()) / self.max_atoms,
            len([c for c in self.composition.values() if c > 0]) / self.config.max_elements,
            self.step_count / (self.max_atoms * 2),
            self._get_charge_neutrality(),
            self._get_composition_entropy(),
        ])
        
        # 组合
        state = np.concatenate([comp_vec, stats])
        
        # 填充
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        
        return state[:self.state_dim]
    
    def _get_charge_neutrality(self) -> float:
        """计算电荷中性程度 (简化)"""
        # 简化: 假设常见氧化态
        oxidation_states = {
            'H': 1, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': -3, 'O': -2, 'F': -1,
            'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': -2, 'Cl': -1,
            'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 3, 'Mn': 2,
            'Fe': 2, 'Co': 2, 'Ni': 2, 'Cu': 2, 'Zn': 2, 'Ga': 3, 'Ge': 4,
        }
        
        total_charge = sum(
            count * oxidation_states.get(elem, 0)
            for elem, count in self.composition.items()
        )
        
        # 归一化
        total_atoms = sum(self.composition.values())
        if total_atoms == 0:
            return 1.0
        
        avg_charge = abs(total_charge) / total_atoms
        return np.exp(-avg_charge)
    
    def _get_composition_entropy(self) -> float:
        """计算成分熵"""
        total = sum(self.composition.values())
        if total == 0:
            return 0.0
        
        probs = [c / total for c in self.composition.values() if c > 0]
        if not probs:
            return 0.0
        
        entropy = -sum(p * np.log(p) for p in probs)
        return entropy / np.log(len(self.config.element_set))  # 归一化
    
    def get_valid_actions(self) -> List[int]:
        """获取有效动作"""
        valid = []
        
        total_atoms = sum(self.composition.values())
        
        # 可以添加元素
        if total_atoms < self.max_atoms:
            valid.extend(range(self.num_elements))
        
        # 如果有原子，可以移除
        if total_atoms > 0:
            valid.append(self.num_elements)
        
        # 可以终止
        if total_atoms >= 2:
            valid.append(self.terminate_action)
        
        return valid
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        self.step_count += 1
        done = False
        info = {}
        
        # 添加元素
        if action < self.num_elements:
            element = self.config.element_set[action]
            self.composition[element] = self.composition.get(element, 0) + 1
        
        # 移除元素
        elif action == self.num_elements:
            # 找到最后一个非零元素并减少
            for elem in reversed(list(self.composition.keys())):
                if self.composition[elem] > 0:
                    self.composition[elem] -= 1
                    break
        
        # 终止
        elif action == self.terminate_action:
            done = True
            info['composition'] = self.composition.copy()
        
        # 检查步数限制
        if self.step_count >= self.max_atoms * 2:
            done = True
        
        reward = self.compute_reward() if done else 0.0
        
        return self._get_state_vector(), reward, done, info
    
    def compute_reward(self) -> float:
        """计算成分奖励"""
        total_atoms = sum(self.composition.values())
        
        if total_atoms < 2:
            return 0.0
        
        reward = 0.0
        
        # 1. 大小奖励
        reward += min(total_atoms / 10, 1.0)
        
        # 2. 电荷中性奖励
        reward += self._get_charge_neutrality()
        
        # 3. 元素多样性奖励
        num_elements = len([c for c in self.composition.values() if c > 0])
        reward += min(num_elements / 3, 1.0)
        
        # 4. 成分熵奖励
        reward += self._get_composition_entropy()
        
        return reward
    
    def get_formula(self) -> str:
        """获取化学式"""
        parts = []
        for elem in self.config.element_set:
            count = self.composition.get(elem, 0)
            if count > 0:
                parts.append(f"{elem}{count}" if count > 1 else elem)
        return ''.join(parts) if parts else "Empty"


class StructureEnv(MaterialDesignEnv):
    """
    晶体结构优化环境
    
    在给定成分下优化晶体结构
    """
    
    def __init__(
        self,
        composition: Optional[Dict[str, int]] = None,
        config: Optional[MaterialEnvConfig] = None
    ):
        super().__init__(config)
        
        self.composition = composition or {'Si': 2}
        self.num_atoms = sum(self.composition.values())
        
        # 晶格参数 [a, b, c, alpha, beta, gamma]
        self.lattice = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        
        # 原子位置 [num_atoms, 3]
        self.positions = np.random.rand(self.num_atoms, 3)
        
        # 动作空间
        # 调整晶格参数(6个) + 调整原子位置(num_atoms * 3) + 终止
        self.action_dim = 6 + self.num_atoms * 3 + 1
        self.terminate_action = self.action_dim - 1
        
        # 状态维度
        self.state_dim = 6 + self.num_atoms * 3 + self.num_atoms * 10
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.lattice = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0])
        self.positions = np.random.rand(self.num_atoms, 3)
        self.step_count = 0
        return self._get_state_vector()
    
    def _get_state_vector(self) -> np.ndarray:
        """获取状态向量"""
        # 晶格参数 (归一化)
        lattice_norm = np.array([
            (self.lattice[0] - self.config.min_lattice) / 
            (self.config.max_lattice - self.config.min_lattice),
            (self.lattice[1] - self.config.min_lattice) / 
            (self.config.max_lattice - self.config.min_lattice),
            (self.lattice[2] - self.config.min_lattice) / 
            (self.config.max_lattice - self.config.min_lattice),
            self.lattice[3] / 180.0,
            self.lattice[4] / 180.0,
            self.lattice[5] / 180.0,
        ])
        
        # 原子位置
        positions_flat = self.positions.flatten()
        
        # 距离矩阵特征
        dist_features = self._get_distance_features()
        
        # 组合
        state = np.concatenate([lattice_norm, positions_flat, dist_features])
        
        # 填充
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        
        return state[:self.state_dim]
    
    def _get_distance_features(self) -> np.ndarray:
        """获取距离特征"""
        # 计算成对距离
        distances = []
        for i in range(self.num_atoms):
            for j in range(i + 1, self.num_atoms):
                # 考虑周期边界条件的简化距离
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                distances.append(dist)
        
        if not distances:
            return np.zeros(self.num_atoms * 10)
        
        # 统计特征
        features = [
            np.mean(distances),
            np.std(distances),
            np.min(distances),
            np.max(distances),
            np.median(distances),
        ]
        
        # 直方图
        hist, _ = np.histogram(distances, bins=10, range=(0, 1))
        features.extend(hist / len(distances))
        
        return np.array(features)
    
    def get_valid_actions(self) -> List[int]:
        """获取有效动作"""
        valid = list(range(self.action_dim))
        return valid
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        self.step_count += 1
        done = False
        info = {}
        
        delta = 0.05
        
        # 调整晶格参数
        if action < 3:
            # 调整 a, b, c
            self.lattice[action] += delta * (self.config.max_lattice - self.config.min_lattice)
            self.lattice[action] = np.clip(
                self.lattice[action],
                self.config.min_lattice,
                self.config.max_lattice
            )
        elif action < 6:
            # 调整 alpha, beta, gamma
            self.lattice[action] += delta * 180.0
            self.lattice[action] = np.clip(self.lattice[action], 60.0, 120.0)
        
        # 调整原子位置
        elif action < self.terminate_action:
            pos_idx = action - 6
            atom_idx = pos_idx // 3
            coord_idx = pos_idx % 3
            
            if atom_idx < self.num_atoms:
                self.positions[atom_idx, coord_idx] += delta
                self.positions[atom_idx, coord_idx] = np.clip(
                    self.positions[atom_idx, coord_idx], 0, 1
                )
        
        # 终止
        elif action == self.terminate_action:
            done = True
            info['structure'] = {
                'lattice': self.lattice.tolist(),
                'positions': self.positions.tolist(),
            }
        
        # 检查步数限制
        if self.step_count >= 100:
            done = True
        
        reward = self.compute_reward() if done else 0.0
        
        return self._get_state_vector(), reward, done, info
    
    def compute_reward(self) -> float:
        """计算结构奖励"""
        reward = 0.0
        
        # 1. 合理的晶格参数
        for i in range(3):
            if self.config.min_lattice <= self.lattice[i] <= self.config.max_lattice:
                reward += 0.1
        
        # 2. 原子间距检查 (避免过近)
        min_dist = float('inf')
        for i in range(self.num_atoms):
            for j in range(i + 1, self.num_atoms):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                min_dist = min(min_dist, dist)
        
        if min_dist > 0.1:  # 最小距离阈值
            reward += 1.0
        
        # 3. 位置分布奖励
        pos_std = np.std(self.positions)
        reward += min(pos_std * 2, 1.0)
        
        return reward


def demo():
    """演示材料环境"""
    print("=" * 60)
    print("Material Design Environments Demo")
    print("=" * 60)
    
    config = MaterialEnvConfig(
        element_set=['Li', 'Na', 'S', 'O', 'P', 'Cl'],
        max_elements=3,
        max_atoms=20
    )
    
    # 1. 成分环境
    print("\n1. Composition Environment")
    env = CompositionEnv(config)
    
    state = env.reset()
    print(f"   Initial state shape: {state.shape}")
    print(f"   Action dim: {env.action_dim}")
    
    # 运行几个步骤
    for i in range(15):
        valid_actions = env.get_valid_actions()
        action = np.random.choice(valid_actions)
        state, reward, done, info = env.step(action)
        
        if done:
            print(f"   Episode finished at step {i+1}")
            print(f"   Final formula: {env.get_formula()}")
            print(f"   Reward: {reward:.3f}")
            break
    
    # 2. 结构环境
    print("\n2. Structure Environment")
    structure_env = StructureEnv(
        composition={'Li': 4, 'S': 2},
        config=config
    )
    
    state = structure_env.reset()
    print(f"   Initial state shape: {state.shape}")
    print(f"   Num atoms: {structure_env.num_atoms}")
    
    # 运行几个步骤
    for i in range(20):
        valid_actions = structure_env.get_valid_actions()
        action = np.random.choice(valid_actions)
        state, reward, done, info = structure_env.step(action)
        
        if done:
            print(f"   Episode finished at step {i+1}")
            print(f"   Final lattice: {structure_env.lattice}")
            print(f"   Reward: {reward:.3f}")
            break
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
