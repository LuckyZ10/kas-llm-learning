#!/usr/bin/env python3
"""
晶体结构环境 - 晶体结构操作的强化学习环境

支持的动作:
- 添加原子 (Add)
- 删除原子 (Remove)
- 移动原子 (Move)
- 替换原子 (Replace)
- 改变晶格 (Change Lattice)
- 交换原子位置 (Swap)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .base_env import (
    MaterialOptEnv, StateRepresentation, ActionSpace, StepResult, EnvConfig,
    ActionType, max_atoms_constraint, min_atoms_constraint
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrystalState:
    """晶体结构状态表示"""
    lattice: np.ndarray  # 3x3 晶格矩阵
    positions: np.ndarray  # Nx3 原子位置 (分数坐标)
    elements: List[str]  # N 元素列表
    energies: Optional[np.ndarray] = None  # 每个原子的能量
    forces: Optional[np.ndarray] = None  # Nx3 力
    
    def copy(self) -> 'CrystalState':
        """创建副本"""
        return CrystalState(
            lattice=self.lattice.copy(),
            positions=self.positions.copy() if len(self.positions) > 0 else np.array([]),
            elements=self.elements.copy(),
            energies=self.energies.copy() if self.energies is not None else None,
            forces=self.forces.copy() if self.forces is not None else None
        )
    
    def get_composition(self) -> Dict[str, int]:
        """获取化学组成"""
        from collections import Counter
        return dict(Counter(self.elements))
    
    def get_volume(self) -> float:
        """计算晶胞体积"""
        return abs(np.linalg.det(self.lattice))
    
    def get_density(self, atomic_masses: Optional[Dict[str, float]] = None) -> float:
        """计算密度"""
        if atomic_masses is None:
            # 简化质量表
            atomic_masses = {
                'H': 1.008, 'Li': 6.94, 'Be': 9.012, 'B': 10.81, 'C': 12.01,
                'N': 14.01, 'O': 16.00, 'F': 19.00, 'Na': 22.99, 'Mg': 24.31,
                'Al': 26.98, 'Si': 28.09, 'P': 30.97, 'S': 32.07, 'Cl': 35.45,
                'K': 39.10, 'Ca': 40.08, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94,
                'Cr': 52.00, 'Mn': 54.94, 'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69,
                'Cu': 63.55, 'Zn': 65.38, 'Ga': 69.72, 'Ge': 72.63
            }
        
        total_mass = sum(atomic_masses.get(e, 50.0) for e in self.elements)
        volume = self.get_volume()
        
        # 转换为 g/cm³ (Å³ to cm³: 1e-24)
        if volume > 0:
            return total_mass / (volume * 1e-24 * 6.022e23)
        return 0.0
    
    def add_atom(self, element: str, position: np.ndarray) -> 'CrystalState':
        """添加原子"""
        new_state = self.copy()
        new_state.elements = self.elements + [element]
        new_state.positions = np.vstack([self.positions, position]) if len(self.positions) > 0 else np.array([position])
        return new_state
    
    def remove_atom(self, index: int) -> 'CrystalState':
        """删除原子"""
        if index < 0 or index >= len(self.elements):
            return self.copy()
        
        new_state = self.copy()
        new_state.elements = self.elements[:index] + self.elements[index+1:]
        mask = np.ones(len(self.positions), dtype=bool)
        mask[index] = False
        new_state.positions = self.positions[mask]
        return new_state
    
    def move_atom(self, index: int, displacement: np.ndarray) -> 'CrystalState':
        """移动原子"""
        new_state = self.copy()
        if 0 <= index < len(new_state.positions):
            new_state.positions[index] += displacement
            # 确保分数坐标在 [0, 1) 范围内
            new_state.positions[index] = new_state.positions[index] % 1.0
        return new_state
    
    def replace_atom(self, index: int, new_element: str) -> 'CrystalState':
        """替换原子"""
        new_state = self.copy()
        if 0 <= index < len(new_state.elements):
            new_state.elements[index] = new_element
        return new_state
    
    def scale_lattice(self, scale: float) -> 'CrystalState':
        """缩放晶格"""
        new_state = self.copy()
        new_state.lattice = self.lattice * scale
        return new_state


class CrystalGraphRepresentation(StateRepresentation):
    """晶体图表示 - 使用图神经网络友好的格式"""
    
    def __init__(self, config: Optional[EnvConfig] = None, max_neighbors: int = 12):
        super().__init__(config)
        self.max_neighbors = max_neighbors
        self.element_to_z = self._build_element_dict()
        self.feature_dim = 128  # 固定特征维度
    
    def _build_element_dict(self) -> Dict[str, int]:
        """构建元素到原子序数的映射"""
        element_z = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34
        }
        return element_z
    
    def encode(self, state: CrystalState) -> np.ndarray:
        """
        编码晶体状态为特征向量
        
        包含:
        - 全局特征 (晶格参数、密度、组成等)
        - 局部特征 (最近邻统计)
        """
        features = []
        
        # 全局特征
        lattice_params = self._get_lattice_params(state.lattice)
        features.extend(lattice_params)
        
        # 密度
        features.append(state.get_density())
        
        # 组成统计
        composition = state.get_composition()
        comp_vector = self._composition_to_vector(composition)
        features.extend(comp_vector)
        
        # 结构指纹 (简化版SOAP-like)
        fingerprint = self._compute_structure_fingerprint(state)
        features.extend(fingerprint)
        
        # 填充到固定维度
        result = np.array(features, dtype=np.float32)
        if len(result) < self.feature_dim:
            result = np.pad(result, (0, self.feature_dim - len(result)))
        else:
            result = result[:self.feature_dim]
        
        return result
    
    def _get_lattice_params(self, lattice: np.ndarray) -> List[float]:
        """获取晶格参数 (a, b, c, alpha, beta, gamma)"""
        a = np.linalg.norm(lattice[0])
        b = np.linalg.norm(lattice[1])
        c = np.linalg.norm(lattice[2])
        
        alpha = np.arccos(np.dot(lattice[1], lattice[2]) / (b * c)) * 180 / np.pi
        beta = np.arccos(np.dot(lattice[0], lattice[2]) / (a * c)) * 180 / np.pi
        gamma = np.arccos(np.dot(lattice[0], lattice[1]) / (a * b)) * 180 / np.pi
        
        return [a, b, c, alpha, beta, gamma]
    
    def _composition_to_vector(self, composition: Dict[str, int], max_elements: int = 20) -> List[float]:
        """将组成转换为向量"""
        vector = []
        for elem in list(self.element_to_z.keys())[:max_elements]:
            vector.append(float(composition.get(elem, 0)))
        
        # 归一化
        total = sum(vector)
        if total > 0:
            vector = [v / total for v in vector]
        
        return vector
    
    def _compute_structure_fingerprint(self, state: CrystalState, n_bins: int = 20) -> List[float]:
        """
        计算结构指纹 (径向分布函数简化版)
        """
        if len(state.positions) < 2:
            return [0.0] * n_bins
        
        # 计算所有原子对距离
        positions_cart = state.positions @ state.lattice  # 转换为笛卡尔坐标
        from scipy.spatial.distance import pdist
        distances = pdist(positions_cart)
        
        # 创建直方图
        hist, _ = np.histogram(distances, bins=n_bins, range=(0, 10.0))
        
        # 归一化
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        return hist.tolist()
    
    def decode(self, state: np.ndarray) -> CrystalState:
        """解码 (简化版，不完全可逆)"""
        # 从特征向量重建粗略结构
        a, b, c = state[0], state[1], state[2]
        lattice = np.diag([a, b, c])
        
        return CrystalState(
            lattice=lattice,
            positions=np.array([]),
            elements=[]
        )


@dataclass
class CrystalAction:
    """晶体结构动作"""
    action_type: ActionType
    params: Dict[str, Any]
    
    @classmethod
    def add_atom(cls, element: str, position: np.ndarray) -> 'CrystalAction':
        return cls(ActionType.ADD_ATOM, {'element': element, 'position': position})
    
    @classmethod
    def remove_atom(cls, index: int) -> 'CrystalAction':
        return cls(ActionType.REMOVE_ATOM, {'atom_index': index})
    
    @classmethod
    def move_atom(cls, index: int, displacement: np.ndarray) -> 'CrystalAction':
        return cls(ActionType.MOVE_ATOM, {'atom_index': index, 'displacement': displacement})
    
    @classmethod
    def replace_atom(cls, index: int, new_element: str) -> 'CrystalAction':
        return cls(ActionType.REPLACE_ATOM, {'atom_index': index, 'new_element': new_element})
    
    @classmethod
    def change_lattice(cls, scale: float) -> 'CrystalAction':
        return cls(ActionType.CHANGE_LATTICE, {'scale': scale})
    
    @classmethod
    def swap_atoms(cls, index1: int, index2: int) -> 'CrystalAction':
        return cls(ActionType.SWAP_ATOMS, {'index1': index1, 'index2': index2})


class StructureModifier:
    """结构修改器 - 执行具体的结构修改操作"""
    
    @staticmethod
    def apply_action(state: CrystalState, action: CrystalAction) -> Tuple[CrystalState, bool, str]:
        """
        应用动作到结构
        
        Returns:
            (新状态, 是否成功, 消息)
        """
        at = action.action_type
        params = action.params
        
        try:
            if at == ActionType.ADD_ATOM:
                new_state = state.add_atom(
                    params['element'],
                    np.array(params['position'])
                )
                return new_state, True, f"Added {params['element']}"
            
            elif at == ActionType.REMOVE_ATOM:
                index = params.get('atom_index')
                if index is None or index >= len(state.elements):
                    # 自动选择能量最高的原子
                    if state.energies is not None:
                        index = np.argmax(state.energies)
                    else:
                        index = len(state.elements) - 1
                new_state = state.remove_atom(index)
                return new_state, True, f"Removed atom at index {index}"
            
            elif at == ActionType.MOVE_ATOM:
                index = params.get('atom_index')
                if index is None:
                    index = np.random.randint(0, len(state.elements))
                new_state = state.move_atom(index, np.array(params['displacement']))
                return new_state, True, f"Moved atom {index}"
            
            elif at == ActionType.REPLACE_ATOM:
                index = params.get('atom_index')
                if index is None:
                    index = np.random.randint(0, len(state.elements))
                new_state = state.replace_atom(index, params['new_element'])
                return new_state, True, f"Replaced atom {index} with {params['new_element']}"
            
            elif at == ActionType.CHANGE_LATTICE:
                new_state = state.scale_lattice(params['scale'])
                return new_state, True, f"Scaled lattice by {params['scale']}"
            
            elif at == ActionType.SWAP_ATOMS:
                idx1, idx2 = params.get('index1', 0), params.get('index2', 1)
                new_state = state.copy()
                if 0 <= idx1 < len(new_state.elements) and 0 <= idx2 < len(new_state.elements):
                    new_state.elements[idx1], new_state.elements[idx2] = \
                        new_state.elements[idx2], new_state.elements[idx1]
                return new_state, True, f"Swapped atoms {idx1} and {idx2}"
            
            elif at == ActionType.TERMINATE:
                return state, True, "Terminated"
            
            else:
                return state, False, f"Unknown action type: {at}"
                
        except Exception as e:
            return state, False, str(e)


class CrystalStructureEnv(MaterialOptEnv):
    """
    晶体结构优化环境
    
    用于优化晶体结构以获得目标性质。
    """
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        reward_calculator: Optional[callable] = None,
        dft_coupling: Optional[Any] = None
    ):
        super().__init__(config)
        self.state_rep = CrystalGraphRepresentation(config)
        self.structure_modifier = StructureModifier()
        self.reward_calculator = reward_calculator
        self.dft_coupling = dft_coupling
        
        # 添加默认约束
        self.add_constraint(max_atoms_constraint(self.config.max_atoms))
        self.add_constraint(min_atoms_constraint(self.config.min_atoms))
    
    def reset(self, initial_structure: Optional[CrystalState] = None) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.episode_reward = 0.0
        self.history = []
        
        if initial_structure is None:
            # 创建默认初始结构 (例如 LiCoO2)
            self.current_structure = self._create_default_structure()
        else:
            self.current_structure = initial_structure.copy()
        
        return self.get_observation()
    
    def _create_default_structure(self) -> CrystalState:
        """创建默认初始结构"""
        # 简单的立方晶格
        lattice = np.eye(3) * 5.0  # 5 Å
        
        # 简单的Li2O结构
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
        ])
        elements = ['Li', 'Li', 'O']
        
        return CrystalState(lattice=lattice, positions=positions, elements=elements)
    
    def step(self, action: CrystalAction) -> StepResult:
        """执行一步"""
        self.current_step += 1
        
        # 应用动作
        new_structure, success, message = self.structure_modifier.apply_action(
            self.current_structure, action
        )
        
        # 检查约束
        is_valid, violations = self.check_constraints(new_structure)
        
        if not is_valid:
            # 违反约束，给予惩罚
            reward = -self.config.penalty_scale * len(violations)
            done = True
            truncated = False
        elif not success:
            # 动作执行失败
            reward = -self.config.penalty_scale
            done = False
            truncated = False
        else:
            # 计算奖励
            reward = self.compute_reward(new_structure, {
                'action': action,
                'success': success,
                'message': message
            })
            
            # 更新结构
            self.current_structure = new_structure
            
            # 检查终止条件
            done = self.is_terminal(new_structure)
            truncated = self.current_step >= self.config.max_steps
        
        self.episode_reward += reward
        
        # 记录历史
        self.history.append({
            'step': self.current_step,
            'action': action,
            'reward': reward,
            'structure': new_structure.copy(),
            'message': message
        })
        
        info = self.get_info()
        info['message'] = message
        info['success'] = success
        
        return StepResult(
            state=self.get_observation(),
            reward=reward * self.config.reward_scale,
            done=done,
            truncated=truncated,
            info=info
        )
    
    def compute_reward(self, structure: CrystalState, action_info: Dict) -> float:
        """计算奖励"""
        if self.reward_calculator is not None:
            return self.reward_calculator(structure, action_info)
        
        # 默认奖励：结构合理性
        reward = 0.0
        
        # 1. 原子间距奖励 (避免太近)
        if len(structure.positions) >= 2:
            positions_cart = structure.positions @ structure.lattice
            from scipy.spatial.distance import pdist
            distances = pdist(positions_cart)
            if len(distances) > 0:
                min_dist = np.min(distances)
                if min_dist < 1.0:  # 原子太近
                    reward -= 1.0
                else:
                    reward += 0.1
        
        # 2. 组成合理性奖励
        composition = structure.get_composition()
        if len(composition) >= 2:
            reward += 0.1  # 多元素奖励
        
        # 3. 体积合理性
        volume = structure.get_volume()
        if 10 < volume < 1000:
            reward += 0.1
        
        return reward
    
    def get_structure(self) -> CrystalState:
        """获取当前结构"""
        return self.current_structure.copy()
    
    def set_structure(self, structure: CrystalState):
        """设置当前结构"""
        self.current_structure = structure.copy()
