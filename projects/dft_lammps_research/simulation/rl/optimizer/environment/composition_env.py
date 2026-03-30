#!/usr/bin/env python3
"""
化学组成环境 - 化学组成优化的强化学习环境

支持的动作:
- 调整元素比例
- 添加/删除元素
- 调整化学计量比
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from .base_env import (
    MaterialOptEnv, StateRepresentation, ActionSpace, StepResult, EnvConfig,
    ActionType
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompositionState:
    """化学组成状态"""
    formula: Dict[str, float]  # 元素 -> 摩尔分数
    constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # 元素约束
    target_properties: Dict[str, float] = field(default_factory=dict)
    
    def copy(self) -> 'CompositionState':
        """创建副本"""
        return CompositionState(
            formula=self.formula.copy(),
            constraints=self.constraints.copy(),
            target_properties=self.target_properties.copy()
        )
    
    def get_total(self) -> float:
        """获取总摩尔数"""
        return sum(self.formula.values())
    
    def get_fractions(self) -> Dict[str, float]:
        """获取摩尔分数"""
        total = self.get_total()
        if total > 0:
            return {elem: count / total for elem, count in self.formula.items()}
        return self.formula.copy()
    
    def normalize(self) -> 'CompositionState':
        """归一化组成"""
        total = self.get_total()
        if total > 0:
            new_formula = {elem: count / total for elem, count in self.formula.items()}
            return CompositionState(
                formula=new_formula,
                constraints=self.constraints.copy(),
                target_properties=self.target_properties.copy()
            )
        return self.copy()
    
    def add_element(self, element: str, amount: float = 0.1) -> 'CompositionState':
        """添加元素"""
        new_state = self.copy()
        new_state.formula[element] = new_state.formula.get(element, 0) + amount
        return new_state.normalize()
    
    def remove_element(self, element: str, amount: float = 0.1) -> 'CompositionState':
        """减少元素"""
        new_state = self.copy()
        if element in new_state.formula:
            new_state.formula[element] = max(0, new_state.formula[element] - amount)
            if new_state.formula[element] <= 1e-6:
                del new_state.formula[element]
        return new_state.normalize()
    
    def adjust_element(self, element: str, delta: float) -> 'CompositionState':
        """调整元素比例"""
        new_state = self.copy()
        if element not in new_state.formula and delta > 0:
            new_state.formula[element] = 0
        
        if element in new_state.formula:
            new_state.formula[element] = max(0, new_state.formula[element] + delta)
            if new_state.formula[element] <= 1e-6:
                del new_state.formula[element]
        
        return new_state.normalize()
    
    def swap_elements(self, elem1: str, elem2: str) -> 'CompositionState':
        """交换两种元素的比例"""
        new_state = self.copy()
        if elem1 in new_state.formula and elem2 in new_state.formula:
            new_state.formula[elem1], new_state.formula[elem2] = \
                new_state.formula[elem2], new_state.formula[elem1]
        return new_state
    
    def to_string(self) -> str:
        """转换为化学式字符串"""
        parts = []
        for elem, count in sorted(self.formula.items()):
            if count > 0:
                if abs(count - round(count)) < 0.01:
                    parts.append(f"{elem}{int(round(count))}")
                else:
                    parts.append(f"{elem}{count:.2f}")
        return "".join(parts) if parts else "Empty"


class CompositionRepresentation(StateRepresentation):
    """化学组成表示"""
    
    def __init__(self, config: Optional[EnvConfig] = None, max_elements: int = 20):
        super().__init__(config)
        self.max_elements = max_elements
        self.element_list = (config.element_set if config else [])[:max_elements]
        self.feature_dim = max_elements * 2 + 5  # 组成 + 约束 + 统计特征
    
    def encode(self, state: CompositionState) -> np.ndarray:
        """编码化学组成状态"""
        features = []
        
        # 组成向量
        fractions = state.get_fractions()
        composition_vec = [fractions.get(elem, 0.0) for elem in self.element_list]
        features.extend(composition_vec)
        
        # 约束向量
        constraint_vec = []
        for elem in self.element_list:
            if elem in state.constraints:
                min_val, max_val = state.constraints[elem]
                current = fractions.get(elem, 0.0)
                # 约束满足程度 (-1 to 1, 0 means satisfied)
                if current < min_val:
                    constraint_vec.append((current - min_val) / (min_val + 1e-6))
                elif current > max_val:
                    constraint_vec.append((current - max_val) / (max_val + 1e-6))
                else:
                    constraint_vec.append(0.0)
            else:
                constraint_vec.append(0.0)
        features.extend(constraint_vec)
        
        # 统计特征
        features.append(len(state.formula))  # 元素数量
        features.append(state.get_total())  # 总摩尔数
        features.append(np.std(list(fractions.values())) if fractions else 0)  # 组成标准差
        features.append(entropy(list(fractions.values())) if fractions else 0)  # 组成熵
        features.append(max(fractions.values()) if fractions else 0)  # 最大组分
        
        return np.array(features, dtype=np.float32)
    
    def decode(self, state: np.ndarray) -> CompositionState:
        """解码组成向量"""
        # 提取组成部分
        composition_part = state[:self.max_elements]
        
        formula = {}
        for i, elem in enumerate(self.element_list):
            if composition_part[i] > 0.01:  # 阈值
                formula[elem] = composition_part[i]
        
        return CompositionState(formula=formula)


def entropy(probs: List[float]) -> float:
    """计算熵"""
    probs = np.array(probs)
    probs = probs[probs > 0]
    if len(probs) > 0:
        return -np.sum(probs * np.log(probs + 1e-10))
    return 0.0


@dataclass
class CompositionAction:
    """化学组成动作"""
    action_type: ActionType
    params: Dict[str, Any]
    
    @classmethod
    def add_element(cls, element: str, amount: float = 0.1) -> 'CompositionAction':
        return cls(ActionType.ADD_ATOM, {'element': element, 'amount': amount})
    
    @classmethod
    def remove_element(cls, element: str, amount: float = 0.1) -> 'CompositionAction':
        return cls(ActionType.REMOVE_ATOM, {'element': element, 'amount': amount})
    
    @classmethod
    def adjust_element(cls, element: str, delta: float) -> 'CompositionAction':
        return cls(ActionType.COMPOSITION_CHANGE, {'element': element, 'delta': delta})
    
    @classmethod
    def swap_elements(cls, elem1: str, elem2: str) -> 'CompositionAction':
        return cls(ActionType.SWAP_ATOMS, {'elem1': elem1, 'elem2': elem2})


class ElementSelector:
    """元素选择器 - 基于化学直觉选择要调整的元素"""
    
    def __init__(self, element_properties: Optional[Dict] = None):
        self.element_properties = element_properties or self._load_element_properties()
    
    def _load_element_properties(self) -> Dict:
        """加载元素性质"""
        # 简化版本 - 实际应加载完整周期表数据
        return {
            'Li': {'group': 1, 'period': 2, 'electronegativity': 0.98, 'radius': 152},
            'Na': {'group': 1, 'period': 3, 'electronegativity': 0.93, 'radius': 186},
            'K': {'group': 1, 'period': 4, 'electronegativity': 0.82, 'radius': 227},
            'Mg': {'group': 2, 'period': 3, 'electronegativity': 1.31, 'radius': 160},
            'Ca': {'group': 2, 'period': 4, 'electronegativity': 1.00, 'radius': 197},
            'Al': {'group': 13, 'period': 3, 'electronegativity': 1.61, 'radius': 143},
            'Si': {'group': 14, 'period': 3, 'electronegativity': 1.90, 'radius': 111},
            'P': {'group': 15, 'period': 3, 'electronegativity': 2.19, 'radius': 100},
            'S': {'group': 16, 'period': 3, 'electronegativity': 2.58, 'radius': 100},
            'O': {'group': 16, 'period': 2, 'electronegativity': 3.44, 'radius': 73},
            'F': {'group': 17, 'period': 2, 'electronegativity': 3.98, 'radius': 72},
            'Cl': {'group': 17, 'period': 3, 'electronegativity': 3.16, 'radius': 99},
            'Ti': {'group': 4, 'period': 4, 'electronegativity': 1.54, 'radius': 176},
            'V': {'group': 5, 'period': 4, 'electronegativity': 1.63, 'radius': 171},
            'Cr': {'group': 6, 'period': 4, 'electronegativity': 1.66, 'radius': 166},
            'Mn': {'group': 7, 'period': 4, 'electronegativity': 1.55, 'radius': 161},
            'Fe': {'group': 8, 'period': 4, 'electronegativity': 1.83, 'radius': 156},
            'Co': {'group': 9, 'period': 4, 'electronegativity': 1.88, 'radius': 152},
            'Ni': {'group': 10, 'period': 4, 'electronegativity': 1.91, 'radius': 149},
            'Cu': {'group': 11, 'period': 4, 'electronegativity': 1.90, 'radius': 145},
            'Zn': {'group': 12, 'period': 4, 'electronegativity': 1.65, 'radius': 142},
        }
    
    def select_substitute(self, element: str, target_property: str) -> List[Tuple[str, float]]:
        """
        选择可能的替代元素
        
        Args:
            element: 要替代的元素
            target_property: 目标性质 (如 'ionic_conductivity', 'strength')
            
        Returns:
            [(候选元素, 相似度分数), ...]
        """
        if element not in self.element_properties:
            return []
        
        elem_props = self.element_properties[element]
        candidates = []
        
        for cand, props in self.element_properties.items():
            if cand == element:
                continue
            
            # 计算相似度 (基于同族、周期相近等)
            similarity = 0.0
            
            # 同族加分
            if props['group'] == elem_props['group']:
                similarity += 0.5
            
            # 相邻周期加分
            if abs(props['period'] - elem_props['period']) <= 1:
                similarity += 0.3
            
            # 电负性相近加分
            en_diff = abs(props['electronegativity'] - elem_props['electronegativity'])
            similarity += max(0, 0.2 - en_diff * 0.1)
            
            candidates.append((cand, similarity))
        
        # 排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:5]  # 返回前5个
    
    def suggest_dopant(self, base_composition: Dict[str, float], target: str) -> List[str]:
        """建议掺杂元素"""
        suggestions = []
        
        # 基于目标选择掺杂
        if target == 'n_type':
            # N型掺杂 - 选择多价电子元素
            suggestions = ['P', 'As', 'Sb']  # 族数更高的元素
        elif target == 'p_type':
            # P型掺杂 - 选择少价电子元素
            suggestions = ['B', 'Al', 'Ga']
        elif target == 'ionic_conductivity':
            suggestions = ['Li', 'Na', 'K', 'Mg', 'Ca']
        elif target == 'strength':
            suggestions = ['Ti', 'V', 'Cr', 'Mo', 'W']
        
        return suggestions


class CompositionEnv(MaterialOptEnv):
    """
    化学组成优化环境
    
    用于优化材料的化学组成以获得目标性质。
    """
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        reward_calculator: Optional[callable] = None,
        target_properties: Optional[Dict[str, float]] = None,
        element_selector: Optional[ElementSelector] = None
    ):
        super().__init__(config)
        self.state_rep = CompositionRepresentation(config)
        self.reward_calculator = reward_calculator
        self.target_properties = target_properties or {}
        self.element_selector = element_selector or ElementSelector()
        
        self.current_composition = None
    
    def reset(self, initial_composition: Optional[CompositionState] = None) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.episode_reward = 0.0
        self.history = []
        
        if initial_composition is None:
            # 创建默认初始组成 (例如 LiFePO4 简化版)
            self.current_composition = CompositionState(
                formula={'Li': 1.0, 'Fe': 1.0, 'P': 1.0, 'O': 4.0}
            ).normalize()
        else:
            self.current_composition = initial_composition.copy()
        
        return self.get_observation()
    
    def step(self, action: CompositionAction) -> StepResult:
        """执行一步"""
        self.current_step += 1
        
        # 应用动作
        new_composition = self._apply_action(self.current_composition, action)
        
        # 检查约束
        is_valid, violations = self.check_constraints(new_composition)
        
        if not is_valid:
            reward = -self.config.penalty_scale * len(violations)
            done = True
            truncated = False
        else:
            # 计算奖励
            reward = self.compute_reward(new_composition, {'action': action})
            
            # 更新组成
            self.current_composition = new_composition
            
            # 检查终止条件
            done = self.is_terminal(new_composition)
            truncated = self.current_step >= self.config.max_steps
        
        self.episode_reward += reward
        
        # 记录历史
        self.history.append({
            'step': self.current_step,
            'action': action,
            'reward': reward,
            'composition': new_composition.copy()
        })
        
        info = self.get_info()
        info['formula'] = new_composition.to_string()
        
        return StepResult(
            state=self.get_observation(),
            reward=reward * self.config.reward_scale,
            done=done,
            truncated=truncated,
            info=info
        )
    
    def _apply_action(
        self,
        composition: CompositionState,
        action: CompositionAction
    ) -> CompositionState:
        """应用动作到组成"""
        at = action.action_type
        params = action.params
        
        if at == ActionType.ADD_ATOM:
            return composition.add_element(params['element'], params.get('amount', 0.1))
        
        elif at == ActionType.REMOVE_ATOM:
            return composition.remove_element(params['element'], params.get('amount', 0.1))
        
        elif at == ActionType.COMPOSITION_CHANGE:
            return composition.adjust_element(params['element'], params['delta'])
        
        elif at == ActionType.SWAP_ATOMS:
            return composition.swap_elements(params['elem1'], params['elem2'])
        
        else:
            return composition.copy()
    
    def compute_reward(self, composition: CompositionState, action_info: Dict) -> float:
        """计算奖励"""
        if self.reward_calculator is not None:
            return self.reward_calculator(composition, action_info)
        
        # 默认奖励函数
        reward = 0.0
        
        # 1. 组成合理性
        fractions = composition.get_fractions()
        
        # 避免单一元素主导
        if fractions:
            max_frac = max(fractions.values())
            if max_frac < 0.8:
                reward += 0.1
        
        # 2. 约束满足
        for elem, (min_val, max_val) in composition.constraints.items():
            current = fractions.get(elem, 0.0)
            if min_val <= current <= max_val:
                reward += 0.1
            else:
                reward -= 0.2
        
        # 3. 目标性质接近度 (简化)
        for prop, target_val in self.target_properties.items():
            # 这里可以调用ML势或DFT计算
            # 简化：给予探索奖励
            reward += 0.05
        
        return reward
    
    def get_observation(self) -> np.ndarray:
        """获取当前观测"""
        if self.current_composition is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return self.state_rep.encode(self.current_composition)
    
    def get_composition(self) -> CompositionState:
        """获取当前组成"""
        return self.current_composition.copy()
    
    def set_composition(self, composition: CompositionState):
        """设置当前组成"""
        self.current_composition = composition.copy()
    
    def suggest_action(self) -> CompositionAction:
        """基于化学直觉建议动作"""
        # 简单启发式：尝试添加/调整可能改善性质的元素
        fractions = self.current_composition.get_fractions()
        
        if not fractions:
            # 空组成，添加基础元素
            return CompositionAction.add_element('Li')
        
        # 随机选择一种调整策略
        strategy = np.random.choice(['add', 'remove', 'adjust', 'swap'])
        
        if strategy == 'add':
            # 建议掺杂元素
            suggestions = self.element_selector.suggest_dopant(fractions, 'ionic_conductivity')
            if suggestions:
                elem = np.random.choice(suggestions)
                return CompositionAction.add_element(elem, 0.05)
        
        elif strategy == 'remove' and len(fractions) > 2:
            # 移除少量占比最大的元素
            max_elem = max(fractions.items(), key=lambda x: x[1])[0]
            return CompositionAction.remove_element(max_elem, 0.05)
        
        elif strategy == 'adjust' and fractions:
            # 调整随机元素
            elem = np.random.choice(list(fractions.keys()))
            delta = np.random.choice([-0.1, 0.1])
            return CompositionAction.adjust_element(elem, delta)
        
        elif strategy == 'swap' and len(fractions) >= 2:
            # 交换两种元素
            elems = list(fractions.keys())
            return CompositionAction.swap_elements(elems[0], elems[1])
        
        return CompositionAction.add_element('Li', 0.05)
