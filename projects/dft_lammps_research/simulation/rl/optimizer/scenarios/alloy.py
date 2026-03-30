#!/usr/bin/env python3
"""
合金优化场景

优化目标:
- 高强度
- 高延展性
- 低密度 (轻量化)
- 耐腐蚀性
- 成本效益
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from ..environment import CompositionEnv, CompositionState, EnvConfig
from ..algorithms import MOEADAgent, MultiObjectiveConfig
from ..rewards import create_alloy_reward
from ..algorithms.multi_objective import ParetoFront

logger = logging.getLogger(__name__)


@dataclass
class AlloyConfig:
    """合金优化配置"""
    # 基体元素
    base_elements: List[str] = None
    
    # 合金类型
    alloy_type: str = 'lightweight'  # lightweight, high_strength, corrosion_resistant
    
    # 目标性质范围
    target_strength_range: Tuple[float, float] = (500, 2000)  # MPa
    target_ductility_range: Tuple[float, float] = (0.1, 0.5)  # 断裂应变
    max_density: float = 5.0  # g/cm³
    
    # 优化参数
    n_generations: int = 200
    population_size: int = 100
    
    def __post_init__(self):
        if self.base_elements is None:
            if self.alloy_type == 'lightweight':
                self.base_elements = ['Al', 'Mg', 'Ti']
            elif self.alloy_type == 'high_strength':
                self.base_elements = ['Fe', 'Ni', 'Co', 'Ti']
            elif self.alloy_type == 'corrosion_resistant':
                self.base_elements = ['Cr', 'Ni', 'Ti', 'Mo']
            else:
                self.base_elements = ['Al', 'Fe', 'Ti']


class AlloyOptimizer:
    """
    合金优化器
    
    使用多目标进化算法优化合金组成，
    平衡强度、延展性、密度等多个目标。
    """
    
    def __init__(
        self,
        config: Optional[AlloyConfig] = None,
        env_config: Optional[EnvConfig] = None
    ):
        self.config = config or AlloyConfig()
        self.env_config = env_config or EnvConfig()
        
        # 设置元素集
        self.env_config.element_set = (
            self.config.base_elements +
            ['Cu', 'Zn', 'Mn', 'Si', 'V', 'Cr', 'Mo', 'W', 'Nb', 'Zr', 'Sc', 'Y']
        )
        
        # 创建奖励函数
        self.reward_composer = create_alloy_reward()
        
        # 创建环境 (使用组成环境)
        self.env = CompositionEnv(
            config=self.env_config,
            reward_calculator=self._compute_reward
        )
        
        # 创建多目标优化器
        mo_config = MultiObjectiveConfig(
            n_objectives=3,  # 强度、延展性、密度
            population_size=self.config.population_size,
            n_generations=self.config.n_generations
        )
        self.agent = MOEADAgent(mo_config)
        
        # 帕累托前沿
        self.pareto_front = None
    
    def _compute_reward(self, composition: CompositionState, info: Dict) -> float:
        """计算合金奖励"""
        reward, details = self.reward_composer.compute(composition, info)
        return reward
    
    def optimize(self) -> ParetoFront:
        """
        执行多目标优化
        
        Returns:
            帕累托前沿
        """
        logger.info(f"Starting alloy optimization ({self.config.alloy_type})")
        logger.info(f"Population size: {self.config.population_size}")
        logger.info(f"Generations: {self.config.n_generations}")
        
        # 定义目标函数
        def objective_strength(composition_vec: np.ndarray) -> float:
            """强度目标 (最大化)"""
            # 简化估算：基于固溶强化和析出强化
            composition = self._vec_to_composition(composition_vec)
            
            strength = 0.0
            
            # 固溶强化贡献
            strengthening_elements = {
                'Mn': 50, 'Si': 80, 'Cu': 60, 'Mg': 40, 'Zn': 45,
                'Cr': 70, 'Mo': 100, 'V': 90, 'Ti': 85, 'Nb': 95
            }
            
            for elem, fraction in composition.items():
                strength += strengthening_elements.get(elem, 0) * fraction
            
            # 基础强度
            base_strength = 100
            
            return -(base_strength + strength)  # 最小化负值 = 最大化
        
        def objective_ductility(composition_vec: np.ndarray) -> float:
            """延展性目标 (最大化)"""
            composition = self._vec_to_composition(composition_vec)
            
            # FCC结构通常延展性更好
            # 简化：高Ni、Cu含量倾向于FCC
            fcc_formers = composition.get('Ni', 0) + composition.get('Cu', 0)
            total = sum(composition.values())
            
            if total > 0:
                fcc_fraction = fcc_formers / total
                ductility = 0.1 + 0.4 * fcc_fraction
            else:
                ductility = 0.1
            
            return -ductility  # 最小化负值
        
        def objective_density(composition_vec: np.ndarray) -> float:
            """密度目标 (最小化)"""
            composition = self._vec_to_composition(composition_vec)
            
            # 元素密度 (g/cm³)
            densities = {
                'Al': 2.7, 'Mg': 1.74, 'Ti': 4.5, 'Fe': 7.87, 'Ni': 8.9,
                'Cu': 8.96, 'Zn': 7.14, 'Mn': 7.21, 'Si': 2.33,
                'Cr': 7.19, 'Mo': 10.28, 'V': 6.11, 'W': 19.25,
                'Nb': 8.57, 'Zr': 6.52, 'Sc': 2.99, 'Y': 4.47
            }
            
            total_mass = 0.0
            total_volume = 0.0
            
            for elem, fraction in composition.items():
                mass = fraction
                density = densities.get(elem, 7.0)
                volume = mass / density
                total_mass += mass
                total_volume += volume
            
            if total_volume > 0:
                density = total_mass / total_volume
            else:
                density = 10.0
            
            return density
        
        # 执行优化
        objective_fns = [objective_strength, objective_ductility, objective_density]
        
        # 初始化种群
        initial_population = self._initialize_population()
        
        self.pareto_front = self.agent.optimize(
            objective_fns,
            initial_population=initial_population,
            bounds=(np.zeros(len(self.env_config.element_set)), np.ones(len(self.env_config.element_set)))
        )
        
        logger.info(f"Optimization complete. Pareto front size: {len(self.pareto_front)}")
        
        return self.pareto_front
    
    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = []
        n_elements = len(self.env_config.element_set)
        
        for _ in range(self.config.population_size):
            # 随机组成 (确保总和为1)
            composition = np.random.random(n_elements)
            composition = composition / composition.sum()
            population.append(composition)
        
        return np.array(population)
    
    def _vec_to_composition(self, vec: np.ndarray) -> Dict[str, float]:
        """向量转组成字典"""
        composition = {}
        for i, elem in enumerate(self.env_config.element_set):
            if i < len(vec) and vec[i] > 0.01:
                composition[elem] = vec[i]
        return composition
    
    def get_best_alloys(self, n: int = 5) -> List[Dict]:
        """获取最佳合金组成"""
        if self.pareto_front is None:
            logger.warning("No Pareto front available. Run optimize() first.")
            return []
        
        alloys = []
        for point in self.pareto_front.points:
            composition = self._vec_to_composition(point.solution)
            
            alloys.append({
                'composition': composition,
                'strength': -point.objectives[0],
                'ductility': -point.objectives[1],
                'density': point.objectives[2]
            })
        
        # 按综合性能排序
        alloys.sort(key=lambda x: x['strength'] + x['ductility'] * 1000 - x['density'] * 10, reverse=True)
        
        return alloys[:n]
    
    def evaluate_alloy(self, composition: Dict[str, float]) -> Dict[str, float]:
        """评估合金性能"""
        properties = {}
        
        # 计算密度
        densities = {
            'Al': 2.7, 'Mg': 1.74, 'Ti': 4.5, 'Fe': 7.87, 'Ni': 8.9,
            'Cu': 8.96, 'Zn': 7.14, 'Mn': 7.21, 'Si': 2.33,
            'Cr': 7.19, 'Mo': 10.28, 'V': 6.11, 'W': 19.25
        }
        
        total = sum(composition.values())
        weighted_density = sum(
            fraction / total * densities.get(elem, 7.0)
            for elem, fraction in composition.items()
        )
        properties['density'] = weighted_density
        
        # 估算强度 (简化)
        strengthening = {
            'Mn': 50, 'Si': 80, 'Cu': 60, 'Mg': 40, 'Zn': 45,
            'Cr': 70, 'Mo': 100, 'V': 90, 'Ti': 85
        }
        
        strength = sum(
            fraction / total * strengthening.get(elem, 0)
            for elem, fraction in composition.items()
        )
        properties['strength'] = 100 + strength
        
        # 估算成本
        costs = {
            'Al': 2.0, 'Mg': 3.0, 'Ti': 10.0, 'Fe': 0.5, 'Ni': 15.0,
            'Cu': 6.0, 'Zn': 2.5, 'Mn': 3.0, 'Si': 2.0,
            'Cr': 8.0, 'Mo': 30.0, 'V': 25.0, 'W': 40.0
        }
        
        cost = sum(
            fraction / total * costs.get(elem, 5.0)
            for elem, fraction in composition.items()
        )
        properties['cost_per_kg'] = cost
        
        return properties
