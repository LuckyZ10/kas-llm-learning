#!/usr/bin/env python3
"""
多目标强化学习算法

包含:
- NSGA-III (Non-dominated Sorting Genetic Algorithm III)
- MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
- 多目标RL框架

适用于材料优化中的多目标场景 (如强度-延展性平衡)。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """帕累托前沿上的点"""
    solution: np.ndarray  # 解向量
    objectives: np.ndarray  # 目标函数值
    crowding_distance: float = 0.0
    rank: int = 0
    
    def dominates(self, other: 'ParetoPoint') -> bool:
        """检查是否支配另一个点"""
        # self支配other如果self在所有目标上都不差且至少一个更好
        not_worse = np.all(self.objectives <= other.objectives)
        better = np.any(self.objectives < other.objectives)
        return not_worse and better
    
    def copy(self) -> 'ParetoPoint':
        return ParetoPoint(
            solution=self.solution.copy(),
            objectives=self.objectives.copy(),
            crowding_distance=self.crowding_distance,
            rank=self.rank
        )


class ParetoFront:
    """帕累托前沿管理器"""
    
    def __init__(self, n_objectives: int):
        self.n_objectives = n_objectives
        self.points: List[ParetoPoint] = []
    
    def add(self, point: ParetoPoint) -> bool:
        """
        添加点到帕累托前沿
        
        Returns:
            是否被接受
        """
        # 检查是否被现有解支配
        for p in self.points:
            if p.dominates(point):
                return False
        
        # 移除被新解支配的解
        self.points = [p for p in self.points if not point.dominates(p)]
        
        # 添加新解
        self.points.append(point)
        
        return True
    
    def get_solutions(self) -> np.ndarray:
        """获取所有解"""
        return np.array([p.solution for p in self.points])
    
    def get_objectives(self) -> np.ndarray:
        """获取所有目标值"""
        return np.array([p.objectives for p in self.points])
    
    def compute_hypervolume(self, reference_point: np.ndarray) -> float:
        """
        计算超体积指标 (Hypervolume indicator)
        
        用于评估帕累托前沿的质量
        """
        if len(self.points) == 0:
            return 0.0
        
        objectives = self.get_objectives()
        
        # 简化: 使用Monte Carlo估计
        n_samples = 10000
        volume = 0.0
        
        # 采样参考区域内的点
        for _ in range(n_samples):
            sample = np.random.uniform(
                objectives.min(axis=0),
                reference_point
            )
            
            # 检查是否被任何帕累托解支配
            dominated = False
            for obj in objectives:
                if np.all(obj <= sample):
                    dominated = True
                    break
            
            if dominated:
                volume += 1.0
        
        # 归一化
        total_volume = np.prod(reference_point - objectives.min(axis=0))
        return volume / n_samples * total_volume
    
    def compute_spread(self) -> float:
        """计算分布度 (Spread)"""
        if len(self.points) < 2:
            return 0.0
        
        objectives = self.get_objectives()
        
        # 计算相邻解的距离
        distances = []
        for i in range(len(objectives)):
            for j in range(i + 1, len(objectives)):
                dist = np.linalg.norm(objectives[i] - objectives[j])
                distances.append(dist)
        
        return np.std(distances) / (np.mean(distances) + 1e-10)
    
    def __len__(self) -> int:
        return len(self.points)


@dataclass
class MultiObjectiveConfig:
    """多目标优化配置"""
    n_objectives: int = 2
    population_size: int = 100
    n_generations: int = 200
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    reference_points: Optional[np.ndarray] = None
    
    # RL相关
    state_dim: int = 128
    action_dim: int = 10
    learning_rate: float = 3e-4


class MultiObjectiveRL(ABC):
    """多目标强化学习基类"""
    
    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        self.config = config or MultiObjectiveConfig()
        self.pareto_front = ParetoFront(self.config.n_objectives)
        self.generation = 0
    
    @abstractmethod
    def optimize(
        self,
        objective_fns: List[Callable],
        initial_population: Optional[np.ndarray] = None
    ) -> ParetoFront:
        """
        执行多目标优化
        
        Args:
            objective_fns: 目标函数列表
            initial_population: 初始种群
            
        Returns:
            帕累托前沿
        """
        pass
    
    def evaluate_objectives(
        self,
        solution: np.ndarray,
        objective_fns: List[Callable]
    ) -> np.ndarray:
        """评估所有目标"""
        return np.array([fn(solution) for fn in objective_fns])


class NSGA3Agent(MultiObjectiveRL):
    """
    NSGA-III算法
    
    参考: Deb & Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based 
    Nondominated Sorting Approach", IEEE TEC 2014
    
    适用于多目标材料优化 (3个或更多目标)。
    """
    
    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        super().__init__(config)
        
        # 生成参考点
        if self.config.reference_points is None:
            self.reference_points = self._generate_reference_points(
                self.config.n_objectives,
                divisions=12
            )
        else:
            self.reference_points = self.config.reference_points
    
    def _generate_reference_points(self, n_objectives: int, divisions: int = 12) -> np.ndarray:
        """生成参考点 (单纯形采样)"""
        if n_objectives == 2:
            return np.array([[i / divisions, 1 - i / divisions] for i in range(divisions + 1)])
        
        # 对于3+目标，使用两层次参考点
        ref_points_1 = self._two_layer_reference_points(n_objectives, divisions, 0)
        ref_points_2 = self._two_layer_reference_points(n_objectives, divisions // 2, 0.5)
        
        return np.vstack([ref_points_1, ref_points_2])
    
    def _two_layer_reference_points(
        self,
        n_objectives: int,
        divisions: int,
        offset: float
    ) -> np.ndarray:
        """生成两层参考点"""
        from itertools import combinations_with_replacement
        
        ref_points = []
        for comb in combinations_with_replacement(range(n_objectives), divisions):
            point = np.zeros(n_objectives)
            for i in comb:
                point[i] += 1.0 / divisions
            ref_points.append(point * (1 - offset) + offset / n_objectives)
        
        return np.array(ref_points)
    
    def optimize(
        self,
        objective_fns: List[Callable],
        initial_population: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> ParetoFront:
        """NSGA-III优化"""
        n_vars = len(initial_population[0]) if initial_population is not None else 10
        
        # 初始化种群
        if initial_population is None:
            population = np.random.random((self.config.population_size, n_vars))
        else:
            population = initial_population.copy()
        
        if bounds is not None:
            lower, upper = bounds
            population = lower + population * (upper - lower)
        
        for generation in range(self.config.n_generations):
            # 评估目标
            objectives = np.array([
                self.evaluate_objectives(ind, objective_fns)
                for ind in population
            ])
            
            # 非支配排序
            fronts = self._non_dominated_sort(population, objectives)
            
            # 计算拥挤距离
            for front in fronts:
                self._compute_crowding_distance(front)
            
            # 选择下一代
            selected = self._environmental_selection(fronts, self.config.population_size)
            
            # 遗传操作
            offspring = self._genetic_operators(population[selected], bounds)
            
            # 合并种群
            population = np.vstack([population, offspring])[:self.config.population_size]
            
            self.generation = generation
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}, Front size: {len(fronts[0])}")
        
        # 最终评估
        objectives = np.array([
            self.evaluate_objectives(ind, objective_fns)
            for ind in population
        ])
        fronts = self._non_dominated_sort(population, objectives)
        
        # 构建帕累托前沿
        for i, idx in enumerate(fronts[0]):
            point = ParetoPoint(
                solution=population[idx],
                objectives=objectives[idx]
            )
            self.pareto_front.add(point)
        
        return self.pareto_front
    
    def _non_dominated_sort(
        self,
        population: np.ndarray,
        objectives: np.ndarray
    ) -> List[List[int]]:
        """非支配排序"""
        n = len(population)
        dominated_count = np.zeros(n, dtype=int)
        dominating_sets = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                    dominating_sets[i].append(j)
                    dominated_count[j] += 1
                elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    dominating_sets[j].append(i)
                    dominated_count[i] += 1
            
            if dominated_count[i] == 0:
                fronts[0].append(i)
        
        # 构建后续前沿
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominating_sets[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # 移除空前沿
    
    def _compute_crowding_distance(self, front_indices: List[int]):
        """计算拥挤距离"""
        # 简化实现
        pass
    
    def _environmental_selection(
        self,
        fronts: List[List[int]],
        n_select: int
    ) -> np.ndarray:
        """环境选择 (基于参考点)"""
        selected = []
        
        for front in fronts:
            if len(selected) + len(front) <= n_select:
                selected.extend(front)
            else:
                # 需要截断
                n_needed = n_select - len(selected)
                selected.extend(front[:n_needed])
                break
        
        return np.array(selected)
    
    def _genetic_operators(
        self,
        parents: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """遗传操作 (交叉和变异)"""
        offspring = []
        n_parents = len(parents)
        
        for i in range(0, n_parents, 2):
            if i + 1 >= n_parents:
                break
            
            parent1, parent2 = parents[i], parents[i + 1]
            
            # 交叉 (SBX - Simulated Binary Crossover)
            if np.random.random() < self.config.crossover_prob:
                child1, child2 = self._sbx_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异 (多项式变异)
            if np.random.random() < self.config.mutation_prob:
                child1 = self._polynomial_mutation(child1)
            if np.random.random() < self.config.mutation_prob:
                child2 = self._polynomial_mutation(child2)
            
            # 边界处理
            if bounds is not None:
                lower, upper = bounds
                child1 = np.clip(child1, lower, upper)
                child2 = np.clip(child2, lower, upper)
            else:
                child1 = np.clip(child1, 0, 1)
                child2 = np.clip(child2, 0, 1)
            
            offspring.extend([child1, child2])
        
        return np.array(offspring)
    
    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        eta: float = 15.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """模拟二进制交叉"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    beta = 1.0 + (2.0 * (y1 - 0) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1))
                    
                    rand = np.random.random()
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (1 - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1))
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    child1[i] = c1
                    child2[i] = c2
        
        return child1, child2
    
    def _polynomial_mutation(
        self,
        individual: np.ndarray,
        eta: float = 20.0
    ) -> np.ndarray:
        """多项式变异"""
        mutant = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < 1.0 / len(individual):
                delta1 = (individual[i] - 0) / (1 - 0)
                delta2 = (1 - individual[i]) / (1 - 0)
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                    delta_q = 1.0 - val ** mut_pow
                
                mutant[i] += delta_q
                mutant[i] = np.clip(mutant[i], 0, 1)
        
        return mutant


class MOEADAgent(MultiObjectiveRL):
    """
    MOEA/D算法 (基于分解的多目标进化算法)
    
    参考: Zhang & Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition",
    IEEE TEC 2007
    
    适用于高维目标空间的材料优化。
    """
    
    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        super().__init__(config)
        
        # 生成权重向量
        self.weight_vectors = self._generate_weight_vectors(
            self.config.n_objectives,
            self.config.population_size
        )
        
        # 邻居结构
        self.neighborhood_size = min(20, self.config.population_size)
        self.neighbors = self._compute_neighbors()
    
    def _generate_weight_vectors(self, n_objectives: int, n_vectors: int) -> np.ndarray:
        """生成均匀分布的权重向量"""
        if n_objectives == 2:
            return np.array([[i / (n_vectors - 1), 1 - i / (n_vectors - 1)] 
                           for i in range(n_vectors)])
        
        # 对于3+目标，使用单纯形采样
        from itertools import combinations_with_replacement
        
        # 确定分割数
        H = int((n_vectors * np.math.factorial(n_objectives - 1)) ** (1.0 / (n_objectives - 1)))
        
        weights = []
        for comb in combinations_with_replacement(range(n_objectives), H):
            weight = np.zeros(n_objectives)
            for i in comb:
                weight[i] += 1.0
            weight = weight / H
            weights.append(weight)
        
        return np.array(weights[:n_vectors])
    
    def _compute_neighbors(self) -> np.ndarray:
        """计算每个权重向量的邻居"""
        from scipy.spatial.distance import cdist
        
        distances = cdist(self.weight_vectors, self.weight_vectors)
        neighbors = np.argsort(distances, axis=1)[:, :self.neighborhood_size]
        
        return neighbors
    
    def optimize(
        self,
        objective_fns: List[Callable],
        initial_population: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> ParetoFront:
        """MOEA/D优化"""
        n_vars = len(initial_population[0]) if initial_population is not None else 10
        
        # 初始化种群
        if initial_population is None:
            population = np.random.random((self.config.population_size, n_vars))
        else:
            population = initial_population.copy()
        
        if bounds is not None:
            lower, upper = bounds
            population = lower + population * (upper - lower)
        
        # 评估目标
        objectives = np.array([
            self.evaluate_objectives(ind, objective_fns)
            for ind in population
        ])
        
        # 参考点 (理想点)
        z = objectives.min(axis=0)
        
        for generation in range(self.config.n_generations):
            for i in range(self.config.population_size):
                # 从邻居中选择父母
                neighbors = self.neighbors[i]
                parent_indices = np.random.choice(neighbors, 2, replace=False)
                parent1, parent2 = population[parent_indices]
                
                # 遗传操作
                child = self._differential_evolution(parent1, parent2, population[i])
                
                if bounds is not None:
                    lower, upper = bounds
                    child = np.clip(child, lower, upper)
                
                # 评估
                child_obj = self.evaluate_objectives(child, objective_fns)
                
                # 更新参考点
                z = np.minimum(z, child_obj)
                
                # 更新邻居
                for j in neighbors:
                    if self._techebycheff(child_obj, self.weight_vectors[j], z) < \
                       self._techebycheff(objectives[j], self.weight_vectors[j], z):
                        population[j] = child
                        objectives[j] = child_obj
            
            self.generation = generation
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}")
        
        # 构建帕累托前沿
        for i in range(self.config.population_size):
            point = ParetoPoint(
                solution=population[i],
                objectives=objectives[i]
            )
            self.pareto_front.add(point)
        
        return self.pareto_front
    
    def _differential_evolution(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        current: np.ndarray,
        F: float = 0.5,
        CR: float = 0.9
    ) -> np.ndarray:
        """差分进化变异和交叉"""
        # 变异
        mutant = current + F * (parent1 - parent2)
        
        # 交叉
        trial = current.copy()
        for i in range(len(current)):
            if np.random.random() < CR:
                trial[i] = mutant[i]
        
        return trial
    
    def _techebycheff(
        self,
        objectives: np.ndarray,
        weights: np.ndarray,
        z: np.ndarray
    ) -> float:
        """Tchebycheff聚合函数"""
        return np.max(weights * np.abs(objectives - z))
