"""
Multi-Objective Optimization
============================

多目标优化集成
- Pareto优化
- 标量化方法
- 约束处理
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class MultiObjectiveConfig:
    """多目标优化配置"""
    num_objectives: int = 2
    reference_point: List[float] = None
    epsilon: float = 0.1
    population_size: int = 100
    num_generations: int = 50


class ParetoFront:
    """Pareto前沿管理"""
    
    def __init__(self):
        self.solutions = []  # [(params, objectives), ...]
    
    def add(self, params: np.ndarray, objectives: np.ndarray) -> bool:
        """
        添加解到Pareto前沿
        
        Returns:
            是否添加成功
        """
        # 检查是否被支配
        to_remove = []
        dominated = False
        
        for i, (p, obj) in enumerate(self.solutions):
            if self._dominates(obj, objectives):
                dominated = True
                break
            if self._dominates(objectives, obj):
                to_remove.append(i)
        
        if not dominated:
            # 移除被支配的解
            for i in reversed(to_remove):
                self.solutions.pop(i)
            
            # 添加新解
            self.solutions.append((params.copy(), objectives.copy()))
            return True
        
        return False
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """检查obj1是否支配obj2"""
        better_in_one = False
        
        for v1, v2 in zip(obj1, obj2):
            if v1 < v2:
                return False
            if v1 > v2:
                better_in_one = True
        
        return better_in_one
    
    def get_solutions(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取所有Pareto解"""
        return self.solutions.copy()
    
    def hypervolume(self, reference_point: np.ndarray) -> float:
        """
        计算超体积指标
        
        衡量Pareto前沿的质量
        """
        if not self.solutions:
            return 0.0
        
        # 简化实现: 使用Lebesgue近似
        volume = 0.0
        
        # 排序
        sorted_solutions = sorted(
            self.solutions,
            key=lambda x: x[1][0]
        )
        
        for i, (params, obj) in enumerate(sorted_solutions):
            # 计算贡献
            ref = reference_point.copy()
            
            for j in range(len(obj)):
                if j == 0:
                    if i < len(sorted_solutions) - 1:
                        ref[j] = sorted_solutions[i + 1][1][j]
                else:
                    ref[j] = min(ref[j], obj[j])
            
            contribution = 1.0
            for j in range(len(obj)):
                contribution *= max(0, ref[j] - obj[j])
            
            volume += contribution
        
        return volume
    
    def get_crowding_distances(self) -> np.ndarray:
        """计算拥挤距离"""
        if len(self.solutions) <= 2:
            return np.ones(len(self.solutions))
        
        num_objectives = len(self.solutions[0][1])
        distances = np.zeros(len(self.solutions))
        
        for m in range(num_objectives):
            # 按目标m排序
            sorted_idx = sorted(
                range(len(self.solutions)),
                key=lambda i: self.solutions[i][1][m]
            )
            
            # 边界点设为无穷
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            # 计算中间点的距离
            obj_range = (
                self.solutions[sorted_idx[-1]][1][m] -
                self.solutions[sorted_idx[0]][1][m]
            )
            
            if obj_range > 0:
                for i in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[i]] += (
                        self.solutions[sorted_idx[i + 1]][1][m] -
                        self.solutions[sorted_idx[i - 1]][1][m]
                    ) / obj_range
        
        return distances


class MultiObjectiveOptimizer:
    """
    多目标优化器
    
    支持多种多目标优化方法
    """
    
    def __init__(
        self,
        objective_funcs: List[Callable],
        param_bounds: List[Tuple[float, float]],
        config: Optional[MultiObjectiveConfig] = None
    ):
        self.objective_funcs = objective_funcs
        self.param_bounds = param_bounds
        self.config = config or MultiObjectiveConfig()
        
        self.pareto_front = ParetoFront()
    
    def optimize(
        self,
        method: str = "nsga2",
        num_iterations: int = None
    ) -> ParetoFront:
        """
        多目标优化
        
        Args:
            method: "nsga2", "weighted_sum", "epsilon_constraint"
        """
        if method == "nsga2":
            return self._nsga2(num_iterations)
        elif method == "weighted_sum":
            return self._weighted_sum(num_iterations)
        elif method == "epsilon_constraint":
            return self._epsilon_constraint(num_iterations)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _nsga2(self, num_iterations: int = None) -> ParetoFront:
        """
        NSGA-II算法
        
        非支配排序遗传算法II
        """
        num_iterations = num_iterations or self.config.num_generations
        pop_size = self.config.population_size
        
        # 初始化种群
        population = self._initialize_population(pop_size)
        
        for generation in range(num_iterations):
            # 评估
            objectives = self._evaluate_population(population)
            
            # 非支配排序
            fronts = self._non_dominated_sort(population, objectives)
            
            # 选择
            offspring = self._tournament_selection(population, objectives, fronts)
            
            # 交叉和变异
            offspring = self._crossover_and_mutate(offspring)
            
            # 合并和选择下一代
            combined = population + offspring
            combined_obj = self._evaluate_population(combined)
            
            population, objectives = self._environmental_selection(
                combined, combined_obj, pop_size
            )
            
            # 更新Pareto前沿
            for params, obj in zip(population, objectives):
                self.pareto_front.add(params, obj)
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Pareto front size = {len(self.pareto_front.solutions)}")
        
        return self.pareto_front
    
    def _initialize_population(self, size: int) -> List[np.ndarray]:
        """初始化种群"""
        population = []
        for _ in range(size):
            params = np.array([
                np.random.uniform(low, high) for low, high in self.param_bounds
            ])
            population.append(params)
        return population
    
    def _evaluate_population(
        self,
        population: List[np.ndarray]
    ) -> List[np.ndarray]:
        """评估种群"""
        objectives = []
        for params in population:
            obj = np.array([f(params) for f in self.objective_funcs])
            objectives.append(obj)
        return objectives
    
    def _non_dominated_sort(
        self,
        population: List[np.ndarray],
        objectives: List[np.ndarray]
    ) -> List[List[int]]:
        """非支配排序"""
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.pareto_front._dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.pareto_front._dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        fronts.pop()  # 移除空前沿
        return fronts
    
    def _tournament_selection(
        self,
        population: List[np.ndarray],
        objectives: List[np.ndarray],
        fronts: List[List[int]],
        tournament_size: int = 2
    ) -> List[np.ndarray]:
        """锦标赛选择"""
        selected = []
        
        # 为每个解计算rank
        ranks = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
        
        # 计算拥挤距离
        temp_pareto = ParetoFront()
        for i, (params, obj) in enumerate(zip(population, objectives)):
            temp_pareto.solutions.append((params, obj))
        
        crowding_distances = temp_pareto.get_crowding_distances()
        
        # 锦标赛
        for _ in range(len(population)):
            contestants = np.random.choice(len(population), tournament_size, replace=False)
            
            winner = contestants[0]
            for c in contestants[1:]:
                # 比较rank
                if ranks[c] < ranks[winner]:
                    winner = c
                elif ranks[c] == ranks[winner]:
                    # 比较拥挤距离
                    if crowding_distances[c] > crowding_distances[winner]:
                        winner = c
            
            selected.append(population[winner].copy())
        
        return selected
    
    def _crossover_and_mutate(
        self,
        parents: List[np.ndarray],
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1
    ) -> List[np.ndarray]:
        """交叉和变异"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            
            # 交叉 (SBX)
            if np.random.random() < crossover_prob:
                child1, child2 = self._sbx_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异 (多项式变异)
            child1 = self._polynomial_mutation(child1, mutation_prob)
            child2 = self._polynomial_mutation(child2, mutation_prob)
            
            offspring.extend([child1, child2])
        
        return offspring
    
    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        eta: float = 20.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """模拟二进制交叉"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    beta = 1.0 + (2.0 * (y1 - self.param_bounds[i][0]) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta + 1.0)
                    
                    rand = np.random.random()
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (self.param_bounds[i][1] - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta + 1.0)
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                    
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    c1 = np.clip(c1, self.param_bounds[i][0], self.param_bounds[i][1])
                    c2 = np.clip(c2, self.param_bounds[i][0], self.param_bounds[i][1])
                    
                    if np.random.random() <= 0.5:
                        child1[i] = c2
                        child2[i] = c1
                    else:
                        child1[i] = c1
                        child2[i] = c2
        
        return child1, child2
    
    def _polynomial_mutation(
        self,
        individual: np.ndarray,
        prob: float,
        eta: float = 20.0
    ) -> np.ndarray:
        """多项式变异"""
        mutant = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < prob:
                delta1 = (individual[i] - self.param_bounds[i][0]) / (
                    self.param_bounds[i][1] - self.param_bounds[i][0]
                )
                delta2 = (self.param_bounds[i][1] - individual[i]) / (
                    self.param_bounds[i][1] - self.param_bounds[i][0]
                )
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                mutant[i] += delta_q * (self.param_bounds[i][1] - self.param_bounds[i][0])
                mutant[i] = np.clip(mutant[i], self.param_bounds[i][0], self.param_bounds[i][1])
        
        return mutant
    
    def _environmental_selection(
        self,
        combined: List[np.ndarray],
        objectives: List[np.ndarray],
        pop_size: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """环境选择"""
        # 非支配排序
        fronts = self._non_dominated_sort(combined, objectives)
        
        new_population = []
        new_objectives = []
        
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                for idx in front:
                    new_population.append(combined[idx])
                    new_objectives.append(objectives[idx])
            else:
                # 需要拥挤距离选择
                remaining = pop_size - len(new_population)
                
                # 计算拥挤距离
                temp_pareto = ParetoFront()
                for idx in front:
                    temp_pareto.solutions.append((combined[idx], objectives[idx]))
                
                distances = temp_pareto.get_crowding_distances()
                
                # 按拥挤距离排序
                sorted_idx = sorted(
                    range(len(front)),
                    key=lambda i: distances[i],
                    reverse=True
                )
                
                for i in range(remaining):
                    idx = front[sorted_idx[i]]
                    new_population.append(combined[idx])
                    new_objectives.append(objectives[idx])
                
                break
        
        return new_population, new_objectives
    
    def _weighted_sum(self, num_iterations: int = None) -> ParetoFront:
        """加权和方法"""
        num_iterations = num_iterations or 50
        
        # 在不同权重组合下优化
        for i in range(num_iterations):
            # 随机权重
            weights = np.random.dirichlet(np.ones(len(self.objective_funcs)))
            
            # 定义加权目标
            def weighted_objective(params):
                obj = np.array([f(params) for f in self.objective_funcs])
                return np.dot(weights, obj)
            
            # 优化 (简化: 随机搜索)
            best_params = None
            best_value = float('-inf')
            
            for _ in range(100):
                params = np.array([
                    np.random.uniform(low, high) for low, high in self.param_bounds
                ])
                value = weighted_objective(params)
                
                if value > best_value:
                    best_value = value
                    best_params = params
            
            # 添加到Pareto前沿
            obj = np.array([f(best_params) for f in self.objective_funcs])
            self.pareto_front.add(best_params, obj)
        
        return self.pareto_front
    
    def _epsilon_constraint(self, num_iterations: int = None) -> ParetoFront:
        """Epsilon约束方法"""
        # 简化实现
        return self._weighted_sum(num_iterations)


def demo():
    """演示多目标优化"""
    print("=" * 60)
    print("Multi-Objective Optimization Demo")
    print("=" * 60)
    
    # 定义测试函数 (ZDT1)
    def f1(x):
        return x[0]
    
    def f2(x):
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        h = 1 - np.sqrt(f1(x) / g)
        return g * h
    
    # 创建优化器
    optimizer = MultiObjectiveOptimizer(
        objective_funcs=[f1, f2],
        param_bounds=[(0, 1)] * 10,
        config=MultiObjectiveConfig(
            population_size=50,
            num_generations=30
        )
    )
    
    # 运行NSGA-II
    print("\nRunning NSGA-II...")
    pareto_front = optimizer.optimize(method="nsga2")
    
    print(f"Pareto front size: {len(pareto_front.solutions)}")
    
    # 计算超体积
    reference = np.array([1.5, 1.5])
    hv = pareto_front.hypervolume(reference)
    print(f"Hypervolume: {hv:.4f}")
    
    # 打印一些解
    print("\nSample Pareto solutions:")
    for i, (params, obj) in enumerate(pareto_front.solutions[:5]):
        print(f"  {i+1}. f1={obj[0]:.3f}, f2={obj[1]:.3f}")
    
    # 运行加权和方法对比
    print("\nRunning Weighted Sum method...")
    optimizer2 = MultiObjectiveOptimizer(
        objective_funcs=[f1, f2],
        param_bounds=[(0, 1)] * 10,
    )
    
    pareto_front2 = optimizer2.optimize(method="weighted_sum", num_iterations=50)
    print(f"Weighted Sum Pareto front size: {len(pareto_front2.solutions)}")
    
    hv2 = pareto_front2.hypervolume(reference)
    print(f"Weighted Sum Hypervolume: {hv2:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
