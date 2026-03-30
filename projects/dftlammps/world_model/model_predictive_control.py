"""
模型预测控制 - Model Predictive Control
========================================

最优策略搜索、实时调整、约束满足的预测控制框架。

This module implements Model Predictive Control (MPC) for materials systems:
- Optimal policy search using learned world models
- Real-time adaptation to changing conditions
- Constraint satisfaction during synthesis
- Multi-objective optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import copy
from collections import deque
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import NonlinearConstraint, Bounds

# 导入世界模型相关类
from .material_world_model import (
    MaterialWorldModel,
    MaterialState,
    MaterialAction,
    ActionType,
    WorldModelConfig
)


class OptimizationMethod(Enum):
    """优化方法枚举"""
    CEM = auto()           # 交叉熵方法
    MPPI = auto()          # 模型预测路径积分
    SHOOTING = auto()      # 直接打靶法
    COLLLOCATION = auto()  # 配点法
    GRADIENT = auto()      # 梯度下降
    GENETIC = auto()       # 遗传算法
    BAYESIAN = auto()      # 贝叶斯优化


class ConstraintType(Enum):
    """约束类型枚举"""
    EQUALITY = auto()      # 等式约束
    INEQUALITY = auto()    # 不等式约束
    BOX = auto()           # 盒约束
    PROBABILISTIC = auto() # 概率约束


@dataclass
class ControlConstraint:
    """
    控制约束
    
    定义优化问题的约束条件
    """
    name: str
    constraint_type: ConstraintType
    
    # 约束函数
    constraint_fn: Optional[Callable[[MaterialState], float]] = None
    
    # 数值约束
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    target_value: Optional[float] = None
    
    # 概率约束参数
    confidence_level: float = 0.95
    
    def evaluate(self, state: MaterialState) -> float:
        """评估约束值"""
        if self.constraint_fn:
            return self.constraint_fn(state)
        return 0.0
    
    def is_satisfied(self, state: MaterialState, tolerance: float = 1e-6) -> bool:
        """检查约束是否满足"""
        value = self.evaluate(state)
        
        if self.constraint_type == ConstraintType.EQUALITY:
            return abs(value - self.target_value) < tolerance
        elif self.constraint_type == ConstraintType.INEQUALITY:
            return self.lower_bound <= value <= self.upper_bound
        elif self.constraint_type == ConstraintType.BOX:
            return self.lower_bound <= value <= self.upper_bound
        
        return True


@dataclass
class MPCConfig:
    """MPC配置"""
    
    # 预测时域
    horizon: int = 20
    
    # 优化参数
    optimization_method: OptimizationMethod = OptimizationMethod.CEM
    num_iterations: int = 100
    num_samples: int = 1000
    elite_fraction: float = 0.1
    
    # 交叉熵方法参数
    cem_alpha: float = 0.25  # 平滑系数
    cem_init_var: float = 1.0
    
    # MPPI参数
    mppi_temperature: float = 1.0
    mppi_lambda: float = 1.0
    
    # 梯度优化参数
    learning_rate: float = 0.01
    max_grad_steps: int = 100
    
    # 控制参数
    control_dim: int = 5
    control_bounds: Tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.array([-1.0] * 5), np.array([1.0] * 5))
    )
    
    # 终端代价权重
    terminal_weight: float = 1.0
    
    # 实时调整
    enable_adaptation: bool = True
    adaptation_rate: float = 0.1
    
    # 约束处理
    constraint_penalty: float = 1000.0
    use_soft_constraints: bool = True


@dataclass
class TrajectoryCost:
    """
    轨迹代价
    
    定义优化目标
    """
    # 阶段代价
    stage_cost_fn: Optional[Callable[[MaterialState, MaterialAction], float]] = None
    
    # 终端代价
    terminal_cost_fn: Optional[Callable[[MaterialState], float]] = None
    
    # 控制代价
    control_cost_fn: Optional[Callable[[MaterialAction], float]] = None
    
    # 平滑代价
    smoothness_weight: float = 0.1
    
    def compute_stage_cost(
        self,
        state: MaterialState,
        action: MaterialAction,
        step: int
    ) -> float:
        """计算阶段代价"""
        cost = 0.0
        
        if self.stage_cost_fn:
            cost += self.stage_cost_fn(state, action)
        
        if self.control_cost_fn:
            cost += self.control_cost_fn(action)
        
        return cost
    
    def compute_terminal_cost(self, state: MaterialState) -> float:
        """计算终端代价"""
        if self.terminal_cost_fn:
            return self.terminal_cost_fn(state)
        return 0.0
    
    def compute_smoothness_cost(self, actions: List[MaterialAction]) -> float:
        """计算平滑代价"""
        if len(actions) < 2:
            return 0.0
        
        cost = 0.0
        for i in range(len(actions) - 1):
            diff = actions[i].magnitude - actions[i+1].magnitude
            cost += diff ** 2
        
        return self.smoothness_weight * cost


class CrossEntropyOptimizer:
    """
    交叉熵方法优化器
    
    基于采样的优化方法，适用于非凸问题
    """
    
    def __init__(
        self,
        horizon: int,
        control_dim: int,
        num_samples: int = 1000,
        elite_fraction: float = 0.1,
        alpha: float = 0.25,
        init_mean: Optional[np.ndarray] = None,
        init_var: Optional[np.ndarray] = None
    ):
        self.horizon = horizon
        self.control_dim = control_dim
        self.num_samples = num_samples
        self.elite_fraction = elite_fraction
        self.alpha = alpha
        
        # 初始化分布参数
        if init_mean is None:
            self.mean = np.zeros((horizon, control_dim))
        else:
            self.mean = init_mean.copy()
        
        if init_var is None:
            self.var = np.ones((horizon, control_dim))
        else:
            self.var = init_var.copy()
        
        self.elite_size = max(1, int(num_samples * elite_fraction))
    
    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        num_iterations: int = 100,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        运行优化
        
        Args:
            cost_fn: 代价函数，输入控制序列，返回标量代价
            num_iterations: 迭代次数
            verbose: 是否打印进度
            
        Returns:
            (最优控制序列, 最优代价)
        """
        best_cost = float('inf')
        best_sequence = None
        
        for iteration in range(num_iterations):
            # 采样控制序列
            samples = self._sample_controls()
            
            # 评估代价
            costs = np.array([cost_fn(sample) for sample in samples])
            
            # 选择精英样本
            elite_indices = np.argsort(costs)[:self.elite_size]
            elite_samples = samples[elite_indices]
            elite_costs = costs[elite_indices]
            
            # 更新分布
            new_mean = np.mean(elite_samples, axis=0)
            new_var = np.var(elite_samples, axis=0) + 1e-6  # 避免零方差
            
            self.mean = self.alpha * new_mean + (1 - self.alpha) * self.mean
            self.var = self.alpha * new_var + (1 - self.alpha) * self.var
            
            # 更新最优
            if elite_costs[0] < best_cost:
                best_cost = elite_costs[0]
                best_sequence = elite_samples[0].copy()
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"CEM Iteration {iteration+1}: cost={best_cost:.4f}")
        
        return best_sequence, best_cost
    
    def _sample_controls(self) -> np.ndarray:
        """采样控制序列"""
        samples = np.random.normal(
            self.mean,
            np.sqrt(self.var),
            size=(self.num_samples, self.horizon, self.control_dim)
        )
        return samples
    
    def reset(self):
        """重置优化器"""
        self.mean = np.zeros((self.horizon, self.control_dim))
        self.var = np.ones((self.horizon, self.control_dim))


class MPPIOptimizer:
    """
    模型预测路径积分优化器
    
    基于信息理论的采样优化方法
    """
    
    def __init__(
        self,
        horizon: int,
        control_dim: int,
        num_samples: int = 1000,
        temperature: float = 1.0,
        lambda_: float = 1.0
    ):
        self.horizon = horizon
        self.control_dim = control_dim
        self.num_samples = num_samples
        self.temperature = temperature
        self.lambda_ = lambda_
        
        # 控制均值
        self.mean = np.zeros((horizon, control_dim))
        self.covariance = np.eye(control_dim) * 0.5
    
    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        num_iterations: int = 50,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        运行MPPI优化
        
        Args:
            cost_fn: 代价函数
            num_iterations: 迭代次数
            verbose: 是否打印进度
            
        Returns:
            (最优控制序列, 最优代价)
        """
        for iteration in range(num_iterations):
            # 采样扰动
            noise = np.random.multivariate_normal(
                np.zeros(self.control_dim),
                self.covariance,
                size=(self.num_samples, self.horizon)
            )
            
            # 生成候选控制序列
            control_samples = self.mean + noise
            
            # 评估代价
            costs = np.array([cost_fn(sample) for sample in control_samples])
            
            # 计算权重
            beta = np.min(costs)
            weights = np.exp(-(costs - beta) / self.temperature)
            weights = weights / np.sum(weights)
            
            # 加权更新
            weighted_noise = np.zeros_like(self.mean)
            for i, noise_sample in enumerate(noise):
                weighted_noise += weights[i] * noise_sample
            
            self.mean += self.lambda_ * weighted_noise
            
            if verbose and (iteration + 1) % 20 == 0:
                print(f"MPPI Iteration {iteration+1}: cost={np.min(costs):.4f}")
        
        final_cost = cost_fn(self.mean)
        return self.mean.copy(), final_cost
    
    def reset(self):
        """重置"""
        self.mean = np.zeros((self.horizon, self.control_dim))


class GradientMPCOptimizer:
    """
    基于梯度的MPC优化器
    
    使用自动微分进行梯度下降优化
    """
    
    def __init__(
        self,
        horizon: int,
        control_dim: int,
        learning_rate: float = 0.01,
        max_steps: int = 100
    ):
        self.horizon = horizon
        self.control_dim = control_dim
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        
        # 可优化参数
        self.controls = torch.nn.Parameter(
            torch.zeros(horizon, control_dim)
        )
    
    def optimize(
        self,
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
        verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        梯度优化
        
        Args:
            cost_fn: 可微代价函数
            verbose: 是否打印进度
            
        Returns:
            (最优控制序列, 最优代价)
        """
        optimizer = optim.Adam([self.controls], lr=self.learning_rate)
        
        best_cost = float('inf')
        best_controls = None
        
        for step in range(self.max_steps):
            optimizer.zero_grad()
            
            cost = cost_fn(self.controls)
            cost.backward()
            
            optimizer.step()
            
            cost_val = cost.item()
            if cost_val < best_cost:
                best_cost = cost_val
                best_controls = self.controls.detach().numpy().copy()
            
            if verbose and (step + 1) % 20 == 0:
                print(f"Gradient Step {step+1}: cost={cost_val:.4f}")
        
        return best_controls if best_controls is not None else self.controls.detach().numpy(), best_cost
    
    def reset(self):
        """重置"""
        self.controls = torch.nn.Parameter(
            torch.zeros(self.horizon, self.control_dim)
        )


class GeneticOptimizer:
    """
    遗传算法优化器
    
    基于进化的全局优化方法
    """
    
    def __init__(
        self,
        horizon: int,
        control_dim: int,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        self.horizon = horizon
        self.control_dim = control_dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population = None
        self.fitness = None
    
    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        num_generations: int = 100,
        verbose: bool = False
    ) -> Tuple[np.ndarray, float]:
        """遗传优化"""
        # 初始化种群
        self.population = np.random.randn(
            self.population_size,
            self.horizon,
            self.control_dim
        )
        
        best_cost = float('inf')
        best_individual = None
        
        for generation in range(num_generations):
            # 评估适应度
            costs = np.array([cost_fn(ind) for ind in self.population])
            self.fitness = -costs  # 最小化代价 = 最大化适应度
            
            # 更新最优
            best_idx = np.argmin(costs)
            if costs[best_idx] < best_cost:
                best_cost = costs[best_idx]
                best_individual = self.population[best_idx].copy()
            
            # 选择
            selected = self._selection()
            
            # 交叉和变异
            offspring = self._crossover_and_mutate(selected)
            
            # 精英保留
            self.population = np.vstack([offspring, best_individual.reshape(1, -1, self.control_dim)])
            
            if verbose and (generation + 1) % 20 == 0:
                print(f"Gen {generation+1}: best_cost={best_cost:.4f}")
        
        return best_individual, best_cost
    
    def _selection(self) -> np.ndarray:
        """锦标赛选择"""
        selected = []
        for _ in range(self.population_size - 1):
            idx = np.random.choice(len(self.population), 3, replace=False)
            winner = idx[np.argmax(self.fitness[idx])]
            selected.append(self.population[winner])
        return np.array(selected)
    
    def _crossover_and_mutate(self, parents: np.ndarray) -> np.ndarray:
        """交叉和变异"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)]
            
            # 交叉
            if np.random.rand() < self.crossover_rate:
                mask = np.random.rand(self.horizon, self.control_dim) > 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            
            # 变异
            mask = np.random.rand(self.horizon, self.control_dim) < self.mutation_rate
            child += mask * np.random.randn(self.horizon, self.control_dim)
            
            offspring.append(child)
        
        return np.array(offspring)
    
    def reset(self):
        """重置"""
        self.population = None
        self.fitness = None


class ConstraintHandler:
    """
    约束处理器
    
    处理各种约束条件的满足
    """
    
    def __init__(self, constraints: List[ControlConstraint]):
        self.constraints = constraints
        self.penalty_weight = 1000.0
    
    def evaluate_constraints(self, state: MaterialState) -> Dict[str, float]:
        """评估所有约束"""
        violations = {}
        for constraint in self.constraints:
            value = constraint.evaluate(state)
            violations[constraint.name] = value
        return violations
    
    def compute_penalty(self, states: List[MaterialState]) -> float:
        """计算约束惩罚"""
        penalty = 0.0
        
        for state in states:
            for constraint in self.constraints:
                value = constraint.evaluate(state)
                
                if constraint.constraint_type == ConstraintType.EQUALITY:
                    if constraint.target_value is not None:
                        penalty += (value - constraint.target_value) ** 2
                
                elif constraint.constraint_type == ConstraintType.INEQUALITY:
                    if value < constraint.lower_bound:
                        penalty += (constraint.lower_bound - value) ** 2
                    if value > constraint.upper_bound:
                        penalty += (value - constraint.upper_bound) ** 2
                
                elif constraint.constraint_type == ConstraintType.BOX:
                    if value < constraint.lower_bound:
                        penalty += (constraint.lower_bound - value) ** 2
                    if value > constraint.upper_bound:
                        penalty += (value - constraint.upper_bound) ** 2
        
        return self.penalty_weight * penalty
    
    def is_feasible(self, states: List[MaterialState], tolerance: float = 1e-3) -> bool:
        """检查轨迹是否可行"""
        for state in states:
            for constraint in self.constraints:
                if not constraint.is_satisfied(state, tolerance):
                    return False
        return True
    
    def project_to_feasible(self, state: MaterialState) -> MaterialState:
        """将状态投影到可行域"""
        # 简化实现：逐步调整直到满足约束
        projected = state.clone()
        max_iterations = 100
        
        for _ in range(max_iterations):
            all_satisfied = True
            for constraint in self.constraints:
                if not constraint.is_satisfied(projected):
                    all_satisfied = False
                    # 简单的梯度投影
                    value = constraint.evaluate(projected)
                    if constraint.constraint_type == ConstraintType.INEQUALITY:
                        if value < constraint.lower_bound:
                            # 调整状态使约束满足
                            projected.temperature = max(projected.temperature, 300)
                        if value > constraint.upper_bound:
                            projected.temperature = min(projected.temperature, 1500)
            
            if all_satisfied:
                break
        
        return projected


class ModelPredictiveController:
    """
    模型预测控制器主类
    
    整合世界模型、优化器和约束处理
    """
    
    def __init__(
        self,
        world_model: MaterialWorldModel,
        config: Optional[MPCConfig] = None
    ):
        self.world_model = world_model
        self.config = config or MPCConfig()
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 约束处理
        self.constraint_handler: Optional[ConstraintHandler] = None
        
        # 轨迹代价
        self.trajectory_cost: Optional[TrajectoryCost] = None
        
        # 控制历史
        self.control_history: deque = deque(maxlen=1000)
        self.state_history: deque = deque(maxlen=1000)
        
        # 自适应参数
        self.adaptation_params = {
            'prediction_error': 0.0,
            'constraint_violation_rate': 0.0
        }
    
    def _create_optimizer(self):
        """创建优化器"""
        method = self.config.optimization_method
        
        if method == OptimizationMethod.CEM:
            return CrossEntropyOptimizer(
                horizon=self.config.horizon,
                control_dim=self.config.control_dim,
                num_samples=self.config.num_samples,
                elite_fraction=self.config.elite_fraction,
                alpha=self.config.cem_alpha
            )
        
        elif method == OptimizationMethod.MPPI:
            return MPPIOptimizer(
                horizon=self.config.horizon,
                control_dim=self.config.control_dim,
                num_samples=self.config.num_samples,
                temperature=self.config.mppi_temperature
            )
        
        elif method == OptimizationMethod.GRADIENT:
            return GradientMPCOptimizer(
                horizon=self.config.horizon,
                control_dim=self.config.control_dim,
                learning_rate=self.config.learning_rate,
                max_steps=self.config.max_grad_steps
            )
        
        elif method == OptimizationMethod.GENETIC:
            return GeneticOptimizer(
                horizon=self.config.horizon,
                control_dim=self.config.control_dim
            )
        
        else:
            return CrossEntropyOptimizer(
                horizon=self.config.horizon,
                control_dim=self.config.control_dim
            )
    
    def set_constraints(self, constraints: List[ControlConstraint]):
        """设置约束"""
        self.constraint_handler = ConstraintHandler(constraints)
    
    def set_cost_function(self, cost: TrajectoryCost):
        """设置代价函数"""
        self.trajectory_cost = cost
    
    def compute_optimal_control(
        self,
        current_state: MaterialState,
        reference_trajectory: Optional[List[MaterialState]] = None
    ) -> Tuple[MaterialAction, Dict[str, Any]]:
        """
        计算最优控制
        
        Args:
            current_state: 当前状态
            reference_trajectory: 参考轨迹 (可选)
            
        Returns:
            (最优动作, 优化信息)
        """
        # 定义代价函数
        def cost_fn(control_sequence: np.ndarray) -> float:
            return self._evaluate_control_sequence(
                current_state,
                control_sequence,
                reference_trajectory
            )
        
        # 运行优化
        if isinstance(self.optimizer, GradientMPCOptimizer):
            # 梯度优化需要可微代价
            def torch_cost_fn(controls: torch.Tensor) -> torch.Tensor:
                # 这里简化处理，实际应该集成PyTorch梯度
                cost_val = cost_fn(controls.detach().numpy())
                return torch.tensor(cost_val, requires_grad=True)
            
            optimal_controls, optimal_cost = self.optimizer.optimize(torch_cost_fn)
        else:
            optimal_controls, optimal_cost = self.optimizer.optimize(
                cost_fn,
                num_iterations=self.config.num_iterations
            )
        
        # 提取第一个控制动作
        first_control = self._vector_to_action(optimal_controls[0])
        
        # 预测未来轨迹
        predicted_trajectory = self._predict_trajectory(
            current_state,
            optimal_controls
        )
        
        info = {
            'optimal_cost': optimal_cost,
            'predicted_trajectory': predicted_trajectory,
            'control_sequence': optimal_controls,
            'constraint_satisfied': True
        }
        
        if self.constraint_handler:
            info['constraint_satisfied'] = self.constraint_handler.is_feasible(
                predicted_trajectory
            )
        
        return first_control, info
    
    def _evaluate_control_sequence(
        self,
        initial_state: MaterialState,
        control_sequence: np.ndarray,
        reference_trajectory: Optional[List[MaterialState]]
    ) -> float:
        """评估控制序列的代价"""
        # 预测轨迹
        predicted_states = self._predict_trajectory(initial_state, control_sequence)
        
        # 转换为动作
        actions = [self._vector_to_action(c) for c in control_sequence]
        
        # 计算代价
        total_cost = 0.0
        
        if self.trajectory_cost:
            # 阶段代价
            for i, (state, action) in enumerate(zip(predicted_states[:-1], actions)):
                total_cost += self.trajectory_cost.compute_stage_cost(state, action, i)
            
            # 终端代价
            total_cost += self.trajectory_cost.compute_terminal_cost(predicted_states[-1])
            
            # 平滑代价
            total_cost += self.trajectory_cost.compute_smoothness_cost(actions)
            
            # 参考轨迹跟踪代价
            if reference_trajectory and len(reference_trajectory) <= len(predicted_states):
                for pred, ref in zip(predicted_states, reference_trajectory):
                    tracking_error = pred.compute_energy_difference(ref)
                    total_cost += 0.1 * tracking_error ** 2
        
        # 约束惩罚
        if self.constraint_handler:
            total_cost += self.constraint_handler.compute_penalty(predicted_states)
        
        return total_cost
    
    def _predict_trajectory(
        self,
        initial_state: MaterialState,
        control_sequence: np.ndarray
    ) -> List[MaterialState]:
        """预测轨迹"""
        states = [initial_state]
        current_state = initial_state
        
        for control in control_sequence:
            action = self._vector_to_action(control)
            next_state, _, _ = self.world_model.predict(current_state, action)
            states.append(next_state)
            current_state = next_state
        
        return states
    
    def _vector_to_action(self, control: np.ndarray) -> MaterialAction:
        """将控制向量转换为动作"""
        # 假设控制向量第一个元素是幅度
        magnitude = control[0] if len(control) > 0 else 0.0
        
        return MaterialAction(
            action_id=f"mpc_action_{id(control)}",
            action_type=ActionType.TEMPERATURE_CHANGE,
            magnitude=float(magnitude),
            duration=100.0
        )
    
    def step(
        self,
        current_state: MaterialState,
        reference_trajectory: Optional[List[MaterialState]] = None
    ) -> Tuple[MaterialAction, Dict[str, Any]]:
        """
        执行单步MPC控制
        
        Args:
            current_state: 当前状态
            reference_trajectory: 参考轨迹
            
        Returns:
            (控制动作, 信息字典)
        """
        # 计算最优控制
        action, info = self.compute_optimal_control(
            current_state,
            reference_trajectory
        )
        
        # 记录历史
        self.control_history.append(action)
        self.state_history.append(current_state)
        
        # 实时调整
        if self.config.enable_adaptation and len(self.state_history) > 1:
            self._adapt_parameters(current_state)
        
        return action, info
    
    def _adapt_parameters(self, current_state: MaterialState):
        """自适应调整参数"""
        if len(self.state_history) < 2:
            return
        
        # 计算预测误差
        prev_predicted = self.state_history[-1]
        actual = current_state
        
        prediction_error = abs(
            prev_predicted.total_energy - actual.total_energy
        )
        
        # 更新自适应参数
        self.adaptation_params['prediction_error'] = (
            0.9 * self.adaptation_params['prediction_error'] +
            0.1 * prediction_error
        )
        
        # 根据误差调整优化参数
        if self.adaptation_params['prediction_error'] > 10.0:
            # 增加采样数以提高鲁棒性
            if isinstance(self.optimizer, CrossEntropyOptimizer):
                self.optimizer.num_samples = min(2000, int(self.optimizer.num_samples * 1.1))
    
    def run_control_loop(
        self,
        initial_state: MaterialState,
        num_steps: int,
        reference_generator: Optional[Callable[[int], MaterialState]] = None
    ) -> Dict[str, Any]:
        """
        运行完整控制循环
        
        Args:
            initial_state: 初始状态
            num_steps: 控制步数
            reference_generator: 参考轨迹生成器
            
        Returns:
            控制结果
        """
        states = [initial_state]
        actions = []
        costs = []
        
        current_state = initial_state
        
        for step in range(num_steps):
            # 生成参考
            reference = None
            if reference_generator:
                reference = reference_generator(step)
            
            # 计算控制
            action, info = self.step(current_state, reference)
            actions.append(action)
            costs.append(info['optimal_cost'])
            
            # 模拟执行 (实际应用中这里会执行真实动作)
            # 这里用世界模型模拟
            next_state, _, _ = self.world_model.predict(current_state, action)
            states.append(next_state)
            current_state = next_state
        
        return {
            'states': states,
            'actions': actions,
            'costs': costs,
            'total_cost': sum(costs),
            'final_state': states[-1]
        }
    
    def get_control_summary(self) -> Dict[str, Any]:
        """获取控制摘要"""
        return {
            'total_steps': len(self.control_history),
            'adaptation_params': self.adaptation_params.copy(),
            'constraint_handler_active': self.constraint_handler is not None,
            'optimization_method': self.config.optimization_method.name,
            'average_control_magnitude': np.mean([
                a.magnitude for a in self.control_history
            ]) if self.control_history else 0.0
        }


class MultiObjectiveMPC:
    """
    多目标MPC
    
    处理多个冲突目标的优化
    """
    
    def __init__(
        self,
        world_model: MaterialWorldModel,
        objectives: List[Callable[[MaterialState], float]],
        config: Optional[MPCConfig] = None
    ):
        self.world_model = world_model
        self.objectives = objectives
        self.config = config or MPCConfig()
        
        # 帕累托前沿权重
        self.weights_history: List[np.ndarray] = []
    
    def optimize_pareto(
        self,
        initial_state: MaterialState,
        num_weight_samples: int = 50
    ) -> List[Dict[str, Any]]:
        """
        帕累托优化
        
        Args:
            initial_state: 初始状态
            num_weight_samples: 权重采样数
            
        Returns:
            帕累托前沿解集
        """
        solutions = []
        
        # 采样不同的权重组合
        num_objectives = len(self.objectives)
        
        for _ in range(num_weight_samples):
            # 随机权重 (归一化)
            weights = np.random.dirichlet(np.ones(num_objectives))
            
            # 创建加权目标函数
            def weighted_objective(state: MaterialState) -> float:
                values = [obj(state) for obj in self.objectives]
                return np.dot(weights, values)
            
            # 创建单目标MPC
            cost = TrajectoryCost(
                terminal_cost_fn=weighted_objective
            )
            
            mpc = ModelPredictiveController(self.world_model, self.config)
            mpc.set_cost_function(cost)
            
            # 优化
            _, info = mpc.compute_optimal_control(initial_state)
            final_state = info['predicted_trajectory'][-1]
            
            # 计算所有目标值
            objective_values = [obj(final_state) for obj in self.objectives]
            
            solutions.append({
                'weights': weights,
                'final_state': final_state,
                'objective_values': objective_values,
                'trajectory': info['predicted_trajectory']
            })
        
        # 提取帕累托前沿
        pareto_solutions = self._extract_pareto_front(solutions)
        
        return pareto_solutions
    
    def _extract_pareto_front(
        self,
        solutions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """提取帕累托前沿"""
        pareto_indices = []
        
        for i, sol_i in enumerate(solutions):
            is_dominated = False
            values_i = np.array(sol_i['objective_values'])
            
            for j, sol_j in enumerate(solutions):
                if i != j:
                    values_j = np.array(sol_j['objective_values'])
                    
                    # 检查j是否支配i (假设最大化)
                    if np.all(values_j >= values_i) and np.any(values_j > values_i):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return [solutions[i] for i in pareto_indices]


class AdaptiveMPC:
    """
    自适应MPC
    
    根据实时反馈调整模型和策略
    """
    
    def __init__(
        self,
        base_world_model: MaterialWorldModel,
        config: Optional[MPCConfig] = None
    ):
        self.base_model = base_world_model
        self.config = config or MPCConfig()
        
        # 在线学习缓冲区
        self.online_buffer: deque = deque(maxlen=1000)
        
        # 自适应模型
        self.adaptive_model = copy.deepcopy(base_world_model)
        
        # 性能跟踪
        self.performance_history: deque = deque(maxlen=100)
    
    def update_model(
        self,
        state: MaterialState,
        action: MaterialAction,
        next_state: MaterialState,
        reward: float
    ):
        """在线更新模型"""
        from .material_world_model import Transition
        
        transition = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward
        )
        
        self.online_buffer.append(transition)
        
        # 定期微调模型
        if len(self.online_buffer) >= 100:
            self._finetune_model()
    
    def _finetune_model(self):
        """微调世界模型"""
        # 使用在线数据进行短时训练
        transitions = list(self.online_buffer)
        
        # 降低学习率进行微调
        original_lr = self.adaptive_model.config.learning_rate
        self.adaptive_model.config.learning_rate *= 0.1
        self.adaptive_model.config.num_epochs = 5  # 少轮数
        
        self.adaptive_model.train(transitions, verbose=False)
        
        # 恢复原始学习率
        self.adaptive_model.config.learning_rate = original_lr
        
        # 清空缓冲区
        self.online_buffer.clear()
    
    def compute_control(self, current_state: MaterialState) -> Tuple[MaterialAction, Dict]:
        """计算控制动作"""
        # 使用自适应模型
        mpc = ModelPredictiveController(self.adaptive_model, self.config)
        return mpc.compute_optimal_control(current_state)


# 应用案例
class SynthesisPathPlanner:
    """
    合成路径规划器
    
    规划最优材料合成路径
    """
    
    def __init__(self, mpc: ModelPredictiveController):
        self.mpc = mpc
    
    def plan_path(
        self,
        initial_material: MaterialState,
        target_properties: Dict[str, Tuple[float, float]],
        max_steps: int = 50
    ) -> Dict[str, Any]:
        """
        规划合成路径
        
        Args:
            initial_material: 初始材料
            target_properties: 目标属性范围
            max_steps: 最大步数
            
        Returns:
            合成路径规划结果
        """
        # 创建目标代价函数
        def terminal_cost(state: MaterialState) -> float:
            cost = 0.0
            for prop, (min_val, max_val) in target_properties.items():
                current_val = getattr(state, prop, 0)
                target = (min_val + max_val) / 2
                cost += (current_val - target) ** 2
            return cost
        
        cost = TrajectoryCost(terminal_cost_fn=terminal_cost)
        self.mpc.set_cost_function(cost)
        
        # 运行控制循环
        result = self.mpc.run_control_loop(initial_material, max_steps)
        
        # 评估最终状态
        final_state = result['final_state']
        property_achievement = {}
        
        for prop, (min_val, max_val) in target_properties.items():
            current_val = getattr(final_state, prop, 0)
            achieved = min_val <= current_val <= max_val
            property_achievement[prop] = {
                'target_range': (min_val, max_val),
                'achieved_value': current_val,
                'achieved': achieved
            }
        
        return {
            'synthesis_path': result,
            'property_achievement': property_achievement,
            'success_rate': sum(1 for p in property_achievement.values() if p['achieved']) / len(target_properties),
            'recommended_steps': [
                {
                    'step': i,
                    'action_type': a.action_type.name,
                    'magnitude': a.magnitude,
                    'predicted_state': {
                        'temperature': s.temperature,
                        'energy': s.total_energy
                    }
                }
                for i, (a, s) in enumerate(zip(result['actions'], result['states'][1:]))
            ]
        }


class RealTimeSynthesisController:
    """
    实时合成控制器
    
    用于实际实验的实时控制
    """
    
    def __init__(self, mpc: ModelPredictiveController):
        self.mpc = mpc
        self.measurement_buffer: deque = deque(maxlen=10)
        self.control_active = False
    
    def start_control(self):
        """启动控制"""
        self.control_active = True
    
    def stop_control(self):
        """停止控制"""
        self.control_active = False
    
    def process_measurement(self, measurement: Dict[str, float]) -> Optional[MaterialAction]:
        """
        处理测量数据并返回控制动作
        
        Args:
            measurement: 测量数据字典
            
        Returns:
            控制动作或None
        """
        if not self.control_active:
            return None
        
        # 更新测量缓冲区
        self.measurement_buffer.append(measurement)
        
        # 构建当前状态
        current_state = MaterialState(
            state_id=f"measured_{len(self.measurement_buffer)}",
            temperature=measurement.get('temperature', 300),
            pressure=measurement.get('pressure', 1.0),
            total_energy=measurement.get('energy', 0)
        )
        
        # 计算控制
        action, info = self.mpc.step(current_state)
        
        return action


if __name__ == "__main__":
    print("Testing Model Predictive Control...")
    
    # 创建测试世界模型
    from .material_world_model import (
        MaterialWorldModel,
        WorldModelConfig,
        create_synthetic_transitions
    )
    
    config = WorldModelConfig(state_dim=20, action_dim=5, num_epochs=5)
    world_model = MaterialWorldModel(config)
    
    # 训练
    transitions = create_synthetic_transitions(200)
    world_model.train(transitions, verbose=False)
    
    # 创建MPC
    mpc_config = MPCConfig(
        horizon=10,
        control_dim=5,
        optimization_method=OptimizationMethod.CEM,
        num_iterations=50,
        num_samples=200
    )
    
    mpc = ModelPredictiveController(world_model, mpc_config)
    
    # 设置代价函数
    def terminal_cost(state: MaterialState) -> float:
        return (state.temperature - 800) ** 2 + state.total_energy ** 2
    
    cost = TrajectoryCost(terminal_cost_fn=terminal_cost)
    mpc.set_cost_function(cost)
    
    # 测试单步控制
    test_state = transitions[0].state
    action, info = mpc.compute_optimal_control(test_state)
    
    print(f"\nMPC Test:")
    print(f"  Optimal action magnitude: {action.magnitude:.4f}")
    print(f"  Optimal cost: {info['optimal_cost']:.4f}")
    print(f"  Constraint satisfied: {info['constraint_satisfied']}")
    
    # 测试控制循环
    result = mpc.run_control_loop(test_state, num_steps=5)
    
    print(f"\nControl Loop Test:")
    print(f"  Total cost: {result['total_cost']:.4f}")
    print(f"  Final temperature: {result['final_state'].temperature:.2f}")
    
    # 测试路径规划
    planner = SynthesisPathPlanner(mpc)
    
    path_result = planner.plan_path(
        test_state,
        target_properties={'temperature': (700, 900)},
        max_steps=10
    )
    
    print(f"\nPath Planning Test:")
    print(f"  Success rate: {path_result['success_rate']:.2%}")
    print(f"  Number of steps: {len(path_result['recommended_steps'])}")
    
    print("\nAll MPC tests passed!")
