"""
机理模型模块 - Mechanistic Models for Materials

本模块实现物理约束神经网络、符号回归和方程发现：
- 物理约束神经网络 (PINN)
- 符号回归 (Symbolic Regression)
- 方程发现 (Equation Discovery)
- 物理信息嵌入

作者: Causal AI Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
import re
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class PhysicalConstraint:
    """物理约束定义"""
    name: str
    constraint_type: str  # 'pde', 'boundary', 'symmetry', 'conservation'
    equation: Callable  # 约束方程函数
    weight: float = 1.0
    domain: Optional[Dict] = None  # 约束域
    
    def evaluate(self, *args, **kwargs) -> float:
        """评估约束违反程度"""
        return self.equation(*args, **kwargs)


@dataclass
class DiscoveredEquation:
    """发现的方程"""
    expression: str
    variables: List[str]
    parameters: Dict[str, float]
    complexity: int
    fitness: float
    r2_score: float
    symbolic_form: Any = None
    
    def __str__(self):
        return f"{self.expression} (R²={self.r2_score:.4f}, complexity={self.complexity})"
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用发现的方程预测"""
        # 创建局部命名空间
        local_vars = {var: X[:, i] for i, var in enumerate(self.variables)}
        local_vars.update(self.parameters)
        
        # 安全评估
        return eval(self.expression, {"np": np, **local_vars})


class PhysicsInformedLayer(nn.Module if HAS_TORCH else object):
    """
    物理信息层 - 将物理约束嵌入神经网络
    """
    
    def __init__(self, in_features: int, out_features: int,
                 physics_constraint: PhysicalConstraint = None):
        super().__init__() if HAS_TORCH else None
        if not HAS_TORCH:
            return
        
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
        self.constraint = physics_constraint
        
    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        
        # 应用物理约束（如果存在）
        if self.constraint is not None:
            # 约束可以通过正则化项在损失函数中应用
            pass
        
        return out


class PhysicsInformedNN(nn.Module if HAS_TORCH else object):
    """
    物理约束神经网络 (Physics-Informed Neural Network)
    
    将物理定律作为软约束嵌入神经网络训练
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = [64, 64, 64],
                 physics_constraints: List[PhysicalConstraint] = None,
                 activation: str = 'tanh'):
        """
        初始化PINN
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度
            physics_constraints: 物理约束列表
            activation: 激活函数类型
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for PhysicsInformedNN")
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_constraints = physics_constraints or []
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播"""
        return self.network(x)
    
    def compute_derivatives(self, x: torch.Tensor, 
                           variable_idx: int = 0,
                           order: int = 1) -> torch.Tensor:
        """
        计算自动微分导数
        
        Args:
            x: 输入张量
            variable_idx: 对哪个输入变量求导
            order: 导数阶数
            
        Returns:
            导数张量
        """
        x_copy = x.clone().requires_grad_(True)
        y = self.forward(x_copy)
        
        if order == 1:
            dy_dx = torch.autograd.grad(
                y.sum(), x_copy, create_graph=True
            )[0]
            return dy_dx[:, variable_idx:variable_idx+1]
        
        elif order == 2:
            dy_dx = torch.autograd.grad(
                y.sum(), x_copy, create_graph=True
            )[0]
            d2y_dx2 = torch.autograd.grad(
                dy_dx[:, variable_idx].sum(), x_copy, create_graph=True
            )[0]
            return d2y_dx2[:, variable_idx:variable_idx+1]
        
        else:
            raise ValueError("Only 1st and 2nd order derivatives supported")
    
    def physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算物理约束损失
        
        Args:
            x: 输入数据点（通常是配点）
            
        Returns:
            物理损失
        """
        if not self.physics_constraints:
            return torch.tensor(0.0)
        
        total_physics_loss = 0.0
        
        for constraint in self.physics_constraints:
            if constraint.constraint_type == 'pde':
                # 计算PDE残差
                residual = self._compute_pde_residual(x, constraint)
                total_physics_loss += constraint.weight * torch.mean(residual ** 2)
            
            elif constraint.constraint_type == 'boundary':
                # 边界条件
                residual = self._compute_boundary_residual(x, constraint)
                total_physics_loss += constraint.weight * torch.mean(residual ** 2)
            
            elif constraint.constraint_type == 'conservation':
                # 守恒定律
                residual = self._compute_conservation_residual(x, constraint)
                total_physics_loss += constraint.weight * torch.mean(residual ** 2)
        
        return total_physics_loss
    
    def _compute_pde_residual(self, x: torch.Tensor, 
                              constraint: PhysicalConstraint) -> torch.Tensor:
        """计算PDE残差"""
        # 获取预测
        u = self.forward(x)
        
        # 计算导数
        u_x = self.compute_derivatives(x, variable_idx=0, order=1)
        u_xx = self.compute_derivatives(x, variable_idx=0, order=2)
        
        if x.shape[1] > 1:
            u_t = self.compute_derivatives(x, variable_idx=1, order=1)
        else:
            u_t = torch.zeros_like(u)
        
        # 评估约束方程
        residual = constraint.equation(u, u_x, u_xx, u_t, x)
        return residual
    
    def _compute_boundary_residual(self, x: torch.Tensor,
                                   constraint: PhysicalConstraint) -> torch.Tensor:
        """计算边界残差"""
        u = self.forward(x)
        return constraint.equation(u, x)
    
    def _compute_conservation_residual(self, x: torch.Tensor,
                                       constraint: PhysicalConstraint) -> torch.Tensor:
        """计算守恒残差"""
        u = self.forward(x)
        u_x = self.compute_derivatives(x, variable_idx=0, order=1)
        return constraint.equation(u, u_x, x)


class PINNTrainer:
    """
    PINN训练器
    
    管理数据损失和物理损失的联合优化
    """
    
    def __init__(self, model: PhysicsInformedNN,
                 physics_weight: float = 1.0,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        """
        初始化训练器
        
        Args:
            model: PINN模型
            physics_weight: 物理损失权重
            learning_rate: 学习率
            device: 计算设备
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required")
        
        self.model = model.to(device)
        self.physics_weight = physics_weight
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.5
        )
        self.history = {'data_loss': [], 'physics_loss': [], 'total_loss': []}
        
    def train(self, X_data: np.ndarray, y_data: np.ndarray,
              X_physics: np.ndarray = None,
              epochs: int = 1000,
              batch_size: int = 32,
              verbose: bool = True) -> Dict:
        """
        训练PINN
        
        Args:
            X_data: 观测数据输入
            y_data: 观测数据输出
            X_physics: 物理配点
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印进度
            
        Returns:
            训练历史
        """
        # 转换数据
        X_data_t = torch.FloatTensor(X_data).to(self.device)
        y_data_t = torch.FloatTensor(y_data).to(self.device)
        
        if X_physics is not None:
            X_physics_t = torch.FloatTensor(X_physics).to(self.device)
        else:
            # 如果没有提供物理配点，使用数据点
            X_physics_t = X_data_t
        
        # 数据加载器
        dataset = TensorDataset(X_data_t, y_data_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_data_loss = 0.0
            epoch_physics_loss = 0.0
            
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                
                # 数据损失
                y_pred = self.model(X_batch)
                data_loss = torch.mean((y_pred - y_batch) ** 2)
                
                # 物理损失
                physics_loss = self.model.physics_loss(X_physics_t)
                
                # 总损失
                total_loss = data_loss + self.physics_weight * physics_loss
                
                # 反向传播
                total_loss.backward()
                self.optimizer.step()
                
                epoch_data_loss += data_loss.item()
                epoch_physics_loss += physics_loss.item()
            
            # 记录历史
            avg_data_loss = epoch_data_loss / len(dataloader)
            avg_physics_loss = epoch_physics_loss / len(dataloader)
            avg_total = avg_data_loss + self.physics_weight * avg_physics_loss
            
            self.history['data_loss'].append(avg_data_loss)
            self.history['physics_loss'].append(avg_physics_loss)
            self.history['total_loss'].append(avg_total)
            
            # 学习率调度
            self.scheduler.step(avg_total)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Data Loss = {avg_data_loss:.6f}, "
                      f"Physics Loss = {avg_physics_loss:.6f}, "
                      f"Total = {avg_total:.6f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            y_pred = self.model(X_t).cpu().numpy()
        return y_pred


class ExpressionNode:
    """
    表达式树节点
    
    用于符号回归的表达式表示
    """
    
    # 允许的运算符
    UNARY_OPS = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'neg']
    BINARY_OPS = ['+', '-', '*', '/', 'pow']
    
    def __init__(self, value: str, children: List['ExpressionNode'] = None):
        self.value = value
        self.children = children or []
        self.depth = 0
        
    def __str__(self):
        if not self.children:
            return str(self.value)
        
        if self.value in self.UNARY_OPS:
            return f"{self.value}({self.children[0]})"
        elif self.value in self.BINARY_OPS:
            return f"({self.children[0]} {self.value} {self.children[1]})"
        else:
            return str(self.value)
    
    def evaluate(self, variables: Dict[str, np.ndarray],
                 parameters: Dict[str, float] = None) -> np.ndarray:
        """评估表达式"""
        parameters = parameters or {}
        
        # 叶节点
        if not self.children:
            if self.value in variables:
                return variables[self.value]
            elif self.value in parameters:
                return np.full_like(list(variables.values())[0], parameters[self.value])
            else:
                try:
                    return np.full_like(list(variables.values())[0], float(self.value))
                except:
                    return np.zeros_like(list(variables.values())[0])
        
        # 一元运算符
        if self.value in self.UNARY_OPS:
            operand = self.children[0].evaluate(variables, parameters)
            if self.value == 'sin':
                return np.sin(operand)
            elif self.value == 'cos':
                return np.cos(operand)
            elif self.value == 'exp':
                return np.exp(np.clip(operand, -700, 700))
            elif self.value == 'log':
                return np.log(np.abs(operand) + 1e-10)
            elif self.value == 'sqrt':
                return np.sqrt(np.abs(operand))
            elif self.value == 'abs':
                return np.abs(operand)
            elif self.value == 'neg':
                return -operand
        
        # 二元运算符
        elif self.value in self.BINARY_OPS:
            left = self.children[0].evaluate(variables, parameters)
            right = self.children[1].evaluate(variables, parameters)
            
            if self.value == '+':
                return left + right
            elif self.value == '-':
                return left - right
            elif self.value == '*':
                return left * right
            elif self.value == '/':
                return left / (right + 1e-10)
            elif self.value == 'pow':
                return np.power(np.abs(left) + 1e-10, right)
        
        return np.zeros_like(list(variables.values())[0])
    
    def complexity(self) -> int:
        """计算表达式复杂度"""
        if not self.children:
            return 1
        return 1 + sum(child.complexity() for child in self.children)
    
    def copy(self) -> 'ExpressionNode':
        """复制节点"""
        new_children = [child.copy() for child in self.children]
        return ExpressionNode(self.value, new_children)
    
    def to_sympy(self):
        """转换为SymPy表达式"""
        try:
            import sympy as sp
            
            if not self.children:
                try:
                    return float(self.value)
                except:
                    return sp.Symbol(self.value)
            
            if self.value in self.UNARY_OPS:
                operand = self.children[0].to_sympy()
                if self.value == 'sin':
                    return sp.sin(operand)
                elif self.value == 'cos':
                    return sp.cos(operand)
                elif self.value == 'exp':
                    return sp.exp(operand)
                elif self.value == 'log':
                    return sp.log(operand)
                elif self.value == 'sqrt':
                    return sp.sqrt(operand)
                elif self.value == 'abs':
                    return sp.Abs(operand)
                elif self.value == 'neg':
                    return -operand
            
            elif self.value in self.BINARY_OPS:
                left = self.children[0].to_sympy()
                right = self.children[1].to_sympy()
                
                if self.value == '+':
                    return left + right
                elif self.value == '-':
                    return left - right
                elif self.value == '*':
                    return left * right
                elif self.value == '/':
                    return left / right
                elif self.value == 'pow':
                    return left ** right
            
            return sp.Symbol(str(self.value))
        except ImportError:
            return None


class SymbolicRegression:
    """
    符号回归
    
    使用遗传编程发现符号方程
    """
    
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 max_depth: int = 5,
                 tournament_size: int = 5,
                 parsimony_coefficient: float = 0.01):
        """
        初始化符号回归
        
        Args:
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
            max_depth: 最大表达式深度
            tournament_size: 锦标赛选择大小
            parsimony_coefficient: 简洁性惩罚系数
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient
        
        self.population: List[ExpressionNode] = []
        self.fitness_scores: List[float] = []
        self.best_individual: Optional[ExpressionNode] = None
        self.best_fitness: float = -np.inf
        self.variables: List[str] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray,
            variable_names: List[str] = None) -> DiscoveredEquation:
        """
        拟合符号回归
        
        Args:
            X: 特征数据
            y: 目标值
            variable_names: 变量名称
            
        Returns:
            发现的最佳方程
        """
        self.variables = variable_names or [f"x{i}" for i in range(X.shape[1])]
        
        # 初始化种群
        self._initialize_population()
        
        # 进化
        for generation in range(self.generations):
            # 评估适应度
            self._evaluate_fitness(X, y)
            
            # 选择
            selected = self._selection()
            
            # 交叉
            offspring = self._crossover(selected)
            
            # 变异
            offspring = self._mutation(offspring)
            
            # 更新种群
            self.population = offspring
            
            # 打印进度
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation+1}/{self.generations}: "
                      f"Best Fitness = {self.best_fitness:.6f}, "
                      f"Best Expression = {self.best_individual}")
        
        # 返回最佳方程
        return self._create_equation(X, y)
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        
        for _ in range(self.population_size):
            tree = self._grow_tree(depth=0, max_depth=self.max_depth)
            self.population.append(tree)
    
    def _grow_tree(self, depth: int, max_depth: int,
                   method: str = 'mixed') -> ExpressionNode:
        """生长表达式树"""
        # 决定是否终止
        if depth >= max_depth:
            return self._create_terminal()
        
        if method == 'grow' and np.random.random() < 0.5:
            return self._create_terminal()
        
        # 创建操作符节点
        if np.random.random() < 0.5 and depth < max_depth - 1:
            # 一元操作
            op = np.random.choice(ExpressionNode.UNARY_OPS)
            child = self._grow_tree(depth + 1, max_depth, method)
            return ExpressionNode(op, [child])
        else:
            # 二元操作
            op = np.random.choice(ExpressionNode.BINARY_OPS)
            left = self._grow_tree(depth + 1, max_depth, method)
            right = self._grow_tree(depth + 1, max_depth, method)
            return ExpressionNode(op, [left, right])
    
    def _create_terminal(self) -> ExpressionNode:
        """创建叶节点"""
        # 变量或常数
        if np.random.random() < 0.7:
            # 变量
            var = np.random.choice(self.variables)
            return ExpressionNode(var)
        else:
            # 常数
            const = np.random.uniform(-5, 5)
            return ExpressionNode(str(const))
    
    def _evaluate_fitness(self, X: np.ndarray, y: np.ndarray):
        """评估适应度"""
        self.fitness_scores = []
        
        variables = {name: X[:, i] for i, name in enumerate(self.variables)}
        
        for individual in self.population:
            try:
                # 预测
                y_pred = individual.evaluate(variables)
                
                # 计算MSE
                mse = np.mean((y - y_pred) ** 2)
                
                # R²分数
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                ss_res = np.sum((y - y_pred) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # 适应度（考虑简洁性）
                complexity_penalty = self.parsimony_coefficient * individual.complexity()
                fitness = -mse - complexity_penalty
                
                self.fitness_scores.append(fitness)
                
                # 更新最佳
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual.copy()
            except:
                self.fitness_scores.append(-np.inf)
    
    def _selection(self) -> List[ExpressionNode]:
        """锦标赛选择"""
        selected = []
        
        for _ in range(self.population_size):
            # 随机选择tournament_size个个体
            tournament_idx = np.random.choice(
                len(self.population), 
                size=self.tournament_size,
                replace=False
            )
            tournament_fitness = [self.fitness_scores[i] for i in tournament_idx]
            
            # 选择最佳
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, population: List[ExpressionNode]) -> List[ExpressionNode]:
        """交叉操作"""
        offspring = []
        
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % len(population)]
            
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._subtree_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return offspring[:self.population_size]
    
    def _subtree_crossover(self, parent1: ExpressionNode,
                          parent2: ExpressionNode) -> Tuple[ExpressionNode, ExpressionNode]:
        """子树交叉"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 随机选择交叉点
        nodes1 = self._get_all_nodes(child1)
        nodes2 = self._get_all_nodes(child2)
        
        if nodes1 and nodes2:
            node1 = np.random.choice(nodes1)
            node2 = np.random.choice(nodes2)
            
            # 交换子树
            node1.value, node2.value = node2.value, node1.value
            node1.children, node2.children = node2.children, node1.children
        
        return child1, child2
    
    def _get_all_nodes(self, root: ExpressionNode) -> List[ExpressionNode]:
        """获取所有节点"""
        nodes = [root]
        for child in root.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _mutation(self, population: List[ExpressionNode]) -> List[ExpressionNode]:
        """变异操作"""
        mutated = []
        
        for individual in population:
            if np.random.random() < self.mutation_rate:
                mutated.append(self._subtree_mutation(individual))
            else:
                mutated.append(individual)
        
        return mutated
    
    def _subtree_mutation(self, individual: ExpressionNode) -> ExpressionNode:
        """子树变异"""
        new_tree = individual.copy()
        nodes = self._get_all_nodes(new_tree)
        
        if nodes:
            node = np.random.choice(nodes)
            # 替换为新子树
            new_subtree = self._grow_tree(depth=0, max_depth=self.max_depth // 2)
            node.value = new_subtree.value
            node.children = new_subtree.children
        
        return new_tree
    
    def _create_equation(self, X: np.ndarray, y: np.ndarray) -> DiscoveredEquation:
        """创建发现的方程对象"""
        if self.best_individual is None:
            return None
        
        # 优化常数参数
        variables = {name: X[:, i] for i, name in enumerate(self.variables)}
        parameters = self._optimize_constants(self.best_individual, variables, y)
        
        # 评估
        y_pred = self.best_individual.evaluate(variables, parameters)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return DiscoveredEquation(
            expression=str(self.best_individual),
            variables=self.variables,
            parameters=parameters,
            complexity=self.best_individual.complexity(),
            fitness=self.best_fitness,
            r2_score=r2,
            symbolic_form=self.best_individual.to_sympy()
        )
    
    def _optimize_constants(self, tree: ExpressionNode,
                           variables: Dict[str, np.ndarray],
                           y: np.ndarray) -> Dict[str, float]:
        """优化方程中的常数参数"""
        # 识别所有常数节点
        constants = {}
        self._find_constants(tree, constants)
        
        if not constants:
            return {}
        
        # 优化常数值
        def objective(params):
            const_dict = {k: v for k, v in zip(constants.keys(), params)}
            y_pred = tree.evaluate(variables, const_dict)
            return np.mean((y - y_pred) ** 2)
        
        # 使用简单优化
        x0 = [constants[k] for k in constants.keys()]
        try:
            result = minimize(objective, x0, method='Nelder-Mead')
            return {k: v for k, v in zip(constants.keys(), result.x)}
        except:
            return constants
    
    def _find_constants(self, node: ExpressionNode, 
                       constants: Dict[str, float], counter: List[int] = None):
        """找到所有常数节点"""
        if counter is None:
            counter = [0]
        
        if not node.children:
            try:
                val = float(node.value)
                const_name = f"c{counter[0]}"
                constants[const_name] = val
                node.value = const_name
                counter[0] += 1
            except:
                pass
        else:
            for child in node.children:
                self._find_constants(child, constants, counter)


class EquationDiscovery:
    """
    方程发现
    
    从数据中发现物理方程的系统方法
    """
    
    def __init__(self,
                 library_size: int = 50,
                 sparsity_threshold: float = 0.1,
                 alpha: float = 0.01):
        """
        初始化方程发现
        
        Args:
            library_size: 候选函数库大小
            sparsity_threshold: 稀疏性阈值
            alpha: 正则化系数
        """
        self.library_size = library_size
        self.sparsity_threshold = sparsity_threshold
        self.alpha = alpha
        
        self.library: List[Callable] = []
        self.coefficients: np.ndarray = None
        self.discovered_terms: List[Tuple[str, float]] = []
        
    def build_library(self, X: np.ndarray, derivatives: np.ndarray = None,
                     variable_names: List[str] = None) -> np.ndarray:
        """
        构建候选函数库
        
        Args:
            X: 输入数据
            derivatives: 导数数据
            variable_names: 变量名称
            
        Returns:
            函数库矩阵 Theta
        """
        n_samples = X.shape[0]
        n_vars = X.shape[1]
        
        self.variables = variable_names or [f"x{i}" for i in range(n_vars)]
        
        library_functions = []
        self.library_names = []
        
        # 1. 常数项
        library_functions.append(np.ones(n_samples))
        self.library_names.append("1")
        
        # 2. 线性项
        for i in range(n_vars):
            library_functions.append(X[:, i])
            self.library_names.append(self.variables[i])
        
        # 3. 多项式项（二次）
        for i in range(n_vars):
            for j in range(i, n_vars):
                library_functions.append(X[:, i] * X[:, j])
                self.library_names.append(f"{self.variables[i]}*{self.variables[j]}")
        
        # 4. 三角函数
        for i in range(n_vars):
            library_functions.append(np.sin(X[:, i]))
            self.library_names.append(f"sin({self.variables[i]})")
            library_functions.append(np.cos(X[:, i]))
            self.library_names.append(f"cos({self.variables[i]})")
        
        # 5. 指数和对数
        for i in range(n_vars):
            library_functions.append(np.exp(X[:, i]))
            self.library_names.append(f"exp({self.variables[i]})")
            library_functions.append(np.log(np.abs(X[:, i]) + 1e-10))
            self.library_names.append(f"log(|{self.variables[i]}|)")
        
        # 6. 导数项（如果提供）
        if derivatives is not None:
            for i in range(derivatives.shape[1]):
                library_functions.append(derivatives[:, i])
                self.library_names.append(f"d/dt({self.variables[i]})")
        
        # 截断到指定大小
        Theta = np.column_stack(library_functions[:self.library_size])
        self.library_names = self.library_names[:self.library_size]
        
        return Theta
    
    def discover(self, X: np.ndarray, y: np.ndarray,
                method: str = "sparse_regression") -> DiscoveredEquation:
        """
        发现方程
        
        Args:
            X: 输入数据
            y: 目标值（通常是导数）
            method: 发现方法
            
        Returns:
            发现的方程
        """
        # 构建函数库
        Theta = self.build_library(X)
        
        if method == "sparse_regression":
            coefficients = self._sparse_regression(Theta, y)
        elif method == "lasso":
            coefficients = self._lasso_regression(Theta, y)
        elif method == "ridge":
            coefficients = self._ridge_regression(Theta, y)
        else:
            coefficients = self._least_squares(Theta, y)
        
        self.coefficients = coefficients
        
        # 提取非零项
        self.discovered_terms = []
        for i, coef in enumerate(coefficients):
            if abs(coef) > self.sparsity_threshold:
                self.discovered_terms.append((self.library_names[i], coef))
        
        # 构建方程表达式
        expression = self._build_expression()
        
        # 评估
        y_pred = Theta @ coefficients
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        return DiscoveredEquation(
            expression=expression,
            variables=self.variables,
            parameters={t: c for t, c in self.discovered_terms},
            complexity=len(self.discovered_terms),
            fitness=-np.mean((y - y_pred) ** 2),
            r2_score=r2
        )
    
    def _sparse_regression(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """稀疏回归 (Sequentially Thresholded Least Squares)"""
        # 初始最小二乘
        coefficients = np.linalg.lstsq(Theta, y, rcond=None)[0]
        
        # 迭代阈值化
        for _ in range(10):
            # 阈值化小系数
            small_coefs = np.abs(coefficients) < self.sparsity_threshold
            coefficients[small_coefs] = 0
            
            # 重新拟合非零系数
            for i in range(len(coefficients)):
                if coefficients[i] != 0:
                    # 固定其他系数，重新估计
                    residual = y - Theta @ coefficients + Theta[:, i] * coefficients[i]
                    coefficients[i] = np.linalg.lstsq(
                        Theta[:, i:i+1], residual, rcond=None
                    )[0][0]
        
        return coefficients
    
    def _lasso_regression(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """LASSO回归"""
        try:
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=self.alpha, max_iter=10000)
            model.fit(Theta, y)
            return model.coef_
        except ImportError:
            return self._least_squares(Theta, y)
    
    def _ridge_regression(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """岭回归"""
        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=self.alpha)
            model.fit(Theta, y)
            return model.coef_
        except ImportError:
            return self._least_squares(Theta, y)
    
    def _least_squares(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        """最小二乘回归"""
        return np.linalg.lstsq(Theta, y, rcond=None)[0]
    
    def _build_expression(self) -> str:
        """构建方程表达式"""
        terms = []
        for name, coef in self.discovered_terms:
            if abs(coef) > 1e-6:
                sign = "+" if coef > 0 else "-"
                terms.append(f"{sign} {abs(coef):.4f}*{name}")
        
        return " ".join(terms) if terms else "0"
    
    def discover_dynamical_system(self, X: np.ndarray, dt: float = 0.01,
                                  variable_names: List[str] = None) -> List[DiscoveredEquation]:
        """
        发现动力系统方程组
        
        Args:
            X: 状态变量时间序列 [n_timesteps, n_variables]
            dt: 时间步长
            variable_names: 变量名称
            
        Returns:
            每个变量的方程列表
        """
        n_vars = X.shape[1]
        self.variables = variable_names or [f"x{i}" for i in range(n_vars)]
        
        # 数值微分
        dX = np.gradient(X, dt, axis=0)
        
        equations = []
        for i in range(n_vars):
            # 为每个变量发现方程
            equation = self.discover(X, dX[:, i])
            equation.expression = f"d{self.variables[i]}/dt = {equation.expression}"
            equations.append(equation)
        
        return equations


class MechanisticModelPipeline:
    """
    机理模型管道
    
    整合物理约束神经网络、符号回归和方程发现的完整流程
    """
    
    def __init__(self):
        self.pinn: Optional[PhysicsInformedNN] = None
        self.pinn_trainer: Optional[PINNTrainer] = None
        self.symbolic_model: Optional[SymbolicRegression] = None
        self.equation_discovery: Optional[EquationDiscovery] = None
        
    def fit_pinn(self, X_data: np.ndarray, y_data: np.ndarray,
                physics_constraints: List[PhysicalConstraint],
                X_physics: np.ndarray = None,
                hidden_dims: List[int] = [64, 64, 64],
                epochs: int = 1000,
                physics_weight: float = 1.0) -> 'MechanisticModelPipeline':
        """
        拟合物理约束神经网络
        
        Args:
            X_data: 观测数据输入
            y_data: 观测数据输出
            physics_constraints: 物理约束
            X_physics: 物理配点
            hidden_dims: 隐藏层维度
            epochs: 训练轮数
            physics_weight: 物理损失权重
            
        Returns:
            self
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for PINN")
        
        input_dim = X_data.shape[1]
        output_dim = y_data.shape[1] if len(y_data.shape) > 1 else 1
        
        self.pinn = PhysicsInformedNN(
            input_dim, output_dim,
            hidden_dims=hidden_dims,
            physics_constraints=physics_constraints
        )
        
        self.pinn_trainer = PINNTrainer(
            self.pinn,
            physics_weight=physics_weight
        )
        
        self.pinn_trainer.train(
            X_data, y_data.reshape(-1, 1) if len(y_data.shape) == 1 else y_data,
            X_physics, epochs=epochs
        )
        
        return self
    
    def fit_symbolic(self, X: np.ndarray, y: np.ndarray,
                    variable_names: List[str] = None,
                    **kwargs) -> DiscoveredEquation:
        """
        拟合符号回归
        
        Args:
            X: 特征数据
            y: 目标值
            variable_names: 变量名称
            **kwargs: 符号回归参数
            
        Returns:
            发现的最佳方程
        """
        self.symbolic_model = SymbolicRegression(**kwargs)
        return self.symbolic_model.fit(X, y, variable_names)
    
    def discover_equations(self, X: np.ndarray, y: np.ndarray = None,
                          variable_names: List[str] = None,
                          **kwargs) -> DiscoveredEquation:
        """
        发现方程
        
        Args:
            X: 输入数据
            y: 目标值
            variable_names: 变量名称
            **kwargs: 方程发现参数
            
        Returns:
            发现的方程
        """
        self.equation_discovery = EquationDiscovery(**kwargs)
        
        if y is None:
            # 假设是动力系统，发现整个方程组
            return self.equation_discovery.discover_dynamical_system(
                X, variable_names=variable_names
            )
        else:
            return self.equation_discovery.discover(X, y)
    
    def predict(self, X: np.ndarray, method: str = 'pinn') -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入数据
            method: 预测方法 ('pinn', 'symbolic')
            
        Returns:
            预测结果
        """
        if method == 'pinn' and self.pinn_trainer is not None:
            return self.pinn_trainer.predict(X)
        elif method == 'symbolic' and self.symbolic_model is not None:
            equation = self.symbolic_model._create_equation(X, np.zeros(len(X)))
            return equation.predict(X)
        else:
            raise ValueError(f"Method {method} not available")
    
    def visualize_results(self, X: np.ndarray, y: np.ndarray = None, ax=None):
        """
        可视化结果
        
        Args:
            X: 输入数据
            y: 真实值（可选）
            ax: matplotlib轴
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available")
            return None
        
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            axes = ax if hasattr(ax, '__len__') else [ax, ax]
        
        # PINN预测
        if self.pinn_trainer is not None:
            y_pinn = self.predict(X, method='pinn')
            axes[0].scatter(X[:, 0], y_pinn, label='PINN Prediction', alpha=0.6)
            if y is not None:
                axes[0].scatter(X[:, 0], y, label='True Data', alpha=0.6)
            axes[0].set_xlabel('Input')
            axes[0].set_ylabel('Output')
            axes[0].set_title('PINN Results')
            axes[0].legend()
        
        # 训练历史
        if self.pinn_trainer is not None and self.pinn_trainer.history:
            axes[1].semilogy(self.pinn_trainer.history['data_loss'], label='Data Loss')
            axes[1].semilogy(self.pinn_trainer.history['physics_loss'], label='Physics Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training History')
            axes[1].legend()
        
        plt.tight_layout()
        return axes


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("机理模型示例")
    print("=" * 60)
    
    # 示例1: 物理约束神经网络
    print("\n" + "-" * 40)
    print("1. 物理约束神经网络 (PINN)")
    print("-" * 40)
    
    if HAS_TORCH:
        # 生成谐振子数据
        np.random.seed(42)
        t = np.linspace(0, 10, 200).reshape(-1, 1)
        omega = 2.0
        u_true = np.sin(omega * t)
        
        # 添加噪声
        u_noisy = u_true + 0.1 * np.random.randn(*u_true.shape)
        
        # 定义物理约束：谐振子方程 u'' + omega^2 * u = 0
        def harmonic_constraint(u, u_x, u_xx, u_t, x):
            # u_tt + omega^2 * u = 0
            return u_xx + omega**2 * u
        
        constraint = PhysicalConstraint(
            name="harmonic_oscillator",
            constraint_type="pde",
            equation=harmonic_constraint,
            weight=1.0
        )
        
        # 训练PINN
        pipeline = MechanisticModelPipeline()
        pipeline.fit_pinn(
            t[:100], u_noisy[:100],
            physics_constraints=[constraint],
            X_physics=t,
            hidden_dims=[32, 32],
            epochs=500,
            physics_weight=0.1
        )
        
        # 预测
        u_pred = pipeline.predict(t, method='pinn')
        mse = np.mean((u_pred.flatten() - u_true.flatten()) ** 2)
        print(f"PINN MSE: {mse:.6f}")
    else:
        print("PyTorch not available, skipping PINN example")
    
    # 示例2: 符号回归
    print("\n" + "-" * 40)
    print("2. 符号回归")
    print("-" * 40)
    
    # 生成符号数据
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (200, 2))
    y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.5 * X[:, 0] * X[:, 1]
    
    sr = SymbolicRegression(
        population_size=50,
        generations=30,
        mutation_rate=0.2,
        crossover_rate=0.7
    )
    
    equation = sr.fit(X, y, variable_names=['x', 'y'])
    print(f"\n发现方程: {equation}")
    print(f"R² 分数: {equation.r2_score:.4f}")
    
    # 示例3: 方程发现
    print("\n" + "-" * 40)
    print("3. 方程发现 (SINDy-like)")
    print("-" * 40)
    
    # 生成Lorenz系统数据
    dt = 0.01
    t = np.arange(0, 10, dt)
    n = len(t)
    
    # 简化的Lorenz系统
    sigma, rho, beta = 10.0, 28.0, 8/3
    
    X_lorenz = np.zeros((n, 3))
    X_lorenz[0] = [1.0, 1.0, 1.0]
    
    for i in range(n-1):
        x, y, z = X_lorenz[i]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        X_lorenz[i+1] = X_lorenz[i] + dt * np.array([dx, dy, dz])
    
    # 发现方程
    ed = EquationDiscovery(library_size=30, sparsity_threshold=0.5)
    equations = ed.discover_dynamical_system(
        X_lorenz, dt=dt, variable_names=['x', 'y', 'z']
    )
    
    print("\n发现的动力学方程:")
    for i, eq in enumerate(equations):
        print(f"  {eq.expression}")
        print(f"  R² = {eq.r2_score:.4f}")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
