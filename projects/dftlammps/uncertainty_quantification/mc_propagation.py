"""
蒙特卡洛误差传播模块 - Monte Carlo Error Propagation

实现不确定性传播的蒙特卡洛方法和高级采样技术。

核心特性:
- 直接蒙特卡洛采样
- 拉丁超立方采样 (LHS)
- 拟蒙特卡洛 (QMC)
- 多项式混沌展开 (PCE)
- 随机配置方法
- 马尔可夫链蒙特卡洛 (MCMC)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

try:
    from scipy import stats
    from scipy.stats import qmc
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==================== 数据结构 ====================

@dataclass
class ConfidenceInterval:
    """置信区间"""
    lower: np.ndarray
    upper: np.ndarray
    confidence: float = 0.95
    
    def width(self) -> np.ndarray:
        """区间宽度"""
        return self.upper - self.lower
    
    def contains(self, value: np.ndarray) -> np.ndarray:
        """检查值是否在区间内"""
        return (value >= self.lower) & (value <= self.upper)
    
    def coverage_probability(self, true_values: np.ndarray) -> float:
        """计算覆盖率"""
        return np.mean(self.contains(true_values))


@dataclass
class ErrorBudget:
    """误差预算分析"""
    total_variance: float
    parameter_contributions: Dict[str, float] = field(default_factory=dict)
    correlation_effects: Dict[str, float] = field(default_factory=dict)
    
    def relative_contribution(self, param: str) -> float:
        """计算相对贡献"""
        if param not in self.parameter_contributions:
            return 0.0
        return self.parameter_contributions[param] / self.total_variance
    
    def get_dominant_sources(self, n: int = 3) -> List[Tuple[str, float]]:
        """获取主要误差来源"""
        sorted_contrib = sorted(
            self.parameter_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_contrib[:n]


@dataclass
class PropagationResult:
    """误差传播结果"""
    mean: np.ndarray
    variance: np.ndarray
    std: np.ndarray
    samples: np.ndarray
    confidence_intervals: Dict[float, ConfidenceInterval] = field(default_factory=dict)
    error_budget: Optional[ErrorBudget] = None
    convergence_history: List[Dict] = field(default_factory=list)
    
    def get_percentile(self, p: float) -> np.ndarray:
        """获取百分位数"""
        return np.percentile(self.samples, p, axis=0)
    
    def get_credible_interval(self, confidence: float = 0.95) -> ConfidenceInterval:
        """获取可信区间"""
        alpha = (1 - confidence) / 2
        lower = np.percentile(self.samples, alpha * 100, axis=0)
        upper = np.percentile(self.samples, (1 - alpha) * 100, axis=0)
        return ConfidenceInterval(lower, upper, confidence)
    
    def probability_exceedance(self, threshold: float) -> float:
        """超过阈值的概率"""
        return np.mean(self.samples > threshold)
    
    def probability_of_failure(self, limit_state: Callable) -> float:
        """失效概率"""
        return np.mean([limit_state(s) for s in self.samples])


# ==================== 采样器基类 ====================

class Sampler(ABC):
    """采样器基类"""
    
    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        """生成样本"""
        pass


class DirectSampler(Sampler):
    """直接采样器（从给定分布采样）"""
    
    def __init__(self,
                 distributions: List[stats.rv_continuous],
                 correlation_matrix: Optional[np.ndarray] = None):
        """
        初始化直接采样器
        
        Args:
            distributions: 各维度的概率分布列表
            correlation_matrix: 相关性矩阵（可选）
        """
        self.distributions = distributions
        self.n_dim = len(distributions)
        self.correlation_matrix = correlation_matrix
        
        if correlation_matrix is not None:
            # Cholesky分解用于相关采样
            self.L = np.linalg.cholesky(correlation_matrix)
    
    def sample(self, n_samples: int) -> np.ndarray:
        """生成相关/独立样本"""
        # 独立采样
        independent_samples = np.array([
            dist.rvs(n_samples) for dist in self.distributions
        ]).T
        
        if self.correlation_matrix is not None:
            # 应用相关性
            correlated_samples = independent_samples @ self.L.T
            return correlated_samples
        
        return independent_samples


class LatinHypercubeSampler(Sampler):
    """拉丁超立方采样器"""
    
    def __init__(self,
                 bounds: np.ndarray,
                 distributions: Optional[List[stats.rv_continuous]] = None):
        """
        初始化LHS采样器
        
        Args:
            bounds: 参数边界 [(lower, upper), ...]
            distributions: 各维度的分布（可选，默认均匀）
        """
        self.bounds = bounds
        self.n_dim = len(bounds)
        self.distributions = distributions
    
    def sample(self, n_samples: int) -> np.ndarray:
        """生成LHS样本"""
        if HAS_SCIPY:
            # 使用scipy的LHS
            sampler = qmc.LatinHypercube(d=self.n_dim)
            unit_samples = sampler.random(n=n_samples)
        else:
            # 手动实现LHS
            unit_samples = self._manual_lhs(n_samples)
        
        # 转换到目标分布
        samples = np.zeros((n_samples, self.n_dim))
        for i in range(self.n_dim):
            if self.distributions is not None and self.distributions[i] is not None:
                # 使用逆变换采样
                samples[:, i] = self.distributions[i].ppf(unit_samples[:, i])
            else:
                # 均匀分布
                lower, upper = self.bounds[i]
                samples[:, i] = lower + unit_samples[:, i] * (upper - lower)
        
        return samples
    
    def _manual_lhs(self, n_samples: int) -> np.ndarray:
        """手动实现LHS"""
        samples = np.zeros((n_samples, self.n_dim))
        
        for i in range(self.n_dim):
            # 生成排列
            perm = np.random.permutation(n_samples)
            # 在区间内随机位置
            samples[:, i] = (perm + np.random.rand(n_samples)) / n_samples
        
        return samples


class QuasiMonteCarloSampler(Sampler):
    """拟蒙特卡洛采样器（Sobol序列等）"""
    
    def __init__(self,
                 bounds: np.ndarray,
                 sequence: str = 'sobol'):
        """
        初始化QMC采样器
        
        Args:
            bounds: 参数边界
            sequence: 序列类型 ('sobol', 'halton', 'lhs')
        """
        self.bounds = bounds
        self.n_dim = len(bounds)
        self.sequence = sequence
    
    def sample(self, n_samples: int) -> np.ndarray:
        """生成QMC样本"""
        if HAS_SCIPY:
            if self.sequence == 'sobol':
                sampler = qmc.Sobol(d=self.n_dim, scramble=True)
            elif self.sequence == 'halton':
                sampler = qmc.Halton(d=self.n_dim)
            else:
                sampler = qmc.LatinHypercube(d=self.n_dim)
            
            unit_samples = sampler.random(n=n_samples)
        else:
            # 回退到随机采样
            unit_samples = np.random.rand(n_samples, self.n_dim)
        
        # 缩放到边界
        samples = np.zeros((n_samples, self.n_dim))
        for i in range(self.n_dim):
            lower, upper = self.bounds[i]
            samples[:, i] = lower + unit_samples[:, i] * (upper - lower)
        
        return samples


class ImportanceSampler(Sampler):
    """重要性采样器"""
    
    def __init__(self,
                 proposal_dist: stats.rv_continuous,
                 target_logpdf: Callable,
                 n_dim: int = 1):
        """
        初始化重要性采样器
        
        Args:
            proposal_dist: 提议分布
            target_logpdf: 目标对数概率密度函数
            n_dim: 维度
        """
        self.proposal_dist = proposal_dist
        self.target_logpdf = target_logpdf
        self.n_dim = n_dim
    
    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成重要性加权样本
        
        Returns:
            samples: 样本
            weights: 重要性权重
        """
        # 从提议分布采样
        samples = self.proposal_dist.rvs((n_samples, self.n_dim))
        
        # 计算权重
        weights = np.zeros(n_samples)
        for i in range(n_samples):
            target_log = self.target_logpdf(samples[i])
            proposal_log = np.sum([
                self.proposal_dist.logpdf(samples[i, j])
                for j in range(self.n_dim)
            ])
            weights[i] = np.exp(target_log - proposal_log)
        
        # 归一化权重
        weights = weights / np.sum(weights)
        
        return samples, weights


# ==================== 误差传播方法 ====================

class UncertaintyPropagator(ABC):
    """不确定性传播基类"""
    
    @abstractmethod
    def propagate(self,
                  model: Callable,
                  input_sampler: Sampler,
                  n_samples: int) -> PropagationResult:
        """传播不确定性"""
        pass


class DirectSampling(UncertaintyPropagator):
    """直接蒙特卡洛采样传播"""
    
    def __init__(self,
                 batch_size: int = 1000,
                 n_workers: int = 1):
        self.batch_size = batch_size
        self.n_workers = n_workers
    
    def propagate(self,
                  model: Callable,
                  input_sampler: Sampler,
                  n_samples: int) -> PropagationResult:
        """
        直接采样传播
        
        Args:
            model: 计算模型 f(x) -> y
            input_sampler: 输入参数采样器
            n_samples: 样本数量
        """
        all_samples = []
        all_outputs = []
        convergence_history = []
        
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            batch_size = min(self.batch_size, n_samples - batch_idx * self.batch_size)
            
            # 采样输入
            inputs = input_sampler.sample(batch_size)
            
            # 计算输出
            outputs = np.array([model(x) for x in inputs])
            
            all_outputs.append(outputs)
            
            # 记录收敛历史
            current_outputs = np.vstack(all_outputs)
            
            if len(current_outputs) > 10:
                conv_stat = {
                    'n_samples': len(current_outputs),
                    'mean': np.mean(current_outputs, axis=0),
                    'std': np.std(current_outputs, axis=0),
                    'stderr': np.std(current_outputs, axis=0) / np.sqrt(len(current_outputs))
                }
                convergence_history.append(conv_stat)
        
        # 合并所有输出
        outputs = np.vstack(all_outputs)
        
        # 计算统计量
        mean = np.mean(outputs, axis=0)
        variance = np.var(outputs, axis=0)
        std = np.sqrt(variance)
        
        # 计算置信区间
        confidence_intervals = {}
        for conf in [0.68, 0.95, 0.99]:
            alpha = (1 - conf) / 2
            lower = np.percentile(outputs, alpha * 100, axis=0)
            upper = np.percentile(outputs, (1 - alpha) * 100, axis=0)
            confidence_intervals[conf] = ConfidenceInterval(lower, upper, conf)
        
        return PropagationResult(
            mean=mean,
            variance=variance,
            std=std,
            samples=outputs,
            confidence_intervals=confidence_intervals,
            convergence_history=convergence_history
        )


class LatinHypercubeSampling(UncertaintyPropagator):
    """拉丁超立方采样传播"""
    
    def propagate(self,
                  model: Callable,
                  input_sampler: LatinHypercubeSampler,
                  n_samples: int) -> PropagationResult:
        """
        LHS传播
        
        LHS在相同样本数下通常比直接采样更均匀
        """
        # 生成LHS样本
        inputs = input_sampler.sample(n_samples)
        
        # 计算输出
        outputs = np.array([model(x) for x in inputs])
        
        mean = np.mean(outputs, axis=0)
        variance = np.var(outputs, axis=0)
        std = np.sqrt(variance)
        
        confidence_intervals = {}
        for conf in [0.68, 0.95, 0.99]:
            alpha = (1 - conf) / 2
            lower = np.percentile(outputs, alpha * 100, axis=0)
            upper = np.percentile(outputs, (1 - alpha) * 100, axis=0)
            confidence_intervals[conf] = ConfidenceInterval(lower, upper, conf)
        
        return PropagationResult(
            mean=mean,
            variance=variance,
            std=std,
            samples=outputs,
            confidence_intervals=confidence_intervals
        )


class QuasiMonteCarlo(UncertaintyPropagator):
    """拟蒙特卡洛传播"""
    
    def propagate(self,
                  model: Callable,
                  input_sampler: QuasiMonteCarloSampler,
                  n_samples: int) -> PropagationResult:
        """
        QMC传播
        
        使用低差异序列实现更快的收敛
        """
        inputs = input_sampler.sample(n_samples)
        
        outputs = np.array([model(x) for x in inputs])
        
        mean = np.mean(outputs, axis=0)
        variance = np.var(outputs, axis=0)
        std = np.sqrt(variance)
        
        confidence_intervals = {}
        for conf in [0.68, 0.95, 0.99]:
            alpha = (1 - conf) / 2
            lower = np.percentile(outputs, alpha * 100, axis=0)
            upper = np.percentile(outputs, (1 - alpha) * 100, axis=0)
            confidence_intervals[conf] = ConfidenceInterval(lower, upper, conf)
        
        return PropagationResult(
            mean=mean,
            variance=variance,
            std=std,
            samples=outputs,
            confidence_intervals=confidence_intervals
        )


# ==================== 高级方法 ====================

class PolynomialChaosExpansion:
    """
    多项式混沌展开 (PCE)
    
    使用正交多项式展开来高效传播不确定性
    """
    
    def __init__(self,
                 degree: int = 3,
                 polynomial_type: str = 'legendre'):
        """
        初始化PCE
        
        Args:
            degree: 多项式阶数
            polynomial_type: 多项式类型 ('legendre', 'hermite', 'laguerre')
        """
        self.degree = degree
        self.polynomial_type = polynomial_type
        self.coefficients = None
        self.multi_indices = None
    
    def fit(self,
            samples: np.ndarray,
            evaluations: np.ndarray) -> 'PolynomialChaosExpansion':
        """
        拟合PCE系数
        
        Args:
            samples: 输入样本 (n_samples, n_dim)
            evaluations: 模型评估值 (n_samples, n_outputs)
        """
        n_samples, n_dim = samples.shape
        
        # 生成多重指标
        self.multi_indices = self._generate_multi_indices(n_dim, self.degree)
        n_terms = len(self.multi_indices)
        
        # 构建设计矩阵
        design_matrix = np.zeros((n_samples, n_terms))
        for i, idx in enumerate(self.multi_indices):
            design_matrix[:, i] = self._evaluate_polynomials(samples, idx)
        
        # 最小二乘拟合
        self.coefficients = np.linalg.lstsq(
            design_matrix, evaluations, rcond=None
        )[0]
        
        return self
    
    def predict(self,
                samples: np.ndarray,
                return_variance: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """使用PCE预测"""
        n_samples = len(samples)
        n_terms = len(self.multi_indices)
        
        design_matrix = np.zeros((n_samples, n_terms))
        for i, idx in enumerate(self.multi_indices):
            design_matrix[:, i] = self._evaluate_polynomials(samples, idx)
        
        predictions = design_matrix @ self.coefficients
        
        if return_variance:
            # 计算预测方差（简化）
            variance = np.var(predictions, axis=0)
            return predictions, variance
        
        return predictions
    
    def _generate_multi_indices(self, n_dim: int, degree: int) -> List[Tuple[int, ...]]:
        """生成多重指标"""
        from itertools import combinations_with_replacement
        
        indices = []
        for d in range(degree + 1):
            for idx in combinations_with_replacement(range(n_dim), d):
                multi_idx = [0] * n_dim
                for i in idx:
                    multi_idx[i] += 1
                indices.append(tuple(multi_idx))
        
        return indices
    
    def _evaluate_polynomials(self, samples: np.ndarray, 
                              multi_idx: Tuple[int, ...]) -> np.ndarray:
        """评估正交多项式"""
        result = np.ones(len(samples))
        
        for dim, power in enumerate(multi_idx):
            if power > 0:
                if self.polynomial_type == 'legendre':
                    # 勒让德多项式
                    from numpy.polynomial.legendre import legval
                    result *= legval(samples[:, dim], [0]*power + [1])
                elif self.polynomial_type == 'hermite':
                    # 埃尔米特多项式
                    from numpy.polynomial.hermite import hermval
                    result *= hermval(samples[:, dim], [0]*power + [1])
                else:
                    # 单项式基
                    result *= samples[:, dim] ** power
        
        return result
    
    def get_sobol_indices(self) -> Dict[str, float]:
        """计算Sobol敏感性指标"""
        # 简化实现
        return {'total_variance': np.sum(self.coefficients**2)}


class StochasticCollocation:
    """
    随机配置方法
    
    在精心选择的配点上求解模型，然后插值
    """
    
    def __init__(self,
                 n_points_per_dim: int = 5,
                 quadrature_rule: str = 'gauss_legendre'):
        """
        初始化随机配置
        
        Args:
            n_points_per_dim: 每维配点数
            quadrature_rule: 求积规则
        """
        self.n_points_per_dim = n_points_per_dim
        self.quadrature_rule = quadrature_rule
        self.collocation_points = None
        self.evaluations = None
    
    def fit(self,
            model: Callable,
            n_dim: int,
            bounds: np.ndarray) -> 'StochasticCollocation':
        """
        在配点上评估模型
        """
        # 生成配点（Gauss-Legendre点）
        from numpy.polynomial.legendre import leggauss
        
        points_1d, weights_1d = leggauss(self.n_points_per_dim)
        
        # 张量积配点
        grids = np.meshgrid(*[points_1d for _ in range(n_dim)], indexing='ij')
        self.collocation_points = np.stack([g.flatten() for g in grids], axis=1)
        
        # 缩放到边界
        for i in range(n_dim):
            lower, upper = bounds[i]
            self.collocation_points[:, i] = 0.5 * (
                (upper + lower) + (upper - lower) * self.collocation_points[:, i]
            )
        
        # 评估模型
        self.evaluations = np.array([
            model(x) for x in self.collocation_points
        ])
        
        return self
    
    def predict(self, samples: np.ndarray) -> np.ndarray:
        """插值预测"""
        # 使用线性插值（简化）
        from scipy.interpolate import LinearNDInterpolator
        
        if HAS_SCIPY and self.collocation_points.shape[1] <= 3:
            interp = LinearNDInterpolator(
                self.collocation_points, self.evaluations
            )
            return interp(samples)
        else:
            # 最近邻插值
            predictions = []
            for s in samples:
                distances = np.linalg.norm(self.collocation_points - s, axis=1)
                nearest_idx = np.argmin(distances)
                predictions.append(self.evaluations[nearest_idx])
            return np.array(predictions)
    
    def compute_statistics(self) -> Dict[str, np.ndarray]:
        """计算统计量（使用求积）"""
        from numpy.polynomial.legendre import leggauss
        
        points_1d, weights_1d = leggauss(self.n_points_per_dim)
        
        # 张量积权重
        weight_grids = np.meshgrid(*[weights_1d for _ in range(
            self.collocation_points.shape[1])], indexing='ij')
        weights = np.prod([g.flatten() for g in weight_grids], axis=0)
        weights = weights / np.sum(weights)  # 归一化
        
        mean = np.sum(weights[:, None] * self.evaluations, axis=0)
        variance = np.sum(weights[:, None] * (self.evaluations - mean)**2, axis=0)
        
        return {'mean': mean, 'variance': variance}


class MarkovChainMonteCarlo:
    """
    马尔可夫链蒙特卡洛 (MCMC)
    
    用于从复杂后验分布采样
    """
    
    def __init__(self,
                 proposal_width: float = 0.1,
                 n_burnin: int = 1000,
                 thinning: int = 10):
        """
        初始化MCMC采样器
        
        Args:
            proposal_width: 提议分布宽度
            n_burnin: 预热样本数
            thinning: 稀释间隔
        """
        self.proposal_width = proposal_width
        self.n_burnin = n_burnin
        self.thinning = thinning
    
    def sample(self,
               log_posterior: Callable,
               initial_state: np.ndarray,
               n_samples: int) -> np.ndarray:
        """
        Metropolis-Hastings采样
        
        Args:
            log_posterior: 对数后验函数
            initial_state: 初始状态
            n_samples: 样本数
        """
        n_dim = len(initial_state)
        
        # 预热
        current_state = initial_state.copy()
        current_log_prob = log_posterior(current_state)
        
        for _ in range(self.n_burnin):
            current_state, current_log_prob = self._mcmc_step(
                current_state, current_log_prob, log_posterior
            )
        
        # 正式采样
        samples = []
        accepted = 0
        
        for i in range(n_samples * self.thinning):
            current_state, current_log_prob = self._mcmc_step(
                current_state, current_log_prob, log_posterior
            )
            
            if i % self.thinning == 0:
                samples.append(current_state.copy())
            
            if current_state is not None:
                accepted += 1
        
        acceptance_rate = accepted / (n_samples * self.thinning)
        print(f"MCMC接受率: {acceptance_rate:.3f}")
        
        return np.array(samples)
    
    def _mcmc_step(self,
                   current_state: np.ndarray,
                   current_log_prob: float,
                   log_posterior: Callable) -> Tuple[np.ndarray, float]:
        """MCMC单步"""
        # 提议新状态
        proposal = current_state + np.random.randn(len(current_state)) * self.proposal_width
        
        # 计算接受概率
        proposal_log_prob = log_posterior(proposal)
        log_accept_prob = proposal_log_prob - current_log_prob
        
        # 接受/拒绝
        if np.log(np.random.rand()) < log_accept_prob:
            return proposal, proposal_log_prob
        else:
            return current_state, current_log_prob


# ==================== 主控类 ====================

class MCErrorPropagation:
    """
    蒙特卡洛误差传播主控类
    
    提供统一的接口进行不确定性传播
    """
    
    def __init__(self):
        self.methods = {
            'direct': DirectSampling(),
            'lhs': LatinHypercubeSampling(),
            'qmc': QuasiMonteCarlo()
        }
    
    def propagate(self,
                  model: Callable,
                  input_distributions: Dict[str, stats.rv_continuous],
                  n_samples: int = 10000,
                  method: str = 'lhs',
                  correlation_matrix: Optional[np.ndarray] = None) -> PropagationResult:
        """
        传播不确定性
        
        Args:
            model: 计算模型
            input_distributions: 输入参数分布
            n_samples: 样本数
            method: 采样方法
            correlation_matrix: 相关性矩阵
        """
        param_names = list(input_distributions.keys())
        distributions = list(input_distributions.values())
        
        # 创建采样器
        if method == 'direct':
            sampler = DirectSampler(distributions, correlation_matrix)
            propagator = self.methods['direct']
        elif method == 'lhs':
            # LHS需要边界
            bounds = np.array([[dist.ppf(0.001), dist.ppf(0.999)] 
                              for dist in distributions])
            sampler = LatinHypercubeSampler(bounds, distributions)
            propagator = self.methods['lhs']
        elif method == 'qmc':
            bounds = np.array([[dist.ppf(0.001), dist.ppf(0.999)]
                              for dist in distributions])
            sampler = QuasiMonteCarloSampler(bounds, 'sobol')
            propagator = self.methods['qmc']
        else:
            raise ValueError(f"未知方法: {method}")
        
        # 包装模型以接受数组输入
        def wrapped_model(params):
            params_dict = {name: params[i] 
                          for i, name in enumerate(param_names)}
            return model(**params_dict)
        
        # 执行传播
        result = propagator.propagate(wrapped_model, sampler, n_samples)
        
        return result
    
    def error_budget_analysis(self,
                             model: Callable,
                             nominal_params: Dict[str, float],
                             param_uncertainties: Dict[str, float],
                             n_samples: int = 5000) -> ErrorBudget:
        """
        误差预算分析
        
        确定各参数对总不确定性的贡献
        """
        param_names = list(nominal_params.keys())
        n_params = len(param_names)
        
        # 全因素分析
        distributions = {
            name: stats.norm(nominal_params[name], param_uncertainties[name])
            for name in param_names
        }
        
        result_full = self.propagate(
            lambda **kwargs: model(**kwargs),
            distributions,
            n_samples,
            method='lhs'
        )
        
        total_variance = np.sum(result_full.variance)
        
        # 单因素分析
        contributions = {}
        for name in param_names:
            # 仅该参数变化，其他固定
            single_dist = {name: distributions[name]}
            for other in param_names:
                if other != name:
                    single_dist[other] = stats.norm(
                        nominal_params[other], 1e-10
                    )
            
            result_single = self.propagate(
                lambda **kwargs: model(**kwargs),
                single_dist,
                n_samples,
                method='lhs'
            )
            
            contributions[name] = np.sum(result_single.variance)
        
        return ErrorBudget(
            total_variance=total_variance,
            parameter_contributions=contributions
        )


# ==================== 示例和测试 ====================

def demo():
    """演示蒙特卡洛误差传播"""
    print("=" * 80)
    print("📊 蒙特卡洛误差传播演示")
    print("=" * 80)
    
    # 定义测试模型：应力-应变关系
    print("\n1. 定义材料模型: 弹性模量计算")
    
    def elastic_model(E: float, nu: float, epsilon: float) -> float:
        """
        弹性模型
        
        Args:
            E: 杨氏模量 (GPa)
            nu: 泊松比
            epsilon: 应变
        
        Returns:
            应力 (GPa)
        """
        # 胡克定律
        stress = E * epsilon
        return stress
    
    # 包装模型用于传播
    def wrapped_model(params):
        E, nu, epsilon = params
        return elastic_model(E, nu, epsilon)
    
    # 定义输入不确定性
    print("\n2. 定义输入参数不确定性:")
    print("   - E (杨氏模量): N(200, 10) GPa")
    print("   - ν (泊松比): N(0.3, 0.02)")
    print("   - ε (应变): N(0.01, 0.001)")
    
    distributions = [
        stats.norm(200, 10),    # E
        stats.norm(0.3, 0.02),  # nu
        stats.norm(0.01, 0.001) # epsilon
    ]
    
    bounds = np.array([
        [150, 250],
        [0.2, 0.4],
        [0.005, 0.015]
    ])
    
    # 演示不同采样方法
    print("\n3. 不同采样方法对比:")
    
    # 直接采样
    print("\n   a) 直接蒙特卡洛采样 (n=10000)")
    direct_sampler = DirectSampler(distributions)
    direct_propagator = DirectSampling()
    
    direct_inputs = direct_sampler.sample(10000)
    direct_outputs = np.array([wrapped_model(x) for x in direct_inputs])
    
    print(f"      应力均值: {np.mean(direct_outputs):.4f} GPa")
    print(f"      应力标准差: {np.std(direct_outputs):.4f} GPa")
    print(f"      95% 置信区间: [{np.percentile(direct_outputs, 2.5):.4f}, "
          f"{np.percentile(direct_outputs, 97.5):.4f}] GPa")
    
    # LHS采样
    print("\n   b) 拉丁超立方采样 (n=10000)")
    lhs_sampler = LatinHypercubeSampler(bounds, distributions)
    lhs_inputs = lhs_sampler.sample(10000)
    lhs_outputs = np.array([wrapped_model(x) for x in lhs_inputs])
    
    print(f"      应力均值: {np.mean(lhs_outputs):.4f} GPa")
    print(f"      应力标准差: {np.std(lhs_outputs):.4f} GPa")
    print(f"      95% 置信区间: [{np.percentile(lhs_outputs, 2.5):.4f}, "
          f"{np.percentile(lhs_outputs, 97.5):.4f}] GPa")
    
    # QMC采样
    if HAS_SCIPY:
        print("\n   c) Sobol序列采样 (n=8192)")
        qmc_sampler = QuasiMonteCarloSampler(bounds, 'sobol')
        qmc_inputs = qmc_sampler.sample(8192)
        qmc_outputs = np.array([wrapped_model(x) for x in qmc_inputs])
        
        print(f"      应力均值: {np.mean(qmc_outputs):.4f} GPa")
        print(f"      应力标准差: {np.std(qmc_outputs):.4f} GPa")
        print(f"      95% 置信区间: [{np.percentile(qmc_outputs, 2.5):.4f}, "
          f"{np.percentile(qmc_outputs, 97.5):.4f}] GPa")
    
    # 误差预算分析
    print("\n4. 误差预算分析:")
    
    mc_prop = MCErrorPropagation()
    
    param_model = lambda E, nu, epsilon: E * epsilon
    
    error_budget = mc_prop.error_budget_analysis(
        model=param_model,
        nominal_params={'E': 200, 'nu': 0.3, 'epsilon': 0.01},
        param_uncertainties={'E': 10, 'nu': 0.02, 'epsilon': 0.001},
        n_samples=5000
    )
    
    print(f"   总方差: {error_budget.total_variance:.6f}")
    print("   参数贡献:")
    for param, contrib in error_budget.get_dominant_sources(3):
        rel_contrib = error_budget.relative_contribution(param)
        print(f"      - {param}: {contrib:.6f} ({rel_contrib*100:.1f}%)")
    
    # PCE演示
    print("\n5. 多项式混沌展开 (PCE):")
    
    # 训练数据
    np.random.seed(42)
    train_samples = np.random.rand(200, 2) * 2 - 1  # [-1, 1]^2
    train_samples[:, 0] = train_samples[:, 0] * 50 + 200  # E: [150, 250]
    train_samples[:, 1] = train_samples[:, 1] * 0.005 + 0.01  # epsilon
    
    train_evals = np.array([wrapped_model([s[0], 0.3, s[1]]) for s in train_samples])
    
    # 拟合PCE
    pce = PolynomialChaosExpansion(degree=3)
    pce.fit(train_samples, train_evals)
    
    # 测试预测
    test_samples = np.random.rand(100, 2)
    test_samples[:, 0] = test_samples[:, 0] * 50 + 175
    test_samples[:, 1] = test_samples[:, 1] * 0.005 + 0.0075
    
    pce_preds = pce.predict(test_samples)
    true_vals = np.array([wrapped_model([s[0], 0.3, s[1]]) for s in test_samples])
    
    rmse = np.sqrt(np.mean((pce_preds - true_vals)**2))
    print(f"   PCE测试RMSE: {rmse:.4f} GPa")
    print(f"   PCE系数数量: {len(pce.coefficients)}")
    
    print("\n" + "=" * 80)
    print("✅ 蒙特卡洛误差传播演示完成")
    print("=" * 80)


if __name__ == "__main__":
    demo()
