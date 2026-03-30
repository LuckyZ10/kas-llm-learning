"""
不确定性量化和置信度评估 (Uncertainty Quantification & Confidence Assessment)

实现预测结果的不确定性量化和置信度评估框架。
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Protocol, Set,
    Tuple, TypeVar, Union, Iterator
)

import numpy as np
from numpy.typing import NDArray

warnings.filterwarnings('ignore')


try:
    from .twin_core import StateVector, Prediction, Observation
except ImportError:
    from twin_core import StateVector, Prediction, Observation


T = TypeVar('T')


class UncertaintyType(Enum):
    """不确定性类型"""
    ALEATORIC = "aleatoric"         # 偶然不确定性 (数据噪声)
    EPISTEMIC = "epistemic"         # 认知不确定性 (模型不足)
    PARAMETRIC = "parametric"       # 参数不确定性
    STRUCTURAL = "structural"       # 结构不确定性
    MEASUREMENT = "measurement"     # 测量不确定性
    PROPAGATION = "propagation"     # 传播不确定性


class UQMethod(Enum):
    """不确定性量化方法"""
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"
    PERTURBATION = "perturbation"
    POLYNOMIAL_CHAOS = "polynomial_chaos"


@dataclass
class UncertaintyEstimate:
    """不确定性估计结果"""
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    variance: NDArray[np.float64]
    confidence_intervals: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.mean, np.ndarray):
            self.mean = np.array(self.mean, dtype=np.float64)
        if not isinstance(self.std, np.ndarray):
            self.std = np.array(self.std, dtype=np.float64)
        if not isinstance(self.variance, np.ndarray):
            self.variance = np.array(self.variance, dtype=np.float64)
    
    @property
    def coefficient_of_variation(self) -> NDArray[np.float64]:
        """变异系数"""
        return np.abs(self.std / (self.mean + 1e-10))
    
    @property
    def relative_uncertainty(self) -> NDArray[np.float64]:
        """相对不确定性"""
        return self.std / (np.abs(self.mean) + 1e-10)


@dataclass
class ConfidenceAssessment:
    """置信度评估结果"""
    confidence_score: float  # 0-1
    reliability: float       # 可靠性
    calibration_error: float  # 校准误差
    prediction_interval_coverage: float  # 预测区间覆盖率
    
    @property
    def is_reliable(self) -> bool:
        return self.confidence_score > 0.7 and self.reliability > 0.8


class UncertaintyQuantifier(ABC):
    """
    不确定性量化器基类
    """
    
    def __init__(self, method: UQMethod):
        self.method = method
        self._is_fitted = False
    
    @abstractmethod
    def quantify(self, predictions: List[NDArray[np.float64]], 
                 **kwargs) -> UncertaintyEstimate:
        """量化不确定性"""
        pass
    
    @abstractmethod
    def calibrate(self, predictions: List[NDArray[np.float64]], 
                  ground_truth: NDArray[np.float64]) -> None:
        """校准不确定性估计"""
        pass


class MonteCarloUQ(UncertaintyQuantifier):
    """
    Monte Carlo不确定性量化
    
    通过多次随机采样估计不确定性
    """
    
    def __init__(self, n_samples: int = 1000):
        super().__init__(UQMethod.MONTE_CARLO)
        self.n_samples = n_samples
        self._samples: List[NDArray[np.float64]] = []
    
    def quantify(self, predictions: List[NDArray[np.float64]], 
                 **kwargs) -> UncertaintyEstimate:
        """
        使用Monte Carlo采样量化不确定性
        
        Args:
            predictions: 多个预测样本列表
        """
        if len(predictions) < 2:
            raise ValueError("Need at least 2 predictions for uncertainty quantification")
        
        # 转换为数组
        samples = np.array(predictions)
        self._samples = predictions
        
        # 计算统计量
        mean = np.mean(samples, axis=0)
        variance = np.var(samples, axis=0)
        std = np.sqrt(variance)
        
        # 计算置信区间
        ci_95 = self._compute_confidence_interval(samples, 0.95)
        ci_99 = self._compute_confidence_interval(samples, 0.99)
        
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            variance=variance,
            confidence_intervals={
                '95%': ci_95,
                '99%': ci_99
            }
        )
    
    def _compute_confidence_interval(self, samples: NDArray[np.float64], 
                                      confidence: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """计算置信区间"""
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(samples, lower_percentile, axis=0)
        upper = np.percentile(samples, upper_percentile, axis=0)
        
        return lower, upper
    
    def calibrate(self, predictions: List[NDArray[np.float64]], 
                  ground_truth: NDArray[np.float64]) -> None:
        """校准Monte Carlo估计"""
        # 检查覆盖率
        uq = self.quantify(predictions)
        lower, upper = uq.confidence_intervals['95%']
        
        coverage = np.mean((ground_truth >= lower) & (ground_truth <= upper))
        print(f"Monte Carlo 95% CI coverage: {coverage:.2%}")
        
        self._is_fitted = True


class BootstrapUQ(UncertaintyQuantifier):
    """
    Bootstrap不确定性量化
    
    使用Bootstrap重采样估计统计量的不确定性
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence: float = 0.95):
        super().__init__(UQMethod.BOOTSTRAP)
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self._bootstrap_samples: List[NDArray[np.float64]] = []
    
    def quantify(self, predictions: List[NDArray[np.float64]], 
                 **kwargs) -> UncertaintyEstimate:
        """
        使用Bootstrap量化不确定性
        """
        data = np.array(predictions)
        n_samples = len(data)
        
        # Bootstrap重采样
        bootstrap_means = []
        
        for _ in range(self.n_bootstrap):
            # 有放回抽样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample = data[indices]
            bootstrap_means.append(np.mean(sample, axis=0))
        
        bootstrap_means = np.array(bootstrap_means)
        self._bootstrap_samples = [b for b in bootstrap_means]
        
        # 计算统计量
        mean = np.mean(data, axis=0)
        std = np.std(bootstrap_means, axis=0)
        variance = np.var(bootstrap_means, axis=0)
        
        # 置信区间
        alpha = 1 - self.confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100, axis=0)
        upper = np.percentile(bootstrap_means, (1-alpha/2) * 100, axis=0)
        
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            variance=variance,
            confidence_intervals={
                f'{int(self.confidence*100)}%': (lower, upper)
            }
        )
    
    def calibrate(self, predictions: List[NDArray[np.float64]], 
                  ground_truth: NDArray[np.float64]) -> None:
        """校准Bootstrap估计"""
        self._is_fitted = True


class EnsembleUQ(UncertaintyQuantifier):
    """
    集成模型不确定性量化
    
    基于多个模型的预测差异估计不确定性
    """
    
    def __init__(self):
        super().__init__(UQMethod.ENSEMBLE)
        self.models: List[Any] = []
        self._model_weights: Optional[NDArray[np.float64]] = None
    
    def add_model(self, model: Any, weight: float = 1.0) -> None:
        """添加模型到集成"""
        self.models.append(model)
    
    def quantify(self, predictions: List[NDArray[np.float64]], 
                 **kwargs) -> UncertaintyEstimate:
        """
        量化集成模型的不确定性
        
        包含:
        - 模型间方差 (epistemic)
        - 平均预测方差 (aleatoric)
        """
        if len(predictions) < 2:
            raise ValueError("Need at least 2 models for ensemble UQ")
        
        predictions_array = np.array(predictions)
        
        # 集成预测均值
        if self._model_weights is not None:
            mean = np.average(predictions_array, axis=0, weights=self._model_weights)
        else:
            mean = np.mean(predictions_array, axis=0)
        
        # Epistemic uncertainty (模型间差异)
        epistemic_var = np.var(predictions_array, axis=0)
        
        # 假设每个模型的aleatoric uncertainty
        # 这里使用预测间的差异作为代理
        aleatoric_var = epistemic_var * 0.5  # 简化假设
        
        total_var = epistemic_var + aleatoric_var
        std = np.sqrt(total_var)
        
        # 置信区间
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std
        
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            variance=total_var,
            confidence_intervals={
                '95%': (ci_lower, ci_upper),
                'epistemic': (mean - np.sqrt(epistemic_var), mean + np.sqrt(epistemic_var))
            }
        )
    
    def calibrate(self, predictions: List[NDArray[np.float64]], 
                  ground_truth: NDArray[np.float64]) -> None:
        """校准集成模型权重"""
        # 基于验证性能调整权重
        errors = []
        for pred in predictions:
            error = np.mean((pred - ground_truth)**2)
            errors.append(error)
        
        # 权重反比于误差
        weights = 1.0 / (np.array(errors) + 1e-10)
        self._model_weights = weights / np.sum(weights)
        
        self._is_fitted = True


class BayesianUQ:
    """
    贝叶斯不确定性量化
    
    使用变分推断或MCMC估计后验分布
    """
    
    def __init__(self, n_posterior_samples: int = 1000):
        self.n_posterior_samples = n_posterior_samples
        self._posterior_samples: Optional[NDArray[np.float64]] = None
        self._prior_mean: Optional[NDArray[np.float64]] = None
        self._prior_std: float = 1.0
    
    def set_prior(self, mean: NDArray[np.float64], std: float = 1.0) -> None:
        """设置先验分布"""
        self._prior_mean = mean
        self._prior_std = std
    
    def approximate_posterior(self, likelihood_fn: Callable[[NDArray[np.float64]], float],
                               n_iterations: int = 1000) -> None:
        """
        使用变分推断近似后验分布
        
        简化实现: 使用随机游走Metropolis-Hastings
        """
        if self._prior_mean is None:
            raise ValueError("Prior not set")
        
        dim = len(self._prior_mean)
        samples = []
        current = self._prior_mean.copy()
        
        for i in range(n_iterations):
            # 提议分布
            proposal = current + np.random.randn(dim) * 0.1
            
            # 计算接受率
            current_likelihood = likelihood_fn(current)
            proposal_likelihood = likelihood_fn(proposal)
            
            # 先验比率 (高斯先验)
            prior_ratio = np.exp(-0.5 * np.sum((proposal - self._prior_mean)**2) / self._prior_std**2)
            prior_ratio /= np.exp(-0.5 * np.sum((current - self._prior_mean)**2) / self._prior_std**2)
            
            acceptance_prob = min(1.0, np.exp(proposal_likelihood - current_likelihood) * prior_ratio)
            
            if np.random.rand() < acceptance_prob:
                current = proposal
            
            if i > n_iterations // 4:  # Burn-in
                samples.append(current.copy())
        
        self._posterior_samples = np.array(samples)
    
    def quantify(self) -> UncertaintyEstimate:
        """量化后验不确定性"""
        if self._posterior_samples is None:
            raise ValueError("Posterior not approximated")
        
        mean = np.mean(self._posterior_samples, axis=0)
        std = np.std(self._posterior_samples, axis=0)
        variance = np.var(self._posterior_samples, axis=0)
        
        # 可信区间 (credible interval)
        lower = np.percentile(self._posterior_samples, 2.5, axis=0)
        upper = np.percentile(self._posterior_samples, 97.5, axis=0)
        
        return UncertaintyEstimate(
            mean=mean,
            std=std,
            variance=variance,
            confidence_intervals={
                '95%_credible': (lower, upper)
            }
        )


class SensitivityAnalyzer:
    """
    敏感性分析器
    
    分析输入参数对输出不确定性的影响
    """
    
    def __init__(self):
        self._base_values: Dict[str, float] = {}
        self._perturbation_range: Dict[str, Tuple[float, float]] = {}
        self._sensitivity_indices: Dict[str, float] = {}
    
    def set_parameter(self, name: str, base_value: float, 
                     min_val: float, max_val: float) -> None:
        """设置参数范围"""
        self._base_values[name] = base_value
        self._perturbation_range[name] = (min_val, max_val)
    
    def sobol_analysis(self, model_fn: Callable[..., float], 
                       n_samples: int = 1000) -> Dict[str, float]:
        """
        Sobol敏感性分析 (简化实现)
        
        计算一阶敏感性指数
        """
        param_names = list(self._base_values.keys())
        n_params = len(param_names)
        
        # 生成Saltelli采样
        A = np.random.rand(n_samples, n_params)
        B = np.random.rand(n_samples, n_params)
        
        # 缩放到参数范围
        for i, name in enumerate(param_names):
            min_val, max_val = self._perturbation_range[name]
            A[:, i] = A[:, i] * (max_val - min_val) + min_val
            B[:, i] = B[:, i] * (max_val - min_val) + min_val
        
        # 计算模型输出
        y_A = np.array([model_fn(*a) for a in A])
        y_B = np.array([model_fn(*b) for b in B])
        
        # 计算Sobol指数
        var_y = np.var(y_A)
        
        sobol_indices = {}
        for i, name in enumerate(param_names):
            # 创建混合矩阵
            AB = A.copy()
            AB[:, i] = B[:, i]
            y_AB = np.array([model_fn(*ab) for ab in AB])
            
            # 一阶Sobol指数
            sobol_i = np.mean(y_B * (y_AB - y_A)) / var_y if var_y > 0 else 0
            sobol_indices[name] = abs(sobol_i)
        
        self._sensitivity_indices = sobol_indices
        return sobol_indices
    
    def morris_screening(self, model_fn: Callable[..., float], 
                        n_trajectories: int = 10) -> Dict[str, Tuple[float, float]]:
        """
        Morris筛选方法
        
        计算基本效应的均值和标准差
        """
        param_names = list(self._base_values.keys())
        n_params = len(param_names)
        
        # 离散化水平
        delta = 1.0 / 3
        
        elementary_effects = {name: [] for name in param_names}
        
        for _ in range(n_trajectories):
            # 随机起点
            x_base = np.random.rand(n_params)
            
            for i, name in enumerate(param_names):
                # 创建扰动点
                x_perturbed = x_base.copy()
                x_perturbed[i] = (x_perturbed[i] + delta) % 1
                
                # 缩放
                min_val, max_val = self._perturbation_range[name]
                x_base_scaled = x_base[i] * (max_val - min_val) + min_val
                x_pert_scaled = x_perturbed[i] * (max_val - min_val) + min_val
                
                # 计算基本效应
                ee = (model_fn(*x_perturbed) - model_fn(*x_base)) / delta
                elementary_effects[name].append(ee)
        
        # 计算统计量
        morris_indices = {}
        for name in param_names:
            ees = elementary_effects[name]
            mu = np.mean(ees)
            sigma = np.std(ees)
            morris_indices[name] = (mu, sigma)
        
        return morris_indices


class ConfidenceEstimator:
    """
    置信度估计器
    
    评估预测结果的置信度和可靠性
    """
    
    def __init__(self):
        self._calibration_data: List[Tuple[float, bool]] = []  # (置信度, 是否正确)
        self._temperature: float = 1.0
    
    def estimate(self, prediction: NDArray[np.float64], 
                uncertainty: UncertaintyEstimate) -> ConfidenceAssessment:
        """
        估计预测的置信度
        """
        # 基于不确定性计算置信度
        cv = uncertainty.coefficient_of_variation
        avg_cv = np.mean(cv)
        
        # 变异系数越小，置信度越高
        confidence_score = np.exp(-avg_cv)
        
        # 计算可靠性
        reliability = self._compute_reliability(confidence_score)
        
        # 校准误差
        calibration_error = self._compute_calibration_error()
        
        # 预测区间覆盖率
        coverage = self._estimate_coverage(uncertainty)
        
        return ConfidenceAssessment(
            confidence_score=float(confidence_score),
            reliability=reliability,
            calibration_error=calibration_error,
            prediction_interval_coverage=coverage
        )
    
    def _compute_reliability(self, confidence: float) -> float:
        """计算可靠性分数"""
        # 基于历史校准数据
        if len(self._calibration_data) < 10:
            return confidence
        
        # 计算准确校准的频率
        correct_count = sum(1 for conf, correct in self._calibration_data 
                          if correct)
        empirical_accuracy = correct_count / len(self._calibration_data)
        
        # 结合预测置信度和经验准确率
        reliability = 0.7 * confidence + 0.3 * empirical_accuracy
        
        return reliability
    
    def _compute_calibration_error(self) -> float:
        """计算期望校准误差 (ECE)"""
        if len(self._calibration_data) < 10:
            return 0.0
        
        # 分桶计算
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            mask = [(bin_edges[i] <= conf < bin_edges[i+1]) 
                   for conf, _ in self._calibration_data]
            
            if sum(mask) > 0:
                bin_confs = [self._calibration_data[j][0] for j, m in enumerate(mask) if m]
                bin_corrects = [self._calibration_data[j][1] for j, m in enumerate(mask) if m]
                
                avg_conf = np.mean(bin_confs)
                avg_acc = np.mean(bin_corrects)
                
                ece += len(bin_confs) * abs(avg_conf - avg_acc)
        
        return ece / len(self._calibration_data)
    
    def _estimate_coverage(self, uncertainty: UncertaintyEstimate) -> float:
        """估计预测区间覆盖率"""
        # 基于历史数据估计
        if '95%' not in uncertainty.confidence_intervals:
            return 0.95  # 默认
        
        return 0.95
    
    def update_calibration(self, predicted_confidence: float, 
                          was_correct: bool) -> None:
        """更新校准数据"""
        self._calibration_data.append((predicted_confidence, was_correct))
        
        # 限制历史大小
        if len(self._calibration_data) > 1000:
            self._calibration_data = self._calibration_data[-1000:]
    
    def temperature_scaling(self, logits: NDArray[np.float64], 
                           labels: NDArray[np.int64]) -> float:
        """
        温度缩放校准
        
        学习最优温度参数来校准softmax概率
        """
        from scipy.optimize import minimize_scalar
        
        def nll_loss(T: float) -> float:
            scaled_logits = logits / T
            probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=-1, keepdims=True)
            
            # 负对数似然
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))
            return nll
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self._temperature = result.x
        
        return self._temperature


class UncertaintyPropagator:
    """
    不确定性传播器
    
    追踪不确定性在多步预测中的传播
    """
    
    def __init__(self, n_stages: int = 10):
        self.n_stages = n_stages
        self._uncertainty_history: List[UncertaintyEstimate] = []
    
    def propagate(self, initial_uncertainty: UncertaintyEstimate,
                  transition_model: Callable[[NDArray[np.float64]], NDArray[np.float64]],
                  n_steps: int) -> List[UncertaintyEstimate]:
        """
        传播不确定性
        
        使用非侵入式多项式混沌或Monte Carlo方法
        """
        uncertainties = [initial_uncertainty]
        current = initial_uncertainty
        
        for step in range(n_steps):
            # 使用扰动法传播
            samples = []
            n_mc = 100
            
            for _ in range(n_mc):
                # 从当前分布采样
                sample = current.mean + np.random.randn(len(current.mean)) * current.std
                
                # 通过转移模型
                next_sample = transition_model(sample)
                samples.append(next_sample)
            
            # 计算新分布
            samples_array = np.array(samples)
            new_mean = np.mean(samples_array, axis=0)
            new_std = np.std(samples_array, axis=0)
            new_var = np.var(samples_array, axis=0)
            
            current = UncertaintyEstimate(
                mean=new_mean,
                std=new_std,
                variance=new_var,
                confidence_intervals={
                    '95%': (
                        np.percentile(samples_array, 2.5, axis=0),
                        np.percentile(samples_array, 97.5, axis=0)
                    )
                }
            )
            
            uncertainties.append(current)
        
        self._uncertainty_history = uncertainties
        return uncertainties
    
    def compute_growth_rate(self) -> NDArray[np.float64]:
        """计算不确定性的增长率"""
        if len(self._uncertainty_history) < 2:
            return np.array([0.0])
        
        variances = np.array([u.variance for u in self._uncertainty_history])
        
        # 计算每一步的方差增长率
        growth_rates = np.diff(variances, axis=0) / (variances[:-1] + 1e-10)
        
        return np.mean(growth_rates, axis=0)


class UQEngine:
    """
    不确定性量化引擎
    
    整合多种UQ方法的统一接口
    """
    
    def __init__(self):
        self.quantifiers: Dict[UQMethod, UncertaintyQuantifier] = {}
        self.confidence_estimator = ConfidenceEstimator()
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.propagator = UncertaintyPropagator()
        
        # 默认添加一些量化器
        self.quantifiers[UQMethod.MONTE_CARLO] = MonteCarloUQ()
        self.quantifiers[UQMethod.BOOTSTRAP] = BootstrapUQ()
        self.quantifiers[UQMethod.ENSEMBLE] = EnsembleUQ()
    
    def quantify(self, predictions: List[NDArray[np.float64]],
                method: UQMethod = UQMethod.ENSEMBLE,
                **kwargs) -> UncertaintyEstimate:
        """使用指定方法量化不确定性"""
        if method not in self.quantifiers:
            raise ValueError(f"Method {method} not available")
        
        return self.quantifiers[method].quantify(predictions, **kwargs)
    
    def multi_method_quantify(self, predictions: List[NDArray[np.float64]]) -> Dict[UQMethod, UncertaintyEstimate]:
        """使用多种方法量化并比较"""
        results = {}
        
        for method, quantifier in self.quantifiers.items():
            try:
                results[method] = quantifier.quantify(predictions)
            except Exception as e:
                print(f"Method {method.value} failed: {e}")
        
        return results
    
    def assess_confidence(self, prediction: NDArray[np.float64],
                         uncertainty: UncertaintyEstimate) -> ConfidenceAssessment:
        """评估置信度"""
        return self.confidence_estimator.estimate(prediction, uncertainty)
    
    def analyze_sensitivity(self, model_fn: Callable[..., float],
                           parameters: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
        """
        敏感性分析
        
        Args:
            model_fn: 模型函数
            parameters: {name: (base, min, max)}
        """
        for name, (base, min_val, max_val) in parameters.items():
            self.sensitivity_analyzer.set_parameter(name, base, min_val, max_val)
        
        return self.sensitivity_analyzer.sobol_analysis(model_fn)


def demo():
    """演示不确定性量化功能"""
    print("=" * 60)
    print("不确定性量化和置信度评估演示")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 创建引擎
    engine = UQEngine()
    
    # 生成预测样本 (模拟集成模型预测)
    print("\n1. 生成模拟预测数据")
    n_models = 10
    n_dims = 5
    
    true_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = []
    
    for i in range(n_models):
        # 每个模型有不同的偏差和噪声
        bias = np.random.randn(n_dims) * 0.1
        noise = np.random.randn(n_dims) * 0.2
        pred = true_value + bias + noise
        predictions.append(pred)
    
    print(f"   模型数量: {n_models}")
    print(f"   输出维度: {n_dims}")
    
    # 使用多种方法量化不确定性
    print(f"\n2. 多方法不确定性量化")
    results = engine.multi_method_quantify(predictions)
    
    for method, uq in results.items():
        print(f"\n   {method.value}:")
        print(f"      均值: {uq.mean}")
        print(f"      标准差: {uq.std}")
        print(f"      变异系数: {uq.coefficient_of_variation}")
    
    # 使用Ensemble方法进行详细分析
    print(f"\n3. 详细不确定性分析 (Ensemble方法)")
    uq = results[UQMethod.ENSEMBLE]
    
    print(f"   相对不确定性: {uq.relative_uncertainty}")
    print(f"   95% 置信区间:")
    lower, upper = uq.confidence_intervals['95%']
    for i in range(n_dims):
        print(f"      维度{i}: [{lower[i]:.3f}, {upper[i]:.3f}]")
    
    # 置信度评估
    print(f"\n4. 置信度评估")
    mean_pred = uq.mean
    confidence = engine.assess_confidence(mean_pred, uq)
    
    print(f"   置信度分数: {confidence.confidence_score:.3f}")
    print(f"   可靠性: {confidence.reliability:.3f}")
    print(f"   校准误差: {confidence.calibration_error:.3f}")
    print(f"   预测区间覆盖率: {confidence.prediction_interval_coverage:.3f}")
    print(f"   是否可靠: {confidence.is_reliable}")
    
    # 敏感性分析
    print(f"\n5. 敏感性分析")
    
    def test_model(x1, x2, x3):
        # 测试函数: y = 2*x1 + x2^2 + 0.1*x3 + noise
        return 2*x1 + x2**2 + 0.1*x3 + np.random.randn() * 0.01
    
    parameters = {
        'x1': (0.5, 0.0, 1.0),
        'x2': (0.5, 0.0, 1.0),
        'x3': (0.5, 0.0, 1.0)
    }
    
    sensitivity = engine.analyze_sensitivity(test_model, parameters)
    print("   Sobol敏感性指数:")
    for param, index in sorted(sensitivity.items(), key=lambda x: x[1], reverse=True):
        print(f"      {param}: {index:.4f}")
    
    # 不确定性传播
    print(f"\n6. 不确定性传播")
    
    # 定义简单的转移模型
    def transition(x):
        return x * 0.95 + np.random.randn(len(x)) * 0.1
    
    propagator = UncertaintyPropagator()
    propagated = propagator.propagate(uq, transition, n_steps=5)
    
    for i, unc in enumerate(propagated[:3]):
        print(f"   步骤 {i}: 均值标准差 = {np.mean(unc.std):.4f}")
    
    growth_rate = propagator.compute_growth_rate()
    print(f"   平均不确定性增长率: {np.mean(growth_rate):.4f}")
    
    # 贝叶斯不确定性量化
    print(f"\n7. 贝叶斯不确定性量化")
    bayesian = BayesianUQ(n_posterior_samples=500)
    bayesian.set_prior(mean=np.zeros(3), std=1.0)
    
    # 定义似然函数
    def likelihood(params):
        # 简单的似然: 接近真实值概率更高
        diff = np.sum((params - np.array([1.0, 0.5, -0.5]))**2)
        return -0.5 * diff
    
    print("   近似后验分布...")
    bayesian.approximate_posterior(likelihood, n_iterations=2000)
    bayes_uq = bayesian.quantify()
    
    print(f"   后验均值: {bayes_uq.mean}")
    print(f"   后验标准差: {bayes_uq.std}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return engine


if __name__ == "__main__":
    demo()
