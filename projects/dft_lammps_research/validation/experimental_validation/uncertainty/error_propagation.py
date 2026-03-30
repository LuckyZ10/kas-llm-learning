"""
Error Propagation
=================
误差传播计算

用于计算计算结果的不确定性和误差传播
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging

logger = logging.getLogger(__name__)


class ErrorPropagator:
    """
    误差传播计算器
    
    基于泰勒展开和蒙特卡洛方法计算误差传播
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.config.setdefault('n_monte_carlo', 10000)
        self.config.setdefault('confidence_level', 0.95)
    
    def linear_propagation(self,
                          values: np.ndarray,
                          uncertainties: np.ndarray,
                          jacobian: np.ndarray) -> Tuple[float, float]:
        """
        线性误差传播
        
        对于 y = f(x)，误差传播公式:
        σ_y² = J · Σ · Jᵀ
        
        其中 J 是雅可比矩阵，Σ 是协方差矩阵
        
        Args:
            values: 输入值
            uncertainties: 输入不确定度（标准差）
            jacobian: 雅可比矩阵 ∂f/∂x
            
        Returns:
            (输出值, 输出不确定度)
        """
        # 计算输出值（假设线性函数 y = J·x）
        output_value = np.dot(jacobian, values)
        
        # 计算输出方差
        # 简化：假设输入变量不相关
        variance = np.sum((jacobian * uncertainties)**2)
        output_uncertainty = np.sqrt(variance)
        
        return float(output_value), float(output_uncertainty)
    
    def monte_carlo_propagation(self,
                               func: Callable,
                               values: np.ndarray,
                               uncertainties: np.ndarray,
                               n_samples: int = None,
                               distribution: str = 'normal') -> Dict[str, float]:
        """
        蒙特卡洛误差传播
        
        通过对输入参数进行随机采样，计算输出的分布
        
        Args:
            func: 目标函数 f(x) -> y
            values: 输入值
            uncertainties: 输入不确定度
            n_samples: 采样数
            distribution: 分布类型 ('normal', 'uniform')
            
        Returns:
            输出统计量
        """
        n_samples = n_samples or self.config['n_monte_carlo']
        
        n_params = len(values)
        samples = np.zeros((n_samples, n_params))
        
        # 生成样本
        if distribution == 'normal':
            for i in range(n_params):
                samples[:, i] = np.random.normal(values[i], uncertainties[i], n_samples)
        elif distribution == 'uniform':
            for i in range(n_params):
                samples[:, i] = np.random.uniform(
                    values[i] - uncertainties[i] * np.sqrt(3),
                    values[i] + uncertainties[i] * np.sqrt(3),
                    n_samples
                )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # 计算输出
        outputs = np.array([func(sample) for sample in samples])
        
        # 计算统计量
        mean = np.mean(outputs)
        std = np.std(outputs, ddof=1)
        
        confidence = self.config['confidence_level']
        ci_lower = np.percentile(outputs, (1 - confidence) / 2 * 100)
        ci_upper = np.percentile(outputs, (1 + confidence) / 2 * 100)
        
        return {
            'mean': float(mean),
            'std': float(std),
            'variance': float(std**2),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'median': float(np.median(outputs)),
            'min': float(np.min(outputs)),
            'max': float(np.max(outputs)),
            'skewness': float(self._calculate_skewness(outputs)),
            'kurtosis': float(self._calculate_kurtosis(outputs)),
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return np.sum(((data - mean) / std)**3) * n / ((n-1) * (n-2))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return np.sum(((data - mean) / std)**4) / n - 3
    
    def finite_difference_jacobian(self,
                                   func: Callable,
                                   values: np.ndarray,
                                   step: float = 1e-6) -> np.ndarray:
        """
        有限差分计算雅可比矩阵
        
        Args:
            func: 目标函数
            values: 输入值
            step: 差分步长
            
        Returns:
            雅可比矩阵
        """
        n = len(values)
        f0 = func(values)
        
        if np.isscalar(f0):
            jacobian = np.zeros(n)
            for i in range(n):
                values_plus = values.copy()
                values_plus[i] += step
                f_plus = func(values_plus)
                jacobian[i] = (f_plus - f0) / step
        else:
            m = len(f0)
            jacobian = np.zeros((m, n))
            for i in range(n):
                values_plus = values.copy()
                values_plus[i] += step
                f_plus = func(values_plus)
                jacobian[:, i] = (f_plus - f0) / step
        
        return jacobian
    
    def correlated_propagation(self,
                              values: np.ndarray,
                              covariance: np.ndarray,
                              jacobian: np.ndarray) -> Tuple[float, float]:
        """
        相关变量的误差传播
        
        σ_y² = J · Σ · Jᵀ
        
        Args:
            values: 输入值
            covariance: 协方差矩阵
            jacobian: 雅可比矩阵
            
        Returns:
            (输出值, 输出不确定度)
        """
        output_value = np.dot(jacobian, values)
        
        # 计算输出方差（考虑相关性）
        output_variance = np.dot(jacobian, np.dot(covariance, jacobian.T))
        output_uncertainty = np.sqrt(output_variance)
        
        return float(output_value), float(output_uncertainty)
    
    def covariance_from_correlation(self,
                                   uncertainties: np.ndarray,
                                   correlation_matrix: np.ndarray) -> np.ndarray:
        """
        从不确定度和相关系数矩阵计算协方差矩阵
        
        Σ_ij = σ_i · σ_j · ρ_ij
        
        Args:
            uncertainties: 标准差数组
            correlation_matrix: 相关系数矩阵
            
        Returns:
            协方差矩阵
        """
        sigma_matrix = np.outer(uncertainties, uncertainties)
        covariance = sigma_matrix * correlation_matrix
        return covariance


class ConfidenceIntervalEstimator:
    """
    置信区间估计器
    
    计算各种统计量的置信区间
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.config.setdefault('confidence_level', 0.95)
        self.config.setdefault('n_bootstrap', 1000)
    
    def mean_ci(self, data: np.ndarray, confidence: float = None) -> Tuple[float, float, float]:
        """
        均值的置信区间（t分布）
        
        Args:
            data: 数据数组
            confidence: 置信水平
            
        Returns:
            (均值, CI下限, CI上限)
        """
        from scipy import stats
        
        confidence = confidence or self.config['confidence_level']
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n)
        
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        
        margin = t_critical * se
        
        return mean, mean - margin, mean + margin
    
    def proportion_ci(self, successes: int, trials: int, 
                     confidence: float = None,
                     method: str = 'wilson') -> Tuple[float, float, float]:
        """
        比例的置信区间
        
        Args:
            successes: 成功次数
            trials: 总试验次数
            confidence: 置信水平
            method: 'wilson', 'normal', 'agresti_coull'
            
        Returns:
            (比例, CI下限, CI上限)
        """
        from scipy import stats
        
        confidence = confidence or self.config['confidence_level']
        alpha = 1 - confidence
        
        p = successes / trials if trials > 0 else 0
        
        if method == 'normal':
            z = stats.norm.ppf(1 - alpha/2)
            se = np.sqrt(p * (1 - p) / trials)
            margin = z * se
            ci_lower = max(0, p - margin)
            ci_upper = min(1, p + margin)
            
        elif method == 'wilson':
            z = stats.norm.ppf(1 - alpha/2)
            denominator = 1 + z**2 / trials
            centre = (p + z**2 / (2 * trials)) / denominator
            margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
            ci_lower = max(0, centre - margin)
            ci_upper = min(1, centre + margin)
            
        elif method == 'agresti_coull':
            z = stats.norm.ppf(1 - alpha/2)
            n_tilde = trials + z**2
            p_tilde = (successes + z**2 / 2) / n_tilde
            se = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
            margin = z * se
            ci_lower = max(0, p_tilde - margin)
            ci_upper = min(1, p_tilde + margin)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return p, ci_lower, ci_upper
    
    def bootstrap_ci(self, data: np.ndarray,
                    statistic_func: Callable = np.mean,
                    confidence: float = None,
                    n_bootstrap: int = None) -> Tuple[float, float, float]:
        """
        Bootstrap置信区间
        
        Args:
            data: 数据数组
            statistic_func: 统计量函数
            confidence: 置信水平
            n_bootstrap: Bootstrap次数
            
        Returns:
            (统计量, CI下限, CI上限)
        """
        confidence = confidence or self.config['confidence_level']
        n_bootstrap = n_bootstrap or self.config['n_bootstrap']
        
        n = len(data)
        bootstrap_statistics = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_statistics.append(statistic_func(sample))
        
        alpha = 1 - confidence
        stat_value = statistic_func(data)
        ci_lower = np.percentile(bootstrap_statistics, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_statistics, (1 - alpha/2) * 100)
        
        return stat_value, ci_lower, ci_upper
    
    def prediction_interval(self, 
                           x: np.ndarray,
                           y: np.ndarray,
                           x_new: float,
                           confidence: float = None) -> Tuple[float, float, float]:
        """
        线性回归的预测区间
        
        Args:
            x: 自变量
            y: 因变量
            x_new: 新的预测点
            confidence: 置信水平
            
        Returns:
            (预测值, PI下限, PI上限)
        """
        from scipy import stats
        
        confidence = confidence or self.config['confidence_level']
        
        n = len(x)
        
        # 线性回归
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x_new + intercept
        
        # 残差标准差
        y_fit = slope * x + intercept
        residuals = y - y_fit
        s = np.sqrt(np.sum(residuals**2) / (n - 2))
        
        # 预测区间
        x_mean = np.mean(x)
        ssx = np.sum((x - x_mean)**2)
        
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df=n-2)
        
        margin = t_critical * s * np.sqrt(1 + 1/n + (x_new - x_mean)**2 / ssx)
        
        return y_pred, y_pred - margin, y_pred + margin
    
    def multiple_comparison_correction(self,
                                      p_values: List[float],
                                      method: str = 'bonferroni') -> List[float]:
        """
        多重比较校正
        
        Args:
            p_values: p值列表
            method: 'bonferroni', 'holm', 'fdr_bh'
            
        Returns:
            校正后的p值
        """
        from scipy import stats
        
        p_values = np.array(p_values)
        m = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni校正
            corrected = np.minimum(p_values * m, 1.0)
            
        elif method == 'holm':
            # Holm-Bonferroni校正
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            corrected_sorted = np.zeros(m)
            for i, p in enumerate(sorted_p):
                corrected_sorted[i] = min(p * (m - i), 1.0)
            
            # 确保单调性
            for i in range(m-1, 0, -1):
                corrected_sorted[i-1] = min(corrected_sorted[i-1], corrected_sorted[i])
            
            corrected = np.zeros(m)
            corrected[sorted_idx] = corrected_sorted
            
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR控制
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            corrected_sorted = np.zeros(m)
            for i, p in enumerate(sorted_p):
                corrected_sorted[i] = p * m / (i + 1)
            
            # 确保单调性
            for i in range(m-1, 0, -1):
                corrected_sorted[i-1] = min(corrected_sorted[i-1], corrected_sorted[i])
            
            corrected = np.minimum(corrected_sorted, 1.0)
            
            corrected_final = np.zeros(m)
            corrected_final[sorted_idx] = corrected
            corrected = corrected_final
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return corrected.tolist()


class SensitivityAnalyzer:
    """
    敏感性分析器
    
    分析输出对输入参数的敏感程度
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.config.setdefault('perturbation_fraction', 0.01)  # 1%扰动
    
    def local_sensitivity(self,
                         func: Callable,
                         values: np.ndarray,
                         parameter_names: List[str] = None) -> Dict[str, Any]:
        """
        局部敏感性分析（一阶导数）
        
        S_i = ∂f/∂x_i · Δx_i / Δf
        
        Args:
            func: 目标函数
            values: 输入值
            parameter_names: 参数名称
            
        Returns:
            敏感性指标
        """
        n = len(values)
        parameter_names = parameter_names or [f"param_{i}" for i in range(n)]
        
        f0 = func(values)
        
        sensitivities = {}
        normalized_sensitivities = {}
        
        for i in range(n):
            # 扰动
            delta = values[i] * self.config['perturbation_fraction']
            if delta == 0:
                delta = 1e-6
            
            values_plus = values.copy()
            values_plus[i] += delta
            
            f_plus = func(values_plus)
            
            # 敏感性系数
            sensitivity = (f_plus - f0) / delta
            sensitivities[parameter_names[i]] = float(sensitivity)
            
            # 归一化敏感性（弹性系数）
            if f0 != 0 and values[i] != 0:
                elastic = (f_plus - f0) / f0 / (delta / values[i])
                normalized_sensitivities[parameter_names[i]] = float(elastic)
        
        return {
            'sensitivities': sensitivities,
            'elasticities': normalized_sensitivities,
            'most_sensitive': max(sensitivities, key=lambda k: abs(sensitivities[k])),
        }
    
    def sobol_analysis(self,
                      func: Callable,
                      bounds: List[Tuple[float, float]],
                      n_samples: int = 1024) -> Dict[str, Any]:
        """
        Sobol全局敏感性分析
        
        基于方差的敏感性分析
        
        Args:
            func: 目标函数
            bounds: 参数范围 [(min, max), ...]
            n_samples: 样本数
            
        Returns:
            Sobol指数
        """
        try:
            from SALib.sample import saltelli
            from SALib.analyze import sobol
        except ImportError:
            logger.warning("SALib not available for Sobol analysis")
            return {'error': 'SALib not installed'}
        
        n_params = len(bounds)
        parameter_names = [f"param_{i}" for i in range(n_params)]
        
        # 定义问题
        problem = {
            'num_vars': n_params,
            'names': parameter_names,
            'bounds': bounds
        }
        
        # 生成样本
        param_values = saltelli.sample(problem, n_samples)
        
        # 评估函数
        outputs = np.array([func(params) for params in param_values])
        
        # 分析
        sobol_indices = sobol.analyze(problem, outputs)
        
        return {
            'S1': dict(zip(parameter_names, sobol_indices['S1'])),
            'ST': dict(zip(parameter_names, sobol_indices['ST'])),
            'S1_conf': dict(zip(parameter_names, sobol_indices['S1_conf'])),
            'ST_conf': dict(zip(parameter_names, sobol_indices['ST_conf'])),
        }
    
    def morris_screening(self,
                        func: Callable,
                        bounds: List[Tuple[float, float]],
                        n_trajectories: int = 10,
                        n_levels: int = 4) -> Dict[str, Any]:
        """
        Morris筛选法
        
        计算基本效应的统计量
        
        Args:
            func: 目标函数
            bounds: 参数范围
            n_trajectories: 轨迹数
            n_levels: 水平数
            
        Returns:
            Morris指标
        """
        try:
            from SALib.sample import morris as morris_sample
            from SALib.analyze import morris as morris_analyze
        except ImportError:
            logger.warning("SALib not available for Morris analysis")
            return {'error': 'SALib not installed'}
        
        n_params = len(bounds)
        parameter_names = [f"param_{i}" for i in range(n_params)]
        
        problem = {
            'num_vars': n_params,
            'names': parameter_names,
            'bounds': bounds
        }
        
        # 生成样本
        param_values = morris_sample.sample(problem, n_trajectories, num_levels=n_levels)
        
        # 评估函数
        outputs = np.array([func(params) for params in param_values])
        
        # 分析
        morris_indices = morris_analyze.analyze(problem, param_values, outputs)
        
        return {
            'mu': dict(zip(parameter_names, morris_indices['mu'])),
            'mu_star': dict(zip(parameter_names, morris_indices['mu_star'])),
            'sigma': dict(zip(parameter_names, morris_indices['sigma'])),
            'mu_star_conf': dict(zip(parameter_names, morris_indices['mu_star_conf'])),
        }
    
    def correlation_analysis(self,
                            parameters: np.ndarray,
                            outputs: np.ndarray) -> Dict[str, float]:
        """
        相关性分析
        
        计算参数与输出的相关系数
        
        Args:
            parameters: 参数矩阵 (n_samples, n_params)
            outputs: 输出数组 (n_samples,)
            
        Returns:
            相关系数字典
        """
        from scipy.stats import pearsonr, spearmanr
        
        n_params = parameters.shape[1]
        
        pearson_corr = []
        pearson_p = []
        spearman_corr = []
        spearman_p = []
        
        for i in range(n_params):
            r, p = pearsonr(parameters[:, i], outputs)
            pearson_corr.append(r)
            pearson_p.append(p)
            
            rho, p = spearmanr(parameters[:, i], outputs)
            spearman_corr.append(rho)
            spearman_p.append(p)
        
        return {
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_corr,
            'spearman_p': spearman_p,
        }
    
    def tornado_diagram_data(self,
                            func: Callable,
                            values: np.ndarray,
                            uncertainties: np.ndarray,
                            parameter_names: List[str] = None) -> Dict[str, List]:
        """
        生成龙卷风图数据
        
        显示各参数变化对输出的影响
        
        Args:
            func: 目标函数
            values: 输入值
            uncertainties: 不确定度
            parameter_names: 参数名称
            
        Returns:
            龙卷风图数据
        """
        n = len(values)
        parameter_names = parameter_names or [f"param_{i}" for i in range(n)]
        
        f0 = func(values)
        
        low_values = []
        high_values = []
        
        for i in range(n):
            # 低值影响
            values_low = values.copy()
            values_low[i] -= uncertainties[i]
            f_low = func(values_low)
            
            # 高值影响
            values_high = values.copy()
            values_high[i] += uncertainties[i]
            f_high = func(values_high)
            
            low_values.append(f_low - f0)
            high_values.append(f_high - f0)
        
        return {
            'parameters': parameter_names,
            'low_impact': low_values,
            'high_impact': high_values,
            'base_value': f0,
        }
