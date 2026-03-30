"""
Statistical Analyzer
====================
统计分析工具

提供:
- MAE, RMSE, R² 计算
- 一致性检验 (Kolmogorov-Smirnov, Chi-square)
- 假设检验
- 异常值检测
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    统计分析器
    
    用于计算计算结果与实验数据的统计差异
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.config.setdefault('confidence_level', 0.95)
        self.config.setdefault('outlier_threshold', 3.0)  # 标准差倍数
    
    def calculate_errors(self,
                        experimental: np.ndarray,
                        predicted: np.ndarray) -> Dict[str, float]:
        """
        计算基本误差指标
        
        Args:
            experimental: 实验值数组
            predicted: 预测值数组
            
        Returns:
            误差指标字典
        """
        # 确保数组长度相同
        min_len = min(len(experimental), len(predicted))
        exp = experimental[:min_len]
        pred = predicted[:min_len]
        
        # 残差
        residuals = exp - pred
        
        # 基本指标
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        
        # 相对误差
        mask = exp != 0
        if np.any(mask):
            mape = np.mean(np.abs(residuals[mask] / exp[mask])) * 100
            mspe = np.mean((residuals[mask] / exp[mask])**2) * 100
            rmspe = np.sqrt(mspe)
        else:
            mape = np.inf
            mspe = np.inf
            rmspe = np.inf
        
        # 最大误差
        max_error = np.max(np.abs(residuals))
        max_error_idx = np.argmax(np.abs(residuals))
        
        # R² 决定系数
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((exp - np.mean(exp))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 调整R²
        n = len(exp)
        p = 1  # 假设1个参数
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        # Pearson相关系数
        pearson_r, pearson_p = stats.pearsonr(exp, pred)
        
        # Spearman相关系数
        spearman_r, spearman_p = stats.spearmanr(exp, pred)
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape_percent': float(mape),
            'mspe_percent': float(mspe),
            'rmspe_percent': float(rmspe),
            'max_error': float(max_error),
            'max_error_index': int(max_error_idx),
            'r2': float(r2),
            'adjusted_r2': float(adj_r2),
            'pearson_r': float(pearson_r),
            'pearson_p_value': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p_value': float(spearman_p),
            'n_samples': n,
        }
    
    def kolmogorov_smirnov_test(self,
                                experimental: np.ndarray,
                                predicted: np.ndarray) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov检验
        
        检验两个样本是否来自同一分布
        
        Args:
            experimental: 实验数据
            predicted: 预测数据
            
        Returns:
            KS检验结果
        """
        statistic, p_value = stats.ks_2samp(experimental, predicted)
        
        alpha = 1 - self.config['confidence_level']
        reject_null = p_value < alpha
        
        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': alpha,
            'reject_null': reject_null,
            'interpretation': 'Distributions are different' if reject_null else 'Distributions are similar',
        }
    
    def chi_square_test(self,
                       experimental: np.ndarray,
                       predicted: np.ndarray,
                       bins: int = 10) -> Dict[str, Any]:
        """
        卡方检验
        
        检验观测值与预测值的一致性
        
        Args:
            experimental: 实验数据
            predicted: 预测数据
            bins: 分组数
            
        Returns:
            卡方检验结果
        """
        # 创建直方图
        min_val = min(np.min(experimental), np.min(predicted))
        max_val = max(np.max(experimental), np.max(predicted))
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        exp_hist, _ = np.histogram(experimental, bins=bin_edges)
        pred_hist, _ = np.histogram(predicted, bins=bin_edges)
        
        # 卡方检验
        # 避免除以零
        mask = exp_hist > 0
        if np.any(mask):
            chi2 = np.sum((exp_hist[mask] - pred_hist[mask])**2 / exp_hist[mask])
            dof = np.sum(mask) - 1
            p_value = 1 - stats.chi2.cdf(chi2, dof)
        else:
            chi2 = np.inf
            p_value = 0
            dof = 0
        
        alpha = 1 - self.config['confidence_level']
        
        return {
            'test': 'Chi-square',
            'chi2_statistic': float(chi2),
            'degrees_of_freedom': int(dof),
            'p_value': float(p_value),
            'alpha': alpha,
            'reject_null': p_value < alpha,
        }
    
    def paired_t_test(self,
                     experimental: np.ndarray,
                     predicted: np.ndarray) -> Dict[str, Any]:
        """
        配对t检验
        
        检验实验值和预测值的均值差异是否显著
        
        Args:
            experimental: 实验数据
            predicted: 预测数据
            
        Returns:
            t检验结果
        """
        statistic, p_value = stats.ttest_rel(experimental, predicted)
        
        alpha = 1 - self.config['confidence_level']
        
        return {
            'test': 'Paired t-test',
            't_statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'interpretation': 'Means are significantly different' if p_value < alpha else 'Means are not significantly different',
        }
    
    def bland_altman_analysis(self,
                             experimental: np.ndarray,
                             predicted: np.ndarray) -> Dict[str, Any]:
        """
        Bland-Altman分析
        
        评估两种测量方法的一致性
        
        Args:
            experimental: 实验数据
            predicted: 预测数据
            
        Returns:
            Bland-Altman分析结果
        """
        # 计算差异和平均值
        differences = experimental - predicted
        means = (experimental + predicted) / 2
        
        # 统计
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # 一致性界限 (LoA)
        loa_lower = mean_diff - 1.96 * std_diff
        loa_upper = mean_diff + 1.96 * std_diff
        
        # 95%置信区间
        n = len(differences)
        se_diff = std_diff / np.sqrt(n)
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        return {
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'limits_of_agreement': {
                'lower': float(loa_lower),
                'upper': float(loa_upper),
            },
            'confidence_interval': {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
            },
            'n_samples': n,
        }
    
    def detect_outliers(self,
                       experimental: np.ndarray,
                       predicted: np.ndarray,
                       method: str = 'residual') -> List[Dict]:
        """
        检测异常值
        
        Args:
            experimental: 实验数据
            predicted: 预测数据
            method: 'residual', 'zscore', 'iqr'
            
        Returns:
            异常值列表
        """
        residuals = np.abs(experimental - predicted)
        outliers = []
        
        if method == 'residual':
            # 基于残差标准差
            threshold = self.config['outlier_threshold'] * np.std(residuals)
            outlier_indices = np.where(residuals > threshold)[0]
            
        elif method == 'zscore':
            # Z-score方法
            z_scores = np.abs(stats.zscore(residuals))
            outlier_indices = np.where(z_scores > self.config['outlier_threshold'])[0]
            
        elif method == 'iqr':
            # IQR方法
            q1 = np.percentile(residuals, 25)
            q3 = np.percentile(residuals, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_indices = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        for idx in outlier_indices:
            outliers.append({
                'index': int(idx),
                'experimental': float(experimental[idx]),
                'predicted': float(predicted[idx]),
                'residual': float(residuals[idx]),
            })
        
        return outliers
    
    def calculate_confidence_intervals(self,
                                      data: np.ndarray,
                                      confidence: float = None) -> Dict[str, float]:
        """
        计算置信区间
        
        Args:
            data: 数据数组
            confidence: 置信水平 (默认使用config中的设置)
            
        Returns:
            置信区间
        """
        confidence = confidence or self.config['confidence_level']
        
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n)
        
        # t分布临界值
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        
        margin = t_critical * se
        
        return {
            'mean': float(mean),
            'std': float(std),
            'confidence_level': confidence,
            'confidence_interval': {
                'lower': float(mean - margin),
                'upper': float(mean + margin),
            },
            'margin_of_error': float(margin),
            'n_samples': n,
        }
    
    def bootstrap_analysis(self,
                          experimental: np.ndarray,
                          predicted: np.ndarray,
                          n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Bootstrap分析
        
        用于估计误差统计量的置信区间
        
        Args:
            experimental: 实验数据
            predicted: 预测数据
            n_bootstrap: Bootstrap迭代次数
            
        Returns:
            Bootstrap分析结果
        """
        n = len(experimental)
        rmse_bootstrap = []
        mae_bootstrap = []
        r2_bootstrap = []
        
        for _ in range(n_bootstrap):
            # 有放回抽样
            indices = np.random.choice(n, size=n, replace=True)
            exp_sample = experimental[indices]
            pred_sample = predicted[indices]
            
            # 计算指标
            residuals = exp_sample - pred_sample
            rmse_bootstrap.append(np.sqrt(np.mean(residuals**2)))
            mae_bootstrap.append(np.mean(np.abs(residuals)))
            
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((exp_sample - np.mean(exp_sample))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2_bootstrap.append(r2)
        
        # 计算置信区间
        confidence = self.config['confidence_level']
        alpha = 1 - confidence
        
        results = {
            'n_bootstrap': n_bootstrap,
            'rmse': {
                'mean': float(np.mean(rmse_bootstrap)),
                'std': float(np.std(rmse_bootstrap)),
                'ci_lower': float(np.percentile(rmse_bootstrap, alpha/2 * 100)),
                'ci_upper': float(np.percentile(rmse_bootstrap, (1 - alpha/2) * 100)),
            },
            'mae': {
                'mean': float(np.mean(mae_bootstrap)),
                'std': float(np.std(mae_bootstrap)),
                'ci_lower': float(np.percentile(mae_bootstrap, alpha/2 * 100)),
                'ci_upper': float(np.percentile(mae_bootstrap, (1 - alpha/2) * 100)),
            },
            'r2': {
                'mean': float(np.mean(r2_bootstrap)),
                'std': float(np.std(r2_bootstrap)),
                'ci_lower': float(np.percentile(r2_bootstrap, alpha/2 * 100)),
                'ci_upper': float(np.percentile(r2_bootstrap, (1 - alpha/2) * 100)),
            },
        }
        
        return results
    
    def residual_analysis(self,
                         experimental: np.ndarray,
                         predicted: np.ndarray) -> Dict[str, Any]:
        """
        残差分析
        
        分析残差的分布特性
        
        Args:
            experimental: 实验数据
            predicted: 预测数据
            
        Returns:
            残差分析结果
        """
        residuals = experimental - predicted
        
        # 正态性检验
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        # 自相关（简化）
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        
        # 异方差性检验（简化）
        # 将数据分成两半，比较方差
        mid = len(residuals) // 2
        var_first = np.var(residuals[:mid])
        var_second = np.var(residuals[mid:])
        
        return {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
            'normality_test': {
                'shapiro_statistic': float(shapiro_stat),
                'shapiro_p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05,
            },
            'autocorrelation': float(autocorr),
            'heteroscedasticity': {
                'variance_first_half': float(var_first),
                'variance_second_half': float(var_second),
                'ratio': float(var_second / var_first) if var_first > 0 else float('inf'),
            },
        }
    
    def comprehensive_analysis(self,
                              experimental: np.ndarray,
                              predicted: np.ndarray) -> Dict[str, Any]:
        """
        综合分析
        
        运行所有统计分析
        
        Args:
            experimental: 实验数据
            predicted: 预测数据
            
        Returns:
            完整的统计分析结果
        """
        return {
            'errors': self.calculate_errors(experimental, predicted),
            'ks_test': self.kolmogorov_smirnov_test(experimental, predicted),
            'chi_square_test': self.chi_square_test(experimental, predicted),
            't_test': self.paired_t_test(experimental, predicted),
            'bland_altman': self.bland_altman_analysis(experimental, predicted),
            'outliers': self.detect_outliers(experimental, predicted),
            'residual_analysis': self.residual_analysis(experimental, predicted),
            'bootstrap': self.bootstrap_analysis(experimental, predicted),
        }


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """便捷函数：计算MAE"""
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """便捷函数：计算RMSE"""
    return np.sqrt(np.mean((y_true - y_pred)**2))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """便捷函数：计算R²"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
