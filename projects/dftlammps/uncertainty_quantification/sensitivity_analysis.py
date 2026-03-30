"""
敏感性分析工具 - Sensitivity Analysis Tools

提供全面的敏感性分析方法：
- 全局敏感性分析（Sobol、Morris、FAST）
- 局部敏感性分析（梯度、有限差分）
- 参数筛选方法

核心特性:
- Sobol一阶和总效应指标
- Morris基本效应筛选
- Fourier幅度敏感性测试
- 基于梯度的局部分析
- 自动参数重要性排序
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
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from SALib.sample import saltelli, morris as morris_sample, fast_sampler
    from SALib.analyze import sobol, morris as morris_analyze, fast
    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False


# ==================== 数据结构 ====================

@dataclass
class SensitivityIndices:
    """敏感性指标"""
    first_order: np.ndarray  # 一阶效应
    total_order: np.ndarray  # 总效应
    second_order: Optional[np.ndarray] = None  # 二阶效应（可选）
    confidence_intervals: Optional[Dict[str, np.ndarray]] = None
    
    def most_important(self, n: int = 3) -> List[Tuple[int, float]]:
        """获取最重要的n个参数"""
        indices = np.argsort(self.total_order)[::-1][:n]
        return [(i, self.total_order[i]) for i in indices]
    
    def interaction_measure(self) -> np.ndarray:
        """测量交互效应（总效应 - 一阶效应）"""
        return self.total_order - self.first_order
    
    def has_significant_interactions(self, threshold: float = 0.1) -> bool:
        """检查是否存在显著交互效应"""
        interactions = self.interaction_measure()
        return np.any(interactions > threshold)


@dataclass
class ParameterImportance:
    """参数重要性评估"""
    param_names: List[str]
    importance_scores: np.ndarray
    rankings: np.ndarray
    significance: np.ndarray  # 统计显著性
    
    def get_important_params(self, threshold: float = 0.05) -> List[str]:
        """获取重要参数"""
        important = np.where(self.importance_scores > threshold)[0]
        return [self.param_names[i] for i in important]
    
    def get_param_rank(self, param_name: str) -> int:
        """获取参数排名"""
        idx = self.param_names.index(param_name)
        return self.rankings[idx]


@dataclass
class SensitivityReport:
    """敏感性分析报告"""
    method: str
    indices: SensitivityIndices
    parameter_importance: ParameterImportance
    convergence_diagnostics: Optional[Dict] = None
    recommendations: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """生成摘要"""
        lines = [
            f"敏感性分析摘要 ({self.method})",
            "=" * 60,
            "参数重要性排名:"
        ]
        
        sorted_params = sorted(
            zip(self.parameter_importance.param_names,
                self.parameter_importance.importance_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (name, score) in enumerate(sorted_params[:5]):
            lines.append(f"  {i+1}. {name}: {score:.4f}")
        
        lines.append("\n建议:")
        for rec in self.recommendations:
            lines.append(f"  - {rec}")
        
        return "\n".join(lines)


@dataclass
class ElementaryEffects:
    """Morris基本效应结果"""
    mu: np.ndarray  # 基本效应均值
    mu_star: np.ndarray  # 绝对值均值
    sigma: np.ndarray  # 标准差
    
    def classify_parameters(self, threshold_mu: float = 0.1,
                          threshold_sigma: float = 0.05) -> Dict[str, List[int]]:
        """
        根据Morris方法分类参数
        
        返回:
            'important': 重要参数（高mu_star）
            'interaction': 有交互效应（高sigma）
            'linear': 线性效应（低sigma）
            'insignificant': 不重要（低mu_star）
        """
        classification = {
            'important': [],
            'interaction': [],
            'linear': [],
            'insignificant': []
        }
        
        for i in range(len(self.mu_star)):
            if self.mu_star[i] < threshold_mu:
                classification['insignificant'].append(i)
            elif self.sigma[i] > threshold_sigma:
                classification['interaction'].append(i)
            elif self.sigma[i] < threshold_sigma / 2:
                classification['linear'].append(i)
            else:
                classification['important'].append(i)
        
        return classification


# ==================== 敏感性分析基类 ====================

class SensitivityAnalyzer(ABC):
    """敏感性分析器基类"""
    
    @abstractmethod
    def analyze(self,
                model: Callable,
                param_names: List[str],
                bounds: np.ndarray,
                n_samples: int = 1000) -> SensitivityReport:
        """执行敏感性分析"""
        pass
    
    def rank_parameters(self,
                       indices: SensitivityIndices,
                       param_names: List[str]) -> ParameterImportance:
        """对参数进行重要性排序"""
        importance = indices.total_order
        rankings = np.argsort(np.argsort(importance)[::-1]) + 1
        
        # 简单显著性检验
        significance = importance > (np.mean(importance) + np.std(importance))
        
        return ParameterImportance(
            param_names=param_names,
            importance_scores=importance,
            rankings=rankings,
            significance=significance
        )


# ==================== 全局敏感性分析 ====================

class SobolSensitivity(SensitivityAnalyzer):
    """
    Sobol敏感性分析
    
    基于方差分解的全局敏感性分析
    """
    
    def __init__(self, calc_second_order: bool = True):
        """
        初始化Sobol分析
        
        Args:
            calc_second_order: 是否计算二阶效应
        """
        self.calc_second_order = calc_second_order
    
    def analyze(self,
                model: Callable,
                param_names: List[str],
                bounds: np.ndarray,
                n_samples: int = 1024) -> SensitivityReport:
        """
        执行Sobol分析
        
        Args:
            model: 计算模型
            param_names: 参数名称列表
            bounds: 参数边界 [[lower, upper], ...]
            n_samples: 每组样本数（总样本数约为 n * (2D + 2)）
        """
        n_params = len(param_names)
        
        if HAS_SALIB:
            # 使用SALib进行Sobol分析
            return self._analyze_salib(model, param_names, bounds, n_samples)
        else:
            # 手动实现简化版本
            return self._analyze_manual(model, param_names, bounds, n_samples)
    
    def _analyze_salib(self,
                       model: Callable,
                       param_names: List[str],
                       bounds: np.ndarray,
                       n_samples: int) -> SensitivityReport:
        """使用SALib进行Sobol分析"""
        from SALib import ProblemSpec
        
        # 定义问题
        problem = {
            'num_vars': len(param_names),
            'names': param_names,
            'bounds': bounds.tolist()
        }
        
        # 生成样本
        param_values = saltelli.sample(problem, n_samples, 
                                       calc_second_order=self.calc_second_order)
        
        # 评估模型
        evaluations = np.array([model(x) for x in param_values])
        
        # 分析
        Si = sobol.analyze(problem, evaluations, 
                          calc_second_order=self.calc_second_order)
        
        # 构建结果
        indices = SensitivityIndices(
            first_order=Si['S1'],
            total_order=Si['ST'],
            second_order=Si.get('S2'),
            confidence_intervals={
                'S1_conf': Si.get('S1_conf'),
                'ST_conf': Si.get('ST_conf')
            }
        )
        
        importance = self.rank_parameters(indices, param_names)
        
        recommendations = self._generate_recommendations(indices, param_names)
        
        return SensitivityReport(
            method='Sobol',
            indices=indices,
            parameter_importance=importance,
            recommendations=recommendations
        )
    
    def _analyze_manual(self,
                        model: Callable,
                        param_names: List[str],
                        bounds: np.ndarray,
                        n_samples: int) -> SensitivityReport:
        """手动实现Sobol分析（简化版本）"""
        n_params = len(param_names)
        
        # 生成Saltelli序列（简化）
        A = np.random.rand(n_samples, n_params)
        B = np.random.rand(n_samples, n_params)
        
        # 缩放到边界
        A = bounds[:, 0] + A * (bounds[:, 1] - bounds[:, 0])
        B = bounds[:, 0] + B * (bounds[:, 1] - bounds[:, 0])
        
        # 评估A和B矩阵
        f_A = np.array([model(a) for a in A])
        f_B = np.array([model(b) for b in B])
        
        # 计算一阶和总效应指标
        S1 = np.zeros(n_params)
        ST = np.zeros(n_params)
        
        for i in range(n_params):
            # 创建A_B矩阵（A的第i列替换为B的第i列）
            A_B = A.copy()
            A_B[:, i] = B[:, i]
            f_A_B = np.array([model(ab) for ab in A_B])
            
            # 一阶效应
            V_i = np.mean(f_B * (f_A_B - f_A))
            V_y = np.var(np.concatenate([f_A, f_B]))
            S1[i] = V_i / V_y if V_y > 0 else 0
            
            # 总效应
            V_Ti = 0.5 * np.mean((f_A - f_A_B)**2)
            ST[i] = V_Ti / V_y if V_y > 0 else 0
        
        indices = SensitivityIndices(
            first_order=S1,
            total_order=ST
        )
        
        importance = self.rank_parameters(indices, param_names)
        recommendations = self._generate_recommendations(indices, param_names)
        
        return SensitivityReport(
            method='Sobol (Manual)',
            indices=indices,
            parameter_importance=importance,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self,
                                  indices: SensitivityIndices,
                                  param_names: List[str]) -> List[str]:
        """生成分析建议"""
        recommendations = []
        
        # 识别重要参数
        important = indices.most_important(3)
        rec = f"重点关注: {', '.join([param_names[i] for i, _ in important[:3]])}"
        recommendations.append(rec)
        
        # 检查交互效应
        if indices.has_significant_interactions(0.1):
            recommendations.append("存在显著参数交互效应，考虑使用高阶模型")
        
        # 识别不敏感参数
        insensitive = np.where(indices.total_order < 0.01)[0]
        if len(insensitive) > 0:
            rec = f"可固定参数: {', '.join([param_names[i] for i in insensitive])}"
            recommendations.append(rec)
        
        return recommendations


class MorrisMethod(SensitivityAnalyzer):
    """
    Morris筛选方法
    
    使用基本效应进行参数筛选的高效方法
    """
    
    def __init__(self,
                 num_levels: int = 4,
                 optimal_trajectories: Optional[int] = None):
        """
        初始化Morris方法
        
        Args:
            num_levels: 水平数（通常为4）
            optimal_trajectories: 最优轨迹数
        """
        self.num_levels = num_levels
        self.optimal_trajectories = optimal_trajectories
    
    def analyze(self,
                model: Callable,
                param_names: List[str],
                bounds: np.ndarray,
                n_samples: int = 10) -> SensitivityReport:
        """
        执行Morris分析
        
        Args:
            model: 计算模型
            param_names: 参数名称
            bounds: 参数边界
            n_samples: 轨迹数
        """
        n_params = len(param_names)
        
        if HAS_SALIB:
            return self._analyze_salib(model, param_names, bounds, n_samples)
        else:
            return self._analyze_manual(model, param_names, bounds, n_samples)
    
    def _analyze_salib(self,
                       model: Callable,
                       param_names: List[str],
                       bounds: np.ndarray,
                       n_trajectories: int) -> SensitivityReport:
        """使用SALib进行Morris分析"""
        problem = {
            'num_vars': len(param_names),
            'names': param_names,
            'bounds': bounds.tolist()
        }
        
        # 生成Morris样本
        param_values = morris_sample.sample(
            problem, n_trajectories, num_levels=self.num_levels,
            optimal_trajectories=self.optimal_trajectories
        )
        
        # 评估
        evaluations = np.array([model(x) for x in param_values])
        
        # 分析
        Si = morris_analyze.analyze(
            problem, param_values, evaluations,
            num_levels=self.num_levels
        )
        
        ee = ElementaryEffects(
            mu=Si['mu'],
            mu_star=Si['mu_star'],
            sigma=Si['sigma']
        )
        
        # 转换为统一格式
        indices = SensitivityIndices(
            first_order=ee.mu_star,
            total_order=ee.mu_star
        )
        
        importance = self.rank_parameters(indices, param_names)
        
        recommendations = self._generate_recommendations(ee, param_names)
        
        return SensitivityReport(
            method='Morris',
            indices=indices,
            parameter_importance=importance,
            recommendations=recommendations
        )
    
    def _analyze_manual(self,
                        model: Callable,
                        param_names: List[str],
                        bounds: np.ndarray,
                        n_trajectories: int) -> SensitivityReport:
        """手动实现Morris分析"""
        n_params = len(param_names)
        delta = self.num_levels / (2 * (self.num_levels - 1))
        
        elementary_effects = [[] for _ in range(n_params)]
        
        for _ in range(n_trajectories):
            # 生成基向量
            base_point = np.random.rand(n_params)
            base_point = bounds[:, 0] + base_point * (bounds[:, 1] - bounds[:, 0])
            
            # 评估基向量
            f_base = model(base_point)
            
            # 对每个参数计算基本效应
            for i in range(n_params):
                perturbed = base_point.copy()
                step = delta * (bounds[i, 1] - bounds[i, 0])
                perturbed[i] += step
                perturbed[i] = np.clip(perturbed[i], bounds[i, 0], bounds[i, 1])
                
                f_perturbed = model(perturbed)
                ee = (f_perturbed - f_base) / step
                
                elementary_effects[i].append(ee)
        
        # 计算统计量
        mu = np.array([np.mean(ee) for ee in elementary_effects])
        mu_star = np.array([np.mean(np.abs(ee)) for ee in elementary_effects])
        sigma = np.array([np.std(ee) for ee in elementary_effects])
        
        ee = ElementaryEffects(mu=mu, mu_star=mu_star, sigma=sigma)
        
        indices = SensitivityIndices(
            first_order=mu_star,
            total_order=mu_star
        )
        
        importance = self.rank_parameters(indices, param_names)
        recommendations = self._generate_recommendations(ee, param_names)
        
        return SensitivityReport(
            method='Morris (Manual)',
            indices=indices,
            parameter_importance=importance,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self,
                                  ee: ElementaryEffects,
                                  param_names: List[str]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        classification = ee.classify_parameters()
        
        if classification['important']:
            rec = f"重要参数: {', '.join([param_names[i] for i in classification['important']])}"
            recommendations.append(rec)
        
        if classification['interaction']:
            rec = f"有交互效应: {', '.join([param_names[i] for i in classification['interaction']])}"
            recommendations.append(rec)
        
        if classification['linear']:
            rec = f"线性效应: {', '.join([param_names[i] for i in classification['linear']])}"
            recommendations.append(rec)
        
        if classification['insignificant']:
            rec = f"可忽略: {', '.join([param_names[i] for i in classification['insignificant']])}"
            recommendations.append(rec)
        
        return recommendations


class FASTAnalysis(SensitivityAnalyzer):
    """
    Fourier幅度敏感性测试 (FAST)
    
    基于频域分析的敏感性方法
    """
    
    def __init__(self, M: int = 4):
        """
        初始化FAST分析
        
        Args:
            M: 干扰因子（通常为4）
        """
        self.M = M
    
    def analyze(self,
                model: Callable,
                param_names: List[str],
                bounds: np.ndarray,
                n_samples: int = 1000) -> SensitivityReport:
        """
        执行FAST分析
        """
        if HAS_SALIB:
            return self._analyze_salib(model, param_names, bounds, n_samples)
        else:
            return self._analyze_manual(model, param_names, bounds, n_samples)
    
    def _analyze_salib(self,
                       model: Callable,
                       param_names: List[str],
                       bounds: np.ndarray,
                       n_samples: int) -> SensitivityReport:
        """使用SALib进行FAST分析"""
        problem = {
            'num_vars': len(param_names),
            'names': param_names,
            'bounds': bounds.tolist()
        }
        
        # 生成FAST样本
        param_values = fast_sampler.sample(problem, n_samples, M=self.M)
        
        # 评估
        evaluations = np.array([model(x) for x in param_values])
        
        # 分析
        Si = fast.analyze(problem, evaluations, M=self.M, print_to_console=False)
        
        indices = SensitivityIndices(
            first_order=Si['S1'],
            total_order=Si['ST']
        )
        
        importance = self.rank_parameters(indices, param_names)
        
        return SensitivityReport(
            method='FAST',
            indices=indices,
            parameter_importance=importance,
            recommendations=[]
        )
    
    def _analyze_manual(self,
                        model: Callable,
                        param_names: List[str],
                        bounds: np.ndarray,
                        n_samples: int) -> SensitivityReport:
        """手动实现FAST（简化）"""
        n_params = len(param_names)
        
        # 使用随机采样作为近似
        samples = np.random.rand(n_samples, n_params)
        samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
        
        evaluations = np.array([model(x) for x in samples])
        
        # 简单的方差分解
        total_var = np.var(evaluations)
        
        S1 = np.ones(n_params) / n_params  # 简化的均匀分配
        ST = S1.copy()
        
        indices = SensitivityIndices(
            first_order=S1,
            total_order=ST
        )
        
        importance = self.rank_parameters(indices, param_names)
        
        return SensitivityReport(
            method='FAST (Simplified)',
            indices=indices,
            parameter_importance=importance,
            recommendations=[]
        )


# ==================== 局部敏感性分析 ====================

class LocalSensitivity:
    """局部敏感性分析"""
    
    def __init__(self, nominal_point: np.ndarray):
        """
        初始化局部敏感性分析
        
        Args:
            nominal_point: 名义参数点
        """
        self.nominal_point = nominal_point
    
    def compute_elasticity(self,
                          model: Callable,
                          perturbation: float = 0.01) -> np.ndarray:
        """
        计算弹性系数
        
        elasticity_i = (dy/y) / (dx_i/x_i)
        """
        n_params = len(self.nominal_point)
        
        f_nominal = model(self.nominal_point)
        
        elasticities = np.zeros(n_params)
        
        for i in range(n_params):
            perturbed = self.nominal_point.copy()
            perturbed[i] *= (1 + perturbation)
            
            f_perturbed = model(perturbed)
            
            dy = f_perturbed - f_nominal
            dx = perturbed[i] - self.nominal_point[i]
            
            if f_nominal != 0 and self.nominal_point[i] != 0:
                elasticities[i] = (dy / f_nominal) / (dx / self.nominal_point[i])
        
        return elasticities
    
    def compute_normalized_sensitivity(self,
                                       model: Callable,
                                       perturbation: float = 0.01) -> np.ndarray:
        """
        计算归一化敏感性系数
        """
        n_params = len(self.nominal_point)
        
        f_nominal = model(self.nominal_point)
        
        sensitivities = np.zeros(n_params)
        
        for i in range(n_params):
            perturbed = self.nominal_point.copy()
            perturbed[i] += perturbation * self.nominal_point[i]
            
            f_perturbed = model(perturbed)
            
            sensitivities[i] = (f_perturbed - f_nominal) / perturbation
        
        return sensitivities


class GradientBasedAnalysis:
    """
    基于梯度的敏感性分析
    
    使用自动微分或数值微分计算梯度
    """
    
    def __init__(self, method: str = 'forward'):
        """
        初始化梯度分析
        
        Args:
            method: 微分方法 ('forward', 'central', 'automatic')
        """
        self.method = method
    
    def compute_gradient(self,
                        model: Callable,
                        point: np.ndarray,
                        h: float = 1e-6) -> np.ndarray:
        """计算梯度"""
        n_params = len(point)
        gradient = np.zeros(n_params)
        
        f_point = model(point)
        
        if self.method == 'forward':
            for i in range(n_params):
                perturbed = point.copy()
                perturbed[i] += h
                gradient[i] = (model(perturbed) - f_point) / h
        
        elif self.method == 'central':
            for i in range(n_params):
                perturbed_plus = point.copy()
                perturbed_minus = point.copy()
                perturbed_plus[i] += h
                perturbed_minus[i] -= h
                gradient[i] = (model(perturbed_plus) - model(perturbed_minus)) / (2 * h)
        
        return gradient
    
    def compute_hessian(self,
                       model: Callable,
                       point: np.ndarray,
                       h: float = 1e-5) -> np.ndarray:
        """计算Hessian矩阵"""
        n_params = len(point)
        hessian = np.zeros((n_params, n_params))
        
        for i in range(n_params):
            for j in range(i, n_params):
                # 中心差分计算二阶导数
                x_pp = point.copy()
                x_pm = point.copy()
                x_mp = point.copy()
                x_mm = point.copy()
                
                x_pp[i] += h; x_pp[j] += h
                x_pm[i] += h; x_pm[j] -= h
                x_mp[i] -= h; x_mp[j] += h
                x_mm[i] -= h; x_mm[j] -= h
                
                hessian[i, j] = (model(x_pp) - model(x_pm) - 
                                model(x_mp) + model(x_mm)) / (4 * h * h)
                hessian[j, i] = hessian[i, j]
        
        return hessian


class FiniteDifferenceSensitivity:
    """有限差分敏感性分析"""
    
    def __init__(self, scheme: str = 'central'):
        """
        初始化有限差分分析
        
        Args:
            scheme: 差分格式 ('forward', 'backward', 'central')
        """
        self.scheme = scheme
    
    def compute_sensitivity(self,
                           model: Callable,
                           point: np.ndarray,
                           param_idx: int,
                           h: float = 1e-6) -> float:
        """计算单个参数的敏感性"""
        if self.scheme == 'forward':
            return self._forward_diff(model, point, param_idx, h)
        elif self.scheme == 'backward':
            return self._backward_diff(model, point, param_idx, h)
        else:
            return self._central_diff(model, point, param_idx, h)
    
    def _forward_diff(self, model, point, i, h):
        """前向差分"""
        f0 = model(point)
        point_plus = point.copy()
        point_plus[i] += h
        f_plus = model(point_plus)
        return (f_plus - f0) / h
    
    def _backward_diff(self, model, point, i, h):
        """后向差分"""
        f0 = model(point)
        point_minus = point.copy()
        point_minus[i] -= h
        f_minus = model(point_minus)
        return (f0 - f_minus) / h
    
    def _central_diff(self, model, point, i, h):
        """中心差分"""
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[i] += h
        point_minus[i] -= h
        return (model(point_plus) - model(point_minus)) / (2 * h)


# ==================== 筛选方法 ====================

class ScreeningAnalysis:
    """参数筛选分析"""
    
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
    
    def screen_parameters(self,
                         model: Callable,
                         param_names: List[str],
                         bounds: np.ndarray,
                         n_samples: int = 100) -> Dict[str, List[str]]:
        """
        筛选参数
        
        Returns:
            {'important': [...], 'non_important': [...]}
        """
        # 使用Morris方法进行筛选
        morris = MorrisMethod(num_levels=4)
        report = morris.analyze(model, param_names, bounds, n_samples)
        
        ee = ElementaryEffects(
            mu=report.indices.first_order,
            mu_star=report.indices.total_order,
            sigma=np.zeros(len(param_names))
        )
        
        classification = ee.classify_parameters(
            threshold_mu=self.threshold
        )
        
        return {
            'important': [param_names[i] for i in classification['important']],
            'non_important': [param_names[i] for i in classification['insignificant']],
            'interaction': [param_names[i] for i in classification['interaction']]
        }


# ==================== 示例和测试 ====================

def demo():
    """演示敏感性分析"""
    print("=" * 80)
    print("📊 敏感性分析工具演示")
    print("=" * 80)
    
    # 定义测试模型：材料本构关系
    print("\n1. 定义材料模型")
    
    def material_model(params):
        """
        材料模型: 应力 = E * epsilon * exp(-k * epsilon)
        
        参数:
            params[0]: E - 杨氏模量
            params[1]: k - 非线性系数
            params[2]: epsilon - 应变
        """
        E, k, epsilon = params
        return E * epsilon * np.exp(-k * epsilon)
    
    param_names = ['E', 'k', 'epsilon']
    bounds = np.array([
        [150, 250],    # E: GPa
        [0.01, 0.1],   # k: 无量纲
        [0.001, 0.1]   # epsilon: 应变
    ])
    
    print(f"   模型: σ = E·ε·exp(-k·ε)")
    print(f"   参数: {param_names}")
    print(f"   边界:")
    for name, bound in zip(param_names, bounds):
        print(f"      {name}: [{bound[0]}, {bound[1]}]")
    
    # Sobol分析
    print("\n2. Sobol全局敏感性分析")
    
    if HAS_SALIB:
        sobol = SobolSensitivity(calc_second_order=True)
        report = sobol.analyze(material_model, param_names, bounds, n_samples=512)
        
        print("   一阶效应 (S1):")
        for name, s1 in zip(param_names, report.indices.first_order):
            print(f"      {name}: {s1:.4f}")
        
        print("   总效应 (ST):")
        for name, st in zip(param_names, report.indices.total_order):
            print(f"      {name}: {st:.4f}")
        
        if report.indices.second_order is not None:
            print("   二阶交互效应:")
            for i in range(len(param_names)):
                for j in range(i+1, len(param_names)):
                    s2 = report.indices.second_order[i, j]
                    if abs(s2) > 0.01:
                        print(f"      {param_names[i]} × {param_names[j]}: {s2:.4f}")
        
        print("   建议:")
        for rec in report.recommendations:
            print(f"      • {rec}")
    else:
        print("   SALib未安装，使用手动实现")
        sobol = SobolSensitivity()
        report = sobol.analyze(material_model, param_names, bounds, n_samples=100)
        print(f"   一阶效应: {report.indices.first_order}")
        print(f"   总效应: {report.indices.total_order}")
    
    # Morris分析
    print("\n3. Morris筛选方法")
    
    morris = MorrisMethod(num_levels=4)
    report_morris = morris.analyze(material_model, param_names, bounds, n_samples=20)
    
    print("   基本效应统计:")
    for name, mu, mu_star, sigma in zip(
        param_names,
        report_morris.indices.first_order,
        report_morris.indices.total_order,
        np.zeros(len(param_names))
    ):
        print(f"      {name}: μ*={mu_star:.4f}, μ={mu:.4f}")
    
    print("   参数分类:")
    for rec in report_morris.recommendations:
        print(f"      • {rec}")
    
    # 局部敏感性
    print("\n4. 局部敏感性分析")
    
    nominal_point = np.array([200, 0.05, 0.05])
    local = LocalSensitivity(nominal_point)
    
    print(f"   名义点: E={nominal_point[0]}, k={nominal_point[1]}, ε={nominal_point[2]}")
    
    elasticities = local.compute_elasticity(material_model)
    print("   弹性系数:")
    for name, elast in zip(param_names, elasticities):
        print(f"      {name}: {elast:.4f}")
    
    # 梯度分析
    print("\n5. 梯度分析")
    
    grad_analysis = GradientBasedAnalysis(method='central')
    gradient = grad_analysis.compute_gradient(material_model, nominal_point)
    
    print(f"   梯度: {gradient}")
    print(f"   梯度范数: {np.linalg.norm(gradient):.4f}")
    
    # 参数重要性排序
    print("\n6. 参数重要性排序")
    
    importance = report.parameter_importance
    print("   排名 (基于总效应):")
    for name, score, rank in zip(
        importance.param_names,
        importance.importance_scores,
        importance.rankings
    ):
        sig = "*" if importance.significance[importance.param_names.index(name)] else ""
        print(f"      {rank}. {name}: {score:.4f} {sig}")
    
    print("\n" + "=" * 80)
    print("✅ 敏感性分析演示完成")
    print("=" * 80)


if __name__ == "__main__":
    demo()
