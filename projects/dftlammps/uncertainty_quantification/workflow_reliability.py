"""
工作流可靠性评估 - Workflow Reliability Assessment

实现DFT-MD工作流的可靠性评估：
- 失效概率计算
- 可靠性指标
- 系统可靠性分析
- 质量保障与监控

核心特性:
- FORM/SORM方法
- 蒙特卡洛可靠性
- 故障树分析
- 实时可靠性监控
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import warnings

try:
    from scipy import stats
    from scipy.optimize import minimize
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==================== 数据结构 ====================

@dataclass
class FailureProbability:
    """失效概率估计"""
    pf: float  # 失效概率
    variance: float  # 估计方差
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    method: str = "unknown"
    
    def reliability_index(self) -> float:
        """计算可靠性指标 β = -Φ⁻¹(pf)"""
        if self.pf <= 0:
            return np.inf
        if self.pf >= 1:
            return -np.inf
        return -norm.ppf(self.pf)
    
    def is_safe(self, threshold: float = 1e-4) -> bool:
        """检查是否安全"""
        return self.pf < threshold


@dataclass
class ReliabilityIndex:
    """可靠性指标"""
    beta: float  # Hasofer-Lind可靠性指标
    pf: float  # 对应失效概率
    design_point: Optional[np.ndarray] = None  # 设计点（MPP）
    importance_factors: Optional[Dict[str, float]] = None  # 重要性因子
    
    def safety_level(self) -> str:
        """安全等级评估"""
        if self.beta > 4.0:
            return "very_safe"
        elif self.beta > 3.0:
            return "safe"
        elif self.beta > 2.0:
            return "marginal"
        else:
            return "unsafe"


@dataclass
class UncertaintyBudget:
    """不确定性预算"""
    total_uncertainty: float
    component_contributions: Dict[str, float] = field(default_factory=dict)
    correlation_effects: float = 0.0
    
    def pareto_analysis(self, n_top: int = 5) -> List[Tuple[str, float]]:
        """帕累托分析"""
        sorted_contrib = sorted(
            self.component_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_contrib[:n_top]


@dataclass
class ReliabilityAssessment:
    """可靠性评估结果"""
    failure_probability: FailureProbability
    reliability_index: ReliabilityIndex
    uncertainty_budget: UncertaintyBudget
    critical_scenarios: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """生成摘要报告"""
        lines = [
            "=" * 60,
            "可靠性评估摘要",
            "=" * 60,
            f"失效概率: {self.failure_probability.pf:.6e}",
            f"可靠性指标 β: {self.reliability_index.beta:.4f}",
            f"安全等级: {self.reliability_index.safety_level()}",
            f"总不确定性: {self.uncertainty_budget.total_uncertainty:.4f}",
            "",
            "主要不确定性来源:"
        ]
        
        for name, contrib in self.uncertainty_budget.pareto_analysis(5):
            lines.append(f"  - {name}: {contrib:.4f}")
        
        lines.extend([
            "",
            "建议:",
        ])
        for rec in self.recommendations:
            lines.append(f"  - {rec}")
        
        return "\n".join(lines)


# ==================== 失效概率计算方法 ====================

class FORMAnalysis:
    """
    一阶可靠性方法 (FORM)
    
    在标准正态空间中寻找设计点
    """
    
    def __init__(self,
                 tolerance: float = 1e-6,
                 max_iterations: int = 100):
        """
        初始化FORM分析
        
        Args:
            tolerance: 收敛容差
            max_iterations: 最大迭代次数
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def analyze(self,
               limit_state: Callable,
               distributions: Dict[str, stats.rv_continuous],
               initial_point: Optional[np.ndarray] = None) -> ReliabilityIndex:
        """
        执行FORM分析
        
        Args:
            limit_state: 极限状态函数 g(x) <= 0 表示失效
            distributions: 参数概率分布
            initial_point: 初始点
        
        Returns:
            可靠性指标
        """
        param_names = list(distributions.keys())
        n_params = len(param_names)
        
        # 转换为标准正态空间
        def transform_to_normal(x):
            """转换到标准正态空间"""
            u = np.zeros(len(x))
            for i, (name, dist) in enumerate(distributions.items()):
                u[i] = norm.ppf(dist.cdf(x[i]))
            return u
        
        def transform_from_normal(u):
            """从标准正态空间转换回来"""
            x = np.zeros(len(u))
            for i, (name, dist) in enumerate(distributions.items()):
                x[i] = dist.ppf(norm.cdf(u[i]))
            return x
        
        # 标准正态空间中的极限状态函数
        def g_u(u):
            x = transform_from_normal(u)
            return limit_state(x)
        
        # 使用HLRF算法求解设计点
        if initial_point is None:
            u = np.zeros(n_params)
        else:
            u = transform_to_normal(initial_point)
        
        for iteration in range(self.max_iterations):
            # 计算函数值和梯度
            g_val = g_u(u)
            
            # 数值梯度
            grad_g = self._numerical_gradient(g_u, u)
            grad_norm = np.linalg.norm(grad_g)
            
            if grad_norm < 1e-10:
                break
            
            # HLRF更新
            alpha = grad_g / grad_norm
            beta = -g_val / grad_norm + np.dot(alpha, u)
            
            u_new = -beta * alpha
            
            # 检查收敛
            if np.linalg.norm(u_new - u) < self.tolerance:
                u = u_new
                break
            
            u = u_new
        
        # 计算结果
        beta = np.linalg.norm(u)
        pf = norm.cdf(-beta)
        
        design_point = transform_from_normal(u)
        
        # 重要性因子
        grad_g = self._numerical_gradient(g_u, u)
        importance = np.abs(grad_g) / np.linalg.norm(grad_g)
        importance_factors = {
            name: importance[i] 
            for i, name in enumerate(param_names)
        }
        
        return ReliabilityIndex(
            beta=beta,
            pf=pf,
            design_point=design_point,
            importance_factors=importance_factors
        )
    
    def _numerical_gradient(self,
                           func: Callable,
                           x: np.ndarray,
                           h: float = 1e-6) -> np.ndarray:
        """数值梯度"""
        n = len(x)
        grad = np.zeros(n)
        f_x = func(x)
        
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            grad[i] = (func(x_plus) - f_x) / h
        
        return grad


class SORMAnalysis:
    """
    二阶可靠性方法 (SORM)
    
    考虑极限状态函数曲率的改进方法
    """
    
    def __init__(self, form_tolerance: float = 1e-6):
        self.form = FORMAnalysis(tolerance=form_tolerance)
    
    def analyze(self,
               limit_state: Callable,
               distributions: Dict[str, stats.rv_continuous],
               initial_point: Optional[np.ndarray] = None) -> ReliabilityIndex:
        """
        执行SORM分析
        
        首先进行FORM分析，然后考虑曲率修正
        """
        # 首先运行FORM
        form_result = self.form.analyze(limit_state, distributions, initial_point)
        
        # SORM修正（简化版本）
        # 实际实现需要计算Hessian和曲率
        beta_form = form_result.beta
        
        # 假设小的曲率修正
        curvature_correction = 1.0  # 简化：无修正
        pf_sorm = norm.cdf(-beta_form) * curvature_correction
        
        return ReliabilityIndex(
            beta=-norm.ppf(pf_sorm),
            pf=pf_sorm,
            design_point=form_result.design_point,
            importance_factors=form_result.importance_factors
        )


class MonteCarloReliability:
    """
    蒙特卡洛可靠性分析
    
    通过直接采样估计失效概率
    """
    
    def __init__(self,
                 batch_size: int = 10000,
                 coefficient_of_variation_target: float = 0.1):
        """
        初始化MC可靠性分析
        
        Args:
            batch_size: 每批样本数
            coefficient_of_variation_target: 目标变异系数
        """
        self.batch_size = batch_size
        self.cv_target = coefficient_of_variation_target
    
    def analyze(self,
               limit_state: Callable,
               distributions: Dict[str, stats.rv_continuous],
               correlation_matrix: Optional[np.ndarray] = None,
               max_samples: int = 1000000) -> FailureProbability:
        """
        执行蒙特卡洛可靠性分析
        
        Args:
            limit_state: 极限状态函数
            distributions: 参数分布
            correlation_matrix: 相关性矩阵
            max_samples: 最大样本数
        """
        param_names = list(distributions.keys())
        n_params = len(param_names)
        
        n_failures = 0
        n_total = 0
        
        # 用于估计方差的累加器
        sum_indicators = 0.0
        sum_squared_indicators = 0.0
        
        while n_total < max_samples:
            # 生成样本
            samples = self._generate_samples(
                distributions, self.batch_size, correlation_matrix
            )
            
            # 评估极限状态函数
            for sample in samples:
                n_total += 1
                
                g_val = limit_state(sample)
                indicator = 1 if g_val <= 0 else 0
                
                n_failures += indicator
                sum_indicators += indicator
                sum_squared_indicators += indicator**2
                
                # 检查收敛
                if n_total >= 100:
                    pf_current = n_failures / n_total
                    if pf_current > 0:
                        variance = (sum_squared_indicators / n_total - 
                                  (sum_indicators / n_total)**2) / n_total
                        cv = np.sqrt(variance) / pf_current if pf_current > 0 else np.inf
                        
                        if cv < self.cv_target:
                            break
            
            if n_total >= max_samples:
                break
        
        pf = n_failures / n_total if n_total > 0 else 0.0
        variance = (sum_squared_indicators / n_total - 
                   (sum_indicators / n_total)**2) / n_total if n_total > 0 else 1.0
        
        # 置信区间
        z_alpha = 1.96  # 95% CI
        ci_half_width = z_alpha * np.sqrt(variance)
        ci_lower = max(0, pf - ci_half_width)
        ci_upper = min(1, pf + ci_half_width)
        
        return FailureProbability(
            pf=pf,
            variance=variance,
            confidence_interval=(ci_lower, ci_upper),
            method=f"Monte Carlo (n={n_total})"
        )
    
    def _generate_samples(self,
                         distributions: Dict[str, stats.rv_continuous],
                         n_samples: int,
                         correlation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """生成相关样本"""
        n_params = len(distributions)
        
        # 独立采样
        independent_samples = np.array([
            dist.rvs(n_samples) for dist in distributions.values()
        ]).T
        
        if correlation_matrix is not None and HAS_SCIPY:
            # 应用相关性
            L = np.linalg.cholesky(correlation_matrix)
            correlated_samples = independent_samples @ L.T
            return correlated_samples
        
        return independent_samples


# ==================== 主控类 ====================

class ReliabilityEngine:
    """
    可靠性评估引擎
    
    提供统一的可靠性评估接口
    """
    
    def __init__(self):
        self.methods = {
            'form': FORMAnalysis(),
            'sorm': SORMAnalysis(),
            'mc': MonteCarloReliability()
        }
    
    def assess(self,
              limit_state: Callable,
              distributions: Dict[str, stats.rv_continuous],
              method: str = 'form') -> ReliabilityAssessment:
        """
        执行可靠性评估
        
        Args:
            limit_state: 极限状态函数
            distributions: 参数分布
            method: 分析方法
        """
        # 失效概率
        if method == 'form':
            ri = self.methods['form'].analyze(limit_state, distributions)
            pf = FailureProbability(
                pf=ri.pf,
                variance=0.0,
                method='FORM'
            )
        elif method == 'mc':
            pf = self.methods['mc'].analyze(limit_state, distributions)
            ri = ReliabilityIndex(beta=pf.reliability_index(), pf=pf.pf)
        else:
            pf = FailureProbability(pf=0.0, variance=0.0)
            ri = ReliabilityIndex(beta=0.0, pf=0.0)
        
        # 不确定性预算（简化）
        uncertainty_budget = UncertaintyBudget(
            total_uncertainty=pf.variance,
            component_contributions={
                name: dist.var() for name, dist in distributions.items()
            }
        )
        
        # 生成建议
        recommendations = self._generate_recommendations(pf, ri)
        
        return ReliabilityAssessment(
            failure_probability=pf,
            reliability_index=ri,
            uncertainty_budget=uncertainty_budget,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self,
                                  pf: FailureProbability,
                                  ri: ReliabilityIndex) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if pf.pf > 1e-3:
            recommendations.append("失效概率过高，建议重新设计")
        
        if ri.beta < 3.0:
            recommendations.append("可靠性指标偏低，建议增加安全裕度")
        
        if ri.importance_factors:
            # 找出最重要的参数
            most_important = max(ri.importance_factors.items(),
                               key=lambda x: x[1])
            recommendations.append(
                f"重点关注参数: {most_important[0]} (重要性因子: {most_important[1]:.3f})"
            )
        
        return recommendations


class WorkflowReliability:
    """
    DFT-MD工作流可靠性评估
    
    针对分子动力学工作流的专门可靠性评估
    """
    
    def __init__(self):
        self.engine = ReliabilityEngine()
        self.steps = {}
    
    def add_workflow_step(self,
                         name: str,
                         limit_state: Callable,
                         distributions: Dict[str, stats.rv_continuous]):
        """添加工作流步骤"""
        self.steps[name] = {
            'limit_state': limit_state,
            'distributions': distributions
        }
    
    def assess_step(self, step_name: str, method: str = 'form') -> ReliabilityAssessment:
        """评估单个步骤"""
        step = self.steps[step_name]
        return self.engine.assess(
            step['limit_state'],
            step['distributions'],
            method
        )
    
    def assess_workflow(self, method: str = 'form') -> Dict[str, ReliabilityAssessment]:
        """评估整个工作流"""
        results = {}
        for name in self.steps.keys():
            results[name] = self.assess_step(name, method)
        return results
    
    def workflow_reliability(self) -> float:
        """
        计算工作流整体可靠性
        
        假设串联系统
        """
        assessments = self.assess_workflow()
        
        # 串联系统：所有步骤都成功
        reliability = 1.0
        for assessment in assessments.values():
            reliability *= (1 - assessment.failure_probability.pf)
        
        return reliability


# ==================== 系统可靠性 ====================

class SystemReliability:
    """系统可靠性分析"""
    
    def __init__(self):
        self.components = {}
    
    def add_component(self,
                     name: str,
                     failure_probability: FailureProbability,
                     failure_mode: str = 'independent'):
        """添加组件"""
        self.components[name] = {
            'pf': failure_probability,
            'mode': failure_mode
        }
    
    def series_system(self, component_names: List[str]) -> FailureProbability:
        """
        串联系统可靠性
        
        任一组件失效则系统失效
        """
        # 假设独立
        pfs = [self.components[name]['pf'].pf for name in component_names]
        pf_system = 1.0 - np.prod([1 - pf for pf in pfs])
        
        return FailureProbability(
            pf=pf_system,
            variance=0.0,
            method="Series System"
        )
    
    def parallel_system(self, component_names: List[str]) -> FailureProbability:
        """
        并联系统可靠性
        
        所有组件失效系统才失效
        """
        pfs = [self.components[name]['pf'].pf for name in component_names]
        pf_system = np.prod(pfs)
        
        return FailureProbability(
            pf=pf_system,
            variance=0.0,
            method="Parallel System"
        )


class FaultTreeAnalysis:
    """故障树分析"""
    
    def __init__(self, top_event: str):
        """
        初始化故障树
        
        Args:
            top_event: 顶事件名称
        """
        self.top_event = top_event
        self.basic_events = {}
        self.gates = {}
    
    def add_basic_event(self,
                       name: str,
                       failure_probability: float,
                       description: str = ""):
        """添加基本事件"""
        self.basic_events[name] = {
            'pf': failure_probability,
            'description': description
        }
    
    def add_gate(self,
                name: str,
                gate_type: str,  # 'AND', 'OR'
                inputs: List[str]):
        """添加逻辑门"""
        self.gates[name] = {
            'type': gate_type,
            'inputs': inputs
        }
    
    def evaluate(self) -> float:
        """评估顶事件概率"""
        return self._evaluate_gate(self.top_event)
    
    def _evaluate_gate(self, event: str) -> float:
        """递归评估门"""
        # 检查是否是基本事件
        if event in self.basic_events:
            return self.basic_events[event]['pf']
        
        # 检查是否是门
        if event not in self.gates:
            return 0.0
        
        gate = self.gates[event]
        input_probs = [self._evaluate_gate(inp) for inp in gate['inputs']]
        
        if gate['type'] == 'AND':
            return np.prod(input_probs)
        elif gate['type'] == 'OR':
            return 1.0 - np.prod([1 - p for p in input_probs])
        
        return 0.0
    
    def minimal_cut_sets(self) -> List[List[str]]:
        """计算最小割集（简化版本）"""
        # 简化实现：返回所有基本事件
        return [[name] for name in self.basic_events.keys()]


class EventTreeAnalysis:
    """事件树分析"""
    
    def __init__(self, initiating_event: str, frequency: float):
        """
        初始化事件树
        
        Args:
            initiating_event: 初始事件
            frequency: 初始事件发生频率
        """
        self.initiating_event = initiating_event
        self.frequency = frequency
        self.branches = []
    
    def add_branch(self,
                  name: str,
                  success_probability: float,
                  consequence: str):
        """添加分支"""
        self.branches.append({
            'name': name,
            'success_prob': success_probability,
            'failure_prob': 1 - success_probability,
            'consequence': consequence
        })
    
    def evaluate(self) -> Dict[str, float]:
        """评估事件树"""
        # 计算各后果序列的概率
        consequences = {}
        
        # 简化：只考虑所有成功和所有失败
        all_success_prob = np.prod([b['success_prob'] for b in self.branches])
        
        for branch in self.branches:
            key = f"{branch['consequence']}_failure"
            consequences[key] = self.frequency * branch['failure_prob']
        
        consequences['all_success'] = self.frequency * all_success_prob
        
        return consequences


# ==================== 监控与质量 ====================

class ReliabilityMonitor:
    """可靠性实时监控系统"""
    
    def __init__(self,
                 warning_threshold: float = 1e-3,
                 critical_threshold: float = 1e-2):
        """
        初始化监控器
        
        Args:
            warning_threshold: 警告阈值
            critical_threshold: 严重阈值
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.history = []
    
    def update(self,
              timestamp: float,
              failure_probability: FailureProbability,
              context: Optional[Dict] = None) -> str:
        """更新监控状态"""
        entry = {
            'timestamp': timestamp,
            'pf': failure_probability.pf,
            'beta': failure_probability.reliability_index(),
            'context': context or {}
        }
        self.history.append(entry)
        
        # 检查警报
        if failure_probability.pf >= self.critical_threshold:
            return 'CRITICAL'
        elif failure_probability.pf >= self.warning_threshold:
            return 'WARNING'
        return 'NORMAL'
    
    def trend_analysis(self, window_size: int = 10) -> Dict:
        """趋势分析"""
        if len(self.history) < window_size:
            return {'trend': 'insufficient_data'}
        
        recent = self.history[-window_size:]
        pfs = [e['pf'] for e in recent]
        
        # 线性趋势
        x = np.arange(len(pfs))
        slope = np.polyfit(x, pfs, 1)[0]
        
        return {
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'slope': slope,
            'current_pf': pfs[-1],
            'average_pf': np.mean(pfs)
        }
    
    def get_alerts(self) -> List[Dict]:
        """获取警报列表"""
        alerts = []
        for entry in self.history:
            if entry['pf'] >= self.critical_threshold:
                alerts.append({
                    'level': 'CRITICAL',
                    'timestamp': entry['timestamp'],
                    'pf': entry['pf']
                })
            elif entry['pf'] >= self.warning_threshold:
                alerts.append({
                    'level': 'WARNING',
                    'timestamp': entry['timestamp'],
                    'pf': entry['pf']
                })
        return alerts


class QualityAssurance:
    """质量保障系统"""
    
    def __init__(self,
                 reliability_target: float = 1e-4,
                 confidence_level: float = 0.95):
        """
        初始化质量保障
        
        Args:
            reliability_target: 可靠性目标
            confidence_level: 置信水平
        """
        self.reliability_target = reliability_target
        self.confidence_level = confidence_level
        self.checks = []
    
    def add_check(self,
                 name: str,
                 check_function: Callable,
                 threshold: float):
        """添加质量检查"""
        self.checks.append({
            'name': name,
            'function': check_function,
            'threshold': threshold,
            'passed': 0,
            'failed': 0
        })
    
    def run_checks(self, data: Dict) -> Dict:
        """运行所有检查"""
        results = {}
        
        for check in self.checks:
            try:
                value = check['function'](data)
                passed = value <= check['threshold']
                
                if passed:
                    check['passed'] += 1
                else:
                    check['failed'] += 1
                
                results[check['name']] = {
                    'value': value,
                    'threshold': check['threshold'],
                    'passed': passed
                }
            except Exception as e:
                results[check['name']] = {
                    'error': str(e),
                    'passed': False
                }
        
        return results
    
    def overall_quality_score(self) -> float:
        """计算整体质量分数"""
        if not self.checks:
            return 1.0
        
        total_checks = sum(c['passed'] + c['failed'] for c in self.checks)
        if total_checks == 0:
            return 1.0
        
        total_passed = sum(c['passed'] for c in self.checks)
        return total_passed / total_checks


# ==================== 示例和测试 ====================

def demo():
    """演示工作流可靠性评估"""
    print("=" * 80)
    print("🛡️ 工作流可靠性评估演示")
    print("=" * 80)
    
    # 定义示例：梁的弯曲失效
    print("\n1. 结构可靠性分析示例")
    print("   问题: 悬臂梁最大应力失效")
    print("   极限状态: g = σ_yield - σ_max")
    
    def beam_limit_state(params):
        """
        梁极限状态函数
        params = [F, L, W, H, sigma_yield]
        F: 载荷 (N)
        L: 长度 (m)
        W: 宽度 (m)
        H: 高度 (m)
        sigma_yield: 屈服强度 (Pa)
        """
        F, L, W, H, sigma_yield = params
        
        # 最大弯矩
        M_max = F * L
        
        # 截面惯性矩 (矩形截面)
        I = W * H**3 / 12
        
        # 最大应力
        y_max = H / 2
        sigma_max = M_max * y_max / I
        
        # 极限状态
        return sigma_yield - sigma_max
    
    # 定义参数分布
    distributions = {
        'F': stats.norm(1000, 100),        # 载荷: N
        'L': stats.norm(2.0, 0.1),         # 长度: m
        'W': stats.norm(0.1, 0.005),       # 宽度: m
        'H': stats.norm(0.2, 0.01),        # 高度: m
        'sigma_yield': stats.norm(250e6, 25e6)  # 屈服强度: Pa
    }
    
    # FORM分析
    print("\n2. FORM一阶可靠性分析")
    
    form = FORMAnalysis()
    ri_form = form.analyze(beam_limit_state, distributions)
    
    print(f"   可靠性指标 β: {ri_form.beta:.4f}")
    print(f"   失效概率 Pf: {ri_form.pf:.6e}")
    print(f"   安全等级: {ri_form.safety_level()}")
    print("   重要性因子:")
    if ri_form.importance_factors:
        for name, factor in sorted(ri_form.importance_factors.items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"      {name}: {factor:.4f}")
    
    # 蒙特卡洛验证
    print("\n3. 蒙特卡洛验证")
    
    mc = MonteCarloReliability(batch_size=5000)
    pf_mc = mc.analyze(beam_limit_state, distributions, max_samples=100000)
    
    print(f"   样本数: 100000")
    print(f"   失效概率: {pf_mc.pf:.6e}")
    print(f"   95% 置信区间: [{pf_mc.confidence_interval[0]:.6e}, "
          f"{pf_mc.confidence_interval[1]:.6e}]")
    print(f"   可靠性指标: {pf_mc.reliability_index():.4f}")
    
    # 使用统一接口
    print("\n4. 可靠性评估引擎")
    
    engine = ReliabilityEngine()
    assessment = engine.assess(beam_limit_state, distributions, method='form')
    
    print(assessment.summary())
    
    # 工作流示例
    print("\n5. DFT-MD工作流可靠性示例")
    
    workflow = WorkflowReliability()
    
    # 添加工作流步骤
    def dft_convergence_limit(params):
        """DFT收敛性极限状态"""
        cutoff, kpoints = params
        # 假设收敛标准
        return min(cutoff - 400, kpoints - 4)  # cutoff >= 400, kpoints >= 4
    
    def md_stability_limit(params):
        """MD稳定性极限状态"""
        timestep, temperature = params
        # 稳定性条件
        return 1.0 - timestep / (2.0 / np.sqrt(temperature / 300))
    
    workflow.add_workflow_step(
        'DFT_Convergence',
        dft_convergence_limit,
        {
            'cutoff': stats.norm(450, 50),
            'kpoints': stats.norm(6, 1)
        }
    )
    
    workflow.add_workflow_step(
        'MD_Stability',
        md_stability_limit,
        {
            'timestep': stats.norm(1.0, 0.1),
            'temperature': stats.norm(300, 10)
        }
    )
    
    wf_reliability = workflow.workflow_reliability()
    print(f"   DFT-MD工作流整体可靠性: {wf_reliability:.6f}")
    print(f"   工作流失效概率: {1 - wf_reliability:.6e}")
    
    print("\n" + "=" * 80)
    print("✅ 工作流可靠性评估演示完成")
    print("=" * 80)


if __name__ == "__main__":
    demo()
