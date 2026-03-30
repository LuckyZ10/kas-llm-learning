"""
反馈优化循环模块
基于计算-实验对比结果，自动优化计算参数和模型
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from enum import Enum
from abc import ABC, abstractmethod
import json
import warnings

from .comparison import ComparisonResult, ValidationReport, AgreementLevel
from .data_formats import ExperimentalProperty


class OptimizationTarget(Enum):
    """优化目标类型"""
    ACCURACY = "accuracy"  # 提高准确度
    PRECISION = "precision"  # 提高精密度
    EFFICIENCY = "efficiency"  # 提高效率
    BALANCE = "balance"  # 平衡优化


class ParameterType(Enum):
    """参数类型"""
    ENCUT = "encut"  # 截断能
    KPOINTS = "kpoints"  # K点网格
    SIGMA = "sigma"  # 展宽
    POTENTIAL = "potential"  # 势函数
    FUNCTIONAL = "functional"  # 交换关联泛函
    BASIS_SET = "basis_set"  # 基组
    CUTOFF = "cutoff"  # 截断半径
    TIMESTEP = "timestep"  # 时间步长
    TEMPERATURE = "temperature"  # 温度
    PRESSURE = "pressure"  # 压力


@dataclass
class ParameterAdjustment:
    """参数调整建议"""
    parameter: str
    current_value: Any
    suggested_value: Any
    parameter_type: ParameterType
    confidence: float  # 置信度 0-1
    expected_improvement: float  # 预期改善百分比
    reason: str = ""
    
    def __str__(self) -> str:
        return (f"{self.parameter}: {self.current_value} → {self.suggested_value} "
                f"(置信度: {self.confidence*100:.1f}%, 预期改善: {self.expected_improvement:.1f}%)")


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    target: OptimizationTarget
    priority: int  # 优先级 1-10
    adjustments: List[ParameterAdjustment] = field(default_factory=list)
    expected_outcome: str = ""
    risk_level: str = "low"  # low, medium, high
    estimated_cost: str = "medium"  # low, medium, high
    
    def __str__(self) -> str:
        lines = [
            f"优化目标: {self.target.value} (优先级: {self.priority}/10)",
            f"预期效果: {self.expected_outcome}",
            f"风险等级: {self.risk_level}, 成本: {self.estimated_cost}",
            "参数调整:"
        ]
        for adj in self.adjustments:
            lines.append(f"  • {adj}")
        return '\n'.join(lines)


@dataclass
class FeedbackCycle:
    """反馈循环记录"""
    cycle_id: int
    timestamp: str
    validation_report: Optional[ValidationReport] = None
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    applied_adjustments: List[ParameterAdjustment] = field(default_factory=list)
    improvement_achieved: Optional[float] = None  # 实际改善百分比
    status: str = "pending"  # pending, applied, verified, failed
    
    def to_dict(self) -> Dict:
        return {
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'status': self.status,
            'improvement': self.improvement_achieved
        }


class ErrorAnalyzer:
    """误差分析器"""
    
    def __init__(self):
        self.error_patterns: Dict[str, List[float]] = {}
    
    def analyze_systematic_errors(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """分析系统误差"""
        if not results:
            return {}
        
        errors = np.array([r.relative_error for r in results])
        
        analysis = {
            'mean_bias': np.mean(errors),
            'std_error': np.std(errors),
            'max_overestimation': np.max(errors),
            'max_underestimation': np.min(errors),
            'skewness': self._calculate_skewness(errors),
            'is_systematic': abs(np.mean(errors)) > 0.5 * np.std(errors),
            'trend': self._detect_trend(results)
        }
        
        return analysis
    
    def _calculate_skewness(self, errors: np.ndarray) -> float:
        """计算偏度"""
        n = len(errors)
        if n < 3:
            return 0.0
        mean = np.mean(errors)
        std = np.std(errors, ddof=1)
        if std == 0:
            return 0.0
        return (np.sum((errors - mean) ** 3) / n) / (std ** 3)
    
    def _detect_trend(self, results: List[ComparisonResult]) -> str:
        """检测误差趋势"""
        if len(results) < 3:
            return "insufficient_data"
        
        errors = np.array([r.relative_error for r in results])
        experimental = np.array([r.experimental_value for r in results])
        
        # 简单线性回归
        if np.std(experimental) > 0:
            slope = np.polyfit(experimental, errors, 1)[0]
            if abs(slope) < 0.01:
                return "constant"
            elif slope > 0:
                return "increasing_with_value"
            else:
                return "decreasing_with_value"
        
        return "no_trend"
    
    def identify_outliers(self, results: List[ComparisonResult], 
                         threshold: float = 2.0) -> List[ComparisonResult]:
        """识别异常值"""
        if len(results) < 3:
            return []
        
        errors = np.array([r.absolute_error for r in results])
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        outliers = []
        for result in results:
            z_score = (result.absolute_error - mean_error) / std_error
            if abs(z_score) > threshold:
                outliers.append(result)
        
        return outliers
    
    def categorize_errors(self, results: List[ComparisonResult]) -> Dict[str, List[ComparisonResult]]:
        """按一致性等级分类误差"""
        categories = {
            'excellent': [],
            'good': [],
            'acceptable': [],
            'poor': [],
            'unacceptable': []
        }
        
        for result in results:
            categories[result.agreement_level.value].append(result)
        
        return categories


class ParameterOptimizer(ABC):
    """参数优化器抽象基类"""
    
    @abstractmethod
    def suggest_adjustments(self, 
                          report: ValidationReport,
                          current_params: Dict[str, Any]) -> List[ParameterAdjustment]:
        """建议参数调整"""
        pass
    
    @abstractmethod
    def estimate_improvement(self, 
                           adjustment: ParameterAdjustment,
                           current_mape: float) -> float:
        """估计改善程度"""
        pass


class DFTParameterOptimizer(ParameterOptimizer):
    """DFT参数优化器"""
    
    def __init__(self):
        self.encut_recommendations = {
            'low': {'value': 400, 'reason': '当前截断能偏低，可能 insufficient'},
            'medium': {'value': 520, 'reason': '标准截断能'},
            'high': {'value': 600, 'reason': '高精度计算'},
            'very_high': {'value': 800, 'reason': '超高精度，用于困难系统'}
        }
    
    def suggest_adjustments(self,
                          report: ValidationReport,
                          current_params: Dict[str, Any]) -> List[ParameterAdjustment]:
        """建议DFT参数调整"""
        adjustments = []
        
        if not report.statistics:
            return adjustments
        
        mape = report.statistics.mape
        
        # 根据误差大小建议ENCUT调整
        current_encut = current_params.get('ENCUT', 400)
        
        if mape > 20:  # 大误差
            if current_encut < 600:
                adjustments.append(ParameterAdjustment(
                    parameter='ENCUT',
                    current_value=current_encut,
                    suggested_value=600,
                    parameter_type=ParameterType.ENCUT,
                    confidence=0.8,
                    expected_improvement=15.0,
                    reason='高MAPE表明截断能可能不足'
                ))
        elif mape > 10:  # 中等误差
            if current_encut < 520:
                adjustments.append(ParameterAdjustment(
                    parameter='ENCUT',
                    current_value=current_encut,
                    suggested_value=520,
                    parameter_type=ParameterType.ENCUT,
                    confidence=0.7,
                    expected_improvement=10.0,
                    reason='中等误差，建议增加截断能'
                ))
        
        # K点建议
        current_kpoints = current_params.get('KPOINTS', [4, 4, 4])
        if report.statistics.systematic_error > 5:
            denser_kpoints = [k + 2 for k in current_kpoints]
            adjustments.append(ParameterAdjustment(
                parameter='KPOINTS',
                current_value=current_kpoints,
                suggested_value=denser_kpoints,
                parameter_type=ParameterType.KPOINTS,
                confidence=0.6,
                expected_improvement=8.0,
                reason='系统误差可能源于K点网格不足'
            ))
        
        # SIGMA建议（对于金属体系）
        if 'ISMEAR' in current_params and current_params.get('ISMEAR', 0) != 0:
            current_sigma = current_params.get('SIGMA', 0.2)
            if report.statistics.error_std > 0.1:
                adjustments.append(ParameterAdjustment(
                    parameter='SIGMA',
                    current_value=current_sigma,
                    suggested_value=current_sigma * 0.5,
                    parameter_type=ParameterType.SIGMA,
                    confidence=0.5,
                    expected_improvement=5.0,
                    reason='尝试减小展宽以提高精度'
                ))
        
        return adjustments
    
    def estimate_improvement(self,
                           adjustment: ParameterAdjustment,
                           current_mape: float) -> float:
        """估计改善程度"""
        return adjustment.expected_improvement


class MDParameterOptimizer(ParameterOptimizer):
    """MD参数优化器"""
    
    def suggest_adjustments(self,
                          report: ValidationReport,
                          current_params: Dict[str, Any]) -> List[ParameterAdjustment]:
        """建议MD参数调整"""
        adjustments = []
        
        if not report.statistics:
            return adjustments
        
        # 时间步长建议
        current_timestep = current_params.get('timestep', 1.0)
        if report.statistics.error_std > 0.15:
            adjustments.append(ParameterAdjustment(
                parameter='timestep',
                current_value=current_timestep,
                suggested_value=current_timestep * 0.5,
                parameter_type=ParameterType.TIMESTEP,
                confidence=0.6,
                expected_improvement=10.0,
                reason='减小时间步长可能提高能量守恒'
            ))
        
        # 截断半径建议
        current_cutoff = current_params.get('cutoff', 8.0)
        if 'pressure' in report.property_name.lower():
            adjustments.append(ParameterAdjustment(
                parameter='cutoff',
                current_value=current_cutoff,
                suggested_value=current_cutoff + 2.0,
                parameter_type=ParameterType.CUTOFF,
                confidence=0.7,
                expected_improvement=12.0,
                reason='压力计算需要更大的截断半径'
            ))
        
        return adjustments
    
    def estimate_improvement(self,
                           adjustment: ParameterAdjustment,
                           current_mape: float) -> float:
        """估计改善程度"""
        return adjustment.expected_improvement


class FeedbackLoop:
    """反馈优化循环"""
    
    def __init__(self):
        self.error_analyzer = ErrorAnalyzer()
        self.optimizers: Dict[str, ParameterOptimizer] = {
            'dft': DFTParameterOptimizer(),
            'md': MDParameterOptimizer()
        }
        self.cycles: List[FeedbackCycle] = []
        self.convergence_threshold: float = 5.0  # MAPE收敛阈值
        self.max_cycles: int = 10
    
    def analyze_validation(self, report: ValidationReport) -> Dict[str, Any]:
        """分析验证结果"""
        analysis = {
            'overall_status': self._assess_overall_status(report),
            'systematic_errors': None,
            'outliers': [],
            'error_categories': {},
            'recommendations': []
        }
        
        if report.results:
            # 系统误差分析
            analysis['systematic_errors'] = self.error_analyzer.analyze_systematic_errors(
                report.results
            )
            
            # 异常值检测
            analysis['outliers'] = self.error_analyzer.identify_outliers(report.results)
            
            # 误差分类
            analysis['error_categories'] = self.error_analyzer.categorize_errors(
                report.results
            )
        
        return analysis
    
    def _assess_overall_status(self, report: ValidationReport) -> str:
        """评估整体状态"""
        if not report.statistics:
            return "insufficient_data"
        
        mape = report.statistics.mape
        
        if mape < 2:
            return "excellent"
        elif mape < 5:
            return "good"
        elif mape < 10:
            return "acceptable"
        elif mape < 20:
            return "needs_improvement"
        else:
            return "critical"
    
    def generate_recommendations(self,
                                report: ValidationReport,
                                current_params: Dict[str, Any],
                                calculation_type: str = 'dft') -> List[OptimizationRecommendation]:
        """生成优化建议"""
        recommendations = []
        
        # 获取优化器
        optimizer = self.optimizers.get(calculation_type.lower())
        if not optimizer:
            return recommendations
        
        # 获取参数调整建议
        adjustments = optimizer.suggest_adjustments(report, current_params)
        
        if not adjustments:
            return recommendations
        
        # 按置信度排序
        adjustments.sort(key=lambda x: x.confidence, reverse=True)
        
        # 生成推荐
        if report.statistics and report.statistics.mape > 10:
            rec = OptimizationRecommendation(
                target=OptimizationTarget.ACCURACY,
                priority=9,
                adjustments=[a for a in adjustments[:3] if a.confidence > 0.6],
                expected_outcome=f"预期MAPE从 {report.statistics.mape:.1f}% 降至 {report.statistics.mape * 0.7:.1f}%",
                risk_level="medium",
                estimated_cost="medium"
            )
            recommendations.append(rec)
        
        if report.statistics and report.statistics.error_std > 0.1:
            rec = OptimizationRecommendation(
                target=OptimizationTarget.PRECISION,
                priority=7,
                adjustments=[a for a in adjustments if 'KPOINTS' in a.parameter or 'timestep' in a.parameter],
                expected_outcome="减小标准偏差，提高结果重现性",
                risk_level="low",
                estimated_cost="high"
            )
            recommendations.append(rec)
        
        # 按优先级排序
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        return recommendations
    
    def run_cycle(self,
                 report: ValidationReport,
                 current_params: Dict[str, Any],
                 calculation_type: str = 'dft') -> FeedbackCycle:
        """运行一个反馈循环"""
        import datetime
        
        cycle_id = len(self.cycles) + 1
        
        # 分析
        analysis = self.analyze_validation(report)
        
        # 生成建议
        recommendations = self.generate_recommendations(
            report, current_params, calculation_type
        )
        
        cycle = FeedbackCycle(
            cycle_id=cycle_id,
            timestamp=str(datetime.datetime.now()),
            validation_report=report,
            recommendations=recommendations,
            status="pending"
        )
        
        self.cycles.append(cycle)
        
        return cycle
    
    def apply_recommendation(self, 
                           cycle: FeedbackCycle,
                           recommendation_idx: int = 0) -> Dict[str, Any]:
        """应用推荐"""
        if recommendation_idx >= len(cycle.recommendations):
            return {'success': False, 'error': 'Invalid recommendation index'}
        
        rec = cycle.recommendations[recommendation_idx]
        cycle.applied_adjustments = rec.adjustments
        cycle.status = "applied"
        
        # 生成新的参数集
        new_params = {}
        for adj in rec.adjustments:
            new_params[adj.parameter] = adj.suggested_value
        
        return {
            'success': True,
            'applied_recommendation': rec,
            'parameter_changes': new_params
        }
    
    def verify_improvement(self,
                         cycle: FeedbackCycle,
                         new_report: ValidationReport) -> float:
        """验证改善效果"""
        if not cycle.validation_report or not cycle.validation_report.statistics:
            return 0.0
        
        if not new_report.statistics:
            return 0.0
        
        old_mape = cycle.validation_report.statistics.mape
        new_mape = new_report.statistics.mape
        
        improvement = (old_mape - new_mape) / old_mape * 100
        cycle.improvement_achieved = improvement
        
        if improvement > 0:
            cycle.status = "verified"
        else:
            cycle.status = "failed"
        
        return improvement
    
    def has_converged(self, cycles_to_check: int = 3) -> bool:
        """检查是否收敛"""
        if len(self.cycles) < cycles_to_check:
            return False
        
        recent_cycles = self.cycles[-cycles_to_check:]
        
        # 检查最近的循环是否有明显改善
        improvements = [c.improvement_achieved for c in recent_cycles 
                       if c.improvement_achieved is not None]
        
        if len(improvements) < 2:
            return False
        
        # 如果连续改进都小于阈值，认为收敛
        return all(abs(imp) < self.convergence_threshold for imp in improvements[-2:])
    
    def get_optimization_history(self) -> List[Dict]:
        """获取优化历史"""
        return [cycle.to_dict() for cycle in self.cycles]
    
    def export_history(self, filepath: str):
        """导出优化历史"""
        history = self.get_optimization_history()
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)


class AdaptiveLearningRate:
    """自适应学习率（用于连续优化）"""
    
    def __init__(self, initial_lr: float = 0.1, min_lr: float = 0.001, max_lr: float = 1.0):
        self.lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.history: List[float] = []
    
    def update(self, improvement: float):
        """根据改善情况更新学习率"""
        self.history.append(improvement)
        
        if len(self.history) >= 2:
            # 如果改善在减小，降低学习率
            if improvement < self.history[-2]:
                self.lr *= 0.8
            # 如果改善在增加，略微提高学习率
            elif improvement > self.history[-2] * 1.1:
                self.lr = min(self.lr * 1.1, self.max_lr)
        
        self.lr = max(self.lr, self.min_lr)
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.lr


# =============================================================================
# 便捷函数
# =============================================================================

def create_feedback_loop() -> FeedbackLoop:
    """创建反馈循环"""
    return FeedbackLoop()


def quick_optimize(report: ValidationReport,
                  current_params: Dict[str, Any],
                  calculation_type: str = 'dft') -> List[OptimizationRecommendation]:
    """快速获取优化建议"""
    loop = FeedbackLoop()
    cycle = loop.run_cycle(report, current_params, calculation_type)
    return cycle.recommendations


# =============================================================================
# 演示
# =============================================================================

def demo():
    """演示反馈优化循环"""
    print("=" * 80)
    print("🔄 反馈优化循环演示")
    print("=" * 80)
    
    # 创建反馈循环
    loop = create_feedback_loop()
    print("\n🔹 创建反馈循环...")
    print(f"   ✓ 收敛阈值: {loop.convergence_threshold}%")
    print(f"   ✓ 最大循环: {loop.max_cycles}")
    
    # 创建模拟验证报告
    print("\n🔹 创建模拟验证报告...")
    
    from .comparison import ComparisonResult, ValidationReport, StatisticalAnalysis
    
    # 生成模拟结果
    np.random.seed(42)
    results = []
    for i in range(30):
        comp = 5.0 + np.random.normal(0, 0.5)
        exp = 5.0 + np.random.normal(0, 0.2)
        result = ComparisonResult(
            property_name='band_gap',
            computed_value=comp,
            experimental_value=exp,
            experimental_uncertainty=0.1,
            unit='eV'
        )
        results.append(result)
    
    report = ValidationReport(
        property_name='band_gap',
        results=results
    )
    
    print(f"   ✓ 属性: {report.property_name}")
    print(f"   ✓ 样本数: {len(results)}")
    if report.statistics:
        print(f"   ✓ MAPE: {report.statistics.mape:.2f}%")
    
    # 分析验证
    print("\n🔹 误差分析...")
    analysis = loop.analyze_validation(report)
    print(f"   整体状态: {analysis['overall_status']}")
    
    if analysis['systematic_errors']:
        se = analysis['systematic_errors']
        print(f"   系统偏差: {se['mean_bias']:.4f}")
        print(f"   有系统误差: {se['is_systematic']}")
        print(f"   趋势: {se['trend']}")
    
    # 误差分类
    categories = analysis['error_categories']
    print("\n   误差分类:")
    for level, items in categories.items():
        if items:
            print(f"      {level}: {len(items)} 个")
    
    # 当前参数
    current_params = {
        'ENCUT': 400,
        'KPOINTS': [4, 4, 4],
        'SIGMA': 0.2,
        'ISMEAR': 0
    }
    
    print("\n🔹 当前DFT参数:")
    for param, value in current_params.items():
        print(f"   {param}: {value}")
    
    # 生成优化建议
    print("\n🔹 生成优化建议...")
    recommendations = loop.generate_recommendations(report, current_params, 'dft')
    
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n   建议 {i}:")
        print(f"   {rec}")
    
    # 运行反馈循环
    print("\n🔹 运行反馈循环...")
    cycle = loop.run_cycle(report, current_params, 'dft')
    print(f"   循环ID: {cycle.cycle_id}")
    print(f"   建议数: {len(cycle.recommendations)}")
    print(f"   状态: {cycle.status}")
    
    # 应用推荐
    if cycle.recommendations:
        print("\n🔹 应用第一个推荐...")
        result = loop.apply_recommendation(cycle, 0)
        if result['success']:
            print(f"   ✓ 已应用")
            print(f"   参数变更: {result['parameter_changes']}")
    
    # 模拟验证改善
    print("\n🔹 模拟验证改善...")
    new_results = []
    for i in range(30):
        comp = 5.0 + np.random.normal(0, 0.3)  # 误差减小
        exp = 5.0 + np.random.normal(0, 0.2)
        result = ComparisonResult(
            property_name='band_gap',
            computed_value=comp,
            experimental_value=exp,
            experimental_uncertainty=0.1,
            unit='eV'
        )
        new_results.append(result)
    
    new_report = ValidationReport(
        property_name='band_gap',
        results=new_results
    )
    
    improvement = loop.verify_improvement(cycle, new_report)
    print(f"   改善程度: {improvement:.1f}%")
    print(f"   新MAPE: {new_report.statistics.mape:.2f}%" if new_report.statistics else "")
    
    # 检查收敛
    print("\n🔹 检查收敛...")
    converged = loop.has_converged()
    print(f"   已收敛: {converged}")
    
    # 优化历史
    print("\n🔹 优化历史:")
    history = loop.get_optimization_history()
    for h in history:
        print(f"   循环 {h['cycle_id']}: {h['status']}, 改善={h.get('improvement', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✅ 反馈优化循环演示完成!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
