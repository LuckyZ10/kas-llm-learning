"""
计算-实验对比分析模块
实现计算结果与实验数据的对比、误差分析和验证
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from enum import Enum
from scipy import stats
import warnings

from .data_formats import ExperimentalProperty, ExperimentalDataset, CrystalStructure


class ComparisonMetric(Enum):
    """对比指标类型"""
    ABSOLUTE_ERROR = "absolute_error"
    RELATIVE_ERROR = "relative_error"
    PERCENTAGE_ERROR = "percentage_error"
    MAE = "mae"  # Mean Absolute Error
    RMSE = "rmse"  # Root Mean Square Error
    R2 = "r2"  # R-squared
    MAPE = "mape"  # Mean Absolute Percentage Error
    MAX_ERROR = "max_error"


class AgreementLevel(Enum):
    """一致性等级"""
    EXCELLENT = "excellent"  # < 1% error
    GOOD = "good"  # 1-5% error
    ACCEPTABLE = "acceptable"  # 5-10% error
    POOR = "poor"  # 10-20% error
    UNACCEPTABLE = "unacceptable"  # > 20% error


@dataclass
class ComparisonResult:
    """对比结果"""
    property_name: str
    computed_value: float
    experimental_value: float
    experimental_uncertainty: Optional[float] = None
    unit: str = ""
    
    # 误差指标
    absolute_error: float = 0.0
    relative_error: float = 0.0
    percentage_error: float = 0.0
    
    # 统计指标
    z_score: Optional[float] = None  # 与实验不确定度的偏差
    within_uncertainty: bool = False  # 是否在实验不确定度范围内
    
    # 评估
    agreement_level: AgreementLevel = AgreementLevel.ACCEPTABLE
    
    def __post_init__(self):
        if self.experimental_value != 0:
            self.relative_error = (self.computed_value - self.experimental_value) / abs(self.experimental_value)
            self.percentage_error = abs(self.relative_error) * 100
        
        self.absolute_error = abs(self.computed_value - self.experimental_value)
        
        # 计算Z-score
        if self.experimental_uncertainty and self.experimental_uncertainty > 0:
            self.z_score = (self.computed_value - self.experimental_value) / self.experimental_uncertainty
            self.within_uncertainty = abs(self.z_score) <= 2.0  # 2σ范围
        
        # 确定一致性等级
        self.agreement_level = self._determine_agreement_level()
    
    def _determine_agreement_level(self) -> AgreementLevel:
        """确定一致性等级"""
        pe = self.percentage_error
        if pe < 1.0:
            return AgreementLevel.EXCELLENT
        elif pe < 5.0:
            return AgreementLevel.GOOD
        elif pe < 10.0:
            return AgreementLevel.ACCEPTABLE
        elif pe < 20.0:
            return AgreementLevel.POOR
        else:
            return AgreementLevel.UNACCEPTABLE
    
    @property
    def is_accurate(self) -> bool:
        """判断是否准确（误差 < 10%）"""
        return self.percentage_error < 10.0
    
    def __str__(self) -> str:
        return (f"{self.property_name}: 计算={self.computed_value:.4f} {self.unit}, "
                f"实验={self.experimental_value:.4f}±{self.experimental_uncertainty or 0:.4f} {self.unit}, "
                f"误差={self.percentage_error:.2f}% ({self.agreement_level.value})")


@dataclass
class StatisticalAnalysis:
    """统计分析结果"""
    n_samples: int = 0
    
    # 误差统计
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Square Error
    max_error: float = 0.0
    mean_percentage_error: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error
    
    # 相关性
    r2: float = 0.0  # R-squared
    pearson_r: float = 0.0
    pearson_p: float = 0.0
    spearman_r: float = 0.0
    spearman_p: float = 0.0
    
    # 分布
    error_std: float = 0.0
    error_skewness: float = 0.0
    error_kurtosis: float = 0.0
    
    # 一致性
    within_2sigma: float = 0.0  # 在2σ内的比例
    within_1sigma: float = 0.0  # 在1σ内的比例
    
    # 系统偏差
    mean_bias: float = 0.0  # 平均偏差（计算-实验）
    systematic_error: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'n_samples': self.n_samples,
            'mae': self.mae,
            'rmse': self.rmse,
            'max_error': self.max_error,
            'mean_percentage_error': self.mean_percentage_error,
            'mape': self.mape,
            'r2': self.r2,
            'pearson_r': self.pearson_r,
            'pearson_p': self.pearson_p,
            'within_2sigma': self.within_2sigma,
            'within_1sigma': self.within_1sigma,
            'mean_bias': self.mean_bias,
            'systematic_error': self.systematic_error
        }


@dataclass
class ValidationReport:
    """验证报告"""
    property_name: str
    results: List[ComparisonResult] = field(default_factory=list)
    statistics: Optional[StatisticalAnalysis] = None
    
    # 验证标准
    accuracy_threshold: float = 10.0  # %
    precision_threshold: float = 5.0  # %
    
    # 评估结果
    is_validated: bool = False
    validation_score: float = 0.0  # 0-100
    confidence_level: float = 0.0  # 置信度
    
    def __post_init__(self):
        if self.results:
            self.statistics = self._compute_statistics()
            self._validate()
    
    def _compute_statistics(self) -> StatisticalAnalysis:
        """计算统计指标"""
        stat = StatisticalAnalysis()
        stat.n_samples = len(self.results)
        
        computed = np.array([r.computed_value for r in self.results])
        experimental = np.array([r.experimental_value for r in self.results])
        errors = np.array([r.absolute_error for r in self.results])
        rel_errors = np.array([r.relative_error for r in self.results])
        
        # 基础统计
        stat.mae = np.mean(errors)
        stat.rmse = np.sqrt(np.mean(errors ** 2))
        stat.max_error = np.max(errors)
        stat.mean_percentage_error = np.mean([r.percentage_error for r in self.results])
        stat.mape = np.mean([abs(r.percentage_error) for r in self.results])
        
        # 相关性
        if len(computed) > 2:
            stat.pearson_r, stat.pearson_p = stats.pearsonr(computed, experimental)
            stat.spearman_r, stat.spearman_p = stats.spearmanr(computed, experimental)
            stat.r2 = stat.pearson_r ** 2
        
        # 误差分布
        stat.error_std = np.std(rel_errors)
        stat.error_skewness = stats.skew(rel_errors)
        stat.error_kurtosis = stats.kurtosis(rel_errors)
        
        # 不确定性覆盖
        within_2sigma = sum(1 for r in self.results if r.within_uncertainty) / len(self.results)
        stat.within_2sigma = within_2sigma
        
        # 系统偏差
        stat.mean_bias = np.mean(computed - experimental)
        stat.systematic_error = stat.mean_bias / np.mean(np.abs(experimental)) * 100
        
        return stat
    
    def _validate(self):
        """执行验证"""
        if not self.results or not self.statistics:
            return
        
        # 计算验证分数
        accuracy_score = max(0, 100 - self.statistics.mape)
        precision_score = max(0, 100 - self.statistics.error_std * 10)
        correlation_score = max(0, self.statistics.r2 * 100)
        
        self.validation_score = (accuracy_score + precision_score + correlation_score) / 3
        
        # 判断是否通过验证
        self.is_validated = (
            self.statistics.mape < self.accuracy_threshold and
            self.statistics.r2 > 0.7 and
            self.statistics.within_2sigma > 0.5
        )
        
        # 置信度
        self.confidence_level = min(1.0, np.sqrt(self.statistics.n_samples) / 10)
    
    def get_summary(self) -> str:
        """获取摘要"""
        lines = [
            f"验证报告: {self.property_name}",
            f"=" * 50,
            f"样本数: {self.statistics.n_samples if self.statistics else 0}",
            f"MAE: {self.statistics.mae:.4f}" if self.statistics else "",
            f"RMSE: {self.statistics.rmse:.4f}" if self.statistics else "",
            f"MAPE: {self.statistics.mape:.2f}%" if self.statistics else "",
            f"R²: {self.statistics.r2:.4f}" if self.statistics else "",
            f"验证分数: {self.validation_score:.1f}/100",
            f"验证状态: {'✓ 通过' if self.is_validated else '✗ 未通过'}",
        ]
        return '\n'.join(filter(None, lines))


class StructureComparator:
    """结构比较器"""
    
    @staticmethod
    def compare_lattice(struct1: CrystalStructure, struct2: CrystalStructure) -> Dict[str, float]:
        """比较晶格参数"""
        lat1 = struct1.lattice
        lat2 = struct2.lattice
        
        results = {
            'a_error': abs(lat1.a - lat2.a) / lat2.a * 100,
            'b_error': abs(lat1.b - lat2.b) / lat2.b * 100,
            'c_error': abs(lat1.c - lat2.c) / lat2.c * 100,
            'alpha_error': abs(lat1.alpha - lat2.alpha),
            'beta_error': abs(lat1.beta - lat2.beta),
            'gamma_error': abs(lat1.gamma - lat2.gamma),
            'volume_error': abs(lat1.volume - lat2.volume) / lat2.volume * 100,
        }
        
        results['mean_lattice_error'] = np.mean([results['a_error'], results['b_error'], results['c_error']])
        results['mean_angle_error'] = np.mean([results['alpha_error'], results['beta_error'], results['gamma_error']])
        
        return results
    
    @staticmethod
    def rms_distance(struct1: CrystalStructure, struct2: CrystalStructure) -> float:
        """计算RMS原子位置偏差（需要超胞匹配）"""
        # 简化实现：假设结构已匹配
        if struct1.num_atoms != struct2.num_atoms:
            return float('inf')
        
        # 获取笛卡尔坐标
        coords1 = np.array([site.to_cartesian(struct1.lattice) for site in struct1.sites])
        coords2 = np.array([site.to_cartesian(struct2.lattice) for site in struct2.sites])
        
        # 计算RMSD（简化：假设原子顺序相同）
        rmsd = np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))
        return rmsd
    
    @staticmethod
    def compare_composition(struct1: CrystalStructure, struct2: CrystalStructure) -> Dict[str, Any]:
        """比较化学组成"""
        comp1 = struct1.composition
        comp2 = struct2.composition
        
        all_elements = set(comp1.keys()) | set(comp2.keys())
        
        element_errors = {}
        for elem in all_elements:
            c1 = comp1.get(elem, 0)
            c2 = comp2.get(elem, 0)
            if c2 > 0:
                element_errors[elem] = abs(c1 - c2) / c2 * 100
            else:
                element_errors[elem] = float('inf') if c1 > 0 else 0
        
        return {
            'formula1': struct1.formula,
            'formula2': struct2.formula,
            'element_errors': element_errors,
            'is_same_composition': comp1 == comp2
        }


class PropertyComparator:
    """属性比较器"""
    
    def __init__(self, accuracy_threshold: float = 10.0):
        self.accuracy_threshold = accuracy_threshold
    
    def compare_single(self, 
                      property_name: str,
                      computed: float,
                      experimental: float,
                      experimental_uncertainty: Optional[float] = None,
                      unit: str = "") -> ComparisonResult:
        """比较单个属性值"""
        return ComparisonResult(
            property_name=property_name,
            computed_value=computed,
            experimental_value=experimental,
            experimental_uncertainty=experimental_uncertainty,
            unit=unit
        )
    
    def compare_datasets(self,
                        computed_dataset: ExperimentalDataset,
                        experimental_dataset: ExperimentalDataset,
                        property_names: Optional[List[str]] = None) -> Dict[str, ComparisonResult]:
        """比较两个数据集的属性"""
        results = {}
        
        # 获取要比较的属性
        if property_names is None:
            prop_names = set(p.name for p in computed_dataset.properties) & \
                        set(p.name for p in experimental_dataset.properties)
        else:
            prop_names = property_names
        
        for name in prop_names:
            comp_prop = computed_dataset.get_property(name)
            exp_prop = experimental_dataset.get_property(name)
            
            if comp_prop and exp_prop:
                result = self.compare_single(
                    name,
                    comp_prop.value,
                    exp_prop.value,
                    exp_prop.uncertainty,
                    exp_prop.unit
                )
                results[name] = result
        
        return results
    
    def validate_against_experiment(self,
                                   computed_values: List[float],
                                   experimental_values: List[float],
                                   property_name: str,
                                   experimental_uncertainties: Optional[List[float]] = None) -> ValidationReport:
        """验证计算值对实验数据的拟合程度"""
        results = []
        
        uncertainties = experimental_uncertainties or [None] * len(experimental_values)
        
        for comp, exp, unc in zip(computed_values, experimental_values, uncertainties):
            result = self.compare_single(property_name, comp, exp, unc)
            results.append(result)
        
        return ValidationReport(property_name=property_name, results=results)


class UncertaintyQuantifier:
    """不确定性量化器"""
    
    @staticmethod
    def calculate_prediction_interval(predictions: np.ndarray, 
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """计算预测区间"""
        mean = np.mean(predictions)
        std = np.std(predictions)
        
        # 正态分布分位数
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        margin = z_score * std
        return mean - margin, mean + margin
    
    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray,
                                     statistic_func: Callable,
                                     n_bootstrap: int = 1000,
                                     confidence: float = 0.95) -> Tuple[float, float, float]:
        """Bootstrap置信区间"""
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        median = np.median(bootstrap_stats)
        
        return lower, median, upper
    
    @staticmethod
    def combine_uncertainties(uncertainties: List[float], method: str = 'quadrature') -> float:
        """组合多个不确定度"""
        if method == 'quadrature':
            # 平方和开方
            return np.sqrt(sum(u**2 for u in uncertainties))
        elif method == 'linear':
            # 线性相加
            return sum(uncertainties)
        elif method == 'max':
            # 取最大
            return max(uncertainties)
        else:
            raise ValueError(f"Unknown combination method: {method}")


class BenchmarkSuite:
    """基准测试套件"""
    
    def __init__(self):
        self.results: Dict[str, ValidationReport] = {}
    
    def add_benchmark(self, name: str, report: ValidationReport):
        """添加基准测试结果"""
        self.results[name] = report
    
    def get_summary_table(self) -> str:
        """获取汇总表格"""
        lines = [
            "=" * 100,
            f"{'Property':<20} {'N':<6} {'MAE':<10} {'RMSE':<10} {'MAPE(%)':<10} {'R²':<8} {'Score':<8} {'Status':<10}",
            "-" * 100
        ]
        
        for name, report in self.results.items():
            if report.statistics:
                stat = report.statistics
                status = "✓" if report.is_validated else "✗"
                lines.append(
                    f"{name:<20} {stat.n_samples:<6} {stat.mae:<10.4f} "
                    f"{stat.rmse:<10.4f} {stat.mape:<10.2f} {stat.r2:<8.4f} "
                    f"{report.validation_score:<8.1f} {status:<10}"
                )
        
        lines.append("=" * 100)
        return '\n'.join(lines)
    
    def get_best_performing(self, metric: str = 'mape') -> List[Tuple[str, float]]:
        """获取表现最好的属性"""
        scores = []
        for name, report in self.results.items():
            if report.statistics:
                value = getattr(report.statistics, metric, None)
                if value is not None:
                    scores.append((name, value))
        
        return sorted(scores, key=lambda x: x[1])
    
    def overall_score(self) -> float:
        """计算总体得分"""
        if not self.results:
            return 0.0
        return np.mean([r.validation_score for r in self.results.values()])


# =============================================================================
# 便捷函数
# =============================================================================

def compare_property(computed: float, 
                    experimental: float,
                    uncertainty: Optional[float] = None,
                    name: str = "property") -> ComparisonResult:
    """便捷函数：比较单个属性"""
    return ComparisonResult(
        property_name=name,
        computed_value=computed,
        experimental_value=experimental,
        experimental_uncertainty=uncertainty
    )


def validate_properties(computed_values: List[float],
                       experimental_values: List[float],
                       property_name: str,
                       uncertainties: Optional[List[float]] = None) -> ValidationReport:
    """便捷函数：验证属性列表"""
    comparator = PropertyComparator()
    return comparator.validate_against_experiment(
        computed_values, experimental_values, property_name, uncertainties
    )


def compare_structures(struct1: CrystalStructure, 
                      struct2: CrystalStructure) -> Dict[str, Any]:
    """便捷函数：比较两个结构"""
    lattice_comparison = StructureComparator.compare_lattice(struct1, struct2)
    composition_comparison = StructureComparator.compare_composition(struct1, struct2)
    
    return {
        'lattice': lattice_comparison,
        'composition': composition_comparison,
        'rmsd': StructureComparator.rms_distance(struct1, struct2)
    }


# =============================================================================
# 演示
# =============================================================================

def demo():
    """演示对比分析功能"""
    print("=" * 80)
    print("🔬 计算-实验对比分析演示")
    print("=" * 80)
    
    # 1. 单属性比较
    print("\n🔹 单属性比较示例:")
    
    test_cases = [
        # (computed, experimental, uncertainty, name)
        (3.2, 3.15, 0.1, "band_gap"),
        (4.25, 4.18, 0.02, "lattice_constant"),
        (25.5, 24.8, 1.5, "bulk_modulus"),
        (5.8, 5.0, 0.2, "band_gap"),  # 较大误差
    ]
    
    for comp, exp, unc, name in test_cases:
        result = compare_property(comp, exp, unc, name)
        print(f"   {result}")
        print(f"      Z-score: {result.z_score:.2f}, 在不确定度内: {result.within_uncertainty}")
    
    # 2. 统计分析
    print("\n🔹 批量验证示例:")
    
    np.random.seed(42)
    
    # 模拟计算值和实验值
    n_samples = 50
    true_values = np.random.uniform(1, 10, n_samples)
    experimental_values = true_values + np.random.normal(0, 0.1, n_samples)
    computed_values = true_values + np.random.normal(0.2, 0.3, n_samples)  # 有轻微系统偏差
    uncertainties = np.random.uniform(0.05, 0.2, n_samples)
    
    report = validate_properties(
        computed_values.tolist(),
        experimental_values.tolist(),
        "band_gap",
        uncertainties.tolist()
    )
    
    print(f"\n   {report.get_summary()}")
    
    # 3. 结构比较
    print("\n🔹 晶体结构比较示例:")
    
    from .data_formats import Lattice, AtomSite
    
    struct1 = CrystalStructure(
        formula="NaCl",
        lattice=Lattice(4.20, 4.20, 4.20, 90, 90, 90),
        sites=[AtomSite('Na', 0, 0, 0), AtomSite('Cl', 0.5, 0.5, 0.5)],
        space_group="Fm-3m"
    )
    
    struct2 = CrystalStructure(
        formula="NaCl",
        lattice=Lattice(4.18, 4.18, 4.18, 90, 90, 90),
        sites=[AtomSite('Na', 0, 0, 0), AtomSite('Cl', 0.5, 0.5, 0.5)],
        space_group="Fm-3m"
    )
    
    comparison = compare_structures(struct1, struct2)
    print(f"   晶格误差:")
    print(f"      a: {comparison['lattice']['a_error']:.2f}%")
    print(f"      体积: {comparison['lattice']['volume_error']:.2f}%")
    print(f"   组成相同: {comparison['composition']['is_same_composition']}")
    
    # 4. 基准测试套件
    print("\n🔹 基准测试套件:")
    
    suite = BenchmarkSuite()
    
    # 添加多个属性的测试结果
    properties = ['band_gap', 'lattice_constant', 'bulk_modulus', 'shear_modulus']
    for prop in properties:
        n = np.random.randint(20, 50)
        true = np.random.uniform(1, 100, n)
        exp = true + np.random.normal(0, 0.05 * true.std(), n)
        comp = true + np.random.normal(0, 0.1 * true.std(), n)
        
        report = validate_properties(comp.tolist(), exp.tolist(), prop)
        suite.add_benchmark(prop, report)
    
    print(suite.get_summary_table())
    print(f"\n   总体得分: {suite.overall_score():.1f}/100")
    
    # 5. 不确定性量化
    print("\n🔹 不确定性量化:")
    
    predictions = np.random.normal(5.0, 0.5, 100)
    lower, upper = UncertaintyQuantifier.calculate_prediction_interval(predictions, 0.95)
    print(f"   预测区间 (95%): [{lower:.2f}, {upper:.2f}]")
    print(f"   均值: {np.mean(predictions):.2f} ± {np.std(predictions):.2f}")
    
    # Bootstrap置信区间
    data = np.random.normal(10, 2, 50)
    lower, median, upper = UncertaintyQuantifier.bootstrap_confidence_interval(
        data, np.mean, n_bootstrap=1000
    )
    print(f"   Bootstrap均值置信区间: [{lower:.2f}, {upper:.2f}] (中位数: {median:.2f})")
    
    print("\n" + "=" * 80)
    print("✅ 对比分析演示完成!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
