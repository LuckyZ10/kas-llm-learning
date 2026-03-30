"""
计算-实验对比模块 - Computation-Experiment Comparison Module

实现：
- 多尺度数据对比
- 结构相似性分析
- 性质对比验证
- 不确定性量化
- 反馈循环
"""

import os
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from scipy import stats, interpolate
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetrics:
    """对比指标"""
    rmse: float
    mae: float
    r2: float
    correlation: float
    cosine_similarity: float
    ks_statistic: float
    ks_pvalue: float
    relative_error: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "correlation": self.correlation,
            "cosine_similarity": self.cosine_similarity,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "relative_error": self.relative_error
        }


@dataclass
class StructureComparison:
    """结构对比结果"""
    rmsd: float  # 均方根偏差
    lattice_mismatch: Dict[str, float] = field(default_factory=dict)
    atomic_displacements: List[float] = field(default_factory=list)
    wyckoff_match: float = 0.0
    space_group_match: bool = False
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rmsd": self.rmsd,
            "lattice_mismatch": self.lattice_mismatch,
            "wyckoff_match": self.wyckoff_match,
            "space_group_match": self.space_group_match,
            "confidence": self.confidence
        }


@dataclass
class PropertyComparison:
    """性质对比结果"""
    property_name: str
    experimental_value: float
    computed_value: float
    uncertainty: float
    relative_error: float
    within_tolerance: bool
    tolerance: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_name": self.property_name,
            "experimental_value": self.experimental_value,
            "computed_value": self.computed_value,
            "uncertainty": self.uncertainty,
            "relative_error": self.relative_error,
            "within_tolerance": self.within_tolerance
        }


class DataComparator(ABC):
    """数据对比器抽象基类"""
    
    @abstractmethod
    def compare(self, experimental: Any, computed: Any) -> ComparisonMetrics:
        """对比实验和计算数据"""
        pass


class SpectrumComparator(DataComparator):
    """光谱数据对比器"""
    
    def __init__(self, interpolation_method: str = 'cubic'):
        self.interpolation_method = interpolation_method
    
    def compare(self, exp_spectrum: Tuple[np.ndarray, np.ndarray],
               calc_spectrum: Tuple[np.ndarray, np.ndarray]) -> ComparisonMetrics:
        """
        对比两个光谱
        
        Args:
            exp_spectrum: (x_exp, y_exp) 实验光谱
            calc_spectrum: (x_calc, y_calc) 计算光谱
        """
        x_exp, y_exp = exp_spectrum
        x_calc, y_calc = calc_spectrum
        
        # 对齐到共同网格
        x_common = np.linspace(
            max(x_exp[0], x_calc[0]),
            min(x_exp[-1], x_calc[-1]),
            1000
        )
        
        # 插值
        f_exp = interpolate.interp1d(x_exp, y_exp, kind=self.interpolation_method,
                                     bounds_error=False, fill_value=0)
        f_calc = interpolate.interp1d(x_calc, y_calc, kind=self.interpolation_method,
                                      bounds_error=False, fill_value=0)
        
        y_exp_interp = f_exp(x_common)
        y_calc_interp = f_calc(x_common)
        
        # 归一化
        y_exp_norm = y_exp_interp / np.max(y_exp_interp) if np.max(y_exp_interp) > 0 else y_exp_interp
        y_calc_norm = y_calc_interp / np.max(y_calc_interp) if np.max(y_calc_interp) > 0 else y_calc_interp
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_exp_norm, y_calc_norm))
        mae = np.mean(np.abs(y_exp_norm - y_calc_norm))
        
        correlation = np.corrcoef(y_exp_norm, y_calc_norm)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # R²
        r2 = r2_score(y_exp_norm, y_calc_norm)
        if np.isnan(r2):
            r2 = -np.inf
        
        # 余弦相似度
        cos_sim = 1 - cosine(y_exp_norm, y_calc_norm)
        if np.isnan(cos_sim):
            cos_sim = 0.0
        
        # KS检验
        ks_stat, ks_pvalue = stats.ks_2samp(y_exp_norm, y_calc_norm)
        
        # 相对误差
        relative_error = np.mean(np.abs(y_exp_norm - y_calc_norm) / 
                                (np.abs(y_exp_norm) + 1e-10))
        
        return ComparisonMetrics(
            rmse=rmse,
            mae=mae,
            r2=r2,
            correlation=correlation,
            cosine_similarity=cos_sim,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            relative_error=relative_error
        )


class XRDComparator(SpectrumComparator):
    """XRD数据对比器"""
    
    def compare_patterns(self, 
                        exp_pattern: Dict[str, Any],
                        calc_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """对比XRD衍射图谱"""
        # 获取2θ和强度数据
        exp_2theta = np.array(exp_pattern['two_theta'])
        exp_intensity = np.array(exp_pattern['intensity'])
        calc_2theta = np.array(calc_pattern['two_theta'])
        calc_intensity = np.array(calc_pattern['intensity'])
        
        # 使用父类方法对比
        metrics = self.compare((exp_2theta, exp_intensity),
                              (calc_2theta, calc_intensity))
        
        # 额外分析
        peak_analysis = self._analyze_peak_match(exp_pattern, calc_pattern)
        
        return {
            "metrics": metrics.to_dict(),
            "peak_analysis": peak_analysis,
            "agreement_score": self._calculate_agreement_score(metrics),
            "validation_passed": self._validate(metrics)
        }
    
    def _analyze_peak_match(self, exp_pattern: Dict, calc_pattern: Dict) -> Dict[str, Any]:
        """分析峰位匹配"""
        exp_peaks = exp_pattern.get('peaks', [])
        calc_peaks = calc_pattern.get('peaks', [])
        
        matched = 0
        position_errors = []
        intensity_ratios = []
        
        for exp_peak in exp_peaks:
            exp_pos = exp_peak['two_theta']
            exp_int = exp_peak['intensity']
            
            # 寻找最接近的计算峰
            best_match = None
            best_error = float('inf')
            
            for calc_peak in calc_peaks:
                calc_pos = calc_peak['two_theta']
                error = abs(exp_pos - calc_pos)
                
                if error < 0.5 and error < best_error:  # 0.5度容差
                    best_error = error
                    best_match = calc_peak
            
            if best_match:
                matched += 1
                position_errors.append(best_error)
                intensity_ratios.append(exp_int / (best_match['intensity'] + 1e-10))
        
        return {
            "num_exp_peaks": len(exp_peaks),
            "num_calc_peaks": len(calc_peaks),
            "matched_peaks": matched,
            "match_ratio": matched / len(exp_peaks) if exp_peaks else 0,
            "mean_position_error": np.mean(position_errors) if position_errors else 0,
            "std_position_error": np.std(position_errors) if position_errors else 0,
            "mean_intensity_ratio": np.mean(intensity_ratios) if intensity_ratios else 1.0
        }
    
    def _calculate_agreement_score(self, metrics: ComparisonMetrics) -> float:
        """计算综合一致度评分"""
        # 加权组合多个指标
        weights = {
            'correlation': 0.3,
            'cosine': 0.2,
            'r2': 0.2,
            'ks_pvalue': 0.15,
            'inv_rmse': 0.15
        }
        
        scores = {
            'correlation': max(0, metrics.correlation),
            'cosine': max(0, metrics.cosine_similarity),
            'r2': max(0, metrics.r2) if metrics.r2 > 0 else 0,
            'ks_pvalue': metrics.ks_pvalue,
            'inv_rmse': 1 / (1 + metrics.rmse)
        }
        
        score = sum(weights[k] * scores[k] for k in weights.keys())
        return min(1.0, max(0.0, score))
    
    def _validate(self, metrics: ComparisonMetrics, 
                 thresholds: Optional[Dict[str, float]] = None) -> bool:
        """验证是否通过"""
        thresholds = thresholds or {
            'correlation': 0.7,
            'r2': 0.5,
            'rmse': 0.3
        }
        
        return (
            metrics.correlation >= thresholds['correlation'] and
            metrics.r2 >= thresholds['r2'] and
            metrics.rmse <= thresholds['rmse']
        )


class StructureComparator:
    """结构对比器"""
    
    def compare_structures(self,
                          exp_structure: Dict[str, Any],
                          calc_structure: Dict[str, Any]) -> StructureComparison:
        """对比晶体结构"""
        
        # 对比晶格参数
        lattice_mismatch = self._compare_lattice_params(
            exp_structure.get('lattice_params', {}),
            calc_structure.get('lattice_params', {})
        )
        
        # 对比空间群
        space_group_match = (
            exp_structure.get('space_group') == calc_structure.get('space_group')
        )
        
        # 计算RMSD（如果有原子位置）
        rmsd = self._calculate_rmsd(
            exp_structure.get('positions', []),
            calc_structure.get('positions', [])
        )
        
        # 计算置信度
        confidence = self._calculate_structure_confidence(
            lattice_mismatch, space_group_match, rmsd
        )
        
        return StructureComparison(
            rmsd=rmsd,
            lattice_mismatch=lattice_mismatch,
            space_group_match=space_group_match,
            confidence=confidence
        )
    
    def _compare_lattice_params(self, exp_params: Dict, 
                               calc_params: Dict) -> Dict[str, float]:
        """对比晶格参数"""
        mismatch = {}
        
        for key in ['a', 'b', 'c']:
            if key in exp_params and key in calc_params:
                exp_val = exp_params[key]
                calc_val = calc_params[key]
                if exp_val != 0:
                    mismatch[key] = abs(exp_val - calc_val) / exp_val
                else:
                    mismatch[key] = abs(calc_val)
        
        for key in ['alpha', 'beta', 'gamma']:
            if key in exp_params and key in calc_params:
                mismatch[key] = abs(exp_params[key] - calc_params[key])
        
        return mismatch
    
    def _calculate_rmsd(self, exp_pos: List, calc_pos: List) -> float:
        """计算RMSD"""
        if not exp_pos or not calc_pos or len(exp_pos) != len(calc_pos):
            return 0.0
        
        exp_array = np.array(exp_pos)
        calc_array = np.array(calc_pos)
        
        # 对齐结构（简化实现）
        diff = exp_array - calc_array
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        return float(rmsd)
    
    def _calculate_structure_confidence(self, lattice_mismatch: Dict,
                                       space_group_match: bool,
                                       rmsd: float) -> float:
        """计算结构匹配置信度"""
        # 晶格参数匹配度
        lattice_score = 1.0
        if lattice_mismatch:
            avg_mismatch = np.mean(list(lattice_mismatch.values()))
            lattice_score = max(0, 1 - avg_mismatch)
        
        # 空间群匹配
        space_group_score = 1.0 if space_group_match else 0.5
        
        # RMSD评分（假设小于0.1Å为优秀）
        rmsd_score = max(0, 1 - rmsd / 0.1)
        
        # 加权组合
        confidence = 0.4 * lattice_score + 0.3 * space_group_score + 0.3 * rmsd_score
        
        return confidence


class PropertyComparator:
    """性质对比器"""
    
    def __init__(self, default_tolerance: float = 0.1):
        self.default_tolerance = default_tolerance
        self.property_units = {
            'band_gap': 'eV',
            'lattice_constant': 'Å',
            'bulk_modulus': 'GPa',
            'ionic_conductivity': 'S/cm',
            'density': 'g/cm³',
            'thermal_conductivity': 'W/mK',
            'dielectric_constant': 'dimensionless'
        }
    
    def compare_property(self,
                        property_name: str,
                        exp_value: float,
                        calc_value: float,
                        exp_uncertainty: Optional[float] = None,
                        tolerance: Optional[float] = None) -> PropertyComparison:
        """对比单个性质"""
        tol = tolerance or self.default_tolerance
        
        # 计算相对误差
        if exp_value != 0:
            relative_error = abs(exp_value - calc_value) / abs(exp_value)
        else:
            relative_error = abs(calc_value)
        
        # 判断是否在容差内
        within_tolerance = relative_error <= tol
        
        # 组合不确定性
        uncertainty = exp_uncertainty or (abs(exp_value) * 0.1)  # 默认10%
        
        return PropertyComparison(
            property_name=property_name,
            experimental_value=exp_value,
            computed_value=calc_value,
            uncertainty=uncertainty,
            relative_error=relative_error,
            within_tolerance=within_tolerance,
            tolerance=tol
        )
    
    def compare_multiple_properties(self,
                                   properties: Dict[str, Tuple[float, float]],
                                   uncertainties: Optional[Dict[str, float]] = None) -> Dict[str, PropertyComparison]:
        """对比多个性质"""
        uncertainties = uncertainties or {}
        
        results = {}
        for prop_name, (exp_val, calc_val) in properties.items():
            results[prop_name] = self.compare_property(
                prop_name,
                exp_val,
                calc_val,
                uncertainties.get(prop_name)
            )
        
        return results
    
    def generate_report(self, comparisons: Dict[str, PropertyComparison]) -> str:
        """生成对比报告"""
        report = ["Property Comparison Report", "=" * 40, ""]
        
        total_passed = 0
        for prop_name, comp in comparisons.items():
            unit = self.property_units.get(prop_name, '')
            status = "✓ PASS" if comp.within_tolerance else "✗ FAIL"
            
            report.append(f"{prop_name} ({unit}):")
            report.append(f"  Experimental: {comp.experimental_value:.4f}")
            report.append(f"  Computed:     {comp.computed_value:.4f}")
            report.append(f"  Relative Error: {comp.relative_error:.2%}")
            report.append(f"  Status: {status}")
            report.append("")
            
            if comp.within_tolerance:
                total_passed += 1
        
        report.append(f"\nOverall: {total_passed}/{len(comparisons)} passed")
        
        return "\n".join(report)


class ImageComparator:
    """图像对比器（用于SEM/TEM）"""
    
    def compare_images(self,
                      exp_image: np.ndarray,
                      calc_image: np.ndarray,
                      metrics: List[str] = None) -> Dict[str, float]:
        """对比两幅图像"""
        metrics = metrics or ['mse', 'ssim', 'correlation', 'histogram']
        
        # 确保尺寸一致
        if exp_image.shape != calc_image.shape:
            calc_image = self._resize_image(calc_image, exp_image.shape)
        
        results = {}
        
        if 'mse' in metrics:
            results['mse'] = np.mean((exp_image - calc_image) ** 2)
        
        if 'ssim' in metrics:
            try:
                from skimage.metrics import structural_similarity
                results['ssim'] = structural_similarity(exp_image, calc_image)
            except ImportError:
                results['ssim'] = 0.0
        
        if 'correlation' in metrics:
            results['correlation'] = np.corrcoef(
                exp_image.flatten(), calc_image.flatten()
            )[0, 1]
        
        if 'histogram' in metrics:
            results['histogram_similarity'] = self._compare_histograms(
                exp_image, calc_image
            )
        
        return results
    
    def _resize_image(self, image: np.ndarray, 
                     target_shape: Tuple[int, ...]) -> np.ndarray:
        """调整图像尺寸"""
        try:
            from scipy.ndimage import zoom
            
            factors = [t / s for t, s in zip(target_shape, image.shape)]
            return zoom(image, factors, order=1)
        except:
            return image
    
    def _compare_histograms(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """对比直方图"""
        # 计算直方图
        hist1, _ = np.histogram(img1.flatten(), bins=256, range=(0, 1), density=True)
        hist2, _ = np.histogram(img2.flatten(), bins=256, range=(0, 1), density=True)
        
        # Bhattacharyya距离
        bc_coeff = np.sum(np.sqrt(hist1 * hist2))
        
        return float(bc_coeff)


class FeedbackLoop:
    """反馈循环 - 根据实验结果改进计算模型"""
    
    def __init__(self):
        self.comparisons_history: List[Dict[str, Any]] = []
        self.convergence_threshold = 0.05
        
    def add_comparison(self, comparison_result: Dict[str, Any]):
        """添加对比结果到历史"""
        self.comparisons_history.append({
            'timestamp': self._get_timestamp(),
            'result': comparison_result
        })
    
    def analyze_trends(self) -> Dict[str, Any]:
        """分析趋势"""
        if len(self.comparisons_history) < 2:
            return {"status": "insufficient_data"}
        
        # 提取误差历史
        errors = []
        for entry in self.comparisons_history:
            if 'relative_error' in entry['result']:
                errors.append(entry['result']['relative_error'])
        
        if not errors:
            return {"status": "no_error_data"}
        
        return {
            "status": "ok",
            "error_trend": "decreasing" if errors[-1] < errors[0] else "increasing",
            "initial_error": errors[0],
            "current_error": errors[-1],
            "improvement": errors[0] - errors[-1],
            "converged": abs(errors[-1] - errors[-2]) < self.convergence_threshold 
                        if len(errors) >= 2 else False
        }
    
    def suggest_improvements(self, comparison_result: Dict[str, Any]) -> List[str]:
        """建议改进方向"""
        suggestions = []
        
        # 基于对比结果给出建议
        if 'xrd_comparison' in comparison_result:
            xrd = comparison_result['xrd_comparison']
            if xrd.get('peak_analysis', {}).get('match_ratio', 0) < 0.8:
                suggestions.append("改进结构模型以更好匹配XRD峰位")
        
        if 'property_comparison' in comparison_result:
            props = comparison_result['property_comparison']
            failed_props = [k for k, v in props.items() if not v.within_tolerance]
            if failed_props:
                suggestions.append(f"重新计算以下性质: {', '.join(failed_props)}")
        
        if 'structure_comparison' in comparison_result:
            struct = comparison_result['structure_comparison']
            if struct.get('rmsd', 0) > 0.1:
                suggestions.append("优化结构弛豫参数以减小RMSD")
        
        if not suggestions:
            suggestions.append("模型与实验数据吻合良好，可进行下一步预测")
        
        return suggestions
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


class ComparisonManager:
    """对比管理器 - 整合所有对比功能"""
    
    def __init__(self):
        self.xrd_comparator = XRDComparator()
        self.structure_comparator = StructureComparator()
        self.property_comparator = PropertyComparator()
        self.image_comparator = ImageComparator()
        self.feedback_loop = FeedbackLoop()
    
    def full_comparison(self,
                       experimental_data: Dict[str, Any],
                       computed_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行全面对比"""
        results = {}
        
        # XRD对比
        if 'xrd' in experimental_data and 'xrd' in computed_data:
            results['xrd_comparison'] = self.xrd_comparator.compare_patterns(
                experimental_data['xrd'],
                computed_data['xrd']
            )
        
        # 结构对比
        if 'structure' in experimental_data and 'structure' in computed_data:
            results['structure_comparison'] = self.structure_comparator.compare_structures(
                experimental_data['structure'],
                computed_data['structure']
            ).to_dict()
        
        # 性质对比
        if 'properties' in experimental_data and 'properties' in computed_data:
            results['property_comparison'] = {
                k: v.to_dict() for k, v in 
                self.property_comparator.compare_multiple_properties(
                    self._align_properties(
                        experimental_data['properties'],
                        computed_data['properties']
                    )
                ).items()
            }
        
        # SEM/TEM对比
        if 'sem_image' in experimental_data and 'sem_image' in computed_data:
            results['sem_comparison'] = self.image_comparator.compare_images(
                experimental_data['sem_image'],
                computed_data['sem_image']
            )
        
        # 更新反馈循环
        self.feedback_loop.add_comparison(results)
        results['feedback'] = self.feedback_loop.analyze_trends()
        results['suggestions'] = self.feedback_loop.suggest_improvements(results)
        
        # 综合评分
        results['overall_score'] = self._calculate_overall_score(results)
        
        return results
    
    def _align_properties(self, exp_props: Dict, calc_props: Dict) -> Dict[str, Tuple[float, float]]:
        """对齐实验和计算性质"""
        aligned = {}
        for key in set(exp_props.keys()) & set(calc_props.keys()):
            aligned[key] = (exp_props[key], calc_props[key])
        return aligned
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """计算综合评分"""
        scores = []
        
        if 'xrd_comparison' in results:
            scores.append(results['xrd_comparison'].get('agreement_score', 0))
        
        if 'structure_comparison' in results:
            scores.append(results['structure_comparison'].get('confidence', 0))
        
        if 'property_comparison' in results:
            prop_scores = [
                1.0 if v['within_tolerance'] else 0.0
                for v in results['property_comparison'].values()
            ]
            scores.append(np.mean(prop_scores) if prop_scores else 0)
        
        return np.mean(scores) if scores else 0.0


# ==================== 主入口函数 ====================

def compare_calculation_experiment(calc_data: Dict[str, Any],
                                  exp_data: Dict[str, Any]) -> Dict[str, Any]:
    """对比计算和实验数据"""
    manager = ComparisonManager()
    return manager.full_comparison(exp_data, calc_data)


def validate_calculation(calc_data: Dict[str, Any],
                        exp_data: Dict[str, Any],
                        tolerance: float = 0.1) -> bool:
    """验证计算结果"""
    comparison = compare_calculation_experiment(calc_data, exp_data)
    
    overall_score = comparison.get('overall_score', 0)
    return overall_score >= (1 - tolerance)


def generate_validation_report(calc_data: Dict[str, Any],
                              exp_data: Dict[str, Any]) -> str:
    """生成验证报告"""
    comparison = compare_calculation_experiment(calc_data, exp_data)
    
    report = ["=" * 60]
    report.append("计算-实验验证报告")
    report.append("=" * 60)
    report.append("")
    
    report.append(f"综合评分: {comparison.get('overall_score', 0):.2%}")
    report.append("")
    
    # 各部分详细结果
    if 'xrd_comparison' in comparison:
        report.append("XRD对比:")
        xrd = comparison['xrd_comparison']
        report.append(f"  匹配分数: {xrd.get('agreement_score', 0):.2%}")
        report.append(f"  通过验证: {'是' if xrd.get('validation_passed') else '否'}")
        report.append("")
    
    if 'property_comparison' in comparison:
        report.append("性质对比:")
        for prop, data in comparison['property_comparison'].items():
            status = "✓" if data['within_tolerance'] else "✗"
            report.append(f"  {status} {prop}: {data['relative_error']:.2%} error")
        report.append("")
    
    # 建议
    if 'suggestions' in comparison:
        report.append("改进建议:")
        for suggestion in comparison['suggestions']:
            report.append(f"  - {suggestion}")
    
    return "\n".join(report)


# 示例用法
if __name__ == "__main__":
    # 模拟对比
    exp_data = {
        "properties": {
            "band_gap": 3.2,
            "lattice_constant": 4.18,
            "density": 4.23
        }
    }
    
    calc_data = {
        "properties": {
            "band_gap": 3.15,
            "lattice_constant": 4.20,
            "density": 4.18
        }
    }
    
    # 对比
    comparison = compare_calculation_experiment(calc_data, exp_data)
    
    print(generate_validation_report(calc_data, exp_data))
