"""
XRD Comparator
==============
XRD图谱比较工具

用于比较实验XRD和模拟XRD图谱
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..connectors.base_connector import ExperimentalData

logger = logging.getLogger(__name__)


class XRDComparator:
    """
    XRD图谱比较器
    
    支持多种相似度度量:
    - Pearson相关系数
    - 余弦相似度
    - Rwp (R-weighted pattern)
    - Rexp (Expected R)
    - GoF (Goodness of Fit)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.config.setdefault('two_theta_range', (5, 90))
        self.config.setdefault('interpolation_points', 1000)
        self.config.setdefault('peak_tolerance', 0.2)  # 2θ容差
    
    def compare(self, 
                experimental: ExperimentalData,
                simulated: ExperimentalData,
                methods: List[str] = None) -> Dict[str, float]:
        """
        比较实验和模拟XRD图谱
        
        Args:
            experimental: 实验XRD数据
            simulated: 模拟XRD数据
            methods: 比较方法列表 ['pearson', 'cosine', 'rwp', 'all']
            
        Returns:
            相似度指标字典
        """
        methods = methods or ['all']
        
        # 对齐数据
        exp_interp, sim_interp = self._align_data(experimental, simulated)
        
        results = {}
        
        if 'all' in methods or 'pearson' in methods:
            results['pearson_r'], results['pearson_p'] = self._pearson_correlation(
                exp_interp, sim_interp
            )
        
        if 'all' in methods or 'cosine' in methods:
            results['cosine_similarity'] = self._cosine_similarity(
                exp_interp, sim_interp
            )
        
        if 'all' in methods or 'rwp' in methods:
            results['rwp'] = self._calculate_rwp(exp_interp, sim_interp)
        
        if 'all' in methods or 'rexp' in methods:
            results['rexp'] = self._calculate_rexp(exp_interp)
        
        if 'all' in methods or 'gof' in methods:
            if 'rwp' not in results:
                results['rwp'] = self._calculate_rwp(exp_interp, sim_interp)
            if 'rexp' not in results:
                results['rexp'] = self._calculate_rexp(exp_interp)
            results['gof'] = results['rwp'] / results['rexp'] if results['rexp'] > 0 else float('inf')
        
        if 'all' in methods or 'mae' in methods:
            results['mae'] = np.mean(np.abs(exp_interp - sim_interp))
        
        if 'all' in methods or 'rmse' in methods:
            results['rmse'] = np.sqrt(np.mean((exp_interp - sim_interp)**2))
        
        # 峰匹配
        if 'all' in methods or 'peak_match' in methods:
            results['peak_match_score'] = self._peak_matching(experimental, simulated)
        
        return results
    
    def _align_data(self, 
                   experimental: ExperimentalData,
                   simulated: ExperimentalData) -> Tuple[np.ndarray, np.ndarray]:
        """
        对齐实验和模拟数据到统一网格
        """
        # 获取原始数据
        exp_data = experimental.processed_data
        sim_data = simulated.processed_data
        
        exp_2theta = exp_data[:, 0]
        exp_intensity = exp_data[:, 1]
        sim_2theta = sim_data[:, 0]
        sim_intensity = sim_data[:, 1]
        
        # 确定共同范围
        min_2theta = max(
            np.min(exp_2theta), 
            np.min(sim_2theta),
            self.config['two_theta_range'][0]
        )
        max_2theta = min(
            np.max(exp_2theta),
            np.max(sim_2theta),
            self.config['two_theta_range'][1]
        )
        
        # 创建统一网格
        n_points = self.config['interpolation_points']
        common_2theta = np.linspace(min_2theta, max_2theta, n_points)
        
        # 插值
        exp_interp_func = interp1d(exp_2theta, exp_intensity, 
                                   kind='linear', bounds_error=False, fill_value=0)
        sim_interp_func = interp1d(sim_2theta, sim_intensity,
                                   kind='linear', bounds_error=False, fill_value=0)
        
        exp_interp = exp_interp_func(common_2theta)
        sim_interp = sim_interp_func(common_2theta)
        
        # 归一化
        exp_interp = self._normalize(exp_interp)
        sim_interp = self._normalize(sim_interp)
        
        return exp_interp, sim_interp
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """归一化到0-1范围"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        return data
    
    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """计算Pearson相关系数"""
        r, p = pearsonr(x, y)
        return float(r), float(p)
    
    def _cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x > 0 and norm_y > 0:
            return float(dot_product / (norm_x * norm_y))
        return 0.0
    
    def _calculate_rwp(self, y_obs: np.ndarray, y_calc: np.ndarray) -> float:
        """
        计算Rwp (R-weighted pattern)
        
        Rwp = sqrt(sum(w_i * (y_obs - y_calc)^2) / sum(w_i * y_obs^2))
        
        这里简化权重w_i = 1
        """
        numerator = np.sum((y_obs - y_calc)**2)
        denominator = np.sum(y_obs**2)
        
        if denominator > 0:
            return float(np.sqrt(numerator / denominator))
        return float('inf')
    
    def _calculate_rexp(self, y_obs: np.ndarray) -> float:
        """
        计算Rexp (Expected R)
        
        Rexp = sqrt((N - P) / sum(w_i * y_obs^2))
        
        这里简化处理
        """
        N = len(y_obs)
        P = 1  # 假设1个参数
        denominator = np.sum(y_obs**2)
        
        if denominator > 0:
            return float(np.sqrt((N - P) / denominator))
        return 0.0
    
    def _peak_matching(self, 
                      experimental: ExperimentalData,
                      simulated: ExperimentalData) -> float:
        """
        峰匹配得分
        
        基于峰位置的匹配程度
        """
        from ..connectors.xrd_connector import XRDConnector
        
        connector = XRDConnector()
        
        # 查找峰
        exp_peaks = connector.find_peaks(experimental, prominence=0.1)
        sim_peaks = connector.find_peaks(simulated, prominence=0.1)
        
        if not exp_peaks or not sim_peaks:
            return 0.0
        
        exp_positions = [p['two_theta'] for p in exp_peaks]
        sim_positions = [p['two_theta'] for p in sim_peaks]
        
        tolerance = self.config['peak_tolerance']
        
        # 匹配峰
        matched = 0
        for exp_pos in exp_positions:
            for sim_pos in sim_positions:
                if abs(exp_pos - sim_pos) < tolerance:
                    matched += 1
                    break
        
        # 匹配得分
        total_peaks = max(len(exp_positions), len(sim_positions))
        if total_peaks > 0:
            return matched / total_peaks
        return 0.0
    
    def calculate_difference_profile(self,
                                    experimental: ExperimentalData,
                                    simulated: ExperimentalData) -> np.ndarray:
        """
        计算差分图谱
        
        Returns:
            差分数据 [2θ, exp, sim, diff]
        """
        # 对齐数据
        exp_data = experimental.processed_data
        sim_data = simulated.processed_data
        
        exp_2theta = exp_data[:, 0]
        exp_intensity = exp_data[:, 1]
        sim_2theta = sim_data[:, 0]
        sim_intensity = sim_data[:, 1]
        
        # 创建共同网格
        min_2theta = max(np.min(exp_2theta), np.min(sim_2theta))
        max_2theta = min(np.max(exp_2theta), np.max(sim_2theta))
        common_2theta = np.linspace(min_2theta, max_2theta, 1000)
        
        # 插值
        exp_interp = interp1d(exp_2theta, exp_intensity, 
                             kind='linear', bounds_error=False, fill_value=0)(common_2theta)
        sim_interp = interp1d(sim_2theta, sim_intensity,
                             kind='linear', bounds_error=False, fill_value=0)(common_2theta)
        
        diff = exp_interp - sim_interp
        
        return np.column_stack([common_2theta, exp_interp, sim_interp, diff])


class StructureComparator:
    """
    结构比较器
    
    比较晶体结构的相似性
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def compare_structures(self, 
                          structure1: Any,
                          structure2: Any) -> Dict[str, float]:
        """
        比较两个晶体结构
        
        Args:
            structure1: 结构1 (pymatgen Structure或类似)
            structure2: 结构2
            
        Returns:
            相似度指标
        """
        try:
            from pymatgen.analysis.structure_matcher import StructureMatcher
            
            matcher = StructureMatcher()
            
            # 检查是否匹配
            is_match = matcher.fit(structure1, structure2)
            
            results = {
                'is_match': is_match,
            }
            
            if is_match:
                # 计算RMS距离
                rms_dist = matcher.get_rms_dist(structure1, structure2)
                results['rms_distance'] = float(rms_dist[0]) if rms_dist else None
            
            # 晶格参数比较
            lattice_diff = self._compare_lattices(structure1.lattice, structure2.lattice)
            results.update(lattice_diff)
            
            return results
            
        except ImportError:
            logger.error("pymatgen is required for structure comparison")
            return {'is_match': False, 'error': 'pymatgen not available'}
    
    def _compare_lattices(self, lattice1, lattice2) -> Dict[str, float]:
        """比较晶格参数"""
        params1 = lattice1.parameters
        params2 = lattice2.parameters
        
        # 计算相对差异
        diff_a = abs(params1[0] - params2[0]) / params1[0] * 100
        diff_b = abs(params1[1] - params2[1]) / params1[1] * 100
        diff_c = abs(params1[2] - params2[2]) / params1[2] * 100
        diff_alpha = abs(params1[3] - params2[3])
        diff_beta = abs(params1[4] - params2[4])
        diff_gamma = abs(params1[5] - params2[5])
        
        return {
            'delta_a_percent': float(diff_a),
            'delta_b_percent': float(diff_b),
            'delta_c_percent': float(diff_c),
            'delta_alpha': float(diff_alpha),
            'delta_beta': float(diff_beta),
            'delta_gamma': float(diff_gamma),
            'max_cell_param_diff_percent': float(max(diff_a, diff_b, diff_c)),
        }
    
    def calculate_xrd_similarity(self,
                                structure1: Any,
                                structure2: Any,
                                wavelength: float = 1.5406) -> float:
        """
        基于XRD图谱的结构相似度
        
        Args:
            structure1: 结构1
            structure2: 结构2
            wavelength: X射线波长
            
        Returns:
            相似度得分 (0-1)
        """
        try:
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            calculator = XRDCalculator(wavelength=wavelength)
            
            pattern1 = calculator.get_pattern(structure1)
            pattern2 = calculator.get_pattern(structure2)
            
            # 转换为实验数据格式
            data1 = ExperimentalData(
                data_type='xrd',
                raw_data=np.column_stack([pattern1.x, pattern1.y]),
                column_names=['two_theta', 'intensity'],
                units={}
            )
            data2 = ExperimentalData(
                data_type='xrd',
                raw_data=np.column_stack([pattern2.x, pattern2.y]),
                column_names=['two_theta', 'intensity'],
                units={}
            )
            
            # 比较
            comparator = XRDComparator()
            results = comparator.compare(data1, data2, methods=['cosine'])
            
            return results.get('cosine_similarity', 0.0)
            
        except ImportError:
            logger.error("pymatgen is required for XRD calculation")
            return 0.0
