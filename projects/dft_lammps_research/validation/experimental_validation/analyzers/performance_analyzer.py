"""
Performance Analyzer
====================
性能对比分析工具

用于比较实验电化学性能与模拟预测
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from ..connectors.base_connector import ExperimentalData

logger = logging.getLogger(__name__)


class ElectrochemicalComparator:
    """
    电化学性能比较器
    
    支持:
    - 充放电曲线对比
    - 容量对比
    - 循环稳定性对比
    - 倍率性能对比
    - 电压曲线对比
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.config.setdefault('voltage_range', (0.0, 5.0))
        self.config.setdefault('capacity_range', (0.0, 500.0))
        self.config.setdefault('interpolation_points', 500)
    
    def compare_gcd_curves(self,
                          experimental: ExperimentalData,
                          simulated: ExperimentalData,
                          cycle_number: int = 1) -> Dict[str, float]:
        """
        比较恒流充放电曲线
        
        Args:
            experimental: 实验数据
            simulated: 模拟数据
            cycle_number: 要比较的循环号
            
        Returns:
            对比指标
        """
        # 提取指定循环的数据
        exp_cycle = self._extract_cycle(experimental, cycle_number)
        sim_cycle = self._extract_cycle(simulated, cycle_number)
        
        if exp_cycle is None or sim_cycle is None:
            logger.warning(f"Cycle {cycle_number} not found in data")
            return {}
        
        results = {}
        
        # 计算容量
        exp_capacity = self._calculate_capacity(exp_cycle)
        sim_capacity = self._calculate_capacity(sim_cycle)
        
        results['exp_capacity_mAh_g'] = float(exp_capacity)
        results['sim_capacity_mAh_g'] = float(sim_capacity)
        results['capacity_error_percent'] = float(
            abs(exp_capacity - sim_capacity) / exp_capacity * 100 if exp_capacity > 0 else 0
        )
        
        # 电压曲线相似度
        voltage_similarity = self._compare_voltage_curves(exp_cycle, sim_cycle)
        results.update(voltage_similarity)
        
        # 平均电压
        exp_avg_v = self._calculate_average_voltage(exp_cycle)
        sim_avg_v = self._calculate_average_voltage(sim_cycle)
        results['exp_avg_voltage'] = float(exp_avg_v)
        results['sim_avg_voltage'] = float(sim_avg_v)
        results['avg_voltage_error'] = float(abs(exp_avg_v - sim_avg_v))
        
        # 能量密度
        exp_energy = self._calculate_energy_density(exp_cycle)
        sim_energy = self._calculate_energy_density(sim_cycle)
        results['exp_energy_density_Wh_kg'] = float(exp_energy)
        results['sim_energy_density_Wh_kg'] = float(sim_energy)
        results['energy_density_error_percent'] = float(
            abs(exp_energy - sim_energy) / exp_energy * 100 if exp_energy > 0 else 0
        )
        
        return results
    
    def _extract_cycle(self, data: ExperimentalData, cycle_number: int) -> Optional[np.ndarray]:
        """提取指定循环的数据"""
        df = data.to_dataframe()
        
        if 'cycle' not in df.columns:
            # 假设所有数据是一个循环
            return data.processed_data
        
        cycle_data = df[df['cycle'] == cycle_number]
        if len(cycle_data) == 0:
            return None
        
        return cycle_data.values
    
    def _calculate_capacity(self, data: np.ndarray, col_idx: int = -1) -> float:
        """计算比容量"""
        if data is None or len(data) == 0:
            return 0.0
        
        # 假设容量列是最后一列或特定列
        # 简化处理：取最大容量值
        try:
            capacity_col = data[:, col_idx]
            return float(np.max(capacity_col))
        except:
            return 0.0
    
    def _compare_voltage_curves(self, 
                               exp_data: np.ndarray,
                               sim_data: np.ndarray) -> Dict[str, float]:
        """比较电压曲线"""
        # 提取电压和容量
        # 假设列为: [time, voltage, current, capacity, ...]
        
        try:
            exp_v = exp_data[:, 1]  # 电压列
            exp_cap = exp_data[:, 3] if exp_data.shape[1] > 3 else exp_data[:, 0]
            
            sim_v = sim_data[:, 1]
            sim_cap = sim_data[:, 3] if sim_data.shape[1] > 3 else sim_data[:, 0]
            
            # 对齐到共同容量网格
            min_cap = max(np.min(exp_cap), np.min(sim_cap))
            max_cap = min(np.max(exp_cap), np.max(sim_cap))
            
            if min_cap >= max_cap:
                return {'voltage_rmse': 0.0, 'voltage_mae': 0.0}
            
            common_cap = np.linspace(min_cap, max_cap, self.config['interpolation_points'])
            
            exp_interp = interp1d(exp_cap, exp_v, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')(common_cap)
            sim_interp = interp1d(sim_cap, sim_v, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')(common_cap)
            
            # 计算误差
            rmse = np.sqrt(np.mean((exp_interp - sim_interp)**2))
            mae = np.mean(np.abs(exp_interp - sim_interp))
            
            return {
                'voltage_rmse': float(rmse),
                'voltage_mae': float(mae),
            }
        except Exception as e:
            logger.warning(f"Failed to compare voltage curves: {e}")
            return {'voltage_rmse': 0.0, 'voltage_mae': 0.0}
    
    def _calculate_average_voltage(self, data: np.ndarray) -> float:
        """计算平均电压"""
        if data is None or len(data) == 0:
            return 0.0
        
        try:
            voltage = data[:, 1]
            return float(np.mean(voltage))
        except:
            return 0.0
    
    def _calculate_energy_density(self, data: np.ndarray) -> float:
        """计算能量密度 (Wh/kg)"""
        if data is None or len(data) < 2:
            return 0.0
        
        try:
            voltage = data[:, 1]
            capacity = data[:, 3] if data.shape[1] > 3 else np.arange(len(data))
            
            # 数值积分计算能量
            energy = np.trapz(voltage, capacity)  # mAh/g * V = mWh/g
            return float(energy)  # Wh/kg = mWh/g
        except:
            return 0.0
    
    def compare_cycling_stability(self,
                                 experimental: ExperimentalData,
                                 simulated: ExperimentalData,
                                 n_cycles: int = 100) -> Dict[str, float]:
        """
        比较循环稳定性
        
        Args:
            experimental: 实验循环数据
            simulated: 模拟循环数据
            n_cycles: 比较的循环数
            
        Returns:
            稳定性指标
        """
        # 提取每循环的容量
        exp_capacities = self._extract_cycle_capacities(experimental, n_cycles)
        sim_capacities = self._extract_cycle_capacities(simulated, n_cycles)
        
        results = {}
        
        if len(exp_capacities) > 0 and len(sim_capacities) > 0:
            # 容量保持率
            exp_retention = exp_capacities[-1] / exp_capacities[0] * 100 if exp_capacities[0] > 0 else 0
            sim_retention = sim_capacities[-1] / sim_capacities[0] * 100 if sim_capacities[0] > 0 else 0
            
            results['exp_capacity_retention_percent'] = float(exp_retention)
            results['sim_capacity_retention_percent'] = float(sim_retention)
            results['retention_error_percent'] = float(abs(exp_retention - sim_retention))
            
            # 容量衰减率
            if len(exp_capacities) > 1:
                exp_decay = (exp_capacities[0] - exp_capacities[-1]) / (len(exp_capacities) - 1)
                sim_decay = (sim_capacities[0] - sim_capacities[-1]) / (len(sim_capacities) - 1)
                
                results['exp_capacity_decay_per_cycle'] = float(exp_decay)
                results['sim_capacity_decay_per_cycle'] = float(sim_decay)
            
            # 循环曲线RMSE
            min_len = min(len(exp_capacities), len(sim_capacities))
            if min_len > 0:
                rmse = np.sqrt(np.mean(
                    (np.array(exp_capacities[:min_len]) - np.array(sim_capacities[:min_len]))**2
                ))
                results['cycling_rmse'] = float(rmse)
        
        return results
    
    def _extract_cycle_capacities(self, data: ExperimentalData, n_cycles: int) -> List[float]:
        """提取每循环的容量"""
        from ..connectors.electrochemical_connector import ElectrochemicalConnector
        
        connector = ElectrochemicalConnector()
        cycles = connector.extract_cycles(data)
        
        capacities = []
        for i in range(1, min(n_cycles + 1, max(cycles.keys()) + 1)):
            if i in cycles:
                cycle_data = cycles[i].to_dataframe()
                if 'capacity' in cycle_data.columns:
                    capacities.append(float(cycle_data['capacity'].max()))
                elif 'specific_capacity' in cycle_data.columns:
                    capacities.append(float(cycle_data['specific_capacity'].max()))
        
        return capacities
    
    def compare_rate_capability(self,
                               experimental: ExperimentalData,
                               simulated: ExperimentalData,
                               current_rates: List[float] = None) -> Dict[str, Any]:
        """
        比较倍率性能
        
        Args:
            experimental: 实验倍率数据
            simulated: 模拟倍率数据
            current_rates: 电流倍率列表 [0.1C, 0.2C, 0.5C, 1C, 2C, 5C]
            
        Returns:
            倍率性能指标
        """
        current_rates = current_rates or [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        
        # 提取各倍率下的容量
        exp_capacities = self._extract_rate_capacities(experimental, current_rates)
        sim_capacities = self._extract_rate_capacities(simulated, current_rates)
        
        results = {
            'current_rates_C': current_rates,
            'exp_capacities_mAh_g': exp_capacities,
            'sim_capacities_mAh_g': sim_capacities,
        }
        
        # 计算各倍率的误差
        if len(exp_capacities) == len(sim_capacities):
            errors = [
                abs(e - s) / e * 100 if e > 0 else 0 
                for e, s in zip(exp_capacities, sim_capacities)
            ]
            results['capacity_errors_percent'] = errors
            results['mean_error_percent'] = float(np.mean(errors))
        
        return results
    
    def _extract_rate_capacities(self, 
                                data: ExperimentalData,
                                current_rates: List[float]) -> List[float]:
        """提取各倍率下的容量"""
        df = data.to_dataframe()
        capacities = []
        
        # 假设数据按倍率分组或有倍率列
        # 简化处理：按数据段分割
        n_rates = len(current_rates)
        n_points = len(df) // n_rates if n_rates > 0 else len(df)
        
        for i in range(n_rates):
            start_idx = i * n_points
            end_idx = min((i + 1) * n_points, len(df))
            
            if start_idx < len(df):
                segment = df.iloc[start_idx:end_idx]
                if 'capacity' in segment.columns:
                    capacities.append(float(segment['capacity'].max()))
                elif 'specific_capacity' in segment.columns:
                    capacities.append(float(segment['specific_capacity'].max()))
                else:
                    capacities.append(0.0)
            else:
                capacities.append(0.0)
        
        return capacities
    
    def compare_cv_curves(self,
                         experimental: ExperimentalData,
                         simulated: ExperimentalData) -> Dict[str, float]:
        """
        比较循环伏安曲线
        
        Args:
            experimental: 实验CV数据
            simulated: 模拟CV数据
            
        Returns:
            CV对比指标
        """
        # 提取峰
        exp_peaks = self._find_cv_peaks(experimental)
        sim_peaks = self._find_cv_peaks(simulated)
        
        results = {
            'exp_anodic_peaks': len(exp_peaks.get('anodic', [])),
            'exp_cathodic_peaks': len(exp_peaks.get('cathodic', [])),
            'sim_anodic_peaks': len(sim_peaks.get('anodic', [])),
            'sim_cathodic_peaks': len(sim_peaks.get('cathodic', [])),
        }
        
        # 比较峰位置
        if exp_peaks.get('anodic') and sim_peaks.get('anodic'):
            exp_positions = [p['potential'] for p in exp_peaks['anodic']]
            sim_positions = [p['potential'] for p in sim_peaks['anodic']]
            
            results['anodic_peak_potential_error'] = float(
                np.mean([abs(e - s) for e, s in zip(exp_positions, sim_positions)])
            )
        
        if exp_peaks.get('cathodic') and sim_peaks.get('cathodic'):
            exp_positions = [p['potential'] for p in exp_peaks['cathodic']]
            sim_positions = [p['potential'] for p in sim_peaks['cathodic']]
            
            results['cathodic_peak_potential_error'] = float(
                np.mean([abs(e - s) for e, s in zip(exp_positions, sim_positions)])
            )
        
        return results
    
    def _find_cv_peaks(self, data: ExperimentalData) -> Dict[str, List[Dict]]:
        """查找CV曲线中的氧化还原峰"""
        from scipy.signal import find_peaks
        
        df = data.to_dataframe()
        
        if 'potential' not in df.columns or 'current' not in df.columns:
            return {'anodic': [], 'cathodic': []}
        
        potential = df['potential'].values
        current = df['current'].values
        
        # 分离氧化峰（正电流）和还原峰（负电流）
        anodic_peaks, _ = find_peaks(current, prominence=0.1 * np.max(np.abs(current)))
        cathodic_peaks, _ = find_peaks(-current, prominence=0.1 * np.max(np.abs(current)))
        
        return {
            'anodic': [
                {'potential': float(potential[i]), 'current': float(current[i])}
                for i in anodic_peaks
            ],
            'cathodic': [
                {'potential': float(potential[i]), 'current': float(current[i])}
                for i in cathodic_peaks
            ],
        }


class PropertyComparator:
    """
    通用属性比较器
    
    用于比较各种物理化学性质
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def compare_scalar_properties(self,
                                 exp_value: float,
                                 sim_value: float,
                                 property_name: str = "",
                                 tolerance_percent: float = 10.0) -> Dict[str, Any]:
        """
        比较标量属性
        
        Args:
            exp_value: 实验值
            sim_value: 模拟值
            property_name: 属性名称
            tolerance_percent: 容差百分比
            
        Returns:
            比较结果
        """
        absolute_error = abs(exp_value - sim_value)
        relative_error_percent = (absolute_error / abs(exp_value) * 100) if exp_value != 0 else float('inf')
        
        within_tolerance = relative_error_percent <= tolerance_percent
        
        return {
            'property': property_name,
            'experimental_value': float(exp_value),
            'simulated_value': float(sim_value),
            'absolute_error': float(absolute_error),
            'relative_error_percent': float(relative_error_percent),
            'tolerance_percent': tolerance_percent,
            'within_tolerance': within_tolerance,
        }
    
    def compare_band_gap(self,
                        exp_band_gap: float,
                        sim_band_gap: float,
                        tolerance_ev: float = 0.2) -> Dict[str, Any]:
        """比较带隙"""
        result = self.compare_scalar_properties(
            exp_band_gap, sim_band_gap, "band_gap", tolerance_percent=10.0
        )
        result['within_absolute_tolerance'] = abs(exp_band_gap - sim_band_gap) <= tolerance_ev
        result['absolute_tolerance_eV'] = tolerance_ev
        return result
    
    def compare_lattice_constants(self,
                                 exp_constants: Tuple[float, float, float],
                                 sim_constants: Tuple[float, float, float],
                                 tolerance_percent: float = 5.0) -> Dict[str, Any]:
        """
        比较晶格常数
        
        Args:
            exp_constants: (a, b, c) 实验值
            sim_constants: (a, b, c) 模拟值
            tolerance_percent: 容差百分比
        """
        exp_a, exp_b, exp_c = exp_constants
        sim_a, sim_b, sim_c = sim_constants
        
        results = {
            'a': self.compare_scalar_properties(exp_a, sim_a, "lattice_a", tolerance_percent),
            'b': self.compare_scalar_properties(exp_b, sim_b, "lattice_b", tolerance_percent),
            'c': self.compare_scalar_properties(exp_c, sim_c, "lattice_c", tolerance_percent),
        }
        
        # 体积比较
        exp_volume = exp_a * exp_b * exp_c
        sim_volume = sim_a * sim_b * sim_c
        results['volume'] = self.compare_scalar_properties(
            exp_volume, sim_volume, "volume", tolerance_percent
        )
        
        results['all_within_tolerance'] = all([
            results['a']['within_tolerance'],
            results['b']['within_tolerance'],
            results['c']['within_tolerance'],
        ])
        
        return results
    
    def compare_ion_conductivity(self,
                                exp_conductivity: float,
                                sim_conductivity: float,
                                temperature_exp: float = 300,
                                temperature_sim: float = 300) -> Dict[str, Any]:
        """
        比较离子电导率
        
        如果温度不同，会尝试进行温度校正
        """
        # 简单的Arrhenius校正（简化）
        if abs(temperature_exp - temperature_sim) > 10:
            # 假设活化能约为0.5 eV
            Ea = 0.5  # eV
            kB = 8.617e-5  # eV/K
            
            # 校正到相同温度
            correction = np.exp(Ea / kB * (1/temperature_sim - 1/temperature_exp))
            sim_conductivity_corrected = sim_conductivity * correction
        else:
            sim_conductivity_corrected = sim_conductivity
        
        result = self.compare_scalar_properties(
            exp_conductivity, sim_conductivity_corrected, "ion_conductivity", tolerance_percent=50.0
        )
        
        result['temperature_experimental_K'] = temperature_exp
        result['temperature_simulated_K'] = temperature_sim
        result['original_simulated_value'] = float(sim_conductivity)
        
        return result
    
    def compare_arrays(self,
                      exp_array: np.ndarray,
                      sim_array: np.ndarray,
                      x_values: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        比较两个数组
        
        Args:
            exp_array: 实验数据数组
            sim_array: 模拟数据数组
            x_values: x轴值（用于插值对齐）
            
        Returns:
            比较指标
        """
        # 确保长度相同
        min_len = min(len(exp_array), len(sim_array))
        exp_array = exp_array[:min_len]
        sim_array = sim_array[:min_len]
        
        # 计算各种误差
        mae = np.mean(np.abs(exp_array - sim_array))
        rmse = np.sqrt(np.mean((exp_array - sim_array)**2))
        
        # 相对误差
        mask = exp_array != 0
        if np.any(mask):
            mape = np.mean(np.abs((exp_array[mask] - sim_array[mask]) / exp_array[mask])) * 100
        else:
            mape = float('inf')
        
        # 最大误差
        max_error = np.max(np.abs(exp_array - sim_array))
        
        # R²
        ss_res = np.sum((exp_array - sim_array)**2)
        ss_tot = np.sum((exp_array - np.mean(exp_array))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape_percent': float(mape),
            'max_error': float(max_error),
            'r2': float(r2),
        }
