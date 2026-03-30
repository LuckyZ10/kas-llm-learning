"""
Parameter Transfer Module
=========================
多尺度参数传递模块

实现DFT/MD与相场之间的自动化参数传递和转换。
处理不同尺度间的单位转换和参数映射。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransferConfig:
    """参数传递配置"""
    # 长度尺度转换
    atomistic_to_meso_scale: float = 1e9  # m to nm
    
    # 能量尺度转换
    ev_to_joule: float = 1.60218e-19
    
    # 时间尺度转换
    fs_to_s: float = 1e-15
    
    # 温度
    reference_temperature: float = 298.15  # K
    
    # 参数平滑
    smoothing_method: str = "gaussian"  # gaussian, spline, linear
    smoothing_width: float = 0.1
    
    # 不确定性量化
    uncertainty_quantification: bool = True
    confidence_level: float = 0.95


class ParameterTransfer:
    """
    多尺度参数传递类
    
    自动化处理从微观(DFT/MD)到介观(相场)的参数传递。
    """
    
    def __init__(self, config: Optional[TransferConfig] = None):
        """
        初始化参数传递
        
        Args:
            config: 传递配置
        """
        self.config = config or TransferConfig()
        
        # 存储转换后的参数
        self.transferred_params = {}
        self.uncertainties = {}
        
        logger.info("Parameter transfer initialized")
    
    def transfer_from_dft(self, dft_params: Dict) -> Dict:
        """
        从DFT参数转换到相场参数
        
        Args:
            dft_params: DFT计算的参数
            
        Returns:
            pf_params: 相场参数
        """
        pf_params = {}
        
        # 1. 化学势
        if 'chemical_potential' in dft_params.get('thermodynamic', {}):
            mu_func = dft_params['thermodynamic']['chemical_potential']['function']
            pf_params['chemical_potential'] = mu_func
        
        # 2. 梯度能量系数 κ
        if 'gradient_energy_coefficient' in dft_params.get('thermodynamic', {}):
            kappa = dft_params['thermodynamic']['gradient_energy_coefficient']
            # 单位转换: J/m -> eV/nm
            kappa_pf = kappa * self.config.ev_to_joule * self.config.atomistic_to_meso_scale
            pf_params['kappa'] = kappa_pf
        
        # 3. 弹性常数
        if 'elastic_constants' in dft_params.get('thermodynamic', {}):
            C = dft_params['thermodynamic']['elastic_constants']
            # 保持GPa单位
            pf_params['elastic'] = {
                'C11': float(C[0, 0]),
                'C12': float(C[0, 1]),
                'C44': float(C[3, 3])
            }
        
        # 4. 界面能
        if 'interface_energy' in dft_params.get('thermodynamic', {}):
            gamma = dft_params['thermodynamic']['interface_energy']
            # 单位已经是J/m²，直接使用
            pf_params['interface_energy'] = gamma
        
        self.transferred_params['from_dft'] = pf_params
        
        logger.info("Transferred parameters from DFT to phase field")
        
        return pf_params
    
    def transfer_from_md(self, md_params: Dict) -> Dict:
        """
        从MD参数转换到相场参数
        
        Args:
            md_params: MD计算的参数
            
        Returns:
            pf_params: 相场参数
        """
        pf_params = {}
        
        # 1. 扩散系数/迁移率
        if 'diffusion_coefficient' in md_params.get('transport', {}):
            D_data = md_params['transport']['diffusion_coefficient']
            D = D_data['value']  # m²/s
            
            # 转换到相场单位: nm²/s
            D_pf = D * (self.config.atomistic_to_meso_scale ** 2)
            
            # 迁移率 M = D / (RT/Vm) (简化)
            M_pf = D_pf / (8.314 * 298.15 / 1e-4)  # 简化单位转换
            
            pf_params['diffusion_coefficient'] = D_pf
            pf_params['mobility'] = M_pf
        
        # 2. 激活能
        if 'activation_energy' in md_params.get('transport', {}):
            Ea = md_params['transport']['activation_energy']  # eV
            pf_params['activation_energy'] = Ea
        
        # 3. 结构参数
        if 'structure' in md_params:
            struct = md_params['structure']
            if 'density_mean' in struct:
                pf_params['density'] = struct['density_mean']
        
        self.transferred_params['from_md'] = pf_params
        
        logger.info("Transferred parameters from MD to phase field")
        
        return pf_params
    
    def merge_parameters(self, dft_params: Dict, md_params: Dict,
                        weights: Optional[Dict] = None) -> Dict:
        """
        合并DFT和MD参数
        
        Args:
            dft_params: DFT参数
            md_params: MD参数
            weights: 权重字典 {'dft': w1, 'md': w2}
            
        Returns:
            merged_params: 合并后的参数
        """
        weights = weights or {'dft': 0.5, 'md': 0.5}
        
        merged = {}
        
        # 共同参数
        common_keys = set(dft_params.keys()) & set(md_params.keys())
        
        for key in common_keys:
            v_dft = dft_params[key]
            v_md = md_params[key]
            
            if isinstance(v_dft, (int, float)) and isinstance(v_md, (int, float)):
                # 加权平均
                merged[key] = weights['dft'] * v_dft + weights['md'] * v_md
                
                # 不确定性估计
                if self.config.uncertainty_quantification:
                    variance = (weights['dft'] * (v_dft - merged[key])**2 + 
                               weights['md'] * (v_md - merged[key])**2)
                    self.uncertainties[key] = np.sqrt(variance)
            else:
                # 非数值参数，优先使用DFT
                merged[key] = v_dft
        
        # DFT特有参数
        for key in set(dft_params.keys()) - common_keys:
            merged[key] = dft_params[key]
        
        # MD特有参数
        for key in set(md_params.keys()) - common_keys:
            merged[key] = md_params[key]
        
        self.transferred_params['merged'] = merged
        
        logger.info("Merged DFT and MD parameters")
        
        return merged
    
    def interpolate_parameters(self, 
                               compositions: np.ndarray,
                               parameters: np.ndarray,
                               target_compositions: np.ndarray) -> np.ndarray:
        """
        插值参数到目标成分
        
        Args:
            compositions: 已知成分数组
            parameters: 对应参数数组
            target_compositions: 目标成分数组
            
        Returns:
            interpolated: 插值后的参数
        """
        from scipy.interpolate import interp1d, UnivariateSpline
        
        if self.config.smoothing_method == "linear":
            interp_func = interp1d(compositions, parameters, 
                                  kind='linear', fill_value='extrapolate')
        elif self.config.smoothing_method == "spline":
            interp_func = UnivariateSpline(compositions, parameters, 
                                          s=self.config.smoothing_width)
        else:  # gaussian or default
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(parameters, 
                                        sigma=len(parameters)*self.config.smoothing_width)
            interp_func = interp1d(compositions, smoothed, 
                                  kind='cubic', fill_value='extrapolate')
        
        interpolated = interp_func(target_compositions)
        
        return interpolated
    
    def validate_transfer(self, pf_params: Dict, 
                         reference_values: Optional[Dict] = None) -> Dict:
        """
        验证参数传递的正确性
        
        Args:
            pf_params: 传递后的相场参数
            reference_values: 参考值 (实验或文献)
            
        Returns:
            validation: 验证结果
        """
        validation = {
            'status': 'ok',
            'warnings': [],
            'errors': [],
            'comparisons': {}
        }
        
        # 检查参数合理性
        if 'mobility' in pf_params:
            M = pf_params['mobility']
            if M < 0 or M > 1e10:
                validation['warnings'].append(
                    f"Unusual mobility value: {M:.2e}"
                )
        
        if 'kappa' in pf_params:
            kappa = pf_params['kappa']
            if kappa < 0:
                validation['errors'].append("Negative gradient energy coefficient!")
        
        # 与参考值对比
        if reference_values:
            for key, ref_val in reference_values.items():
                if key in pf_params:
                    calc_val = pf_params[key]
                    if isinstance(calc_val, (int, float)):
                        error = abs(calc_val - ref_val) / abs(ref_val) * 100
                        validation['comparisons'][key] = {
                            'calculated': calc_val,
                            'reference': ref_val,
                            'error_percent': error
                        }
                        
                        if error > 50:
                            validation['warnings'].append(
                                f"Large discrepancy in {key}: {error:.1f}% error"
                            )
        
        return validation
    
    def generate_parameter_file(self, output_file: str = "phase_field_parameters.json"):
        """
        生成相场参数文件
        
        Args:
            output_file: 输出文件名
        """
        output_path = Path(output_file)
        
        # 准备可序列化的数据
        data = {
            'transferred_parameters': {},
            'uncertainties': self.uncertainties,
            'metadata': {
                'transfer_config': {
                    'reference_temperature': self.config.reference_temperature,
                    'smoothing_method': self.config.smoothing_method
                }
            }
        }
        
        # 转换参数为可序列化格式
        for source, params in self.transferred_params.items():
            data['transferred_parameters'][source] = {}
            for key, value in params.items():
                if isinstance(value, (int, float, str, list, dict)):
                    data['transferred_parameters'][source][key] = value
                elif isinstance(value, np.ndarray):
                    data['transferred_parameters'][source][key] = value.tolist()
                elif callable(value):
                    data['transferred_parameters'][source][key] = 'Function'
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Parameter file generated: {output_path}")
    
    def transfer_back_to_atomistic(self, phase_field_result: Dict) -> Dict:
        """
        将相场结果转换回原子尺度
        
        Args:
            phase_field_result: 相场模拟结果
            
        Returns:
            atomistic_params: 原子尺度参数
        """
        atomistic = {}
        
        # 1. 界面结构
        if 'interface_profile' in phase_field_result:
            profile = phase_field_result['interface_profile']
            # 转换到原子坐标
            atomistic['interface_width_atoms'] = profile['width'] * 0.1  # nm to Å
        
        # 2. 相畴尺寸
        if 'domain_size' in phase_field_result:
            domain_size = phase_field_result['domain_size']
            atomistic['domain_size_m'] = domain_size / self.config.atomistic_to_meso_scale
        
        # 3. 应力场
        if 'stress_field' in phase_field_result:
            stress_pf = phase_field_result['stress_field']
            # GPa保持不变
            atomistic['stress'] = stress_pf
        
        return atomistic
    
    def get_parameter_sensitivity(self, param_name: str,
                                  variation_range: Tuple[float, float],
                                  n_points: int = 20) -> Dict:
        """
        分析参数敏感性
        
        Args:
            param_name: 参数名称
            variation_range: 变化范围 (min, max)
            n_points: 采样点数
            
        Returns:
            sensitivity: 敏感性分析结果
        """
        param_values = np.linspace(variation_range[0], variation_range[1], n_points)
        
        sensitivity = {
            'parameter': param_name,
            'values': param_values.tolist(),
            'range': variation_range
        }
        
        return sensitivity
