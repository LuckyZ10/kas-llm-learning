"""
Spectroscopy Data Connector
===========================
光谱数据连接器

支持的数据类型:
- XPS (X射线光电子能谱)
- Raman (拉曼光谱)
- FTIR (傅里叶变换红外光谱)
- UV-Vis (紫外-可见光谱)
- NMR (核磁共振)
- XAS/XANES (X射线吸收谱)

支持格式:
- CSV/TXT (通用格式)
- VMS (XPS)
- SPC (通用光谱)
- DPT (红外)
- JCAMP-DX
- HDF5
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import logging

from .base_connector import BaseConnector, ExperimentalData, DataMetadata

logger = logging.getLogger(__name__)


class SpectroscopyConnector(BaseConnector):
    """
    光谱数据连接器
    
    支持多种光谱类型的读取和处理
    """
    
    SUPPORTED_FORMATS = ['.csv', '.txt', '.xy', '.vms', '.spc', '.dpt', 
                        '.dx', '.jdx', '.json', '.h5', '.hdf5']
    
    # 光谱类型及其单位
    SPECTRUM_TYPES = {
        'xps': {'x_unit': 'eV', 'y_unit': 'counts', 'x_name': 'Binding Energy'},
        'raman': {'x_unit': 'cm-1', 'y_unit': 'intensity', 'x_name': 'Raman Shift'},
        'ftir': {'x_unit': 'cm-1', 'y_unit': 'transmittance', 'x_name': 'Wavenumber'},
        'uvvis': {'x_unit': 'nm', 'y_unit': 'absorbance', 'x_name': 'Wavelength'},
        'nmr': {'x_unit': 'ppm', 'y_unit': 'intensity', 'x_name': 'Chemical Shift'},
        'xas': {'x_unit': 'eV', 'y_unit': 'absorbance', 'x_name': 'Energy'},
        'xanes': {'x_unit': 'eV', 'y_unit': 'absorbance', 'x_name': 'Energy'},
        'exafs': {'x_unit': 'Angstrom', 'y_unit': 'chi(k)', 'x_name': 'k'},
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.config.setdefault('spectrum_type', 'auto')
        self.config.setdefault('normalize', True)
        self.config.setdefault('baseline_correction', False)
        self.config.setdefault('smoothing', False)
    
    def validate_format(self, filepath: str) -> bool:
        """验证光谱文件格式"""
        path = Path(filepath)
        if not path.exists():
            return False
        return path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def read(self, filepath: str, 
             spectrum_type: str = 'auto',
             **kwargs) -> ExperimentalData:
        """
        读取光谱数据文件
        
        Args:
            filepath: 文件路径
            spectrum_type: 光谱类型 ('xps', 'raman', 'ftir', 'uvvis', 'auto')
            **kwargs: 额外参数
            
        Returns:
            ExperimentalData对象
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        
        # 自动检测光谱类型
        if spectrum_type == 'auto':
            spectrum_type = self._detect_spectrum_type(filepath)
        
        # 根据格式选择读取方法
        if ext in ['.csv', '.txt', '.xy']:
            data = self._read_ascii(filepath, spectrum_type, **kwargs)
        elif ext == '.vms':
            data = self._read_xps_vms(filepath, **kwargs)
        elif ext == '.spc':
            data = self._read_spc(filepath, **kwargs)
        elif ext == '.dpt':
            data = self._read_dpt(filepath, **kwargs)
        elif ext in ['.dx', '.jdx']:
            data = self._read_jcamp(filepath, **kwargs)
        elif ext == '.json':
            data = self._read_json(filepath, **kwargs)
        elif ext in ['.h5', '.hdf5']:
            data = self._read_hdf5(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        
        # 应用后处理
        if self.config.get('baseline_correction', False):
            data = self.subtract_baseline(data)
        
        if self.config.get('smoothing', False):
            data = self.preprocess(data, ['smooth'])
        
        if self.config.get('normalize', False):
            data = self._normalize_spectrum(data)
        
        # 设置元数据
        data.metadata.data_type = spectrum_type
        data.metadata.source = str(filepath)
        
        self.data.append(data)
        logger.info(f"Loaded {spectrum_type} spectrum from {filepath}: {len(data.raw_data)} points")
        
        return data
    
    def _detect_spectrum_type(self, filepath: str) -> str:
        """自动检测光谱类型"""
        path = Path(filepath)
        filename_lower = path.name.lower()
        
        # 从文件名检测
        for spec_type in self.SPECTRUM_TYPES.keys():
            if spec_type in filename_lower:
                return spec_type
        
        # 从文件扩展名和内容检测
        if filename_lower.endswith('.vms'):
            return 'xps'
        elif filename_lower.endswith('.dpt'):
            return 'ftir'
        
        # 尝试从文件内容检测
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.read(1000).lower()
            
            if any(x in header for x in ['binding energy', 'pass energy', 'xps']):
                return 'xps'
            elif any(x in header for x in ['raman', 'raman shift']):
                return 'raman'
            elif any(x in header for x in ['wavenumber', 'transmittance', 'ftir']):
                return 'ftir'
            elif any(x in header for x in ['wavelength', 'absorbance', 'uv-vis', 'uvvis']):
                return 'uvvis'
            elif any(x in header for x in ['##jcamp-dx']):
                # JCAMP格式，尝试进一步检测
                if 'nmr' in header or 'chemical shift' in header:
                    return 'nmr'
                elif 'infrared' in header:
                    return 'ftir'
        except:
            pass
        
        return 'unknown'
    
    def _read_ascii(self, filepath: str, spectrum_type: str,
                   delimiter: Optional[str] = None,
                   skip_rows: int = 0) -> ExperimentalData:
        """读取ASCII格式的光谱数据"""
        # 检测分隔符
        if delimiter is None:
            with open(filepath, 'r') as f:
                first_lines = [f.readline() for _ in range(10)]
                sample_line = ''.join(first_lines)
                if ',' in sample_line:
                    delimiter = ','
                elif '\t' in sample_line:
                    delimiter = '\t'
                else:
                    delimiter = None
        
        # 尝试读取，找到数据开始行
        data_start = 0
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                try:
                    # 尝试解析为数字
                    parts = line.strip().split(delimiter if delimiter else None)
                    float(parts[0])
                    float(parts[1])
                    data_start = i
                    break
                except:
                    continue
        
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=data_start)
        
        # 获取单位信息
        units = self.SPECTRUM_TYPES.get(spectrum_type, {})
        
        return ExperimentalData(
            data_type=spectrum_type,
            raw_data=data,
            column_names=[units.get('x_name', 'x'), units.get('y_unit', 'y')],
            units={
                'x': units.get('x_unit', ''),
                'y': units.get('y_unit', '')
            }
        )
    
    def _read_xps_vms(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取XPS VMS格式（CasaXPS等软件使用）
        
        VMS是文本格式，包含多个区域的数据
        """
        spectra = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        current_spectrum = []
        in_data = False
        
        for line in lines:
            line = line.strip()
            
            # 检测数据块开始
            if line.startswith('#') or line.startswith('*'):
                if current_spectrum:
                    spectra.append(np.array(current_spectrum))
                    current_spectrum = []
                in_data = False
                continue
            
            # 尝试解析数据行
            try:
                parts = line.split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    current_spectrum.append([x, y])
                    in_data = True
            except:
                if in_data and current_spectrum:
                    spectra.append(np.array(current_spectrum))
                    current_spectrum = []
                    in_data = False
        
        if current_spectrum:
            spectra.append(np.array(current_spectrum))
        
        # 返回第一个光谱（或合并多个）
        if spectra:
            data = spectra[0]
        else:
            data = np.array([])
        
        return ExperimentalData(
            data_type='xps',
            raw_data=data,
            column_names=['Binding Energy (eV)', 'Intensity (counts)'],
            units={'x': 'eV', 'y': 'counts'}
        )
    
    def _read_spc(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取SPC格式（通用光谱文件格式）
        
        SPC是二进制格式
        """
        with open(filepath, 'rb') as f:
            # 读取文件头
            header = f.read(512)
            
            # 解析基本信息
            ftflg = header[0]  # 文件类型标志
            fversn = header[1]  # 版本
            fexper = header[2]  # 实验类型
            
            # 根据标志解析数据
            # 这是简化实现
            # 完整解析需要遵循SPC格式规范
            
            # 假设数据在头部之后
            f.seek(512)
            data_bytes = f.read()
        
        # 尝试解析为双精度浮点数
        try:
            data = np.frombuffer(data_bytes, dtype=np.float64)
            n_points = len(data) // 2
            data = data[:n_points * 2].reshape(n_points, 2)
        except:
            # 尝试单精度
            data = np.frombuffer(data_bytes, dtype=np.float32)
            n_points = len(data) // 2
            data = data[:n_points * 2].reshape(n_points, 2)
        
        return ExperimentalData(
            data_type='unknown',
            raw_data=data,
            column_names=['x', 'y'],
            units={}
        )
    
    def _read_dpt(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取DPT格式（红外光谱常见格式）
        
        DPT通常是两列ASCII数据
        """
        data = np.loadtxt(filepath)
        
        return ExperimentalData(
            data_type='ftir',
            raw_data=data,
            column_names=['Wavenumber (cm-1)', 'Transmittance (%)'],
            units={'x': 'cm-1', 'y': '%'}
        )
    
    def _read_jcamp(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取JCAMP-DX格式
        
        JCAMP是标准的光谱交换格式
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 解析头部
        metadata = {}
        data_lines = []
        in_data = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('##'):
                # 元数据行
                if '=' in line:
                    key, value = line[2:].split('=', 1)
                    metadata[key.strip()] = value.strip()
                
                if 'XYDATA' in line or 'XYPOINTS' in line:
                    in_data = True
            elif in_data:
                data_lines.append(line)
        
        # 解析数据
        # JCAMP支持多种数据格式，这里是简化处理
        data_points = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    data_points.append([x, y])
                except:
                    continue
        
        data = np.array(data_points)
        
        # 确定光谱类型
        spec_type = 'unknown'
        data_type = metadata.get('DATA TYPE', '').lower()
        if 'infrared' in data_type:
            spec_type = 'ftir'
        elif 'raman' in data_type:
            spec_type = 'raman'
        elif 'nmr' in data_type:
            spec_type = 'nmr'
        elif 'uv' in data_type:
            spec_type = 'uvvis'
        
        x_unit = metadata.get('XUNITS', '')
        y_unit = metadata.get('YUNITS', '')
        
        return ExperimentalData(
            data_type=spec_type,
            raw_data=data,
            column_names=[f'X ({x_unit})', f'Y ({y_unit})'],
            units={'x': x_unit, 'y': y_unit},
            metadata=DataMetadata(conditions=metadata)
        )
    
    def _read_json(self, filepath: str, **kwargs) -> ExperimentalData:
        """读取JSON格式的光谱数据"""
        import json
        
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        
        data_array = np.array(json_data.get('data', []))
        spectrum_type = json_data.get('spectrum_type', 'unknown')
        
        return ExperimentalData(
            data_type=spectrum_type,
            raw_data=data_array,
            column_names=json_data.get('columns', ['x', 'y']),
            units=json_data.get('units', {}),
            metadata=DataMetadata(**json_data.get('metadata', {}))
        )
    
    def _read_hdf5(self, filepath: str, **kwargs) -> ExperimentalData:
        """读取HDF5格式的光谱数据"""
        try:
            import h5py
            with h5py.File(filepath, 'r') as f:
                dataset_path = kwargs.get('dataset_path', 'spectrum')
                data = f[dataset_path][:]
                
                attrs = dict(f[dataset_path].attrs)
                spectrum_type = attrs.get('spectrum_type', 'unknown')
                
                return ExperimentalData(
                    data_type=spectrum_type,
                    raw_data=data,
                    column_names=['x', 'y'],
                    units={
                        'x': attrs.get('x_unit', ''),
                        'y': attrs.get('y_unit', '')
                    }
                )
        except ImportError:
            raise ImportError("h5py is required for HDF5 format")
    
    def _normalize_spectrum(self, data: ExperimentalData) -> ExperimentalData:
        """归一化光谱"""
        y = data.processed_data[:, 1]
        max_val = np.max(np.abs(y))
        if max_val > 0:
            data.processed_data[:, 1] = y / max_val
        return data
    
    def subtract_baseline(self, data: ExperimentalData,
                         method: str = 'als',
                         **kwargs) -> ExperimentalData:
        """
        减去基线
        
        Args:
            data: 光谱数据
            method: 'als' (非对称最小二乘), 'polynomial', 'linear'
            
        Returns:
            基线校正后的数据
        """
        x = data.processed_data[:, 0]
        y = data.processed_data[:, 1]
        
        if method == 'als':
            baseline = self._als_baseline(y, **kwargs)
        elif method == 'polynomial':
            degree = kwargs.get('degree', 3)
            coeffs = np.polyfit(x, y, degree)
            baseline = np.polyval(coeffs, x)
        elif method == 'linear':
            coeffs = np.polyfit(x, y, 1)
            baseline = np.polyval(coeffs, x)
        else:
            raise ValueError(f"Unknown baseline method: {method}")
        
        data.processed_data[:, 1] = y - baseline
        return data
    
    def _als_baseline(self, y: np.ndarray, 
                     lam: float = 1e5, 
                     p: float = 0.01, 
                     niter: int = 10) -> np.ndarray:
        """
        非对称最小二乘基线校正
        
        基于Whittaker平滑的基线校正算法
        
        Args:
            y: 输入信号
            lam: 平滑参数
            p: 非对称参数
            niter: 迭代次数
            
        Returns:
            基线
        """
        L = len(y)
        D = np.diff(np.eye(L), 2)
        w = np.ones(L)
        
        for _ in range(niter):
            W = np.diag(w)
            Z = W + lam * np.dot(D, D.T)
            baseline = np.linalg.solve(Z, w * y)
            w = p * (y > baseline) + (1 - p) * (y <= baseline)
        
        return baseline
    
    def find_peaks(self, data: ExperimentalData,
                  prominence: float = 0.05,
                  width: Optional[int] = None,
                  distance: int = 5) -> List[Dict]:
        """
        查找光谱中的峰
        
        Args:
            data: 光谱数据
            prominence: 峰的最小显著性
            width: 峰的最小宽度
            distance: 峰之间的最小距离
            
        Returns:
            峰列表
        """
        from scipy.signal import find_peaks, peak_widths
        
        x = data.processed_data[:, 0]
        y = data.processed_data[:, 1]
        
        peaks, properties = find_peaks(y, prominence=prominence, width=width, distance=distance)
        
        # 计算峰宽
        if len(peaks) > 0:
            widths, width_heights, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)
            
            # 转换为x单位
            x_step = np.mean(np.diff(x))
            fwhm = widths * x_step
            left_x = x[0] + left_ips * x_step
            right_x = x[0] + right_ips * x_step
        else:
            fwhm = []
            left_x = []
            right_x = []
        
        peak_list = []
        for i, peak_idx in enumerate(peaks):
            peak_list.append({
                'index': int(peak_idx),
                'position': float(x[peak_idx]),
                'intensity': float(y[peak_idx]),
                'fwhm': float(fwhm[i]) if i < len(fwhm) else None,
                'area': float(properties['prominences'][i]) if 'prominences' in properties else None,
                'left_base': float(left_x[i]) if i < len(left_x) else None,
                'right_base': float(right_x[i]) if i < len(right_x) else None,
            })
        
        return peak_list
    
    def calibrate_wavelength(self, data: ExperimentalData,
                            reference_peaks: List[float],
                            observed_peaks: List[float]) -> ExperimentalData:
        """
        波长校准
        
        使用参考峰位置校准光谱
        
        Args:
            data: 光谱数据
            reference_peaks: 参考峰位置
            observed_peaks: 观测峰位置
            
        Returns:
            校准后的数据
        """
        if len(reference_peaks) != len(observed_peaks):
            raise ValueError("Reference and observed peaks must have same length")
        
        # 线性校准
        coeffs = np.polyfit(observed_peaks, reference_peaks, 1)
        
        x = data.processed_data[:, 0]
        x_calibrated = np.polyval(coeffs, x)
        
        data.processed_data[:, 0] = x_calibrated
        
        logger.info(f"Wavelength calibration: offset={coeffs[1]:.4f}, scale={coeffs[0]:.6f}")
        
        return data


# 便捷函数
def load_spectrum(filepath: str, spectrum_type: str = 'auto', **kwargs) -> ExperimentalData:
    """便捷函数：加载光谱数据"""
    connector = SpectroscopyConnector()
    return connector.read(filepath, spectrum_type=spectrum_type, **kwargs)
