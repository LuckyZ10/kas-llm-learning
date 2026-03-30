"""
XRD Data Connector
==================
X射线衍射数据连接器

支持格式:
- CSV/TXT (两列: 2θ, intensity)
- XRDML (帕纳科格式)
- Bruker RAW
- CIF (用于参考结构)
- HDF5 (处理后的数据)
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import logging

from .base_connector import BaseConnector, ExperimentalData, DataMetadata

logger = logging.getLogger(__name__)


class XRDConnector(BaseConnector):
    """
    X射线衍射数据连接器
    
    用于读取和处理XRD实验数据，支持多种仪器格式
    """
    
    # 支持的文件扩展名
    SUPPORTED_FORMATS = ['.csv', '.txt', '.xy', '.dat', '.xrdml', '.raw', '.h5', '.hdf5', '.cif']
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.config.setdefault('two_theta_min', 5.0)
        self.config.setdefault('two_theta_max', 90.0)
        self.config.setdefault('wavelength', 1.5406)  # Cu Kα1
        self.config.setdefault('normalize', True)
        self.config.setdefault('remove_kalpha2', False)
    
    def validate_format(self, filepath: str) -> bool:
        """验证XRD文件格式"""
        path = Path(filepath)
        if not path.exists():
            return False
        
        ext = path.suffix.lower()
        return ext in self.SUPPORTED_FORMATS
    
    def read(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取XRD数据文件
        
        Args:
            filepath: 文件路径
            **kwargs: 额外参数
            
        Returns:
            ExperimentalData对象，包含2θ和强度数据
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        
        if ext in ['.csv', '.txt', '.xy', '.dat']:
            data = self._read_ascii(filepath, **kwargs)
        elif ext == '.xrdml':
            data = self._read_xrdml(filepath, **kwargs)
        elif ext == '.raw':
            data = self._read_bruker_raw(filepath, **kwargs)
        elif ext in ['.h5', '.hdf5']:
            data = self._read_hdf5(filepath, **kwargs)
        elif ext == '.cif':
            data = self._read_cif(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported XRD format: {ext}")
        
        # 应用配置
        if self.config.get('normalize', False):
            data = self._normalize_intensity(data)
        
        # 裁剪范围
        data = self._crop_range(data)
        
        # 设置元数据
        data.metadata = DataMetadata(
            source=str(filepath),
            data_type='xrd',
            instrument=kwargs.get('instrument', 'unknown'),
            conditions={
                'wavelength': self.config['wavelength'],
                'two_theta_range': [self.config['two_theta_min'], self.config['two_theta_max']]
            }
        )
        
        self.data.append(data)
        logger.info(f"Loaded XRD data from {filepath}: {len(data.raw_data)} points")
        
        return data
    
    def _read_ascii(self, filepath: str, 
                    delimiter: Optional[str] = None,
                    skip_rows: int = 0,
                    column_indices: Optional[Tuple[int, int]] = None) -> ExperimentalData:
        """读取ASCII格式的XRD数据"""
        # 尝试自动检测分隔符
        if delimiter is None:
            with open(filepath, 'r') as f:
                first_line = f.readline()
                if ',' in first_line:
                    delimiter = ','
                elif '\t' in first_line:
                    delimiter = '\t'
                else:
                    delimiter = None  # 空格/任意空白
        
        # 读取数据
        data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_rows)
        
        if data.ndim == 1:
            # 单列表数据，假设是强度，2θ需要推断
            intensity = data
            two_theta = np.linspace(self.config['two_theta_min'], 
                                   self.config['two_theta_max'], 
                                   len(intensity))
            data = np.column_stack([two_theta, intensity])
        elif column_indices:
            two_theta = data[:, column_indices[0]]
            intensity = data[:, column_indices[1]]
            data = np.column_stack([two_theta, intensity])
        else:
            # 假设前两列是2θ和强度
            data = data[:, :2]
        
        return ExperimentalData(
            data_type='xrd',
            raw_data=data,
            column_names=['two_theta', 'intensity'],
            units={'two_theta': 'deg', 'intensity': 'a.u.'}
        )
    
    def _read_xrdml(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取XRDML格式（帕纳科X射线衍射仪）
        
        XRDML是基于XML的格式
        """
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # 命名空间
        ns = {'xrdml': 'http://www.xrdml.com/XRDMeasurement/1.0'}
        
        # 查找扫描数据
        scans = root.findall('.//xrdml:scan', ns)
        
        all_two_theta = []
        all_intensity = []
        
        for scan in scans:
            # 获取2θ起始点和步长
            data_points = scan.find('.//xrdml:dataPoints', ns)
            
            if data_points is not None:
                positions = data_points.find('xrdml:positions', ns)
                if positions is not None:
                    start_pos = float(positions.find('xrdml:startPosition', ns).text)
                    end_pos = float(positions.find('xrdml:endPosition', ns).text)
                
                intensities_elem = data_points.find('xrdml:intensities', ns)
                if intensities_elem is not None:
                    intensity_text = intensities_elem.text.strip()
                    intensities = [float(x) for x in intensity_text.split()]
                    
                    # 生成2θ值
                    two_theta = np.linspace(start_pos, end_pos, len(intensities))
                    
                    all_two_theta.extend(two_theta)
                    all_intensity.extend(intensities)
        
        if not all_two_theta:
            raise ValueError("No data found in XRDML file")
        
        data = np.column_stack([all_two_theta, all_intensity])
        
        return ExperimentalData(
            data_type='xrd',
            raw_data=data,
            column_names=['two_theta', 'intensity'],
            units={'two_theta': 'deg', 'intensity': 'counts'}
        )
    
    def _read_bruker_raw(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取Bruker RAW格式
        
        注意：这需要额外的库或二进制解析
        简化版本，可能需要安装bruker_raw或其他库
        """
        try:
            # 尝试使用可选依赖
            import bruker_raw
            raw_data = bruker_raw.read_raw(filepath)
            
            two_theta = raw_data['two_theta']
            intensity = raw_data['intensity']
            data = np.column_stack([two_theta, intensity])
            
            return ExperimentalData(
                data_type='xrd',
                raw_data=data,
                column_names=['two_theta', 'intensity'],
                units={'two_theta': 'deg', 'intensity': 'counts'}
            )
        except ImportError:
            logger.warning("bruker_raw not available, attempting manual parsing...")
            return self._read_bruker_raw_manual(filepath)
    
    def _read_bruker_raw_manual(self, filepath: str) -> ExperimentalData:
        """手动解析Bruker RAW文件（简化版）"""
        # 这是一个简化的实现，实际RAW格式更复杂
        # 通常需要专门的库
        
        with open(filepath, 'rb') as f:
            header = f.read(512)  # 读取文件头
            
            # 尝试提取基本信息
            # 这只是一个示例，实际格式需要逆向工程
            try:
                start_angle = np.frombuffer(header[80:84], dtype=np.float32)[0]
                end_angle = np.frombuffer(header[84:88], dtype=np.float32)[0]
                step_size = np.frombuffer(header[88:92], dtype=np.float32)[0]
            except:
                start_angle, end_angle, step_size = 5.0, 90.0, 0.02
            
            # 读取数据部分
            f.seek(512)
            raw_bytes = f.read()
            
            # 假设数据是32位整数
            n_points = (end_angle - start_angle) / step_size + 1
            intensity = np.frombuffer(raw_bytes, dtype=np.int32)[:int(n_points)]
            two_theta = np.linspace(start_angle, end_angle, len(intensity))
            
            data = np.column_stack([two_theta, intensity])
            
            return ExperimentalData(
                data_type='xrd',
                raw_data=data,
                column_names=['two_theta', 'intensity'],
                units={'two_theta': 'deg', 'intensity': 'counts'}
            )
    
    def _read_hdf5(self, filepath: str, dataset_path: str = 'xrd_data') -> ExperimentalData:
        """读取HDF5格式的XRD数据"""
        try:
            import h5py
            with h5py.File(filepath, 'r') as f:
                dataset = f[dataset_path]
                data = dataset[:]
                
                # 读取属性
                units = {}
                for key in ['two_theta_unit', 'intensity_unit']:
                    if key in dataset.attrs:
                        units[key.replace('_unit', '')] = dataset.attrs[key]
                
                return ExperimentalData(
                    data_type='xrd',
                    raw_data=data,
                    column_names=['two_theta', 'intensity'],
                    units=units
                )
        except ImportError:
            raise ImportError("h5py is required for HDF5 format")
    
    def _read_cif(self, filepath: str, wavelength: Optional[float] = None, **kwargs) -> ExperimentalData:
        """
        从CIF文件计算理论XRD图谱
        
        用于与实验数据对比
        """
        try:
            from pymatgen.core import Structure
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            structure = Structure.from_file(filepath)
            
            wavelength = wavelength or self.config['wavelength']
            calculator = XRDCalculator(wavelength=wavelength)
            
            pattern = calculator.get_pattern(structure)
            
            two_theta = pattern.x
            intensity = pattern.y
            
            data = np.column_stack([two_theta, intensity])
            
            return ExperimentalData(
                data_type='xrd_theoretical',
                raw_data=data,
                column_names=['two_theta', 'intensity'],
                units={'two_theta': 'deg', 'intensity': 'a.u.'},
                metadata=DataMetadata(
                    source=str(filepath),
                    data_type='xrd_theoretical',
                    conditions={'wavelength': wavelength, 'structure': structure.formula}
                )
            )
        except ImportError:
            raise ImportError("pymatgen is required for CIF processing")
    
    def _normalize_intensity(self, data: ExperimentalData) -> ExperimentalData:
        """归一化强度到0-1范围"""
        intensity = data.processed_data[:, 1]
        max_intensity = np.max(intensity)
        if max_intensity > 0:
            data.processed_data[:, 1] = intensity / max_intensity
        return data
    
    def _crop_range(self, data: ExperimentalData) -> ExperimentalData:
        """裁剪2θ范围"""
        two_theta = data.processed_data[:, 0]
        mask = (two_theta >= self.config['two_theta_min']) & (two_theta <= self.config['two_theta_max'])
        data.processed_data = data.processed_data[mask]
        return data
    
    def find_peaks(self, data: ExperimentalData, 
                   prominence: float = 0.05,
                   width: Optional[int] = None,
                   distance: int = 10) -> List[Dict]:
        """
        查找XRD图谱中的峰
        
        Args:
            data: XRD数据
            prominence: 峰的最小显著性
            width: 峰的最小宽度
            distance: 峰之间的最小距离
            
        Returns:
            峰列表，每个峰包含位置、强度、宽度等信息
        """
        from scipy.signal import find_peaks, peak_widths
        
        two_theta = data.processed_data[:, 0]
        intensity = data.processed_data[:, 1]
        
        # 查找峰
        peaks, properties = find_peaks(intensity, 
                                      prominence=prominence,
                                      width=width,
                                      distance=distance)
        
        # 计算峰宽
        if len(peaks) > 0:
            widths, width_heights, left_ips, right_ips = peak_widths(
                intensity, peaks, rel_height=0.5
            )
            
            # 转换为2θ单位
            two_theta_step = np.mean(np.diff(two_theta))
            fwhm = widths * two_theta_step
            
            # 左、右边界
            left_theta = two_theta[0] + left_ips * two_theta_step
            right_theta = two_theta[0] + right_ips * two_theta_step
        else:
            fwhm = []
            left_theta = []
            right_theta = []
        
        peak_list = []
        for i, peak_idx in enumerate(peaks):
            peak_list.append({
                'index': int(peak_idx),
                'two_theta': float(two_theta[peak_idx]),
                'intensity': float(intensity[peak_idx]),
                'fwhm': float(fwhm[i]) if i < len(fwhm) else None,
                'left_base': float(left_theta[i]) if i < len(left_theta) else None,
                'right_base': float(right_theta[i]) if i < len(right_theta) else None,
                'area': float(properties['prominences'][i]) if 'prominences' in properties else None
            })
        
        return peak_list
    
    def calculate_d_spacing(self, two_theta: np.ndarray, 
                           wavelength: Optional[float] = None) -> np.ndarray:
        """
        根据布拉格定律计算晶面间距d
        
        nλ = 2d sin(θ)
        => d = λ / (2 sin(θ))
        """
        wavelength = wavelength or self.config['wavelength']
        theta = np.radians(two_theta / 2)
        d = wavelength / (2 * np.sin(theta))
        return d
    
    def index_peaks(self, peaks: List[Dict], 
                   structure: Any,
                   tolerance: float = 0.05) -> List[Dict]:
        """
        为峰标定晶面指数
        
        Args:
            peaks: 峰列表
            structure: 晶体结构（pymatgen Structure）
            tolerance: 匹配容差（2θ偏差）
            
        Returns:
            带晶面指数的峰列表
        """
        try:
            from pymatgen.analysis.diffraction.xrd import XRDCalculator
            
            calculator = XRDCalculator(wavelength=self.config['wavelength'])
            pattern = calculator.get_pattern(structure, scaled=False)
            
            indexed_peaks = []
            for peak in peaks:
                exp_2theta = peak['two_theta']
                
                # 找到最近的理论峰
                diff = np.abs(pattern.x - exp_2theta)
                closest_idx = np.argmin(diff)
                
                if diff[closest_idx] < tolerance:
                    peak['hkl'] = pattern.hkls[closest_idx]
                    peak['theoretical_2theta'] = float(pattern.x[closest_idx])
                    peak['d_spacing'] = self.calculate_d_spacing(exp_2theta)
                
                indexed_peaks.append(peak)
            
            return indexed_peaks
        except ImportError:
            raise ImportError("pymatgen is required for peak indexing")
    
    def subtract_background(self, data: ExperimentalData,
                           method: str = 'spline',
                           **kwargs) -> ExperimentalData:
        """
        减去背景
        
        Args:
            data: XRD数据
            method: 'spline', 'polynomial', 或 'auto'
            
        Returns:
            背景扣除后的数据
        """
        two_theta = data.processed_data[:, 0]
        intensity = data.processed_data[:, 1]
        
        if method == 'spline':
            from scipy.interpolate import UnivariateSpline
            
            # 使用低阶样条拟合背景
            spline = UnivariateSpline(two_theta, intensity, s=kwargs.get('smoothness', len(two_theta)))
            background = spline(two_theta)
            
        elif method == 'polynomial':
            degree = kwargs.get('degree', 5)
            coeffs = np.polyfit(two_theta, intensity, degree)
            background = np.polyval(coeffs, two_theta)
            
        elif method == 'auto':
            # 使用形态学操作估计背景
            from scipy.ndimage import minimum_filter1d, maximum_filter1d
            
            window = kwargs.get('window', 50)
            # 最小值滤波器
            background = minimum_filter1d(intensity, size=window)
            # 平滑
            background = maximum_filter1d(background, size=window//2)
            
        else:
            raise ValueError(f"Unknown background method: {method}")
        
        data.processed_data[:, 1] = intensity - background
        data.processed_data[:, 1] = np.maximum(data.processed_data[:, 1], 0)  # 防止负值
        
        return data


# 便捷函数
def load_xrd(filepath: str, **kwargs) -> ExperimentalData:
    """便捷函数：加载XRD数据"""
    connector = XRDConnector()
    return connector.read(filepath, **kwargs)
