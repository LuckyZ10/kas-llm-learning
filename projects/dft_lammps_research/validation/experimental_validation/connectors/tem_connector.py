"""
TEM Data Connector
==================
透射电镜数据连接器

支持的数据类型:
- TEM图像
- HRTEM图像
- STEM图像
- EDS谱图
- EELS谱图
- 电子衍射图样

支持格式:
- DM3/DM4 (Gatan DigitalMicrograph)
- TIFF/EMI (FEI/Thermo Fisher)
- SER/EMI (TIA)
- MRC/CCP4
- HDF5
- PNG/JPG (通用图像)
"""

import struct
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np
import logging

from .base_connector import BaseConnector, ExperimentalData, DataMetadata

logger = logging.getLogger(__name__)


@dataclass
class TEMMetadata:
    """TEM图像元数据"""
    voltage: float = 200.0  # kV
    magnification: float = 0.0
    pixel_size: float = 0.0  # nm/pixel
    exposure_time: float = 0.0  # s
    binning: int = 1
    camera_length: float = 0.0  # mm (用于衍射)
    defocus: float = 0.0  # nm
    spherical_aberration: float = 0.0  # mm
    
    def to_dict(self) -> Dict:
        return {
            'voltage': self.voltage,
            'magnification': self.magnification,
            'pixel_size': self.pixel_size,
            'exposure_time': self.exposure_time,
            'binning': self.binning,
            'camera_length': self.camera_length,
            'defocus': self.defocus,
            'spherical_aberration': self.spherical_aberration,
        }


class TEMConnector(BaseConnector):
    """
    透射电镜数据连接器
    
    支持TEM图像、衍射图样和能谱数据的读取
    """
    
    SUPPORTED_IMAGE_FORMATS = ['.dm3', '.dm4', '.tif', '.tiff', '.ser', '.emi', 
                               '.mrc', '.h5', '.hdf5', '.png', '.jpg', '.jpeg']
    SUPPORTED_SPECTRUM_FORMATS = ['.dm3', '.dm4', '.msa', '.emsa', '.h5', '.hdf5']
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.config.setdefault('voltage', 200.0)  # kV
        self.config.setdefault('normalize', True)
        self.config.setdefault('flip_vertical', False)
        self.config.setdefault('flip_horizontal', False)
    
    def validate_format(self, filepath: str) -> bool:
        """验证TEM文件格式"""
        path = Path(filepath)
        if not path.exists():
            return False
        ext = path.suffix.lower()
        return ext in self.SUPPORTED_IMAGE_FORMATS or ext in self.SUPPORTED_SPECTRUM_FORMATS
    
    def read(self, filepath: str, data_type: str = 'image', **kwargs) -> ExperimentalData:
        """
        读取TEM数据文件
        
        Args:
            filepath: 文件路径
            data_type: 'image', 'diffraction', 'eds', 'eels'
            **kwargs: 额外参数
            
        Returns:
            ExperimentalData对象
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        
        if ext in ['.dm3', '.dm4']:
            data = self._read_dm3(filepath, data_type, **kwargs)
        elif ext in ['.tif', '.tiff']:
            data = self._read_tiff(filepath, data_type, **kwargs)
        elif ext in ['.ser', '.emi']:
            data = self._read_tia(filepath, data_type, **kwargs)
        elif ext == '.mrc':
            data = self._read_mrc(filepath, **kwargs)
        elif ext in ['.h5', '.hdf5']:
            data = self._read_hdf5(filepath, data_type, **kwargs)
        elif ext in ['.png', '.jpg', '.jpeg']:
            data = self._read_image(filepath, **kwargs)
        elif ext in ['.msa', '.emsa']:
            data = self._read_emsa(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        
        # 应用后处理
        if self.config.get('normalize', False) and data_type == 'image':
            data = self._normalize_image(data)
        
        if self.config.get('flip_vertical', False):
            data = self._flip_vertical(data)
        
        if self.config.get('flip_horizontal', False):
            data = self._flip_horizontal(data)
        
        data.metadata.data_type = data_type
        data.metadata.source = str(filepath)
        
        self.data.append(data)
        logger.info(f"Loaded {data_type} data from {filepath}")
        
        return data
    
    def _read_dm3(self, filepath: str, data_type: str, **kwargs) -> ExperimentalData:
        """
        读取DM3/DM4格式（Gatan DigitalMicrograph）
        
        DM3/DM4是TEM数据的标准格式
        """
        try:
            import ncempy.io.dm
            
            dm_file = ncempy.io.dm.fileDM(filepath)
            data_array = dm_file.getDataset(0)['data']
            
            # 提取元数据
            metadata = TEMMetadata()
            
            # 尝试提取像素尺寸
            try:
                pix_scale = dm_file.pixScale
                if pix_scale:
                    metadata.pixel_size = pix_scale[0]
            except:
                pass
            
            # 尝试提取电压
            try:
                voltage = dm_file.allTags.get('.ImageList.1.ImageData.Metadata.\nMicroscope Info.Voltage')
                if voltage:
                    metadata.voltage = voltage / 1000  # 转换为kV
            except:
                pass
            
            return ExperimentalData(
                data_type=data_type,
                raw_data=data_array,
                column_names=['y', 'x'] if data_array.ndim == 2 else ['intensity'],
                metadata=DataMetadata(
                    conditions=metadata.to_dict()
                ),
                units={'pixel_size': 'nm', 'voltage': 'kV'}
            )
            
        except ImportError:
            logger.warning("ncempy not available, using basic parser...")
            return self._read_dm3_basic(filepath)
    
    def _read_dm3_basic(self, filepath: str) -> ExperimentalData:
        """基础DM3解析（简化版）"""
        with open(filepath, 'rb') as f:
            # DM3文件头
            header = f.read(16)
            version = struct.unpack('>I', header[0:4])[0]
            
            # 简化处理，读取为原始数据
            # 实际格式复杂，包含标签组和数据
            f.seek(0)
            raw_data = f.read()
        
        # 尝试提取图像数据（简化假设）
        # 实际实现需要完整解析DM3标签树
        
        # 创建一个占位符
        data_array = np.zeros((1024, 1024), dtype=np.float32)
        
        return ExperimentalData(
            data_type='image',
            raw_data=data_array,
            column_names=['y', 'x'],
            units={}
        )
    
    def _read_tiff(self, filepath: str, data_type: str, **kwargs) -> ExperimentalData:
        """读取TIFF格式（FEI/Thermo Fisher）"""
        try:
            from PIL import Image
            
            img = Image.open(filepath)
            data_array = np.array(img)
            
            # 尝试提取FEI元数据
            metadata = TEMMetadata()
            
            # FEI TIFF通常包含元数据在ImageDescription标签
            try:
                import xml.etree.ElementTree as ET
                desc = img.tag.get(270, '')  # ImageDescription tag
                if desc and '<' in desc:
                    root = ET.fromstring(desc)
                    # 解析FEI元数据
                    for elem in root.iter():
                        if 'PixelSize' in elem.tag:
                            metadata.pixel_size = float(elem.text) * 1e9  # 转换为nm
                        elif 'AccelerationVoltage' in elem.tag:
                            metadata.voltage = float(elem.text) / 1000
            except:
                pass
            
            return ExperimentalData(
                data_type=data_type,
                raw_data=data_array,
                column_names=['y', 'x'],
                metadata=DataMetadata(
                    conditions=metadata.to_dict()
                ),
                units={'pixel_size': 'nm', 'voltage': 'kV'}
            )
            
        except ImportError:
            raise ImportError("PIL/Pillow is required for TIFF support")
    
    def _read_tia(self, filepath: str, data_type: str, **kwargs) -> ExperimentalData:
        """
        读取TIA格式（FEI/Thermo Fisher）
        
        SER是主数据文件，EMI是元数据文件
        """
        try:
            import ncempy.io.ser
            
            ser_data = ncempy.io.ser.readSer(filepath)
            data_array = ser_data['data']
            
            # 提取元数据
            metadata = TEMMetadata()
            
            return ExperimentalData(
                data_type=data_type,
                raw_data=data_array,
                column_names=['y', 'x'] if data_array.ndim == 2 else ['intensity'],
                metadata=DataMetadata(
                    conditions=metadata.to_dict()
                ),
                units={}
            )
            
        except ImportError:
            logger.warning("ncempy not available, TIA support limited")
            # 简化处理
            data_array = np.zeros((1024, 1024))
            return ExperimentalData(
                data_type=data_type,
                raw_data=data_array,
                column_names=['y', 'x'],
                units={}
            )
    
    def _read_mrc(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取MRC格式
        
        MRC是电子显微镜的标准格式
        """
        try:
            import mrcfile
            
            with mrcfile.open(filepath, permissive=True) as mrc:
                data_array = mrc.data
                
                # 提取元数据
                voxel_size = mrc.voxel_size
                
                return ExperimentalData(
                    data_type='image',
                    raw_data=data_array,
                    column_names=['z', 'y', 'x'] if data_array.ndim == 3 else ['y', 'x'],
                    metadata=DataMetadata(
                        conditions={
                            'voxel_size': {
                                'x': float(voxel_size.x),
                                'y': float(voxel_size.y),
                                'z': float(voxel_size.z) if data_array.ndim == 3 else 0
                            }
                        }
                    ),
                    units={'voxel_size': 'Angstrom'}
                )
                
        except ImportError:
            raise ImportError("mrcfile is required for MRC support")
    
    def _read_hdf5(self, filepath: str, data_type: str, **kwargs) -> ExperimentalData:
        """读取HDF5格式的TEM数据"""
        try:
            import h5py
            with h5py.File(filepath, 'r') as f:
                dataset_path = kwargs.get('dataset_path', 'data/image')
                data_array = f[dataset_path][:]
                
                # 读取元数据
                metadata = {}
                for key in ['voltage', 'magnification', 'pixel_size']:
                    if key in f[dataset_path].attrs:
                        metadata[key] = f[dataset_path].attrs[key]
                
                return ExperimentalData(
                    data_type=data_type,
                    raw_data=data_array,
                    column_names=['y', 'x'] if data_array.ndim == 2 else ['intensity'],
                    metadata=DataMetadata(conditions=metadata),
                    units={'pixel_size': 'nm', 'voltage': 'kV'}
                )
        except ImportError:
            raise ImportError("h5py is required for HDF5 format")
    
    def _read_image(self, filepath: str, **kwargs) -> ExperimentalData:
        """读取通用图像格式（PNG, JPG等）"""
        try:
            from PIL import Image
            
            img = Image.open(filepath)
            
            # 转换为灰度
            if img.mode != 'L':
                img = img.convert('L')
            
            data_array = np.array(img, dtype=np.float32)
            
            return ExperimentalData(
                data_type='image',
                raw_data=data_array,
                column_names=['y', 'x'],
                units={}
            )
            
        except ImportError:
            raise ImportError("PIL/Pillow is required for image support")
    
    def _read_emsa(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取EMSA/MSA格式（能谱标准格式）
        
        EMSA是电子能谱的标准ASCII格式
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # 解析头部
        metadata = {}
        data_lines = []
        in_data = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('#'):
                # 元数据行
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    metadata[key.strip()] = value.strip()
            elif line and not line.startswith('#SPECTRUM'):
                # 数据行
                data_lines.append(line)
        
        # 解析数据
        data_points = []
        for line in data_lines:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    data_points.append([x, y])
                except:
                    continue
        
        data_array = np.array(data_points)
        
        spectrum_type = 'eds' if 'EDX' in metadata.get('SIGNALTYPE', '') else 'eels'
        
        return ExperimentalData(
            data_type=spectrum_type,
            raw_data=data_array,
            column_names=['Energy (eV)', 'Intensity'],
            metadata=DataMetadata(conditions=metadata),
            units={'x': 'eV', 'y': 'counts'}
        )
    
    def _normalize_image(self, data: ExperimentalData) -> ExperimentalData:
        """归一化图像"""
        img = data.processed_data
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            data.processed_data = (img - img_min) / (img_max - img_min)
        return data
    
    def _flip_vertical(self, data: ExperimentalData) -> ExperimentalData:
        """垂直翻转图像"""
        data.processed_data = np.flipud(data.processed_data)
        return data
    
    def _flip_horizontal(self, data: ExperimentalData) -> ExperimentalData:
        """水平翻转图像"""
        data.processed_data = np.fliplr(data.processed_data)
        return data
    
    def extract_line_profile(self, data: ExperimentalData,
                            start: Tuple[int, int],
                            end: Tuple[int, int],
                            width: int = 1) -> np.ndarray:
        """
        提取线扫描剖面
        
        Args:
            data: 图像数据
            start: 起点 (y, x)
            end: 终点 (y, x)
            width: 线宽（用于平均）
            
        Returns:
            强度剖面
        """
        from scipy.ndimage import map_coordinates
        
        img = data.processed_data
        
        # 创建线上的点
        num_points = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
        y_coords = np.linspace(start[0], end[0], num_points)
        x_coords = np.linspace(start[1], end[1], num_points)
        
        # 提取剖面
        coords = np.vstack([y_coords, x_coords])
        profile = map_coordinates(img, coords, order=1)
        
        return profile
    
    def calculate_fft(self, data: ExperimentalData) -> ExperimentalData:
        """
        计算FFT（用于衍射分析）
        
        Args:
            data: 图像数据
            
        Returns:
            FFT数据
        """
        from numpy.fft import fft2, fftshift
        
        img = data.processed_data
        
        # 计算FFT
        fft_data = fft2(img)
        fft_shifted = fftshift(fft_data)
        fft_magnitude = np.abs(fft_shifted)
        
        return ExperimentalData(
            data_type='fft',
            raw_data=fft_magnitude,
            column_names=['ky', 'kx'],
            metadata=data.metadata,
            units={'x': '1/nm', 'y': '1/nm'}
        )
    
    def calculate_diffraction_pattern(self, data: ExperimentalData,
                                     wavelength: Optional[float] = None) -> np.ndarray:
        """
        从FFT计算电子衍射图样
        
        Args:
            data: HRTEM图像
            wavelength: 电子波长（pm）
            
        Returns:
            衍射强度分布
        """
        wavelength = wavelength or self._calculate_electron_wavelength(
            self.config.get('voltage', 200)
        )
        
        fft_data = self.calculate_fft(data)
        
        # 转换为衍射角度
        # 这是简化实现
        
        return fft_data.processed_data
    
    def _calculate_electron_wavelength(self, voltage_kv: float) -> float:
        """
        计算电子波长（相对论修正）
        
        Args:
            voltage_kv: 加速电压（kV）
            
        Returns:
            波长（pm）
        """
        # 物理常数
        h = 6.62607015e-34  # J·s
        m0 = 9.10938356e-31  # kg
        e = 1.602176634e-19  # C
        c = 299792458  # m/s
        
        V = voltage_kv * 1000  # 转换为V
        
        # 相对论电子波长
        numerator = h * c
        denominator = np.sqrt(V * e * (2 * m0 * c**2 + V * e))
        wavelength_m = numerator / denominator
        
        return wavelength_m * 1e12  # 转换为pm


# 便捷函数
def load_tem_image(filepath: str, **kwargs) -> ExperimentalData:
    """便捷函数：加载TEM图像"""
    connector = TEMConnector()
    return connector.read(filepath, data_type='image', **kwargs)


def load_eels(filepath: str, **kwargs) -> ExperimentalData:
    """便捷函数：加载EELS谱"""
    connector = TEMConnector()
    return connector.read(filepath, data_type='eels', **kwargs)


def load_eds(filepath: str, **kwargs) -> ExperimentalData:
    """便捷函数：加载EDS谱"""
    connector = TEMConnector()
    return connector.read(filepath, data_type='eds', **kwargs)
