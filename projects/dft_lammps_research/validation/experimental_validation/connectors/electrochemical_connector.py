"""
Electrochemical Data Connector
==============================
电化学数据连接器

支持的数据类型:
- 充放电曲线 (Galvanostatic cycling)
- 循环伏安 (Cyclic Voltammetry)
- 电化学阻抗谱 (EIS)
- 恒电位/恒电流测试

支持格式:
- CSV/TXT (通用格式)
- MPR/Biologic (Bio-Logic仪器)
- NDT/Neware (新威仪器)
- HDF5
- JSON
"""

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging

from .base_connector import BaseConnector, ExperimentalData, DataMetadata

logger = logging.getLogger(__name__)


@dataclass
class ElectrochemicalTest:
    """电化学测试参数"""
    test_type: str  # 'gcd', 'cv', 'eis', 'potentiostatic', 'galvanostatic'
    cell_config: Dict[str, Any] = None
    electrolyte: Dict[str, Any] = None
    electrode: Dict[str, Any] = None
    test_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.cell_config is None:
            self.cell_config = {}
        if self.electrolyte is None:
            self.electrolyte = {}
        if self.electrode is None:
            self.electrode = {}
        if self.test_parameters is None:
            self.test_parameters = {}


class ElectrochemicalConnector(BaseConnector):
    """
    电化学数据连接器
    
    支持各种电化学测试数据的读取和处理
    """
    
    SUPPORTED_FORMATS = ['.csv', '.txt', '.mpr', '.mpt', '.ndt', '.json', '.h5', '.hdf5']
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.config.setdefault('capacity_unit', 'mAh/g')
        self.config.setdefault('voltage_unit', 'V')
        self.config.setdefault('current_unit', 'mA')
        self.config.setdefault('mass_active_material', None)
        self.config.setdefault('area_electrode', None)
    
    def validate_format(self, filepath: str) -> bool:
        """验证电化学数据文件格式"""
        path = Path(filepath)
        if not path.exists():
            return False
        return path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def read(self, filepath: str, test_type: str = 'auto', **kwargs) -> ExperimentalData:
        """
        读取电化学数据文件
        
        Args:
            filepath: 文件路径
            test_type: 测试类型 ('gcd', 'cv', 'eis', 'auto')
            **kwargs: 额外参数
            
        Returns:
            ExperimentalData对象
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        
        # 自动检测测试类型
        if test_type == 'auto':
            test_type = self._detect_test_type(filepath)
        
        if ext in ['.csv', '.txt']:
            data = self._read_csv(filepath, test_type, **kwargs)
        elif ext in ['.mpr', '.mpt']:
            data = self._read_biologic(filepath, test_type, **kwargs)
        elif ext == '.ndt':
            data = self._read_neware(filepath, test_type, **kwargs)
        elif ext == '.json':
            data = self._read_json(filepath, **kwargs)
        elif ext in ['.h5', '.hdf5']:
            data = self._read_hdf5(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        
        # 后处理
        if test_type == 'gcd':
            data = self._process_gcd_data(data, **kwargs)
        elif test_type == 'cv':
            data = self._process_cv_data(data, **kwargs)
        elif test_type == 'eis':
            data = self._process_eis_data(data, **kwargs)
        
        data.metadata.data_type = test_type
        data.metadata.source = str(filepath)
        
        self.data.append(data)
        logger.info(f"Loaded {test_type} data from {filepath}: {len(data.raw_data)} points")
        
        return data
    
    def _detect_test_type(self, filepath: str) -> str:
        """自动检测测试类型"""
        path = Path(filepath)
        filename_lower = path.name.lower()
        
        if any(x in filename_lower for x in ['gcd', 'charge', 'discharge', 'cycle', '充放电']):
            return 'gcd'
        elif any(x in filename_lower for x in ['cv', 'voltammetry', '循环伏安']):
            return 'cv'
        elif any(x in filename_lower for x in ['eis', 'impedance', '交流阻抗', 'nyquist']):
            return 'eis'
        elif any(x in filename_lower for x in ['rate', '倍率']):
            return 'rate'
        else:
            # 尝试从文件内容推断
            return self._detect_from_content(filepath)
    
    def _detect_from_content(self, filepath: str) -> str:
        """从文件内容推断测试类型"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.read(2000).lower()
        
        if any(x in header for x in ['freq', 'frequency', 'impedance', 'zre', 'zim', 'phase']):
            return 'eis'
        elif any(x in header for x in ['scan rate', 'scanrate', 'we(1).current', 'we(1).potential']):
            return 'cv'
        elif any(x in header for x in ['cycle', 'charge', 'discharge', 'specific capacity']):
            return 'gcd'
        else:
            return 'unknown'
    
    def _read_csv(self, filepath: str, test_type: str, 
                  delimiter: Optional[str] = None,
                  skip_rows: int = 0,
                  encoding: str = 'utf-8') -> ExperimentalData:
        """读取CSV格式的电化学数据"""
        # 读取文件确定分隔符
        if delimiter is None:
            with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
                first_line = f.readline()
                if ',' in first_line:
                    delimiter = ','
                elif '\t' in first_line:
                    delimiter = '\t'
                elif ';' in first_line:
                    delimiter = ';'
                else:
                    delimiter = r'\s+'
        
        df = pd.read_csv(filepath, delimiter=delimiter, skiprows=skip_rows, encoding=encoding)
        
        # 标准化列名
        df.columns = [col.strip().lower() for col in df.columns]
        
        return ExperimentalData(
            data_type=test_type,
            raw_data=df.values,
            column_names=list(df.columns),
            units={}
        )
    
    def _read_biologic(self, filepath: str, test_type: str, **kwargs) -> ExperimentalData:
        """
        读取Bio-Logic仪器的.mpr或.mpt文件
        
        MPR是二进制格式，MPT是文本格式
        """
        if filepath.endswith('.mpt'):
            return self._read_biologic_text(filepath, test_type)
        else:
            return self._read_biologic_binary(filepath, test_type)
    
    def _read_biologic_text(self, filepath: str, test_type: str) -> ExperimentalData:
        """读取Bio-Logic MPT文本文件"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 查找数据开始行
        header_lines = 0
        for i, line in enumerate(lines):
            if line.startswith('mode') or line.startswith('freq/Hz'):
                header_lines = i
                break
        
        df = pd.read_csv(filepath, skiprows=header_lines, delimiter='\t', encoding='utf-8')
        
        # 标准化列名
        column_mapping = {
            'freq/Hz': 'frequency',
            'z/ohm': 'impedance_magnitude',
            'phase(z)/deg': 'phase_angle',
            '|z|/ohm': 'impedance_magnitude',
            'time/s': 'time',
            'ewe/V': 'voltage',
            '<i>/mA': 'current',
            'cycle number': 'cycle',
            'ox/red': 'direction',
            'capacity/mA.h': 'capacity',
        }
        
        df = df.rename(columns=column_mapping)
        df.columns = [col.strip().lower() for col in df.columns]
        
        return ExperimentalData(
            data_type=test_type,
            raw_data=df.values,
            column_names=list(df.columns),
            units={'frequency': 'Hz', 'voltage': 'V', 'current': 'mA', 'capacity': 'mAh'}
        )
    
    def _read_biologic_binary(self, filepath: str, test_type: str) -> ExperimentalData:
        """
        读取Bio-Logic MPR二进制文件
        
        这是简化实现，完整的解析需要更多细节
        """
        with open(filepath, 'rb') as f:
            # MPR文件头解析
            header = f.read(20)
            
            # 读取剩余数据（简化处理）
            # 实际格式需要逆向工程
            data_bytes = f.read()
        
        # 假设数据是浮点数数组
        # 这只是一个占位符实现
        data_array = np.frombuffer(data_bytes, dtype=np.float32)
        n_cols = 5  # 假设5列
        n_rows = len(data_array) // n_cols
        data_array = data_array[:n_rows * n_cols].reshape(n_rows, n_cols)
        
        return ExperimentalData(
            data_type=test_type,
            raw_data=data_array,
            column_names=['col1', 'col2', 'col3', 'col4', 'col5'],
            units={}
        )
    
    def _read_neware(self, filepath: str, test_type: str, **kwargs) -> ExperimentalData:
        """
        读取新威(NEWARE)仪器的NDT文件
        
        新威是常见的电池测试设备制造商
        """
        # NDT通常是二进制格式
        with open(filepath, 'rb') as f:
            # 读取文件头
            header = f.read(512)
            
            # 解析数据
            # 这是简化实现
            data = []
            while True:
                record = f.read(48)  # 假设每条记录48字节
                if not record or len(record) < 48:
                    break
                
                # 解析记录
                try:
                    record_data = struct.unpack('iffffff', record[:28])  # 示例格式
                    data.append(record_data)
                except:
                    continue
        
        data_array = np.array(data)
        
        return ExperimentalData(
            data_type=test_type,
            raw_data=data_array,
            column_names=['record_id', 'cycle', 'step', 'time', 'voltage', 'current', 'capacity'],
            units={'time': 's', 'voltage': 'V', 'current': 'mA', 'capacity': 'mAh'}
        )
    
    def _read_json(self, filepath: str, **kwargs) -> ExperimentalData:
        """读取JSON格式的电化学数据"""
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        
        # 提取数据
        if 'data' in json_data:
            data_array = np.array(json_data['data'])
        elif 'cycles' in json_data:
            # 合并所有循环数据
            all_data = []
            for cycle_data in json_data['cycles']:
                all_data.extend(cycle_data)
            data_array = np.array(all_data)
        else:
            data_array = np.array(list(json_data.values()))
        
        metadata = json_data.get('metadata', {})
        column_names = json_data.get('columns', [f'col_{i}' for i in range(data_array.shape[1])])
        
        return ExperimentalData(
            data_type=json_data.get('test_type', 'unknown'),
            raw_data=data_array,
            column_names=column_names,
            metadata=DataMetadata(**metadata),
            units=json_data.get('units', {})
        )
    
    def _read_hdf5(self, filepath: str, **kwargs) -> ExperimentalData:
        """读取HDF5格式的电化学数据"""
        try:
            import h5py
            with h5py.File(filepath, 'r') as f:
                dataset_path = kwargs.get('dataset_path', 'data')
                data = f[dataset_path][:]
                
                attrs = dict(f[dataset_path].attrs)
                column_names = json.loads(attrs.get('columns', '[]'))
                test_type = attrs.get('test_type', 'unknown')
                units = json.loads(attrs.get('units', '{}'))
                
                return ExperimentalData(
                    data_type=test_type,
                    raw_data=data,
                    column_names=column_names,
                    units=units
                )
        except ImportError:
            raise ImportError("h5py is required for HDF5 format")
    
    def _process_gcd_data(self, data: ExperimentalData, 
                          mass: Optional[float] = None,
                          area: Optional[float] = None) -> ExperimentalData:
        """处理恒流充放电数据"""
        mass = mass or self.config.get('mass_active_material')
        area = area or self.config.get('area_electrode')
        
        df = data.to_dataframe()
        
        # 标准化列名映射
        col_mapping = {
            'voltage': ['voltage', 'v', 'ewe/v', 'potential', 'ewe'],
            'current': ['current', 'i', 'i/mA', '<i>/mA', 'current/mA'],
            'time': ['time', 't', 'time/s', 't/s'],
            'capacity': ['capacity', 'cap', 'capacity/mAh', 'specific_capacity'],
            'cycle': ['cycle', 'cycle_number', 'cyclenumber', 'n'],
        }
        
        # 查找列
        for standard_name, possible_names in col_mapping.items():
            for col in df.columns:
                if any(name in col for name in possible_names):
                    df.rename(columns={col: standard_name}, inplace=True)
                    break
        
        # 计算比容量（如果提供了质量）
        if mass and 'capacity' in df.columns:
            df['specific_capacity'] = df['capacity'] / mass * 1000  # mAh/g
        
        data.processed_data = df.values
        data.column_names = list(df.columns)
        
        return data
    
    def _process_cv_data(self, data: ExperimentalData, **kwargs) -> ExperimentalData:
        """处理循环伏安数据"""
        df = data.to_dataframe()
        
        # 标准化列名
        col_mapping = {
            'potential': ['potential', 'v', 'ewe/v', 'voltage'],
            'current': ['current', 'i', 'i/mA', '<i>/mA'],
            'scan_rate': ['scan_rate', 'scanrate', 'v/mv/s'],
            'cycle': ['cycle', 'cycle_number'],
        }
        
        for standard_name, possible_names in col_mapping.items():
            for col in df.columns:
                if any(name in col for name in possible_names):
                    df.rename(columns={col: standard_name}, inplace=True)
                    break
        
        # 计算电流密度（如果提供了面积）
        area = kwargs.get('area', self.config.get('area_electrode'))
        if area and 'current' in df.columns:
            df['current_density'] = df['current'] / area  # mA/cm²
        
        data.processed_data = df.values
        data.column_names = list(df.columns)
        
        return data
    
    def _process_eis_data(self, data: ExperimentalData, **kwargs) -> ExperimentalData:
        """处理EIS数据"""
        df = data.to_dataframe()
        
        # 标准化列名
        col_mapping = {
            'frequency': ['frequency', 'freq', 'freq/hz', 'f'],
            'z_real': ['zreal', 'z_real', 're(z)', 'z\'/ohm', 'z_re'],
            'z_imag': ['zimag', 'z_imag', '-im(z)', '-z\'/ohm', 'z_im'],
            'z_mag': ['z', '|z|', 'z/ohm', 'magnitude'],
            'phase': ['phase', 'phase_angle', 'phase(z)/deg'],
        }
        
        for standard_name, possible_names in col_mapping.items():
            for col in df.columns:
                if any(name in col for name in possible_names):
                    df.rename(columns={col: standard_name}, inplace=True)
                    break
        
        # 计算阻抗分量（如果没有直接提供）
        if 'z_real' not in df.columns and 'z_mag' in df.columns and 'phase' in df.columns:
            z_mag = df['z_mag']
            phase_rad = np.radians(df['phase'])
            df['z_real'] = z_mag * np.cos(phase_rad)
            df['z_imag'] = -z_mag * np.sin(phase_rad)  # 负号按照电化学惯例
        
        data.processed_data = df.values
        data.column_names = list(df.columns)
        
        return data
    
    def calculate_capacity(self, data: ExperimentalData, 
                          mass: float,
                          unit: str = 'mAh/g') -> ExperimentalData:
        """
        计算比容量
        
        Args:
            data: 原始数据
            mass: 活性物质质量 (mg)
            unit: 输出单位
            
        Returns:
            添加了比容量列的数据
        """
        df = data.to_dataframe()
        
        if 'capacity' in df.columns:
            if unit == 'mAh/g':
                df['specific_capacity'] = df['capacity'] / mass * 1000
            elif unit == 'Ah/kg':
                df['specific_capacity'] = df['capacity'] / mass
            elif unit == 'mAh/cm2':
                area = self.config.get('area_electrode', 1.0)
                df['areal_capacity'] = df['capacity'] / area
        
        data.processed_data = df.values
        if 'specific_capacity' not in data.column_names:
            data.column_names.append('specific_capacity')
        
        return data
    
    def extract_cycles(self, data: ExperimentalData, 
                      cycle_numbers: Optional[List[int]] = None) -> Dict[int, ExperimentalData]:
        """
        提取特定循环的数据
        
        Args:
            data: 包含多个循环的数据
            cycle_numbers: 要提取的循环号列表，None表示提取所有
            
        Returns:
            循环号到数据的映射
        """
        df = data.to_dataframe()
        
        if 'cycle' not in df.columns:
            logger.warning("No cycle column found")
            return {0: data}
        
        cycles = {}
        unique_cycles = df['cycle'].unique()
        
        if cycle_numbers is None:
            cycle_numbers = unique_cycles
        
        for cycle_num in cycle_numbers:
            if cycle_num in unique_cycles:
                cycle_df = df[df['cycle'] == cycle_num]
                cycles[cycle_num] = ExperimentalData(
                    data_type=data.data_type,
                    raw_data=cycle_df.values,
                    column_names=data.column_names,
                    units=data.units,
                    metadata=data.metadata
                )
        
        return cycles
    
    def calculate_coulombic_efficiency(self, data: ExperimentalData) -> List[float]:
        """
        计算库伦效率
        
        对于每个循环，CE = 放电容量 / 充电容量
        
        Returns:
            每个循环的库伦效率列表
        """
        cycles = self.extract_cycles(data)
        efficiencies = []
        
        for cycle_num, cycle_data in sorted(cycles.items()):
            df = cycle_data.to_dataframe()
            
            if 'capacity' not in df.columns:
                continue
            
            # 假设负电流是充电，正电流是放电（或相反，取决于惯例）
            if 'current' in df.columns:
                charge_cap = df[df['current'] < 0]['capacity'].max()
                discharge_cap = df[df['current'] > 0]['capacity'].max()
            else:
                # 如果没有电流信息，使用第一个和最后一个容量值
                charge_cap = df['capacity'].iloc[0]
                discharge_cap = df['capacity'].iloc[-1]
            
            if charge_cap > 0:
                ce = discharge_cap / charge_cap * 100
                efficiencies.append(ce)
            else:
                efficiencies.append(0.0)
        
        return efficiencies


# 便捷函数
def load_electrochemical(filepath: str, test_type: str = 'auto', **kwargs) -> ExperimentalData:
    """便捷函数：加载电化学数据"""
    connector = ElectrochemicalConnector()
    return connector.read(filepath, test_type=test_type, **kwargs)
