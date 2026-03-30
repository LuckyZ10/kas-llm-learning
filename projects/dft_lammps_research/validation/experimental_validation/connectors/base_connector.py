"""
Base Data Connector
===================
所有数据连接器的基础类
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class DataMetadata:
    """实验数据元数据"""
    source: str = ""
    data_type: str = ""
    sample_id: str = ""
    experiment_date: str = ""
    operator: str = ""
    instrument: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataMetadata':
        return cls(**data)


@dataclass
class ExperimentalData:
    """统一的实验数据结构"""
    data_type: str
    raw_data: np.ndarray
    processed_data: Optional[np.ndarray] = None
    metadata: DataMetadata = field(default_factory=DataMetadata)
    units: Dict[str, str] = field(default_factory=dict)
    column_names: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.processed_data is None:
            self.processed_data = self.raw_data.copy()
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        if len(self.column_names) == self.raw_data.shape[1]:
            return pd.DataFrame(self.raw_data, columns=self.column_names)
        return pd.DataFrame(self.raw_data)
    
    def save(self, filepath: str, format: str = "auto"):
        """保存数据"""
        path = Path(filepath)
        if format == "auto":
            format = path.suffix.lower().lstrip('.')
        
        if format in ['json']:
            data = {
                'data_type': self.data_type,
                'metadata': self.metadata.to_dict(),
                'units': self.units,
                'column_names': self.column_names,
                'raw_data': self.raw_data.tolist(),
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format in ['csv']:
            df = self.to_dataframe()
            df.to_csv(path, index=False)
        elif format in ['h5', 'hdf5']:
            self._save_hdf5(path)
        elif format in ['npy', 'npz']:
            np.savez(path, 
                    raw_data=self.raw_data,
                    processed_data=self.processed_data,
                    metadata=json.dumps(self.metadata.to_dict()),
                    units=json.dumps(self.units))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved experimental data to {filepath}")
    
    def _save_hdf5(self, path: Path):
        """保存为HDF5格式"""
        try:
            import h5py
            with h5py.File(path, 'w') as f:
                f.create_dataset('raw_data', data=self.raw_data)
                f.create_dataset('processed_data', data=self.processed_data)
                f.attrs['data_type'] = self.data_type
                f.attrs['metadata'] = json.dumps(self.metadata.to_dict())
                f.attrs['units'] = json.dumps(self.units)
                f.attrs['column_names'] = json.dumps(self.column_names)
        except ImportError:
            raise ImportError("h5py is required for HDF5 format")
    
    @classmethod
    def load(cls, filepath: str, format: str = "auto") -> 'ExperimentalData':
        """加载数据"""
        path = Path(filepath)
        if format == "auto":
            format = path.suffix.lower().lstrip('.')
        
        if format in ['json']:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(
                data_type=data['data_type'],
                raw_data=np.array(data['raw_data']),
                metadata=DataMetadata.from_dict(data['metadata']),
                units=data.get('units', {}),
                column_names=data.get('column_names', [])
            )
        elif format in ['npy', 'npz']:
            loaded = np.load(path, allow_pickle=True)
            return cls(
                data_type='unknown',
                raw_data=loaded['raw_data'],
                processed_data=loaded.get('processed_data'),
                metadata=DataMetadata.from_dict(json.loads(str(loaded['metadata']))),
                units=json.loads(str(loaded['units']))
            )
        else:
            raise ValueError(f"Unsupported format: {format}")


class BaseConnector(ABC):
    """
    数据连接器基类
    
    所有具体连接器都应该继承此类并实现抽象方法
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.data: List[ExperimentalData] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def read(self, filepath: str, **kwargs) -> ExperimentalData:
        """
        读取数据文件
        
        Args:
            filepath: 数据文件路径
            **kwargs: 额外参数
            
        Returns:
            ExperimentalData对象
        """
        pass
    
    @abstractmethod
    def validate_format(self, filepath: str) -> bool:
        """
        验证文件格式是否正确
        
        Args:
            filepath: 文件路径
            
        Returns:
            格式是否有效
        """
        pass
    
    def read_batch(self, filepaths: List[str], **kwargs) -> List[ExperimentalData]:
        """
        批量读取多个文件
        
        Args:
            filepaths: 文件路径列表
            **kwargs: 传递给read方法的参数
            
        Returns:
            ExperimentalData对象列表
        """
        results = []
        for filepath in filepaths:
            try:
                data = self.read(filepath, **kwargs)
                results.append(data)
            except Exception as e:
                self.logger.error(f"Failed to read {filepath}: {e}")
        return results
    
    def read_directory(self, directory: str, pattern: str = "*", **kwargs) -> List[ExperimentalData]:
        """
        读取整个目录中的数据文件
        
        Args:
            directory: 目录路径
            pattern: 文件名匹配模式
            **kwargs: 传递给read方法的参数
            
        Returns:
            ExperimentalData对象列表
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = list(dir_path.glob(pattern))
        self.logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")
        
        return self.read_batch([str(f) for f in files], **kwargs)
    
    def preprocess(self, data: ExperimentalData, operations: List[str]) -> ExperimentalData:
        """
        预处理数据
        
        Args:
            data: 输入数据
            operations: 预处理操作列表 ['normalize', 'smooth', 'baseline']
            
        Returns:
            处理后的数据
        """
        processed = data.processed_data.copy()
        
        for op in operations:
            if op == 'normalize':
                processed = self._normalize(processed)
            elif op == 'smooth':
                processed = self._smooth(processed)
            elif op == 'baseline':
                processed = self._remove_baseline(processed)
            elif op == 'clip':
                processed = self._clip_outliers(processed)
            else:
                self.logger.warning(f"Unknown preprocessing operation: {op}")
        
        data.processed_data = processed
        return data
    
    def _normalize(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """归一化数据"""
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        return (data - min_val) / range_val
    
    def _smooth(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """平滑数据（移动平均）"""
        kernel = np.ones(window) / window
        if data.ndim == 1:
            return np.convolve(data, kernel, mode='same')
        else:
            # 对每列应用平滑
            result = np.zeros_like(data)
            for i in range(data.shape[1]):
                result[:, i] = np.convolve(data[:, i], kernel, mode='same')
            return result
    
    def _remove_baseline(self, data: np.ndarray, method: str = 'linear') -> np.ndarray:
        """去除基线"""
        if method == 'linear':
            # 简单的线性基线拟合
            if data.ndim == 1:
                x = np.arange(len(data))
                coeffs = np.polyfit(x, data, 1)
                baseline = np.polyval(coeffs, x)
                return data - baseline
            else:
                result = data.copy()
                for i in range(data.shape[1]):
                    x = np.arange(len(data))
                    coeffs = np.polyfit(x, data[:, i], 1)
                    baseline = np.polyval(coeffs, x)
                    result[:, i] = data[:, i] - baseline
                return result
        return data
    
    def _clip_outliers(self, data: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """裁剪异常值"""
        mean = np.mean(data)
        std = np.std(data)
        lower = mean - n_std * std
        upper = mean + n_std * std
        return np.clip(data, lower, upper)
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要"""
        return {
            'n_datasets': len(self.data),
            'data_types': list(set(d.data_type for d in self.data)),
            'total_samples': sum(len(d.raw_data) for d in self.data),
        }
