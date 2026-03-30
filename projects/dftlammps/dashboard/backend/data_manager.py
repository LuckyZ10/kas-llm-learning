"""
DFT/LAMMPS Dashboard - 核心数据管理模块
=============================================
提供数据加载、处理和缓存功能

参考设计理念:
- Clean Architecture: 数据层与展示层分离
- Reactive Patterns: 数据流驱动UI更新
"""

import os
import json
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
from functools import lru_cache
import threading
import time

import numpy as np
import pandas as pd

# ASE for structure handling
try:
    from ase.io import read, write
    from ase import Atoms
    from ase.geometry import get_distances
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    logging.warning("ASE not available. Structure handling limited.")

# pymatgen for crystal analysis
try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.vasp import Vasprun, Outcar
    from pymatgen.io.lammps.data import LammpsData
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    logging.warning("pymatgen not available. Crystal analysis limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """仪表板配置类"""
    # 数据路径
    work_dir: str = "./workspace"
    dft_results_path: str = "./workspace/dft_results"
    md_results_path: str = "./workspace/md_results"
    ml_models_path: str = "./workspace/ml_models"
    screening_db_path: str = "./workspace/screening_db"
    workflows_path: str = "./workspace/workflows"
    
    # 缓存设置
    cache_enabled: bool = True
    cache_ttl: int = 60  # 秒
    max_cache_size: int = 100
    
    # 性能设置
    max_points: int = 50000
    time_window: int = 7200  # 2小时
    
    # 可视化阈值
    force_error_threshold: float = 0.05  # eV/Å
    energy_error_threshold: float = 0.001  # eV/atom
    virial_error_threshold: float = 0.5  # eV
    
    # 实时更新
    auto_refresh: bool = True
    refresh_interval: int = 5  # 秒
    
    # 服务器设置
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False
    
    # 文件模式
    file_patterns: Dict[str, str] = field(default_factory=lambda: {
        'training_log': '**/lcurve*.out',
        'md_log': '**/log.lammps',
        'trajectory': '**/*.lammpstrj',
        'vasp_outcar': '**/OUTCAR',
        'vasp_vasprun': '**/vasprun.xml',
        'structure': '**/*.xyz',
        'screening_csv': '**/screening_results.csv',
        'screening_json': '**/screening_results.json',
    })
    
    @classmethod
    def from_yaml(cls, path: str) -> 'DashboardConfig':
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """保存配置到YAML文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)


class DataCache:
    """线程安全的数据缓存系统"""
    
    def __init__(self, ttl: int = 60, max_size: int = 100):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        with self._lock:
            if key not in self._cache:
                return None
            
            data, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None
            return data
    
    def set(self, key: str, data: Any):
        """设置缓存数据"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                # 移除最旧的条目
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            self._cache[key] = (data, time.time())
    
    def invalidate(self, pattern: Optional[str] = None):
        """使缓存失效"""
        with self._lock:
            if pattern is None:
                self._cache.clear()
            else:
                keys_to_remove = [k for k in self._cache if pattern in k]
                for k in keys_to_remove:
                    del self._cache[k]
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()


class StructureDataManager:
    """分子/晶体结构数据管理器"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.cache = DataCache(config.cache_ttl, config.max_cache_size)
    
    def load_structure(self, filepath: str) -> Optional[Dict[str, Any]]:
        """加载分子/晶体结构"""
        cache_key = f"structure_{filepath}_{os.path.getmtime(filepath) if os.path.exists(filepath) else 0}"
        
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            if not os.path.exists(filepath):
                return None
            
            # 尝试使用ASE
            if ASE_AVAILABLE:
                atoms = read(filepath)
                data = self._atoms_to_dict(atoms)
                self.cache.set(cache_key, data)
                return data
            
            # 回退：读取原始XYZ
            return self._read_xyz_raw(filepath)
            
        except Exception as e:
            logger.error(f"Error loading structure {filepath}: {e}")
            return None
    
    def load_trajectory(self, filepath: str, max_frames: int = 100) -> Optional[Dict[str, Any]]:
        """加载MD轨迹"""
        cache_key = f"traj_{filepath}_{max_frames}_{os.path.getmtime(filepath) if os.path.exists(filepath) else 0}"
        
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            if not os.path.exists(filepath):
                return None
            
            if ASE_AVAILABLE:
                frames = read(filepath, index=':')
                if len(frames) > max_frames:
                    # 均匀采样
                    indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
                    frames = [frames[i] for i in indices]
                
                data = {
                    'frames': [self._atoms_to_dict(f) for f in frames],
                    'n_frames': len(frames),
                    'n_atoms': len(frames[0]) if frames else 0,
                }
                self.cache.set(cache_key, data)
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading trajectory {filepath}: {e}")
            return None
    
    def _atoms_to_dict(self, atoms: 'Atoms') -> Dict[str, Any]:
        """将ASE Atoms转换为字典"""
        return {
            'positions': atoms.positions.tolist(),
            'symbols': atoms.get_chemical_symbols(),
            'cell': atoms.cell.tolist() if atoms.cell is not None else None,
            'pbc': atoms.pbc.tolist(),
            'numbers': atoms.numbers.tolist(),
            'n_atoms': len(atoms),
            'energy': atoms.get_potential_energy() if atoms.calc else None,
            'forces': atoms.get_forces().tolist() if atoms.calc and atoms.calc.results.get('forces') is not None else None,
        }
    
    def _read_xyz_raw(self, filepath: str) -> Dict[str, Any]:
        """原始XYZ文件读取"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        n_atoms = int(lines[0].strip())
        comment = lines[1].strip()
        
        symbols = []
        positions = []
        
        for line in lines[2:2+n_atoms]:
            parts = line.split()
            symbols.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])
        
        return {
            'positions': positions,
            'symbols': symbols,
            'cell': None,
            'pbc': [False, False, False],
            'numbers': [],
            'n_atoms': n_atoms,
            'comment': comment,
            'energy': None,
            'forces': None,
        }
    
    def analyze_structure(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析结构特征"""
        analysis = {
            'composition': {},
            'bond_stats': {},
            'density': None,
        }
        
        # 计算组成
        symbols = structure_data.get('symbols', [])
        for sym in symbols:
            analysis['composition'][sym] = analysis['composition'].get(sym, 0) + 1
        
        # 计算密度
        cell = structure_data.get('cell')
        if cell is not None and ASE_AVAILABLE:
            try:
                atoms = Atoms(
                    symbols=structure_data['symbols'],
                    positions=structure_data['positions'],
                    cell=cell,
                    pbc=structure_data.get('pbc', [True, True, True])
                )
                volume = atoms.get_volume()
                mass = sum(atoms.get_masses())
                analysis['density'] = mass / volume * 1.66054  # g/cm³
            except Exception as e:
                logger.warning(f"Density calculation failed: {e}")
        
        return analysis


class TrainingDataManager:
    """ML训练数据管理器"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.cache = DataCache(config.cache_ttl, config.max_cache_size)
    
    def load_training_log(self, log_path: Optional[str] = None) -> pd.DataFrame:
        """加载DeePMD训练日志"""
        if log_path is None:
            log_path = self._find_training_log()
        
        if not log_path or not os.path.exists(log_path):
            return pd.DataFrame()
        
        cache_key = f"training_{log_path}_{os.path.getmtime(log_path)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            df = pd.read_csv(log_path, sep=r'\s+', comment='#')
            
            # 标准化列名
            column_mapping = {
                'batch': 'step',
                'step': 'step',
                'lr': 'learning_rate',
                'loss': 'total_loss',
                'energy_rmse': 'energy_rmse',
                'force_rmse': 'force_rmse',
                'virial_rmse': 'virial_rmse',
            }
            
            df.columns = [column_mapping.get(c.lower(), c) for c in df.columns]
            
            # 添加时间戳（如果没有）
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(
                    end=datetime.now(),
                    periods=len(df),
                    freq='10S'
                )
            
            self.cache.set(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"Error loading training log: {e}")
            return pd.DataFrame()
    
    def get_training_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算训练统计信息"""
        if df.empty:
            return {
                'current_step': 0,
                'current_loss': 0,
                'energy_rmse': 0,
                'force_rmse': 0,
                'converged': False,
            }
        
        latest = df.iloc[-1]
        
        energy_rmse = latest.get('energy_rmse', float('inf'))
        force_rmse = latest.get('force_rmse', float('inf'))
        
        return {
            'current_step': int(latest.get('step', 0)),
            'current_loss': float(latest.get('total_loss', latest.get('loss', 0))),
            'learning_rate': float(latest.get('learning_rate', 0)),
            'energy_rmse': float(energy_rmse) if pd.notna(energy_rmse) else None,
            'force_rmse': float(force_rmse) if pd.notna(force_rmse) else None,
            'converged': (
                energy_rmse < self.config.energy_error_threshold and
                force_rmse < self.config.force_error_threshold
            ),
            'total_steps': len(df),
        }
    
    def _find_training_log(self) -> Optional[str]:
        """查找训练日志文件"""
        work_dir = Path(self.config.work_dir)
        pattern = self.config.file_patterns.get('training_log', '**/lcurve*.out')
        
        for path in work_dir.glob(pattern):
            return str(path)
        
        return None


class MDDataManager:
    """分子动力学数据管理器"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.cache = DataCache(config.cache_ttl, config.max_cache_size)
    
    def load_md_log(self, log_path: Optional[str] = None) -> pd.DataFrame:
        """加载LAMMPS日志"""
        if log_path is None:
            log_path = self._find_md_log()
        
        if not log_path or not os.path.exists(log_path):
            return pd.DataFrame()
        
        cache_key = f"md_{log_path}_{os.path.getmtime(log_path)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            # 解析LAMMPS日志
            sections = content.split('Loop time of')
            
            data_rows = []
            for section in sections:
                lines = section.strip().split('\n')
                
                # 查找列标题行
                header_idx = None
                for i, line in enumerate(lines):
                    if line.startswith('Step '):
                        header_idx = i
                        break
                
                if header_idx is None:
                    continue
                
                headers = lines[header_idx].split()
                
                # 读取数据行
                for line in lines[header_idx+1:]:
                    if line.strip() == '' or line.startswith('Loop'):
                        break
                    values = line.split()
                    if len(values) == len(headers):
                        row = dict(zip(headers, values))
                        data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            
            # 转换数值列
            numeric_cols = ['Step', 'Temp', 'PotEng', 'KinEng', 'TotEng', 'Press', 'Volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['Step'])
            
            self.cache.set(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"Error loading MD log: {e}")
            return pd.DataFrame()
    
    def calculate_msd(self, trajectory: List[Dict], dt: float = 1.0) -> pd.DataFrame:
        """计算均方位移(MSD)"""
        if not trajectory or len(trajectory) < 2:
            return pd.DataFrame()
        
        n_frames = len(trajectory)
        msd_data = []
        
        # 获取初始位置
        initial_pos = np.array(trajectory[0]['positions'])
        
        for i, frame in enumerate(trajectory):
            positions = np.array(frame['positions'])
            displacement = positions - initial_pos
            msd = np.mean(np.sum(displacement**2, axis=1))
            
            msd_data.append({
                'frame': i,
                'time': i * dt,
                'msd': msd,
            })
        
        return pd.DataFrame(msd_data)
    
    def get_md_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算MD统计信息"""
        if df.empty:
            return {
                'current_step': 0,
                'avg_temp': 0,
                'avg_pressure': 0,
                'total_energy': 0,
            }
        
        return {
            'current_step': int(df['Step'].max()) if 'Step' in df.columns else 0,
            'avg_temp': float(df['Temp'].mean()) if 'Temp' in df.columns else 0,
            'max_temp': float(df['Temp'].max()) if 'Temp' in df.columns else 0,
            'min_temp': float(df['Temp'].min()) if 'Temp' in df.columns else 0,
            'avg_pressure': float(df['Press'].mean()) if 'Press' in df.columns else 0,
            'total_energy': float(df['TotEng'].iloc[-1]) if 'TotEng' in df.columns else 0,
            'n_steps': len(df),
        }
    
    def _find_md_log(self) -> Optional[str]:
        """查找MD日志文件"""
        work_dir = Path(self.config.work_dir)
        pattern = self.config.file_patterns.get('md_log', '**/log.lammps')
        
        for path in work_dir.glob(pattern):
            return str(path)
        
        return None


class ScreeningDataManager:
    """高通量筛选数据管理器"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.cache = DataCache(config.cache_ttl, config.max_cache_size)
    
    def load_screening_results(self) -> pd.DataFrame:
        """加载筛选结果"""
        db_path = Path(self.config.screening_db_path)
        
        # 尝试不同的文件格式
        for ext, loader in [
            ('screening_results.csv', lambda p: pd.read_csv(p)),
            ('screening_results.json', lambda p: pd.read_json(p)),
            ('results.csv', lambda p: pd.read_csv(p)),
        ]:
            filepath = db_path / ext
            if filepath.exists():
                cache_key = f"screening_{filepath}_{os.path.getmtime(filepath)}"
                cached = self.cache.get(cache_key)
                if cached is not None:
                    return cached
                
                try:
                    df = loader(filepath)
                    self.cache.set(cache_key, df)
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
        
        return pd.DataFrame()
    
    def get_pareto_front(self, df: pd.DataFrame, x_col: str, y_col: str, minimize: bool = True) -> pd.DataFrame:
        """计算Pareto前沿"""
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            return pd.DataFrame()
        
        data = df[[x_col, y_col]].dropna()
        
        if minimize:
            # 两个目标都最小化
            pareto_mask = np.ones(len(data), dtype=bool)
            for i, (x, y) in enumerate(data.values):
                if not pareto_mask[i]:
                    continue
                for j, (x2, y2) in enumerate(data.values):
                    if i != j and x2 <= x and y2 <= y and (x2 < x or y2 < y):
                        pareto_mask[i] = False
                        break
        else:
            # 默认：x最大化，y最小化
            pareto_mask = np.ones(len(data), dtype=bool)
            for i, (x, y) in enumerate(data.values):
                if not pareto_mask[i]:
                    continue
                for j, (x2, y2) in enumerate(data.values):
                    if i != j and x2 >= x and y2 <= y and (x2 > x or y2 < y):
                        pareto_mask[i] = False
                        break
        
        return df.iloc[data.index[pareto_mask]]
    
    def filter_data(self, df: pd.DataFrame, filters: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """应用筛选条件"""
        filtered = df.copy()
        
        for col, (min_val, max_val) in filters.items():
            if col in filtered.columns:
                filtered = filtered[
                    (filtered[col] >= min_val) & 
                    (filtered[col] <= max_val)
                ]
        
        return filtered


class DataManager:
    """统一数据管理器"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        
        # 子管理器
        self.structure = StructureDataManager(self.config)
        self.training = TrainingDataManager(self.config)
        self.md = MDDataManager(self.config)
        self.screening = ScreeningDataManager(self.config)
        
        # 全局缓存
        self.cache = DataCache(self.config.cache_ttl, self.config.max_cache_size)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统整体状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'work_dir_exists': os.path.exists(self.config.work_dir),
            'modules': {
                'ase': ASE_AVAILABLE,
                'pymatgen': PYMATGEN_AVAILABLE,
            },
            'cache_stats': {
                'structure': len(self.structure.cache._cache),
                'training': len(self.training.cache._cache),
                'md': len(self.md.cache._cache),
                'screening': len(self.screening.cache._cache),
            }
        }
        
        # 检查各模块数据可用性
        status['data_available'] = {
            'training_log': self.training._find_training_log() is not None,
            'md_log': self.md._find_md_log() is not None,
            'screening': not self.screening.load_screening_results().empty,
        }
        
        return status
    
    def invalidate_cache(self, pattern: Optional[str] = None):
        """使所有缓存失效"""
        self.structure.cache.invalidate(pattern)
        self.training.cache.invalidate(pattern)
        self.md.cache.invalidate(pattern)
        self.screening.cache.invalidate(pattern)


def load_config(path: str = "dashboard_config.yaml") -> DashboardConfig:
    """加载配置"""
    if os.path.exists(path):
        return DashboardConfig.from_yaml(path)
    return DashboardConfig()
