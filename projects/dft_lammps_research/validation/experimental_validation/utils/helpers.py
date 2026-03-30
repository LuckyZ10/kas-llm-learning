"""
Utility Functions
=================
实用工具函数
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    加载配置文件
    
    支持JSON和YAML格式
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.json']:
            return json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict, config_path: str):
    """保存配置文件"""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix in ['.json']:
            json.dump(config, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                yaml.dump(config, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")


def compute_file_hash(filepath: str, algorithm: str = 'md5') -> str:
    """计算文件哈希值"""
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def ensure_dir(directory: str) -> Path:
    """确保目录存在"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_number(value: float, precision: int = 4) -> str:
    """格式化数字"""
    if abs(value) < 0.01 or abs(value) > 10000:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def calculate_progress(current: int, total: int) -> Dict[str, Any]:
    """计算进度信息"""
    percent = (current / total * 100) if total > 0 else 0
    
    return {
        'current': current,
        'total': total,
        'percent': percent,
        'remaining': total - current,
    }


def merge_dicts(*dicts: Dict) -> Dict:
    """合并多个字典"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def filter_dict(d: Dict, keys: List[str]) -> Dict:
    """筛选字典中的指定键"""
    return {k: v for k, v in d.items() if k in keys}


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    展平嵌套字典
    
    例如: {'a': {'b': 1}} -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = '.') -> Dict:
    """反展平字典"""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return result


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    return a / b if b != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """限制值在范围内"""
    return max(min_val, min(max_val, value))


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    归一化数组
    
    Args:
        arr: 输入数组
        method: 'minmax', 'zscore', 'robust'
    """
    if method == 'minmax':
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        return arr
    
    elif method == 'zscore':
        mean = np.mean(arr)
        std = np.std(arr)
        if std > 0:
            return (arr - mean) / std
        return arr
    
    elif method == 'robust':
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad > 0:
            return (arr - median) / (1.4826 * mad)
        return arr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def remove_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
    """
    去除异常值
    
    Args:
        data: 输入数据
        method: 'iqr', 'zscore', 'mad'
        threshold: 阈值
    """
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (data >= lower) & (data <= upper)
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        mask = z_scores <= threshold
    
    elif method == 'mad':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z = 0.6745 * (data - median) / mad if mad > 0 else np.zeros_like(data)
        mask = np.abs(modified_z) <= threshold
    
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    
    return data[mask]


def estimate_runtime(start_time: float, current: int, total: int) -> Dict[str, float]:
    """
    估计运行时间
    
    Args:
        start_time: 开始时间（time.time()）
        current: 当前进度
        total: 总进度
        
    Returns:
        时间估计信息
    """
    import time
    
    elapsed = time.time() - start_time
    
    if current > 0:
        rate = elapsed / current
        remaining = rate * (total - current)
        total_time = rate * total
    else:
        remaining = 0
        total_time = 0
    
    return {
        'elapsed_seconds': elapsed,
        'remaining_seconds': remaining,
        'total_estimate_seconds': total_time,
        'percent_complete': (current / total * 100) if total > 0 else 0,
    }


def create_summary_table(data: Dict[str, Dict], 
                        columns: List[str] = None) -> str:
    """
    创建Markdown格式的汇总表
    
    Args:
        data: {行名: {列名: 值}}
        columns: 列名列表
        
    Returns:
        Markdown表格字符串
    """
    if not data:
        return ""
    
    # 获取所有列
    if columns is None:
        columns = []
        for row_data in data.values():
            columns.extend(row_data.keys())
        columns = sorted(set(columns))
    
    # 创建表头
    lines = ['| 项目 | ' + ' | '.join(columns) + ' |']
    lines.append('|------|' + '|'.join(['------'] * len(columns)) + '|')
    
    # 添加数据行
    for name, row_data in data.items():
        row_values = [str(row_data.get(col, '')) for col in columns]
        lines.append(f"| {name} | " + ' | '.join(row_values) + ' |')
    
    return '\n'.join(lines)


def batch_process(items: List[Any], 
                 process_func: callable,
                 n_workers: int = 1,
                 progress_callback: callable = None) -> List[Any]:
    """
    批处理函数
    
    Args:
        items: 待处理项列表
        process_func: 处理函数
        n_workers: 工作进程数
        progress_callback: 进度回调函数
        
    Returns:
        处理结果列表
    """
    results = []
    
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_func, item): i for i, item in enumerate(items)}
            
            for future in as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    results.append((i, result))
                except Exception as e:
                    logger.error(f"Failed to process item {i}: {e}")
                    results.append((i, None))
                
                if progress_callback:
                    progress_callback(len(results), len(items))
    else:
        for i, item in enumerate(items):
            try:
                result = process_func(item)
                results.append((i, result))
            except Exception as e:
                logger.error(f"Failed to process item {i}: {e}")
                results.append((i, None))
            
            if progress_callback:
                progress_callback(i + 1, len(items))
    
    # 按原始顺序排序
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]
