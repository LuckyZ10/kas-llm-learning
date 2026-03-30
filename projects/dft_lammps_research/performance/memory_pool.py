#!/usr/bin/env python3
"""
memory_pool.py
==============
内存池管理模块

提供高效的内存管理功能：
- NumPy数组预分配池
- 大对象分块管理
- 内存使用监控
- GC优化策略

作者: Performance Optimization Expert
日期: 2026-03-22
"""

import gc
import sys
import threading
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArraySpec:
    """数组规格"""
    shape: Tuple[int, ...]
    dtype: np.dtype
    
    @property
    def size_bytes(self) -> int:
        """计算字节大小"""
        return np.prod(self.shape) * np.dtype(self.dtype).itemsize
    
    def __hash__(self) -> int:
        return hash((self.shape, str(self.dtype)))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ArraySpec):
            return False
        return self.shape == other.shape and self.dtype == other.dtype


@dataclass
class PooledArray:
    """池化数组"""
    array: np.ndarray
    spec: ArraySpec
    in_use: bool = False
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


class ArrayPool:
    """
    NumPy数组池
    
    预分配和复用NumPy数组，减少内存分配开销。
    
    示例:
        pool = ArrayPool()
        
        # 从池中获取数组
        with pool.acquire((1000, 1000), dtype=np.float64) as arr:
            # 使用数组
            arr[:] = compute_result
        # 自动归还到池中
        
        # 预分配常用规格
        pool.preallocate([
            ((1000, 3), np.float64),
            ((100, 100, 100), np.float32),
        ])
    """
    
    def __init__(self, 
                 max_pools_per_spec: int = 10,
                 max_memory_mb: Optional[float] = None):
        """
        初始化数组池
        
        Args:
            max_pools_per_spec: 每种规格的最大池数量
            max_memory_mb: 最大内存限制（MB）
        """
        self.max_pools_per_spec = max_pools_per_spec
        self.max_memory = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        
        self._pools: Dict[ArraySpec, deque] = defaultdict(deque)
        self._in_use: Dict[int, PooledArray] = {}
        self._total_memory = 0
        self._lock = threading.RLock()
    
    def acquire(self, 
                shape: Tuple[int, ...], 
                dtype: Union[str, np.dtype] = np.float64,
                zero: bool = False) -> 'ArrayPoolContext':
        """
        获取数组（上下文管理器）
        
        Args:
            shape: 数组形状
            dtype: 数据类型
            zero: 是否初始化为零
        
        Returns:
            数组上下文管理器
        """
        return ArrayPoolContext(self, shape, dtype, zero)
    
    def get(self, 
            shape: Tuple[int, ...], 
            dtype: Union[str, np.dtype] = np.float64,
            zero: bool = False) -> np.ndarray:
        """
        获取数组（手动管理）
        
        Args:
            shape: 数组形状
            dtype: 数据类型
            zero: 是否初始化为零
        
        Returns:
            数组
        """
        spec = ArraySpec(shape, np.dtype(dtype))
        
        with self._lock:
            # 尝试从池中获取
            pool = self._pools[spec]
            for pa in pool:
                if not pa.in_use:
                    pa.in_use = True
                    pa.last_used = time.time()
                    self._in_use[id(pa.array)] = pa
                    
                    if zero:
                        pa.array.fill(0)
                    
                    return pa.array
            
            # 池中没有可用数组，创建新的
            arr = np.empty(shape, dtype=dtype)
            pa = PooledArray(array=arr, spec=spec, in_use=True)
            self._in_use[id(arr)] = pa
            
            # 检查内存限制
            if self.max_memory:
                arr_size = arr.nbytes
                while self._total_memory + arr_size > self.max_memory:
                    if not self._evict_oldest():
                        break
                self._total_memory += arr_size
            
            return arr
    
    def release(self, arr: np.ndarray) -> None:
        """
        释放数组回池中
        
        Args:
            arr: 要释放的数组
        """
        arr_id = id(arr)
        
        with self._lock:
            if arr_id not in self._in_use:
                # 不是池管理的数组，忽略
                return
            
            pa = self._in_use.pop(arr_id)
            pa.in_use = False
            pa.last_used = time.time()
            
            # 归还到池中
            pool = self._pools[pa.spec]
            if len(pool) < self.max_pools_per_spec:
                pool.append(pa)
            else:
                # 池已满，释放内存
                self._total_memory -= pa.spec.size_bytes
    
    def preallocate(self, specs: List[Tuple[Tuple[int, ...], Union[str, np.dtype]]]) -> None:
        """
        预分配数组
        
        Args:
            specs: 规格列表 [(shape, dtype), ...]
        """
        for shape, dtype in specs:
            spec = ArraySpec(shape, np.dtype(dtype))
            
            with self._lock:
                pool = self._pools[spec]
                current_count = len(pool)
                
                for _ in range(self.max_pools_per_spec - current_count):
                    arr = np.empty(shape, dtype=dtype)
                    pa = PooledArray(array=arr, spec=spec, in_use=False)
                    pool.append(pa)
                    self._total_memory += spec.size_bytes
    
    def clear(self) -> None:
        """清空所有池"""
        with self._lock:
            self._pools.clear()
            self._in_use.clear()
            self._total_memory = 0
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息"""
        with self._lock:
            stats = {
                "total_memory_mb": self._total_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory / (1024 * 1024) if self.max_memory else None,
                "spec_count": len(self._pools),
                "in_use_count": len(self._in_use),
                "specs": {}
            }
            
            for spec, pool in self._pools.items():
                available = sum(1 for pa in pool if not pa.in_use)
                stats["specs"][f"{spec.shape}-{spec.dtype}"] = {
                    "total": len(pool),
                    "available": available,
                    "memory_mb": spec.size_bytes * len(pool) / (1024 * 1024)
                }
            
            return stats
    
    def _evict_oldest(self) -> bool:
        """淘汰最老的未使用数组"""
        oldest = None
        oldest_time = float('inf')
        
        for pool in self._pools.values():
            for pa in pool:
                if not pa.in_use and pa.last_used < oldest_time:
                    oldest = pa
                    oldest_time = pa.last_used
        
        if oldest:
            pool = self._pools[oldest.spec]
            pool.remove(oldest)
            self._total_memory -= oldest.spec.size_bytes
            return True
        
        return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()


class ArrayPoolContext:
    """数组池上下文管理器"""
    
    def __init__(self, pool: ArrayPool, 
                 shape: Tuple[int, ...], 
                 dtype: Union[str, np.dtype],
                 zero: bool = False):
        self.pool = pool
        self.shape = shape
        self.dtype = dtype
        self.zero = zero
        self.array: Optional[np.ndarray] = None
    
    def __enter__(self) -> np.ndarray:
        self.array = self.pool.get(self.shape, self.dtype, self.zero)
        return self.array
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.array is not None:
            self.pool.release(self.array)


class MemoryPool:
    """
    通用内存池
    
    管理通用内存分配，支持对象复用。
    
    示例:
        pool = MemoryPool()
        
        # 获取对象
        obj = pool.acquire(dict)
        # 使用...
        pool.release(obj)
    """
    
    def __init__(self, max_size: int = 100):
        """
        初始化内存池
        
        Args:
            max_size: 每种类型最大池大小
        """
        self.max_size = max_size
        self._pools: Dict[type, deque] = defaultdict(deque)
        self._lock = threading.RLock()
    
    def acquire(self, obj_type: type, *args, **kwargs) -> Any:
        """
        获取对象
        
        Args:
            obj_type: 对象类型
            *args: 构造参数
            **kwargs: 构造关键字参数
        
        Returns:
            对象实例
        """
        with self._lock:
            pool = self._pools[obj_type]
            
            if pool:
                obj = pool.popleft()
                # 重置对象状态（如果可能）
                if hasattr(obj, 'clear'):
                    obj.clear()
                return obj
        
        # 池为空，创建新对象
        return obj_type(*args, **kwargs)
    
    def release(self, obj: Any) -> None:
        """
        释放对象回池中
        
        Args:
            obj: 要释放的对象
        """
        obj_type = type(obj)
        
        with self._lock:
            pool = self._pools[obj_type]
            if len(pool) < self.max_size:
                pool.append(obj)
    
    def clear(self) -> None:
        """清空所有池"""
        with self._lock:
            self._pools.clear()


class ChunkedAllocator:
    """
    分块分配器
    
    将大对象分块管理，提高内存利用率和缓存效率。
    
    示例:
        allocator = ChunkedAllocator(chunk_size=1000)
        
        # 分块处理大数据
        for chunk in allocator.iter_chunks(large_array):
            process(chunk)
    """
    
    def __init__(self, chunk_size: int = 10000):
        """
        初始化分块分配器
        
        Args:
            chunk_size: 默认分块大小
        """
        self.chunk_size = chunk_size
    
    def iter_chunks(self, 
                    data: Union[np.ndarray, List],
                    chunk_size: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        迭代分块
        
        Args:
            data: 数据数组或列表
            chunk_size: 分块大小（None使用默认值）
        
        Yields:
            数据块
        """
        size = chunk_size or self.chunk_size
        n = len(data)
        
        for i in range(0, n, size):
            yield data[i:i+size]
    
    def create_chunks(self,
                      shape: Tuple[int, ...],
                      dtype: np.dtype,
                      chunk_axis: int = 0) -> List[np.ndarray]:
        """
        创建分块数组
        
        Args:
            shape: 完整形状
            dtype: 数据类型
            chunk_axis: 分块轴
        
        Returns:
            块列表
        """
        total_size = shape[chunk_axis]
        chunk_size = self.chunk_size
        
        chunks = []
        for start in range(0, total_size, chunk_size):
            end = min(start + chunk_size, total_size)
            chunk_shape = list(shape)
            chunk_shape[chunk_axis] = end - start
            chunk = np.empty(chunk_shape, dtype=dtype)
            chunks.append(chunk)
        
        return chunks
    
    def process_chunks(self,
                       data: np.ndarray,
                       func: Callable[[np.ndarray], np.ndarray],
                       chunk_axis: int = 0,
                       out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        分块处理数组
        
        Args:
            data: 输入数组
            func: 处理函数
            chunk_axis: 分块轴
            out: 输出数组（可选）
        
        Returns:
            处理后的数组
        """
        if out is None:
            out = np.empty_like(data)
        
        total_size = data.shape[chunk_axis]
        
        for start in range(0, total_size, self.chunk_size):
            end = min(start + self.chunk_size, total_size)
            
            # 提取切片
            in_slice = [slice(None)] * data.ndim
            in_slice[chunk_axis] = slice(start, end)
            
            out_slice = [slice(None)] * out.ndim
            out_slice[chunk_axis] = slice(start, end)
            
            # 处理块
            chunk_result = func(data[tuple(in_slice)])
            out[tuple(out_slice)] = chunk_result
        
        return out


class MemoryMonitor:
    """
    内存监控器
    
    监控内存使用情况，提供预警和自动优化。
    
    示例:
        monitor = MemoryMonitor(warning_threshold=0.8)
        monitor.start_monitoring()
        
        # 检查内存状态
        if monitor.is_memory_critical():
            monitor.emergency_cleanup()
    """
    
    _instance: Optional['MemoryMonitor'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, 
                 warning_threshold: float = 0.8,
                 critical_threshold: float = 0.9):
        """
        初始化内存监控器
        
        Args:
            warning_threshold: 警告阈值（内存使用比例）
            critical_threshold: 临界阈值
        """
        if self._initialized:
            return
        
        self._initialized = True
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self._monitoring = False
        self._history: deque = deque(maxlen=100)
        self._lock = threading.Lock()
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        获取内存信息
        
        Returns:
            内存信息字典
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "used_gb": mem.used / (1024**3),
                "percent": mem.percent / 100,
                "cached_gb": getattr(mem, 'cached', 0) / (1024**3),
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def is_memory_critical(self) -> bool:
        """检查内存是否临界"""
        info = self.get_memory_info()
        if "percent" in info:
            return info["percent"] > self.critical_threshold
        return False
    
    def is_memory_warning(self) -> bool:
        """检查内存是否警告"""
        info = self.get_memory_info()
        if "percent" in info:
            return info["percent"] > self.warning_threshold
        return False
    
    def emergency_cleanup(self) -> None:
        """紧急清理内存"""
        logger.warning("Emergency memory cleanup initiated")
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理NumPy缓存
        # np._distributor_init._clear_cache_if_memory_low()
        
        logger.info("Emergency cleanup completed")
    
    def optimize_gc(self) -> None:
        """优化垃圾回收设置"""
        # 降低GC频率以提高性能
        gc.set_threshold(700, 10, 10)
        logger.debug("GC thresholds optimized for performance")
    
    def get_object_count(self) -> Dict[str, int]:
        """获取对象计数统计"""
        counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            counts[obj_type] = counts.get(obj_type, 0) + 1
        
        # 返回前20
        return dict(sorted(counts.items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:20])
    
    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """获取内存统计"""
        monitor = MemoryMonitor()
        return {
            "memory_info": monitor.get_memory_info(),
            "is_critical": monitor.is_memory_critical(),
            "is_warning": monitor.is_memory_warning(),
            "gc_count": gc.get_count(),
        }
    
    def get_large_objects(self, min_size_mb: float = 10) -> List[Tuple[str, float]]:
        """
        获取大对象列表
        
        Args:
            min_size_mb: 最小大小（MB）
        
        Returns:
            (类型, 大小MB) 列表
        """
        large_objects = []
        
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj) / (1024 * 1024)  # MB
                if size > min_size_mb:
                    large_objects.append((type(obj).__name__, size))
            except:
                pass
        
        return sorted(large_objects, key=lambda x: x[1], reverse=True)


# 便捷函数

def get_array_pool() -> ArrayPool:
    """获取全局数组池"""
    return ArrayPool()


def optimize_memory() -> None:
    """优化内存使用"""
    gc.collect()
    
    # 监控器检查
    monitor = MemoryMonitor()
    if monitor.is_memory_critical():
        monitor.emergency_cleanup()
