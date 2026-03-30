#!/usr/bin/env python3
"""
cache_manager.py
================
智能缓存管理模块

提供多种缓存策略：
- LRU-K缓存
- 文件级缓存
- 内存映射缓存
- 缓存一致性管理

作者: Performance Optimization Expert
日期: 2026-03-22
"""

import os
import pickle
import hashlib
import json
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum, auto
import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """缓存策略"""
    LRU = "lru"          # 最近最少使用
    LFU = "lfu"          # 最不经常使用
    FIFO = "fifo"        # 先进先出
    RANDOM = "random"    # 随机淘汰
    TTL = "ttl"          # 生存时间


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    size: int
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # 生存时间（秒）
    
    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """条目年龄（秒）"""
        return time.time() - self.created_at
    
    def touch(self) -> None:
        """更新访问时间"""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """缓存后端抽象基类"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """获取所有键"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """获取缓存大小"""
        pass


class LRUCache(CacheBackend):
    """
    LRU缓存实现
    
    基于OrderedDict实现，保证O(1)的访问和更新。
    
    示例:
        cache = LRUCache(maxsize=100)
        cache.set("key", value)
        value = cache.get("key")
    """
    
    def __init__(self, maxsize: int = 128, max_memory_mb: Optional[float] = None):
        """
        初始化LRU缓存
        
        Args:
            maxsize: 最大条目数
            max_memory_mb: 最大内存限制（MB）
        """
        self.maxsize = maxsize
        self.max_memory = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_memory = 0
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
        
        Returns:
            缓存值或None
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # 检查过期
            if entry.is_expired:
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # 更新访问信息
            entry.touch()
            self._cache.move_to_end(key)
            self._hits += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒）
        """
        # 估算大小
        size = self._estimate_size(value)
        
        with self._lock:
            # 如果键已存在，更新
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_memory -= old_entry.size
            
            # 检查是否需要淘汰
            while (len(self._cache) >= self.maxsize or 
                   (self.max_memory and self._current_memory + size > self.max_memory)):
                if not self._cache:
                    break
                self._evict_lru()
            
            # 添加新条目
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                ttl=ttl
            )
            self._cache[key] = entry
            self._current_memory += size
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            self._hits = 0
            self._misses = 0
    
    def keys(self) -> List[str]:
        """获取所有键"""
        with self._lock:
            return list(self._cache.keys())
    
    def size(self) -> int:
        """获取缓存条目数"""
        with self._lock:
            return len(self._cache)
    
    @property
    def memory_usage(self) -> int:
        """获取内存使用量（字节）"""
        with self._lock:
            return self._current_memory
    
    @property
    def hit_rate(self) -> float:
        """获取命中率"""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "memory_usage_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory / (1024 * 1024) if self.max_memory else None,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
            }
    
    def _evict_lru(self) -> None:
        """淘汰最久未使用的条目"""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_memory -= entry.size
            logger.debug(f"Evicted LRU entry: {key}")
    
    def _remove_entry(self, key: str) -> None:
        """移除条目"""
        entry = self._cache.pop(key)
        self._current_memory -= entry.size
    
    def _estimate_size(self, value: Any) -> int:
        """估算值的大小"""
        if isinstance(value, np.ndarray):
            return value.nbytes
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # 默认估算


class FileCache(CacheBackend):
    """
    文件级缓存
    
    将缓存持久化到文件系统，支持多种序列化格式。
    
    示例:
        cache = FileCache(cache_dir="/tmp/cache", format="pickle")
        cache.set("data", large_array)
        # 程序重启后
        data = cache.get("data")  # 仍然可用
    """
    
    def __init__(self, cache_dir: Union[str, Path], 
                 format: str = "pickle",
                 max_size_gb: Optional[float] = None):
        """
        初始化文件缓存
        
        Args:
            cache_dir: 缓存目录
            format: 序列化格式 ("pickle", "npz", "json")
            max_size_gb: 最大缓存大小（GB）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.max_size = max_size_gb * 1024 * 1024 * 1024 if max_size_gb else None
        self._index_file = self.cache_dir / ".index.json"
        self._index: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        
        self._load_index()
    
    def _load_index(self) -> None:
        """加载索引"""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
    
    def _save_index(self) -> None:
        """保存索引"""
        with self._lock:
            try:
                with open(self._index_file, 'w') as f:
                    json.dump(self._index, f)
            except Exception as e:
                logger.warning(f"Failed to save cache index: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用哈希避免文件名问题
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.{self.format}"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._index:
                return None
            
            entry = self._index[key]
            
            # 检查过期
            if entry.get("ttl"):
                if time.time() - entry["created_at"] > entry["ttl"]:
                    self.delete(key)
                    return None
            
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                del self._index[key]
                self._save_index()
                return None
            
            try:
                value = self._load_value(cache_path)
                entry["access_count"] = entry.get("access_count", 0) + 1
                entry["last_accessed"] = time.time()
                self._save_index()
                return value
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """设置缓存值"""
        cache_path = self._get_cache_path(key)
        
        # 检查大小限制
        if self.max_size:
            current_size = sum(
                self._get_cache_path(k).stat().st_size 
                for k in self._index.keys()
                if self._get_cache_path(k).exists()
            )
            
            # 估算新条目大小
            temp_path = self.cache_dir / ".temp_cache"
            self._save_value(value, temp_path)
            new_size = temp_path.stat().st_size
            temp_path.unlink()
            
            # 清理空间
            while current_size + new_size > self.max_size and self._index:
                oldest_key = min(
                    self._index.keys(),
                    key=lambda k: self._index[k].get("last_accessed", 0)
                )
                self.delete(oldest_key)
                current_size = sum(
                    self._get_cache_path(k).stat().st_size 
                    for k in self._index.keys()
                    if self._get_cache_path(k).exists()
                )
        
        # 保存值
        self._save_value(value, cache_path)
        
        # 更新索引
        with self._lock:
            self._index[key] = {
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 0,
                "ttl": ttl,
                "size": cache_path.stat().st_size,
            }
            self._save_index()
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key not in self._index:
                return False
            
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            
            del self._index[key]
            self._save_index()
            return True
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            for key in list(self._index.keys()):
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
            
            self._index.clear()
            self._save_index()
    
    def keys(self) -> List[str]:
        """获取所有键"""
        with self._lock:
            return list(self._index.keys())
    
    def size(self) -> int:
        """获取缓存条目数"""
        with self._lock:
            return len(self._index)
    
    def _load_value(self, path: Path) -> Any:
        """加载值"""
        if self.format == "pickle":
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif self.format == "npz":
            return dict(np.load(path))
        elif self.format == "json":
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unknown format: {self.format}")
    
    def _save_value(self, value: Any, path: Path) -> None:
        """保存值"""
        if self.format == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(value, f)
        elif self.format == "npz":
            if isinstance(value, dict):
                np.savez_compressed(path, **value)
            else:
                np.savez_compressed(path, data=value)
        elif self.format == "json":
            with open(path, 'w') as f:
                json.dump(value, f)
        else:
            raise ValueError(f"Unknown format: {self.format}")


class MemoryMappedCache(CacheBackend):
    """
    内存映射缓存
    
    使用内存映射文件处理大数组，避免内存复制。
    
    示例:
        cache = MemoryMappedCache(cache_dir="/tmp/mmap_cache")
        cache.set("large_array", np.random.rand(1000000))
        # 内存高效访问
        arr = cache.get("large_array")
    """
    
    def __init__(self, cache_dir: Union[str, Path]):
        """
        初始化内存映射缓存
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mmaps: Dict[str, np.memmap] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        获取内存映射数组
        
        Returns:
            内存映射数组（只读）
        """
        with self._lock:
            if key in self._mmaps:
                return self._mmaps[key]
            
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                return None
            
            try:
                # 读取元数据
                meta_path = cache_path.with_suffix('.meta.json')
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                # 创建内存映射
                mmap = np.memmap(
                    cache_path,
                    dtype=meta['dtype'],
                    mode='r',
                    shape=tuple(meta['shape'])
                )
                self._mmaps[key] = mmap
                return mmap
                
            except Exception as e:
                logger.warning(f"Failed to mmap {key}: {e}")
                return None
    
    def set(self, key: str, value: np.ndarray, **kwargs) -> None:
        """
        设置内存映射数组
        
        Args:
            key: 缓存键
            value: NumPy数组
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("MemoryMappedCache only supports numpy arrays")
        
        cache_path = self._get_cache_path(key)
        
        # 保存数组
        mmap = np.memmap(cache_path, dtype=value.dtype, mode='w+', shape=value.shape)
        mmap[:] = value[:]
        mmap.flush()
        
        # 保存元数据
        meta = {
            'dtype': str(value.dtype),
            'shape': list(value.shape),
            'created_at': time.time(),
        }
        meta_path = cache_path.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        
        # 添加到映射表
        with self._lock:
            self._mmaps[key] = mmap
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._mmaps:
                del self._mmaps[key]
            
            cache_path = self._get_cache_path(key)
            meta_path = cache_path.with_suffix('.meta.json')
            
            deleted = False
            if cache_path.exists():
                cache_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
            
            return deleted
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._mmaps.clear()
            
            for f in self.cache_dir.glob("*.mmap"):
                f.unlink()
            for f in self.cache_dir.glob("*.meta.json"):
                f.unlink()
    
    def keys(self) -> List[str]:
        """获取所有键"""
        keys = []
        for f in self.cache_dir.glob("*.meta.json"):
            keys.append(f.stem.replace('.meta', ''))
        return keys
    
    def size(self) -> int:
        """获取缓存条目数"""
        return len(list(self.cache_dir.glob("*.mmap")))
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存路径"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.mmap"


class CacheManager:
    """
    缓存管理器
    
    统一管理多级缓存，提供统一的接口。
    
    示例:
        manager = CacheManager()
        
        # 多级缓存自动管理
        manager.set("key", value, ttl=3600)
        value = manager.get("key")
        
        # 使用装饰器
        @manager.cached(ttl=3600)
        def expensive_function(x):
            return x ** 2
    """
    
    _instance: Optional['CacheManager'] = None
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
                 memory_cache_size: int = 1000,
                 file_cache_dir: Optional[str] = None,
                 mmap_cache_dir: Optional[str] = None):
        """
        初始化缓存管理器
        
        Args:
            memory_cache_size: 内存缓存大小
            file_cache_dir: 文件缓存目录
            mmap_cache_dir: 内存映射缓存目录
        """
        if self._initialized:
            return
        
        self._initialized = True
        
        # 多级缓存
        self._memory_cache = LRUCache(maxsize=memory_cache_size)
        
        self._file_cache = None
        if file_cache_dir:
            self._file_cache = FileCache(file_cache_dir)
        
        self._mmap_cache = None
        if mmap_cache_dir:
            self._mmap_cache = MemoryMappedCache(mmap_cache_dir)
        
        # 统计
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        按内存 -> 文件 -> 内存映射的顺序查找。
        
        Args:
            key: 缓存键
            default: 默认值
        
        Returns:
            缓存值或默认值
        """
        # L1: 内存缓存
        value = self._memory_cache.get(key)
        if value is not None:
            with self._lock:
                self._hits += 1
            return value
        
        # L2: 文件缓存
        if self._file_cache:
            value = self._file_cache.get(key)
            if value is not None:
                # 回填内存缓存
                self._memory_cache.set(key, value)
                with self._lock:
                    self._hits += 1
                return value
        
        # L3: 内存映射缓存
        if self._mmap_cache:
            value = self._mmap_cache.get(key)
            if value is not None:
                with self._lock:
                    self._hits += 1
                return value
        
        with self._lock:
            self._misses += 1
        return default
    
    def set(self, key: str, value: Any, 
            ttl: Optional[float] = None,
            use_file: bool = False,
            use_mmap: bool = False) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间
            use_file: 是否同时存入文件缓存
            use_mmap: 是否使用内存映射（仅支持数组）
        """
        # L1: 内存缓存
        self._memory_cache.set(key, value, ttl=ttl)
        
        # L2: 文件缓存
        if use_file and self._file_cache:
            self._file_cache.set(key, value, ttl=ttl)
        
        # L3: 内存映射缓存
        if use_mmap and self._mmap_cache and isinstance(value, np.ndarray):
            self._mmap_cache.set(key, value)
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        deleted = False
        
        if self._memory_cache.delete(key):
            deleted = True
        
        if self._file_cache and self._file_cache.delete(key):
            deleted = True
        
        if self._mmap_cache and self._mmap_cache.delete(key):
            deleted = True
        
        return deleted
    
    def clear(self) -> None:
        """清空所有缓存"""
        self._memory_cache.clear()
        
        if self._file_cache:
            self._file_cache.clear()
        
        if self._mmap_cache:
            self._mmap_cache.clear()
    
    def cached(self, ttl: Optional[float] = None, 
               key_func: Optional[Callable] = None):
        """
        缓存装饰器
        
        Args:
            ttl: 生存时间
            key_func: 自定义键生成函数
        
        示例:
            @manager.cached(ttl=3600)
            def expensive_calculation(x, y):
                return x ** y
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func, args, kwargs)
                
                # 尝试从缓存获取
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 存入缓存
                self.set(cache_key, result, ttl=ttl)
                
                return result
            
            # 添加缓存控制方法
            wrapper.cache_clear = lambda: self._clear_function_cache(func)
            wrapper.cache_info = lambda: self._get_function_cache_info(func)
            
            return wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "memory_cache": self._memory_cache.get_stats(),
                "file_cache_size": self._file_cache.size() if self._file_cache else 0,
                "mmap_cache_size": self._mmap_cache.size() if self._mmap_cache else 0,
            }
    
    @staticmethod
    def get_global_stats() -> Dict[str, Any]:
        """获取全局缓存统计"""
        manager = CacheManager()
        return manager.get_stats()
    
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_parts = [
            func.__module__,
            func.__name__,
            str(args),
            str(sorted(kwargs.items()))
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _clear_function_cache(self, func: Callable) -> None:
        """清除函数的缓存"""
        # 这里简化处理，实际应该跟踪每个函数的缓存键
        pass
    
    def _get_function_cache_info(self, func: Callable) -> Dict:
        """获取函数缓存信息"""
        return self.get_stats()
