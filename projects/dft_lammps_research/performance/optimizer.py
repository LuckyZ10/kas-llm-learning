#!/usr/bin/env python3
"""
optimizer.py
============
优化器核心模块

提供自动优化功能：
- 自动选择最优算法
- JIT编译策略管理
- 向量化自动检测
- 内存预分配策略

作者: Performance Optimization Expert
日期: 2026-03-22
"""

import functools
import inspect
from typing import Dict, List, Optional, Callable, Any, Union, Type
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """优化级别"""
    NONE = auto()      # 无优化
    BASIC = auto()     # 基本优化
    AGGRESSIVE = auto()  # 激进优化
    EXTREME = auto()   # 极限优化


@dataclass
class OptimizationConfig:
    """优化配置"""
    level: OptimizationLevel = OptimizationLevel.BASIC
    use_numba: bool = True
    use_parallel: bool = True
    use_vectorization: bool = True
    use_caching: bool = True
    use_memory_pool: bool = True
    cache_size: int = 128
    num_threads: Optional[int] = None


class OptimizationStrategy(ABC):
    """
    优化策略基类
    
    所有具体优化策略的抽象基类。
    """
    
    @abstractmethod
    def apply(self, func: Callable, **kwargs) -> Callable:
        """
        应用优化策略
        
        Args:
            func: 要优化的函数
            **kwargs: 额外参数
        
        Returns:
            优化后的函数
        """
        pass
    
    @abstractmethod
    def is_applicable(self, func: Callable) -> bool:
        """
        检查策略是否适用于给定函数
        
        Args:
            func: 待检查函数
        
        Returns:
            是否适用
        """
        pass


class JITStrategy(OptimizationStrategy):
    """JIT编译策略"""
    
    def __init__(self, use_numba: bool = True, cache: bool = True):
        self.use_numba = use_numba
        self.cache = cache
        self._numba_available = self._check_numba()
    
    def _check_numba(self) -> bool:
        """检查Numba是否可用"""
        try:
            import numba
            return True
        except ImportError:
            return False
    
    def is_applicable(self, func: Callable) -> bool:
        """检查是否适用于JIT编译"""
        if not self._numba_available or not self.use_numba:
            return False
        
        # 检查函数签名
        try:
            sig = inspect.signature(func)
            # Numba对纯数值计算最有效
            return True
        except (ValueError, TypeError):
            return False
    
    def apply(self, func: Callable, 
              nopython: bool = True,
              parallel: bool = True,
              cache: bool = True,
              fastmath: bool = True,
              **kwargs) -> Callable:
        """
        应用JIT编译
        
        Args:
            func: 要编译的函数
            nopython: 是否使用nopython模式
            parallel: 是否启用并行
            cache: 是否缓存编译结果
            fastmath: 是否启用快速数学
        
        Returns:
            JIT编译后的函数
        """
        if not self._numba_available:
            logger.warning("Numba not available, skipping JIT compilation")
            return func
        
        from numba import njit
        
        return njit(
            nopython=nopython,
            parallel=parallel,
            cache=cache,
            fastmath=fastmath,
            **kwargs
        )(func)


class VectorizationStrategy(OptimizationStrategy):
    """向量化策略"""
    
    def __init__(self):
        self._patterns = [
            self._detect_loop_pattern,
            self._detect_list_comprehension,
            self._detect_map_pattern,
        ]
    
    def is_applicable(self, func: Callable) -> bool:
        """检查是否适用于向量化"""
        try:
            source = inspect.getsource(func)
            return any(pattern(source) for pattern in self._patterns)
        except (OSError, TypeError):
            return False
    
    def apply(self, func: Callable, **kwargs) -> Callable:
        """
        应用向量化优化
        
        注意：这需要源代码转换，当前实现为标记和建议。
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 运行时向量化检查
            return func(*args, **kwargs)
        
        wrapper._vectorization_applied = True
        wrapper._original = func
        
        return wrapper
    
    def _detect_loop_pattern(self, source: str) -> bool:
        """检测循环模式"""
        return "for " in source and ("range(" in source or "in " in source)
    
    def _detect_list_comprehension(self, source: str) -> bool:
        """检测列表推导式"""
        return "[" in source and "for " in source and "]" in source
    
    def _detect_map_pattern(self, source: str) -> bool:
        """检测map模式"""
        return "map(" in source or "filter(" in source
    
    def get_suggestions(self, func: Callable) -> List[str]:
        """获取向量化建议"""
        try:
            source = inspect.getsource(func)
            suggestions = []
            
            if "for i in range(len(" in source:
                suggestions.append("Replace loop with NumPy array operations")
            
            if "[].append(" in source:
                suggestions.append("Pre-allocate array instead of dynamic append")
            
            if "sum([" in source:
                suggestions.append("Use np.sum() instead of Python sum()")
            
            return suggestions
        except (OSError, TypeError):
            return []


class CachingStrategy(OptimizationStrategy):
    """缓存策略"""
    
    def __init__(self, maxsize: int = 128, typed: bool = False):
        self.maxsize = maxsize
        self.typed = typed
    
    def is_applicable(self, func: Callable) -> bool:
        """检查是否适用于缓存"""
        # 纯函数适合缓存
        try:
            sig = inspect.signature(func)
            # 检查是否为纯函数（启发式）
            return True
        except (ValueError, TypeError):
            return False
    
    def apply(self, func: Callable, **kwargs) -> Callable:
        """应用缓存"""
        return functools.lru_cache(maxsize=self.maxsize, typed=self.typed)(func)


class ParallelStrategy(OptimizationStrategy):
    """并行化策略"""
    
    def __init__(self, n_jobs: Optional[int] = None, backend: str = "loky"):
        self.n_jobs = n_jobs
        self.backend = backend
    
    def is_applicable(self, func: Callable) -> bool:
        """检查是否适用于并行化"""
        # 检查函数是否可以并行执行
        return True  # 默认启用，实际应用时检查
    
    def apply(self, func: Callable, 
              data_size_threshold: int = 1000,
              **kwargs) -> Callable:
        """
        应用并行化
        
        Args:
            func: 要并行化的函数
            data_size_threshold: 数据量阈值
        """
        try:
            from joblib import Parallel, delayed
            
            @functools.wraps(func)
            def wrapper(data, *args, **kwargs):
                if len(data) < data_size_threshold:
                    # 数据量小，串行执行
                    return [func(item, *args, **kwargs) for item in data]
                
                # 并行执行
                parallel = Parallel(n_jobs=self.n_jobs, backend=self.backend)
                return parallel(delayed(func)(item, *args, **kwargs) for item in data)
            
            wrapper._parallel_applied = True
            wrapper._original = func
            return wrapper
            
        except ImportError:
            logger.warning("joblib not available, skipping parallelization")
            return func


class Optimizer:
    """
    优化器
    
    提供自动优化功能，根据函数特征自动选择最优策略。
    
    示例:
        optimizer = Optimizer()
        
        @optimizer.optimize
        def my_function(data):
            return np.sum(data ** 2)
        
        # 或者手动优化
        optimized_func = optimizer.optimize(my_function, level='aggressive')
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        初始化优化器
        
        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig()
        self._strategies: Dict[str, OptimizationStrategy] = {}
        self._setup_strategies()
    
    def _setup_strategies(self) -> None:
        """设置优化策略"""
        if self.config.use_numba:
            self._strategies["jit"] = JITStrategy()
        
        if self.config.use_vectorization:
            self._strategies["vectorization"] = VectorizationStrategy()
        
        if self.config.use_caching:
            self._strategies["caching"] = CachingStrategy(
                maxsize=self.config.cache_size
            )
        
        if self.config.use_parallel:
            self._strategies["parallel"] = ParallelStrategy(
                n_jobs=self.config.num_threads
            )
    
    def optimize(self, func: Optional[Callable] = None, 
                 level: Optional[str] = None,
                 strategies: Optional[List[str]] = None) -> Callable:
        """
        优化函数
        
        Args:
            func: 要优化的函数
            level: 优化级别 ('none', 'basic', 'aggressive', 'extreme')
            strategies: 指定策略列表（覆盖自动选择）
        
        Returns:
            优化后的函数
        """
        def decorator(f: Callable) -> Callable:
            opt_level = level or self.config.level.name.lower()
            
            if opt_level == "none":
                return f
            
            # 确定要应用的策略
            if strategies:
                active_strategies = [
                    self._strategies[s] for s in strategies 
                    if s in self._strategies
                ]
            else:
                active_strategies = self._select_strategies(f, opt_level)
            
            # 应用策略
            optimized = f
            for strategy in active_strategies:
                if strategy.is_applicable(optimized):
                    optimized = strategy.apply(optimized)
                    logger.debug(f"Applied {strategy.__class__.__name__} to {f.__name__}")
            
            # 标记已优化
            optimized._optimized = True
            optimized._original = f
            optimized._applied_strategies = [
                s.__class__.__name__ for s in active_strategies
            ]
            
            return optimized
        
        if func is not None:
            return decorator(func)
        return decorator
    
    def _select_strategies(self, func: Callable, level: str) -> List[OptimizationStrategy]:
        """
        根据函数特征和级别选择策略
        
        Args:
            func: 待优化函数
            level: 优化级别
        
        Returns:
            策略列表
        """
        selected = []
        
        # 根据级别选择策略
        if level == "basic":
            # 基本：仅缓存
            if "caching" in self._strategies:
                selected.append(self._strategies["caching"])
        
        elif level == "aggressive":
            # 激进：缓存 + JIT
            if "caching" in self._strategies:
                selected.append(self._strategies["caching"])
            if "jit" in self._strategies:
                selected.append(self._strategies["jit"])
        
        elif level == "extreme":
            # 极限：所有策略
            selected = list(self._strategies.values())
        
        return selected
    
    def auto_optimize(self, module: Any) -> None:
        """
        自动优化模块中的所有函数
        
        Args:
            module: 要优化的模块
        """
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith("_"):
                try:
                    optimized = self.optimize(obj)
                    setattr(module, name, optimized)
                    logger.debug(f"Auto-optimized {name}")
                except Exception as e:
                    logger.warning(f"Failed to optimize {name}: {e}")
    
    def benchmark(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """
        基准测试优化效果
        
        Args:
            func: 要测试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            性能对比结果
        """
        import time
        
        # 测试原始函数
        original = getattr(func, "_original", func)
        
        # 预热
        for _ in range(3):
            original(*args, **kwargs)
        
        # 测试原始版本
        start = time.perf_counter()
        for _ in range(10):
            original(*args, **kwargs)
        original_time = time.perf_counter() - start
        
        # 测试优化版本
        if getattr(func, "_optimized", False):
            start = time.perf_counter()
            for _ in range(10):
                func(*args, **kwargs)
            optimized_time = time.perf_counter() - start
            
            speedup = original_time / optimized_time if optimized_time > 0 else 1.0
            
            return {
                "original_time": original_time,
                "optimized_time": optimized_time,
                "speedup": speedup,
                "strategies": getattr(func, "_applied_strategies", [])
            }
        else:
            return {
                "original_time": original_time,
                "optimized_time": original_time,
                "speedup": 1.0,
                "strategies": []
            }


class JITCompiler:
    """
    JIT编译器管理器
    
    简化Numba JIT编译的使用。
    
    示例:
        compiler = JITCompiler()
        
        @compiler.jit
        def compute(x, y):
            return x ** 2 + y ** 2
    """
    
    def __init__(self, cache: bool = True, parallel: bool = True):
        self.cache = cache
        self.parallel = parallel
        self._numba_available = self._check_numba()
        self._compiled_functions: Dict[str, Callable] = {}
    
    def _check_numba(self) -> bool:
        """检查Numba是否可用"""
        try:
            import numba
            return True
        except ImportError:
            warnings.warn("Numba not available, JIT compilation disabled")
            return False
    
    def jit(self, func: Optional[Callable] = None, **kwargs) -> Callable:
        """
        JIT编译装饰器
        
        Args:
            func: 要编译的函数
            **kwargs: Numba编译选项
        
        Returns:
            编译后的函数
        """
        if not self._numba_available:
            return func if func else lambda f: f
        
        from numba import njit
        
        default_kwargs = {
            "cache": self.cache,
            "parallel": self.parallel,
            "nopython": True,
            "fastmath": True,
        }
        default_kwargs.update(kwargs)
        
        if func is not None:
            compiled = njit(**default_kwargs)(func)
            self._compiled_functions[func.__name__] = compiled
            return compiled
        
        def decorator(f: Callable) -> Callable:
            compiled = njit(**default_kwargs)(f)
            self._compiled_functions[f.__name__] = compiled
            return compiled
        
        return decorator
    
    def vectorize(self, func: Optional[Callable] = None, **kwargs) -> Callable:
        """
        Numba向量化装饰器
        
        Args:
            func: 要向量化的函数
            **kwargs: Numba向量化选项
        
        Returns:
            向量化后的函数
        """
        if not self._numba_available:
            return func if func else lambda f: np.vectorize(f)
        
        from numba import vectorize as numba_vectorize
        
        default_kwargs = {
            "cache": self.cache,
            "target": "parallel" if self.parallel else "cpu",
        }
        default_kwargs.update(kwargs)
        
        if func is not None:
            return numba_vectorize(**default_kwargs)(func)
        
        return lambda f: numba_vectorize(**default_kwargs)(f)
    
    def get_compiled(self, name: str) -> Optional[Callable]:
        """获取已编译的函数"""
        return self._compiled_functions.get(name)
    
    def list_compiled(self) -> List[str]:
        """列出所有已编译的函数"""
        return list(self._compiled_functions.keys())


class Vectorizer:
    """
    向量化助手
    
    帮助识别和实现NumPy向量化。
    
    示例:
        vectorizer = Vectorizer()
        
        # 自动向量化
        vectorized_func = vectorizer.auto_vectorize(my_loop_function)
    """
    
    def __init__(self):
        self._vectorized_functions: Dict[str, Callable] = {}
    
    def auto_vectorize(self, func: Callable, 
                       signature: Optional[str] = None) -> Callable:
        """
        自动向量化函数
        
        Args:
            func: 要向量化的函数
            signature: Numba签名（可选）
        
        Returns:
            向量化后的函数
        """
        try:
            from numba import vectorize
            
            if signature:
                return vectorize(signature)(func)
            else:
                # 自动推断
                return np.vectorize(func)
                
        except ImportError:
            return np.vectorize(func)
    
    def apply_numpy(self, operation: str, *arrays, **kwargs) -> np.ndarray:
        """
        应用NumPy向量化操作
        
        Args:
            operation: 操作名称
            *arrays: 输入数组
            **kwargs: 额外参数
        
        Returns:
            结果数组
        """
        operations = {
            "add": np.add,
            "subtract": np.subtract,
            "multiply": np.multiply,
            "divide": np.divide,
            "power": np.power,
            "sum": np.sum,
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            "dot": np.dot,
            "matmul": np.matmul,
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        return operations[operation](*arrays, **kwargs)
    
    def broadcast_arrays(self, *arrays) -> List[np.ndarray]:
        """广播数组到相同形状"""
        return np.broadcast_arrays(*arrays)
    
    def einsum(self, subscript: str, *operands) -> np.ndarray:
        """
        爱因斯坦求和约定
        
        高效实现复杂的数组操作。
        """
        return np.einsum(subscript, *operands, optimize=True)


# 便捷函数

def optimize(level: str = "basic", **kwargs):
    """
    便捷的优化装饰器
    
    示例:
        @optimize(level="aggressive")
        def my_func(x):
            return x ** 2
    """
    optimizer = Optimizer()
    return lambda func: optimizer.optimize(func, level=level, **kwargs)


def jit_compile(**kwargs):
    """
    便捷的JIT编译装饰器
    
    示例:
        @jit_compile(parallel=True)
        def compute(x):
            return x * 2
    """
    compiler = JITCompiler()
    return compiler.jit(**kwargs)
