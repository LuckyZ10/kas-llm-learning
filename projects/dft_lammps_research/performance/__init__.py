#!/usr/bin/env python3
"""
performance/__init__.py
=======================
性能优化模块 - Phase 66

提供全面的性能优化工具，包括：
- 性能分析
- 自动优化
- 智能缓存
- 并行计算
- 内存管理
- I/O加速
- Numba加速核

用法:
    from performance import Profiler, Optimizer, CacheManager
    from performance import profile_function, ParallelEngine

作者: Performance Optimization Expert
日期: 2026-03-22
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Performance Optimization Expert"

# 性能分析
from .profiler import (
    Profiler,
    profile_function,
    PerformanceContext,
    FunctionStats,
    BottleneckAnalyzer
)

# 优化器
from .optimizer import (
    Optimizer,
    OptimizationStrategy,
    JITCompiler,
    Vectorizer
)

# 缓存管理
from .cache_manager import (
    CacheManager,
    LRUCache,
    FileCache,
    MemoryMappedCache,
    CachePolicy
)

# 并行引擎
from .parallel_engine import (
    ParallelEngine,
    TaskScheduler,
    WorkerPool,
    AsyncExecutor,
    parallel_map
)

# 内存池
from .memory_pool import (
    MemoryPool,
    ArrayPool,
    ChunkedAllocator,
    MemoryMonitor
)

# I/O加速
from .io_accelerator import (
    IOAccelerator,
    MemoryMappedFile,
    AsyncReader,
    BatchIO,
    CompressionCache
)

# Numba加速核
from .numba_kernels import (
    NUMBA_AVAILABLE,
    calculate_distance_matrix,
    calculate_rdf_parallel,
    calculate_msd_parallel,
    build_neighbor_list,
    calculate_lennard_jones_energy,
    calculate_lennard_jones_forces,
)

# 便捷导入
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 性能分析
    "Profiler",
    "profile_function", 
    "PerformanceContext",
    "FunctionStats",
    "BottleneckAnalyzer",
    
    # 优化器
    "Optimizer",
    "OptimizationStrategy",
    "JITCompiler",
    "Vectorizer",
    
    # 缓存管理
    "CacheManager",
    "LRUCache",
    "FileCache",
    "MemoryMappedCache",
    "CachePolicy",
    
    # 并行引擎
    "ParallelEngine",
    "TaskScheduler",
    "WorkerPool",
    "AsyncExecutor",
    "parallel_map",
    
    # 内存池
    "MemoryPool",
    "ArrayPool",
    "ChunkedAllocator",
    "MemoryMonitor",
    
    # I/O加速
    "IOAccelerator",
    "MemoryMappedFile",
    "AsyncReader",
    "BatchIO",
    "CompressionCache",
    
    # Numba加速核
    "NUMBA_AVAILABLE",
    "calculate_distance_matrix",
    "calculate_rdf_parallel",
    "calculate_msd_parallel",
    "build_neighbor_list",
    "calculate_lennard_jones_energy",
    "calculate_lennard_jones_forces",
]

# 模块级配置
PERFORMANCE_CONFIG = {
    "numba_enabled": True,
    "parallel_enabled": True,
    "cache_enabled": True,
    "memory_pool_enabled": True,
    "async_io_enabled": True,
    "profiling_enabled": False,  # 默认关闭，避免开销
}


def configure_performance(
    numba: bool = True,
    parallel: bool = True,
    cache: bool = True,
    memory_pool: bool = True,
    async_io: bool = True,
    profiling: bool = False
) -> None:
    """
    配置性能模块全局设置
    
    Args:
        numba: 是否启用Numba加速
        parallel: 是否启用并行计算
        cache: 是否启用缓存
        memory_pool: 是否启用内存池
        async_io: 是否启用异步I/O
        profiling: 是否启用性能分析
    """
    PERFORMANCE_CONFIG["numba_enabled"] = numba
    PERFORMANCE_CONFIG["parallel_enabled"] = parallel
    PERFORMANCE_CONFIG["cache_enabled"] = cache
    PERFORMANCE_CONFIG["memory_pool_enabled"] = memory_pool
    PERFORMANCE_CONFIG["async_io_enabled"] = async_io
    PERFORMANCE_CONFIG["profiling_enabled"] = profiling
    
    # 应用配置
    if numba and not NUMBA_AVAILABLE:
        import warnings
        warnings.warn("Numba not available, JIT acceleration disabled")
        PERFORMANCE_CONFIG["numba_enabled"] = False


def get_performance_status() -> dict:
    """获取性能模块状态报告"""
    return {
        "version": __version__,
        "config": PERFORMANCE_CONFIG.copy(),
        "numba_available": NUMBA_AVAILABLE,
        "cache_stats": CacheManager.get_global_stats() if PERFORMANCE_CONFIG["cache_enabled"] else None,
        "memory_stats": MemoryMonitor.get_stats() if PERFORMANCE_CONFIG["memory_pool_enabled"] else None,
    }


# 初始化检查
def _check_dependencies():
    """检查并报告依赖状态"""
    deps = {
        "numba": NUMBA_AVAILABLE,
        "numpy": True,  # 必须依赖
        "pandas": True,
    }
    
    # 可选依赖
    try:
        import pyarrow
        deps["pyarrow"] = True
    except ImportError:
        deps["pyarrow"] = False
    
    try:
        import zstd
        deps["zstd"] = True
    except ImportError:
        deps["zstd"] = False
    
    try:
        import lz4
        deps["lz4"] = True
    except ImportError:
        deps["lz4"] = False
    
    return deps


# 模块加载时执行
_DEPENDENCIES = _check_dependencies()
