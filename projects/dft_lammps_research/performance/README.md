# Performance Module

DFT+LAMMPS研究平台性能优化模块 - Phase 66

## 概述

本模块提供全面的性能优化工具，包括性能分析、自动优化、智能缓存、并行计算、内存管理和I/O加速。

## 模块结构

```
performance/
├── __init__.py           # 模块入口和配置
├── profiler.py           # 性能分析器
├── optimizer.py          # 自动优化器
├── cache_manager.py      # 三级缓存系统
├── parallel_engine.py    # 并行计算引擎
├── memory_pool.py        # 内存池管理
├── io_accelerator.py     # I/O加速
├── numba_kernels.py      # Numba加速核
└── tests/                # 测试套件
    ├── test_profiler.py
    ├── test_optimizer.py
    └── benchmark_suite.py
```

## 快速开始

```python
from performance import Profiler, CacheManager, parallel_map
from performance.numba_kernels import calculate_distance_matrix

# 1. 性能分析
profiler = Profiler()

@profiler.profile
def my_function(x):
    return x ** 2

# 2. 智能缓存
cache = CacheManager()
cache.set("key", value, ttl=3600)
result = cache.get("key")

# 3. 并行计算
results = parallel_map(process_func, data_list, workers=4)

# 4. Numba加速
import numpy as np
positions = np.random.rand(1000, 3) * 10
box = np.array([10.0, 10.0, 10.0])
dist_matrix = calculate_distance_matrix(positions, box)
```

## 性能提升

| 优化项 | 预期加速 | 适用场景 |
|--------|----------|----------|
| Numba JIT | 10-100x | 数值计算循环 |
| 并行处理 | 2-8x | 独立任务处理 |
| 智能缓存 | 100-1000x | 重复数据访问 |
| I/O加速 | 2-10x | 大文件读写 |
| 内存池 | 1.5-3x | 频繁数组分配 |

## 详细文档

### Profiler - 性能分析器

```python
from performance import Profiler, profile_function

# 方法1: 装饰器
profiler = Profiler()

@profiler.profile(name="custom_name")
def expensive_func():
    # 耗时操作
    pass

# 方法2: 上下文管理器
with profiler.measure("code_block"):
    # 代码块
    pass

# 生成报告
print(profiler.generate_report())

# 导出JSON
profiler.export_json("profile_stats.json")
```

### CacheManager - 缓存管理

```python
from performance import CacheManager

cache = CacheManager(
    memory_cache_size=1000,
    file_cache_dir="/tmp/cache",
    mmap_cache_dir="/tmp/mmap"
)

# 多级缓存自动管理
cache.set("key", large_array, use_mmap=True)
value = cache.get("key")  # 自动从合适层级获取

# 装饰器缓存
@cache.cached(ttl=3600)
def expensive_calculation(x):
    return x ** 2
```

### ParallelEngine - 并行引擎

```python
from performance import ParallelEngine, parallel_map

engine = ParallelEngine()

# 并行映射
results = engine.parallel_map(
    func, 
    data,
    backend='process',  # 'thread', 'process', 'auto'
    workers=8
)

# 并行for循环
results = engine.parallel_for(
    lambda i: compute(i),
    n_iterations=1000
)

# 便捷函数
results = parallel_map(func, data, workers=4)
```

### Numba Kernels - 加速核

```python
from performance.numba_kernels import (
    calculate_distance_matrix,
    calculate_rdf_parallel,
    calculate_msd_parallel,
    build_neighbor_list,
    calculate_lennard_jones_energy,
    calculate_lennard_jones_forces
)

# 所有函数自动使用Numba并行加速
positions = np.random.rand(1000, 3)
box = np.array([10.0, 10.0, 10.0])

# 距离矩阵（自动并行）
dist_matrix = calculate_distance_matrix(positions, box)

# RDF计算
r_bins, g_r = calculate_rdf_parallel(
    positions, box, 
    r_max=10.0, 
    n_bins=100
)

# MSD计算
msd = calculate_msd_parallel(trajectory, box)

# 邻居列表
neighbors, n_neighbors = build_neighbor_list(
    positions, box, cutoff=3.0
)

# LJ能量和力
energy = calculate_lennard_jones_energy(
    positions, box, sigma=1.0, epsilon=1.0
)
forces = calculate_lennard_jones_forces(
    positions, box, sigma=1.0, epsilon=1.0
)
```

### MemoryPool - 内存池

```python
from performance import ArrayPool

pool = ArrayPool(max_pools_per_spec=10)

# 方法1: 上下文管理器（推荐）
with pool.acquire((1000, 1000), dtype=np.float64) as arr:
    arr[:] = compute_result
    # 自动归还到池

# 方法2: 手动管理
arr = pool.get((1000, 1000), dtype=np.float64)
# 使用...
pool.release(arr)

# 预分配常用规格
pool.preallocate([
    ((1000, 3), np.float64),
    ((100, 100, 100), np.float32),
])
```

### IOAccelerator - I/O加速

```python
from performance import IOAccelerator, MemoryMappedFile

io = IOAccelerator()

# 快速读取大文件
data = io.fast_read("large_file.bin", use_mmap=True)

# 内存映射文件
with MemoryMappedFile("data.bin") as mmf:
    chunk = mmf.read_chunk(0, 1024)
    # 或访问整个缓冲区
    buffer = mmf.buffer

# 并发读取多个文件
contents = io.concurrent_read(["file1.txt", "file2.txt"])

# 流式读取
for line in io.stream_lines("large_text.txt"):
    process(line)
```

## 运行测试

```bash
# 运行所有测试
pytest performance/tests/

# 运行基准测试
python performance/tests/benchmark_suite.py
```

## 依赖

必需:
- numpy
- numba (可选，但推荐)

可选（增强功能）:
- psutil (内存监控)
- aiofiles (异步I/O)
- lz4 (压缩缓存)
- zstandard (压缩缓存)
- joblib (并行后端)

安装所有可选依赖:
```bash
pip install psutil aiofiles lz4 zstandard joblib
```

## 配置

```python
from performance import configure_performance

# 全局配置
configure_performance(
    numba=True,           # 启用Numba加速
    parallel=True,        # 启用并行计算
    cache=True,           # 启用缓存
    memory_pool=True,     # 启用内存池
    async_io=True,        # 启用异步I/O
    profiling=False       # 默认关闭分析（避免开销）
)

# 查看状态
from performance import get_performance_status
print(get_performance_status())
```

## 最佳实践

1. **性能分析优先**: 使用Profiler找出真正的瓶颈
2. **逐步优化**: 从最容易的优化（缓存）开始
3. **测试验证**: 每次优化后用benchmark验证效果
4. **内存管理**: 大数组使用ArrayPool避免频繁分配
5. **并行粒度**: 任务执行时间 > 10ms时才考虑并行

## 许可证

MIT License
