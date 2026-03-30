#!/usr/bin/env python3
"""
tests/benchmark_suite.py
========================
性能模块综合基准测试

运行: python benchmark_suite.py

测试项目:
- 性能分析器开销
- 缓存系统性能
- 并行引擎效率
- Numba加速核速度
- I/O加速效果
"""

import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from performance import Profiler, CacheManager, ParallelEngine
from performance.numba_kernels import NUMBA_AVAILABLE


class Benchmark:
    """基准测试基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.results = {}
    
    def run(self) -> dict:
        """运行基准测试"""
        raise NotImplementedError
    
    def report(self) -> str:
        """生成报告"""
        lines = [f"\n{'='*60}", f"{self.name}", f"{'='*60}"]
        
        for test_name, result in self.results.items():
            lines.append(f"\n{test_name}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {result}")
        
        return "\n".join(lines)


class ProfilerBenchmark(Benchmark):
    """性能分析器基准测试"""
    
    def __init__(self):
        super().__init__("Profiler Benchmark")
    
    def run(self) -> dict:
        profiler = Profiler()
        
        # 测试1: 装饰器开销
        def test_func():
            return sum(range(1000))
        
        # 无分析
        start = time.perf_counter()
        for _ in range(1000):
            test_func()
        baseline_time = time.perf_counter() - start
        
        # 有分析
        @profiler.profile
        def profiled_func():
            return sum(range(1000))
        
        start = time.perf_counter()
        for _ in range(1000):
            profiled_func()
        profiled_time = time.perf_counter() - start
        
        overhead = (profiled_time - baseline_time) / baseline_time * 100
        
        self.results["decorator_overhead"] = {
            "baseline_time_ms": baseline_time * 1000,
            "profiled_time_ms": profiled_time * 1000,
            "overhead_percent": f"{overhead:.2f}%"
        }
        
        return self.results


class CacheBenchmark(Benchmark):
    """缓存系统基准测试"""
    
    def __init__(self):
        super().__init__("Cache System Benchmark")
    
    def run(self) -> dict:
        from performance.cache_manager import LRUCache
        
        cache = LRUCache(maxsize=1000)
        
        # 测试1: 写入性能
        start = time.perf_counter()
        for i in range(10000):
            cache.set(f"key_{i}", f"value_{i}")
        write_time = time.perf_counter() - start
        
        # 测试2: 读取性能（部分命中）
        start = time.perf_counter()
        hits = 0
        for i in range(5000, 15000):
            if cache.get(f"key_{i}") is not None:
                hits += 1
        read_time = time.perf_counter() - start
        
        self.results["write_performance"] = {
            "operations": 10000,
            "time_ms": write_time * 1000,
            "ops_per_sec": 10000 / write_time
        }
        
        self.results["read_performance"] = {
            "operations": 10000,
            "time_ms": read_time * 1000,
            "ops_per_sec": 10000 / read_time,
            "hit_rate": f"{hits / 10000 * 100:.1f}%"
        }
        
        return self.results


class ParallelBenchmark(Benchmark):
    """并行引擎基准测试"""
    
    def __init__(self):
        super().__init__("Parallel Engine Benchmark")
    
    def run(self) -> dict:
        engine = ParallelEngine()
        
        def compute_intensive(n):
            return sum(i ** 2 for i in range(n))
        
        data = [10000] * 100
        
        # 测试1: 串行执行
        start = time.perf_counter()
        serial_results = [compute_intensive(x) for x in data]
        serial_time = time.perf_counter() - start
        
        # 测试2: 并行执行
        start = time.perf_counter()
        parallel_results = engine.parallel_map(compute_intensive, data, workers=4)
        parallel_time = time.perf_counter() - start
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
        
        self.results["serial_vs_parallel"] = {
            "serial_time_ms": serial_time * 1000,
            "parallel_time_ms": parallel_time * 1000,
            "speedup": f"{speedup:.2f}x"
        }
        
        return self.results


class NumbaBenchmark(Benchmark):
    """Numba加速核基准测试"""
    
    def __init__(self):
        super().__init__("Numba Kernels Benchmark")
    
    def run(self) -> dict:
        if not NUMBA_AVAILABLE:
            self.results["status"] = "Numba not available"
            return self.results
        
        from performance.numba_kernels import calculate_distance_matrix
        
        # 生成测试数据
        n_atoms = 1000
        positions = np.random.rand(n_atoms, 3) * 10
        box = np.array([10.0, 10.0, 10.0])
        
        # 测试: 距离矩阵计算
        start = time.perf_counter()
        dist_matrix = calculate_distance_matrix(positions, box)
        numba_time = time.perf_counter() - start
        
        # Python版本对比
        start = time.perf_counter()
        py_dist = self._python_distance_matrix(positions, box)
        python_time = time.perf_counter() - start
        
        speedup = python_time / numba_time if numba_time > 0 else 1.0
        
        self.results["distance_matrix"] = {
            "numba_time_ms": numba_time * 1000,
            "python_time_ms": python_time * 1000,
            "speedup": f"{speedup:.2f}x"
        }
        
        return self.results
    
    def _python_distance_matrix(self, positions, box):
        """纯Python距离矩阵计算"""
        n = positions.shape[0]
        result = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = 0.0
                for k in range(3):
                    diff = positions[i, k] - positions[j, k]
                    if diff > box[k] * 0.5:
                        diff -= box[k]
                    elif diff < -box[k] * 0.5:
                        diff += box[k]
                    dist_sq += diff ** 2
                dist = np.sqrt(dist_sq)
                result[i, j] = dist
                result[j, i] = dist
        return result


class MemoryPoolBenchmark(Benchmark):
    """内存池基准测试"""
    
    def __init__(self):
        super().__init__("Memory Pool Benchmark")
    
    def run(self) -> dict:
        from performance.memory_pool import ArrayPool
        
        pool = ArrayPool()
        
        shape = (1000, 1000)
        dtype = np.float64
        
        # 测试1: 普通分配
        start = time.perf_counter()
        for _ in range(100):
            arr = np.empty(shape, dtype=dtype)
            # 使用一下避免优化
            arr.fill(1.0)
        normal_time = time.perf_counter() - start
        
        # 测试2: 内存池分配
        start = time.perf_counter()
        for _ in range(100):
            with pool.acquire(shape, dtype) as arr:
                arr.fill(1.0)
        pooled_time = time.perf_counter() - start
        
        speedup = normal_time / pooled_time if pooled_time > 0 else 1.0
        
        self.results["allocation"] = {
            "normal_time_ms": normal_time * 1000,
            "pooled_time_ms": pooled_time * 1000,
            "speedup": f"{speedup:.2f}x"
        }
        
        return self.results


def run_all_benchmarks():
    """运行所有基准测试"""
    print("=" * 60)
    print("PERFORMANCE MODULE BENCHMARK SUITE")
    print("=" * 60)
    
    benchmarks = [
        ProfilerBenchmark(),
        CacheBenchmark(),
        ParallelBenchmark(),
        NumbaBenchmark(),
        MemoryPoolBenchmark(),
    ]
    
    all_results = {}
    
    for benchmark in benchmarks:
        try:
            results = benchmark.run()
            all_results[benchmark.name] = results
            print(benchmark.report())
        except Exception as e:
            print(f"\n{benchmark.name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, results in all_results.items():
        print(f"\n{name}:")
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"  {key}: {value}")


if __name__ == "__main__":
    run_all_benchmarks()
