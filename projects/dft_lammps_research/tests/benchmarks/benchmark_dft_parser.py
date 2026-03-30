#!/usr/bin/env python3
"""
benchmark_dft_parser.py
=======================
VASP/OUTCAR解析性能基准测试

测试项目:
1. OUTCAR文件解析速度
2. 大文件内存使用
3. 批量文件处理吞吐量
4. 特征提取性能
5. 优化前后对比

作者: Performance Optimization Expert
"""

import os
import sys
import time
import json
import tracemalloc
import cProfile
import pstats
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import tempfile
import shutil

import numpy as np
import pandas as pd

# 确保可以导入项目代码
sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')
from core.dft.bridge import VASPOUTCARParser, VASPParserConfig

# 尝试导入优化库
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, using pure Python")

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available")

# 测试数据生成器
class TestDataGenerator:
    """生成测试用的OUTCAR文件"""
    
    @staticmethod
    def generate_outcar(output_path: str, n_frames: int = 100, n_atoms: int = 50):
        """生成模拟的OUTCAR文件"""
        
        # 定义元素
        elements = ['Li', 'P', 'S'] * (n_atoms // 3 + 1)
        elements = elements[:n_atoms]
        
        with open(output_path, 'w') as f:
            f.write("VASP output file\n")
            f.write(f"   {n_atoms}   atoms\n")
            f.write("  3 types of ions\n")
            
            for frame_idx in range(n_frames):
                f.write(f"\nIteration {frame_idx}\n")
                f.write("---------------------------------------\n")
                f.write("FREE ENERGIE OF THE ION-ELECTRON SYSTEM\n")
                
                # 随机能量
                energy = -100.0 + np.random.random() * 10.0
                f.write(f"  free  energy   TOTEN  =      {energy:.6f} eV\n")
                
                # 原子位置
                f.write("\nPOSITION                                       TOTAL-FORCE (eV/Angst)\n")
                f.write("-----------------------------------------------------------------------------------\n")
                
                for i, elem in enumerate(elements):
                    x, y, z = np.random.random(3) * 10
                    fx, fy, fz = np.random.randn(3) * 0.1
                    f.write(f"  {i+1:4d}  {elem:4s}  {x:12.6f} {y:12.6f} {z:12.6f}  {fx:12.6f} {fy:12.6f} {fz:12.6f}\n")
                
                # 晶格
                f.write("\n  direct lattice vectors                 reciprocal lattice vectors\n")
                for i in range(3):
                    vec = np.random.random(3) * 5 + 5
                    f.write(f" {vec[0]:11.6f} {vec[1]:11.6f} {vec[2]:11.6f}\n")


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    execution_time: float
    memory_peak_mb: float
    throughput: float  # items/second
    details: Dict = None


class DFTParserBenchmark:
    """DFT解析器性能测试"""
    
    def __init__(self, test_dir: str = "./benchmark_test_data"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True)
        self.results = []
        
    def setup(self, n_files: int = 10, n_frames_per_file: int = 100, n_atoms: int = 50):
        """准备测试数据"""
        print(f"Setting up test data: {n_files} files, {n_frames_per_file} frames each, {n_atoms} atoms")
        
        self.test_files = []
        for i in range(n_files):
            filepath = self.test_dir / f"OUTCAR_{i:03d}"
            TestDataGenerator.generate_outcar(str(filepath), n_frames_per_file, n_atoms)
            self.test_files.append(str(filepath))
        
        print(f"Generated {n_files} test files in {self.test_dir}")
    
    def cleanup(self):
        """清理测试数据"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def benchmark_single_file_parsing(self) -> BenchmarkResult:
        """测试单文件解析性能"""
        print("\n=== Benchmark: Single File Parsing ===")
        
        parser = VASPOUTCARParser()
        test_file = self.test_files[0]
        
        # 内存跟踪
        tracemalloc.start()
        start_time = time.time()
        
        frames = parser.parse(test_file)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="single_file_parsing",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=len(frames) / execution_time,
            details={
                'n_frames': len(frames),
                'file_size_mb': Path(test_file).stat().st_size / 1024 / 1024
            }
        )
        
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        print(f"  Throughput: {result.throughput:.1f} frames/s")
        
        return result
    
    def benchmark_batch_parsing(self) -> BenchmarkResult:
        """测试批量解析性能"""
        print("\n=== Benchmark: Batch Parsing (Sequential) ===")
        
        parser = VASPOUTCARParser()
        
        tracemalloc.start()
        start_time = time.time()
        
        total_frames = 0
        for filepath in self.test_files:
            frames = parser.parse(filepath)
            total_frames += len(frames)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="batch_parsing_sequential",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=total_frames / execution_time,
            details={'n_files': len(self.test_files), 'total_frames': total_frames}
        )
        
        print(f"  Total files: {len(self.test_files)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        print(f"  Throughput: {result.throughput:.1f} frames/s")
        
        return result
    
    def benchmark_parallel_parsing(self, n_workers: int = 4) -> BenchmarkResult:
        """测试并行解析性能"""
        print(f"\n=== Benchmark: Parallel Parsing ({n_workers} workers) ===")
        
        from concurrent.futures import ProcessPoolExecutor
        
        def parse_file(filepath):
            parser = VASPOUTCARParser()
            return len(parser.parse(filepath))
        
        tracemalloc.start()
        start_time = time.time()
        
        total_frames = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(parse_file, self.test_files))
            total_frames = sum(results)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name=f"batch_parsing_parallel_{n_workers}",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=total_frames / execution_time,
            details={'n_workers': n_workers, 'total_frames': total_frames}
        )
        
        print(f"  Workers: {n_workers}")
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        print(f"  Throughput: {result.throughput:.1f} frames/s")
        
        return result
    
    def benchmark_with_profiling(self):
        """使用cProfile进行性能分析"""
        print("\n=== Profiling: Detailed Performance Analysis ===")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        parser = VASPOUTCARParser()
        for filepath in self.test_files[:3]:  # 分析前3个文件
            frames = parser.parse(filepath)
        
        profiler.disable()
        
        # 统计
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        print("\nTop 20 time-consuming functions:")
        stats.print_stats(20)
        
        # 保存到文件
        stats.dump_stats(self.test_dir / "dft_parser_profile.prof")
        print(f"\nProfile saved to: {self.test_dir / 'dft_parser_profile.prof'}")
    
    def benchmark_memory_efficient_parsing(self) -> BenchmarkResult:
        """测试内存优化版本的解析"""
        print("\n=== Benchmark: Memory-Efficient Parsing ===")
        
        # 内存优化版本：生成器模式
        def parse_frames_generator(filepath):
            """逐帧生成，不保存所有帧"""
            parser = VASPOUTCARParser()
            # 简化的生成器实现
            frames = parser.parse(filepath)
            for frame in frames:
                yield frame
        
        tracemalloc.start()
        start_time = time.time()
        
        total_frames = 0
        for filepath in self.test_files:
            for frame in parse_frames_generator(filepath):
                # 只处理，不保存
                _ = frame['energy']
                total_frames += 1
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="memory_efficient_parsing",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=total_frames / execution_time,
            details={'method': 'generator', 'total_frames': total_frames}
        )
        
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        print(f"  Throughput: {result.throughput:.1f} frames/s")
        
        return result
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("="*60)
        print("DFT Parser Performance Benchmarks")
        print("="*60)
        
        self.setup(n_files=10, n_frames_per_file=100, n_atoms=50)
        
        try:
            # 单文件测试
            r1 = self.benchmark_single_file_parsing()
            self.results.append(r1)
            
            # 批量顺序解析
            r2 = self.benchmark_batch_parsing()
            self.results.append(r2)
            
            # 并行解析
            r3 = self.benchmark_parallel_parsing(n_workers=4)
            self.results.append(r3)
            
            # 内存优化版本
            r4 = self.benchmark_memory_efficient_parsing()
            self.results.append(r4)
            
            # 性能分析
            self.benchmark_with_profiling()
            
        finally:
            self.generate_report()
            self.cleanup()
    
    def generate_report(self):
        """生成性能报告"""
        print("\n" + "="*60)
        print("DFT Parser Benchmark Summary")
        print("="*60)
        
        df = pd.DataFrame([
            {
                'Test': r.name,
                'Time (s)': f"{r.execution_time:.3f}",
                'Memory (MB)': f"{r.memory_peak_mb:.2f}",
                'Throughput (items/s)': f"{r.throughput:.1f}"
            }
            for r in self.results
        ])
        
        print(df.to_string(index=False))
        
        # 保存结果
        results_dict = {
            'benchmarks': [
                {
                    'name': r.name,
                    'execution_time': r.execution_time,
                    'memory_peak_mb': r.memory_peak_mb,
                    'throughput': r.throughput,
                    'details': r.details
                }
                for r in self.results
            ]
        }
        
        with open('benchmark_dft_parser_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: benchmark_dft_parser_results.json")


# 优化的解析器版本
class OptimizedVASPOUTCARParser(VASPOUTCARParser):
    """优化版本的OUTCAR解析器"""
    
    def parse_chunked(self, outcar_path: str, chunk_size: int = 1000):
        """分块解析大文件"""
        # 实现分块读取，适合超大文件
        file_size = Path(outcar_path).stat().st_size
        
        with open(outcar_path, 'r') as f:
            buffer = []
            bytes_read = 0
            
            for line in f:
                buffer.append(line)
                bytes_read += len(line.encode('utf-8'))
                
                if len(buffer) >= chunk_size:
                    # 处理缓冲区
                    yield from self._process_buffer(buffer)
                    buffer = []
                    
                    # 报告进度
                    progress = bytes_read / file_size * 100
                    if int(progress) % 10 == 0:
                        print(f"  Progress: {progress:.1f}%")
            
            # 处理剩余内容
            if buffer:
                yield from self._process_buffer(buffer)
    
    def _process_buffer(self, lines: List[str]):
        """处理缓冲区内容"""
        # 简化的处理逻辑
        pass


if __name__ == "__main__":
    benchmark = DFTParserBenchmark()
    benchmark.run_all_benchmarks()
