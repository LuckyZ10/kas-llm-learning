#!/usr/bin/env python3
"""
benchmark_md_simulation.py
==========================
MD模拟性能基准测试

测试项目:
1. 轨迹文件读写性能
2. RDF计算性能
3. MSD计算性能
4. 力场计算性能
5. 并行分析性能
6. 大轨迹文件处理

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

# 尝试导入优化库
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available")

sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')
from core.common.parallel import ParallelConfig, create_executor


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    execution_time: float
    memory_peak_mb: float
    throughput: float
    details: Dict = None


class MDTestDataGenerator:
    """生成MD测试数据"""
    
    @staticmethod
    def generate_lammps_dump(output_file: str, n_frames: int = 1000, n_atoms: int = 1000):
        """生成LAMMPS dump文件"""
        elements = ['Li', 'P', 'S']
        
        with open(output_file, 'w') as f:
            for frame_idx in range(n_frames):
                # TIMESTEP
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{frame_idx * 100}\n")
                
                # NUMBER OF ATOMS
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{n_atoms}\n")
                
                # BOX BOUNDS
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("0.0000000000000000e+00 1.0000000000000000e+01\n")
                f.write("0.0000000000000000e+00 1.0000000000000000e+01\n")
                f.write("0.0000000000000000e+00 1.0000000000000000e+01\n")
                
                # ATOMS
                f.write("ITEM: ATOMS id type x y z vx vy vz\n")
                
                for atom_idx in range(n_atoms):
                    atom_type = (atom_idx % 3) + 1
                    elem = elements[atom_type - 1]
                    
                    # 位置（带一点随机位移）
                    x = (atom_idx % 10) + np.random.random() * 0.1
                    y = ((atom_idx // 10) % 10) + np.random.random() * 0.1
                    z = (atom_idx // 100) + np.random.random() * 0.1
                    
                    vx, vy, vz = np.random.randn(3) * 0.01
                    
                    f.write(f"{atom_idx+1} {atom_type} {x:.6f} {y:.6f} {z:.6f} "
                           f"{vx:.6f} {vy:.6f} {vz:.6f}\n")
        
        return output_file
    
    @staticmethod
    def generate_xyz_trajectory(output_file: str, n_frames: int = 1000, n_atoms: int = 1000):
        """生成XYZ轨迹文件"""
        elements = ['Li', 'P', 'S']
        
        with open(output_file, 'w') as f:
            for frame_idx in range(n_frames):
                f.write(f"{n_atoms}\n")
                f.write(f"Frame {frame_idx}\n")
                
                for atom_idx in range(n_atoms):
                    elem = elements[atom_idx % 3]
                    x = np.random.random() * 10
                    y = np.random.random() * 10
                    z = np.random.random() * 10
                    f.write(f"{elem} {x:.6f} {y:.6f} {z:.6f}\n")
        
        return output_file


class MDSimulationBenchmark:
    """MD模拟性能测试"""
    
    def __init__(self, test_dir: str = "./benchmark_md_data"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True)
        self.results = []
        self.trajectory_file = None
        
    def setup(self):
        """准备测试数据"""
        print("Setting up MD simulation test data...")
        
        # 生成不同大小的轨迹文件
        self.trajectory_small = MDTestDataGenerator.generate_lammps_dump(
            str(self.test_dir / "traj_small.dump"),
            n_frames=100,
            n_atoms=100
        )
        
        self.trajectory_medium = MDTestDataGenerator.generate_lammps_dump(
            str(self.test_dir / "traj_medium.dump"),
            n_frames=1000,
            n_atoms=500
        )
        
        self.trajectory_large = MDTestDataGenerator.generate_lammps_dump(
            str(self.test_dir / "traj_large.dump"),
            n_frames=5000,
            n_atoms=1000
        )
        
        # XYZ格式
        self.xyz_file = MDTestDataGenerator.generate_xyz_trajectory(
            str(self.test_dir / "traj.xyz"),
            n_frames=1000,
            n_atoms=500
        )
        
        print(f"Test data ready in {self.test_dir}")
    
    def cleanup(self):
        """清理测试数据"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def benchmark_trajectory_reading(self) -> BenchmarkResult:
        """测试轨迹文件读取性能"""
        print("\n=== Benchmark: Trajectory Reading ===")
        
        tracemalloc.start()
        start_time = time.time()
        
        frame_count = 0
        with open(self.trajectory_large, 'r') as f:
            for line in f:
                if "ITEM: TIMESTEP" in line:
                    frame_count += 1
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="trajectory_reading",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=frame_count / execution_time,
            details={'frame_count': frame_count}
        )
        
        print(f"  Scanned {frame_count} frames in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_trajectory_parsing(self) -> BenchmarkResult:
        """测试轨迹解析性能"""
        print("\n=== Benchmark: Trajectory Parsing ===")
        
        def parse_dump_frame(lines):
            """解析单帧数据"""
            data = {'atoms': []}
            for line in lines:
                if line.startswith("ITEM: TIMESTEP"):
                    continue
                elif line.startswith("ITEM: NUMBER OF ATOMS"):
                    continue
                elif line.startswith("ITEM: BOX BOUNDS"):
                    continue
                elif line.startswith("ITEM: ATOMS"):
                    continue
                else:
                    parts = line.split()
                    if len(parts) >= 7:
                        data['atoms'].append({
                            'id': int(parts[0]),
                            'type': int(parts[1]),
                            'x': float(parts[2]),
                            'y': float(parts[3]),
                            'z': float(parts[4])
                        })
            return data
        
        tracemalloc.start()
        start_time = time.time()
        
        frames_parsed = 0
        with open(self.trajectory_medium, 'r') as f:
            buffer = []
            for line in f:
                if "ITEM: TIMESTEP" in line and buffer:
                    _ = parse_dump_frame(buffer)
                    frames_parsed += 1
                    buffer = [line]
                else:
                    buffer.append(line)
            
            # 处理最后一帧
            if buffer:
                _ = parse_dump_frame(buffer)
                frames_parsed += 1
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="trajectory_parsing",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=frames_parsed / execution_time,
            details={'frames_parsed': frames_parsed}
        )
        
        print(f"  Parsed {frames_parsed} frames in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_rdf_calculation(self) -> BenchmarkResult:
        """测试RDF计算性能"""
        print("\n=== Benchmark: RDF Calculation ===")
        
        # 读取一帧数据
        positions = []
        with open(self.trajectory_small, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 7 and not line.startswith("ITEM"):
                    positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
        
        positions = np.array(positions)
        
        def calculate_rdf(positions, bin_width=0.1, r_max=10.0):
            """计算RDF"""
            n_atoms = len(positions)
            distances = []
            
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < r_max:
                        distances.append(dist)
            
            hist, bin_edges = np.histogram(distances, bins=int(r_max/bin_width), range=(0, r_max))
            return hist, bin_edges
        
        tracemalloc.start()
        start_time = time.time()
        
        # 计算多次
        for _ in range(10):
            hist, edges = calculate_rdf(positions)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="rdf_calculation",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=10 / execution_time,
            details={'n_atoms': len(positions), 'iterations': 10}
        )
        
        print(f"  Calculated RDF 10 times in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_rdf_optimized(self) -> BenchmarkResult:
        """测试优化的RDF计算"""
        print("\n=== Benchmark: Optimized RDF (Numba) ===")
        
        if not NUMBA_AVAILABLE:
            print("  Numba not available, skipping")
            return None
        
        positions = []
        with open(self.trajectory_small, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 7 and not line.startswith("ITEM"):
                    positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
        
        positions = np.array(positions)
        
        @njit(parallel=True)
        def calculate_rdf_numba(positions, r_max=10.0, n_bins=100):
            """优化的RDF计算"""
            n_atoms = len(positions)
            hist = np.zeros(n_bins)
            bin_width = r_max / n_bins
            
            for i in prange(n_atoms):
                for j in range(i+1, n_atoms):
                    dist = 0.0
                    for k in range(3):
                        dist += (positions[i, k] - positions[j, k]) ** 2
                    dist = np.sqrt(dist)
                    
                    if dist < r_max:
                        bin_idx = int(dist / bin_width)
                        if bin_idx < n_bins:
                            hist[bin_idx] += 1
            
            return hist
        
        # 预热
        _ = calculate_rdf_numba(positions[:50])
        
        tracemalloc.start()
        start_time = time.time()
        
        for _ in range(10):
            hist = calculate_rdf_numba(positions)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="rdf_calculation_optimized",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=10 / execution_time,
            details={'method': 'numba_parallel'}
        )
        
        print(f"  Calculated RDF 10 times (optimized) in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        print(f"  Speedup vs naive: {result.throughput / self.results[-1].throughput:.1f}x")
        
        return result
    
    def benchmark_msd_calculation(self) -> BenchmarkResult:
        """测试MSD计算性能"""
        print("\n=== Benchmark: MSD Calculation ===")
        
        # 模拟轨迹数据
        n_frames = 1000
        n_atoms = 100
        n_selected = 20  # 只跟踪部分原子
        
        # 生成随机轨迹
        trajectories = np.cumsum(np.random.randn(n_frames, n_selected, 3) * 0.1, axis=0)
        
        def calculate_msd(trajectories):
            """计算MSD"""
            n_frames = len(trajectories)
            msd = np.zeros(n_frames)
            
            for dt in range(1, n_frames):
                displacements = trajectories[dt:] - trajectories[:-dt]
                squared_disp = np.sum(displacements ** 2, axis=2)
                msd[dt] = np.mean(squared_disp)
            
            return msd
        
        tracemalloc.start()
        start_time = time.time()
        
        msd = calculate_msd(trajectories)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="msd_calculation",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=n_frames / execution_time,
            details={'n_frames': n_frames, 'n_atoms': n_selected}
        )
        
        print(f"  Calculated MSD for {n_frames} frames in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_parallel_analysis(self, n_workers: int = 4) -> BenchmarkResult:
        """测试并行分析性能"""
        print(f"\n=== Benchmark: Parallel Analysis ({n_workers} workers) ===")
        
        # 分块读取大轨迹
        def count_frames_in_chunk(filepath, start_line, end_line):
            """计算块中的帧数"""
            count = 0
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        continue
                    if i >= end_line:
                        break
                    if "ITEM: TIMESTEP" in line:
                        count += 1
            return count
        
        # 先统计总行数
        total_lines = 0
        with open(self.trajectory_large, 'r') as f:
            for _ in f:
                total_lines += 1
        
        # 分块
        chunk_size = total_lines // n_workers
        chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_workers)]
        
        tracemalloc.start()
        start_time = time.time()
        
        from concurrent.futures import ProcessPoolExecutor
        
        total_frames = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(count_frames_in_chunk, self.trajectory_large, start, end)
                for start, end in chunks
            ]
            for future in futures:
                total_frames += future.result()
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name=f"parallel_analysis_{n_workers}",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=total_frames / execution_time,
            details={'n_workers': n_workers, 'total_frames': total_frames}
        )
        
        print(f"  Analyzed {total_frames} frames with {n_workers} workers in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_large_trajectory_streaming(self) -> BenchmarkResult:
        """测试大轨迹文件流式处理"""
        print("\n=== Benchmark: Large Trajectory Streaming ===")
        
        def stream_process_trajectory(filepath, process_func):
            """流式处理轨迹，不保存所有帧"""
            frame_count = 0
            current_frame_lines = []
            
            with open(filepath, 'r') as f:
                for line in f:
                    if "ITEM: TIMESTEP" in line:
                        if current_frame_lines:
                            # 处理上一帧
                            process_func(current_frame_lines)
                            frame_count += 1
                            current_frame_lines = []
                    current_frame_lines.append(line)
                
                # 处理最后一帧
                if current_frame_lines:
                    process_func(current_frame_lines)
                    frame_count += 1
            
            return frame_count
        
        def dummy_process(frame_lines):
            """虚拟处理函数 - 只计算原子数"""
            for line in frame_lines:
                if "ITEM: NUMBER OF ATOMS" in line:
                    return
        
        tracemalloc.start()
        start_time = time.time()
        
        frame_count = stream_process_trajectory(self.trajectory_large, dummy_process)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="large_trajectory_streaming",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=frame_count / execution_time,
            details={'frame_count': frame_count, 'strategy': 'streaming'}
        )
        
        print(f"  Streamed {frame_count} frames in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def profile_md_code(self):
        """对MD代码进行性能分析"""
        print("\n=== Profiling: MD Code Analysis ===")
        
        def simulate_force_calculation():
            """模拟力场计算"""
            n_atoms = 100
            positions = np.random.randn(n_atoms, 3) * 10
            
            # 计算所有原子对距离
            forces = np.zeros((n_atoms, 3))
            
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    r = positions[j] - positions[i]
                    dist = np.linalg.norm(r)
                    
                    if dist < 5.0:
                        # 简化的LJ力
                        f = 24 * (2/dist**13 - 1/dist**7) * r / dist
                        forces[i] -= f
                        forces[j] += f
            
            return forces
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(100):
            simulate_force_calculation()
        
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print("\nTop 15 time-consuming operations:")
        stats.print_stats(15)
        
        stats.dump_stats(self.test_dir / "md_simulation_profile.prof")
        print(f"\nProfile saved to: {self.test_dir / 'md_simulation_profile.prof'}")
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("="*60)
        print("MD Simulation Performance Benchmarks")
        print("="*60)
        
        self.setup()
        
        try:
            # 轨迹读取
            r1 = self.benchmark_trajectory_reading()
            self.results.append(r1)
            
            # 轨迹解析
            r2 = self.benchmark_trajectory_parsing()
            self.results.append(r2)
            
            # RDF计算
            r3 = self.benchmark_rdf_calculation()
            self.results.append(r3)
            
            # 优化的RDF
            r3_opt = self.benchmark_rdf_optimized()
            if r3_opt:
                self.results.append(r3_opt)
            
            # MSD计算
            r4 = self.benchmark_msd_calculation()
            self.results.append(r4)
            
            # 并行分析
            r5 = self.benchmark_parallel_analysis(n_workers=4)
            self.results.append(r5)
            
            # 大轨迹流式处理
            r6 = self.benchmark_large_trajectory_streaming()
            self.results.append(r6)
            
            # 性能分析
            self.profile_md_code()
            
        finally:
            self.generate_report()
            self.cleanup()
    
    def generate_report(self):
        """生成性能报告"""
        print("\n" + "="*60)
        print("MD Simulation Benchmark Summary")
        print("="*60)
        
        df = pd.DataFrame([
            {
                'Test': r.name,
                'Time (s)': f"{r.execution_time:.3f}",
                'Memory (MB)': f"{r.memory_peak_mb:.2f}",
                'Throughput': f"{r.throughput:.1f}"
            }
            for r in self.results if r
        ])
        
        print(df.to_string(index=False))
        
        results_dict = {
            'benchmarks': [
                {
                    'name': r.name,
                    'execution_time': r.execution_time,
                    'memory_peak_mb': r.memory_peak_mb,
                    'throughput': r.throughput,
                    'details': r.details
                }
                for r in self.results if r
            ]
        }
        
        with open('benchmark_md_simulation_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: benchmark_md_simulation_results.json")


if __name__ == "__main__":
    benchmark = MDSimulationBenchmark()
    benchmark.run_all_benchmarks()
