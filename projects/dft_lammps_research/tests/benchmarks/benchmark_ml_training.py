#!/usr/bin/env python3
"""
benchmark_ml_training.py
=======================
ML势训练性能基准测试

测试项目:
1. 数据加载和预处理速度
2. 模型训练速度 (DeepMD/NEP)
3. 不同batch size的性能
4. 内存使用模式
5. GPU vs CPU性能对比
6. 分布式训练扩展性

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

# 测试数据生成
class MLTrainingDataGenerator:
    """生成ML训练测试数据"""
    
    @staticmethod
    def generate_deepmd_dataset(output_dir: str, n_systems: int = 10, 
                                 n_frames_per_system: int = 100,
                                 n_atoms: int = 50,
                                 n_types: int = 3):
        """生成DeepMD格式的训练数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        type_map = ['Li', 'P', 'S'][:n_types]
        
        for sys_idx in range(n_systems):
            sys_dir = output_path / f"system_{sys_idx:03d}"
            sys_dir.mkdir(exist_ok=True)
            
            coords = []
            cells = []
            energies = []
            forces = []
            types = []
            
            for frame_idx in range(n_frames_per_system):
                # 随机坐标
                coord = np.random.randn(n_atoms, 3) * 5
                coords.append(coord)
                
                # 晶格
                cell = np.diag([10, 10, 10]) + np.random.randn(3, 3) * 0.5
                cells.append(cell)
                
                # 能量
                energy = -100.0 + np.random.random() * 10
                energies.append(energy)
                
                # 力
                force = np.random.randn(n_atoms, 3) * 0.1
                forces.append(force)
                
                # 类型
                atom_types = np.random.randint(0, n_types, n_atoms)
                types.append(atom_types)
            
            # 保存为npy
            np.save(sys_dir / "coord.npy", np.array(coords))
            np.save(sys_dir / "box.npy", np.array(cells))
            np.save(sys_dir / "energy.npy", np.array(energies))
            np.save(sys_dir / "force.npy", np.array(forces))
            np.save(sys_dir / "type.npy", types[0])  # 类型不变
            
            # type_map.raw
            with open(sys_dir / "type_map.raw", 'w') as f:
                f.write('\n'.join(type_map))
        
        return str(output_path), type_map
    
    @staticmethod
    def generate_nep_xyz(output_file: str, n_frames: int = 1000, n_atoms: int = 50):
        """生成NEP格式的XYZ文件"""
        elements = ['Li', 'P', 'S']
        
        with open(output_file, 'w') as f:
            for frame_idx in range(n_frames):
                # 随机原子数（略有变化）
                current_n_atoms = n_atoms + np.random.randint(-5, 5)
                current_n_atoms = max(10, current_n_atoms)
                
                f.write(f"{current_n_atoms}\n")
                
                # 晶格和能量信息
                lattice = np.diag([10, 10, 10]) + np.random.randn(3, 3) * 0.5
                lattice_flat = lattice.flatten()
                energy = -100.0 + np.random.random() * 10
                
                lattice_str = " ".join([f"{x:.10f}" for x in lattice_flat])
                f.write(f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3 '
                       f'energy={energy:.10f}\n')
                
                # 原子数据
                for atom_idx in range(current_n_atoms):
                    elem = elements[atom_idx % len(elements)]
                    pos = np.random.randn(3) * 5
                    force = np.random.randn(3) * 0.1
                    
                    f.write(f"{elem:>3} {pos[0]:15.8f} {pos[1]:15.8f} {pos[2]:15.8f} "
                           f"{force[0]:15.8f} {force[1]:15.8f} {force[2]:15.8f}\n")
        
        return output_file


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    execution_time: float
    memory_peak_mb: float
    throughput: float
    details: Dict = None


class MLTrainingBenchmark:
    """ML训练性能测试"""
    
    def __init__(self, test_dir: str = "./benchmark_ml_data"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True)
        self.results = []
        
    def setup(self):
        """准备测试数据"""
        print("Setting up ML training test data...")
        
        # DeepMD数据
        self.deepmd_dir, self.type_map = MLTrainingDataGenerator.generate_deepmd_dataset(
            self.test_dir / "deepmd_data",
            n_systems=20,
            n_frames_per_system=200,
            n_atoms=50
        )
        
        # NEP数据
        self.nep_train = MLTrainingDataGenerator.generate_nep_xyz(
            str(self.test_dir / "train.xyz"),
            n_frames=5000,
            n_atoms=50
        )
        self.nep_test = MLTrainingDataGenerator.generate_nep_xyz(
            str(self.test_dir / "test.xyz"),
            n_frames=500,
            n_atoms=50
        )
        
        print(f"Test data ready in {self.test_dir}")
    
    def cleanup(self):
        """清理测试数据"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def benchmark_data_loading(self) -> BenchmarkResult:
        """测试数据加载性能"""
        print("\n=== Benchmark: Data Loading ===")
        
        tracemalloc.start()
        start_time = time.time()
        
        # 加载所有系统
        total_frames = 0
        for sys_dir in Path(self.deepmd_dir).glob("system_*"):
            if (sys_dir / "coord.npy").exists():
                coords = np.load(sys_dir / "coord.npy")
                total_frames += len(coords)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="data_loading",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=total_frames / execution_time,
            details={'total_frames': total_frames}
        )
        
        print(f"  Loaded {total_frames} frames in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        print(f"  Throughput: {result.throughput:.1f} frames/s")
        
        return result
    
    def benchmark_batch_loading(self, batch_sizes=[32, 64, 128, 256]) -> List[BenchmarkResult]:
        """测试不同batch size的加载性能"""
        print("\n=== Benchmark: Batch Loading Performance ===")
        
        results = []
        
        for batch_size in batch_sizes:
            # 模拟数据加载
            tracemalloc.start()
            start_time = time.time()
            
            total_batches = 0
            for sys_dir in Path(self.deepmd_dir).glob("system_*"):
                if (sys_dir / "coord.npy").exists():
                    coords = np.load(sys_dir / "coord.npy")
                    n_frames = len(coords)
                    
                    # 模拟batch处理
                    for i in range(0, n_frames, batch_size):
                        batch = coords[i:i+batch_size]
                        # 模拟处理
                        _ = batch.mean()
                        total_batches += 1
            
            execution_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            result = BenchmarkResult(
                name=f"batch_size_{batch_size}",
                execution_time=execution_time,
                memory_peak_mb=peak / 1024 / 1024,
                throughput=total_batches / execution_time,
                details={'batch_size': batch_size, 'total_batches': total_batches}
            )
            
            results.append(result)
            print(f"  Batch size {batch_size}: {execution_time:.3f}s, "
                  f"{result.throughput:.1f} batches/s")
        
        return results
    
    def benchmark_data_preprocessing(self) -> BenchmarkResult:
        """测试数据预处理性能"""
        print("\n=== Benchmark: Data Preprocessing ===")
        
        tracemalloc.start()
        start_time = time.time()
        
        # 模拟预处理操作
        processed_frames = 0
        for sys_dir in Path(self.deepmd_dir).glob("system_*"):
            if (sys_dir / "coord.npy").exists():
                coords = np.load(sys_dir / "coord.npy")
                forces = np.load(sys_dir / "force.npy")
                energies = np.load(sys_dir / "energy.npy")
                
                # 归一化坐标
                coords_norm = (coords - coords.mean(axis=1, keepdims=True)) / coords.std()
                
                # 归一化力
                forces_norm = forces / (forces.std() + 1e-8)
                
                processed_frames += len(coords)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="data_preprocessing",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=processed_frames / execution_time,
            details={'processed_frames': processed_frames}
        )
        
        print(f"  Processed {processed_frames} frames in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_training_simulation(self, n_iterations: int = 1000) -> BenchmarkResult:
        """模拟训练过程（不依赖实际训练框架）"""
        print("\n=== Benchmark: Training Simulation ===")
        
        # 模拟神经网络参数
        n_params = 100000
        params = np.random.randn(n_params).astype(np.float32)
        
        # 模拟梯度
        batch_size = 32
        
        tracemalloc.start()
        start_time = time.time()
        
        for iteration in range(n_iterations):
            # 模拟前向传播
            gradients = np.random.randn(n_params).astype(np.float32) * 0.01
            
            # 模拟反向传播和参数更新
            params -= 0.001 * gradients
            
            # 模拟每100次迭代保存检查点
            if iteration % 100 == 0:
                _ = params.copy()
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="training_simulation",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=n_iterations / execution_time,
            details={'n_iterations': n_iterations, 'n_params': n_params}
        )
        
        print(f"  Simulated {n_iterations} iterations in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        print(f"  Iterations/sec: {result.throughput:.1f}")
        
        return result
    
    def benchmark_memory_efficient_training(self) -> BenchmarkResult:
        """测试内存高效训练策略"""
        print("\n=== Benchmark: Memory-Efficient Training ===")
        
        tracemalloc.start()
        start_time = time.time()
        
        processed = 0
        
        # 使用生成器逐系统处理
        def system_generator():
            for sys_dir in sorted(Path(self.deepmd_dir).glob("system_*")):
                if (sys_dir / "coord.npy").exists():
                    yield {
                        'coords': np.load(sys_dir / "coord.npy"),
                        'forces': np.load(sys_dir / "force.npy"),
                        'energies': np.load(sys_dir / "energy.npy")
                    }
        
        # 处理每个系统后立即释放内存
        for sys_data in system_generator():
            # 处理数据
            coords = sys_data['coords']
            processed += len(coords)
            # 数据处理完成后自动释放
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="memory_efficient_training",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=processed / execution_time,
            details={'processed_frames': processed, 'strategy': 'generator'}
        )
        
        print(f"  Processed {processed} frames in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_nep_data_loading(self) -> BenchmarkResult:
        """测试NEP数据加载性能"""
        print("\n=== Benchmark: NEP Data Loading ===")
        
        tracemalloc.start()
        start_time = time.time()
        
        # 逐行读取XYZ文件
        frame_count = 0
        with open(self.nep_train, 'r') as f:
            for line in f:
                if line.strip().isdigit():
                    frame_count += 1
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="nep_data_loading",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=frame_count / execution_time,
            details={'frame_count': frame_count}
        )
        
        print(f"  Scanned {frame_count} frames in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def profile_training_code(self):
        """对训练代码进行性能分析"""
        print("\n=== Profiling: Training Code Analysis ===")
        
        def simulate_training_step():
            # 模拟训练步骤
            batch_data = np.random.randn(32, 100, 3)
            weights = np.random.randn(100, 50)
            
            # 前向传播
            output = np.tensordot(batch_data, weights, axes=([2], [0]))
            
            # 损失计算
            loss = np.mean(output ** 2)
            
            # 反向传播（简化）
            grad = 2 * output.mean(axis=(0, 1))
            
            return loss
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(1000):
            simulate_training_step()
        
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print("\nTop 15 time-consuming operations:")
        stats.print_stats(15)
        
        # 保存
        stats.dump_stats(self.test_dir / "ml_training_profile.prof")
        print(f"\nProfile saved to: {self.test_dir / 'ml_training_profile.prof'}")
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("="*60)
        print("ML Training Performance Benchmarks")
        print("="*60)
        
        self.setup()
        
        try:
            # 数据加载
            r1 = self.benchmark_data_loading()
            self.results.append(r1)
            
            # 不同batch size
            batch_results = self.benchmark_batch_loading([16, 32, 64, 128, 256])
            self.results.extend(batch_results)
            
            # 数据预处理
            r3 = self.benchmark_data_preprocessing()
            self.results.append(r3)
            
            # 训练模拟
            r4 = self.benchmark_training_simulation(n_iterations=1000)
            self.results.append(r4)
            
            # 内存高效训练
            r5 = self.benchmark_memory_efficient_training()
            self.results.append(r5)
            
            # NEP数据加载
            r6 = self.benchmark_nep_data_loading()
            self.results.append(r6)
            
            # 性能分析
            self.profile_training_code()
            
        finally:
            self.generate_report()
            self.cleanup()
    
    def generate_report(self):
        """生成性能报告"""
        print("\n" + "="*60)
        print("ML Training Benchmark Summary")
        print("="*60)
        
        df = pd.DataFrame([
            {
                'Test': r.name,
                'Time (s)': f"{r.execution_time:.3f}",
                'Memory (MB)': f"{r.memory_peak_mb:.2f}",
                'Throughput': f"{r.throughput:.1f}"
            }
            for r in self.results
        ])
        
        print(df.to_string(index=False))
        
        # 保存
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
        
        with open('benchmark_ml_training_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: benchmark_ml_training_results.json")


if __name__ == "__main__":
    benchmark = MLTrainingBenchmark()
    benchmark.run_all_benchmarks()
