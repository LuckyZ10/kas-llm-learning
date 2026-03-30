#!/usr/bin/env python3
"""
benchmark_screening.py
======================
高通量筛选性能基准测试

测试项目:
1. 特征计算吞吐量
2. 并行筛选性能
3. 数据库操作性能
4. ML预测性能
5. 工作流编排开销
6. 大规模候选集处理

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

sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')

# 尝试导入优化库
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    execution_time: float
    memory_peak_mb: float
    throughput: float
    details: Dict = None


class MockStructure:
    """模拟结构对象"""
    def __init__(self, n_atoms: int = 50, formula: str = "Li3PS4"):
        self.n_atoms = n_atoms
        self.formula = formula
        self.elements = list(set(formula.replace('3', '').replace('4', '')))
        self.positions = np.random.randn(n_atoms, 3) * 10
        self.cell = np.diag([10, 10, 10]) + np.random.randn(3, 3)
    
    def get_volume(self):
        return abs(np.linalg.det(self.cell))


class ScreeningBenchmarkDataGenerator:
    """生成筛选测试数据"""
    
    @staticmethod
    def generate_candidate_database(output_file: str, n_candidates: int = 1000):
        """生成候选材料数据库"""
        formulas = ["Li3PS4", "Li2S", "Li7P3S11", "Li10GeP2S12", "Na3PS4"]
        
        data = []
        for i in range(n_candidates):
            formula = formulas[i % len(formulas)]
            n_atoms = np.random.randint(20, 100)
            
            candidate = {
                'material_id': f"mp-{i+1000}",
                'formula': formula,
                'n_atoms': n_atoms,
                'band_gap': np.random.random() * 5,
                'energy_above_hull': np.random.random() * 0.5,
                'formation_energy': -100 - np.random.random() * 50,
                'volume': np.random.random() * 500 + 100,
                'density': np.random.random() * 5 + 1,
            }
            data.append(candidate)
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        return output_file
    
    @staticmethod
    def generate_features(n_samples: int = 1000, n_features: int = 100):
        """生成特征矩阵"""
        return np.random.randn(n_samples, n_features)
    
    @staticmethod
    def generate_mock_structures(n_structures: int = 100, n_atoms: int = 50):
        """生成模拟结构"""
        return [MockStructure(n_atoms) for _ in range(n_structures)]


class ScreeningBenchmark:
    """高通量筛选性能测试"""
    
    def __init__(self, test_dir: str = "./benchmark_screening_data"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True)
        self.results = []
        self.candidates_df = None
        self.features = None
        
    def setup(self):
        """准备测试数据"""
        print("Setting up screening test data...")
        
        # 候选数据库
        db_file = self.test_dir / "candidates.csv"
        ScreeningBenchmarkDataGenerator.generate_candidate_database(
            str(db_file), n_candidates=5000
        )
        self.candidates_df = pd.read_csv(db_file)
        
        # 特征矩阵
        self.features = ScreeningBenchmarkDataGenerator.generate_features(
            n_samples=5000, n_features=200
        )
        
        # 结构
        self.structures = ScreeningBenchmarkDataGenerator.generate_mock_structures(
            n_structures=1000, n_atoms=50
        )
        
        print(f"Test data ready: {len(self.candidates_df)} candidates")
    
    def cleanup(self):
        """清理测试数据"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def benchmark_candidate_loading(self) -> BenchmarkResult:
        """测试候选加载性能"""
        print("\n=== Benchmark: Candidate Loading ===")
        
        tracemalloc.start()
        start_time = time.time()
        
        # 从CSV加载
        df = pd.read_csv(self.test_dir / "candidates.csv")
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="candidate_loading",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=len(df) / execution_time,
            details={'n_candidates': len(df)}
        )
        
        print(f"  Loaded {len(df)} candidates in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_candidate_filtering(self) -> BenchmarkResult:
        """测试候选筛选性能"""
        print("\n=== Benchmark: Candidate Filtering ===")
        
        tracemalloc.start()
        start_time = time.time()
        
        df = self.candidates_df.copy()
        
        # 应用多个筛选条件
        filtered = df[
            (df['band_gap'] > 2.0) &
            (df['energy_above_hull'] < 0.1) &
            (df['n_atoms'] < 100) &
            (df['density'] > 2.0)
        ]
        
        # 排序
        filtered = filtered.sort_values('formation_energy', ascending=True)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="candidate_filtering",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=len(df) / execution_time,
            details={
                'input_candidates': len(df),
                'filtered_candidates': len(filtered)
            }
        )
        
        print(f"  Filtered {len(df)} to {len(filtered)} candidates in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_feature_calculation(self) -> BenchmarkResult:
        """测试特征计算性能"""
        print("\n=== Benchmark: Feature Calculation ===")
        
        def calculate_basic_features(structure):
            """计算基本特征"""
            features = {
                'n_atoms': structure.n_atoms,
                'volume': structure.get_volume(),
                'density': structure.n_atoms / structure.get_volume(),
            }
            
            # 坐标统计
            positions = structure.positions
            features['x_mean'] = positions[:, 0].mean()
            features['y_mean'] = positions[:, 1].mean()
            features['z_mean'] = positions[:, 2].mean()
            features['x_std'] = positions[:, 0].std()
            features['y_std'] = positions[:, 1].std()
            features['z_std'] = positions[:, 2].std()
            
            return features
        
        tracemalloc.start()
        start_time = time.time()
        
        all_features = []
        for struct in self.structures:
            feats = calculate_basic_features(struct)
            all_features.append(feats)
        
        features_df = pd.DataFrame(all_features)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="feature_calculation",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=len(self.structures) / execution_time,
            details={'n_structures': len(self.structures), 'n_features': len(features_df.columns)}
        )
        
        print(f"  Calculated features for {len(self.structures)} structures in {execution_time:.3f}s")
        print(f"  Features per structure: {len(features_df.columns)}")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_parallel_feature_calculation(self, n_workers: int = 4) -> BenchmarkResult:
        """测试并行特征计算"""
        print(f"\n=== Benchmark: Parallel Feature Calculation ({n_workers} workers) ===")
        
        def calculate_features_chunk(structures_chunk):
            """计算一批结构的特征"""
            results = []
            for struct in structures_chunk:
                feats = {
                    'n_atoms': struct.n_atoms,
                    'volume': struct.get_volume(),
                    'density': struct.n_atoms / struct.get_volume(),
                    'x_mean': struct.positions[:, 0].mean(),
                    'y_mean': struct.positions[:, 1].mean(),
                    'z_mean': struct.positions[:, 2].mean(),
                }
                results.append(feats)
            return results
        
        # 分块
        chunk_size = len(self.structures) // n_workers
        chunks = [
            self.structures[i:i+chunk_size]
            for i in range(0, len(self.structures), chunk_size)
        ]
        
        tracemalloc.start()
        start_time = time.time()
        
        from concurrent.futures import ProcessPoolExecutor
        
        all_features = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(calculate_features_chunk, chunk) for chunk in chunks]
            for future in futures:
                all_features.extend(future.result())
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name=f"parallel_feature_calculation_{n_workers}",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=len(self.structures) / execution_time,
            details={'n_workers': n_workers}
        )
        
        print(f"  Calculated features with {n_workers} workers in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_ml_prediction(self) -> BenchmarkResult:
        """测试ML预测性能"""
        print("\n=== Benchmark: ML Prediction ===")
        
        # 模拟ML预测
        def predict_conductivity(features_batch):
            """简化的电导率预测"""
            # 模拟神经网络前向传播
            weights = np.random.randn(features_batch.shape[1])
            logits = np.dot(features_batch, weights)
            return 1 / (1 + np.exp(-logits))  # sigmoid
        
        tracemalloc.start()
        start_time = time.time()
        
        # 批量预测
        batch_size = 100
        predictions = []
        
        for i in range(0, len(self.features), batch_size):
            batch = self.features[i:i+batch_size]
            pred = predict_conductivity(batch)
            predictions.extend(pred)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="ml_prediction",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=len(self.features) / execution_time,
            details={'n_predictions': len(predictions), 'batch_size': batch_size}
        )
        
        print(f"  Made {len(predictions)} predictions in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_batch_database_operations(self) -> BenchmarkResult:
        """测试批量数据库操作"""
        print("\n=== Benchmark: Batch Database Operations ===")
        
        tracemalloc.start()
        start_time = time.time()
        
        # 模拟批量插入
        batch_size = 100
        n_records = 10000
        
        records_inserted = 0
        for i in range(0, n_records, batch_size):
            # 模拟批量插入
            batch = [
                {
                    'id': j,
                    'formula': f"Li{j%10}S{j%5}",
                    'property': np.random.random()
                }
                for j in range(i, min(i+batch_size, n_records))
            ]
            records_inserted += len(batch)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="batch_database_operations",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=records_inserted / execution_time,
            details={'records_inserted': records_inserted, 'batch_size': batch_size}
        )
        
        print(f"  Inserted {records_inserted} records in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_workflow_orchestration(self) -> BenchmarkResult:
        """测试工作流编排性能"""
        print("\n=== Benchmark: Workflow Orchestration ===")
        
        # 模拟工作流步骤
        def step1_data_loading():
            time.sleep(0.01)
            return pd.DataFrame({'a': range(100)})
        
        def step2_feature_calculation(df):
            time.sleep(0.02)
            return df.assign(b=df['a'] * 2)
        
        def step3_filtering(df):
            time.sleep(0.01)
            return df[df['b'] > 50]
        
        def step4_prediction(df):
            time.sleep(0.02)
            return df.assign(pred=np.random.random(len(df)))
        
        tracemalloc.start()
        start_time = time.time()
        
        # 运行多批次工作流
        n_workflows = 50
        for i in range(n_workflows):
            df = step1_data_loading()
            df = step2_feature_calculation(df)
            df = step3_filtering(df)
            df = step4_prediction(df)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="workflow_orchestration",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=n_workflows / execution_time,
            details={'n_workflows': n_workflows}
        )
        
        print(f"  Executed {n_workflows} workflows in {execution_time:.3f}s")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def benchmark_large_scale_screening(self) -> BenchmarkResult:
        """测试大规模筛选"""
        print("\n=== Benchmark: Large-Scale Screening (10k candidates) ===")
        
        # 生成更大的数据集
        large_features = np.random.randn(10000, 200)
        
        def screen_candidates(features, threshold=0.5):
            """筛选候选"""
            # 计算分数
            scores = np.sum(features ** 2, axis=1)
            # 筛选
            mask = scores > threshold
            return mask
        
        tracemalloc.start()
        start_time = time.time()
        
        # 多轮筛选
        selected_indices = []
        for round_idx in range(5):
            mask = screen_candidates(large_features)
            selected = np.where(mask)[0]
            selected_indices.extend(selected)
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name="large_scale_screening",
            execution_time=execution_time,
            memory_peak_mb=peak / 1024 / 1024,
            throughput=50000 / execution_time,  # 5 rounds * 10k
            details={
                'n_candidates': 10000,
                'n_rounds': 5,
                'selected_count': len(selected_indices)
            }
        )
        
        print(f"  Screened 50k candidate-rounds in {execution_time:.3f}s")
        print(f"  Selected {len(selected_indices)} candidates")
        print(f"  Peak memory: {result.memory_peak_mb:.2f} MB")
        
        return result
    
    def profile_screening_code(self):
        """对筛选代码进行性能分析"""
        print("\n=== Profiling: Screening Code Analysis ===")
        
        def simulate_screening_pipeline():
            """模拟筛选流程"""
            # 数据加载
            df = pd.DataFrame({
                'id': range(1000),
                'band_gap': np.random.random(1000) * 5,
                'energy': np.random.random(1000) * 100
            })
            
            # 筛选
            filtered = df[df['band_gap'] > 2.0]
            
            # 特征计算
            filtered = filtered.copy()
            filtered['score'] = filtered['band_gap'] * 0.5 - filtered['energy'] * 0.01
            
            # 排序
            ranked = filtered.sort_values('score', ascending=False)
            
            return ranked
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(100):
            simulate_screening_pipeline()
        
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print("\nTop 15 time-consuming operations:")
        stats.print_stats(15)
        
        stats.dump_stats(self.test_dir / "screening_profile.prof")
        print(f"\nProfile saved to: {self.test_dir / 'screening_profile.prof'}")
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("="*60)
        print("High-Throughput Screening Performance Benchmarks")
        print("="*60)
        
        self.setup()
        
        try:
            # 候选加载
            r1 = self.benchmark_candidate_loading()
            self.results.append(r1)
            
            # 候选筛选
            r2 = self.benchmark_candidate_filtering()
            self.results.append(r2)
            
            # 特征计算
            r3 = self.benchmark_feature_calculation()
            self.results.append(r3)
            
            # 并行特征计算
            r4 = self.benchmark_parallel_feature_calculation(n_workers=4)
            self.results.append(r4)
            
            # ML预测
            r5 = self.benchmark_ml_prediction()
            self.results.append(r5)
            
            # 数据库操作
            r6 = self.benchmark_batch_database_operations()
            self.results.append(r6)
            
            # 工作流编排
            r7 = self.benchmark_workflow_orchestration()
            self.results.append(r7)
            
            # 大规模筛选
            r8 = self.benchmark_large_scale_screening()
            self.results.append(r8)
            
            # 性能分析
            self.profile_screening_code()
            
        finally:
            self.generate_report()
            self.cleanup()
    
    def generate_report(self):
        """生成性能报告"""
        print("\n" + "="*60)
        print("Screening Benchmark Summary")
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
        
        with open('benchmark_screening_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: benchmark_screening_results.json")


if __name__ == "__main__":
    benchmark = ScreeningBenchmark()
    benchmark.run_all_benchmarks()
