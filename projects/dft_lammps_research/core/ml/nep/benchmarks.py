#!/usr/bin/env python3
"""
NEP Training Benchmarks
=======================
性能基准测试
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')

from nep_training import (
    NEPDataConfig, NEPModelConfig, NEPTrainingConfig,
    NEPDataset, NEPDataPreparer, DataAugmenter,
    NEPTrainerV2, EnsembleTrainer, EnsembleConfig,
    GPUMemoryOptimizer, DataLoaderOptimizer, TrainingSpeedOptimizer,
    InferenceOptimizer
)

from ase import Atoms
from ase.build import bulk, molecule


def create_test_dataset(n_structures: int = 100, n_atoms: int = 64) -> NEPDataset:
    """创建测试数据集"""
    frames = []
    
    for i in range(n_structures):
        # 创建Si晶体并添加随机扰动
        atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
        
        # 添加随机扰动
        positions = atoms.get_positions()
        noise = np.random.randn(*positions.shape) * 0.05
        atoms.set_positions(positions + noise)
        
        # 添加模拟的能量和力
        atoms.calc = None
        atoms.info['energy'] = -4.5 * len(atoms) + np.random.randn() * 0.1
        atoms.arrays['forces'] = np.random.randn(len(atoms), 3) * 0.01
        
        from nep_training.data import NEPFrame
        frame = NEPFrame(
            atoms=atoms,
            energy=atoms.info['energy'],
            forces=atoms.arrays['forces']
        )
        frames.append(frame)
    
    return NEPDataset(frames=frames)


def benchmark_data_loading():
    """测试数据加载性能"""
    print("\n" + "=" * 60)
    print("Benchmark: Data Loading")
    print("=" * 60)
    
    dataset = create_test_dataset(n_structures=100)
    
    # 测试保存速度
    start = time.time()
    dataset.save_xyz("/tmp/test_benchmark.xyz")
    save_time = time.time() - start
    
    # 测试加载速度
    start = time.time()
    loaded = NEPDataset(xyz_file="/tmp/test_benchmark.xyz")
    load_time = time.time() - start
    
    print(f"Dataset size: {len(dataset)} structures")
    print(f"Save time: {save_time:.3f}s ({len(dataset)/save_time:.1f} structures/sec)")
    print(f"Load time: {load_time:.3f}s ({len(dataset)/load_time:.1f} structures/sec)")
    
    # 清理
    os.remove("/tmp/test_benchmark.xyz")
    
    return {
        'save_throughput': len(dataset) / save_time,
        'load_throughput': len(dataset) / load_time
    }


def benchmark_data_augmentation():
    """测试数据增强性能"""
    print("\n" + "=" * 60)
    print("Benchmark: Data Augmentation")
    print("=" * 60)
    
    dataset = create_test_dataset(n_structures=100)
    
    augmenter = DataAugmenter(
        rotation=True,
        translation=False,
        noise=True,
        noise_std=0.01
    )
    
    start = time.time()
    augmented = augmenter.augment_dataset(dataset, n_augment=2, augment_ratio=0.5)
    aug_time = time.time() - start
    
    print(f"Original size: {len(dataset)}")
    print(f"Augmented size: {len(augmented)}")
    print(f"Augmentation time: {aug_time:.3f}s")
    print(f"Speed: {len(dataset)/aug_time:.1f} structures/sec")
    
    return {
        'original_size': len(dataset),
        'augmented_size': len(augmented),
        'time': aug_time
    }


def benchmark_model_configuration():
    """测试模型配置生成"""
    print("\n" + "=" * 60)
    print("Benchmark: Model Configuration")
    print("=" * 60)
    
    from nep_training.core import NEP_PRESETS
    
    for preset_name in ['fast', 'balanced', 'accurate', 'light']:
        preset = NEP_PRESETS[preset_name]
        
        config = NEPModelConfig(
            type_list=["Si", "O"],
            **{k: v for k, v in preset.items() if k in NEPModelConfig.__dataclass_fields__}
        )
        
        nep_dict = config.to_nep_in_dict()
        
        print(f"\nPreset: {preset_name}")
        print(f"  Description: {preset['description']}")
        print(f"  Parameters: {len(nep_dict)}")
        print(f"  Expected memory: {config.neuron * 1000:.0f} KB (est.)")


def benchmark_memory_optimizer():
    """测试内存优化器"""
    print("\n" + "=" * 60)
    print("Benchmark: GPU Memory Optimizer")
    print("=" * 60)
    
    optimizer = GPUMemoryOptimizer(gpu_id=0)
    
    # 获取GPU统计
    stats = optimizer.get_gpu_stats()
    
    if stats:
        print(f"GPU {stats.gpu_id}:")
        print(f"  Memory: {stats.memory_used_mb}/{stats.memory_total_mb} MB")
        print(f"  Utilization: {stats.utilization_percent:.1f}%")
        print(f"  Temperature: {stats.temperature_c:.0f}°C")
        
        # 测试batch size推荐
        recommended = optimizer.recommend_batch_size(1000)
        print(f"\nRecommended batch size: {recommended}")
        
        return {
            'memory_used_mb': stats.memory_used_mb,
            'utilization': stats.utilization_percent,
            'recommended_batch': recommended
        }
    else:
        print("GPU not available or nvidia-smi not found")
        return {}


def benchmark_data_loader():
    """测试数据加载优化"""
    print("\n" + "=" * 60)
    print("Benchmark: Data Loader Optimization")
    print("=" * 60)
    
    optimizer = DataLoaderOptimizer()
    
    # 系统优化配置
    config = optimizer.optimize_for_system()
    print(f"Optimized configuration:")
    print(f"  num_workers: {config['num_workers']}")
    print(f"  prefetch_size: {config['prefetch_size']}")
    print(f"  pin_memory: {config['pin_memory']}")
    
    return config


def benchmark_inference():
    """测试推理性能"""
    print("\n" + "=" * 60)
    print("Benchmark: Inference (Simulated)")
    print("=" * 60)
    
    # 创建测试结构
    structures = []
    for i in range(100):
        atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
        structures.append(atoms)
    
    # 模拟推理
    start = time.time()
    
    for atoms in structures:
        # 模拟NEP计算
        _ = np.random.randn(len(atoms))
    
    elapsed = time.time() - start
    
    print(f"Structures: {len(structures)}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Throughput: {len(structures)/elapsed:.1f} structures/sec")
    print(f"Latency: {elapsed/len(structures)*1000:.2f} ms/structure")
    
    return {
        'throughput': len(structures) / elapsed,
        'latency_ms': elapsed / len(structures) * 1000
    }


def benchmark_dataset_statistics():
    """测试数据集统计计算"""
    print("\n" + "=" * 60)
    print("Benchmark: Dataset Statistics")
    print("=" * 60)
    
    dataset = create_test_dataset(n_structures=1000)
    
    start = time.time()
    stats = dataset.get_statistics()
    elapsed = time.time() - start
    
    print(f"Dataset size: {stats['n_frames']} frames")
    print(f"Elements: {stats['elements']}")
    print(f"Energy mean: {stats['energy']['total_mean']:.3f} eV")
    print(f"Energy std: {stats['energy']['total_std']:.3f} eV")
    print(f"Force max abs: {stats['forces']['max_abs']:.3f} eV/Å")
    print(f"\nStatistics computed in {elapsed*1000:.1f} ms")
    
    return stats


def run_all_benchmarks():
    """运行所有基准测试"""
    print("\n" + "=" * 60)
    print("NEP Training Enhanced - Performance Benchmarks")
    print("=" * 60)
    
    results = {}
    
    results['data_loading'] = benchmark_data_loading()
    results['data_augmentation'] = benchmark_data_augmentation()
    results['model_configuration'] = benchmark_model_configuration()
    results['memory_optimizer'] = benchmark_memory_optimizer()
    results['data_loader'] = benchmark_data_loader()
    results['inference'] = benchmark_inference()
    results['dataset_statistics'] = benchmark_dataset_statistics()
    
    # 总结
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    print(f"\nData Operations:")
    print(f"  Load throughput: {results['data_loading']['load_throughput']:.1f} structures/sec")
    print(f"  Augmentation speed: {results['data_augmentation']['time']:.3f}s for {results['data_augmentation']['augmented_size']} structures")
    
    print(f"\nInference:")
    print(f"  Throughput: {results['inference']['throughput']:.1f} structures/sec")
    print(f"  Latency: {results['inference']['latency_ms']:.2f} ms/structure")
    
    if results['memory_optimizer']:
        print(f"\nGPU:")
        print(f"  Memory used: {results['memory_optimizer']['memory_used_mb']} MB")
        print(f"  Utilization: {results['memory_optimizer']['utilization']:.1f}%")
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()
