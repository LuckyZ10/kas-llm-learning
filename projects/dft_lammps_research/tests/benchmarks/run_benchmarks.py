#!/usr/bin/env python3
"""
run_benchmarks.py
=================
Benchmark runner script - 运行所有性能基准测试

用法:
    python run_benchmarks.py [--quick] [--module MODULE]

选项:
    --quick         快速模式（只运行基本测试）
    --module        指定要运行的模块 (dft|ml|md|screening)
    --compare       比较优化前后的性能
"""

import argparse
import sys
import time
import json
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))


def run_dft_benchmarks(quick=False):
    """运行DFT解析器基准测试"""
    print("\n" + "="*60)
    print("Running DFT Parser Benchmarks")
    print("="*60)
    
    try:
        from benchmark_dft_parser import DFTParserBenchmark
        
        benchmark = DFTParserBenchmark()
        benchmark.setup(n_files=5 if quick else 10, 
                       n_frames_per_file=50 if quick else 100,
                       n_atoms=30 if quick else 50)
        
        results = []
        
        # 单文件解析
        r1 = benchmark.benchmark_single_file_parsing()
        results.append(r1)
        
        # 批量解析
        r2 = benchmark.benchmark_batch_parsing()
        results.append(r2)
        
        if not quick:
            # 并行解析
            r3 = benchmark.benchmark_parallel_parsing(n_workers=4)
            results.append(r3)
            
            # 内存优化
            r4 = benchmark.benchmark_memory_efficient_parsing()
            results.append(r4)
        
        benchmark.results = results
        benchmark.generate_report()
        benchmark.cleanup()
        
        return True
        
    except Exception as e:
        print(f"DFT benchmarks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_ml_benchmarks(quick=False):
    """运行ML训练基准测试"""
    print("\n" + "="*60)
    print("Running ML Training Benchmarks")
    print("="*60)
    
    try:
        from benchmark_ml_training import MLTrainingBenchmark
        
        benchmark = MLTrainingBenchmark()
        benchmark.setup()
        
        results = []
        
        # 数据加载
        r1 = benchmark.benchmark_data_loading()
        results.append(r1)
        
        # Batch size测试
        batch_sizes = [16, 32] if quick else [16, 32, 64, 128]
        batch_results = benchmark.benchmark_batch_loading(batch_sizes)
        results.extend(batch_results)
        
        benchmark.results = results
        benchmark.generate_report()
        benchmark.cleanup()
        
        return True
        
    except Exception as e:
        print(f"ML benchmarks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_md_benchmarks(quick=False):
    """运行MD模拟基准测试"""
    print("\n" + "="*60)
    print("Running MD Simulation Benchmarks")
    print("="*60)
    
    try:
        from benchmark_md_simulation import MDSimulationBenchmark
        
        benchmark = MDSimulationBenchmark()
        benchmark.setup()
        
        results = []
        
        # 轨迹读取
        r1 = benchmark.benchmark_trajectory_reading()
        results.append(r1)
        
        # RDF计算
        r2 = benchmark.benchmark_rdf_calculation()
        results.append(r2)
        
        if not quick:
            # 优化的RDF
            r3_opt = benchmark.benchmark_rdf_optimized()
            if r3_opt:
                results.append(r3_opt)
            
            # MSD计算
            r4 = benchmark.benchmark_msd_calculation()
            results.append(r4)
        
        benchmark.results = [r for r in results if r]
        benchmark.generate_report()
        benchmark.cleanup()
        
        return True
        
    except Exception as e:
        print(f"MD benchmarks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_screening_benchmarks(quick=False):
    """运行高通量筛选基准测试"""
    print("\n" + "="*60)
    print("Running Screening Benchmarks")
    print("="*60)
    
    try:
        from benchmark_screening import ScreeningBenchmark
        
        benchmark = ScreeningBenchmark()
        benchmark.setup()
        
        results = []
        
        # 候选加载
        r1 = benchmark.benchmark_candidate_loading()
        results.append(r1)
        
        # 候选筛选
        r2 = benchmark.benchmark_candidate_filtering()
        results.append(r2)
        
        # 特征计算
        r3 = benchmark.benchmark_feature_calculation()
        results.append(r3)
        
        if not quick:
            # ML预测
            r4 = benchmark.benchmark_ml_prediction()
            results.append(r4)
        
        benchmark.results = results
        benchmark.generate_report()
        benchmark.cleanup()
        
        return True
        
    except Exception as e:
        print(f"Screening benchmarks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_optimized_comparison():
    """运行优化前后对比"""
    print("\n" + "="*60)
    print("Running Optimized vs Original Comparison")
    print("="*60)
    
    try:
        from optimized_dft_parser import compare_parsers
        from optimized_md_analysis import benchmark_analysis
        
        # 这里需要实际的测试文件
        print("Note: Requires actual VASP/trajectory files for comparison")
        print("Optimization modules loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        return False


def generate_summary_report(all_results):
    """生成综合报告"""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    total_tests = sum(len(r) for r in all_results.values() if isinstance(r, list))
    print(f"\nTotal tests run: {total_tests}")
    
    for module, results in all_results.items():
        if isinstance(results, list):
            print(f"\n{module.upper()}:")
            for r in results:
                print(f"  - {r.get('name', 'unknown')}: "
                      f"{r.get('throughput', 0):.1f} items/s")
    
    # 保存综合报告
    summary = {
        'timestamp': time.time(),
        'results': all_results
    }
    
    with open('benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: benchmark_summary.json")


def main():
    parser = argparse.ArgumentParser(description='Run performance benchmarks')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--module', choices=['dft', 'ml', 'md', 'screening', 'all'],
                       default='all', help='Module to benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare optimized vs original')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DFT-MD-LAMMPS Performance Benchmark Suite")
    print("="*60)
    
    all_results = {}
    
    if args.module in ['dft', 'all']:
        success = run_dft_benchmarks(args.quick)
        if success:
            try:
                with open('benchmark_dft_parser_results.json') as f:
                    all_results['dft'] = json.load(f)
            except:
                pass
    
    if args.module in ['ml', 'all']:
        success = run_ml_benchmarks(args.quick)
        if success:
            try:
                with open('benchmark_ml_training_results.json') as f:
                    all_results['ml'] = json.load(f)
            except:
                pass
    
    if args.module in ['md', 'all']:
        success = run_md_benchmarks(args.quick)
        if success:
            try:
                with open('benchmark_md_simulation_results.json') as f:
                    all_results['md'] = json.load(f)
            except:
                pass
    
    if args.module in ['screening', 'all']:
        success = run_screening_benchmarks(args.quick)
        if success:
            try:
                with open('benchmark_screening_results.json') as f:
                    all_results['screening'] = json.load(f)
            except:
                pass
    
    if args.compare:
        run_optimized_comparison()
    
    # 生成综合报告
    if all_results:
        generate_summary_report(all_results)
    
    print("\n" + "="*60)
    print("Benchmarking Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
