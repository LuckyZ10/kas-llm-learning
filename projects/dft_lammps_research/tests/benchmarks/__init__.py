"""
Benchmarks Module for DFT-LAMMPS Research Pipeline

This module provides performance benchmarking and optimization tools
for the materials simulation workflow.

Modules:
    benchmark_dft_parser: VASP/OUTCAR parsing performance tests
    benchmark_ml_training: ML potential training performance tests
    benchmark_md_simulation: MD simulation performance tests
    benchmark_screening: High-throughput screening performance tests
    optimized_dft_parser: Optimized DFT parser with Numba acceleration
    optimized_md_analysis: Optimized MD analysis with Numba acceleration

Usage:
    from benchmarks import DFTParserBenchmark
    benchmark = DFTParserBenchmark()
    benchmark.run_all_benchmarks()
"""

__version__ = "1.0.0"
__author__ = "Performance Optimization Expert"

from .benchmark_dft_parser import DFTParserBenchmark, TestDataGenerator
from .benchmark_ml_training import MLTrainingBenchmark, MLTrainingDataGenerator
from .benchmark_md_simulation import MDSimulationBenchmark, MDTestDataGenerator
from .benchmark_screening import ScreeningBenchmark, ScreeningBenchmarkDataGenerator

try:
    from .optimized_dft_parser import (
        OptimizedVASPOUTCARParser,
        BatchDFTParser,
        OptimizedParserConfig,
        compare_parsers
    )
    from .optimized_md_analysis import (
        OptimizedTrajectoryAnalyzer,
        BatchTrajectoryAnalyzer,
        AnalysisConfig,
        benchmark_analysis
    )
    OPTIMIZED_MODULES_AVAILABLE = True
except ImportError as e:
    OPTIMIZED_MODULES_AVAILABLE = False
    import warnings
    warnings.warn(f"Optimized modules not available: {e}")

__all__ = [
    # Benchmark classes
    'DFTParserBenchmark',
    'MLTrainingBenchmark',
    'MDSimulationBenchmark',
    'ScreeningBenchmark',
    # Data generators
    'TestDataGenerator',
    'MLTrainingDataGenerator',
    'MDTestDataGenerator',
    'ScreeningBenchmarkDataGenerator',
]

if OPTIMIZED_MODULES_AVAILABLE:
    __all__.extend([
        # Optimized parsers
        'OptimizedVASPOUTCARParser',
        'BatchDFTParser',
        'OptimizedParserConfig',
        # Optimized analyzers
        'OptimizedTrajectoryAnalyzer',
        'BatchTrajectoryAnalyzer',
        'AnalysisConfig',
        # Utilities
        'compare_parsers',
        'benchmark_analysis',
    ])

def run_all_benchmarks():
    """Run all available benchmarks"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Running All Performance Benchmarks")
    print("="*60)
    
    # DFT Parser
    print("\n1. DFT Parser Benchmarks")
    dft_bench = DFTParserBenchmark()
    dft_bench.setup()
    r1 = dft_bench.benchmark_single_file_parsing()
    r2 = dft_bench.benchmark_batch_parsing()
    dft_bench.cleanup()
    
    # ML Training
    print("\n2. ML Training Benchmarks")
    ml_bench = MLTrainingBenchmark()
    ml_bench.setup()
    r3 = ml_bench.benchmark_data_loading()
    ml_bench.cleanup()
    
    # MD Simulation
    print("\n3. MD Simulation Benchmarks")
    md_bench = MDSimulationBenchmark()
    md_bench.setup()
    r4 = md_bench.benchmark_trajectory_reading()
    md_bench.cleanup()
    
    # Screening
    print("\n4. Screening Benchmarks")
    screen_bench = ScreeningBenchmark()
    screen_bench.setup()
    r5 = screen_bench.benchmark_candidate_loading()
    screen_bench.cleanup()
    
    print("\n" + "="*60)
    print("All Benchmarks Complete")
    print("="*60)
    
    return {
        'dft_parser': [r1, r2],
        'ml_training': [r3],
        'md_simulation': [r4],
        'screening': [r5]
    }

if __name__ == "__main__":
    run_all_benchmarks()
