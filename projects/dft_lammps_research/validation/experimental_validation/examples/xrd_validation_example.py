"""
Example: XRD Validation
=======================
XRD数据验证示例

展示如何使用实验验证模块比较实验XRD和模拟XRD数据
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入实验验证模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimental_validation import (
    XRDConnector,
    XRDComparator,
    ValidationWorkflow,
    ValidationConfig,
)


def create_synthetic_xrd_data(output_dir: str):
    """
    创建合成XRD数据用于演示
    
    生成模拟的实验和理论XRD图谱
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2θ范围
    two_theta = np.linspace(5, 90, 850)
    
    # 模拟LiFePO4的XRD峰
    peak_positions = [17.1, 20.8, 25.6, 29.7, 32.3, 36.4, 39.2, 41.3, 
                     44.2, 47.5, 51.7, 54.1, 57.5, 60.3, 63.8, 67.2]
    
    # 理论图谱（模拟）
    sim_intensity = np.zeros_like(two_theta)
    for pos in peak_positions:
        sim_intensity += np.exp(-((two_theta - pos) / 0.3)**2)
    
    # 实验图谱（添加噪声和峰位移）
    exp_intensity = np.zeros_like(two_theta)
    for pos in peak_positions:
        # 添加轻微峰位移和宽度变化
        shifted_pos = pos + np.random.normal(0, 0.1)
        width = 0.3 + np.random.uniform(-0.05, 0.05)
        exp_intensity += np.exp(-((two_theta - shifted_pos) / width)**2)
    
    # 添加背景噪声
    background = 0.05 + 0.02 * np.sin(two_theta / 20)
    exp_intensity += background
    
    # 添加随机噪声
    noise = np.random.normal(0, 0.02, len(two_theta))
    exp_intensity += noise
    
    # 归一化
    sim_intensity = sim_intensity / np.max(sim_intensity)
    exp_intensity = exp_intensity / np.max(exp_intensity)
    
    # 保存数据
    sim_data = np.column_stack([two_theta, sim_intensity])
    exp_data = np.column_stack([two_theta, exp_intensity])
    
    np.savetxt(output_dir / "simulated_xrd.csv", sim_data, 
               delimiter=',', header='two_theta,intensity', comments='')
    np.savetxt(output_dir / "experimental_xrd.csv", exp_data,
               delimiter=',', header='two_theta,intensity', comments='')
    
    print(f"Created synthetic XRD data in {output_dir}")
    return str(output_dir / "experimental_xrd.csv"), str(output_dir / "simulated_xrd.csv")


def example_basic_xrd_comparison():
    """基础XRD对比示例"""
    print("\n" + "="*60)
    print("Example 1: Basic XRD Comparison")
    print("="*60)
    
    # 创建合成数据
    exp_file, sim_file = create_synthetic_xrd_data("./example_data/xrd")
    
    # 加载数据
    connector = XRDConnector()
    exp_data = connector.read(exp_file)
    sim_data = connector.read(sim_file)
    
    print(f"\nLoaded experimental data: {len(exp_data.raw_data)} points")
    print(f"Loaded simulated data: {len(sim_data.raw_data)} points")
    
    # 比较图谱
    comparator = XRDComparator()
    results = comparator.compare(exp_data, sim_data, methods=['all'])
    
    print("\nComparison Results:")
    print("-" * 40)
    for metric, value in results.items():
        print(f"{metric:25s}: {value:.4f}")
    
    # 查找峰
    exp_peaks = connector.find_peaks(exp_data, prominence=0.1)
    sim_peaks = connector.find_peaks(sim_data, prominence=0.1)
    
    print(f"\nFound {len(exp_peaks)} peaks in experimental data")
    print(f"Found {len(sim_peaks)} peaks in simulated data")
    
    # 显示前5个峰
    print("\nTop 5 Experimental Peaks:")
    for i, peak in enumerate(exp_peaks[:5]):
        print(f"  Peak {i+1}: 2θ = {peak['two_theta']:.2f}°, Intensity = {peak['intensity']:.3f}")
    
    # 计算差分图谱
    diff_profile = comparator.calculate_difference_profile(exp_data, sim_data)
    
    # 保存结果
    output_dir = Path("./example_results/xrd")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(output_dir / "difference_profile.csv", diff_profile,
               delimiter=',', header='two_theta,exp,sim,diff')
    
    print(f"\nResults saved to {output_dir}")


def example_validation_workflow():
    """验证工作流示例"""
    print("\n" + "="*60)
    print("Example 2: Validation Workflow")
    print("="*60)
    
    # 创建数据
    exp_file, sim_file = create_synthetic_xrd_data("./example_data/xrd")
    
    # 配置验证工作流
    config = ValidationConfig(
        data_type='xrd',
        output_dir='./example_results/xrd_workflow',
        generate_report=True,
        generate_plots=True,
        report_format='html',
        rmse_threshold=0.2,
        r2_threshold=0.85,
    )
    
    # 创建并运行工作流
    workflow = ValidationWorkflow(config)
    
    print("\nLoading data...")
    workflow.load_experimental_data(exp_file)
    workflow.load_computational_data(sim_file)
    
    print("Running validation...")
    result = workflow.run_validation()
    
    print("\nValidation Results:")
    print("-" * 40)
    print(f"Success: {result.success}")
    print(f"Data Type: {result.data_type}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
    
    # 生成报告
    report_path = workflow.generate_report()
    print(f"\nReport saved to: {report_path}")


def example_batch_validation():
    """批量验证示例"""
    print("\n" + "="*60)
    print("Example 3: Batch Validation")
    print("="*60)
    
    from experimental_validation import BatchValidator, BatchValidationConfig
    
    # 创建多组数据
    data_dir = Path("./example_data/xrd_batch")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(3):
        # 为每个系统创建略有不同的数据
        exp_dir = data_dir / "experimental"
        sim_dir = data_dir / "simulated"
        exp_dir.mkdir(exist_ok=True)
        sim_dir.mkdir(exist_ok=True)
        
        # 创建数据（简化版）
        two_theta = np.linspace(5, 90, 850)
        
        # 实验数据（带噪声）
        exp_intensity = np.random.rand(len(two_theta)) * 0.1
        exp_intensity[100] = 1.0  # 主峰
        exp_intensity += np.random.normal(0, 0.02, len(two_theta))
        
        # 模拟数据（较平滑）
        sim_intensity = np.zeros_like(two_theta)
        sim_intensity[100] = 0.95
        
        np.savetxt(exp_dir / f"system_{i+1}.csv",
                  np.column_stack([two_theta, exp_intensity]),
                  delimiter=',', header='two_theta,intensity', comments='')
        np.savetxt(sim_dir / f"system_{i+1}.csv",
                  np.column_stack([two_theta, sim_intensity]),
                  delimiter=',', header='two_theta,intensity', comments='')
    
    # 配置批量验证
    config = BatchValidationConfig(
        exp_data_dir=str(exp_dir),
        sim_data_dir=str(sim_dir),
        data_type='xrd',
        output_dir='./example_results/batch_validation',
        individual_reports=True,
        summary_report=True,
        n_workers=1,
    )
    
    # 运行批量验证
    validator = BatchValidator(config)
    results = validator.run_batch()
    
    print("\nBatch Validation Summary:")
    print("-" * 40)
    summary = results.summary
    print(f"Total validations: {summary.get('total_validations', 0)}")
    print(f"Success: {summary.get('success', 0)}")
    print(f"Partial: {summary.get('partial', 0)}")
    print(f"Failure: {summary.get('failure', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0):.1f}%")
    
    # 生成汇总图表
    validator.generate_summary_plots()
    print(f"\nResults saved to {config.output_dir}")


def example_structure_comparison():
    """结构比较示例"""
    print("\n" + "="*60)
    print("Example 4: Structure Comparison")
    print("="*60)
    
    from experimental_validation import StructureComparator
    
    try:
        from pymatgen.core import Structure, Lattice
        
        # 创建两个相似的结构
        lattice1 = Lattice.cubic(4.0)
        structure1 = Structure(
            lattice1,
            species=['Li', 'Fe', 'P', 'O', 'O', 'O', 'O'],
            coords=[
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0.75, 0.25, 0.25],
                [0.25, 0.75, 0.25],
                [0.25, 0.25, 0.75],
                [0.75, 0.75, 0.75],
            ]
        )
        
        # 第二个结构（略有畸变）
        lattice2 = Lattice.cubic(4.05)  # 稍大的晶格
        structure2 = Structure(
            lattice2,
            species=['Li', 'Fe', 'P', 'O', 'O', 'O', 'O'],
            coords=[
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0.75, 0.25, 0.25],
                [0.25, 0.75, 0.25],
                [0.25, 0.25, 0.75],
                [0.75, 0.75, 0.75],
            ]
        )
        
        # 比较结构
        comparator = StructureComparator()
        results = comparator.compare_structures(structure1, structure2)
        
        print("\nStructure Comparison Results:")
        print("-" * 40)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:.4f}")
            else:
                print(f"{key:30s}: {value}")
        
        # 基于XRD的相似度
        xrd_sim = comparator.calculate_xrd_similarity(structure1, structure2)
        print(f"\nXRD-based similarity: {xrd_sim:.4f}")
        
    except ImportError:
        print("pymatgen not available. Install it to run this example.")


if __name__ == "__main__":
    # 运行所有示例
    example_basic_xrd_comparison()
    example_validation_workflow()
    example_batch_validation()
    example_structure_comparison()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
