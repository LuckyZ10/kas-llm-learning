"""
Example: Electrochemical Validation
====================================
电化学数据验证示例

展示如何比较实验电化学数据与模拟预测
"""

import numpy as np
from pathlib import Path

# 导入实验验证模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimental_validation import (
    ElectrochemicalConnector,
    ElectrochemicalComparator,
    ValidationWorkflow,
    ValidationConfig,
)


def create_synthetic_gcd_data(output_dir: str):
    """
    创建合成充放电数据
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 电压范围（磷酸铁锂的典型电压）
    voltage = np.linspace(2.5, 4.2, 200)
    
    # 理论容量曲线
    # 模拟锂化/脱锂过程
    capacity_theory = np.zeros_like(voltage)
    
    # 脱锂平台 (3.4-3.5 V)
    plateau_mask = (voltage > 3.4) & (voltage < 3.5)
    capacity_theory[plateau_mask] = np.linspace(0, 150, np.sum(plateau_mask))
    
    # 低电压区
    low_v_mask = voltage <= 3.4
    capacity_theory[low_v_mask] = 150 * (voltage[low_v_mask] - 2.5) / 0.9
    
    # 高电压区
    high_v_mask = voltage >= 3.5
    capacity_theory[high_v_mask] = 150 + 20 * (voltage[high_v_mask] - 3.5) / 0.7
    
    # 实验数据（添加噪声和衰减）
    capacity_exp = capacity_theory * 0.95  # 5%容量衰减
    capacity_exp += np.random.normal(0, 2, len(voltage))  # 添加噪声
    capacity_exp = np.clip(capacity_exp, 0, 200)
    
    # 创建数据表
    # 假设电流为负表示充电
    current = np.ones(len(voltage)) * (-0.1)  # -0.1 C充电
    time = np.cumsum(np.ones(len(voltage)) * 36)  # 假设每步36秒
    
    sim_data = np.column_stack([time, voltage, current, capacity_theory])
    exp_data = np.column_stack([time, voltage, current, capacity_exp])
    
    # 保存为CSV
    header = 'time_s,voltage_V,current_mA,capacity_mAh_g'
    np.savetxt(output_dir / "simulated_gcd.csv", sim_data,
               delimiter=',', header=header, comments='')
    np.savetxt(output_dir / "experimental_gcd.csv", exp_data,
               delimiter=',', header=header, comments='')
    
    print(f"Created synthetic GCD data in {output_dir}")
    return str(output_dir / "experimental_gcd.csv"), str(output_dir / "simulated_gcd.csv")


def create_synthetic_cv_data(output_dir: str):
    """
    创建合成CV数据
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 电压扫描
    voltage = np.concatenate([
        np.linspace(2.5, 4.2, 200),  # 正向扫描
        np.linspace(4.2, 2.5, 200),  # 反向扫描
    ])
    
    # 模拟电流响应
    current_theory = np.zeros_like(voltage)
    
    # 氧化峰（正向扫描，约3.5V）
    forward_mask = np.arange(len(voltage)) < 200
    oxidation_mask = forward_mask & (voltage > 3.4) & (voltage < 3.6)
    current_theory[oxidation_mask] += 0.5 * np.exp(-((voltage[oxidation_mask] - 3.5) / 0.05)**2)
    
    # 还原峰（反向扫描，约3.4V）
    reverse_mask = np.arange(len(voltage)) >= 200
    reduction_mask = reverse_mask & (voltage > 3.3) & (voltage < 3.5)
    current_theory[reduction_mask] -= 0.5 * np.exp(-((voltage[reduction_mask] - 3.4) / 0.05)**2)
    
    # 双电层电流
    current_theory += 0.1 * (voltage - 3.3)
    
    # 实验数据（添加噪声）
    current_exp = current_theory + np.random.normal(0, 0.02, len(voltage))
    
    # 保存
    sim_data = np.column_stack([voltage, current_theory])
    exp_data = np.column_stack([voltage, current_exp])
    
    header = 'potential_V,current_mA'
    np.savetxt(output_dir / "simulated_cv.csv", sim_data,
               delimiter=',', header=header, comments='')
    np.savetxt(output_dir / "experimental_cv.csv", exp_data,
               delimiter=',', header=header, comments='')
    
    print(f"Created synthetic CV data in {output_dir}")
    return str(output_dir / "experimental_cv.csv"), str(output_dir / "simulated_cv.csv")


def create_synthetic_cycling_data(output_dir: str):
    """
    创建合成循环数据
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模拟100个循环
    cycles = np.arange(1, 101)
    
    # 理论容量（略有衰减）
    capacity_theory = 160 * np.exp(-cycles / 1000)
    
    # 实验容量（更多衰减和噪声）
    capacity_exp = 155 * np.exp(-cycles / 500) + np.random.normal(0, 2, len(cycles))
    
    # 库伦效率
    ce_theory = np.ones_like(cycles) * 99.9
    ce_exp = 99.5 + np.random.normal(0, 0.3, len(cycles))
    
    # 保存
    sim_data = np.column_stack([cycles, capacity_theory, ce_theory])
    exp_data = np.column_stack([cycles, capacity_exp, ce_exp])
    
    header = 'cycle,capacity_mAh_g,coulombic_efficiency_percent'
    np.savetxt(output_dir / "simulated_cycling.csv", sim_data,
               delimiter=',', header=header, comments='')
    np.savetxt(output_dir / "experimental_cycling.csv", exp_data,
               delimiter=',', header=header, comments='')
    
    print(f"Created synthetic cycling data in {output_dir}")
    return str(output_dir / "experimental_cycling.csv"), str(output_dir / "simulated_cycling.csv")


def example_gcd_comparison():
    """充放电曲线对比示例"""
    print("\n" + "="*60)
    print("Example 1: GCD Curve Comparison")
    print("="*60)
    
    # 创建数据
    exp_file, sim_file = create_synthetic_gcd_data("./example_data/electrochemical")
    
    # 加载数据
    connector = ElectrochemicalConnector()
    exp_data = connector.read(exp_file, test_type='gcd')
    sim_data = connector.read(sim_file, test_type='gcd')
    
    print(f"\nLoaded experimental data: {len(exp_data.raw_data)} points")
    print(f"Loaded simulated data: {len(sim_data.raw_data)} points")
    
    # 对比
    comparator = ElectrochemicalComparator()
    results = comparator.compare_gcd_curves(exp_data, sim_data)
    
    print("\nGCD Comparison Results:")
    print("-" * 40)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:35s}: {value:.4f}")
        else:
            print(f"{key:35s}: {value}")
    
    # 使用工作流
    print("\n" + "-" * 40)
    print("Running Validation Workflow...")
    
    config = ValidationConfig(
        data_type='gcd',
        output_dir='./example_results/electrochemical',
        generate_report=True,
    )
    
    workflow = ValidationWorkflow(config)
    workflow.load_experimental_data(exp_file)
    workflow.load_computational_data(sim_file)
    result = workflow.run_validation()
    
    print(f"Validation success: {result.success}")
    report_path = workflow.generate_report()
    print(f"Report saved to: {report_path}")


def example_cv_comparison():
    """CV曲线对比示例"""
    print("\n" + "="*60)
    print("Example 2: CV Curve Comparison")
    print("="*60)
    
    # 创建数据
    exp_file, sim_file = create_synthetic_cv_data("./example_data/electrochemical")
    
    # 加载数据
    connector = ElectrochemicalConnector()
    exp_data = connector.read(exp_file, test_type='cv')
    sim_data = connector.read(sim_file, test_type='cv')
    
    # 对比
    comparator = ElectrochemicalComparator()
    results = comparator.compare_cv_curves(exp_data, sim_data)
    
    print("\nCV Comparison Results:")
    print("-" * 40)
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key:35s}: {value:.4f}")
        else:
            print(f"{key:35s}: {value}")


def example_cycling_stability():
    """循环稳定性对比示例"""
    print("\n" + "="*60)
    print("Example 3: Cycling Stability Comparison")
    print("="*60)
    
    # 创建数据
    exp_file, sim_file = create_synthetic_cycling_data("./example_data/electrochemical")
    
    # 加载数据
    connector = ElectrochemicalConnector()
    exp_data = connector.read(exp_file, test_type='gcd')
    sim_data = connector.read(sim_file, test_type='gcd')
    
    # 对比
    comparator = ElectrochemicalComparator()
    results = comparator.compare_cycling_stability(exp_data, sim_data, n_cycles=100)
    
    print("\nCycling Stability Results:")
    print("-" * 40)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:35s}: {value:.4f}")
        else:
            print(f"{key:35s}: {value}")


def example_capacity_calculation():
    """容量计算示例"""
    print("\n" + "="*60)
    print("Example 4: Capacity Calculation")
    print("="*60)
    
    exp_file, _ = create_synthetic_gcd_data("./example_data/electrochemical")
    
    connector = ElectrochemicalConnector()
    data = connector.read(exp_file, test_type='gcd')
    
    # 计算比容量
    data_with_capacity = connector.calculate_capacity(data, mass=10.0, unit='mAh/g')
    
    df = data_with_capacity.to_dataframe()
    
    print("\nCapacity Information:")
    print("-" * 40)
    if 'capacity' in df.columns:
        print(f"Max capacity: {df['capacity'].max():.2f} mAh/g")
    if 'specific_capacity' in df.columns:
        print(f"Max specific capacity: {df['specific_capacity'].max():.2f} mAh/g")


if __name__ == "__main__":
    # 运行所有示例
    example_gcd_comparison()
    example_cv_comparison()
    example_cycling_stability()
    example_capacity_calculation()
    
    print("\n" + "="*60)
    print("All electrochemical examples completed!")
    print("="*60)
