"""
Example: Uncertainty Quantification
===================================
不确定性量化示例

展示误差传播、置信区间估计和敏感性分析
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入实验验证模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimental_validation import (
    ErrorPropagator,
    ConfidenceIntervalEstimator,
    SensitivityAnalyzer,
)


def example_linear_error_propagation():
    """线性误差传播示例"""
    print("\n" + "="*60)
    print("Example 1: Linear Error Propagation")
    print("="*60)
    
    propagator = ErrorPropagator()
    
    # 示例：计算密度
    # ρ = m / V
    # 假设测量质量和体积都有不确定度
    
    mass = 10.0  # g
    mass_uncertainty = 0.1  # g
    
    volume = 5.0  # cm³
    volume_uncertainty = 0.2  # cm³
    
    # 计算密度
    density = mass / volume
    
    # 使用蒙特卡洛方法计算误差传播
    def density_func(params):
        m, v = params
        return m / v if v != 0 else 0
    
    values = np.array([mass, volume])
    uncertainties = np.array([mass_uncertainty, volume_uncertainty])
    
    results = propagator.monte_carlo_propagation(
        density_func, values, uncertainties, n_samples=100000
    )
    
    print(f"\nDensity Calculation:")
    print(f"  Formula: ρ = m / V")
    print(f"  m = {mass} ± {mass_uncertainty} g")
    print(f"  V = {volume} ± {volume_uncertainty} cm³")
    print(f"\n  Nominal density: {density:.4f} g/cm³")
    print(f"\n  Monte Carlo Results:")
    print(f"    Mean: {results['mean']:.4f} g/cm³")
    print(f"    Std:  {results['std']:.4f} g/cm³")
    print(f"    95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}] g/cm³")
    
    # 解析方法
    # ∂ρ/∂m = 1/V, ∂ρ/∂V = -m/V²
    jacobian = np.array([1/volume, -mass/volume**2])
    
    density_val, density_unc = propagator.linear_propagation(
        values, uncertainties, jacobian
    )
    
    print(f"\n  Linear Propagation:")
    print(f"    Density: {density_val:.4f} ± {density_unc:.4f} g/cm³")


def example_band_gap_uncertainty():
    """带隙计算的不确定度示例"""
    print("\n" + "="*60)
    print("Example 2: Band Gap Uncertainty")
    print("="*60)
    
    # 模拟DFT计算带隙的误差来源
    # 1. k点网格密度
    # 2. 截断能
    # 3. 交换关联泛函
    
    # 输入参数
    cutoff_energy = 500  # eV
    cutoff_uncertainty = 50  # eV（假设不确定性）
    
    k_points = 8  # 每维度k点
    k_uncertainty = 2  # k点不确定性
    
    # 简化的带隙计算函数
    def band_gap_func(params):
        cutoff, k = params
        # 简化的收敛行为
        bg_converged = 2.5  # eV（收敛值）
        # 截断能误差
        bg = bg_converged * (1 - 0.1 * np.exp(-cutoff / 300))
        # k点误差
        bg = bg * (1 - 0.05 * np.exp(-k / 4))
        return bg
    
    values = np.array([cutoff_energy, k_points])
    uncertainties = np.array([cutoff_uncertainty, k_uncertainty])
    
    propagator = ErrorPropagator()
    results = propagator.monte_carlo_propagation(
        band_gap_func, values, uncertainties, n_samples=50000
    )
    
    nominal = band_gap_func(values)
    
    print(f"\nBand Gap Calculation:")
    print(f"  Cutoff energy: {cutoff_energy} ± {cutoff_uncertainty} eV")
    print(f"  K-points: {k_points} ± {k_uncertainty} per dimension")
    print(f"\n  Nominal band gap: {nominal:.3f} eV")
    print(f"  Mean: {results['mean']:.3f} eV")
    print(f"  Std:  {results['std']:.3f} eV")
    print(f"  95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}] eV")
    
    # 与实验值比较
    exp_band_gap = 2.2  # eV（假设实验值）
    exp_uncertainty = 0.05  # eV
    
    print(f"\n  Experimental value: {exp_band_gap} ± {exp_uncertainty} eV")
    
    # 一致性检验
    diff = abs(results['mean'] - exp_band_gap)
    combined_unc = np.sqrt(results['std']**2 + exp_uncertainty**2)
    
    if diff < 2 * combined_unc:
        print(f"  ✓ Consistent with experiment (diff = {diff:.3f} eV)")
    else:
        print(f"  ✗ Inconsistent with experiment (diff = {diff:.3f} eV)")


def example_confidence_intervals():
    """置信区间示例"""
    print("\n" + "="*60)
    print("Example 3: Confidence Intervals")
    print("="*60)
    
    estimator = ConfidenceIntervalEstimator()
    
    # 模拟实验数据
    np.random.seed(42)
    data = np.random.normal(100, 10, 50)  # 50个样本
    
    print(f"\nSample Statistics:")
    print(f"  n = {len(data)}")
    print(f"  mean = {np.mean(data):.2f}")
    print(f"  std = {np.std(data, ddof=1):.2f}")
    
    # 均值置信区间
    mean, ci_lower, ci_upper = estimator.mean_ci(data, confidence=0.95)
    print(f"\n  95% CI for mean: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    # Bootstrap置信区间
    mean_bs, ci_lower_bs, ci_upper_bs = estimator.bootstrap_ci(
        data, np.mean, confidence=0.95, n_bootstrap=10000
    )
    print(f"  Bootstrap 95% CI: [{ci_lower_bs:.2f}, {ci_upper_bs:.2f}]")
    
    # 比例的置信区间
    successes = 45
    trials = 50
    prop, ci_lower_p, ci_upper_p = estimator.proportion_ci(
        successes, trials, method='wilson'
    )
    print(f"\nProportion CI (45/50 successes):")
    print(f"  p = {prop:.3f}")
    print(f"  95% CI: [{ci_lower_p:.3f}, {ci_upper_p:.3f}]")


def example_sensitivity_analysis():
    """敏感性分析示例"""
    print("\n" + "="*60)
    print("Example 4: Sensitivity Analysis")
    print("="*60)
    
    analyzer = SensitivityAnalyzer()
    
    # 示例：离子电导率计算
    # σ = n * q * μ
    # n: 载流子浓度
    # q: 电荷
    # μ: 迁移率
    
    def ionic_conductivity(params):
        n, q, mu = params
        return n * q * mu
    
    # 参数值
    n = 1e20  # cm⁻³
    q = 1.6e-19  # C
    mu = 0.1  # cm²/(V·s)
    
    values = np.array([n, q, mu])
    param_names = ['carrier_concentration', 'charge', 'mobility']
    
    # 局部敏感性分析
    sensitivities = analyzer.local_sensitivity(
        ionic_conductivity, values, param_names
    )
    
    print(f"\nIonic Conductivity Sensitivity:")
    print(f"  σ = n · q · μ")
    print(f"\n  Parameter values:")
    print(f"    n = {n:.2e} cm⁻³")
    print(f"    q = {q:.2e} C")
    print(f"    μ = {mu:.2f} cm²/(V·s)")
    print(f"  σ = {ionic_conductivity(values):.2e} S/cm")
    
    print(f"\n  Sensitivity coefficients:")
    for param, sens in sensitivities['sensitivities'].items():
        print(f"    ∂σ/∂{param:20s} = {sens:.2e}")
    
    print(f"\n  Elasticities (normalized):")
    for param, elas in sensitivities['elasticities'].items():
        print(f"    {param:20s}: {elas:.3f}")
    
    print(f"\n  Most sensitive parameter: {sensitivities['most_sensitive']}")
    
    # 龙卷风图数据
    uncertainties = np.array([n * 0.1, q * 0.01, mu * 0.2])
    tornado_data = analyzer.tornado_diagram_data(
        ionic_conductivity, values, uncertainties, param_names
    )
    
    print(f"\n  Tornado diagram data:")
    print(f"    Base value: {tornado_data['base_value']:.2e}")
    for i, param in enumerate(tornado_data['parameters']):
        print(f"    {param}: [{tornado_data['low_impact'][i]:+.2e}, "
              f"{tornado_data['high_impact'][i]:+.2e}]")


def example_monte_carlo_validation():
    """蒙特卡洛验证示例"""
    print("\n" + "="*60)
    print("Example 5: Monte Carlo Validation")
    print("="*60)
    
    # 模拟计算与实验的系统性偏差
    # 假设真实值是100，但计算有系统误差和随机误差
    
    np.random.seed(42)
    
    # 实验数据（有噪声）
    n_exp = 30
    exp_data = np.random.normal(100, 2, n_exp)
    
    # 计算数据（有系统偏差+噪声）
    n_sim = 1000
    systematic_bias = 5  # 系统偏差5%
    random_error = 3
    sim_data = np.random.normal(100 + systematic_bias, random_error, n_sim)
    
    print(f"\nMonte Carlo Validation:")
    print(f"  Experimental data: n={n_exp}, mean={np.mean(exp_data):.2f}, std={np.std(exp_data):.2f}")
    print(f"  Simulated data: n={n_sim}, mean={np.mean(sim_data):.2f}, std={np.std(sim_data):.2f}")
    
    # 使用置信区间估计器
    estimator = ConfidenceIntervalEstimator()
    
    # 实验均值CI
    _, exp_ci_low, exp_ci_high = estimator.mean_ci(exp_data)
    # 模拟均值CI
    _, sim_ci_low, sim_ci_high = estimator.mean_ci(sim_data)
    
    print(f"\n  Experimental 95% CI: [{exp_ci_low:.2f}, {exp_ci_high:.2f}]")
    print(f"  Simulated 95% CI: [{sim_ci_low:.2f}, {sim_ci_high:.2f}]")
    
    # 检查是否有重叠
    overlap = not (exp_ci_high < sim_ci_low or sim_ci_high < exp_ci_low)
    
    if overlap:
        print(f"  \n  ✓ CIs overlap - systematic bias not statistically significant")
    else:
        print(f"  \n  ✗ CIs do not overlap - systematic bias detected")
        bias = np.mean(sim_data) - np.mean(exp_data)
        print(f"    Estimated bias: {bias:.2f}")


def example_correlation_analysis():
    """相关性分析示例"""
    print("\n" + "="*60)
    print("Example 6: Correlation Analysis")
    print("="*60)
    
    analyzer = SensitivityAnalyzer()
    
    # 生成参数和输出的相关性数据
    np.random.seed(42)
    n_samples = 1000
    
    # 三个参数
    param1 = np.random.uniform(0, 1, n_samples)  # 强相关
    param2 = np.random.uniform(0, 1, n_samples)  # 中等相关
    param3 = np.random.uniform(0, 1, n_samples)  # 弱相关
    
    # 输出（与参数有不同相关性）
    output = (
        2.0 * param1 +
        1.0 * param2 +
        0.2 * param3 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    parameters = np.column_stack([param1, param2, param3])
    
    correlation = analyzer.correlation_analysis(parameters, output)
    
    print(f"\nCorrelation Analysis:")
    print(f"  n_samples = {n_samples}")
    print(f"\n  Pearson correlation:")
    print(f"    Param 1: r = {correlation['pearson_r'][0]:.3f} (p = {correlation['pearson_p'][0]:.3e})")
    print(f"    Param 2: r = {correlation['pearson_r'][1]:.3f} (p = {correlation['pearson_p'][1]:.3e})")
    print(f"    Param 3: r = {correlation['pearson_r'][2]:.3f} (p = {correlation['pearson_p'][2]:.3e})")
    
    print(f"\n  Spearman correlation:")
    print(f"    Param 1: ρ = {correlation['spearman_rho'][0]:.3f}")
    print(f"    Param 2: ρ = {correlation['spearman_rho'][1]:.3f}")
    print(f"    Param 3: ρ = {correlation['spearman_rho'][2]:.3f}")
    
    # 排序重要性
    importance = np.argsort(np.abs(correlation['pearson_r']))[::-1]
    print(f"\n  Parameter importance (by |r|):")
    for i, idx in enumerate(importance, 1):
        print(f"    {i}. Param {idx+1}: |r| = {abs(correlation['pearson_r'][idx]):.3f}")


if __name__ == "__main__":
    # 运行所有示例
    example_linear_error_propagation()
    example_band_gap_uncertainty()
    example_confidence_intervals()
    example_sensitivity_analysis()
    example_monte_carlo_validation()
    example_correlation_analysis()
    
    print("\n" + "="*60)
    print("All uncertainty quantification examples completed!")
    print("="*60)
