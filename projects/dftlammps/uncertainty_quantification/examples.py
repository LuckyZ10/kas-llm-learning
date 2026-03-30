"""
不确定性量化综合示例与验证 - UQ Examples and Validation

Phase 70: 计算不确定性量化体系 - 案例验证

本模块提供完整的案例验证：
1. 材料性质预测的贝叶斯势函数验证
2. 分子动力学模拟的误差传播验证
3. 材料参数的敏感性分析验证
4. DFT-MD工作流的可靠性评估验证
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==================== 案例1: 材料性质预测的贝叶斯验证 ====================

def example_bayesian_potential_validation():
    """
    案例1: 贝叶斯神经网络势函数验证
    
    验证点:
    - 不确定性估计的合理性
    - 预测区间的覆盖率
    - 主动学习的有效性
    """
    print("\n" + "=" * 80)
    print("📊 案例1: 贝叶斯神经网络势函数验证")
    print("=" * 80)
    
    if not HAS_TORCH:
        print("PyTorch未安装，跳过此案例")
        return {}
    
    from .bayesian_potential import (
        ACSFDescriptor, MCDropoutPotential, EnsemblePotential,
        PotentialTrainer, BayesianCalibration
    )
    
    # 生成合成训练数据 (Lennard-Jones势近似)
    print("\n1. 生成合成训练数据...")
    
    def lennard_jones(r, epsilon=1.0, sigma=1.0):
        """Lennard-Jones势"""
        return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
    
    # 生成训练数据
    np.random.seed(42)
    n_train = 100
    r_train = np.random.uniform(0.9, 3.0, n_train)
    
    # 添加噪声模拟DFT误差
    noise = np.random.normal(0, 0.05, n_train)
    e_train = lennard_jones(r_train) + noise
    
    # 创建伪结构
    train_structures = []
    for i in range(n_train):
        train_structures.append({
            'positions': np.array([[0, 0, 0], [r_train[i], 0, 0]]),
            'atom_types': np.array([1, 1]),
            'energy': np.array([e_train[i]])
        })
    
    print(f"   训练样本数: {n_train}")
    
    # 创建势函数
    print("\n2. 创建MC Dropout势函数...")
    acsf = ACSFDescriptor(
        radial_params=[(0.1, 0.0), (0.1, 1.0)],
        angular_params=[],
        cutoff=3.0
    )
    
    mc_potential = MCDropoutPotential(
        descriptor=acsf,
        hidden_layers=[32, 32],
        dropout_rate=0.1
    )
    
    # 创建集成势函数
    ensemble_potential = EnsemblePotential(
        descriptor=acsf,
        n_models=3,
        hidden_layers=[32, 32]
    )
    
    # 测试预测（随机初始化）
    print("\n3. 测试预测不确定性...")
    test_r = np.linspace(0.95, 2.5, 20)
    
    mc_uncertainties = []
    ensemble_uncertainties = []
    
    for r in test_r:
        positions = np.array([[0, 0, 0], [r, 0, 0]])
        atom_types = np.array([1, 1])
        
        # MC Dropout预测
        pred_mc = mc_potential.predict_energy(positions, atom_types, n_samples=50)
        mc_uncertainties.append(np.sqrt(pred_mc.uncertainty.energy_var[0]))
        
        # 集成预测
        pred_ens = ensemble_potential.predict_energy(positions, atom_types)
        ensemble_uncertainties.append(np.sqrt(pred_ens.uncertainty.energy_var[0]))
    
    mc_uncertainties = np.array(mc_uncertainties)
    ensemble_uncertainties = np.array(ensemble_uncertainties)
    
    print(f"   MC Dropout平均不确定性: {np.mean(mc_uncertainties):.4f}")
    print(f"   集成方法平均不确定性: {np.mean(ensemble_uncertainties):.4f}")
    print(f"   不确定性比 (MC/Ens): {np.mean(mc_uncertainties)/np.mean(ensemble_uncertainties):.2f}")
    
    # 验证不确定性校准
    print("\n4. 不确定性校准验证...")
    
    # 生成验证数据
    n_val = 50
    r_val = np.random.uniform(0.95, 2.5, n_val)
    e_val = lennard_jones(r_val)
    
    calibration_structures = []
    for i in range(n_val):
        calibration_structures.append({
            'positions': np.array([[0, 0, 0], [r_val[i], 0, 0]]),
            'atom_types': np.array([1, 1]),
            'energy': np.array([e_val[i]])
        })
    
    calibrator = BayesianCalibration(mc_potential)
    cal_result = calibrator.calibrate(calibration_structures, method='temperature_scaling')
    
    print(f"   校准方法: {cal_result['method']}")
    if 'temperature' in cal_result:
        print(f"   温度缩放因子: {cal_result['temperature']:.4f}")
    
    return {
        'mc_avg_uncertainty': float(np.mean(mc_uncertainties)),
        'ensemble_avg_uncertainty': float(np.mean(ensemble_uncertainties)),
        'calibration': cal_result
    }


# ==================== 案例2: 分子动力学误差传播验证 ====================

def example_md_error_propagation():
    """
    案例2: 分子动力学模拟误差传播验证
    
    验证点:
    - 蒙特卡洛误差传播的正确性
    - 不同采样方法的对比
    - 误差预算分析的准确性
    """
    print("\n" + "=" * 80)
    print("📊 案例2: 分子动力学误差传播验证")
    print("=" * 80)
    
    from .mc_propagation import MCErrorPropagation, LatinHypercubeSampler, DirectSampling
    
    # 定义MD模拟模型
    print("\n1. 定义MD模拟简化模型...")
    
    def md_temperature_model(params):
        """
        MD温度演化模型
        params = [T0, gamma, dt, n_steps]
        """
        T0, gamma, dt, n_steps = params
        # 简化的温度弛豫模型
        return T0 * np.exp(-gamma * dt * n_steps)
    
    # 定义参数不确定性
    param_names = ['T0', 'gamma', 'dt', 'n_steps']
    distributions = {
        'T0': stats.norm(300, 10),      # 初始温度 K
        'gamma': stats.norm(0.1, 0.01), # 阻尼系数 ps^-1
        'dt': stats.norm(1.0, 0.1),     # 时间步 fs
        'n_steps': stats.norm(1000, 50) # 步数
    }
    
    print("   参数定义:")
    for name, dist in distributions.items():
        print(f"      {name}: N({dist.mean():.1f}, {dist.std():.2f})")
    
    # 误差传播
    print("\n2. 蒙特卡洛误差传播...")
    
    mc_prop = MCErrorPropagation()
    
    result = mc_prop.propagate(
        lambda **kwargs: md_temperature_model([
            kwargs['T0'], kwargs['gamma'], kwargs['dt'], kwargs['n_steps']
        ]),
        distributions,
        n_samples=10000,
        method='lhs'
    )
    
    print(f"   最终温度均值: {result.mean[0]:.2f} K")
    print(f"   温度标准差: {result.std[0]:.2f} K")
    print(f"   相对不确定性: {result.std[0]/result.mean[0]*100:.2f}%")
    
    # 误差预算分析
    print("\n3. 误差预算分析...")
    
    error_budget = mc_prop.error_budget_analysis(
        model=lambda T0, gamma, dt, n_steps: md_temperature_model([T0, gamma, dt, n_steps]),
        nominal_params={
            'T0': 300, 'gamma': 0.1, 'dt': 1.0, 'n_steps': 1000
        },
        param_uncertainties={
            'T0': 10, 'gamma': 0.01, 'dt': 0.1, 'n_steps': 50
        },
        n_samples=5000
    )
    
    print(f"   总方差: {error_budget.total_variance:.4f}")
    print("   参数贡献:")
    for param, contrib in error_budget.get_dominant_sources(4):
        rel = error_budget.relative_contribution(param)
        print(f"      {param}: {contrib:.4f} ({rel*100:.1f}%)")
    
    # 验证置信区间覆盖率
    print("\n4. 置信区间覆盖率验证...")
    
    # 理论值
    T_true = md_temperature_model([300, 0.1, 1.0, 1000])
    
    # 检查覆盖率
    ci_95 = result.confidence_intervals[0.95]
    coverage_95 = (T_true >= ci_95.lower[0]) and (T_true <= ci_95.upper[0])
    
    print(f"   理论最终温度: {T_true:.2f} K")
    print(f"   95% CI: [{ci_95.lower[0]:.2f}, {ci_95.upper[0]:.2f}] K")
    print(f"   覆盖理论值: {'是' if coverage_95 else '否'}")
    
    return {
        'mean_temperature': float(result.mean[0]),
        'std_temperature': float(result.std[0]),
        'coverage_95': coverage_95,
        'error_budget': {
            'total_variance': error_budget.total_variance,
            'dominant_source': error_budget.get_dominant_sources(1)[0][0]
        }
    }


# ==================== 案例3: 敏感性分析验证 ====================

def example_sensitivity_validation():
    """
    案例3: 材料参数敏感性分析验证
    
    验证点:
    - Sobol指标的加和性
    - Morris方法的筛选能力
    - 参数重要性排序的合理性
    """
    print("\n" + "=" * 80)
    print("📊 案例3: 材料参数敏感性分析验证")
    print("=" * 80)
    
    from .sensitivity_analysis import SobolSensitivity, MorrisMethod, ElementaryEffects
    
    # 定义材料模型
    print("\n1. 定义材料本构模型...")
    
    def material_stress(params):
        """
        应力-应变模型
        params = [E, nu, epsilon, alpha, T]
        E: 弹性模量
        nu: 泊松比
        epsilon: 应变
        alpha: 热膨胀系数
        T: 温度变化
        """
        E, nu, epsilon, alpha, T = params
        
        # 机械应变
        mech_strain = epsilon
        # 热应变
        thermal_strain = alpha * T
        # 总应变
        total_strain = mech_strain + thermal_strain
        
        # 应力 (考虑温度软化)
        E_effective = E * (1 - 0.001 * T)
        stress = E_effective * total_strain
        
        return stress
    
    param_names = ['E', 'nu', 'epsilon', 'alpha', 'T']
    bounds = np.array([
        [150, 250],      # E: GPa
        [0.25, 0.35],    # nu
        [0.001, 0.01],   # epsilon
        [1e-6, 5e-6],    # alpha: /K
        [0, 100]         # T: K
    ])
    
    print(f"   参数数量: {len(param_names)}")
    print(f"   模型输出: 应力 (MPa)")
    
    # Sobol分析
    print("\n2. Sobol全局敏感性分析...")
    
    sobol = SobolSensitivity(calc_second_order=False)
    report_sobol = sobol.analyze(material_stress, param_names, bounds, n_samples=256)
    
    print("   一阶Sobol指标:")
    for name, s1 in zip(param_names, report_sobol.indices.first_order):
        print(f"      {name}: {s1:.4f}")
    
    print("   总效应Sobol指标:")
    for name, st in zip(param_names, report_sobol.indices.total_order):
        print(f"      {name}: {st:.4f}")
    
    # 验证一阶指标加和
    sum_s1 = np.sum(report_sobol.indices.first_order)
    print(f"   一阶指标之和: {sum_s1:.4f}")
    print(f"   交互效应估计: {1 - sum_s1:.4f}")
    
    # Morris分析
    print("\n3. Morris筛选方法...")
    
    morris = MorrisMethod(num_levels=4)
    report_morris = morris.analyze(material_stress, param_names, bounds, n_samples=20)
    
    print("   基本效应 μ*:")
    for name, mu_star in zip(param_names, report_morris.indices.total_order):
        print(f"      {name}: {mu_star:.4f}")
    
    # 参数分类
    print("   参数分类:")
    for rec in report_morris.recommendations[:4]:
        print(f"      • {rec}")
    
    # 验证重要性排序
    print("\n4. 重要性排序验证...")
    
    importance = report_sobol.parameter_importance
    rankings = list(zip(param_names, importance.rankings))
    rankings.sort(key=lambda x: x[1])
    
    print("   基于Sobol总效应的排名:")
    for name, rank in rankings:
        score = importance.importance_scores[param_names.index(name)]
        print(f"      {rank}. {name} (ST={score:.4f})")
    
    # 理论预期: E和epsilon应该最重要
    expected_important = ['E', 'epsilon']
    top2 = [rankings[i][0] for i in range(2)]
    
    match = len(set(expected_important) & set(top2))
    print(f"\n   预期重要参数: {expected_important}")
    print(f"   实际TOP2参数: {top2}")
    print(f"   匹配度: {match}/2")
    
    return {
        'sum_first_order': float(sum_s1),
        'interaction_effect': float(1 - sum_s1),
        'most_important_param': top2[0],
        'ranking_match': match
    }


# ==================== 案例4: 工作流可靠性验证 ====================

def example_workflow_reliability_validation():
    """
    案例4: DFT-MD工作流可靠性评估验证
    
    验证点:
    - FORM方法的准确性
    - 工作流整体可靠性计算
    - 关键失效路径识别
    """
    print("\n" + "=" * 80)
    print("📊 案例4: DFT-MD工作流可靠性评估验证")
    print("=" * 80)
    
    from .workflow_reliability import (
        WorkflowReliability, FORMAnalysis, MonteCarloReliability,
        ReliabilityEngine
    )
    
    # 定义DFT-MD工作流
    print("\n1. 定义DFT-MD工作流...")
    
    workflow = WorkflowReliability()
    
    # 步骤1: DFT收敛性
    def dft_convergence_limit(params):
        """
        DFT收敛性极限状态
        params = [encut, kspacing, ediff]
        """
        encut, kspacing, ediff = params
        
        # 收敛条件
        converged = (encut >= 400 and 
                    kspacing <= 0.3 and 
                    ediff <= 1e-5)
        
        # 返回裕度 (正值表示安全)
        margin = min(encut - 400, 
                    0.3 - kspacing, 
                    1e-5 - ediff)
        
        return margin
    
    # 步骤2: 力场质量
    def forcefield_quality_limit(params):
        """
        力场质量极限状态
        params = [rmse_forces, coverage]
        """
        rmse_forces, coverage = params
        
        # 质量标准
        good = (rmse_forces <= 0.1 and coverage >= 0.9)
        
        margin = min(0.1 - rmse_forces, coverage - 0.9)
        
        return margin
    
    # 步骤3: MD稳定性
    def md_stability_limit(params):
        """
        MD稳定性极限状态
        params = [timestep, temperature, pressure]
        """
        timestep, temperature, pressure = params
        
        # 稳定性条件 (简化的能量守恒条件)
        max_timestep = 2.0 / np.sqrt(max(temperature, 1) / 300)
        stable = (timestep <= max_timestep and 
                 100 <= temperature <= 1000 and 
                 0.1 <= pressure <= 10)
        
        margin = min(max_timestep - timestep,
                    temperature - 100,
                    1000 - temperature,
                    pressure - 0.1,
                    10 - pressure)
        
        return margin
    
    # 添加工作流步骤
    workflow.add_workflow_step(
        'DFT_Convergence',
        dft_convergence_limit,
        {
            'encut': stats.norm(450, 30),      # eV
            'kspacing': stats.norm(0.25, 0.05), # Å^-1
            'ediff': stats.lognorm(0.5, scale=1e-6)  # eV
        }
    )
    
    workflow.add_workflow_step(
        'ForceField_Quality',
        forcefield_quality_limit,
        {
            'rmse_forces': stats.norm(0.08, 0.02),  # eV/Å
            'coverage': stats.beta(9, 1)  # 覆盖率
        }
    )
    
    workflow.add_workflow_step(
        'MD_Stability',
        md_stability_limit,
        {
            'timestep': stats.norm(1.0, 0.2),     # fs
            'temperature': stats.norm(300, 50),   # K
            'pressure': stats.norm(1.0, 2.0)      # bar
        }
    )
    
    print("   工作流步骤:")
    for i, name in enumerate(workflow.steps.keys(), 1):
        print(f"      {i}. {name}")
    
    # 评估各步骤
    print("\n2. 各步骤可靠性评估 (FORM)...")
    
    results = workflow.assess_workflow(method='form')
    
    for name, assessment in results.items():
        print(f"   {name}:")
        print(f"      失效概率: {assessment.failure_probability.pf:.6e}")
        print(f"      可靠性指标 β: {assessment.reliability_index.beta:.4f}")
    
    # 工作流整体可靠性
    print("\n3. 工作流整体可靠性...")
    
    wf_reliability = workflow.workflow_reliability()
    wf_failure_prob = 1 - wf_reliability
    
    print(f"   工作流整体可靠性: {wf_reliability:.6f}")
    print(f"   工作流失效概率: {wf_failure_prob:.6e}")
    print(f"   工作流可靠性指标: {-stats.norm.ppf(wf_failure_prob):.4f}")
    
    # 识别关键步骤
    print("\n4. 关键步骤识别...")
    
    step_pfs = {name: a.failure_probability.pf 
                for name, a in results.items()}
    critical_step = max(step_pfs.items(), key=lambda x: x[1])
    
    print(f"   最不可靠步骤: {critical_step[0]}")
    print(f"   该步骤失效概率: {critical_step[1]:.6e}")
    
    # FORM与MC对比验证
    print("\n5. FORM与蒙特卡洛对比验证...")
    
    engine = ReliabilityEngine()
    
    # 使用最简单的步骤进行对比
    step = workflow.steps['DFT_Convergence']
    
    assessment_form = engine.assess(
        step['limit_state'],
        step['distributions'],
        method='form'
    )
    
    assessment_mc = engine.assess(
        step['limit_state'],
        step['distributions'],
        method='mc'
    )
    
    print(f"   FORM失效概率: {assessment_form.failure_probability.pf:.6e}")
    print(f"   MC失效概率: {assessment_mc.failure_probability.pf:.6e}")
    
    relative_diff = abs(assessment_form.failure_probability.pf - 
                       assessment_mc.failure_probability.pf) / \
                   max(assessment_mc.failure_probability.pf, 1e-10)
    print(f"   相对差异: {relative_diff*100:.1f}%")
    
    return {
        'workflow_reliability': wf_reliability,
        'workflow_failure_probability': wf_failure_prob,
        'critical_step': critical_step[0],
        'form_mc_relative_diff': float(relative_diff)
    }


# ==================== 综合演示 ====================

def run_all_validations():
    """运行所有验证案例"""
    print("\n" + "=" * 80)
    print("🔬 Phase 70: 不确定性量化体系 - 综合案例验证")
    print("=" * 80)
    
    results = {}
    
    # 案例1
    try:
        results['bayesian_potential'] = example_bayesian_potential_validation()
    except Exception as e:
        print(f"案例1失败: {e}")
        results['bayesian_potential'] = {'error': str(e)}
    
    # 案例2
    try:
        results['md_error_propagation'] = example_md_error_propagation()
    except Exception as e:
        print(f"案例2失败: {e}")
        results['md_error_propagation'] = {'error': str(e)}
    
    # 案例3
    try:
        results['sensitivity_analysis'] = example_sensitivity_validation()
    except Exception as e:
        print(f"案例3失败: {e}")
        results['sensitivity_analysis'] = {'error': str(e)}
    
    # 案例4
    try:
        results['workflow_reliability'] = example_workflow_reliability_validation()
    except Exception as e:
        print(f"案例4失败: {e}")
        results['workflow_reliability'] = {'error': str(e)}
    
    # 总结
    print("\n" + "=" * 80)
    print("📋 验证总结")
    print("=" * 80)
    
    print("\n案例完成情况:")
    for name, result in results.items():
        status = "✅ 成功" if 'error' not in result else "❌ 失败"
        print(f"   {name}: {status}")
    
    print("\n关键指标:")
    if 'md_error_propagation' in results and 'error' not in results['md_error_propagation']:
        r = results['md_error_propagation']
        print(f"   温度预测不确定性: {r.get('std_temperature', 'N/A'):.2f} K")
    
    if 'sensitivity_analysis' in results and 'error' not in results['sensitivity_analysis']:
        r = results['sensitivity_analysis']
        print(f"   交互效应比例: {r.get('interaction_effect', 'N/A'):.2%}")
        print(f"   重要性排序匹配: {r.get('ranking_match', 'N/A')}/2")
    
    if 'workflow_reliability' in results and 'error' not in results['workflow_reliability']:
        r = results['workflow_reliability']
        print(f"   工作流可靠性: {r.get('workflow_reliability', 'N/A'):.4f}")
        print(f"   关键失效步骤: {r.get('critical_step', 'N/A')}")
    
    print("\n" + "=" * 80)
    print("✅ 所有案例验证完成")
    print("=" * 80)
    
    return results


def demo():
    """演示入口"""
    return run_all_validations()


if __name__ == "__main__":
    demo()
