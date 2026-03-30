"""
self_driving_perovskite.py
自动驾驶钙钛矿发现

演示闭环发现系统用于钙钛矿太阳能电池材料的自动发现。

应用场景:
- 新型钙钛矿组成发现
- 带隙工程
- 稳定性优化
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
sys.path.insert(0, '/root/.openclaw/workspace')

from dftlammps.frontier.self_driving_lab import (
    SelfDrivingLab, SynthesisPlanner, SynthesisParameters
)
from dftlammps.frontier.closed_loop_discovery import (
    ClosedLoopDiscovery, BayesianOptimizer
)


def discover_perovskite_composition(
    target_bandgap: float = 1.5,
    tolerance_factor_range: Tuple[float, float] = (0.8, 1.0),
    max_experiments: int = 20
) -> List[Dict]:
    """
    发现钙钛矿组成
    
    Args:
        target_bandgap: 目标带隙 (eV)
        tolerance_factor_range: 容忍因子范围
        max_experiments: 最大实验次数
    """
    print("=" * 60)
    print("Self-Driving Perovskite Discovery")
    print("=" * 60)
    
    # 初始化闭环发现系统
    discovery = ClosedLoopDiscovery(
        target_properties={
            'band_gap': (target_bandgap, 0.1),
            'stability_score': (0.8, 0.1)
        },
        max_iterations=max_experiments
    )
    
    print(f"\nTarget bandgap: {target_bandgap} eV")
    print(f"Tolerance factor range: {tolerance_factor_range}")
    
    # 运行发现
    results = discovery.run()
    
    # 输出结果
    print("\n" + "=" * 60)
    print("Discovery Results")
    print("=" * 60)
    
    summary = discovery.get_summary()
    
    print(f"\nTotal iterations: {summary['total_iterations']}")
    print(f"Successful discoveries: {summary['successful_iterations']}")
    
    if summary['best_samples']:
        print("\nBest compositions found:")
        for sample in summary['best_samples'][:5]:
            print(f"  Iteration {sample['iteration']}: {sample['properties']}")
    
    return summary['best_samples']


def plan_perovskite_synthesis(
    composition: Dict[str, float],
    method: str = 'solution'
) -> List[SynthesisParameters]:
    """
    规划钙钛矿合成路线
    
    Args:
        composition: 化学组成
        method: 合成方法 ('solution', 'vapor', 'mechanochemical')
    """
    print("\n" + "=" * 60)
    print("Perovskite Synthesis Planning")
    print("=" * 60)
    
    planner = SynthesisPlanner()
    
    # 构建化学式字符串
    formula = ''.join(f"{k}{v}" for k, v in composition.items())
    
    print(f"\nComposition: {formula}")
    print(f"Method preference: {method}")
    
    # 生成候选方案
    candidates = planner.plan_synthesis(formula)
    
    # 筛选适合的方法
    if method == 'solution':
        suitable = ['sol_gel', 'co_precipitation']
    elif method == 'vapor':
        suitable = ['vapor_deposition']
    else:
        suitable = ['mechanochemical', 'solid_state']
    
    # 输出方案
    print(f"\nGenerated {len(candidates)} synthesis candidates")
    
    for i, cand in enumerate(candidates[:3]):
        print(f"\nCandidate {i+1}:")
        print(f"  Temperature: {cand.temperature}°C")
        print(f"  Time: {cand.time} hours")
        print(f"  Atmosphere: {cand.atmosphere}")
        print(f"  Precursors: {cand.precursors}")
    
    return candidates


def optimize_synthesis_conditions(
    composition: Dict[str, float],
    initial_params: SynthesisParameters,
    target_properties: Dict[str, float],
    n_iterations: int = 20
) -> Dict[str, float]:
    """
    优化合成条件
    
    使用贝叶斯优化寻找最佳合成参数
    """
    print("\n" + "=" * 60)
    print("Optimizing Synthesis Conditions")
    print("=" * 60)
    
    # 定义参数空间
    bounds = {
        'temperature': (100, 200),  # 退火温度
        'annealing_time': (10, 60),  # 退火时间 (分钟)
        'precursor_concentration': (0.5, 2.0),  # 前驱体浓度 (M)
    }
    
    optimizer = BayesianOptimizer(bounds)
    
    best_params = None
    best_score = -float('inf')
    
    history = []
    
    for iteration in range(n_iterations):
        # 建议参数
        params = optimizer.suggest()
        
        # 模拟合成和表征
        result = simulate_perovskite_synthesis(composition, params)
        
        # 评分
        score = score_perovskite_result(result, target_properties)
        
        # 更新优化器
        optimizer.update(params, score)
        
        history.append((params, score, result))
        
        if score > best_score:
            best_score = score
            best_params = params
        
        if iteration % 5 == 0:
            print(f"Iteration {iteration}: Score = {score:.3f}, "
                  f"Bandgap = {result['band_gap']:.3f} eV")
    
    print(f"\nBest conditions found:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.2f}")
    print(f"  Best score: {best_score:.3f}")
    
    return best_params


def simulate_perovskite_synthesis(
    composition: Dict[str, float],
    params: Dict[str, float]
) -> Dict[str, float]:
    """
    模拟钙钛矿合成和表征
    
    简化模型用于演示
    """
    # 基于参数计算性质 (带噪声)
    
    # 带隙 (受温度影响)
    base_bandgap = 1.5
    temp_effect = (params['temperature'] - 150) * 0.002
    noise = np.random.randn() * 0.05
    bandgap = base_bandgap + temp_effect + noise
    
    # 稳定性
    stability = 0.7 + np.random.rand() * 0.2
    if params['annealing_time'] > 30:
        stability += 0.05
    
    # 结晶度
    crystallinity = min(1.0, params['annealing_time'] / 40 + np.random.rand() * 0.2)
    
    return {
        'band_gap': np.clip(bandgap, 1.2, 2.0),
        'stability_score': np.clip(stability, 0, 1),
        'crystallinity': np.clip(crystallinity, 0, 1),
        'efficiency': np.clip(15 + (1.5 - bandgap) * 10 + np.random.randn() * 2, 5, 25)
    }


def score_perovskite_result(
    result: Dict[str, float],
    target: Dict[str, float]
) -> float:
    """评分结果"""
    score = 0.0
    
    # 带隙匹配
    if 'band_gap' in target:
        error = abs(result['band_gap'] - target['band_gap'])
        score += max(0, 1 - error / 0.3)
    
    # 稳定性
    if 'stability_score' in target:
        score += result['stability_score']
    
    # 效率
    if 'efficiency' in result:
        score += result['efficiency'] / 25 * 0.5
    
    return score


def design_stable_perovskite():
    """
    设计稳定钙钛矿
    
    针对稳定性进行优化
    """
    print("\n" + "=" * 60)
    print("Designing Stable Perovskite")
    print("=" * 60)
    
    # 混合阳离子策略
    cations = {
        'FA': 0.8,   # 甲脒
        'MA': 0.15,  # 甲胺
        'Cs': 0.05   # 铯
    }
    
    halides = {
        'I': 0.85,
        'Br': 0.15
    }
    
    print("Mixed cation strategy:")
    print(f"  A-site: {cations}")
    print(f"  X-site: {halides}")
    
    # 容忍因子计算
    tolerance_factor = calculate_tolerance_factor(cations, 'Pb', halides)
    print(f"\nTolerance factor: {tolerance_factor:.3f}")
    
    if 0.8 <= tolerance_factor <= 1.0:
        print("  ✓ Within stable range")
    else:
        print("  ✗ Outside stable range")
    
    # 八体因子
    octahedral_factor = calculate_octahedral_factor('Pb', halides)
    print(f"Octahedral factor: {octahedral_factor:.3f}")
    
    return {
        'cations': cations,
        'halides': halides,
        'tolerance_factor': tolerance_factor,
        'predicted_stable': 0.8 <= tolerance_factor <= 1.0
    }


def calculate_tolerance_factor(
    a_cations: Dict[str, float],
    b_cation: str,
    x_anions: Dict[str, float]
) -> float:
    """
    计算Goldschmidt容忍因子
    
    t = (r_A + r_X) / (sqrt(2) * (r_B + r_X))
    """
    # 离子半径 (简化值)
    ionic_radii = {
        'FA': 2.53, 'MA': 2.17, 'Cs': 1.67,
        'Pb': 1.19, 'Sn': 1.10,
        'I': 2.20, 'Br': 1.96, 'Cl': 1.81
    }
    
    # 加权平均
    r_A = sum(ionic_radii.get(c, 2.0) * frac for c, frac in a_cations.items())
    r_B = ionic_radii.get(b_cation, 1.2)
    r_X = sum(ionic_radii.get(x, 2.0) * frac for x, frac in x_anions.items())
    
    t = (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))
    
    return t


def calculate_octahedral_factor(b_cation: str, x_anions: Dict[str, float]) -> float:
    """计算八体因子"""
    ionic_radii = {
        'Pb': 1.19, 'Sn': 1.10,
        'I': 2.20, 'Br': 1.96, 'Cl': 1.81
    }
    
    r_B = ionic_radii.get(b_cation, 1.2)
    r_X = sum(ionic_radii.get(x, 2.0) * frac for x, frac in x_anions.items())
    
    return r_B / r_X


def run_full_workflow():
    """
    运行完整工作流
    
    从计算筛选到合成优化
    """
    print("\n" + "#" * 60)
    print("# Full Self-Driving Perovskite Workflow")
    print("#" * 60)
    
    # 1. 发现组成
    print("\n## Phase 1: Computational Screening")
    candidates = discover_perovskite_composition(
        target_bandgap=1.5,
        max_experiments=10
    )
    
    if not candidates:
        print("No successful candidates found")
        return
    
    best_composition = candidates[0].get('properties', {})
    
    # 2. 合成规划
    print("\n## Phase 2: Synthesis Planning")
    # 构建组成字典
    comp_dict = {'Pb': 1, 'I': 3}  # 简化
    synthesis_plans = plan_perovskite_synthesis(comp_dict, method='solution')
    
    # 3. 条件优化
    print("\n## Phase 3: Condition Optimization")
    if synthesis_plans:
        best_conditions = optimize_synthesis_conditions(
            composition=comp_dict,
            initial_params=synthesis_plans[0],
            target_properties={'band_gap': 1.5},
            n_iterations=15
        )
    
    # 4. 稳定性设计
    print("\n## Phase 4: Stability Optimization")
    stable_design = design_stable_perovskite()
    
    print("\n" + "#" * 60)
    print("# Workflow Completed!")
    print("#" * 60)
    
    return {
        'composition': best_composition,
        'synthesis_plans': synthesis_plans,
        'optimal_conditions': best_conditions if synthesis_plans else None,
        'stability_design': stable_design
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Self-Driving Perovskite Discovery Demo")
    print("=" * 60)
    
    # 演示1: 组成发现
    print("\n" + "#" * 60)
    print("# Demo 1: Composition Discovery")
    print("#" * 60)
    
    candidates = discover_perovskite_composition(
        target_bandgap=1.6,
        max_experiments=8
    )
    
    # 演示2: 合成规划
    print("\n" + "#" * 60)
    print("# Demo 2: Synthesis Planning")
    print("#" * 60)
    
    composition = {'Cs': 0.1, 'FA': 0.9, 'Pb': 1, 'I': 3}
    plans = plan_perovskite_synthesis(composition)
    
    # 演示3: 条件优化
    print("\n" + "#" * 60)
    print("# Demo 3: Condition Optimization")
    print("#" * 60)
    
    if plans:
        optimal = optimize_synthesis_conditions(
            composition,
            plans[0],
            {'band_gap': 1.5},
            n_iterations=10
        )
    
    # 演示4: 稳定性设计
    print("\n" + "#" * 60)
    print("# Demo 4: Stability Design")
    print("#" * 60)
    
    stability_result = design_stable_perovskite()
    
    # 演示5: 完整工作流
    print("\n" + "#" * 60)
    print("# Demo 5: Full Workflow")
    print("#" * 60)
    
    full_result = run_full_workflow()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
