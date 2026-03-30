#!/usr/bin/env python3
"""
电池筛选管道使用示例
演示如何使用battery_screening_pipeline进行固态电解质筛选
"""

import os
import sys
sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')

from workflows.battery.screening import (
    BatteryScreeningPipeline,
    BatteryScreeningConfig,
    ScreeningCriteria,
    print_results_summary
)


def example_1_quick_screening():
    """示例1: 快速筛选 (仅使用MP数据，无DFT计算)"""
    
    print("\n" + "="*60)
    print("示例1: 快速筛选 (仅MP数据)")
    print("="*60)
    
    # 配置
    config = BatteryScreeningConfig(
        target_ion="Li",
        use_ml_acceleration=False,  # 不使用ML加速
    )
    
    # 筛选标准
    criteria = ScreeningCriteria(
        target_ion="Li",
        max_natoms=50,
        min_gap=3.0,
        max_ehull=0.05,
    )
    
    # 创建管道
    pipeline = BatteryScreeningPipeline(config)
    
    # 获取候选
    candidates = pipeline.fetch_candidates(criteria, max_entries=50, generate_variants=False)
    
    # 计算特征
    features_df = pipeline.compute_features(use_soap=False)
    
    print(f"\n找到 {len(candidates)} 个候选材料")
    print(f"特征维度: {features_df.shape}")
    print("\n前5个候选:")
    print(features_df.head())
    
    return candidates, features_df


def example_2_lithium_conductors():
    """示例2: Li离子导体筛选 (完整流程)"""
    
    print("\n" + "="*60)
    print("示例2: Li离子导体筛选")
    print("="*60)
    
    # 检查API密钥
    if not os.environ.get("MP_API_KEY"):
        print("警告: 请设置MP_API_KEY环境变量")
        print("export MP_API_KEY='your_api_key_here'")
        return
    
    # 配置
    config = BatteryScreeningConfig(
        target_ion="Li",
        max_natoms=80,
        use_ml_acceleration=True,
        md_temperatures=[300, 600, 900],
        md_nsteps_equil=10000,  # 减少平衡步数以加快演示
        md_nsteps_prod=50000,   # 减少生产步数以加快演示
    )
    
    # 筛选标准
    criteria = ScreeningCriteria(
        target_ion="Li",
        allowed_anions=["O", "S", "Se", "F", "Cl"],
        allowed_cations=["P", "Si", "Ge", "Al", "B", "Zr", "Ti", "Sn"],
        max_natoms=80,
        min_gap=2.0,
        max_ehull=0.1,
    )
    
    # 运行完整管道
    pipeline = BatteryScreeningPipeline(config)
    
    try:
        results = pipeline.run_full_pipeline(
            criteria=criteria,
            max_candidates=20,
            n_dft_screen=5
        )
        
        print_results_summary(results, top_n=10)
        
    except Exception as e:
        print(f"运行失败: {e}")
        import traceback
        traceback.print_exc()


def example_3_sodium_conductors():
    """示例3: Na离子导体筛选"""
    
    print("\n" + "="*60)
    print("示例3: Na离子导体筛选")
    print("="*60)
    
    config = BatteryScreeningConfig(
        target_ion="Na",
        max_natoms=100,
        use_ml_acceleration=False,  # 简化版本
    )
    
    criteria = ScreeningCriteria(
        target_ion="Na",
        allowed_anions=["O", "S", "Se", "F", "Cl"],
        max_natoms=100,
        min_gap=2.0,
        max_ehull=0.15,
    )
    
    pipeline = BatteryScreeningPipeline(config)
    
    # 获取候选
    candidates = pipeline.fetch_candidates(criteria, max_entries=30)
    
    # 计算特征
    features_df = pipeline.compute_features(use_soap=True)
    
    # 打印特征重要性高的列
    print("\n特征统计:")
    print(features_df.describe())
    
    return candidates, features_df


def example_4_custom_analysis():
    """示例4: 自定义分析 - 对特定化学系统筛选"""
    
    print("\n" + "="*60)
    print("示例4: 自定义分析 - Li-P-S系统")
    print("="*60)
    
    if not os.environ.get("MP_API_KEY"):
        print("警告: 请设置MP_API_KEY环境变量")
        return
    
    config = BatteryScreeningConfig(target_ion="Li")
    pipeline = BatteryScreeningPipeline(config)
    
    # 直接查询特定化学系统
    from mp_api.client import MPRester
    
    mpr = MPRester()
    docs = mpr.summary.search(
        chemsys="Li-P-S",
        fields=["material_id", "formula_pretty", "structure", 
               "band_gap", "energy_above_hull"],
        num_chunks=1,
        chunk_size=50
    )
    
    print(f"\nLi-P-S系统找到 {len(docs)} 个材料")
    
    for doc in docs[:5]:
        print(f"  {doc.material_id}: {doc.formula_pretty}")
        print(f"    带隙: {doc.band_gap:.2f} eV")
        print(f"    能量高于凸包: {doc.energy_above_hull:.3f} eV/atom")


def example_5_batch_processing():
    """示例5: 批量处理 - 多温度电导率计算"""
    
    print("\n" + "="*60)
    print("示例5: 批量处理 - 多温度电导率")
    print("="*60)
    
    config = BatteryScreeningConfig(
        target_ion="Li",
        md_temperatures=[300, 400, 500, 600, 700, 800, 900],
    )
    
    print("温度扫描范围:", config.md_temperatures)
    
    # 模拟Arrhenius分析
    import numpy as np
    
    # 模拟数据
    temperatures = np.array(config.md_temperatures)
    kB = 8.617e-5  # eV/K
    
    # 模拟Arrhenius行为
    Ea = 0.3  # eV
    D0 = 1e-4  # cm²/s
    D = D0 * np.exp(-Ea / (kB * temperatures))
    
    print("\n模拟扩散系数:")
    for T, d in zip(temperatures, D):
        print(f"  {T}K: D = {d:.2e} cm²/s")
    
    # Arrhenius拟合
    ln_D = np.log(D)
    inv_T = 1 / temperatures
    slope, intercept = np.polyfit(inv_T, ln_D, 1)
    fitted_Ea = -slope * kB
    
    print(f"\n拟合活化能: {fitted_Ea:.3f} eV (真实: {Ea:.3f} eV)")


def main():
    """运行示例"""
    
    print("\n" + "="*60)
    print("电池筛选管道示例")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Battery Screening Examples')
    parser.add_argument('--example', type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help='要运行的示例编号')
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_quick_screening,
        2: example_2_lithium_conductors,
        3: example_3_sodium_conductors,
        4: example_4_custom_analysis,
        5: example_5_batch_processing,
    }
    
    if args.example in examples:
        examples[args.example]()
    else:
        print(f"未知示例: {args.example}")


if __name__ == "__main__":
    main()
