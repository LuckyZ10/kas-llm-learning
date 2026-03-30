#!/usr/bin/env python3
"""
Li3PS4工作流完整示例
==================

演示从Materials Project获取Li3PS4结构，经过完整的DFT+ML+MD工作流，
最终预测离子电导率的完整过程。

本示例展示：
1. 从MP获取Li3PS4结构
2. VASP DFT结构优化
3. DeepMD机器学习势训练
4. 多温度LAMMPS MD模拟
5. 扩散系数和离子电导率分析
6. Arrhenius拟合得到活化能

Usage:
    python Li3PS4_workflow_example.py [--skip-dft] [--skip-ml] [--use-existing]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')

from core.common.workflow_engine import (
    IntegratedMaterialsWorkflow,
    IntegratedWorkflowConfig,
    MaterialsProjectConfig,
    DFTStageConfig,
    MLPotentialConfig,
    MDStageConfig,
    AnalysisConfig,
    WorkflowStage,
)


def setup_li3ps4_workflow(output_dir: str = "./Li3PS4_workflow",
                          skip_dft: bool = False,
                          skip_ml: bool = False) -> IntegratedWorkflowConfig:
    """
    设置Li3PS4专用工作流配置
    
    针对硫化物固态电解质优化的参数设置
    """
    
    print("="*70)
    print("Li3PS4固态电解质计算工作流")
    print("="*70)
    print(f"\n工作目录: {output_dir}")
    print("目标体系: Li3PS4 (硫化物固态电解质)")
    print("\n配置信息:")
    
    config = IntegratedWorkflowConfig(
        workflow_name="Li3PS4_Solid_Electrolyte",
        working_dir=output_dir,
        
        # Materials Project配置
        mp_config=MaterialsProjectConfig(
            max_entries=10,
            query_criteria={
                'chemsys': 'Li-P-S',
                'nsites': {'$lte': 50},
            }
        ),
        
        # DFT配置 - 针对Li3PS4优化
        dft_config=DFTStageConfig(
            code="vasp",
            functional="PBE",
            encut=520,  # 对S元素足够
            kpoints_density=0.2,  # 对超胞足够
            ediff=1e-6,
            ncores=32,
            max_steps=200,
            fmax=0.01,  # 严格收敛
        ),
        
        # ML势配置
        ml_config=MLPotentialConfig(
            framework="deepmd",
            preset="accurate",  # 为了精确预测扩散
            num_models=4,  # 集成4个模型用于不确定性量化
            max_iterations=5,
            uncertainty_threshold=0.15,
        ),
        
        # MD配置 - 针对离子传导优化
        md_config=MDStageConfig(
            ensemble="nvt",
            temperatures=[300, 400, 500, 600, 700, 800, 900],  # 覆盖工作温度范围
            timestep=1.0,  # 1 fs，对Li离子足够
            nsteps_equil=100000,  # 100 ps平衡
            nsteps_prod=500000,   # 500 ps生产运行
            nprocs=4,
        ),
        
        # 分析配置
        analysis_config=AnalysisConfig(
            compute_diffusion=True,
            compute_conductivity=True,
            compute_activation_energy=True,
            compute_vibration=False,
        ),
        
        max_parallel=4,
        save_intermediate=True,
        generate_report=True,
    )
    
    # 配置阶段
    config.stages["fetch_structure"] = WorkflowStage("fetch_structure", enabled=True)
    config.stages["dft_calculation"] = WorkflowStage(
        "dft_calculation", 
        enabled=not skip_dft,
        retry_count=2
    )
    config.stages["ml_training"] = WorkflowStage(
        "ml_training",
        enabled=not skip_ml,
        depends_on=["dft_calculation"],
        retry_count=1
    )
    config.stages["md_simulation"] = WorkflowStage(
        "md_simulation",
        enabled=True,
        depends_on=["ml_training"],
        timeout=7200  # 2小时超时
    )
    config.stages["analysis"] = WorkflowStage(
        "analysis",
        enabled=True,
        depends_on=["md_simulation"]
    )
    
    # 打印配置详情
    print(f"  DFT代码: {config.dft_config.code}")
    print(f"  截断能: {config.dft_config.encut} eV")
    print(f"  ML框架: {config.ml_config.framework}")
    print(f"  模型数量: {config.ml_config.num_models} (集成)")
    print(f"  MD系综: {config.md_config.ensemble}")
    print(f"  MD温度: {config.md_config.temperatures} K")
    print(f"  MD步数: {config.md_config.nsteps_prod} (生产)")
    print(f"  阶段: {', '.join([k for k, v in config.stages.items() if v.enabled])}")
    print()
    
    return config


def print_results_summary(results: Dict):
    """打印结果摘要"""
    
    print("\n" + "="*70)
    print("计算结果摘要")
    print("="*70)
    
    # 基本信息
    formula = results.get('formula', 'Unknown')
    print(f"\n化学式: {formula}")
    
    # DFT结果
    if 'dft' in results and results['dft'].get('success'):
        dft = results['dft']
        print(f"\n【DFT结构优化】")
        print(f"  总能量: {dft['energy']:.4f} eV")
        print(f"  单原子能量: {dft['energy_per_atom']:.4f} eV/atom")
        print(f"  最大力: {np.max(np.abs(dft['forces'])):.4f} eV/Å")
        if dft.get('stress'):
            print(f"  应力: {np.array(dft['stress']).mean():.4f} GPa")
    
    # ML模型
    if 'ml_models' in results:
        print(f"\n【ML势训练】")
        print(f"  模型文件:")
        for i, model in enumerate(results['ml_models']):
            print(f"    {i+1}. {model}")
    
    # MD轨迹
    if 'trajectories' in results:
        print(f"\n【MD模拟】")
        print(f"  生成轨迹:")
        for T, path in results['trajectories'].items():
            print(f"    {T}K: {path}")
    
    # 分析结果
    if 'analysis' in results:
        analysis = results['analysis']
        print(f"\n【扩散与电导率分析】")
        
        # 扩散系数
        if 'diffusion_coefficients' in analysis:
            print(f"\n  扩散系数 D (cm²/s):")
            print(f"  {'温度 (K)':<12} {'D':<15} {'log10(D)'}")
            print(f"  {'-'*40}")
            for T, D in sorted(analysis['diffusion_coefficients'].items()):
                if D > 0:
                    print(f"  {T:<12} {D:.2e}      {np.log10(D):.2f}")
        
        # 离子电导率
        if 'conductivities' in analysis:
            print(f"\n  离子电导率 σ (S/cm):")
            print(f"  {'温度 (K)':<12} {'σ':<15} {'log10(σ)'}")
            print(f"  {'-'*40}")
            for T, sigma in sorted(analysis['conductivities'].items()):
                if sigma > 0:
                    print(f"  {T:<12} {sigma:.2e}      {np.log10(sigma):.2f}")
        
        # 活化能
        if 'activation_energy' in analysis and analysis['activation_energy']:
            Ea = analysis['activation_energy']
            print(f"\n  【Arrhenius分析】")
            print(f"  活化能 Ea = {Ea:.3f} eV")
            if 'pre_exponential' in analysis:
                D0 = analysis['pre_exponential']
                print(f"  前置因子 D0 = {D0:.2e} cm²/s")
            
            # 与实验值比较
            print(f"\n  文献参考:")
            print(f"    Li3PS4 实验Ea: ~0.2-0.3 eV (晶相)")
            print(f"    计算Ea: {Ea:.3f} eV")
            if 0.15 < Ea < 0.4:
                print(f"    状态: ✓ 与实验值吻合良好")
            else:
                print(f"    状态: ⚠ 与实验值有偏差")
        
        # 预测室温电导率
        if 'diffusion_coefficients' in analysis:
            D_300 = analysis['diffusion_coefficients'].get(300, 0)
            sigma_300 = analysis['conductivities'].get(300, 0)
            if sigma_300 > 0:
                print(f"\n  【室温性能预测 (300K)】")
                print(f"  扩散系数: {D_300:.2e} cm²/s")
                print(f"  离子电导率: {sigma_300:.2e} S/cm")
                
                if sigma_300 > 1e-4:
                    print(f"  评价: 优秀固态电解质 (σ > 10⁻⁴ S/cm)")
                elif sigma_300 > 1e-5:
                    print(f"  评价: 良好固态电解质 (σ > 10⁻⁵ S/cm)")
                else:
                    print(f"  评价: 需要改进 (σ < 10⁻⁵ S/cm)")
    
    print("\n" + "="*70)


def generate_detailed_report(results: Dict, output_dir: str):
    """生成详细报告文件"""
    
    report_file = Path(output_dir) / "Li3PS4_detailed_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Li3PS4固态电解质计算详细报告\n")
        f.write("="*70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"工作目录: {output_dir}\n\n")
        
        # 详细结果
        f.write("1. 体系信息\n")
        f.write("-"*70 + "\n")
        f.write(f"化学式: {results.get('formula', 'N/A')}\n")
        f.write(f"材料ID: mp-30406 (Li3PS4)\n")
        f.write(f"晶体类型: 硫化物固态电解质\n\n")
        
        # DFT细节
        if 'dft' in results:
            f.write("2. DFT计算细节\n")
            f.write("-"*70 + "\n")
            dft = results['dft']
            f.write(f"代码: VASP\n")
            f.write(f"泛函: PBE\n")
            f.write(f"截断能: 520 eV\n")
            f.write(f"总能量: {dft.get('energy', 'N/A')} eV\n")
            f.write(f"单原子能量: {dft.get('energy_per_atom', 'N/A')} eV/atom\n\n")
        
        # ML训练细节
        if 'ml_models' in results:
            f.write("3. ML势训练细节\n")
            f.write("-"*70 + "\n")
            f.write(f"框架: DeepMD-kit\n")
            f.write(f"描述符: se_e2_a\n")
            f.write(f"模型数量: {len(results['ml_models'])} (集成学习)\n")
            for i, model in enumerate(results['ml_models']):
                f.write(f"  模型 {i+1}: {model}\n")
            f.write("\n")
        
        # MD细节
        if 'trajectories' in results:
            f.write("4. MD模拟细节\n")
            f.write("-"*70 + "\n")
            f.write(f"模拟引擎: LAMMPS\n")
            f.write(f"系综: NVT\n")
            f.write(f"时间步长: 1 fs\n")
            f.write(f"平衡步数: 100,000 (100 ps)\n")
            f.write(f"生产步数: 500,000 (500 ps)\n")
            f.write(f"模拟温度:\n")
            for T in sorted(results['trajectories'].keys()):
                f.write(f"  {T}K\n")
            f.write("\n")
        
        # 分析结果
        if 'analysis' in results:
            f.write("5. 离子传输分析\n")
            f.write("-"*70 + "\n")
            analysis = results['analysis']
            
            # 表格
            f.write(f"{'T (K)':<10} {'D (cm²/s)':<15} {'σ (S/cm)':<15}\n")
            f.write("-"*40 + "\n")
            
            temps = sorted(analysis.get('diffusion_coefficients', {}).keys())
            for T in temps:
                D = analysis['diffusion_coefficients'].get(T, 0)
                sigma = analysis['conductivities'].get(T, 0)
                f.write(f"{T:<10} {D:<15.2e} {sigma:<15.2e}\n")
            
            f.write("\n")
            
            # Arrhenius
            if 'activation_energy' in analysis:
                f.write(f"活化能: {analysis['activation_energy']:.3f} eV\n")
                f.write(f"前置因子: {analysis.get('pre_exponential', 0):.2e} cm²/s\n\n")
                
                f.write("Arrhenius方程:\n")
                f.write(f"  D = D0 * exp(-Ea/kT)\n")
                f.write(f"  其中 D0 = {analysis.get('pre_exponential', 0):.2e} cm²/s\n")
                f.write(f"       Ea = {analysis['activation_energy']:.3f} eV\n\n")
        
        f.write("6. 结论\n")
        f.write("-"*70 + "\n")
        
        if 'analysis' in results and 'conductivities' in results['analysis']:
            sigma_300 = results['analysis']['conductivities'].get(300, 0)
            Ea = results['analysis'].get('activation_energy', 0)
            
            f.write(f"Li3PS4预测的室温离子电导率: {sigma_300:.2e} S/cm\n")
            f.write(f"预测的活化能: {Ea:.3f} eV\n\n")
            
            if sigma_300 > 1e-4:
                f.write("该材料表现出优秀的离子导电性能，适合作为固态电解质。\n")
            elif sigma_300 > 1e-5:
                f.write("该材料表现出良好的离子导电性能，可用作固态电解质。\n")
            else:
                f.write("该材料的离子导电性需要进一步改进。\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("报告生成完成\n")
        f.write("="*70 + "\n")
    
    print(f"\n详细报告已保存: {report_file}")


def create_plots(results: Dict, output_dir: str):
    """创建分析图表"""
    
    try:
        import matplotlib.pyplot as plt
        
        analysis = results.get('analysis', {})
        
        if not analysis.get('diffusion_coefficients'):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        temps = sorted(analysis['diffusion_coefficients'].keys())
        D_values = [analysis['diffusion_coefficients'][T] for T in temps]
        sigma_values = [analysis['conductivities'].get(T, 0) for T in temps]
        
        # 1. Arrhenius plot
        ax = axes[0, 0]
        inv_T = 1000 / np.array(temps)  # 1000/T for better scale
        log_D = np.log(D_values)
        
        ax.scatter(inv_T, log_D, s=100, c='blue', label='MD Data')
        
        # Fit line
        if len(temps) >= 2:
            coeffs = np.polyfit(inv_T, log_D, 1)
            fit_line = np.poly1d(coeffs)
            ax.plot(inv_T, fit_line(inv_T), 'r--', 
                   label=f'Fit: Ea={analysis.get("activation_energy", 0):.3f} eV')
        
        ax.set_xlabel('1000/T (K⁻¹)')
        ax.set_ylabel('ln(D) [cm²/s]')
        ax.set_title('Arrhenius Plot - Li Diffusion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Temperature dependence of diffusion
        ax = axes[0, 1]
        ax.semilogy(temps, D_values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Diffusion Coefficient (cm²/s)')
        ax.set_title('Diffusion Coefficient vs Temperature')
        ax.grid(True, alpha=0.3)
        
        # 3. Ionic conductivity
        ax = axes[1, 0]
        valid_sigma = [(T, s) for T, s in zip(temps, sigma_values) if s > 0]
        if valid_sigma:
            T_valid, s_valid = zip(*valid_sigma)
            ax.semilogy(T_valid, s_valid, 's-', linewidth=2, markersize=8, color='green')
            ax.axhline(y=1e-4, color='r', linestyle='--', label='Target (10⁻⁴ S/cm)')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Ionic Conductivity (S/cm)')
        ax.set_title('Ionic Conductivity vs Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Summary bar chart
        ax = axes[1, 1]
        metrics = ['Activation\nEnergy (eV)', 'log10(D) at\n300K', 'log10(σ) at\n300K']
        
        Ea = analysis.get('activation_energy', 0)
        D_300 = analysis['diffusion_coefficients'].get(300, 1e-20)
        sigma_300 = analysis['conductivities'].get(300, 1e-20)
        
        values = [
            Ea,
            np.log10(D_300) if D_300 > 0 else -20,
            np.log10(sigma_300) if sigma_300 > 0 else -20
        ]
        
        colors = ['skyblue', 'lightgreen', 'salmon']
        bars = ax.bar(metrics, values, color=colors, edgecolor='black')
        ax.set_ylabel('Value')
        ax.set_title('Key Performance Metrics')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = Path(output_dir) / "Li3PS4_analysis_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"分析图表已保存: {plot_file}")
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping plots")
    except Exception as e:
        print(f"Failed to create plots: {e}")


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description="Li3PS4 workflow example - Full DFT+ML+MD pipeline"
    )
    parser.add_argument('-o', '--output', type=str, default='./Li3PS4_workflow',
                       help='Output directory')
    parser.add_argument('--skip-dft', action='store_true',
                       help='Skip DFT calculation (use existing)')
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip ML training (use existing)')
    parser.add_argument('--use-existing', action='store_true',
                       help='Use existing workflow results if available')
    
    args = parser.parse_args()
    
    # Check for existing results
    existing_report = Path(args.output) / "workflow_report.json"
    if args.use_existing and existing_report.exists():
        print(f"\nLoading existing results from {args.output}")
        with open(existing_report, 'r') as f:
            report = json.load(f)
            results = report.get('results', {})
        print_results_summary(results)
        return
    
    # Setup workflow
    config = setup_li3ps4_workflow(
        output_dir=args.output,
        skip_dft=args.skip_dft,
        skip_ml=args.skip_ml
    )
    
    # Create workflow
    workflow = IntegratedMaterialsWorkflow(config)
    
    # Run workflow
    print("\n开始执行工作流...\n")
    
    try:
        # Li3PS4 from Materials Project
        results = workflow.run(material_id="mp-30406")  # Li3PS4
        
        # Print results
        print_results_summary(results)
        
        # Generate detailed report
        generate_detailed_report(results, args.output)
        
        # Create plots
        create_plots(results, args.output)
        
        print(f"\n✓ 工作流成功完成!")
        print(f"所有结果保存在: {args.output}")
        
    except Exception as e:
        print(f"\n✗ 工作流失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
