#!/usr/bin/env python3
"""
快速入门示例 - 最小工作流
Quick Start Example - Minimal Workflow

运行方法:
    python simple_workflow.py

输出:
    ./quick_start_output/ 目录包含完整结果
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace/dft_lammps_research')

from core.common.workflow_engine import (
    IntegratedMaterialsWorkflow,
    IntegratedWorkflowConfig,
    MaterialsProjectConfig,
    DFTStageConfig,
    MLPotentialConfig,
    MDStageConfig
)


def main():
    """快速入门主函数"""
    
    print("="*60)
    print("DFT+LAMMPS 快速入门示例")
    print("DFT+LAMMPS Quick Start Example")
    print("="*60)
    
    # 步骤1: 配置工作流
    print("\n[1/3] 配置工作流...")
    config = IntegratedWorkflowConfig(
        workflow_name="quick_start_demo",
        working_dir="./quick_start_output",
        
        # Materials Project配置
        mp_config=MaterialsProjectConfig(
            api_key=None,  # 将从环境变量MP_API_KEY读取
            max_entries=10
        ),
        
        # DFT配置 (使用较低精度以加快示例速度)
        dft_config=DFTStageConfig(
            code="vasp",
            functional="PBE",
            encut=400,      # 降低截断能以加快速度
            kpoints_density=0.3,
            ncores=4,
            max_steps=100   # 减少优化步数
        ),
        
        # ML势训练配置
        ml_config=MLPotentialConfig(
            framework="deepmd",
            preset="fast",      # 快速训练模式
            num_models=2,       # 减少模型数量
            max_iterations=3    # 限制主动学习迭代
        ),
        
        # MD模拟配置
        md_config=MDStageConfig(
            temperatures=[300, 500],  # 两个温度点
            timestep=1.0,
            nsteps_equil=1000,        # 减少平衡步数
            nsteps_prod=5000,         # 减少生产步数
            nprocs=2
        )
    )
    
    # 步骤2: 创建工作流
    print("[2/3] 创建工作流...")
    workflow = IntegratedMaterialsWorkflow(config)
    
    # 步骤3: 运行工作流
    print("[3/3] 运行工作流 (这可能需要几分钟)...")
    print("-" * 60)
    
    try:
        # 从Materials Project获取Li3PS4结构
        results = workflow.run(formula="Li3PS4")
        
        # 输出结果
        print("\n" + "="*60)
        print("✓ 工作流完成! | Workflow Completed!")
        print("="*60)
        print(f"\n结果摘要 | Results Summary:")
        print(f"  化学式 | Formula: {results['formula']}")
        print(f"  DFT能量 | DFT Energy: {results['dft']['energy_per_atom']:.4f} eV/atom")
        
        if 'analysis' in results:
            analysis = results['analysis']
            print(f"  扩散系数 | Diffusion Coefficients:")
            for T, D in analysis.get('diffusion_coefficients', {}).items():
                print(f"    {T}K: {D:.2e} cm²/s")
            
            if analysis.get('activation_energy'):
                print(f"  活化能 | Activation Energy: {analysis['activation_energy']:.3f} eV")
        
        print(f"\n输出目录 | Output Directory: {config.working_dir}")
        print(f"报告文件 | Report: {config.working_dir}/workflow_report.json")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 错误 | Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
