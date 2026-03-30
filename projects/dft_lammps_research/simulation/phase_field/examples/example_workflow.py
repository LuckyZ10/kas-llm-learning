"""
Phase Field - DFT Coupling Workflow Example
===========================================
相场-DFT多尺度耦合工作流示例

演示如何使用工作流模块自动化从DFT到相场的完整模拟流程。
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_field.workflow import PhaseFieldWorkflow, WorkflowConfig


def run_dft_to_phasefield_workflow():
    """运行DFT到相场的完整工作流"""
    
    print("=" * 70)
    print("Phase Field - DFT Multi-scale Coupling Workflow")
    print("=" * 70)
    
    # 配置工作流
    config = WorkflowConfig(
        name="LiCoO2_phase_separation",
        work_dir="./workflow_output",
        
        # 计算阶段
        run_dft=True,
        run_md=False,
        run_phase_field=True,
        
        # DFT输入 (模拟路径，实际应指向DFT计算结果)
        dft_output_path="./vasp_results",
        
        # 相场模型选择
        pf_model_type="electrochemical",  # 电化学相场
        pf_config={
            'nx': 128,
            'ny': 128,
            'dx': 2.0,
            'dt': 0.001,
            'total_steps': 5000,
            'temperature': 298.15,
            'E0': 3.9,
        },
        
        # 耦合模式
        coupling_mode="one_way",  # 单向耦合
        
        # 验证
        validate_results=True,
        compare_with_experiment=True,
        experimental_data={
            'domain_size': 50.0,  # nm (文献值)
            'diffusion_coefficient': 1e-15  # m²/s
        },
        
        # 输出
        save_intermediate=True,
        generate_report=True
    )
    
    # 创建工作流
    workflow = PhaseFieldWorkflow(config)
    
    # 定义回调函数
    def on_dft_complete(dft_params):
        print("\n[Callback] DFT extraction complete!")
        print(f"  - Diffusion coefficient: {dft_params.get('M', 'N/A')}")
        print(f"  - Gradient energy coefficient: {dft_params.get('kappa', 'N/A')}")
    
    def on_transfer_complete(transferred_params):
        print("\n[Callback] Parameter transfer complete!")
        print(f"  - Transferred {len(transferred_params)} parameters")
    
    def on_pf_complete(pf_result):
        print("\n[Callback] Phase field simulation complete!")
        print(f"  - Converged: {pf_result.get('converged')}")
        print(f"  - Final energy: {pf_result.get('final_energy', 0):.4f}")
    
    callbacks = {
        'on_dft_complete': on_dft_complete,
        'on_transfer_complete': on_transfer_complete,
        'on_pf_complete': on_pf_complete
    }
    
    # 运行工作流
    print("\nStarting workflow execution...")
    print("-" * 70)
    
    # 注意：此示例使用模拟数据
    # 实际运行时需要真实的DFT计算结果
    
    # 模拟DFT参数 (实际应从DFT计算获取)
    mock_dft_params = {
        'thermodynamic': {
            'chemical_potential': {
                'function': lambda c: 8.314 * 298.15 * np.log(c / (1 - c)),
                'coefficients': [1.0, -2.0, 1.0]
            },
            'gradient_energy_coefficient': 1.5e-9  # J/m
        },
        'kinetic': {
            'diffusion_coefficient': {
                'value': 1e-15,  # m²/s
                'barrier': 0.3,  # eV
                'temperature': 298.15
            }
        }
    }
    
    # 模拟运行
    print("\n[Step 1] Extracting DFT parameters (simulated)...")
    workflow.results['dft_params'] = mock_dft_params
    on_dft_complete(mock_dft_params)
    
    print("\n[Step 2] Transferring parameters...")
    from phase_field.coupling.parameter_transfer import ParameterTransfer
    transfer = ParameterTransfer()
    transferred = transfer.transfer_from_dft(mock_dft_params)
    workflow.results['transferred_params'] = transferred
    on_transfer_complete(transferred)
    
    print("\n[Step 3] Running phase field simulation...")
    # 使用简化的相场模拟
    from phase_field.core.cahn_hilliard import CahnHilliardSolver, CahnHilliardConfig
    
    pf_config = CahnHilliardConfig(
        nx=64, ny=64, dx=2.0, dt=0.001, total_steps=2000
    )
    pf_config.M = transferred.get('M', 1.0)
    pf_config.kappa = transferred.get('kappa', 1.0)
    
    solver = CahnHilliardSolver(pf_config)
    solver.initialize_fields(seed=42)
    
    print(f"  Running {pf_config.total_steps} steps...")
    result = solver.run()
    
    pf_result = {
        'model_type': 'cahn_hilliard',
        'total_steps': result['total_steps'],
        'converged': result['converged'],
        'final_energy': result['final_energy'],
        'domain_size': solver.get_domain_size()
    }
    workflow.results['phase_field_result'] = pf_result
    on_pf_complete(pf_result)
    
    print("\n[Step 4] Validating results...")
    validation = workflow._validate_results()
    print(f"  Validation status: {validation['status']}")
    for check in validation.get('checks', []):
        print(f"  - {check['name']}: {check['status']}")
    
    print("\n[Step 5] Generating report...")
    workflow._generate_report()
    
    # 最终结果
    print("\n" + "=" * 70)
    print("Workflow Summary")
    print("=" * 70)
    print(f"\nTransmitted Parameters:")
    for key, value in transferred.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4e}")
    
    print(f"\nPhase Field Results:")
    print(f"  Domain size: {pf_result['domain_size']:.2f} nm")
    print(f"  Final energy: {pf_result['final_energy']:.4f}")
    print(f"  Converged: {pf_result['converged']}")
    
    print(f"\nOutput files saved to: {config.work_dir}/")
    print("  - workflow_report.md")
    print("  - dft_parameters.json")
    print("  - transferred_parameters.json")
    print("  - phase_field_result.json")
    print("  - validation.json")
    
    return workflow


def demonstrate_parameter_conversion():
    """演示参数转换过程"""
    
    print("\n" + "=" * 70)
    print("Parameter Conversion Demonstration")
    print("=" * 70)
    
    from phase_field.coupling.parameter_transfer import ParameterTransfer
    
    transfer = ParameterTransfer()
    
    # 模拟DFT参数
    print("\nInput DFT Parameters:")
    dft_params = {
        'thermodynamic': {
            'gradient_energy_coefficient': 1.5e-9,  # J/m
            'elastic_constants': np.array([[200, 100, 0],
                                           [100, 200, 0],
                                           [0, 0, 50]])  # GPa
        },
        'kinetic': {
            'diffusion_coefficient': {
                'value': 1e-15,  # m²/s
                'barrier': 0.3   # eV
            }
        }
    }
    print(f"  κ (DFT): {dft_params['thermodynamic']['gradient_energy_coefficient']:.2e} J/m")
    print(f"  D (DFT): {dft_params['kinetic']['diffusion_coefficient']['value']:.2e} m²/s")
    
    # 转换
    pf_params = transfer.transfer_from_dft(dft_params)
    
    print("\nConverted Phase Field Parameters:")
    print(f"  κ (PF): {pf_params.get('kappa', 0):.2e} eV/nm")
    print(f"  D (PF): {pf_params.get('diffusion_coefficient', 0):.2e} nm²/s")
    print(f"  M (PF): {pf_params.get('mobility', 0):.2e} (PF units)")
    
    return pf_params


if __name__ == "__main__":
    # 运行工作流示例
    workflow = run_dft_to_phasefield_workflow()
    
    # 演示参数转换
    demonstrate_parameter_conversion()
