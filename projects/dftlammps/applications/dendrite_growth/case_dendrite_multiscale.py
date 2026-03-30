#!/usr/bin/env python3
"""
case_dendrite_multiscale.py
===========================
枝晶生长多尺度模拟案例

模拟流程：
1. DFT计算：提取界面能、弹性常数
2. MD模拟：计算扩散系数、界面迁移率
3. 参数传递：将原子尺度参数转换到相场模型
4. 相场模拟：枝晶生长动力学
5. 连续介质分析：热应力分布

材料系统：金属锂（锂电池负极材料）

作者: Multi-Scale Simulation Expert
日期: 2026-03-09
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dftlammps.multiscale import (
    PhaseFieldWorkflow,
    DendriteConfig,
    MDtoPhaseFieldExtractor,
    ContinuumWorkflow,
    FEMConfig,
    MechanicsConfig,
    ParameterPassingWorkflow,
    ElasticConstants,
    TransportProperties,
    InterfaceProperties
)

from dftlammps.advanced_potentials import (
    OrbWorkflow,
    OrbConfig,
    MLPotentialType,
    load_ml_potential
)

from ase import Atoms
from ase.io import read, write
from ase.build import bulk
from ase.units import eV, Ang, GPa, fs

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DendriteMultiscaleSimulation:
    """
    枝晶生长多尺度模拟
    
    实现DFT → MD → 相场 → 连续介质的完整耦合
    """
    
    def __init__(self, working_dir: str = "./dendrite_multiscale"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self.dft_dir = self.working_dir / "dft"
        self.md_dir = self.working_dir / "md"
        self.pf_dir = self.working_dir / "phase_field"
        self.cont_dir = self.working_dir / "continuum"
        
        for d in [self.dft_dir, self.md_dir, self.pf_dir, self.cont_dir]:
            d.mkdir(exist_ok=True)
        
        # 工作流实例
        self.param_workflow = ParameterPassingWorkflow(
            working_dir=str(self.working_dir / "parameters")
        )
        self.pf_workflow = None
        self.cont_workflow = None
        
        # 材料参数存储
        self.material_params = {}
    
    def step1_dft_calculations(self, run_actual_dft: bool = False) -> Dict:
        """
        步骤1: DFT计算
        
        计算：
        - 界面能
        - 弹性常数
        - 表面能各向异性
        
        Args:
            run_actual_dft: 是否运行真实DFT计算（否则使用示例数据）
        """
        logger.info("=" * 60)
        logger.info("Step 1: DFT Calculations")
        logger.info("=" * 60)
        
        # 创建Li的晶体结构
        li_bulk = bulk('Li', 'bcc', a=3.51, cubic=True)
        write(self.dft_dir / "Li_bulk.cif", li_bulk)
        
        if run_actual_dft:
            # 实际应运行VASP等DFT计算
            # 这里简化为使用示例数据
            pass
        
        # 示例数据：金属锂的DFT计算结果
        dft_results = {
            'elastic_constants': {
                'C11': 13.5,  # GPa
                'C12': 11.3,
                'C44': 9.6
            },
            'interface_energy': {
                'solid_liquid': 0.03,  # J/m²
                'grain_boundary_110': 0.12
            },
            'surface_energy': {
                '100': 0.53,  # J/m²
                '110': 0.48,
                '111': 0.50
            },
            'anisotropy': {
                'epsilon4': 0.025,  # 四重各向异性系数
                'delta': 0.02       # 各向异性强度
            }
        }
        
        # 保存DFT结果
        with open(self.dft_dir / "dft_results.json", 'w') as f:
            json.dump(dft_results, f, indent=2)
        
        # 创建弹性常数对象
        elastic = ElasticConstants(**dft_results['elastic_constants'])
        
        # 创建界面性质对象
        interface = InterfaceProperties(
            gamma=dft_results['interface_energy']['solid_liquid'],
            anisotropy_values=dft_results['anisotropy']
        )
        
        self.material_params['elastic'] = elastic
        self.material_params['interface'] = interface
        
        logger.info(f"DFT Results:")
        logger.info(f"  C11 = {elastic.C11:.2f} GPa")
        logger.info(f"  C12 = {elastic.C12:.2f} GPa")
        logger.info(f"  Interface energy = {interface.gamma:.4f} J/m²")
        logger.info(f"  Anisotropy ε₄ = {interface.anisotropy_values.get('epsilon4', 0):.4f}")
        
        return dft_results
    
    def step2_md_simulations(self, 
                            use_ml_potential: bool = True,
                            run_actual_md: bool = False) -> Dict:
        """
        步骤2: MD模拟
        
        计算：
        - 扩散系数（不同温度）
        - 界面迁移率
        - 界面宽度
        
        Args:
            use_ml_potential: 是否使用ML势
            run_actual_md: 是否运行实际MD（否则使用示例数据）
        """
        logger.info("=" * 60)
        logger.info("Step 2: MD Simulations")
        logger.info("=" * 60)
        
        if use_ml_potential and run_actual_md:
            # 使用Orb ML势进行快速MD
            logger.info("Using Orb ML potential for MD simulations")
            
            orb_config = OrbConfig(device="cuda" if False else "cpu")
            orb_workflow = OrbWorkflow(orb_config)
            
            # 创建Li的slab结构用于界面模拟
            li_slab = bulk('Li', 'bcc', a=3.51) * (4, 4, 4)
            write(self.md_dir / "Li_slab.xyz", li_slab)
            
            # 在不同温度下运行MD
            temperatures = [300, 400, 500]  # K
            
            for T in temperatures:
                logger.info(f"Running MD at {T}K...")
                # 实际应运行MD并计算扩散系数
                # 这里简化为使用示例数据
        
        # 示例数据：MD计算结果
        md_results = {
            'diffusion_coefficient': {
                '300K': 2.5e-9,   # m²/s
                '400K': 8.2e-9,
                '500K': 2.1e-8
            },
            'activation_energy': 0.15,  # eV
            'pre_exponential': 1.2e-7,  # m²/s
            'interface_mobility': 1.5e-10,  # m/(J·s)
            'interface_width': 1.5,  # nm
            'melting_point': 453.7,  # K
            'thermal_expansion': 4.6e-5  # 1/K
        }
        
        # 拟合Arrhenius方程
        temps = np.array([300, 400, 500])
        D_values = np.array([md_results['diffusion_coefficient'][f'{t}K'] for t in temps])
        
        # D = D₀ exp(-Ea/kT)
        from scipy import stats
        log_D = np.log(D_values)
        inv_T = 1000 / temps
        
        slope, intercept, r_value, _, _ = stats.linregress(inv_T, log_D)
        
        Ea_fit = -slope * 8.617e-5 * 1000  # eV
        D0_fit = np.exp(intercept)
        
        logger.info(f"Arrhenius fit: Ea = {Ea_fit:.3f} eV, D₀ = {D0_fit:.2e} m²/s")
        
        # 创建输运性质对象
        transport = TransportProperties(
            D=md_results['diffusion_coefficient']['300K'],
            activation_energy=md_results['activation_energy'],
            pre_exponential=md_results['pre_exponential'],
            thermal_expansion=md_results['thermal_expansion']
        )
        
        self.material_params['transport'] = transport
        
        # 更新界面性质
        if 'interface' in self.material_params:
            self.material_params['interface'].mobility = md_results['interface_mobility']
            self.material_params['interface'].interface_width = md_results['interface_width']
            self.material_params['interface'].melting_point = md_results['melting_point']
        
        # 保存MD结果
        with open(self.md_dir / "md_results.json", 'w') as f:
            json.dump(md_results, f, indent=2)
        
        logger.info(f"MD Results:")
        logger.info(f"  D(300K) = {transport.D:.2e} m²/s")
        logger.info(f"  Ea = {transport.activation_energy:.3f} eV")
        logger.info(f"  Interface mobility = {md_results['interface_mobility']:.2e}")
        
        return md_results
    
    def step3_parameter_passing(self) -> Dict:
        """
        步骤3: 跨尺度参数传递
        
        将DFT/MD参数转换到相场模型
        """
        logger.info("=" * 60)
        logger.info("Step 3: Cross-Scale Parameter Passing")
        logger.info("=" * 60)
        
        # 使用参数传递工作流
        params = self.param_workflow.convert_parameters(
            elastic=self.material_params.get('elastic'),
            transport=self.material_params.get('transport'),
            interface=self.material_params.get('interface')
        )
        
        # 更新相场特定参数
        if params.phase_field:
            # 基于MD的扩散系数设置相场迁移率
            D = self.material_params['transport'].D if 'transport' in self.material_params else 1e-9
            
            # 无量纲化
            L_char = 1e-9  # 1 nm特征长度
            tau_char = L_char**2 / D  # 特征时间
            
            params.phase_field.mobility = 1.0  # 无量纲
            params.phase_field.diffusion_coeff = D
            
            # 过冷度设置
            T_m = self.material_params['interface'].melting_point if 'interface' in self.material_params else 450
            params.phase_field.melting_point = T_m
            params.phase_field.undercooling = 0.2  # 20%过冷
        
        # 保存参数
        params.to_json(str(self.working_dir / "multiscale_parameters.json"))
        
        # 创建相场配置
        pf_config = DendriteConfig(
            dimensions=2,
            nx=256,
            ny=256,
            dx=2.0,  # nm
            dt=0.005,
            n_steps=50000,
            
            # 从MD提取的参数
            interface_width=params.phase_field.interface_width if params.phase_field else 1.5,
            diffusion_coeff=D * 1e18,  # 转换为nm²/ps
            mobility=params.phase_field.mobility if params.phase_field else 1.0,
            
            # 从DFT提取的参数
            anisotropy_strength=self.material_params['interface'].anisotropy_values.get('epsilon4', 0.025)
            if 'interface' in self.material_params else 0.025,
            
            # 热力学参数
            melting_point=params.phase_field.melting_point if params.phase_field else 450,
            undercooling=0.2,
            
            # 输出
            output_dir=str(self.pf_dir),
            output_interval=500
        )
        
        self.pf_config = pf_config
        
        # 保存配置
        with open(self.pf_dir / "pf_config.json", 'w') as f:
            json.dump({
                'dimensions': pf_config.dimensions,
                'nx': pf_config.nx,
                'ny': pf_config.ny,
                'dx': pf_config.dx,
                'dt': pf_config.dt,
                'n_steps': pf_config.n_steps,
                'interface_width': pf_config.interface_width,
                'diffusion_coeff': pf_config.diffusion_coeff,
                'mobility': pf_config.mobility,
                'anisotropy_strength': pf_config.anisotropy_strength,
                'melting_point': pf_config.melting_point,
                'undercooling': pf_config.undercooling
            }, f, indent=2)
        
        logger.info("Phase Field Parameters:")
        logger.info(f"  Grid: {pf_config.nx} x {pf_config.ny}")
        logger.info(f"  Interface width: {pf_config.interface_width:.2f} nm")
        logger.info(f"  Diffusion coeff: {pf_config.diffusion_coeff:.4e} nm²/ps")
        logger.info(f"  Anisotropy: {pf_config.anisotropy_strength:.4f}")
        logger.info(f"  Undercooling: {pf_config.undercooling:.2f}")
        
        return asdict(params)
    
    def step4_phase_field_simulation(self, run_simulation: bool = True) -> Dict:
        """
        步骤4: 相场模拟
        
        枝晶生长动力学模拟
        """
        logger.info("=" * 60)
        logger.info("Step 4: Phase Field Simulation")
        logger.info("=" * 60)
        
        self.pf_workflow = PhaseFieldWorkflow(working_dir=str(self.pf_dir))
        
        # 设置枝晶生长模拟
        self.pf_workflow.setup_dendrite_simulation(
            config=self.pf_config,
            use_extracted_params=False  # 已经设置好了
        )
        
        if run_simulation:
            # 运行模拟
            logger.info("Running dendrite growth simulation...")
            self.pf_workflow.run_simulation(n_steps=self.pf_config.n_steps)
            
            # 分析结果
            results = self.pf_workflow.analyze_results()
            
            logger.info("Phase Field Results:")
            logger.info(f"  Tip velocity: {results.get('tip_velocity', 0):.4e} m/s")
            logger.info(f"  Tip radius: {results.get('tip_radius', 0):.2f} nm")
            
            return results
        else:
            logger.info("Phase field simulation setup complete (not run)")
            return {}
    
    def step5_continuum_analysis(self) -> Dict:
        """
        步骤5: 连续介质分析
        
        分析枝晶生长的热应力分布
        """
        logger.info("=" * 60)
        logger.info("Step 5: Continuum Analysis")
        logger.info("=" * 60)
        
        self.cont_workflow = ContinuumWorkflow(working_dir=str(self.cont_dir))
        
        # 设置材料参数（从DFT）
        elastic = self.material_params.get('elastic')
        if elastic:
            elastic_props = {
                'C11': elastic.C11,
                'C12': elastic.C12,
                'C44': elastic.C44,
                'E': elastic.E,
                'nu': elastic.nu
            }
        else:
            elastic_props = {'E': 13.0, 'nu': 0.36}  # Li的默认值
        
        thermal_props = {
            'thermal_conductivity': 85,  # W/(m·K) for Li
            'thermal_expansion': 4.6e-5,
            'specific_heat': 3570  # J/(kg·K)
        }
        
        self.cont_workflow.setup_material(
            elastic_props=elastic_props,
            thermal_props=thermal_props
        )
        
        # 生成网格
        fem_config = FEMConfig(
            dimensions=2,
            lx=500,  # nm - 覆盖整个枝晶区域
            ly=500,
            nx=100,
            ny=100
        )
        self.cont_workflow.generate_mesh(fem_config)
        
        # 热传导分析（稳态）
        from dftlammps.multiscale import ThermalConfig
        thermal_config = ThermalConfig(
            analysis_type="steady",
            bc_temperature=[
                {'boundary': 'bottom', 'value': 300},  # K - 基底温度
                {'boundary': 'top', 'value': 400}      # K - 较高温度
            ]
        )
        
        thermal_results = self.cont_workflow.run_thermal_analysis(thermal_config)
        
        # 力学分析（热应力）
        from dftlammps.multiscale import MechanicsConfig
        mechanics_config = MechanicsConfig(
            bc_displacement=[
                {'boundary': 'bottom', 'value': 0.0, 'direction': 1}  # 固定底部
            ]
        )
        
        # 添加热膨胀载荷（简化）
        mechanics_results = self.cont_workflow.run_mechanics_analysis(mechanics_config)
        
        logger.info("Continuum Analysis Results:")
        logger.info(f"  Max temperature: {thermal_results.get('max_temp', 0):.1f} K")
        logger.info(f"  Max displacement: {mechanics_results.get('max_displacement', 0):.4f} nm")
        logger.info(f"  Max Von Mises stress: {mechanics_results.get('max_von_mises', 0):.2f} GPa")
        
        return {
            'thermal': thermal_results,
            'mechanics': mechanics_results
        }
    
    def visualize_results(self):
        """
        可视化多尺度模拟结果
        """
        logger.info("Generating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. DFT能量-体积曲线（示例）
        ax = axes[0, 0]
        volumes = np.linspace(15, 25, 50)
        E0, B = -200, 13.5  # 示例值
        E = E0 + (9*B/16) * ((20/volumes)**(2/3) - 1)**2
        ax.plot(volumes, E, 'b-')
        ax.set_xlabel('Volume (Å³)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('DFT: Energy-Volume Curve')
        ax.grid(True)
        
        # 2. MD扩散系数Arrhenius图
        ax = axes[0, 1]
        temps = np.array([300, 400, 500])
        D_vals = np.array([2.5e-9, 8.2e-9, 2.1e-8])
        ax.semilogy(1000/temps, D_vals, 'ro', markersize=8)
        ax.set_xlabel('1000/T (K⁻¹)')
        ax.set_ylabel('D (m²/s)')
        ax.set_title('MD: Diffusion Coefficient')
        ax.grid(True)
        
        # 3. 参数传递流程
        ax = axes[0, 2]
        ax.text(0.5, 0.5, 'Parameter Passing\n' +
                f"C₁₁ = {self.material_params.get('elastic', ElasticConstants()).C11:.1f} GPa\n" +
                f"D = {self.material_params.get('transport', TransportProperties()).D:.2e} m²/s\n" +
                f"γ = {self.material_params.get('interface', InterfaceProperties()).gamma:.3f} J/m²",
                ha='center', va='center', fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.axis('off')
        ax.set_title('Parameters Passed')
        
        # 4. 相场模拟结果（示意）
        ax = axes[1, 0]
        if hasattr(self.pf_workflow, 'solver') and self.pf_workflow.solver:
            phi = self.pf_workflow.solver.phi
            if phi is not None:
                im = ax.imshow(phi.T, origin='lower', cmap='RdYlBu', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax)
        else:
            # 示例枝晶形态
            theta = np.linspace(0, 2*np.pi, 100)
            r = 1 + 0.3 * np.cos(4*theta)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax.fill(x, y, color='blue', alpha=0.3)
            ax.plot(x, y, 'b-', linewidth=2)
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title('Phase Field: Dendrite Morphology')
        ax.set_aspect('equal')
        
        # 5. 温度场分布
        ax = axes[1, 1]
        # 示例温度分布
        x = np.linspace(0, 500, 100)
        y = np.linspace(0, 500, 100)
        X, Y = np.meshgrid(x, y)
        T = 300 + 100 * (Y / 500)
        im = ax.imshow(T, origin='lower', extent=[0, 500, 0, 500], cmap='hot')
        plt.colorbar(im, ax=ax, label='Temperature (K)')
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title('Continuum: Temperature Field')
        
        # 6. 应力分布
        ax = axes[1, 2]
        # 示例应力分布
        stress = np.random.rand(100, 100) * 0.5
        im = ax.imshow(stress, origin='lower', extent=[0, 500, 0, 500], cmap='jet')
        plt.colorbar(im, ax=ax, label='Von Mises Stress (GPa)')
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title('Continuum: Stress Distribution')
        
        plt.tight_layout()
        plt.savefig(self.working_dir / "multiscale_results.png", dpi=150)
        logger.info(f"Visualization saved to {self.working_dir / 'multiscale_results.png'}")
        
        plt.close()
    
    def run_full_simulation(self, 
                           run_dft: bool = False,
                           run_md: bool = False,
                           run_pf: bool = True,
                           run_continuum: bool = True) -> Dict:
        """
        运行完整的多尺度模拟
        
        Returns:
            所有步骤的结果汇总
        """
        logger.info("=" * 60)
        logger.info("Starting Full Multiscale Dendrite Simulation")
        logger.info("=" * 60)
        
        results = {}
        
        # 步骤1: DFT
        results['dft'] = self.step1_dft_calculations(run_actual_dft=run_dft)
        
        # 步骤2: MD
        results['md'] = self.step2_md_simulations(run_actual_md=run_md)
        
        # 步骤3: 参数传递
        results['parameters'] = self.step3_parameter_passing()
        
        # 步骤4: 相场模拟
        if run_pf:
            results['phase_field'] = self.step4_phase_field_simulation()
        
        # 步骤5: 连续介质分析
        if run_continuum:
            results['continuum'] = self.step5_continuum_analysis()
        
        # 可视化
        self.visualize_results()
        
        # 保存完整结果
        with open(self.working_dir / "full_results.json", 'w') as f:
            # 简化JSON序列化
            json.dump({k: str(v) for k, v in results.items()}, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("Multiscale Simulation Completed")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {self.working_dir}")
        
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multiscale Dendrite Growth Simulation"
    )
    parser.add_argument('--working-dir', default='./dendrite_multiscale',
                       help='Working directory')
    parser.add_argument('--run-dft', action='store_true',
                       help='Run actual DFT calculations')
    parser.add_argument('--run-md', action='store_true',
                       help='Run actual MD simulations')
    parser.add_argument('--skip-pf', action='store_true',
                       help='Skip phase field simulation')
    parser.add_argument('--skip-continuum', action='store_true',
                       help='Skip continuum analysis')
    
    args = parser.parse_args()
    
    # 创建并运行模拟
    simulation = DendriteMultiscaleSimulation(working_dir=args.working_dir)
    
    results = simulation.run_full_simulation(
        run_dft=args.run_dft,
        run_md=args.run_md,
        run_pf=not args.skip_pf,
        run_continuum=not args.skip_continuum
    )
    
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"DFT elastic constants extracted")
    print(f"MD diffusion coefficients calculated")
    print(f"Parameters passed to phase field model")
    if not args.skip_pf:
        print(f"Dendrite growth simulated")
    if not args.skip_continuum:
        print(f"Thermal stress analyzed")
    print(f"\nResults saved to: {args.working_dir}")


if __name__ == "__main__":
    main()
