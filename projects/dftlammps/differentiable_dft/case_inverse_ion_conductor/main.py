"""
Case Study: Inverse Ion Conductor Design
========================================

离子导体逆向设计案例研究

本案例展示如何使用可微分DFT进行离子导体逆向设计，包括：
1. 锂离子电池固态电解质设计
2. 钠离子电池NASICON型电解质优化
3. 硫化物高电导率电解质设计

运行方式:
    python case_inverse_ion_conductor/main.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from inverse_design import (
    FractionalCoordinateStructure,
    InverseDesignOptimizer,
    IonConductorTarget,
    IonMigrationAnalyzer,
    SolidElectrolyteDesigner,
    NASICONDesigner,
    SulfideElectrolyteDesigner
)


class MockDFTEngine:
    """模拟DFT引擎用于演示"""
    
    def __init__(self):
        pass
    
    def calculate_ion_migration(self, positions, atomic_numbers, cell, ion_type='Li'):
        """模拟离子迁移计算"""
        # 基于通道几何的启发式模型
        
        # 计算平均原子间距 (通道尺寸近似)
        n_atoms = len(positions)
        volume = jnp.abs(jnp.linalg.det(cell))
        avg_spacing = (volume / n_atoms) ** (1/3)
        
        # 通道半径近似
        channel_radius = avg_spacing * 0.3
        
        # 离子半径
        ion_radii = {'Li': 0.76, 'Na': 1.02, 'K': 1.38, 'Ag': 1.15}
        r_ion = ion_radii.get(ion_type, 1.0)
        
        # 尺寸匹配因子 (越大越好)
        size_match = channel_radius / r_ion
        
        # 迁移势垒 (eV) - 尺寸匹配越好，势垒越低
        Ea = 0.5 - 0.2 * jnp.tanh(size_match - 1.0)
        Ea = jnp.clip(Ea, 0.1, 1.0)
        
        # 添加位置无序度的影响
        disorder = jnp.std(positions) * 0.1
        Ea += disorder
        
        # 计算电导率 (Arrhenius)
        kT = 0.0258  # 300K
        sigma_0 = 1e3  # S/cm
        sigma = sigma_0 * jnp.exp(-Ea / kT)
        
        return {
            'activation_energy': float(Ea),
            'ionic_conductivity': float(sigma),
            'channel_radius': float(channel_radius)
        }


class MockIonCalculator:
    """离子计算器包装类"""
    
    def __init__(self, dft_engine):
        self.dft = dft_engine
    
    def calculate(self, positions, atomic_numbers, cell, ion_type='Li'):
        return self.dft.calculate_ion_migration(positions, atomic_numbers, cell, ion_type)


def case_lithium_conductor():
    """
    案例1: 锂离子电池固态电解质设计
    
    目标: 设计具有高锂离子电导率的固态电解质
    """
    print("=" * 70)
    print("案例1: 锂离子电池固态电解质设计")
    print("=" * 70)
    
    dft_engine = MockDFTEngine()
    ion_calc = MockIonCalculator(dft_engine)
    designer = SolidElectrolyteDesigner(dft_engine)
    
    # 创建初始结构 (硫化物电解质类似物)
    print("\n创建初始结构...")
    structure = FractionalCoordinateStructure(
        n_atoms=8,
        atomic_numbers=jnp.array([3, 3, 16, 16, 16, 16, 15, 15]),  # Li2P2S4-like
        initial_cell=jnp.eye(3) * 9.0,
        fix_cell=False
    )
    
    init_pos, init_nums, init_cell = structure.to_structure()
    init_props = ion_calc.calculate(init_pos, init_nums, init_cell, 'Li')
    
    print(f"初始结构: Li-P-S体系")
    print(f"初始电导率: {init_props['ionic_conductivity']:.2e} S/cm")
    print(f"初始活化能: {init_props['activation_energy']:.3f} eV")
    
    # 设计目标
    target = IonConductorTarget(
        ion_type='Li',
        target_conductivity=1e-3,  # 1 mS/cm
        min_migration_barrier=0.25,
        temperature=300.0
    )
    
    print(f"\n设计目标:")
    print(f"  目标电导率: {target.target_conductivity:.0e} S/cm")
    print(f"  最大迁移势垒: {target.min_migration_barrier} eV")
    print(f"  温度: {target.temperature} K")
    
    # 定义目标函数
    def objective(params, struct):
        struct.set_params(params)
        pos, nums, cell = struct.to_structure()
        return designer._ion_conductor_loss(pos, nums, cell, target)
    
    # 优化
    optimizer = InverseDesignOptimizer(
        None, 'adam', 0.02, 300
    )
    optimizer.objective = objective
    
    print("\n开始优化...")
    result = optimizer.optimize(structure)
    
    # 评估结果
    final_pos, final_nums, final_cell = result.to_structure()
    final_props = ion_calc.calculate(final_pos, final_nums, final_cell, 'Li')
    
    # 迁移分析
    analyzer = IonMigrationAnalyzer(dft_engine)
    percolation = analyzer.percolation_analysis(
        final_pos, final_nums, final_cell, 'Li'
    )
    
    print(f"\n优化完成!")
    print(f"最终电导率: {final_props['ionic_conductivity']:.2e} S/cm")
    print(f"最终活化能: {final_props['activation_energy']:.3f} eV")
    print(f"渗流维度: {percolation['dimensionality']}")
    print(f"低势垒路径数: {percolation['n_low_barrier_paths']}")
    
    # 温度依赖性
    print(f"\n电导率温度依赖性:")
    for T in [250, 300, 350, 400]:
        kT = 8.617e-5 * T
        sigma_T = 1e3 * np.exp(-final_props['activation_energy'] / kT)
        print(f"  {T}K: {sigma_T:.2e} S/cm")
    
    return {
        'initial_conductivity': init_props['ionic_conductivity'],
        'final_conductivity': final_props['ionic_conductivity'],
        'activation_energy': final_props['activation_energy'],
        'percolation': percolation,
        'history': optimizer.history
    }


def case_nasicon_optimization():
    """
    案例2: NASICON钠离子导体优化
    
    优化Si/P比例以获得最佳电导率
    """
    print("\n" + "=" * 70)
    print("案例2: NASICON钠离子导体优化")
    print("=" * 70)
    
    dft_engine = MockDFTEngine()
    nasicon_designer = NASICONDesigner(dft_engine)
    
    print("\n扫描NASICON组成空间...")
    print("化学式: Na1+xZr2SixP3-xO12")
    print("x范围: 0.0 - 2.0")
    
    results = []
    x_values = np.linspace(0, 2, 5)
    
    for x in x_values:
        print(f"\n--- 组成 x = {x:.1f} ---")
        
        # 创建NASICON结构
        structure = nasicon_designer.create_nasicon_template(
            x_composition=x,
            lattice_param=15.0
        )
        
        # 快速评估 (不运行完整优化)
        pos, nums, cell = structure.to_structure()
        props = dft_engine.calculate_ion_migration(pos, nums, cell, 'Na')
        
        print(f"  电导率: {props['ionic_conductivity']:.2e} S/cm")
        print(f"  活化能: {props['activation_energy']:.3f} eV")
        
        results.append({
            'x': x,
            'conductivity': props['ionic_conductivity'],
            'activation_energy': props['activation_energy']
        })
    
    # 找到最佳组成
    best = max(results, key=lambda r: r['conductivity'])
    
    print("\n" + "-" * 50)
    print("NASICON组成优化结果:")
    print("-" * 50)
    print(f"{'Si含量 (x)':<15} {'电导率 (S/cm)':<20} {'活化能 (eV)':<15}")
    print("-" * 50)
    for r in results:
        marker = " *" if r == best else ""
        print(f"{r['x']:<15.1f} {r['conductivity']:<20.2e} {r['activation_energy']:<15.3f}{marker}")
    
    print(f"\n最佳组成: x = {best['x']:.1f}")
    print(f"  电导率: {best['conductivity']:.2e} S/cm")
    print(f"  活化能: {best['activation_energy']:.3f} eV")
    
    return results


def case_sulfide_superionic():
    """
    案例3: 硫化物超离子导体设计
    
    设计类LGPS的高电导率硫化物电解质
    """
    print("\n" + "=" * 70)
    print("案例3: 硫化物超离子导体设计")
    print("=" * 70)
    
    dft_engine = MockDFTEngine()
    sulfide_designer = SulfideElectrolyteDesigner(dft_engine)
    ion_calc = MockIonCalculator(dft_engine)
    
    print("\n目标: 设计类LGPS的高电导率硫化物电解质")
    print("参考: Li10GeP2S12 (LGPS), σ ≈ 10 mS/cm")
    
    # 创建初始结构
    structure = FractionalCoordinateStructure(
        n_atoms=12,
        atomic_numbers=jnp.array([
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # Li (10个)
            32, 15, 15  # Ge, P, P
        ]),
        initial_cell=jnp.array([
            [8.7, 0, 0],
            [0, 8.7, 0],
            [0, 0, 12.6]
        ]),
        fix_cell=False
    )
    
    init_pos, init_nums, init_cell = structure.to_structure()
    init_props = ion_calc.calculate(init_pos, init_nums, init_cell, 'Li')
    
    print(f"\n初始结构: Li-rich硫化物")
    print(f"初始电导率: {init_props['ionic_conductivity']:.2e} S/cm")
    print(f"初始活化能: {init_props['activation_energy']:.3f} eV")
    
    # 高目标电导率
    target = IonConductorTarget(
        ion_type='Li',
        target_conductivity=1e-2,  # 10 mS/cm
        min_migration_barrier=0.20,
        conductivity_weight=2.0  # 更强调电导率
    )
    
    print(f"\n设计目标:")
    print(f"  目标电导率: {target.target_conductivity:.0e} S/cm")
    print(f"  参考材料: LGPS (Li10GeP2S12)")
    
    # 优化
    designer = SolidElectrolyteDesigner(dft_engine)
    
    def objective(params, struct):
        struct.set_params(params)
        pos, nums, cell = struct.to_structure()
        
        loss = designer._ion_conductor_loss(pos, nums, cell, target)
        
        # 额外鼓励三维渗流
        analyzer = IonMigrationAnalyzer(dft_engine)
        perc = analyzer.percolation_analysis(pos, nums, cell, 'Li')
        if perc['dimensionality'] == '3D':
            loss *= 0.85  # 15%奖励
        
        return loss
    
    optimizer = InverseDesignOptimizer(
        None, 'adam', 0.015, 400
    )
    optimizer.objective = objective
    
    print("\n开始优化...")
    result = optimizer.optimize(structure)
    
    # 评估
    final_pos, final_nums, final_cell = result.to_structure()
    final_props = ion_calc.calculate(final_pos, final_nums, final_cell, 'Li')
    
    analyzer = IonMigrationAnalyzer(dft_engine)
    percolation = analyzer.percolation_analysis(final_pos, final_nums, final_cell, 'Li')
    
    print(f"\n优化结果:")
    print(f"最终电导率: {final_props['ionic_conductivity']:.2e} S/cm")
    print(f"最终活化能: {final_props['activation_energy']:.3f} eV")
    print(f"渗流维度: {percolation['dimensionality']}")
    
    # 与参考材料比较
    lgps_conductivity = 1.2e-2  # S/cm
    ratio = final_props['ionic_conductivity'] / lgps_conductivity
    print(f"\n与LGPS比较:")
    print(f"  LGPS电导率: {lgps_conductivity:.1e} S/cm")
    print(f"  优化材料 / LGPS: {ratio:.2f}x")
    
    return {
        'initial_conductivity': init_props['ionic_conductivity'],
        'final_conductivity': final_props['ionic_conductivity'],
        'activation_energy': final_props['activation_energy'],
        'percolation': percolation,
        'history': optimizer.history
    }


def plot_results(li_results, nasicon_results, sulfide_results):
    """绘制结果图表"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 图1: 锂离子导体优化历史
        ax1 = axes[0]
        history = li_results['history']
        steps = [h['step'] for h in history]
        losses = [h['loss'] for h in history]
        ax1.semilogy(steps, losses, 'b-', linewidth=2)
        ax1.set_xlabel('Optimization Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Li-Ion Conductor: Optimization')
        ax1.grid(True, alpha=0.3)
        
        # 图2: NASICON组成优化
        ax2 = axes[1]
        x_vals = [r['x'] for r in nasicon_results]
        conds = [r['conductivity'] * 1e3 for r in nasicon_results]  # mS/cm
        eas = [r['activation_energy'] for r in nasicon_results]
        
        ax2_twin = ax2.twinx()
        ax2.bar(x_vals, conds, width=0.3, color='blue', alpha=0.6, label='Conductivity')
        ax2_twin.plot(x_vals, eas, 'ro-', linewidth=2, markersize=8, label='Ea')
        
        ax2.set_xlabel('Si content (x)')
        ax2.set_ylabel('Conductivity (mS/cm)', color='blue')
        ax2_twin.set_ylabel('Activation Energy (eV)', color='red')
        ax2.set_title('NASICON: Composition Optimization')
        ax2.grid(True, alpha=0.3)
        
        # 图3: 硫化物电导率对比
        ax3 = axes[2]
        materials = ['Initial', 'Optimized', 'LGPS (Ref)']
        conductivities = [
            sulfide_results['initial_conductivity'] * 1e3,
            sulfide_results['final_conductivity'] * 1e3,
            12  # LGPS: 12 mS/cm
        ]
        colors = ['gray', 'green', 'blue']
        bars = ax3.bar(materials, conductivities, color=colors, alpha=0.7)
        ax3.set_ylabel('Ionic Conductivity (mS/cm)')
        ax3.set_title('Sulfide Electrolyte Performance')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, conductivities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "ion_conductor_results.png", dpi=150)
        print(f"\n图表已保存至: {output_dir / 'ion_conductor_results.png'}")
        
    except Exception as e:
        print(f"绘图失败: {e}")


def main():
    """主函数"""
    print("\n" + "#" * 70)
    print("# 离子导体逆向设计案例研究")
    print("#" * 70)
    
    # 运行案例
    li_results = case_lithium_conductor()
    nasicon_results = case_nasicon_optimization()
    sulfide_results = case_sulfide_superionic()
    
    # 绘制结果
    print("\n" + "=" * 70)
    print("生成结果图表...")
    plot_results(li_results, nasicon_results, sulfide_results)
    
    # 总结
    print("\n" + "=" * 70)
    print("案例研究总结")
    print("=" * 70)
    
    print(f"\n1. 锂离子电池固态电解质:")
    print(f"   - 电导率: {li_results['initial_conductivity']:.2e} → {li_results['final_conductivity']:.2e} S/cm")
    print(f"   - 活化能: {li_results['activation_energy']:.3f} eV")
    print(f"   - 渗流维度: {li_results['percolation']['dimensionality']}")
    
    print(f"\n2. NASICON钠离子导体:")
    best_nasicon = max(nasicon_results, key=lambda r: r['conductivity'])
    print(f"   - 最佳Si含量: x = {best_nasicon['x']:.1f}")
    print(f"   - 最佳电导率: {best_nasicon['conductivity']:.2e} S/cm")
    
    print(f"\n3. 硫化物超离子导体:")
    print(f"   - 电导率: {sulfide_results['initial_conductivity']:.2e} → {sulfide_results['final_conductivity']:.2e} S/cm")
    print(f"   - 与LGPS比值: {sulfide_results['final_conductivity']/1.2e-2:.2f}x")
    
    print("\n" + "#" * 70)
    print("# 案例研究完成!")
    print("#" * 70)


if __name__ == "__main__":
    main()
