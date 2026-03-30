"""
Case Study: Inverse Band Gap Design
===================================

带隙逆向设计案例研究

本案例展示如何使用可微分DFT进行带隙逆向设计，包括：
1. 太阳能电池材料设计 (目标带隙 ~1.3 eV)
2. LED材料设计 (特定发射波长)
3. 透明导电氧化物设计 (宽带隙 + 高导电性)

运行方式:
    python case_inverse_bandgap/main.py
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
    BandGapTarget,
    BandGapObjective,
    SolarCellOptimizer,
    LEDMaterialDesigner,
    TransparentConductorOptimizer
)


class MockDFTEngine:
    """模拟DFT引擎用于演示"""
    
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level
    
    def calculate_band_gap(self, positions, atomic_numbers, cell):
        """模拟带隙计算"""
        # 基于结构的启发式模型
        volume = jnp.abs(jnp.linalg.det(cell))
        n_electrons = jnp.sum(atomic_numbers)
        density = n_electrons / volume
        
        # 简化的带隙-密度关系
        base_gap = 3.0 - 0.5 * density + 0.3 * jnp.sin(density * 5)
        
        # 添加与位置相关的变化
        position_var = jnp.std(positions)
        gap = base_gap + 0.2 * position_var
        
        # 添加噪声
        noise = np.random.normal(0, self.noise_level)
        
        return {
            'band_gap': float(gap + noise),
            'direct_gap': float(gap + 0.15 + noise),
            'indirect_gap': float(gap + noise)
        }


class BandGapCalculator:
    """带隙计算器包装类"""
    
    def __init__(self, dft_engine):
        self.dft = dft_engine
    
    def calculate(self, positions, atomic_numbers, cell):
        return self.dft.calculate_band_gap(positions, atomic_numbers, cell)


def case_solar_cell():
    """
    案例1: 太阳能电池材料设计
    
    目标: 设计带隙约1.3 eV的材料，接近单结太阳能电池的效率极限
    """
    print("=" * 70)
    print("案例1: 太阳能电池材料设计")
    print("=" * 70)
    
    # 初始化DFT引擎
    dft_engine = MockDFTEngine()
    bg_calc = BandGapCalculator(dft_engine)
    
    # 创建初始结构 (AB型二元化合物)
    print("\n创建初始结构...")
    structure = FractionalCoordinateStructure(
        n_atoms=4,
        atomic_numbers=jnp.array([14, 14, 16, 16]),  # Si2O2-like
        initial_cell=jnp.eye(3) * 10.0,
        fix_cell=False
    )
    
    # 显示初始结构参数
    print(f"初始分数坐标: {structure.fractional_coords}")
    print(f"初始晶胞参数: {structure.cell_params if not structure.fix_cell else 'fixed'}")
    
    # 计算初始带隙
    init_pos, init_nums, init_cell = structure.to_structure()
    init_props = bg_calc.calculate(init_pos, init_nums, init_cell)
    print(f"\n初始带隙: {init_props['band_gap']:.3f} eV")
    
    # 设置设计目标: 1.3 eV 直接带隙 (太阳能电池优选)
    target = BandGapTarget(
        target_gap=1.3,
        gap_type='direct',
        tolerance=0.05
    )
    
    print(f"\n设计目标:")
    print(f"  目标带隙: {target.target_gap} eV")
    print(f"  带隙类型: {target.gap_type}")
    print(f"  容差: ±{target.tolerance} eV")
    
    # 构建目标函数
    objective = BandGapObjective(bg_calc, target)
    
    # 包装为优化器使用的函数
    def wrapped_objective(params, struct):
        struct.set_params(params)
        pos, nums, cell = struct.to_structure()
        return objective.loss(pos, nums, cell)
    
    # 创建优化器
    optimizer = InverseDesignOptimizer(
        objective=None,
        optimizer_type='adam',
        learning_rate=0.02,
        max_iter=300
    )
    optimizer.objective = wrapped_objective
    
    # 运行优化
    print("\n开始优化...")
    result = optimizer.optimize(structure)
    
    # 分析结果
    final_pos, final_nums, final_cell = result.to_structure()
    final_props = bg_calc.calculate(final_pos, final_nums, final_cell)
    
    print(f"\n优化完成!")
    print(f"最终带隙: {final_props['band_gap']:.3f} eV")
    print(f"目标带隙: {target.target_gap} eV")
    print(f"误差: {abs(final_props['band_gap'] - target.target_gap):.3f} eV")
    
    # 计算理论效率
    solar_optimizer = SolarCellOptimizer(bg_calc)
    efficiency = solar_optimizer.detailed_balance_efficiency(
        final_props['band_gap']
    )
    
    print(f"\n理论太阳能电池性能:")
    print(f"  开路电压 (Voc): {efficiency['voc']:.3f} V")
    print(f"  短路电流 (Jsc): {efficiency['jsc']:.2f} mA/cm²")
    print(f"  填充因子 (FF): {efficiency['fill_factor']:.2%}")
    print(f"  理论效率: {efficiency['efficiency']:.2f}%")
    
    return {
        'initial_gap': init_props['band_gap'],
        'final_gap': final_props['band_gap'],
        'target_gap': target.target_gap,
        'efficiency': efficiency,
        'history': optimizer.history
    }


def case_led_design():
    """
    案例2: LED材料设计
    
    设计红、绿、蓝三色LED材料
    """
    print("\n" + "=" * 70)
    print("案例2: RGB LED材料设计")
    print("=" * 70)
    
    dft_engine = MockDFTEngine(noise_level=0.02)
    bg_calc = BandGapCalculator(dft_engine)
    led_designer = LEDMaterialDesigner(bg_calc)
    
    results = {}
    
    for color, wavelength in [('red', 650), ('green', 530), ('blue', 470)]:
        print(f"\n--- 设计 {color.upper()} LED ---")
        
        # 创建初始结构
        structure = FractionalCoordinateStructure(
            n_atoms=3,
            atomic_numbers=jnp.array([31, 33, 33]),  # GaAs2-like
            initial_cell=jnp.eye(3) * 8.0,
            fix_cell=False
        )
        
        # 设计目标
        target_gap = led_designer.wavelength_to_gap(wavelength)
        print(f"目标波长: {wavelength} nm")
        print(f"目标带隙: {target_gap:.3f} eV")
        
        target = BandGapTarget(
            target_gap=target_gap,
            gap_type='direct',  # LED需要直接带隙
            tolerance=0.02
        )
        
        objective = BandGapObjective(bg_calc, target)
        
        def wrapped_obj(params, struct):
            struct.set_params(params)
            pos, nums, cell = struct.to_structure()
            return objective.loss(pos, nums, cell)
        
        optimizer = InverseDesignOptimizer(
            None, 'adam', 0.03, 250
        )
        optimizer.objective = wrapped_obj
        
        result = optimizer.optimize(structure)
        
        # 评估
        final_pos, final_nums, final_cell = result.to_structure()
        final_props = bg_calc.calculate(final_pos, final_nums, final_cell)
        
        final_wavelength = led_designer.gap_to_wavelength(
            final_props['direct_gap']
        )
        
        print(f"优化后带隙: {final_props['direct_gap']:.3f} eV")
        print(f"优化后波长: {final_wavelength:.1f} nm")
        print(f"波长误差: {abs(final_wavelength - wavelength):.1f} nm")
        
        results[color] = {
            'target_wavelength': wavelength,
            'final_wavelength': final_wavelength,
            'band_gap': final_props['direct_gap']
        }
    
    # 显示RGB总结
    print("\n" + "-" * 50)
    print("RGB LED设计总结:")
    print("-" * 50)
    print(f"{'颜色':<10} {'目标波长(nm)':<15} {'实际波长(nm)':<15} {'带隙(eV)':<10}")
    print("-" * 50)
    for color, data in results.items():
        print(f"{color:<10} {data['target_wavelength']:<15.0f} "
              f"{data['final_wavelength']:<15.1f} {data['band_gap']:<10.3f}")
    
    return results


def case_tco_design():
    """
    案例3: 透明导电氧化物(TCO)设计
    
    目标: 宽带隙 (>3.0 eV) 确保透明，同时保持合理电导率
    """
    print("\n" + "=" * 70)
    print("案例3: 透明导电氧化物设计")
    print("=" * 70)
    
    dft_engine = MockDFTEngine()
    bg_calc = BandGapCalculator(dft_engine)
    tco_optimizer = TransparentConductorOptimizer(bg_calc)
    
    # 创建初始结构 (氧化物)
    structure = FractionalCoordinateStructure(
        n_atoms=6,
        atomic_numbers=jnp.array([22, 22, 8, 8, 8, 8]),  # Ti2O4-like
        initial_cell=jnp.eye(3) * 9.0,
        fix_cell=False
    )
    
    print("初始结构: Ti-O体系")
    
    # 计算初始透明度
    init_pos, init_nums, init_cell = structure.to_structure()
    init_props = bg_calc.calculate(init_pos, init_nums, init_cell)
    init_transparency = tco_optimizer.transparency_at_gap(init_props['band_gap'])
    
    print(f"初始带隙: {init_props['band_gap']:.3f} eV")
    print(f"初始可见光透明度: {init_transparency:.1%}")
    
    # 优化目标: 带隙 > 3.2 eV (确保对可见光透明)
    target_gap = 3.3
    print(f"\n设计目标:")
    print(f"  目标带隙: {target_gap} eV")
    print(f"  最小透明度: 80%")
    
    target = BandGapTarget(
        target_gap=target_gap,
        gap_type='direct',
        tolerance=0.1
    )
    
    objective = BandGapObjective(bg_calc, target)
    
    def tco_objective(params, struct):
        struct.set_params(params)
        pos, nums, cell = struct.to_structure()
        
        # 基础带隙损失
        loss = objective.loss(pos, nums, cell)
        
        # 透明度约束
        props = bg_calc.calculate(pos, nums, cell)
        transparency = tco_optimizer.transparency_at_gap(props['band_gap'])
        
        if transparency < 0.8:
            loss += 5.0 * (0.8 - transparency)**2
        
        return loss
    
    optimizer = InverseDesignOptimizer(
        None, 'adam', 0.015, 350
    )
    optimizer.objective = tco_objective
    
    print("\n开始优化...")
    result = optimizer.optimize(structure)
    
    # 评估
    final_pos, final_nums, final_cell = result.to_structure()
    final_props = bg_calc.calculate(final_pos, final_nums, final_cell)
    final_transparency = tco_optimizer.transparency_at_gap(final_props['band_gap'])
    
    print(f"\n优化结果:")
    print(f"  最终带隙: {final_props['band_gap']:.3f} eV")
    print(f"  可见光透明度: {final_transparency:.1%}")
    print(f"  带隙类型: {target.gap_type}")
    
    # 透明度 vs 波长分析
    print("\n透明度分析:")
    wavelengths = [380, 450, 550, 650, 780]  # nm
    for wl in wavelengths:
        E_photon = 1240 / wl  # eV
        if E_photon < final_props['band_gap']:
            T = 0.95  # 光子能量小于带隙，透明
        else:
            T = 0.05  # 吸收
        print(f"  {wl}nm ({E_photon:.2f}eV): {T:.0%} 透明度")
    
    return {
        'band_gap': final_props['band_gap'],
        'transparency': final_transparency
    }


def plot_results(solar_results, led_results, tco_results):
    """绘制优化结果图表"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 图1: 太阳能案例的优化历史
        ax1 = axes[0]
        history = solar_results['history']
        steps = [h['step'] for h in history]
        losses = [h['loss'] for h in history]
        ax1.semilogy(steps, losses)
        ax1.set_xlabel('Optimization Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Solar Cell: Optimization Convergence')
        ax1.grid(True)
        
        # 图2: LED带隙与波长
        ax2 = axes[1]
        colors = ['red', 'green', 'blue']
        target_wavelengths = [led_results[c]['target_wavelength'] for c in colors]
        final_wavelengths = [led_results[c]['final_wavelength'] for c in colors]
        x = np.arange(len(colors))
        width = 0.35
        ax2.bar(x - width/2, target_wavelengths, width, label='Target', alpha=0.8)
        ax2.bar(x + width/2, final_wavelengths, width, label='Optimized', alpha=0.8)
        ax2.set_ylabel('Wavelength (nm)')
        ax2.set_title('LED: Target vs Optimized Wavelengths')
        ax2.set_xticks(x)
        ax2.set_xticklabels([c.upper() for c in colors])
        ax2.legend()
        ax2.grid(True, axis='y')
        
        # 图3: TCO透明度
        ax3 = axes[2]
        band_gaps = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        transparencies = [min(1.0, max(0, (Eg - 1.6) / 1.5)) for Eg in band_gaps]
        ax3.plot(band_gaps, transparencies, 'b-', linewidth=2, label='Transparency')
        ax3.axvline(tco_results['band_gap'], color='r', linestyle='--', 
                   label=f'Optimized Eg={tco_results["band_gap"]:.2f}eV')
        ax3.axhline(0.8, color='g', linestyle=':', label='80% threshold')
        ax3.set_xlabel('Band Gap (eV)')
        ax3.set_ylabel('Visible Light Transparency')
        ax3.set_title('TCO: Band Gap vs Transparency')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "bandgap_design_results.png", dpi=150)
        print(f"\n图表已保存至: {output_dir / 'bandgap_design_results.png'}")
        
    except Exception as e:
        print(f"绘图失败: {e}")


def main():
    """主函数"""
    print("\n" + "#" * 70)
    print("# 带隙逆向设计案例研究")
    print("#" * 70)
    
    # 运行案例
    solar_results = case_solar_cell()
    led_results = case_led_design()
    tco_results = case_tco_design()
    
    # 绘制结果
    print("\n" + "=" * 70)
    print("生成结果图表...")
    plot_results(solar_results, led_results, tco_results)
    
    # 总结
    print("\n" + "=" * 70)
    print("案例研究总结")
    print("=" * 70)
    print(f"\n1. 太阳能电池材料:")
    print(f"   - 带隙优化: {solar_results['initial_gap']:.2f} eV → {solar_results['final_gap']:.2f} eV")
    print(f"   - 目标带隙: {solar_results['target_gap']:.2f} eV")
    print(f"   - 理论效率: {solar_results['efficiency']['efficiency']:.2f}%")
    
    print(f"\n2. LED材料:")
    for color, data in led_results.items():
        error = abs(data['final_wavelength'] - data['target_wavelength'])
        print(f"   - {color.upper()}: {data['target_wavelength']:.0f}nm → {data['final_wavelength']:.1f}nm "
              f"(误差: {error:.1f}nm)")
    
    print(f"\n3. 透明导电氧化物:")
    print(f"   - 优化带隙: {tco_results['band_gap']:.2f} eV")
    print(f"   - 可见光透明度: {tco_results['transparency']:.1%}")
    
    print("\n" + "#" * 70)
    print("# 案例研究完成!")
    print("#" * 70)


if __name__ == "__main__":
    main()
