"""
Precipitation Simulation Example
================================
合金沉淀相演化模拟示例

演示如何使用相场模型模拟Al-Cu合金的沉淀相演化。
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_field.applications.precipitation import PrecipitationSimulator, PrecipConfig


def run_precipitation_simulation():
    """运行沉淀相演化模拟"""
    
    print("=" * 60)
    print("Al-Cu Alloy Precipitation Simulation")
    print("=" * 60)
    
    # 配置Al-Cu合金
    config = PrecipConfig(
        # 网格设置
        nx=128,
        ny=128,
        dx=2.0,  # nm
        dt=0.0005,
        total_steps=10000,
        save_interval=200,
        
        # 合金成分
        n_components=2,
        element_names=['Al', 'Cu'],
        nominal_composition={'Al': 0.96, 'Cu': 0.04},
        precipitate_composition={'Al': 0.67, 'Cu': 0.33},  # Al2Cu (θ')
        
        # 平衡成分
        equilibrium_composition={'matrix': 0.02, 'precipitate': 0.33},
        
        # 时效参数
        temperature=473.15,  # 200°C
        aging_time=10.0,  # hours
        
        # 动力学参数
        diffusivity_prefactor=1e-5,  # m²/s
        activation_energy=1.0,  # eV
        
        # 形核参数
        nucleation_density=1e21,  # m^-3
        critical_radius=1.0,  # nm
        
        # 弹性参数
        lattice_mismatch=0.02,
        
        # 自由能参数
        free_energy_type='regular_solution',
        free_energy_params={'Omega': 2.0}
    )
    
    # 创建模拟器
    simulator = PrecipitationSimulator(config)
    
    # 初始化
    print("\nInitializing concentration field...")
    simulator.initialize_fields(uniform_composition=True, seed=42)
    
    # 运行模拟
    print("\nRunning precipitation simulation...")
    print(f"{'Step':<10}{'Time':<12}{'N_precip':<12}{'<R> (nm)':<12}{'f_v':<10}")
    print("-" * 60)
    
    precip_history = []
    
    def callback(step, time, info):
        if step % 1000 == 0:
            stats = simulator.get_precipitate_statistics()
            print(f"{step:<10}{time:<12.3f}{stats['count']:<12}{stats['average_radius']:<12.2f}"
                  f"{stats['total_volume_fraction']:<10.4f}")
            precip_history.append(stats)
    
    result = simulator.run(n_steps=10000, callback=callback)
    
    # 最终结果
    print("\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    
    stats = simulator.get_precipitate_statistics()
    print(f"\nPrecipitate Statistics:")
    print(f"  Number of precipitates: {stats['count']}")
    print(f"  Number density: {stats['number_density']:.2e} m^-3")
    print(f"  Average radius: {stats['average_radius']:.2f} nm")
    print(f"  Radius std: {stats['radius_std']:.2f} nm")
    print(f"  Volume fraction: {stats['total_volume_fraction']:.4f}")
    print(f"  Nucleation events: {stats['nucleation_count']}")
    
    # 尺寸分布
    bins, counts = simulator.get_size_distribution(n_bins=15)
    print(f"\nSize Distribution:")
    for i in range(len(counts)):
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f} nm: {counts[i]} precipitates")
    
    # 估算硬度
    hardness = simulator.estimate_hardness_increase()
    print(f"\nEstimated Hardness Increase: {hardness:.1f}%")
    
    # 可视化
    plot_results(simulator, precip_history)
    
    return simulator, result


def plot_results(simulator, precip_history):
    """绘制结果"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 浓度场
        ax = axes[0, 0]
        im = ax.imshow(simulator.c.T, origin='lower', cmap='hot', 
                      vmin=0, vmax=0.4, aspect='auto')
        ax.set_title('Cu Concentration Field')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        plt.colorbar(im, ax=ax, label='Cu fraction')
        
        # 2. 沉淀相标记
        ax = axes[0, 1]
        grain_ids = simulator.grain_ids
        im = ax.imshow(grain_ids.T, origin='lower', cmap='tab20', aspect='auto')
        ax.set_title('Precipitate Identification')
        plt.colorbar(im, ax=ax)
        
        # 3. 序参量
        ax = axes[0, 2]
        eta = simulator.eta_precipitate
        im = ax.imshow(eta.T, origin='lower', cmap='viridis', aspect='auto')
        ax.set_title('Precipitate Order Parameter')
        plt.colorbar(im, ax=ax)
        
        # 4. 沉淀相数量演化
        if precip_history:
            ax = axes[1, 0]
            steps = [i * 1000 for i in range(len(precip_history))]
            counts = [p['count'] for p in precip_history]
            ax.plot(steps, counts, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Step')
            ax.set_ylabel('Number of Precipitates')
            ax.set_title('Precipitate Nucleation')
            ax.grid(True)
        
        # 5. 平均尺寸演化
        if precip_history:
            ax = axes[1, 1]
            radii = [p['average_radius'] for p in precip_history]
            ax.plot(steps, radii, 's-', color='red', linewidth=2, markersize=6)
            ax.set_xlabel('Step')
            ax.set_ylabel('Average Radius (nm)')
            ax.set_title('Precipitate Growth')
            ax.grid(True)
        
        # 6. 尺寸分布
        ax = axes[1, 2]
        bins, counts = simulator.get_size_distribution(n_bins=15)
        ax.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7, color='green')
        ax.set_xlabel('Radius (nm)')
        ax.set_ylabel('Count')
        ax.set_title('Size Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('precipitation_results.png', dpi=150)
        print("\nVisualization saved to: precipitation_results.png")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization")


if __name__ == "__main__":
    simulator, result = run_precipitation_simulation()
