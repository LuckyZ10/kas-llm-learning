"""
SEI Growth Example
==================
锂离子电池SEI生长模拟示例

演示如何使用电化学相场模型模拟SEI层的生长过程。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入相场模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_field.applications.sei_growth import SEIGrowthSimulator, SEIConfig


def run_sei_simulation():
    """运行SEI生长模拟"""
    
    print("=" * 60)
    print("SEI Growth Simulation Example")
    print("=" * 60)
    
    # 配置参数
    config = SEIConfig(
        # 网格设置
        nx=128,
        ny=64,
        dx=1.0,  # nm
        dt=0.001,
        total_steps=5000,
        save_interval=100,
        
        # SEI组分
        n_components=3,
        component_names=['organic', 'Li2CO3', 'LiF'],
        component_fractions={'organic': 0.3, 'Li2CO3': 0.5, 'LiF': 0.2},
        
        # 电化学参数
        temperature=298.15,  # K
        applied_voltage=0.1,  # V vs Li/Li+
        exchange_current_density=10.0,  # A/m²
        
        # 机械参数
        sei_modulus=10.0,  # GPa
        include_mechanical_failure=True,
        
        # 初始条件
        initial_sei_thickness=5.0,  # nm
    )
    
    # 创建模拟器
    simulator = SEIGrowthSimulator(config)
    
    # 初始化
    print("\nInitializing fields...")
    simulator.initialize_fields(seed=42)
    
    # 记录结果
    sei_thickness_history = []
    time_history = []
    voltage_history = []
    
    # 运行模拟
    print("\nRunning simulation...")
    print(f"{'Step':<10}{'Time':<12}{'SEI Thickness':<15}{'Voltage':<12}")
    print("-" * 60)
    
    def callback(step, time, info):
        if step % 500 == 0:
            thickness = info.get('sei_thickness', 0)
            voltage = info.get('phi_mean', 0)
            print(f"{step:<10}{time:<12.3f}{thickness:<15.2f}{voltage:<12.4f}")
            
            sei_thickness_history.append(thickness)
            time_history.append(time)
            voltage_history.append(voltage)
    
    # 运行
    result = simulator.run(n_steps=5000, callback=callback)
    
    # 最终统计
    print("\n" + "=" * 60)
    print("Simulation Results:")
    print("=" * 60)
    
    sei_props = simulator.get_sei_properties()
    print(f"\nFinal SEI Properties:")
    print(f"  Thickness: {sei_props['thickness']:.2f} nm")
    print(f"  Growth Rate: {sei_props['growth_rate']:.4f} nm/s")
    print(f"  Porosity: {sei_props['porosity']:.4f}")
    print(f"  Impedance: {simulator.get_impedance_contribution():.2e} Ω·m²")
    
    print(f"\nComponent Volume Fractions:")
    for comp, frac in sei_props['volume_fractions'].items():
        print(f"  {comp}: {frac:.3f}")
    
    # 可视化
    plot_results(simulator, time_history, sei_thickness_history)
    
    return simulator, result


def plot_results(simulator, time_history, sei_thickness_history):
    """绘制结果"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. SEI厚度演化
        ax = axes[0, 0]
        ax.plot(time_history, sei_thickness_history, 'b-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('SEI Thickness (nm)')
        ax.set_title('SEI Growth Kinetics')
        ax.grid(True)
        
        # 2. 相场分布
        ax = axes[0, 1]
        total_sei = sum(simulator.phi[name] for name in simulator.config.component_names)
        im = ax.imshow(total_sei.T, origin='lower', cmap='viridis', aspect='auto')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_title('SEI Distribution')
        plt.colorbar(im, ax=ax, label='SEI fraction')
        
        # 3. 各组分分布
        ax = axes[0, 2]
        for i, name in enumerate(simulator.config.component_names):
            ax.imshow(simulator.phi[name].T, origin='lower', 
                     cmap='Blues', alpha=0.3*(i+1), aspect='auto')
        ax.set_title('Component Distribution')
        
        # 4. 电势分布
        ax = axes[1, 0]
        im = ax.imshow(simulator.phi.T, origin='lower', cmap='RdBu_r', aspect='auto')
        ax.set_title('Potential Distribution')
        plt.colorbar(im, ax=ax, label='Potential (V)')
        
        # 5. 应力分布 (如果启用)
        if simulator.config.include_mechanical_failure and simulator.fields.get('stress') is not None:
            ax = axes[1, 1]
            stress = simulator.fields['stress']
            im = ax.imshow(stress.T, origin='lower', cmap='coolwarm', aspect='auto')
            ax.set_title('Stress Distribution')
            plt.colorbar(im, ax=ax, label='Stress (GPa)')
        
        # 6. 损伤分布
        if simulator.damage is not None:
            ax = axes[1, 2]
            im = ax.imshow(simulator.damage.T, origin='lower', cmap='Reds', aspect='auto')
            ax.set_title('Damage Distribution')
            plt.colorbar(im, ax=ax, label='Damage')
        
        plt.tight_layout()
        plt.savefig('sei_simulation_results.png', dpi=150)
        print("\nVisualization saved to: sei_simulation_results.png")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization")


if __name__ == "__main__":
    simulator, result = run_sei_simulation()
