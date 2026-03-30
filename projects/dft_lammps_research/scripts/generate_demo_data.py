#!/usr/bin/env python3
"""
Dashboard Demo Data Generator
=============================
生成示例数据用于测试DFT+LAMMPS监控仪表盘

使用方法:
    python generate_demo_data.py

这将创建以下示例数据:
    - ML训练日志 (lcurve.out)
    - MD模拟日志 (log.lammps)
    - 高通量筛选结果 (screening_results.csv)
    - 主动学习进度数据
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


def generate_training_log(output_dir: str):
    """生成DeePMD训练日志"""
    output_path = Path(output_dir) / "lcurve.out"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 模拟训练过程
    n_steps = 10000
    batch_size = 100
    
    batches = []
    lrs = []
    losses = []
    energy_rmses = []
    force_rmses = []
    virial_rmses = []
    
    start_lr = 0.001
    stop_lr = 3.51e-8
    decay_steps = 5000
    
    for i in range(0, n_steps, batch_size):
        step = i
        # 指数衰减学习率
        lr = start_lr * (stop_lr / start_lr) ** (step / (decay_steps * 20))
        
        # 模拟损失下降
        base_loss = 100 * np.exp(-step / 2000)
        noise = np.random.normal(0, base_loss * 0.1)
        loss = max(base_loss + noise, 0.01)
        
        # RMSE也随训练下降
        energy_rmse = 0.1 * np.exp(-step / 3000) + np.random.normal(0, 0.001)
        force_rmse = 1.0 * np.exp(-step / 2500) + np.random.normal(0, 0.01)
        virial_rmse = 0.05 * np.exp(-step / 3500) + np.random.normal(0, 0.005)
        
        batches.append(step)
        lrs.append(lr)
        losses.append(loss)
        energy_rmses.append(max(energy_rmse, 0.0001))
        force_rmses.append(max(force_rmse, 0.001))
        virial_rmses.append(max(virial_rmse, 0.0001))
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write("# batch lr loss energy_rmse energy_rmse_traj force_rmse force_rmse_traj virial_rmse virial_rmse_traj\n")
        for i in range(len(batches)):
            f.write(f"{batches[i]} {lrs[i]:.6e} {losses[i]:.6e} "
                   f"{energy_rmses[i]:.6e} {energy_rmses[i]:.6e} "
                   f"{force_rmses[i]:.6e} {force_rmses[i]:.6e} "
                   f"{virial_rmses[i]:.6e} {virial_rmses[i]:.6e}\n")
    
    print(f"✓ Generated training log: {output_path}")


def generate_md_log(output_dir: str):
    """生成LAMMPS MD日志"""
    output_path = Path(output_dir) / "log.lammps"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("LAMMPS simulation log\n")
        f.write("="*50 + "\n\n")
        
        # 模拟NVT系综
        f.write("Step Temp PotEng KinEng TotEng Press Volume\n")
        
        n_steps = 50000
        target_temp = 300
        
        for step in range(0, n_steps, 100):
            # 温度逐渐平衡到目标值
            if step < 5000:
                temp = target_temp + np.random.normal(0, 50)
            else:
                temp = target_temp + np.random.normal(0, 10)
            
            # 能量波动
            pot_eng = -5000 + np.random.normal(0, 5)
            kin_eng = temp * 0.5 + np.random.normal(0, 5)
            tot_eng = pot_eng + kin_eng
            
            # 压强
            press = 1.0 + np.random.normal(0, 50)
            
            # 体积
            volume = 1000 + np.random.normal(0, 5)
            
            f.write(f"{step} {temp:.4f} {pot_eng:.4f} {kin_eng:.4f} "
                   f"{tot_eng:.4f} {press:.4f} {volume:.4f}\n")
    
    print(f"✓ Generated MD log: {output_path}")


def generate_screening_results(output_dir: str, n_materials: int = 100):
    """生成高通量筛选结果"""
    output_path = Path(output_dir) / "screening_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # 生成候选材料
    materials = []
    formulas = ["Li2O", "Li3N", "LiF", "LiCl", "Li2S", "Li3P", "Li2CO3", 
                "LiAlO2", "Li4SiO4", "Li3PO4", "Li2SO4", "LiGaO2"]
    
    for i in range(n_materials):
        formula = np.random.choice(formulas)
        
        # 各种物理属性
        ionic_conductivity = 10 ** np.random.uniform(-8, -3)  # S/cm
        formation_energy = np.random.uniform(-4, -1)  # eV/atom
        band_gap = np.random.uniform(2, 6)  # eV
        bulk_modulus = np.random.uniform(20, 150)  # GPa
        energy_above_hull = np.random.uniform(0, 0.1)  # eV/atom
        
        materials.append({
            'structure_id': f'mp-{1000+i}',
            'formula': formula,
            'ionic_conductivity': ionic_conductivity,
            'formation_energy': formation_energy,
            'band_gap': band_gap,
            'bulk_modulus': bulk_modulus,
            'energy_above_hull': energy_above_hull,
        })
    
    df = pd.DataFrame(materials)
    df.to_csv(output_path, index=False)
    print(f"✓ Generated screening results: {output_path}")


def generate_active_learning_data(output_dir: str, n_iterations: int = 5):
    """生成主动学习进度数据"""
    al_path = Path(output_dir)
    
    for iter_num in range(n_iterations):
        iter_dir = al_path / f"iter_{iter_num:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成探索统计
        total_structs = 500
        # 随着迭代进行，候选结构减少（收敛）
        candidate_ratio = max(0.05, 0.3 - iter_num * 0.05)
        n_candidates = int(total_structs * candidate_ratio)
        n_accurate = int(total_structs * 0.6)
        n_failed = total_structs - n_candidates - n_accurate
        
        stats = {
            'total': total_structs,
            'accurate': n_accurate,
            'candidate': n_candidates,
            'failed': n_failed,
            'iteration': iter_num
        }
        
        with open(iter_dir / "exploration_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 生成模型偏差数据
        deviations = []
        for _ in range(n_candidates):
            # 偏差随迭代降低
            base_devi = 0.15 * np.exp(-iter_num / 3)
            devi = max(0.02, base_devi + np.random.normal(0, 0.02))
            deviations.append({'forces': devi})
        
        with open(iter_dir / "model_deviations.json", 'w') as f:
            json.dump(deviations, f, indent=2)
    
    print(f"✓ Generated active learning data: {al_path}")


def main():
    """主函数"""
    print("="*60)
    print("DFT+LAMMPS Dashboard Demo Data Generator")
    print("="*60)
    
    # 创建示例数据目录
    base_dir = "./dashboard_demo_data"
    
    # ML训练数据
    generate_training_log(f"{base_dir}/models")
    
    # MD模拟数据
    generate_md_log(f"{base_dir}/md_results")
    
    # 高通量筛选数据
    generate_screening_results(f"{base_dir}/screening_db", n_materials=100)
    
    # 主动学习数据
    generate_active_learning_data(f"{base_dir}/active_learning_workflow", n_iterations=5)
    
    print("="*60)
    print("Demo data generation complete!")
    print(f"Data location: {base_dir}")
    print()
    print("To view the dashboard with demo data:")
    print(f"  1. Update dashboard_config.yaml paths to use '{base_dir}'")
    print("  2. Run: python monitoring_dashboard.py")
    print("  3. Open http://localhost:8050 in your browser")
    print("="*60)


if __name__ == "__main__":
    main()
