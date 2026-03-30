#!/usr/bin/env python3
"""
DFT+LAMMPS Framework - 完整演示工作流
=====================================

一键运行的完整演示脚本，使用模拟数据（无需安装VASP/LAMMPS/DeepMD）
展示完整的DFT→ML势→MD→分析工作流

运行方式:
    python demo_workflow.py

预计运行时间: 2-5分钟
输出: ./demo_output/ 目录包含完整结果和可视化

功能展示:
    ✓ 结构获取与预处理
    ✓ DFT计算模拟（使用模拟数据）
    ✓ ML势训练过程（模拟训练曲线）
    ✓ MD模拟轨迹生成（模拟数据）
    ✓ 扩散系数和电导率分析
    ✓ 结果可视化图表
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，将跳过图表生成")
    print("  安装: pip install matplotlib")

# 尝试导入ASE
try:
    from ase import Atoms
    from ase.io import write
    from ase.build import bulk
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("警告: ASE未安装")
    print("  安装: pip install ase")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# 模拟数据生成器
# =============================================================================

class MockDataGenerator:
    """生成逼真的模拟数据用于演示"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.materials_db = {
            "Li3PS4": {
                "space_group": "P-6m2",
                "band_gap": 3.2,
                "formation_energy": -1.85,
                "lattice": [[7.70, 0, 0], [0, 7.70, 0], [0, 0, 6.16]],
                "atoms": [("Li", 0.0, 0.0, 0.0), ("Li", 0.5, 0.0, 0.0), ("Li", 0.0, 0.5, 0.0),
                         ("P", 0.333, 0.667, 0.25), ("S", 0.333, 0.667, 0.75),
                         ("S", 0.667, 0.333, 0.25), ("S", 0.0, 0.0, 0.5)]
            },
            "Li7P3S11": {
                "space_group": "P42/mnm",
                "band_gap": 2.8,
                "formation_energy": -1.72,
                "lattice": [[12.6, 0, 0], [0, 12.6, 0], [0, 0, 12.4]],
                "atoms": []
            },
            "Li6PS5Cl": {
                "space_group": "F-43m",
                "band_gap": 3.5,
                "formation_energy": -1.95,
                "lattice": [[9.8, 0, 0], [0, 9.8, 0], [0, 0, 9.8]],
                "atoms": []
            }
        }
    
    def get_structure(self, formula: str) -> Dict:
        """获取模拟结构数据"""
        if formula in self.materials_db:
            data = self.materials_db[formula].copy()
            data["formula"] = formula
            return data
        else:
            # 生成随机结构
            return {
                "formula": formula,
                "space_group": "P1",
                "band_gap": np.random.uniform(1.5, 4.5),
                "formation_energy": np.random.uniform(-3.0, -0.5),
                "lattice": np.eye(3) * np.random.uniform(5, 10),
                "atoms": [("Li", 0, 0, 0), ("P", 0.5, 0.5, 0.5)]
            }
    
    def generate_dft_results(self, structure: Dict) -> Dict:
        """生成模拟DFT计算结果"""
        n_atoms = len(structure.get("atoms", [])) or 10
        
        # 模拟能量和力
        energy = structure["formation_energy"] * n_atoms
        
        results = {
            "energy": energy,
            "energy_per_atom": structure["formation_energy"],
            "forces": np.random.normal(0, 0.01, (n_atoms, 3)).tolist(),
            "stress": np.diag([0.1, 0.05, -0.02]).tolist(),
            "cell": structure["lattice"],
            "converged": True,
            "n_steps": np.random.randint(20, 80),
            "band_gap": structure.get("band_gap", 2.5),
            "magnetic_moment": 0.0
        }
        
        return results
    
    def generate_training_log(self, n_steps: int = 1000) -> pd.DataFrame:
        """生成模拟ML训练日志"""
        steps = np.arange(0, n_steps, 10)
        
        # 指数衰减的学习率
        start_lr = 0.001
        stop_lr = 1e-6
        lrs = start_lr * (stop_lr / start_lr) ** (steps / n_steps)
        
        # 下降的损失函数（带噪声）
        base_loss = 100 * np.exp(-steps / 300)
        noise = np.random.normal(0, base_loss * 0.05)
        losses = base_loss + noise
        
        # 能量RMSE
        energy_rmse = 0.1 * np.exp(-steps / 250) + 0.001
        energy_rmse += np.random.normal(0, 0.0005, len(steps))
        
        # 力RMSE
        force_rmse = 0.5 * np.exp(-steps / 200) + 0.01
        force_rmse += np.random.normal(0, 0.005, len(steps))
        
        df = pd.DataFrame({
            'step': steps,
            'lr': lrs,
            'loss': losses,
            'energy_rmse': energy_rmse,
            'force_rmse': force_rmse,
            'virial_rmse': energy_rmse * 0.5
        })
        
        return df
    
    def generate_md_trajectory(self, n_frames: int = 1000, 
                                temperature: float = 300) -> pd.DataFrame:
        """生成模拟MD轨迹数据"""
        dt = 1.0  # fs
        times = np.arange(n_frames) * dt
        
        # 温度波动（正弦+噪声）
        temp_base = temperature
        temp_fluctuation = 10 * np.sin(2 * np.pi * times / 500) + np.random.normal(0, 5, n_frames)
        temps = temp_base + temp_fluctuation
        
        # 能量
        pe = -5000 + 5 * np.sin(2 * np.pi * times / 800) + np.random.normal(0, 2, n_frames)
        ke = temps * 0.5 + np.random.normal(0, 1, n_frames)
        
        # 压强
        pressures = 1.0 + 100 * np.sin(2 * np.pi * times / 1000) + np.random.normal(0, 50, n_frames)
        
        # 体积
        volumes = 1000 + np.random.normal(0, 2, n_frames)
        
        # 扩散相关数据（MSD）
        msd = 0.5 * times * 1e-3  # 线性增长
        msd += np.random.normal(0, 0.1, n_frames)
        
        df = pd.DataFrame({
            'step': np.arange(n_frames),
            'time_fs': times,
            'temperature': temps,
            'potential_energy': pe,
            'kinetic_energy': ke,
            'total_energy': pe + ke,
            'pressure': pressures,
            'volume': volumes,
            'msd': msd
        })
        
        return df
    
    def calculate_diffusion(self, trajectory: pd.DataFrame, 
                           atom_type: str = "Li") -> float:
        """从轨迹计算扩散系数"""
        times = trajectory['time_fs'].values * 1e-15  # 转换为秒
        msd = trajectory['msd'].values * 1e-20  # 转换为m²
        
        # 线性拟合
        slope = np.polyfit(times[len(times)//4:3*len(times)//4], 
                          msd[len(msd)//4:3*len(msd)//4], 1)[0]
        
        # D = slope / (6 * dimension)
        D = slope / 6  # m²/s
        D_cm2_s = D * 1e4  # 转换为cm²/s
        
        return max(D_cm2_s, 1e-12)  # 最小值限制
    
    def calculate_conductivity(self, D: float, structure: Dict,
                               temperature: float) -> float:
        """使用Nernst-Einstein方程计算电导率"""
        # 简化计算
        n = 1e22  # 离子数密度 cm^-3
        q = 1.6e-19  # 电荷 C
        kB = 1.38e-23  # J/K
        
        sigma = n * q**2 * D / (kB * temperature)  # S/cm
        return sigma
    
    def fit_arrhenius(self, temps: List[float], 
                     diffs: List[float]) -> Tuple[float, float]:
        """拟合Arrhenius方程"""
        temps_arr = np.array(temps)
        diffs_arr = np.array(diffs)
        
        # ln(D) vs 1/T
        inv_T = 1000 / temps_arr  # 1000/T (K^-1)
        ln_D = np.log(diffs_arr)
        
        # 线性拟合
        slope, intercept = np.polyfit(inv_T, ln_D, 1)
        
        # Ea = -slope * kB (转换为eV)
        kB = 8.617e-5  # eV/K
        Ea = -slope * kB * 1000  # eV
        
        # D0 = exp(intercept)
        D0 = np.exp(intercept)
        
        return Ea, D0


# =============================================================================
# 可视化模块
# =============================================================================

class Visualizer:
    """结果可视化器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_training_curves(self, training_df: pd.DataFrame, 
                            save_name: str = "training_curves.png"):
        """绘制训练曲线"""
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss曲线
        ax = axes[0, 0]
        ax.semilogy(training_df['step'], training_df['loss'], 
                   color=self.colors[0], linewidth=1.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        
        # 学习率
        ax = axes[0, 1]
        ax.semilogy(training_df['step'], training_df['lr'],
                   color=self.colors[1], linewidth=1.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        
        # 能量RMSE
        ax = axes[1, 0]
        ax.semilogy(training_df['step'], training_df['energy_rmse'],
                   color=self.colors[2], linewidth=1.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Energy RMSE (eV)')
        ax.set_title('Energy RMSE vs Step')
        ax.axhline(y=0.01, color='r', linestyle='--', label='Target: 0.01 eV')
        ax.legend()
        
        # 力RMSE
        ax = axes[1, 1]
        ax.semilogy(training_df['step'], training_df['force_rmse'],
                   color=self.colors[3], linewidth=1.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Force RMSE (eV/Å)')
        ax.set_title('Force RMSE vs Step')
        ax.axhline(y=0.1, color='r', linestyle='--', label='Target: 0.1 eV/Å')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  保存训练曲线: {save_path}")
    
    def plot_md_trajectory(self, trajectory: pd.DataFrame, temperature: float,
                          save_name: str = None):
        """绘制MD轨迹分析"""
        if not HAS_MATPLOTLIB:
            return
        
        if save_name is None:
            save_name = f"md_analysis_T{int(temperature)}.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 温度
        ax = axes[0, 0]
        ax.plot(trajectory['time_fs'] / 1000, trajectory['temperature'],
               color=self.colors[0], linewidth=1)
        ax.axhline(y=temperature, color='r', linestyle='--', label=f'Target: {temperature}K')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title(f'Temperature Profile @ {temperature}K')
        ax.legend()
        
        # 能量
        ax = axes[0, 1]
        ax.plot(trajectory['time_fs'] / 1000, trajectory['total_energy'],
               color=self.colors[1], linewidth=1)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Total Energy')
        
        # 压强
        ax = axes[1, 0]
        ax.plot(trajectory['time_fs'] / 1000, trajectory['pressure'],
               color=self.colors[2], linewidth=1)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Pressure (atm)')
        ax.set_title('Pressure Profile')
        
        # MSD
        ax = axes[1, 1]
        ax.plot(trajectory['time_fs'] / 1000, trajectory['msd'],
               color=self.colors[3], linewidth=1.5)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('MSD (Å²)')
        ax.set_title('Mean Square Displacement')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  保存MD轨迹: {save_path}")
    
    def plot_diffusion_analysis(self, temps: List[float], diffs: List[float],
                                Ea: float, D0: float,
                                save_name: str = "diffusion_analysis.png"):
        """绘制扩散分析结果"""
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 扩散系数vs温度
        ax = axes[0]
        bars = ax.bar(range(len(temps)), diffs, color=self.colors[:len(temps)])
        ax.set_xticks(range(len(temps)))
        ax.set_xticklabels([f'{T}K' for T in temps])
        ax.set_ylabel('Diffusion Coefficient (cm²/s)')
        ax.set_title('Diffusion Coefficient vs Temperature')
        ax.set_yscale('log')
        
        # 添加数值标签
        for bar, diff in zip(bars, diffs):
            height = bar.get_height()
            ax.annotate(f'{diff:.2e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        # Arrhenius图
        ax = axes[1]
        inv_T = 1000 / np.array(temps)
        ln_D = np.log(diffs)
        
        ax.scatter(inv_T, ln_D, s=100, color=self.colors[0], zorder=5)
        
        # 拟合线
        fit_T = np.linspace(inv_T.min() * 0.9, inv_T.max() * 1.1, 100)
        fit_ln_D = np.log(D0) - Ea / (8.617e-5) / (1000 / fit_T)
        ax.plot(fit_T, fit_ln_D, 'r--', linewidth=2, 
               label=f'Ea = {Ea:.3f} eV')
        
        ax.set_xlabel('1000/T (K⁻¹)')
        ax.set_ylabel('ln(D)')
        ax.set_title('Arrhenius Plot')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  保存扩散分析: {save_path}")
    
    def plot_workflow_summary(self, results: Dict, save_name: str = "workflow_summary.png"):
        """绘制工作流总览"""
        if not HAS_MATPLOTLIB:
            return
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 标题
        fig.suptitle(f"DFT+LAMMPS Workflow Results: {results['formula']}", 
                    fontsize=16, fontweight='bold')
        
        # DFT结果摘要
        ax1 = fig.add_subplot(gs[0, 0])
        dft_data = results.get('dft', {})
        metrics = ['Energy\n(eV/atom)', 'Band Gap\n(eV)', 'Steps\n(DFT)']
        values = [
            dft_data.get('energy_per_atom', 0),
            dft_data.get('band_gap', 0),
            dft_data.get('n_steps', 0) / 10  # 缩放
        ]
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        bars = ax1.bar(metrics, values, color=colors)
        ax1.set_title('DFT Results', fontweight='bold')
        ax1.set_ylabel('Value')
        
        # 训练收敛
        ax2 = fig.add_subplot(gs[0, 1])
        if 'training_df' in results:
            final_rmse = results['training_df']['force_rmse'].iloc[-1]
            target_rmse = 0.1
            ax2.bar(['Final RMSE', 'Target'], [final_rmse, target_rmse],
                   color=['#e74c3c' if final_rmse > target_rmse else '#2ecc71', '#95a5a6'])
            ax2.set_title('Training Convergence', fontweight='bold')
            ax2.set_ylabel('Force RMSE (eV/Å)')
        
        # 扩散系数
        ax3 = fig.add_subplot(gs[0, 2])
        if 'analysis' in results:
            diff_data = results['analysis'].get('diffusion_coefficients', {})
            if diff_data:
                temps = list(diff_data.keys())
                diffs = list(diff_data.values())
                ax3.barh(range(len(temps)), diffs, color=self.colors[:len(temps)])
                ax3.set_yticks(range(len(temps)))
                ax3.set_yticklabels([f'{T}K' for T in temps])
                ax3.set_xlabel('D (cm²/s)')
                ax3.set_title('Diffusion Coefficients', fontweight='bold')
                ax3.set_xscale('log')
        
        # 电导率
        ax4 = fig.add_subplot(gs[1, :2])
        if 'analysis' in results:
            cond_data = results['analysis'].get('conductivities', {})
            if cond_data:
                temps = list(cond_data.keys())
                conds = list(cond_data.values())
                ax4.plot(temps, conds, 'o-', linewidth=2, markersize=10, color='#3498db')
                ax4.set_xlabel('Temperature (K)')
                ax4.set_ylabel('Ionic Conductivity (S/cm)')
                ax4.set_title('Ionic Conductivity vs Temperature', fontweight='bold')
                ax4.set_yscale('log')
        
        # 活化能
        ax5 = fig.add_subplot(gs[1, 2])
        Ea = results.get('analysis', {}).get('activation_energy', 0)
        reference_Ea = 0.3  # 典型固体电解质
        ax5.bar(['Calculated', 'Reference\n(typical)'], [Ea, reference_Ea],
               color=['#e74c3c' if Ea > reference_Ea else '#2ecc71', '#95a5a6'])
        ax5.set_title('Activation Energy', fontweight='bold')
        ax5.set_ylabel('Ea (eV)')
        ax5.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Good conductor')
        
        # 时间线
        ax6 = fig.add_subplot(gs[2, :])
        stages = ['Structure\nFetch', 'DFT\nCalculation', 'ML\nTraining', 
                 'MD\nSimulation', 'Analysis']
        durations = [0.5, 2.0, 3.0, 2.5, 1.0]  # 模拟时间（分钟）
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c']
        
        cumulative = 0
        for i, (stage, duration, color) in enumerate(zip(stages, durations, colors)):
            ax6.barh(0, duration, left=cumulative, color=color, label=stage, height=0.5)
            ax6.text(cumulative + duration/2, 0, f'{duration}min', 
                    ha='center', va='center', fontweight='bold')
            cumulative += duration
        
        ax6.set_xlim(0, sum(durations) * 1.1)
        ax6.set_ylim(-0.5, 0.5)
        ax6.set_xlabel('Time (minutes)')
        ax6.set_title('Workflow Timeline (Estimated)', fontweight='bold')
        ax6.legend(loc='upper right', ncol=len(stages))
        ax6.set_yticks([])
        ax6.spines['left'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  保存工作流总览: {save_path}")


# =============================================================================
# 演示工作流主类
# =============================================================================

class DemoWorkflow:
    """
    完整演示工作流
    
    展示从结构获取到分析的所有步骤，使用模拟数据
    """
    
    def __init__(self, working_dir: str = "./demo_output"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.mock_gen = MockDataGenerator(seed=42)
        self.visualizer = Visualizer(str(self.working_dir / "figures"))
        
        self.results = {}
        self.timing = {}
    
    def print_header(self, text: str):
        """打印章节标题"""
        print(f"\n{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}")
    
    def print_progress(self, step: int, total: int, message: str):
        """打印进度"""
        progress = (step / total) * 100
        bar_len = 30
        filled = int(bar_len * step / total)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"  [{bar}] {progress:.0f}% {message}")
    
    def stage_1_fetch_structure(self, formula: str = "Li3PS4") -> Dict:
        """阶段1: 获取结构"""
        self.print_header("Stage 1: Structure Fetching")
        start_time = time.time()
        
        print(f"  查询材料: {formula}")
        structure = self.mock_gen.get_structure(formula)
        
        print(f"  ✓ 找到结构:")
        print(f"    - 化学式: {structure['formula']}")
        print(f"    - 空间群: {structure['space_group']}")
        print(f"    - 带隙: {structure['band_gap']:.2f} eV")
        print(f"    - 形成能: {structure['formation_energy']:.3f} eV/atom")
        
        # 保存结构信息
        structure_file = self.working_dir / "structure.json"
        with open(structure_file, 'w') as f:
            json.dump(structure, f, indent=2)
        
        elapsed = time.time() - start_time
        self.timing['structure_fetch'] = elapsed
        print(f"  ⏱ 耗时: {elapsed:.2f}s")
        
        return structure
    
    def stage_2_dft_calculation(self, structure: Dict) -> Dict:
        """阶段2: DFT计算（模拟）"""
        self.print_header("Stage 2: DFT Calculation (Simulated)")
        start_time = time.time()
        
        print("  设置DFT计算参数:")
        print("    - 软件: VASP (模拟)")
        print("    - 泛函: PBE")
        print("    - 截断能: 520 eV")
        print("    - K点密度: 0.25 Å⁻¹")
        
        print("\n  运行结构优化...")
        for i in range(5):
            time.sleep(0.2)  # 模拟计算时间
            self.print_progress(i + 1, 5, f"SCF迭代 {i+1}/5")
        
        results = self.mock_gen.generate_dft_results(structure)
        
        print(f"\n  ✓ DFT计算完成:")
        print(f"    - 总能量: {results['energy']:.4f} eV")
        print(f"    - 单原子能量: {results['energy_per_atom']:.4f} eV/atom")
        print(f"    - 带隙: {results['band_gap']:.2f} eV")
        print(f"    - 收敛步数: {results['n_steps']}")
        
        # 保存结果
        dft_file = self.working_dir / "dft_results.json"
        with open(dft_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        elapsed = time.time() - start_time
        self.timing['dft_calculation'] = elapsed
        print(f"  ⏱ 耗时: {elapsed:.2f}s")
        
        return results
    
    def stage_3_ml_training(self) -> pd.DataFrame:
        """阶段3: ML势训练（模拟）"""
        self.print_header("Stage 3: ML Potential Training (Simulated)")
        start_time = time.time()
        
        print("  准备训练数据...")
        print("    - 读取DFT计算结果")
        print("    - 生成训练/验证数据集")
        
        print("\n  训练DeepMD模型:")
        print("    - 描述符: se_e2_a")
        print("    - 截断半径: 6.0 Å")
        print("    - 神经网络: [25, 50, 100] -> [240, 240, 240]")
        
        # 生成训练日志
        training_df = self.mock_gen.generate_training_log(n_steps=1000)
        
        for i in range(0, len(training_df), 200):
            row = training_df.iloc[i]
            self.print_progress(i + 1, len(training_df), 
                              f"Step {int(row['step'])}, Loss={row['loss']:.4f}")
            time.sleep(0.05)
        
        final_row = training_df.iloc[-1]
        print(f"\n  ✓ 训练完成:")
        print(f"    - 最终Loss: {final_row['loss']:.6f}")
        print(f"    - 能量RMSE: {final_row['energy_rmse']:.6f} eV")
        print(f"    - 力RMSE: {final_row['force_rmse']:.6f} eV/Å")
        
        # 绘制训练曲线
        self.visualizer.plot_training_curves(training_df)
        
        # 保存训练日志
        training_file = self.working_dir / "training_log.csv"
        training_df.to_csv(training_file, index=False)
        
        elapsed = time.time() - start_time
        self.timing['ml_training'] = elapsed
        print(f"  ⏱ 耗时: {elapsed:.2f}s")
        
        return training_df
    
    def stage_4_md_simulation(self, structure: Dict,
                              temperatures: List[float] = None) -> Dict[float, pd.DataFrame]:
        """阶段4: MD模拟（模拟）"""
        self.print_header("Stage 4: MD Simulation (Simulated)")
        start_time = time.time()
        
        if temperatures is None:
            temperatures = [300, 500, 700, 900]
        
        print(f"  设置MD模拟参数:")
        print(f"    - 系综: NVT")
        print(f"    - 时间步长: 1.0 fs")
        print(f"    - 温度点: {temperatures}")
        print(f"    - 模拟时长: 100 ps")
        
        trajectories = {}
        
        for i, T in enumerate(temperatures):
            print(f"\n  在 {T}K 下运行MD模拟...")
            
            # 生成轨迹
            traj = self.mock_gen.generate_md_trajectory(n_frames=1000, temperature=T)
            trajectories[T] = traj
            
            # 绘制轨迹
            self.visualizer.plot_md_trajectory(traj, T)
            
            print(f"    ✓ 完成: {len(traj)} 帧")
            
            # 保存轨迹
            traj_file = self.working_dir / f"md_trajectory_T{T}.csv"
            traj.to_csv(traj_file, index=False)
        
        elapsed = time.time() - start_time
        self.timing['md_simulation'] = elapsed
        print(f"\n  ⏱ 耗时: {elapsed:.2f}s")
        
        return trajectories
    
    def stage_5_analysis(self, structure: Dict,
                        trajectories: Dict[float, pd.DataFrame]) -> Dict:
        """阶段5: 分析计算"""
        self.print_header("Stage 5: Analysis")
        start_time = time.time()
        
        results = {
            'diffusion_coefficients': {},
            'conductivities': {},
            'temperatures': [],
            'diffusion_values': []
        }
        
        print("  计算扩散系数...")
        for T, traj in trajectories.items():
            D = self.mock_gen.calculate_diffusion(traj, "Li")
            results['diffusion_coefficients'][T] = D
            results['temperatures'].append(T)
            results['diffusion_values'].append(D)
            print(f"    {T}K: D = {D:.2e} cm²/s")
        
        print("\n  计算离子电导率...")
        for T in results['temperatures']:
            D = results['diffusion_coefficients'][T]
            sigma = self.mock_gen.calculate_conductivity(D, structure, T)
            results['conductivities'][T] = sigma
            print(f"    {T}K: σ = {sigma:.2e} S/cm")
        
        print("\n  拟合Arrhenius方程...")
        Ea, D0 = self.mock_gen.fit_arrhenius(results['temperatures'],
                                             results['diffusion_values'])
        results['activation_energy'] = Ea
        results['pre_exponential'] = D0
        
        print(f"    ✓ 活化能 Ea = {Ea:.3f} eV")
        print(f"    ✓ 前置因子 D0 = {D0:.2e} cm²/s")
        
        # 绘制扩散分析
        self.visualizer.plot_diffusion_analysis(
            results['temperatures'],
            results['diffusion_values'],
            Ea, D0
        )
        
        # 保存分析结果
        analysis_file = self.working_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            # 转换numpy类型
            serializable_results = {
                'diffusion_coefficients': {str(k): v for k, v in results['diffusion_coefficients'].items()},
                'conductivities': {str(k): v for k, v in results['conductivities'].items()},
                'activation_energy': float(Ea),
                'pre_exponential': float(D0)
            }
            json.dump(serializable_results, f, indent=2)
        
        elapsed = time.time() - start_time
        self.timing['analysis'] = elapsed
        print(f"  ⏱ 耗时: {elapsed:.2f}s")
        
        return results
    
    def run(self, formula: str = "Li3PS4") -> Dict:
        """运行完整演示工作流"""
        print("\n" + "="*70)
        print("  DFT + LAMMPS 多尺度材料计算框架 - 完整演示")
        print("  DFT + LAMMPS Multi-Scale Materials Simulation Framework")
        print("="*70)
        print(f"\n  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  工作目录: {self.working_dir.absolute()}")
        print(f"  目标材料: {formula}")
        print("\n  ⚠ 注意: 本演示使用模拟数据，无需安装VASP/LAMMPS/DeepMD")
        
        total_start = time.time()
        
        try:
            # 阶段1: 获取结构
            structure = self.stage_1_fetch_structure(formula)
            self.results['formula'] = formula
            self.results['structure'] = structure
            
            # 阶段2: DFT计算
            dft_results = self.stage_2_dft_calculation(structure)
            self.results['dft'] = dft_results
            
            # 阶段3: ML训练
            training_df = self.stage_3_ml_training()
            self.results['training_df'] = training_df
            
            # 阶段4: MD模拟
            trajectories = self.stage_4_md_simulation(structure)
            self.results['trajectories'] = trajectories
            
            # 阶段5: 分析
            analysis_results = self.stage_5_analysis(structure, trajectories)
            self.results['analysis'] = analysis_results
            
            # 生成总览图
            self.visualizer.plot_workflow_summary(self.results)
            
            # 生成最终报告
            self.generate_report()
            
            total_elapsed = time.time() - total_start
            
            # 打印总结
            self.print_summary(total_elapsed)
            
            return self.results
            
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_report(self):
        """生成最终报告"""
        report = {
            'workflow_name': 'DFT+LAMMPS Demo Workflow',
            'timestamp': datetime.now().isoformat(),
            'formula': self.results.get('formula'),
            'structure': {
                'space_group': self.results.get('structure', {}).get('space_group'),
                'band_gap': self.results.get('structure', {}).get('band_gap'),
                'formation_energy': self.results.get('structure', {}).get('formation_energy'),
            },
            'dft_results': self.results.get('dft'),
            'ml_training': {
                'final_loss': float(self.results['training_df']['loss'].iloc[-1]) if 'training_df' in self.results else None,
                'final_force_rmse': float(self.results['training_df']['force_rmse'].iloc[-1]) if 'training_df' in self.results else None,
            },
            'analysis': self.results.get('analysis'),
            'timing': self.timing,
            'total_time': sum(self.timing.values())
        }
        
        # 移除DataFrame
        if 'training_df' in report['ml_training']:
            del report['ml_training']['training_df']
        
        report_file = self.working_dir / "workflow_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n  报告已保存: {report_file}")
    
    def print_summary(self, total_time: float):
        """打印执行摘要"""
        print("\n" + "="*70)
        print("  演示工作流完成！| Demo Workflow Completed!")
        print("="*70)
        
        print(f"\n  📊 结果摘要:")
        print(f"    化学式: {self.results['formula']}")
        print(f"    DFT能量: {self.results['dft']['energy_per_atom']:.4f} eV/atom")
        
        analysis = self.results.get('analysis', {})
        print(f"\n  📈 离子传输性质:")
        for T, D in analysis.get('diffusion_coefficients', {}).items():
            print(f"    {T}K: D = {D:.2e} cm²/s")
        
        Ea = analysis.get('activation_energy', 0)
        print(f"\n    活化能 Ea = {Ea:.3f} eV")
        
        # 解释活化能
        if Ea < 0.3:
            conductivity_level = "超离子导体"
        elif Ea < 0.5:
            conductivity_level = "良好导体"
        elif Ea < 0.8:
            conductivity_level = "中等导体"
        else:
            conductivity_level = "绝缘体/差导体"
        
        print(f"    评级: {conductivity_level}")
        
        print(f"\n  ⏱ 执行时间:")
        for stage, duration in self.timing.items():
            print(f"    {stage}: {duration:.2f}s")
        print(f"    总计: {total_time:.2f}s ({total_time/60:.1f}min)")
        
        print(f"\n  📁 输出文件:")
        print(f"    工作目录: {self.working_dir.absolute()}")
        print(f"    - workflow_report.json  (完整报告)")
        print(f"    - figures/              (可视化图表)")
        print(f"    - *.csv                 (数据文件)")
        
        print(f"\n  🔗 下一步:")
        print(f"    1. 查看图表: {self.working_dir / 'figures'}")
        print(f"    2. 读取报告: {self.working_dir / 'workflow_report.json'}")
        print(f"    3. 尝试真实计算: 安装VASP/LAMMPS/DeepMD并运行完整工作流")
        
        print("\n" + "="*70)


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DFT+LAMMPS Framework Demo Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python demo_workflow.py                    # 使用默认材料 Li3PS4
    python demo_workflow.py -f Li7P3S11        # 指定材料
    python demo_workflow.py -o ./my_output     # 自定义输出目录
        """
    )
    
    parser.add_argument('-f', '--formula', type=str, default='Li3PS4',
                       help='目标材料化学式 (默认: Li3PS4)')
    parser.add_argument('-o', '--output', type=str, default='./demo_output',
                       help='输出目录 (默认: ./demo_output)')
    parser.add_argument('-t', '--temps', type=float, nargs='+',
                       default=[300, 500, 700, 900],
                       help='MD模拟温度点 (默认: 300 500 700 900)')
    
    args = parser.parse_args()
    
    # 创建工作流并运行
    workflow = DemoWorkflow(working_dir=args.output)
    
    try:
        results = workflow.run(formula=args.formula)
        return 0
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
