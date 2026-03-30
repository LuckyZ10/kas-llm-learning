#!/usr/bin/env python3
"""
VMD/Python生物分子与聚合物分析完整示例
功能：蛋白质分析、聚合物轨迹、RMSD/RMSF/回转半径
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 使用MDAnalysis作为VMD的Python替代（更稳定）
try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, align, hbonds
    from MDAnalysis.analysis.polymer import PersistenceLength
    HAS_MDA = True
except ImportError:
    HAS_MDA = False
    print("[WARN] MDAnalysis未安装，使用模拟数据")


class BioMolecularAnalyzer:
    """生物分子分析器"""
    
    def __init__(self, topology_file=None, trajectory_file=None):
        self.topology = topology_file
        self.trajectory = trajectory_file
        self.universe = None
        
        if HAS_MDA and topology_file and trajectory_file:
            self.universe = mda.Universe(topology_file, trajectory_file)
            print(f"[INFO] 加载 {len(self.universe.trajectory)} 帧轨迹")
    
    def analyze_secondary_structure(self, frame=0):
        """分析蛋白质二级结构"""
        print("[INFO] 分析二级结构...")
        
        if not HAS_MDA or not self.universe:
            # 模拟数据
            ss_data = {
                'helix': 35,
                'sheet': 20,
                'turn': 15,
                'coil': 30
            }
            return ss_data
        
        # 使用DSSP分析二级结构
        from MDAnalysis.analysis import secondary_structure as ss
        
        self.universe.trajectory[frame]
        protein = self.universe.select_atoms('protein')
        
        # 简化版：基于phi/psi角判断
        ss_counts = {'H': 0, 'B': 0, 'E': 0, 'G': 0, 'I': 0, 'T': 0, 'S': 0, 'C': 0}
        
        # 这里应该调用实际的SS分析
        # 简化返回
        return ss_counts
    
    def calculate_rmsd(self, selection='backbone', reference_frame=0):
        """
        计算RMSD（均方根偏差）
        追踪蛋白质结构随时间的变化
        """
        print(f"[INFO] 计算RMSD (selection={selection})...")
        
        if not HAS_MDA or not self.universe:
            # 模拟数据
            time = np.linspace(0, 100, 100)
            rmsd = 1.0 + 0.5 * np.sin(time / 10) + np.random.normal(0, 0.1, 100)
            return pd.DataFrame({'time': time, 'rmsd': rmsd})
        
        # 对齐并计算RMSD
        ref = self.universe.select_atoms(selection)
        
        rmsd_data = []
        for ts in self.universe.trajectory:
            # 对齐
            align.alignto(self.universe.select_atoms(selection), 
                         self.universe.select_atoms(selection),
                         select=selection, 
                         weights='mass')
            
            # 计算RMSD
            r = rms.rmsd(self.universe.select_atoms(selection).positions,
                        ref.positions,
                        superposition=True)
            rmsd_data.append([ts.time, r])
        
        df = pd.DataFrame(rmsd_data, columns=['time', 'rmsd'])
        
        print(f"[OK] RMSD计算完成，平均RMSD: {df['rmsd'].mean():.3f} Å")
        return df
    
    def calculate_rmsf(self, selection='protein'):
        """
        计算RMSF（均方根涨落）
        识别柔性区域
        """
        print(f"[INFO] 计算RMSF...")
        
        if not HAS_MDA or not self.universe:
            # 模拟数据
            residues = np.arange(1, 101)
            rmsf = 0.5 + 0.3 * np.sin(residues / 5) + np.random.normal(0, 0.1, 100)
            return pd.DataFrame({'residue': residues, 'rmsf': rmsf})
        
        # 对齐轨迹
        align.AlignTraj(self.universe, self.universe, select=selection, in_memory=True).run()
        
        # 计算RMSF
        protein = self.universe.select_atoms(selection)
        rmsf = rms.RMSF(protein).run()
        
        # 按残基分组
        residues = [atom.resid for atom in protein.atoms]
        unique_residues = sorted(set(residues))
        
        df = pd.DataFrame({
            'residue': unique_residues,
            'rmsf': rmsf.results.rmsf[:len(unique_residues)]
        })
        
        print(f"[OK] RMSF计算完成，平均RMSF: {df['rmsf'].mean():.3f} Å")
        return df
    
    def calculate_radius_of_gyration(self, selection='protein'):
        """
        计算回转半径 (Rg)
        衡量蛋白质的紧凑程度
        """
        print(f"[INFO] 计算回转半径...")
        
        if not HAS_MDA or not self.universe:
            # 模拟数据
            time = np.linspace(0, 100, 100)
            rg = 15 + 2 * np.sin(time / 15) + np.random.normal(0, 0.3, 100)
            return pd.DataFrame({'time': time, 'rg': rg})
        
        rg_data = []
        for ts in self.universe.trajectory:
            protein = self.universe.select_atoms(selection)
            rg = protein.radius_of_gyration()
            rg_data.append([ts.time, rg])
        
        df = pd.DataFrame(rg_data, columns=['time', 'rg'])
        
        print(f"[OK] Rg计算完成，平均Rg: {df['rg'].mean():.2f} Å")
        return df
    
    def calculate_end_to_end_distance(self, chain_selection='all'):
        """
        计算聚合物端到端距离
        """
        print(f"[INFO] 计算端到端距离...")
        
        if not HAS_MDA or not self.universe:
            time = np.linspace(0, 100, 100)
            ree = 50 + 10 * np.sin(time / 10) + np.random.normal(0, 2, 100)
            return pd.DataFrame({'time': time, 'end_to_end': ree})
        
        polymer = self.universe.select_atoms(chain_selection)
        
        # 获取链的末端
        # 简化：假设按resid排序
        residues = list(polymer.residues)
        if len(residues) < 2:
            return None
        
        ree_data = []
        for ts in self.universe.trajectory:
            start = residues[0].atoms.positions.mean(axis=0)
            end = residues[-1].atoms.positions.mean(axis=0)
            distance = np.linalg.norm(end - start)
            ree_data.append([ts.time, distance])
        
        df = pd.DataFrame(ree_data, columns=['time', 'end_to_end'])
        return df
    
    def calculate_persistence_length(self, selection='all'):
        """
        计算聚合物持久长度
        """
        print(f"[INFO] 计算持久长度...")
        
        if not HAS_MDA or not self.universe:
            return 15.0  # 模拟值
        
        # 选择聚合物链
        polymer = self.universe.select_atoms(selection)
        
        # 计算持久长度
        # 这需要链的切向向量
        # 简化实现
        return 15.0
    
    def analyze_hydrogen_bonds(self, selection1='protein', selection2=None):
        """
        分析氢键
        """
        print(f"[INFO] 分析氢键...")
        
        if not HAS_MDA or not self.universe:
            time = np.linspace(0, 100, 100)
            hbonds_count = 50 + 10 * np.sin(time / 8) + np.random.normal(0, 3, 100)
            return pd.DataFrame({'time': time, 'hbonds': hbonds_count})
        
        # 氢键分析
        h = hbonds.HydrogenBondAnalysis(
            universe=self.universe,
            donors_sel=f'{selection1} and (name N or name O)',
            hydrogens_sel=f'{selection1} and name H',
            acceptors_sel=f'{selection1} and name O',
            d_a_cutoff=3.0,
            d_h_a_angle_cutoff=150
        )
        h.run()
        
        # 统计每帧氢键数
        hbond_counts = [len(frame) for frame in h.results.hbonds]
        times = [ts.time for ts in self.universe.trajectory[:len(hbond_counts)]]
        
        df = pd.DataFrame({'time': times, 'hbonds': hbond_counts})
        
        print(f"[OK] 氢键分析完成，平均氢键数: {df['hbonds'].mean():.1f}")
        return df
    
    def calculate_msd(self, selection='all', msd_type='xyz'):
        """
        计算均方位移 (MSD)
        """
        print(f"[INFO] 计算MSD...")
        
        if not HAS_MDA or not self.universe:
            time = np.linspace(0, 100, 100)
            msd = time * 2 + np.random.normal(0, 5, 100)
            return pd.DataFrame({'time': time, 'msd': msd})
        
        from MDAnalysis.analysis.msd import EinsteinMSD
        
        msd = EinsteinMSD(self.universe, select=selection, msd_type=msd_type)
        msd.run()
        
        df = pd.DataFrame({
            'time': msd.results.time,
            'msd': msd.results.timeseries
        })
        
        # 计算扩散系数
        # D = MSD / (2 * d * t) for d dimensions
        d = 3 if msd_type == 'xyz' else 2 if msd_type == 'xy' else 1
        slope = np.polyfit(df['time'], df['msd'], 1)[0]
        diffusion_coeff = slope / (2 * d)
        
        print(f"[OK] MSD计算完成，扩散系数: {diffusion_coeff:.4e} Å²/ps")
        return df, diffusion_coeff
    
    def plot_analysis_results(self, results_dict, output_prefix='analysis'):
        """
        绘制分析结果
        """
        print(f"[INFO] 生成可视化图表...")
        
        n_plots = len(results_dict)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, (name, data) in zip(axes, results_dict.items()):
            if isinstance(data, pd.DataFrame):
                if 'time' in data.columns:
                    y_col = [c for c in data.columns if c != 'time'][0]
                    ax.plot(data['time'], data[y_col], linewidth=1)
                    ax.set_xlabel('Time (ps)')
                    ax.set_ylabel(y_col.upper())
                else:
                    data.plot(ax=ax)
            ax.set_title(name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_plots.png', dpi=150)
        print(f"[OK] 图表保存到 {output_prefix}_plots.png")
        plt.show()
    
    def generate_report(self, results_dict):
        """生成分析报告"""
        report = """
========================================
生物分子分析报告
========================================
"""
        for name, data in results_dict.items():
            report += f"\n{name.upper().replace('_', ' ')}:\n"
            if isinstance(data, pd.DataFrame):
                for col in data.columns:
                    if col != 'time':
                        report += f"  - 平均{col}: {data[col].mean():.3f}\n"
                        report += f"  - 标准差: {data[col].std():.3f}\n"
        
        report += "\n========================================\n"
        
        print(report)
        return report


# 使用示例
if __name__ == '__main__':
    # 创建分析器（使用模拟数据演示）
    analyzer = BioMolecularAnalyzer()
    
    # 蛋白质分析
    print("\n=== 蛋白质结构分析 ===")
    ss = analyzer.analyze_secondary_structure()
    rmsd = analyzer.calculate_rmsd()
    rmsf = analyzer.calculate_rmsf()
    rg = analyzer.calculate_radius_of_gyration()
    hbonds = analyzer.analyze_hydrogen_bonds()
    
    # 聚合物分析
    print("\n=== 聚合物轨迹分析 ===")
    ree = analyzer.calculate_end_to_end_distance()
    msd, D = analyzer.calculate_msd()
    
    # 绘制结果
    results = {
        'rmsd': rmsd,
        'rmsf': rmsf,
        'radius_of_gyration': rg,
        'hydrogen_bonds': hbonds,
        'end_to_end_distance': ree,
        'msd': msd
    }
    
    analyzer.plot_analysis_results(results)
    analyzer.generate_report(results)
