#!/usr/bin/env python3
"""
LAMMPS分析工具包 - 高级后处理
analyze_lammps.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, integrate
from scipy.optimize import curve_fit
import argparse
import json
from pathlib import Path

class LAMMPSAnalyzer:
    """LAMMPS模拟数据分析器"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.data = {}
        
    def read_log(self, log_file):
        """读取LAMMPS log文件"""
        properties = []
        data_started = False
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith('Step') or line.startswith('Loop'):
                if line.startswith('Step'):
                    headers = line.split()
                    data_started = True
            elif data_started and line.strip():
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) == len(headers):
                        properties.append(dict(zip(headers, values)))
                except:
                    pass
        
        return properties
    
    def analyze_rdf(self, rdf_file):
        """分析径向分布函数"""
        data = np.loadtxt(rdf_file, comments='#')
        r = data[:, 1]
        g_r = data[:, 2]
        
        # 找第一个峰
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(g_r, height=1.5, distance=10)
        
        results = {
            'r': r.tolist(),
            'g_r': g_r.tolist(),
            'first_peak_r': r[peaks[0]] if len(peaks) > 0 else None,
            'first_peak_height': g_r[peaks[0]] if len(peaks) > 0 else None
        }
        
        # 计算配位数
        rho = 0.085  # 假设密度
        cn = 4 * np.pi * rho * np.trapz(r**2 * g_r, r)
        results['coordination_number'] = float(cn)
        
        return results
    
    def analyze_diffusion(self, msd_file, dt=1.0):
        """分析扩散系数"""
        data = np.loadtxt(msd_file, comments='#')
        time = data[:, 0] * dt / 1000  # 转换为ps
        msd = data[:, 1]
        
        # 线性拟合
        slope, intercept, r_value, _, _ = stats.linregress(time, msd)
        D = slope / 6  # 3D扩散
        D_cm2s = D * 1e-16 / 1e-12
        
        return {
            'time': time.tolist(),
            'msd': msd.tolist(),
            'diffusion_coefficient': float(D_cm2s),
            'r_squared': float(r_value**2),
            'slope': float(slope),
            'intercept': float(intercept)
        }
    
    def analyze_energetics(self, log_file):
        """分析能量演化"""
        data = self.read_log(log_file)
        
        temps = [d.get('Temp', 0) for d in data]
        pes = [d.get('PotEng', 0) for d in data]
        steps = [d.get('Step', 0) for d in data]
        
        return {
            'steps': steps,
            'temperature': temps,
            'potential_energy': pes,
            'mean_temp': np.mean(temps),
            'std_temp': np.std(temps),
            'mean_pe': np.mean(pes),
            'std_pe': np.std(pes)
        }
    
    def generate_report(self, output_file='analysis_report.html'):
        """生成HTML报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LAMMPS Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>LAMMPS Simulation Analysis Report</h1>
            <p>Generated: {np.datetime64('now')}</p>
            
            <h2>Summary</h2>
            <div class="metric">
                <p>Results Directory: {self.results_dir}</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"Report generated: {output_file}")

def plot_comparison(data_files, labels, output='comparison.png'):
    """比较多组数据"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for data_file, label in zip(data_files, labels):
        data = np.loadtxt(data_file, comments='#')
        
        # 温度
        axes[0, 0].plot(data[:, 0], data[:, 1], label=label)
        # 能量
        axes[0, 1].plot(data[:, 0], data[:, 2], label=label)
        # 压力
        axes[1, 0].plot(data[:, 0], data[:, 3], label=label)
        # 密度
        axes[1, 1].plot(data[:, 0], data[:, 4], label=label)
    
    for ax in axes.flat:
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300)

def main():
    parser = argparse.ArgumentParser(description='LAMMPS Analysis Tool')
    parser.add_argument('--rdf', help='RDF file to analyze')
    parser.add_argument('--msd', help='MSD file to analyze')
    parser.add_argument('--log', help='Log file to analyze')
    parser.add_argument('--results-dir', help='Results directory')
    parser.add_argument('--output', default='analysis_results.json', help='Output file')
    
    args = parser.parse_args()
    
    analyzer = LAMMPSAnalyzer(args.results_dir or '.')
    results = {}
    
    if args.rdf:
        results['rdf'] = analyzer.analyze_rdf(args.rdf)
        print(f"RDF Analysis: First peak at {results['rdf']['first_peak_r']:.3f} Å")
    
    if args.msd:
        results['diffusion'] = analyzer.analyze_diffusion(args.msd)
        print(f"Diffusion Coefficient: {results['diffusion']['diffusion_coefficient']:.2e} cm²/s")
    
    if args.log:
        results['energetics'] = analyzer.analyze_energetics(args.log)
        print(f"Mean Temperature: {results['energetics']['mean_temp']:.1f} K")
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
