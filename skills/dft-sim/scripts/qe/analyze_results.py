#!/usr/bin/env python3
"""
QE结果分析脚本
分析scf.out文件并提取关键信息
"""

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_qe_output(filename):
    """解析QE输出文件"""
    
    with open(filename, 'r') as f:
        content = f.read()
    
    data = {}
    
    # 提取总能量
    energy_match = re.search(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry', content)
    if energy_match:
        data['energy_ry'] = float(energy_match.group(1))
        data['energy_ev'] = data['energy_ry'] * 13.6057
    
    # 提取费米能
    fermi_match = re.search(r'the Fermi energy is\s+([\d.]+)\s+ev', re.IGNORECASE, content)
    if fermi_match:
        data['fermi_energy'] = float(fermi_match.group(1))
    
    # 提取晶格参数
    cell_match = re.search(r'celldm\(1\)\s*=\s*([\d.]+)', content)
    if cell_match:
        data['celldm1'] = float(cell_match.group(1))
    
    # 提取k点
    kpoints_match = re.search(r'number of k points=\s*(\d+)', content)
    if kpoints_match:
        data['nkpoints'] = int(kpoints_match.group(1))
    
    # 提取收敛步数
    scf_steps = len(re.findall(r'iteration #', content))
    data['scf_steps'] = scf_steps
    
    # 提取计算时间
    time_match = re.search(r'PWSCF\s+:\s+.*?([\d.]+)\s+CPU', content)
    if time_match:
        data['cpu_time'] = float(time_match.group(1))
    
    # 提取力 (如果有)
    forces = []
    for match in re.finditer(r'total force\s*=\s+([\d.]+)', content):
        forces.append(float(match.group(1)))
    if forces:
        data['total_force'] = forces[-1]
    
    # 提取应力 (如果有)
    stress_match = re.search(r'total\s+stress\s+.*?([\d.]+)\s+Ry', content, re.DOTALL)
    if stress_match:
        data['stress'] = float(stress_match.group(1))
    
    return data

def analyze_directory(directory='.'):
    """分析目录中的所有QE输出"""
    
    results = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.out'):
                filepath = os.path.join(root, file)
                
                try:
                    data = parse_qe_output(filepath)
                    data['file'] = filepath
                    data['name'] = os.path.basename(root)
                    results.append(data)
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")
    
    return results

def create_summary_table(results, output='summary.csv'):
    """创建汇总表格"""
    
    if not results:
        print("No results to summarize")
        return
    
    df = pd.DataFrame(results)
    
    # 选择关键列
    columns = ['name', 'energy_ev', 'fermi_energy', 'nkpoints', 'scf_steps', 'cpu_time']
    available_cols = [c for c in columns if c in df.columns]
    
    df_summary = df[available_cols]
    
    # 保存CSV
    df_summary.to_csv(output, index=False)
    print(f"Summary saved to {output}")
    
    # 打印表格
    print("\n" + "=" * 100)
    print(df_summary.to_string(index=False))
    print("=" * 100)

def plot_results(results, output='qe_analysis.png'):
    """绘制结果图表"""
    
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [r['name'] for r in results]
    
    # 能量对比
    if 'energy_ev' in results[0]:
        energies = [r.get('energy_ev', 0) for r in results]
        axes[0, 0].bar(range(len(names)), energies, color='steelblue')
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Energy (eV)')
        axes[0, 0].set_title('Total Energy')
        axes[0, 0].grid(True, alpha=0.3)
    
    # SCF步数
    if 'scf_steps' in results[0]:
        steps = [r.get('scf_steps', 0) for r in results]
        axes[0, 1].bar(range(len(names)), steps, color='coral')
        axes[0, 1].set_xticks(range(len(names)))
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('SCF Steps')
        axes[0, 1].set_title('Convergence Steps')
        axes[0, 1].grid(True, alpha=0.3)
    
    # CPU时间
    if 'cpu_time' in results[0]:
        times = [r.get('cpu_time', 0) for r in results]
        axes[1, 0].bar(range(len(names)), times, color='seagreen')
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('CPU Time (s)')
        axes[1, 0].set_title('Computation Time')
        axes[1, 0].grid(True, alpha=0.3)
    
    # k点数量
    if 'nkpoints' in results[0]:
        kpoints = [r.get('nkpoints', 0) for r in results]
        axes[1, 1].bar(range(len(names)), kpoints, color='mediumpurple')
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Number of k-points')
        axes[1, 1].set_title('k-point Sampling')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to {output}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze QE results')
    parser.add_argument('--dir', default='.', help='Directory to analyze')
    parser.add_argument('--csv', default='summary.csv', help='CSV output')
    parser.add_argument('--plot', default='qe_analysis.png', help='Plot output')
    
    args = parser.parse_args()
    
    print("Analyzing QE results...")
    results = analyze_directory(args.dir)
    
    if results:
        print(f"\nFound {len(results)} calculations\n")
        create_summary_table(results, args.csv)
        plot_results(results, args.plot)
    else:
        print("No QE output files found!")
