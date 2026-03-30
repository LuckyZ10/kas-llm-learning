#!/usr/bin/env python3
"""
VASP结果对比分析脚本
对比不同计算的结果
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_all_data(directory='.'):
    """提取所有VASP计算数据"""
    results = []
    
    for root, dirs, files in os.walk(directory):
        if 'OUTCAR' in files and 'POSCAR' in files:
            data = {'directory': root}
            
            # 提取能量
            outcar_path = os.path.join(root, 'OUTCAR')
            with open(outcar_path, 'r') as f:
                content = f.read()
                
                # 总能量
                import re
                energy_match = re.findall(r'TOTEN\s+=\s+([-\d.]+)\s+eV', content)
                if energy_match:
                    data['energy'] = float(energy_match[-1])
                
                # 体积
                vol_match = re.findall(r'volume of cell\s*:\s+([\d.]+)', content)
                if vol_match:
                    data['volume'] = float(vol_match[-1])
                
                # 压强
                press_match = re.findall(r'pressure\s*=\s+([-\d.]+)', content)
                if press_match:
                    data['pressure'] = float(press_match[-1])
                
                # 带隙 (如果有)
                gap_match = re.search(r'band gap\s*=\s+([\d.]+)', content, re.IGNORECASE)
                if gap_match:
                    data['band_gap'] = float(gap_match.group(1))
            
            results.append(data)
    
    return results

def plot_comparison(results, output='comparison.png'):
    """绘制结果对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    dirs = [r['directory'].split('/')[-1] for r in results]
    
    # 能量对比
    if 'energy' in results[0]:
        energies = [r.get('energy', 0) for r in results]
        axes[0, 0].bar(range(len(dirs)), energies)
        axes[0, 0].set_xticks(range(len(dirs)))
        axes[0, 0].set_xticklabels(dirs, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Energy (eV)')
        axes[0, 0].set_title('Total Energy')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 体积对比
    if 'volume' in results[0]:
        volumes = [r.get('volume', 0) for r in results]
        axes[0, 1].bar(range(len(dirs)), volumes, color='orange')
        axes[0, 1].set_xticks(range(len(dirs)))
        axes[0, 1].set_xticklabels(dirs, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Volume (Å³)')
        axes[0, 1].set_title('Cell Volume')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 压强对比
    if 'pressure' in results[0]:
        pressures = [r.get('pressure', 0) for r in results]
        axes[1, 0].bar(range(len(dirs)), pressures, color='green')
        axes[1, 0].set_xticks(range(len(dirs)))
        axes[1, 0].set_xticklabels(dirs, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Pressure (kB)')
        axes[1, 0].set_title('Pressure')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 带隙对比
    if 'band_gap' in results[0]:
        gaps = [r.get('band_gap', 0) for r in results]
        axes[1, 1].bar(range(len(dirs)), gaps, color='red')
        axes[1, 1].set_xticks(range(len(dirs)))
        axes[1, 1].set_xticklabels(dirs, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Band Gap (eV)')
        axes[1, 1].set_title('Band Gap')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output}")

def generate_report(results, output='report.txt'):
    """生成文本报告"""
    
    with open(output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VASP Calculation Results Report\n")
        f.write(f"Generated: {__import__('datetime').datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        
        # 表头
        headers = ['Directory', 'Energy (eV)', 'Volume (Å³)', 'Pressure (kB)', 'Band Gap (eV)']
        f.write(f"{headers[0]:<30} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15}\n")
        f.write("-" * 90 + "\n")
        
        # 数据
        for r in results:
            dir_name = r['directory'].split('/')[-1][:28]
            energy = f"{r.get('energy', 'N/A'):.6f}" if 'energy' in r else 'N/A'
            volume = f"{r.get('volume', 'N/A'):.2f}" if 'volume' in r else 'N/A'
            pressure = f"{r.get('pressure', 'N/A'):.2f}" if 'pressure' in r else 'N/A'
            gap = f"{r.get('band_gap', 'N/A'):.3f}" if 'band_gap' in r else 'N/A'
            
            f.write(f"{dir_name:<30} {energy:<15} {volume:<15} {pressure:<15} {gap:<15}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Report saved to {output}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare VASP results')
    parser.add_argument('--dir', default='.', help='Root directory')
    parser.add_argument('--output', default='comparison.png', help='Plot output')
    parser.add_argument('--report', default='report.txt', help='Report output')
    
    args = parser.parse_args()
    
    print("Extracting data...")
    results = extract_all_data(args.dir)
    
    if results:
        print(f"Found {len(results)} calculations")
        plot_comparison(results, args.output)
        generate_report(results, args.report)
    else:
        print("No calculations found!")
