#!/usr/bin/env python3
"""
VASP结果提取脚本
用于从OUTCAR中提取关键信息
"""

import os
import re
import sys
import json
from pathlib import Path

def extract_energy(outcar_path):
    """提取总能量"""
    with open(outcar_path, 'r') as f:
        content = f.read()
    
    # 提取最终能量
    pattern = r'TOTEN\s+=\s+([-\d.]+)\s+eV'
    matches = re.findall(pattern, content)
    if matches:
        return float(matches[-1])
    return None

def extract_forces(outcar_path):
    """提取最大力"""
    with open(outcar_path, 'r') as f:
        lines = f.readlines()
    
    # 找最后一个IONIC STEP的力
    max_force = None
    for i, line in enumerate(lines):
        if 'TOTAL-FORCE' in line:
            # 读取接下来的几行
            forces = []
            for j in range(i+2, len(lines)):
                if lines[j].strip() == '' or '----' in lines[j]:
                    break
                parts = lines[j].split()
                if len(parts) >= 6:
                    fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                    force = (fx**2 + fy**2 + fz**2)**0.5
                    forces.append(force)
            if forces:
                max_force = max(forces)
    
    return max_force

def extract_convergence(outcar_path):
    """检查是否收敛"""
    with open(outcar_path, 'r') as f:
        content = f.read()
    
    converged = 'reached required accuracy' in content
    return converged

def extract_timing(outcar_path):
    """提取计算时间"""
    with open(outcar_path, 'r') as f:
        content = f.read()
    
    pattern = r'Total CPU time used.*?([\d.]+)\s+sec'
    match = re.search(pattern, content)
    if match:
        return float(match.group(1))
    return None

def extract_volume(outcar_path):
    """提取晶胞体积"""
    with open(outcar_path, 'r') as f:
        content = f.read()
    
    pattern = r'volume of cell\s*:\s+([\d.]+)'
    matches = re.findall(pattern, content)
    if matches:
        return float(matches[-1])
    return None

def analyze_vasp_results(directory='.'):
    """分析目录中的所有VASP结果"""
    results = []
    
    for root, dirs, files in os.walk(directory):
        if 'OUTCAR' in files:
            outcar_path = os.path.join(root, 'OUTCAR')
            
            result = {
                'directory': root,
                'energy': extract_energy(outcar_path),
                'max_force': extract_forces(outcar_path),
                'converged': extract_convergence(outcar_path),
                'cpu_time': extract_timing(outcar_path),
                'volume': extract_volume(outcar_path)
            }
            results.append(result)
    
    return results

def print_summary(results):
    """打印结果摘要"""
    print("=" * 80)
    print(f"{'Directory':<40} {'Energy (eV)':<15} {'Max Force':<12} {'Converged':<10}")
    print("=" * 80)
    
    for r in results:
        energy = f"{r['energy']:.6f}" if r['energy'] else 'N/A'
        force = f"{r['max_force']:.4f}" if r['max_force'] else 'N/A'
        conv = 'Yes' if r['converged'] else 'No'
        print(f"{r['directory']:<40} {energy:<15} {force:<12} {conv:<10}")
    
    print("=" * 80)

if __name__ == '__main__':
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    print(f"Analyzing VASP results in: {directory}")
    results = analyze_vasp_results(directory)
    
    if results:
        print_summary(results)
        
        # 保存为JSON
        with open('vasp_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to vasp_results.json")
    else:
        print("No VASP results found!")
