#!/usr/bin/env python3
"""
QE结果提取脚本
用于从pw.x输出中提取关键信息
"""

import os
import re
import sys
import json

def extract_energy(output_path):
    """提取总能量"""
    with open(output_path, 'r') as f:
        content = f.read()
    
    # 提取最终能量
    pattern = r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry'
    matches = re.findall(pattern, content)
    if matches:
        return float(matches[-1])
    return None

def extract_convergence(output_path):
    """检查是否收敛"""
    with open(output_path, 'r') as f:
        content = f.read()
    
    converged = 'convergence has been achieved' in content
    return converged

def extract_timing(output_path):
    """提取计算时间"""
    with open(output_path, 'r') as f:
        content = f.read()
    
    pattern = r'PWSCF\s+:\s+.*?([\d.]+)\s+CPU'
    match = re.search(pattern, content)
    if match:
        return float(match.group(1))
    return None

def extract_scf_cycles(output_path):
    """提取SCF迭代次数"""
    with open(output_path, 'r') as f:
        content = f.read()
    
    pattern = r'iteration #\s+(\d+)'
    matches = re.findall(pattern, content)
    if matches:
        return int(matches[-1])
    return None

def extract_fermi_energy(output_path):
    """提取费米能级"""
    with open(output_path, 'r') as f:
        content = f.read()
    
    pattern = r'the Fermi energy is\s+([\d.]+)\s+ev'
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def analyze_qe_results(directory='.'):
    """分析目录中的所有QE结果"""
    results = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.out') or file == 'pw.out' or file == 'scf.out':
                output_path = os.path.join(root, file)
                
                result = {
                    'file': output_path,
                    'energy_ry': extract_energy(output_path),
                    'energy_ev': extract_energy(output_path) * 13.6057 if extract_energy(output_path) else None,
                    'converged': extract_convergence(output_path),
                    'cpu_time': extract_timing(output_path),
                    'scf_cycles': extract_scf_cycles(output_path),
                    'fermi_energy': extract_fermi_energy(output_path)
                }
                results.append(result)
    
    return results

def print_summary(results):
    """打印结果摘要"""
    print("=" * 100)
    print(f"{'File':<50} {'Energy (Ry)':<15} {'SCF Cycles':<12} {'Converged':<10}")
    print("=" * 100)
    
    for r in results:
        energy = f"{r['energy_ry']:.8f}" if r['energy_ry'] else 'N/A'
        cycles = str(r['scf_cycles']) if r['scf_cycles'] else 'N/A'
        conv = 'Yes' if r['converged'] else 'No'
        print(f"{r['file']:<50} {energy:<15} {cycles:<12} {conv:<10}")
    
    print("=" * 100)

if __name__ == '__main__':
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    print(f"Analyzing QE results in: {directory}")
    results = analyze_qe_results(directory)
    
    if results:
        print_summary(results)
        
        # 保存为JSON
        with open('qe_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to qe_results.json")
    else:
        print("No QE results found!")
