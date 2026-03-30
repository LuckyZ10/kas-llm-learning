#!/usr/bin/env python3
"""
Quantum ESPRESSO能带结构后处理脚本
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def read_bands_output(filename='bands.out'):
    """读取QE bands输出"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    kpoints = []
    energies = []
    
    current_k = None
    current_bands = []
    
    for line in lines:
        if 'k =' in line:
            # 保存之前的k点数据
            if current_k is not None and current_bands:
                kpoints.append(current_k)
                energies.append(current_bands)
            
            # 解析新的k点
            parts = line.split('=')[1].split()
            current_k = [float(x) for x in parts[:3]]
            current_bands = []
        
        elif line.strip() and not line.startswith('k') and '=' not in line:
            # 能带能量行
            try:
                vals = [float(x) for x in line.split()]
                current_bands.extend(vals)
            except:
                pass
    
    # 保存最后一个k点
    if current_k is not None and current_bands:
        kpoints.append(current_k)
        energies.append(current_bands)
    
    return np.array(kpoints), np.array(energies)

def read_bands_dat(filename='bands.dat.gnu'):
    """读取gnuplot格式的能带数据"""
    data = np.loadtxt(filename)
    
    # 分离不同的能带 (由空行分隔)
    kpoints = []
    energies_list = []
    
    current_k = []
    current_e = []
    
    for row in data:
        if len(row) >= 2:
            current_k.append(row[0])
            current_e.append(row[1])
        else:
            # 能带结束
            if current_k:
                kpoints = current_k  # 所有能带k点相同
                energies_list.append(current_e)
            current_k = []
            current_e = []
    
    # 最后一个能带
    if current_k:
        energies_list.append(current_e)
    
    return np.array(kpoints), np.array(energies_list).T

def plot_bands_qe(kpath, energies, fermi_energy=0, output='bands_qe.png',
                  emin=-10, emax=10, title='QE Band Structure'):
    """绘制QE能带图"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 减去费米能
    energies_shifted = energies - fermi_energy
    
    # 绘制能带
    for ib in range(energies.shape[1]):
        ax.plot(kpath, energies_shifted[:, ib], 'b-', linewidth=1.0, alpha=0.8)
    
    # 费米能级线
    ax.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
    
    ax.set_xlim(kpath[0], kpath[-1])
    ax.set_ylim(emin, emax)
    ax.set_xlabel('k-path', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Band structure saved to {output}")
    
    return fig, ax

def read_fermi_from_scf(scf_out='scf.out'):
    """从SCF输出读取费米能级"""
    try:
        with open(scf_out, 'r') as f:
            for line in f:
                if 'the Fermi energy is' in line.lower():
                    return float(line.split()[4])
    except:
        pass
    return 0.0

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot QE band structure')
    parser.add_argument('--input', default='bands.dat.gnu', help='Input file')
    parser.add_argument('--scf', default='scf.out', help='SCF output for Fermi energy')
    parser.add_argument('--output', default='bands_qe.png', help='Output image')
    parser.add_argument('--emin', type=float, default=-10, help='Minimum energy')
    parser.add_argument('--emax', type=float, default=10, help='Maximum energy')
    parser.add_argument('--fermi', type=float, default=None, help='Fermi energy')
    
    args = parser.parse_args()
    
    # 读取数据
    kpath, energies = read_bands_dat(args.input)
    
    # 获取费米能
    if args.fermi is None:
        fermi = read_fermi_from_scf(args.scf)
        print(f"Fermi energy: {fermi:.4f} eV")
    else:
        fermi = args.fermi
    
    # 绘制
    plot_bands_qe(kpath, energies, fermi, args.output, args.emin, args.emax)
