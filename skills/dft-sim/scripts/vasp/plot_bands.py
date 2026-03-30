#!/usr/bin/env python3
"""
VASP能带结构后处理脚本
从EIGENVAL文件提取能带数据并绘图
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.collections import LineCollection

def read_eigenval(filename='EIGENVAL'):
    """读取VASP EIGENVAL文件"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 读取基本信息
    line0 = lines[0].split()
    nelect = int(line0[0])
    nkpoints = int(line0[1])
    nbands = int(line0[2])
    
    # 读取k点和能带
    kpoints = []
    energies = []
    
    line_idx = 5  # 从第6行开始
    
    for ik in range(nkpoints):
        # 读取k点坐标和权重
        k_line = lines[line_idx].split()
        kx, ky, kz = float(k_line[0]), float(k_line[1]), float(k_line[2])
        kpoints.append([kx, ky, kz])
        line_idx += 1
        
        # 读取能带
        band_energies = []
        for ib in range(nbands):
            e_line = lines[line_idx].split()
            energy = float(e_line[1])
            band_energies.append(energy)
            line_idx += 1
        energies.append(band_energies)
    
    return np.array(kpoints), np.array(energies), nelect

def calculate_kpath(kpoints):
    """计算k点路径距离"""
    kpath = [0.0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i-1])
        kpath.append(kpath[-1] + dk)
    return np.array(kpath)

def plot_bands(kpath, energies, fermi_energy=0, output='bands.png',
               klabels=None, klabel_positions=None,
               emin=-10, emax=10, title='Band Structure'):
    """绘制能带图"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 减去费米能
    energies_shifted = energies - fermi_energy
    
    # 绘制能带
    for ib in range(energies.shape[1]):
        ax.plot(kpath, energies_shifted[:, ib], 'b-', linewidth=1.0, alpha=0.8)
    
    # 费米能级线
    ax.axhline(y=0, color='r', linestyle='--', linewidth=0.8, label='Fermi Level')
    
    # 高对称点标记
    if klabel_positions and klabels:
        for pos in klabel_positions:
            ax.axvline(x=kpath[pos], color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks([kpath[p] for p in klabel_positions])
        ax.set_xticklabels(klabels)
    
    ax.set_xlim(kpath[0], kpath[-1])
    ax.set_ylim(emin, emax)
    ax.set_xlabel('k-point', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Band structure saved to {output}")
    
    return fig, ax

def extract_fermi_energy(outcar='OUTCAR'):
    """从OUTCAR提取费米能级"""
    try:
        with open(outcar, 'r') as f:
            for line in f:
                if 'E-fermi' in line:
                    return float(line.split()[2])
    except:
        pass
    return 0.0

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot VASP band structure')
    parser.add_argument('--eigenval', default='EIGENVAL', help='EIGENVAL file')
    parser.add_argument('--outcar', default='OUTCAR', help='OUTCAR file')
    parser.add_argument('--output', default='bands.png', help='Output image')
    parser.add_argument('--emin', type=float, default=-10, help='Minimum energy')
    parser.add_argument('--emax', type=float, default=10, help='Maximum energy')
    parser.add_argument('--fermi', type=float, default=None, help='Fermi energy')
    
    args = parser.parse_args()
    
    # 读取数据
    kpoints, energies, nelect = read_eigenval(args.eigenval)
    kpath = calculate_kpath(kpoints)
    
    # 获取费米能
    if args.fermi is None:
        fermi_energy = extract_fermi_energy(args.outcar)
        print(f"Fermi energy: {fermi_energy:.4f} eV")
    else:
        fermi_energy = args.fermi
    
    # 绘制能带
    plot_bands(kpath, energies, fermi_energy, args.output,
               emin=args.emin, emax=args.emax)
