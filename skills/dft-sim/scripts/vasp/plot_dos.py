#!/usr/bin/env python3
"""
VASP态密度后处理脚本
从DOSCAR提取DOS数据并绘图
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def read_doscar(filename='DOSCAR'):
    """读取VASP DOSCAR文件"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 读取头信息
    header = lines[0].split()
    natoms = int(header[0])
    
    # 第6行读取能量信息
    line6 = lines[5].split()
    emax = float(line6[0])
    emin = float(line6[1])
    nedos = int(line6[2])
    fermi = float(line6[3])
    
    # 读取总DOS
    dos_start = 6
    energy = []
    dos_total = []
    dos_integrated = []
    
    for i in range(nedos):
        line = lines[dos_start + i].split()
        energy.append(float(line[0]))
        dos_total.append(float(line[1]))
        dos_integrated.append(float(line[2]))
    
    energy = np.array(energy)
    dos_total = np.array(dos_total)
    dos_integrated = np.array(dos_integrated)
    
    # 读取投影DOS (如果存在)
    pdos_data = []
    line_idx = dos_start + nedos
    
    for iatom in range(natoms):
        # 跳过原子头行
        line_idx += 1
        
        atom_pdos = []
        for i in range(nedos):
            line = lines[line_idx + i].split()
            # 格式: energy s p d (f)
            atom_pdos.append([float(x) for x in line[1:]])
        
        pdos_data.append(np.array(atom_pdos))
        line_idx += nedos
    
    return energy, dos_total, dos_integrated, fermi, pdos_data

def plot_dos(energy, dos_total, fermi_energy, output='dos.png',
             emin=-10, emax=10, title='Density of States'):
    """绘制总DOS"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 减去费米能
    energy_shifted = energy - fermi_energy
    
    # 绘制DOS
    ax.fill_between(energy_shifted, 0, dos_total, alpha=0.5, color='blue')
    ax.plot(energy_shifted, dos_total, 'b-', linewidth=1.5, label='Total DOS')
    
    # 费米能级线
    ax.axvline(x=0, color='r', linestyle='--', linewidth=0.8, label='Fermi Level')
    
    ax.set_xlim(emin, emax)
    ax.set_ylim(0, max(dos_total) * 1.1)
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('DOS (states/eV)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"DOS plot saved to {output}")
    
    return fig, ax

def plot_pdos(energy, pdos_data, fermi_energy, output='pdos.png',
              emin=-10, emax=10, title='Projected DOS'):
    """绘制投影DOS"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    energy_shifted = energy - fermi_energy
    
    # 合并所有原子的PDOS
    total_pdos = np.zeros((len(energy), pdos_data[0].shape[1]))
    for atom_pdos in pdos_data:
        total_pdos += atom_pdos
    
    # 绘制各轨道贡献
    colors = ['red', 'green', 'blue', 'orange']
    labels = ['s', 'p', 'd', 'f']
    
    for i in range(min(total_pdos.shape[1], 4)):
        ax.fill_between(energy_shifted, 0, total_pdos[:, i], 
                        alpha=0.3, color=colors[i])
        ax.plot(energy_shifted, total_pdos[:, i], 
                color=colors[i], linewidth=1.5, label=labels[i])
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlim(emin, emax)
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('PDOS (states/eV)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"PDOS plot saved to {output}")
    
    return fig, ax

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot VASP DOS')
    parser.add_argument('--doscar', default='DOSCAR', help='DOSCAR file')
    parser.add_argument('--output', default='dos.png', help='Output image')
    parser.add_argument('--pdos', action='store_true', help='Plot PDOS')
    parser.add_argument('--emin', type=float, default=-10, help='Minimum energy')
    parser.add_argument('--emax', type=float, default=10, help='Maximum energy')
    
    args = parser.parse_args()
    
    # 读取数据
    energy, dos_total, dos_int, fermi, pdos_data = read_doscar(args.doscar)
    print(f"Fermi energy: {fermi:.4f} eV")
    print(f"Energy range: {energy.min():.2f} to {energy.max():.2f} eV")
    
    # 绘制DOS
    plot_dos(energy, dos_total, fermi, args.output, args.emin, args.emax)
    
    # 绘制PDOS
    if args.pdos and pdos_data:
        plot_pdos(energy, pdos_data, fermi, 
                  args.output.replace('.png', '_pdos.png'), args.emin, args.emax)
