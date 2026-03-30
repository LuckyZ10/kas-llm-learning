#!/usr/bin/env python3
"""
Quantum ESPRESSO DOS后处理脚本
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

def read_pdos(filename):
    """读取QE PDOS文件"""
    data = np.loadtxt(filename, skiprows=1)
    
    # 第一列是能量，后面是各轨道贡献
    energy = data[:, 0]
    pdos = data[:, 1:]
    
    return energy, pdos

def read_total_dos(filename='pwscf.dos'):
    """读取总DOS文件"""
    data = np.loadtxt(filename, skiprows=1)
    energy = data[:, 0]
    dos = data[:, 1]
    return energy, dos

def plot_qe_dos(energy, dos, fermi_energy=0, output='dos_qe.png',
                emin=-10, emax=10, title='QE DOS'):
    """绘制DOS"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 减去费米能
    energy_shifted = energy - fermi_energy
    
    # 绘制DOS
    ax.fill_between(energy_shifted, 0, dos, alpha=0.5, color='blue')
    ax.plot(energy_shifted, dos, 'b-', linewidth=1.5)
    
    # 费米能级线
    ax.axvline(x=0, color='r', linestyle='--', linewidth=0.8)
    
    ax.set_xlim(emin, emax)
    ax.set_ylim(0, max(dos) * 1.1)
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('DOS (states/eV)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"DOS plot saved to {output}")
    
    return fig, ax

def plot_qe_pdos(pdos_files, labels=None, fermi_energy=0, 
                 output='pdos_qe.png', emin=-10, emax=10):
    """绘制多个PDOS"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(pdos_files)))
    
    for i, (pdos_file, color) in enumerate(zip(pdos_files, colors)):
        energy, pdos = read_pdos(pdos_file)
        energy_shifted = energy - fermi_energy
        
        # 对所有轨道求和
        total_pdos = np.sum(pdos, axis=1)
        
        label = labels[i] if labels else f'Atom {i+1}'
        ax.plot(energy_shifted, total_pdos, color=color, 
                linewidth=1.5, label=label)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlim(emin, emax)
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('PDOS (states/eV)', fontsize=12)
    ax.set_title('Projected DOS', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"PDOS plot saved to {output}")
    
    return fig, ax

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot QE DOS')
    parser.add_argument('--dos', default='pwscf.dos', help='Total DOS file')
    parser.add_argument('--pdos', nargs='+', help='PDOS files')
    parser.add_argument('--output', default='dos_qe.png', help='Output image')
    parser.add_argument('--emin', type=float, default=-10, help='Minimum energy')
    parser.add_argument('--emax', type=float, default=10, help='Maximum energy')
    parser.add_argument('--fermi', type=float, default=0, help='Fermi energy')
    
    args = parser.parse_args()
    
    # 绘制总DOS
    if args.dos:
        energy, dos = read_total_dos(args.dos)
        plot_qe_dos(energy, dos, args.fermi, args.output, args.emin, args.emax)
    
    # 绘制PDOS
    if args.pdos:
        labels = [f.split('.')[-2] for f in args.pdos]
        plot_qe_pdos(args.pdos, labels, args.fermi,
                     args.output.replace('.png', '_pdos.png'), args.emin, args.emax)
