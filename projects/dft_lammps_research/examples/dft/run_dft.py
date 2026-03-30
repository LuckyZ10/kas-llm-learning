#!/usr/bin/env python3
"""
DFT计算示例 - 结构优化
DFT Calculation Example - Structure Relaxation
"""

from ase import Atoms
from ase.io import read, write
from ase.calculators.vasp import Vasp
from ase.optimize import BFGS
import numpy as np


def relax_structure(poscar_file: str, output_dir: str = "./dft_output"):
    """
    运行VASP结构优化
    
    Args:
        poscar_file: 输入POSCAR文件路径
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取结构
    atoms = read(poscar_file)
    print(f"Loaded structure: {atoms.get_chemical_formula()}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Initial cell:\n{atoms.get_cell()}")
    
    # 设置VASP计算器
    calc = Vasp(
        xc='PBE',
        encut=520,
        kpts=(5, 5, 5),
        ibrion=2,       # 共轭梯度优化
        isif=3,         # 优化晶胞和原子位置
        nsw=200,        # 最大离子步数
        ediffg=-0.01,   # 力收敛标准 (eV/Å)
        ediff=1e-6,     # 电子收敛标准
        ismear=0,
        sigma=0.05,
        lwave=False,    # 不保存WAVECAR
        lcharg=True,    # 保存CHGCAR
        directory=output_dir,
        command='mpirun -np 4 vasp_std'  # 根据系统修改
    )
    
    atoms.calc = calc
    
    # 运行优化
    print("\nStarting structure relaxation...")
    
    # 方法1: 使用VASP内置优化
    energy = atoms.get_potential_energy()
    
    # 方法2: 使用ASE优化器 (可选)
    # optimizer = BFGS(atoms, logfile=f'{output_dir}/relax.log')
    # optimizer.run(fmax=0.01)
    
    # 输出结果
    print(f"\n✓ Optimization complete!")
    print(f"Final energy: {energy:.6f} eV")
    print(f"Final energy per atom: {energy/len(atoms):.6f} eV/atom")
    
    # 读取优化后的结构
    final_atoms = read(f'{output_dir}/CONTCAR')
    print(f"\nFinal cell:\n{final_atoms.get_cell()}")
    
    # 保存结果
    write(f'{output_dir}/final_structure.vasp', final_atoms)
    
    return final_atoms, energy


def run_aimd(poscar_file: str, 
             temperature: float = 300,
             nsteps: int = 1000,
             output_dir: str = "./aimd_output"):
    """
    运行AIMD生成训练数据
    
    Args:
        poscar_file: 输入结构
        temperature: MD温度 (K)
        nsteps: MD步数
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    atoms = read(poscar_file)
    
    calc = Vasp(
        xc='PBE',
        encut=400,      # MD可用较低截断能
        kpts=(2, 2, 2), # MD可用较少k点
        ibrion=0,       # MD模拟
        mdalgo=2,       # Nose-Hoover
        smass=0,
        tebeg=temperature,
        teend=temperature,
        potim=1.0,      # 时间步长 (fs)
        nsw=nsteps,
        ismear=0,
        sigma=0.1,
        lwave=False,
        lcharg=False,
        directory=output_dir,
        command='mpirun -np 4 vasp_std'
    )
    
    atoms.calc = calc
    
    print(f"Running AIMD at {temperature}K for {nsteps} steps...")
    energy = atoms.get_potential_energy()
    print(f"✓ AIMD complete!")
    
    return output_dir


if __name__ == "__main__":
    # 运行结构优化
    relax_structure(
        poscar_file="Li3PS4.POSCAR",
        output_dir="./relax_output"
    )
    
    # 运行AIMD (可选)
    # run_aimd(
    #     poscar_file="Li3PS4.POSCAR",
    #     temperature=300,
    #     nsteps=1000
    # )
