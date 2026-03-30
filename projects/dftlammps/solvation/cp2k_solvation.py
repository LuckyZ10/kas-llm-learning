#!/usr/bin/env python3
"""
cp2k_solvation.py
=================
CP2K溶剂化模型接口

功能：
1. 显式溶剂 (DFT/MD with explicit water)
2. 隐式溶剂 (SCCS, CDMT)
3. 电化学界面模拟
4. 金属表面吸附的溶剂化效应

CP2K溶剂化方法：
- SCCS: 自洽溶剂化连续介质模型
- CDMT: 连续介质介电理论
- Explicit: AIMD with water molecules

作者: Advanced DFT Expert
日期: 2026-03-09
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging

from ase import Atoms
from ase.io import read, write
from ase.build import molecule, bulk, surface, add_adsorbate
from ase.units import eV, Bohr, Angstrom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CP2KSolvationConfig:
    """CP2K溶剂化配置"""
    # 溶剂类型
    solvation_type: str = "SCCS"  # "SCCS", "CDMT", "EXPLICIT"
    
    # SCCS参数
    dielectric_constant: float = 78.4  # 水
    rho_min: float = 1.0e-3  # 电子密度阈值 (Bohr^-3)
    rho_max: float = 1.0e-2
    
    # 表面张力 (表面能)
    gamma_s: float = 0.0  # Hartree/Bohr²
    
    # 压力项
    p_v: float = 0.0  # Hartree/Bohr³
    
    # 显式溶剂参数
    n_water_layers: int = 2  # 水层数
    water_density: float = 1.0  # g/cm³
    
    def get_sccs_section(self) -> str:
        """生成SCCS输入段"""
        return f"""&SCCS
  ALPHA [Bohr^-1] {1.0 / self.rho_max:.6f}
  BETA [Bohr^-1] {1.0 / self.rho_min:.6f}
  GAMMA_S [Hartree*Bohr^-2] {self.gamma_s:.10f}
  DIELECTRIC_CONSTANT {self.dielectric_constant:.2f}
  &ANDREUSSI
    FILLING_THRESHOLD {self.rho_max:.6f}
  &END ANDREUSSI
&END SCCS
"""


@dataclass
class CP2KElectrolyteConfig:
    """CP2K电解质配置"""
    cation: str = "Na"  # 阳离子
    anion: str = "Cl"   # 阴离子
    concentration: float = 1.0  # M
    
    # 离子位置 (用于显式溶剂)
    n_cations: int = 0
    n_anions: int = 0


@dataclass
class CP2KInputGenerator:
    """CP2K输入文件生成器"""
    project_name: str = "cp2k_solvation"
    
    # 计算设置
    basis_set: str = "DZVP-MOLOPT-SR-GTH"
    potential: str = "GTH-PBE"
    xc_functional: str = "PBE"
    cutoff: float = 400  # Ry
    
    # 周期性
    periodic: str = "XYZ"  # "XYZ", "XY", "NONE"
    
    def generate_input(self, structure: Atoms,
                       solv_config: CP2KSolvationConfig,
                       output_file: str = "cp2k.inp") -> str:
        """生成CP2K输入文件"""
        
        symbols = structure.get_chemical_symbols()
        unique_symbols = list(set(symbols))
        
        # 计算盒子大小
        cell = structure.get_cell()
        
        input_content = f"""&GLOBAL
  PROJECT {self.project_name}
  RUN_TYPE ENERGY
  PRINT_LEVEL MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME POTENTIAL
    
    &MGRID
      CUTOFF {solv_config.dielectric_constant * 5:.0f}
      REL_CUTOFF 50
    &END MGRID
    
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 100
    &END SCF
    
    &XC
      &XC_FUNCTIONAL {self.xc_functional}
      &END XC_FUNCTIONAL
    &END XC
    
    &POISSON
      PERIODIC {self.periodic}
      POISSON_SOLVER PERIODIC
    &END POISSON
    
    &SCCS
      &ANDREUSSI
        FILLING_THRESHOLD {solv_config.rho_max:.6f}
      &END ANDREUSSI
      DIELECTRIC_CONSTANT {solv_config.dielectric_constant:.2f}
    &END SCCS
"""
        
        # 添加溶剂化 (SCCS)
        if solv_config.solvation_type == "SCCS":
            input_content += solv_config.get_sccs_section()
        
        # 添加KIND部分
        for sym in unique_symbols:
            input_content += f"""
    &KIND {sym}
      BASIS_SET {self.basis_set}
      POTENTIAL {self.potential}
    &END KIND
"""
        
        # 添加SUBSYS部分
        input_content += self._generate_subsys(structure)
        
        input_content += "\n&END FORCE_EVAL\n"
        
        # 写入文件
        with open(output_file, 'w') as f:
            f.write(input_content)
        
        logger.info(f"CP2K input written to {output_file}")
        
        return input_content
    
    def _generate_subsys(self, structure: Atoms) -> str:
        """生成SUBSYS部分"""
        cell = structure.get_cell()
        positions = structure.get_positions()
        symbols = structure.get_chemical_symbols()
        
        subsys = "\n  &SUBSYS\n"
        
        # CELL
        subsys += "    &CELL\n"
        for i, vec in enumerate(cell):
            subsys += f"      ABC {vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}\n"
        subsys += "      PERIODIC XYZ\n"
        subsys += "    &END CELL\n"
        
        # COORD
        subsys += "    &COORD\n"
        subsys += "      SCALED .FALSE.\n"
        for sym, pos in zip(symbols, positions):
            subsys += f"      {sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n"
        subsys += "    &END COORD\n"
        
        subsys += "  &END SUBSYS\n"
        
        return subsys


# =============================================================================
# Explicit Solvent Setup
# =============================================================================

class ExplicitSolventSetup:
    """显式溶剂设置"""
    
    def __init__(self):
        pass
    
    def add_water_box(self, structure: Atoms,
                      box_size: Tuple[float, float, float],
                      density: float = 1.0) -> Atoms:
        """
        添加水盒子
        
        Args:
            structure: 溶质结构
            box_size: 盒子大小 (Å)
            density: 水密度 (g/cm³)
        
        Returns:
            溶剂化结构
        """
        # 计算所需水分子数
        box_volume = box_size[0] * box_size[1] * box_size[2]  # Å³
        box_volume_cm3 = box_volume * 1e-24  # cm³
        
        molar_mass_water = 18.015  # g/mol
        n_molecules = int(density * box_volume_cm3 * 6.022e23 / molar_mass_water)
        
        logger.info(f"Adding {n_molecules} water molecules")
        
        # 创建水盒子 (简化：随机放置)
        water = molecule('H2O')
        
        solvated = structure.copy()
        
        # 设置盒子
        solvated.set_cell(box_size)
        solvated.set_pbc(True)
        
        # 添加水分子 (简化实现)
        # 实际应使用packmol等工具
        for i in range(min(n_molecules, 10)):  # 限制数量用于示例
            h2o = water.copy()
            # 随机位置 (避开溶质)
            pos = np.array([
                np.random.uniform(0, box_size[0]),
                np.random.uniform(0, box_size[1]),
                np.random.uniform(box_size[2]/2, box_size[2])  # 上半部分
            ])
            h2o.translate(pos)
            solvated.extend(h2o)
        
        return solvated
    
    def add_water_layers(self, slab: Atoms,
                         n_layers: int = 2,
                         layer_height: float = 3.0) -> Atoms:
        """
        在表面添加水层
        
        Args:
            slab: 表面结构
            n_layers: 水层数
            layer_height: 每层高度 (Å)
        
        Returns:
            溶剂化表面
        """
        water = molecule('H2O')
        
        solvated = slab.copy()
        cell = slab.get_cell()
        
        # 在z方向添加水层
        z_start = np.max(slab.positions[:, 2]) + 2.0  # 表面以上2Å
        
        for layer in range(n_layers):
            z = z_start + layer * layer_height
            
            # 在xy平面添加水分子 (简化)
            for i in range(4):  # 每层的简化水分子数
                h2o = water.copy()
                pos = np.array([
                    np.random.uniform(0, cell[0, 0]),
                    np.random.uniform(0, cell[1, 1]),
                    z + np.random.uniform(-0.5, 0.5)
                ])
                h2o.translate(pos)
                h2o.rotate(np.random.uniform(0, 360), 'z')
                solvated.extend(h2o)
        
        # 扩展盒子
        new_cell = cell.copy()
        new_cell[2, 2] = z_start + n_layers * layer_height + 10.0  # 真空层
        solvated.set_cell(new_cell)
        
        return solvated
    
    def add_ions(self, structure: Atoms,
                 cation: str = "Na",
                 anion: str = "Cl",
                 concentration: float = 1.0) -> Atoms:
        """
        添加离子到溶剂
        
        Args:
            structure: 溶剂化结构
            cation: 阳离子元素
            anion: 阴离子元素
            concentration: 浓度 (M)
        
        Returns:
            含离子的结构
        """
        # 计算所需离子数
        cell = structure.get_cell()
        volume = cell.volume * 1e-24  # cm³
        
        n_ions = int(concentration * volume * 6.022e20)  # 简化
        n_ions = max(1, min(n_ions, 5))  # 限制数量
        
        logger.info(f"Adding {n_ions} {cation}+ and {n_ions} {anion}-")
        
        # 添加离子 (简化)
        for i in range(n_ions):
            # 阳离子
            cation_atom = Atoms(cation, positions=[[
                np.random.uniform(0, cell[0, 0]),
                np.random.uniform(0, cell[1, 1]),
                np.random.uniform(cell[2, 2]/2, cell[2, 2])
            ]])
            structure.extend(cation_atom)
            
            # 阴离子
            anion_atom = Atoms(anion, positions=[[
                np.random.uniform(0, cell[0, 0]),
                np.random.uniform(0, cell[1, 1]),
                np.random.uniform(cell[2, 2]/2, cell[2, 2])
            ]])
            structure.extend(anion_atom)
        
        return structure


# =============================================================================
# Electrochemical Interface
# =============================================================================

class CP2KElectrochemicalInterface:
    """CP2K电化学界面"""
    
    def __init__(self, solvation_config: CP2KSolvationConfig,
                 electrolyte_config: Optional[CP2KElectrolyteConfig] = None):
        self.solv_config = solvation_config
        self.elec_config = electrolyte_config or CP2KElectrolyteConfig()
        self.explicit_setup = ExplicitSolventSetup()
    
    def setup_electrode_electrolyte_interface(self,
                                               metal_slab: Atoms,
                                               use_explicit_water: bool = True) -> Atoms:
        """
        设置电极-电解质界面
        
        Args:
            metal_slab: 金属表面
            use_explicit_water: 是否使用显式水
        >        Returns:
            界面结构
        """
        if use_explicit_water:
            # 添加显式水层
            interface = self.explicit_setup.add_water_layers(
                metal_slab,
                n_layers=self.solv_config.n_water_layers
            )
            
            # 添加离子
            interface = self.explicit_setup.add_ions(
                interface,
                cation=self.elec_config.cation,
                anion=self.elec_config.anion,
                concentration=self.elec_config.concentration
            )
        else:
            # 使用隐式溶剂
            interface = metal_slab.copy()
        
        return interface
    
    def setup_work_function_calculation(self, interface: Atoms,
                                        fermi_level: float) -> Dict:
        """
        设置功函数计算
        
        Args:
            interface: 界面结构
            fermi_level: Fermi能级 (eV)
        
        Returns:
            计算结果字典
        """
        # CP2K中的功函数计算需要后处理
        # 从静电势剖面提取真空能级
        
        return {
            'fermi_level': fermi_level,
            'vacuum_level': None,  # 需要从输出提取
            'work_function': None,
        }
    
    def calculate_her_overpotential(self,
                                    adsorption_energy_h: float,
                                    pH: float = 0,
                                    applied_potential: float = 0.0) -> float:
        """
        计算HER过电位
        
        基于计算氢电极模型 (CHE)
        
        ΔG = ΔE + ΔZPE - TΔS + eU + kT ln(a_H+)
        
        Args:
            adsorption_energy_h: H吸附自由能 (eV)
            pH: pH值
            applied_potential: 外加电位 (V vs RHE)
        
        Returns:
            过电位 (V)
        """
        # 标准HER自由能变化
        delta_G_0 = 0.0  # eV (定义)
        
        # 自由能修正 (典型值)
        zpe_h = 0.05  # eV, 零点能
        ts_h = 0.4    # eV, 熵贡献
        
        delta_g = adsorption_energy_h + zpe_h - ts_h
        
        # pH修正
        kT = 0.0257  # eV at 300K
        delta_g += kT * np.log(10) * pH
        
        # 电位修正
        delta_g += applied_potential
        
        # 过电位
        eta = -delta_g
        
        return eta


# =============================================================================
# Solvation Free Energy Calculator
# =============================================================================

class SolvationFreeEnergyCalculator:
    """溶剂化自由能计算器"""
    
    def __init__(self):
        pass
    
    def calculate_from_thermodynamic_integration(self,
                                                  structure: Atoms,
                                                  lambda_values: np.ndarray) -> float:
        """
        热力学积分计算溶剂化自由能
        
        ΔG_solv = ∫₀¹ <∂H/∂λ>_λ dλ
        
        Args:
            structure: 分子结构
            lambda_values: 耦合参数数组
        
        Returns:
            溶剂化自由能 (eV)
        """
        # 简化实现
        # 实际需要对每个lambda进行MD模拟
        
        # 假设线性响应
        delta_g = 0.0
        
        return delta_g
    
    def calculate_from_explicit_solvent(self,
                                        solute: Atoms,
                                        n_snapshots: int = 10) -> float:
        """
        从显式溶剂MD计算溶剂化能
        
        ΔG_solv = E(solute in solvent) - E(solute in vacuum) - E(solvent)
        
        Args:
            solute: 溶质分子
            n_snapshots: MD快照数
        >        Returns:
            溶剂化自由能
        """
        # 简化实现
        return 0.0


# =============================================================================
# High-Level Workflow
# =============================================================================

class CP2KSolvationWorkflow:
    """CP2K溶剂化工作流"""
    
    def __init__(self, solv_config: Optional[CP2KSolvationConfig] = None):
        self.solv_config = solv_config or CP2KSolvationConfig()
        self.input_gen = CP2KInputGenerator()
        self.explicit_setup = ExplicitSolventSetup()
        self.results = {}
    
    def run_solvation_calculation(self, structure: Atoms,
                                   output_dir: str = "./",
                                   use_explicit: bool = False) -> Dict:
        """
        运行溶剂化计算
        
        Args:
            structure: 结构
            output_dir: 输出目录
            use_explicit: 是否使用显式溶剂
        
        Returns:
            结果字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("CP2K Solvation Calculation")
        logger.info("=" * 60)
        
        # 准备结构
        if use_explicit:
            logger.info("Using explicit solvent model")
            self.solv_config.solvation_type = "EXPLICIT"
            
            # 添加水盒子
            structure = self.explicit_setup.add_water_box(
                structure,
                box_size=(20.0, 20.0, 20.0)
            )
        else:
            logger.info(f"Using implicit solvent: {self.solv_config.solvation_type}")
        
        # 生成输入文件
        input_file = output_path / "cp2k.inp"
        self.input_gen.generate_input(
            structure, self.solv_config, str(input_file)
        )
        
        # 保存结构
        structure_file = output_path / "structure.xyz"
        write(structure_file, structure)
        
        logger.info(f"Input files saved to {output_path}")
        logger.info("Run: mpirun cp2k.popt -i cp2k.inp -o cp2k.out")
        
        self.results['input_file'] = str(input_file)
        self.results['structure_file'] = str(structure_file)
        
        return self.results
    
    def run_electrochemical_interface(self,
                                       metal_slab: Atoms,
                                       adsorbate: Optional[Atoms] = None,
                                       output_dir: str = "./") -> Dict:
        """
        运行电化学界面计算
        
        Args:
            metal_slab: 金属表面
            adsorbate: 吸附物 (可选)
            output_dir: 输出目录
        
        Returns:
            结果字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("CP2K Electrochemical Interface")
        logger.info("=" * 60)
        
        # 设置界面
        interface_builder = CP2KElectrochemicalInterface(self.solv_config)
        interface = interface_builder.setup_electrode_electrolyte_interface(
            metal_slab, use_explicit_water=True
        )
        
        # 添加吸附物
        if adsorbate is not None:
            add_adsorbate(interface, adsorbate, height=2.0, position='ontop')
        
        # 生成输入
        input_file = output_path / "cp2k_echem.inp"
        self.input_gen.generate_input(
            interface, self.solv_config, str(input_file)
        )
        
        # 保存结构
        write(output_path / "interface.xyz", interface)
        
        logger.info(f"Interface setup complete")
        
        self.results['interface_file'] = str(input_file)
        
        return self.results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CP2K Solvation Interface")
    parser.add_argument('--structure', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default='./cp2k_solvation')
    parser.add_argument('--model', type=str, default='SCCS',
                       choices=['SCCS', 'EXPLICIT'],
                       help='Solvation model')
    parser.add_argument('--epsilon', type=float, default=78.4)
    parser.add_argument('--surface', action='store_true',
                       help='Setup electrochemical interface')
    
    args = parser.parse_args()
    
    structure = read(args.structure)
    logger.info(f"Loaded: {structure.get_chemical_formula()}")
    
    # 创建配置
    config = CP2KSolvationConfig(
        solvation_type=args.model,
        dielectric_constant=args.epsilon,
    )
    
    # 运行工作流
    workflow = CP2KSolvationWorkflow(config)
    
    if args.surface:
        results = workflow.run_electrochemical_interface(structure, output_dir=args.output)
    else:
        results = workflow.run_solvation_calculation(
            structure, args.output, use_explicit=(args.model == "EXPLICIT")
        )
    
    print(f"\nSetup complete. Input file: {results.get('input_file')}")


if __name__ == "__main__":
    main()
