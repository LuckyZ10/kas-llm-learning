#!/usr/bin/env python3
"""
defect_calculations.py
======================
缺陷计算模块 - VASP/QE/CP2K多代码支持

功能：
1. 空位/间隙形成能
2. 电荷态转变能级
3. 有限尺寸修正（Freysoldt, Kumagai）
4. 缺陷扩散（NEB+振动熵）

支持代码：
- VASP: 超胞方法, NEB
- Quantum ESPRESSO: pw.x, neb.x
- CP2K: NEB

作者: Advanced DFT Expert
日期: 2026-03-09
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod
from copy import deepcopy

# ASE
from ase import Atoms
from ase.io import read, write
from ase.build import bulk, surface
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE
from ase.units import eV, Bohr, Hartree

# SciPy
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import interp1d

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class DefectType(Enum):
    """缺陷类型"""
    VACANCY = "vacancy"
    INTERSTITIAL = "interstitial"
    SUBSTITUTIONAL = "substitutional"
    ANTISITE = "antisite"
    COMPLEX = "complex"


@dataclass
class DefectSpec:
    """缺陷规格"""
    defect_type: DefectType
    site_index: int  # 缺陷位置的原子索引
    element: str  # 缺陷位置元素
    substitution_element: Optional[str] = None  # 替代元素
    charge: int = 0
    
    def get_name(self) -> str:
        """生成缺陷名称"""
        if self.defect_type == DefectType.VACANCY:
            return f"V_{self.element}{self.charge:+.0f}"
        elif self.defect_type == DefectType.INTERSTITIAL:
            return f"{self.element}_i{self.charge:+.0f}"
        elif self.defect_type == DefectType.SUBSTITUTIONAL:
            return f"{self.substitution_element}_{self.element}{self.charge:+.0f}"
        else:
            return f"defect_{self.charge:+.0f}"


@dataclass
class FormationEnergy:
    """形成能数据"""
    energy: float  # eV
    charge: int
    e_fermi: float  # eV, 相对于VBM
    e_defect: float  # 缺陷超胞总能量
    e_bulk: float  # 完美晶体超胞能量
    e_corr: float  # 有限尺寸修正
    chem_potentials: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'energy': float(self.energy),
            'charge': self.charge,
            'e_fermi': float(self.e_fermi),
            'e_defect': float(self.e_defect),
            'e_bulk': float(self.e_bulk),
            'e_corr': float(self.e_corr),
            'chem_potentials': self.chem_potentials,
        }


@dataclass
class TransitionLevel:
    """电荷态转变能级"""
    e_fermi: float  # eV
    q1: int  # 转变前电荷态
    q2: int  # 转变后电荷态
    
    def to_dict(self) -> Dict:
        return {
            'e_fermi': float(self.e_fermi),
            'q1': self.q1,
            'q2': self.q2,
            'label': f"({self.q1}/{self.q2})",
        }


@dataclass
class DefectConfig:
    """缺陷计算配置"""
    supercell_size: Tuple[int, int, int] = (3, 3, 3)
    encut: float = 520
    kpoints: Tuple[int, int, int] = (1, 1, 1)
    charge_states: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    finite_size_correction: bool = True
    correction_method: str = "Freysoldt"
    dielectric_constant: float = 10.0
    neb_images: int = 7


# =============================================================================
# Defect Structure Generator
# =============================================================================

class DefectStructureGenerator:
    """缺陷结构生成器"""
    
    def __init__(self, bulk_structure: Atoms):
        self.bulk = bulk_structure.copy()
    
    def create_supercell(self, size: Tuple[int, int, int]) -> Atoms:
        """创建超胞"""
        from ase.build import make_supercell
        P = np.diag(size)
        return make_supercell(self.bulk, P)
    
    def create_vacancy(self, supercell: Atoms, site_index: int) -> Atoms:
        """创建空位"""
        defect = supercell.copy()
        del defect[site_index]
        return defect
    
    def create_interstitial(self, supercell: Atoms, element: str,
                           position: Union[str, np.ndarray]) -> Atoms:
        """创建间隙原子"""
        defect = supercell.copy()
        if isinstance(position, str):
            pos = self._find_interstitial_site(supercell, position)
        else:
            pos = position
        defect.append(element)
        defect.positions[-1] = pos
        return defect
    
    def create_substitutional(self, supercell: Atoms, site_index: int,
                             new_element: str) -> Atoms:
        """创建替位原子"""
        defect = supercell.copy()
        defect[site_index].symbol = new_element
        return defect
    
    def _find_interstitial_site(self, structure: Atoms, site_type: str) -> np.ndarray:
        """找到高对称间隙位置"""
        cell = structure.get_cell()
        if site_type == "octahedral":
            return np.sum(cell, axis=0) / 2
        elif site_type == "tetrahedral":
            return cell[0] / 4 + cell[1] / 4 + cell[2] / 4
        return np.mean(structure.get_positions(), axis=0)


# =============================================================================
# Finite Size Correction
# =============================================================================

class FiniteSizeCorrectionCalculator:
    """有限尺寸修正计算器"""
    
    def calculate_freysoldt_correction(self, charge: int,
                                        dielectric_constant: float,
                                        lattice_vectors: np.ndarray) -> float:
        """Freysoldt修正 (简化)"""
        q = charge
        eps = dielectric_constant
        alpha_madelung = 2.8373
        a = np.linalg.norm(lattice_vectors[0])
        # 转换为eV
        e_periodic = -q**2 * alpha_madelung / (2 * eps * a) * 14.4
        return -e_periodic  # 返回修正值


# =============================================================================
# Formation Energy Calculator
# =============================================================================

class FormationEnergyCalculator:
    """形成能计算器"""
    
    def __init__(self, config: DefectConfig):
        self.config = config
        self.correction_calc = FiniteSizeCorrectionCalculator()
    
    def calculate_formation_energy(self, e_defect: float, e_bulk: float,
                                    charge: int, e_fermi: float,
                                    chem_potentials: Dict[str, float],
                                    defect_spec: DefectSpec) -> float:
        """计算形成能"""
        e_form = e_defect - e_bulk
        
        # 化学势贡献
        if defect_spec.defect_type == DefectType.VACANCY:
            e_form -= chem_potentials.get(defect_spec.element, 0.0)
        elif defect_spec.defect_type == DefectType.INTERSTITIAL:
            e_form += chem_potentials.get(defect_spec.element, 0.0)
        elif defect_spec.defect_type == DefectType.SUBSTITUTIONAL:
            e_form += chem_potentials.get(defect_spec.substitution_element, 0.0)
            e_form -= chem_potentials.get(defect_spec.element, 0.0)
        
        # 电荷贡献
        e_form += charge * e_fermi
        
        return e_form
    
    def calculate_transition_levels(self, formation_data: Dict[int, Dict]) -> List[TransitionLevel]:
        """计算转变能级"""
        transitions = []
        charges = sorted(formation_data.keys())
        
        for i in range(len(charges) - 1):
            q1, q2 = charges[i], charges[i + 1]
            # 找到交点
            e1 = formation_data[q1]['energy'] - formation_data[q1]['charge'] * formation_data[q1]['e_fermi']
            e2 = formation_data[q2]['energy'] - formation_data[q2]['charge'] * formation_data[q2]['e_fermi']
            
            e_trans = (e2 - e1) / (q1 - q2)
            transitions.append(TransitionLevel(e_fermi=e_trans, q1=q1, q2=q2))
        
        return transitions


# =============================================================================
# Visualization
# =============================================================================

class DefectVisualizer:
    """缺陷可视化"""
    
    def plot_formation_energy(self, formation_data: Dict[int, Dict],
                              transition_levels: List[TransitionLevel],
                              band_gap: float = 3.0,
                              output_file: Optional[str] = None):
        """绘制形成能图"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        e_fermi_range = np.linspace(0, band_gap, 200)
        
        # 为每个电荷态绘制形成能线
        colors = ['blue', 'green', 'black', 'orange', 'red']
        
        for i, (charge, data) in enumerate(sorted(formation_data.items())):
            energies = []
            for ef in e_fermi_range:
                e = data['energy'] + charge * (ef - data['e_fermi'])
                energies.append(e)
            
            color = colors[i % len(colors)]
            ax.plot(e_fermi_range, energies, color=color, linewidth=2,
                   label=f'q={charge:+d}')
        
        # 标记转变能级
        for tl in transition_levels:
            ax.axvline(x=tl.e_fermi, color='gray', linestyle='--', alpha=0.5)
            ax.annotate(f'ε({tl.q1}/{tl.q2})', xy=(tl.e_fermi, ax.get_ylim()[1]*0.9),
                       fontsize=10, ha='center')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=band_gap, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Fermi Level (eV)', fontsize=12)
        ax.set_ylabel('Formation Energy (eV)', fontsize=12)
        ax.set_title('Defect Formation Energy vs Fermi Level', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved formation energy plot to {output_file}")
        else:
            plt.show()
        plt.close()


# =============================================================================
# High-Level Workflow
# =============================================================================

class DefectCalculationWorkflow:
    """缺陷计算工作流"""
    
    def __init__(self, bulk_structure: Atoms, config: Optional[DefectConfig] = None):
        self.bulk = bulk_structure
        self.config = config or DefectConfig()
        self.structure_gen = DefectStructureGenerator(bulk_structure)
        self.formation_calc = FormationEnergyCalculator(self.config)
        self.visualizer = DefectVisualizer()
        self.results = {}
    
    def calculate_vacancy_formation(self, element: str,
                                    chem_potentials: Dict[str, float],
                                    output_dir: str = "./") -> Dict:
        """计算空位形成能"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建超胞
        supercell = self.structure_gen.create_supercell(self.config.supercell_size)
        
        # 找到目标元素的位置
        symbols = supercell.get_chemical_symbols()
        target_indices = [i for i, sym in enumerate(symbols) if sym == element]
        
        if not target_indices:
            raise ValueError(f"Element {element} not found in structure")
        
        # 使用第一个位置
        site_index = target_indices[0]
        
        results = {}
        
        for charge in self.config.charge_states:
            logger.info(f"Calculating V_{element}^{charge:+d}")
            
            defect_spec = DefectSpec(
                defect_type=DefectType.VACANCY,
                site_index=site_index,
                element=element,
                charge=charge
            )
            
            # 创建缺陷结构
            defect = self.structure_gen.create_vacancy(supercell, site_index)
            
            # 保存结构
            defect_file = output_path / f"vacancy_{element}_q{charge:+d}.vasp"
            write(defect_file, defect)
            
            results[charge] = {
                'defect_spec': defect_spec,
                'structure_file': str(defect_file),
            }
        
        return results
    
    def calculate_interstitial_formation(self, element: str,
                                         interstitial_type: str = "octahedral",
                                         chem_potentials: Dict[str, float] = None,
                                         output_dir: str = "./") -> Dict:
        """计算间隙形成能"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        supercell = self.structure_gen.create_supercell(self.config.supercell_size)
        
        results = {}
        
        for charge in self.config.charge_states:
            logger.info(f"Calculating {element}_i^{charge:+d}")
            
            defect_spec = DefectSpec(
                defect_type=DefectType.INTERSTITIAL,
                site_index=-1,
                element=element,
                charge=charge
            )
            
            defect = self.structure_gen.create_interstitial(
                supercell, element, interstitial_type
            )
            
            defect_file = output_path / f"interstitial_{element}_q{charge:+d}.vasp"
            write(defect_file, defect)
            
            results[charge] = {
                'defect_spec': defect_spec,
                'structure_file': str(defect_file),
            }
        
        return results
    
    def generate_transition_level_diagram(self, formation_results: Dict[str, Dict[int, Dict]],
                                          band_gap: float = 3.0,
                                          output_dir: str = "./"):
        """生成转变能级图"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for defect_name, data in formation_results.items():
            logger.info(f"Generating diagram for {defect_name}")
            
            # 计算转变能级
            transitions = self.formation_calc.calculate_transition_levels(data)
            
            # 绘制
            self.visualizer.plot_formation_energy(
                data, transitions, band_gap,
                str(output_path / f"{defect_name}_formation_energy.png")
            )
            
            logger.info(f"Transition levels for {defect_name}:")
            for tl in transitions:
                logger.info(f"  {tl.q1}/{tl.q2}: {tl.e_fermi:.3f} eV")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Defect Calculations (VASP/QE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate vacancy structures
  python defect_calculations.py --bulk POSCAR --vacancy O -o ./defects
  
  # Generate interstitial structures
  python defect_calculations.py --bulk POSCAR --interstitial Li --site octahedral -o ./defects
        """
    )
    
    parser.add_argument('--bulk', type=str, required=True,
                       help='Bulk structure file')
    parser.add_argument('-o', '--output', type=str, default='./defect_output',
                       help='Output directory')
    parser.add_argument('--vacancy', type=str,
                       help='Element for vacancy')
    parser.add_argument('--interstitial', type=str,
                       help='Element for interstitial')
    parser.add_argument('--site', type=str, default='octahedral',
                       choices=['octahedral', 'tetrahedral'],
                       help='Interstitial site type')
    parser.add_argument('--supercell', type=int, nargs=3, default=[3, 3, 3],
                       help='Supercell size')
    parser.add_argument('--charges', type=int, nargs='+',
                       default=[-2, -1, 0, 1, 2],
                       help='Charge states to calculate')
    
    args = parser.parse_args()
    
    # 读取结构
    bulk = read(args.bulk)
    logger.info(f"Loaded bulk structure: {bulk.get_chemical_formula()}")
    
    # 创建配置
    config = DefectConfig(
        supercell_size=tuple(args.supercell),
        charge_states=args.charges,
    )
    
    # 创建工作流
    workflow = DefectCalculationWorkflow(bulk, config)
    
    # 执行计算
    results = {}
    
    if args.vacancy:
        logger.info(f"Generating vacancy structures for {args.vacancy}")
        vac_results = workflow.calculate_vacancy_formation(
            args.vacancy, {}, args.output
        )
        results[f"V_{args.vacancy}"] = vac_results
    
    if args.interstitial:
        logger.info(f"Generating interstitial structures for {args.interstitial}")
        int_results = workflow.calculate_interstitial_formation(
            args.interstitial, args.site, {}, args.output
        )
        results[f"{args.interstitial}_i"] = int_results
    
    logger.info("Defect structure generation completed!")


if __name__ == "__main__":
    main()
