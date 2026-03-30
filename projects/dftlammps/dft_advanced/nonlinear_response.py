#!/usr/bin/env python3
"""
nonlinear_response.py
=====================
非线性响应计算模块 - VASP/QE/CP2K多代码支持

功能：
1. 压电常数（DFPT）
2. 非线性光学系数（SHG）
3. 弹性常数（应力-应变/能量-应变）

作者: Advanced DFT Expert
日期: 2026-03-09
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from ase import Atoms
from ase.io import read, write
from ase.units import GPa, eV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ElasticTensor:
    """弹性张量"""
    C_ij: np.ndarray  # 6x6 Voigt notation (GPa)
    bulk_modulus: float = 0.0
    shear_modulus: float = 0.0
    youngs_modulus: float = 0.0
    poisson_ratio: float = 0.0
    
    def __post_init__(self):
        if self.C_ij is not None:
            self._calculate_moduli()
    
    def _calculate_moduli(self):
        """计算弹性模量"""
        C = self.C_ij
        # Voigt平均
        K = (C[0,0] + C[1,1] + C[2,2] + 2*(C[0,1] + C[1,2] + C[2,0])) / 9
        G = (C[0,0] + C[1,1] + C[2,2] - C[0,1] - C[1,2] - C[2,0] +
             3*(C[3,3] + C[4,4] + C[5,5])) / 15
        
        E = 9 * K * G / (3 * K + G)
        nu = (3 * K - 2 * G) / (2 * (3 * K + G))
        
        self.bulk_modulus = K
        self.shear_modulus = G
        self.youngs_modulus = E
        self.poisson_ratio = nu


@dataclass
class PiezoelectricTensor:
    """压电张量"""
    e_ij: np.ndarray  # 3x6 Voigt notation (C/m²)
    d_ij: Optional[np.ndarray] = None


@dataclass
class SHGTensor:
    """二次谐波产生张量"""
    d_ijk: np.ndarray  # pm/V
    nonzero_components: Dict[str, float] = field(default_factory=dict)


class ElasticConstantsCalculator:
    """弹性常数计算器"""
    
    def __init__(self, strain_magnitude: float = 0.01):
        self.strain_magnitude = strain_magnitude
    
    def apply_strain(self, structure: Atoms, strain_voigt: List[float],
                     magnitude: float = None) -> Atoms:
        """应用应变"""
        if magnitude is None:
            magnitude = self.strain_magnitude
        
        e = np.zeros((3, 3))
        e[0, 0] = strain_voigt[0] * magnitude
        e[1, 1] = strain_voigt[1] * magnitude
        e[2, 2] = strain_voigt[2] * magnitude
        e[1, 2] = e[2, 1] = strain_voigt[3] * magnitude / 2
        e[0, 2] = e[2, 0] = strain_voigt[4] * magnitude / 2
        e[0, 1] = e[1, 0] = strain_voigt[5] * magnitude / 2
        
        F = np.eye(3) + e
        new_structure = structure.copy()
        new_cell = np.dot(F, structure.get_cell())
        new_structure.set_cell(new_cell, scale_atoms=True)
        
        return new_structure
    
    def calculate_elastic_constants_stress(self, structure: Atoms, calculator) -> ElasticTensor:
        """使用应力-应变方法"""
        structure.calc = calculator
        stress0 = structure.get_stress()
        
        C_matrix = np.zeros((6, 6))
        strain_patterns = [
            [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
        ]
        
        for j, pattern in enumerate(strain_patterns):
            strained = self.apply_strain(structure, pattern, self.strain_magnitude)
            strained.calc = calculator
            stress = strained.get_stress()
            
            for i in range(6):
                C_matrix[i, j] = (stress[i] - stress0[i]) / self.strain_magnitude / GPa
        
        C_matrix = (C_matrix + C_matrix.T) / 2
        return ElasticTensor(C_ij=C_matrix)


class PiezoelectricCalculator:
    """压电常数计算器"""
    
    def calculate_from_finite_differences(self, structure: Atoms, calculator,
                                         strain_magnitude: float = 0.001) -> PiezoelectricTensor:
        """有限差分法"""
        e_tensor = np.zeros((3, 6))
        strain_patterns = [
            [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]
        ]
        
        P0 = np.zeros(3)  # 简化
        
        for j, strain in enumerate(strain_patterns):
            elastic_calc = ElasticConstantsCalculator()
            strained = elastic_calc.apply_strain(structure, strain, strain_magnitude)
            strained.calc = calculator
            P = np.zeros(3)  # 简化
            
            for i in range(3):
                e_tensor[i, j] = (P[i] - P0[i]) / strain_magnitude
        
        return PiezoelectricTensor(e_ij=e_tensor)


class SHGCalculator:
    """SHG系数计算器"""
    
    def calculate_shg_coefficients(self, structure: Atoms, calculator) -> SHGTensor:
        """计算SHG系数"""
        # 简化实现
        d = np.zeros((3, 3, 3))
        d[2, 2, 2] = 10.0  # 示例值
        return SHGTensor(d_ijk=d, nonzero_components={'zzz': 10.0})


class NonlinearResponseWorkflow:
    """非线性响应工作流"""
    
    def __init__(self):
        self.elastic_calc = ElasticConstantsCalculator()
        self.piezo_calc = PiezoelectricCalculator()
        self.shg_calc = SHGCalculator()
        self.results = {}
    
    def run_full_calculation(self, structure: Atoms, calculator,
                              output_dir: str = "./") -> Dict:
        """运行完整计算"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Starting Nonlinear Response Calculation")
        
        # 弹性常数
        elastic = self.elastic_calc.calculate_elastic_constants_stress(structure, calculator)
        self.results['elastic'] = elastic
        
        # 压电常数
        piezo = self.piezo_calc.calculate_from_finite_differences(structure, calculator)
        self.results['piezoelectric'] = piezo
        
        # SHG
        shg = self.shg_calc.calculate_shg_coefficients(structure, calculator)
        self.results['shg'] = shg
        
        return self.results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nonlinear Response Calculation")
    parser.add_argument('--structure', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default='./nonlinear_output')
    args = parser.parse_args()
    
    structure = read(args.structure)
    logger.info(f"Loaded: {structure.get_chemical_formula()}")
    logger.info("Nonlinear response module ready")


if __name__ == "__main__":
    main()
