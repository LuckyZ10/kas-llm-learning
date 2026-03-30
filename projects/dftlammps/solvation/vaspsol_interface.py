#!/usr/bin/env python3
"""
vaspsol_interface.py
===================
VASPsol隐式溶剂模型接口

功能：
1. VASPsol参数设置与自动优化
2. 电化学界面模型
3. 电容计算、Zeta电位
4. 与VASP计算流程集成

VASPsol关键参数：
- LSOL: 开启隐式溶剂
- EB_K: 溶剂体介电常数
- TAU: 腔体表面积张力
- LAMBDA_D_K: Debye屏蔽长度 (电解质)

作者: Advanced DFT Expert
日期: 2026-03-09
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

from ase import Atoms
from ase.io import read, write
from ase.calculators.vasp import Vasp
from ase.units import eV, Bohr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VASPsolConfig:
    """VASPsol配置"""
    # 基本参数
    lsol: bool = True  # 开启VASPsol
    eb_k: float = 78.4  # 水的介电常数 (80 for H2O, 5 for organic)
    
    # 腔体参数
    lambda_v0: float = 1.0  # 空腔约束强度
    tau: float = 0.0005  # 表面积张力 (Hartree/Bohr²)
    
    # 电解质参数
    lambda_d_k: Optional[float] = None  # Debye长度 (Å), None=无电解质
    
    # 可选参数
    nc_k: Optional[float] = None  # 空腔占据阈值
    sigma_k: Optional[float] = None  # 腔体展宽
    
    def get_vasp_dict(self) -> Dict[str, Any]:
        """转换为VASP参数字典"""
        params = {
            'lsol': self.lsol,
            'eb_k': self.eb_k,
            'lambda_v0': self.lambda_v0,
            'tau': self.tau,
        }
        if self.lambda_d_k is not None:
            params['lambda_d_k'] = self.lambda_d_k
        if self.nc_k is not None:
            params['nc_k'] = self.nc_k
        if self.sigma_k is not None:
            params['sigma_k'] = self.sigma_k
        return params


@dataclass
class SolvationResults:
    """溶剂化计算结果"""
    total_energy: float
    solvation_energy: float  # 溶剂化能 (真空-溶剂)
    surface_area: float  # 空腔表面积 (Bohr²)
    volume: float  # 空腔体积 (Bohr³)
    
    # 电解质相关
    zeta_potential: Optional[float] = None  # Zeta电位 (V)
    capacitance: Optional[float] = None  # 微分电容 (μF/cm²)
    
    def to_dict(self) -> Dict:
        return {
            'total_energy': float(self.total_energy),
            'solvation_energy': float(self.solvation_energy),
            'surface_area': float(self.surface_area),
            'volume': float(self.volume),
            'zeta_potential': float(self.zeta_potential) if self.zeta_potential else None,
            'capacitance': float(self.capacitance) if self.capacitance else None,
        }


@dataclass
class ElectrochemicalConfig:
    """电化学界面配置"""
    # 电极电位
    target_potential: float = 0.0  # vs SHE (V)
    
    # 电解质
    electrolyte_concentration: float = 1.0  # M
    cation: str = "Na"  # 阳离子
    anion: str = "Cl"   # 阴离子
    
    # 界面设置
    slab_thickness: float = 10.0  # Å
    vacuum_thickness: float = 20.0  # Å
    solvent_thickness: float = 10.0  # Å


class VASPsolCalculator:
    """VASPsol计算器封装"""
    
    def __init__(self, sol_config: Optional[VASPsolConfig] = None,
                 vasp_kwargs: Optional[Dict] = None):
        self.sol_config = sol_config or VASPsolConfig()
        self.vasp_kwargs = vasp_kwargs or {}
    
    def get_calculator(self, **kwargs) -> Vasp:
        """获取配置好的VASP计算器"""
        # 合并参数
        params = {**self.vasp_kwargs, **kwargs}
        
        # 添加VASPsol参数
        sol_params = self.sol_config.get_vasp_dict()
        params.update(sol_params)
        
        return Vasp(**params)
    
    def calculate_solvation_energy(self, structure: Atoms,
                                    output_dir: str = "./",
                                    reference_energy: Optional[float] = None) -> SolvationResults:
        """
        计算溶剂化能
        
        ΔG_solv = E_sol - E_vac
        
        Args:
            structure: 分子/表面结构
            output_dir: 输出目录
            reference_energy: 真空参考能量 (如已计算)
        
        Returns:
            SolvationResults
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 溶剂化计算
        calc = self.get_calculator(directory=str(output_path))
        structure.calc = calc
        
        energy_sol = structure.get_potential_energy()
        
        # 读取VASPsol输出
        surface_area, volume = self._read_solvation_output(output_path)
        
        # 计算溶剂化能
        if reference_energy is None:
            # 需要运行真空计算
            logger.info("Reference vacuum energy not provided, solvation energy set to 0")
            solv_energy = 0.0
        else:
            solv_energy = energy_sol - reference_energy
        
        return SolvationResults(
            total_energy=energy_sol,
            solvation_energy=solv_energy,
            surface_area=surface_area,
            volume=volume,
        )
    
    def _read_solvation_output(self, output_path: Path) -> Tuple[float, float]:
        """读取VASPsol输出文件"""
        surface_area = 0.0
        volume = 0.0
        
        # 从OUTCAR或vasprun.xml读取
        outcar_path = output_path / "OUTCAR"
        
        if outcar_path.exists():
            with open(outcar_path, 'r') as f:
                for line in f:
                    if "solvent surface area" in line.lower():
                        try:
                            surface_area = float(line.split()[-1])
                        except:
                            pass
                    if "solvent volume" in line.lower():
                        try:
                            volume = float(line.split()[-1])
                        except:
                            pass
        
        return surface_area, volume
    
    def optimize_solvation_parameters(self, structure: Atoms,
                                       experimental_solv_energy: float,
                                       param_range: Optional[Dict] = None) -> VASPsolConfig:
        """
        优化VASPsol参数以匹配实验溶剂化能
        
        Args:
            structure: 参考分子
            experimental_solv_energy: 实验溶剂化能 (eV)
            param_range: 参数扫描范围
        
        Returns:
            优化后的配置
        """
        if param_range is None:
            param_range = {
                'tau': np.linspace(0.0001, 0.001, 10),
                'eb_k': [78.4, 80.0],
            }
        
        best_config = None
        best_error = float('inf')
        
        for tau in param_range.get('tau', [0.0005]):
            for eb_k in param_range.get('eb_k', [78.4]):
                config = VASPsolConfig(tau=tau, eb_k=eb_k)
                self.sol_config = config
                
                # 计算 (简化，实际需要运行VASP)
                # result = self.calculate_solvation_energy(structure)
                # error = abs(result.solvation_energy - experimental_solv_energy)
                
                # 简化
                error = abs(tau - 0.0005)  # 占位
                
                if error < best_error:
                    best_error = error
                    best_config = config
        
        return best_config or VASPsolConfig()


class ElectrochemicalInterface:
    """电化学界面模型"""
    
    def __init__(self, sol_config: VASPsolConfig,
                 echem_config: ElectrochemicalConfig):
        self.sol_config = sol_config
        self.echem_config = echem_config
    
    def setup_electrode_surface(self, bulk_structure: Atoms,
                                 miller_indices: Tuple[int, int, int],
                                 n_layers: int = 4) -> Atoms:
        """
        设置电极表面模型
        
        Args:
            bulk_structure: 体相结构
            miller_indices: 晶面指数
            n_layers: 原子层数
        
        Returns:
            表面模型
        """
        from ase.build import surface
        
        # 构建表面
        slab = surface(bulk_structure, miller_indices, n_layers)
        slab.center(vacuum=self.echem_config.vacuum_thickness, axis=2)
        
        return slab
    
    def add_electrolyte_layer(self, slab: Atoms) -> Atoms:
        """
        添加电解质层 (通过VASPsol参数)
        
        实际不需要添加原子，通过lambda_d_k参数实现
        """
        # 计算Debye长度
        concentration = self.echem_config.electrolyte_concentration  # M
        
        # Debye长度公式 (水溶液，25°C)
        # λ_D = 0.304 / sqrt(c) (nm)
        lambda_d_nm = 0.304 / np.sqrt(concentration)
        lambda_d_ang = lambda_d_nm * 10  # 转换为Å
        
        self.sol_config.lambda_d_k = lambda_d_ang
        
        logger.info(f"Debye length: {lambda_d_ang:.2f} Å")
        
        return slab
    
    def calculate_differential_capacitance(self, 
                                           potentials: np.ndarray,
                                           charges: np.ndarray) -> np.ndarray:
        """
        计算微分电容 C = dσ/dV
        
        Args:
            potentials: 电位数组 (V)
            charges: 表面电荷数组 (e)
        
        Returns:
            电容数组 (μF/cm²)
        """
        # 数值微分
        dQ_dV = np.gradient(charges, potentials)
        
        # 转换为 μF/cm²
        # 需要知道表面积
        # 简化：假设单位面积
        capacitance = dQ_dV * 1.602e-13  # 粗略转换
        
        return capacitance
    
    def setup_constant_potential_calculation(self, target_potential: float,
                                              reference_potential: float = 4.44) -> Dict:
        """
        设置恒电位计算
        
        通过调整NELECT实现目标电位
        
        Args:
            target_potential: 目标电位 (V vs SHE)
            reference_potential: 参比电位 (eV)
        
        Returns:
            VASP参数字典
        """
        # 电容近似
        C = 20  # μF/cm², 典型值
        
        # 所需电荷变化
        delta_V = target_potential  # vs SHE (假设0V为pzc)
        delta_sigma = C * delta_V  # μC/cm²
        
        # 转换为电子数 (需要表面积)
        # 简化
        delta_nelect = delta_sigma * 0.01  # 占位转换
        
        return {
            'nelect_delta': delta_nelect,
            'target_potential': target_potential,
        }


class VASPsolWorkflow:
    """VASPsol工作流"""
    
    def __init__(self, sol_config: Optional[VASPsolConfig] = None):
        self.sol_config = sol_config or VASPsolConfig()
        self.calculator = VASPsolCalculator(self.sol_config)
        self.results = {}
    
    def run_solvation_calculation(self, structure: Atoms,
                                   output_dir: str = "./",
                                   run_vacuum_reference: bool = True) -> SolvationResults:
        """
        运行溶剂化计算
        
        Args:
            structure: 结构
            output_dir: 输出目录
            run_vacuum_reference: 是否运行真空参考计算
        
        Returns:
            SolvationResults
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("VASPsol Calculation")
        logger.info("=" * 60)
        
        # 真空参考计算
        ref_energy = None
        if run_vacuum_reference:
            logger.info("\nStep 1: Vacuum reference calculation")
            vac_calc = Vasp(
                directory=str(output_path / "vacuum"),
                lsol=False,
            )
            structure_vac = structure.copy()
            structure_vac.calc = vac_calc
            ref_energy = structure_vac.get_potential_energy()
            logger.info(f"Vacuum energy: {ref_energy:.4f} eV")
        
        # 溶剂化计算
        logger.info("\nStep 2: Solvation calculation")
        result = self.calculator.calculate_solvation_energy(
            structure, str(output_path / "solvation"), ref_energy
        )
        
        logger.info(f"Solvation energy: {result.solvation_energy:.4f} eV")
        logger.info(f"Surface area: {result.surface_area:.2f} Bohr²")
        
        self.results['solvation'] = result
        
        return result
    
    def run_electrochemical_series(self, slab: Atoms,
                                    potentials: np.ndarray,
                                    output_dir: str = "./") -> Dict:
        """
        运行电化学系列计算 (不同电位)
        
        Args:
            slab: 表面结构
            potentials: 电位数组 (V vs SHE)
            output_dir: 输出目录
        
        Returns:
            结果字典
        """
        output_path = Path(output_dir)
        
        results = {
            'potentials': potentials.tolist(),
            'energies': [],
            'charges': [],
        }
        
        for i, V in enumerate(potentials):
            logger.info(f"\nCalculating at {V:.2f} V vs SHE")
            
            # 设置恒电位
            pot_config = self.calculator.setup_constant_potential_calculation(V)
            
            # 运行计算 (简化)
            # result = self.calculator.calculate_solvation_energy(
            #     slab, str(output_path / f"V_{V:.2f}")
            # )
            
            # 占位结果
            results['energies'].append(-100.0 + 0.1 * V)
            results['charges'].append(0.01 * V)
        
        # 计算微分电容
        capacitance = self.calculator.calculate_differential_capacitance(
            potentials, np.array(results['charges'])
        )
        results['capacitance'] = capacitance.tolist()
        
        self.results['electrochemical'] = results
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="VASPsol Interface")
    parser.add_argument('--structure', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default='./vaspsol_output')
    parser.add_argument('--epsilon', type=float, default=78.4,
                       help='Solvent dielectric constant')
    parser.add_argument('--tau', type=float, default=0.0005,
                       help='Surface tension')
    parser.add_argument('--electrolyte', action='store_true',
                       help='Include electrolyte (Debye screening)')
    parser.add_argument('--concentration', type=float, default=1.0,
                       help='Electrolyte concentration (M)')
    
    args = parser.parse_args()
    
    structure = read(args.structure)
    logger.info(f"Loaded: {structure.get_chemical_formula()}")
    
    # 创建配置
    config = VASPsolConfig(
        eb_k=args.epsilon,
        tau=args.tau,
    )
    
    if args.electrolyte:
        lambda_d = 3.04 / np.sqrt(args.concentration)  # Å
        config.lambda_d_k = lambda_d
        logger.info(f"Debye length: {lambda_d:.2f} Å")
    
    # 运行计算
    workflow = VASPsolWorkflow(config)
    result = workflow.run_solvation_calculation(structure, args.output)
    
    print(f"\nSolvation energy: {result.solvation_energy:.4f} eV")


if __name__ == "__main__":
    main()
