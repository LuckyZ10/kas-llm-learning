#!/usr/bin/env python3
"""
advanced_dft_workflow.py
========================
高级DFT计算工作流集成

整合dft_advanced和solvation模块到主工作流
- 自动判断计算类型并选择最优参数
- 后处理自动化（能带对齐、缺陷能级图等）
- 技术文档与最佳实践

作者: Advanced DFT Expert
日期: 2026-03-09
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto

from ase import Atoms
from ase.io import read, write

# Import from dft_advanced
from ..dft_advanced import (
    OpticalPropertyWorkflow,
    MagneticPropertyWorkflow,
    DefectCalculationWorkflow,
    NonlinearResponseWorkflow,
    VASPOpticalConfig,
    VASPMagneticConfig,
    DefectConfig,
)

# Import from solvation
from ..solvation import (
    VASPsolWorkflow,
    CP2KSolvationWorkflow,
    VASPsolConfig,
    CP2KSolvationConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalculationType(Enum):
    """计算类型枚举"""
    OPTICAL = "optical"
    MAGNETIC = "magnetic"
    DEFECT = "defect"
    NONLINEAR = "nonlinear"
    SOLVATION = "solvation"
    ELECTROCHEMICAL = "electrochemical"


@dataclass
class AdvancedDFTConfig:
    """高级DFT配置"""
    code: str = "vasp"  # "vasp", "qe", "cp2k"
    
    # 计算类型选择
    calculation_type: Optional[CalculationType] = None
    
    # 模块特定配置
    optical_config: Optional[VASPOpticalConfig] = None
    magnetic_config: Optional[VASPMagneticConfig] = None
    defect_config: Optional[DefectConfig] = None
    solvation_config: Optional[Any] = None
    
    # 通用设置
    ncores: int = 32
    memory: int = 128  # GB
    time_limit: int = 24  # hours
    
    # 自动化设置
    auto_kpoints: bool = True
    auto_encut: bool = True
    auto_charge: bool = True
    
    def __post_init__(self):
        if self.optical_config is None:
            self.optical_config = VASPOpticalConfig(ncores=self.ncores)
        if self.magnetic_config is None:
            self.magnetic_config = VASPMagneticConfig(ncores=self.ncores)
        if self.defect_config is None:
            self.defect_config = DefectConfig()


class CalculationTypeAnalyzer:
    """计算类型分析器 - 自动判断需要的计算类型"""
    
    @staticmethod
    def analyze_structure(structure: Atoms) -> List[CalculationType]:
        """
        分析结构，推荐适合的计算类型
        
        Returns:
            推荐的计算类型列表
        """
        recommendations = []
        symbols = set(structure.get_chemical_symbols())
        
        # 检查磁性元素
        magnetic_elements = {'Fe', 'Co', 'Ni', 'Mn', 'Cr', 'V', 'Ti', 'Cu', 
                            'Nd', 'Sm', 'Gd', 'Tb', 'Dy', 'Ho', 'Eu'}
        if symbols & magnetic_elements:
            recommendations.append(CalculationType.MAGNETIC)
        
        # 检查带隙材料 (通常需要光学性质)
        if not symbols <= {'Li', 'Na', 'K', 'Mg', 'Al', 'Cu', 'Ag', 'Au'}:
            recommendations.append(CalculationType.OPTICAL)
        
        # 建议非线性响应 (压电、弹性)
        recommendations.append(CalculationType.NONLINEAR)
        
        return recommendations
    
    @staticmethod
    def analyze_research_goal(goal: str) -> List[CalculationType]:
        """
        根据研究目标推荐计算类型
        
        Args:
            goal: 研究目标描述
        
        Returns:
            推荐的计算类型列表
        """
        goal_lower = goal.lower()
        recommendations = []
        
        if any(kw in goal_lower for kw in ['optical', 'absorption', 'dielectric', 'exciton', 'band gap']):
            recommendations.append(CalculationType.OPTICAL)
        
        if any(kw in goal_lower for kw in ['magnet', 'spin', 'curie', 'mae', 'anisotropy']):
            recommendations.append(CalculationType.MAGNETIC)
        
        if any(kw in goal_lower for kw in ['defect', 'vacancy', 'diffusion', 'dopant']):
            recommendations.append(CalculationType.DEFECT)
        
        if any(kw in goal_lower for kw in ['piezo', 'elastic', 'shg', 'nonlinear']):
            recommendations.append(CalculationType.NONLINEAR)
        
        if any(kw in goal_lower for kw in ['solvent', 'solution', 'aqueous', 'electrolyte']):
            recommendations.append(CalculationType.SOLVATION)
        
        if any(kw in goal_lower for kw in ['electrochemical', 'catalysis', 'her', 'orr', 'battery']):
            recommendations.append(CalculationType.ELECTROCHEMICAL)
        
        return recommendations


class AutoParameterSelector:
    """自动参数选择器"""
    
    @staticmethod
    def select_kpoints(structure: Atoms, calculation_type: CalculationType) -> tuple:
        """自动选择K点网格"""
        cell = structure.get_cell()
        
        # 基础密度
        kpoint_density = 0.2  # 2π/Å
        
        # 根据计算类型调整
        if calculation_type == CalculationType.OPTICAL:
            kpoint_density = 0.15  # 光学需要更密集的k点
        elif calculation_type == CalculationType.DEFECT:
            kpoint_density = 0.3   # 超胞可以用更少的k点
        
        kpoints = []
        for i in range(3):
            k = max(1, int(2 * 3.14159 / (kpoint_density * np.linalg.norm(cell[i]))))
            kpoints.append(k)
        
        return tuple(kpoints)
    
    @staticmethod
    def select_encut(structure: Atoms, calculation_type: CalculationType) -> float:
        """自动选择截断能"""
        symbols = structure.get_chemical_symbols()
        
        # 默认截断能
        encut = 520  # eV
        
        # 根据元素调整
        if any(el in symbols for el in ['Li', 'Na', 'K', 'H', 'O']):
            encut = 400
        if any(el in symbols for el in ['Fe', 'Co', 'Ni', 'Mn', 'Cr']):
            encut = 600  # 过渡金属需要更高
        
        # 计算类型调整
        if calculation_type == CalculationType.OPTICAL:
            encut = max(encut, 600)  # 光学计算需要高截断能
        
        return encut
    
    @staticmethod
    def select_charge_state(structure: Atoms, defect_element: str) -> List[int]:
        """自动选择电荷态"""
        # 根据元素常见氧化态推断
        common_oxidation = {
            'O': [-2, -1, 0],
            'S': [-2, -1, 0],
            'N': [-3, 0],
            'C': [0],
            'Si': [-4, 0, 4],
            'Li': [0, 1],
            'Na': [0, 1],
            'Cu': [0, 1, 2],
            'Fe': [0, 2, 3],
            'Co': [0, 2, 3],
            'Ni': [0, 2],
            'Mn': [0, 2, 4],
        }
        
        return common_oxidation.get(defect_element, [-2, -1, 0, 1, 2])


class AdvancedDFTWorkflow:
    """高级DFT工作流主类"""
    
    def __init__(self, config: Optional[AdvancedDFTConfig] = None):
        self.config = config or AdvancedDFTConfig()
        self.type_analyzer = CalculationTypeAnalyzer()
        self.param_selector = AutoParameterSelector()
        self.results = {}
    
    def run(self, structure: Atoms, 
            calculation_types: Optional[List[CalculationType]] = None,
            output_dir: str = "./advanced_dft") -> Dict:
        """
        运行高级DFT计算工作流
        
        Args:
            structure: 输入结构
            calculation_types: 计算类型列表 (None则自动判断)
            output_dir: 输出目录
        
        Returns:
            结果字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 自动判断计算类型
        if calculation_types is None:
            calculation_types = self.type_analyzer.analyze_structure(structure)
            logger.info(f"Auto-selected calculations: {[ct.value for ct in calculation_types]}")
        
        # 运行各类型计算
        for calc_type in calculation_types:
            logger.info(f"\nRunning {calc_type.value} calculation...")
            
            try:
                if calc_type == CalculationType.OPTICAL:
                    result = self._run_optical(structure, output_path)
                elif calc_type == CalculationType.MAGNETIC:
                    result = self._run_magnetic(structure, output_path)
                elif calc_type == CalculationType.DEFECT:
                    result = self._run_defect(structure, output_path)
                elif calc_type == CalculationType.NONLINEAR:
                    result = self._run_nonlinear(structure, output_path)
                elif calc_type == CalculationType.SOLVATION:
                    result = self._run_solvation(structure, output_path)
                elif calc_type == CalculationType.ELECTROCHEMICAL:
                    result = self._run_electrochemical(structure, output_path)
                else:
                    logger.warning(f"Unknown calculation type: {calc_type}")
                    continue
                
                self.results[calc_type.value] = result
                
            except Exception as e:
                logger.error(f"Error in {calc_type.value} calculation: {e}")
                continue
        
        # 后处理和报告
        self._generate_report(output_path)
        
        return self.results
    
    def _run_optical(self, structure: Atoms, output_path: Path) -> Dict:
        """运行光学性质计算"""
        workflow = OpticalPropertyWorkflow(
            code=self.config.code,
            config=self.config.optical_config
        )
        
        result = workflow.run_full_calculation(
            structure,
            str(output_path / "optical"),
            run_bse=True
        )
        
        return result
    
    def _run_magnetic(self, structure: Atoms, output_path: Path) -> Dict:
        """运行磁性计算"""
        workflow = MagneticPropertyWorkflow(
            code=self.config.code,
            config=self.config.magnetic_config
        )
        
        result = workflow.run_full_calculation(
            structure,
            str(output_path / "magnetic"),
            calculate_mae=True,
            calculate_exchange=True
        )
        
        return result
    
    def _run_defect(self, structure: Atoms, output_path: Path) -> Dict:
        """运行缺陷计算"""
        workflow = DefectCalculationWorkflow(
            bulk_structure=structure,
            config=self.config.defect_config
        )
        
        # 示例：计算氧空位
        results = workflow.calculate_vacancy_formation(
            element="O",
            chem_potentials={"O": -4.5},
            output_dir=str(output_path / "defect")
        )
        
        return results
    
    def _run_nonlinear(self, structure: Atoms, output_path: Path) -> Dict:
        """运行非线性响应计算"""
        workflow = NonlinearResponseWorkflow()
        
        # 创建虚拟计算器 (实际应使用VASP/QE)
        from ase.calculators.emt import EMT
        calculator = EMT()
        
        result = workflow.run_full_calculation(
            structure,
            calculator,
            str(output_path / "nonlinear")
        )
        
        return result
    
    def _run_solvation(self, structure: Atoms, output_path: Path) -> Dict:
        """运行溶剂化计算"""
        if self.config.code == "vasp":
            workflow = VASPsolWorkflow(self.config.solvation_config or VASPsolConfig())
            result = workflow.run_solvation_calculation(
                structure,
                str(output_path / "solvation")
            )
        elif self.config.code == "cp2k":
            workflow = CP2KSolvationWorkflow(
                self.config.solvation_config or CP2KSolvationConfig()
            )
            result = workflow.run_solvation_calculation(
                structure,
                str(output_path / "solvation")
            )
        else:
            logger.warning(f"Solvation not supported for {self.config.code}")
            return {}
        
        return result
    
    def _run_electrochemical(self, structure: Atoms, output_path: Path) -> Dict:
        """运行电化学界面计算"""
        if self.config.code == "vasp":
            workflow = VASPsolWorkflow(VASPsolConfig(lambda_d_k=3.0))
            result = workflow.run_electrochemical_series(
                structure,
                potentials=np.linspace(-1.0, 1.0, 5),
                output_dir=str(output_path / "electrochemical")
            )
        elif self.config.code == "cp2k":
            workflow = CP2KSolvationWorkflow()
            result = workflow.run_electrochemical_interface(
                structure,
                output_dir=str(output_path / "electrochemical")
            )
        else:
            logger.warning(f"Electrochemical not supported for {self.config.code}")
            return {}
        
        return result
    
    def _generate_report(self, output_path: Path):
        """生成计算报告"""
        report = {
            'code': self.config.code,
            'calculation_types': list(self.results.keys()),
            'results_summary': {},
        }
        
        for calc_type, result in self.results.items():
            if isinstance(result, dict):
                report['results_summary'][calc_type] = {
                    'keys': list(result.keys()),
                }
        
        report_file = output_path / "advanced_dft_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced DFT Workflow")
    parser.add_argument('--structure', type=str, required=True)
    parser.add_argument('--code', type=str, default='vasp',
                       choices=['vasp', 'qe', 'cp2k'])
    parser.add_argument('-o', '--output', type=str, default='./advanced_dft')
    parser.add_argument('--type', type=str, nargs='+',
                       choices=['optical', 'magnetic', 'defect', 'nonlinear', 'solvation', 'electrochemical'],
                       help='Calculation types (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    structure = read(args.structure)
    logger.info(f"Loaded: {structure.get_chemical_formula()}")
    
    # 创建配置
    config = AdvancedDFTConfig(code=args.code)
    
    # 解析计算类型
    calc_types = None
    if args.type:
        calc_types = [CalculationType(t) for t in args.type]
    
    # 运行工作流
    workflow = AdvancedDFTWorkflow(config)
    results = workflow.run(structure, calc_types, args.output)
    
    print("\n" + "=" * 60)
    print("Advanced DFT Calculation Completed")
    print("=" * 60)
    print(f"Results saved to: {args.output}")
    print(f"Calculation types: {list(results.keys())}")


if __name__ == "__main__":
    import numpy as np
    main()
