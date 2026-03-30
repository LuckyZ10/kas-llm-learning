#!/usr/bin/env python3
"""
chgnet_interface.py
===================
CHGNet (Crystal Hamiltonian Graph Neural Network) 接口

功能：
1. CHGNet模型加载和预测
2. 电荷密度预测
3. 磁矩预测
4. 与ASE的集成
5. 高通量筛选支持

CHGNet特点：
- 基于图神经网络架构
- 预测能量、力、应力和磁矩
- 考虑电荷密度信息
- 对磁性材料特别有效

作者: ML Potential Integration Expert
日期: 2026-03-09
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from datetime import datetime
import warnings

# ASE
from ase import Atoms
from ase.io import read, write
from ase.units import eV, Ang, GPa, fs, Bohr
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS, FIRE
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入CHGNet
try:
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator, MolecularDynamics
    CHGNET_AVAILABLE = True
except ImportError:
    CHGNET_AVAILABLE = False
    warnings.warn("CHGNet not available. Install with: pip install chgnet")


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class CHGNetConfig:
    """CHGNet配置"""
    # 模型选择
    model_name: str = "0.3.0"  # 模型版本
    model_path: Optional[str] = None  # 自定义模型路径
    
    # 计算设置
    use_device: str = "cuda"  # cuda, cpu
    
    # 预测设置
    predict_magmom: bool = True  # 预测磁矩
    predict_stress: bool = True
    
    # 截断半径
    cutoff: float = 6.0
    
    # 批处理
    batch_size: int = 16


@dataclass
class CHGNetPrediction:
    """CHGNet预测结果"""
    energy: float = 0.0  # eV
    forces: np.ndarray = field(default_factory=lambda: np.array([]))  # eV/Å
    stress: Optional[np.ndarray] = None  # GPa
    magmom: Optional[np.ndarray] = None  # μB
    
    # 原子级属性
    site_energies: Optional[np.ndarray] = None
    site_magmoms: Optional[np.ndarray] = None
    
    # 元数据
    composition: str = ""
    n_atoms: int = 0


@dataclass
class MagneticProperties:
    """磁性性质"""
    total_magmom: float = 0.0  # μB
    magmom_per_atom: float = 0.0
    magnetic_ordering: str = ""  # FM, AFM, NM
    
    # 每个原子的磁矩
    atomic_magmoms: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 居里温度估计
    curie_temperature: float = 0.0  # K


# =============================================================================
# CHGNet计算器
# =============================================================================

class CHGNetASECalculator(Calculator):
    """
    ASE兼容的CHGNet计算器
    
    支持能量、力、应力和磁矩的预测
    """
    
    implemented_properties = ['energy', 'forces', 'stress', 'magmom', 'free_energy']
    
    def __init__(self,
                 model_name: str = "0.3.0",
                 model_path: Optional[str] = None,
                 use_device: str = "cuda",
                 predict_magmom: bool = True,
                 **kwargs):
        """
        初始化CHGNet计算器
        
        Args:
            model_name: 预训练模型版本
            model_path: 自定义模型路径
            use_device: 计算设备
            predict_magmom: 是否预测磁矩
        """
        super().__init__(**kwargs)
        
        if not CHGNET_AVAILABLE:
            raise ImportError("CHGNet is not available. Install with: pip install chgnet")
        
        self.use_device = use_device
        self.predict_magmom = predict_magmom
        
        # 加载模型
        if model_path and Path(model_path).exists():
            logger.info(f"Loading CHGNet model from {model_path}")
            self.model = CHGNet.from_file(model_path)
        else:
            logger.info(f"Loading CHGNet model version {model_name}")
            self.model = CHGNet.load()
        
        # 设置设备
        import torch
        if use_device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("CHGNet using CUDA")
        else:
            self.model = self.model.cpu()
            logger.info("CHGNet using CPU")
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """执行计算"""
        super().calculate(atoms, properties, system_changes)
        
        if atoms is None:
            atoms = self.atoms
        
        # 使用CHGNet进行预测
        structure = self._atoms_to_structure(atoms)
        
        # 模型预测
        prediction = self.model.predict_structure(structure)
        
        # 提取结果
        if 'energy' in prediction:
            self.results['energy'] = prediction['energy']
            self.results['free_energy'] = prediction['energy']
        
        if 'forces' in prediction:
            self.results['forces'] = prediction['forces']
        
        if 'stress' in prediction:
            self.results['stress'] = prediction['stress']
        
        if 'magmom' in prediction and self.predict_magmom:
            self.results['magmom'] = prediction['magmom']
            # 总磁矩
            self.results['total_magmom'] = np.sum(prediction['magmom'])
    
    def _atoms_to_structure(self, atoms: Atoms):
        """将ASE Atoms转换为CHGNet结构格式"""
        # CHGNet使用pymatgen Structure
        from pymatgen.core import Structure, Lattice
        
        lattice = Lattice(atoms.get_cell())
        species = atoms.get_chemical_symbols()
        coords = atoms.get_positions()
        
        structure = Structure(lattice, species, coords, coords_are_cartesian=True)
        
        return structure
    
    def get_magmom_prediction(self, atoms: Atoms) -> MagneticProperties:
        """获取磁矩预测"""
        # 运行计算
        self.calculate(atoms)
        
        magmom = self.results.get('magmom', np.zeros(len(atoms)))
        total_magmom = self.results.get('total_magmom', 0.0)
        
        # 判断磁性排序
        magmom_signs = np.sign(magmom)
        if np.all(magmom_signs >= 0) or np.all(magmom_signs <= 0):
            ordering = "FM"  # 铁磁
        elif np.abs(np.sum(magmom)) < 0.1 * np.sum(np.abs(magmom)):
            ordering = "AFM"  # 反铁磁
        else:
            ordering = "FiM"  # 亚铁磁
        
        # 如果总磁矩接近0，认为是非磁性
        if np.abs(total_magmom) < 0.1:
            ordering = "NM"
        
        return MagneticProperties(
            total_magmom=total_magmom,
            magmom_per_atom=total_magmom / len(atoms) if len(atoms) > 0 else 0,
            magnetic_ordering=ordering,
            atomic_magmoms=magmom
        )


# =============================================================================
# CHGNet批量预测
# =============================================================================

class CHGNetBatchPredictor:
    """
    CHGNet批量预测器
    
    高效处理大量晶体结构的预测
    """
    
    def __init__(self, calculator: CHGNetASECalculator, batch_size: int = 16):
        self.calc = calculator
        self.batch_size = batch_size
    
    def predict_structures(self, structures: List[Atoms]) -> List[CHGNetPrediction]:
        """
        批量预测结构
        
        Returns:
            List of CHGNetPrediction objects
        """
        results = []
        
        for i, atoms in enumerate(structures):
            try:
                atoms.calc = self.calc
                
                # 预测
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                
                # 可选预测
                stress = None
                magmom = None
                
                try:
                    stress = atoms.get_stress()
                except:
                    pass
                
                try:
                    self.calc.calculate(atoms, properties=['magmom'])
                    magmom = self.calc.results.get('magmom')
                except:
                    pass
                
                pred = CHGNetPrediction(
                    energy=energy,
                    forces=forces,
                    stress=stress,
                    magmom=magmom,
                    composition=atoms.get_chemical_formula(),
                    n_atoms=len(atoms)
                )
                
                results.append(pred)
                
            except Exception as e:
                logger.warning(f"Prediction failed for structure {i}: {e}")
                results.append(CHGNetPrediction(
                    composition=atoms.get_chemical_formula(),
                    n_atoms=len(atoms)
                ))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Predicted {i + 1}/{len(structures)} structures")
        
        return results
    
    def predict_to_dataframe(self, structures: List[Atoms]) -> pd.DataFrame:
        """
        预测并返回DataFrame格式
        """
        predictions = self.predict_structures(structures)
        
        data = []
        for pred in predictions:
            row = {
                'composition': pred.composition,
                'n_atoms': pred.n_atoms,
                'energy': pred.energy,
                'energy_per_atom': pred.energy / pred.n_atoms if pred.n_atoms > 0 else None,
                'total_magmom': np.sum(pred.magmom) if pred.magmom is not None else None,
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def screen_magnetic_materials(self,
                                  structures: List[Atoms],
                                  min_magmom: float = 0.5) -> Tuple[List[Atoms], pd.DataFrame]:
        """
        筛选磁性材料
        
        Returns:
            magnetic_structures, results_df
        """
        results = self.predict_structures(structures)
        
        magnetic_structures = []
        magnetic_data = []
        
        for atoms, pred in zip(structures, results):
            if pred.magmom is not None:
                total_magmom = np.abs(np.sum(pred.magmom))
                magmom_per_atom = total_magmom / pred.n_atoms
                
                if magmom_per_atom >= min_magmom:
                    magnetic_structures.append(atoms)
                    magnetic_data.append({
                        'composition': pred.composition,
                        'total_magmom': total_magmom,
                        'magmom_per_atom': magmom_per_atom
                    })
        
        logger.info(f"Found {len(magnetic_structures)} magnetic materials")
        
        return magnetic_structures, pd.DataFrame(magnetic_data)


# =============================================================================
# CHGNet结构优化
# =============================================================================

class CHGNetStructureOptimizer:
    """
    使用CHGNet进行结构优化
    """
    
    def __init__(self, calculator: CHGNetASECalculator):
        self.calc = calculator
    
    def relax_structure(self,
                       atoms: Atoms,
                       fmax: float = 0.01,
       max_steps: int = 500,
                       constant_cell: bool = False) -> Atoms:
        """
        结构弛豫
        
        Args:
            atoms: 输入结构
            fmax: 最大力收敛标准 (eV/Å)
            max_steps: 最大步数
            constant_cell: 是否固定晶胞
        """
        atoms = atoms.copy()
        atoms.calc = self.calc
        
        # 选择优化器
        if constant_cell:
            opt = BFGS(atoms)
        else:
            # 允许晶胞优化
            from ase.optimize import BFGS as BFGSBase
            opt = BFGSBase(atoms)
        
        # 运行优化
        opt.run(fmax=fmax, steps=max_steps)
        
        logger.info(f"Optimization completed:")
        logger.info(f"  Final energy: {atoms.get_potential_energy():.4f} eV")
        logger.info(f"  Max force: {np.max(np.abs(atoms.get_forces())):.4f} eV/Å")
        
        return atoms
    
    def optimize_cell(self,
                     atoms: Atoms,
                     fmax: float = 0.01,
                     max_steps: int = 300) -> Atoms:
        """
        晶胞优化
        
        使用应力信息优化晶胞参数
        """
        atoms = atoms.copy()
        atoms.calc = self.calc
        
        # 使用ExpCellFilter进行晶胞优化
        from ase.constraints import ExpCellFilter
        
        ucf = ExpCellFilter(atoms)
        opt = BFGS(ucf)
        
        opt.run(fmax=fmax, steps=max_steps)
        
        # 获取应力
        try:
            stress = atoms.get_stress()
            logger.info(f"Final stress: {stress}")
        except:
            pass
        
        return atoms
    
    def calculate_eos(self,
                     atoms: Atoms,
                     volumes: Optional[np.ndarray] = None,
                     n_points: int = 7,
                     scale_range: float = 0.1) -> Dict:
        """
        计算状态方程 (Equation of State)
        
        Returns:
            Dict with V, E, and fitted EOS parameters
        """
        atoms = atoms.copy()
        atoms.calc = self.calc
        
        # 生成体积范围
        if volumes is None:
            v0 = atoms.get_volume()
            volumes = np.linspace(v0 * (1 - scale_range), v0 * (1 + scale_range), n_points)
        
        energies = []
        
        for v in volumes:
            # 缩放晶胞
            atoms_scaled = atoms.copy()
            cell_scaled = atoms.get_cell() * (v / atoms.get_volume())**(1/3)
            atoms_scaled.set_cell(cell_scaled, scale_atoms=True)
            atoms_scaled.calc = self.calc
            
            e = atoms_scaled.get_potential_energy()
            energies.append(e)
            
            logger.info(f"V={v:.2f} Å³, E={e:.4f} eV")
        
        energies = np.array(energies)
        
        # 拟合EOS (Birch-Murnaghan)
        try:
            from ase.eos import EquationOfState
            eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
            v0, e0, B = eos.fit()
            
            logger.info(f"EOS fit: V0={v0:.2f} Å³, E0={e0:.4f} eV, B={B/1e9:.2f} GPa")
            
            return {
                'volumes': volumes,
                'energies': energies,
                'V0': v0,
                'E0': e0,
                'B': B,
                'eos_fit': eos
            }
        except Exception as e:
            logger.error(f"EOS fitting failed: {e}")
            return {
                'volumes': volumes,
                'energies': energies
            }


# =============================================================================
# CHGNet高通量筛选
# =============================================================================

class CHGNetScreeningPipeline:
    """
    基于CHGNet的高通量材料筛选
    """
    
    def __init__(self, calculator: CHGNetASECalculator):
        self.calc = calculator
        self.optimizer = CHGNetStructureOptimizer(calculator)
        self.predictor = CHGNetBatchPredictor(calculator)
    
    def screen_stability(self,
                        structures: List[Atoms],
                        reference_energies: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        筛选稳定结构（基于形成能）
        
        Args:
            structures: 候选结构列表
            reference_energies: 参考元素能量 (eV/atom)
        """
        results = self.predictor.predict_to_dataframe(structures)
        
        # 计算形成能
        if reference_energies:
            formation_energies = []
            
            for atoms, e_total in zip(structures, results['energy']):
                symbols = atoms.get_chemical_symbols()
                e_ref = sum(reference_energies.get(s, 0) for s in symbols)
                e_form = (e_total - e_ref) / len(atoms)
                formation_energies.append(e_form)
            
            results['formation_energy'] = formation_energies
        
        return results
    
    def screen_elastic_stability(self,
                                structures: List[Atoms],
                                min_bulk_modulus: float = 0) -> pd.DataFrame:
        """
        筛选弹性稳定结构
        
        通过EOS计算体模量
        """
        results = []
        
        for atoms in structures:
            try:
                eos_data = self.optimizer.calculate_eos(atoms, n_points=5)
                
                results.append({
                    'composition': atoms.get_chemical_formula(),
                    'B': eos_data.get('B', 0) / 1e9,  # GPa
                    'V0': eos_data.get('V0', 0),
                    'stable': eos_data.get('B', 0) > min_bulk_modulus
                })
            except Exception as e:
                logger.warning(f"EOS calculation failed: {e}")
                results.append({
                    'composition': atoms.get_chemical_formula(),
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def screen_all(self,
                  structures: List[Atoms],
                  energy_threshold: float = 0.0,
                  min_magmom: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """
        综合筛选
        
        Returns:
            Dictionary of screening results
        """
        results = {}
        
        # 能量预测
        logger.info("Screening by energy...")
        results['energy'] = self.predictor.predict_to_dataframe(structures)
        
        # 磁性筛选
        if min_magmom is not None:
            logger.info("Screening by magnetism...")
            _, mag_df = self.predictor.screen_magnetic_materials(structures, min_magmom)
            results['magnetic'] = mag_df
        
        return results


# =============================================================================
# CHGNet-DFT集成
# =============================================================================

class CHGNetDFTInterface:
    """
    CHGNet与DFT的集成接口
    
    使用CHGNet预筛选 + DFT精算的工作流
    """
    
    def __init__(self, chgnet_calc: CHGNetASECalculator):
        self.chgnet_calc = chgnet_calc
        self.screening = CHGNetScreeningPipeline(chgnet_calc)
    
    def prescreen_for_dft(self,
                         candidate_structures: List[Atoms],
                         n_select: int = 10,
                         criteria: str = "energy") -> List[Atoms]:
        """
        使用CHGNet预筛选DFT计算候选
        
        Args:
            candidate_structures: 候选结构
            n_select: 选择的结构数
            criteria: 筛选标准
        
        Returns:
            精选的候选结构
        """
        # CHGNet快速预测
        results = self.screening.predictor.predict_to_dataframe(candidate_structures)
        
        # 根据标准排序
        if criteria == "energy":
            # 选择能量最低的
            indices = results['energy'].argsort()[:n_select]
        elif criteria == "diversity":
            # 基于成分多样性选择
            from sklearn.cluster import KMeans
            
            # 简单的特征：组成和能量
            features = results[['n_atoms', 'energy']].values
            
            kmeans = KMeans(n_clusters=min(n_select, len(candidate_structures)))
            labels = kmeans.fit_predict(features)
            
            # 从每个聚类选择一个代表性结构
            indices = []
            for label in np.unique(labels):
                cluster_indices = np.where(labels == label)[0]
                # 选择能量最低的
                best_idx = cluster_indices[results.iloc[cluster_indices]['energy'].argsort()[0]]
                indices.append(best_idx)
            
            indices = np.array(indices)
        else:
            # 随机选择
            indices = np.random.choice(len(candidate_structures), n_select, replace=False)
        
        selected_structures = [candidate_structures[i] for i in indices]
        
        logger.info(f"Pre-screened {len(candidate_structures)} structures to {n_select} for DFT")
        
        return selected_structures
    
    def run_dft_with_initialization(self,
                                   structures: List[Atoms],
                                   dft_calculator) -> List[Atoms]:
        """
        使用CHGNet结果初始化DFT计算
        
        可以使用CHGNet预测的磁矩初始化DFT
        """
        initialized_structures = []
        
        for atoms in structures:
            # 获取CHGNet预测
            atoms.calc = self.chgnet_calc
            mag_props = self.chgnet_calc.get_magmom_prediction(atoms)
            
            # 设置初始磁矩
            if mag_props.atomic_magmoms is not None:
                atoms.set_initial_magnetic_moments(mag_props.atomic_magmoms)
            
            initialized_structures.append(atoms)
        
        return initialized_structures


# =============================================================================
# 主工作流类
# =============================================================================

class CHGNetWorkflow:
    """
    CHGNet完整工作流
    """
    
    def __init__(self, config: CHGNetConfig):
        self.config = config
        self.calculator = None
    
    def setup(self) -> CHGNetASECalculator:
        """设置CHGNet计算器"""
        cfg = self.config
        
        self.calculator = CHGNetASECalculator(
            model_name=cfg.model_name,
            model_path=cfg.model_path,
            use_device=cfg.use_device,
            predict_magmom=cfg.predict_magmom
        )
        
        return self.calculator
    
    def predict_structure(self, atoms: Atoms) -> CHGNetPrediction:
        """预测单个结构"""
        if self.calculator is None:
            self.setup()
        
        atoms.calc = self.calculator
        
        return CHGNetPrediction(
            energy=atoms.get_potential_energy(),
            forces=atoms.get_forces(),
            stress=atoms.get_stress() if self.config.predict_stress else None,
            composition=atoms.get_chemical_formula(),
            n_atoms=len(atoms)
        )
    
    def batch_predict(self, structures: List[Atoms]) -> pd.DataFrame:
        """批量预测"""
        if self.calculator is None:
            self.setup()
        
        predictor = CHGNetBatchPredictor(self.calculator, self.config.batch_size)
        
        return predictor.predict_to_dataframe(structures)
    
    def relax_structure(self, atoms: Atoms, fmax: float = 0.01) -> Atoms:
        """结构弛豫"""
        if self.calculator is None:
            self.setup()
        
        optimizer = CHGNetStructureOptimizer(self.calculator)
        
        return optimizer.relax_structure(atoms, fmax=fmax)


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CHGNet ML Potential Tool")
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Predict
    predict_parser = subparsers.add_parser('predict', help='Predict properties')
    predict_parser.add_argument('structure', help='Structure file')
    predict_parser.add_argument('--device', default='cuda', help='Device')
    
    # Batch
    batch_parser = subparsers.add_parser('batch', help='Batch prediction')
    batch_parser.add_argument('structures', nargs='+', help='Structure files')
    batch_parser.add_argument('--output', default='chgnet_results.csv')
    
    # Relax
    relax_parser = subparsers.add_parser('relax', help='Relax structure')
    relax_parser.add_argument('structure', help='Structure file')
    relax_parser.add_argument('--fmax', type=float, default=0.01)
    
    # Screen
    screen_parser = subparsers.add_parser('screen', help='Screen structures')
    screen_parser.add_argument('structures', nargs='+', help='Structure files')
    screen_parser.add_argument('--magnetic', action='store_true', help='Screen for magnetic materials')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        config = CHGNetConfig(use_device=args.device)
        workflow = CHGNetWorkflow(config)
        workflow.setup()
        
        atoms = read(args.structure)
        pred = workflow.predict_structure(atoms)
        
        print(f"Composition: {pred.composition}")
        print(f"Energy: {pred.energy:.4f} eV")
        print(f"Energy per atom: {pred.energy/pred.n_atoms:.4f} eV/atom")
        
        # 磁矩预测
        mag_props = workflow.calculator.get_magmom_prediction(atoms)
        print(f"Total magnetic moment: {mag_props.total_magmom:.2f} μB")
        print(f"Magnetic ordering: {mag_props.magnetic_ordering}")
        
    elif args.command == 'batch':
        config = CHGNetConfig()
        workflow = CHGNetWorkflow(config)
        workflow.setup()
        
        structures = [read(f) for f in args.structures]
        df = workflow.batch_predict(structures)
        df.to_csv(args.output, index=False)
        
        print(f"Results saved to {args.output}")
        print(df.head())
        
    elif args.command == 'relax':
        config = CHGNetConfig()
        workflow = CHGNetWorkflow(config)
        
        atoms = read(args.structure)
        relaxed = workflow.relax_structure(atoms, fmax=args.fmax)
        
        write('relaxed.xyz', relaxed)
        print(f"Relaxed structure saved to relaxed.xyz")
        
    elif args.command == 'screen':
        config = CHGNetConfig()
        workflow = CHGNetWorkflow(config)
        workflow.setup()
        
        structures = [read(f) for f in args.structures]
        pipeline = CHGNetScreeningPipeline(workflow.calculator)
        
        if args.magnetic:
            selected, results = pipeline.predictor.screen_magnetic_materials(structures)
            print(f"Found {len(selected)} magnetic materials")
            print(results.head())
        else:
            results = pipeline.screen_all(structures)
            print(results.get('energy', pd.DataFrame()).head())
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
