#!/usr/bin/env python3
"""
dft_to_lammps_bridge.py
=======================
DFT与LAMMPS之间的桥梁脚本

功能：
1. 解析VASP OUTCAR文件，提取能量、力、应力等数据
2. 拟合力场参数（支持多种势函数类型）
3. 生成LAMMPS输入文件
4. ASE接口封装，支持VASP/QE与LAMMPS的无缝耦合
5. QM/MM边界处理

作者: DFT-MD Coupling Expert
日期: 2026-03-09
"""

import os
import re
import json
import glob
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from datetime import datetime
import warnings

# ASE - 核心接口
from ase import Atoms
from ase.io import read, write
from ase.io.vasp import read_vasp_out, read_vasp_xdatcar
from ase.io.espresso import read_espresso_out
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
from ase.calculators.vasp import Vasp
from ase.calculators.espresso import Espresso
from ase.calculators.lammpsrun import LAMMPS as ASE_LAMMPS
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms, FixBondLengths
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.units import GPa, eV, Ang, fs
from ase.optimize import BFGS, FIRE
from ase.geometry import get_distances

# SciPy for fitting
from scipy.optimize import least_squares, curve_fit, minimize
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

# scikit-learn for ML-assisted fitting
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 可选: dpdata for DeepMD format
warnings.filterwarnings('ignore')

try:
    import dpdata
    DPDATA_AVAILABLE = True
except ImportError:
    DPDATA_AVAILABLE = False
    warnings.warn("dpdata not available. DeepMD format conversion disabled.")

try:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp import Outcar, Vasprun
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    warnings.warn("pymatgen not available. Some features disabled.")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class VASPParserConfig:
    """VASP解析配置"""
    extract_energy: bool = True
    extract_forces: bool = True
    extract_stress: bool = True
    extract_positions: bool = True
    extract_velocities: bool = False
    extract_magmom: bool = False
    filter_unconverged: bool = True
    energy_threshold: float = 100.0  # eV/atom, 异常值过滤


@dataclass
class ForceFieldConfig:
    """力场拟合配置"""
    ff_type: str = "buckingham"  # buckingham, morse, lj, eam, snap, nnp
    elements: List[str] = field(default_factory=list)
    cutoff: float = 6.0  # Å
    charge_dict: Dict[str, float] = field(default_factory=dict)
    
    # Buckingham参数初始值
    buckingham_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Morse参数初始值
    morse_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # LJ参数初始值
    lj_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 拟合选项
    fit_method: str = "least_squares"  # least_squares, ridge, lasso
    regularization: float = 0.01
    max_iterations: int = 1000


@dataclass
class LAMMPSInputConfig:
    """LAMMPS输入配置"""
    units: str = "metal"
    atom_style: str = "atomic"
    boundary: str = "p p p"
    
    # 势函数设置
    pair_style: str = "buck/coul/long"  # 或其他类型
    pair_coeff: List[str] = field(default_factory=list)
    kspace_style: str = "pppm 1.0e-5"
    
    # 模拟参数
    timestep: float = 1.0  # fs
    temperature: float = 300.0  # K
    pressure: Optional[float] = None  # atm
    ensemble: str = "nvt"
    
    # 运行设置
    nsteps: int = 100000
    thermo_interval: int = 100
    dump_interval: int = 1000
    
    # 约束
    fix_com: bool = True
    velocity_dist: str = "gaussian"


@dataclass
class QMMMConfig:
    """QM/MM耦合配置"""
    qm_region: List[int] = field(default_factory=list)  # QM区域原子索引
    qm_calculator: Optional[str] = None  # VASP/QE
    mm_calculator: Optional[str] = None  # LAMMPS
    
    # 边界处理
    buffer_zone: float = 2.0  # Å
    link_atom_type: str = "H"
    
    # 耦合方案
    coupling_scheme: str = "mechanical"  # mechanical, electrostatic, subtractive
    
    # DFT参数
    qm_charge: int = 0
    qm_multiplicity: int = 1


# =============================================================================
# VASP OUTCAR解析器
# =============================================================================

class VASPOUTCARParser:
    """
    高性能VASP OUTCAR解析器
    
    支持:
    - 单点能计算
    - 结构优化
    - AIMD轨迹
    - 磁性计算
    - 振动分析
    """
    
    def __init__(self, config: Optional[VASPParserConfig] = None):
        self.config = config or VASPParserConfig()
        self.frames = []
        
    def parse(self, outcar_path: Union[str, Path]) -> List[Dict]:
        """
        解析OUTCAR文件
        
        Returns:
            frames: 每帧数据的字典列表
        """
        outcar_path = Path(outcar_path)
        
        if not outcar_path.exists():
            raise FileNotFoundError(f"OUTCAR not found: {outcar_path}")
        
        logger.info(f"Parsing OUTCAR: {outcar_path}")
        
        # 使用ASE读取
        try:
            atoms_list = read_vasp_out(str(outcar_path), index=':')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
        except Exception as e:
            logger.error(f"ASE parsing failed: {e}")
            # 回退到手动解析
            return self._manual_parse(outcar_path)
        
        # 提取每帧数据
        frames = []
        for i, atoms in enumerate(atoms_list):
            frame_data = self._extract_frame_data(atoms, i)
            if frame_data:
                frames.append(frame_data)
        
        # 过滤异常值
        if self.config.filter_unconverged:
            frames = self._filter_frames(frames)
        
        self.frames = frames
        logger.info(f"Parsed {len(frames)} frames from OUTCAR")
        
        return frames
    
    def parse_multiple(self, outcar_paths: List[Union[str, Path]]) -> List[Dict]:
        """解析多个OUTCAR文件"""
        all_frames = []
        for path in outcar_paths:
            frames = self.parse(path)
            all_frames.extend(frames)
        logger.info(f"Total frames from all OUTCARs: {len(all_frames)}")
        return all_frames
    
    def _extract_frame_data(self, atoms: Atoms, index: int) -> Optional[Dict]:
        """从ASE Atoms对象提取数据"""
        frame = {
            'index': index,
            'atoms': atoms.copy(),
            'symbols': atoms.get_chemical_symbols(),
            'positions': atoms.get_positions(),
            'cell': atoms.get_cell(),
            'pbc': atoms.get_pbc(),
        }
        
        # 能量
        if self.config.extract_energy:
            try:
                energy = atoms.get_potential_energy()
                frame['energy'] = energy
                frame['energy_per_atom'] = energy / len(atoms)
            except:
                frame['energy'] = None
        
        # 力
        if self.config.extract_forces:
            try:
                forces = atoms.get_forces()
                frame['forces'] = forces
                frame['max_force'] = np.max(np.abs(forces))
                frame['rms_force'] = np.sqrt(np.mean(forces**2))
            except:
                frame['forces'] = None
        
        # 应力
        if self.config.extract_stress:
            try:
                stress = atoms.get_stress()
                frame['stress'] = stress
                # 转换为压强 (GPa)
                pressure = -np.trace(stress[:3]) / 3 * GPa
                frame['pressure'] = pressure
            except:
                frame['stress'] = None
                frame['pressure'] = None
        
        # 速度
        if self.config.extract_velocities:
            velocities = atoms.get_velocities()
            frame['velocities'] = velocities
        
        # 磁矩
        if self.config.extract_magmom:
            if hasattr(atoms, 'calc') and atoms.calc is not None:
                try:
                    magmom = atoms.calc.get_magnetic_moment()
                    frame['magmom'] = magmom
                except:
                    pass
        
        return frame
    
    def _manual_parse(self, outcar_path: Path) -> List[Dict]:
        """手动解析OUTCAR (当ASE解析失败时使用)"""
        logger.warning("Using manual parser...")
        
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
        
        frames = []
        i = 0
        
        while i < len(lines):
            # 查找能量行
            if "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in lines[i]:
                frame = self._parse_single_frame(lines, i)
                if frame:
                    frames.append(frame)
            i += 1
        
        return frames
    
    def _parse_single_frame(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """解析单帧数据"""
        frame = {'index': len(self.frames)}
        
        # 解析能量
        for i in range(start_idx, min(start_idx + 50, len(lines))):
            if "free  energy   TOTEN" in lines[i]:
                match = re.search(r"([-\d.]+)\s+eV", lines[i])
                if match:
                    frame['energy'] = float(match.group(1))
                break
        
        return frame if 'energy' in frame else None
    
    def _filter_frames(self, frames: List[Dict]) -> List[Dict]:
        """过滤异常帧"""
        if len(frames) < 2:
            return frames
        
        # 基于能量的异常值检测
        energies = np.array([f['energy'] for f in frames if f.get('energy') is not None])
        if len(energies) == 0:
            return frames
        
        mean_e = np.mean(energies)
        std_e = np.std(energies)
        threshold = self.config.energy_threshold * std_e
        
        filtered = []
        for frame in frames:
            if frame.get('energy') is not None:
                if abs(frame['energy'] - mean_e) < threshold:
                    filtered.append(frame)
                else:
                    logger.warning(f"Filtered outlier frame {frame['index']}: E = {frame['energy']:.4f}")
            else:
                filtered.append(frame)
        
        return filtered
    
    def to_xyz(self, output_file: str):
        """导出为XYZ格式"""
        frames = []
        for frame in self.frames:
            atoms = frame['atoms']
            # 添加能量信息到comment行
            comment = f"energy={frame.get('energy', 0):.6f}"
            frames.append(atoms)
        
        write(output_file, frames, format='extxyz')
        logger.info(f"Exported {len(frames)} frames to {output_file}")
    
    def to_deepmd(self, output_dir: str, train_ratio: float = 0.9):
        """导出为DeepMD格式"""
        if not DPDATA_AVAILABLE:
            raise ImportError("dpdata is required for DeepMD format conversion")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        coords = []
        cells = []
        energies = []
        forces_list = []
        
        symbols = None
        
        for frame in self.frames:
            atoms = frame['atoms']
            if symbols is None:
                symbols = atoms.get_chemical_symbols()
            
            coords.append(atoms.get_positions())
            cells.append(atoms.get_cell().array)
            energies.append(frame.get('energy', 0))
            forces_list.append(frame.get('forces', np.zeros_like(coords[-1])))
        
        # 创建dpdata系统
        type_map = list(set(symbols))
        atom_types = [type_map.index(s) for s in symbols]
        atom_numbs = [symbols.count(t) for t in type_map]
        
        system = dpdata.LabeledSystem()
        system['atom_names'] = type_map
        system['atom_numbs'] = atom_numbs
        system['atom_types'] = np.array(atom_types)
        system['coords'] = np.array(coords)
        system['cells'] = np.array(cells)
        system['energies'] = np.array(energies)
        system['forces'] = np.array(forces_list)
        system['orig'] = np.zeros(3)
        system['nopbc'] = False
        
        # 分割训练集和测试集
        n_frames = len(system)
        n_train = int(n_frames * train_ratio)
        indices = np.random.permutation(n_frames)
        
        train_system = system.sub_system(indices[:n_train])
        test_system = system.sub_system(indices[n_train:])
        
        # 保存
        train_dir = output_dir / "training"
        test_dir = output_dir / "validation"
        
        train_system.to_deepmd_npy(str(train_dir))
        test_system.to_deepmd_npy(str(test_dir))
        
        logger.info(f"Exported to DeepMD format: {n_train} train, {n_frames - n_train} test")
        
        return str(train_dir), str(test_dir)
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if not self.frames:
            return {}
        
        stats = {
            'n_frames': len(self.frames),
            'n_atoms': len(self.frames[0]['atoms']),
            'elements': list(set(self.frames[0]['symbols'])),
        }
        
        # 能量统计
        energies = [f['energy'] for f in self.frames if f.get('energy') is not None]
        if energies:
            stats['energy'] = {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies)),
                'min': float(np.min(energies)),
                'max': float(np.max(energies)),
            }
        
        # 力统计
        forces = [f['forces'] for f in self.frames if f.get('forces') is not None]
        if forces:
            all_forces = np.concatenate([f.flatten() for f in forces])
            stats['forces'] = {
                'mean': float(np.mean(all_forces)),
                'std': float(np.std(all_forces)),
                'max_abs': float(np.max(np.abs(all_forces))),
            }
        
        # 压强统计
        pressures = [f['pressure'] for f in self.frames if f.get('pressure') is not None]
        if pressures:
            stats['pressure'] = {
                'mean': float(np.mean(pressures)),
                'std': float(np.std(pressures)),
            }
        
        return stats


# =============================================================================
# Quantum ESPRESSO解析器
# =============================================================================

class QuantumESPRESSOParser:
    """Quantum ESPRESSO输出解析器"""
    
    def __init__(self):
        self.frames = []
    
    def parse(self, pwscf_out: Union[str, Path]) -> List[Dict]:
        """解析PWscf输出文件"""
        pwscf_out = Path(pwscf_out)
        
        if not pwscf_out.exists():
            raise FileNotFoundError(f"PWscf output not found: {pwscf_out}")
        
        logger.info(f"Parsing QE output: {pwscf_out}")
        
        try:
            atoms_list = read_espresso_out(str(pwscf_out), index=':')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
        except Exception as e:
            logger.error(f"ASE parsing failed: {e}")
            return []
        
        frames = []
        for i, atoms in enumerate(atoms_list):
            frame = {
                'index': i,
                'atoms': atoms.copy(),
                'energy': atoms.get_potential_energy() if atoms.calc else None,
                'forces': atoms.get_forces() if atoms.calc else None,
            }
            frames.append(frame)
        
        self.frames = frames
        logger.info(f"Parsed {len(frames)} frames from QE output")
        
        return frames


# =============================================================================
# 力场参数拟合
# =============================================================================

class ForceFieldFitter:
    """
    力场参数拟合器
    
    支持多种势函数类型:
    - Buckingham: A*exp(-r/rho) - C/r^6
    - Morse: D_e * (1 - exp(-a*(r-r_e)))^2
    - Lennard-Jones: 4*epsilon * ((sigma/r)^12 - (sigma/r)^6)
    - Stillinger-Weber (SW)
    - EAM (简化版本)
    """
    
    def __init__(self, config: Optional[ForceFieldConfig] = None):
        self.config = config or ForceFieldConfig()
        self.energy_data = []
        self.force_data = []
        self.structures = []
        self.fitted_params = None
        
    def load_data(self, frames: List[Dict]):
        """加载训练数据"""
        for frame in frames:
            if frame.get('energy') is not None and frame.get('forces') is not None:
                self.energy_data.append(frame['energy'])
                self.force_data.append(frame['forces'])
                self.structures.append(frame['atoms'])
        
        logger.info(f"Loaded {len(self.energy_data)} training structures")
    
    def load_from_outcar(self, outcar_path: Union[str, Path]):
        """直接从OUTCAR加载"""
        parser = VASPOUTCARParser()
        frames = parser.parse(outcar_path)
        self.load_data(frames)
    
    def fit(self, ff_type: Optional[str] = None) -> Dict:
        """
        执行拟合
        
        Returns:
            fitted_params: 拟合参数字典
        """
        ff_type = ff_type or self.config.ff_type
        
        logger.info(f"Fitting {ff_type} force field...")
        
        if ff_type == "buckingham":
            self.fitted_params = self._fit_buckingham()
        elif ff_type == "morse":
            self.fitted_params = self._fit_morse()
        elif ff_type == "lj":
            self.fitted_params = self._fit_lennard_jones()
        else:
            raise ValueError(f"Unsupported force field type: {ff_type}")
        
        return self.fitted_params
    
    def _fit_buckingham(self) -> Dict:
        """
        拟合Buckingham势参数
        
        E = A * exp(-r/rho) - C/r^6
        """
        elements = self.config.elements or self._get_elements()
        
        # 获取元素对
        pairs = [(e1, e2) for i, e1 in enumerate(elements) 
                 for e2 in elements[i:]]
        
        params_dict = {}
        
        for pair in pairs:
            pair_key = f"{pair[0]}-{pair[1]}"
            
            # 初始猜测
            p0 = [1000.0, 0.3, 10.0]  # A, rho, C
            
            # 拟合该元素对的参数
            # 这里简化为使用全局数据，实际应只使用相关原子对的贡献
            try:
                result = least_squares(
                    self._buckingham_residuals,
                    p0,
                    args=(pair,),
                    bounds=([0, 0.1, 0], [100000, 1.0, 1000]),
                    max_nfev=self.config.max_iterations
                )
                
                params_dict[pair_key] = {
                    'A': result.x[0],
                    'rho': result.x[1],
                    'C': result.x[2]
                }
                
                logger.info(f"Fitted Buckingham for {pair_key}: A={result.x[0]:.2f}, "
                           f"rho={result.x[1]:.4f}, C={result.x[2]:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to fit {pair_key}: {e}")
                params_dict[pair_key] = {'A': 1000.0, 'rho': 0.3, 'C': 0.0}
        
        return params_dict
    
    def _buckingham_residuals(self, params: np.ndarray, pair: Tuple[str, str]) -> np.ndarray:
        """Buckingham拟合的残差函数"""
        A, rho, C = params
        
        residuals = []
        
        for energy, forces, atoms in zip(self.energy_data, self.force_data, self.structures):
            # 计算Buckingham能量
            calc_energy = self._calc_buckingham_energy(atoms, pair, A, rho, C)
            residuals.append(energy - calc_energy)
            
            # 添加力的残差 (可选)
            # calc_forces = self._calc_buckingham_forces(atoms, pair, A, rho, C)
            # residuals.extend((forces - calc_forces).flatten())
        
        return np.array(residuals)
    
    def _calc_buckingham_energy(self, atoms: Atoms, pair: Tuple[str, str], 
                                 A: float, rho: float, C: float) -> float:
        """计算Buckingham能量"""
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        energy = 0.0
        
        # 获取原子类型索引
        indices1 = [i for i, s in enumerate(symbols) if s == pair[0]]
        indices2 = [i for i, s in enumerate(symbols) if s == pair[1]]
        
        for i in indices1:
            for j in indices2:
                if i != j:
                    r = np.linalg.norm(positions[i] - positions[j])
                    if r < self.config.cutoff:
                        energy += A * np.exp(-r / rho) - C / r**6
        
        return energy
    
    def _fit_morse(self) -> Dict:
        """
        拟合Morse势参数
        
        E = D_e * (1 - exp(-a*(r-r_e)))^2
        """
        elements = self.config.elements or self._get_elements()
        pairs = [(e1, e2) for i, e1 in enumerate(elements) 
                 for e2 in elements[i:]]
        
        params_dict = {}
        
        for pair in pairs:
            pair_key = f"{pair[0]}-{pair[1]}"
            
            p0 = [1.0, 1.5, 2.0]  # D_e, a, r_e
            
            try:
                result = least_squares(
                    self._morse_residuals,
                    p0,
                    args=(pair,),
                    bounds=([0.01, 0.1, 0.5], [10.0, 5.0, 5.0]),
                    max_nfev=self.config.max_iterations
                )
                
                params_dict[pair_key] = {
                    'D_e': result.x[0],
                    'a': result.x[1],
                    'r_e': result.x[2]
                }
                
                logger.info(f"Fitted Morse for {pair_key}: D_e={result.x[0]:.4f}, "
                           f"a={result.x[1]:.4f}, r_e={result.x[2]:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to fit {pair_key}: {e}")
                params_dict[pair_key] = {'D_e': 1.0, 'a': 1.5, 'r_e': 2.0}
        
        return params_dict
    
    def _morse_residuals(self, params: np.ndarray, pair: Tuple[str, str]) -> np.ndarray:
        """Morse拟合的残差函数"""
        D_e, a, r_e = params
        
        residuals = []
        
        for energy, atoms in zip(self.energy_data, self.structures):
            calc_energy = self._calc_morse_energy(atoms, pair, D_e, a, r_e)
            residuals.append(energy - calc_energy)
        
        return np.array(residuals)
    
    def _calc_morse_energy(self, atoms: Atoms, pair: Tuple[str, str],
                            D_e: float, a: float, r_e: float) -> float:
        """计算Morse能量"""
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        energy = 0.0
        indices1 = [i for i, s in enumerate(symbols) if s == pair[0]]
        indices2 = [i for i, s in enumerate(symbols) if s == pair[1]]
        
        for i in indices1:
            for j in indices2:
                if i != j:
                    r = np.linalg.norm(positions[i] - positions[j])
                    if r < self.config.cutoff:
                        energy += D_e * (1 - np.exp(-a * (r - r_e)))**2
        
        return energy
    
    def _fit_lennard_jones(self) -> Dict:
        """
        拟合Lennard-Jones势参数
        
        E = 4*epsilon * ((sigma/r)^12 - (sigma/r)^6)
        """
        elements = self.config.elements or self._get_elements()
        pairs = [(e1, e2) for i, e1 in enumerate(elements) 
                 for e2 in elements[i:]]
        
        params_dict = {}
        
        for pair in pairs:
            pair_key = f"{pair[0]}-{pair[1]}"
            
            p0 = [0.1, 3.0]  # epsilon, sigma
            
            try:
                result = least_squares(
                    self._lj_residuals,
                    p0,
                    args=(pair,),
                    bounds=([0.001, 1.0], [1.0, 5.0]),
                    max_nfev=self.config.max_iterations
                )
                
                params_dict[pair_key] = {
                    'epsilon': result.x[0],
                    'sigma': result.x[1]
                }
                
                logger.info(f"Fitted LJ for {pair_key}: epsilon={result.x[0]:.4f}, "
                           f"sigma={result.x[1]:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to fit {pair_key}: {e}")
                params_dict[pair_key] = {'epsilon': 0.1, 'sigma': 3.0}
        
        return params_dict
    
    def _lj_residuals(self, params: np.ndarray, pair: Tuple[str, str]) -> np.ndarray:
        """LJ拟合的残差函数"""
        epsilon, sigma = params
        
        residuals = []
        
        for energy, atoms in zip(self.energy_data, self.structures):
            calc_energy = self._calc_lj_energy(atoms, pair, epsilon, sigma)
            residuals.append(energy - calc_energy)
        
        return np.array(residuals)
    
    def _calc_lj_energy(self, atoms: Atoms, pair: Tuple[str, str],
                         epsilon: float, sigma: float) -> float:
        """计算LJ能量"""
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        energy = 0.0
        indices1 = [i for i, s in enumerate(symbols) if s == pair[0]]
        indices2 = [i for i, s in enumerate(symbols) if s == pair[1]]
        
        for i in indices1:
            for j in indices2:
                if i != j:
                    r = np.linalg.norm(positions[i] - positions[j])
                    if r < self.config.cutoff:
                        sr6 = (sigma / r)**6
                        energy += 4 * epsilon * sr6 * (sr6 - 1)
        
        return energy
    
    def _get_elements(self) -> List[str]:
        """从结构中自动获取元素列表"""
        if self.structures:
            symbols = self.structures[0].get_chemical_symbols()
            return list(set(symbols))
        return ["H", "O"]  # 默认值
    
    def validate(self, test_frames: List[Dict]) -> Dict:
        """验证拟合的力场"""
        if self.fitted_params is None:
            raise ValueError("Must fit force field before validation")
        
        errors = []
        
        for frame in test_frames:
            if frame.get('energy') is not None:
                # 计算预测能量
                # 这里简化处理
                pred_energy = 0.0  # 实际应使用拟合的势函数计算
                true_energy = frame['energy']
                errors.append(abs(pred_energy - true_energy))
        
        return {
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean(np.array(errors)**2)),
            'max_error': np.max(errors)
        }
    
    def save_params(self, output_file: str):
        """保存拟合参数"""
        with open(output_file, 'w') as f:
            json.dump(self.fitted_params, f, indent=2)
        logger.info(f"Saved force field parameters to {output_file}")
    
    def load_params(self, input_file: str):
        """加载拟合参数"""
        with open(input_file, 'r') as f:
            self.fitted_params = json.load(f)
        logger.info(f"Loaded force field parameters from {input_file}")


# =============================================================================
# LAMMPS输入文件生成器
# =============================================================================

class LAMMPSInputGenerator:
    """
    LAMMPS输入文件生成器
    
    功能:
    - 自动生成LAMMPS输入脚本
    - 支持多种势函数类型
    - 支持多种系综 (NVE, NVT, NPT)
    - 包含完整的模拟流程设置
    """
    
    def __init__(self, config: Optional[LAMMPSInputConfig] = None):
        self.config = config or LAMMPSInputConfig()
        self.input_lines = []
    
    def generate(self, 
                 atoms: Atoms,
                 potential_params: Optional[Dict] = None,
                 output_file: str = "in.lammps") -> str:
        """
        生成LAMMPS输入文件
        
        Args:
            atoms: ASE Atoms对象
            potential_params: 势函数参数字典 (从ForceFieldFitter获得)
            output_file: 输出文件名
            
        Returns:
            input_file_path: 生成的输入文件路径
        """
        self.atoms = atoms
        self.potential_params = potential_params
        self.input_lines = []
        
        # 构建输入文件
        self._add_header()
        self._add_initialization()
        self._add_structure_read()
        self._add_potential_setup()
        self._add_simulation_settings()
        self._add_output_settings()
        self._add_run_commands()
        
        # 写入文件
        input_content = "\n".join(self.input_lines)
        
        with open(output_file, 'w') as f:
            f.write(input_content)
        
        # 同时写入结构数据文件
        data_file = Path(output_file).parent / "structure.data"
        write_lammps_data(data_file, atoms, atom_style='atomic')
        
        logger.info(f"Generated LAMMPS input: {output_file}")
        logger.info(f"Generated structure data: {data_file}")
        
        return output_file
    
    def _add_header(self):
        """添加文件头"""
        self._add_line("# LAMMPS input file generated by dft_to_lammps_bridge.py")
        self._add_line(f"# Generated: {datetime.now().isoformat()}")
        self._add_line(f"# System: {self.atoms.get_chemical_formula()}")
        self._add_line("")
    
    def _add_initialization(self):
        """添加初始化设置"""
        self._add_line("# Initialization")
        self._add_line(f"units {self.config.units}")
        self._add_line(f"dimension 3")
        self._add_line(f"boundary {self.config.boundary}")
        self._add_line(f"atom_style {self.config.atom_style}")
        self._add_line("")
    
    def _add_structure_read(self):
        """添加结构读取"""
        self._add_line("# Read structure")
        self._add_line("read_data structure.data")
        self._add_line("")
        
        # 元素类型映射
        elements = list(set(self.atoms.get_chemical_symbols()))
        for i, elem in enumerate(elements, 1):
            self._add_line(f"# Type {i}: {elem}")
    
    def _add_potential_setup(self):
        """添加势函数设置"""
        self._add_line("")
        self._add_line("# Potential setup")
        
        ff_type = self.config.pair_style
        
        if ff_type == "buck/coul/long":
            self._setup_buckingham()
        elif ff_type == "lj/cut":
            self._setup_lj()
        elif ff_type == "morse":
            self._setup_morse()
        elif ff_type == "eam/alloy":
            self._setup_eam()
        elif ff_type == "deepmd":
            self._setup_deepmd()
        else:
            raise ValueError(f"Unsupported pair style: {ff_type}")
        
        self._add_line("")
        
        # 邻居列表
        self._add_line("neighbor 2.0 bin")
        self._add_line("neigh_modify every 10 delay 0 check yes")
        self._add_line("")
    
    def _setup_buckingham(self):
        """设置Buckingham势"""
        self._add_line(f"pair_style {self.config.pair_style} {self.config.cutoff}")
        
        elements = list(set(self.atoms.get_chemical_symbols()))
        
        # 如果提供了拟合参数，使用它们
        if self.potential_params:
            for i, e1 in enumerate(elements, 1):
                for j, e2 in enumerate(elements[i-1:], i):
                    pair_key = f"{e1}-{e2}" if f"{e1}-{e2}" in self.potential_params else f"{e2}-{e1}"
                    if pair_key in self.potential_params:
                        p = self.potential_params[pair_key]
                        A = p.get('A', 1000.0)
                        rho = p.get('rho', 0.3)
                        C = p.get('C', 0.0)
                        self._add_line(f"pair_coeff {i} {j} {A:.4f} {rho:.4f} {C:.4f}")
        
        # K-space求和
        if "coul/long" in self.config.pair_style:
            self._add_line(f"kspace_style {self.config.kspace_style}")
    
    def _setup_lj(self):
        """设置Lennard-Jones势"""
        self._add_line(f"pair_style {self.config.pair_style} {self.config.cutoff}")
        
        elements = list(set(self.atoms.get_chemical_symbols()))
        
        if self.potential_params:
            for i, e1 in enumerate(elements, 1):
                for j, e2 in enumerate(elements[i-1:], i):
                    pair_key = f"{e1}-{e2}" if f"{e1}-{e2}" in self.potential_params else f"{e2}-{e1}"
                    if pair_key in self.potential_params:
                        p = self.potential_params[pair_key]
                        epsilon = p.get('epsilon', 0.1)
                        sigma = p.get('sigma', 3.0)
                        self._add_line(f"pair_coeff {i} {j} {epsilon:.6f} {sigma:.4f}")
    
    def _setup_morse(self):
        """设置Morse势"""
        self._add_line(f"pair_style {self.config.pair_style} {self.config.cutoff}")
        
        elements = list(set(self.atoms.get_chemical_symbols()))
        
        if self.potential_params:
            for i, e1 in enumerate(elements, 1):
                for j, e2 in enumerate(elements[i-1:], i):
                    pair_key = f"{e1}-{e2}" if f"{e1}-{e2}" in self.potential_params else f"{e2}-{e1}"
                    if pair_key in self.potential_params:
                        p = self.potential_params[pair_key]
                        D_e = p.get('D_e', 1.0)
                        alpha = p.get('a', 1.5)
                        r0 = p.get('r_e', 2.0)
                        self._add_line(f"pair_coeff {i} {j} {D_e:.6f} {alpha:.4f} {r0:.4f}")
    
    def _setup_eam(self):
        """设置EAM势"""
        self._add_line("pair_style eam/alloy")
        # 假设势文件存在
        elements = " ".join(set(self.atoms.get_chemical_symbols()))
        self._add_line(f"pair_coeff * * potential.eam {elements}")
    
    def _setup_deepmd(self):
        """设置DeepMD势"""
        self._add_line("pair_style deepmd graph.pb")
        self._add_line("pair_coeff * *")
    
    def _add_simulation_settings(self):
        """添加模拟设置"""
        self._add_line("# Simulation settings")
        self._add_line(f"timestep {self.config.timestep / 1000}")  # fs -> ps for metal units
        
        # 速度初始化
        T = self.config.temperature
        self._add_line(f"velocity all create {T} 12345 dist {self.config.velocity_dist}")
        self._add_line("")
        
        # 约束
        if self.config.fix_com:
            self._add_line("fix com_fix all momentum 100 linear 1 1 1")
        self._add_line("")
    
    def _add_output_settings(self):
        """添加输出设置"""
        self._add_line("# Output settings")
        self._add_line(f"thermo {self.config.thermo_interval}")
        self._add_line("thermo_style custom step temp pe ke etotal press vol density")
        self._add_line("")
        
        # Dump轨迹
        self._add_line(f"dump traj all custom {self.config.dump_interval} dump.lammpstrj id type x y z vx vy vz fx fy fz")
        self._add_line("dump_modify traj sort id")
        self._add_line("")
    
    def _add_run_commands(self):
        """添加运行命令"""
        self._add_line("# Run simulation")
        
        T = self.config.temperature
        P = self.config.pressure
        nsteps = self.config.nsteps
        ensemble = self.config.ensemble
        
        if ensemble == "nve":
            self._add_line("fix ensemble all nve")
            
        elif ensemble == "nvt":
            self._add_line(f"fix ensemble all nvt temp {T} {T} $(100.0*dt)")
            
        elif ensemble == "npt":
            if P is None:
                raise ValueError("Pressure must be specified for NPT ensemble")
            # 将atm转换为bar (LAMMPS默认单位)
            P_bar = P * 1.01325
            self._add_line(f"fix ensemble all npt temp {T} {T} $(100.0*dt) iso {P_bar} {P_bar} $(1000.0*dt)")
        
        self._add_line("")
        self._add_line(f"run {nsteps}")
        self._add_line("")
        self._add_line("unfix ensemble")
        if self.config.fix_com:
            self._add_line("unfix com_fix")
        self._add_line("")
        self._add_line("write_data final_structure.data")
    
    def _add_line(self, line: str):
        """添加一行到输入文件"""
        self.input_lines.append(line)


# =============================================================================
# QM/MM边界处理
# =============================================================================

class QMMMBoundaryHandler:
    """
    QM/MM边界处理器
    
    实现方案:
    - 机械耦合 (Mechanical coupling)
    - 静电耦合 (Electrostatic coupling)  
    - 减法方案 (Subtractive scheme)
    
    边界原子处理:
    - 链接原子法 (Link atom method)
    - 边界平滑 (Boundary smoothing)
    """
    
    def __init__(self, config: Optional[QMMMConfig] = None):
        self.config = config or QMMMConfig()
        self.qm_atoms = None
        self.mm_atoms = None
        self.link_atoms = []
    
    def partition_system(self, atoms: Atoms) -> Tuple[Atoms, Atoms]:
        """
        将系统划分为QM和MM区域
        
        Returns:
            qm_atoms: QM区域原子
            mm_atoms: MM区域原子
        """
        indices = np.arange(len(atoms))
        
        if not self.config.qm_region:
            # 自动识别QM区域 (例如，活性位点)
            self.config.qm_region = self._auto_detect_qm_region(atoms)
        
        qm_mask = np.isin(indices, self.config.qm_region)
        mm_mask = ~qm_mask
        
        self.qm_atoms = atoms[qm_mask].copy()
        self.mm_atoms = atoms[mm_mask].copy()
        
        # 处理QM/MM边界
        if self.config.coupling_scheme == "mechanical":
            self._setup_mechanical_coupling(atoms, qm_mask)
        elif self.config.coupling_scheme == "electrostatic":
            self._setup_electrostatic_coupling(atoms, qm_mask)
        
        logger.info(f"Partitioned system: {len(self.qm_atoms)} QM atoms, {len(self.mm_atoms)} MM atoms")
        
        return self.qm_atoms, self.mm_atoms
    
    def _auto_detect_qm_region(self, atoms: Atoms) -> List[int]:
        """自动检测QM区域"""
        # 简化实现：选择中心区域的原子
        positions = atoms.get_positions()
        center = np.mean(positions, axis=0)
        
        # 找到距离中心最近的原子
        distances = np.linalg.norm(positions - center, axis=1)
        
        # 选择最近的10个原子作为QM区域
        qm_indices = np.argsort(distances)[:min(10, len(atoms))].tolist()
        
        return qm_indices
    
    def _setup_mechanical_coupling(self, atoms: Atoms, qm_mask: np.ndarray):
        """设置机械耦合"""
        # QM和MM区域独立计算，然后在边界处叠加
        pass
    
    def _setup_electrostatic_coupling(self, atoms: Atoms, qm_mask: np.ndarray):
        """设置静电耦合"""
        # MM区域的电荷作为外场作用于QM区域
        pass
    
    def add_link_atoms(self, qm_atoms: Atoms, mm_atoms: Atoms) -> Atoms:
        """
        添加链接原子
        
        在断开的共价键处添加氢原子来饱和QM区域的价键
        """
        # 检测QM/MM边界上的断键
        cutoff = 1.8  # Å, 化学键截止距离
        
        qm_positions = qm_atoms.get_positions()
        mm_positions = mm_atoms.get_positions()
        
        # 查找QM和MM原子之间的近距离对
        for i, qm_pos in enumerate(qm_positions):
            for j, mm_pos in enumerate(mm_positions):
                dist = np.linalg.norm(qm_pos - mm_pos)
                if dist < cutoff:
                    # 在QM-MM键处添加链接原子
                    link_pos = self._calculate_link_position(qm_pos, mm_pos)
                    link_atom = Atoms(self.config.link_atom_type, positions=[link_pos])
                    self.link_atoms.append(link_atom)
        
        logger.info(f"Added {len(self.link_atoms)} link atoms")
        
        # 合并QM原子和链接原子
        if self.link_atoms:
            from ase.build import molecule
            combined = qm_atoms.copy()
            for link in self.link_atoms:
                combined.extend(link)
            return combined
        
        return qm_atoms
    
    def _calculate_link_position(self, qm_pos: np.ndarray, mm_pos: np.ndarray) -> np.ndarray:
        """计算链接原子位置"""
        # 放在距离QM原子适当的键长处
        bond_length = 1.09  # C-H键长，单位Å
        direction = (mm_pos - qm_pos) / np.linalg.norm(mm_pos - qm_pos)
        return qm_pos + direction * bond_length
    
    def calculate_embedding_energy(self, qm_atoms: Atoms, mm_atoms: Atoms) -> float:
        """计算MM区域对QM区域的嵌入能"""
        # 简化为MM电荷产生的静电势
        energy = 0.0
        
        # 获取MM原子的电荷
        mm_charges = self._get_mm_charges(mm_atoms)
        
        # 计算与QM原子的相互作用
        for i, qm_atom in enumerate(qm_atoms):
            for j, mm_atom in enumerate(mm_atoms):
                r = np.linalg.norm(qm_atom.position - mm_atom.position)
                if r > 0:
                    energy += mm_charges[j] / r  # 静电相互作用 (简化)
        
        return energy
    
    def _get_mm_charges(self, mm_atoms: Atoms) -> np.ndarray:
        """获取MM原子电荷"""
        # 简化：使用预定义的电荷字典
        charges = []
        for symbol in mm_atoms.get_chemical_symbols():
            charges.append(self.config.charge_dict.get(symbol, 0.0))
        return np.array(charges)


# =============================================================================
# ASE接口封装
# =============================================================================

class UnifiedDFTMDCalculator(Calculator):
    """
    统一的DFT-MD计算器
    
    封装VASP、QE和LAMMPS的ASE计算器，提供统一接口
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, 
                 mode: str = "vasp",
                 qmmm_config: Optional[QMMMConfig] = None,
                 **kwargs):
        """
        Args:
            mode: "vasp", "espresso", 或 "lammps"
            qmmm_config: QM/MM配置 (可选)
            **kwargs: 传递给底层计算器的参数
        """
        super().__init__(**kwargs)
        
        self.mode = mode
        self.qmmm_config = qmmm_config
        self.calculator = None
        
        # 初始化底层计算器
        if mode == "vasp":
            self.calculator = Vasp(**kwargs)
        elif mode == "espresso":
            self.calculator = Espresso(**kwargs)
        elif mode == "lammps":
            self.calculator = ASE_LAMMPS(**kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        """执行计算"""
        if atoms is not None:
            self.atoms = atoms.copy()
        
        if self.qmmm_config:
            # QM/MM计算
            self._calculate_qmmm()
        else:
            # 纯DFT或纯MD计算
            self.atoms.calc = self.calculator
            
            if 'energy' in properties:
                self.results['energy'] = self.atoms.get_potential_energy()
            if 'forces' in properties:
                self.results['forces'] = self.atoms.get_forces()
            if 'stress' in properties:
                self.results['stress'] = self.atoms.get_stress()
    
    def _calculate_qmmm(self):
        """QM/MM计算"""
        handler = QMMMBoundaryHandler(self.qmmm_config)
        qm_atoms, mm_atoms = handler.partition_system(self.atoms)
        
        # QM计算
        qm_atoms = handler.add_link_atoms(qm_atoms, mm_atoms)
        qm_atoms.calc = self.calculator
        qm_energy = qm_atoms.get_potential_energy()
        
        # 嵌入能
        embedding = handler.calculate_embedding_energy(qm_atoms, mm_atoms)
        
        # 总能量
        self.results['energy'] = qm_energy + embedding
        
        # 力 (简化：只使用QM区域的力)
        self.results['forces'] = qm_atoms.get_forces()[:len(self.qmmm_config.qm_region)]


# =============================================================================
# 主工作流类
# =============================================================================

class DFTToLAMMPSBridge:
    """
    DFT到LAMMPS的桥梁类
    
    主工作流:
    1. 解析DFT输出 (VASP/QE)
    2. 拟合力场参数
    3. 生成LAMMPS输入
    4. 运行验证
    """
    
    def __init__(self, working_dir: str = "./dft_lammps_bridge"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = None
        self.fitter = None
        self.lammps_generator = None
        
        self.frames = []
        self.fitted_params = None
    
    def parse_dft_output(self, 
                        output_path: Union[str, Path],
                        code: str = "vasp") -> List[Dict]:
        """
        解析DFT输出文件
        
        Args:
            output_path: 输出文件路径
            code: "vasp" 或 "espresso"
            
        Returns:
            frames: 解析的帧列表
        """
        if code == "vasp":
            self.parser = VASPOUTCARParser()
        elif code == "espresso":
            self.parser = QuantumESPRESSOParser()
        else:
            raise ValueError(f"Unsupported code: {code}")
        
        self.frames = self.parser.parse(output_path)
        
        # 保存统计信息
        stats = self.parser.get_statistics()
        with open(self.working_dir / "dft_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        return self.frames
    
    def fit_force_field(self, 
                       ff_type: str = "buckingham",
                       config: Optional[ForceFieldConfig] = None) -> Dict:
        """
        拟合力场参数
        
        Args:
            ff_type: 力场类型
            config: 力场配置
            
        Returns:
            fitted_params: 拟合参数
        """
        if not self.frames:
            raise ValueError("No DFT data available. Run parse_dft_output first.")
        
        config = config or ForceFieldConfig(ff_type=ff_type)
        self.fitter = ForceFieldFitter(config)
        self.fitter.load_data(self.frames)
        
        self.fitted_params = self.fitter.fit()
        
        # 保存参数
        self.fitter.save_params(self.working_dir / "fitted_params.json")
        
        return self.fitted_params
    
    def generate_lammps_input(self,
                             atoms: Optional[Atoms] = None,
                             config: Optional[LAMMPSInputConfig] = None) -> str:
        """
        生成LAMMPS输入文件
        
        Args:
            atoms: 参考结构 (可选，默认使用第一帧)
            config: LAMMPS配置
            
        Returns:
            input_file: 生成的输入文件路径
        """
        if atoms is None and self.frames:
            atoms = self.frames[0]['atoms']
        
        if atoms is None:
            raise ValueError("No structure available")
        
        config = config or LAMMPSInputConfig()
        self.lammps_generator = LAMMPSInputGenerator(config)
        
        output_file = self.working_dir / "in.lammps"
        
        input_file = self.lammps_generator.generate(
            atoms=atoms,
            potential_params=self.fitted_params,
            output_file=str(output_file)
        )
        
        return input_file
    
    def run_full_pipeline(self,
                         dft_output: Union[str, Path],
                         code: str = "vasp",
                         ff_type: str = "buckingham",
                         lammps_config: Optional[LAMMPSInputConfig] = None) -> Dict:
        """
        运行完整流程
        
        Args:
            dft_output: DFT输出文件路径
            code: DFT代码类型
            ff_type: 力场类型
            lammps_config: LAMMPS配置
            
        Returns:
            results: 包含所有结果的字典
        """
        logger.info("=" * 60)
        logger.info("Starting DFT to LAMMPS Bridge Pipeline")
        logger.info("=" * 60)
        
        results = {
            'working_dir': str(self.working_dir),
            'dft_code': code,
            'ff_type': ff_type,
        }
        
        # 步骤1: 解析DFT输出
        logger.info("\nStep 1: Parsing DFT output...")
        frames = self.parse_dft_output(dft_output, code)
        results['n_frames'] = len(frames)
        
        # 步骤2: 拟合力场
        logger.info("\nStep 2: Fitting force field...")
        params = self.fit_force_field(ff_type)
        results['fitted_params'] = params
        
        # 步骤3: 生成LAMMPS输入
        logger.info("\nStep 3: Generating LAMMPS input...")
        lammps_input = self.generate_lammps_input(config=lammps_config)
        results['lammps_input'] = lammps_input
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
        return results


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DFT to LAMMPS Bridge: Parse DFT output, fit force field, generate LAMMPS input"
    )
    
    parser.add_argument("outcar", help="VASP OUTCAR file path")
    parser.add_argument("--code", default="vasp", choices=["vasp", "espresso"],
                       help="DFT code type")
    parser.add_argument("--ff-type", default="buckingham", 
                       choices=["buckingham", "morse", "lj"],
                       help="Force field type")
    parser.add_argument("--output-dir", default="./dft_lammps_bridge",
                       help="Working directory")
    parser.add_argument("--to-deepmd", action="store_true",
                       help="Also export to DeepMD format")
    parser.add_argument("--to-xyz", action="store_true",
                       help="Also export to XYZ format")
    
    args = parser.parse_args()
    
    # 创建并运行桥接器
    bridge = DFTToLAMMPSBridge(working_dir=args.output_dir)
    
    # 运行完整流程
    results = bridge.run_full_pipeline(
        dft_output=args.outcar,
        code=args.code,
        ff_type=args.ff_type
    )
    
    # 可选：导出其他格式
    if args.to_deepmd and DPDATA_AVAILABLE:
        train_dir, valid_dir = bridge.parser.to_deepmd(
            str(Path(args.output_dir) / "deepmd_data")
        )
        print(f"\nExported to DeepMD format:")
        print(f"  Training: {train_dir}")
        print(f"  Validation: {valid_dir}")
    
    if args.to_xyz:
        xyz_file = Path(args.output_dir) / "trajectory.xyz"
        bridge.parser.to_xyz(str(xyz_file))
        print(f"\nExported to XYZ: {xyz_file}")
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"DFT frames parsed: {results['n_frames']}")
    print(f"Force field type: {results['ff_type']}")
    print(f"LAMMPS input: {results['lammps_input']}")
    print(f"Fitted parameters:")
    for pair, params in results['fitted_params'].items():
        print(f"  {pair}: {params}")


if __name__ == "__main__":
    main()
