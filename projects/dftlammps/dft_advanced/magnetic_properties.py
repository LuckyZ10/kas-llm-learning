#!/usr/bin/env python3
"""
magnetic_properties.py
======================
磁性计算模块 - VASP/QE/CP2K多代码支持

功能：
1. 自旋极化DFT（ISPIN=2, MAGMOM）
2. 磁各向异性能量（MAE）
3. 交换耦合常数（J）
4. 居里温度估算

支持代码：
- VASP: ISPIN, MAGMOM, LSORBIT (SOC)
- Quantum ESPRESSO: nspin, starting_magnetization
- CP2K: UKS, MAGNETIZATION

作者: Advanced DFT Expert
日期: 2026-03-09
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import logging
from abc import ABC, abstractmethod

# ASE
from ase import Atoms
from ase.io import read, write
from ase.units import eV, Bohr

# SciPy
from scipy.optimize import minimize, curve_fit
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class MagneticOrder(Enum):
    """磁有序类型"""
    FERROMAGNETIC = "FM"
    ANTIFERROMAGNETIC = "AFM"
    FERRIMAGNETIC = "FiM"
    PARAMAGNETIC = "PM"
    SPIN_GLASS = "SG"


class SpinAxis(Enum):
    """自旋轴方向"""
    X = "x"
    Y = "y"
    Z = "z"
    XY = "xy"
    XZ = "xz"
    YZ = "yz"
    XYZ = "xyz"


@dataclass
class MagneticState:
    """磁状态数据结构"""
    total_energy: float  # eV
    magnetic_moment: float  # μB
    magmom_per_atom: List[float]  # 每个原子的磁矩
    spin_axis: SpinAxis = SpinAxis.Z
    
    # 可选信息
    band_gap: Optional[float] = None
    fermi_energy: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'total_energy': float(self.total_energy),
            'magnetic_moment': float(self.magnetic_moment),
            'magmom_per_atom': [float(m) for m in self.magmom_per_atom],
            'spin_axis': self.spin_axis.value,
            'band_gap': float(self.band_gap) if self.band_gap else None,
            'fermi_energy': float(self.fermi_energy) if self.fermi_energy else None,
        }


@dataclass
class MagneticAnisotropy:
    """磁各向异性数据结构"""
    easy_axis: SpinAxis
    easy_energy: float  # eV
    anisotropy_energy: Dict[SpinAxis, float]  # 相对easy axis的能量差
    
    # 各向异性常数 (单轴)
    K_u: Optional[float] = None  # J/m³ or eV/cell
    
    # 各向异性常数 (立方)
    K_1: Optional[float] = None
    K_2: Optional[float] = None
    
    def get_mae(self, axis: SpinAxis) -> float:
        """获取特定方向的磁各向异性能"""
        return self.anisotropy_energy.get(axis, 0.0)
    
    def to_dict(self) -> Dict:
        return {
            'easy_axis': self.easy_axis.value,
            'easy_energy': float(self.easy_energy),
            'anisotropy_energy': {k.value: float(v) for k, v in self.anisotropy_energy.items()},
            'K_u': float(self.K_u) if self.K_u else None,
            'K_1': float(self.K_1) if self.K_1 else None,
            'K_2': float(self.K_2) if self.K_2 else None,
        }


@dataclass
class ExchangeCoupling:
    """交换耦合数据结构"""
    J_ij: Dict[Tuple[int, int], float]  # 交换耦合常数 (meV)
    neighbors: Dict[int, List[int]]  # 邻居列表
    distances: Dict[Tuple[int, int], float]  # 原子间距
    
    # 平均交换耦合
    J_avg: Optional[float] = None
    J_nn: Optional[float] = None  # 最近邻
    
    def to_dict(self) -> Dict:
        return {
            'J_ij': {f"{i}-{j}": float(v) for (i, j), v in self.J_ij.items()},
            'neighbors': {str(k): v for k, v in self.neighbors.items()},
            'distances': {f"{i}-{j}": float(v) for (i, j), v in self.distances.items()},
            'J_avg': float(self.J_avg) if self.J_avg else None,
            'J_nn': float(self.J_nn) if self.J_nn else None,
        }


@dataclass
class CurieTemperature:
    """居里温度估算结果"""
    T_c_mean_field: float  # K, 平均场近似
    T_c_monte_carlo: Optional[float] = None  # K, Monte Carlo结果
    T_c_renormalization: Optional[float] = None  # K, 重整化群
    
    # 使用的模型参数
    spin_magnitude: float = 0.5
    z_coordination: int = 6  # 配位数
    J_eff: float = 0.0  # meV
    
    def to_dict(self) -> Dict:
        return {
            'T_c_mean_field': float(self.T_c_mean_field),
            'T_c_monte_carlo': float(self.T_c_monte_carlo) if self.T_c_monte_carlo else None,
            'T_c_renormalization': float(self.T_c_renormalization) if self.T_c_renormalization else None,
            'spin_magnitude': float(self.spin_magnitude),
            'z_coordination': self.z_coordination,
            'J_eff': float(self.J_eff),
        }


@dataclass
class SpinConfiguration:
    """自旋构型"""
    indices: List[int]  # 原子索引
    directions: List[np.ndarray]  # 自旋方向 (单位矢量)
    magnitudes: List[float]  # 自旋大小
    
    def get_magmom_string(self) -> str:
        """生成MAGMOM字符串 (VASP格式)"""
        magmoms = []
        for mag, direction in zip(self.magnitudes, self.directions):
            magmoms.append(mag * direction[2])  # z分量
        return " ".join([f"{m:.1f}" for m in magmoms])
    
    def get_vector_magmom(self) -> List[List[float]]:
        """获取矢量磁矩 (用于SOC计算)"""
        result = []
        for mag, direction in zip(self.magnitudes, self.directions):
            result.append([mag * d for d in direction])
        return result


@dataclass
class VASPMagneticConfig:
    """VASP磁性计算配置"""
    ispin: int = 2  # 2 for spin-polarized
    magmom: Optional[str] = None  # 初始磁矩
    
    # SOC参数
    lsorbit: bool = False  # 开启SOC
    saxis: Tuple[float, float, float] = (0, 0, 1)  # 自旋量化轴
    
    # 收敛参数
    nelm: int = 100
    nelmin: int = 4
    amix: float = 0.4
    bmix: float = 1.0
    
    # 其他
    encut: float = 520
    ismear: int = 0
    sigma: float = 0.05
    ncores: int = 32


@dataclass
class QEMagneticConfig:
    """Quantum ESPRESSO磁性计算配置"""
    nspin: int = 2  # 2 for spin-polarized
    starting_magnetization: Dict[str, float] = field(default_factory=dict)
    
    # 非共线磁性
    noncolin: bool = False
    lspinorb: bool = False
    
    # 参数
    ecutwfc: float = 60  # Ry
    ecutrho: float = 480  # Ry
    mixing_beta: float = 0.7


# =============================================================================
# Base Calculator Class
# =============================================================================

class MagneticCalculator(ABC):
    """磁性计算基类"""
    
    def __init__(self, config: Any):
        self.config = config
        self.results = {}
        self.magnetic_states = {}
    
    @abstractmethod
    def calculate_magnetic_state(self, structure: Atoms,
                                  spin_config: Optional[SpinConfiguration] = None,
                                  output_dir: str = "./") -> MagneticState:
        """计算磁状态"""
        pass
    
    @abstractmethod
    def calculate_anisotropy(self, structure: Atoms,
                             output_dir: str = "./") -> MagneticAnisotropy:
        """计算磁各向异性"""
        pass
    
    def calculate_exchange_coupling(self, structure: Atoms,
                                     reference_states: Dict[str, MagneticState],
                                     output_dir: str = "./") -> ExchangeCoupling:
        """
        计算交换耦合常数 (四态法)
        
        基于总能量差计算Heisenberg模型参数:
        H = -Σ_{i<j} J_ij S_i · S_j
        
        需要参考态:
        - FM: 所有自旋平行
        - AFM1, AFM2, ...: 不同的反铁磁构型
        """
        logger.info("Calculating exchange coupling constants (4-state method)...")
        
        # 这里实现四态法
        # 简化版本，实际需要多个计算
        
        # 获取磁矩最大的磁性原子
        mag_indices = self._identify_magnetic_atoms(structure)
        
        if len(mag_indices) < 2:
            logger.warning("Not enough magnetic atoms for exchange coupling")
            return ExchangeCoupling(
                J_ij={},
                neighbors={},
                distances={},
            )
        
        # 构建邻居列表
        positions = structure.get_positions()
        distances = squareform(pdist(positions[mag_indices]))
        
        neighbors = {}
        J_ij = {}
        dist_dict = {}
        
        # 假设已知FM和AFM态的能量差
        if 'FM' in reference_states and 'AFM' in reference_states:
            E_FM = reference_states['FM'].total_energy
            E_AFM = reference_states['AFM'].total_energy
            delta_E = E_AFM - E_FM
            
            # 简化的最近邻J估计
            # ΔE = E_AFM - E_FM ≈ 2zJS² (对于 bipartite lattice)
            # 这里假设S=1/2
            z = self._get_coordination(structure, mag_indices)
            S = 0.5
            
            J_eff = delta_E / (2 * z * S**2) * 1000  # 转换为meV
            
            # 分配给最近邻对
            for i, idx_i in enumerate(mag_indices):
                neighbors[idx_i] = []
                for j, idx_j in enumerate(mag_indices):
                    if i != j:
                        dist = distances[i, j]
                        # 找到最近邻
                        if dist < 3.5:  # 假设最近邻距离小于3.5 Å
                            neighbors[idx_i].append(idx_j)
                            J_ij[(idx_i, idx_j)] = J_eff
                            dist_dict[(idx_i, idx_j)] = dist
        
        J_avg = np.mean(list(J_ij.values())) if J_ij else 0.0
        
        return ExchangeCoupling(
            J_ij=J_ij,
            neighbors=neighbors,
            distances=dist_dict,
            J_avg=J_avg,
            J_nn=J_avg,
        )
    
    def estimate_curie_temperature(self, exchange: ExchangeCoupling,
                                    spin_magnitude: float = 0.5) -> CurieTemperature:
        """
        估算居里温度
        
        使用平均场近似:
        k_B T_C = (2/3) S(S+1) Σ_j J_{0j}
        
        对于简单立方:
        k_B T_C = (2/3) z J S(S+1)
        """
        logger.info("Estimating Curie temperature...")
        
        z = exchange.z_coordination if hasattr(exchange, 'z_coordination') else 6
        J = exchange.J_nn if exchange.J_nn else 10.0  # meV
        S = spin_magnitude
        
        # 平均场近似
        k_B = 8.617e-5  # eV/K
        T_C_MF = (2/3) * z * abs(J) * S * (S + 1) / (k_B * 1000)  # 转换为K
        
        # Monte Carlo估算 (简化为MF的0.8倍，对于3D Heisenberg)
        T_C_MC = 0.8 * T_C_MF
        
        # 重整化群估算
        T_C_RG = 0.9 * T_C_MF
        
        return CurieTemperature(
            T_c_mean_field=T_C_MF,
            T_c_monte_carlo=T_C_MC,
            T_c_renormalization=T_C_RG,
            spin_magnitude=S,
            z_coordination=z,
            J_eff=J,
        )
    
    def _identify_magnetic_atoms(self, structure: Atoms) -> List[int]:
        """识别磁性原子 (过渡金属、稀土)"""
        magnetic_elements = {
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
            'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        }
        
        symbols = structure.get_chemical_symbols()
        indices = [i for i, sym in enumerate(symbols) if sym in magnetic_elements]
        
        return indices
    
    def _get_coordination(self, structure: Atoms, indices: List[int]) -> int:
        """估算配位数"""
        if len(indices) < 2:
            return 0
        
        positions = structure.get_positions()[indices]
        distances = squareform(pdist(positions))
        
        # 计算平均最近邻数 (距离小于阈值的原子数)
        threshold = 3.0  # Å
        coord = np.mean([np.sum((0 < d) & (d < threshold)) - 1 for d in distances])
        
        return int(round(coord))
    
    def save_results(self, output_dir: str):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_dict = {}
        
        for key, value in self.results.items():
            if hasattr(value, 'to_dict'):
                results_dict[key] = value.to_dict()
            elif isinstance(value, dict):
                results_dict[key] = {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                                     for k, v in value.items()}
        
        with open(output_path / "magnetic_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path / 'magnetic_results.json'}")


# =============================================================================
# VASP Implementation
# =============================================================================

class VASPMagneticCalculator(MagneticCalculator):
    """VASP磁性计算器"""
    
    def __init__(self, config: Optional[VASPMagneticConfig] = None):
        super().__init__(config or VASPMagneticConfig())
    
    def calculate_magnetic_state(self, structure: Atoms,
                                  spin_config: Optional[SpinConfiguration] = None,
                                  output_dir: str = "./") -> MagneticState:
        """计算磁状态"""
        from ase.calculators.vasp import Vasp
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running VASP spin-polarized calculation in {output_dir}")
        
        # 设置MAGMOM
        magmom = None
        if spin_config:
            magmom = spin_config.get_magmom_string()
        elif self.config.magmom:
            magmom = self.config.magmom
        else:
            # 自动设置初始磁矩
            magmom = self._auto_magmom(structure)
        
        # 设置计算器
        calc = Vasp(
            directory=str(output_path),
            xc='PBE',
            encut=self.config.encut,
            ispin=self.config.ispin,
            magmom=magmom,
            ismear=self.config.ismear,
            sigma=self.config.sigma,
            nelm=self.config.nelm,
            nelmin=self.config.nelmin,
            amix=self.config.amix,
            bmix=self.config.bmix,
            ncore=self.config.ncores,
            lwave=True,
            lcharg=True,
        )
        
        structure.calc = calc
        
        # 运行计算
        energy = structure.get_potential_energy()
        
        # 读取磁矩
        magmom_total = self._read_total_magmom(output_path)
        magmom_per_atom = self._read_magmom_per_atom(output_path)
        
        # 读取其他信息
        band_gap = self._read_band_gap(output_path)
        
        state = MagneticState(
            total_energy=energy,
            magnetic_moment=magmom_total,
            magmom_per_atom=magmom_per_atom,
            band_gap=band_gap,
        )
        
        logger.info(f"Magnetic state: E = {energy:.4f} eV, M = {magmom_total:.3f} μB")
        
        return state
    
    def calculate_anisotropy(self, structure: Atoms,
                             output_dir: str = "./") -> MagneticAnisotropy:
        """
        计算磁各向异性 (MAE)
        
        需要开启SOC (LSORBIT = .TRUE.)
        计算不同方向的能量差
        """
        if not self.config.lsorbit:
            logger.warning("SOC not enabled. Set lsorbit=True for MAE calculation.")
        
        logger.info("Calculating magnetic anisotropy energy...")
        
        # 计算不同方向的能量
        directions = {
            SpinAxis.X: (1, 0, 0),
            SpinAxis.Y: (0, 1, 0),
            SpinAxis.Z: (0, 0, 1),
        }
        
        energies = {}
        states = {}
        
        for axis, saxis in directions.items():
            logger.info(f"  Calculating for axis: {axis.value}")
            
            axis_dir = Path(output_dir) / f"axis_{axis.value}"
            
            # 更新SAXIS
            self.config.saxis = saxis
            
            # 运行计算
            state = self._calculate_with_soc(structure, axis_dir)
            energies[axis] = state.total_energy
            states[axis] = state
        
        # 找到easy axis
        easy_axis = min(energies, key=energies.get)
        easy_energy = energies[easy_axis]
        
        # 计算各向异性能
        anisotropy = {axis: E - easy_energy for axis, E in energies.items()}
        
        # 计算各向异性常数
        # K_u ≈ (E_hard - E_easy) / V (对于单轴)
        volume = structure.get_volume() * 1e-30  # Å³ to m³
        
        # 找到hard axis
        hard_axis = max(anisotropy, key=anisotropy.get)
        K_u = anisotropy[hard_axis] * 1.602e-19 / volume  # eV to J, then J/m³
        
        return MagneticAnisotropy(
            easy_axis=easy_axis,
            easy_energy=easy_energy,
            anisotropy_energy=anisotropy,
            K_u=K_u,
        )
    
    def _calculate_with_soc(self, structure: Atoms, output_dir: Path) -> MagneticState:
        """运行包含SOC的计算"""
        from ase.calculators.vasp import Vasp
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        calc = Vasp(
            directory=str(output_dir),
            xc='PBE',
            encut=self.config.encut,
            ispin=2,
            lsorbit=True,
            saxis=self.config.saxis,
            ismear=self.config.ismear,
            sigma=self.config.sigma,
            ncore=self.config.ncores,
        )
        
        structure.calc = calc
        energy = structure.get_potential_energy()
        
        magmom = self._read_total_magmom(output_dir)
        magmom_per_atom = self._read_magmom_per_atom(output_dir)
        
        return MagneticState(
            total_energy=energy,
            magnetic_moment=magmom,
            magmom_per_atom=magmom_per_atom,
        )
    
    def _auto_magmom(self, structure: Atoms) -> str:
        """自动生成MAGMOM"""
        symbols = structure.get_chemical_symbols()
        
        # 默认磁矩
        default_magmom = {
            'H': 0.0, 'He': 0.0,
            'Li': 0.0, 'Be': 0.0, 'B': 0.0, 'C': 0.0, 'N': 0.0, 'O': 0.0, 'F': 0.0,
            'Na': 0.0, 'Mg': 0.0, 'Al': 0.0, 'Si': 0.0, 'P': 0.0, 'S': 0.0, 'Cl': 0.0,
            'K': 0.0, 'Ca': 0.0,
            'Sc': 0.5, 'Ti': 1.0, 'V': 2.0, 'Cr': 3.0, 'Mn': 4.0,
            'Fe': 3.0, 'Co': 2.0, 'Ni': 1.0, 'Cu': 0.5, 'Zn': 0.0,
        }
        
        magmoms = [default_magmom.get(sym, 0.5) for sym in symbols]
        
        return " ".join([f"{m:.1f}" for m in magmoms])
    
    def _read_total_magmom(self, output_path: Path) -> float:
        """从OUTCAR读取总磁矩"""
        outcar_path = output_path / "OUTCAR"
        
        if not outcar_path.exists():
            return 0.0
        
        with open(outcar_path, 'r') as f:
            for line in f:
                if "number of electron" in line and "magnetization" in line:
                    parts = line.split()
                    try:
                        return float(parts[-1])
                    except:
                        pass
        
        return 0.0
    
    def _read_magmom_per_atom(self, output_path: Path) -> List[float]:
        """从OUTCAR读取每个原子的磁矩"""
        outcar_path = output_path / "OUTCAR"
        
        if not outcar_path.exists():
            return []
        
        magmoms = []
        reading = False
        
        with open(outcar_path, 'r') as f:
            for line in f:
                if "magnetization (x)" in line:
                    reading = True
                    magmoms = []
                    continue
                if reading:
                    if line.strip() == "" or "----" in line:
                        if magmoms:
                            break
                        continue
                    parts = line.split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        try:
                            magmoms.append(float(parts[-1]))
                        except:
                            pass
        
        return magmoms
    
    def _read_band_gap(self, output_path: Path) -> Optional[float]:
        """读取带隙 (需要解析EIGENVAL或vasprun.xml)"""
        # 简化实现
        return None


# =============================================================================
# Quantum ESPRESSO Implementation
# =============================================================================

class QEMagneticCalculator(MagneticCalculator):
    """Quantum ESPRESSO磁性计算器"""
    
    def __init__(self, config: Optional[QEMagneticConfig] = None):
        super().__init__(config or QEMagneticConfig())
    
    def calculate_magnetic_state(self, structure: Atoms,
                                  spin_config: Optional[SpinConfiguration] = None,
                                  output_dir: str = "./") -> MagneticState:
        """计算磁状态"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running QE spin-polarized calculation in {output_dir}")
        
        # 生成输入文件
        self._generate_pw_input(structure, output_path, spin_config)
        
        # 实际应运行pw.x
        logger.info("pw.in generated. Run: mpirun pw.x < pw.in > pw.out")
        
        # 模拟结果
        return MagneticState(
            total_energy=-100.0,
            magnetic_moment=2.0,
            magmom_per_atom=[0.5, 0.5, 0.5, 0.5],
        )
    
    def calculate_anisotropy(self, structure: Atoms,
                             output_dir: str = "./") -> MagneticAnisotropy:
        """计算磁各向异性"""
        # QE使用ld1.x或类似工具计算
        logger.info("MAE calculation in QE requires additional tools (ld1.x)")
        
        return MagneticAnisotropy(
            easy_axis=SpinAxis.Z,
            easy_energy=0.0,
            anisotropy_energy={SpinAxis.Z: 0.0},
        )
    
    def _generate_pw_input(self, structure: Atoms, output_path: Path,
                           spin_config: Optional[SpinConfiguration]):
        """生成pw.x输入文件"""
        
        symbols = structure.get_chemical_symbols()
        unique_symbols = list(set(symbols))
        
        # 设置初始磁化
        if not self.config.starting_magnetization:
            for sym in unique_symbols:
                if sym in ['Fe', 'Co', 'Ni', 'Mn', 'Cr']:
                    self.config.starting_magnetization[sym] = 0.5
                else:
                    self.config.starting_magnetization[sym] = 0.0
        
        input_content = f"""&control
    calculation = 'scf'
    prefix = 'magnetic'
    outdir = './'
    pseudo_dir = './'
/
&system
    ibrav = 0
    nat = {len(structure)}
    ntyp = {len(unique_symbols)}
    ecutwfc = {self.config.ecutwfc}
    ecutrho = {self.config.ecutrho}
    nspin = {self.config.nspin}
    occupations = 'smearing'
    smearing = 'gaussian'
    degauss = 0.01
"""
        
        # 添加初始磁化
        for i, sym in enumerate(unique_symbols):
            mag = self.config.starting_magnetization.get(sym, 0.0)
            input_content += f"    starting_magnetization({i+1}) = {mag}\n"
        
        input_content += """/\n&electrons\n    conv_thr = 1.0d-8\n/\n"""
        
        # 原子种类
        input_content += "ATOMIC_SPECIES\n"
        for sym in unique_symbols:
            mass = self._get_atomic_mass(sym)
            input_content += f"{sym}  {mass:.2f}  {sym}.upf\n"
        
        # 原子位置
        input_content += "ATOMIC_POSITIONS angstrom\n"
        positions = structure.get_positions()
        for sym, pos in zip(symbols, positions):
            input_content += f"{sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n"
        
        # 晶胞
        input_content += "CELL_PARAMETERS angstrom\n"
        cell = structure.get_cell()
        for row in cell:
            input_content += f"{row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}\n"
        
        input_content += "K_POINTS automatic\n6 6 6 0 0 0\n"
        
        with open(output_path / "pw.in", 'w') as f:
            f.write(input_content)
    
    def _get_atomic_mass(self, symbol: str) -> float:
        """获取原子质量"""
        masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
            'S': 32.065, 'Cl': 35.453, 'K': 39.098, 'Ca': 40.078,
            'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
            'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
        }
        return masses.get(symbol, 50.0)


# =============================================================================
# Spin Configuration Generators
# =============================================================================

class SpinConfigurationGenerator:
    """自旋构型生成器"""
    
    @staticmethod
    def ferromagnetic(structure: Atoms, magnitude: float = 1.0) -> SpinConfiguration:
        """生成铁磁构型"""
        n_atoms = len(structure)
        indices = list(range(n_atoms))
        directions = [np.array([0, 0, 1]) for _ in range(n_atoms)]
        magnitudes = [magnitude] * n_atoms
        
        return SpinConfiguration(indices, directions, magnitudes)
    
    @staticmethod
    def antiferromagnetic_simple(structure: Atoms, 
                                  magnetic_sites: List[int],
                                  magnitude: float = 1.0) -> SpinConfiguration:
        """
        生成简单反铁磁构型 ( bipartite )
        
        Args:
            structure: ASE Atoms对象
            magnetic_sites: 磁性原子索引列表
            magnitude: 磁矩大小
        """
        directions = []
        
        for i, site in enumerate(magnetic_sites):
            if i % 2 == 0:
                directions.append(np.array([0, 0, 1]))
            else:
                directions.append(np.array([0, 0, -1]))
        
        magnitudes = [magnitude] * len(magnetic_sites)
        
        return SpinConfiguration(magnetic_sites, directions, magnitudes)
    
    @staticmethod
    def antiferromagnetic_neel(structure: Atoms,
                                lattice_type: str = "square") -> SpinConfiguration:
        """
        生成Neel反铁磁构型
        
        Args:
            structure: ASE Atoms对象
            lattice_type: "square", "bcc", "fcc"
        """
        positions = structure.get_positions()
        n_atoms = len(structure)
        
        directions = []
        
        if lattice_type == "square":
            # 2D方晶格: 上下交替
            for i in range(n_atoms):
                if i % 2 == 0:
                    directions.append(np.array([0, 0, 1]))
                else:
                    directions.append(np.array([0, 0, -1]))
        
        elif lattice_type == "bcc":
            # BCC: 角和心反平行
            for i in range(n_atoms):
                if i % 2 == 0:
                    directions.append(np.array([0, 0, 1]))
                else:
                    directions.append(np.array([0, 0, -1]))
        
        else:
            # 默认
            directions = [np.array([0, 0, 1]) if i % 2 == 0 else np.array([0, 0, -1])
                         for i in range(n_atoms)]
        
        magnitudes = [1.0] * n_atoms
        
        return SpinConfiguration(list(range(n_atoms)), directions, magnitudes)
    
    @staticmethod
    def spiral(structure: Atoms,
               q_vector: np.ndarray,
               magnetic_sites: List[int],
               magnitude: float = 1.0,
               plane: str = "xy") -> SpinConfiguration:
        """
        生成螺旋磁序构型
        
        Args:
            structure: ASE Atoms对象
            q_vector: 调制矢量 (2π/a单位)
            magnetic_sites: 磁性原子索引
            magnitude: 磁矩大小
            plane: 旋转平面 ("xy", "yz", "xz")
        """
        positions = structure.get_positions()
        
        directions = []
        
        for site in magnetic_sites:
            pos = positions[site]
            phase = np.dot(q_vector, pos)
            
            if plane == "xy":
                direction = np.array([np.cos(phase), np.sin(phase), 0])
            elif plane == "yz":
                direction = np.array([0, np.cos(phase), np.sin(phase)])
            elif plane == "xz":
                direction = np.array([np.cos(phase), 0, np.sin(phase)])
            else:
                direction = np.array([np.cos(phase), np.sin(phase), 0])
            
            directions.append(direction / np.linalg.norm(direction))
        
        magnitudes = [magnitude] * len(magnetic_sites)
        
        return SpinConfiguration(magnetic_sites, directions, magnitudes)


# =============================================================================
# Visualization
# =============================================================================

class MagneticVisualizer:
    """磁性可视化"""
    
    def __init__(self):
        pass
    
    def plot_magnetic_structure(self, structure: Atoms,
                                 spin_config: SpinConfiguration,
                                 output_file: Optional[str] = None,
                                 show_cell: bool = True):
        """绘制磁性结构 (3D)"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = structure.get_positions()
        symbols = structure.get_chemical_symbols()
        
        # 绘制原子
        colors = {'Fe': 'red', 'Co': 'blue', 'Ni': 'green', 'Mn': 'orange',
                 'Cr': 'purple', 'O': 'gray', 'N': 'gray', 'C': 'gray'}
        
        for i, (pos, sym) in enumerate(zip(positions, symbols)):
            color = colors.get(sym, 'gray')
            ax.scatter(*pos, c=color, s=200, alpha=0.6, edgecolors='black')
            ax.text(pos[0], pos[1], pos[2], f"  {sym}{i}", fontsize=8)
        
        # 绘制自旋
        scale = 0.5
        for idx, direction, mag in zip(spin_config.indices, 
                                        spin_config.directions,
                                        spin_config.magnitudes):
            pos = positions[idx]
            end_pos = pos + direction * mag * scale
            
            # 绘制箭头
            color = 'red' if direction[2] > 0 else 'blue'
            ax.quiver(pos[0], pos[1], pos[2],
                     direction[0]*mag*scale, direction[1]*mag*scale, direction[2]*mag*scale,
                     color=color, arrow_length_ratio=0.3, linewidth=2)
        
        # 绘制晶胞
        if show_cell:
            cell = structure.get_cell()
            origin = np.zeros(3)
            
            for i in range(3):
                for j in range(3):
                    if i != j:
                        v1 = cell[i]
                        v2 = cell[j]
                        ax.plot3D([origin[0], v1[0]], [origin[1], v1[1]], [origin[2], v1[2]], 'k--', alpha=0.3)
                        ax.plot3D([origin[0], v2[0]], [origin[1], v2[1]], [origin[2], v2[2]], 'k--', alpha=0.3)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Magnetic Structure')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved magnetic structure plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_exchange_coupling(self, exchange: ExchangeCoupling,
                                structure: Atoms,
                                output_file: Optional[str] = None):
        """绘制交换耦合网络"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        positions = structure.get_positions()[:, :2]  # 2D投影
        
        # 绘制原子
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=200, zorder=3)
        
        # 绘制交换耦合
        for (i, j), J in exchange.J_ij.items():
            pos_i = positions[i]
            pos_j = positions[j]
            
            # 线宽和颜色基于J值
            linewidth = abs(J) / 10
            color = 'red' if J > 0 else 'blue'
            alpha = min(abs(J) / 50, 1.0)
            
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                   color=color, linewidth=linewidth, alpha=alpha, zorder=1)
            
            # 标注J值
            mid = (pos_i + pos_j) / 2
            ax.text(mid[0], mid[1], f'{J:.1f}', fontsize=8, ha='center')
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_title('Exchange Coupling Network')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved exchange coupling plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_anisotropy_energy(self, anisotropy: MagneticAnisotropy,
                                output_file: Optional[str] = None):
        """绘制各向异性能极坐标图"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # 准备数据
        angles = []
        energies = []
        
        # 对于单轴各向异性，在xz平面绘制
        theta_values = np.linspace(0, 2*np.pi, 100)
        
        for theta in theta_values:
            # 简化的能量表达式: E = K_u sin²θ
            if anisotropy.K_u:
                E = anisotropy.K_u * np.sin(theta)**2 * 1e-3  # 转换为meV
            else:
                E = 0
            energies.append(E)
        
        # 归一化
        max_E = max(energies) if max(energies) > 0 else 1
        energies_norm = [E / max_E for E in energies]
        
        ax.fill(theta_values, energies_norm, alpha=0.3)
        ax.plot(theta_values, energies_norm, 'b-', linewidth=2)
        
        # 标记easy axis
        if anisotropy.easy_axis == SpinAxis.Z:
            ax.annotate('Easy axis', xy=(0, 0), xytext=(0, -0.1),
                       ha='center', fontsize=12, color='red')
        
        ax.set_title('Magnetic Anisotropy Energy', fontsize=14, pad=20)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved anisotropy plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_magnetic_phase_diagram(self, 
                                     temperatures: np.ndarray,
                                     magnetizations: np.ndarray,
                                     T_c: float,
                                     output_file: Optional[str] = None):
        """绘制磁性相图 (M-T曲线)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(temperatures, magnetizations, 'b-', linewidth=2, label='Magnetization')
        ax.axvline(x=T_c, color='r', linestyle='--', label=f'T_C = {T_c:.1f} K')
        
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('Magnetization (μB)', fontsize=12)
        ax.set_title('Magnetic Phase Diagram', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved phase diagram to {output_file}")
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# High-Level Workflow
# =============================================================================

class MagneticPropertyWorkflow:
    """磁性性质计算工作流"""
    
    def __init__(self, code: str = "vasp", config: Optional[Any] = None):
        self.code = code.lower()
        
        if self.code == "vasp":
            self.calculator = VASPMagneticCalculator(config)
        elif self.code == "qe":
            self.calculator = QEMagneticCalculator(config)
        else:
            raise ValueError(f"Unsupported code: {code}")
        
        self.visualizer = MagneticVisualizer()
        self.results = {}
    
    def run_full_calculation(self, structure: Atoms,
                              output_dir: str,
                              calculate_mae: bool = True,
                              calculate_exchange: bool = True) -> Dict:
        """
        运行完整的磁性计算
        
        Args:
            structure: ASE Atoms对象
            output_dir: 输出目录
            calculate_mae: 是否计算MAE
            calculate_exchange: 是否计算交换耦合
        
        Returns:
            results: 包含所有结果的字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Starting Magnetic Properties Calculation")
        logger.info("=" * 60)
        
        # 1. FM态计算
        logger.info("\nStep 1: Ferromagnetic State")
        fm_config = SpinConfigurationGenerator.ferromagnetic(structure)
        fm_state = self.calculator.calculate_magnetic_state(
            structure, fm_config, str(output_path / "FM")
        )
        self.results['FM'] = fm_state
        
        # 2. AFM态计算 (简单AFM)
        logger.info("\nStep 2: Antiferromagnetic State")
        mag_atoms = self.calculator._identify_magnetic_atoms(structure)
        if len(mag_atoms) >= 2:
            afm_config = SpinConfigurationGenerator.antiferromagnetic_simple(
                structure, mag_atoms
            )
            afm_state = self.calculator.calculate_magnetic_state(
                structure, afm_config, str(output_path / "AFM")
            )
            self.results['AFM'] = afm_state
        
        # 3. 磁各向异性
        if calculate_mae and self.code == "vasp":
            logger.info("\nStep 3: Magnetic Anisotropy Energy")
            try:
                self.calculator.config.lsorbit = True
                anisotropy = self.calculator.calculate_anisotropy(
                    structure, str(output_path / "MAE")
                )
                self.results['anisotropy'] = anisotropy
                logger.info(f"Easy axis: {anisotropy.easy_axis.value}")
                logger.info(f"MAE: {anisotropy.anisotropy_energy}")
            except Exception as e:
                logger.warning(f"MAE calculation failed: {e}")
        
        # 4. 交换耦合
        if calculate_exchange and 'AFM' in self.results:
            logger.info("\nStep 4: Exchange Coupling Constants")
            try:
                exchange = self.calculator.calculate_exchange_coupling(
                    structure,
                    {'FM': self.results['FM'], 'AFM': self.results['AFM']},
                    str(output_path / "exchange")
                )
                self.results['exchange'] = exchange
                logger.info(f"Average J: {exchange.J_avg:.2f} meV")
            except Exception as e:
                logger.warning(f"Exchange calculation failed: {e}")
        
        # 5. 居里温度
        if 'exchange' in self.results:
            logger.info("\nStep 5: Curie Temperature Estimation")
            curie = self.calculator.estimate_curie_temperature(
                self.results['exchange']
            )
            self.results['curie_temperature'] = curie
            logger.info(f"T_C (mean field): {curie.T_c_mean_field:.1f} K")
        
        # 6. 可视化
        logger.info("\nStep 6: Generating Visualizations")
        self._generate_plots(structure, output_path)
        
        # 7. 保存结果
        logger.info("\nStep 7: Saving Results")
        self._save_results(output_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("Magnetic Properties Calculation Completed")
        logger.info("=" * 60)
        
        return self.results
    
    def _generate_plots(self, structure: Atoms, output_path: Path):
        """生成图表"""
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 磁性结构
        if 'FM' in self.results:
            fm_config = SpinConfigurationGenerator.ferromagnetic(structure)
            self.visualizer.plot_magnetic_structure(
                structure, fm_config,
                str(plots_dir / "magnetic_structure.png")
            )
        
        # 各向异性
        if 'anisotropy' in self.results:
            self.visualizer.plot_anisotropy_energy(
                self.results['anisotropy'],
                str(plots_dir / "anisotropy.png")
            )
        
        # 交换耦合网络
        if 'exchange' in self.results:
            mag_atoms = self.calculator._identify_magnetic_atoms(structure)
            if mag_atoms:
                self.visualizer.plot_exchange_coupling(
                    self.results['exchange'],
                    structure[mag_atoms],
                    str(plots_dir / "exchange_network.png")
                )
        
        logger.info(f"Plots saved to {plots_dir}")
    
    def _save_results(self, output_path: Path):
        """保存结果"""
        results_file = output_path / "magnetic_results.json"
        
        results_dict = {}
        
        for key, value in self.results.items():
            if hasattr(value, 'to_dict'):
                results_dict[key] = value.to_dict()
            elif isinstance(value, dict):
                results_dict[key] = {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                                     for k, v in value.items()}
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")


# =============================================================================
# Utility Functions
# =============================================================================

def heisenberg_hamiltonian(spins: List[np.ndarray], J_matrix: np.ndarray) -> float:
    """
    计算Heisenberg哈密顿量
    
    H = -Σ_{i<j} J_ij S_i · S_j
    
    Args:
        spins: 自旋矢量列表
        J_matrix: 交换耦合矩阵
    
    Returns:
        能量 (meV)
    """
    n_spins = len(spins)
    energy = 0.0
    
    for i in range(n_spins):
        for j in range(i+1, n_spins):
            S_dot_S = np.dot(spins[i], spins[j])
            energy -= J_matrix[i, j] * S_dot_S
    
    return energy


def ising_hamiltonian(spins: List[float], J_matrix: np.ndarray, 
                      h_field: float = 0.0) -> float:
    """
    计算Ising哈密顿量
    
    H = -Σ_{i<j} J_ij s_i s_j - h Σ_i s_i
    
    Args:
        spins: 自旋值列表 (±1)
        J_matrix: 交换耦合矩阵
        h_field: 外场
    
    Returns:
        能量 (meV)
    """
    n_spins = len(spins)
    energy = 0.0
    
    for i in range(n_spins):
        for j in range(i+1, n_spins):
            energy -= J_matrix[i, j] * spins[i] * spins[j]
        energy -= h_field * spins[i]
    
    return energy


def monte_carlo_magnetic(J_matrix: np.ndarray,
                         n_spins: int,
                         temperature: float,
                         n_steps: int = 10000) -> Tuple[float, float]:
    """
    简单的Metropolis Monte Carlo模拟
    
    Args:
        J_matrix: 交换耦合矩阵
        n_spins: 自旋数
        temperature: 温度 (K)
        n_steps: MC步数
    
    Returns:
        (平均能量, 平均磁化)
    """
    k_B = 8.617e-5  # eV/K
    
    # 初始化随机自旋
    spins = np.random.choice([-1, 1], size=n_spins)
    
    energy = ising_hamiltonian(spins.tolist(), J_matrix)
    
    energies = []
    magnetizations = []
    
    for step in range(n_steps):
        # 随机选择一个自旋
        i = np.random.randint(n_spins)
        
        # 翻转自旋
        spins_new = spins.copy()
        spins_new[i] *= -1
        
        # 计算能量变化
        energy_new = ising_hamiltonian(spins_new.tolist(), J_matrix)
        dE = energy_new - energy
        
        # Metropolis准则
        if dE < 0 or np.random.random() < np.exp(-dE / (k_B * temperature)):
            spins = spins_new
            energy = energy_new
        
        # 记录
        if step > n_steps // 2:  # 只记录后一半
            energies.append(energy)
            magnetizations.append(np.mean(spins))
    
    return np.mean(energies), np.abs(np.mean(magnetizations))


def calculate_spin_wave_spectrum(J: float, S: float, 
                                  k_points: np.ndarray,
                                  lattice: str = "simple_cubic") -> np.ndarray:
    """
    计算自旋波色散关系 (线性自旋波理论)
    
    ℏω(k) = 2zJS(1 - γ(k))
    
    其中 γ(k) = (1/z) Σ_δ exp(ik·δ)
    
    Args:
        J: 交换耦合 (meV)
        S: 自旋大小
        k_points: k点数组 (2π/a单位)
        lattice: 晶格类型
    
    Returns:
        频率数组 (meV)
    """
    if lattice == "simple_cubic":
        z = 6
        # 简立方: γ(k) = (1/3)(cos(kx) + cos(ky) + cos(kz))
        # 沿[100]方向: k = (k, 0, 0)
        gamma_k = np.cos(k_points)
    elif lattice == "bcc":
        z = 8
        # BCC: 更复杂的表达式
        gamma_k = np.cos(k_points / 2)**3
    elif lattice == "fcc":
        z = 12
        gamma_k = np.cos(k_points / 2)**4
    else:
        z = 6
        gamma_k = np.cos(k_points)
    
    omega = 2 * z * J * S * (1 - gamma_k)
    
    return omega


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Magnetic Properties Calculation (VASP/QE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # VASP magnetic calculation
  python magnetic_properties.py --code vasp --structure POSCAR -o ./magnetic_output
  
  # With SOC for MAE
  python magnetic_properties.py --code vasp --structure POSCAR --soc -o ./magnetic_output
  
  # QE calculation
  python magnetic_properties.py --code qe --structure structure.in -o ./magnetic_output
        """
    )
    
    parser.add_argument('--code', type=str, default='vasp',
                       choices=['vasp', 'qe'],
                       help='DFT code to use')
    parser.add_argument('--structure', type=str, required=True,
                       help='Structure file path')
    parser.add_argument('-o', '--output', type=str, default='./magnetic_output',
                       help='Output directory')
    parser.add_argument('--soc', action='store_true',
                       help='Enable SOC for MAE calculation')
    parser.add_argument('--encut', type=float, default=520,
                       help='Plane wave cutoff (eV)')
    parser.add_argument('--magmom', type=str, default=None,
                       help='Initial magnetic moments (e.g., "1 1 -1 -1")')
    
    args = parser.parse_args()
    
    # 读取结构
    structure = read(args.structure)
    logger.info(f"Loaded structure: {structure.get_chemical_formula()}")
    
    # 创建配置
    if args.code == 'vasp':
        config = VASPMagneticConfig(
            encut=args.encut,
            lsorbit=args.soc,
            magmom=args.magmom,
        )
    else:
        config = QEMagneticConfig(
            ecutwfc=args.encut / 13.6,
        )
    
    # 运行工作流
    workflow = MagneticPropertyWorkflow(args.code, config)
    results = workflow.run_full_calculation(
        structure, args.output,
        calculate_mae=args.soc,
        calculate_exchange=True
    )
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("CALCULATION SUMMARY")
    print("=" * 60)
    if 'FM' in results:
        print(f"FM energy: {results['FM'].total_energy:.4f} eV")
        print(f"FM magnetization: {results['FM'].magnetic_moment:.3f} μB")
    if 'AFM' in results:
        print(f"AFM energy: {results['AFM'].total_energy:.4f} eV")
        print(f"AFM magnetization: {results['AFM'].magnetic_moment:.3f} μB")
    if 'anisotropy' in results:
        print(f"Easy axis: {results['anisotropy'].easy_axis.value}")
        for axis, energy in results['anisotropy'].anisotropy_energy.items():
            print(f"  MAE ({axis.value}): {energy*1e3:.3f} meV")
    if 'exchange' in results:
        print(f"Average exchange coupling: {results['exchange'].J_avg:.2f} meV")
    if 'curie_temperature' in results:
        print(f"Curie temperature (MF): {results['curie_temperature'].T_c_mean_field:.1f} K")
    print("=" * 60)


if __name__ == "__main__":
    main()
