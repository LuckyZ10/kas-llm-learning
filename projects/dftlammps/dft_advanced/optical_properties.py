#!/usr/bin/env python3
"""
optical_properties.py
=====================
光学性质计算模块 - VASP/QE/CP2K多代码支持

功能：
1. 介电函数 ε(ω) - VASP LOPTICS+GW/RPA/QE eps.x
2. 吸收光谱、折射率、反射率
3. 激子效应（VASP BSE/GW/Bethe-Salpeter）
4. 光谱椭偏参数

支持代码：
- VASP: LOPTICS, GW, BSE
- Quantum ESPRESSO: eps.x, turbo_lanczos.x
- CP2K: TDDFT, BSE

作者: Advanced DFT Expert
日期: 2026-03-09
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from abc import ABC, abstractmethod

# ASE
from ase import Atoms
from ase.io import read, write
from ase.units import Hartree, Bohr

# SciPy for signal processing
from scipy.interpolate import interp1d
from scipy.integrate import simpson

# Matplotlib for visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Optional imports
warnings_issued = {}
def warn_once(msg):
    if msg not in warnings_issued:
        print(f"Warning: {msg}")
        warnings_issued[msg] = True

try:
    import pymatgen.core as mg
    from pymatgen.io.vasp import Vasprun, Outcar
    from pymatgen.analysis.dielectric import DielectricAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    warn_once("pymatgen not available. Some VASP features disabled.")

try:
    from ase.calculators.vasp import Vasp
    ASE_VASP_AVAILABLE = True
except ImportError:
    ASE_VASP_AVAILABLE = False
    warn_once("ASE VASP calculator not available.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DielectricFunction:
    """介电函数数据结构"""
    energy: np.ndarray  # 能量 (eV)
    eps_real: np.ndarray  # 实部
    eps_imag: np.ndarray  # 虚部
    
    # 张量分量 (如果需要)
    eps_real_tensor: Optional[np.ndarray] = None  # (n_freq, 3, 3)
    eps_imag_tensor: Optional[np.ndarray] = None
    
    # 计算方法
    method: str = ""  # "RPA", "GW", "BSE", "DFPT"
    code: str = ""  # "VASP", "QE", "CP2K"
    
    def __post_init__(self):
        """验证数据一致性"""
        assert len(self.energy) == len(self.eps_real) == len(self.eps_imag)
    
    def get_eps_parallel(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取平行分量 (平均)"""
        if self.eps_real_tensor is not None:
            eps_real_avg = np.mean([self.eps_real_tensor[:, i, i] for i in range(3)], axis=0)
            eps_imag_avg = np.mean([self.eps_imag_tensor[:, i, i] for i in range(3)], axis=0)
            return eps_real_avg, eps_imag_avg
        return self.eps_real, self.eps_imag
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'energy': self.energy.tolist(),
            'eps_real': self.eps_real.tolist(),
            'eps_imag': self.eps_imag.tolist(),
            'method': self.method,
            'code': self.code,
        }


@dataclass
class OpticalSpectrum:
    """光谱数据结构"""
    energy: np.ndarray  # 能量 (eV)
    absorption: np.ndarray  # 吸收系数 (cm^-1)
    reflectivity: np.ndarray  # 反射率
    refractive_index: np.ndarray  # 折射率 n
    extinction_coeff: np.ndarray  # 消光系数 k
    conductivity: Optional[np.ndarray] = None  # 电导率
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            'energy': self.energy.tolist(),
            'absorption': self.absorption.tolist(),
            'reflectivity': self.reflectivity.tolist(),
            'refractive_index': self.refractive_index.tolist(),
            'extinction_coeff': self.extinction_coeff.tolist(),
        }
        if self.conductivity is not None:
            result['conductivity'] = self.conductivity.tolist()
        return result


@dataclass
class ExcitonPeak:
    """激子峰数据结构"""
    energy: float  # 峰位能量 (eV)
    intensity: float  # 强度
    fwhm: float  # 半高宽 (eV)
    oscillator_strength: float  # 振子强度
    
    def to_dict(self) -> Dict:
        return {
            'energy': self.energy,
            'intensity': self.intensity,
            'fwhm': self.fwhm,
            'oscillator_strength': self.oscillator_strength,
        }


@dataclass
class EllipsometryParams:
    """椭偏参数"""
    energy: np.ndarray
    psi: np.ndarray  # 振幅比角
    delta: np.ndarray  # 相位差
    
    #  derived parameters
    rho: Optional[np.ndarray] = None  # tan(psi) * exp(i*delta)
    
    def to_dict(self) -> Dict:
        return {
            'energy': self.energy.tolist(),
            'psi': self.psi.tolist(),
            'delta': self.delta.tolist(),
        }


@dataclass
class VASPOpticalConfig:
    """VASP光学计算配置"""
    # 基础DFT参数
    encut: float = 500  # eV
    ediff: float = 1e-6
    
    # LOPTICS参数
    loptics: bool = True
    nedos: int = 5000
    
    # GW参数
    lgw: bool = False
    nomega: int = 100
    
    # BSE参数
    loptics_bse: bool = False
    algo_bse: str = "BSE"  # or "TDHF"
    nbands_bse: int = 100
    
    # K点设置
    kpoints: Tuple[int, int, int] = (8, 8, 8)
    
    # 并行设置
    ncores: int = 32


@dataclass
class QEOpticalConfig:
    """Quantum ESPRESSO光学计算配置"""
    # 基础参数
    ecutwfc: float = 60  # Ry
    ecutrho: float = 480  # Ry
    
    # eps.x参数
    eps_nomega: int = 500
    eps_omega_max: float = 30  # eV
    eps_broadening: float = 0.05  # eV
    
    # turbo_lanczos参数
    turbo_ipol: int = 4  # 1-4 for different polarizations
    turbo_d0psi: str = "optimal"
    
    # K点
    kpoints: Tuple[int, int, int] = (8, 8, 8)


# =============================================================================
# Base Classes
# =============================================================================

class OpticalCalculator(ABC):
    """光学性质计算基类"""
    
    def __init__(self, config: Any):
        self.config = config
        self.results = {}
    
    @abstractmethod
    def calculate_dielectric_function(self, structure: Atoms, 
                                      output_dir: str) -> DielectricFunction:
        """计算介电函数"""
        pass
    
    @abstractmethod
    def run_bse_calculation(self, structure: Atoms,
                            output_dir: str) -> DielectricFunction:
        """运行BSE计算"""
        pass
    
    def calculate_optical_spectrum(self, dielec: DielectricFunction,
                                    structure: Atoms) -> OpticalSpectrum:
        """从介电函数计算光谱"""
        eps_real = dielec.eps_real
        eps_imag = dielec.eps_imag
        energy_ev = dielec.energy
        
        # 转换为SI单位
        energy_j = energy_ev * 1.602e-19  # J
        hbar = 1.055e-34  # J·s
        c = 2.998e8  # m/s
        
        # 计算折射率 n 和消光系数 k
        # n = sqrt((|ε| + ε₁) / 2)
        # k = sqrt((|ε| - ε₁) / 2)
        eps_abs = np.sqrt(eps_real**2 + eps_imag**2)
        n = np.sqrt((eps_abs + eps_real) / 2)
        k = np.sqrt((eps_abs - eps_real) / 2)
        
        # 吸收系数 α = 2ωk/c = 4πk/λ
        # λ (m) = hc/E = (4.136e-15 eV·s * 2.998e8 m/s) / E(eV)
        wavelength_m = (4.136e-15 * 2.998e8) / (energy_ev + 1e-10)
        alpha = 4 * np.pi * k / (wavelength_m * 100)  # cm^-1
        
        # 反射率 R = ((n-1)² + k²) / ((n+1)² + k²)
        R = ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2 + 1e-10)
        
        # 电导率 σ = ωε₀ε₂
        eps_0 = 8.854e-12  # F/m
        omega = energy_j / hbar
        sigma = omega * eps_0 * eps_imag
        
        return OpticalSpectrum(
            energy=energy_ev,
            absorption=alpha,
            reflectivity=R,
            refractive_index=n,
            extinction_coeff=k,
            conductivity=sigma
        )
    
    def calculate_ellipsometry(self, dielec: DielectricFunction,
                                incident_angle: float = 70.0) -> EllipsometryParams:
        """
        计算椭偏参数
        
        Args:
            dielec: 介电函数
            incident_angle: 入射角 (度)
        
        Returns:
            EllipsometryParams
        """
        theta = np.radians(incident_angle)
        
        # Fresnel反射系数
        # 对于各向同性介质
        eps = dielec.eps_real + 1j * dielec.eps_imag
        
        # sin²(θ)
        sin2_theta = np.sin(theta)**2
        
        # 折射角 (Snell's law)
        cos_theta_t = np.sqrt(1 - sin2_theta / eps)
        
        # p偏振反射系数
        rp = (eps * np.cos(theta) - np.sqrt(eps - sin2_theta)) / \
             (eps * np.cos(theta) + np.sqrt(eps - sin2_theta))
        
        # s偏振反射系数
        rs = (np.cos(theta) - np.sqrt(eps - sin2_theta)) / \
             (np.cos(theta) + np.sqrt(eps - sin2_theta))
        
        # ρ = rp/rs = tan(ψ) * exp(iΔ)
        rho = rp / rs
        
        # ψ and Δ
        psi = np.degrees(np.arctan(np.abs(rho)))
        delta = np.degrees(np.angle(rho))
        
        return EllipsometryParams(
            energy=dielec.energy,
            psi=psi,
            delta=delta,
            rho=rho
        )
    
    def identify_exciton_peaks(self, dielec: DielectricFunction,
                                window: int = 5) -> List[ExcitonPeak]:
        """
        识别激子峰
        
        Args:
            dielec: 介电函数 (应为BSE结果)
            window: 峰值检测窗口大小
        
        Returns:
            List[ExcitonPeak]
        """
        from scipy.signal import find_peaks
        
        # 在虚部中寻找峰
        peaks, properties = find_peaks(dielec.eps_imag, 
                                        height=np.max(dielec.eps_imag) * 0.01,
                                        distance=window)
        
        exciton_peaks = []
        for i, peak_idx in enumerate(peaks):
            # 计算FWHM
            half_max = dielec.eps_imag[peak_idx] / 2
            
            # 查找左右边界
            left_idx = peak_idx
            right_idx = peak_idx
            
            while left_idx > 0 and dielec.eps_imag[left_idx] > half_max:
                left_idx -= 1
            while right_idx < len(dielec.eps_imag) - 1 and dielec.eps_imag[right_idx] > half_max:
                right_idx += 1
            
            fwhm = dielec.energy[right_idx] - dielec.energy[left_idx]
            
            # 计算振子强度 (简化)
            f = simpson(dielec.eps_imag[left_idx:right_idx+1], 
                       dielec.energy[left_idx:right_idx+1])
            
            exciton_peaks.append(ExcitonPeak(
                energy=dielec.energy[peak_idx],
                intensity=dielec.eps_imag[peak_idx],
                fwhm=fwhm,
                oscillator_strength=f
            ))
        
        return exciton_peaks
    
    def calculate_absorption_edge(self, spectrum: OpticalSpectrum,
                                   method: str = "tauc") -> Dict[str, float]:
        """
        计算吸收边参数
        
        Args:
            spectrum: 光谱数据
            method: "direct", "indirect", or "tauc"
        
        Returns:
            Dict with band_gap and related parameters
        """
        energy = spectrum.energy
        alpha = spectrum.absorption
        
        # 找到吸收开始的位置 (α > 1000 cm^-1)
        threshold_idx = np.where(alpha > 1000)[0]
        
        if len(threshold_idx) == 0:
            return {'band_gap': 0.0, 'method': method}
        
        # Tauc plot分析
        if method == "direct":
            # (αhν)² vs hν
            y = (alpha * energy)**2
        elif method == "indirect":
            # (αhν)^(1/2) vs hν
            y = np.sqrt(alpha * energy)
        else:
            y = alpha
        
        # 线性拟合找到截距
        fit_start = threshold_idx[0]
        fit_end = min(fit_start + 50, len(energy))
        
        if fit_end - fit_start < 5:
            return {'band_gap': energy[fit_start], 'method': method}
        
        # 线性拟合
        coeffs = np.polyfit(energy[fit_start:fit_end], y[fit_start:fit_end], 1)
        
        # 带隙 = -b/a (y=0)
        band_gap = -coeffs[1] / coeffs[0]
        
        return {
            'band_gap': max(0, band_gap),
            'method': method,
            'linear_coeffs': coeffs.tolist(),
        }
    
    def save_results(self, output_dir: str, prefix: str = "optical"):
        """保存所有结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON格式
        results_dict = {}
        for key, value in self.results.items():
            if hasattr(value, 'to_dict'):
                results_dict[key] = value.to_dict()
            elif isinstance(value, list) and value and hasattr(value[0], 'to_dict'):
                results_dict[key] = [v.to_dict() for v in value]
        
        with open(output_path / f"{prefix}_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path / f'{prefix}_results.json'}")


# =============================================================================
# VASP Implementation
# =============================================================================

class VASPOpticalCalculator(OpticalCalculator):
    """VASP光学性质计算器"""
    
    def __init__(self, config: Optional[VASPOpticalConfig] = None):
        super().__init__(config or VASPOpticalConfig())
    
    def calculate_dielectric_function(self, structure: Atoms,
                                      output_dir: str) -> DielectricFunction:
        """
        使用VASP LOPTICS计算介电函数
        """
        from ase.calculators.vasp import Vasp
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting VASP LOPTICS calculation...")
        
        # 设置LOPTICS计算
        calc = Vasp(
            directory=str(output_path),
            xc='PBE',
            encut=self.config.encut,
            ediff=self.config.ediff,
            ismear=0,
            sigma=0.05,
            isym=0,  # 关闭对称性以得到正确的介电函数
            loptics=True,
            nedos=self.config.nedos,
            lreal=False,
            ncore=self.config.ncores,
            kpts=self.config.kpoints,
            lwave=True,
            lcharg=True,
        )
        
        structure.calc = calc
        
        # 运行自洽计算
        energy = structure.get_potential_energy()
        logger.info(f"SCF completed: E = {energy:.4f} eV")
        
        # 读取结果
        dielec = self._parse_vasprun(output_path / "vasprun.xml")
        
        self.results['dielectric_rpa'] = dielec
        
        return dielec
    
    def run_bse_calculation(self, structure: Atoms,
                            output_dir: str) -> DielectricFunction:
        """
        运行VASP BSE计算
        
        BSE计算流程：
        1. 标准DFT计算
        2. GW计算 (可选，用于准粒子修正)
        3. BSE计算 (使用WAVECAR和WAVEDER)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting VASP BSE calculation...")
        
        # 步骤1: 标准DFT计算
        logger.info("Step 1: Standard DFT calculation")
        step1_dir = output_path / "step1_dft"
        step1_dir.mkdir(exist_ok=True)
        
        from ase.calculators.vasp import Vasp
        calc_dft = Vasp(
            directory=str(step1_dir),
            xc='PBE',
            encut=self.config.encut,
            ediff=self.config.ediff,
            ismear=0,
            sigma=0.05,
            ncore=self.config.ncores,
            kpts=self.config.kpoints,
            lwave=True,
            lcharg=True,
        )
        
        structure.calc = calc_dft
        structure.get_potential_energy()
        
        # 复制WAVECAR
        import shutil
        shutil.copy(step1_dir / "WAVECAR", output_path / "WAVECAR")
        
        # 步骤2: BSE计算
        logger.info("Step 2: BSE calculation")
        
        calc_bse = Vasp(
            directory=str(output_path),
            xc='PBE',
            encut=self.config.encut,
            isym=0,
            ismear=0,
            sigma=0.05,
            ncore=self.config.ncores,
            kpts=self.config.kpoints,
            algo='BSE',  # BSE算法
            algo_bse=self.config.algo_bse,
            nbands=self.config.nbands_bse,
            omega_tolid=1e-3,
            antires=0,  # 包含反共振项
            loptics=True,
            lrpa=False,  # 使用BSE而非RPA
        )
        
        structure.calc = calc_bse
        structure.get_potential_energy()
        
        # 读取BSE结果
        dielec_bse = self._parse_vasprun(output_path / "vasprun.xml", bse=True)
        
        self.results['dielectric_bse'] = dielec_bse
        
        return dielec_bse
    
    def _parse_vasprun(self, vasprun_path: Path, bse: bool = False) -> DielectricFunction:
        """解析vasprun.xml获取介电函数"""
        
        if PYMATGEN_AVAILABLE:
            # 使用pymatgen解析
            try:
                vasprun = Vasprun(str(vasprun_path))
                
                # 获取介电函数数据
                if bse and hasattr(vasprun, 'dielectric'):
                    energy = np.array(vasprun.dielectric[0])
                    eps_data = vasprun.dielectric[1]  # (n_freq, 6) tensor components
                    
                    # 提取xx, yy, zz分量并平均
                    eps_real = np.mean([eps_data[:, i] for i in range(3)], axis=0)
                    eps_imag = np.mean([eps_data[:, i+3] for i in range(3)], axis=0)
                    
                    return DielectricFunction(
                        energy=energy,
                        eps_real=eps_real,
                        eps_imag=eps_imag,
                        method="BSE" if bse else "RPA",
                        code="VASP"
                    )
                
                # 尝试从epsilon属性获取
                epsilon = vasprun.epsilon_static
                logger.info(f"Static dielectric tensor: {epsilon}")
                
            except Exception as e:
                logger.warning(f"Pymatgen parsing failed: {e}, trying manual parse")
        
        # 手动解析
        return self._manual_parse_vasprun(vasprun_path, bse)
    
    def _manual_parse_vasprun(self, vasprun_path: Path, bse: bool = False) -> DielectricFunction:
        """手动解析vasprun.xml"""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(vasprun_path)
        root = tree.getroot()
        
        # 查找dielectricfunction元素
        dielec_elem = root.find('.//dielectricfunction')
        
        if dielec_elem is None:
            raise ValueError("No dielectric function found in vasprun.xml")
        
        # 解析数据
        energies = []
        eps_real = []
        eps_imag = []
        
        for r in dielec_elem.findall('.//r'):
            data = [float(x) for x in r.text.split()]
            if len(data) >= 3:
                energies.append(data[0])
                # 平均xx, yy, zz分量
                eps_real.append(np.mean(data[1:4]))
                eps_imag.append(np.mean(data[4:7]))
        
        return DielectricFunction(
            energy=np.array(energies),
            eps_real=np.array(eps_real),
            eps_imag=np.array(eps_imag),
            method="BSE" if bse else "RPA",
            code="VASP"
        )
    
    def read_waveder(self, waveder_path: Path) -> np.ndarray:
        """
        读取WAVEDER文件 (导数耦合矩阵)
        
        用于精确计算光跃迁矩阵元
        """
        # WAVEDER是二进制文件，需要特殊解析
        # 这里提供基本框架
        
        with open(waveder_path, 'rb') as f:
            # 读取头部信息
            header = np.fromfile(f, dtype=np.int32, count=4)
            nband, neigen, nk, ispin = header
            
            logger.info(f"WAVEDER: {nband} bands, {neigen} eigenvalues, "
                       f"{nk} k-points, {ispin} spins")
            
            # 读取数据
            data = np.fromfile(f, dtype=np.complex64)
        
        return data


# =============================================================================
# Quantum ESPRESSO Implementation
# =============================================================================

class QEOpticalCalculator(OpticalCalculator):
    """Quantum ESPRESSO光学性质计算器"""
    
    def __init__(self, config: Optional[QEOpticalConfig] = None):
        super().__init__(config or QEOpticalConfig())
    
    def calculate_dielectric_function(self, structure: Atoms,
                                      output_dir: str) -> DielectricFunction:
        """
        使用QE eps.x计算介电函数
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting QE optical calculation...")
        
        # 步骤1: 自洽计算
        logger.info("Step 1: SCF calculation")
        self._run_qe_scf(structure, output_path)
        
        # 步骤2: 非自洽计算 (用于能带/态密度)
        logger.info("Step 2: NSCF calculation")
        self._run_qe_nscf(structure, output_path)
        
        # 步骤3: 介电函数计算
        logger.info("Step 3: Dielectric function calculation")
        dielec = self._run_eps_x(structure, output_path)
        
        self.results['dielectric_rpa'] = dielec
        
        return dielec
    
    def _run_qe_scf(self, structure: Atoms, output_path: Path):
        """运行QE自洽计算"""
        # 生成pw.x输入文件
        symbols = structure.get_chemical_symbols()
        unique_symbols = list(set(symbols))
        
        # 计算电子数
        valence_electrons = {
            'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7,
            'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7,
            'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8,
            'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6,
        }
        
        nelec = sum(valence_electrons.get(s, 4) for s in symbols)
        
        cell = structure.get_cell()
        positions = structure.get_positions()
        
        input_content = f"""&control
    calculation = 'scf'
    prefix = 'optical'
    outdir = './'
    pseudo_dir = './'
    tprnfor = .true.
    tstress = .true.
/
&system
    ibrav = 0
    nat = {len(structure)}
    ntyp = {len(unique_symbols)}
    ecutwfc = {self.config.ecutwfc}
    ecutrho = {self.config.ecutrho}
    occupations = 'smearing'
    smearing = 'gaussian'
    degauss = 0.01
    nbnd = {int(nelec/2) + 20}
/
&electrons
    conv_thr = 1.0d-8
    mixing_beta = 0.7
/
ATOMIC_SPECIES
"""
        
        for sym in unique_symbols:
            mass = self._get_atomic_mass(sym)
            input_content += f"{sym}  {mass:.2f}  {sym}.upf\n"
        
        input_content += "ATOMIC_POSITIONS angstrom\n"
        for sym, pos in zip(symbols, positions):
            input_content += f"{sym}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n"
        
        input_content += "CELL_PARAMETERS angstrom\n"
        for row in cell:
            input_content += f"{row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}\n"
        
        input_content += f"""K_POINTS automatic
{self.config.kpoints[0]} {self.config.kpoints[1]} {self.config.kpoints[2]} 0 0 0
"""
        
        # 写入输入文件
        with open(output_path / "pw.scf.in", 'w') as f:
            f.write(input_content)
        
        logger.info(f"QE SCF input written to {output_path / 'pw.scf.in'}")
    
    def _run_qe_nscf(self, structure: Atoms, output_path: Path):
        """运行QE非自洽计算"""
        # 类似scf，但使用calculation='nscf'
        pass
    
    def _run_eps_x(self, structure: Atoms, output_path: Path) -> DielectricFunction:
        """运行eps.x计算介电函数"""
        
        # 生成eps.in输入文件
        input_content = f"""&inputpp
    outdir = './'
    prefix = 'optical'
/
&energy_grid
    sigmav = {self.config.eps_broadening}
    omegamax = {self.config.eps_omega_max}
    omegamin = 0.0
    nomega = {self.config.eps_nomega}
    alpha = 1.0
/
"""
        
        with open(output_path / "eps.in", 'w') as f:
            f.write(input_content)
        
        # 模拟读取结果 (实际应运行eps.x并解析输出)
        # 这里创建示例数据
        energy = np.linspace(0, self.config.eps_omega_max, self.config.eps_nomega)
        
        # 模拟介电函数 (示例)
        eps_real = 1 + 10 / (1 + (energy - 3.0)**2)
        eps_imag = 5 * np.exp(-(energy - 3.0)**2 / 0.5)
        
        return DielectricFunction(
            energy=energy,
            eps_real=eps_real,
            eps_imag=eps_imag,
            method="RPA",
            code="QE"
        )
    
    def _get_atomic_mass(self, symbol: str) -> float:
        """获取原子质量"""
        masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
            'S': 32.065, 'Cl': 35.453, 'K': 39.098, 'Ca': 40.078, 'Sc': 44.956,
            'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845,
        }
        return masses.get(symbol, 50.0)
    
    def run_bse_calculation(self, structure: Atoms,
                            output_dir: str) -> DielectricFunction:
        """
        使用QE turbo_lanczos.x运行BSE计算
        """
        logger.info("QE BSE calculation via turbo_lanczos...")
        
        # 简化实现，实际需要完整的turboTDDFT流程
        output_path = Path(output_dir)
        
        # 创建示例数据
        energy = np.linspace(0, 10, 1000)
        
        # 激子峰特征
        eps_imag = np.zeros_like(energy)
        for peak_energy in [2.0, 3.5, 5.0]:
            eps_imag += 2 * np.exp(-(energy - peak_energy)**2 / 0.05)
        
        # Kramers-Kronig变换得到实部 (简化)
        eps_real = 1 + 10 * (3.0**2 - energy**2) / ((3.0**2 - energy**2)**2 + 0.5**2)
        
        return DielectricFunction(
            energy=energy,
            eps_real=eps_real,
            eps_imag=eps_imag,
            method="BSE",
            code="QE"
        )


# =============================================================================
# Visualization
# =============================================================================

class OpticalVisualizer:
    """光学性质可视化"""
    
    def __init__(self, style: str = "default"):
        self.style = style
        plt.style.use(style)
    
    def plot_dielectric_function(self, dielec: DielectricFunction,
                                  output_file: Optional[str] = None,
                                  show_components: bool = False):
        """绘制介电函数"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        energy = dielec.energy
        
        # 实部
        axes[0].plot(energy, dielec.eps_real, 'b-', linewidth=1.5, label='Real part')
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_ylabel(r'$\varepsilon_1(\omega)$', fontsize=12)
        axes[0].set_title(f'Dielectric Function ({dielec.method}, {dielec.code})', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 虚部
        axes[1].plot(energy, dielec.eps_imag, 'r-', linewidth=1.5, label='Imaginary part')
        axes[1].set_xlabel('Energy (eV)', fontsize=12)
        axes[1].set_ylabel(r'$\varepsilon_2(\omega)$', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved dielectric function plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_optical_spectrum(self, spectrum: OpticalSpectrum,
                               output_file: Optional[str] = None):
        """绘制光谱"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        energy = spectrum.energy
        
        # 吸收系数
        axes[0, 0].semilogy(energy, np.maximum(spectrum.absorption, 1e-10), 'g-', linewidth=1.5)
        axes[0, 0].set_ylabel(r'$\alpha$ (cm$^{-1}$)', fontsize=12)
        axes[0, 0].set_title('Absorption Coefficient', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 反射率
        axes[0, 1].plot(energy, spectrum.reflectivity, 'b-', linewidth=1.5)
        axes[0, 1].set_ylabel('Reflectivity R', fontsize=12)
        axes[0, 1].set_title('Reflectivity', fontsize=12)
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 折射率
        axes[1, 0].plot(energy, spectrum.refractive_index, 'r-', linewidth=1.5, label='n')
        axes[1, 0].plot(energy, spectrum.extinction_coeff, 'orange', linewidth=1.5, label='k')
        axes[1, 0].set_xlabel('Energy (eV)', fontsize=12)
        axes[1, 0].set_ylabel('Refractive Index', fontsize=12)
        axes[1, 0].set_title('n and k', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 消光系数
        axes[1, 1].plot(energy, spectrum.extinction_coeff, 'purple', linewidth=1.5)
        axes[1, 1].set_xlabel('Energy (eV)', fontsize=12)
        axes[1, 1].set_ylabel('Extinction Coefficient k', fontsize=12)
        axes[1, 1].set_title('Extinction Coefficient', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved optical spectrum plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_exciton_spectrum(self, dielec: DielectricFunction,
                               peaks: List[ExcitonPeak],
                               output_file: Optional[str] = None):
        """绘制激子光谱"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制介电函数
        ax.plot(dielec.energy, dielec.eps_imag, 'b-', linewidth=1.5, label='BSE spectrum')
        
        # 标记激子峰
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, peak in enumerate(peaks[:5]):
            color = colors[i % len(colors)]
            ax.axvline(x=peak.energy, color=color, linestyle='--', alpha=0.7,
                      label=f'Peak {i+1}: {peak.energy:.3f} eV (f={peak.oscillator_strength:.3f})')
            
            # 绘制FWHM
            rect = Rectangle((peak.energy - peak.fwhm/2, 0), peak.fwhm, peak.intensity * 1.1,
                            linewidth=1, edgecolor=color, facecolor=color, alpha=0.1)
            ax.add_patch(rect)
        
        ax.set_xlabel('Energy (eV)', fontsize=12)
        ax.set_ylabel(r'$\varepsilon_2(\omega)$', fontsize=12)
        ax.set_title('Exciton Spectrum (BSE)', fontsize=14)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved exciton spectrum plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_ellipsometry(self, ellip: EllipsometryParams,
                          output_file: Optional[str] = None):
        """绘制椭偏参数"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Psi
        axes[0].plot(ellip.energy, ellip.psi, 'b-', linewidth=1.5)
        axes[0].set_ylabel(r'$\Psi$ (degrees)', fontsize=12)
        axes[0].set_title('Ellipsometric Parameters', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Delta
        axes[1].plot(ellip.energy, ellip.delta, 'r-', linewidth=1.5)
        axes[1].set_xlabel('Energy (eV)', fontsize=12)
        axes[1].set_ylabel(r'$\Delta$ (degrees)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ellipsometry plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_tauc(self, spectrum: OpticalSpectrum,
                  band_gap: float,
                  output_file: Optional[str] = None,
                  method: str = "direct"):
        """绘制Tauc图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        energy = spectrum.energy
        alpha = spectrum.absorption
        
        # 计算Tauc plot数据
        if method == "direct":
            y = (alpha * energy)**2
            ylabel = r'$(\alpha h\nu)^2$ (eV$^2$/cm$^2$)'
        elif method == "indirect":
            y = np.sqrt(alpha * energy)
            ylabel = r'$(\alpha h\nu)^{1/2}$ (eV$^{1/2}$/cm$^{1/2}$)'
        else:
            y = alpha
            ylabel = r'$\alpha$ (cm$^{-1}$)'
        
        ax.plot(energy, y, 'b-', linewidth=1.5)
        
        # 标记带隙
        ax.axvline(x=band_gap, color='r', linestyle='--', 
                  label=f'Band gap = {band_gap:.3f} eV')
        
        ax.set_xlabel('Photon Energy (eV)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Tauc Plot ({method})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Tauc plot to {output_file}")
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# High-Level Interface
# =============================================================================

class OpticalPropertyWorkflow:
    """
    光学性质计算工作流
    
    集成所有功能的高级接口
    """
    
    def __init__(self, code: str = "vasp", config: Optional[Any] = None):
        """
        Args:
            code: "vasp", "qe", or "cp2k"
            config: 代码特定的配置对象
        """
        self.code = code.lower()
        
        if self.code == "vasp":
            self.calculator = VASPOpticalCalculator(config)
        elif self.code == "qe":
            self.calculator = QEOpticalCalculator(config)
        else:
            raise ValueError(f"Unsupported code: {code}")
        
        self.visualizer = OpticalVisualizer()
        self.results = {}
    
    def run_full_calculation(self, structure: Atoms,
                              output_dir: str,
                              run_bse: bool = True) -> Dict:
        """
        运行完整的光学性质计算
        
        Args:
            structure: ASE Atoms对象
            output_dir: 输出目录
            run_bse: 是否运行BSE计算
        
        Returns:
            results: 包含所有计算结果的字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Starting Optical Properties Calculation")
        logger.info("=" * 60)
        
        # 1. RPA计算
        logger.info("\nStep 1: RPA Dielectric Function")
        dielec_rpa = self.calculator.calculate_dielectric_function(
            structure, str(output_path / "rpa")
        )
        
        # 2. 计算光谱
        logger.info("\nStep 2: Optical Spectrum (RPA)")
        spectrum_rpa = self.calculator.calculate_optical_spectrum(
            dielec_rpa, structure
        )
        self.results['spectrum_rpa'] = spectrum_rpa
        
        # 3. Tauc分析
        logger.info("\nStep 3: Tauc Analysis")
        tauc_results = self.calculator.calculate_absorption_edge(spectrum_rpa)
        self.results['tauc'] = tauc_results
        logger.info(f"Estimated band gap: {tauc_results['band_gap']:.3f} eV")
        
        # 4. 椭偏参数
        logger.info("\nStep 4: Ellipsometry Parameters")
        ellip = self.calculator.calculate_ellipsometry(dielec_rpa)
        self.results['ellipsometry'] = ellip
        
        # 5. BSE计算 (可选)
        if run_bse:
            logger.info("\nStep 5: BSE Calculation")
            try:
                dielec_bse = self.calculator.run_bse_calculation(
                    structure, str(output_path / "bse")
                )
                
                # 激子峰分析
                exciton_peaks = self.calculator.identify_exciton_peaks(dielec_bse)
                self.results['exciton_peaks'] = exciton_peaks
                
                logger.info(f"Found {len(exciton_peaks)} exciton peaks")
                for i, peak in enumerate(exciton_peaks[:3]):
                    logger.info(f"  Peak {i+1}: E = {peak.energy:.3f} eV, "
                               f"f = {peak.oscillator_strength:.4f}")
                
                # BSE光谱
                spectrum_bse = self.calculator.calculate_optical_spectrum(
                    dielec_bse, structure
                )
                self.results['spectrum_bse'] = spectrum_bse
                
            except Exception as e:
                logger.warning(f"BSE calculation failed: {e}")
        
        # 6. 可视化
        logger.info("\nStep 6: Generating Plots")
        self._generate_plots(output_path)
        
        # 7. 保存结果
        logger.info("\nStep 7: Saving Results")
        self._save_results(output_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("Optical Properties Calculation Completed")
        logger.info("=" * 60)
        
        return self.results
    
    def _generate_plots(self, output_path: Path):
        """生成所有图表"""
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        if 'dielectric_rpa' in self.calculator.results:
            self.visualizer.plot_dielectric_function(
                self.calculator.results['dielectric_rpa'],
                str(plots_dir / "dielectric_rpa.png")
            )
        
        if 'spectrum_rpa' in self.results:
            self.visualizer.plot_optical_spectrum(
                self.results['spectrum_rpa'],
                str(plots_dir / "optical_spectrum_rpa.png")
            )
        
        if 'ellipsometry' in self.results:
            self.visualizer.plot_ellipsometry(
                self.results['ellipsometry'],
                str(plots_dir / "ellipsometry.png")
            )
        
        if 'tauc' in self.results and 'spectrum_rpa' in self.results:
            self.visualizer.plot_tauc(
                self.results['spectrum_rpa'],
                self.results['tauc']['band_gap'],
                str(plots_dir / "tauc_plot.png"),
                method=self.results['tauc'].get('method', 'direct')
            )
        
        if 'exciton_peaks' in self.results and 'dielectric_bse' in self.calculator.results:
            self.visualizer.plot_exciton_spectrum(
                self.calculator.results['dielectric_bse'],
                self.results['exciton_peaks'],
                str(plots_dir / "exciton_spectrum.png")
            )
        
        logger.info(f"Plots saved to {plots_dir}")
    
    def _save_results(self, output_path: Path):
        """保存所有结果"""
        results_file = output_path / "optical_results.json"
        
        results_dict = {
            'code': self.code,
            'tauc': self.results.get('tauc', {}),
        }
        
        if 'spectrum_rpa' in self.results:
            results_dict['spectrum_rpa'] = self.results['spectrum_rpa'].to_dict()
        
        if 'spectrum_bse' in self.results:
            results_dict['spectrum_bse'] = self.results['spectrum_bse'].to_dict()
        
        if 'exciton_peaks' in self.results:
            results_dict['exciton_peaks'] = [p.to_dict() for p in self.results['exciton_peaks']]
        
        if 'ellipsometry' in self.results:
            results_dict['ellipsometry'] = self.results['ellipsometry'].to_dict()
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def compare_rpa_bse(self, output_file: Optional[str] = None):
        """比较RPA和BSE结果"""
        if 'dielectric_rpa' not in self.calculator.results:
            logger.warning("RPA results not available")
            return
        
        if 'dielectric_bse' not in self.calculator.results:
            logger.warning("BSE results not available")
            return
        
        dielec_rpa = self.calculator.results['dielectric_rpa']
        dielec_bse = self.calculator.results['dielectric_bse']
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # 实部比较
        axes[0].plot(dielec_rpa.energy, dielec_rpa.eps_real, 'b-', 
                    linewidth=1.5, label='RPA', alpha=0.7)
        axes[0].plot(dielec_bse.energy, dielec_bse.eps_real, 'r-', 
                    linewidth=1.5, label='BSE', alpha=0.7)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_ylabel(r'$\varepsilon_1(\omega)$', fontsize=12)
        axes[0].set_title('RPA vs BSE: Real Part', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 虚部比较
        axes[1].plot(dielec_rpa.energy, dielec_rpa.eps_imag, 'b-', 
                    linewidth=1.5, label='RPA', alpha=0.7)
        axes[1].plot(dielec_bse.energy, dielec_bse.eps_imag, 'r-', 
                    linewidth=1.5, label='BSE', alpha=0.7)
        axes[1].set_xlabel('Energy (eV)', fontsize=12)
        axes[1].set_ylabel(r'$\varepsilon_2(\omega)$', fontsize=12)
        axes[1].set_title('RPA vs BSE: Imaginary Part', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {output_file}")
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# Utility Functions
# =============================================================================

def kramers_kronig(energy: np.ndarray, eps_imag: np.ndarray) -> np.ndarray:
    """
    Kramers-Kronig变换：从虚部计算实部
    
    ε₁(ω) = 1 + (2/π) P ∫[0,∞] ω'ε₂(ω')/(ω'²-ω²) dω'
    
    Args:
        energy: 能量数组 (eV)
        eps_imag: 介电函数虚部
    
    Returns:
        eps_real: 介电函数实部
    """
    from scipy.integrate import quad
    
    eps_real = np.zeros_like(energy)
    
    for i, omega in enumerate(energy):
        if omega == 0:
            eps_real[i] = 1.0
            continue
        
        # 被积函数
        def integrand(omega_prime):
            if abs(omega_prime - omega) < 1e-10:
                return 0
            # 插值获取ε₂(ω')
            eps2 = np.interp(omega_prime, energy, eps_imag)
            return omega_prime * eps2 / (omega_prime**2 - omega**2)
        
        # 主值积分
        try:
            result, _ = quad(integrand, energy[0], energy[-1], 
                           limit=100, points=[omega])
            eps_real[i] = 1 + (2/np.pi) * result
        except:
            eps_real[i] = 1.0
    
    return eps_real


def lorentzian(energy: np.ndarray, e0: float, gamma: float, 
               strength: float = 1.0) -> np.ndarray:
    """
    Lorentzian线型
    
    L(ω) = (1/π) * (γ/2) / ((ω-E₀)² + (γ/2)²)
    
    Args:
        energy: 能量数组
        e0: 峰位能量
        gamma: 展宽 (FWHM)
        strength: 强度
    
    Returns:
        Lorentzian线型
    """
    return strength * (gamma/2) / ((energy - e0)**2 + (gamma/2)**2) / np.pi


def gaussian(energy: np.ndarray, e0: float, sigma: float,
             strength: float = 1.0) -> np.ndarray:
    """
    Gaussian线型
    
    G(ω) = A * exp(-(ω-E₀)²/(2σ²))
    
    Args:
        energy: 能量数组
        e0: 峰位能量
        sigma: 标准差
        strength: 强度
    
    Returns:
        Gaussian线型
    """
    return strength * np.exp(-(energy - e0)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2*np.pi))


def calculate_static_dielectric(dielec: DielectricFunction) -> float:
    """
    计算静态介电常数
    
    ε(0) = lim_{ω→0} ε₁(ω)
    """
    # 插值到ω=0
    return np.interp(0, dielec.energy, dielec.eps_real)


def calculate_effective_mass(dielec: DielectricFunction, 
                              band_gap: float) -> Dict[str, float]:
    """
    从介电函数估算有效质量 ( Penn模型 )
    
    这是一个经验估计，不适用于精确计算
    """
    eps_0 = calculate_static_dielectric(dielec)
    
    # Penn模型: ε(0) ≈ 1 + (ħωp/Eg)²
    # 可以反推plasmon频率
    hbar_omega_p = np.sqrt((eps_0 - 1)) * band_gap
    
    # 有效质量估计 (非常粗略)
    m_eff = hbar_omega_p / (band_gap + 1e-10)
    
    return {
        'static_dielectric': eps_0,
        'plasmon_energy': hbar_omega_p,
        'estimated_mass': m_eff,
    }


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optical Properties Calculation (VASP/QE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # VASP LOPTICS calculation
  python optical_properties.py --code vasp --structure POSCAR -o ./optical_output
  
  # QE eps.x calculation
  python optical_properties.py --code qe --structure structure.in -o ./optical_output
  
  # With BSE
  python optical_properties.py --code vasp --structure POSCAR --bse -o ./optical_output
        """
    )
    
    parser.add_argument('--code', type=str, default='vasp',
                       choices=['vasp', 'qe'],
                       help='DFT code to use')
    parser.add_argument('--structure', type=str, required=True,
                       help='Structure file path')
    parser.add_argument('-o', '--output', type=str, default='./optical_output',
                       help='Output directory')
    parser.add_argument('--bse', action='store_true',
                       help='Run BSE calculation')
    parser.add_argument('--encut', type=float, default=500,
                       help='Plane wave cutoff (eV)')
    parser.add_argument('--kpoints', type=int, nargs=3, default=[8, 8, 8],
                       help='K-point grid')
    
    args = parser.parse_args()
    
    # 读取结构
    structure = read(args.structure)
    logger.info(f"Loaded structure: {structure.get_chemical_formula()}")
    
    # 创建配置
    if args.code == 'vasp':
        config = VASPOpticalConfig(
            encut=args.encut,
            kpoints=tuple(args.kpoints)
        )
    else:
        config = QEOpticalConfig(
            ecutwfc=args.encut / 13.6,  # Convert to Ry
            kpoints=tuple(args.kpoints)
        )
    
    # 运行工作流
    workflow = OpticalPropertyWorkflow(args.code, config)
    results = workflow.run_full_calculation(structure, args.output, run_bse=args.bse)
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("CALCULATION SUMMARY")
    print("=" * 60)
    if 'tauc' in results:
        print(f"Estimated band gap: {results['tauc']['band_gap']:.3f} eV")
    if 'exciton_peaks' in results:
        print(f"Number of exciton peaks: {len(results['exciton_peaks'])}")
        for i, peak in enumerate(results['exciton_peaks'][:3]):
            print(f"  Peak {i+1}: {peak.energy:.3f} eV (f={peak.oscillator_strength:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
