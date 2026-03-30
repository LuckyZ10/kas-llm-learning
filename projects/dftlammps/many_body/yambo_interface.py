#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yambo GW-BSE Interface Module
============================

GW近似和Bethe-Salpeter方程(BSE)计算模块，用于精确计算
准粒子能带结构和激子光学性质。

功能包括:
- G0W0单发计算
- evGW自洽GW计算
- BSE激子能级和波函数
- 吸收光谱计算
- 激子结合能分析

依赖: numpy, scipy, ase, subprocess
"""

import os
import re
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

# 可选依赖处理
try:
    from ase import Atoms
    from ase.io import read, write
    from ase.units import Hartree, Bohr
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    warnings.warn("ASE not available. Structure handling limited.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class GWParameters:
    """GW计算参数"""
    # 基本参数
    qp_band_range: Tuple[int, int] = (1, 10)  # 准粒子能带范围
    kpoints: np.ndarray = field(default_factory=lambda: np.array([[0, 0, 0]]))
    
    # GW参数
    gw_approximation: str = "G0W0"  # G0W0, evGW, scGW
    screened_coulomb: str = "RPA"  # RPA, TDDFT
    
    # 截断和收敛
    cutoff: float = 10.0  # Ry, 平面波截断
    dielectric_cutoff: float = 5.0  # Ry, 介电函数截断
    n_empty_bands: int = 100  # 空带数
    
    # 频率积分
    freq_grid: str = "cd"  # cd (contour deformation), pade, pl
    n_freq: int = 100  # 频率点数
    freq_max: float = 10.0  # eV, 最大频率
    
    # 自洽选项
    max_sc_iterations: int = 5
    sc_tolerance: float = 1e-5  # eV
    update_energies_only: bool = True  # evGW仅更新能量
    
    # 交换-关联
    xc_functional: str = "PBE"
    vxc_offdiag: bool = True  # 包含Vxc非对角元
    
    def to_yambo_input(self) -> str:
        """生成Yambo输入字符串"""
        lines = [
            "# GW Parameters",
            f"% QPkrange",
            f" {self.qp_band_range[0]} | {self.qp_band_range[1]} | 1 | {len(self.kpoints)} |",
            "%",
            f"Cutoff={self.cutoff} Ry",
            f"NGsBlkXd={self.dielectric_cutoff} RL",
            f"BndsRnXd= 1 | {self.n_empty_bands}",
            f"GbndRnge= 1 | {self.n_empty_bands}",
            f"GTermKind= " + ("BG" if self.screened_coulomb == "RPA" else "BRS"),
        ]
        
        if self.freq_grid == "cd":
            lines.extend([
                "DysSolver= " + ("n" if self.gw_approximation == "G0W0" else "s"),
                "CohSex= (self.gw_approximation == 'COHSEX')",
            ])
        
        return "\n".join(lines)


@dataclass
class BSEParameters:
    """BSE计算参数"""
    # 能带范围
    occupied_bands: Tuple[int, int] = (1, 4)  # 价带
    unoccupied_bands: Tuple[int, int] = (5, 10)  # 导带
    
    # 动量转移
    q_point: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    # 截断
    bse_cutoff: float = 5.0  # Ry
    n_empty_bands_bse: int = 50
    
    # 近似
    approximation: str = "TDHF"  # TDHF, RPA, LRC (long-range correction)
    
    # 交换项
    use_exchange: bool = True  # 包含交换项
    exchange_factor: float = 1.0  # 交换项系数
    
    # 屏蔽势
    use_screening: bool = True
    screening_type: str = "RPA"
    
    # 激子态数
    n_excitons: int = 10
    
    # 光谱
    energy_range: Tuple[float, float] = (0.0, 10.0)  # eV
    energy_step: float = 0.01  # eV
    broadening: float = 0.1  # eV, 洛伦兹展宽
    
    # 振动耦合
    include_phonons: bool = False
    elph_cutoff: float = 0.001  # eV
    
    def to_yambo_input(self) -> str:
        """生成Yambo BSE输入字符串"""
        lines = [
            "# BSE Parameters",
            f"% BSEBands",
            f" {self.occupied_bands[0]} | {self.occupied_bands[1]} |",
            f" {self.unoccupied_bands[0]} | {self.unoccupied_bands[1]} |",
            "%",
            f"BSENGBlk={self.bse_cutoff} RL",
            f"BSENGexx={self.bse_cutoff} RL",
            f"BSEEhEny= {self.energy_range[0]} | {self.energy_range[1]} | eV",
            f"% BEnSteps",
            f" {int((self.energy_range[1]-self.energy_range[0])/self.energy_step)} |",
            "%",
            f"% BEnRange",
            f" {self.energy_range[0]} | {self.energy_range[1]} | eV",
            "%",
            f"BSSmod= " + ("h" if self.approximation == "TDHF" else "r"),
            f"BSEmod= " + ("resonant" if self.use_screening else "coupling"),
            f"BSKmod= " + ("SEX" if self.use_screening else "HF"),
            f"% BndsRnXs",
            f" 1 | {self.n_empty_bands_bse} |",
            "%",
            f"% NGsBlkXs",
            f" {self.bse_cutoff} | RL |",
            "%",
        ]
        return "\n".join(lines)


@dataclass
class ExcitonState:
    """激子态数据类"""
    index: int  # 激子索引
    energy: float  # eV, 激子能量
    binding_energy: float  # eV, 相对于带边的结合能
    oscillator_strength: float  # 振子强度
    
    # 组成
    coefficients: Dict[Tuple[int, int, int], complex] = field(default_factory=dict)
    # (v, c, k) -> amplitude, v=价带, c=导带, k=k点
    
    # 实空间波函数
    wavefunction_realspace: Optional[np.ndarray] = None
    
    # 有效半径
    effective_radius: Optional[float] = None  # Bohr
    
    # 对称性
    symmetry: Optional[str] = None
    
    def get_eh_amplitude(self, k_index: int) -> complex:
        """获取特定k点的电子-空穴振幅"""
        total = 0.0 + 0.0j
        for (v, c, k), coeff in self.coefficients.items():
            if k == k_index:
                total += abs(coeff)**2
        return np.sqrt(total)
    
    def get_band_contribution(self, v_band: int, c_band: int) -> float:
        """获取特定带间跃迁的贡献"""
        total = 0.0
        for (v, c, k), coeff in self.coefficients.items():
            if v == v_band and c == c_band:
                total += abs(coeff)**2
        return total
    
    @property
    def eh_separation(self) -> float:
        """电子-空穴平均分离距离 (Bohr)"""
        if self.effective_radius is not None:
            return self.effective_radius
        # 简化估计
        return 1.0 / np.sqrt(self.binding_energy / 13.6) if self.binding_energy > 0 else float('inf')


@dataclass  
class QPEigenvalue:
    """准粒子本征值"""
    k_point: np.ndarray
    band_index: int
    dft_energy: float  # eV
    qp_energy: float  # eV, GW修正后
    
    # 自能贡献
    vxc_expectation: float  # eV, <Vxc>
    sx_expectation: float  # eV, <Σx>
    ch_expectation: float  # eV, <Σc>
    
    # Z因子
    z_factor: float  # 重整化因子
    
    @property
    def qp_correction(self) -> float:
        """准粒子修正 = qp_energy - dft_energy"""
        return self.qp_energy - self.dft_energy
    
    @property
    def self_energy(self) -> float:
        """总自能"""
        return self.sx_expectation + self.ch_expectation - self.vxc_expectation


# =============================================================================
# Yambo接口主类
# =============================================================================

class YamboGWBSE:
    """
    Yambo GW-BSE 计算接口
    
    提供高层次的API来运行Yambo计算，包括:
    - DFT预处理 (通过 Quantum ESPRESSO)
    - GW准粒子能带计算
    - BSE激子能级计算
    - 光学吸收光谱
    
    Example:
        >>> yambo = YamboGWBSE(work_dir="./calc", prefix="si")
        >>> atoms = bulk('Si', 'diamond', a=5.43)
        >>> yambo.set_structure(atoms)
        >>> yambo.run_dft(scf_params={"ecutwfc": 60})
        >>> 
        >>> # GW计算
        >>> gw_params = GWParameters(gw_approximation="G0W0")
        >>> yambo.run_gw(gw_params)
        >>> 
        >>> # BSE计算
        >>> bse_params = BSEParameters(n_excitons=20)
        >>> yambo.run_bse(bse_params)
        >>> 
        >>> # 获取结果
        >>> spectrum = yambo.get_absorption_spectrum()
        >>> excitons = yambo.get_exciton_states()
    """
    
    def __init__(self, work_dir: str = "./yambo_calc", prefix: str = "system",
                 yambo_path: Optional[str] = None, p2y_path: Optional[str] = None):
        """
        初始化Yambo接口
        
        Parameters:
        -----------
        work_dir : str
            工作目录路径
        prefix : str
            计算前缀
        yambo_path : str, optional
            yambo可执行文件路径
        p2y_path : str, optional
            p2y可执行文件路径 (从pw.x输出转换)
        """
        self.work_dir = Path(work_dir)
        self.prefix = prefix
        self.yambo_path = yambo_path or self._find_yambo()
        self.p2y_path = p2y_path or self._find_p2y()
        
        # 创建目录结构
        self.save_dir = self.work_dir / "SAVE"
        self.gw_dir = self.work_dir / "gw_calc"
        self.bse_dir = self.work_dir / "bse_calc"
        
        for d in [self.work_dir, self.save_dir, self.gw_dir, self.bse_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 结构信息
        self.atoms: Optional['Atoms'] = None
        self.kpoints: Optional[np.ndarray] = None
        
        # 计算结果缓存
        self._qp_energies: Dict[Tuple[int, int], QPEigenvalue] = {}
        self._exciton_states: List[ExcitonState] = []
        self._absorption_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._gw_parameters: Optional[GWParameters] = None
        self._bse_parameters: Optional[BSEParameters] = None
        
        # DFT计算标志
        self._dft_done = False
        self._gw_done = False
        self._bse_done = False
    
    def _find_yambo(self) -> str:
        """查找yambo可执行文件"""
        result = subprocess.run(["which", "yambo"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        # 常见安装路径
        candidates = [
            "/usr/bin/yambo",
            "/usr/local/bin/yambo",
            os.path.expanduser("~/yambo/bin/yambo"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise RuntimeError("Yambo not found. Please specify yambo_path.")
    
    def _find_p2y(self) -> str:
        """查找p2y可执行文件"""
        yambo_bin = Path(self.yambo_path).parent
        p2y_candidates = [
            str(yambo_bin / "p2y"),
            "/usr/bin/p2y",
            "/usr/local/bin/p2y",
        ]
        for path in p2y_candidates:
            if os.path.exists(path):
                return path
        raise RuntimeError("p2y not found. Please specify p2y_path.")
    
    def set_structure(self, atoms: 'Atoms', kpoints: Optional[np.ndarray] = None):
        """设置晶体结构"""
        if not HAS_ASE:
            raise ImportError("ASE required for structure handling")
        self.atoms = atoms
        if kpoints is not None:
            self.kpoints = kpoints
        else:
            # 自动生成k点网格
            self.kpoints = self._generate_kgrid(atoms)
    
    def _generate_kgrid(self, atoms: 'Atoms', density: float = 0.1) -> np.ndarray:
        """根据k点密度生成均匀网格"""
        cell = atoms.cell
        nks = [max(1, int(np.linalg.norm(cell[i]) / density)) for i in range(3)]
        kgrid = []
        for i in range(nks[0]):
            for j in range(nks[1]):
                for k in range(nks[2]):
                    kgrid.append([i/nks[0], j/nks[1], k/nks[2]])
        return np.array(kgrid)
    
    def run_dft(self, scf_params: Optional[Dict] = None, 
                nscf_params: Optional[Dict] = None,
                run: bool = True) -> str:
        """
        运行DFT计算 (Quantum ESPRESSO)
        
        Parameters:
        -----------
        scf_params : dict
            SCF计算参数
        nscf_params : dict
            NSCF计算参数 (用于GW/BSE)
        run : bool
            是否实际运行，False则只生成输入文件
        
        Returns:
        --------
        str : QE输入文件路径
        """
        if self.atoms is None:
            raise ValueError("Structure not set. Call set_structure() first.")
        
        # 默认参数
        default_scf = {
            "calculation": "scf",
            "prefix": self.prefix,
            "outdir": "./out",
            "pseudo_dir": "./pseudo",
            "ecutwfc": 60,
            "ecutrho": 480,
            "occupations": "smearing",
            "smearing": "gauss",
            "degauss": 0.01,
        }
        
        default_nscf = {
            "calculation": "nscf",
            "prefix": self.prefix,
            "outdir": "./out",
            "nbnd": 100,
        }
        
        scf_params = {**default_scf, **(scf_params or {})}
        nscf_params = {**default_nscf, **(nscf_params or {})}
        
        # 生成输入文件
        scf_input = self._generate_qe_input(scf_params, "scf")
        nscf_input = self._generate_qe_input(nscf_params, "nscf")
        
        scf_file = self.work_dir / f"{self.prefix}.scf.in"
        nscf_file = self.work_dir / f"{self.prefix}.nscf.in"
        
        with open(scf_file, 'w') as f:
            f.write(scf_input)
        with open(nscf_file, 'w') as f:
            f.write(nscf_input)
        
        if run:
            # 运行SCF
            self._run_command(f"pw.x < {scf_file} > {self.prefix}.scf.out")
            # 运行NSCF
            self._run_command(f"pw.x < {nscf_file} > {self.prefix}.nscf.out")
            # 转换为Yambo格式
            os.chdir(self.save_dir.parent)
            self._run_command(f"{self.p2y_path} -F {nscf_file}")
            self._dft_done = True
        
        return str(scf_file)
    
    def _generate_qe_input(self, params: Dict, calc_type: str) -> str:
        """生成Quantum ESPRESSO输入文件"""
        lines = ["&CONTROL"]
        for key, val in params.items():
            if isinstance(val, str):
                lines.append(f"   {key}='{val}'")
            else:
                lines.append(f"   {key}={val}")
        lines.append("/")
        
        lines.extend([
            "&SYSTEM",
            f"   ibrav=0",
            f"   nat={len(self.atoms)}",
            f"   ntyp={len(set(self.atoms.get_chemical_symbols()))}",
        ])
        
        # 电子参数
        if "nbnd" in params:
            lines.append(f"   nbnd={params['nbnd']}")
        lines.append("/")
        
        lines.extend([
            "&ELECTRONS",
            "   conv_thr=1.0d-8",
            "/",
        ])
        
        # 晶胞
        lines.append("CELL_PARAMETERS angstrom")
        for vec in self.atoms.cell:
            lines.append(f"   {vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")
        
        # 原子类型
        lines.append("ATOMIC_SPECIES")
        symbols = list(set(self.atoms.get_chemical_symbols()))
        for sym in symbols:
            lines.append(f"   {sym} 1.0 {sym}.upf")
        
        # 原子位置
        lines.append("ATOMIC_POSITIONS crystal")
        positions = self.atoms.get_scaled_positions()
        for i, (sym, pos) in enumerate(zip(self.atoms.get_chemical_symbols(), positions)):
            lines.append(f"   {sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")
        
        # K点
        lines.append("K_POINTS automatic")
        if self.kpoints is not None:
            # 使用生成的k点
            lines.append(f"{len(self.kpoints)}")
            for k in self.kpoints:
                lines.append(f"{k[0]:.6f} {k[1]:.6f} {k[2]:.6f} 1.0")
        else:
            # 均匀网格
            lines.append("6 6 6 0 0 0")
        
        return "\n".join(lines)
    
    def _run_command(self, cmd: str, cwd: Optional[str] = None):
        """运行shell命令"""
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
        return result.stdout
    
    def run_gw(self, params: Optional[GWParameters] = None, 
               run: bool = True) -> str:
        """
        运行GW计算
        
        Parameters:
        -----------
        params : GWParameters
            GW计算参数
        run : bool
            是否实际运行
        
        Returns:
        --------
        str : 输入文件路径
        """
        if params is None:
            params = GWParameters()
        self._gw_parameters = params
        
        # 准备计算目录
        os.chdir(self.gw_dir)
        
        # 链接SAVE目录
        save_link = self.gw_dir / "SAVE"
        if not save_link.exists():
            save_link.symlink_to(self.save_dir)
        
        # 生成输入文件
        input_file = self.gw_dir / "gw.in"
        with open(input_file, 'w') as f:
            f.write(params.to_yambo_input())
        
        if run:
            # 分步运行
            # 1. 初始化
            self._run_command(f"{self.yambo_path} -F gw.in -J gw")
            
            # 2. 介电函数
            self._run_command(f"{self.yambo_path} -b -F gw.in -J gw")
            
            # 3. 自能
            self._run_command(f"{self.yambo_path} -g n -F gw.in -J gw")
            
            self._parse_gw_output()
            self._gw_done = True
        
        return str(input_file)
    
    def _parse_gw_output(self):
        """解析GW输出文件"""
        output_file = self.gw_dir / "gw.o"
        if not output_file.exists():
            return
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # 解析准粒子能量
        # Yambo输出格式: k =    1, band =    1 Eo =    -5.234 E =    -5.123 Z = 0.78
        pattern = r'k\s*=\s*(\d+),\s*band\s*=\s*(\d+)\s*Eo\s*=\s*([-\d.]+)\s*E\s*=\s*([-\d.]+)\s*Z\s*=\s*([\d.]+)'
        
        for match in re.finditer(pattern, content):
            k_idx = int(match.group(1))
            band_idx = int(match.group(2))
            dft_e = float(match.group(3))
            qp_e = float(match.group(4))
            z_factor = float(match.group(5))
            
            qp = QPEigenvalue(
                k_point=self.kpoints[k_idx-1] if self.kpoints is not None else np.zeros(3),
                band_index=band_idx,
                dft_energy=dft_e,
                qp_energy=qp_e,
                vxc_expectation=0.0,
                sx_expectation=0.0,
                ch_expectation=0.0,
                z_factor=z_factor
            )
            self._qp_energies[(k_idx, band_idx)] = qp
    
    def run_bse(self, params: Optional[BSEParameters] = None,
                run: bool = True) -> str:
        """
        运行BSE计算
        
        Parameters:
        -----------
        params : BSEParameters
            BSE计算参数
        run : bool
            是否实际运行
        
        Returns:
        --------
        str : 输入文件路径
        """
        if params is None:
            params = BSEParameters()
        self._bse_parameters = params
        
        # 准备目录
        os.chdir(self.bse_dir)
        
        # 链接SAVE目录
        save_link = self.bse_dir / "SAVE"
        if not save_link.exists():
            save_link.symlink_to(self.save_dir)
        
        # 生成输入文件
        input_file = self.bse_dir / "bse.in"
        with open(input_file, 'w') as f:
            f.write(params.to_yambo_input())
        
        if run:
            # 运行BSE
            self._run_command(f"{self.yambo_path} -y d -F bse.in -J bse")
            
            self._parse_bse_output()
            self._bse_done = True
        
        return str(input_file)
    
    def _parse_bse_output(self):
        """解析BSE输出文件"""
        # 解析激子能量和振子强度
        exc_file = self.bse_dir / "bse.exc"
        eig_file = self.bse_dir / "bse.exc_E_sorted"
        
        if not eig_file.exists():
            return
        
        # 解析本征值文件
        with open(eig_file, 'r') as f:
            lines = f.readlines()
        
        excitons = []
        for i, line in enumerate(lines):
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0])
                energy = float(parts[1])
                strength = float(parts[2])
                
                exc = ExcitonState(
                    index=idx,
                    energy=energy,
                    binding_energy=0.0,  # 将在后续计算
                    oscillator_strength=strength,
                    coefficients={}
                )
                excitons.append(exc)
        
        self._exciton_states = excitons
    
    def get_qp_energies(self) -> Dict[Tuple[int, int], QPEigenvalue]:
        """获取准粒子能量"""
        if not self._gw_done:
            raise RuntimeError("GW calculation not done yet")
        return self._qp_energies
    
    def get_exciton_states(self) -> List[ExcitonState]:
        """获取激子态列表"""
        if not self._bse_done:
            raise RuntimeError("BSE calculation not done yet")
        return self._exciton_states
    
    def get_band_gap(self, gw_corrected: bool = True) -> float:
        """
        计算带隙
        
        Parameters:
        -----------
        gw_corrected : bool
            使用GW修正后的能量
        
        Returns:
        --------
        float : 带隙 (eV)
        """
        if not self._qp_energies:
            raise RuntimeError("No QP energies available")
        
        energies = list(self._qp_energies.values())
        
        # 找到HOMO和LUMO
        occupied = [e for e in energies if e.qp_energy < 0]
        unoccupied = [e for e in energies if e.qp_energy >= 0]
        
        if not occupied or not unoccupied:
            # 使用DFT费米能级估计
            homo = max(e.qp_energy for e in energies if e.band_index <= self.atoms.get_global_number_of_atoms())
            lumo = min(e.qp_energy for e in energies if e.band_index > self.atoms.get_global_number_of_atoms())
        else:
            homo = max(e.qp_energy for e in occupied)
            lumo = min(e.qp_energy for e in unoccupied)
        
        return lumo - homo
    
    def get_exciton_binding_energy(self, exciton_index: int = 0) -> float:
        """
        计算激子结合能
        
        Parameters:
        -----------
        exciton_index : int
            激子态索引 (0为基态)
        
        Returns:
        --------
        float : 结合能 (eV)
        """
        if not self._bse_done:
            raise RuntimeError("BSE calculation not done")
        
        gap = self.get_band_gap(gw_corrected=True)
        exciton_energy = self._exciton_states[exciton_index].energy
        
        binding_energy = gap - exciton_energy
        self._exciton_states[exciton_index].binding_energy = binding_energy
        
        return binding_energy
    
    def get_absorption_spectrum(self, broaden: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算光学吸收光谱
        
        Parameters:
        -----------
        broaden : float, optional
            展宽参数 (eV)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (energy, absorption)
        """
        if not self._bse_done:
            raise RuntimeError("BSE calculation not done")
        
        if self._absorption_spectrum is not None and broaden is None:
            return self._absorption_spectrum
        
        params = self._bse_parameters
        broaden = broaden or params.broadening
        
        # 能量网格
        energies = np.arange(params.energy_range[0], params.energy_range[1], params.energy_step)
        absorption = np.zeros_like(energies)
        
        # 洛伦兹展宽
        for exc in self._exciton_states:
            strength = exc.oscillator_strength
            energy = exc.energy
            gamma = broaden
            
            absorption += strength * (gamma / np.pi) / ((energies - energy)**2 + gamma**2)
        
        self._absorption_spectrum = (energies, absorption)
        return energies, absorption
    
    def plot_spectrum(self, save_path: Optional[str] = None):
        """绘制吸收光谱"""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for plotting")
        
        energies, absorption = self.get_absorption_spectrum()
        
        plt.figure(figsize=(10, 6))
        plt.plot(energies, absorption, 'b-', linewidth=2)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Absorption (arb. units)')
        plt.title('Optical Absorption Spectrum (GW-BSE)')
        plt.grid(True, alpha=0.3)
        
        # 标注激子峰
        for exc in self._exciton_states[:5]:
            plt.axvline(x=exc.energy, color='r', linestyle='--', alpha=0.5)
            plt.annotate(f'Exc {exc.index}', xy=(exc.energy, 0), 
                        xytext=(exc.energy, max(absorption)*0.1),
                        rotation=90, fontsize=8)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, filename: str = "yambo_results.json"):
        """保存计算结果到JSON"""
        results = {
            "gw_parameters": self._gw_parameters.__dict__ if self._gw_parameters else None,
            "bse_parameters": self._bse_parameters.__dict__ if self._bse_parameters else None,
            "band_gap": self.get_band_gap() if self._gw_done else None,
            "exciton_states": [
                {
                    "index": e.index,
                    "energy": e.energy,
                    "binding_energy": e.binding_energy,
                    "oscillator_strength": e.oscillator_strength,
                }
                for e in self._exciton_states
            ] if self._bse_done else None,
        }
        
        # 序列化numpy数组
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=serialize)
    
    def load_results(self, filename: str = "yambo_results.json"):
        """从JSON加载计算结果"""
        with open(filename, 'r') as f:
            results = json.load(f)
        
        # 恢复激子态
        if results.get("exciton_states"):
            self._exciton_states = [
                ExcitonState(
                    index=e["index"],
                    energy=e["energy"],
                    binding_energy=e["binding_energy"],
                    oscillator_strength=e["oscillator_strength"],
                    coefficients={}
                )
                for e in results["exciton_states"]
            ]


# =============================================================================
# 高级功能: 自洽GW和混合计算
# =============================================================================

class SelfConsistentGW(YamboGWBSE):
    """自洽GW计算类"""
    
    def run_sc_gw(self, params: GWParameters, max_iter: int = 5) -> Dict:
        """
        运行自洽GW计算
        
        Parameters:
        -----------
        params : GWParameters
            GW参数
        max_iter : int
            最大迭代次数
        
        Returns:
        --------
        Dict : 迭代历史和收敛信息
        """
        history = {"iterations": [], "converged": False}
        
        for iteration in range(max_iter):
            print(f"SC-GW Iteration {iteration + 1}/{max_iter}")
            
            # 运行GW
            params.gw_approximation = "evGW"
            self.run_gw(params, run=True)
            
            # 检查收敛
            converged = self._check_sc_convergence(history, params.sc_tolerance)
            history["iterations"].append({
                "iteration": iteration + 1,
                "qp_energies": {k: v.qp_energy for k, v in self._qp_energies.items()}
            })
            
            if converged:
                history["converged"] = True
                print(f"SC-GW converged in {iteration + 1} iterations")
                break
        
        return history
    
    def _check_sc_convergence(self, history: Dict, tolerance: float) -> bool:
        """检查自洽收敛"""
        if len(history["iterations"]) < 2:
            return False
        
        prev = history["iterations"][-2]["qp_energies"]
        curr = history["iterations"][-1]["qp_energies"]
        
        max_diff = 0.0
        for key in prev:
            if key in curr:
                diff = abs(prev[key] - curr[key])
                max_diff = max(max_diff, diff)
        
        return max_diff < tolerance


class BSEWithGW(YamboGWBSE):
    """使用GW准粒子能量的BSE计算"""
    
    def run_gw_bse(self, gw_params: GWParameters, bse_params: BSEParameters,
                   run_gw: bool = True) -> Dict:
        """
        顺序运行GW+BSE
        
        Parameters:
        -----------
        gw_params : GWParameters
            GW参数
        bse_params : BSEParameters
            BSE参数
        run_gw : bool
            是否运行GW (如果False，假设GW已完成)
        
        Returns:
        --------
        Dict : 计算结果汇总
        """
        results = {}
        
        # 1. GW计算
        if run_gw:
            print("Running GW calculation...")
            self.run_gw(gw_params)
            results["gw_gap"] = self.get_band_gap()
        
        # 2. 使用GW能量更新BSE参数
        # Yambo会自动使用之前计算的GW准粒子能量
        
        # 3. BSE计算
        print("Running BSE calculation...")
        self.run_bse(bse_params)
        
        # 4. 计算激子结合能
        binding_energy = self.get_exciton_binding_energy()
        
        results.update({
            "exciton_states": self._exciton_states,
            "exciton_binding_energy": binding_energy,
            "absorption_spectrum": self.get_absorption_spectrum(),
        })
        
        return results


# =============================================================================
# 示例和测试
# =============================================================================

def example_silicon_g0w0():
    """硅G0W0计算示例"""
    print("=" * 60)
    print("Example: Silicon G0W0 Calculation")
    print("=" * 60)
    
    # 创建硅晶体结构
    if HAS_ASE:
        from ase.build import bulk
        atoms = bulk('Si', 'diamond', a=5.43)
    else:
        # 手动创建
        atoms = None
    
    # 初始化Yambo接口 (不查找可执行文件)
    try:
        yambo = YamboGWBSE(work_dir="./si_g0w0", prefix="si", 
                          yambo_path="/tmp/yambo_mock", p2y_path="/tmp/p2y_mock")
    except:
        # 创建模拟实例
        yambo = YamboGWBSE.__new__(YamboGWBSE)
        yambo.work_dir = "./si_g0w0"
        yambo.prefix = "si"
    
    if atoms:
        yambo.set_structure(atoms)
    
    # GW参数
    gw_params = GWParameters(
        gw_approximation="G0W0",
        qp_band_range=(1, 8),
        cutoff=10.0,
        n_empty_bands=100,
        freq_grid="cd"
    )
    
    print("\nGW Parameters:")
    print(gw_params.to_yambo_input())
    print(f"\nYambo input would be saved to: ./si_g0w0/gw.in")
    
    return yambo


def example_mos2_bse():
    """MoS2 BSE计算示例"""
    print("=" * 60)
    print("Example: MoS2 BSE Calculation")
    print("=" * 60)
    
    yambo = YamboGWBSE(work_dir="./mos2_bse", prefix="mos2")
    
    # BSE参数 (针对2D材料优化)
    bse_params = BSEParameters(
        occupied_bands=(1, 16),  # 价带
        unoccupied_bands=(17, 24),  # 导带
        bse_cutoff=10.0,
        n_excitons=20,
        approximation="TDHF",
        use_screening=True,
        energy_range=(0.0, 5.0),
        broadening=0.05,  # 较小展宽用于2D材料
    )
    
    # 生成BSE输入
    bse_input = yambo.run_bse(bse_params, run=False)
    print(f"BSE input file generated: {bse_input}")
    print("\nBSE Parameters:")
    print(bse_params.to_yambo_input())
    
    return yambo


def example_gw_bse_workflow():
    """完整GW+BSE工作流示例"""
    print("=" * 60)
    print("Example: Complete GW+BSE Workflow")
    print("=" * 60)
    
    # 创建工作流
    workflow = BSEWithGW(work_dir="./workflow", prefix="system")
    
    # GW参数
    gw_params = GWParameters(
        gw_approximation="G0W0",
        cutoff=20.0,
        dielectric_cutoff=5.0,
        n_empty_bands=200,
        qp_band_range=(1, 20),
    )
    
    # BSE参数
    bse_params = BSEParameters(
        occupied_bands=(1, 8),
        unoccupied_bands=(9, 16),
        n_excitons=15,
        energy_range=(0.0, 8.0),
        broadening=0.1,
    )
    
    # 生成输入文件
    gw_input = workflow.run_gw(gw_params, run=False)
    bse_input = workflow.run_bse(bse_params, run=False)
    
    print("\nWorkflow files generated:")
    print(f"  GW input: {gw_input}")
    print(f"  BSE input: {bse_input}")
    print("\nTo run actual calculations:")
    print("  1. Prepare DFT calculations with Quantum ESPRESSO")
    print("  2. Run yambo -F gw.in")
    print("  3. Run yambo -y d -F bse.in")
    
    return workflow


if __name__ == "__main__":
    # 运行示例
    example_silicon_g0w0()
    print("\n")
    example_mos2_bse()
    print("\n")
    example_gw_bse_workflow()
