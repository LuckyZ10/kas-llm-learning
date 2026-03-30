#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BerkeleyGW Interface Module
===========================

基于平面波的GW计算软件BerkeleyGW的Python接口。
BerkeleyGW适用于大规模GW计算和激子BSE模拟。

特点:
- 高效的平面波基组
- 支持超胞和能带结构计算
- 先进的插值技术
- 大规模并行优化

功能包括:
- Epsilon: 介电函数计算
- Sigma: 自能计算
- Kernel: BSE核矩阵
- Absorption: 光学吸收
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

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.units import Hartree, Bohr, Ry
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    Ry = 13.605698066  # eV

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class BGWEpsilonParameters:
    """BerkeleyGW介电函数计算参数"""
    
    # 频率网格
    frequency_dependence: int = 2  # 0=静态, 2=全频率
    n_frequency: int = 40
    max_frequency: float = 30.0  # eV
    freq_grid_type: str = "umklapp"  # umklapp, sehift
    
    # 截断
    epsilon_cutoff: float = 10.0  # Ry
    number_bands: int = 128  # 参与计算的能带数
    
    # 交换-关联
    screening_model: str = "RPA"  # RPA, LDA, GGA
    broadening: float = 0.1  # eV
    
    # 并行
    number_kpoints: int = 1
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "screening_semiconductor": True,
            "eqp_corrections": False,
            "number_bands": self.number_bands,
            "band_occupation": 1,  # 半导体
            "frequency_dependence": self.frequency_dependence,
            "frequency_max": self.max_frequency,
            "delta_frequency": self.max_frequency / self.n_frequency,
            "broadening": self.broadening,
            "epsilon_cutoff": self.epsilon_cutoff,
            "write_vkb": False,
        }
    
    def write_input_file(self, filename: str = "epsilon.inp"):
        """写入epsilon.inp输入文件"""
        lines = [
            "# BerkeleyGW Epsilon Input",
            f"screening_semiconductor  .true.",
            f"eqp_corrections  .false.",
            f"number_bands  {self.number_bands}",
            f"band_occupation  1",
            f"frequency_dependence  {self.frequency_dependence}",
            f"frequency_max  {self.max_frequency}",
            f"delta_frequency  {self.max_frequency / self.n_frequency}",
            f"broadening  {self.broadening}",
            f"epsilon_cutoff  {self.epsilon_cutoff}",
            "write_vkb  .false.",
        ]
        
        with open(filename, 'w') as f:
            f.write("\n".join(lines))


@dataclass
class BGWSigmaParameters:
    """BerkeleyGW自能计算参数"""
    
    # 准粒子能带
    band_range: Tuple[int, int] = (1, 10)  # 计算自能的能带
    
    # 交换项
    use_exchange: bool = True
    
    # 关联项
    correlation_method: str = "CD"  # CD=contour deformation, PPA=plasmon pole
    n_frequency_corr: int = 32  # 关联项频率点数
    
    # 截断
    truncation_scheme: str = "box"  # box, spherical, wire, slab
    cell_truncation: bool = False
    
    # 周期性系统的截断
    truncation_axis: Optional[int] = None  # 用于准2D系统
    truncation_height: Optional[float] = None
    
    # GW近似
    gw_approximation: str = "G0W0"  # G0W0, evGW
    
    # 能带插值
    use_interpolation: bool = True
    interpolation_scheme: str = "lanczos"  # lanczos, csi
    
    def write_input_file(self, filename: str = "sigma.inp"):
        """写入sigma.inp输入文件"""
        lines = [
            "# BerkeleyGW Sigma Input",
            f"band_range_min  {self.band_range[0]}",
            f"band_range_max  {self.band_range[1]}",
            f"screening_semiconductor  .true.",
            f"eqp_corrections  {' .true.' if self.gw_approximation == 'evGW' else ' .false.'}",
            f"max_grad_iterations  {5 if self.gw_approximation == 'evGW' else 0}",
            f"number_bands  {self.band_range[1] + 50}",
            f"band_occupation  1",
            f"frequency_dependence  2",
            f"cd_integral_method  2",  # contour deformation
            f"n_freq  {self.n_frequency_corr}",
        ]
        
        # 截断方案
        if self.cell_truncation:
            lines.append("cell_truncation  .true.")
            if self.truncation_axis is not None:
                lines.append(f"truncation_axis  {self.truncation_axis}")
                lines.append(f"truncation_height  {self.truncation_height or 100.0}")
        
        lines.append("dont_use_vxcdat  .false.")
        
        with open(filename, 'w') as f:
            f.write("\n".join(lines))


@dataclass
class BGWBSEParameters:
    """BerkeleyGW BSE参数"""
    
    # 能带范围
    valence_bands: Tuple[int, int] = (1, 4)
    conduction_bands: Tuple[int, int] = (5, 8)
    
    # 激子态
    n_excitons: int = 10
    
    # BSE近似
    use_hartree: bool = True
    use_exchange: bool = True
    use_correlation: bool = True
    
    # 插值
    interpolate_weights: bool = True
    interpolation_kgrid: Tuple[int, int, int] = (10, 10, 10)
    
    # 光谱
    energy_range: Tuple[float, float] = (0.0, 10.0)
    energy_step: float = 0.01
    broadening: float = 0.1
    
    def write_kernel_input(self, filename: str = "kernel.inp"):
        """写入kernel.inp"""
        lines = [
            "# BerkeleyGW Kernel Input",
            f"number_val_bands  {self.valence_bands[1] - self.valence_bands[0] + 1}",
            f"number_cond_bands  {self.conduction_bands[1] - self.conduction_bands[0] + 1}",
            f"use_hartree  {' .true.' if self.use_hartree else ' .false.'}",
            f"use_exchange  {' .true.' if self.use_exchange else ' .false.'}",
            f"use_correlation  {' .true.' if self.use_correlation else ' .false.'}",
            f"screening_semiconductor  .true.",
            f"eqp_corrections  .true.",
        ]
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
    
    def write_absorption_input(self, filename: str = "absorption.inp"):
        """写入absorption.inp"""
        lines = [
            "# BerkeleyGW Absorption Input",
            f"energy_range  {self.energy_range[0]} {self.energy_range[1]}",
            f"energy_step  {self.energy_step}",
            f"broadening  {self.broadening}",
            f"interpolate_weights  {' .true.' if self.interpolate_weights else ' .false.'}",
            f"polarization  1 0 0",  # x方向
            f"polarization  0 1 0",  # y方向
            f"polarization  0 0 1",  # z方向
        ]
        with open(filename, 'w') as f:
            f.write("\n".join(lines))


@dataclass
class BGWBandStructure:
    """BerkeleyGW能带结构数据"""
    
    kpoints: np.ndarray  # (nk, 3) 晶体坐标
    kpath_ticks: List[str]  # 高对称点标签
    kpath_indices: List[int]  # 高对称点索引
    
    dft_energies: np.ndarray  # (nk, nbands) eV
    gw_energies: np.ndarray  # (nk, nbands) eV
    
    # 沿路径的距离
    kdistances: Optional[np.ndarray] = None
    
    def get_band_gap(self, use_gw: bool = True) -> Tuple[float, str]:
        """获取带隙"""
        energies = self.gw_energies if use_gw else self.dft_energies
        
        # 找到最小导带和最大价带
        nval = energies.shape[1] // 2
        vbm = np.max(energies[:, :nval])
        cbm = np.min(energies[:, nval:])
        
        # 确定带隙类型
        vbm_idx = np.where(energies == vbm)
        cbm_idx = np.where(energies == cbm)
        
        if vbm_idx[0][0] == cbm_idx[0][0]:
            gap_type = "Direct"
        else:
            gap_type = "Indirect"
        
        return cbm - vbm, gap_type
    
    def plot(self, save_path: Optional[str] = None, show_dft: bool = True):
        """绘制能带结构"""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")
        
        # 计算k点距离
        if self.kdistances is None:
            self._calculate_kdistances()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制能带
        for ib in range(self.dft_energies.shape[1]):
            if show_dft:
                ax.plot(self.kdistances, self.dft_energies[:, ib], 'k--', alpha=0.5, linewidth=1)
            ax.plot(self.kdistances, self.gw_energies[:, ib], 'b-', linewidth=2)
        
        # 高对称点标记
        for idx, label in zip(self.kpath_indices, self.kpath_ticks):
            ax.axvline(x=self.kdistances[idx], color='gray', linestyle='-', alpha=0.3)
        ax.set_xticks(self.kdistances[self.kpath_indices])
        ax.set_xticklabels(self.kpath_ticks)
        
        ax.set_xlabel('k-path')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('GW Band Structure')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 费米能级
        gap, _ = self.get_band_gap(use_gw=True)
        vbm = np.max(self.gw_energies[:, :self.gw_energies.shape[1]//2])
        ax.axhline(y=vbm, color='r', linestyle='--', alpha=0.5, label='VBM')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def _calculate_kdistances(self):
        """计算k点沿路径的距离"""
        distances = [0.0]
        for i in range(1, len(self.kpoints)):
            dk = np.linalg.norm(self.kpoints[i] - self.kpoints[i-1])
            distances.append(distances[-1] + dk)
        self.kdistances = np.array(distances)


# =============================================================================
# BerkeleyGW主接口类
# =============================================================================

class BerkeleyGW:
    """
    BerkeleyGW 计算接口
    
    提供高层次的API来运行BerkeleyGW计算。
    
    Example:
        >>> bgw = BerkeleyGW(work_dir="./bgw_calc", prefix="si")
        >>> atoms = bulk('Si', 'diamond', a=5.43)
        >>> bgw.set_structure(atoms)
        >>> 
        >>> # 运行计算流程
        >>> bgw.run_epsilon(BGWEpsilonParameters())
        >>> bgw.run_sigma(BGWSigmaParameters())
        >>> bgw.run_bse(BGWBSEParameters())
    """
    
    def __init__(self, work_dir: str = "./bgw_calc", prefix: str = "system",
                 bgw_path: Optional[str] = None):
        """
        初始化BerkeleyGW接口
        
        Parameters:
        -----------
        work_dir : str
            工作目录
        prefix : str
            计算前缀
        bgw_path : str, optional
            BerkeleyGW可执行文件路径
        """
        self.work_dir = Path(work_dir)
        self.prefix = prefix
        self.bgw_path = bgw_path or self._find_bgw()
        
        # 子目录
        self.epsilon_dir = self.work_dir / "1-epsilon"
        self.sigma_dir = self.work_dir / "2-sigma"
        self.kernel_dir = self.work_dir / "3-kernel"
        self.absorption_dir = self.work_dir / "4-absorption"
        
        for d in [self.work_dir, self.epsilon_dir, self.sigma_dir, 
                  self.kernel_dir, self.absorption_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 结构信息
        self.atoms: Optional['Atoms'] = None
        
        # 结果
        self._band_structure: Optional[BGWBandStructure] = None
        self._epsilon_data: Optional[Dict] = None
        self._sigma_data: Optional[Dict] = None
        self._bse_data: Optional[Dict] = None
        
        # 计算状态
        self._epsilon_done = False
        self._sigma_done = False
        self._bse_done = False
    
    def _find_bgw(self) -> str:
        """查找BerkeleyGW安装"""
        candidates = [
            "/usr/local/berkeleygw/bin",
            os.path.expanduser("~/berkeleygw/bin"),
            os.environ.get("BGW_BIN", ""),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return "/usr/local/berkeleygw/bin"  # 默认路径
    
    def set_structure(self, atoms: 'Atoms'):
        """设置结构"""
        if not HAS_ASE:
            raise ImportError("ASE required")
        self.atoms = atoms
    
    def run_epsilon(self, params: Optional[BGWEpsilonParameters] = None,
                    run: bool = True) -> str:
        """
        运行介电函数计算
        
        Parameters:
        -----------
        params : BGWEpsilonParameters
            介电函数参数
        run : bool
            是否实际运行
        
        Returns:
        --------
        str : 输入文件路径
        """
        if params is None:
            params = BGWEpsilonParameters()
        
        os.chdir(self.epsilon_dir)
        
        # 写入输入文件
        input_file = self.epsilon_dir / "epsilon.inp"
        params.write_input_file(str(input_file))
        
        # 写入辅助文件
        self._write_kpoints_file(self.epsilon_dir / "kpoints.inp")
        
        if run:
            epsilon_exe = Path(self.bgw_path) / "epsilon.cplx.x"  # 复数版本
            self._run_command(f"{epsilon_exe} > epsilon.out")
            self._parse_epsilon_output()
            self._epsilon_done = True
        
        return str(input_file)
    
    def run_sigma(self, params: Optional[BGWSigmaParameters] = None,
                  run: bool = True) -> str:
        """运行自能计算"""
        if params is None:
            params = BGWSigmaParameters()
        
        os.chdir(self.sigma_dir)
        
        # 链接epsilon结果
        eps_link = self.sigma_dir / "epsilon_inv_rpa.dat"
        if not eps_link.exists():
            eps_link.symlink_to(self.epsilon_dir / "epsilon_inv_rpa.dat")
        
        # 写入输入
        input_file = self.sigma_dir / "sigma.inp"
        params.write_input_file(str(input_file))
        
        # 写入k点
        self._write_kpoints_file(self.sigma_dir / "kpoints.inp", band=True)
        
        if run:
            sigma_exe = Path(self.bgw_path) / "sigma.cplx.x"
            self._run_command(f"{sigma_exe} > sigma.out")
            self._parse_sigma_output()
            self._sigma_done = True
        
        return str(input_file)
    
    def run_bse(self, params: Optional[BGWBSEParameters] = None,
                run: bool = True) -> Tuple[str, str]:
        """
        运行BSE计算
        
        Returns:
        --------
        Tuple[str, str] : (kernel输入, absorption输入)路径
        """
        if params is None:
            params = BGWBSEParameters()
        
        # Kernel计算
        os.chdir(self.kernel_dir)
        
        # 链接必要文件
        for f in ["epsilon_inv_rpa.dat", "sigma_hp.log"]:
            src = self.sigma_dir / f if "sigma" in f else self.epsilon_dir / f
            link = self.kernel_dir / f
            if not link.exists() and src.exists():
                link.symlink_to(src)
        
        kernel_input = self.kernel_dir / "kernel.inp"
        params.write_kernel_input(str(kernel_input))
        self._write_kpoints_file(self.kernel_dir / "kpoints.inp")
        
        if run:
            kernel_exe = Path(self.bgw_path) / "kernel.cplx.x"
            self._run_command(f"{kernel_exe} > kernel.out")
        
        # Absorption计算
        os.chdir(self.absorption_dir)
        
        abs_input = self.absorption_dir / "absorption.inp"
        params.write_absorption_input(str(abs_input))
        
        # 链接BSE对角化结果
        bse_link = self.absorption_dir / "bse_eigenvectors.dat"
        if not bse_link.exists():
            bse_link.symlink_to(self.kernel_dir / "bse_eigenvectors.dat")
        
        if run:
            abs_exe = Path(self.bgw_path) / "absorption.cplx.x"
            self._run_command(f"{abs_exe} > absorption.out")
            self._parse_absorption_output()
            self._bse_done = True
        
        return str(kernel_input), str(abs_input)
    
    def _write_kpoints_file(self, filename: Path, band: bool = False):
        """写入k点文件"""
        if band and self._band_structure:
            kpts = self._band_structure.kpoints
        else:
            # 生成均匀网格
            kpts = self._generate_kgrid((6, 6, 6))
        
        lines = [f"{len(kpts)}"]
        for k in kpts:
            lines.append(f"{k[0]:.8f} {k[1]:.8f} {k[2]:.8f} 1.0")
        
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
    
    def _generate_kgrid(self, nks: Tuple[int, int, int]) -> np.ndarray:
        """生成k点网格"""
        kgrid = []
        for i in range(nks[0]):
            for j in range(nks[1]):
                for k in range(nks[2]):
                    kgrid.append([
                        i/nks[0] - 0.5,
                        j/nks[1] - 0.5,
                        k/nks[2] - 0.5
                    ])
        return np.array(kgrid)
    
    def _run_command(self, cmd: str):
        """运行命令"""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
        return result.stdout
    
    def _parse_epsilon_output(self):
        """解析epsilon输出"""
        output_file = self.epsilon_dir / "epsilon.out"
        if not output_file.exists():
            return
        
        # 读取介电函数数据
        eps_file = self.epsilon_dir / "epsilon_inv_rpa.dat"
        if eps_file.exists():
            # 解析二进制或文本格式
            self._epsilon_data = {
                "epsilon_file": str(eps_file),
                "status": "completed"
            }
    
    def _parse_sigma_output(self):
        """解析sigma输出"""
        output_file = self.sigma_dir / "sigma_hp.log"
        if not output_file.exists():
            return
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # 解析准粒子能量
        # BerkeleyGW格式: kpoint, band, E_DFT, E_QP, Vxc, Sigma_x, Sigma_c, Z
        qp_data = []
        
        for line in content.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    qp_data.append({
                        'kpoint': int(parts[0]),
                        'band': int(parts[1]),
                        'E_dft': float(parts[2]),
                        'E_qp': float(parts[3]),
                        'Vxc': float(parts[4]),
                        'Sigma_x': float(parts[5]),
                        'Sigma_c': float(parts[6]),
                        'Z': float(parts[7]),
                    })
                except ValueError:
                    continue
        
        self._sigma_data = {"qp_data": qp_data}
        
        # 构建能带结构
        if qp_data:
            self._build_band_structure(qp_data)
    
    def _build_band_structure(self, qp_data: List[Dict]):
        """从QP数据构建能带结构"""
        # 按k点和能带组织数据
        kpoints_set = sorted(set(d['kpoint'] for d in qp_data))
        bands_set = sorted(set(d['band'] for d in qp_data))
        
        nk = len(kpoints_set)
        nb = len(bands_set)
        
        dft_energies = np.zeros((nk, nb))
        gw_energies = np.zeros((nk, nb))
        
        for d in qp_data:
            ik = kpoints_set.index(d['kpoint'])
            ib = bands_set.index(d['band'])
            dft_energies[ik, ib] = d['E_dft']
            gw_energies[ik, ib] = d['E_qp']
        
        self._band_structure = BGWBandStructure(
            kpoints=np.zeros((nk, 3)),  # 简化
            kpath_ticks=[],
            kpath_indices=[],
            dft_energies=dft_energies,
            gw_energies=gw_energies
        )
    
    def _parse_absorption_output(self):
        """解析吸收光谱输出"""
        # 读取吸收系数
        abs_file = self.absorption_dir / "absorption_coeff.dat"
        if abs_file.exists():
            data = np.loadtxt(abs_file)
            self._bse_data = {
                "energy": data[:, 0],
                "absorption": data[:, 1]
            }
    
    def get_band_structure(self) -> Optional[BGWBandStructure]:
        """获取能带结构"""
        return self._band_structure
    
    def get_absorption_spectrum(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取吸收光谱"""
        if self._bse_data:
            return self._bse_data["energy"], self._bse_data["absorption"]
        return None
    
    def get_qp_corrections(self) -> Dict[int, float]:
        """获取准粒子修正 (eV)"""
        if not self._sigma_data:
            return {}
        
        corrections = {}
        for d in self._sigma_data.get("qp_data", []):
            band = d['band']
            corr = d['E_qp'] - d['E_dft']
            if band not in corrections:
                corrections[band] = []
            corrections[band].append(corr)
        
        # 平均所有k点
        return {b: np.mean(c) for b, c in corrections.items()}


# =============================================================================
# 高级功能
# =============================================================================

class BGWBandStructureCalculator:
    """能带结构计算器"""
    
    def __init__(self, bgw: BerkeleyGW):
        self.bgw = bgw
    
    def setup_kpath(self, path: List[Tuple[np.ndarray, str]], 
                    npoints_per_segment: int = 20) -> np.ndarray:
        """
        设置能带路径
        
        Parameters:
        -----------
        path : List[Tuple[np.ndarray, str]]
            [(kpoint, label), ...]
        npoints_per_segment : int
            每段k点数
        
        Returns:
        --------
        np.ndarray : k点列表
        """
        kpoints = []
        labels = []
        indices = []
        
        for i in range(len(path) - 1):
            k1, label1 = path[i]
            k2, label2 = path[i + 1]
            
            if i == 0:
                kpoints.append(k1)
                labels.append(label1)
                indices.append(len(kpoints) - 1)
            
            for j in range(1, npoints_per_segment):
                t = j / npoints_per_segment
                k = k1 + t * (k2 - k1)
                kpoints.append(k)
            
            kpoints.append(k2)
            labels.append(label2)
            indices.append(len(kpoints) - 1)
        
        return np.array(kpoints), labels, indices
    
    def get_standard_kpath(self, crystal_system: str) -> Tuple[List[np.ndarray], List[str]]:
        """获取标准高对称路径"""
        paths = {
            "cubic": {
                "points": [
                    ([0.0, 0.0, 0.0], "Γ"),
                    ([0.5, 0.0, 0.0], "X"),
                    ([0.5, 0.5, 0.0], "M"),
                    ([0.0, 0.0, 0.0], "Γ"),
                    ([0.5, 0.5, 0.5], "R"),
                ],
            },
            "hexagonal": {
                "points": [
                    ([0.0, 0.0, 0.0], "Γ"),
                    ([1/3, 1/3, 0.0], "K"),
                    ([0.5, 0.0, 0.0], "M"),
                    ([0.0, 0.0, 0.0], "Γ"),
                    ([0.0, 0.0, 0.5], "A"),
                ],
            },
        }
        
        if crystal_system not in paths:
            raise ValueError(f"Unknown crystal system: {crystal_system}")
        
        return paths[crystal_system]["points"]


class BGW2DMaterials:
    """2D材料专用计算接口"""
    
    def __init__(self, bgw: BerkeleyGW):
        self.bgw = bgw
    
    def setup_truncation(self, vacuum_axis: int = 2,
                         truncation_height: float = 100.0) -> BGWSigmaParameters:
        """
        设置2D材料截断
        
        Parameters:
        -----------
        vacuum_axis : int
            真空方向 (0, 1, or 2)
        truncation_height : float
            截断高度 (Bohr)
        
        Returns:
        --------
        BGWSigmaParameters : 配置好的参数
        """
        params = BGWSigmaParameters(
            cell_truncation=True,
            truncation_axis=vacuum_axis,
            truncation_height=truncation_height,
        )
        return params
    
    def setup_epsilon_2d(self, epsilon_cutoff: float = 10.0) -> BGWEpsilonParameters:
        """
        设置2D材料介电函数参数
        
        2D材料需要更大的截断和更密集的k点
        """
        return BGWEpsilonParameters(
            epsilon_cutoff=epsilon_cutoff,
            number_bands=256,  # 2D材料需要更多能带
            frequency_dependence=2,
        )


# =============================================================================
# 示例和测试
# =============================================================================

def example_silicon_bgw():
    """硅BerkeleyGW计算示例"""
    print("=" * 60)
    print("Example: Silicon BerkeleyGW Calculation")
    print("=" * 60)
    
    try:
        bgw = BerkeleyGW(work_dir="./si_bgw", prefix="si", bgw_path="/tmp")
    except:
        bgw = BerkeleyGW.__new__(BerkeleyGW)
        bgw.work_dir = "./si_bgw"
        bgw.prefix = "si"
    
    # Epsilon参数
    eps_params = BGWEpsilonParameters(
        epsilon_cutoff=10.0,
        number_bands=128,
        frequency_dependence=2,
    )
    print(f"\nEpsilon input would be saved to: ./si_bgw/epsilon.inp")
    
    # Sigma参数
    sigma_params = BGWSigmaParameters(
        band_range=(1, 8),
        correlation_method="CD",
        gw_approximation="G0W0",
    )
    print(f"Sigma input would be saved to: ./si_bgw/sigma.inp")
    
    # BSE参数
    bse_params = BGWBSEParameters(
        valence_bands=(1, 4),
        conduction_bands=(5, 8),
        n_excitons=10,
    )
    print(f"Kernel input would be saved to: ./si_bgw/kernel.inp")
    print(f"Absorption input would be saved to: ./si_bgw/absorption.inp")


def example_2d_mos2():
    """2D MoS2计算示例"""
    print("\n" + "=" * 60)
    print("Example: 2D MoS2 with Truncation")
    print("=" * 60)
    
    bgw = BerkeleyGW(work_dir="./mos2_bgw", prefix="mos2")
    
    # 2D材料专用设置
    bgw_2d = BGW2DMaterials(bgw)
    
    eps_params = bgw_2d.setup_epsilon_2d(epsilon_cutoff=15.0)
    sigma_params = bgw_2d.setup_truncation(vacuum_axis=2, truncation_height=80.0)
    
    print("\n2D MoS2 Parameters:")
    print(f"Epsilon cutoff: {eps_params.epsilon_cutoff} Ry")
    print(f"Truncation axis: {sigma_params.truncation_axis}")
    print(f"Truncation height: {sigma_params.truncation_height} Bohr")


def example_band_structure():
    """能带结构计算示例"""
    print("\n" + "=" * 60)
    print("Example: Band Structure Setup")
    print("=" * 60)
    
    bgw = BerkeleyGW(work_dir="./bands", prefix="gaas")
    
    # 设置高对称路径
    calc = BGWBandStructureCalculator(bgw)
    path = calc.get_standard_kpath("cubic")
    
    print("\nStandard cubic k-path:")
    for kpt, label in path:
        print(f"  {label}: ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f})")


if __name__ == "__main__":
    example_silicon_bgw()
    example_2d_mos2()
    example_band_structure()
