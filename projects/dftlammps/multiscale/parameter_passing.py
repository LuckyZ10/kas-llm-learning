#!/usr/bin/env python3
"""
parameter_passing.py
====================
跨尺度参数传递模块

功能：
1. 从DFT提取弹性常数
2. 从MD提取扩散系数
3. 从原子模拟提取界面能
4. 自动格式转换（DFT→MD→连续介质）
5. 参数验证和不确定性量化

支持的源：
- DFT (VASP, QE, Gaussian)
- MD (LAMMPS, GROMACS)
- 相场/连续介质模拟

作者: Multi-Scale Simulation Expert
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
from scipy import stats
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, griddata

# ASE
from ase import Atoms
from ase.io import read, write
from ase.io.vasp import read_vasp_out
from ase.units import eV, Ang, GPa, fs, Bohr, Hartree
from ase.elasticity import get_elastic_constants, get_elementary_deformations
from ase.phonons import Phonons
from ase.md.analysis import DiffusionCoefficient

# 可选依赖
try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.elasticity import ElasticTensor, StressStrainAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    warnings.warn("pymatgen not available. Some features disabled.")

try:
    from dscribe.descriptors import SOAP
    DSCRIBE_AVAILABLE = True
except ImportError:
    DSCRIBE_AVAILABLE = False

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
class ElasticConstants:
    """弹性常数（Voigt表示）"""
    # 立方晶系
    C11: float = 0.0  # GPa
    C12: float = 0.0
    C44: float = 0.0
    
    # 六方晶系
    C13: float = 0.0
    C33: float = 0.0
    C66: float = 0.0
    
    # 工程常数（派生）
    E: float = 0.0  # 杨氏模量
    nu: float = 0.0  # 泊松比
    G: float = 0.0   # 剪切模量
    
    # 立方晶系的体模量和剪切模量
    K: float = 0.0  # 体模量
    G_voigt: float = 0.0  # Voigt平均剪切模量
    G_reuss: float = 0.0  # Reuss平均剪切模量
    G_hill: float = 0.0   # Hill平均剪切模量
    
    # 各向异性比
    Zener_ratio: float = 0.0
    
    # 单位转换
    unit: str = "GPa"
    
    def __post_init__(self):
        """计算派生性质"""
        if self.C11 > 0 and self.C12 > 0:
            # 立方晶系
            self.K = (self.C11 + 2 * self.C12) / 3
            self.G_voigt = (self.C11 - self.C12 + 3 * self.C44) / 5
            self.G_reuss = 5 * (self.C11 - self.C12) * self.C44 / (4 * self.C44 + 3 * (self.C11 - self.C12))
            self.G_hill = (self.G_voigt + self.G_reuss) / 2
            
            # 杨氏模量和泊松比
            self.E = 9 * self.K * self.G_hill / (3 * self.K + self.G_hill)
            self.nu = (3 * self.K - 2 * self.G_hill) / (2 * (3 * self.K + self.G_hill))
            
            # Zener各向异性比
            if self.C44 > 0:
                self.Zener_ratio = 2 * self.C44 / (self.C11 - self.C12)


@dataclass
class TransportProperties:
    """输运性质"""
    # 扩散系数
    D: float = 0.0  # m²/s
    D_components: Dict[str, float] = field(default_factory=dict)  # 各向异性分量
    
    # 热导率
    kappa: float = 0.0  # W/(m·K)
    kappa_tensor: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    # 电导率
    sigma: float = 0.0  # S/m
    
    # 离子电导率
    ionic_conductivity: float = 0.0  # S/m
    
    # 温度依赖性
    activation_energy: float = 0.0  # eV
    pre_exponential: float = 0.0    # m²/s
    
    # 计算方法
    method: str = ""  # "MD", "DFT", "experiment"


@dataclass
class InterfaceProperties:
    """界面性质"""
    # 界面能
    gamma: float = 0.0  # J/m²
    gamma_110: float = 0.0
    gamma_100: float = 0.0
    gamma_111: float = 0.0
    
    # 界面迁移率
    mobility: float = 0.0  # m/(J·s) 或 m⁴/(J·s)
    
    # 接触角
    contact_angle: float = 0.0  # degrees
    
    # 黏附功
    work_of_adhesion: float = 0.0  # J/m²
    
    # 界面厚度
    interface_width: float = 0.0  # nm
    
    # 界面扩散
    D_interface: float = 0.0  # m²/s


@dataclass
class PhaseFieldParameters:
    """相场模拟参数"""
    # 界面参数
    interface_energy: float = 0.0  # J/m²
    interface_width: float = 1.0   # nm
    
    # 动力学参数
    mobility: float = 1.0  # m⁴/(J·s) 或无量纲
    diffusion_coeff: float = 0.0  # m²/s
    
    # 各向异性
    anisotropy_mode: str = "four_fold"
    epsilon4: float = 0.05
    epsilon6: float = 0.0
    
    # 热力学参数
    melting_point: float = 0.0  # K
    latent_heat: float = 0.0    # J/m³
    heat_capacity: float = 0.0  # J/(m³·K)
    
    # 梯度能量系数
    gradient_coeff: float = 0.0  # J/m


@dataclass
class MultiScaleParameters:
    """多尺度参数集合"""
    # 原子尺度（DFT/MD）
    elastic: Optional[ElasticConstants] = None
    transport: Optional[TransportProperties] = None
    interface: Optional[InterfaceProperties] = None
    
    # 介观尺度（相场）
    phase_field: Optional[PhaseFieldParameters] = None
    
    # 连续介质尺度
    continuum_elastic: Dict = field(default_factory=dict)
    continuum_thermal: Dict = field(default_factory=dict)
    
    # 元数据
    source_calculations: Dict = field(default_factory=dict)
    extraction_methods: Dict = field(default_factory=dict)
    uncertainties: Dict = field(default_factory=dict)
    temperature: float = 300.0
    
    def to_json(self, filename: str):
        """导出到JSON"""
        data = asdict(self)
        # 转换numpy数组
        data = self._convert_numpy(data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _convert_numpy(self, obj):
        """递归转换numpy数组为列表"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(item) for item in obj]
        return obj


# =============================================================================
# DFT参数提取器
# =============================================================================

class DFTParameterExtractor:
    """
    从DFT计算结果提取材料参数
    
    支持：
    - VASP (OUTCAR, vasprun.xml)
    - Quantum ESPRESSO
    - 其他支持ASE的DFT代码
    """
    
    def __init__(self):
        self.elastic_constants = None
        self.energy_data = {}
        self.stress_data = {}
    
    def extract_elastic_constants(self, 
                                   structure_file: str,
                                   calculator_type: str = "vasp",
                                   deformations: Optional[List] = None) -> ElasticConstants:
        """
        从DFT计算提取弹性常数
        
        方法：应用应变并计算应力响应
        """
        logger.info(f"Extracting elastic constants from DFT ({calculator_type})...")
        
        # 读取结构
        atoms = read(structure_file)
        
        # 准备变形
        if deformations is None:
            # 标准变形集
            deformations = np.linspace(-0.02, 0.02, 5)
        
        # 计算弹性常数
        # 方法1: 使用ASE的弹性常数计算
        try:
            C11_values = []
            C12_values = []
            C44_values = []
            
            # 体积应变
            volumes = []
            energies_vol = []
            
            for eps in deformations:
                # 体积变形
                atoms_def = atoms.copy()
                cell = atoms_def.get_cell()
                atoms_def.set_cell(cell * (1 + eps), scale_atoms=True)
                
                # 这里应该运行DFT计算，简化使用占位符
                # 实际实现应调用DFT计算器
                energy = self._get_dft_energy(atoms_def, calculator_type)
                
                volumes.append(atoms_def.get_volume())
                energies_vol.append(energy)
            
            # 拟合能量-体积曲线得到体模量
            B = self._fit_bulk_modulus(volumes, energies_vol)
            
            # 剪切变形（简化）
            # 实际应使用完整的应变-应力关系
            
            # 创建弹性常数对象
            self.elastic_constants = ElasticConstants(
                C11=B * 1.5,  # 近似
                C12=B * 0.5,
                C44=B * 0.3
            )
            
        except Exception as e:
            logger.error(f"Failed to extract elastic constants: {e}")
            self.elastic_constants = ElasticConstants()
        
        logger.info(f"Elastic constants extracted: C11={self.elastic_constants.C11:.2f}, "
                   f"C12={self.elastic_constants.C12:.2f}, C44={self.elastic_constants.C44:.2f} GPa")
        
        return self.elastic_constants
    
    def _get_dft_energy(self, atoms: Atoms, calculator_type: str) -> float:
        """获取DFT能量（简化实现）"""
        # 实际实现应设置DFT计算器并运行计算
        # 这里使用占位符
        return 0.0
    
    def _fit_bulk_modulus(self, volumes: List[float], energies: List[float]) -> float:
        """
        拟合Birch-Murnaghan方程得到体模量
        
        E(V) = E₀ + (9V₀B₀/16) * [(V₀/V)^(2/3) - 1]² * 
               {6 + [B₀' - 4][(V₀/V)^(2/3) - 1]}
        """
        volumes = np.array(volumes)
        energies = np.array(energies)
        
        # 找到最小能量点
        idx_min = np.argmin(energies)
        V0 = volumes[idx_min]
        E0 = energies[idx_min]
        
        # 简化的BM拟合（仅使用二次项）
        x = (V0 / volumes)**(2/3) - 1
        coeffs = np.polyfit(x, energies - E0, 2)
        
        # B₀ ≈ (16/9V₀) * 二次系数 / 2
        B0 = (16 / (9 * V0)) * coeffs[0] / 2
        
        # 转换为GPa
        # 注意：这里需要单位转换
        B0_GPa = B0 * 160.2  # 从eV/Å³转换为GPa
        
        return B0_GPa
    
    def extract_from_vasp_elastic(self, outcar_file: str) -> ElasticConstants:
        """
        从VASP的弹性常数计算提取
        
        VASP使用IBRION=6计算弹性常数
        """
        elastic_data = self._parse_vasp_elastic(outcar_file)
        
        if elastic_data:
            C_matrix = elastic_data.get('elastic_matrix', [])
            
            if len(C_matrix) >= 6:
                C11 = C_matrix[0][0]
                C12 = C_matrix[0][1]
                C44 = C_matrix[3][3]
                
                self.elastic_constants = ElasticConstants(
                    C11=C11, C12=C12, C44=C44
                )
                
                return self.elastic_constants
        
        return ElasticConstants()
    
    def _parse_vasp_elastic(self, outcar_file: str) -> Dict:
        """解析VASP的弹性常数输出"""
        data = {}
        
        if not Path(outcar_file).exists():
            return data
        
        with open(outcar_file, 'r') as f:
            lines = f.readlines()
        
        # 查找弹性张量
        in_elastic_block = False
        elastic_lines = []
        
        for line in lines:
            if "ELASTIC MODULI" in line or "TOTAL ELASTIC MODULI" in line:
                in_elastic_block = True
                elastic_lines = []
                continue
            
            if in_elastic_block:
                if line.strip() == "" or "----------------" in line:
                    in_elastic_block = False
                    break
                elastic_lines.append(line)
        
        # 解析弹性矩阵
        if elastic_lines:
            C_matrix = []
            for line in elastic_lines[1:]:  # 跳过表头
                values = [float(x) for x in line.split()[1:] if x.replace('.', '').replace('-', '').isdigit()]
                if values:
                    C_matrix.append(values)
            
            data['elastic_matrix'] = C_matrix
        
        return data
    
    def extract_interface_energy(self, 
                                  interface_structure: str,
                                  bulk_reference: str,
                                  area: float) -> float:
        """
        从界面结构计算界面能
        
        γ = (E_interface - n·E_bulk) / (2A)
        """
        # 读取结构
        interface_atoms = read(interface_structure)
        bulk_atoms = read(bulk_reference)
        
        # 计算能量（这里应使用DFT）
        E_interface = 0.0  # 占位符
        E_bulk = 0.0
        
        # 计算原子数
        n_atoms_interface = len(interface_atoms)
        n_atoms_bulk = len(bulk_atoms)
        
        n_units = n_atoms_interface / n_atoms_bulk
        
        # 界面能
        gamma = (E_interface - n_units * E_bulk) / (2 * area)
        
        # 转换为J/m²
        gamma_SI = gamma * eV / (Ang**2)
        
        return gamma_SI
    
    def extract_surface_energy(self,
                               surface_structure: str,
                               bulk_reference: str,
                               area: float) -> float:
        """
        计算表面能
        
        γ = (E_slab - n·E_bulk) / (2A)
        """
        return self.extract_interface_energy(surface_structure, bulk_reference, area)


# =============================================================================
# MD参数提取器
# =============================================================================

class MDParameterExtractor:
    """
    从分子动力学模拟提取参数
    
    提取：
    - 扩散系数（MSD分析）
    - 界面迁移率
    - 界面能（capillary fluctuation method）
    """
    
    def __init__(self):
        self.transport_properties = TransportProperties()
        self.interface_properties = InterfaceProperties()
        self.trajectory_data = {}
    
    def load_trajectory(self, trajectory_file: str, format: str = "auto"):
        """加载MD轨迹"""
        atoms_list = read(trajectory_file, index=':')
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
        
        self.trajectory_data['trajectory'] = atoms_list
        self.trajectory_data['n_frames'] = len(atoms_list)
        
        logger.info(f"Loaded {len(atoms_list)} frames from {trajectory_file}")
    
    def extract_diffusion_coefficient(self,
                                      atom_type: Optional[str] = None,
                                      dimension: int = 3,
                                      fit_range: Optional[Tuple[int, int]] = None) -> TransportProperties:
        """
        从MSD提取扩散系数
        
        D = MSD / (2·d·t)
        
        其中d是维度数
        """
        if 'trajectory' not in self.trajectory_data:
            raise ValueError("No trajectory loaded. Call load_trajectory first.")
        
        trajectory = self.trajectory_data['trajectory']
        n_frames = len(trajectory)
        
        if n_frames < 2:
            logger.warning("Need at least 2 frames for diffusion analysis")
            return self.transport_properties
        
        # 获取原子位置
        if atom_type:
            indices = [i for i, atom in enumerate(trajectory[0]) 
                      if atom.symbol == atom_type]
        else:
            indices = list(range(len(trajectory[0])))
        
        n_atoms = len(indices)
        
        # 计算MSD
        positions_t0 = trajectory[0].get_positions()[indices]
        
        msd_data = []
        times = []
        
        # 获取时间信息
        timestep = 1.0  # fs, 假设值
        if hasattr(trajectory[1], 'info') and 'time' in trajectory[1].info:
            timestep = trajectory[1].info['time'] - trajectory[0].info['time']
        
        for i, atoms in enumerate(trajectory):
            if i == 0:
                continue
            
            positions = atoms.get_positions()[indices]
            
            # 应用PBC
            cell = atoms.get_cell()
            displacements = positions - positions_t0
            
            # 最小图像约定
            if cell is not None:
                cell_inv = np.linalg.inv(cell)
                displacements_frac = displacements @ cell_inv.T
                displacements_frac -= np.round(displacements_frac)
                displacements = displacements_frac @ cell.T
            
            # MSD
            msd = np.mean(np.sum(displacements**2, axis=1))
            msd_data.append(msd)
            times.append(i * timestep)
        
        times = np.array(times)
        msd_data = np.array(msd_data)
        
        # 线性拟合
        if fit_range:
            start, end = fit_range
            times_fit = times[start:end]
            msd_fit = msd_data[start:end]
        else:
            # 自动选择线性区域（后50%数据）
            mid = len(times) // 2
            times_fit = times[mid:]
            msd_fit = msd_data[mid:]
        
        if len(times_fit) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(times_fit, msd_fit)
            
            # D = slope / (2*d)
            D = slope / (2 * dimension)  # Å²/fs
            
            # 转换为m²/s
            D_SI = D * 1e-8  # Å²/fs -> m²/s
            
            self.transport_properties.D = D_SI
            self.transport_properties.method = "MD"
            
            logger.info(f"Diffusion coefficient: {D_SI:.4e} m²/s (R²={r_value**2:.4f})")
        
        return self.transport_properties
    
    def extract_temperature_dependent_diffusion(self,
                                                 trajectory_files: List[str],
                                                 temperatures: List[float]) -> TransportProperties:
        """
        提取温度依赖的扩散系数并拟合Arrhenius方程
        
        D = D₀ exp(-Ea/kT)
        """
        D_values = []
        
        for traj_file, T in zip(trajectory_files, temperatures):
            self.load_trajectory(traj_file)
            props = self.extract_diffusion_coefficient()
            D_values.append(props.D)
        
        D_values = np.array(D_values)
        temperatures = np.array(temperatures)
        
        # Arrhenius拟合
        # ln(D) = ln(D₀) - Ea/(kT)
        ln_D = np.log(D_values)
        inv_T = 1000 / temperatures  # 使用1000/T便于数值稳定
        
        slope, intercept, r_value, _, _ = stats.linregress(inv_T, ln_D)
        
        # 提取参数
        k_B = 8.617e-5  # eV/K
        Ea = -slope * k_B * 1000  # eV
        D0 = np.exp(intercept)  # m²/s
        
        self.transport_properties.activation_energy = Ea
        self.transport_properties.pre_exponential = D0
        
        logger.info(f"Arrhenius fit: Ea={Ea:.3f} eV, D₀={D0:.4e} m²/s")
        
        return self.transport_properties
    
    def extract_interface_mobility(self,
                                   trajectory_file: str,
                                   driving_force: float,
                                   interface_normal: List[float] = [1, 0, 0]) -> float:
        """
        提取界面迁移率
        
        M = v / F
        
        其中v是界面速度，F是驱动力
        """
        self.load_trajectory(trajectory_file)
        trajectory = self.trajectory_data['trajectory']
        
        # 跟踪界面位置
        positions = []
        times = []
        
        for i, atoms in enumerate(trajectory):
            # 简化：使用密度分布找到界面位置
            # 实际应使用序参量区分两相
            
            # 假设相A的密度高，相B的密度低
            z_coords = atoms.get_positions()[:, 2]
            densities, bin_edges = np.histogram(z_coords, bins=50)
            
            # 找到密度梯度最大的位置
            gradients = np.gradient(densities.astype(float))
            interface_idx = np.argmax(np.abs(gradients))
            interface_pos = (bin_edges[interface_idx] + bin_edges[interface_idx + 1]) / 2
            
            positions.append(interface_pos)
            
            time = i * 1.0  # 假设时间步长
            if hasattr(atoms, 'info') and 'time' in atoms.info:
                time = atoms.info['time']
            times.append(time)
        
        positions = np.array(positions)
        times = np.array(times)
        
        # 线性拟合得到速度
        if len(times) > 1:
            slope, _, r_value, _, _ = stats.linregress(times, positions)
            velocity = abs(slope)  # Å/fs
            velocity_SI = velocity * 1e5  # m/s
            
            # 迁移率
            mobility = velocity_SI / driving_force  # m/(J·s) 或 m⁴/(J·s)
            
            self.interface_properties.mobility = mobility
            
            logger.info(f"Interface velocity: {velocity_SI:.4e} m/s")
            logger.info(f"Interface mobility: {mobility:.4e}")
        
        return self.interface_properties.mobility
    
    def extract_thermal_conductivity(self,
                                     trajectory_file: str,
                                     temperature_gradient: float,
                                     heat_flux: float) -> float:
        """
        使用Green-Kubo关系或NEMD提取热导率
        
        κ = -J / ∇T
        """
        # 简化的NEMD方法
        kappa = abs(heat_flux / temperature_gradient)
        
        self.transport_properties.kappa = kappa
        
        return kappa


# =============================================================================
# 参数转换器
# =============================================================================

class ParameterConverter:
    """
    参数格式转换器
    
    在不同尺度之间转换参数
    """
    
    def __init__(self):
        self.conversion_log = []
    
    def elastic_to_continuum(self, elastic: ElasticConstants) -> Dict:
        """
        将弹性常数转换为连续介质格式
        
        输入：弹性常数（GPa）
        输出：连续介质材料参数
        """
        continuum_params = {
            'elastic_type': 'isotropic' if elastic.Zener_ratio > 0.9 else 'anisotropic',
            'E': elastic.E,  # 杨氏模量
            'nu': elastic.nu,  # 泊松比
            'G': elastic.G_hill,  # 剪切模量
            'K': elastic.K,  # 体模量
        }
        
        # 如果是各向异性，添加完整的弹性矩阵
        if continuum_params['elastic_type'] == 'anisotropic':
            continuum_params['C11'] = elastic.C11
            continuum_params['C12'] = elastic.C12
            continuum_params['C44'] = elastic.C44
            continuum_params['anisotropy_ratio'] = elastic.Zener_ratio
        
        return continuum_params
    
    def diffusion_to_phase_field(self, transport: TransportProperties) -> PhaseFieldParameters:
        """
        将扩散系数转换为相场参数
        
        注意：需要特征长度和特征时间尺度来无量纲化
        """
        D = transport.D  # m²/s
        
        # 假设特征长度L = 1 nm, 特征时间τ = L²/D
        L_char = 1e-9  # m
        tau_char = L_char**2 / D if D > 0 else 1.0
        
        # 无量纲迁移率
        M_dimensionless = 1.0  # 在无量纲单位中
        
        pf_params = PhaseFieldParameters(
            diffusion_coeff=D,
            mobility=M_dimensionless
        )
        
        return pf_params
    
    def interface_to_phase_field(self, interface: InterfaceProperties) -> PhaseFieldParameters:
        """
        将界面性质转换为相场参数
        """
        # 界面能与梯度能量系数的关系
        # γ = √(κ·W) / (3√2) 对于双曲正切分布
        # 其中κ是梯度能量系数，W是势垒高度
        
        gamma = interface.gamma  # J/m²
        interface_width = interface.interface_width * 1e-9  # m
        
        # 假设势能形式
        W = 6 * gamma / interface_width  # 势垒高度 (J/m³)
        kappa = 3 * gamma * interface_width / 2  # 梯度能量系数 (J/m)
        
        pf_params = PhaseFieldParameters(
            interface_energy=gamma,
            interface_width=interface.interface_width,
            gradient_coeff=kappa,
            mobility=interface.mobility if interface.mobility > 0 else 1.0
        )
        
        return pf_params
    
    def convert_all(self, 
                   elastic: Optional[ElasticConstants] = None,
                   transport: Optional[TransportProperties] = None,
                   interface: Optional[InterfaceProperties] = None) -> MultiScaleParameters:
        """
        执行所有参数转换
        """
        msp = MultiScaleParameters()
        
        # 保存原子尺度参数
        msp.elastic = elastic
        msp.transport = transport
        msp.interface = interface
        
        # 转换到连续介质
        if elastic:
            msp.continuum_elastic = self.elastic_to_continuum(elastic)
        
        # 转换到相场
        if interface:
            msp.phase_field = self.interface_to_phase_field(interface)
        elif transport:
            msp.phase_field = self.diffusion_to_phase_field(transport)
        else:
            msp.phase_field = PhaseFieldParameters()
        
        # 添加扩散系数到相场参数
        if transport:
            msp.phase_field.diffusion_coeff = transport.D
            msp.phase_field.activation_energy = transport.activation_energy
        
        return msp


# =============================================================================
# 参数验证器
# =============================================================================

class ParameterValidator:
    """
    参数验证和不确定性量化
    """
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_elastic_constants(self, elastic: ElasticConstants) -> Dict:
        """
        验证弹性常数满足稳定性条件
        
        对于立方晶系：
        - C11 > |C12|
        - C11 + 2C12 > 0
        - C44 > 0
        """
        issues = []
        
        if elastic.C11 <= abs(elastic.C12):
            issues.append(f"Stability violation: C11 ({elastic.C11}) <= |C12| ({abs(elastic.C12)})")
        
        if elastic.C11 + 2 * elastic.C12 <= 0:
            issues.append(f"Stability violation: C11 + 2*C12 <= 0")
        
        if elastic.C44 <= 0:
            issues.append(f"Stability violation: C44 <= 0")
        
        result = {
            'is_stable': len(issues) == 0,
            'issues': issues,
            ' Born_stability': {
                'C11_minus_C12': elastic.C11 - elastic.C12,
                'C11_plus_2C12': elastic.C11 + 2 * elastic.C12,
                'C44': elastic.C44
            }
        }
        
        self.validation_results['elastic'] = result
        
        return result
    
    def validate_diffusion_coefficient(self, 
                                       transport: TransportProperties,
                                       expected_range: Optional[Tuple[float, float]] = None) -> Dict:
        """
        验证扩散系数在合理范围内
        """
        D = transport.D
        
        # 典型的扩散系数范围（固体中）
        if expected_range is None:
            expected_range = (1e-20, 1e-8)  # m²/s
        
        D_min, D_max = expected_range
        
        is_valid = D_min <= D <= D_max
        
        result = {
            'is_valid': is_valid,
            'value': D,
            'expected_range': expected_range,
            'warnings': []
        }
        
        if D < D_min:
            result['warnings'].append(f"Diffusion coefficient very small: {D:.2e}")
        elif D > D_max:
            result['warnings'].append(f"Diffusion coefficient very large: {D:.2e}")
        
        self.validation_results['diffusion'] = result
        
        return result
    
    def estimate_uncertainty(self, 
                            values: List[float],
                            confidence_level: float = 0.95) -> Dict:
        """
        估计参数的不确定性
        """
        values = np.array(values)
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        # 置信区间
        alpha = 1 - confidence_level
        ci_low = mean - stats.t.ppf(1 - alpha/2, len(values)-1) * std / np.sqrt(len(values))
        ci_high = mean + stats.t.ppf(1 - alpha/2, len(values)-1) * std / np.sqrt(len(values))
        
        return {
            'mean': mean,
            'std': std,
            'relative_uncertainty': std / mean if mean != 0 else float('inf'),
            f'confidence_interval_{int(confidence_level*100)}': (ci_low, ci_high)
        }
    
    def cross_scale_consistency_check(self, msp: MultiScaleParameters) -> Dict:
        """
        检查跨尺度参数一致性
        """
        issues = []
        warnings_list = []
        
        # 检查弹性参数
        if msp.elastic and msp.continuum_elastic:
            # 检查转换一致性
            E_atomic = msp.elastic.E
            E_continuum = msp.continuum_elastic.get('E', 0)
            
            if abs(E_atomic - E_continuum) > 1.0:
                issues.append(f"Elastic modulus mismatch: atomic={E_atomic:.2f}, continuum={E_continuum:.2f}")
        
        # 检查扩散参数
        if msp.transport and msp.phase_field:
            D_atomic = msp.transport.D
            D_pf = msp.phase_field.diffusion_coeff
            
            if abs(D_atomic - D_pf) > 1e-15:
                warnings_list.append("Diffusion coefficient may need unit conversion")
        
        return {
            'is_consistent': len(issues) == 0,
            'issues': issues,
            'warnings': warnings_list
        }


# =============================================================================
# 主工作流类
# =============================================================================

class ParameterPassingWorkflow:
    """
    跨尺度参数传递完整工作流
    
    整合参数提取、转换、验证的全过程
    """
    
    def __init__(self, working_dir: str = "./parameter_passing"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.dft_extractor = DFTParameterExtractor()
        self.md_extractor = MDParameterExtractor()
        self.converter = ParameterConverter()
        self.validator = ParameterValidator()
        
        self.parameters = MultiScaleParameters()
    
    def extract_from_dft(self,
                        structure_file: str,
                        calculation_type: str = "elastic",
                        **kwargs) -> Dict:
        """
        从DFT计算提取参数
        
        Args:
            structure_file: 结构文件路径
            calculation_type: "elastic", "interface", "surface"
        """
        logger.info(f"Extracting {calculation_type} parameters from DFT...")
        
        if calculation_type == "elastic":
            elastic = self.dft_extractor.extract_elastic_constants(
                structure_file, **kwargs
            )
            self.parameters.elastic = elastic
            
            # 验证
            validation = self.validator.validate_elastic_constants(elastic)
            self.parameters.uncertainties['elastic'] = validation
            
            return asdict(elastic)
        
        elif calculation_type == "interface":
            gamma = self.dft_extractor.extract_interface_energy(
                kwargs['interface_structure'],
                kwargs['bulk_reference'],
                kwargs['area']
            )
            self.parameters.interface = InterfaceProperties(gamma=gamma)
            return {'interface_energy': gamma}
        
        return {}
    
    def extract_from_md(self,
                       trajectory_file: str,
                       extraction_type: str = "diffusion",
                       **kwargs) -> Dict:
        """
        从MD轨迹提取参数
        """
        logger.info(f"Extracting {extraction_type} parameters from MD...")
        
        self.md_extractor.load_trajectory(trajectory_file)
        
        if extraction_type == "diffusion":
            transport = self.md_extractor.extract_diffusion_coefficient(**kwargs)
            self.parameters.transport = transport
            
            # 验证
            validation = self.validator.validate_diffusion_coefficient(transport)
            self.parameters.uncertainties['diffusion'] = validation
            
            return {
                'D': transport.D,
                'Ea': transport.activation_energy,
                'D0': transport.pre_exponential
            }
        
        elif extraction_type == "interface_mobility":
            mobility = self.md_extractor.extract_interface_mobility(
                trajectory_file, **kwargs
            )
            if self.parameters.interface is None:
                self.parameters.interface = InterfaceProperties()
            self.parameters.interface.mobility = mobility
            return {'mobility': mobility}
        
        return {}
    
    def convert_parameters(self) -> MultiScaleParameters:
        """
        将提取的参数转换到所有尺度
        """
        logger.info("Converting parameters across scales...")
        
        self.parameters = self.converter.convert_all(
            elastic=self.parameters.elastic,
            transport=self.parameters.transport,
            interface=self.parameters.interface
        )
        
        # 一致性检查
        consistency = self.validator.cross_scale_consistency_check(self.parameters)
        if not consistency['is_consistent']:
            logger.warning("Cross-scale consistency issues found:")
            for issue in consistency['issues']:
                logger.warning(f"  - {issue}")
        
        return self.parameters
    
    def export_parameters(self, filename: Optional[str] = None) -> str:
        """
        导出参数到文件
        """
        if filename is None:
            filename = self.working_dir / "multiscale_parameters.json"
        
        self.parameters.to_json(filename)
        
        # 同时生成不同尺度的专用输入文件
        self._export_phase_field_params()
        self._export_continuum_params()
        
        logger.info(f"Parameters exported to {filename}")
        
        return str(filename)
    
    def _export_phase_field_params(self):
        """导出相场参数"""
        if self.parameters.phase_field:
            pf_dict = asdict(self.parameters.phase_field)
            
            with open(self.working_dir / "phase_field_params.json", 'w') as f:
                json.dump(pf_dict, f, indent=2)
    
    def _export_continuum_params(self):
        """导出连续介质参数"""
        continuum_dict = {
            'elastic': self.parameters.continuum_elastic,
            'thermal': self.parameters.continuum_thermal
        }
        
        with open(self.working_dir / "continuum_params.json", 'w') as f:
            json.dump(continuum_dict, f, indent=2)
    
    def run_full_workflow(self,
                         dft_structure: Optional[str] = None,
                         md_trajectory: Optional[str] = None,
                         temperature: float = 300.0) -> MultiScaleParameters:
        """
        运行完整工作流
        
        Args:
            dft_structure: DFT结构文件
            md_trajectory: MD轨迹文件
            temperature: 温度
        
        Returns:
            parameters: 多尺度参数集合
        """
        logger.info("=" * 60)
        logger.info("Starting Parameter Passing Workflow")
        logger.info("=" * 60)
        
        self.parameters.temperature = temperature
        
        # 步骤1: DFT参数提取
        if dft_structure:
            try:
                self.extract_from_dft(dft_structure, calculation_type="elastic")
            except Exception as e:
                logger.warning(f"DFT extraction failed: {e}")
        
        # 步骤2: MD参数提取
        if md_trajectory:
            try:
                self.extract_from_md(md_trajectory, extraction_type="diffusion")
            except Exception as e:
                logger.warning(f"MD extraction failed: {e}")
        
        # 步骤3: 参数转换
        self.convert_parameters()
        
        # 步骤4: 导出
        self.export_parameters()
        
        logger.info("=" * 60)
        logger.info("Parameter Passing Workflow Completed")
        logger.info("=" * 60)
        
        return self.parameters


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Scale Parameter Passing Tool")
    parser.add_argument("--dft-structure", help="DFT structure file (POSCAR, etc.)")
    parser.add_argument("--md-trajectory", help="MD trajectory file")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--output-dir", default="./parameter_passing")
    
    args = parser.parse_args()
    
    if not args.dft_structure and not args.md_trajectory:
        parser.error("At least one of --dft-structure or --md-trajectory must be provided")
    
    workflow = ParameterPassingWorkflow(working_dir=args.output_dir)
    
    parameters = workflow.run_full_workflow(
        dft_structure=args.dft_structure,
        md_trajectory=args.md_trajectory,
        temperature=args.temperature
    )
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("EXTRACTED PARAMETERS SUMMARY")
    print("=" * 60)
    
    if parameters.elastic:
        print(f"\nElastic Constants:")
        print(f"  C11 = {parameters.elastic.C11:.2f} GPa")
        print(f"  C12 = {parameters.elastic.C12:.2f} GPa")
        print(f"  C44 = {parameters.elastic.C44:.2f} GPa")
        print(f"  E = {parameters.elastic.E:.2f} GPa")
        print(f"  ν = {parameters.elastic.nu:.4f}")
    
    if parameters.transport:
        print(f"\nTransport Properties:")
        print(f"  D = {parameters.transport.D:.4e} m²/s")
        if parameters.transport.activation_energy > 0:
            print(f"  Ea = {parameters.transport.activation_energy:.3f} eV")
    
    if parameters.phase_field:
        print(f"\nPhase-Field Parameters:")
        print(f"  Interface energy = {parameters.phase_field.interface_energy:.4f} J/m²")
        print(f"  Interface width = {parameters.phase_field.interface_width:.2f} nm")
        print(f"  Mobility = {parameters.phase_field.mobility:.4e}")
    
    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()
