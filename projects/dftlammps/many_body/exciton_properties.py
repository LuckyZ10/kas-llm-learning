#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exciton Properties Module
=========================

激子物理性质计算模块，包括:
- 激子结合能计算
- 激子玻尔半径和有效质量
- 激子-声子耦合
- 温度依赖的光学性质
- 激子扩散动力学

适用于:
- 3D块体半导体
- 2D材料 (过渡金属硫化物等)
- 1D纳米线
- 量子点
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from scipy import integrate, optimize
from scipy.interpolate import interp1d
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 物理常数
# =============================================================================

# 氢原子单位
RYDBERG = 13.6056980659  # eV
BOHR_RADIUS = 0.529177210903  # Angstrom

# 材料常数
EPSILON_0 = 8.8541878128e-12  # F/m, 真空介电常数
H_BAR = 6.582119569e-16  # eV·s
KB = 8.617333262e-5  # eV/K, 玻尔兹曼常数


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class MaterialParameters:
    """材料参数"""
    name: str
    dimensionality: int = 3  # 3, 2, or 1
    
    # 介电常数
    epsilon_static: float = 10.0  # 静态介电常数
    epsilon_high: float = 10.0  # 高频介电常数
    epsilon_parallel: Optional[float] = None  # 2D材料平行方向
    epsilon_perp: Optional[float] = None  # 2D材料垂直方向
    
    # 有效质量 (电子质量为单位)
    m_e_eff: float = 1.0  # 电子有效质量
    m_h_eff: float = 1.0  # 空穴有效质量
    
    # 带隙 (eV)
    band_gap: float = 1.0
    spin_orbit_splitting: float = 0.0  # 自旋轨道耦合分裂
    
    # 晶格参数
    lattice_constant: Optional[float] = None  # Angstrom
    layer_thickness: Optional[float] = None  # 2D材料层厚 (Angstrom)
    
    # 声子
    optical_phonon_energy: float = 0.03  # eV, LO声子能量
    acoustic_deformation_potential: float = 7.0  # eV
    sound_velocity: float = 5e5  # cm/s
    
    @property
    def reduced_mass(self) -> float:
        """约化质量"""
        return (self.m_e_eff * self.m_h_eff) / (self.m_e_eff + self.m_h_eff)
    
    @property
    def average_mass(self) -> float:
        """平均质量 (用于激子质心运动)"""
        return self.m_e_eff + self.m_h_eff


@dataclass
class ExcitonState:
    """激子态"""
    n: int  # 主量子数 (1, 2, 3, ...)
    l: int = 0  # 角动量量子数
    m: int = 0  # 磁量子数
    
    # 能量 (eV)
    binding_energy: float = 0.0  # 结合能 (正值)
    transition_energy: float = 0.0  # 跃迁能量 = gap - binding_energy
    
    # 空间特性
    bohr_radius: float = 0.0  # Bohr (Angstrom)
    rms_radius: float = 0.0  # RMS半径
    
    # 光学性质
    oscillator_strength: float = 0.0
    lifetime: float = 0.0  # ps
    
    # 温度展宽
    homogeneous_width: float = 0.0  # eV
    inhomogeneous_width: float = 0.0  # eV
    
    def decay_rate(self, mechanism: str = "radiative") -> float:
        """衰减率 (ns^-1)"""
        if mechanism == "radiative":
            if self.lifetime > 0:
                return 1.0 / self.lifetime  # ps^-1
        return 0.0


@dataclass
class ExcitonPhononCoupling:
    """激子-声子耦合参数"""
    
    # Frohlich耦合常数
    g_LO: float = 0.0  # LO声子耦合强度 (无量纲)
    
    # 形变势耦合
    D_ac: float = 0.0  # 声学声子形变势 (eV)
    
    #  Huang-Rhys因子 (电声耦合强度)
    huang_rhys_factor: float = 0.0
    
    # 声子边带
    phonon_sideband_energy: float = 0.0  # eV
    
    # 自能修正
    self_energy_real: float = 0.0  # eV
    self_energy_imag: float = 0.0  # eV (寿命)


# =============================================================================
# 预定义材料数据库
# =============================================================================

MATERIAL_DATABASE = {
    "GaAs": MaterialParameters(
        name="GaAs",
        dimensionality=3,
        epsilon_static=12.9,
        epsilon_high=10.9,
        m_e_eff=0.067,
        m_h_eff=0.50,
        band_gap=1.519,
        optical_phonon_energy=0.036,
        lattice_constant=5.65,
    ),
    
    "Si": MaterialParameters(
        name="Silicon",
        dimensionality=3,
        epsilon_static=11.7,
        epsilon_high=11.7,
        m_e_eff=0.98,  # 纵向
        m_h_eff=0.49,
        band_gap=1.17,
        optical_phonon_energy=0.062,
        lattice_constant=5.43,
    ),
    
    "MoS2": MaterialParameters(
        name="MoS2",
        dimensionality=2,
        epsilon_static=4.5,
        epsilon_high=4.5,
        epsilon_parallel=4.5,
        epsilon_perp=1.0,  # 真空
        m_e_eff=0.47,
        m_h_eff=0.54,
        band_gap=1.89,  # 单层，光学带隙
        optical_phonon_energy=0.048,
        layer_thickness=6.5,
    ),
    
    "WSe2": MaterialParameters(
        name="WSe2",
        dimensionality=2,
        epsilon_static=5.0,
        epsilon_high=5.0,
        m_e_eff=0.34,
        m_h_eff=0.44,
        band_gap=1.65,
        optical_phonon_energy=0.030,
        layer_thickness=6.5,
    ),
    
    "hBN": MaterialParameters(
        name="hBN",
        dimensionality=2,
        epsilon_static=3.5,
        epsilon_high=3.5,
        m_e_eff=0.78,
        m_h_eff=0.74,
        band_gap=6.08,
        optical_phonon_energy=0.170,
        layer_thickness=3.3,
    ),
    
    "GaN": MaterialParameters(
        name="GaN",
        dimensionality=3,
        epsilon_static=8.9,
        epsilon_high=5.35,
        m_e_eff=0.20,
        m_h_eff=0.80,
        band_gap=3.51,
        optical_phonon_energy=0.091,
        lattice_constant=3.19,
    ),
}


def get_material(name: str) -> MaterialParameters:
    """从数据库获取材料参数"""
    if name not in MATERIAL_DATABASE:
        raise ValueError(f"Unknown material: {name}. Available: {list(MATERIAL_DATABASE.keys())}")
    return MATERIAL_DATABASE[name]


# =============================================================================
# 激子结合能计算
# =============================================================================

class ExcitonBindingEnergy:
    """
    激子结合能计算器
    
    使用Wannier-Mott模型计算激子性质
    """
    
    def __init__(self, material: MaterialParameters):
        self.mat = material
        self.dimensionality = material.dimensionality
    
    def hydrogenic_binding_energy(self, n: int = 1) -> float:
        """
        计算类氢激子结合能
        
        Parameters:
        -----------
        n : int
            主量子数
        
        Returns:
        --------
        float : 结合能 (eV)
        """
        # 3D情况
        if self.dimensionality == 3:
            return self._binding_energy_3d(n)
        # 2D情况
        elif self.dimensionality == 2:
            return self._binding_energy_2d(n)
        # 1D情况
        elif self.dimensionality == 1:
            return self._binding_energy_1d(n)
        else:
            raise ValueError(f"Invalid dimensionality: {self.dimensionality}")
    
    def _binding_energy_3d(self, n: int) -> float:
        """3D激子结合能 (Rydberg公式)"""
        mu = self.mat.reduced_mass
        eps = self.mat.epsilon_static
        
        # E_n = - (mu / eps^2) * Ry / n^2
        E_b = (mu / eps**2) * RYDBERG / (n**2)
        return E_b
    
    def _binding_energy_2d(self, n: int) -> float:
        """2D激子结合能"""
        mu = self.mat.reduced_mass
        eps = self.mat.epsilon_static
        
        # 严格2D极限: E_n = 4 * Ry / (2n-1)^2 (非屏蔽)
        # 考虑屏蔽: 使用Rytova-Keldysh势
        # 简化: E_b = 4 * mu / eps^2 * Ry / (2n-1)^2
        E_b = 4.0 * (mu / eps**2) * RYDBERG / ((2*n - 1)**2)
        
        return E_b
    
    def _binding_energy_1d(self, n: int) -> float:
        """1D激子结合能"""
        # 1D激子结合能更强
        mu = self.mat.reduced_mass
        eps = self.mat.epsilon_static
        
        # 近似公式
        E_b = 4.0 * (mu / eps**2) * RYDBERG / (n**2)
        return E_b
    
    def rydberg_series(self, n_max: int = 5) -> List[ExcitonState]:
        """
        计算Rydberg激子系列
        
        Parameters:
        -----------
        n_max : int
            最大主量子数
        
        Returns:
        --------
        List[ExcitonState] : 激子态列表
        """
        states = []
        
        for n in range(1, n_max + 1):
            # 1s, 2s, 2p, ...
            for l in range(n):
                E_b = self.hydrogenic_binding_energy(n)
                a_b = self.bohr_radius(n)
                
                # 振子强度 ~ 1/n^3 for s态
                if l == 0:
                    f = 1.0 / (n**3)
                else:
                    f = 0.1 / (n**3)  # p, d态较弱
                
                state = ExcitonState(
                    n=n,
                    l=l,
                    m=0,
                    binding_energy=E_b,
                    transition_energy=self.mat.band_gap - E_b,
                    bohr_radius=a_b,
                    rms_radius=a_b * np.sqrt(n**2 * (5*n**2 + 1 - 3*l*(l+1)) / 2) / n,
                    oscillator_strength=f,
                )
                states.append(state)
        
        return sorted(states, key=lambda x: x.binding_energy, reverse=True)
    
    def bohr_radius(self, n: int = 1, l: int = 0) -> float:
        """
        计算激子Bohr半径
        
        Parameters:
        -----------
        n : int
            主量子数
        l : int
            角动量量子数
        
        Returns:
        --------
        float : Bohr半径 (Angstrom)
        """
        mu = self.mat.reduced_mass
        eps = self.mat.epsilon_static
        
        # a_n = (eps / mu) * a_0 * n^2 / (对于l=0)
        a_0 = BOHR_RADIUS  # Angstrom
        
        if self.dimensionality == 3:
            a_b = (eps / mu) * a_0 * n**2
        elif self.dimensionality == 2:
            # 2D激子半径
            a_b = (eps / mu) * a_0 * (2*n - 1)
        else:
            a_b = (eps / mu) * a_0 * n**2
        
        return a_b
    
    def effective_rydberg(self) -> float:
        """有效Rydberg能量 (eV)"""
        mu = self.mat.reduced_mass
        eps = self.mat.epsilon_static
        return (mu / eps**2) * RYDBERG
    
    def keldysh_parameter(self) -> float:
        """
        计算Keldysh参数 (2D材料)
        
        r* = (epsilon_parallel / epsilon_perp) * (layer_thickness / a_0)
        
        Returns:
        --------
        float : 无量纲Keldysh参数
        """
        if self.dimensionality != 2:
            return float('inf')
        
        eps_para = self.mat.epsilon_parallel or self.mat.epsilon_static
        eps_perp = self.mat.epsilon_perp or 1.0
        d = self.mat.layer_thickness or 6.0  # Angstrom
        
        r_star = (eps_para / eps_perp) * (d / BOHR_RADIUS)
        return r_star
    
    def rytova_keldysh_potential(self, r: np.ndarray) -> np.ndarray:
        """
        Rytova-Keldysh势 (2D材料屏蔽库仑势)
        
        V(r) = - (e^2 / (2*epsilon_0)) * (1 / (r_star * a_0)) * 
               [H_0(r/r_0) - Y_0(r/r_0)]
        
        其中 r_0 = (epsilon_parallel / epsilon_perp) * d
        """
        from scipy.special import struve, yn
        
        r_star = self.keldysh_parameter()
        r_0 = r_star * BOHR_RADIUS  # 有效Bohr半径
        
        # 简化形式 (大r极限)
        x = r / r_0
        
        # 对于大r，V ~ -1/r (库仑)
        # 对于小r，V ~ -ln(r) (对数)
        
        # 使用近似
        V = np.zeros_like(r)
        mask_small = x < 1.0
        mask_large = x >= 1.0
        
        V[mask_small] = -np.log(1.0 / x[mask_small] + 1.0)
        V[mask_large] = -1.0 / x[mask_large]
        
        # 单位转换
        prefactor = (RYDBERG * BOHR_RADIUS) / (self.mat.epsilon_static * r_0)
        V *= prefactor
        
        return V
    
    def solve_2d_keldysh_exciton(self, n_states: int = 5) -> List[ExcitonState]:
        """
        数值求解2D Keldysh激子
        
        使用变分法或数值积分
        """
        # 变分波函数: psi(r) = sqrt(2/pi) * (1/a) * exp(-r/a)
        
        def variational_energy(a: float) -> float:
            """计算变分能量"""
            # 动能项
            T = (H_BAR**2) / (2 * self.mat.reduced_mass * 9.109e-31 * a**2 * 1e-20)
            T /= 1.602e-19  # 转换为eV
            
            # 势能项 (数值积分)
            r = np.linspace(0.01, 10*a, 1000)
            V = self.rytova_keldysh_potential(r)
            
            # |psi|^2 = (2/pi) * (1/a^2) * exp(-2r/a)
            psi2 = (2/np.pi) * (1/a**2) * np.exp(-2*r/a)
            
            # <V> = integral psi^2 V 2*pi*r dr
            V_avg = np.trapz(psi2 * V * 2 * np.pi * r, r)
            
            return T + V_avg
        
        # 最小化能量
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(variational_energy, bounds=(0.1, 100), method='bounded')
        a_opt = result.x
        E_opt = result.fun
        
        # 结合能 = -E_opt
        E_binding = -E_opt
        
        # 构建激子态
        states = []
        for n in range(1, n_states + 1):
            # 简化: 假设类似氢原子能级缩放
            E_n = E_binding / (n - 0.5)**2  # 2D修正
            a_n = a_opt * (n - 0.5)
            
            state = ExcitonState(
                n=n,
                l=0,
                binding_energy=E_n,
                transition_energy=self.mat.band_gap - E_n,
                bohr_radius=a_n,
                oscillator_strength=1.0/n**3,
            )
            states.append(state)
        
        return states


# =============================================================================
# 激子-声子耦合
# =============================================================================

class ExcitonPhononInteraction:
    """
    激子-声子相互作用
    
    包括:
    - Frohlich耦合 (LO声子)
    - 形变势耦合 (声学声子)
    - 自能修正
    """
    
    def __init__(self, material: MaterialParameters, exciton: ExcitonState):
        self.mat = material
        self.exciton = exciton
    
    def frohlich_coupling(self, phonon_energy: Optional[float] = None) -> float:
        """
        计算Frohlich耦合常数
        
        g_F = sqrt(alpha_F * hbar*omega_LO)
        
        其中 alpha_F = (e^2 / (4*pi*epsilon_0*hbar*omega_LO)) * 
                      (1/epsilon_inf - 1/epsilon_0) * sqrt(2*m*omega_LO/hbar)
        """
        hbar_omega = phonon_energy or self.mat.optical_phonon_energy
        
        eps_0 = self.mat.epsilon_static
        eps_inf = self.mat.epsilon_high
        m = self.mat.reduced_mass
        
        # Frohlich耦合常数
        alpha_F = (1 / hbar_omega) * (1/eps_inf - 1/eps_0) * np.sqrt(m / hbar_omega)
        
        # 无量纲耦合常数 (多体系统)
        g = np.sqrt(alpha_F * hbar_omega)
        
        return g
    
    def deformation_potential_coupling(self, q: np.ndarray) -> np.ndarray:
        """
        形变势耦合矩阵元
        
        |M(q)|^2 = (D^2 * hbar * q) / (2 * rho * V * c_s)
        """
        D = self.mat.acoustic_deformation_potential  # eV
        rho = 5.0  # g/cm^3, 密度 (应作为参数传入)
        c_s = self.mat.sound_velocity  # cm/s
        
        # 单位转换
        D_J = D * 1.602e-19  # J
        rho_SI = rho * 1000  # kg/m^3
        c_s_SI = c_s * 0.01  # m/s
        
        # 矩阵元
        M_sq = (D_J**2 * H_BAR * q * 1e10) / (2 * rho_SI * c_s_SI)  # J^2
        M_sq /= (1.602e-19)**2  # 转换为 eV^2
        
        return np.sqrt(M_sq)
    
    def self_energy(self, temperature: float = 0.0) -> Tuple[float, float]:
        """
        计算激子自能
        
        包括:
        - 实部: 能级移动
        - 虚部: 寿命展宽
        
        Returns:
        --------
        Tuple[float, float] : (Re[Σ], Im[Σ]) in eV
        """
        # 简化模型
        # Re[Σ] ~ -g^2 * N_LO (温度依赖)
        # Im[Σ] ~ g^2 * (1 + N_LO) (衰减率)
        
        hbar_omega = self.mat.optical_phonon_energy
        
        # Bose-Einstein分布
        if temperature > 0:
            n_LO = 1.0 / (np.exp(hbar_omega / (KB * temperature)) - 1.0)
        else:
            n_LO = 0.0
        
        # Frohlich耦合
        g = self.frohlich_coupling()
        
        # 自能 (简化)
        Re_Sigma = -g**2 * n_LO
        Im_Sigma = -g**2 * (1.0 + n_LO) * 0.1  # 衰减
        
        return Re_Sigma, Im_Sigma
    
    def huang_rhys_factor(self) -> float:
        """
        Huang-Rhys因子
        
        S = g^2 / (hbar*omega_LO)^2
        
        表征电声耦合强度:
        - S << 1: 弱耦合，零声子线主导
        - S ~ 1: 中等耦合
        - S >> 1: 强耦合，多声子边带
        """
        g = self.frohlich_coupling()
        hbar_omega = self.mat.optical_phonon_energy
        
        S = (g / hbar_omega)**2
        return S
    
    def phonon_sideband(self, energy_range: Tuple[float, float] = (-0.5, 0.5),
                       n_points: int = 500,
                       temperature: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算声子边带光谱
        
        使用Pekarian线型 (泊松分布)
        
        I(E) = sum_n (S^n / n!) * exp(-S) * L(E - E_0 - n*hbar*omega)
        """
        S = self.huang_rhys_factor()
        hbar_omega = self.mat.optical_phonon_energy
        E_0 = 0.0  # 零声子线位置
        
        energies = np.linspace(energy_range[0], energy_range[1], n_points)
        intensity = np.zeros_like(energies)
        
        # 声子边带贡献 (n = 0, 1, 2, ...)
        for n in range(10):  # 截断
            weight = (S**n / np.math.factorial(n)) * np.exp(-S)
            peak_position = E_0 - n * hbar_omega  # Stokes边带 (发射)
            
            # 洛伦兹线型
            gamma = 0.01  # 展宽
            lineshape = (gamma / np.pi) / ((energies - peak_position)**2 + gamma**2)
            
            intensity += weight * lineshape
        
        return energies, intensity
    
    def temperature_dependent_gap_shift(self, temperatures: np.ndarray) -> np.ndarray:
        """
        温度依赖的带隙移动 (Varshni公式)
        
        E_g(T) = E_g(0) - alpha * T^2 / (T + beta)
        """
        # 典型参数
        alpha = 5e-4  # eV/K
        beta = 200.0  # K, Debye温度相关
        
        shift = -alpha * temperatures**2 / (temperatures + beta)
        return shift
    
    def linewidth_temperature_dependence(self, temperatures: np.ndarray) -> np.ndarray:
        """
        温度依赖的线宽
        
        Gamma(T) = Gamma_0 + Gamma_1 / (exp(hbar*omega_LO / kT) - 1)
        """
        Gamma_0 = 0.001  # eV, 零温展宽
        Gamma_1 = 0.05   # eV, 耦合强度
        
        hbar_omega = self.mat.optical_phonon_energy
        
        n_phonon = 1.0 / (np.exp(hbar_omega / (KB * temperatures)) - 1.0)
        Gamma = Gamma_0 + Gamma_1 * n_phonon
        
        return Gamma


# =============================================================================
# 激子扩散和动力学
# =============================================================================

class ExcitonDynamics:
    """
    激子扩散和动力学
    """
    
    def __init__(self, material: MaterialParameters, exciton: ExcitonState):
        self.mat = material
        self.exciton = exciton
    
    def diffusion_coefficient(self, temperature: float = 300.0) -> float:
        """
        激子扩散系数
        
        D = kB * T * tau / M
        
        其中 tau 是动量弛豫时间, M 是激子质量
        """
        M = self.mat.average_mass  # 激子质量 (电子质量)
        
        # 动量弛豫时间 (从激子-声子散射估算)
        tau = self.momentum_relaxation_time(temperature)
        
        # 扩散系数 (cm^2/s)
        D = (KB * temperature * tau * 1e-12) / (M * 9.109e-31)
        D *= 1e4  # m^2/s to cm^2/s
        
        return D
    
    def momentum_relaxation_time(self, temperature: float) -> float:
        """动量弛豫时间 (ps)"""
        # 简化: tau ~ 1/T
        tau_0 = 1.0  # ps at 300K
        return tau_0 * (300.0 / temperature)
    
    def diffusion_length(self, temperature: float = 300.0,
                        lifetime: Optional[float] = None) -> float:
        """
        激子扩散长度
        
        L_D = sqrt(D * tau)
        """
        D = self.diffusion_coefficient(temperature)  # cm^2/s
        tau = (lifetime or self.exciton.lifetime) * 1e-12  # ps to s
        
        L_D = np.sqrt(D * tau)  # cm
        L_D *= 1e7  # to nm
        
        return L_D
    
    def radiative_lifetime(self) -> float:
        """
        辐射寿命
        
        tau_rad = (3 * epsilon * m0 * c^3) / (4 * alpha * omega^3 * |mu|^2)
        """
        # 简化估算
        omega = self.exciton.transition_energy / H_BAR  # s^-1
        
        # 跃迁偶极矩 (约化)
        a_b = self.exciton.bohr_radius * 1e-10  # m
        mu = 1.602e-19 * a_b  # C·m
        
        epsilon = self.mat.epsilon_static
        c = 3e8  # m/s
        alpha = 1/137  # 精细结构常数
        m0 = 9.109e-31  # kg
        
        tau_rad = (3 * epsilon * m0 * c**3) / (4 * alpha * omega**3 * mu**2)
        tau_rad *= 1e12  # s to ps
        
        return tau_rad
    
    def nonradiative_lifetime(self, defect_density: float = 1e15) -> float:
        """
        非辐射寿命
        
        tau_NR ~ 1 / (sigma * v * N_defect)
        
        Parameters:
        -----------
        defect_density : float
            缺陷密度 (cm^-3)
        """
        # 捕获截面 (cm^2)
        sigma = 1e-14
        
        # 热速度 (cm/s)
        kT = KB * 300  # eV
        m = self.mat.average_mass * 9.109e-31  # kg
        v = np.sqrt(3 * kT * 1.602e-19 / m) * 100  # cm/s
        
        tau_nr = 1.0 / (sigma * v * defect_density)
        tau_nr *= 1e12  # s to ps
        
        return tau_nr


# =============================================================================
# 可视化工具
# =============================================================================

class ExcitonVisualizer:
    """激子性质可视化"""
    
    def __init__(self, material: MaterialParameters):
        self.mat = material
        self.calculator = ExcitonBindingEnergy(material)
    
    def plot_rydberg_series(self, save_path: Optional[str] = None):
        """绘制Rydberg激子系列"""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")
        
        states = self.calculator.rydberg_series(n_max=5)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 能级图
        n_values = [s.n for s in states if s.l == 0]  # s态
        energies = [s.transition_energy for s in states if s.l == 0]
        
        for n, E in zip(n_values, energies):
            ax1.hlines(E, 0, 1, colors='blue', linewidth=2)
            ax1.text(1.05, E, f'n={n}', va='center')
        
        ax1.set_xlim(-0.5, 2)
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title(f'{self.mat.name} Exciton Rydberg Series')
        ax1.set_xticks([])
        
        # 结合能vs主量子数
        binding_energies = [s.binding_energy for s in states if s.l == 0]
        ax2.plot(n_values, binding_energies, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Principal quantum number n')
        ax2.set_ylabel('Binding energy (eV)')
        ax2.set_title('Binding Energy vs n')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_temperature_dependence(self, temperatures: np.ndarray,
                                    save_path: Optional[str] = None):
        """绘制温度依赖性质"""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")
        
        exciton = self.calculator.rydberg_series(n_max=1)[0]
        eph = ExcitonPhononInteraction(self.mat, exciton)
        
        # 计算温度依赖
        gap_shift = eph.temperature_dependent_gap_shift(temperatures)
        linewidth = eph.linewidth_temperature_dependence(temperatures)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 带隙移动
        ax1.plot(temperatures, gap_shift, 'b-', linewidth=2)
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Gap shift (eV)')
        ax1.set_title(f'{self.mat.name} Temperature-dependent Gap')
        ax1.grid(True, alpha=0.3)
        
        # 线宽
        ax2.plot(temperatures, linewidth * 1000, 'r-', linewidth=2)  # meV
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Linewidth (meV)')
        ax2.set_title('Homogeneous Linewidth vs T')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# 示例和测试
# =============================================================================

def example_exciton_properties():
    """激子性质计算示例"""
    print("=" * 60)
    print("Example: Exciton Properties Calculation")
    print("=" * 60)
    
    # 获取材料
    for mat_name in ["GaAs", "MoS2", "WSe2"]:
        print(f"\n--- {mat_name} ---")
        mat = get_material(mat_name)
        
        calc = ExcitonBindingEnergy(mat)
        
        # 有效Rydberg
        eff_ryd = calc.effective_rydberg()
        print(f"Effective Rydberg: {eff_ryd:.3f} meV")
        
        # 激子系列
        states = calc.rydberg_series(n_max=3)
        print("Exciton states:")
        for s in states[:3]:
            print(f"  n={s.n}, l={s.l}: E_b = {s.binding_energy*1000:.1f} meV, "
                  f"a_b = {s.bohr_radius:.1f} Å")
        
        # Keldysh参数 (2D材料)
        if mat.dimensionality == 2:
            r_star = calc.keldysh_parameter()
            print(f"Keldysh parameter r*: {r_star:.2f}")


def example_phonon_coupling():
    """激子-声子耦合示例"""
    print("\n" + "=" * 60)
    print("Example: Exciton-Phonon Coupling")
    print("=" * 60)
    
    mat = get_material("MoS2")
    calc = ExcitonBindingEnergy(mat)
    exciton = calc.rydberg_series(n_max=1)[0]
    
    eph = ExcitonPhononInteraction(mat, exciton)
    
    # Huang-Rhys因子
    S = eph.huang_rhys_factor()
    print(f"\nHuang-Rhys factor S: {S:.3f}")
    
    # 温度依赖
    temperatures = np.linspace(0, 500, 100)
    gap_shift = eph.temperature_dependent_gap_shift(temperatures)
    print(f"\nGap shift at 300K: {gap_shift[-1]*1000:.2f} meV")


def example_exciton_dynamics():
    """激子动力学示例"""
    print("\n" + "=" * 60)
    print("Example: Exciton Dynamics")
    print("=" * 60)
    
    mat = get_material("GaAs")
    calc = ExcitonBindingEnergy(mat)
    exciton = calc.rydberg_series(n_max=1)[0]
    
    dyn = ExcitonDynamics(mat, exciton)
    
    # 辐射寿命
    tau_rad = dyn.radiative_lifetime()
    print(f"\nRadiative lifetime: {tau_rad:.1f} ps")
    
    # 扩散系数
    D = dyn.diffusion_coefficient(temperature=300)
    print(f"Diffusion coefficient at 300K: {D:.2f} cm²/s")
    
    # 扩散长度
    L_D = dyn.diffusion_length(temperature=300, lifetime=100)
    print(f"Diffusion length: {L_D:.0f} nm")


if __name__ == "__main__":
    example_exciton_properties()
    example_phonon_coupling()
    example_exciton_dynamics()
