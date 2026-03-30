#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Optical Properties Module
==================================

高级光学性质计算模块，包括:
- 二次谐波产生 (SHG)
- 圆二色性 (CD)
- 高次谐波产生 (HHG)
- 非线性光学响应

适用于:
- 非中心对称材料
- 手性材料
- 强场物理
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy import integrate, fft
import warnings

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 物理常数
# =============================================================================

HBAR_EV = 6.582119569e-16  # eV·s
ALPHA_FINE = 1/137.035999084  # 精细结构常数


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class SHGParameters:
    """SHG计算参数"""
    
    # 频率范围
    omega_min: float = 0.0  # eV
    omega_max: float = 10.0  # eV
    n_omega: int = 500
    
    # 展宽
    broadening: float = 0.1  # eV (洛伦兹展宽)
    
    # 能带
    valence_bands: List[int] = None
    conduction_bands: List[int] = None
    
    # 响应张量
    tensor_components: List[Tuple[int, int, int]] = None
    
    def __post_init__(self):
        if self.valence_bands is None:
            self.valence_bands = [1, 2, 3, 4]
        if self.conduction_bands is None:
            self.conduction_bands = [5, 6, 7, 8]
        if self.tensor_components is None:
            # 常见非零分量
            self.tensor_components = [(0,0,0), (1,1,1), (2,2,2),  # xxx, yyy, zzz
                                      (0,1,2), (1,2,0), (2,0,1)]  # xyz等


@dataclass
class CDParameters:
    """圆二色性计算参数"""
    
    # 频率范围
    omega_min: float = 0.5  # eV
    omega_max: float = 10.0  # eV
    n_omega: int = 500
    
    # 展宽
    broadening: float = 0.05  # eV
    
    # 偏振方向
    light_direction: np.ndarray = None  # 光传播方向
    
    def __post_init__(self):
        if self.light_direction is None:
            self.light_direction = np.array([0, 0, 1])  # z方向
        self.light_direction = self.light_direction / np.linalg.norm(self.light_direction)


@dataclass
class HHGParameters:
    """HHG计算参数"""
    
    # 激光参数
    wavelength: float = 800.0  # nm, 基频波长
    intensity: float = 1e14  # W/cm^2, 激光强度
    pulse_duration: float = 30.0  # fs, 脉冲长度
    n_cycles: int = 6  # 光学周期数
    
    # 计算参数
    max_harmonic: int = 20  # 最大谐波阶数
    time_step: float = 0.1  # fs, 时间步长
    
    # 理论模型
    model: str = "tdse"  # tdse (含时薛定谔方程), sfa (强场近似), tddft
    
    @property
    def photon_energy(self) -> float:
        """光子能量 (eV)"""
        return 1239.8 / self.wavelength
    
    @property
    def optical_period(self) -> float:
        """光学周期 (fs)"""
        return self.wavelength / 299.792  # fs


# =============================================================================
# 二次谐波产生 (SHG)
# =============================================================================

class SHGCalculator:
    """
    二次谐波产生 (Second Harmonic Generation) 计算器
    
    SHG是非线性光学二阶过程，要求材料不具有反演对称性。
    响应由二阶非线性极化率 χ^(2) 描述。
    
    χ^(2)_{ijk}(-2ω; ω, ω) = (e^3/ℏ^2) * sum_{v,c,c'} 
        [ r^i_{vc'} * r^j_{c'c} * r^k_{cv} / ((ω - ω_{c'v} + iη)(2ω - ω_{cv} + iη)) ]
    
    其中:
    - r: 位置矩阵元
    - v: 价带, c, c': 导带
    - ω_{cv}: 带间跃迁能量
    - η: 展宽参数
    """
    
    def __init__(self, energies: np.ndarray, 
                 dipole_matrix: np.ndarray,
                 occupations: np.ndarray):
        """
        初始化SHG计算器
        
        Parameters:
        -----------
        energies : np.ndarray
            能带能量 (nk, nbands) in eV
        dipole_matrix : np.ndarray
            偶极矩阵元 (nk, nbands, nbands, 3)
        occupations : np.ndarray
            能带占据数 (nk, nbands)
        """
        self.energies = energies
        self.dipole = dipole_matrix
        self.occ = occupations
        
        self.nk = energies.shape[0]
        self.nbands = energies.shape[1]
    
    def calculate_chi2(self, params: SHGParameters,
                       crystal_class: str = " zincblende") -> Dict[str, np.ndarray]:
        """
        计算二阶非线性极化率 χ^(2)
        
        Parameters:
        -----------
        params : SHGParameters
            计算参数
        crystal_class : str
            晶体类别，决定张量形式
        
        Returns:
        --------
        Dict[str, np.ndarray] : 各独立张量分量的光谱
        """
        omegas = np.linspace(params.omega_min, params.omega_max, params.n_omega)
        eta = params.broadening
        
        chi2 = {}
        
        # 根据晶体类别确定独立分量
        independent_components = self._get_independent_components(crystal_class)
        
        for comp in independent_components:
            i, j, k = comp
            chi2_tensor = np.zeros(len(omegas), dtype=complex)
            
            for io, omega in enumerate(omegas):
                chi2_val = 0.0 + 0.0j
                
                # 遍历k点
                for ik in range(self.nk):
                    # 遍历价带v和导带c, c'
                    for iv in params.valence_bands:
                        for ic in params.conduction_bands:
                            for icp in params.conduction_bands:
                                if ic == icp:
                                    continue
                                
                                E_v = self.energies[ik, iv]
                                E_c = self.energies[ik, ic]
                                E_cp = self.energies[ik, icp]
                                
                                # 跃迁能量
                                omega_cv = E_c - E_v
                                omega_cpv = E_cp - E_v
                                
                                # 选择规则 (占据数检查)
                                if self.occ[ik, iv] < 0.5 or self.occ[ik, ic] > 0.5:
                                    continue
                                
                                # 矩阵元
                                r_vc = self.dipole[ik, iv, ic, :]
                                r_cp_c = self.dipole[ik, icp, ic, :]
                                r_cv = self.dipole[ik, ic, iv, :]
                                
                                # 两项贡献 (ω, ω) 和 (ω, ω) 交换
                                term1 = (r_vc[i] * r_cp_c[j] * r_cv[k]) / \
                                        ((omega - omega_cpv + 1j*eta) * 
                                         (2*omega - omega_cv + 1j*eta))
                                
                                chi2_val += term1
                
                chi2_tensor[io] = chi2_val
            
            chi2[f"{i}{j}{k}"] = chi2_tensor
        
        return chi2
    
    def _get_independent_components(self, crystal_class: str) -> List[Tuple[int, int, int]]:
        """获取晶体类别对应的独立分量"""
        components_map = {
            "zincblende": [(0,1,2), (1,2,0), (2,0,1),  # xyz, yzx, zxy
                          (0,2,1), (1,0,2), (2,1,0)],  # xzy, yxz, zyx
            "wurtzite": [(2,2,2), (2,0,0), (2,1,1)],  # zzz, zxx, zyy
            "perovskite": [(0,0,0), (1,1,1), (2,2,2),
                          (0,1,2), (1,2,0), (2,0,1)],
        }
        return components_map.get(crystal_class, [(0,1,2)])
    
    def shg_susceptibility(self, chi2: Dict[str, np.ndarray],
                          omega: np.ndarray) -> np.ndarray:
        """
        计算SHG极化率 |χ^(2)|^2
        
        Returns:
        --------
        np.ndarray : SHG信号强度 vs 频率
        """
        intensity = np.zeros_like(omega)
        
        for comp, chi2_val in chi2.items():
            intensity += np.abs(chi2_val)**2
        
        return intensity
    
    def plot_shg_spectrum(self, chi2: Dict[str, np.ndarray],
                         omega: np.ndarray,
                         save_path: Optional[str] = None):
        """绘制SHG光谱"""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 张量分量
        for comp, chi2_val in chi2.items():
            ax1.plot(omega, np.abs(chi2_val), label=f'χ^(2)_{comp}')
        
        ax1.set_xlabel('Photon energy (eV)')
        ax1.set_ylabel('|χ^(2)| (arb. units)')
        ax1.set_title('SHG Susceptibility Tensor Components')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 总SHG信号
        intensity = self.shg_susceptibility(chi2, omega)
        ax2.plot(omega, intensity, 'r-', linewidth=2)
        ax2.set_xlabel('Photon energy (eV)')
        ax2.set_ylabel('SHG Intensity (arb. units)')
        ax2.set_title('Total SHG Signal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# 圆二色性 (Circular Dichroism)
# =============================================================================

class CDCalculator:
    """
    圆二色性计算器
    
    CD测量左圆偏振光和右圆偏振光吸收的差异。
    对于手性材料，CD光谱呈现特征性的峰。
    
    CD信号正比于旋光张量的虚部:
    θ(ω) ∝ Im[α_L(ω) - α_R(ω)]
    
    其中α是极化率，下标L/R表示左/右圆偏振。
    """
    
    def __init__(self, energies: np.ndarray,
                 dipole_matrix: np.ndarray,
                 orbital_angular: Optional[np.ndarray] = None):
        """
        初始化CD计算器
        
        Parameters:
        -----------
        energies : np.ndarray
            能带能量 (nk, nbands)
        dipole_matrix : np.ndarray
            速度规范动量矩阵元 (nk, nbands, nbands, 3)
        orbital_angular : np.ndarray, optional
            轨道角动量矩阵元
        """
        self.energies = energies
        self.dipole = dipole_matrix
        self.L = orbital_angular
        
        self.nk = energies.shape[0]
        self.nbands = energies.shape[1]
    
    def calculate_cd_spectrum(self, params: CDParameters) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算CD光谱
        
        Parameters:
        -----------
        params : CDParameters
            CD计算参数
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (能量, CD信号)
        """
        omegas = np.linspace(params.omega_min, params.omega_max, params.n_omega)
        eta = params.broadening
        
        cd_signal = np.zeros(len(omegas))
        
        # 光传播方向的单位向量
        q_hat = params.light_direction
        
        for io, omega in enumerate(omegas):
            cd_val = 0.0
            
            for ik in range(self.nk):
                for v in range(self.nbands // 2):  # 价带
                    for c in range(self.nbands // 2, self.nbands):  # 导带
                        
                        # 跃迁能量
                        E_cv = self.energies[ik, c] - self.energies[ik, v]
                        
                        # 洛伦兹线型
                        lorentz = (eta / np.pi) / ((omega - E_cv)**2 + eta**2)
                        
                        # 旋转强度 (Rosenfeld理论)
                        # R ∝ Im[μ · m]
                        # 其中μ是电偶极矩，m是磁偶极矩
                        
                        # 电偶极矩矩阵元
                        mu_vc = self.dipole[ik, v, c, :]
                        
                        # 磁偶极矩 (正比于轨道角动量)
                        if self.L is not None:
                            m_vc = self.L[ik, v, c, :]
                        else:
                            # 使用k·p近似
                            m_vc = np.cross(q_hat, mu_vc)
                        
                        # CD信号 ∝ (mu · m) · q_hat
                        rotatory_strength = np.dot(np.cross(mu_vc, m_vc), q_hat)
                        
                        cd_val += rotatory_strength * lorentz
            
            cd_signal[io] = cd_val
        
        return omegas, cd_signal
    
    def calculate_g_factor(self, cd_signal: np.ndarray,
                          absorption: np.ndarray,
                          energy_range: Optional[Tuple[float, float]] = None) -> float:
        """
        计算不对称因子 g = Δε / ε
        
        Parameters:
        -----------
        cd_signal : np.ndarray
            CD信号
        absorption : np.ndarray
            吸收光谱
        energy_range : tuple, optional
            计算g因子的能量范围
        
        Returns:
        --------
        float : g因子 (无量纲)
        """
        if energy_range:
            mask = (energy_range[0] <= self.energies) & (self.energies <= energy_range[1])
            cd_max = np.max(np.abs(cd_signal[mask]))
            abs_max = np.max(absorption[mask])
        else:
            cd_max = np.max(np.abs(cd_signal))
            abs_max = np.max(absorption)
        
        g_factor = cd_max / abs_max if abs_max > 0 else 0
        return g_factor
    
    def plot_cd_spectrum(self, omegas: np.ndarray,
                        cd_signal: np.ndarray,
                        absorption: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None):
        """绘制CD光谱"""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # CD信号
        ax1.fill_between(omegas, cd_signal, alpha=0.3, color='blue')
        ax1.plot(omegas, cd_signal, 'b-', linewidth=1.5)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('CD Signal (arb. units)')
        ax1.set_title('Circular Dichroism Spectrum')
        ax1.grid(True, alpha=0.3)
        
        # 吸收光谱 (可选)
        if absorption is not None:
            ax2.plot(omegas, absorption, 'r-', linewidth=1.5, label='Absorption')
            ax2.set_ylabel('Absorption (arb. units)')
            ax2.set_xlabel('Photon energy (eV)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.plot(omegas, np.abs(cd_signal), 'r-', linewidth=1.5, label='|CD|')
            ax2.set_ylabel('|CD| (arb. units)')
            ax2.set_xlabel('Photon energy (eV)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# 高次谐波产生 (HHG)
# =============================================================================

class HHGCalculator:
    """
    高次谐波产生计算器
    
    HHG是强激光场与物质相互作用产生的非微扰非线性光学过程。
    产生奇次谐波: ω, 3ω, 5ω, ..., (2n+1)ω
    
    三种理论模型:
    1. TDSE: 含时薛定谔方程 (最精确)
    2. SFA: 强场近似 (半经典)
    3. TDDFT: 含时密度泛函理论
    """
    
    def __init__(self, params: HHGParameters):
        self.params = params
        
        # 激光电场
        self.E0 = self._intensity_to_field(params.intensity)  # V/m
        self.omega0 = params.photon_energy / HBAR_EV  # s^-1
        
        # 时间网格
        self.T = params.optical_period  # fs
        self.dt = params.time_step  # fs
        self.t = np.arange(0, params.n_cycles * self.T, self.dt)
    
    def _intensity_to_field(self, I: float) -> float:
        """将强度 (W/cm^2) 转换为电场振幅 (V/m)"""
        c = 3e8  # m/s
        epsilon0 = 8.854e-12  # F/m
        
        I_SI = I * 1e4  # W/m^2
        E0 = np.sqrt(2 * I_SI / (c * epsilon0))
        return E0
    
    def laser_field(self, t: np.ndarray,
                   envelope: str = "gaussian") -> np.ndarray:
        """
        生成激光电场
        
        E(t) = E0 * f(t) * cos(ω0*t)
        
        Parameters:
        -----------
        t : np.ndarray
            时间数组 (fs)
        envelope : str
            脉冲包络形状: "gaussian", "sin2", "trapezoidal"
        
        Returns:
        --------
        np.ndarray : 电场 (V/m)
        """
        t_fs = t  # fs
        t_s = t_fs * 1e-15  # s
        
        # 载波
        carrier = np.cos(self.omega0 * t_s)
        
        # 包络
        if envelope == "gaussian":
            sigma = self.params.pulse_duration / (2 * np.sqrt(2 * np.log(2)))
            env = np.exp(-0.5 * ((t_fs - self.params.n_cycles * self.T / 2) / sigma)**2)
        elif envelope == "sin2":
            env = np.sin(np.pi * t_fs / (self.params.n_cycles * self.T))**2
            env = np.where((t_fs > 0) & (t_fs < self.params.n_cycles * self.T), env, 0)
        else:
            env = np.ones_like(t)
        
        E_t = self.E0 * env * carrier
        return E_t
    
    def solve_tdse_1d(self, potential: Callable,
                     x: np.ndarray,
                     psi0: np.ndarray) -> np.ndarray:
        """
        求解一维含时薛定谔方程
        
        iℏ ∂ψ/∂t = [-ℏ²/2m ∂²/∂x² + V(x) + x*E(t)] ψ
        
        Parameters:
        -----------
        potential : callable
            势能函数 V(x)
        x : np.ndarray
            空间网格 (m)
        psi0 : np.ndarray
            初始波函数
        
        Returns:
        --------
        np.ndarray : 时间演化的偶极矩
        """
        dx = x[1] - x[0]
        nx = len(x)
        nt = len(self.t)
        
        # 哈密顿量 (Crank-Nicolson)
        hbar = 1.054e-34  # J·s
        m = 9.109e-31  # kg
        
        # 动能矩阵
        T_diag = np.ones(nx) * hbar**2 / (m * dx**2)
        T_off = np.ones(nx-1) * (-0.5) * hbar**2 / (m * dx**2)
        
        # 偶极矩时间序列
        dipole = np.zeros(nt)
        
        psi = psi0.copy()
        E_t = self.laser_field(self.t)
        
        for it in range(nt):
            # 势能 + 激光场
            V = potential(x) + x * E_t[it]
            
            # 计算偶极矩
            dipole[it] = np.real(np.trapz(np.conj(psi) * x * psi, x))
            
            # 时间演化 (简化，使用小时间步长)
            H_psi = np.zeros_like(psi, dtype=complex)
            H_psi[1:-1] = T_off[:-1] * psi[:-2] + T_diag[1:-1] * psi[1:-1] + T_off[:-1] * psi[2:]
            H_psi += V * psi
            
            # 显式欧拉 (需要非常小的时间步长)
            dt_s = self.dt * 1e-15
            psi = psi - 1j * dt_s / hbar * H_psi
            
            # 归一化
            norm = np.sqrt(np.trapz(np.abs(psi)**2, x))
            psi = psi / norm
        
        return dipole
    
    def strong_field_approximation(self, Ip: float) -> np.ndarray:
        """
        强场近似 (Lewenstein模型)
        
        用于计算原子/分子的HHG。
        
        Parameters:
        -----------
        Ip : float
            电离势 (eV)
        
        Returns:
        --------
        np.ndarray : 谐波谱
        """
        # 简化的SFA实现
        E_t = self.laser_field(self.t)
        
        # 经典轨迹计算
        harmonics = []
        
        for n in range(1, self.params.max_harmonic + 1, 2):  # 奇次谐波
            omega_n = n * self.params.photon_energy
            
            # 振幅 (简化模型)
            cutoff_energy = Ip + 3.17 * (self.E0**2 / (4 * self.omega0**2))
            
            if omega_n < cutoff_energy:
                # 截止定律以下
                amp = np.exp(-omega_n / (2 * cutoff_energy))
            else:
                # 截止以上快速衰减
                amp = np.exp(-(omega_n - cutoff_energy) / 0.5)
            
            harmonics.append(amp)
        
        return np.array(harmonics)
    
    def calculate_hhg_spectrum(self, dipole_moment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算HHG光谱
        
        通过对偶极加速度做FFT得到谐波谱。
        
        Parameters:
        -----------
        dipole_moment : np.ndarray
            时间依赖的偶极矩
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (谐波阶数, 强度)
        """
        # 计算加速度 (偶极矩的二阶导数)
        dt_s = self.dt * 1e-15
        acceleration = np.gradient(np.gradient(dipole_moment, dt_s), dt_s)
        
        # FFT
        spectrum = np.abs(fft.fft(acceleration))**2
        freqs = fft.fftfreq(len(acceleration), dt_s)
        
        # 只保留正频率和奇次谐波
        mask = (freqs > 0)
        freqs = freqs[mask]
        spectrum = spectrum[mask]
        
        # 转换为谐波阶数
        harmonics = freqs / (self.omega0 / (2 * np.pi))
        
        # 选择整数谐波
        harmonic_orders = np.arange(1, self.params.max_harmonic + 1, 2)
        harmonic_intensity = []
        
        for n in harmonic_orders:
            idx = np.argmin(np.abs(harmonics - n))
            harmonic_intensity.append(spectrum[idx])
        
        return harmonic_orders, np.array(harmonic_intensity)
    
    def plot_hhg_spectrum(self, harmonics: np.ndarray,
                         intensity: np.ndarray,
                         save_path: Optional[str] = None):
        """绘制HHG光谱"""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 谐波谱 (线状)
        for n, I in zip(harmonics, intensity):
            ax1.vlines(n, 0, I, colors='blue', linewidth=2)
        
        ax1.set_xlabel('Harmonic order')
        ax1.set_ylabel('Intensity (arb. units)')
        ax1.set_title(f'HHG Spectrum (I = {self.params.intensity:.0e} W/cm²)')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 激光电场
        E_t = self.laser_field(self.t)
        ax2.plot(self.t, E_t / 1e9)  # GV/m
        ax2.set_xlabel('Time (fs)')
        ax2.set_ylabel('Electric field (GV/m)')
        ax2.set_title('Laser Pulse')
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

def example_shg_zincblende():
    """闪锌矿结构SHG示例"""
    print("=" * 60)
    print("Example: SHG in Zinc-Blende Structure")
    print("=" * 60)
    
    # 创建模拟数据
    nk, nbands = 10, 10
    energies = np.random.rand(nk, nbands) * 10  # 0-10 eV
    
    # 排序能带
    for ik in range(nk):
        energies[ik].sort()
    
    # 偶极矩阵元
    dipole = np.random.rand(nk, nbands, nbands, 3) + 0.0j
    
    # 占据数
    occupations = np.zeros((nk, nbands))
    occupations[:, :4] = 1.0  # 价带填满
    
    # 计算SHG
    shg = SHGCalculator(energies, dipole, occupations)
    params = SHGParameters(
        omega_min=0.5, omega_max=5.0, n_omega=100,
        valence_bands=[0, 1, 2, 3],
        conduction_bands=[4, 5, 6, 7]
    )
    
    chi2 = shg.calculate_chi2(params, crystal_class="zincblende")
    
    print("\nSHG independent components:")
    for comp in chi2:
        print(f"  χ^(2)_{comp}: max = {np.max(np.abs(chi2[comp])):.4f}")
    
    return shg, chi2


def example_cd_chiral():
    """手性材料CD示例"""
    print("\n" + "=" * 60)
    print("Example: CD in Chiral Material")
    print("=" * 60)
    
    nk, nbands = 10, 8
    energies = np.zeros((nk, nbands))
    
    # 创建简单的能带结构
    for ik in range(nk):
        for ib in range(nbands):
            energies[ik, ib] = -5 + ib * 2 + 0.1 * ik
    
    dipole = np.random.rand(nk, nbands, nbands, 3) + 0.0j
    
    cd = CDCalculator(energies, dipole)
    params = CDParameters(omega_min=1.0, omega_max=8.0, n_omega=200)
    
    omegas, cd_signal = cd.calculate_cd_spectrum(params)
    
    print(f"\nCD signal range: [{np.min(cd_signal):.4f}, {np.max(cd_signal):.4f}]")
    print(f"CD anisotropy factor g ≈ {np.max(np.abs(cd_signal)):.4f}")
    
    return cd, omegas, cd_signal


def example_hhg_atom():
    """原子HHG示例"""
    print("\n" + "=" * 60)
    print("Example: HHG in Strong Laser Field")
    print("=" * 60)
    
    params = HHGParameters(
        wavelength=800.0,  # nm
        intensity=1e14,    # W/cm^2
        pulse_duration=30.0,  # fs
        n_cycles=6,
        max_harmonic=15,
    )
    
    hhg = HHGCalculator(params)
    
    print(f"\nLaser parameters:")
    print(f"  Photon energy: {params.photon_energy:.2f} eV")
    print(f"  Optical period: {params.optical_period:.2f} fs")
    print(f"  Electric field: {hhg.E0/1e9:.2f} GV/m")
    
    # 强场近似计算
    Ip = 13.6  # 氢原子电离势 (eV)
    harmonics = hhg.strong_field_approximation(Ip)
    
    harmonic_orders = np.arange(1, params.max_harmonic + 1, 2)
    print(f"\nHarmonic intensities:")
    for n, I in zip(harmonic_orders, harmonics):
        print(f"  H{n}: {I:.4f}")
    
    return hhg, harmonic_orders, harmonics


if __name__ == "__main__":
    example_shg_zincblende()
    example_cd_chiral()
    example_hhg_atom()
