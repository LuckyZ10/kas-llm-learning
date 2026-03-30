#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defect Excited States Module
============================

缺陷激发态计算模块，包括:
- 色心缺陷 (NV中心, SiV, GeV等)
- 点缺陷发光谱
- 自旋-光子耦合
- 量子比特性质

适用于:
- 金刚石NV中心
- SiC色心
- 二维材料缺陷
"""

import os
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from scipy import linalg, integrate
from scipy.linalg import expm
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 物理常数
# =============================================================================

HBAR = 6.582119569e-16  # eV·s
MU_B = 5.7883818012e-5  # eV/T, 玻尔磁子
G_ELECTRON = 2.002319  # 电子g因子
KB = 8.617333262e-5  # eV/K


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class ColorCenterParameters:
    """色心参数"""
    
    name: str
    host_material: str
    
    # 零声子线 (Zero-Phonon Line)
    zpl_energy: float = 1.0  # eV
    zpl_wavelength: Optional[float] = None  # nm
    
    # 德拜-沃勒因子 (零声子线强度占比)
    debye_waller_factor: float = 0.05
    
    # 自旋性质
    electron_spin: float = 1.0  # S=1 for NV, S=1/2 for other
    g_factor: float = 2.0028
    
    # 零场分裂 (Zero-Field Splitting)
    zfs_d: float = 2.87e-3  # eV, D参数
    zfs_e: float = 0.0  # eV, E参数 (应变导致)
    
    # 超精细耦合
    hyperfine_coupling: Dict[str, float] = field(default_factory=dict)
    # e.g., {"N": 2.3e-6, "C": 0.1e-6}  # eV
    
    # 荧光寿命
    radiative_lifetime: float = 12.0  # ns
    
    # 自旋极化
    optical_pumping_rate: float = 1e7  # s^-1
    spin_flip_rate: float = 1e3  # s^-1
    
    @property
    def wavelength(self) -> float:
        """发射波长 (nm)"""
        if self.zpl_wavelength:
            return self.zpl_wavelength
        return 1239.8 / self.zpl_energy
    
    @property
    def zfs_d_ghz(self) -> float:
        """零场分裂D参数 (GHz)"""
        return self.zfs_d * 1e9 / (6.626e-34 * 1e9)  # eV to GHz


@dataclass
class PhononSideband:
    """声子边带数据"""
    
    # 声子频率
    local_phonon_energy: float = 0.065  # eV, ~500 cm^-1
    
    # Huang-Rhys因子
    huang_rhys_s: float = 3.0
    
    # 声子谱密度
    phonon_spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def poisson_distribution(self, n_max: int = 10) -> np.ndarray:
        """
        计算声子边带的泊松分布
        
        P(n) = S^n * exp(-S) / n!
        """
        S = self.huang_rhys_s
        n = np.arange(n_max + 1)
        P = (S**n * np.exp(-S)) / np.array([np.math.factorial(i) for i in n])
        return P


@dataclass
class QuantumBit:
    """量子比特参数"""
    
    # 基态自旋子能级
    gs_levels: np.ndarray = field(default_factory=lambda: np.array([0, 2.87e-3, 5.74e-3]))
    
    # 激发态自旋子能级
    es_levels: np.ndarray = field(default_factory=lambda: np.array([0, 1.4e-3, 2.8e-3]))
    
    # 相干时间
    t1_relaxation: float = 6e3  # us, 自旋晶格弛豫
    t2_decoherence: float = 1.5  # ms, 退相干时间
    t2_star: float = 10.0  # us, 非均匀展宽
    
    # 拉比频率
    rabi_frequency: float = 10.0  # MHz
    
    # 读出对比度
    readout_contrast: float = 0.3  # 30%


# =============================================================================
# 预定义色心数据库
# =============================================================================

COLOR_CENTER_DATABASE = {
    "NV": ColorCenterParameters(
        name="NV",
        host_material="diamond",
        zpl_energy=1.945,
        zpl_wavelength=637.0,
        debye_waller_factor=0.04,
        electron_spin=1.0,
        g_factor=2.0028,
        zfs_d=2.87e-3,
        zfs_e=0.0,
        hyperfine_coupling={"N": 2.3e-6, "C": 0.11e-6},
        radiative_lifetime=12.0,
    ),
    
    "NV-": ColorCenterParameters(
        name="NV-",
        host_material="diamond",
        zpl_energy=1.945,
        zpl_wavelength=637.0,
        debye_waller_factor=0.04,
        electron_spin=1.0,
        g_factor=2.0028,
        zfs_d=2.87e-3,
        zfs_e=0.0,
        hyperfine_coupling={"N": 2.3e-6},
        radiative_lifetime=12.0,
    ),
    
    "SiV": ColorCenterParameters(
        name="SiV",
        host_material="diamond",
        zpl_energy=1.682,
        zpl_wavelength=736.0,
        debye_waller_factor=0.70,  # 高ZPL产额
        electron_spin=0.5,
        g_factor=2.0028,
        zfs_d=0.0,
        zfs_e=0.0,
        hyperfine_coupling={"Si": 40e-6},  # 强超精细
        radiative_lifetime=8.0,
    ),
    
    "GeV": ColorCenterParameters(
        name="GeV",
        host_material="diamond",
        zpl_energy=1.602,
        zpl_wavelength=774.0,
        debye_waller_factor=0.60,
        electron_spin=0.5,
        g_factor=2.0028,
        radiative_lifetime=6.0,
    ),
    
    "SnV": ColorCenterParameters(
        name="SnV",
        host_material="diamond",
        zpl_energy=1.480,
        zpl_wavelength=838.0,
        debye_waller_factor=0.50,
        electron_spin=0.5,
        radiative_lifetime=5.0,
    ),
    
    "PbV": ColorCenterParameters(
        name="PbV",
        host_material="diamond",
        zpl_energy=1.314,
        zpl_wavelength=944.0,
        debye_waller_factor=0.40,
        electron_spin=0.5,
        radiative_lifetime=4.0,
    ),
    
    # SiC色心
    "VV_SiC": ColorCenterParameters(
        name="VV",
        host_material="SiC",
        zpl_energy=1.096,  # PL1
        zpl_wavelength=1131.0,
        debye_waller_factor=0.10,
        electron_spin=1.0,
        zfs_d=1.3e-3,  # 较小
        radiative_lifetime=20.0,
    ),
}


def get_color_center(name: str) -> ColorCenterParameters:
    """获取色心参数"""
    if name not in COLOR_CENTER_DATABASE:
        raise ValueError(f"Unknown color center: {name}")
    return COLOR_CENTER_DATABASE[name]


# =============================================================================
# NV中心哈密顿量和动力学
# =============================================================================

class NVHamiltonian:
    """
    NV中心自旋哈密顿量
    
    H = D * S_z^2 + E * (S_x^2 - S_y^2) + g * μ_B * B · S
    
    对于基态 (S=1):
    - D = 2.87 GHz (零场分裂)
    - E ~ 0 (应变参数)
    """
    
    def __init__(self, params: ColorCenterParameters):
        self.params = params
        self.S = params.electron_spin
        
        # 构建自旋算符
        self._build_spin_operators()
    
    def _build_spin_operators(self):
        """构建自旋算符矩阵"""
        S = self.S
        dim = int(2 * S + 1)
        
        # S_z (对角)
        m_values = np.arange(S, -S - 1, -1)
        self.Sz = np.diag(m_values)
        
        # S_+ 和 S_-
        S_plus = np.zeros((dim, dim), dtype=complex)
        for i, m in enumerate(m_values[:-1]):
            S_plus[i, i+1] = np.sqrt(S*(S+1) - m*(m-1))
        
        S_minus = S_plus.T.conj()
        
        # S_x 和 S_y
        self.Sx = (S_plus + S_minus) / 2
        self.Sy = (S_plus - S_minus) / (2j)
        
        # S^2
        self.S2 = S * (S + 1) * np.eye(dim)
    
    def ground_state_hamiltonian(self, B: np.ndarray = np.zeros(3)) -> np.ndarray:
        """
        基态哈密顿量
        
        Parameters:
        -----------
        B : np.ndarray
            磁场 (Tesla), 默认为零
        
        Returns:
        --------
        np.ndarray : 哈密顿矩阵
        """
        D = self.params.zfs_d  # eV
        E = self.params.zfs_e  # eV
        g = self.params.g_factor
        mu_B = MU_B  # eV/T
        
        # 零场分裂
        H_zfs = D * (self.Sz @ self.Sz - self.S2 / 3)
        H_zfs = H_zfs.astype(complex)  # 转换为复数类型
        H_zfs += E * (self.Sx @ self.Sx - self.Sy @ self.Sy)
        
        # 塞曼项
        H_zeeman = g * mu_B * (B[0] * self.Sx + B[1] * self.Sy + B[2] * self.Sz)
        
        return H_zfs + H_zeeman
    
    def get_eigenstates(self, B: np.ndarray = np.zeros(3)) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算本征态和本征能量
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (本征能量, 本征矢)
        """
        H = self.ground_state_hamiltonian(B)
        eigenvalues, eigenvectors = linalg.eigh(H)
        return eigenvalues, eigenvectors
    
    def transition_frequencies(self, B: np.ndarray = np.zeros(3)) -> np.ndarray:
        """
        计算允许的自旋跃迁频率
        
        Returns:
        --------
        np.ndarray : 跃迁频率 (eV)
        """
        energies, _ = self.get_eigenstates(B)
        
        transitions = []
        for i in range(len(energies)):
            for j in range(i + 1, len(energies)):
                transitions.append(energies[j] - energies[i])
        
        return np.array(transitions)
    
    def odmr_spectrum(self, B: np.ndarray = np.zeros(3),
                     frequency_range: Tuple[float, float] = (2.5e-3, 3.2e-3),
                     n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算ODMR (光探测磁共振) 谱
        
        Parameters:
        -----------
        B : np.ndarray
            外加磁场
        frequency_range : tuple
            频率范围 (eV)
        n_points : int
            频率点数
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (频率, 强度)
        """
        frequencies = np.linspace(frequency_range[0], frequency_range[1], n_points)
        
        # 获取跃迁频率
        transitions = self.transition_frequencies(B)
        
        # 构建洛伦兹线型
        spectrum = np.zeros_like(frequencies)
        linewidth = 1e-6  # eV
        
        for trans in transitions:
            spectrum += (linewidth / np.pi) / ((frequencies - trans)**2 + linewidth**2)
        
        return frequencies, spectrum


# =============================================================================
# 色心发光谱
# =============================================================================

class ColorCenterSpectrum:
    """色心发光谱计算器"""
    
    def __init__(self, params: ColorCenterParameters):
        self.params = params
        self.phonon = PhononSideband(
            local_phonon_energy=0.065,  # eV
            huang_rhys_s=3.5 if params.name.startswith("NV") else 1.0
        )
    
    def zpl_spectrum(self, energies: np.ndarray,
                    broadening: float = 1e-5) -> np.ndarray:
        """
        零声子线光谱
        
        Parameters:
        -----------
        energies : np.ndarray
            能量网格 (eV)
        broadening : float
            展宽 (eV)
        
        Returns:
        --------
        np.ndarray : 光谱强度
        """
        E_zpl = self.params.zpl_energy
        
        # 洛伦兹线型
        spectrum = (broadening / np.pi) / ((energies - E_zpl)**2 + broadening**2)
        
        # 归一化到德拜-沃勒因子
        spectrum *= self.params.debye_waller_factor
        
        return spectrum
    
    def phonon_sideband(self, energies: np.ndarray,
                       broadening: float = 0.01) -> np.ndarray:
        """
        声子边带光谱
        
        使用Pekarian线型
        """
        E_zpl = self.params.zpl_energy
        hbar_omega_ph = self.phonon.local_phonon_energy
        S = self.phonon.huang_rhys_s
        
        spectrum = np.zeros_like(energies)
        
        # 多声子贡献
        n_max = 15
        for n in range(1, n_max + 1):
            # 泊松权重
            weight = (S**n / math.factorial(n)) * np.exp(-S)
            
            # 峰位
            E_peak = E_zpl - n * hbar_omega_ph
            
            # 高斯线型 (声子展宽)
            spectrum += weight * np.exp(-((energies - E_peak) / broadening)**2 / 2)
        
        # 归一化
        spectrum *= (1 - self.params.debye_waller_factor) / np.max(spectrum + 1e-10)
        
        return spectrum
    
    def full_spectrum(self, energies: np.ndarray,
                     zpl_broadening: float = 1e-5,
                     psb_broadening: float = 0.02) -> np.ndarray:
        """
        完整发光谱 (ZPL + PSB)
        
        Parameters:
        -----------
        energies : np.ndarray
            能量网格 (eV)
        zpl_broadening : float
            ZPL展宽 (eV)
        psb_broadening : float
            PSB展宽 (eV)
        
        Returns:
        --------
        np.ndarray : 总光谱
        """
        zpl = self.zpl_spectrum(energies, zpl_broadening)
        psb = self.phonon_sideband(energies, psb_broadening)
        
        return zpl + psb
    
    def plot_spectrum(self, energies: np.ndarray,
                     save_path: Optional[str] = None):
        """绘制发光谱"""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")
        
        zpl = self.zpl_spectrum(energies)
        psb = self.phonon_sideband(energies)
        total = zpl + psb
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.fill_between(energies, total, alpha=0.3, color='blue')
        ax.plot(energies, total, 'b-', linewidth=2, label='Total')
        ax.plot(energies, zpl, 'r-', linewidth=1, label='ZPL')
        ax.plot(energies, psb, 'g-', linewidth=1, label='PSB')
        
        ax.axvline(x=self.params.zpl_energy, color='k', linestyle='--', alpha=0.5,
                  label=f'ZPL = {self.params.zpl_energy:.3f} eV')
        
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Intensity (arb. units)')
        ax.set_title(f'{self.params.name} in {self.params.host_material} Luminescence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# 量子比特操作
# =============================================================================

class QuantumBitOperations:
    """
    色心量子比特操作
    
    实现量子比特的初始化、操作和读出。
    """
    
    def __init__(self, params: ColorCenterParameters):
        self.params = params
        self.hamiltonian = NVHamiltonian(params)
        
        # 密度矩阵
        self.rho = np.diag([1, 0, 0])  # 初始化为m_s=0
    
    def initialize(self, state: str = "m0"):
        """
        初始化量子比特
        
        Parameters:
        -----------
        state : str
            "m0" |0>, "m1" |-1>, "superposition" (|0> + |-1>)/sqrt(2)
        """
        dim = int(2 * self.params.electron_spin + 1)
        
        if state == "m0":
            self.rho = np.zeros((dim, dim))
            self.rho[0, 0] = 1.0
        elif state == "m1":
            self.rho = np.zeros((dim, dim))
            self.rho[2, 2] = 1.0
        elif state == "superposition":
            psi = np.array([1, 0, 1]) / np.sqrt(2)
            self.rho = np.outer(psi, psi.conj())
    
    def apply_mw_pulse(self, frequency: float,  # Hz
                      amplitude: float,  # V/m
                      duration: float,  # s
                      phase: float = 0.0):
        """
        施加微波脉冲 (Rabi振荡)
        
        Parameters:
        -----------
        frequency : float
            微波频率 (Hz)
        amplitude : float
            场强 (V/m)
        duration : float
            脉冲长度 (s)
        phase : float
            脉冲相位 (rad)
        """
        # 拉比频率
        # Ω = g * μ_B * B_⊥ / ℏ
        mu_B = 9.274e-24  # J/T
        hbar = 1.055e-34  # J·s
        g = self.params.g_factor
        
        # 假设微波沿x方向
        B_mw = amplitude * duration  # 简化
        omega_rabi = g * mu_B * B_mw / hbar
        
        # 旋转算符
        theta = omega_rabi * duration
        
        # 绕x轴旋转
        Rx = np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2), 0],
            [-1j*np.sin(theta/2), np.cos(theta/2), 0],
            [0, 0, 1]
        ])
        
        # 应用演化
        self.rho = Rx @ self.rho @ Rx.conj().T
    
    def apply_pi_pulse(self, frequency: float, amplitude: float):
        """施加π脉冲 (反转)"""
        # π = Ω * t
        mu_B = 9.274e-24
        hbar = 1.055e-34
        g = self.params.g_factor
        
        omega_rabi = g * mu_B * amplitude / hbar
        t_pi = np.pi / omega_rabi
        
        self.apply_mw_pulse(frequency, amplitude, t_pi)
    
    def apply_pi_half_pulse(self, frequency: float, amplitude: float):
        """施加π/2脉冲 (制备叠加态)"""
        mu_B = 9.274e-24
        hbar = 1.055e-34
        g = self.params.g_factor
        
        omega_rabi = g * mu_B * amplitude / hbar
        t_pi2 = np.pi / (2 * omega_rabi)
        
        self.apply_mw_pulse(frequency, amplitude, t_pi2)
    
    def cpmg_sequence(self, n_pulses: int = 1,
                     tau: float = 1e-6) -> float:
        """
        Carr-Purcell-Meiboom-Gill (CPMG) 序列
        
        用于测量T2退相干时间
        
        Parameters:
        -----------
        n_pulses : int
            π脉冲数
        tau : float
            脉冲间隔 (s)
        
        Returns:
        --------
        float : 读出信号强度
        """
        # 初始化到 |0>
        self.initialize("m0")
        
        # π/2脉冲
        self.apply_pi_half_pulse(2.87e9, 1e-3)
        
        # 自由演化 + π脉冲序列
        T2 = self.hamiltonian.params.radiative_lifetime * 1e-9  # 简化
        
        for i in range(n_pulses):
            # 退相干
            decay = np.exp(-tau / T2)
            self.rho *= decay
            
            # π脉冲
            self.apply_pi_pulse(2.87e9, 1e-3)
        
        # π/2脉冲
        self.apply_pi_half_pulse(2.87e9, 1e-3)
        
        # 读出
        return np.real(self.rho[0, 0])
    
    def spin_echo_decay(self, tau_list: np.ndarray) -> np.ndarray:
        """
        计算自旋回波衰减
        
        V(2τ) = exp(-(2τ/T2)^3)
        """
        T2 = 1.5e-3  # s, NV中心典型值
        
        signal = np.exp(-(2 * tau_list / T2)**3)
        return signal


# =============================================================================
# 示例和测试
# =============================================================================

def example_nv_hamiltonian():
    """NV中心哈密顿量示例"""
    print("=" * 60)
    print("Example: NV Center Hamiltonian")
    print("=" * 60)
    
    nv = get_color_center("NV")
    H = NVHamiltonian(nv)
    
    # 零场
    E0, psi0 = H.get_eigenstates(B=np.zeros(3))
    print("\nZero-field eigenvalues (eV):")
    for i, E in enumerate(E0):
        print(f"  |{i}>: {E*1e6:.3f} μeV = {E/HBAR*1e-9:.3f} GHz")
    
    # 有磁场 (沿z轴)
    Bz = 0.1  # Tesla = 100 Gauss
    E_B, psi_B = H.get_eigenstates(B=np.array([0, 0, Bz]))
    print(f"\nWith Bz = {Bz*1e3:.0f} mT:")
    for i, E in enumerate(E_B):
        print(f"  |{i}>: {E*1e6:.3f} μeV")
    
    # 跃迁频率
    trans = H.transition_frequencies(B=np.array([0, 0, Bz]))
    print(f"\nTransition frequencies:")
    for t in trans:
        print(f"  {t*1e6:.3f} μeV = {t/HBAR*1e-9:.3f} GHz")


def example_nv_spectrum():
    """NV发光谱示例"""
    print("\n" + "=" * 60)
    print("Example: NV Center Luminescence Spectrum")
    print("=" * 60)
    
    nv = get_color_center("NV")
    spectrum = ColorCenterSpectrum(nv)
    
    energies = np.linspace(1.7, 2.1, 1000)
    
    print(f"\nNV center properties:")
    print(f"  ZPL energy: {nv.zpl_energy:.3f} eV")
    print(f"  Wavelength: {nv.wavelength:.1f} nm")
    print(f"  Debye-Waller factor: {nv.debye_waller_factor:.2%}")
    print(f"  Huang-Rhys factor S: {spectrum.phonon.huang_rhys_s:.2f}")
    
    total = spectrum.full_spectrum(energies)
    print(f"\nSpectrum integral: {np.trapezoid(total, energies):.4f}")


def example_quantum_bit():
    """量子比特操作示例"""
    print("\n" + "=" * 60)
    print("Example: Quantum Bit Operations")
    print("=" * 60)
    
    nv = get_color_center("NV")
    qubit = QuantumBitOperations(nv)
    
    # 初始化
    qubit.initialize("m0")
    print(f"\nInitial state population m=0: {np.real(qubit.rho[0,0]):.3f}")
    
    # π脉冲 (应该翻转到m=-1)
    qubit.apply_pi_pulse(2.87e9, 1e-3)
    print(f"After π pulse: m=0 = {np.real(qubit.rho[0,0]):.3f}, m=-1 = {np.real(qubit.rho[2,2]):.3f}")
    
    # CPMG序列
    signal = qubit.cpmg_sequence(n_pulses=1, tau=1e-6)
    print(f"\nCPMG signal (1 pulse): {signal:.3f}")
    
    # T2衰减
    tau = np.linspace(0, 3e-3, 100)  # up to 3 ms
    echo = qubit.spin_echo_decay(tau)
    print(f"Spin echo at 1 ms: {echo[33]:.3f}")


def example_compare_color_centers():
    """比较不同色心"""
    print("\n" + "=" * 60)
    print("Example: Color Center Comparison")
    print("=" * 60)
    
    centers = ["NV", "SiV", "GeV", "SnV"]
    
    print("\n{:<10} {:>10} {:>12} {:>12} {:>10}".format(
        "Name", "ZPL (eV)", "λ (nm)", "DWF (%)", "Lifetime"))
    print("-" * 60)
    
    for name in centers:
        cc = get_color_center(name)
        print("{:<10} {:>10.3f} {:>12.1f} {:12.1f} {:>9.1f}ns".format(
            name, cc.zpl_energy, cc.wavelength, 
            cc.debye_waller_factor*100, cc.radiative_lifetime))


if __name__ == "__main__":
    example_nv_hamiltonian()
    example_nv_spectrum()
    example_quantum_bit()
    example_compare_color_centers()
