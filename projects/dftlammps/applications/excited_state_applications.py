#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excited State Applications Module
=================================

激发态计算应用案例，包括:
- 太阳能电池吸收优化
- 量子阱LED设计
- 色心量子比特

这些应用展示了GW+BSE方法在实际器件中的应用。
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import integrate, interpolate
import warnings

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 应用1: 太阳能电池吸收优化
# =============================================================================

@dataclass
class SolarCellParameters:
    """太阳能电池参数"""
    
    # 材料
    absorber_material: str = "Si"
    band_gap: float = 1.1  # eV
    
    # 结构
    thickness: float = 100.0  # μm
    
    # 光谱
    spectrum_source: str = "AM1.5G"
    
    # 器件参数
    temperature: float = 300.0  # K
    
    # 效率限制
    shockley_queisser_limit: float = 0.0  # 将计算


class SolarCellOptimizer:
    """
    太阳能电池吸收优化器
    
    使用GW-BSE计算的精确吸收谱来优化太阳能电池效率。
    
    关键优化策略:
    1. 带隙工程 - 找到最优带隙
    2. 激子收集 - 最大化激子解离
    3. 吸收层厚度优化
    """
    
    def __init__(self, params: SolarCellParameters):
        self.params = params
        
        # 加载AM1.5G光谱
        self.am15_energies, self.am15_flux = self._load_am15_spectrum()
    
    def _load_am15_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载AM1.5G标准太阳光谱"""
        # 简化的AM1.5G光谱 (ASTM G173)
        # 能量范围: 0.3 - 4.5 eV
        energies = np.linspace(0.3, 4.5, 1000)
        
        # 黑体近似 (简化)
        kT = 0.026  # eV at 300K
        flux = (energies**2) / (np.exp(energies / kT) - 1)
        
        # 归一化到1000 W/m^2
        flux = flux / np.trapezoid(flux, energies) * 1000  # W/m^2/eV
        
        return energies, flux
    
    def calculate_absorption(self, energies: np.ndarray,
                            band_gap: Optional[float] = None) -> np.ndarray:
        """
        计算吸收系数
        
        使用Tauc-Lorentz模型 + 激子效应
        """
        Eg = band_gap or self.params.band_gap
        
        # Tauc-Lorentz吸收
        alpha = np.zeros_like(energies)
        
        for i, E in enumerate(energies):
            if E > Eg:
                # 直接带隙吸收 (E - Eg)^0.5
                alpha[i] = 1e5 * np.sqrt(E - Eg)  # cm^-1
            else:
                # Urbach尾
                alpha[i] = 100 * np.exp((E - Eg) / 0.025)
        
        # 激子增强 (在带边)
        exciton_binding = 0.015  # eV, 典型值
        exciton_peak = Eg - exciton_binding
        
        # 添加激子峰
        gamma = 0.01  # eV
        exciton_contrib = 5e4 * (gamma / np.pi) / ((energies - exciton_peak)**2 + gamma**2)
        alpha += exciton_contrib
        
        return alpha
    
    def calculate_photocurrent(self, thickness: Optional[float] = None) -> float:
        """
        计算光生电流密度
        
        J_ph = e * ∫ A(E) * Φ(E) dE
        
        其中 A(E) 是吸收概率
        """
        thickness_cm = (thickness or self.params.thickness) * 1e-4  # μm to cm
        
        # 计算每个能量的吸收
        alpha = self.calculate_absorption(self.am15_energies)
        
        # Beer-Lambert吸收概率
        # A = 1 - exp(-α*d)
        absorption_prob = 1 - np.exp(-alpha * thickness_cm)
        
        # 光生电流 (A/m^2)
        e_charge = 1.602e-19  # C
        
        # 光子通量 Φ = flux / E
        photon_flux = self.am15_flux / (self.am15_energies * e_charge)  # photons / m^2 / s / eV
        
        # 积分
        absorbed_photons = photon_flux * absorption_prob
        J_ph = e_charge * integrate.trapezoid(absorbed_photons, self.am15_energies)
        
        return J_ph  # A/m^2
    
    def calculate_shockley_queisser(self) -> Dict[str, float]:
        """
        计算Shockley-Queisser极限
        
        单结太阳能电池的详细平衡极限效率。
        """
        Eg = self.params.band_gap
        T = self.params.temperature
        kT = 8.617e-5 * T  # eV
        
        # 归一化带隙
        xg = Eg / kT
        
        # 开路电压 (近似)
        # V_oc = (kT/e) * ln(J_sc/J_0 + 1)
        # 简化: V_oc ≈ Eg/e - kT/e * ln(...) 
        
        # 短路电流密度
        J_sc = self.calculate_photocurrent()
        
        # 饱和电流密度 (简化)
        J_0 = 1e-8  # A/m^2
        
        # 开路电压
        V_oc = (kT / 1) * np.log(J_sc / J_0 + 1)  # 简化，电压单位为eV
        
        # 填充因子 (近似)
        voc = V_oc / kT  # 归一化
        FF = (voc - np.log(voc + 0.72)) / (voc + 1)
        
        # 效率
        P_in = 1000.0  # W/m^2 (AM1.5G)
        efficiency = (J_sc * V_oc * FF) / P_in
        
        return {
            "J_sc": J_sc,  # A/m^2
            "V_oc": V_oc,  # eV (需要转换为V)
            "FF": FF,
            "efficiency": efficiency * 100,  # %
        }
    
    def optimize_band_gap(self, gap_range: Tuple[float, float] = (0.5, 2.5),
                         n_points: int = 50) -> Dict:
        """
        优化带隙以最大化效率
        
        Returns:
        --------
        Dict : 优化结果
        """
        gaps = np.linspace(gap_range[0], gap_range[1], n_points)
        efficiencies = []
        
        for gap in gaps:
            self.params.band_gap = gap
            sq = self.calculate_shockley_queisser()
            efficiencies.append(sq["efficiency"])
        
        # 找到最优带隙
        idx_max = np.argmax(efficiencies)
        optimal_gap = gaps[idx_max]
        max_efficiency = efficiencies[idx_max]
        
        return {
            "optimal_gap": optimal_gap,
            "max_efficiency": max_efficiency,
            "gaps": gaps,
            "efficiencies": np.array(efficiencies),
        }
    
    def optimize_thickness(self, thickness_range: Tuple[float, float] = (0.1, 500),
                          n_points: int = 50) -> Dict:
        """
        优化吸收层厚度
        
        平衡吸收效率和载流子收集。
        """
        thicknesses = np.logspace(np.log10(thickness_range[0]),
                                  np.log10(thickness_range[1]),
                                  n_points)
        
        currents = []
        for d in thicknesses:
            J = self.calculate_photocurrent(d)
            currents.append(J)
        
        currents = np.array(currents)
        
        # 找到饱和点
        J_sat = currents[-1]
        idx_90 = np.where(currents > 0.9 * J_sat)[0][0]
        optimal_thickness = thicknesses[idx_90]
        
        return {
            "optimal_thickness": optimal_thickness,
            "thicknesses": thicknesses,
            "currents": currents,
            "J_saturation": J_sat,
        }
    
    def plot_solar_cell_performance(self, save_path: Optional[str] = None):
        """绘制太阳能电池性能图"""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. AM1.5G光谱
        ax1.fill_between(self.am15_energies, self.am15_flux, alpha=0.3, color='orange')
        ax1.plot(self.am15_energies, self.am15_flux, 'r-', linewidth=1.5)
        ax1.set_xlabel('Photon energy (eV)')
        ax1.set_ylabel('Spectral irradiance (W/m²/eV)')
        ax1.set_title('AM1.5G Solar Spectrum')
        ax1.set_xlim([0.3, 4.5])
        ax1.grid(True, alpha=0.3)
        
        # 2. 吸收系数
        alpha = self.calculate_absorption(self.am15_energies)
        ax2.semilogy(self.am15_energies, alpha, 'b-', linewidth=2)
        ax2.axvline(x=self.params.band_gap, color='r', linestyle='--',
                   label=f'E_g = {self.params.band_gap:.2f} eV')
        ax2.set_xlabel('Photon energy (eV)')
        ax2.set_ylabel('Absorption coefficient (cm⁻¹)')
        ax2.set_title('Absorption Coefficient')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 带隙优化
        opt_result = self.optimize_band_gap()
        ax3.plot(opt_result["gaps"], opt_result["efficiencies"], 'g-', linewidth=2)
        ax3.axvline(x=opt_result["optimal_gap"], color='r', linestyle='--',
                   label=f'Optimal E_g = {opt_result["optimal_gap"]:.2f} eV')
        ax3.set_xlabel('Band gap (eV)')
        ax3.set_ylabel('Efficiency (%)')
        ax3.set_title('Shockley-Queisser Limit')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 厚度优化
        thick_result = self.optimize_thickness()
        ax4.semilogx(thick_result["thicknesses"],
                    thick_result["currents"] / thick_result["currents"][-1] * 100,
                    'm-', linewidth=2)
        ax4.axvline(x=thick_result["optimal_thickness"], color='r', linestyle='--',
                   label=f'Optimal d = {thick_result["optimal_thickness"]:.1f} μm')
        ax4.set_xlabel('Thickness (μm)')
        ax4.set_ylabel('Relative current (%)')
        ax4.set_title('Thickness Optimization')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


# =============================================================================
# 应用2: 量子阱LED
# =============================================================================

@dataclass
class QuantumWellLEDParameters:
    """量子阱LED参数"""
    
    # 材料
    barrier_material: str = "GaAs"
    well_material: str = "InGaAs"
    
    # 几何
    well_width: float = 10.0  # nm
    barrier_width: float = 20.0  # nm
    n_wells: int = 3  # 量子阱数量
    
    # 组分
    indium_composition: float = 0.2  # InGaAs中In的组分
    
    # 温度
    temperature: float = 300.0  # K


class QuantumWellLED:
    """
    量子阱LED模拟器
    
    计算量子限制效应导致的能级移动和发光特性。
    """
    
    def __init__(self, params: QuantumWellLEDParameters):
        self.params = params
        
        # 材料参数
        self.m_e_eff = 0.067  # 电子有效质量 (GaAs)
        self.m_hh_eff = 0.50  # 重空穴有效质量
        self.Eg_bulk = 1.42  # eV (GaAs带隙)
        
        # 量子限制能量
        self.calculate_confinement()
    
    def calculate_confinement(self):
        """计算量子限制能级"""
        hbar = 1.055e-34  # J·s
        m_e = 9.109e-31  # kg
        eV = 1.602e-19  # J/eV
        
        L_w = self.params.well_width * 1e-9  # m
        
        # 无限深势阱近似
        # E_n = (ℏ²π²n²)/(2m*L²)
        
        # 电子限制能量 (n=1)
        E1e = (hbar**2 * np.pi**2) / (2 * self.m_e_eff * m_e * L_w**2) / eV
        
        # 空穴限制能量
        E1h = (hbar**2 * np.pi**2) / (2 * self.m_hh_eff * m_e * L_w**2) / eV
        
        # 有效带隙
        self.Eg_effective = self.Eg_bulk + E1e + E1h
        
        # 量子限制能量存储
        self.E_confinement_e = E1e
        self.E_confinement_h = E1h
    
    def emission_wavelength(self) -> float:
        """计算发射波长"""
        wavelength_nm = 1239.8 / self.Eg_effective
        return wavelength_nm
    
    def quantum_confined_stark_effect(self, electric_field: float) -> float:
        """
        量子限制Stark效应 (QCSE)
        
        电场导致红移。
        
        Parameters:
        -----------
        electric_field : float
            电场 (V/m)
        
        Returns:
        --------
        float : 带隙移动 (eV)
        """
        # 简化公式
        # ΔE ≈ - (e²F²L⁴)/(24ℏ²) * (1/m_e + 1/m_h)
        e = 1.602e-19
        hbar = 1.055e-34
        m_e = 9.109e-31
        L = self.params.well_width * 1e-9
        
        m_reduced = (self.m_e_eff * self.m_hh_eff) / (self.m_e_eff + self.m_hh_eff)
        
        delta_E_J = -(e**2 * electric_field**2 * L**4) / (24 * hbar**2) * (1/m_reduced / m_e)
        delta_E_eV = delta_E_J / e
        
        return delta_E_eV
    
    def exciton_binding_2d(self) -> float:
        """
        计算2D激子结合能
        
        2D激子结合能是3D的4倍。
        """
        # 3D激子结合能 (GaAs)
        E_b_3d = 0.004  # eV
        
        # 2D增强
        # 实际值在3D和严格2D之间
        E_b_2d = 4 * E_b_3d  # 近似
        
        return E_b_2d
    
    def luminescence_spectrum(self, energies: np.ndarray,
                             temperature: Optional[float] = None,
                             broadening: float = 0.01) -> np.ndarray:
        """
        计算发光光谱
        
        包括:
        - 带边发射
        - 激子峰
        - 热展宽
        """
        T = temperature or self.params.temperature
        kT = 8.617e-5 * T  # eV
        
        spectrum = np.zeros_like(energies)
        
        # 1. 带边发射 (联合态密度)
        for i, E in enumerate(energies):
            if E > self.Eg_effective:
                # 阶梯状DOS
                spectrum[i] = 1.0
        
        # 2. 激子峰
        E_b = self.exciton_binding_2d()
        E_exciton = self.Eg_effective - E_b
        
        # 激子贡献 (洛伦兹线型)
        exciton_strength = 5.0  # 激子增强
        spectrum += exciton_strength * (broadening / np.pi) / \
                   ((energies - E_exciton)**2 + broadening**2)
        
        # 3. 热展宽 (费米-狄拉克)
        fermi_factor = 1 / (1 + np.exp((energies - self.Eg_effective) / kT))
        spectrum *= fermi_factor
        
        return spectrum
    
    def internal_quantum_efficiency(self, carrier_density: float = 1e18) -> float:
        """
        计算内量子效率
        
        IQE = radiative_rate / (radiative_rate + nonradiative_rate)
        """
        # 辐射复合率 (简化)
        B = 2e-10  # cm³/s, 辐射复合系数
        R_rad = B * carrier_density
        
        # 非辐射复合率 (SRH + Auger)
        tau_nr = 10e-9  # s
        R_nr = 1 / tau_nr
        
        IQE = R_rad / (R_rad + R_nr)
        
        return IQE


# =============================================================================
# 应用3: 色心量子比特集成
# =============================================================================

class ColorCenterQuantumComputing:
    """
    色心量子比特量子计算应用
    
    实现量子计算的基本操作和算法模拟。
    """
    
    def __init__(self, color_center_type: str = "NV"):
        from many_body.exciton_properties import get_material
        from defect_excited.color_center import get_color_center
        
        self.cc_type = color_center_type
        self.cc = get_color_center(color_center_type)
        
        # 量子比特参数
        self.T1 = self.cc.radiative_lifetime * 1e-9  # s
        self.T2 = 1e-3  # s, 典型值
    
    def coherence_metrics(self) -> Dict[str, float]:
        """计算相干性指标"""
        # 品质因子
        Q_T1 = self.T1 * (self.cc.zfs_d / (6.626e-34 / 1.602e-19))
        Q_T2 = self.T2 * (self.cc.zfs_d / (6.626e-34 / 1.602e-19))
        
        # 单量子比特门数
        gate_time = 10e-9  # 10 ns
        n_gates_T2 = self.T2 / gate_time
        
        return {
            "T1": self.T1,
            "T2": self.T2,
            "T2_star": self.T2 / 100,  # 简化
            "Q_T1": Q_T1,
            "Q_T2": Q_T2,
            "max_gates": n_gates_T2,
        }
    
    def rabi_oscillations(self, pulse_durations: np.ndarray,
                         rabi_freq: float = 10e6) -> np.ndarray:
        """
        模拟Rabi振荡
        
        P(|1>) = sin²(Ωt/2)
        """
        Omega = rabi_freq
        population = np.sin(Omega * pulse_durations / 2)**2
        
        # 添加T2*衰减
        T2_star = 10e-6  # s
        decay = np.exp(-pulse_durations / T2_star)
        
        return population * decay
    
    def ramsey_fringes(self, delays: np.ndarray,
                      detuning: float = 1e6) -> np.ndarray:
        """
        Ramsey干涉条纹
        
        S(t) = cos(2πΔf·t) · exp(-t/T2*)
        """
        fringes = np.cos(2 * np.pi * detuning * delays)
        decay = np.exp(-delays / (self.T2 / 100))
        
        return fringes * decay
    
    def spin_echo_decay(self, total_times: np.ndarray) -> np.ndarray:
        """
        Hahn自旋回波衰减
        
        抑制低频噪声，测量T2。
        """
        # exp(-(t/T2)^3) for spectral diffusion
        signal = np.exp(-(total_times / self.T2)**3)
        return signal
    
    def quantum_error_rate(self) -> Dict[str, float]:
        """计算量子错误率"""
        gate_time = 10e-9  # s
        
        # 退相干错误
        p_decoherence = 1 - np.exp(-gate_time / self.T2)
        
        # 初始化错误 (简化)
        p_init = 0.01
        
        # 读出错误
        p_readout = 0.05
        
        return {
            "gate_error": p_decoherence,
            "initialization_error": p_init,
            "readout_error": p_readout,
            "total_error": p_decoherence + p_init + p_readout,
        }


# =============================================================================
# 示例和测试
# =============================================================================

def example_solar_cell():
    """太阳能电池优化示例"""
    print("=" * 60)
    print("Application: Solar Cell Optimization")
    print("=" * 60)
    
    params = SolarCellParameters(
        absorber_material="GaAs",
        band_gap=1.42,
        thickness=100.0,
    )
    
    sc = SolarCellOptimizer(params)
    
    # Shockley-Queisser极限
    sq = sc.calculate_shockley_queisser()
    print("\nShockley-Queisser Analysis:")
    print(f"  J_sc = {sq['J_sc']:.2f} A/m²")
    print(f"  V_oc = {sq['V_oc']:.3f} V")
    print(f"  FF = {sq['FF']:.3f}")
    print(f"  Efficiency = {sq['efficiency']:.2f}%")
    
    # 带隙优化
    opt = sc.optimize_band_gap()
    print(f"\nOptimal band gap: {opt['optimal_gap']:.2f} eV")
    print(f"Maximum efficiency: {opt['max_efficiency']:.2f}%")
    
    # 厚度优化
    thick = sc.optimize_thickness()
    print(f"\nOptimal thickness: {thick['optimal_thickness']:.1f} μm")


def example_qw_led():
    """量子阱LED示例"""
    print("\n" + "=" * 60)
    print("Application: Quantum Well LED")
    print("=" * 60)
    
    params = QuantumWellLEDParameters(
        well_width=10.0,  # nm
        n_wells=3,
    )
    
    led = QuantumWellLED(params)
    
    print(f"\nQuantum Well Parameters:")
    print(f"  Well width: {params.well_width} nm")
    print(f"  Electron confinement: {led.E_confinement_e*1000:.1f} meV")
    print(f"  Hole confinement: {led.E_confinement_h*1000:.1f} meV")
    print(f"  Effective band gap: {led.Eg_effective:.3f} eV")
    print(f"  Emission wavelength: {led.emission_wavelength():.1f} nm")
    
    print(f"\nExciton binding energy: {led.exciton_binding_2d()*1000:.1f} meV")
    print(f"Internal quantum efficiency: {led.internal_quantum_efficiency()*100:.1f}%")
    
    # QCSE
    E_field = 1e7  # V/m
    stark_shift = led.quantum_confined_stark_effect(E_field)
    print(f"\nQCSE at {E_field/1e6:.1f} MV/m: {stark_shift*1000:.2f} meV")


def example_quantum_computing():
    """量子计算应用示例"""
    print("\n" + "=" * 60)
    print("Application: Color Center Quantum Computing")
    print("=" * 60)
    
    qc = ColorCenterQuantumComputing("NV")
    
    # 相干性指标
    metrics = qc.coherence_metrics()
    print("\nCoherence Metrics:")
    print(f"  T1 = {metrics['T1']*1e3:.1f} ms")
    print(f"  T2 = {metrics['T2']*1e3:.1f} ms")
    print(f"  T2* = {metrics['T2_star']*1e6:.1f} μs")
    print(f"  Max single-qubit gates: {metrics['max_gates']:.0f}")
    
    # 错误率
    errors = qc.quantum_error_rate()
    print(f"\nQuantum Error Rates:")
    print(f"  Gate error: {errors['gate_error']*100:.4f}%")
    print(f"  Readout error: {errors['readout_error']*100:.1f}%")


if __name__ == "__main__":
    example_solar_cell()
    example_qw_led()
    example_quantum_computing()
