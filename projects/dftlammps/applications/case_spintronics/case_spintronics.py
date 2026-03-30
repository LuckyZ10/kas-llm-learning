"""
case_spintronics.py

Spintronics Devices - Application Case Study

This module demonstrates advanced spintronic device simulations:
- Magnetic Tunnel Junction (MTJ) memory cells
- Spin-orbit torque (SOT) MRAM
- Spin transistors (Datta-Das spinFET)
- Domain wall racetrack memory

References:
- Datta & Das, Appl. Phys. Lett. 56, 665 (1990) - SpinFET
- Parkin et al., Science 320, 190 (2008) - Racetrack memory
- Miron et al., Nature 476, 189 (2011) - SOT switching
"""

import numpy as np
from scipy import linalg, integrate
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, '/root/.openclaw/workspace/dftlammps')

from spin_transport.spin_transport import (
    MagneticLayer, TunnelBarrier, MagneticTunnelJunction,
    SpinTransferTorque, SpinHallEffect, SpinOrbitTorque
)


class MTJMemoryCell:
    """
    Magnetic Tunnel Junction memory cell simulation.
    
    Models reading and writing operations for STT-MRAM.
    """
    
    def __init__(self,
                 free_layer: MagneticLayer,
                 pinned_layer: MagneticLayer,
                 barrier: TunnelBarrier,
                 cell_area: float = 100e-9 * 100e-9):  # m²
        self.mtj = MagneticTunnelJunction(free_layer, pinned_layer, barrier)
        self.cell_area = cell_area
        
        # Memory parameters
        self.write_pulse_width = 10e-9  # s
        self.read_voltage = 0.1  # V
    
    def read_state(self) -> int:
        """
        Read memory state by measuring resistance.
        
        Returns:
            0 for parallel (low resistance), 1 for antiparallel (high resistance)
        """
        R = self.mtj.calculate_resistance()
        R_P = self.mtj.calculate_resistance(angle=0)
        R_AP = self.mtj.calculate_resistance(angle=np.pi)
        
        # Threshold between P and AP states
        R_mid = (R_P + R_AP) / 2
        
        return 0 if R < R_mid else 1
    
    def calculate_read_disturbance(self, 
                                  num_read_cycles: int = 1e6) -> float:
        """
        Calculate probability of read disturbance.
        
        Read current should not switch the free layer.
        """
        # STT from read current
        I_read = self.read_voltage / self.mtj.calculate_resistance()
        J_read = I_read / self.cell_area
        
        # Critical current for switching
        stt = SpinTransferTorque(self.mtj.free_layer)
        # Simplified calculation
        J_c = 1e10  # A/m² (typical)
        
        # Read disturb probability (simplified model)
        if J_read < 0.1 * J_c:
            return 1e-15  # Essentially zero
        else:
            return (J_read / J_c) ** 2
    
    def calculate_write_current(self, 
                               switching_time: float = 10e-9) -> float:
        """
        Calculate required write current for given switching time.
        
        Uses Slonczewski switching model.
        """
        # Critical current density
        # J_c0 = (2e/ℏ) (α M_s t_f) (H_k + H_d + 2πM_s) / g(θ)
        
        layer = self.mtj.free_layer
        
        # Physical constants
        e = 1.602e-19
        hbar = 1.055e-34
        mu0 = 4 * np.pi * 1e-7
        
        # Anisotropy field (simplified)
        H_k = 2 * layer.anisotropy_constant / layer.saturation_magnetization
        
        # Critical current density (A/m²)
        J_c0 = (2 * e / hbar) * layer.damping * \
               layer.saturation_magnetization * 1e6 * \
               layer.thickness * 1e-9 * H_k / 0.5  # g(θ) ≈ 0.5
        
        # Thermal stability factor
        Delta = (layer.anisotropy_constant * self.cell_area * 
                 layer.thickness * 1e-9) / (1.38e-23 * layer.damping * 300)
        
        # Switching current for given pulse width
        # J_sw = J_c0 [1 - (1/Δ) ln(π/2 × τ_sw/τ0)]
        tau_0 = 1e-9  # s
        
        if switching_time > tau_0:
            correction = 1 - (1/Delta) * np.log(np.pi/2 * switching_time/tau_0)
        else:
            correction = 1
        
        J_sw = J_c0 * correction
        
        return J_sw * self.cell_area  # Convert to current
    
    def calculate_thermal_stability(self, temperature: float = 300.0) -> float:
        """
        Calculate thermal stability factor Δ = E_b / k_B T.
        
        Should be > 40 for 10-year retention.
        """
        layer = self.mtj.free_layer
        
        # Energy barrier
        E_b = layer.anisotropy_constant * self.cell_area * layer.thickness * 1e-9
        
        # Thermal stability
        kB = 1.38e-23
        Delta = E_b / (kB * temperature)
        
        return Delta


class SpinFET:
    """
    Datta-Das spin field-effect transistor.
    
    Uses spin-orbit coupling (Rashba effect) for spin manipulation.
    """
    
    def __init__(self,
                 channel_length: float = 1e-6,  # m
                 channel_width: float = 100e-9,  # m
                 rashba_coefficient: float = 1e-10):  # eV·m
        self.L = channel_length
        self.W = channel_width
        self.alpha_R = rashba_coefficient
        
        # Material parameters
        self.mobility = 1e4  # cm²/V·s
        self.m_eff = 0.05  # Effective mass (m₀)
    
    def calculate_spin_precession(self, 
                                 gate_voltage: float,
                                 drain_voltage: float = 0.01) -> float:
        """
        Calculate spin precession angle through channel.
        
        θ = 2m*α_R L / ℏ²
        """
        # Electric field from gate voltage
        E_field = gate_voltage / 10e-9  # V/m (assuming 10nm oxide)
        
        # Rashba spin-orbit coupling
        # α_R depends on electric field
        alpha_eff = self.alpha_R * E_field / 1e6  # Scale appropriately
        
        # Spin precession angle
        hbar = 1.055e-34
        m0 = 9.11e-31
        m_star = self.m_eff * m0
        
        theta = 2 * m_star * alpha_eff * self.L / hbar**2
        
        return theta
    
    def calculate_transconductance(self,
                                  Vg_range: np.ndarray,
                                  Vds: float = 0.01) -> np.ndarray:
        """
        Calculate spin-dependent transconductance.
        
        Current depends on spin precession angle:
        I ∝ 1 + cos(θ)
        """
        gm_values = []
        
        for Vg in Vg_range:
            theta = self.calculate_spin_precession(Vg, Vds)
            
            # Channel conductance
            sigma = self.mobility * 1e-4  # Convert to m²/V·s
            
            # Spin-dependent conductance
            G = sigma * self.W / self.L * (1 + np.cos(theta)) / 2
            
            Ids = G * Vds
            gm_values.append(Ids)
        
        return np.array(gm_values)
    
    def calculate_on_off_ratio(self,
                              Vg_on: float = 1.0,
                              Vg_off: float = 0.0) -> float:
        """
        Calculate ON/OFF current ratio.
        """
        theta_on = self.calculate_spin_precession(Vg_on)
        theta_off = self.calculate_spin_precession(Vg_off)
        
        I_on = (1 + np.cos(theta_on)) / 2
        I_off = (1 + np.cos(theta_off)) / 2
        
        if I_off > 1e-10:
            ratio = I_on / I_off
        else:
            ratio = float('inf')
        
        return ratio


class SOTMRAM:
    """
    Spin-orbit torque MRAM using heavy metal/ferromagnet bilayers.
    
    Uses SOT for deterministic switching with in-plane current.
    """
    
    def __init__(self,
                 heavy_metal: str = "Pt",
                 free_layer: MagneticLayer = None,
                 dimensions: Tuple[float, float, float] = (100e-9, 100e-9, 1e-9)):
        self.heavy_metal = heavy_metal
        self.dimensions = dimensions  # (length, width, thickness)
        
        self.free_layer = free_layer or MagneticLayer(
            name="CoFeB",
            thickness=1.0,
            magnetization=np.array([0, 0, 1]),
            exchange_stiffness=20.0,
            damping=0.01,
            saturation_magnetization=1.2,
            anisotropy_constant=1e5
        )
        
        self.sot = SpinOrbitTorque(heavy_metal, self.free_layer)
    
    def calculate_critical_current_density(self,
                                          switching_time: float = 1e-9) -> float:
        """
        Calculate critical SOT switching current density.
        
        Uses anti-damping torque mechanism.
        """
        # Physical constants
        e = 1.602e-19
        hbar = 1.055e-34
        mu0 = 4 * np.pi * 1e-7
        
        layer = self.free_layer
        
        # Spin Hall angle
        spin_hall_angles = {'Pt': 0.08, 'Ta': -0.12, 'W': -0.30}
        theta_SH = spin_hall_angles.get(self.heavy_metal, 0.1)
        
        # Effective field from anisotropy
        H_eff = 2 * layer.anisotropy_constant / (mu0 * layer.saturation_magnetization**2 * 1e6)
        
        # Critical current density
        # J_c = (2e/ℏ) (M_s t) (H_eff) / θ_SH
        
        M_s = layer.saturation_magnetization * 1e6  # A/m
        t = layer.thickness * 1e-9  # m
        
        J_c = (2 * e / hbar) * M_s * t * H_eff / abs(theta_SH)
        
        return J_c
    
    def calculate_switching_efficiency(self) -> float:
        """
        Calculate switching efficiency (switching per unit current).
        
        Returns radians per A/m².
        """
        layer = self.free_layer
        
        # Spin Hall angle
        spin_hall_angles = {'Pt': 0.08, 'Ta': -0.12, 'W': -0.30}
        theta_SH = spin_hall_angles.get(self.heavy_metal, 0.1)
        
        # Efficiency depends on spin Hall angle and layer properties
        efficiency = theta_SH / (layer.saturation_magnetization * 
                                 layer.thickness * 1e-9)
        
        return efficiency
    
    def estimate_write_energy(self, 
                             switching_time: float = 1e-9) -> float:
        """
        Calculate energy per write operation.
        """
        J_c = self.calculate_critical_current_density(switching_time)
        
        length, width, _ = self.dimensions
        area = width * 1e-9  # Cross-section of heavy metal
        
        I_write = J_c * area
        
        # Resistivity of heavy metal
        resistivities = {'Pt': 10.6e-8, 'Ta': 13.1e-8, 'W': 5.6e-8}
        rho = resistivities.get(self.heavy_metal, 10e-8)
        
        # Resistance
        R = rho * length / area
        
        # Energy
        E_write = I_write**2 * R * switching_time
        
        return E_write


class DomainWallRacetrack:
    """
    Domain wall racetrack memory simulation.
    
    Information stored in magnetic domain walls moved by current.
    """
    
    def __init__(self,
                 wire_width: float = 100e-9,
                 wire_thickness: float = 10e-9,
                 material: str = "CoNi"):
        self.width = wire_width
        self.thickness = wire_thickness
        self.material = material
        
        # Material parameters
        self.M_s = 8e5  # A/m
        self.A = 1e-11  # Exchange stiffness (J/m)
        self.K_u = 1e5  # Anisotropy (J/m³)
        self.alpha = 0.01  # Damping
        self.P = 0.5  # Spin polarization
        
        # Domain wall width
        self.delta_DW = np.sqrt(self.A / self.K_u)
    
    def calculate_domain_wall_velocity(self,
                                      current_density: float) -> float:
        """
        Calculate domain wall velocity from spin-transfer torque.
        
        v_DW = (γ ℏ P / 2eM_s) J / (1 + α²)
        """
        gamma = 1.76e11  # Gyromagnetic ratio (rad/s·T)
        hbar = 1.055e-34
        e = 1.602e-19
        
        # Non-adiabatic STT contribution
        beta = 0.01  # Non-adiabaticity parameter
        
        v_DW = (gamma * hbar * self.P / (2 * e * self.M_s)) * \
               (current_density / (self.alpha + 1/self.alpha))
        
        return v_DW
    
    def calculate_critical_current(self) -> float:
        """
        Calculate critical current density for domain wall motion.
        """
        # Walker breakdown threshold
        # J_c = (2eM_s/hP) (γ α K_u / μ_0 M_s) δ_DW
        
        hbar = 1.055e-34
        e = 1.602e-19
        mu0 = 4 * np.pi * 1e-7
        gamma = 1.76e11
        
        H_anisotropy = 2 * self.K_u / (mu0 * self.M_s)
        
        J_c = (2 * e * self.M_s / (hbar * self.P)) * \
              (gamma * self.alpha * H_anisotropy * self.delta_DW)
        
        return J_c
    
    def estimate_data_rate(self,
                          bit_spacing: float = 50e-9,
                          current_density: float = 1e11) -> float:
        """
        Estimate data read/write rate.
        
        Returns bits per second.
        """
        v_DW = self.calculate_domain_wall_velocity(current_density)
        
        # Bit rate
        data_rate = v_DW / bit_spacing
        
        return data_rate
    
    def calculate_density(self, 
                         bit_spacing: float = 50e-9) -> float:
        """
        Calculate storage density.
        
        Returns bits per m².
        """
        area_per_bit = self.width * bit_spacing
        
        density = 1 / area_per_bit
        
        return density


class SkyrmionDevice:
    """
    Magnetic skyrmion-based memory and logic devices.
    
    Skyrmions are topologically protected spin textures.
    """
    
    def __init__(self,
                 film_thickness: float = 1e-9,
                 DMI_constant: float = 3e-3):  # J/m²
        self.thickness = film_thickness
        self.D = DMI_constant  # Dzyaloshinskii-Moriya interaction
        
        # Material parameters
        self.A = 1e-11  # Exchange stiffness
        self.M_s = 1e6  # Saturation magnetization
        self.K_u = 1e6  # Anisotropy
    
    def calculate_skyrmion_size(self) -> float:
        """
        Calculate skyrmion diameter.
        
        d ≈ 4A / D
        """
        diameter = 4 * self.A / abs(self.D)
        
        return diameter
    
    def calculate_creation_current(self) -> float:
        """
        Estimate current required for skyrmion creation.
        """
        # Simplified model
        J_create = 1e11  # A/m² (typical)
        
        return J_create
    
    def calculate_stability(self, temperature: float = 300.0) -> float:
        """
        Calculate thermal stability of skyrmion.
        
        Returns energy barrier in k_B T units.
        """
        # Skyrmion energy
        E_skyrmion = np.pi * self.thickness * (4 * self.A + 
                                                np.pi * abs(self.D) * self.calculate_skyrmion_size())
        
        kB = 1.38e-23
        stability = E_skyrmion / (kB * temperature)
        
        return stability


def example_mtj_memory():
    """
    Example: MTJ memory cell performance.
    """
    print("=" * 70)
    print("Example: MTJ Memory Cell (STT-MRAM)")
    print("=" * 70)
    
    # Create magnetic layers
    free_layer = MagneticLayer(
        name="CoFeB_Free",
        thickness=2.0,
        magnetization=np.array([0, 0, 1]),
        exchange_stiffness=20.0,
        damping=0.01,
        saturation_magnetization=1.2,
        anisotropy_constant=1e5
    )
    
    pinned_layer = MagneticLayer(
        name="CoFeB_Pinned",
        thickness=3.0,
        magnetization=np.array([0, 0, -1]),
        exchange_stiffness=20.0,
        damping=0.01,
        saturation_magnetization=1.2
    )
    
    barrier = TunnelBarrier(
        material="MgO",
        thickness=1.0,
        height=2.0
    )
    
    # Create memory cell
    cell = MTJMemoryCell(free_layer, pinned_layer, barrier)
    
    print(f"\nMTJ Memory Cell Parameters:")
    print(f"  Cell area: {cell.cell_area*1e12:.0f} nm²")
    print(f"  TMR ratio: {cell.mtj.calculate_tmr_ratio()*100:.0f}%")
    
    # Read characteristics
    R_P = cell.mtj.calculate_resistance(angle=0)
    R_AP = cell.mtj.calculate_resistance(angle=np.pi)
    
    print(f"\nRead characteristics:")
    print(f"  Parallel resistance: {R_P:.1f} kΩ")
    print(f"  Antiparallel resistance: {R_AP:.1f} kΩ")
    print(f"  Read voltage: {cell.read_voltage*1000:.0f} mV")
    
    # Write characteristics
    I_write = cell.calculate_write_current()
    Delta = cell.calculate_thermal_stability()
    
    print(f"\nWrite characteristics:")
    print(f"  Write current: {I_write*1e6:.1f} μA")
    print(f"  Thermal stability Δ: {Delta:.1f} (need > 40)")
    
    return cell


def example_spinfet():
    """
    Example: Datta-Das spinFET simulation.
    """
    print("\n" + "=" * 70)
    print("Example: Datta-Das SpinFET")
    print("=" * 70)
    
    # Create spinFET
    spinfet = SpinFET(
        channel_length=1e-6,
        channel_width=100e-9,
        rashba_coefficient=1e-10
    )
    
    print(f"\nSpinFET Parameters:")
    print(f"  Channel length: {spinfet.L*1e6:.1f} μm")
    print(f"  Channel width: {spinfet.W*1e9:.0f} nm")
    print(f"  Rashba coefficient: {spinfet.alpha_R*1e10:.1f} × 10⁻¹⁰ eV·m")
    
    # Spin precession vs gate voltage
    print(f"\nSpin precession angle vs gate voltage:")
    print(f"{'Vg (V)':>10} {'θ (rad)':>12} {'θ (°)':>12}")
    print("-" * 40)
    
    for Vg in [0, 0.25, 0.5, 0.75, 1.0]:
        theta = spinfet.calculate_spin_precession(Vg)
        print(f"{Vg:>10.2f} {theta:>12.2f} {np.degrees(theta):>12.1f}")
    
    # ON/OFF ratio
    ratio = spinfet.calculate_on_off_ratio()
    print(f"\nON/OFF ratio: {ratio:.1e}")
    
    return spinfet


def example_sot_mram():
    """
    Example: SOT-MRAM cell.
    """
    print("\n" + "=" * 70)
    print("Example: SOT-MRAM Cell")
    print("=" * 70)
    
    # Create SOT-MRAM
    sot_mram = SOTMRAM(
        heavy_metal="Pt",
        dimensions=(100e-9, 100e-9, 1e-9)
    )
    
    print(f"\nSOT-MRAM Parameters:")
    print(f"  Heavy metal: {sot_mram.heavy_metal}")
    print(f"  Cell dimensions: {sot_mram.dimensions[0]*1e9:.0f} × "
          f"{sot_mram.dimensions[1]*1e9:.0f} nm²")
    
    # Critical current
    J_c = sot_mram.calculate_critical_current_density()
    print(f"\nSwitching characteristics:")
    print(f"  Critical current density: {J_c/1e10:.2f} × 10¹⁰ A/m²")
    
    # Write energy
    E_write = sot_mram.estimate_write_energy()
    print(f"  Write energy: {E_write*1e15:.1f} fJ")
    
    # Efficiency
    efficiency = sot_mram.calculate_switching_efficiency()
    print(f"  Switching efficiency: {efficiency:.2e} rad/(A/m²)")
    
    return sot_mram


def example_racetrack():
    """
    Example: Domain wall racetrack memory.
    """
    print("\n" + "=" * 70)
    print("Example: Domain Wall Racetrack Memory")
    print("=" * 70)
    
    # Create racetrack
    racetrack = DomainWallRacetrack(
        wire_width=100e-9,
        wire_thickness=10e-9
    )
    
    print(f"\nRacetrack Parameters:")
    print(f"  Wire width: {racetrack.width*1e9:.0f} nm")
    print(f"  Wire thickness: {racetrack.thickness*1e9:.0f} nm")
    print(f"  Domain wall width: {racetrack.delta_DW*1e9:.1f} nm")
    
    # Domain wall motion
    print(f"\nDomain wall velocity vs current:")
    print(f"{'J (10¹⁰ A/m²)':>15} {'v_DW (m/s)':>15}")
    print("-" * 35)
    
    for J in [0.5, 1.0, 1.5, 2.0, 2.5]:
        J_Am2 = J * 1e10
        v = racetrack.calculate_domain_wall_velocity(J_Am2)
        print(f"{J:>15.1f} {v:>15.1f}")
    
    # Data rate
    data_rate = racetrack.estimate_data_rate(
        bit_spacing=50e-9,
        current_density=1e11
    )
    
    print(f"\nData rate: {data_rate/1e6:.0f} Mbps")
    
    # Storage density
    density = racetrack.calculate_density(bit_spacing=50e-9)
    print(f"Storage density: {density/1e9:.2f} Gbit/cm²")
    
    return racetrack


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Spintronics Devices - Application Cases")
    print("=" * 70)
    
    # Run examples
    mtj = example_mtj_memory()
    spinfet = example_spinfet()
    sot = example_sot_mram()
    racetrack = example_racetrack()
    
    print("\n" + "=" * 70)
    print("Spintronics Devices Cases - Complete")
    print("=" * 70)
