"""
spin_transport.py

Spin Transport and Spintronics Module

This module provides tools for spin-dependent transport calculations including:
- Magnetic Tunnel Junctions (MTJ)
- Spin Transfer Torque (STT)
- Spin Hall Effect
- Non-local Spin Valves

References:
- Slonczewski, J. Magn. Magn. Mater. 159, L1 (1996) - STT
- Hirsch, PRL 83, 1834 (1999) - Spin Hall effect
- Ralph & Stiles, JMMM 320, 1190 (2008) - STT review
"""

import numpy as np
from scipy import linalg, optimize
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class MagnetizationDirection(Enum):
    """Predefined magnetization directions."""
    PARALLEL = "parallel"  # ↑↑
    ANTIPARALLEL = "antiparallel"  # ↑↓
    PERPENDICULAR = "perpendicular"  # ↑→
    CUSTOM = "custom"


@dataclass
class MagneticLayer:
    """
    Configuration for a magnetic layer in a spintronic device.
    """
    
    name: str
    thickness: float  # nm
    magnetization: np.ndarray  # 3D unit vector
    exchange_stiffness: float  # pJ/m
    damping: float  # Gilbert damping
    saturation_magnetization: float  # MA/m
    anisotropy_constant: float = 0.0  # J/m³
    anisotropy_axis: Optional[np.ndarray] = None
    
    def __post_init__(self):
        # Normalize magnetization
        self.magnetization = self.magnetization / np.linalg.norm(self.magnetization)
        
        if self.anisotropy_axis is not None:
            self.anisotropy_axis = self.anisotropy_axis / np.linalg.norm(self.anisotropy_axis)
    
    def get_magnetization_angle(self) -> Tuple[float, float]:
        """
        Get spherical angles (theta, phi) of magnetization.
        """
        m = self.magnetization
        theta = np.arccos(m[2])
        phi = np.arctan2(m[1], m[0])
        return theta, phi


@dataclass
class TunnelBarrier:
    """
    Configuration for tunnel barrier in MTJ.
    """
    
    material: str
    thickness: float  # nm
    height: float  # eV (barrier height)
    effective_mass: float = 0.5  # in units of electron mass
    
    def calculate_transmission_coefficient(self, energy: float,
                                          spin: int = 0) -> float:
        """
        Calculate tunneling transmission using WKB approximation.
        
        T ≈ exp(-2 ∫ √(2m*(V-E))/ℏ dx)
        """
        hbar = 6.582e-16  # eV·s
        m0 = 9.109e-31  # kg
        
        if energy >= self.height:
            return 1.0
        
        # WKB approximation
        m_eff = self.effective_mass * m0
        barrier_height_eV = self.height - energy
        
        # Convert thickness to meters
        thickness_m = self.thickness * 1e-9
        
        # Decay constant
        kappa = np.sqrt(2 * m_eff * barrier_height_eV * 1.602e-19) / hbar
        
        transmission = np.exp(-2 * kappa * thickness_m)
        
        return transmission


class MagneticTunnelJunction:
    """
    Magnetic Tunnel Junction (MTJ) simulator.
    
    Models the TMR (Tunnel Magnetoresistance) effect and spin-dependent
    tunneling in MTJ devices.
    """
    
    def __init__(self,
                 free_layer: MagneticLayer,
                 pinned_layer: MagneticLayer,
                 barrier: TunnelBarrier):
        self.free_layer = free_layer
        self.pinned_layer = pinned_layer
        self.barrier = barrier
        
        # Spin polarization (material dependent)
        self.spin_polarization = 0.5  # Typical value
        
        # Julliere model parameter
        self.julliere_factor = self._calculate_julliere_factor()
    
    def _calculate_julliere_factor(self) -> float:
        """
        Calculate Julliere TMR factor:
        
        TMR = 2P₁P₂ / (1 - P₁P₂)
        """
        P = self.spin_polarization
        tmr = 2 * P * P / (1 - P * P)
        return tmr
    
    def calculate_resistance(self, 
                            angle: Optional[float] = None) -> float:
        """
        Calculate MTJ resistance as function of magnetization angle.
        
        Uses the Slonczewski model for angular dependence.
        """
        # Get angle between magnetizations
        if angle is None:
            m1 = self.free_layer.magnetization
            m2 = self.pinned_layer.magnetization
            cos_theta = np.dot(m1, m2)
            angle = np.arccos(np.clip(cos_theta, -1, 1))
        
        # Parallel and antiparallel resistances
        R_P = 1.0  # Normalized
        R_AP = R_P * (1 + self.julliere_factor)
        
        # Angular dependence (Slonczewski formula)
        # R(θ) = R_P [1 + (TMR/(2+TMR)) (1 - cosθ)]
        R = R_P * (1 + (self.julliere_factor / (2 + self.julliere_factor)) * 
                   (1 - np.cos(angle)))
        
        return R
    
    def calculate_tmr_ratio(self) -> float:
        """
        Calculate TMR ratio:
        
        TMR = (R_AP - R_P) / R_P
        """
        R_AP = self.calculate_resistance(angle=np.pi)
        R_P = self.calculate_resistance(angle=0)
        
        tmr = (R_AP - R_P) / R_P
        return tmr
    
    def calculate_spin_current(self, bias_voltage: float,
                              angle: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate spin-up and spin-down current components.
        
        Returns:
            (I_up, I_down) in arbitrary units
        """
        P = self.spin_polarization
        
        # Transmission depends on relative magnetization orientation
        if angle is None:
            m1 = self.free_layer.magnetization
            m2 = self.pinned_layer.magnetization
            angle = np.arccos(np.clip(np.dot(m1, m2), -1, 1))
        
        # Spin-dependent transmission
        T_up = 1 + P * np.cos(angle / 2)
        T_down = 1 - P * np.cos(angle / 2)
        
        # Current proportional to transmission and voltage
        I_up = T_up * bias_voltage
        I_down = T_down * bias_voltage
        
        return I_up, I_down
    
    def calculate_iv_curve(self, bias_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate I-V characteristics for both parallel and antiparallel states.
        """
        I_parallel = []
        I_antiparallel = []
        
        for V in bias_range:
            # Parallel state
            I_up_P, I_down_P = self.calculate_spin_current(V, angle=0)
            I_parallel.append(I_up_P + I_down_P)
            
            # Antiparallel state
            I_up_AP, I_down_AP = self.calculate_spin_current(V, angle=np.pi)
            I_antiparallel.append(I_up_AP + I_down_AP)
        
        return np.array(I_parallel), np.array(I_antiparallel)


class SpinTransferTorque:
    """
    Spin Transfer Torque (STT) calculator.
    
    Implements the Landau-Lifshitz-Gilbert-Slonczewski (LLGS) equation
    for magnetization dynamics under spin-polarized current.
    """
    
    def __init__(self, magnetic_layer: MagneticLayer):
        self.layer = magnetic_layer
        
        # Physical constants
        self.hbar = 1.055e-34  # J·s
        self.e_charge = 1.602e-19  # C
        self.mu_B = 9.274e-24  # J/T
        self.gyromagnetic_ratio = 1.76e11  # rad/(s·T)
    
    def calculate_slonczewski_torque(self,
                                    current_density: float,
                                    pinned_magnetization: np.ndarray,
                                    polarization: float = 0.5) -> np.ndarray:
        """
        Calculate Slonczewski spin transfer torque.
        
        τ_STT = (ℏJ/2eM_s t) m × (m × m_p)
        
        where J is current density, m is free layer magnetization,
        m_p is pinned layer magnetization.
        """
        m = self.layer.magnetization
        m_p = pinned_magnetization / np.linalg.norm(pinned_magnetization)
        
        # Cross product m × m_p
        m_cross_mp = np.cross(m, m_p)
        
        # Double cross product m × (m × m_p)
        torque_direction = np.cross(m, m_cross_mp)
        
        # Torque magnitude
        # Simplified: actual formula depends on device geometry
        thickness_m = self.layer.thickness * 1e-9
        Ms = self.layer.saturation_magnetization * 1e6  # A/m
        
        prefactor = (self.hbar * current_density * polarization / 
                    (2 * self.e_charge * Ms * thickness_m))
        
        torque = prefactor * torque_direction
        
        return torque
    
    def calculate_field_like_torque(self,
                                   current_density: float,
                                   pinned_magnetization: np.ndarray,
                                   coefficient: float = 0.1) -> np.ndarray:
        """
        Calculate field-like (anti-damping) torque.
        
        τ_FL = coefficient × m × m_p
        """
        m = self.layer.magnetization
        m_p = pinned_magnetization / np.linalg.norm(pinned_magnetization)
        
        torque = coefficient * current_density * np.cross(m, m_p)
        
        return torque
    
    def llg_equation(self, m: np.ndarray, H_eff: np.ndarray,
                    current_density: float,
                    pinned_magnetization: np.ndarray,
                    time: float) -> np.ndarray:
        """
        Landau-Lifshitz-Gilbert equation with STT:
        
        dm/dt = -γ m × H_eff + α m × dm/dt + τ_STT
        
        where α is the Gilbert damping.
        """
        gamma = self.gyromagnetic_ratio
        alpha = self.layer.damping
        
        # Precession term
        precession = -gamma * np.cross(m, H_eff)
        
        # Damping term
        damping = -alpha * gamma * np.cross(m, np.cross(m, H_eff))
        
        # Spin transfer torque
        stt = self.calculate_slonczewski_torque(
            current_density, pinned_magnetization
        )
        
        # Total time derivative
        dm_dt = precession + damping + stt
        
        return dm_dt
    
    def integrate_llg(self, H_eff: np.ndarray,
                     current_density: float,
                     pinned_magnetization: np.ndarray,
                     time_span: Tuple[float, float],
                     dt: float = 1e-12) -> List[np.ndarray]:
        """
        Integrate LLG equation using simple Euler method.
        """
        times = np.arange(time_span[0], time_span[1], dt)
        m_trajectory = [self.layer.magnetization.copy()]
        
        m = self.layer.magnetization.copy()
        
        for t in times[:-1]:
            dm_dt = self.llg_equation(m, H_eff, current_density, 
                                     pinned_magnetization, t)
            m = m + dm_dt * dt
            
            # Normalize to keep |m| = 1
            m = m / np.linalg.norm(m)
            
            m_trajectory.append(m.copy())
        
        return m_trajectory


class SpinHallEffect:
    """
    Spin Hall Effect calculator.
    
    The spin Hall effect generates a spin current perpendicular to
    a charge current due to spin-orbit coupling.
    """
    
    def __init__(self, material: str = "Pt"):
        self.material = material
        
        # Spin Hall angle for common materials
        self.spin_hall_angles = {
            'Pt': 0.08,
            'Ta': -0.12,
            'W': -0.30,
            'Au': 0.003,
            'Cu': 0.0006,
        }
        
        self.spin_hall_angle = self.spin_hall_angles.get(material, 0.1)
        
        # Conductivity (S/m)
        self.conductivities = {
            'Pt': 9.4e6,
            'Ta': 6.3e6,
            'W': 1.8e7,
            'Au': 4.5e7,
            'Cu': 5.8e7,
        }
        
        self.conductivity = self.conductivities.get(material, 1e7)
    
    def calculate_spin_current(self, charge_current_density: float,
                              thickness: float) -> float:
        """
        Calculate generated spin current density.
        
        J_s = θ_SH × J_c
        
        where θ_SH is the spin Hall angle.
        """
        spin_current = self.spin_hall_angle * charge_current_density
        return spin_current
    
    def calculate_inverse_she_voltage(self, 
                                     spin_current_density: float,
                                     length: float,
                                     width: float) -> float:
        """
        Calculate voltage generated by inverse spin Hall effect.
        
        V_ISHE = θ_SH × (2e/ℏ) × (J_s / σ) × L
        """
        e = self.spin_hall_angle * spin_current_density * length
        voltage = e / (self.conductivity * width)
        
        return voltage
    
    def calculate_spin_accumulation(self, 
                                   charge_current_density: float,
                                   diffusion_length: float = 10e-9) -> float:
        """
        Calculate spin accumulation at interface.
        
        μ_s = θ_SH × λ_sf × e × ρ × J_c
        
        where λ_sf is the spin-flip length.
        """
        resistivity = 1.0 / self.conductivity
        
        spin_accumulation = (self.spin_hall_angle * 
                           diffusion_length * 
                           charge_current_density * 
                           resistivity)
        
        return spin_accumulation


class NonLocalSpinValve:
    """
    Non-local spin valve geometry for spin transport measurements.
    
    Used to measure spin diffusion lengths and spin Hall effects.
    """
    
    def __init__(self, 
                 spin_diffusion_length: float = 1e-6,  # m
                 spin_flip_time: float = 1e-9,  # s
                 conductivity: float = 1e7):  # S/m
        self.lambda_sf = spin_diffusion_length
        self.tau_sf = spin_flip_time
        self.sigma = conductivity
        
        # Spin diffusion constant
        self.D = self.lambda_sf**2 / self.tau_sf
    
    def calculate_nonlocal_resistance(self, 
                                     distance: float,
                                     injector_width: float = 100e-9,
                                     detector_width: float = 100e-9) -> float:
        """
        Calculate non-local resistance in spin valve geometry.
        
        R_NL = (P² × R_N) × exp(-L/λ_sf)
        
        where P is spin polarization and R_N is normal resistance.
        """
        # Assume spin polarization
        P = 0.5
        
        # Resistance of channel
        R_N = 1.0 / (self.sigma * injector_width * detector_width)
        
        # Non-local resistance
        R_NL = P**2 * R_N * np.exp(-distance / self.lambda_sf)
        
        return R_NL
    
    def calculate_spin_signal(self, distances: np.ndarray) -> np.ndarray:
        """
        Calculate spin signal as function of injector-detector distance.
        """
        signals = []
        
        for L in distances:
            R_NL = self.calculate_nonlocal_resistance(L)
            signals.append(R_NL)
        
        return np.array(signals)
    
    def extract_spin_diffusion_length(self,
                                     distances: np.ndarray,
                                     signals: np.ndarray) -> float:
        """
        Extract spin diffusion length by fitting exponential decay.
        """
        # Fit log(signals) vs distance to linear function
        log_signals = np.log(signals)
        
        # Linear fit: log(R_NL) = log(R_0) - L/λ_sf
        coeffs = np.polyfit(distances, log_signals, 1)
        
        # λ_sf = -1 / slope
        lambda_sf = -1.0 / coeffs[0]
        
        return lambda_sf


class SpinOrbitTorque:
    """
    Spin-orbit torque calculations for heavy metal/ferromagnet bilayers.
    
    Includes both damping-like and field-like torque contributions.
    """
    
    def __init__(self, 
                 heavy_metal: str = "Pt",
                 ferromagnet: MagneticLayer = None):
        self.heavy_metal = heavy_metal
        self.ferromagnet = ferromagnet or MagneticLayer(
            name="CoFeB",
            thickness=1.0,
            magnetization=np.array([0, 0, 1]),
            exchange_stiffness=20.0,
            damping=0.01,
            saturation_magnetization=1.2
        )
        
        self.spin_hall = SpinHallEffect(heavy_metal)
    
    def calculate_damping_like_torque(self,
                                     charge_current: float,
                                     layer_thickness: float = 5e-9) -> np.ndarray:
        """
        Calculate damping-like torque from spin Hall effect.
        
        τ_DL ∝ m × (m × σ)
        
        where σ is the spin polarization direction.
        """
        # Spin current generated by SHE
        J_s = self.spin_hall.calculate_spin_current(charge_current, layer_thickness)
        
        # Spin polarization perpendicular to plane
        sigma = np.array([0, 1, 0])
        
        m = self.ferromagnet.magnetization
        
        # Damping-like torque direction
        torque_dir = np.cross(m, np.cross(m, sigma))
        
        # Torque magnitude depends on spin current
        torque_magnitude = J_s * layer_thickness
        
        return torque_magnitude * torque_dir
    
    def calculate_field_like_torque(self,
                                   charge_current: float,
                                   rashba_coefficient: float = 0.1) -> np.ndarray:
        """
        Calculate field-like torque from Rashba-Edelstein effect.
        
        τ_FL ∝ m × E_Rashba
        """
        # Rashba field perpendicular to current
        E_rashba = np.array([0, 0, 1]) * rashba_coefficient * charge_current
        
        m = self.ferromagnet.magnetization
        
        torque = np.cross(m, E_rashba)
        
        return torque
    
    def calculate_switching_current(self,
                                   external_field: np.ndarray,
                                   anisotropy_field: float = 0.1) -> float:
        """
        Estimate critical current for magnetization switching.
        """
        # Simplified critical current estimate
        # Actual calculation requires solving LLG
        
        H_k = anisotropy_field  # Anisotropy field in T
        H_ext = np.linalg.norm(external_field)
        
        # Critical current proportional to effective field
        J_c = (H_k + H_ext) / (self.spin_hall.spin_hall_angle * 1e10)
        
        return J_c


def example_mtj_simulation():
    """
    Example: Magnetic Tunnel Junction simulation.
    """
    print("=" * 60)
    print("Example: Magnetic Tunnel Junction")
    print("=" * 60)
    
    # Create magnetic layers
    free_layer = MagneticLayer(
        name="Free_CoFeB",
        thickness=2.0,
        magnetization=np.array([1, 0, 0]),  # Along x
        exchange_stiffness=20.0,
        damping=0.01,
        saturation_magnetization=1.2,
        anisotropy_constant=1e5
    )
    
    pinned_layer = MagneticLayer(
        name="Pinned_CoFeB",
        thickness=3.0,
        magnetization=np.array([0, 0, 1]),  # Along z (perpendicular to free)
        exchange_stiffness=20.0,
        damping=0.01,
        saturation_magnetization=1.2
    )
    
    # Tunnel barrier
    barrier = TunnelBarrier(
        material="MgO",
        thickness=1.0,
        height=2.0,
        effective_mass=0.4
    )
    
    # Create MTJ
    mtj = MagneticTunnelJunction(free_layer, pinned_layer, barrier)
    
    print(f"\nMTJ Configuration:")
    print(f"  Free layer thickness: {free_layer.thickness} nm")
    print(f"  Barrier thickness: {barrier.thickness} nm")
    print(f"  Barrier height: {barrier.height} eV")
    
    # Calculate TMR
    tmr = mtj.calculate_tmr_ratio()
    print(f"\nTMR ratio: {tmr*100:.1f}%")
    
    # Calculate angular dependence
    angles = np.linspace(0, np.pi, 50)
    resistances = [mtj.calculate_resistance(theta) for theta in angles]
    
    print(f"\nResistance:")
    print(f"  Parallel state: {resistances[0]:.3f} Ω")
    print(f"  Antiparallel state: {resistances[-1]:.3f} Ω")
    
    return mtj


def example_stt_dynamics():
    """
    Example: Spin transfer torque dynamics.
    """
    print("\n" + "=" * 60)
    print("Example: STT Magnetization Dynamics")
    print("=" * 60)
    
    # Magnetic layer
    layer = MagneticLayer(
        name="Free_Layer",
        thickness=2.0,
        magnetization=np.array([1, 0, 0]),
        exchange_stiffness=20.0,
        damping=0.01,
        saturation_magnetization=1.2
    )
    
    # Pinned layer magnetization (reference)
    m_pinned = np.array([0, 0, 1])
    
    # STT calculator
    stt = SpinTransferTorque(layer)
    
    # Calculate torque
    J = 1e10  # A/m²
    torque = stt.calculate_slonczewski_torque(J, m_pinned)
    
    print(f"\nSpin transfer torque at J = {J/1e10:.1f} × 10¹⁰ A/m²:")
    print(f"  Torque magnitude: {np.linalg.norm(torque):.3e}")
    print(f"  Torque direction: {torque / np.linalg.norm(torque)}")
    
    # Integrate LLG
    print("\nIntegrating LLG equation...")
    H_eff = np.array([0, 0, 0.1])  # 0.1 T along z
    
    trajectory = stt.integrate_llg(
        H_eff, J, m_pinned, 
        time_span=(0, 5e-9), 
        dt=1e-13
    )
    
    print(f"  Time steps: {len(trajectory)}")
    print(f"  Final magnetization: {trajectory[-1]}")
    print(f"  Switching achieved: {np.dot(trajectory[0], trajectory[-1]) < 0}")
    
    return stt, trajectory


def example_spin_hall():
    """
    Example: Spin Hall effect calculation.
    """
    print("\n" + "=" * 60)
    print("Example: Spin Hall Effect in Platinum")
    print("=" * 60)
    
    # Pt spin Hall effect
    she = SpinHallEffect("Pt")
    
    J_c = 1e10  # A/m²
    thickness = 5e-9  # m
    
    J_s = she.calculate_spin_current(J_c, thickness)
    
    print(f"\nSpin Hall effect in {she.material}:")
    print(f"  Spin Hall angle: {she.spin_hall_angle}")
    print(f"  Charge current: {J_c/1e10:.1f} × 10¹⁰ A/m²")
    print(f"  Generated spin current: {J_s/1e8:.2f} × 10⁸ A/m²")
    
    # Spin accumulation
    mu_s = she.calculate_spin_accumulation(J_c)
    print(f"  Spin accumulation: {mu_s*1e6:.2f} μeV")
    
    return she


if __name__ == "__main__":
    # Run examples
    mtj = example_mtj_simulation()
    stt, traj = example_stt_dynamics()
    she = example_spin_hall()
    
    print("\n" + "=" * 60)
    print("Spin Transport Module - Test Complete")
    print("=" * 60)
