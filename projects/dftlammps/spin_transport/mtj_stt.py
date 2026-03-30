#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spin Transport Module
=====================

This module provides comprehensive tools for spin-dependent transport
calculations including magnetic tunnel junctions (MTJ), spin transfer
torque (STT), and spin Hall effects.

References:
-----------
[1] Slonczewski, J.C. (1996). JMMM, 159, L1-L7.
[2] Berger, L. (1996). PRB, 54, 9353.
[3] Hirsch, J.E. (1999). PRL, 83, 1834.

Author: Spin Transport Team
Date: 2025
"""

import numpy as np
from numpy.linalg import inv, eigh, eigvals
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Physical constants
HBAR = 6.62607015e-34 / (2 * np.pi)
Q_E = 1.602176634e-19
K_B = 1.380649e-23
G_0 = 2 * Q_E**2 / (2 * np.pi * HBAR)
MU_B = 9.274009994e-24
GAMMA_LG = 1.76085963023e11
MU_0 = 4 * np.pi * 1e-7


class SpinPolarization(Enum):
    """Types of spin polarization."""
    PARALLEL = "parallel"
    ANTIPARALLEL = "antiparallel"


@dataclass
class MTJParameters:
    """Parameters for magnetic tunnel junction."""
    area: float = 100e-18
    barrier_thickness: float = 1.0e-9
    barrier_height: float = 1.0
    free_layer_thickness: float = 2.0e-9
    pinned_layer_thickness: float = 5.0e-9
    anisotropy_constant: float = 1e5
    anisotropy_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    polarization_pinned: float = 0.6
    polarization_free: float = 0.6
    resistance_area_product: float = 10.0
    tmr_ratio: float = 1.0
    temperature: float = 300.0

    def __post_init__(self):
        self.anisotropy_direction = self.anisotropy_direction / np.linalg.norm(self.anisotropy_direction)


@dataclass
class STTParameters:
    """Parameters for spin transfer torque calculations."""
    lambda_sf: float = 5e-9
    epsilon_prime: float = 0.1
    alpha_damping: float = 0.01
    gamma_lg: float = GAMMA_LG
    thermal_noise: bool = True
    temperature: float = 300.0
    cell_volume: float = 1e-25
    saturation_magnetization: float = 8e5


class MagneticTunnelJunction:
    """Magnetic Tunnel Junction (MTJ) calculator."""
    
    def __init__(self, params: MTJParameters):
        self.params = params
        self.magnetization_pinned = np.array([0.0, 0.0, 1.0])
        self.magnetization_free = np.array([0.0, 0.0, 1.0])
        
    def calculate_tunnel_resistance(self) -> Tuple[float, float]:
        """Calculate tunnel resistance for P and AP configurations."""
        P1 = self.params.polarization_pinned
        P2 = self.params.polarization_free
        
        RA = self.params.resistance_area_product
        area = self.params.area * 1e12
        
        R_P = RA / area
        tmr = 2 * P1 * P2 / (1 - P1 * P2)
        R_AP = R_P * (1 + tmr)
        
        return R_P, R_AP
    
    def calculate_resistance_vs_angle(self, angles: np.ndarray) -> np.ndarray:
        """Calculate resistance as function of relative angle."""
        R_P, R_AP = self.calculate_tunnel_resistance()
        tmr = (R_AP - R_P) / R_P
        return R_P * (1 + tmr/2 * (1 - np.cos(angles)))
    
    def calculate_current(self, voltage: float, 
                         configuration: SpinPolarization = SpinPolarization.PARALLEL) -> float:
        """Calculate tunnel current at given voltage."""
        R_P, R_AP = self.calculate_tunnel_resistance()
        
        if configuration == SpinPolarization.PARALLEL:
            R = R_P
        else:
            R = R_AP
        
        I = voltage / R
        
        # Non-linear correction
        if abs(voltage) > 0.1:
            barrier = self.params.barrier_height
            phi = barrier - abs(voltage) / 2
            if phi > 0:
                correction = 1 + (voltage**2 / (24 * phi**2))
                I *= correction
        
        return I
    
    def calculate_iv_curve(self, voltage_range: Tuple[float, float],
                          n_points: int = 100,
                          configuration: SpinPolarization = SpinPolarization.PARALLEL
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate I-V characteristic."""
        voltages = np.linspace(voltage_range[0], voltage_range[1], n_points)
        currents = np.array([self.calculate_current(V, configuration) for V in voltages])
        return voltages, currents


class SpinTransferTorque:
    """Spin Transfer Torque (STT) calculator."""
    
    def __init__(self, params: STTParameters, mtj_params: MTJParameters):
        self.params = params
        self.mtj = MagneticTunnelJunction(mtj_params)
        
    def calculate_slonczewski_torque(self, current_density: float, theta: float) -> Tuple[float, float]:
        """Calculate Slonczewski spin transfer torque."""
        P = self.mtj.params.polarization_pinned
        M_s = self.params.saturation_magnetization
        t = self.mtj.params.free_layer_thickness
        
        cos_theta = np.cos(theta)
        denominator = -4 + (1 + P)**3 * (3 + cos_theta) / (4 * P**1.5)
        g_theta = 1.0 / denominator if abs(denominator) > 1e-10 else 0
        
        prefactor = HBAR * current_density / (2 * Q_E * M_s * t)
        
        tau_parallel = prefactor * g_theta * np.sin(theta)
        tau_perp = prefactor * self.params.epsilon_prime * g_theta * np.sin(theta)
        
        return tau_parallel, tau_perp
    
    def calculate_critical_current(self, switching_type: str = "thermal") -> float:
        """Calculate critical current for magnetization switching."""
        M_s = self.params.saturation_magnetization
        t = self.mtj.params.free_layer_thickness
        V = self.mtj.params.area * t
        
        K_u = self.mtj.params.anisotropy_constant
        H_k = 2 * K_u / (M_s * MU_0)
        
        gamma = self.params.gamma_lg
        alpha = self.params.alpha_damping
        
        Delta = K_u * V / (K_B * self.params.temperature)
        P = self.mtj.params.polarization_pinned
        
        if switching_type == "thermal":
            g_0 = 1 / (-4 + (1+P)**3 / P**1.5)
            I_c = (2 * Q_E * alpha * M_s * V * H_k / (HBAR * abs(g_0))) * (1 + Delta**(-1))
        else:
            I_c = (2 * Q_E * alpha * M_s * V * H_k / (HBAR * P))
        
        return I_c / self.mtj.params.area


class SpinHallEffect:
    """Spin Hall effect calculator."""
    
    def __init__(self, spin_hall_angle: float = 0.1,
                 conductivity: float = 1e6,
                 spin_diffusion_length: float = 1e-9,
                 thickness: float = 5e-9):
        self.theta_sh = spin_hall_angle
        self.sigma = conductivity
        self.lambda_sf = spin_diffusion_length
        self.thickness = thickness
        
    def calculate_spin_current(self, charge_current_density: float) -> float:
        """Calculate spin current from charge current via SHE."""
        return self.theta_sh * (HBAR / (2 * Q_E)) * charge_current_density
    
    def calculate_spin_accumulation(self, charge_current_density: float, z: Optional[float] = None) -> float:
        """Calculate spin accumulation profile."""
        t = self.thickness
        l_sf = self.lambda_sf
        
        if z is None:
            z = t / 2
        
        prefactor = self.theta_sh * l_sf * charge_current_density / self.sigma
        
        if t / l_sf > 5:
            mu_s = prefactor * np.exp(-np.abs(z - t/2) / l_sf)
        else:
            mu_s = prefactor * np.sinh(z / l_sf) / np.cosh(t / (2 * l_sf))
        
        return mu_s


def example_mtj():
    """Example: MTJ characteristics."""
    print("=" * 70)
    print("Spin Transport Example: MTJ")
    print("=" * 70)
    
    params = MTJParameters(area=100e-18, barrier_thickness=1e-9,
                          polarization_pinned=0.6, polarization_free=0.6,
                          tmr_ratio=1.0)
    
    mtj = MagneticTunnelJunction(params)
    
    R_P, R_AP = mtj.calculate_tunnel_resistance()
    print(f"\nTunnel resistances:")
    print(f"  Parallel: {R_P:.2f} Ω")
    print(f"  Antiparallel: {R_AP:.2f} Ω")
    print(f"  TMR: {(R_AP/R_P - 1)*100:.1f}%")
    
    # I-V characteristics
    V, I_P = mtj.calculate_iv_curve((-0.5, 0.5), 50, SpinPolarization.PARALLEL)
    V, I_AP = mtj.calculate_iv_curve((-0.5, 0.5), 50, SpinPolarization.ANTIPARALLEL)
    
    print(f"\nI-V characteristics:")
    print(f"  Max current (P): {np.max(np.abs(I_P))*1e6:.2f} μA")
    print(f"  Max current (AP): {np.max(np.abs(I_AP))*1e6:.2f} μA")
    
    return mtj


def example_stt():
    """Example: Spin transfer torque."""
    print("\n" + "=" * 70)
    print("Spin Transport Example: STT")
    print("=" * 70)
    
    mtj_params = MTJParameters(area=100e-18, polarization_pinned=0.6,
                               free_layer_thickness=2e-9)
    stt_params = STTParameters(alpha_damping=0.01, saturation_magnetization=8e5)
    
    stt = SpinTransferTorque(stt_params, mtj_params)
    
    # Angular dependence
    angles = np.linspace(0, np.pi, 50)
    current_density = 1e11  # A/m²
    
    tau_par = []
    tau_perp = []
    for theta in angles:
        t_p, t_pp = stt.calculate_slonczewski_torque(current_density, theta)
        tau_par.append(t_p)
        tau_perp.append(t_pp)
    
    print(f"\nTorque at 90°:")
    print(f"  Parallel: {tau_par[25]:.4e} J/m³")
    print(f"  Perpendicular: {tau_perp[25]:.4e} J/m³")
    
    # Critical current
    J_c = stt.calculate_critical_current("thermal")
    print(f"\nCritical current density: {J_c*1e-10:.2f} × 10¹⁰ A/m²")
    
    return stt


def example_spin_hall():
    """Example: Spin Hall effect."""
    print("\n" + "=" * 70)
    print("Spin Transport Example: Spin Hall Effect")
    print("=" * 70)
    
    # Pt parameters
    she = SpinHallEffect(spin_hall_angle=0.1, conductivity=1e6,
                        spin_diffusion_length=1e-9, thickness=5e-9)
    
    j_c = 1e11  # A/m²
    j_s = she.calculate_spin_current(j_c)
    
    print(f"\nCharge current density: {j_c*1e-10:.2f} × 10¹⁰ A/m²")
    print(f"Spin current density: {j_s*1e-10:.4f} × 10¹⁰ A/m²")
    print(f"Spin Hall angle: {she.theta_sh}")
    
    # Spin accumulation
    mu_s = she.calculate_spin_accumulation(j_c)
    print(f"\nSpin accumulation: {mu_s*1e3:.3f} meV")
    
    return she


if __name__ == "__main__":
    print("Running Spin Transport Examples\n")
    
    mtj = example_mtj()
    stt = example_stt()
    she = example_spin_hall()
    
    print("\n" + "=" * 70)
    print("All spin transport examples completed!")
    print("=" * 70)
