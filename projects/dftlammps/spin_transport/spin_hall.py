#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spin Hall Effect and Spin-Orbit Coupling Module
==============================================

This module provides detailed calculations for:
- Intrinsic and extrinsic spin Hall effect
- Rashba spin-orbit coupling
- Dresselhaus spin-orbit coupling
- Spin-orbit torques (SOT)
- Edelstein effect

References:
-----------
[1] Sinova, J., et al. (2015). Rev. Mod. Phys., 87, 1213.
[2] Manchon, A., et al. (2019). Nature Mater., 18, 1196.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass

# Constants
HBAR = 6.62607015e-34 / (2 * np.pi)
Q_E = 1.602176634e-19
K_B = 1.380649e-23


@dataclass
class SpinOrbitParameters:
    """Parameters for spin-orbit coupling calculations."""
    # Rashba SOC
    alpha_rashba: float = 0.0  # eV·Å
    
    # Dresselhaus SOC
    beta_dresselhaus: float = 0.0  # eV·Å
    
    # Spin Hall
    spin_hall_angle: float = 0.1
    spin_hall_conductivity: float = 100.0  # (ℏ/e) Ω⁻¹cm⁻¹
    
    # Material parameters
    effective_mass: float = 0.1  # m_e
    relaxation_time: float = 1e-14  # s
    

def calculate_rashba_hamiltonian(kx: float, ky: float, 
                                 alpha: float,
                                 g_factor: float = 2.0) -> np.ndarray:
    """
    Calculate Rashba Hamiltonian.
    
    H_R = α (k_y σ_x - k_x σ_y)
    
    Parameters:
    -----------
    kx, ky : float
        Wavevector components [Å⁻¹]
    alpha : float
        Rashba parameter [eV·Å]
    g_factor : float
        Electron g-factor
        
    Returns:
    --------
    np.ndarray
        2×2 Hamiltonian matrix
    """
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Rashba term
    H_R = alpha * (ky * sigma_x - kx * sigma_y)
    
    return H_R


def calculate_dresselhaus_hamiltonian(kx: float, ky: float,
                                      beta: float) -> np.ndarray:
    """
    Calculate Dresselhaus Hamiltonian.
    
    H_D = β (k_x σ_x - k_y σ_y)  [001] direction
    
    Parameters:
    -----------
    kx, ky : float
        Wavevector components [Å⁻¹]
    beta : float
        Dresselhaus parameter [eV·Å]
        
    Returns:
    --------
    np.ndarray
        2×2 Hamiltonian matrix
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    H_D = beta * (kx * sigma_x - ky * sigma_y)
    
    return H_D


def calculate_spin_splitting(k: np.ndarray, alpha: float, 
                             beta: float = 0.0) -> np.ndarray:
    """
    Calculate spin-split band structure.
    
    Parameters:
    -----------
    k : np.ndarray
        Wavevectors [Å⁻¹], shape (N, 2) for (kx, ky)
    alpha : float
        Rashba parameter
    beta : float
        Dresselhaus parameter
        
    Returns:
    --------
    np.ndarray
        Spin-split energies, shape (N, 2)
    """
    k = np.atleast_2d(k)
    n_k = len(k)
    
    energies = np.zeros((n_k, 2))
    
    for i, (kx, ky) in enumerate(k):
        # Kinetic energy
        E_k = 3.81 * (kx**2 + ky**2)  # eV (for m* = 0.1 m_e)
        
        # SOC contribution
        H_soc = calculate_rashba_hamiltonian(kx, ky, alpha)
        if beta > 0:
            H_soc += calculate_dresselhaus_hamiltonian(kx, ky, beta)
        
        # Total energy
        eigenvalues = np.linalg.eigvalsh(H_soc)
        energies[i] = E_k + eigenvalues
    
    return energies


def calculate_spin_texture(k_grid: np.ndarray, alpha: float,
                          beta: float = 0.0) -> Dict:
    """
    Calculate spin texture in k-space.
    
    For Rashba SOC, spins rotate in-plane perpendicular to k.
    
    Parameters:
    -----------
    k_grid : np.ndarray
        k-point grid, shape (N, 2)
    alpha : float
        Rashba parameter
    beta : float
        Dresselhaus parameter
        
    Returns:
    --------
    Dict
        Spin texture data
    """
    n_k = len(k_grid)
    
    S_x = np.zeros((n_k, 2))  # Two bands
    S_y = np.zeros((n_k, 2))
    S_z = np.zeros((n_k, 2))
    
    for i, (kx, ky) in enumerate(k_grid):
        H = calculate_rashba_hamiltonian(kx, ky, alpha)
        if beta > 0:
            H += calculate_dresselhaus_hamiltonian(kx, ky, beta)
        
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Calculate expectation values of spin
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        for band in range(2):
            psi = eigenvectors[:, band]
            S_x[i, band] = np.real(np.conj(psi) @ sigma_x @ psi)
            S_y[i, band] = np.real(np.conj(psi) @ sigma_y @ psi)
            S_z[i, band] = np.real(np.conj(psi) @ sigma_z @ psi)
    
    return {
        'k_points': k_grid,
        'S_x': S_x,
        'S_y': S_y,
        'S_z': S_z
    }


def calculate_intrinsic_spin_hall_conductivity(
    berry_curvature: np.ndarray,
    occupation: np.ndarray,
    k_weights: np.ndarray) -> float:
    """
    Calculate intrinsic spin Hall conductivity from Berry curvature.
    
    σ_xy^z = -(e²/ℏ) ∑_n ∫ (d³k/(2π)³) f_n(k) Ω_n,xy(k)
    
    Parameters:
    -----------
    berry_curvature : np.ndarray
        Berry curvature Ω_n,xy [Å²]
    occupation : np.ndarray
        Occupation numbers f_n(k)
    k_weights : np.ndarray
        k-point weights
        
    Returns:
    --------
    float
        Spin Hall conductivity [(ℏ/e) Ω⁻¹cm⁻¹]
    """
    # Integrate Berry curvature
    spin_hall = -np.sum(berry_curvature * occupation * k_weights)
    
    # Convert units
    # σ_SH [ℏ/e] = σ_xy * (ℏ/e)
    spin_hall_conductivity = spin_hall * 1e4  # Convert to practical units
    
    return spin_hall_conductivity


def calculate_extrinsic_spin_hall(skew_scattering_rate: float,
                                  side_jump_rate: float,
                                  charge_conductivity: float) -> float:
    """
    Calculate extrinsic spin Hall effect contributions.
    
    Parameters:
    -----------
    skew_scattering_rate : float
        Skew scattering rate [s⁻¹]
    side_jump_rate : float
        Side jump rate [s⁻¹]
    charge_conductivity : float
        Charge conductivity [Ω⁻¹m⁻¹]
        
    Returns:
    --------
    float
        Spin Hall angle
    """
    # Skew scattering contribution
    theta_skew = skew_scattering_rate / (skew_scattering_rate + side_jump_rate)
    
    # Side jump contribution
    theta_side = side_jump_rate / (skew_scattering_rate + side_jump_rate)
    
    # Total extrinsic SHE
    theta_sh = theta_skew + theta_side
    
    return theta_sh


def calculate_edelstein_effect(electric_field: float,
                               alpha_rashba: float,
                               density_of_states: float,
                               relaxation_time: float) -> float:
    """
    Calculate Edelstein effect (inverse of Rashba-Edelstein effect).
    
    Generates spin accumulation from electric field in 2D Rashba systems.
    
    Parameters:
    -----------
    electric_field : float
        Applied electric field [V/m]
    alpha_rashba : float
        Rashba parameter [eV·Å]
    density_of_states : float
        DOS at Fermi level [states/eV/area]
    relaxation_time : float
        Spin relaxation time [s]
        
    Returns:
    --------
    float
        Spin accumulation (nonequilibrium spin density) [m⁻²]
    """
    # Edelstein effect
    # δs = (e τ α / ℏ) E × ẑ
    
    spin_accumulation = (Q_E * relaxation_time * alpha_rashba * 1e-10 / HBAR) * electric_field
    
    return spin_accumulation


def calculate_spin_orbit_torque_efficiency(
    alpha_rashba: float,
    fermi_velocity: float,
    spin_lifetime: float,
    torque_type: str = "damping") -> float:
    """
    Calculate spin-orbit torque efficiency.
    
    Parameters:
    -----------
    alpha_rashba : float
        Rashba parameter [eV·Å]
    fermi_velocity : float
        Fermi velocity [m/s]
    spin_lifetime : float
        Spin lifetime [s]
    torque_type : str
        "damping" or "field"
        
    Returns:
    --------
    float
        Torque efficiency [ℏ/(2e)]
    """
    # Rashba wavevector
    k_R = alpha_rashba * 1e-10 / HBAR / fermi_velocity
    
    # Spin precession length
    lambda_soc = 2 * np.pi / k_R if k_R > 0 else np.inf
    
    # Spin diffusion length
    l_sf = np.sqrt(fermi_velocity**2 * spin_lifetime * 1e-14)
    
    if torque_type == "damping":
        # Damping-like torque efficiency
        efficiency = k_R * l_sf / (1 + (k_R * l_sf)**2)
    else:
        # Field-like torque efficiency
        efficiency = (k_R * l_sf)**2 / (1 + (k_R * l_sf)**2)
    
    return efficiency


def example_rashba_spin_splitting():
    """Example: Rashba spin splitting."""
    print("=" * 70)
    print("Spin-Orbit Example: Rashba Spin Splitting")
    print("=" * 70)
    
    # Parameters for Ag(111) surface state
    alpha = 0.1  # eV·Å
    
    # Create k-grid
    k = np.linspace(-0.5, 0.5, 100)  # Å⁻¹
    k_grid = np.array([[kx, 0] for kx in k])
    
    # Calculate spin splitting
    energies = calculate_spin_splitting(k_grid, alpha)
    
    print(f"\nRashba parameter: α = {alpha} eV·Å")
    print(f"k_F ≈ {alpha/3.81:.3f} Å⁻¹")
    print(f"Maximum splitting: {2*alpha*np.max(np.abs(k)):.3f} eV")
    
    # Rashba energy
    E_R = alpha**2 / (2 * 3.81)  # Using effective mass approximation
    print(f"Rashba energy: E_R = {E_R:.4f} eV")
    
    return k, energies


def example_spin_texture():
    """Example: Spin texture in k-space."""
    print("\n" + "=" * 70)
    print("Spin-Orbit Example: Spin Texture")
    print("=" * 70)
    
    alpha = 0.1
    
    # Circular k-grid
    theta = np.linspace(0, 2*np.pi, 36)
    k_f = 0.2  # Fermi wavevector
    k_grid = np.array([[k_f * np.cos(t), k_f * np.sin(t)] for t in theta])
    
    # Calculate spin texture
    texture = calculate_spin_texture(k_grid, alpha)
    
    print(f"\nRashba spin texture:")
    print(f"  S_y/S_x ≈ -k_x/k_y (perpendicular to k)")
    print(f"  S_z ≈ 0 (in-plane spins)")
    
    # Check orthogonality
    kx, ky = k_grid[5]
    Sx = texture['S_x'][5, 0]
    Sy = texture['S_y'][5, 0]
    dot_product = kx * Sx + ky * Sy
    print(f"\nOrthogonality check (k·S): {dot_product:.6f}")
    
    return texture


def example_spin_hall_angles():
    """Example: Spin Hall angles in different materials."""
    print("\n" + "=" * 70)
    print("Spin-Orbit Example: Spin Hall Angles")
    print("=" * 70)
    
    materials = {
        'Pt': {'theta_sh': 0.1, 'source': 'intrinsic'},
        'Au': {'theta_sh': 0.02, 'source': 'extrinsic'},
        'Ta': {'theta_sh': -0.15, 'source': 'intrinsic'},
        'W': {'theta_sh': -0.3, 'source': 'intrinsic'},
        'CuBi': {'theta_sh': 0.003, 'source': 'extrinsic'}
    }
    
    print("\nSpin Hall angles:")
    for material, data in materials.items():
        print(f"  {material:6s}: θ_SH = {data['theta_sh']:+6.3f} ({data['source']})")
    
    # Calculate spin Hall conductivity for Pt
    theta_sh = 0.1
    sigma_c = 1e6  # S/m
    
    # σ_SH = θ_SH × σ_c × (ℏ/2e)
    sigma_sh = theta_sh * sigma_c * HBAR / (2 * Q_E)
    print(f"\nPt spin Hall conductivity: {sigma_sh*1e4:.1f} (ℏ/e) Ω⁻¹cm⁻¹")
    
    return materials


def example_edelstein_effect():
    """Example: Edelstein effect in 2DEG."""
    print("\n" + "=" * 70)
    print("Spin-Orbit Example: Edelstein Effect")
    print("=" * 70)
    
    # 2DEG parameters
    alpha = 0.1  # eV·Å
    E_field = 1e5  # V/m (0.1 V across 1 μm)
    tau_s = 1e-12  # s
    
    spin_acc = calculate_edelstein_effect(E_field, alpha, 1e14, tau_s)
    
    print(f"\nElectric field: E = {E_field*1e-5:.1f} × 10⁵ V/m")
    print(f"Rashba parameter: α = {alpha} eV·Å")
    print(f"Spin relaxation time: τ_s = {tau_s*1e12:.1f} ps")
    print(f"\nSpin accumulation: δs = {spin_acc:.3e} m⁻²")
    print(f"                   = {spin_acc*1e-14:.3f} × 10¹⁴ m⁻²")
    
    # Corresponding magnetic field
    mu_B = 9.27e-24  # J/T
    B_eff = alpha * 1e-10 * E_field / mu_B
    print(f"\nEffective magnetic field: B_eff ≈ {B_eff:.2f} T")
    
    return spin_acc


if __name__ == "__main__":
    print("Running Spin-Orbit Coupling Examples\n")
    
    k, energies = example_rashba_spin_splitting()
    texture = example_spin_texture()
    materials = example_spin_hall_angles()
    spin_acc = example_edelstein_effect()
    
    print("\n" + "=" * 70)
    print("All spin-orbit examples completed!")
    print("=" * 70)
