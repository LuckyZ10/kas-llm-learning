"""
Electron-Phonon Coupling Calculator
====================================

Calculation of electron-phonon coupling properties.

Features:
- Electron-phonon coupling matrix elements
- Eliashberg spectral function α²F(ω)
- Electron-phonon coupling constant λ
- Mode-resolved coupling constants λ_ν(q)
- Temperature-dependent coupling

Author: DFTLammps Electron-Phonon Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Phonopy imports
try:
    from phonopy import Phonopy
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Physical constants
EV_TO_THZ = 241.799  # eV to THz
THZ_TO_EV = 1.0 / EV_TO_THZ
KB = 8.617e-5  # eV/K


@dataclass
class ElPhConfig:
    """Configuration for electron-phonon calculations."""
    
    # Energy grids
    energy_window: Tuple[float, float] = (-1.0, 1.0)  # eV around Fermi
    n_energy: int = 100
    
    # Smearing
    smearing: float = 0.05  # eV
    smearing_type: str = 'gaussian'
    
    # Temperature
    temperature: float = 300.0  # K
    
    # Output
    output_dir: str = "./elph_output"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ElPhResults:
    """Results from electron-phonon calculation."""
    
    # Coupling constants
    lambda_total: Optional[float] = None
    lambda_q: Optional[np.ndarray] = None  # q-dependent
    lambda_nu_q: Optional[np.ndarray] = None  # Mode-resolved
    
    # Spectral function
    omega: Optional[np.ndarray] = None  # meV
    a2f: Optional[np.ndarray] = None
    
    # Superconducting properties
    omega_log: Optional[float] = None  # meV
    tc_mcmillan: Optional[float] = None  # K
    tc_allen_dynes: Optional[float] = None  # K
    
    # Matrix elements
    g_kq_nu: Optional[np.ndarray] = None  # e-ph matrix elements
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'lambda_total': self.lambda_total,
            'omega_log_meV': self.omega_log,
            'tc_mcmillan_K': self.tc_mcmillan,
            'tc_allen_dynes_K': self.tc_allen_dynes
        }
    
    def save(self, filepath: str):
        """Save results."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ElectronPhononCalculator:
    """
    Calculator for electron-phonon coupling properties.
    
    Calculates:
    - Electron-phonon matrix elements g(k,q,ν)
    - Eliashberg spectral function α²F(ω)
    - Coupling constant λ
    - Superconducting properties
    """
    
    def __init__(self, config: Optional[ElPhConfig] = None):
        """
        Initialize e-ph calculator.
        
        Args:
            config: Configuration
        """
        self.config = config or ElPhConfig()
        self.results: Optional[ElPhResults] = None
        
        logger.info("Initialized ElectronPhononCalculator")
    
    def calculate_elph_matrix_elements(
        self,
        k_points: np.ndarray,
        q_points: np.ndarray,
        phonopy: Phonopy,
        dVscf: np.ndarray  # Change in self-consistent potential
    ) -> np.ndarray:
        """
        Calculate electron-phonon matrix elements.
        
        g(k,q,ν) = ⟨ψ_k+q|δV_q,ν·Q_ν|ψ_k⟩
        
        Args:
            k_points: k-points in reduced coordinates (N_k, 3)
            q_points: q-points in reduced coordinates (N_q, 3)
            phonopy: Phonopy object
            dVscf: Self-consistent potential derivative (N_q, n_modes, N_R, n_G)
            
        Returns:
            Matrix elements g(k,q,ν) with shape (N_k, N_q, n_modes)
        """
        if not PHONOPY_AVAILABLE:
            raise ImportError("Phonopy required")
        
        n_k = len(k_points)
        n_q = len(q_points)
        
        # Get phonon modes at q-points
        phonopy.run_qpoints(q_points)
        frequencies = phonopy.qpoints.frequencies  # THz
        eigenvectors = phonopy.qpoints.eigenvectors
        
        n_modes = frequencies.shape[1]
        
        # Calculate matrix elements (simplified)
        # Real calculation requires electronic wavefunctions
        g_kq_nu = np.zeros((n_k, n_q, n_modes), dtype=complex)
        
        for ik, k in enumerate(k_points):
            for iq, q in enumerate(q_points):
                for nu in range(n_modes):
                    # Simplified: matrix element proportional to deformation potential
                    # Real implementation needs actual electronic states
                    freq = frequencies[iq, nu]
                    if freq > 0.01:  # Skip acoustic modes
                        g_kq_nu[ik, iq, nu] = np.sqrt(1.0 / (2 * freq))  # Simplified
        
        return g_kq_nu
    
    def calculate_eliashberg_function(
        self,
        frequencies: np.ndarray,  # Phonon frequencies (THz)
        g_kq_nu: np.ndarray,  # Matrix elements
        energies_k: np.ndarray,  # Electronic energies
        energies_kq: np.ndarray,
        smearing: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Eliashberg spectral function α²F(ω).
        
        α²F(ω) = N(0)⁻¹ Σ_k,q,ν |g(k,q,ν)|² δ(ε_k) δ(ε_k+q) δ(ω - ω_qν)
        
        Args:
            frequencies: Phonon frequencies (THz)
            g_kq_nu: Matrix elements
            energies_k: Electronic energies at k (eV)
            energies_kq: Electronic energies at k+q (eV)
            smearing: Energy smearing (eV)
            
        Returns:
            Tuple of (omega_grid, alpha2f)
        """
        smearing = smearing or self.config.smearing
        
        # Create frequency grid
        omega_min = max(0.1, frequencies.min() * 0.5)
        omega_max = frequencies.max() * 1.5
        omega_grid = np.linspace(omega_min, omega_max, 200)
        
        alpha2f = np.zeros(len(omega_grid))
        
        # Fermi surface average
        n_k, n_q, n_modes = g_kq_nu.shape
        
        for ik in range(n_k):
            for iq in range(n_q):
                for nu in range(n_modes):
                    # Energy conservation
                    delta_ek = self._gaussian(energies_k[ik], 0, smearing)
                    delta_ekq = self._gaussian(energies_kq[ik, iq], 0, smearing)
                    
                    # Phonon frequency delta
                    omega_nu = frequencies[iq, nu]
                    g2 = np.abs(g_kq_nu[ik, iq, nu])**2
                    
                    for i_omega, omega in enumerate(omega_grid):
                        delta_ph = self._gaussian(omega, omega_nu, smearing * EV_TO_THZ)
                        alpha2f[i_omega] += g2 * delta_ek * delta_ekq * delta_ph
        
        # Normalize
        norm = integrate.simpson(alpha2f, omega_grid)
        if norm > 0:
            alpha2f /= norm
        
        return omega_grid, alpha2f
    
    def _gaussian(self, x: float, x0: float, sigma: float) -> float:
        """Gaussian function."""
        return np.exp(-((x - x0) / sigma)**2 / 2) / (sigma * np.sqrt(2 * np.pi))
    
    def calculate_lambda(
        self,
        omega: np.ndarray,  # meV
        a2f: np.ndarray,
        omega_max: Optional[float] = None
    ) -> float:
        """
        Calculate electron-phonon coupling constant λ.
        
        λ = 2∫₀^∞ α²F(ω)/ω dω
        
        Args:
            omega: Frequency grid (meV)
            a2f: Eliashberg function
            omega_max: Maximum frequency for integration
            
        Returns:
            Coupling constant λ
        """
        if omega_max is not None:
            mask = omega <= omega_max
            omega = omega[mask]
            a2f = a2f[mask]
        
        # Avoid division by zero
        omega_safe = np.where(omega > 0.1, omega, 0.1)
        
        integrand = 2 * a2f / omega_safe
        lambda_eph = integrate.simpson(integrand, omega)
        
        return lambda_eph
    
    def calculate_omega_log(
        self,
        omega: np.ndarray,
        a2f: np.ndarray,
        lambda_eph: float
    ) -> float:
        """
        Calculate logarithmic average frequency ω_log.
        
        ω_log = exp[(2/λ) ∫ ln(ω) α²F(ω)/ω dω]
        
        Args:
            omega: Frequency grid (meV)
            a2f: Eliashberg function
            lambda_eph: Coupling constant
            
        Returns:
            Logarithmic average frequency (meV)
        """
        omega_safe = np.where(omega > 0.1, omega, 0.1)
        
        integrand = (2 / lambda_eph) * a2f * np.log(omega_safe) / omega_safe
        
        # Exclude negative contributions
        mask = omega > 0
        omega_log = np.exp(integrate.simpson(integrand[mask], omega[mask]))
        
        return omega_log
    
    def calculate_tc_mcmillan(
        self,
        omega_log: float,  # meV
        lambda_eph: float,
        mu_star: float = 0.1
    ) -> float:
        """
        Calculate superconducting Tc using McMillan formula.
        
        Tc = (ω_log / 1.2) * exp[-1.04(1 + λ) / (λ - μ*(1 + 0.62λ))]
        
        Args:
            omega_log: Logarithmic average frequency (meV)
            lambda_eph: Coupling constant
            mu_star: Coulomb pseudopotential
            
        Returns:
            Critical temperature (K)
        """
        if lambda_eph <= mu_star * (1 + 0.62 * lambda_eph):
            return 0.0
        
        # Convert ω_log from meV to K (1 meV = 11.605 K)
        omega_log_k = omega_log * 11.605
        
        numerator = -1.04 * (1 + lambda_eph)
        denominator = lambda_eph - mu_star * (1 + 0.62 * lambda_eph)
        
        tc = (omega_log_k / 1.2) * np.exp(numerator / denominator)
        
        return tc
    
    def calculate_tc_allen_dynes(
        self,
        omega_log: float,
        lambda_eph: float,
        mu_star: float = 0.1
    ) -> float:
        """
        Calculate Tc using Allen-Dynes formula.
        
        More accurate than McMillan for strong coupling (λ > 1).
        
        Args:
            omega_log: Logarithmic average frequency (meV)
            lambda_eph: Coupling constant
            mu_star: Coulomb pseudopotential
            
        Returns:
            Critical temperature (K)
        """
        omega_log_k = omega_log * 11.605
        
        # Allen-Dynes correction factor
        f = lambda_eph * (1 + (lambda_eph / (2.46 * (1 + 3.8 * mu_star))**1.5))
        
        # Modified McMillan
        numerator = -1.04 * (1 + lambda_eph * (1 - 0.62 * mu_star - 0.62 * lambda_eph))
        denominator = lambda_eph - mu_star * (1 + 0.62 * lambda_eph)
        
        if denominator <= 0:
            return 0.0
        
        tc = (omega_log_k / 1.2) * np.exp(numerator / denominator) * f
        
        return tc
    
    def calculate_mode_lambda(
        self,
        frequencies: np.ndarray,  # (N_q, N_modes) in THz
        g_kq_nu: np.ndarray,  # (N_k, N_q, N_modes)
        energies_k: np.ndarray,
        energies_kq: np.ndarray
    ) -> np.ndarray:
        """
        Calculate mode-resolved coupling constants λ_ν(q).
        
        Args:
            frequencies: Phonon frequencies (THz)
            g_kq_nu: Matrix elements
            energies_k: Electronic energies
            energies_kq: Electronic energies at k+q
            
        Returns:
            λ_ν(q) array (N_q, N_modes)
        """
        n_q, n_modes = frequencies.shape
        lambda_nu_q = np.zeros((n_q, n_modes))
        
        for iq in range(n_q):
            for nu in range(n_modes):
                omega = frequencies[iq, nu]
                if omega < 0.01:
                    continue
                
                # Sum over k
                lambda_sum = 0.0
                for ik in range(len(energies_k)):
                    g2 = np.abs(g_kq_nu[ik, iq, nu])**2
                    delta_ek = self._gaussian(energies_k[ik], 0, self.config.smearing)
                    delta_ekq = self._gaussian(energies_kq[ik, iq], 0, self.config.smearing)
                    lambda_sum += g2 * delta_ek * delta_ekq
                
                lambda_nu_q[iq, nu] = 2 * lambda_sum / omega
        
        return lambda_nu_q
    
    def run_full_calculation(
        self,
        frequencies: np.ndarray,  # THz
        g_kq_nu: np.ndarray,
        energies_k: np.ndarray,
        energies_kq: np.ndarray,
        mu_star: float = 0.1
    ) -> ElPhResults:
        """
        Run full e-ph calculation.
        
        Args:
            frequencies: Phonon frequencies (THz)
            g_kq_nu: Matrix elements
            energies_k: Electronic energies (eV)
            energies_kq: Electronic energies at k+q (eV)
            mu_star: Coulomb pseudopotential
            
        Returns:
            ElPhResults object
        """
        # Calculate Eliashberg function
        omega, a2f = self.calculate_eliashberg_function(
            frequencies, g_kq_nu, energies_k, energies_kq
        )
        
        # Calculate λ
        lambda_eph = self.calculate_lambda(omega, a2f)
        
        # Calculate ω_log
        omega_log = self.calculate_omega_log(omega, a2f, lambda_eph)
        
        # Calculate Tc
        tc_mc = self.calculate_tc_mcmillan(omega_log, lambda_eph, mu_star)
        tc_ad = self.calculate_tc_allen_dynes(omega_log, lambda_eph, mu_star)
        
        # Calculate mode-resolved λ
        lambda_nu_q = self.calculate_mode_lambda(
            frequencies, g_kq_nu, energies_k, energies_kq
        )
        
        self.results = ElPhResults(
            lambda_total=lambda_eph,
            lambda_nu_q=lambda_nu_q,
            omega=omega,
            a2f=a2f,
            omega_log=omega_log,
            tc_mcmillan=tc_mc,
            tc_allen_dynes=tc_ad,
            g_kq_nu=g_kq_nu
        )
        
        logger.info(f"Calculated λ = {lambda_eph:.3f}, ω_log = {omega_log:.2f} meV")
        logger.info(f"Tc (McMillan) = {tc_mc:.2f} K, Tc (Allen-Dynes) = {tc_ad:.2f} K")
        
        return self.results
    
    def plot_eliashberg_function(
        self,
        results: Optional[ElPhResults] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Figure:
        """Plot Eliashberg spectral function."""
        results = results or self.results
        if results is None:
            raise ValueError("No results available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(results.omega, results.a2f, 'b-', lw=2)
        ax.fill_between(results.omega, 0, results.a2f, alpha=0.3)
        
        ax.set_xlabel('ℏω (meV)', fontsize=12)
        ax.set_ylabel('α²F(ω)', fontsize=12)
        ax.set_title(f'Eliashberg Function (λ = {results.lambda_total:.3f})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        if results.omega_log:
            ax.axvline(x=results.omega_log, color='r', linestyle='--', 
                      label=f'ω_log = {results.omega_log:.1f} meV')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved α²F plot to {save_path}")
        
        return fig
    
    def plot_lambda_vs_temperature(
        self,
        temperatures: np.ndarray,
        omega: np.ndarray,
        a2f: np.ndarray,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot temperature-dependent λ.
        
        Note: This is a simplified calculation assuming
        temperature dependence comes mainly from Fermi occupation.
        """
        lambda_t = np.zeros(len(temperatures))
        
        for i, T in enumerate(temperatures):
            # Temperature correction (simplified)
            correction = 1.0 + (T / 300.0)**2 * 0.1
            lambda_t[i] = self.calculate_lambda(omega, a2f) / correction
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(temperatures, lambda_t, 'b-', lw=2)
        
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('λ(T)', fontsize=12)
        ax.set_title('Temperature-Dependent Electron-Phonon Coupling', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == '__main__':
    print("Electron-Phonon Calculator - use within Python for full functionality")
