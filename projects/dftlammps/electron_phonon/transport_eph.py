"""
Electron-Phonon Transport Calculator
=====================================

Calculation of electronic transport properties from electron-phonon coupling.

Features:
- Electrical resistivity ρ(T)
- Electron mobility μ(T)
- Seebeck coefficient S(T)
- Lorenz number L(T)
- Wiedemann-Franz law verification
- Temperature-dependent transport

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

logger = logging.getLogger(__name__)

# Physical constants
KB = 8.617333e-5  # eV/K
EV_TO_J = 1.602e-19
HBAR = 6.582e-16  # eV*s
ELECTRON_MASS = 9.109e-31  # kg
ELECTRON_CHARGE = 1.602e-19  # C


@dataclass
class TransportConfig:
    """Configuration for transport calculations."""
    
    # Temperature range
    t_min: float = 10.0  # K
    t_max: float = 1000.0  # K
    n_temps: int = 50
    
    # Doping concentration
    carrier_concentration: float = 1e21  # cm^-3
    carrier_type: str = 'electron'  # 'electron' or 'hole'
    
    # Material parameters
    effective_mass: float = 1.0  # m_e
    deformation_potential: float = 10.0  # eV
    elastic_constants: Optional[np.ndarray] = None
    
    # Output
    output_dir: str = "./transport_output"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class TransportResults:
    """Results from transport calculations."""
    
    temperatures: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Transport coefficients
    resistivity: Optional[np.ndarray] = None  # μΩ·cm
    conductivity: Optional[np.ndarray] = None  # S/cm
    mobility: Optional[np.ndarray] = None  # cm²/V/s
    seebeck: Optional[np.ndarray] = None  # μV/K
    lorenz_number: Optional[np.ndarray] = None  # W·Ω/K²
    thermal_conductivity_e: Optional[np.ndarray] = None  # W/m/K (electronic)
    
    # Relaxation times
    relaxation_time: Optional[np.ndarray] = None  # fs
    mean_free_path: Optional[np.ndarray] = None  # nm
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'temperatures': self.temperatures.tolist(),
            'resistivity_microohm_cm': self.resistivity.tolist() if self.resistivity is not None else None,
            'conductivity_S_cm': self.conductivity.tolist() if self.conductivity is not None else None,
            'mobility_cm2_V_s': self.mobility.tolist() if self.mobility is not None else None,
            'seebeck_microV_K': self.seebeck.tolist() if self.seebeck is not None else None
        }
    
    def save(self, filepath: str):
        """Save results."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ElectronPhononTransport:
    """
    Calculator for electron-phonon transport properties.
    
    Calculates:
    - Electrical resistivity from e-ph scattering
    - Electron mobility
    - Seebeck coefficient
    - Electronic thermal conductivity
    """
    
    def __init__(self, config: Optional[TransportConfig] = None):
        """
        Initialize transport calculator.
        
        Args:
            config: Transport configuration
        """
        self.config = config or TransportConfig()
        self.results: Optional[TransportResults] = None
        
        logger.info("Initialized ElectronPhononTransport")
    
    def calculate_scattering_rate_acoustic(
        self,
        temperature: float,
        effective_mass: float,
        deformation_potential: float,
        longitudinal_sound_velocity: float,
        density: float
    ) -> float:
        """
        Calculate acoustic phonon scattering rate (deformation potential).
        
        Using Bardeen-Shockley theory:
        τ⁻¹ = (D² m* k_B T) / (ħ² ρ v_s²)
        
        Args:
            temperature: Temperature (K)
            effective_mass: Effective mass (m_e)
            deformation_potential: Deformation potential (eV)
            longitudinal_sound_velocity: Sound velocity (m/s)
            density: Mass density (kg/m³)
            
        Returns:
            Scattering rate (s^-1)
        """
        m_star = effective_mass * ELECTRON_MASS  # kg
        D = deformation_potential * EV_TO_J  # J
        
        rate = (D**2 * m_star * KB * temperature * EV_TO_J) / \
               (HBAR**2 * EV_TO_J**2 * density * longitudinal_sound_velocity**2)
        
        return rate
    
    def calculate_scattering_rate_optical(
        self,
        temperature: float,
        optical_phonon_energy: float,
        coupling_constant: float
    ) -> float:
        """
        Calculate optical phonon scattering rate.
        
        Args:
            temperature: Temperature (K)
            optical_phonon_energy: Optical phonon energy (meV)
            coupling_constant: Dimensionless coupling constant
            
        Returns:
            Scattering rate (s^-1)
        """
        omega_opt = optical_phonon_energy * 1e-3 * EV_TO_J / HBAR  # Hz
        
        # Bose-Einstein occupation
        n_bose = 1.0 / (np.exp(optical_phonon_energy / (0.08617 * temperature)) - 1.0)
        
        # Scattering rate
        rate = coupling_constant * omega_opt * (n_bose + 0.5)
        
        return rate
    
    def calculate_mobility_acoustic(
        self,
        temperature: float,
        effective_mass: float,
        deformation_potential: float,
        sound_velocity: float,
        density: float
    ) -> float:
        """
        Calculate electron mobility limited by acoustic phonons.
        
        μ = (e τ) / m*
        
        Args:
            temperature: Temperature (K)
            effective_mass: Effective mass (m_e)
            deformation_potential: Deformation potential (eV)
            sound_velocity: Longitudinal sound velocity (m/s)
            density: Mass density (kg/m³)
            
        Returns:
            Mobility (cm²/V/s)
        """
        # Scattering rate
        rate = self.calculate_scattering_rate_acoustic(
            temperature, effective_mass, deformation_potential, 
            sound_velocity, density
        )
        
        # Relaxation time
        tau = 1.0 / rate  # s
        
        # Mobility
        m_star = effective_mass * ELECTRON_MASS
        mobility = ELECTRON_CHARGE * tau / m_star  # m²/V/s
        mobility_cm2 = mobility * 1e4  # cm²/V/s
        
        return mobility_cm2
    
    def calculate_resistivity_bloch_gruneisen(
        self,
        temperatures: np.ndarray,
        debye_temperature: float,
        resistivity_0: float,
        exponent: float = 5.0
    ) -> np.ndarray:
        """
        Calculate resistivity using Bloch-Grüneisen formula.
        
        ρ(T) = ρ₀ (T/Θ_D)^n ∫[0 to Θ_D/T] x^n/(e^x - 1)(1 - e^(-x)) dx
        
        For n=5: standard Bloch-Grüneisen for phonons
        
        Args:
            temperatures: Temperature array (K)
            debye_temperature: Debye temperature (K)
            resistivity_0: Residual resistivity (μΩ·cm)
            exponent: Exponent n (default 5)
            
        Returns:
            Resistivity array (μΩ·cm)
        """
        def integrand(x, n):
            if x < 1e-10:
                return x**(n-1)
            return x**n / ((np.exp(x) - 1) * (1 - np.exp(-x)))
        
        resistivity = np.zeros(len(temperatures))
        
        for i, T in enumerate(temperatures):
            if T == 0:
                resistivity[i] = resistivity_0
                continue
            
            # Integrate
            x_max = debye_temperature / T
            x_vals = np.linspace(0, x_max, 1000)
            integrals = [integrand(x, exponent) for x in x_vals]
            integral = integrate.simpson(integrals, x_vals)
            
            # Bloch-Grüneisen formula
            rho_T = (T / debye_temperature)**exponent * integral
            
            # Scale and add residual
            resistivity[i] = resistivity_0 * (1 + rho_T)
        
        return resistivity
    
    def calculate_resistivity_from_lambda(
        self,
        temperatures: np.ndarray,
        lambda_eph: float,
        omega_log: float,  # meV
        carrier_density: float,  # cm^-3
        effective_mass: float
    ) -> np.ndarray:
        """
        Calculate resistivity from electron-phonon coupling constant.
        
        Using Ziman formula:
        ρ = (m* v_F) / (n e² τ)
        
        Args:
            temperatures: Temperature array (K)
            lambda_eph: Coupling constant
            omega_log: Logarithmic average phonon frequency (meV)
            carrier_density: Carrier concentration (cm^-3)
            effective_mass: Effective mass (m_e)
            
        Returns:
            Resistivity array (μΩ·cm)
        """
        n = carrier_density * 1e6  # m^-3
        m_star = effective_mass * ELECTRON_MASS
        
        # Fermi velocity (simplified)
        v_f = HBAR * (3 * np.pi**2 * n)**(1/3) / m_star  # m/s
        
        resistivity = np.zeros(len(temperatures))
        
        for i, T in enumerate(temperatures):
            # Temperature-dependent scattering rate
            # τ⁻¹ ∝ λ (T/Θ_D) for T > Θ_D
            theta_D = omega_log * 11.605  # Convert meV to K
            
            if T < theta_D / 10:
                # Low-T: ρ ∝ T^5
                rate = lambda_eph * KB * T * EV_TO_J / (HBAR * EV_TO_J)
                rate *= (T / theta_D)**4
            elif T > theta_D:
                # High-T: ρ ∝ T
                rate = lambda_eph * KB * T * EV_TO_J / (HBAR * EV_TO_J)
            else:
                # Intermediate
                rate = lambda_eph * KB * T * EV_TO_J / (HBAR * EV_TO_J)
                rate *= (T / theta_D)**2
            
            # Resistivity
            rho = m_star * v_f * rate / (n * ELECTRON_CHARGE**2)  # Ω·m
            resistivity[i] = rho * 1e8  # μΩ·cm
        
        return resistivity
    
    def calculate_seebeck_coefficient(
        self,
        temperatures: np.ndarray,
        chemical_potential: float,  # eV
        effective_mass: float,
        carrier_density: float,
        scattering_exponent: float = 0.0
    ) -> np.ndarray:
        """
        Calculate Seebeck coefficient (thermopower).
        
        Using Mott formula for degenerate semiconductors:
        S = -(π²/3)(k_B/e)(k_B T / μ) [∂lnσ/∂lnE]_{E=μ}
        
        Args:
            temperatures: Temperature array (K)
            chemical_potential: Chemical potential / Fermi energy (eV)
            effective_mass: Effective mass (m_e)
            carrier_density: Carrier concentration (cm^-3)
            scattering_exponent: Exponent in energy-dependent scattering
            
        Returns:
            Seebeck coefficient array (μV/K)
        """
        seebeck = np.zeros(len(temperatures))
        
        for i, T in enumerate(temperatures):
            # Mott formula
            # S = (k_B/e) * (π²/3) * (k_B T / E_F) * (s + 3/2)
            # where s is scattering exponent
            
            prefactor = (np.pi**2 / 3) * (KB / ELECTRON_CHARGE)  # V/K
            energy_factor = KB * T / chemical_potential
            scattering_factor = scattering_exponent + 1.5
            
            S = prefactor * energy_factor * scattering_factor  # V/K
            seebeck[i] = S * 1e6  # μV/K
        
        return seebeck
    
    def calculate_electronic_thermal_conductivity(
        self,
        temperatures: np.ndarray,
        electrical_conductivity: np.ndarray,  # S/cm
        lorenz_number: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate electronic thermal conductivity using Wiedemann-Franz law.
        
        κ_e = L σ T
        
        where L is the Lorenz number (π²/3)(k_B/e)² ≈ 2.44e-8 WΩ/K² for metals
        
        Args:
            temperatures: Temperature array (K)
            electrical_conductivity: Electrical conductivity (S/cm)
            lorenz_number: Lorenz number (W·Ω/K²), uses Sommerfeld value if None
            
        Returns:
            Electronic thermal conductivity (W/m/K)
        """
        # Sommerfeld value
        L_0 = (np.pi**2 / 3) * (KB / ELECTRON_CHARGE)**2  # W·Ω/K²
        
        if lorenz_number is None:
            L = np.full(len(temperatures), L_0)
        else:
            L = lorenz_number
        
        sigma = electrical_conductivity * 100  # Convert S/cm to S/m
        
        kappa_e = L * sigma * temperatures  # W/m/K
        
        return kappa_e
    
    def calculate_lorenz_number(
        self,
        temperatures: np.ndarray,
        chemical_potential: float,
        scattering_exponent: float = 0.0
    ) -> np.ndarray:
        """
        Calculate temperature-dependent Lorenz number.
        
        For non-degenerate semiconductors, L deviates from Sommerfeld value.
        
        Args:
            temperatures: Temperature array (K)
            chemical_potential: Chemical potential (eV)
            scattering_exponent: Scattering exponent
            
        Returns:
            Lorenz number array (W·Ω/K²)
        """
        L_0 = (np.pi**2 / 3) * (KB / ELECTRON_CHARGE)**2
        
        lorenz = np.zeros(len(temperatures))
        
        for i, T in enumerate(temperatures):
            # Correction factor for non-degenerate statistics
            eta = chemical_potential / (KB * T)  # Reduced chemical potential
            
            if eta > 5:  # Degenerate limit
                lorenz[i] = L_0
            elif eta < -5:  # Non-degenerate limit
                lorenz[i] = 2 * L_0  # Different scattering mechanisms
            else:
                # Interpolation (simplified)
                f = 1 / (1 + np.exp(-eta))
                lorenz[i] = L_0 * (1 + f)
        
        return lorenz
    
    def run_full_transport_calculation(
        self,
        lambda_eph: float,
        omega_log: float,
        debye_temperature: float,
        effective_mass: float = 1.0,
        carrier_density: Optional[float] = None,
        elastic_constants: Optional[Dict] = None
    ) -> TransportResults:
        """
        Run complete transport calculation.
        
        Args:
            lambda_eph: Electron-phonon coupling constant
            omega_log: Logarithmic average phonon frequency (meV)
            debye_temperature: Debye temperature (K)
            effective_mass: Effective mass (m_e)
            carrier_density: Carrier concentration (cm^-3)
            elastic_constants: Dictionary with elastic constants
            
        Returns:
            TransportResults object
        """
        carrier_density = carrier_density or self.config.carrier_concentration
        
        temps = np.linspace(self.config.t_min, self.config.t_max, self.config.n_temps)
        
        # Calculate resistivity
        resistivity = self.calculate_resistivity_from_lambda(
            temps, lambda_eph, omega_log, carrier_density, effective_mass
        )
        
        # Conductivity
        conductivity = 1.0 / resistivity * 1e4  # S/cm (from μΩ·cm)
        
        # Mobility
        mobility = conductivity / (carrier_density * ELECTRON_CHARGE)  # cm²/V/s
        
        # Seebeck coefficient
        # Estimate chemical potential
        E_F = HBAR**2 * (3 * np.pi**2 * carrier_density * 1e6)**(2/3) / \
              (2 * effective_mass * ELECTRON_MASS) / EV_TO_J  # eV
        
        seebeck = self.calculate_seebeck_coefficient(
            temps, E_F, effective_mass, carrier_density
        )
        
        # Lorenz number
        lorenz = self.calculate_lorenz_number(temps, E_F)
        
        # Electronic thermal conductivity
        kappa_e = self.calculate_electronic_thermal_conductivity(
            temps, conductivity, lorenz
        )
        
        self.results = TransportResults(
            temperatures=temps,
            resistivity=resistivity,
            conductivity=conductivity,
            mobility=mobility,
            seebeck=seebeck,
            lorenz_number=lorenz,
            thermal_conductivity_e=kappa_e
        )
        
        logger.info(f"Completed transport calculation for {len(temps)} temperatures")
        logger.info(f"Resistivity at 300K: {resistivity[len(temps)//3]:.2f} μΩ·cm")
        
        return self.results
    
    def plot_transport_properties(
        self,
        results: Optional[TransportResults] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot comprehensive transport properties.
        
        Args:
            results: TransportResults (uses self.results if None)
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        results = results or self.results
        if results is None:
            raise ValueError("No results available")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        temps = results.temperatures
        
        # Resistivity
        ax = axes[0, 0]
        ax.semilogy(temps, results.resistivity, 'b-', lw=2)
        ax.set_xlabel('T (K)', fontsize=11)
        ax.set_ylabel('ρ (μΩ·cm)', fontsize=11)
        ax.set_title('Electrical Resistivity', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Conductivity
        ax = axes[0, 1]
        ax.semilogy(temps, results.conductivity, 'r-', lw=2)
        ax.set_xlabel('T (K)', fontsize=11)
        ax.set_ylabel('σ (S/cm)', fontsize=11)
        ax.set_title('Electrical Conductivity', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Mobility
        ax = axes[0, 2]
        ax.semilogy(temps, results.mobility, 'g-', lw=2)
        ax.set_xlabel('T (K)', fontsize=11)
        ax.set_ylabel('μ (cm²/V/s)', fontsize=11)
        ax.set_title('Electron Mobility', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Seebeck
        ax = axes[1, 0]
        ax.plot(temps, results.seebeck, 'm-', lw=2)
        ax.axhline(y=0, color='k', linestyle='--', lw=0.5)
        ax.set_xlabel('T (K)', fontsize=11)
        ax.set_ylabel('S (μV/K)', fontsize=11)
        ax.set_title('Seebeck Coefficient', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Lorenz number
        ax = axes[1, 1]
        ax.plot(temps, results.lorenz_number * 1e8, 'c-', lw=2)
        ax.axhline(y=2.44, color='k', linestyle='--', lw=1, label='L₀ (Sommerfeld)')
        ax.set_xlabel('T (K)', fontsize=11)
        ax.set_ylabel('L (10⁻⁸ W·Ω/K²)', fontsize=11)
        ax.set_title('Lorenz Number', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Thermal conductivity
        ax = axes[1, 2]
        ax.plot(temps, results.thermal_conductivity_e, 'orange', lw=2)
        ax.set_xlabel('T (K)', fontsize=11)
        ax.set_ylabel('κₑ (W/m/K)', fontsize=11)
        ax.set_title('Electronic Thermal Conductivity', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transport properties plot to {save_path}")
        
        return fig
    
    def plot_resistivity_comparison(
        self,
        temperatures: np.ndarray,
        rho_ziman: np.ndarray,
        rho_bg: np.ndarray,
        experimental: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Compare different resistivity models.
        
        Args:
            temperatures: Temperature array
            rho_ziman: Ziman formula results
            rho_bg: Bloch-Grüneisen results
            experimental: Experimental data (optional)
            figsize: Figure size
            save_path: Path to save
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(temperatures, rho_ziman, 'b-', lw=2, label='Ziman formula')
        ax.plot(temperatures, rho_bg, 'r--', lw=2, label='Bloch-Grüneisen')
        
        if experimental is not None:
            ax.scatter(temperatures[::5], experimental[::5], 
                      c='g', s=50, label='Experimental', zorder=5)
        
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('Resistivity (μΩ·cm)', fontsize=12)
        ax.set_title('Resistivity: Theory vs Experiment', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == '__main__':
    print("Electron-Phonon Transport Calculator - use within Python")
