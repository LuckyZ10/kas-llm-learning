"""
Infrared (IR) Spectroscopy Calculator
======================================

Calculation of infrared absorption spectra from phonon data.

Features:
- IR intensity from Born effective charges
- Dielectric function calculation
- Absorption coefficient
- Reflectivity and transmittance
- Temperature-dependent IR spectra

Author: DFTLammps Spectroscopy Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import norm, inv, eigvals
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Phonopy imports
try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False

# Pymatgen imports
try:
    from pymatgen.core import Structure
    PMG_AVAILABLE = True
except ImportError:
    PMG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Physical constants
THZ_TO_CMM1 = 33.356  # THz to cm^-1
EPSILON_0 = 8.854e-12  # F/m
C = 2.998e10  # Speed of light (cm/s)


@dataclass
class IRConfig:
    """Configuration for IR calculations."""
    
    # Frequency range
    frequency_min: float = 0.0  # cm^-1
    frequency_max: float = 2000.0  # cm^-1
    n_points: int = 2000
    
    # Broadening
    gamma: float = 5.0  # cm^-1 (damping constant)
    broadening_type: str = 'lorentzian'  # 'lorentzian' or 'gaussian'
    
    # Dielectric background
    epsilon_infinity: Optional[np.ndarray] = None  # High-frequency dielectric tensor
    
    # Experimental conditions
    temperature: float = 300.0  # K
    incident_angle: float = 0.0  # degrees (normal incidence)
    
    # Output
    output_dir: str = "./ir_output"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class IRTensor:
    """IR data for a phonon mode."""
    
    mode_index: int
    frequency: float  # cm^-1
    frequency_thz: float  # THz
    
    # Infrared intensity (related to oscillator strength)
    intensity: float
    
    # Oscillator strength tensor
    oscillator_strength: np.ndarray  # (3, 3)
    
    # Effective charge derivative
    debye_waller_factor: float = 1.0
    
    def calculate_intensity_from_charges(
        self,
        born_charges: np.ndarray,
        eigenvector: np.ndarray,
        masses: np.ndarray
    ):
        """
        Calculate IR intensity from Born effective charges.
        
        I ∝ |Σ_α Z*_α · e_α/√M_α|²
        """
        n_atoms = len(masses)
        eig = eigenvector.reshape(n_atoms, 3)
        
        # Calculate mode effective charge
        Z_mode = np.zeros(3)
        for atom in range(n_atoms):
            Z_mode += np.dot(born_charges[atom], eig[atom]) / np.sqrt(masses[atom])
        
        self.intensity = np.sum(Z_mode**2)
        
        # Oscillator strength
        self.oscillator_strength = np.outer(Z_mode, Z_mode)
        
        return self.intensity


@dataclass
class IRSpectrum:
    """Complete IR spectrum."""
    
    frequencies: np.ndarray  # cm^-1
    
    # Dielectric function
    epsilon_real: np.ndarray  # Real part
    epsilon_imag: np.ndarray  # Imaginary part
    
    # Optical properties
    absorption_coeff: np.ndarray  # cm^-1
    reflectivity: np.ndarray
    transmittance: Optional[np.ndarray] = None
    
    # Mode information
    modes: Optional[List[IRTensor]] = None
    
    def get_band_positions(self, threshold: float = 0.1) -> List[Tuple[float, float]]:
        """Get IR active band positions and intensities."""
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(
            self.epsilon_imag,
            height=threshold * np.max(self.epsilon_imag),
            prominence=0.05 * np.max(self.epsilon_imag)
        )
        
        return [(self.frequencies[p], self.epsilon_imag[p]) for p in peaks]
    
    def save(self, filepath: str):
        """Save spectrum to file."""
        ext = Path(filepath).suffix
        if ext == '.dat':
            np.savetxt(filepath, np.column_stack([
                self.frequencies,
                self.epsilon_real,
                self.epsilon_imag,
                self.absorption_coeff
            ]), header='Freq(cm-1)  Eps_real  Eps_imag  Abs_coeff(cm-1)')
        elif ext == '.json':
            data = {
                'frequencies': self.frequencies.tolist(),
                'epsilon_real': self.epsilon_real.tolist(),
                'epsilon_imag': self.epsilon_imag.tolist(),
                'absorption_coeff': self.absorption_coeff.tolist()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)


class IRCalculator:
    """
    Calculator for IR spectra from phonon calculations.
    
    Uses Born effective charges and phonon frequencies to calculate
    dielectric function and optical properties.
    """
    
    def __init__(self, config: Optional[IRConfig] = None):
        """
        Initialize IR calculator.
        
        Args:
            config: IR calculation configuration
        """
        self.config = config or IRConfig()
        self.ir_tensors: List[IRTensor] = []
        self.spectrum: Optional[IRSpectrum] = None
        
        logger.info("Initialized IRCalculator")
    
    def calculate_ir_tensors(
        self,
        phonopy: Phonopy,
        born_charges: np.ndarray
    ) -> List[IRTensor]:
        """
        Calculate IR tensors from phonon modes and Born charges.
        
        Args:
            phonopy: Phonopy object with force constants
            born_charges: Born effective charges (n_atoms, 3, 3)
            
        Returns:
            List of IRTensor objects
        """
        if not PHONOPY_AVAILABLE:
            raise ImportError("Phonopy required")
        
        # Get phonon data at Γ
        phonopy.run_qpoints([[0, 0, 0]])
        frequencies = phonopy.qpoints.frequencies[0]  # THz
        eigenvectors = phonopy.qpoints.eigenvectors[0]
        
        masses = phonopy.unitcell.masses
        n_atoms = len(phonopy.unitcell)
        
        self.ir_tensors = []
        
        for mode in range(len(frequencies)):
            freq_thz = frequencies[mode]
            freq_cm = freq_thz * THZ_TO_CMM1
            
            # Skip acoustic modes and imaginary frequencies
            if freq_thz < 0.5:
                continue
            
            # Create IR tensor
            ir_tensor = IRTensor(
                mode_index=mode,
                frequency=freq_cm,
                frequency_thz=freq_thz,
                intensity=0.0,
                oscillator_strength=np.zeros((3, 3))
            )
            
            # Calculate intensity
            eig = eigenvectors[mode].reshape(n_atoms, 3)
            ir_tensor.calculate_intensity_from_charges(
                born_charges, eig, masses
            )
            
            self.ir_tensors.append(ir_tensor)
        
        logger.info(f"Calculated {len(self.ir_tensors)} IR-active modes")
        return self.ir_tensors
    
    def calculate_dielectric_function(
        self,
        epsilon_infinity: Optional[np.ndarray] = None,
        gamma: Optional[float] = None,
        frequency_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate frequency-dependent dielectric function.
        
        Using Lorentz oscillator model:
        ε(ω) = ε_∞ + Σ_j S_j / (ω_j² - ω² - iγω)
        
        Args:
            epsilon_infinity: High-frequency dielectric tensor
            gamma: Damping parameter (cm^-1)
            frequency_range: (min, max) frequency range
            
        Returns:
            Tuple of (frequencies, epsilon_real, epsilon_imag)
        """
        if not self.ir_tensors:
            raise ValueError("Calculate IR tensors first")
        
        eps_inf = epsilon_infinity or self.config.epsilon_infinity
        if eps_inf is None:
            eps_inf = np.eye(3)  # Default to identity
        
        gamma = gamma or self.config.gamma
        freq_range = frequency_range or (self.config.frequency_min, 
                                         self.config.frequency_max)
        
        # Frequency grid
        frequencies = np.linspace(freq_range[0], freq_range[1], 
                                  self.config.n_points)
        
        # Calculate dielectric function (trace)
        epsilon_real = np.full(len(frequencies), np.trace(eps_inf) / 3.0)
        epsilon_imag = np.zeros(len(frequencies))
        
        for ir_mode in self.ir_tensors:
            omega_j = ir_mode.frequency  # cm^-1
            S_j = np.trace(ir_mode.oscillator_strength) / 3.0  # Oscillator strength
            
            # Lorentzian contribution
            for i, omega in enumerate(frequencies):
                if abs(omega - omega_j) < 1e-6:
                    continue
                
                denom = (omega_j**2 - omega**2)**2 + (gamma * omega)**2
                epsilon_real[i] += S_j * (omega_j**2 - omega**2) / denom
                epsilon_imag[i] += S_j * gamma * omega / denom
        
        return frequencies, epsilon_real, epsilon_imag
    
    def calculate_optical_properties(
        self,
        frequencies: np.ndarray,
        epsilon_real: np.ndarray,
        epsilon_imag: np.ndarray,
        thickness: Optional[float] = None  # cm
    ) -> Dict[str, np.ndarray]:
        """
        Calculate optical properties from dielectric function.
        
        Args:
            frequencies: Frequency array (cm^-1)
            epsilon_real: Real part of dielectric function
            epsilon_imag: Imaginary part
            thickness: Sample thickness for transmittance (cm)
            
        Returns:
            Dictionary with absorption, reflectivity, transmittance
        """
        # Complex refractive index: n + iκ = √ε
        epsilon = epsilon_real + 1j * epsilon_imag
        n_complex = np.sqrt(epsilon)
        n = n_complex.real
        kappa = n_complex.imag
        
        # Absorption coefficient: α = 4πκν/c = 4πκω̃
        # where ω̃ is wavenumber in cm^-1
        absorption_coeff = 4 * np.pi * frequencies * kappa  # cm^-1
        
        # Reflectivity at normal incidence
        R = ((n - 1)**2 + kappa**2) / ((n + 1)**2 + kappa**2)
        
        properties = {
            'refractive_index': n,
            'extinction_coefficient': kappa,
            'absorption_coefficient': absorption_coeff,
            'reflectivity': R
        }
        
        # Transmittance (if thickness given)
        if thickness is not None:
            T = (1 - R)**2 * np.exp(-absorption_coeff * thickness)
            properties['transmittance'] = T
        
        return properties
    
    def calculate_spectrum(
        self,
        epsilon_infinity: Optional[np.ndarray] = None,
        gamma: Optional[float] = None,
        thickness: Optional[float] = None
    ) -> IRSpectrum:
        """
        Calculate complete IR spectrum.
        
        Args:
            epsilon_infinity: High-frequency dielectric tensor
            gamma: Damping parameter
            thickness: Sample thickness for transmittance
            
        Returns:
            IRSpectrum object
        """
        # Calculate dielectric function
        frequencies, eps_real, eps_imag = self.calculate_dielectric_function(
            epsilon_infinity, gamma
        )
        
        # Calculate optical properties
        optical_props = self.calculate_optical_properties(
            frequencies, eps_real, eps_imag, thickness
        )
        
        self.spectrum = IRSpectrum(
            frequencies=frequencies,
            epsilon_real=eps_real,
            epsilon_imag=eps_imag,
            absorption_coeff=optical_props['absorption_coefficient'],
            reflectivity=optical_props['reflectivity'],
            transmittance=optical_props.get('transmittance'),
            modes=self.ir_tensors
        )
        
        return self.spectrum
    
    def plot_dielectric_function(
        self,
        spectrum: Optional[IRSpectrum] = None,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> Figure:
        """Plot dielectric function."""
        spectrum = spectrum or self.spectrum
        if spectrum is None:
            raise ValueError("No spectrum available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Real part
        ax1.plot(spectrum.frequencies, spectrum.epsilon_real, 'b-', lw=1.5)
        ax1.axhline(y=0, color='k', linestyle='--', lw=0.5)
        ax1.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
        ax1.set_ylabel('ε\' (Real)', fontsize=12)
        ax1.set_title('Real Part of Dielectric Function', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # Imaginary part
        ax2.plot(spectrum.frequencies, spectrum.epsilon_imag, 'r-', lw=1.5)
        ax2.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
        ax2.set_ylabel('ε\'\' (Imaginary)', fontsize=12)
        ax2.set_title('Imaginary Part of Dielectric Function', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_absorption_spectrum(
        self,
        spectrum: Optional[IRSpectrum] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Figure:
        """Plot IR absorption spectrum."""
        spectrum = spectrum or self.spectrum
        if spectrum is None:
            raise ValueError("No spectrum available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(spectrum.frequencies, spectrum.absorption_coeff, 'b-', lw=1.5)
        ax.fill_between(spectrum.frequencies, 0, spectrum.absorption_coeff,
                        alpha=0.3, color='b')
        
        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Absorption Coefficient (cm⁻¹)', fontsize=12)
        ax.set_title('IR Absorption Spectrum', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved IR spectrum to {save_path}")
        
        return fig


def calculate_ir_spectrum_workflow(
    phonopy: Phonopy,
    born_charges: np.ndarray,
    epsilon_infinity: Optional[np.ndarray] = None,
    gamma: float = 5.0,
    output_dir: str = './ir_output'
) -> IRSpectrum:
    """
    Complete workflow for IR spectrum calculation.
    
    Args:
        phonopy: Phonopy object
        born_charges: Born effective charges
        epsilon_infinity: High-frequency dielectric tensor
        gamma: Damping parameter (cm^-1)
        output_dir: Output directory
        
    Returns:
        IRSpectrum object
    """
    config = IRConfig(gamma=gamma, output_dir=output_dir)
    
    calc = IRCalculator(config)
    calc.calculate_ir_tensors(phonopy, born_charges)
    spectrum = calc.calculate_spectrum(epsilon_infinity=epsilon_infinity)
    
    # Save outputs
    calc.plot_dielectric_function(save_path=f"{output_dir}/dielectric_function.png")
    calc.plot_absorption_spectrum(save_path=f"{output_dir}/ir_absorption.png")
    spectrum.save(f"{output_dir}/ir_spectrum.dat")
    
    return spectrum


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='IR Spectrum Calculator')
    parser.add_argument('--phonopy', type=str, help='Phonopy file')
    parser.add_argument('--born', type=str, required=True, help='Born charges file')
    parser.add_argument('--gamma', type=float, default=5.0, help='Damping (cm-1)')
    parser.add_argument('--outdir', type=str, default='./ir_output')
    
    args = parser.parse_args()
    
    print("IR Calculator - use within Python for full functionality")
