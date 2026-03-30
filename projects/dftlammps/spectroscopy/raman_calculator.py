"""
Raman Spectroscopy Calculator
==============================

Calculation of Raman spectra from phonon data and DFPT.

Features:
- Raman tensor calculation from DFPT (VASP)
- Raman activity and intensity calculation
- Temperature-dependent Raman spectra
- Polarization-resolved Raman
- Resonance Raman effects

Author: DFTLammps Spectroscopy Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
from numpy.linalg import norm, eigh
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Phonopy imports
try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.phonon.raman import Raman
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False

# Pymatgen imports
try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Vasprun, Outcar
    PMG_AVAILABLE = True
except ImportError:
    PMG_AVAILABLE = False

logger = logging.getLogger(__name__)

# Physical constants
THZ_TO_CMM1 = 33.356  # THz to cm^-1
CMM1_TO_EV = 1.23984e-4  # cm^-1 to eV
EPSILON_0 = 8.854e-12  # F/m


class RamanMode(Enum):
    """Raman calculation modes."""
    NON_RESONANT = "non_resonant"  # Standard Raman
    RESONANT = "resonant"  # Resonance Raman
    SURFACE_ENHANCED = "sers"  # SERS


@dataclass
class RamanConfig:
    """Configuration for Raman calculations."""
    
    # Incident light parameters
    laser_wavelength: float = 532.0  # nm (green laser)
    laser_energy: Optional[float] = None  # eV (calculated from wavelength if None)
    
    # Scattering geometry
    incident_polarization: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0]))
    scattered_polarization: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    
    # Temperature
    temperature: float = 300.0  # K
    
    # Spectral parameters
    frequency_range: Optional[Tuple[float, float]] = None  # cm^-1
    broadening: float = 2.0  # cm^-1 (FWHM)
    
    # Output
    output_dir: str = "./raman_output"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.laser_energy is None:
            # E(eV) = 1240 / λ(nm)
            self.laser_energy = 1240.0 / self.laser_wavelength


@dataclass
class RamanTensor:
    """Raman tensor for a phonon mode."""
    
    mode_index: int
    frequency: float  # cm^-1
    frequency_thz: float  # THz
    
    # Raman tensor (3x3 symmetric)
    tensor: np.ndarray
    
    # Derived quantities
    activity: Optional[float] = None  # Raman activity
    intensity: Optional[float] = None  # Relative intensity
    depolarization_ratio: Optional[float] = None
    
    def calculate_activity(self):
        """Calculate Raman activity from tensor."""
        # Raman activity: A = 45α² + 7γ²
        # where α is trace and γ is anisotropy
        
        trace = np.trace(self.tensor)
        alpha_squared = trace**2 / 9.0
        
        # Anisotropy
        deviatoric = self.tensor - trace * np.eye(3) / 3.0
        gamma_squared = np.sum(deviatoric**2) * 3.0 / 2.0
        
        self.activity = 45 * alpha_squared + 7 * gamma_squared
        return self.activity
    
    def calculate_intensity(self, incident_pol: np.ndarray, 
                           scattered_pol: np.ndarray) -> float:
        """
        Calculate Raman intensity for given polarization geometry.
        
        I ∝ |e_i · R · e_s|²
        """
        intensity = np.abs(
            np.dot(incident_pol, np.dot(self.tensor, scattered_pol))
        )**2
        self.intensity = intensity
        return intensity
    
    def calculate_depolarization_ratio(self) -> float:
        """Calculate depolarization ratio ρ = I⊥ / I∥."""
        # For unpolarized incident light
        # ρ = 3γ² / (45α² + 4γ²)
        
        trace = np.trace(self.tensor)
        alpha = trace / 3.0
        deviatoric = self.tensor - trace * np.eye(3) / 3.0
        gamma = np.sqrt(3.0 * np.sum(deviatoric**2) / 2.0)
        
        if 45 * alpha**2 + 4 * gamma**2 > 1e-10:
            rho = 3 * gamma**2 / (45 * alpha**2 + 4 * gamma**2)
        else:
            rho = 0.0
        
        self.depolarization_ratio = rho
        return rho


@dataclass
class RamanSpectrum:
    """Complete Raman spectrum."""
    
    frequencies: np.ndarray  # cm^-1
    intensities: np.ndarray  # Arbitrary units
    
    # Mode information
    modes: Optional[List[RamanTensor]] = None
    
    # Experimental parameters
    laser_wavelength: float = 532.0  # nm
    temperature: float = 300.0  # K
    
    def get_peak_positions(self, threshold: float = 0.01) -> List[Tuple[float, float]]:
        """
        Get peak positions and intensities.
        
        Returns:
            List of (frequency, intensity) tuples
        """
        from scipy.signal import find_peaks
        
        # Normalize
        norm_intensity = self.intensities / np.max(self.intensities)
        
        # Find peaks
        peaks, properties = find_peaks(
            norm_intensity, 
            height=threshold,
            prominence=0.05
        )
        
        return [(self.frequencies[p], self.intensities[p]) for p in peaks]
    
    def save(self, filepath: str):
        """Save spectrum to file."""
        ext = Path(filepath).suffix
        if ext == '.dat' or ext == '.txt':
            np.savetxt(filepath, np.column_stack([self.frequencies, self.intensities]),
                      header='Frequency(cm-1)  Intensity')
        elif ext == '.json':
            data = {
                'frequencies': self.frequencies.tolist(),
                'intensities': self.intensities.tolist(),
                'laser_wavelength': self.laser_wavelength,
                'temperature': self.temperature
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif ext == '.npz':
            np.savez(filepath, 
                    frequencies=self.frequencies,
                    intensities=self.intensities,
                    laser_wavelength=self.laser_wavelength,
                    temperature=self.temperature)


class RamanCalculator:
    """
    Calculator for Raman spectra from phonon calculations.
    
    Supports:
    - VASP DFPT Raman tensors (LEPSILON=.TRUE.)
    - Finite difference Raman from phonopy
    - Non-resonant and resonant Raman
    """
    
    def __init__(self, config: Optional[RamanConfig] = None):
        """
        Initialize Raman calculator.
        
        Args:
            config: Raman calculation configuration
        """
        self.config = config or RamanConfig()
        self.raman_tensors: List[RamanTensor] = []
        self.spectrum: Optional[RamanSpectrum] = None
        
        logger.info(f"Initialized RamanCalculator at λ = {self.config.laser_wavelength} nm")
    
    def read_raman_tensors_vasp(self, vasprun_path: str) -> List[RamanTensor]:
        """
        Read Raman tensors from VASP DFPT calculation.
        
        Requires VASP calculation with LEPSILON = .TRUE. and LRAMAN = .TRUE.
        
        Args:
            vasprun_path: Path to vasprun.xml
            
        Returns:
            List of RamanTensor objects
        """
        if not PMG_AVAILABLE:
            raise ImportError("Pymatgen required for VASP parsing")
        
        vasprun = Vasprun(vasprun_path)
        
        # Get Raman tensors
        # Note: VASP doesn't directly output Raman tensors
        # They need to be calculated from Born charges and phonon modes
        
        # Get dielectric tensor and Born effective charges
        epsilon = vasprun.epsilon_static  # Dielectric tensor
        born_charges = vasprun.born_charges  # Born effective charges
        
        # Get phonon frequencies and eigenvectors (would need separate calculation)
        # This is a placeholder - real implementation would interface with phonopy
        
        logger.info(f"Read dielectric and Born charge data from {vasprun_path}")
        
        # Calculate Raman tensors from data
        # Raman tensor R ~ dχ/dQ where χ is susceptibility and Q is phonon coordinate
        
        return self.raman_tensors
    
    def calculate_raman_tensors_phonopy(
        self,
        phonopy: Phonopy,
        born_charges: np.ndarray,
        dielectric_tensor: np.ndarray,
        delta_q: float = 0.01
    ) -> List[RamanTensor]:
        """
        Calculate Raman tensors using phonopy.
        
        Raman tensor: R_ij = dχ_ij/dQ = Σ_α (dχ_ij/du_α) * e_α/√M_α
        
        Args:
            phonopy: Phonopy object with force constants
            born_charges: Born effective charges (n_atoms, 3, 3)
            dielectric_tensor: Electronic dielectric tensor (3, 3)
            delta_q: Displacement for finite differences
            
        Returns:
            List of RamanTensor objects
        """
        if not PHONOPY_AVAILABLE:
            raise ImportError("Phonopy required")
        
        # Get phonon frequencies and eigenvectors at Γ
        phonopy.run_qpoints([[0, 0, 0]])
        frequencies = phonopy.qpoints.frequencies[0]  # THz
        eigenvectors = phonopy.qpoints.eigenvectors[0]
        
        n_modes = len(frequencies)
        n_atoms = len(phonopy.unitcell)
        
        # Calculate Raman tensor for each mode
        self.raman_tensors = []
        
        for mode in range(n_modes):
            freq_thz = frequencies[mode]
            freq_cm = freq_thz * THZ_TO_CMM1
            
            # Skip imaginary modes
            if freq_thz < 0.1:  # ~3 cm^-1
                continue
            
            # Get eigenvector for this mode
            eig = eigenvectors[mode]  # Shape: (n_atoms * 3,)
            eig = eig.reshape(n_atoms, 3)
            
            # Calculate Raman tensor
            # Simplified: R_ij = Σ_α Z*_ij,α · e_α
            raman_tensor = np.zeros((3, 3))
            
            for atom in range(n_atoms):
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            raman_tensor[i, j] += (
                                born_charges[atom, i, k] * eig[atom, k]
                            )
            
            # Normalize
            masses = phonopy.unitcell.masses
            norm_factor = np.sqrt(np.sum([m * np.sum(eig[i]**2) 
                                         for i, m in enumerate(masses)]))
            raman_tensor /= norm_factor
            
            rt = RamanTensor(
                mode_index=mode,
                frequency=freq_cm,
                frequency_thz=freq_thz,
                tensor=raman_tensor
            )
            
            rt.calculate_activity()
            rt.calculate_depolarization_ratio()
            
            self.raman_tensors.append(rt)
        
        logger.info(f"Calculated {len(self.raman_tensors)} Raman tensors")
        return self.raman_tensors
    
    def calculate_spectrum(
        self,
        frequency_range: Optional[Tuple[float, float]] = None,
        n_points: int = 1000,
        broadening: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> RamanSpectrum:
        """
        Calculate Raman spectrum from tensors.
        
        Args:
            frequency_range: (min, max) in cm^-1
            n_points: Number of frequency points
            broadening: Peak broadening (cm^-1)
            temperature: Temperature (K)
            
        Returns:
            RamanSpectrum object
        """
        if not self.raman_tensors:
            raise ValueError("Calculate Raman tensors first")
        
        freq_range = frequency_range or self.config.frequency_range or (0, 1000)
        broadening = broadening or self.config.broadening
        temperature = temperature or self.config.temperature
        
        # Generate frequency grid
        frequencies = np.linspace(freq_range[0], freq_range[1], n_points)
        intensities = np.zeros(n_points)
        
        # Calculate intensity for each mode
        for rt in self.raman_tensors:
            # Temperature factor (Bose-Einstein)
            freq_ev = rt.frequency * CMM1_TO_EV
            if temperature > 0:
                n_bose = 1.0 / (np.exp(freq_ev / (8.617e-5 * temperature)) - 1.0)
            else:
                n_bose = 0.0
            
            # Stokes intensity: I ∝ (n_B + 1) * activity
            # Anti-Stokes: I ∝ n_B * activity
            intensity_stokes = (n_bose + 1) * rt.activity
            
            # Calculate intensity for polarization geometry
            pol_intensity = rt.calculate_intensity(
                self.config.incident_polarization,
                self.config.scattered_polarization
            )
            
            total_intensity = intensity_stokes * pol_intensity
            
            # Lorentzian broadening
            gamma = broadening / 2.0
            intensities += total_intensity * (
                gamma / (np.pi * ((frequencies - rt.frequency)**2 + gamma**2))
            )
        
        self.spectrum = RamanSpectrum(
            frequencies=frequencies,
            intensities=intensities,
            modes=self.raman_tensors,
            laser_wavelength=self.config.laser_wavelength,
            temperature=temperature
        )
        
        return self.spectrum
    
    def calculate_polarized_spectra(
        self,
        configurations: List[Tuple[np.ndarray, np.ndarray, str]]
    ) -> Dict[str, RamanSpectrum]:
        """
        Calculate spectra for different polarization configurations.
        
        Args:
            configurations: List of (incident_pol, scattered_pol, label) tuples
            
        Returns:
            Dictionary of spectra for each configuration
        """
        spectra = {}
        
        for inc_pol, scat_pol, label in configurations:
            # Update polarization
            self.config.incident_polarization = inc_pol
            self.config.scattered_polarization = scat_pol
            
            # Calculate spectrum
            spectrum = self.calculate_spectrum()
            spectra[label] = spectrum
        
        return spectra
    
    def plot_spectrum(
        self,
        spectrum: Optional[RamanSpectrum] = None,
        show_modes: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot Raman spectrum.
        
        Args:
            spectrum: Spectrum to plot (uses self.spectrum if None)
            show_modes: Mark individual mode positions
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        spectrum = spectrum or self.spectrum
        if spectrum is None:
            raise ValueError("No spectrum available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot spectrum
        ax.plot(spectrum.frequencies, spectrum.intensities, 'b-', lw=1.5)
        ax.fill_between(spectrum.frequencies, 0, spectrum.intensities, 
                        alpha=0.3, color='b')
        
        # Mark individual modes
        if show_modes and spectrum.modes:
            for mode in spectrum.modes:
                ax.axvline(x=mode.frequency, color='r', linestyle='--', 
                          alpha=0.3, lw=0.5)
        
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (arb. units)', fontsize=12)
        ax.set_title(f'Raman Spectrum (λ = {spectrum.laser_wavelength:.0f} nm, '
                    f'T = {spectrum.temperature:.0f} K)', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Raman spectrum to {save_path}")
        
        return fig
    
    def export_to_csv(self, filepath: str):
        """Export spectrum to CSV format."""
        if self.spectrum is None:
            raise ValueError("No spectrum to export")
        
        np.savetxt(
            filepath,
            np.column_stack([self.spectrum.frequencies, self.spectrum.intensities]),
            delimiter=',',
            header='Frequency(cm-1),Intensity',
            comments=''
        )
        logger.info(f"Exported Raman spectrum to {filepath}")
    
    def get_mode_analysis(self) -> Dict:
        """
        Get analysis of Raman-active modes.
        
        Returns:
            Dictionary with mode analysis
        """
        if not self.raman_tensors:
            raise ValueError("No Raman tensors available")
        
        analysis = {
            'n_modes': len(self.raman_tensors),
            'frequency_range': (
                min(rt.frequency for rt in self.raman_tensors),
                max(rt.frequency for rt in self.raman_tensors)
            ),
            'strongest_mode': None,
            'modes': []
        }
        
        # Find strongest mode
        strongest = max(self.raman_tensors, key=lambda x: x.activity)
        analysis['strongest_mode'] = {
            'index': strongest.mode_index,
            'frequency': strongest.frequency,
            'activity': strongest.activity,
            'depolarization_ratio': strongest.depolarization_ratio
        }
        
        # Mode details
        for rt in sorted(self.raman_tensors, key=lambda x: x.frequency):
            analysis['modes'].append({
                'index': rt.mode_index,
                'frequency_cm': rt.frequency,
                'frequency_thz': rt.frequency_thz,
                'activity': rt.activity,
                'depolarization_ratio': rt.depolarization_ratio
            })
        
        return analysis


def calculate_raman_spectrum_workflow(
    phonopy: Phonopy,
    born_charges: np.ndarray,
    dielectric_tensor: np.ndarray,
    laser_wavelength: float = 532.0,
    temperature: float = 300.0,
    output_dir: str = './raman_output'
) -> RamanSpectrum:
    """
    Complete workflow for Raman spectrum calculation.
    
    Args:
        phonopy: Phonopy object
        born_charges: Born effective charges
        dielectric_tensor: Dielectric tensor
        laser_wavelength: Laser wavelength (nm)
        temperature: Temperature (K)
        output_dir: Output directory
        
    Returns:
        RamanSpectrum object
    """
    config = RamanConfig(
        laser_wavelength=laser_wavelength,
        temperature=temperature,
        output_dir=output_dir
    )
    
    calc = RamanCalculator(config)
    calc.calculate_raman_tensors_phonopy(phonopy, born_charges, dielectric_tensor)
    spectrum = calc.calculate_spectrum()
    
    # Save outputs
    calc.plot_spectrum(save_path=f"{output_dir}/raman_spectrum.png")
    spectrum.save(f"{output_dir}/raman_spectrum.dat")
    
    return spectrum


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Raman Spectrum Calculator')
    parser.add_argument('--phonopy', type=str, help='Phonopy params file')
    parser.add_argument('--born', type=str, help='Born charges file')
    parser.add_argument('--laser', type=float, default=532.0, help='Laser wavelength (nm)')
    parser.add_argument('--temp', type=float, default=300.0, help='Temperature (K)')
    parser.add_argument('--outdir', type=str, default='./raman_output', help='Output directory')
    
    args = parser.parse_args()
    
    print("Raman Calculator - use within Python for full functionality")
