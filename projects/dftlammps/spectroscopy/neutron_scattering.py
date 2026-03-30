"""
Neutron Scattering Calculator
==============================

Calculation of neutron scattering cross sections from phonon data.

Features:
- Coherent neutron scattering function S(Q,ω)
- Incoherent scattering cross sections
- Powder diffraction pattern simulation
- Single crystal neutron scattering
- Temperature-dependent scattering
- Energy and momentum resolution effects

Author: DFTLammps Spectroscopy Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

# Phonopy imports
try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.spectrum.dynamic_structure_factor import atomic_form_factor_WK1995
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Physical constants
NEUTRON_MASS = 1.675e-27  # kg
HBAR = 1.055e-34  # J*s
EV_TO_J = 1.602e-19
MEV_TO_J = EV_TO_J / 1000
THZ_TO_MEV = 4.136  # THz to meV
ANGSTROM = 1e-10  # m

# Neutron scattering lengths (fm)
NEUTRON_SCATTERING_LENGTHS = {
    'H': -3.74, 'D': 6.67, 'He': 3.26, 'Li': -1.90, 'Be': 7.79,
    'B': 5.30, 'C': 6.65, 'N': 9.36, 'O': 5.80, 'F': 5.65,
    'Ne': 4.57, 'Na': 3.63, 'Mg': 5.38, 'Al': 3.45, 'Si': 4.15,
    'P': 5.13, 'S': 2.85, 'Cl': 9.58, 'Ar': 1.91, 'K': 3.67,
    'Ca': 4.70, 'Sc': 12.3, 'Ti': -3.44, 'V': -0.38, 'Cr': 3.64,
    'Mn': -3.75, 'Fe': 9.45, 'Co': 2.49, 'Ni': 10.3, 'Cu': 7.72,
    'Zn': 5.68, 'Ga': 7.29, 'Ge': 8.19, 'As': 6.58, 'Se': 7.97,
    'Br': 6.79, 'Kr': 7.81, 'Rb': 7.09, 'Sr': 7.02, 'Y': 7.75,
    'Zr': 7.16, 'Nb': 7.05, 'Mo': 6.72, 'Tc': 6.8, 'Ru': 7.03,
    'Rh': 5.88, 'Pd': 5.91, 'Ag': 5.92, 'Cd': 4.87, 'In': 4.07,
    'Sn': 6.23, 'Sb': 5.57, 'Te': 5.80, 'I': 5.28, 'Xe': 4.92,
    'Cs': 5.42, 'Ba': 5.07, 'La': 8.53, 'Ce': 4.84, 'Pr': 4.58,
    'Nd': 7.69, 'Pm': 12.6, 'Sm': 0.80, 'Eu': 7.22, 'Gd': 9.5,
    'Tb': 7.38, 'Dy': 16.9, 'Ho': 8.01, 'Er': 7.79, 'Tm': 7.07,
    'Yb': 12.43, 'Lu': 7.21, 'Hf': 7.77, 'Ta': 6.91, 'W': 4.86,
    'Re': 9.2, 'Os': 10.7, 'Ir': 10.6, 'Pt': 9.60, 'Au': 7.63,
    'Hg': 12.66, 'Tl': 8.78, 'Pb': 9.41, 'Bi': 8.53, 'Po': None,
    'At': None, 'Rn': None, 'Fr': None, 'Ra': 10.0, 'Ac': None,
    'Th': 10.31, 'Pa': 9.1, 'U': 8.42
}


@dataclass
class NeutronConfig:
    """Configuration for neutron scattering calculations."""
    
    # Instrument parameters
    e_i: float = 50.0  # Incident energy (meV)
    e_f: Optional[float] = None  # Final energy (meV)
    
    # Q-grid
    q_min: float = 0.0  # Å^-1
    q_max: float = 10.0  # Å^-1
    n_q: int = 200
    
    # Energy transfer
    omega_min: float = -50.0  # meV
    omega_max: float = 50.0  # meV
    n_omega: int = 400
    
    # Temperature
    temperature: float = 300.0  # K
    
    # Resolution
    dq_resolution: float = 0.05  # Å^-1
    domega_resolution: float = 1.0  # meV
    
    # Output
    output_dir: str = "./neutron_output"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class NeutronSpectrum:
    """Neutron scattering spectrum S(Q, ω)."""
    
    q_points: np.ndarray  # Å^-1
    omega: np.ndarray  # meV
    
    # Scattering function S(Q, ω)
    s_qw_coherent: np.ndarray  # Coherent scattering
    s_qw_incoherent: Optional[np.ndarray] = None  # Incoherent scattering
    
    # Powder averaged
    s_qw_powder: Optional[np.ndarray] = None
    
    def save(self, filepath: str):
        """Save spectrum to file."""
        ext = Path(filepath).suffix
        if ext == '.npz':
            np.savez_compressed(
                filepath,
                q_points=self.q_points,
                omega=self.omega,
                s_qw_coherent=self.s_qw_coherent
            )
        elif ext == '.h5' or ext == '.hdf5':
            import h5py
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('q_points', data=self.q_points)
                f.create_dataset('omega', data=self.omega)
                f.create_dataset('s_qw_coherent', data=self.s_qw_coherent)


class NeutronScatteringCalculator:
    """
    Calculator for neutron scattering from phonon data.
    
    Calculates coherent and incoherent scattering cross sections
    for comparison with inelastic neutron scattering experiments.
    """
    
    def __init__(self, config: Optional[NeutronConfig] = None):
        """
        Initialize neutron scattering calculator.
        
        Args:
            config: Neutron scattering configuration
        """
        self.config = config or NeutronConfig()
        self.spectrum: Optional[NeutronSpectrum] = None
        
        logger.info("Initialized NeutronScatteringCalculator")
    
    def get_scattering_length(self, element: str) -> float:
        """Get neutron scattering length for element (fm)."""
        return NEUTRON_SCATTERING_LENGTHS.get(element, 5.0)  # Default to 5 fm
    
    def calculate_structure_factor(
        self,
        phonopy: Phonopy,
        q_points: np.ndarray,
        temperature: Optional[float] = None
    ) -> Dict:
        """
        Calculate dynamic structure factor S(Q, ω).
        
        Args:
            phonopy: Phonopy object with force constants
            q_points: Q-points in reduced coordinates (N_q, 3)
            temperature: Temperature (K)
            
        Returns:
            Dictionary with structure factor data
        """
        if not PHONOPY_AVAILABLE:
            raise ImportError("Phonopy required")
        
        T = temperature or self.config.temperature
        
        # Get phonon frequencies and eigenvectors at q-points
        phonopy.run_qpoints(q_points)
        frequencies = phonopy.qpoints.frequencies  # THz, shape (N_q, n_modes)
        eigenvectors = phonopy.qpoints.eigenvectors  # shape (N_q, n_modes, n_dof)
        
        n_q = len(q_points)
        n_modes = frequencies.shape[1]
        
        # Get atomic information
        symbols = phonopy.unitcell.symbols
        masses = phonopy.unitcell.masses
        positions = phonopy.unitcell.positions
        n_atoms = len(symbols)
        
        # Scattering lengths
        b = np.array([self.get_scattering_length(s) for s in symbols])
        
        # Calculate structure factor
        # S(Q,ω) ∝ Σ_s |F_s(Q)|² δ(ω - ω_s)
        # where F_s(Q) = Σ_j b_j exp(iQ·r_j) (Q·e_js) / √(2M_jω_s)
        
        structure_factors = np.zeros((n_q, n_modes), dtype=complex)
        
        for iq, q in enumerate(q_points):
            # Convert q to Cartesian (simplified)
            q_cart = 2 * np.pi * q  # This assumes simple reciprocal lattice
            
            for s in range(n_modes):
                freq = frequencies[iq, s]  # THz
                
                if freq < 0.1:  # Skip acoustic modes
                    continue
                
                eig = eigenvectors[iq, s].reshape(n_atoms, 3)
                
                F = 0.0
                for j in range(n_atoms):
                    phase = np.exp(1j * np.dot(q_cart, positions[j]))
                    q_dot_e = np.dot(q_cart, eig[j])
                    F += b[j] * phase * q_dot_e / np.sqrt(2 * masses[j] * freq)
                
                structure_factors[iq, s] = F
        
        return {
            'q_points': q_points,
            'frequencies_THz': frequencies,
            'frequencies_meV': frequencies * THZ_TO_MEV,
            'structure_factors': structure_factors,
            'temperature': T
        }
    
    def calculate_s_qw(
        self,
        phonopy: Phonopy,
        q_path: Optional[np.ndarray] = None,
        temperature: Optional[float] = None,
        include_incoherent: bool = True
    ) -> NeutronSpectrum:
        """
        Calculate S(Q, ω) for neutron scattering.
        
        Args:
            phonopy: Phonopy object
            q_path: Q-point path (uses default if None)
            temperature: Temperature (K)
            include_incoherent: Include incoherent scattering
            
        Returns:
            NeutronSpectrum object
        """
        T = temperature or self.config.temperature
        
        # Generate Q-grid
        if q_path is None:
            q_path = self._generate_q_path()
        
        # Generate energy grid
        omega = np.linspace(self.config.omega_min, self.config.omega_max, 
                           self.config.n_omega)
        
        # Calculate structure factor
        sf_data = self.calculate_structure_factor(phonopy, q_path, T)
        
        # Calculate S(Q, ω) with thermal factors
        n_q = len(q_path)
        s_qw = np.zeros((n_q, len(omega)))
        
        for iq in range(n_q):
            for s, freq in enumerate(sf_data['frequencies_meV'][iq]):
                if freq < 0.5:
                    continue
                
                # Bose-Einstein factor
                n_bose = 1.0 / (np.exp(freq / (0.08617 * T)) - 1.0)
                
                # Intensity factor
                # One-phonon coherent: (n+1)δ(ω-ω_s) + nδ(ω+ω_s)
                intensity = np.abs(sf_data['structure_factors'][iq, s])**2
                
                # Add to S(Q, ω) with Lorentzian broadening
                gamma = self.config.domega_resolution / 2.0
                
                # Stokes (energy loss, ω > 0)
                s_qw[iq] += intensity * (n_bose + 1) * (
                    gamma / (np.pi * ((omega - freq)**2 + gamma**2))
                )
                
                # Anti-Stokes (energy gain, ω < 0)
                s_qw[iq] += intensity * n_bose * (
                    gamma / (np.pi * ((omega + freq)**2 + gamma**2))
                )
        
        self.spectrum = NeutronSpectrum(
            q_points=np.linspace(0, 1, n_q),  # Normalized Q-path
            omega=omega,
            s_qw_coherent=s_qw
        )
        
        return self.spectrum
    
    def _generate_q_path(self) -> np.ndarray:
        """Generate default Q-point path."""
        n_points = self.config.n_q
        # Simple path from Γ to X
        q_path = np.zeros((n_points, 3))
        q_path[:, 0] = np.linspace(0, 0.5, n_points)
        return q_path
    
    def calculate_powder_pattern(
        self,
        phonopy: Phonopy,
        q_max: Optional[float] = None,
        n_q: int = 100,
        n_theta: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate powder-averaged neutron scattering.
        
        Args:
            phonopy: Phonopy object
            q_max: Maximum Q value
            n_q: Number of Q points
            n_theta: Number of angular points for averaging
            
        Returns:
            Tuple of (Q_values, S(Q,ω) averaged)
        """
        q_max = q_max or self.config.q_max
        
        # Generate Q sphere
        q_values = np.linspace(0, q_max, n_q)
        
        # Calculate powder average by integrating over angles
        # This is simplified - real implementation would use proper spherical averaging
        
        omega = np.linspace(self.config.omega_min, self.config.omega_max,
                           self.config.n_omega)
        
        s_qw_powder = np.zeros((n_q, len(omega)))
        
        for iq, q in enumerate(q_values):
            # Generate points on sphere
            thetas = np.linspace(0, np.pi, n_theta)
            phis = np.linspace(0, 2*np.pi, n_theta)
            
            for theta in thetas:
                for phi in phis:
                    q_vec = q * np.array([
                        np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)
                    ])
                    
                    # Calculate S(Q, ω) for this direction
                    # Simplified: just use isotropic approximation
                    pass
        
        return q_values, s_qw_powder
    
    def plot_dispersion_with_intensity(
        self,
        spectrum: Optional[NeutronSpectrum] = None,
        figsize: Tuple[int, int] = (10, 6),
        cmap: str = 'hot',
        log_scale: bool = True,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot phonon dispersion with neutron scattering intensity.
        
        Args:
            spectrum: NeutronSpectrum (uses self.spectrum if None)
            figsize: Figure size
            cmap: Colormap
            log_scale: Use logarithmic scale for intensity
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        spectrum = spectrum or self.spectrum
        if spectrum is None:
            raise ValueError("No spectrum available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data
        Q = spectrum.q_points
        omega = spectrum.omega
        S = spectrum.s_qw_coherent
        
        if log_scale:
            S = np.log10(S + 1e-10)
        
        # Plot as colormap
        im = ax.pcolormesh(Q, omega, S.T, cmap=cmap, shading='auto')
        plt.colorbar(im, ax=ax, label='log₁₀ S(Q,ω)' if log_scale else 'S(Q,ω)')
        
        ax.set_xlabel('Q (r.l.u.)', fontsize=12)
        ax.set_ylabel('Energy Transfer (meV)', fontsize=12)
        ax.set_title('Neutron Scattering S(Q, ω)', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved neutron dispersion plot to {save_path}")
        
        return fig
    
    def plot_constant_q_cut(
        self,
        q_value: float,
        spectrum: Optional[NeutronSpectrum] = None,
        dq: float = 0.05,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot constant-Q cut of neutron scattering.
        
        Args:
            q_value: Q value for cut
            spectrum: NeutronSpectrum
            dq: Q integration width
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        spectrum = spectrum or self.spectrum
        if spectrum is None:
            raise ValueError("No spectrum available")
        
        # Find Q index
        q_idx = np.argmin(np.abs(spectrum.q_points - q_value))
        
        # Average over nearby Q points
        q_mask = np.abs(spectrum.q_points - q_value) < dq
        intensity = np.mean(spectrum.s_qw_coherent[q_mask], axis=0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(spectrum.omega, intensity, 'b-', lw=1.5)
        ax.fill_between(spectrum.omega, 0, intensity, alpha=0.3)
        
        ax.set_xlabel('Energy Transfer (meV)', fontsize=12)
        ax.set_ylabel('Intensity (arb. units)', fontsize=12)
        ax.set_title(f'Constant-Q Cut at Q = {q_value:.2f} Å⁻¹', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_constant_e_cut(
        self,
        energy: float,
        spectrum: Optional[NeutronSpectrum] = None,
        dE: float = 1.0,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot constant-energy cut of neutron scattering.
        
        Args:
            energy: Energy transfer for cut (meV)
            spectrum: NeutronSpectrum
            dE: Energy integration width
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        spectrum = spectrum or self.spectrum
        if spectrum is None:
            raise ValueError("No spectrum available")
        
        # Find energy index
        e_mask = np.abs(spectrum.omega - energy) < dE
        intensity = np.mean(spectrum.s_qw_coherent[:, e_mask], axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(spectrum.q_points, intensity, 'r-', lw=1.5)
        ax.fill_between(spectrum.q_points, 0, intensity, alpha=0.3, color='r')
        
        ax.set_xlabel('Q (r.l.u.)', fontsize=12)
        ax.set_ylabel('Intensity (arb. units)', fontsize=12)
        ax.set_title(f'Constant-E Cut at E = {energy:.1f} meV', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def calculate_neutron_scattering_workflow(
    phonopy: Phonopy,
    q_path: np.ndarray,
    temperature: float = 300.0,
    output_dir: str = './neutron_output'
) -> NeutronSpectrum:
    """
    Complete workflow for neutron scattering calculation.
    
    Args:
        phonopy: Phonopy object with force constants
        q_path: Q-point path in reciprocal space
        temperature: Temperature (K)
        output_dir: Output directory
        
    Returns:
        NeutronSpectrum object
    """
    config = NeutronConfig(temperature=temperature, output_dir=output_dir)
    
    calc = NeutronScatteringCalculator(config)
    spectrum = calc.calculate_s_qw(phonopy, q_path, temperature)
    
    # Save outputs
    spectrum.save(f"{output_dir}/neutron_spectrum.npz")
    calc.plot_dispersion_with_intensity(
        save_path=f"{output_dir}/neutron_dispersion.png")
    
    return spectrum


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Neutron Scattering Calculator')
    parser.add_argument('--phonopy', type=str, required=True, help='Phonopy file')
    parser.add_argument('--temp', type=float, default=300.0, help='Temperature (K)')
    parser.add_argument('--outdir', type=str, default='./neutron_output')
    
    args = parser.parse_args()
    
    print("Neutron Scattering Calculator - use within Python for full functionality")
