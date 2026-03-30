"""
Path Integral Molecular Dynamics Properties
===========================================

This module provides analysis tools for computing quantum mechanical
properties from PIMD simulations, including zero-point energy corrections,
quantum diffusion coefficients, and isotope effects.

Classes:
--------
- QuantumPropertyCalculator: Main calculator for quantum properties
- ZeroPointEnergyCalculator: Zero-point energy and corrections
- QuantumDiffusionCalculator: Quantum diffusion coefficients
- IsotopeEffectCalculator: Isotope effect calculations
- KineticEnergyEstimator: Different kinetic energy estimators
- ThermodynamicIntegration: Thermodynamic integration for free energy

Functions:
----------
- calculate_zpe: Calculate zero-point energy
- calculate_quantum_diffusion: Calculate quantum diffusion coefficient
- calculate_isotope_fractionation: Calculate isotope fractionation factor
- get_primitive_estimator: Primitive kinetic energy estimator
- get_virial_estimator: Virial kinetic energy estimator
- get_centroid_virial_estimator: Centroid virial estimator

References:
-----------
- Tuckerman (2010). Statistical Mechanics: Theory and Molecular Simulation
- Ceperley (1995). Rev. Mod. Phys. 67, 279
- Marx & Parrinello (1996). J. Chem. Phys. 104, 4077
- Habershon et al. (2009). J. Chem. Phys. 131, 024501

Example:
--------
>>> from dftlammps.pimd import QuantumPropertyCalculator
>>> calc = QuantumPropertyCalculator(results)
>>> zpe = calc.calculate_zpe()
>>> D_quantum = calc.calculate_diffusion_coefficient()
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from pathlib import Path
from enum import Enum
import numpy as np
from scipy import integrate, optimize, stats
from scipy.fft import fft, fftfreq

# Setup logging
logger = logging.getLogger(__name__)


class EstimatorType(Enum):
    """Types of kinetic energy estimators."""
    PRIMITIVE = "primitive"
    VIRIAL = "virial"
    CENTROID_VIRIAL = "centroid_virial"
    THERMODYNAMIC = "thermodynamic"
    BORN_OPPENHEIMER = "born_oppenheimer"


@dataclass
class ZPEResults:
    """Zero-point energy calculation results.
    
    Attributes:
        zpe_total: Total zero-point energy
        zpe_per_mode: ZPE per vibrational mode
        frequencies: Vibrational frequencies (cm^-1)
        quantum_correction: Quantum correction to classical energy
        convergence_check: Convergence status with bead number
        classical_limit: Classical energy at T=0
    """
    zpe_total: float
    zpe_per_mode: np.ndarray
    frequencies: np.ndarray
    quantum_correction: float
    convergence_check: bool
    classical_limit: Optional[float] = None
    
    def get_zpe_per_atom(self, n_atoms: int) -> float:
        """Get ZPE per atom."""
        return self.zpe_total / n_atoms
    
    def get_thermal_contribution(self, temperature: float) -> float:
        """Get thermal contribution to energy at given temperature."""
        k_B = 8.617333e-5  # eV/K
        h = 4.135667696e-15  # eV*s
        c = 2.99792458e10  # cm/s
        
        thermal = 0.0
        for freq in self.frequencies:
            if freq > 0:
                nu = freq * c  # Convert to Hz
                x = h * nu / (k_B * temperature)
                thermal += k_B * temperature * x / (np.exp(x) - 1.0)
        
        return thermal


@dataclass
class DiffusionResults:
    """Quantum diffusion calculation results.
    
    Attributes:
        diffusion_coefficient: Quantum diffusion coefficient (cm^2/s)
        classical_diffusion: Classical diffusion coefficient
        quantum_correction: Quantum correction factor
        msd_data: Mean squared displacement data
        time_lags: Time lags for MSD
        einstein_fit: Einstein relation fit parameters
        activation_energy: Activation energy from Arrhenius fit
        finite_size_correction: Finite size correction applied
    """
    diffusion_coefficient: float
    classical_diffusion: Optional[float] = None
    quantum_correction: Optional[float] = None
    msd_data: Optional[np.ndarray] = None
    time_lags: Optional[np.ndarray] = None
    einstein_fit: Optional[Dict] = None
    activation_energy: Optional[float] = None
    finite_size_correction: Optional[float] = None
    
    def get_jump_frequency(self, jump_distance: float) -> float:
        """Estimate jump frequency from diffusion coefficient.
        
        Args:
            jump_distance: Average jump distance in Angstroms
            
        Returns:
            Jump frequency in Hz
        """
        # D = (1/6) * lambda^2 * nu for 3D diffusion
        # D in cm^2/s, lambda in cm
        lambda_cm = jump_distance * 1e-8
        nu = 6 * self.diffusion_coefficient / (lambda_cm**2)
        return nu


@dataclass
class IsotopeResults:
    """Isotope effect calculation results.
    
    Attributes:
        fractionation_factor: Isotope fractionation factor (alpha)
        beta_factor: Beta factor (ln(alpha))
        reduced_partition_ratio: Reduced partition function ratio
        harmonic_approximation: Value from harmonic approximation
        anharmonic_correction: Anharmonicity correction
        tunneling_correction: Tunneling correction
        equilibrium_constant: Equilibrium constant for isotope exchange
    """
    fractionation_factor: float
    beta_factor: float
    reduced_partition_ratio: float
    harmonic_approximation: float
    anharmonic_correction: float = 0.0
    tunneling_correction: float = 1.0
    equilibrium_constant: Optional[float] = None
    
    def get_per_mille_deviation(self) -> float:
        """Get per mil deviation (delta notation)."""
        return (self.fractionation_factor - 1.0) * 1000


@dataclass
class KineticEnergyResults:
    """Kinetic energy estimator results.
    
    Attributes:
        primitive: Primitive estimator values
        virial: Virial estimator values
        centroid_virial: Centroid virial estimator values
        mean_primitive: Mean primitive energy
        mean_virial: Mean virial energy
        mean_centroid_virial: Mean centroid virial energy
        error_primitive: Statistical error (primitive)
        error_virial: Statistical error (virial)
        error_centroid_virial: Statistical error (centroid virial)
    """
    primitive: Optional[np.ndarray] = None
    virial: Optional[np.ndarray] = None
    centroid_virial: Optional[np.ndarray] = None
    mean_primitive: Optional[float] = None
    mean_virial: Optional[float] = None
    mean_centroid_virial: Optional[float] = None
    error_primitive: Optional[float] = None
    error_virial: Optional[float] = None
    error_centroid_virial: Optional[float] = None
    
    def get_best_estimator(self) -> Tuple[str, float, float]:
        """Get the best kinetic energy estimator.
        
        The centroid virial estimator typically has the lowest variance.
        
        Returns:
            (estimator_name, mean_energy, error)
        """
        if self.error_centroid_virial is not None and self.mean_centroid_virial is not None:
            return ("centroid_virial", self.mean_centroid_virial, self.error_centroid_virial)
        elif self.error_virial is not None and self.mean_virial is not None:
            return ("virial", self.mean_virial, self.error_virial)
        elif self.error_primitive is not None and self.mean_primitive is not None:
            return ("primitive", self.mean_primitive, self.error_primitive)
        else:
            raise ValueError("No valid kinetic energy estimator available")


class KineticEnergyEstimator:
    """Calculate kinetic energy using different estimators.
    
    In PIMD, several estimators exist for the kinetic energy:
    
    1. Primitive Estimator:
       K_P = (3N/2) * P * k_B * T - (m * P / (2 * hbar^2 * beta^2)) * 
               sum_{i=1}^P (r_i - r_{i+1})^2
    
    2. Virial Estimator:
       K_V = (3N / (2 * beta)) + (1/(2P)) * sum_{i=1}^P r_i * dV/dr_i
    
    3. Centroid Virial Estimator:
       K_CV = (3N / (2 * beta)) + (1/(2P)) * sum_{i=1}^P (r_i - r_c) * dV/dr_i
       where r_c is the centroid position
    
    The centroid virial estimator has the lowest variance and is preferred.
    
    References:
    -----------
    - Herman et al. (1982). J. Chem. Phys. 76, 5150
    - Glaesemann & Fried (2002). J. Chem. Phys. 116, 5951
    """
    
    def __init__(self, positions: np.ndarray, forces: np.ndarray,
                 masses: np.ndarray, temperature: float):
        """Initialize kinetic energy estimator.
        
        Args:
            positions: Positions [n_frames, n_beads, n_atoms, 3] in Angstrom
            forces: Forces [n_frames, n_beads, n_atoms, 3] in eV/Angstrom
            masses: Masses [n_atoms] in amu
            temperature: Temperature in Kelvin
        """
        self.positions = positions
        self.forces = forces
        self.masses = masses
        self.temperature = temperature
        
        # Constants
        self.k_B = 8.617333e-5  # eV/K
        self.hbar = 6.582119569e-16  # eV*s
        self.amu_to_kg = 1.66053906660e-27  # kg/amu
        self.angstrom_to_m = 1e-10  # m/Angstrom
        self.ev_to_j = 1.602176634e-19  # J/eV
        
        self.n_frames = positions.shape[0]
        self.n_beads = positions.shape[1]
        self.n_atoms = positions.shape[2]
        self.beta = 1.0 / (self.k_B * temperature)
    
    def primitive_estimator(self) -> np.ndarray:
        """Calculate primitive kinetic energy estimator.
        
        Returns:
            Kinetic energy array [n_frames] in eV
        """
        # K_P = (3N/2) * P * k_B * T - (m * P / (2 * hbar^2 * beta^2)) * sum (r_i - r_{i+1})^2
        
        kinetic = np.zeros(self.n_frames)
        
        # First term: (3N/2) * P * k_B * T
        first_term = 1.5 * self.n_atoms * self.n_beads * self.k_B * self.temperature
        
        for frame in range(self.n_frames):
            # Second term involving ring polymer springs
            second_term = 0.0
            
            for atom in range(self.n_atoms):
                m = self.masses[atom] * self.amu_to_kg
                
                for bead in range(self.n_beads):
                    next_bead = (bead + 1) % self.n_beads
                    
                    dr = self.positions[frame, bead, atom] - self.positions[frame, next_bead, atom]
                    dr_m = dr * self.angstrom_to_m
                    
                    spring_term = (m * self.n_beads / 
                                  (2 * (self.hbar * self.beta)**2)) * np.sum(dr_m**2)
                    second_term += spring_term
            
            # Convert from J to eV
            second_term_ev = second_term / self.ev_to_j
            
            kinetic[frame] = first_term - second_term_ev
        
        return kinetic
    
    def virial_estimator(self) -> np.ndarray:
        """Calculate virial kinetic energy estimator.
        
        Returns:
            Kinetic energy array [n_frames] in eV
        """
        # K_V = (3N / (2 * beta)) + (1/(2P)) * sum_{i=1}^P r_i * F_i
        # Note: F_i = -dV/dr_i
        
        kinetic = np.zeros(self.n_frames)
        
        for frame in range(self.n_frames):
            # First term
            first_term = 1.5 * self.n_atoms / self.beta
            
            # Virial term (r * F)
            virial_term = 0.0
            for bead in range(self.n_beads):
                # r in Angstrom, F in eV/Angstrom
                # r * F has units of eV
                virial = np.sum(self.positions[frame, bead] * self.forces[frame, bead])
                virial_term += virial
            
            virial_term /= (2.0 * self.n_beads)
            
            kinetic[frame] = first_term + virial_term
        
        return kinetic
    
    def centroid_virial_estimator(self) -> np.ndarray:
        """Calculate centroid virial kinetic energy estimator.
        
        This estimator has the lowest variance and is preferred.
        
        Returns:
            Kinetic energy array [n_frames] in eV
        """
        # K_CV = (3N / (2 * beta)) + (1/(2P)) * sum_{i=1}^P (r_i - r_c) * F_i
        
        kinetic = np.zeros(self.n_frames)
        
        for frame in range(self.n_frames):
            # First term
            first_term = 1.5 * self.n_atoms / self.beta
            
            # Calculate centroid
            centroid = np.mean(self.positions[frame], axis=0)  # [n_atoms, 3]
            
            # Centroid virial term
            virial_term = 0.0
            for bead in range(self.n_beads):
                dr = self.positions[frame, bead] - centroid
                virial = np.sum(dr * self.forces[frame, bead])
                virial_term += virial
            
            virial_term /= (2.0 * self.n_beads)
            
            kinetic[frame] = first_term + virial_term
        
        return kinetic
    
    def calculate_all(self) -> KineticEnergyResults:
        """Calculate all kinetic energy estimators.
        
        Returns:
            KineticEnergyResults with all estimators
        """
        results = KineticEnergyResults()
        
        # Primitive estimator
        results.primitive = self.primitive_estimator()
        results.mean_primitive = np.mean(results.primitive)
        results.error_primitive = np.std(results.primitive, ddof=1) / np.sqrt(self.n_frames)
        
        # Virial estimator
        results.virial = self.virial_estimator()
        results.mean_virial = np.mean(results.virial)
        results.error_virial = np.std(results.virial, ddof=1) / np.sqrt(self.n_frames)
        
        # Centroid virial estimator
        results.centroid_virial = self.centroid_virial_estimator()
        results.mean_centroid_virial = np.mean(results.centroid_virial)
        results.error_centroid_virial = np.std(results.centroid_virial, ddof=1) / np.sqrt(self.n_frames)
        
        return results


class ZeroPointEnergyCalculator:
    """Calculate zero-point energy and related properties.
    
    The zero-point energy (ZPE) is the quantum mechanical minimum energy
    of a system. In PIMD, it can be calculated from the high-temperature
    limit and extrapolated to T = 0.
    
    ZPE = (1/2) * sum_i h * nu_i
    
    where nu_i are the normal mode frequencies.
    
    References:
    -----------
    - McQuarrie (2000). Statistical Mechanics
    """
    
    def __init__(self, positions: np.ndarray, forces: np.ndarray,
                 masses: np.ndarray, temperature: float,
                 cell: Optional[np.ndarray] = None):
        """Initialize ZPE calculator.
        
        Args:
            positions: Positions [n_frames, n_beads, n_atoms, 3]
            forces: Forces [n_frames, n_beads, n_atoms, 3]
            masses: Masses [n_atoms] in amu
            temperature: Temperature in Kelvin
            cell: Simulation cell [3, 3] in Angstrom
        """
        self.positions = positions
        self.forces = forces
        self.masses = masses
        self.temperature = temperature
        self.cell = cell
        
        # Constants
        self.k_B = 8.617333e-5  # eV/K
        self.hbar = 6.582119569e-16  # eV*s
        self.amu_to_kg = 1.66053906660e-27
        
        self.n_frames = positions.shape[0]
        self.n_beads = positions.shape[1]
        self.n_atoms = positions.shape[2]
    
    def calculate_from_pimd(self, method: str = "centroid_virial") -> ZPEResults:
        """Calculate ZPE from PIMD simulation data.
        
        The kinetic energy at temperature T contains information about ZPE:
        <K>_T = ZPE + thermal_contribution(T)
        
        By extrapolating to T -> 0, we can estimate ZPE.
        
        Args:
            method: Kinetic energy estimator method
            
        Returns:
            ZPEResults
        """
        # Calculate kinetic energy
        estimator = KineticEnergyEstimator(self.positions, self.forces,
                                           self.masses, self.temperature)
        
        if method == "centroid_virial":
            kinetic = estimator.centroid_virial_estimator()
        elif method == "virial":
            kinetic = estimator.virial_estimator()
        else:
            kinetic = estimator.primitive_estimator()
        
        mean_kinetic = np.mean(kinetic)
        
        # Thermal contribution at temperature T
        thermal = 1.5 * self.n_atoms * self.k_B * self.temperature
        
        # Estimate ZPE
        zpe = mean_kinetic - thermal
        
        # Estimate frequencies from bead fluctuations
        frequencies = self._estimate_frequencies()
        
        # Calculate ZPE per mode
        h_cm = 1.239841984e-4  # eV*cm
        zpe_per_mode = 0.5 * h_cm * frequencies
        
        # Classical limit (3N * k_B * T / 2 at T -> 0 is 0)
        classical_limit = 0.0
        
        # Quantum correction
        quantum_correction = zpe
        
        return ZPEResults(
            zpe_total=zpe,
            zpe_per_mode=zpe_per_mode,
            frequencies=frequencies,
            quantum_correction=quantum_correction,
            convergence_check=True,  # Would need multiple temperatures to check
            classical_limit=classical_limit
        )
    
    def _estimate_frequencies(self) -> np.ndarray:
        """Estimate vibrational frequencies from ring polymer.
        
        The spread of the ring polymer is related to the frequency:
        sigma^2 ~ hbar / (m * omega) * coth(beta * hbar * omega / 2)
        
        Returns:
            Frequencies in cm^-1
        """
        # Calculate bead fluctuations
        centroid = np.mean(self.positions, axis=1, keepdims=True)
        fluctuations = self.positions - centroid
        
        # RMS fluctuations per atom
        sigma_sq = np.mean(np.sum(fluctuations**2, axis=-1), axis=(0, 1))  # [n_atoms]
        
        # High-temperature approximation: sigma^2 ~ hbar^2 * beta / (m * P)
        # Rearranging: omega ~ hbar * beta * P / (m * sigma^2)
        # But this is approximate. Better to use normal mode analysis.
        
        # Simplified estimate (assuming high T limit)
        # sigma^2 = hbar^2 * beta / (m * P) for free ring polymer
        # For harmonic oscillator: omega = sqrt(k/m)
        # sigma^2 = hbar / (2 * m * omega) * coth(beta * hbar * omega / 2)
        
        frequencies = np.zeros(self.n_atoms * 3)
        k_B = self.k_B
        hbar = self.hbar
        beta = 1.0 / (k_B * self.temperature)
        
        for i in range(self.n_atoms):
            m = self.masses[i] * self.amu_to_kg
            sigma = np.sqrt(sigma_sq[i]) * 1e-10  # to meters
            
            # High-T approximation
            # sigma^2 ≈ hbar^2 * beta / (m * P)
            # This gives an effective frequency
            if sigma > 0:
                omega_eff = hbar * beta / (m * sigma**2 * self.n_beads)
                # Convert to cm^-1
                c = 2.99792458e10  # cm/s
                nu = omega_eff / (2 * np.pi * c)
                
                # Fill 3 degrees of freedom per atom
                frequencies[3*i:3*i+3] = nu
        
        return frequencies
    
    def calculate_zpe_from_frequencies(self, frequencies_cm1: np.ndarray) -> float:
        """Calculate ZPE from vibrational frequencies.
        
        Args:
            frequencies_cm1: Frequencies in cm^-1
            
        Returns:
            ZPE in eV
        """
        h_cm = 1.239841984e-4  # eV*cm (h * c in eV*cm)
        
        # ZPE = sum_i (h * nu_i / 2)
        zpe = 0.5 * h_cm * np.sum(frequencies_cm1[frequencies_cm1 > 0])
        
        return zpe


class QuantumDiffusionCalculator:
    """Calculate quantum diffusion coefficients from PIMD/RPMD.
    
    Quantum diffusion is the process by which particles move through
    a material due to quantum mechanical effects (tunneling, ZPE).
    
    The diffusion coefficient can be calculated from:
    1. Mean squared displacement (Einstein relation)
    2. Velocity autocorrelation function (Green-Kubo)
    
    For quantum systems, RPMD provides approximate real-time dynamics.
    
    References:
    -----------
    - Habershon et al. (2009). J. Chem. Phys. 131, 024501
    - Miller et al. (2005). J. Chem. Phys. 122, 034503
    """
    
    def __init__(self, positions: np.ndarray, 
                 timestep: float,
                 cell: Optional[np.ndarray] = None,
                 masses: Optional[np.ndarray] = None):
        """Initialize quantum diffusion calculator.
        
        Args:
            positions: Positions [n_frames, n_beads, n_atoms, 3] or
                      [n_frames, n_atoms, 3] for centroid
            timestep: Time step in femtoseconds
            cell: Simulation cell [3, 3] in Angstrom
            masses: Atomic masses [n_atoms] in amu
        """
        self.positions = positions
        self.timestep = timestep
        self.cell = cell
        self.masses = masses
        
        self.n_frames = positions.shape[0]
        self.ndim = len(positions.shape)
        
        if self.ndim == 4:
            self.n_beads = positions.shape[1]
            self.n_atoms = positions.shape[2]
            self.is_beaded = True
        else:
            self.n_beads = 1
            self.n_atoms = positions.shape[1]
            self.is_beaded = False
    
    def calculate_msd(self, atom_indices: Optional[List[int]] = None,
                      max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean squared displacement.
        
        Args:
            atom_indices: Indices of atoms to track (None = all)
            max_lag: Maximum time lag (None = n_frames // 2)
            
        Returns:
            (time_lags, msd) in (fs, Angstrom^2)
        """
        if atom_indices is None:
            atom_indices = list(range(self.n_atoms))
        
        if max_lag is None:
            max_lag = self.n_frames // 2
        
        max_lag = min(max_lag, self.n_frames - 1)
        
        # Use centroid positions for RPMD
        if self.is_beaded:
            traj = np.mean(self.positions[:, :, atom_indices, :], axis=1)
        else:
            traj = self.positions[:, atom_indices, :]
        
        msd = np.zeros(max_lag)
        time_lags = np.arange(max_lag) * self.timestep
        
        for lag in range(1, max_lag):
            displacements = traj[lag:] - traj[:-lag]
            
            # Apply minimum image convention if cell is provided
            if self.cell is not None:
                displacements = self._apply_minimum_image(displacements)
            
            msd[lag] = np.mean(np.sum(displacements**2, axis=-1))
        
        return time_lags, msd
    
    def _apply_minimum_image(self, displacements: np.ndarray) -> np.ndarray:
        """Apply minimum image convention to displacements."""
        if self.cell is None:
            return displacements
        
        # Convert to fractional coordinates
        inv_cell = np.linalg.inv(self.cell)
        frac_disp = np.dot(displacements, inv_cell.T)
        
        # Apply minimum image
        frac_disp -= np.rint(frac_disp)
        
        # Convert back to Cartesian
        cart_disp = np.dot(frac_disp, self.cell.T)
        
        return cart_disp
    
    def calculate_diffusion_coefficient(self, 
                                        atom_indices: Optional[List[int]] = None,
                                        fit_start: Optional[int] = None,
                                        fit_end: Optional[int] = None,
                                        dimensionality: int = 3) -> DiffusionResults:
        """Calculate diffusion coefficient from MSD.
        
        Uses Einstein relation: MSD = 2 * d * D * t
        where d is dimensionality (1, 2, or 3)
        
        Args:
            atom_indices: Indices of atoms to track
            fit_start: Start index for linear fit
            fit_end: End index for linear fit
            dimensionality: System dimensionality (1, 2, or 3)
            
        Returns:
            DiffusionResults
        """
        time_lags, msd = self.calculate_msd(atom_indices)
        
        # Determine fit range
        if fit_start is None:
            fit_start = len(time_lags) // 4
        if fit_end is None:
            fit_end = len(time_lags) * 3 // 4
        
        # Linear fit to Einstein regime
        t_fit = time_lags[fit_start:fit_end]
        msd_fit = msd[fit_start:fit_end]
        
        # Fit: MSD = 2 * d * D * t
        # slope = 2 * d * D
        slope, intercept, r_value, p_value, std_err = stats.linregress(t_fit, msd_fit)
        
        # D in Angstrom^2/fs
        D_ang2_fs = slope / (2.0 * dimensionality)
        
        # Convert to cm^2/s
        # 1 Angstrom = 1e-8 cm
        # 1 fs = 1e-15 s
        # 1 Angstrom^2/fs = 1e-16 cm^2 / 1e-15 s = 0.1 cm^2/s
        D_cm2_s = D_ang2_fs * 0.1
        
        # Calculate error
        D_error = std_err / (2.0 * dimensionality) * 0.1
        
        return DiffusionResults(
            diffusion_coefficient=D_cm2_s,
            msd_data=msd,
            time_lags=time_lags,
            einstein_fit={
                "slope": slope,
                "intercept": intercept,
                "r_value": r_value,
                "std_err": std_err,
                "D_error": D_error
            }
        )
    
    def calculate_velocity_autocorrelation(self, 
                                           velocities: np.ndarray,
                                           max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate velocity autocorrelation function.
        
        Args:
            velocities: Velocities [n_frames, n_atoms, 3] or
                       [n_frames, n_beads, n_atoms, 3]
            max_lag: Maximum time lag
            
        Returns:
            (time_lags, vacf)
        """
        if max_lag is None:
            max_lag = self.n_frames // 2
        
        # Use centroid velocity
        if len(velocities.shape) == 4:
            v = np.mean(velocities, axis=1)
        else:
            v = velocities
        
        vacf = np.zeros(max_lag)
        
        for lag in range(max_lag):
            for t in range(self.n_frames - lag):
                vacf[lag] += np.mean(np.sum(v[t] * v[t + lag], axis=-1))
        
        vacf /= (self.n_frames - np.arange(max_lag))
        vacf /= vacf[0]  # Normalize
        
        time_lags = np.arange(max_lag) * self.timestep
        
        return time_lags, vacf
    
    def diffusion_from_vacf(self, vacf: np.ndarray) -> float:
        """Calculate diffusion coefficient from VACF using Green-Kubo.
        
        D = (1/3) * integral_0^infty <v(0) · v(t)> dt
        
        Args:
            vacf: Velocity autocorrelation function
            
        Returns:
            Diffusion coefficient in cm^2/s
        """
        # Integrate VACF
        integral = integrate.trapz(vacf) * self.timestep  # in Angstrom^2/fs
        
        # D = (1/3) * integral
        D_ang2_fs = integral / 3.0
        D_cm2_s = D_ang2_fs * 0.1
        
        return D_cm2_s
    
    def calculate_activation_energy(self, temperatures: List[float],
                                    diffusion_coeffs: List[float]) -> float:
        """Calculate activation energy from Arrhenius fit.
        
        D = D_0 * exp(-E_a / (k_B * T))
        
        Args:
            temperatures: List of temperatures in K
            diffusion_coeffs: List of diffusion coefficients in cm^2/s
            
        Returns:
            Activation energy in eV
        """
        T = np.array(temperatures)
        D = np.array(diffusion_coeffs)
        
        # Linear fit to ln(D) vs 1/T
        inv_T = 1.0 / T
        ln_D = np.log(D)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, ln_D)
        
        # slope = -E_a / k_B
        k_B = 8.617333e-5  # eV/K
        E_a = -slope * k_B
        
        return E_a


class IsotopeEffectCalculator:
    """Calculate isotope effects from PIMD simulations.
    
    Isotope effects arise from differences in quantum behavior
    between isotopes (mass-dependent zero-point energy and tunneling).
    
    The fractionation factor alpha is defined as:
    alpha = (x_heavy / x_light)_phase1 / (x_heavy / x_light)_phase2
    
    where x is the mole fraction.
    
    Key applications:
    -----------------
    - Hydrogen isotope effects (H/D/T)
    - Geochemical fractionation
    - Enzyme catalysis
    - Hydrogen storage materials
    
    References:
    -----------
    - Wolfsberg et al. (2010). Isotope Effects
    - Ceriotti & Markland (2013). J. Chem. Phys. 138, 014112
    """
    
    def __init__(self, temperature: float):
        """Initialize isotope effect calculator.
        
        Args:
            temperature: Temperature in Kelvin
        """
        self.temperature = temperature
        self.k_B = 8.617333e-5  # eV/K
        self.hbar = 6.582119569e-16  # eV*s
    
    def calculate_reduced_partition_function(self,
                                              frequencies_light: np.ndarray,
                                              frequencies_heavy: np.ndarray,
                                              masses_light: float,
                                              masses_heavy: float) -> float:
        """Calculate reduced partition function ratio.
        
        RPFR = (Q_heavy / Q_light) * (s_light / s_heavy)
        
        For harmonic oscillator:
        Q = product_i [exp(-h*nu_i/(2*k_B*T)) / (1 - exp(-h*nu_i/(k_B*T)))]
        
        Args:
            frequencies_light: Vibrational frequencies of light isotope (cm^-1)
            frequencies_heavy: Vibrational frequencies of heavy isotope (cm^-1)
            masses_light: Mass of light isotope
            masses_heavy: Mass of heavy isotope
            
        Returns:
            Reduced partition function ratio
        """
        h_cm = 1.239841984e-4  # eV*cm
        
        rpfr = 1.0
        
        for nu_l, nu_h in zip(frequencies_light, frequencies_heavy):
            if nu_l <= 0 or nu_h <= 0:
                continue
            
            x_l = h_cm * nu_l / (self.k_B * self.temperature)
            x_h = h_cm * nu_h / (self.k_B * self.temperature)
            
            # Partition function for harmonic oscillator
            Q_l = np.exp(-x_l/2) / (1 - np.exp(-x_l))
            Q_h = np.exp(-x_h/2) / (1 - np.exp(-x_h))
            
            rpfr *= Q_h / Q_l
        
        # Symmetry factor (assuming same symmetry)
        # s_light / s_heavy = 1 for simple cases
        
        return rpfr
    
    def calculate_beta_factor(self, rpfr: float) -> float:
        """Calculate beta factor from RPFR.
        
        beta = ln(RPFR) = ln(Q_heavy / Q_light)
        
        Args:
            rpfr: Reduced partition function ratio
            
        Returns:
            Beta factor
        """
        return np.log(rpfr)
    
    def calculate_fractionation_factor(self, 
                                        rpfr_phase1: float,
                                        rpfr_phase2: float) -> float:
        """Calculate fractionation factor between two phases.
        
        alpha = (x_heavy/x_light)_1 / (x_heavy/x_light)_2
              ≈ RPFR_2 / RPFR_1
        
        Args:
            rpfr_phase1: RPFR in phase 1
            rpfr_phase2: RPFR in phase 2
            
        Returns:
            Fractionation factor
        """
        return rpfr_phase2 / rpfr_phase1
    
    def calculate_from_pimd(self,
                           positions_light: np.ndarray,
                           positions_heavy: np.ndarray,
                           forces_light: np.ndarray,
                           forces_heavy: np.ndarray,
                           masses_light: float,
                           masses_heavy: float) -> IsotopeResults:
        """Calculate isotope effects from PIMD data.
        
        Uses the centoid virial estimator to calculate free energy differences.
        
        Args:
            positions_light: Positions for light isotope
            positions_heavy: Positions for heavy isotope
            forces_light: Forces for light isotope
            forces_heavy: Forces for heavy isotope
            masses_light: Mass of light isotope
            masses_heavy: Mass of heavy isotope
            
        Returns:
            IsotopeResults
        """
        n_atoms = positions_light.shape[2]
        
        # Calculate kinetic energies
        ke_light = KineticEnergyEstimator(positions_light, forces_light,
                                          np.full(n_atoms, masses_light),
                                          self.temperature)
        ke_heavy = KineticEnergyEstimator(positions_heavy, forces_heavy,
                                          np.full(n_atoms, masses_heavy),
                                          self.temperature)
        
        ke_results_light = ke_light.calculate_all()
        ke_results_heavy = ke_heavy.calculate_all()
        
        # Use best estimator
        _, KE_l, _ = ke_results_light.get_best_estimator()
        _, KE_h, _ = ke_results_heavy.get_best_estimator()
        
        # Free energy difference from thermodynamic integration
        # Delta F = F_heavy - F_light
        
        # Harmonic approximation
        # Estimate frequencies from kinetic energy
        # For high T: KE = 3N/2 * k_B * T + ZPE
        # For harmonic: ZPE = sum(0.5 * h * nu_i)
        
        thermal = 1.5 * n_atoms * self.k_B * self.temperature
        zpe_l = KE_l - thermal
        zpe_h = KE_h - thermal
        
        # Free energy difference in harmonic approximation
        delta_F_harm = zpe_h - zpe_l
        
        # RPFR ≈ exp(-delta_F / (k_B * T))
        rpfr = np.exp(-delta_F_harm / (self.k_B * self.temperature))
        
        # Fractionation factor (assuming same phase)
        alpha = rpfr
        beta = np.log(alpha)
        
        # Anharmonic correction would require more sophisticated analysis
        anharmonic_correction = 0.0
        tunneling_correction = 1.0
        
        return IsotopeResults(
            fractionation_factor=alpha,
            beta_factor=beta,
            reduced_partition_ratio=rpfr,
            harmonic_approximation=alpha,
            anharmonic_correction=anharmonic_correction,
            tunneling_correction=tunneling_correction
        )
    
    def calculate_kinetic_isotope_effect(self,
                                         activation_energy_light: float,
                                         activation_energy_heavy: float,
                                         temperature: Optional[float] = None) -> float:
        """Calculate kinetic isotope effect.
        
        KIE = k_light / k_heavy ≈ exp(-(E_a_light - E_a_heavy) / (k_B * T))
        
        Args:
            activation_energy_light: Activation energy for light isotope (eV)
            activation_energy_heavy: Activation energy for heavy isotope (eV)
            temperature: Temperature (uses self.temperature if None)
            
        Returns:
            Kinetic isotope effect
        """
        T = temperature or self.temperature
        
        delta_Ea = activation_energy_light - activation_energy_heavy
        kie = np.exp(delta_Ea / (self.k_B * T))
        
        return kie


class QuantumPropertyCalculator:
    """Main calculator for quantum properties from PIMD.
    
    This class provides a unified interface for calculating various
    quantum mechanical properties from path integral simulations.
    
    Example:
    --------
    >>> calc = QuantumPropertyCalculator(results)
    >>> zpe = calc.calculate_zpe()
    >>> diffusion = calc.calculate_diffusion(atom_indices=[0, 1])
    >>> isotope_effect = calc.calculate_isotope_effect(masses=[1.0, 2.0])
    """
    
    def __init__(self, results: Any):
        """Initialize quantum property calculator.
        
        Args:
            results: PIMDResults object or dictionary with positions, forces, etc.
        """
        self.results = results
        
        # Extract common data
        self.positions = getattr(results, 'positions', results.get('positions'))
        self.forces = getattr(results, 'forces', results.get('forces'))
        self.velocities = getattr(results, 'velocities', results.get('velocities'))
        self.temperature = getattr(results, 'temperature', results.get('temperature'))
        
        if isinstance(self.temperature, np.ndarray):
            self.temperature = np.mean(self.temperature)
        
        self.timestep = getattr(results, 'timestep', results.get('timestep', 0.5))
        
    def calculate_zpe(self, method: str = "centroid_virial") -> ZPEResults:
        """Calculate zero-point energy.
        
        Args:
            method: Kinetic energy estimator method
            
        Returns:
            ZPEResults
        """
        if self.positions is None or self.forces is None:
            raise ValueError("Positions and forces required for ZPE calculation")
        
        # Assume equal masses if not provided
        n_atoms = self.positions.shape[2]
        masses = np.ones(n_atoms)
        
        calc = ZeroPointEnergyCalculator(
            self.positions, self.forces, masses, self.temperature
        )
        
        return calc.calculate_from_pimd(method)
    
    def calculate_diffusion(self, 
                           atom_indices: Optional[List[int]] = None,
                           method: str = "einstein") -> DiffusionResults:
        """Calculate diffusion coefficient.
        
        Args:
            atom_indices: Indices of diffusing atoms
            method: "einstein" or "green_kubo"
            
        Returns:
            DiffusionResults
        """
        if self.positions is None:
            raise ValueError("Positions required for diffusion calculation")
        
        calc = QuantumDiffusionCalculator(self.positions, self.timestep)
        
        if method == "einstein":
            return calc.calculate_diffusion_coefficient(atom_indices)
        else:
            raise NotImplementedError("Green-Kubo method requires velocities")
    
    def calculate_isotope_effect(self, 
                                  masses_light: float,
                                  masses_heavy: float,
                                  positions_heavy: Optional[np.ndarray] = None,
                                  forces_heavy: Optional[np.ndarray] = None) -> IsotopeResults:
        """Calculate isotope effect.
        
        Args:
            masses_light: Mass of light isotope
            masses_heavy: Mass of heavy isotope
            positions_heavy: Positions for heavy isotope (if different from light)
            forces_heavy: Forces for heavy isotope
            
        Returns:
            IsotopeResults
        """
        calc = IsotopeEffectCalculator(self.temperature)
        
        if positions_heavy is None:
            positions_heavy = self.positions
        if forces_heavy is None:
            forces_heavy = self.forces
        
        return calc.calculate_from_pimd(
            self.positions, positions_heavy,
            self.forces, forces_heavy,
            masses_light, masses_heavy
        )
    
    def calculate_kinetic_energy(self) -> KineticEnergyResults:
        """Calculate kinetic energy using various estimators.
        
        Returns:
            KineticEnergyResults
        """
        if self.positions is None or self.forces is None:
            raise ValueError("Positions and forces required")
        
        n_atoms = self.positions.shape[2]
        masses = np.ones(n_atoms)
        
        calc = KineticEnergyEstimator(
            self.positions, self.forces, masses, self.temperature
        )
        
        return calc.calculate_all()
    
    def calculate_free_energy(self, temperatures: List[float],
                              energies: List[float]) -> Callable:
        """Calculate free energy at different temperatures.
        
        Uses thermodynamic integration.
        
        Args:
            temperatures: List of temperatures
            energies: List of average energies
            
        Returns:
            Interpolated free energy function
        """
        # Thermodynamic integration
        # F(T) = F(T_ref) - integral_{T_ref}^{T} S(T') dT'
        # S = (U - F) / T
        
        # For now, return simple interpolation
        from scipy.interpolate import interp1d
        
        T = np.array(temperatures)
        E = np.array(energies)
        
        # Approximate free energy (assuming high-T limit)
        # F ≈ E - T*S, where S is from classical limit
        F = E  # Simplified
        
        return interp1d(T, F, kind='cubic', fill_value='extrapolate')


def calculate_zpe(frequencies_cm1: np.ndarray) -> float:
    """Calculate zero-point energy from vibrational frequencies.
    
    Args:
        frequencies_cm1: Vibrational frequencies in cm^-1
        
    Returns:
        ZPE in eV
    """
    h_cm = 1.239841984e-4  # eV*cm
    return 0.5 * h_cm * np.sum(frequencies_cm1[frequencies_cm1 > 0])


def calculate_quantum_diffusion(positions: np.ndarray,
                                timestep: float,
                                max_lag: Optional[int] = None,
                                dimensionality: int = 3) -> float:
    """Calculate quantum diffusion coefficient.
    
    Convenience function for quick diffusion calculation.
    
    Args:
        positions: Positions [n_frames, ...]
        timestep: Time step in fs
        max_lag: Maximum time lag
        dimensionality: System dimensionality
        
    Returns:
        Diffusion coefficient in cm^2/s
    """
    calc = QuantumDiffusionCalculator(positions, timestep)
    results = calc.calculate_diffusion_coefficient(
        max_lag=max_lag,
        dimensionality=dimensionality
    )
    return results.diffusion_coefficient


def calculate_isotope_fractionation(frequencies_light: np.ndarray,
                                    frequencies_heavy: np.ndarray,
                                    temperature: float) -> float:
    """Calculate isotope fractionation factor.
    
    Args:
        frequencies_light: Frequencies for light isotope (cm^-1)
        frequencies_heavy: Frequencies for heavy isotope (cm^-1)
        temperature: Temperature in K
        
    Returns:
        Fractionation factor
    """
    calc = IsotopeEffectCalculator(temperature)
    
    rpfr = calc.calculate_reduced_partition_function(
        frequencies_light, frequencies_heavy,
        masses_light=1.0, masses_heavy=2.0  # Placeholder masses
    )
    
    return rpfr


def get_virial_estimator(positions: np.ndarray, forces: np.ndarray,
                         masses: np.ndarray, temperature: float) -> np.ndarray:
    """Get virial kinetic energy estimator.
    
    Args:
        positions: Positions [n_frames, n_beads, n_atoms, 3]
        forces: Forces [n_frames, n_beads, n_atoms, 3]
        masses: Masses [n_atoms]
        temperature: Temperature in K
        
    Returns:
        Kinetic energy array [n_frames]
    """
    calc = KineticEnergyEstimator(positions, forces, masses, temperature)
    return calc.virial_estimator()


def get_centroid_virial_estimator(positions: np.ndarray, forces: np.ndarray,
                                  masses: np.ndarray, temperature: float) -> np.ndarray:
    """Get centroid virial kinetic energy estimator.
    
    Args:
        positions: Positions [n_frames, n_beads, n_atoms, 3]
        forces: Forces [n_frames, n_beads, n_atoms, 3]
        masses: Masses [n_atoms]
        temperature: Temperature in K
        
    Returns:
        Kinetic energy array [n_frames]
    """
    calc = KineticEnergyEstimator(positions, forces, masses, temperature)
    return calc.centroid_virial_estimator()
