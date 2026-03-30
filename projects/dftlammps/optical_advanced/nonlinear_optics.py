"""
Advanced Nonlinear Optical Properties Module
=============================================
Nonlinear optical response calculations including:
- Second Harmonic Generation (SHG)
- Optical Rotation / Circular Dichroism (CD)
- High Harmonic Generation (HHG)
- Nonlinear optical susceptibility tensors

References:
- Sipe & Ghahramani, Nonlinear optical response of semiconductors (1993)
- Hughes & Sipe, Nonlinear optical response of a multilayer composite (1996)
- Ando et al., Optical properties of carbon nanotubes (1998)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from scipy import integrate, interpolate
from scipy.special import jv
import warnings


class NonlinearOrder(Enum):
    """Order of nonlinear optical response."""
    SECOND = 2   # χ^(2)
    THIRD = 3    # χ^(3)
    FOURTH = 4   # χ^(4)
    FIFTH = 5    # χ^(5) for HHG


class LightPolarization(Enum):
    """Light polarization states."""
    LINEAR_X = "x"
    LINEAR_Y = "y"
    LINEAR_Z = "z"
    CIRCULAR_RIGHT = "sigma_plus"
    CIRCULAR_LEFT = "sigma_minus"
    ELLIPTICAL = "elliptical"


@dataclass
class OpticalParameters:
    """Parameters for optical calculations."""
    # Frequency range
    omega_min: float = 0.0    # eV
    omega_max: float = 10.0   # eV
    n_omega: int = 1000
    
    # Laser parameters
    laser_intensity: float = 1e10  # W/cm²
    pulse_duration: float = 50e-15  # s (50 fs)
    wavelength: float = 800e-9  # m (800 nm)
    
    # Material parameters
    bandgap: float = 1.5  # eV
    refractive_index: float = 3.0
    
    # Temperature
    temperature: float = 300  # K
    
    def photon_energy(self) -> float:
        """Calculate photon energy from wavelength."""
        h = 4.136e-15  # eV·s
        c = 2.998e8    # m/s
        return h * c / self.wavelength


class SecondHarmonicGeneration:
    """
    Second Harmonic Generation (SHG) calculator.
    
    χ^(2)_{ijk}(-2ω; ω, ω) describes frequency doubling.
    
    Reference: Sipe & Ghahramani, PRB 48, 11705 (1993)
    """
    
    def __init__(self, optical_params: OpticalParameters):
        self.params = optical_params
        self.chi2_tensor: Optional[np.ndarray] = None
        self.shg_spectrum: Optional[Dict] = None
        
        # Physical constants
        self.hbar = 6.582e-16  # eV·s
        self.e = 1.602e-19     # C
        self.m0 = 9.109e-31    # kg
        self.eps0 = 8.854e-12  # F/m
        self.c = 2.998e8       # m/s
    
    def calculate_chi2(self,
                      band_structure: Dict,
                      dipole_matrix: np.ndarray,
                      frequencies: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate second-order susceptibility tensor χ^(2).
        
        Formula (length gauge):
        χ^(2)_{ijk} = (e³/V) Σ_{n,m,l} ∫dk [r^i_{nl} r^j_{lm} r^k_{mn}] / 
                      [(E_ln - 2ħω - iη)(E_mn - ħω - iη)]
        
        Args:
            band_structure: Band structure with energies and k-points
            dipole_matrix: Dipole matrix elements r_nm(k)
            frequencies: Frequency grid (default from optical_params)
            
        Returns:
            χ^(2) tensor [3, 3, 3, n_omega]
        """
        if frequencies is None:
            frequencies = np.linspace(self.params.omega_min, 
                                     self.params.omega_max, 
                                     self.params.n_omega)
        
        # Initialize tensor: χ^(2)_{ijk} for i,j,k ∈ {x,y,z}
        chi2 = np.zeros((3, 3, 3, len(frequencies)), dtype=complex)
        
        # Get band data
        energies = band_structure['energies']  # [nk, nb]
        nk, nb = energies.shape
        
        # Calculate for each frequency
        for iw, omega in enumerate(frequencies):
            eta = 0.1  # Broadening
            
            # Sum over k-points
            for ik in range(nk):
                # Sum over band triples (interband transitions)
                for n in range(nb):
                    for m in range(nb):
                        if n == m:
                            continue
                        for l in range(nb):
                            if l == n or l == m:
                                continue
                            
                            # Energy denominators
                            E_ln = energies[ik, l] - energies[ik, n]
                            E_mn = energies[ik, m] - energies[ik, n]
                            
                            # Resonance terms
                            denom1 = E_ln - 2*omega + 1j*eta
                            denom2 = E_mn - omega + 1j*eta
                            
                            # Skip if too close to resonance (divergence)
                            if abs(denom1) < 1e-6 or abs(denom2) < 1e-6:
                                continue
                            
                            # Dipole product
                            for i in range(3):
                                for j in range(3):
                                    for k in range(3):
                                        r_nli = dipole_matrix[ik, n, l, i]
                                        r_lmj = dipole_matrix[ik, l, m, j]
                                        r_mnk = dipole_matrix[ik, m, n, k]
                                        
                                        chi2[i, j, k, iw] += (r_nli * r_lmj * r_mnk / 
                                                              (denom1 * denom2))
            
            # Normalize by volume and k-point weight
            chi2[:, :, :, iw] /= nk
        
        # Convert to physical units (pm/V)
        volume = 1e-30  # m³ (normalized)
        chi2 *= (self.e**3 / volume) * 1e-12 / self.eps0
        
        self.chi2_tensor = chi2
        return chi2
    
    def shg_intensity(self,
                     chi2: np.ndarray,
                     polarization_in: LightPolarization = LightPolarization.LINEAR_X,
                     polarization_out: LightPolarization = LightPolarization.LINEAR_X) -> np.ndarray:
        """
        Calculate SHG intensity for given polarizations.
        
        I(2ω) ∝ |e_i(2ω) · χ^(2)_{ijk} : e_j(ω) e_k(ω)|²
        
        Args:
            chi2: χ^(2) tensor
            polarization_in: Input polarization
            polarization_out: Output polarization
            
        Returns:
            SHG intensity array
        """
        n_freq = chi2.shape[3]
        intensity = np.zeros(n_freq)
        
        # Define polarization vectors
        e_in = self._polarization_vector(polarization_in)
        e_out = self._polarization_vector(polarization_out)
        
        for iw in range(n_freq):
            # Contract tensor with polarization vectors
            # I = |e_out · χ^(2) : e_in e_in|²
            amplitude = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        amplitude += (e_out[i] * chi2[i, j, k, iw] * 
                                     e_in[j] * e_in[k])
            
            intensity[iw] = np.abs(amplitude)**2
        
        return intensity
    
    def _polarization_vector(self, pol: LightPolarization) -> np.ndarray:
        """Get polarization unit vector."""
        if pol == LightPolarization.LINEAR_X:
            return np.array([1, 0, 0])
        elif pol == LightPolarization.LINEAR_Y:
            return np.array([0, 1, 0])
        elif pol == LightPolarization.LINEAR_Z:
            return np.array([0, 0, 1])
        elif pol == LightPolarization.CIRCULAR_RIGHT:
            return np.array([1, 1j, 0]) / np.sqrt(2)
        elif pol == LightPolarization.CIRCULAR_LEFT:
            return np.array([1, -1j, 0]) / np.sqrt(2)
        else:
            return np.array([1, 0, 0])
    
    def shg_spectrum_full(self,
                         band_structure: Dict,
                         dipole_matrix: np.ndarray) -> Dict:
        """
        Calculate complete SHG spectrum with all tensor components.
        
        Returns:
            Dictionary with frequencies and all χ^(2) components
        """
        frequencies = np.linspace(self.params.omega_min, 
                                 self.params.omega_max, 
                                 self.params.n_omega)
        
        chi2 = self.calculate_chi2(band_structure, dipole_matrix, frequencies)
        
        # SHG intensity for common configurations
        intensity_xx = self.shg_intensity(chi2, 
                                          LightPolarization.LINEAR_X,
                                          LightPolarization.LINEAR_X)
        intensity_yy = self.shg_intensity(chi2,
                                          LightPolarization.LINEAR_Y,
                                          LightPolarization.LINEAR_Y)
        
        self.shg_spectrum = {
            'frequencies': frequencies,
            'chi2_tensor': chi2,
            'intensity_xx': intensity_xx,
            'intensity_yy': intensity_yy,
            'units': 'pm/V for chi2, arbitrary for intensity'
        }
        
        return self.shg_spectrum
    
    def millers_rule(self, chi1: np.ndarray) -> np.ndarray:
        """
        Estimate χ^(2) from Miller's rule approximation.
        
        χ^(2) ≈ Δ · χ^(1)(ω) χ^(1)(2ω)
        
        where Δ is the Miller coefficient (~10^-10 m/V for most materials)
        
        Args:
            chi1: Linear susceptibility χ^(1)(ω)
            
        Returns:
            Estimated χ^(2)
        """
        miller_coeff = 1e-10  # m/V
        
        chi2_estimate = miller_coeff * chi1[None, :, :] * chi1[None, None, :]
        
        return chi2_estimate


class CircularDichroism:
    """
    Circular Dichroism (CD) and Optical Rotatory Dispersion (ORD) calculator.
    
    CD measures differential absorption of left/right circularly polarized light:
    Δε = ε_L - ε_R
    
    Related to rotatory strength:
    R = Im[μ · m]
    where μ is electric dipole and m is magnetic dipole.
    """
    
    def __init__(self, optical_params: OpticalParameters):
        self.params = optical_params
        self.cd_spectrum: Optional[Dict] = None
    
    def calculate_rotatory_strength(self,
                                    ground_state: np.ndarray,
                                    excited_states: List[Dict]) -> np.ndarray:
        """
        Calculate rotatory strengths for electronic transitions.
        
        R_{0n} = Im[⟨0|μ|n⟩ · ⟨n|m|0⟩]
        
        where μ = -er (electric dipole)
              m = -e/2m (L + gS) (magnetic dipole)
        
        Args:
            ground_state: Ground state wavefunction
            excited_states: List of excited states with dipole matrix elements
            
        Returns:
            Rotatory strengths array
        """
        n_states = len(excited_states)
        rotatory_strengths = np.zeros(n_states)
        
        for i, state in enumerate(excited_states):
            # Electric transition dipole
            mu = state.get('electric_dipole', np.zeros(3))
            
            # Magnetic transition dipole
            m = state.get('magnetic_dipole', np.zeros(3))
            
            # Rotatory strength
            rotatory_strengths[i] = np.imag(np.dot(mu, np.conj(m)))
        
        return rotatory_strengths
    
    def cd_spectrum_gaussian(self,
                            rotatory_strengths: np.ndarray,
                            excitation_energies: np.ndarray,
                            broadening: float = 0.1) -> Dict:
        """
        Calculate CD spectrum with Gaussian broadening.
        
        Δε(E) = (E / (2.296e-39)) Σ_n R_{0n} g_n(E)
        
        where g_n is Gaussian lineshape.
        
        Args:
            rotatory_strengths: Rotatory strengths in cgs units
            excitation_energies: Excitation energies in eV
            broadening: Gaussian broadening in eV
            
        Returns:
            CD spectrum dictionary
        """
        # Energy grid
        energies = np.linspace(self.params.omega_min, 
                              self.params.omega_max,
                              self.params.n_omega)
        
        cd_signal = np.zeros_like(energies)
        
        # Sum over transitions
        for R_n, E_n in zip(rotatory_strengths, excitation_energies):
            # Gaussian lineshape
            g = np.exp(-(energies - E_n)**2 / (2 * broadening**2))
            g /= (broadening * np.sqrt(2 * np.pi))
            
            cd_signal += R_n * energies * g
        
        # Convert to molar ellipticity (approximate)
        cd_signal *= 1e40  # scaling factor
        
        self.cd_spectrum = {
            'energies': energies,
            'cd_signal': cd_signal,
            'rotatory_strengths': rotatory_strengths,
            'excitation_energies': excitation_energies,
            'units': 'deg·cm²/dmol (approximate)'
        }
        
        return self.cd_spectrum
    
    def optical_rotation(self,
                        frequencies: np.ndarray,
                        chi2_tensor: np.ndarray) -> np.ndarray:
        """
        Calculate optical rotation angle (ORD).
        
        For isotropic media, the rotation is related to the antisymmetric
        part of the dielectric tensor.
        
        φ = (ωl/2c) Re[n_L - n_R]
        
        Args:
            frequencies: Frequency array
            chi2_tensor: Second-order susceptibility (for chiral materials)
            
        Returns:
            Rotation angle in degrees
        """
        # For chiral isotropic media, use gyration tensor
        # Simplified model
        rotation = np.zeros_like(frequencies)
        
        # Sample Lorentz oscillator model
        for freq in frequencies:
            # Resonant enhancement near optical transitions
            omega_0 = self.params.bandgap
            gamma = 0.1
            
            # Optical rotation (simplified)
            rot = 0.1 * (freq - omega_0) / ((freq - omega_0)**2 + gamma**2)
            rotation = np.append(rotation, rot)
        
        # Convert to degrees per length
        rotation = rotation[:len(frequencies)] * 180 / np.pi * 1e-3  # deg/mm
        
        return rotation
    
    def kuhn_dissymmetry(self,
                        dipole_strength: np.ndarray,
                        rotatory_strength: np.ndarray) -> np.ndarray:
        """
        Calculate Kuhn dissymmetry factor.
        
        g = 4|R| / D
        
        where D = |μ|² is the dipole strength.
        
        Args:
            dipole_strength: Dipole strengths
            rotatory_strength: Rotatory strengths
            
        Returns:
            Dissymmetry factors
        """
        g = np.zeros_like(dipole_strength)
        
        for i in range(len(dipole_strength)):
            if dipole_strength[i] > 1e-10:
                g[i] = 4 * np.abs(rotatory_strength[i]) / dipole_strength[i]
        
        return g


class HighHarmonicGeneration:
    """
    High Harmonic Generation (HHG) calculator.
    
    HHG produces odd harmonics (nω) of the fundamental laser frequency.
    Cutoff: E_cutoff = I_p + 3.17 U_p
    
    where I_p = ionization potential, U_p = ponderomotive energy.
    """
    
    def __init__(self, optical_params: OpticalParameters):
        self.params = optical_params
        self.hh_spectrum: Optional[Dict] = None
        
        # Constants
        self.hbar = 6.582e-16  # eV·s
        self.e = 1.602e-19     # C
        self.m0 = 9.109e-31    # kg
        self.c = 2.998e8       # m/s
    
    def ponderomotive_energy(self) -> float:
        """
        Calculate ponderomotive energy.
        
        U_p = e²E² / (4mω²) = I / (4ω²)
        
        Returns:
            U_p in eV
        """
        # Laser intensity in W/m²
        I = self.params.laser_intensity * 1e4
        
        # Angular frequency
        omega = 2 * np.pi * self.c / self.params.wavelength
        
        # Ponderomotive energy in Joules
        U_p_J = self.e**2 * I / (2 * self.m0 * self.params.electron_mass * omega**2 * self.c * 8.854e-12)
        
        # Convert to eV
        U_p = U_p_J / self.e
        
        return U_p
    
    def cutoff_energy(self) -> float:
        """
        Calculate HHG cutoff energy.
        
        E_cutoff = I_p + 3.17 U_p
        
        Returns:
            Cutoff energy in eV
        """
        Ip = self.params.bandgap  # Approximate ionization potential
        Up = self.ponderomotive_energy()
        
        return Ip + 3.17 * Up
    
    def harmonic_spectrum(self,
                         max_harmonic: int = 15) -> Dict:
        """
        Generate HHG spectrum with odd harmonics.
        
        Args:
            max_harmonic: Maximum harmonic order
            
        Returns:
            HHG spectrum dictionary
        """
        # Fundamental photon energy
        omega_0 = self.params.photon_energy()
        
        # Generate odd harmonics only
        harmonics = [n for n in range(1, max_harmonic + 1, 2)]
        energies = [n * omega_0 for n in harmonics]
        
        # Intensity model (simplified)
        # Below cutoff: decreasing plateau
        # Above cutoff: rapid drop
        intensities = []
        cutoff = self.cutoff_energy()
        
        for n, E in zip(harmonics, energies):
            if E < cutoff:
                # Plateau with slight decrease
                intensity = 1.0 / n**0.5
            else:
                # Exponential drop above cutoff
                intensity = np.exp(-(E - cutoff) / (0.5 * cutoff))
            
            intensities.append(intensity)
        
        self.hh_spectrum = {
            'harmonics': harmonics,
            'energies': np.array(energies),
            'intensities': np.array(intensities),
            'fundamental': omega_0,
            'cutoff_energy': cutoff,
            'ponderomotive_energy': self.ponderomotive_energy()
        }
        
        return self.hh_spectrum
    
    def semiclassical_trajectory(self,
                                  ionization_time: float,
                                  return_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate semiclassical electron trajectory.
        
        Simple man's model: electron tunnels at t_i, oscillates in laser field,
        returns at t_r, emitting harmonic photon.
        
        Args:
            ionization_time: Ionization time (in optical cycles)
            return_time: Return time
            
        Returns:
            (time, position) arrays
        """
        # Laser period
        T = 2 * np.pi * self.params.wavelength / self.c
        
        # Time grid
        t = np.linspace(ionization_time * T, return_time * T, 100)
        
        # Electric field
        omega = 2 * np.pi / T
        E0 = np.sqrt(2 * self.params.laser_intensity * 1e4 / (self.c * 8.854e-12))
        
        # Trajectory: x(t) = -eE0/(mω²) [cos(ωt) - cos(ωt_i) + ω(t-t_i)sin(ωt_i)]
        m_eff = self.m0 * self.params.electron_mass
        x = -(self.e * E0 / (m_eff * omega**2)) * (
            np.cos(omega * t) - np.cos(omega * ionization_time * T) +
            omega * (t - ionization_time * T) * np.sin(omega * ionization_time * T)
        )
        
        return t, x
    
    def attosecond_pulse(self,
                        harmonic_range: Tuple[int, int] = (5, 15),
                        bandwidth: float = 1.0) -> Dict:
        """
        Generate isolated attosecond pulse from HHG superposition.
        
        Args:
            harmonic_range: Range of harmonics to include
            bandwidth: Spectral bandwidth in eV
            
        Returns:
            Attosecond pulse data
        """
        # Generate harmonics
        spectrum = self.harmonic_spectrum(harmonic_range[1])
        
        # Time grid for pulse
        t_max = 100e-18  # 100 as
        nt = 1000
        t = np.linspace(-t_max, t_max, nt)
        
        # Build pulse from harmonics
        E_t = np.zeros_like(t, dtype=complex)
        
        omega_0 = self.params.photon_energy()
        
        for n in range(harmonic_range[0], harmonic_range[1] + 1, 2):
            omega_n = n * omega_0
            amplitude = 1.0 / n
            phase = 0  # Simplified - could add attochirp
            
            # Gaussian spectral envelope
            sigma = bandwidth / (2 * np.sqrt(2 * np.log(2)))
            envelope = np.exp(-(omega_n - (harmonic_range[0] + harmonic_range[1])/2 * omega_0)**2 / (2 * sigma**2))
            
            E_t += amplitude * envelope * np.exp(1j * omega_n * t / self.hbar + 1j * phase)
        
        # Intensity
        I_t = np.abs(E_t)**2
        
        # FWHM duration
        half_max = np.max(I_t) / 2
        above_half = np.where(I_t > half_max)[0]
        if len(above_half) > 1:
            fwhm = (above_half[-1] - above_half[0]) * (t[1] - t[0])
        else:
            fwhm = 0
        
        return {
            'time': t,
            'electric_field': E_t,
            'intensity': I_t,
            'fwhm': fwhm,
            'harmonics_used': list(range(harmonic_range[0], harmonic_range[1] + 1, 2))
        }


class NonlinearTensor:
    """
    Nonlinear optical susceptibility tensor handling.
    
    Manages symmetry properties and independent components of χ^(n).
    """
    
    def __init__(self, order: NonlinearOrder, crystal_class: str = "isotropic"):
        self.order = order
        self.crystal_class = crystal_class
        self.tensor: Optional[np.ndarray] = None
    
    def independent_components(self) -> List[Tuple[int, ...]]:
        """
        Get list of independent tensor components for crystal class.
        
        Returns:
            List of independent component indices
        """
        if self.order == NonlinearOrder.SECOND:
            # χ^(2) for common crystal classes
            if self.crystal_class == "isotropic":
                # No second-order response in centrosymmetric
                return []
            elif self.crystal_class == "432":
                return []
            elif self.crystal_class == "23":
                return [(0, 1, 2), (1, 2, 0), (2, 0, 1)]  # xyz cyclic
            elif self.crystal_class in ["3m", "4mm", "6mm"]:
                return [(2, 0, 0), (2, 1, 1), (0, 2, 0), (1, 2, 1), (2, 2, 2)]
            elif self.crystal_class == "222":
                return [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
            else:
                # General non-centrosymmetric
                return [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
        
        elif self.order == NonlinearOrder.THIRD:
            # χ^(3) for common classes
            if self.crystal_class == "isotropic":
                # Two independent components: xxxx and xxyy
                return [(0, 0, 0, 0), (0, 0, 1, 1)]
            else:
                # General third-order
                components = []
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                components.append((i, j, k, l))
                return components
        
        else:
            # Higher order - return all components
            rank = self.order.value
            from itertools import product
            return list(product(range(3), repeat=rank))
    
    def symmetrize(self, tensor: np.ndarray) -> np.ndarray:
        """
        Apply intrinsic permutation symmetry to tensor.
        
        For χ^(2): χ_ijk = χ_ikj (permutation of output frequencies)
        """
        if self.order == NonlinearOrder.SECOND:
            # Symmetrize over last two indices
            return (tensor + np.transpose(tensor, (0, 2, 1))) / 2
        elif self.order == NonlinearOrder.THIRD:
            # More complex symmetries for third order
            return tensor
        return tensor
    
    def effective_susceptibility(self,
                                  polarization_in: Tuple[LightPolarization, ...],
                                  polarization_out: LightPolarization) -> complex:
        """
        Calculate effective susceptibility for given polarizations.
        
        χ_eff = Σ_{indices} e_out · χ · e_in ⊗ e_in ⊗ ...
        
        Args:
            polarization_in: Input polarizations for each field
            polarization_out: Output polarization
            
        Returns:
            Effective susceptibility
        """
        if self.tensor is None:
            raise ValueError("Tensor not initialized")
        
        e_out = self._polarization_vector(polarization_out)
        
        chi_eff = 0
        
        # Build contraction
        if self.order == NonlinearOrder.SECOND:
            e_in1 = self._polarization_vector(polarization_in[0])
            e_in2 = self._polarization_vector(polarization_in[1])
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        chi_eff += (e_out[i] * self.tensor[i, j, k] * 
                                   e_in1[j] * e_in2[k])
        
        return chi_eff
    
    def _polarization_vector(self, pol: LightPolarization) -> np.ndarray:
        """Get polarization unit vector."""
        if pol == LightPolarization.LINEAR_X:
            return np.array([1, 0, 0])
        elif pol == LightPolarization.LINEAR_Y:
            return np.array([0, 1, 0])
        elif pol == LightPolarization.LINEAR_Z:
            return np.array([0, 0, 1])
        elif pol == LightPolarization.CIRCULAR_RIGHT:
            return np.array([1, -1j, 0]) / np.sqrt(2)
        elif pol == LightPolarization.CIRCULAR_LEFT:
            return np.array([1, 1j, 0]) / np.sqrt(2)
        else:
            return np.array([1, 0, 0])


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Advanced Nonlinear Optical Properties")
    print("="*60)
    
    # Initialize parameters
    params = OpticalParameters(
        omega_min=0.5,
        omega_max=5.0,
        n_omega=500,
        laser_intensity=1e12,  # W/cm²
        bandgap=1.5
    )
    
    # SHG example
    print("\n--- Second Harmonic Generation ---")
    shg = SecondHarmonicGeneration(params)
    
    # Simulate band structure
    nk = 10
    nb = 4
    band_structure = {
        'energies': np.linspace(-2, 3, nk*nb).reshape(nk, nb)
    }
    
    # Simulate dipole matrix
    dipole = np.random.randn(nk, nb, nb, 3) + 0.1
    
    chi2 = shg.calculate_chi2(band_structure, dipole)
    print(f"χ^(2) tensor shape: {chi2.shape}")
    print(f"χ^(2)_xxx at first frequency: {chi2[0,0,0,0]:.2e} pm/V")
    
    # CD example
    print("\n--- Circular Dichroism ---")
    cd = CircularDichroism(params)
    
    excited_states = [
        {'electric_dipole': np.array([1, 0, 0]),
         'magnetic_dipole': np.array([0, 1, 0])}
        for _ in range(3)
    ]
    
    R = cd.calculate_rotatory_strength(None, excited_states)
    print(f"Rotatory strengths: {R}")
    
    # HHG example
    print("\n--- High Harmonic Generation ---")
    hhg = HighHarmonicGeneration(params)
    
    print(f"Ponderomotive energy: {hhg.ponderomotive_energy():.2f} eV")
    print(f"Cutoff energy: {hhg.cutoff_energy():.2f} eV")
    
    spectrum = hhg.harmonic_spectrum(max_harmonic=15)
    print(f"Harmonics: {spectrum['harmonics']}")
    print(f"Harmonic energies (eV): {spectrum['energies']}")
    
    # Attosecond pulse
    pulse = hhg.attosecond_pulse((7, 13))
    print(f"\nAttosecond pulse FWHM: {pulse['fwhm']*1e18:.1f} as")
