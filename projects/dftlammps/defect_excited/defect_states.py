"""
Defect Excited States Module
============================
Calculation and analysis of defect-related excited states including:
- Color center excited states (NV centers, etc.)
- Defect luminescence spectra
- Quantum defect spin manipulation

References:
- Doherty et al., The nitrogen-vacancy colour centre in diamond (2013)
- Awschalom et al., Quantum spintronics (2013)
- Davies, The Jahn-Teller effect and vibronic coupling at deep levels (1981)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy import linalg, integrate
from scipy.special import factorial
import warnings


class DefectType(Enum):
    """Types of point defects."""
    VACANCY = "vacancy"              # Missing atom
    SUBSTITUTIONAL = "substitutional" # Foreign atom on lattice site
    INTERSTITIAL = "interstitial"     # Atom in void space
    NV_CENTER = "nv_center"           # Nitrogen-vacancy in diamond
    SI_VACANCY = "si_vacancy"         # Silicon vacancy
    DIVACANCY = "divacancy"          # Two adjacent vacancies
    COMPLEX = "complex"              # Multi-atom defect


class ChargeState(Enum):
    """Defect charge states."""
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    DOUBLE_NEGATIVE = -2
    DOUBLE_POSITIVE = 2


class SpinState(Enum):
    """Electronic spin states."""
    SINGLET = 0
    DOUBLET = 1
    TRIPLET = 2
    QUARTET = 3


@dataclass
class DefectHamiltonian:
    """Hamiltonian parameters for defect system."""
    # Electronic energies
    ground_state_energy: float = 0.0  # eV
    excited_state_energy: float = 1.5  # eV
    
    # Spin parameters
    zero_field_splitting_D: float = 2.87e-3  # eV (NV center value)
    zero_field_splitting_E: float = 0.0      # eV
    g_factor: float = 2.0023  # Landé g-factor
    
    # Spin-orbit coupling
    spin_orbit_lambda: float = 0.0  # eV
    
    # Strain coupling
    strain_coupling: float = 0.01  # eV
    
    # Hyperfine coupling
    hyperfine_A_parallel: float = 2.3e-6  # eV (N nuclear spin)
    hyperfine_A_perp: float = 2.1e-6      # eV


@dataclass
class VibronicParameters:
    """Parameters for vibronic coupling."""
    # Huang-Rhys factor
    S: float = 3.0
    
    # Phonon energy
    phonon_energy: float = 0.07  # eV (NV center)
    
    # Effective frequency
    omega_eff: float = 70e-3  # eV
    
    # Linewidths
    homogeneous_width: float = 0.01  # eV
    inhomogeneous_width: float = 0.1  # eV
    
    # Temperature
    temperature: float = 300  # K


class ColorCenter:
    """
    Color center defect model (e.g., NV center in diamond).
    
    NV center electronic structure:
    - Ground state: ³A₂ (triplet, S=1)
    - Excited state: ³E (triplet, S=1)
    - Metastable: ¹A₁ (singlet)
    
    Key properties:
    - ZFS: D_gs = 2.87 GHz (ground), D_es ≈ 1.42 GHz (excited)
    - ODMR: Optically detected magnetic resonance
    """
    
    def __init__(self,
                 defect_type: DefectType = DefectType.NV_CENTER,
                 charge_state: ChargeState = ChargeState.NEGATIVE,
                 hamiltonian: Optional[DefectHamiltonian] = None,
                 vibronic: Optional[VibronicParameters] = None):
        """
        Initialize color center.
        
        Args:
            defect_type: Type of color center
            charge_state: Charge state
            hamiltonian: Hamiltonian parameters
            vibronic: Vibronic coupling parameters
        """
        self.defect_type = defect_type
        self.charge_state = charge_state
        self.H = hamiltonian or DefectHamiltonian()
        self.vibronic = vibronic or VibronicParameters()
        
        # Spin operators
        self.S = self._init_spin_operators()
        
        # Internal states
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        
    def _init_spin_operators(self) -> Dict:
        """
        Initialize spin-1 operators for triplet state.
        """
        # S=1 matrices
        Sx = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]]) / np.sqrt(2)
        
        Sy = np.array([[0, -1j, 0],
                       [1j, 0, -1j],
                       [0, 1j, 0]]) / np.sqrt(2)
        
        Sz = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, -1]])
        
        return {'x': Sx, 'y': Sy, 'z': Sz}
    
    def build_spin_hamiltonian(self,
                                B_field: np.ndarray = np.zeros(3),
                                strain: np.ndarray = np.zeros(3)) -> np.ndarray:
        """
        Build spin Hamiltonian in magnetic field.
        
        H = D(S_z² - S(S+1)/3) + E(S_x² - S_y²) + gμ_B B·S + H_strain
        
        Args:
            B_field: Magnetic field in Tesla [Bx, By, Bz]
            strain: Strain tensor components [ε_xx, ε_yy, ε_zz]
            
        Returns:
            3x3 Hamiltonian matrix
        """
        # Zero-field splitting
        D = self.H.zero_field_splitting_D
        E = self.H.zero_field_splitting_E
        
        Sx = self.S['x']
        Sy = self.S['y']
        Sz = self.S['z']
        
        # ZFS term
        Sz2 = Sz @ Sz
        S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
        
        H_zfs = D * (Sz2 - S2 / 3) + E * (Sx @ Sx - Sy @ Sy)
        
        # Zeeman term
        mu_B = 5.788e-5  # eV/T (Bohr magneton)
        g = self.H.g_factor
        
        H_zeeman = g * mu_B * (B_field[0] * Sx + B_field[1] * Sy + B_field[2] * Sz)
        
        # Strain coupling (simplified)
        H_strain = self.H.strain_coupling * strain[2] * Sz2
        
        H_total = H_zfs + H_zeeman + H_strain
        
        return H_total
    
    def diagonalize(self,
                    B_field: np.ndarray = np.zeros(3)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize spin Hamiltonian.
        
        Args:
            B_field: Magnetic field
            
        Returns:
            (eigenvalues, eigenvectors)
        """
        H = self.build_spin_hamiltonian(B_field)
        
        self.eigenvalues, self.eigenvectors = linalg.eigh(H)
        
        return self.eigenvalues, self.eigenvectors
    
    def odmr_frequencies(self,
                         B_field: np.ndarray = np.zeros(3)) -> np.ndarray:
        """
        Calculate ODMR (optically detected magnetic resonance) frequencies.
        
        These are the transition frequencies between spin sublevels.
        
        Args:
            B_field: Magnetic field
            
        Returns:
            Transition frequencies in GHz
        """
        eigenvalues, _ = self.diagonalize(B_field)
        
        # Transition frequencies
        freqs = []
        for i in range(len(eigenvalues)):
            for j in range(i+1, len(eigenvalues)):
                freq = abs(eigenvalues[j] - eigenvalues[i]) / (6.582e-16)  # Convert to Hz
                freqs.append(freq / 1e9)  # Convert to GHz
        
        return np.array(freqs)
    
    def rabi_frequency(self,
                       microwave_field: float,
                       transition: Tuple[int, int] = (0, 1)) -> float:
        """
        Calculate Rabi frequency for spin manipulation.
        
        Ω_R = g μ_B B_mw |⟨i|S⊥|j⟩| / ħ
        
        Args:
            microwave_field: MW magnetic field amplitude in Tesla
            transition: Transition indices
            
        Returns:
            Rabi frequency in MHz
        """
        if self.eigenvectors is None:
            self.diagonalize()
        
        i, j = transition
        
        # Transition matrix element (perpendicular component)
        S_perp = self.S['x']  # Assuming x-polarized MW
        
        matrix_element = np.abs(
            self.eigenvectors[:, i].conj() @ S_perp @ self.eigenvectors[:, j]
        )
        
        mu_B = 5.788e-5  # eV/T
        g = self.H.g_factor
        
        # Rabi frequency in eV
        omega_eV = g * mu_B * microwave_field * matrix_element
        
        # Convert to MHz
        omega_MHz = omega_eV / (6.582e-16) / 1e6
        
        return omega_MHz


class DefectLuminescence:
    """
    Defect luminescence (PL) spectrum calculator.
    
    Includes:
    - Zero-phonon line (ZPL)
    - Phonon sideband (PSB)
    - Huang-Rhys factor analysis
    """
    
    def __init__(self,
                 zero_phonon_energy: float = 1.945,  # NV center ZPL (eV)
                 vibronic: Optional[VibronicParameters] = None):
        """
        Initialize luminescence calculator.
        
        Args:
            zero_phonon_energy: ZPL energy in eV
            vibronic: Vibronic coupling parameters
        """
        self.E_zpl = zero_phonon_energy
        self.vibronic = vibronic or VibronicParameters()
        
        # Constants
        self.kB = 8.617e-5  # eV/K
    
    def phonon_sideband(self,
                       energy_range: Tuple[float, float] = None,
                       n_points: int = 1000) -> Dict:
        """
        Calculate phonon sideband using Pekarian or Gaussian approximation.
        
        I(E) ∝ Σ_n P(n) δ(E - E_ZPL + nħω)
        
        where P(n) = e^(-S) S^n / n! (Poisson distribution)
        
        Args:
            energy_range: Energy range [min, max] in eV
            n_points: Number of energy points
            
        Returns:
            Spectrum dictionary
        """
        if energy_range is None:
            energy_range = (self.E_zpl - 0.5, self.E_zpl + 0.5)
        
        energies = np.linspace(energy_range[0], energy_range[1], n_points)
        intensity = np.zeros_like(energies)
        
        S = self.vibronic.S
        hw = self.vibronic.phonon_energy
        
        # Include phonon replicas
        max_phonons = min(20, int(0.5 / hw) + 1)
        
        for n in range(max_phonons):
            # Poisson weight
            if n < 100:
                P_n = np.exp(-S) * S**n / factorial(n)
            else:
                # Stirling approximation for large n
                P_n = np.exp(-S + n * np.log(S) - n * np.log(n) + n)
            
            # Peak position (Stokes shift)
            E_n = self.E_zpl - n * hw
            
            # Gaussian broadening
            sigma = self.vibronic.homogeneous_width
            if self.vibronic.temperature > 0:
                # Temperature-dependent broadening
                sigma += self.kB * self.vibronic.temperature * 0.1
            
            intensity += P_n * np.exp(-(energies - E_n)**2 / (2 * sigma**2))
        
        # Normalize
        intensity /= np.max(intensity)
        
        return {
            'energies': energies,
            'intensity': intensity,
            'zpl_energy': self.E_zpl,
            'huang_rhys': S,
            'phonon_energy': hw,
            'debye_waller': np.exp(-S)
        }
    
    def debye_waller_factor(self, temperature: float = 300) -> float:
        """
        Calculate Debye-Waller factor (ZPL intensity fraction).
        
        DW = e^(-S coth(ħω/2kT))
        
        Args:
            temperature: Temperature in K
            
        Returns:
            Debye-Waller factor
        """
        S = self.vibronic.S
        hw = self.vibronic.phonon_energy
        kT = self.kB * temperature
        
        if temperature > 0:
            n_bose = 1 / (np.exp(hw / (2 * kT)) - 1)
            S_eff = S * (2 * n_bose + 1)
        else:
            S_eff = S
        
        return np.exp(-S_eff)
    
    def lineshape_function(self,
                          t: np.ndarray,
                          temperature: float = 300) -> np.ndarray:
        """
        Calculate lineshape function g(t) for optical dephasing.
        
        Used in nonlinear spectroscopy calculations.
        
        Args:
            t: Time array
            temperature: Temperature
            
        Returns:
            Lineshape function
        """
        S = self.vibronic.S
        hw = self.vibronic.phonon_energy
        kT = self.kB * temperature
        
        # Bose factor
        if temperature > 0:
            n = 1 / (np.exp(hw / kT) - 1)
        else:
            n = 0
        
        # Lineshape function
        g_t = S * ((2 * n + 1) * (1 - np.cos(hw * t / 6.582e-16)) +
                   1j * np.sin(hw * t / 6.582e-16) - 1j * hw * t / 6.582e-16)
        
        return g_t
    
    def absorption_emission_spectra(self,
                                   temperature: float = 300,
                                   energy_range: Tuple[float, float] = None) -> Dict:
        """
        Calculate both absorption and emission spectra.
        
        Absorption: E_ZPL + nħω (anti-Stokes)
        Emission: E_ZPL - nħω (Stokes)
        
        Args:
            temperature: Temperature
            energy_range: Energy range
            
        Returns:
            Spectra dictionary
        """
        if energy_range is None:
            energy_range = (self.E_zpl - 0.8, self.E_zpl + 0.3)
        
        energies = np.linspace(energy_range[0], energy_range[1], 1000)
        
        S = self.vibronic.S
        hw = self.vibronic.phonon_energy
        kT = self.kB * temperature
        
        absorption = np.zeros_like(energies)
        emission = np.zeros_like(energies)
        
        max_phonons = min(15, int(0.8 / hw))
        
        for n in range(max_phonons):
            # Poisson weight
            P_n = np.exp(-S) * S**n / factorial(n)
            
            # Temperature-dependent occupation
            if temperature > 0:
                n_occ = 1 / (np.exp(n * hw / kT) - 1)
            else:
                n_occ = 0 if n > 0 else 1
            
            # Absorption (ZPL + nħω)
            E_abs = self.E_zpl + n * hw
            sigma = self.vibronic.inhomogeneous_width
            absorption += P_n * (n_occ + 1) * np.exp(-(energies - E_abs)**2 / (2 * sigma**2))
            
            # Emission (ZPL - nħω)
            E_em = self.E_zpl - n * hw
            emission += P_n * np.exp(-(energies - E_em)**2 / (2 * sigma**2))
        
        # Normalize
        absorption /= np.max(absorption) if np.max(absorption) > 0 else 1
        emission /= np.max(emission) if np.max(emission) > 0 else 1
        
        return {
            'energies': energies,
            'absorption': absorption,
            'emission': emission,
            'stokes_shift': self.calculate_stokes_shift(),
            'temperature': temperature
        }
    
    def calculate_stokes_shift(self) -> float:
        """
        Calculate Stokes shift from Huang-Rhys factor.
        
        ΔE_Stokes = 2Sħω
        """
        return 2 * self.vibronic.S * self.vibronic.phonon_energy
    
    def frank_condon_factors(self,
                             n_max: int = 10) -> np.ndarray:
        """
        Calculate Franck-Condon factors |⟨m|n⟩|².
        
        For displaced harmonic oscillators:
        |⟨m|n⟩|² = e^(-S) S^(m+n) (m! n!) / (Σ_k C(m,k) C(n,k) k! (-S)^(-k))²
        
        Args:
            n_max: Maximum vibrational quantum number
            
        Returns:
            FC factor matrix [n_max, n_max]
        """
        S = self.vibronic.S
        
        fc_matrix = np.zeros((n_max, n_max))
        
        for m in range(n_max):
            for n in range(n_max):
                # Simplified FC factor (displaced oscillator)
                # |⟨m|n⟩|² for m=0 (ground state)
                if m == 0:
                    fc = np.exp(-S) * S**n / factorial(n)
                else:
                    # More general case (approximate)
                    fc = np.exp(-S) * S**(m+n) / np.sqrt(factorial(m) * factorial(n))
                
                fc_matrix[m, n] = fc
        
        return fc_matrix


class QuantumSpinManipulation:
    """
    Quantum control and spin manipulation for defect qubits.
    
    Implements:
    - Rabi oscillations
    - Ramsey interferometry
    - Hahn echo
    - Dynamical decoupling
    """
    
    def __init__(self, color_center: ColorCenter):
        self.defect = color_center
        
        # Time step for simulations
        self.dt = 1e-12  # s
    
    def rabi_oscillations(self,
                         pulse_duration: float,
                         rabi_frequency: float,
                         detuning: float = 0.0) -> Dict:
        """
        Simulate Rabi oscillations.
        
        P_1(t) = (Ω_R² / (Ω_R² + Δ²)) sin²(√(Ω_R² + Δ²) t / 2)
        
        Args:
            pulse_duration: Pulse duration in s
            rabi_frequency: Rabi frequency in Hz
            detuning: Detuning from resonance in Hz
            
        Returns:
            Rabi oscillation data
        """
        # Time array
        t = np.arange(0, pulse_duration, self.dt)
        
        # Generalized Rabi frequency
        Omega = np.sqrt(rabi_frequency**2 + detuning**2)
        
        # Rabi formula
        if Omega > 0:
            P1 = (rabi_frequency**2 / Omega**2) * np.sin(Omega * t / 2)**2
        else:
            P1 = np.zeros_like(t)
        
        return {
            'time': t,
            'excited_state_population': P1,
            'ground_state_population': 1 - P1,
            'rabi_frequency': rabi_frequency,
            'detuning': detuning
        }
    
    def ramsey_fringes(self,
                      free_evolution_time: float,
                      detuning: float,
                      phase: float = 0.0) -> Dict:
        """
        Simulate Ramsey interferometry sequence.
        
        π/2 - τ - π/2 with phase accumulation
        
        Args:
            free_evolution_time: Free evolution time τ
            detuning: Detuning from resonance
            phase: Phase of second pulse
            
        Returns:
            Ramsey signal
        """
        # Time array
        t = np.arange(0, free_evolution_time, self.dt)
        
        # Ramsey fringes: P ∝ cos²(Δτ/2 + φ)
        signal = 0.5 * (1 + np.cos(detuning * t + phase))
        
        return {
            'time': t,
            'signal': signal,
            'detuning': detuning,
            'contrast': np.max(signal) - np.min(signal)
        }
    
    def hahn_echo(self,
                  total_time: float,
                  detuning: float,
                  dephasing_rate: float = 1e6) -> Dict:
        """
        Simulate Hahn echo sequence.
        
        π/2 - τ/2 - π - τ/2 - measure
        
        Refocuses static dephasing (T2* → T2).
        
        Args:
            total_time: Total sequence time
            detuning: Static detuning
            dephasing_rate: Dephasing rate (1/T2)
            
        Returns:
            Echo signal
        """
        # Time array
        t = np.arange(0, total_time, self.dt)
        
        # Echo signal: exp(-t/T2) (refocuses static detuning)
        signal = np.exp(-dephasing_rate * t)
        
        # Add revival at t = total_time (echo)
        echo_time = total_time / 2
        echo_width = total_time / 20
        echo = np.exp(-((t - echo_time) / echo_width)**2)
        
        signal = signal + 0.5 * echo
        
        return {
            'time': t,
            'signal': signal,
            'echo_time': echo_time,
            'dephasing_time': 1 / dephasing_rate
        }
    
    def dynamical_decoupling(self,
                            total_time: float,
                            n_pulses: int,
                            pulse_sequence: str = "CPMG") -> Dict:
        """
        Simulate dynamical decoupling pulse sequences.
        
        CPMG: π/2 - (τ - π - τ)^N - measure
        XY8: Alternating X/Y pulses
        
        Args:
            total_time: Total sequence time
            n_pulses: Number of π pulses
            pulse_sequence: Sequence type ("CPMG", "XY8", "UDD")
            
        Returns:
            DD signal
        """
        tau = total_time / (2 * n_pulses + 1)
        
        # Time array
        t = np.arange(0, total_time, self.dt)
        
        # Signal with DD (suppressed dephasing)
        # DD extends T2 by decoupling from low-frequency noise
        T2_dd = total_time * (n_pulses**0.7)  # Scaling with N pulses
        
        signal = np.exp(-t / T2_dd)
        
        return {
            'time': t,
            'signal': signal,
            'n_pulses': n_pulses,
            'sequence': pulse_sequence,
            'effective_T2': T2_dd
        }
    
    def entanglement_fidelity(self,
                             gate_time: float,
                             coherence_time: float) -> float:
        """
        Estimate entanglement gate fidelity.
        
        F ≈ exp(-gate_time / (2 * T2))
        
        Args:
            gate_time: Gate operation time
            coherence_time: Spin coherence time T2
            
        Returns:
            Fidelity estimate
        """
        return np.exp(-gate_time / (2 * coherence_time))


# Utility functions
def calculate_zfs_tensor(positions: np.ndarray,
                         spin_densities: np.ndarray) -> np.ndarray:
    """
    Calculate zero-field splitting tensor from spin density.
    
    D_{ij} = (μ0/4π) (gμ_B)² Σ_{kl} [δ_{ij} - 3r_i r_j/r²] / r³
    
    Args:
        positions: Atomic positions [N, 3]
        spin_densities: Spin density at each position
        
    Returns:
        ZFS tensor [3, 3]
    """
    D = np.zeros((3, 3))
    
    mu0 = 4 * np.pi * 1e-7  # H/m
    mu_B = 9.274e-24  # J/T
    g = 2.0023
    
    prefactor = mu0 / (4 * np.pi) * (g * mu_B)**2
    
    for i, r_i in enumerate(positions):
        for j, r_j in enumerate(positions):
            if i == j:
                continue
            
            r = r_j - r_i
            r_norm = np.linalg.norm(r)
            
            if r_norm > 0:
                dipole_tensor = (np.eye(3) - 3 * np.outer(r, r) / r_norm**2) / r_norm**3
                D += prefactor * spin_densities[i] * spin_densities[j] * dipole_tensor
    
    return D


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Defect Excited States - NV Center Example")
    print("="*60)
    
    # Initialize NV center
    H_params = DefectHamiltonian(
        zero_field_splitting_D=2.87e-3,  # 2.87 GHz in eV
        zero_field_splitting_E=0.0,
        g_factor=2.0023
    )
    
    nv = ColorCenter(
        defect_type=DefectType.NV_CENTER,
        charge_state=ChargeState.NEGATIVE,
        hamiltonian=H_params
    )
    
    # Zero-field ODMR
    print("\n--- Zero-field ODMR ---")
    freqs_zf = nv.odmr_frequencies(B_field=np.zeros(3))
    print(f"Transition frequencies: {freqs_zf} GHz")
    
    # With magnetic field
    print("\n--- ODMR with B-field ---")
    B = np.array([0, 0, 100e-4])  # 100 G along z
    freqs_B = nv.odmr_frequencies(B_field=B)
    print(f"Transition frequencies: {freqs_B} GHz")
    
    # Rabi frequency
    print("\n--- Rabi Frequency ---")
    omega_R = nv.rabi_frequency(microwave_field=1e-7)  # 1 μT
    print(f"Rabi frequency: {omega_R:.2f} MHz")
    
    # Luminescence
    print("\n--- Defect Luminescence ---")
    lum = DefectLuminescence(
        zero_phonon_energy=1.945,  # NV ZPL
        vibronic=VibronicParameters(S=3.0, phonon_energy=0.07)
    )
    
    spectrum = lum.phonon_sideband()
    print(f"ZPL energy: {spectrum['zpl_energy']:.3f} eV")
    print(f"Huang-Rhys factor: {spectrum['huang_rhys']:.2f}")
    print(f"Debye-Waller factor: {spectrum['debye_waller']:.3f}")
    
    # Stokes shift
    print(f"Stokes shift: {lum.calculate_stokes_shift():.3f} eV")
    
    # Quantum control
    print("\n--- Quantum Spin Manipulation ---")
    control = QuantumSpinManipulation(nv)
    
    rabi = control.rabi_oscillations(
        pulse_duration=1e-6,  # 1 μs
        rabi_frequency=10e6   # 10 MHz
    )
    print(f"Rabi π-pulse time: {0.5/rabi['rabi_frequency']*1e9:.1f} ns")
    
    # Fidelity
    fidelity = control.entanglement_fidelity(
        gate_time=100e-9,  # 100 ns
        coherence_time=1e-3  # 1 ms
    )
    print(f"Estimated gate fidelity: {fidelity:.4f}")
    
    print("\n" + "="*60)
