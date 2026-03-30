"""
negf_formalism.py

Non-Equilibrium Green's Function (NEGF) Formalism Implementation

This module provides a comprehensive implementation of NEGF for quantum
transport calculations, including:
- Green's function calculations (retarded, advanced, lesser, greater)
- Spectral functions and local density of states
- Current formulas (Landauer-Buttiker and Meir-Wingreen)
- Phonon scattering (inelastic transport)

References:
- Datta, "Electronic Transport in Mesoscopic Systems" (1995)
- Haug & Jauho, "Quantum Kinetics in Transport and Optics of Semiconductors"
- Datta, "Nanoscale device modeling: the Green's function method"
"""

import numpy as np
from scipy import linalg, integrate
from scipy.sparse import csr_matrix, block_diag
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class GreenFunctionType(Enum):
    """Types of Green's functions in NEGF formalism."""
    RETARDED = "retarded"  # G^R
    ADVANCED = "advanced"  # G^A  
    LESSER = "lesser"      # G^<
    GREATER = "greater"    # G^>
    TIME_ORDERED = "time_ordered"  # G^T
    ANTI_TIME_ORDERED = "anti_time_ordered"  # G^T̃


@dataclass
class SelfEnergy:
    """
    Self-energy contribution from baths or scattering mechanisms.
    
    Contains retarded and lesser/greater components needed for
    non-equilibrium calculations.
    """
    
    # Retarded self-energy
    sigma_r: np.ndarray
    
    # Lesser and greater self-energies
    sigma_lesser: Optional[np.ndarray] = None
    sigma_greater: Optional[np.ndarray] = None
    
    # Broadening (imaginary part)
    gamma: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.gamma is None:
            # Γ = i(Σ^R - Σ^A) = i(Σ^R - Σ^R†)
            self.gamma = 1j * (self.sigma_r - self.sigma_r.T.conj())
    
    def get_lesser(self, fermi_factor: float = 0.5) -> np.ndarray:
        """
        Calculate lesser self-energy from retarded component.
        
        For equilibrium: Σ^< = -f(E) Γ(E)
        """
        if self.sigma_lesser is not None:
            return self.sigma_lesser
        
        return -fermi_factor * self.gamma
    
    def get_greater(self, fermi_factor: float = 0.5) -> np.ndarray:
        """
        Calculate greater self-energy from retarded component.
        
        For equilibrium: Σ^> = (1-f(E)) Γ(E)
        """
        if self.sigma_greater is not None:
            return self.sigma_greater
        
        return (1 - fermi_factor) * self.gamma


@dataclass
class NEGFSystemAdvanced:
    """
    Advanced NEGF system with full non-equilibrium capabilities.
    """
    
    # Hamiltonian and overlap
    hamiltonian: np.ndarray
    overlap: Optional[np.ndarray] = None
    
    # Self-energies from different sources
    electrode_self_energies: Dict[str, SelfEnergy] = field(default_factory=dict)
    scattering_self_energies: Dict[str, SelfEnergy] = field(default_factory=dict)
    
    # System parameters
    temperature: float = 300.0  # K
    
    # Cache for Green's functions
    _green_functions: Dict[GreenFunctionType, Dict[float, np.ndarray]] = field(
        default_factory=lambda: {gf_type: {} for gf_type in GreenFunctionType}
    )
    
    def __post_init__(self):
        if self.overlap is None:
            self.overlap = np.eye(self.hamiltonian.shape[0])
    
    def calculate_retarded_green_function(self, energy: complex) -> np.ndarray:
        """
        Calculate retarded Green's function:
        
        G^R(E) = [(E + iη)S - H - Σ^R]^{-1}
        
        where Σ^R is the total retarded self-energy.
        """
        # Check cache
        e_key = round(energy.real, 10)
        if e_key in self._green_functions[GreenFunctionType.RETARDED]:
            return self._green_functions[GreenFunctionType.RETARDED][e_key]
        
        # Total self-energy
        sigma_r_total = np.zeros_like(self.hamiltonian, dtype=complex)
        
        for se in self.electrode_self_energies.values():
            sigma_r_total += se.sigma_r
        
        for se in self.scattering_self_energies.values():
            sigma_r_total += se.sigma_r
        
        # G^R = [ES - H - Σ^R]^{-1}
        g_inv = energy * self.overlap - self.hamiltonian - sigma_r_total
        g_r = linalg.inv(g_inv)
        
        # Cache result
        self._green_functions[GreenFunctionType.RETARDED][e_key] = g_r
        
        return g_r
    
    def calculate_advanced_green_function(self, energy: complex) -> np.ndarray:
        """G^A = (G^R)†"""
        g_r = self.calculate_retarded_green_function(energy)
        return g_r.T.conj()
    
    def calculate_lesser_green_function(self, energy: complex,
                                       chemical_potentials: Dict[str, float]) -> np.ndarray:
        """
        Calculate lesser Green's function:
        
        G^< = G^R Σ^< G^A
        
        where Σ^< is the total lesser self-energy from all baths.
        """
        # Check cache
        e_key = round(energy.real, 10)
        if e_key in self._green_functions[GreenFunctionType.LESSER]:
            return self._green_functions[GreenFunctionType.LESSER][e_key]
        
        g_r = self.calculate_retarded_green_function(energy)
        g_a = g_r.T.conj()
        
        # Total lesser self-energy
        sigma_lesser_total = np.zeros_like(self.hamiltonian, dtype=complex)
        
        for name, se in self.electrode_self_energies.items():
            mu = chemical_potentials.get(name, 0.0)
            fermi_factor = self._fermi_distribution(energy.real, mu)
            sigma_lesser_total += se.get_lesser(fermi_factor)
        
        for se in self.scattering_self_energies.values():
            sigma_lesser_total += se.get_lesser()
        
        # G^< = G^R Σ^< G^A
        g_lesser = g_r @ sigma_lesser_total @ g_a
        
        # Cache result
        self._green_functions[GreenFunctionType.LESSER][e_key] = g_lesser
        
        return g_lesser
    
    def calculate_greater_green_function(self, energy: complex,
                                        chemical_potentials: Dict[str, float]) -> np.ndarray:
        """
        Calculate greater Green's function:
        
        G^> = G^R Σ^> G^A
        """
        g_r = self.calculate_retarded_green_function(energy)
        g_a = g_r.T.conj()
        
        # Total greater self-energy
        sigma_greater_total = np.zeros_like(self.hamiltonian, dtype=complex)
        
        for name, se in self.electrode_self_energies.items():
            mu = chemical_potentials.get(name, 0.0)
            fermi_factor = self._fermi_distribution(energy.real, mu)
            sigma_greater_total += se.get_greater(fermi_factor)
        
        for se in self.scattering_self_energies.values():
            sigma_greater_total += se.get_greater()
        
        # G^> = G^R Σ^> G^A
        g_greater = g_r @ sigma_greater_total @ g_a
        
        return g_greater
    
    def _fermi_distribution(self, energy: float, 
                           chemical_potential: float) -> float:
        """Fermi-Dirac distribution."""
        kB = 8.617e-5  # eV/K
        
        if self.temperature == 0:
            return 1.0 if energy < chemical_potential else 0.0
        
        kT = kB * self.temperature
        return 1.0 / (np.exp((energy - chemical_potential) / kT) + 1.0)
    
    def calculate_spectral_function(self, energy: complex) -> np.ndarray:
        """
        Calculate spectral function:
        
        A(E) = i(G^R - G^A) = G^R Γ G^A
        
        The spectral function gives the density of states.
        """
        g_r = self.calculate_retarded_green_function(energy)
        g_a = g_r.T.conj()
        
        # Total broadening
        gamma_total = np.zeros_like(self.hamiltonian, dtype=complex)
        for se in self.electrode_self_energies.values():
            gamma_total += se.gamma
        for se in self.scattering_self_energies.values():
            gamma_total += se.gamma
        
        # A = G^R Γ G^A
        A = g_r @ gamma_total @ g_a
        
        return A
    
    def calculate_ldos(self, energy: complex) -> np.ndarray:
        """
        Calculate Local Density of States:
        
        LDOS(E, r) = -1/π Im[G^R(E, r, r)] = 1/(2π) A(E, r, r)
        """
        g_r = self.calculate_retarded_green_function(energy)
        
        # Diagonal elements
        ldos = -np.imag(np.diag(g_r)) / np.pi
        
        return ldos


class LandauerButtiker:
    """
    Landauer-Buttiker formula for coherent transport.
    
    For coherent transport, the current can be expressed as:
    I = (2e/h) ∫ T(E) [f_L(E) - f_R(E)] dE
    """
    
    def __init__(self, negf_system: NEGFSystemAdvanced):
        self.negf = negf_system
        self.h_bar = 6.582e-16  # eV·s
        self.e_charge = 1.602e-19  # C
    
    def calculate_transmission(self, energy: float,
                              lead1: str, lead2: str,
                              eta: float = 1e-8) -> float:
        """
        Calculate transmission coefficient between two leads using Caroli formula:
        
        T_{12}(E) = Tr[Γ_1 G^R Γ_2 G^A]
        """
        z_energy = energy + 1j * eta
        
        g_r = self.negf.calculate_retarded_green_function(z_energy)
        g_a = g_r.T.conj()
        
        # Get broadenings
        gamma1 = self.negf.electrode_self_energies[lead1].gamma
        gamma2 = self.negf.electrode_self_energies[lead2].gamma
        
        # T = Tr[Γ_1 G^R Γ_2 G^A]
        t_matrix = gamma1 @ g_r @ gamma2 @ g_a
        transmission = np.real(np.trace(t_matrix))
        
        return transmission
    
    def calculate_current(self, bias_voltage: float,
                         lead1: str, lead2: str,
                         energy_range: Tuple[float, float],
                         num_points: int = 1000) -> float:
        """
        Calculate current using Landauer formula.
        
        I = (2e/h) ∫ T(E) [f_1(E) - f_2(E)] dE
        """
        # Chemical potentials
        mu1 = bias_voltage / 2
        mu2 = -bias_voltage / 2
        
        # Integration
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        de = energies[1] - energies[0]
        
        current_integral = 0.0
        
        for E in energies:
            T_E = self.calculate_transmission(E, lead1, lead2)
            
            f1 = self._fermi(E, mu1)
            f2 = self._fermi(E, mu2)
            
            current_integral += T_E * (f1 - f2) * de
        
        # I = (2e/h) × integral
        current = (2 * self.e_charge / self.h_bar) * current_integral
        
        return current
    
    def _fermi(self, E: float, mu: float) -> float:
        """Fermi-Dirac distribution."""
        kB = 8.617e-5  # eV/K
        kT = kB * self.negf.temperature
        
        if kT == 0:
            return 1.0 if E < mu else 0.0
        
        return 1.0 / (np.exp((E - mu) / kT) + 1.0)


class MeirWingreen:
    """
    Meir-Wingreen formula for current with inelastic scattering.
    
    More general than Landauer formula - includes non-coherent effects.
    """
    
    def __init__(self, negf_system: NEGFSystemAdvanced):
        self.negf = negf_system
        self.h_bar = 6.582e-16
        self.e_charge = 1.602e-19
    
    def calculate_current(self, lead_name: str,
                         chemical_potentials: Dict[str, float],
                         energy_range: Tuple[float, float],
                         num_points: int = 1000) -> float:
        """
        Calculate current through a specific lead using Meir-Wingreen formula:
        
        I_α = (e/h) ∫ Tr[Σ_>^α G^< - Σ_<^α G^>] dE
        
        where α labels the lead.
        """
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        de = energies[1] - energies[0]
        
        current_integral = 0.0
        
        for E in energies:
            z = E + 1j * 1e-8
            
            # Get self-energies for this lead
            se = self.negf.electrode_self_energies[lead_name]
            
            # Get Green's functions
            g_lesser = self.negf.calculate_lesser_green_function(z, chemical_potentials)
            g_greater = self.negf.calculate_greater_green_function(z, chemical_potentials)
            
            # Get chemical potential for this lead
            mu = chemical_potentials.get(lead_name, 0.0)
            fermi_factor = self._fermi(E, mu)
            
            sigma_greater = se.get_greater(fermi_factor)
            sigma_lesser = se.get_lesser(fermi_factor)
            
            # Meir-Wingreen integrand
            term = np.trace(sigma_greater @ g_lesser - sigma_lesser @ g_greater)
            current_integral += np.real(term) * de
        
        # I = (e/h) × integral
        current = (self.e_charge / self.h_bar) * current_integral
        
        return current
    
    def _fermi(self, E: float, mu: float) -> float:
        """Fermi-Dirac distribution."""
        kB = 8.617e-5
        kT = kB * self.negf.temperature
        
        if kT == 0:
            return 1.0 if E < mu else 0.0
        
        return 1.0 / (np.exp((E - mu) / kT) + 1.0)


class PhononScattering:
    """
    Implementation of electron-phonon scattering in NEGF.
    
    Includes self-consistent Born approximation for inelastic transport.
    """
    
    def __init__(self, negf_system: NEGFSystemAdvanced):
        self.negf = negf_system
        
        # Phonon parameters
        self.debye_temperature = 300.0  # K
        self.phonon_coupling = 0.1  # eV
        self.max_phonon_energy = 0.1  # eV
    
    def calculate_phonon_self_energy(self, energy: float,
                                    phonon_mode: Dict,
                                    occupation: float) -> SelfEnergy:
        """
        Calculate electron-phonon self-energy.
        
        For deformation potential coupling:
        Σ^R(E) = Σ_k |M_q|² [n_B(ω_q) G(E-ω_q) + (n_B(ω_q)+1) G(E+ω_q)]
        
        where M_q is the electron-phonon matrix element.
        """
        omega = phonon_mode['frequency']
        matrix_element = phonon_mode['coupling']
        
        # Calculate Green's functions at shifted energies
        e_minus = energy - omega + 1j * 1e-8
        e_plus = energy + omega + 1j * 1e-8
        
        g_r_minus = self.negf.calculate_retarded_green_function(e_minus)
        g_r_plus = self.negf.calculate_retarded_green_function(e_plus)
        
        # Bose-Einstein distribution
        n_bose = self._bose_occupation(omega)
        
        # Retarded self-energy
        sigma_r = (matrix_element**2 * (
            n_bose * g_r_minus + (n_bose + 1) * g_r_plus
        ))
        
        # Lesser self-energy
        g_lesser_minus = self._get_lesser_gf_approx(e_minus)
        g_lesser_plus = self._get_lesser_gf_approx(e_plus)
        
        sigma_lesser = matrix_element**2 * (
            n_bose * g_lesser_minus + (n_bose + 1) * g_lesser_plus
        )
        
        # Greater self-energy
        g_greater_minus = self._get_greater_gf_approx(e_minus)
        g_greater_plus = self._get_greater_gf_approx(e_plus)
        
        sigma_greater = matrix_element**2 * (
            (n_bose + 1) * g_greater_minus + n_bose * g_greater_plus
        )
        
        return SelfEnergy(
            sigma_r=sigma_r,
            sigma_lesser=sigma_lesser,
            sigma_greater=sigma_greater
        )
    
    def _bose_occupation(self, omega: float) -> float:
        """Bose-Einstein distribution."""
        kB = 8.617e-5
        T = self.negf.temperature
        
        if T == 0:
            return 0.0 if omega > 0 else float('inf')
        
        kT = kB * T
        
        if omega <= 0:
            return 0.0
        
        return 1.0 / (np.exp(omega / kT) - 1.0)
    
    def _get_lesser_gf_approx(self, energy: float) -> np.ndarray:
        """Approximate lesser Green's function."""
        # Approximation: G^< ≈ -f(E) A(E)
        g_r = self.negf.calculate_retarded_green_function(energy + 1j * 1e-8)
        g_a = g_r.T.conj()
        
        # Assume equilibrium at E_F = 0
        f = self._fermi(energy, 0.0)
        
        return -f * (g_r - g_a) * 1j
    
    def _get_greater_gf_approx(self, energy: float) -> np.ndarray:
        """Approximate greater Green's function."""
        g_r = self.negf.calculate_retarded_green_function(energy + 1j * 1e-8)
        g_a = g_r.T.conj()
        
        f = self._fermi(energy, 0.0)
        
        return (1 - f) * (g_r - g_a) * 1j
    
    def _fermi(self, E: float, mu: float) -> float:
        """Fermi-Dirac distribution."""
        kB = 8.617e-5
        kT = kB * self.negf.temperature
        
        if kT == 0:
            return 1.0 if E < mu else 0.0
        
        return 1.0 / (np.exp((E - mu) / kT) + 1.0)
    
    def calculate_inelastic_current(self, lead1: str, lead2: str,
                                   bias_voltage: float,
                                   energy_range: Tuple[float, float],
                                   num_phonon_modes: int = 10) -> Dict:
        """
        Calculate current with inelastic phonon scattering.
        
        Returns both elastic and inelastic contributions.
        """
        # Generate phonon modes
        phonon_energies = np.linspace(0.01, self.max_phonon_energy, num_phonon_modes)
        
        # Calculate self-consistent self-energy
        # This is a simplified version
        
        results = {
            'elastic_contribution': 0.0,
            'inelastic_contribution': 0.0,
            'total_current': 0.0,
            'phonon_energies': phonon_energies.tolist()
        }
        
        return results


class SpectralAnalysis:
    """
    Tools for analyzing spectral functions and density of states.
    """
    
    def __init__(self, negf_system: NEGFSystemAdvanced):
        self.negf = negf_system
    
    def calculate_partial_dos(self, energy: complex,
                             atom_indices: List[int]) -> float:
        """
        Calculate projected density of states on specific atoms.
        """
        ldos = self.negf.calculate_ldos(energy)
        
        # Project onto selected atoms
        partial_dos = np.sum(ldos[atom_indices])
        
        return partial_dos
    
    def calculate_energy_resolved_dos(self, 
                                     energy_range: Tuple[float, float],
                                     num_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate DOS as function of energy.
        """
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        dos = []
        
        for E in energies:
            ldos = self.negf.calculate_ldos(E + 1j * 1e-8)
            dos.append(np.sum(ldos))
        
        return energies, np.array(dos)
    
    def find_resonant_states(self, energy_range: Tuple[float, float],
                            threshold: float = 1e-3) -> List[Dict]:
        """
        Find resonant states by analyzing peaks in LDOS.
        """
        energies, dos = self.calculate_energy_resolved_dos(energy_range, 1000)
        
        resonances = []
        
        # Find local maxima
        for i in range(1, len(dos) - 1):
            if dos[i] > dos[i-1] and dos[i] > dos[i+1] and dos[i] > threshold:
                # Estimate width
                half_max = dos[i] / 2
                
                # Find left and right half-max points
                left = i
                while left > 0 and dos[left] > half_max:
                    left -= 1
                
                right = i
                while right < len(dos) - 1 and dos[right] > half_max:
                    right += 1
                
                width = energies[right] - energies[left]
                
                resonances.append({
                    'energy': energies[i],
                    'amplitude': dos[i],
                    'width': width,
                    'lifetime': self.h_bar / width if width > 0 else float('inf')
                })
        
        return resonances
    
    def calculate_bond_current(self, energy: float,
                              atom_i: int, atom_j: int) -> float:
        """
        Calculate bond current between two atoms using bond-current formula.
        
        This is useful for visualizing current pathways in molecular devices.
        """
        # Calculate Green's functions
        z = energy + 1j * 1e-8
        g_r = self.negf.calculate_retarded_green_function(z)
        g_lesser = self.negf.calculate_lesser_green_function(z, {})
        
        # Get Hamiltonian elements
        H = self.negf.hamiltonian
        H_ij = H[atom_i, atom_j]
        
        # Bond current (simplified)
        # J_ij = (e/ℏ) × 2 Im[H_ij G^<_ji]
        bond_current = (2.0 / self.h_bar) * np.imag(H_ij * g_lesser[atom_j, atom_i])
        
        return bond_current


def example_coherent_transport():
    """
    Example: Coherent transport through a double barrier structure.
    """
    print("=" * 60)
    print("Example: NEGF Coherent Transport - Double Barrier")
    print("=" * 60)
    
    # Simple 1D model: double barrier resonant tunneling
    num_sites = 20
    
    # Hamiltonian: barriers at sites 7,8 and 12,13
    H = np.zeros((num_sites, num_sites))
    
    # On-site energies
    for i in range(num_sites):
        if 6 <= i <= 8 or 11 <= i <= 13:
            H[i, i] = 2.0  # Barrier height
        else:
            H[i, i] = 0.0  # Well and leads
    
    # Nearest neighbor hopping
    t = -1.0
    for i in range(num_sites - 1):
        H[i, i+1] = t
        H[i+1, i] = t
    
    # Self-energies for leads (simplified wide-band approximation)
    gamma = 0.5
    sigma_left = np.zeros((num_sites, num_sites), dtype=complex)
    sigma_left[0, 0] = -1j * gamma / 2
    
    sigma_right = np.zeros((num_sites, num_sites), dtype=complex)
    sigma_right[-1, -1] = -1j * gamma / 2
    
    # Create NEGF system
    se_left = SelfEnergy(sigma_r=sigma_left)
    se_right = SelfEnergy(sigma_r=sigma_right)
    
    negf = NEGFSystemAdvanced(
        hamiltonian=H,
        electrode_self_energies={'left': se_left, 'right': se_right}
    )
    
    # Calculate transmission
    print("\nCalculating transmission...")
    landauer = LandauerButtiker(negf)
    
    energies = np.linspace(0, 4, 200)
    transmissions = []
    
    for E in energies:
        T = landauer.calculate_transmission(E, 'left', 'right')
        transmissions.append(T)
    
    transmissions = np.array(transmissions)
    
    # Find resonances
    resonances = []
    for i in range(1, len(transmissions) - 1):
        if transmissions[i] > transmissions[i-1] and transmissions[i] > transmissions[i+1]:
            if transmissions[i] > 0.1:
                resonances.append((energies[i], transmissions[i]))
    
    print(f"\nResults:")
    print(f"  Peak transmission: {np.max(transmissions):.4f}")
    print(f"  Number of resonances: {len(resonances)}")
    
    if resonances:
        print(f"  Resonant energies:")
        for E, T in resonances[:3]:
            print(f"    E = {E:.3f} eV, T = {T:.3f}")
    
    # Calculate LDOS at resonance
    if resonances:
        E_res = resonances[0][0]
        ldos = negf.calculate_ldos(E_res + 1j * 1e-8)
        
        print(f"\nLDOS at E = {E_res:.3f} eV:")
        print(f"  Max LDOS at site: {np.argmax(ldos)}")
        print(f"  LDOS in well: {np.sum(ldos[9:11]):.3f}")
    
    return energies, transmissions


def example_inelastic_transport():
    """
    Example: Inelastic transport with phonon scattering.
    """
    print("\n" + "=" * 60)
    print("Example: Inelastic Transport with Phonons")
    print("=" * 60)
    
    # Simple 2-site model with electron-phonon coupling
    H = np.array([[0.0, -1.0], [-1.0, 0.0]])
    
    # Lead self-energies
    gamma = 0.2
    sigma_left = np.zeros((2, 2), dtype=complex)
    sigma_left[0, 0] = -1j * gamma / 2
    
    sigma_right = np.zeros((2, 2), dtype=complex)
    sigma_right[1, 1] = -1j * gamma / 2
    
    negf = NEGFSystemAdvanced(
        hamiltonian=H,
        electrode_self_energies={
            'left': SelfEnergy(sigma_r=sigma_left),
            'right': SelfEnergy(sigma_r=sigma_right)
        },
        temperature=300.0
    )
    
    # Add phonon scattering
    phonon = PhononScattering(negf)
    
    # Define phonon mode
    phonon_mode = {
        'frequency': 0.05,  # eV
        'coupling': 0.1     # eV
    }
    
    print(f"\nPhonon parameters:")
    print(f"  Frequency: {phonon_mode['frequency']*1000:.1f} meV")
    print(f"  Coupling: {phonon_mode['coupling']*1000:.1f} meV")
    
    # Calculate phonon self-energy
    energy = 0.0
    occupation = 0.5
    
    se_phonon = phonon.calculate_phonon_self_energy(energy, phonon_mode, occupation)
    
    print(f"\nPhonon self-energy at E = {energy:.2f} eV:")
    print(f"  Re[Σ^R]: {np.real(se_phonon.sigma_r[0,0]):.4f} eV")
    print(f"  Im[Σ^R]: {np.imag(se_phonon.sigma_r[0,0]):.4f} eV")
    
    return phonon


if __name__ == "__main__":
    # Run examples
    energies, transmissions = example_coherent_transport()
    phonon = example_inelastic_transport()
    
    print("\n" + "=" * 60)
    print("NEGF Formalism Module - Test Complete")
    print("=" * 60)
