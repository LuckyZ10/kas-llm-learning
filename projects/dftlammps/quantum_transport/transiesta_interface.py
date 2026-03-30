"""
transiesta_interface.py

TranSIESTA/NEGF Interface for Quantum Transport Simulations

This module provides interfaces to SIESTA and TranSIESTA for non-equilibrium
green's function (NEGF) calculations of electronic transport in nanostructures.

Features:
- Electrode-Scattering-Electrode structure construction
- Self-energy calculations and level alignment
- Transmission coefficient T(E) calculations
- I-V characteristic curves
- Interface with SIESTA/TranSIESTA input/output files

References:
- Brandbyge et al., PRB 65, 165401 (2002) - TranSIESTA
- Taylor et al., PRB 63, 245407 (2001) - NEGF formalism
"""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import csr_matrix, lil_matrix, block_diag
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from enum import Enum
import json


class ElectrodeType(Enum):
    """Types of electrodes for transport calculations."""
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass
class ElectrodeConfig:
    """Configuration for electrode in transport setup."""
    
    name: str
    electrode_type: ElectrodeType
    num_atoms: int
    fermi_level: float  # eV
    num_orbitals: int
    repetition: Tuple[int, int, int] = (1, 1, 1)
    
    # Hamiltonian and overlap matrices (sparse)
    h0: Optional[csr_matrix] = None  # On-site Hamiltonian
    h1: Optional[csr_matrix] = None  # Coupling to next unit cell
    s0: Optional[csr_matrix] = None  # On-site overlap
    s1: Optional[csr_matrix] = None  # Overlap coupling
    
    # Surface Green's function (computed)
    surface_gf: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate electrode configuration."""
        if self.h0 is not None and self.s0 is not None:
            if self.h0.shape != self.s0.shape:
                raise ValueError("H0 and S0 must have same dimensions")


@dataclass  
class TransportStructure:
    """
    Complete transport structure with electrodes and scattering region.
    
    Structure: Left Electrode | Scattering Region | Right Electrode
    """
    
    left_electrode: ElectrodeConfig
    right_electrode: ElectrodeConfig
    
    # Scattering region
    num_scatter_atoms: int
    num_scatter_orbitals: int
    
    # Hamiltonian matrices (full structure)
    h_scatter: Optional[csr_matrix] = None  # Scattering region Hamiltonian
    s_scatter: Optional[csr_matrix] = None  # Scattering region overlap
    
    # Coupling matrices
    h_lc: Optional[csr_matrix] = None  # Left electrode to scattering
    h_cr: Optional[csr_matrix] = None  # Scattering to right electrode
    s_lc: Optional[csr_matrix] = None
    s_cr: Optional[csr_matrix] = None
    
    # Metadata
    temperature: float = 300.0  # K
    bias_voltage: float = 0.0  # V
    
    def get_total_size(self) -> int:
        """Get total number of basis functions."""
        return (self.left_electrode.num_orbitals + 
                self.num_scatter_orbitals + 
                self.right_electrode.num_orbitals)
    
    def validate_structure(self) -> bool:
        """Validate the transport structure."""
        if self.h_scatter is None or self.s_scatter is None:
            raise ValueError("Scattering region matrices not set")
        
        if self.h_lc is None or self.h_cr is None:
            raise ValueError("Electrode coupling matrices not set")
            
        # Check dimensions
        nL = self.left_electrode.num_orbitals
        nS = self.num_scatter_orbitals
        nR = self.right_electrode.num_orbitals
        
        assert self.h_scatter.shape == (nS, nS), "H_scatter dimension mismatch"
        assert self.h_lc.shape == (nL, nS), "H_LC dimension mismatch"
        assert self.h_cr.shape == (nS, nR), "H_CR dimension mismatch"
        
        return True


class SurfaceGFCalculator:
    """
    Calculator for surface Green's functions using iterative methods.
    
    The surface Green's function is essential for calculating self-energies
    of semi-infinite electrodes.
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-10):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def calculate_surface_gf(self, energy: complex, 
                            h0: np.ndarray, h1: np.ndarray,
                            s0: np.ndarray, s1: np.ndarray,
                            method: str = "decimation") -> np.ndarray:
        """
        Calculate surface Green's function g_s(E) for semi-infinite lead.
        
        Args:
            energy: Complex energy (E + i*eta)
            h0: On-site Hamiltonian
            h1: Coupling to next unit cell (h1 = h_{-1}^\dagger)
            s0: On-site overlap
            s1: Overlap coupling
            method: Method for calculation ("decimation" or "recursive")
            
        Returns:
            Surface Green's function matrix
        """
        if method == "decimation":
            return self._decimation_method(energy, h0, h1, s0, s1)
        elif method == "recursive":
            return self._recursive_method(energy, h0, h1, s0, s1)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _decimation_method(self, energy: complex,
                          h0: np.ndarray, h1: np.ndarray,
                          s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
        """
        Calculate surface Green's function using the decimation technique.
        
        This is an efficient O(N) method for calculating surface Green's functions.
        """
        n = h0.shape[0]
        
        # Initial matrices
        alpha = h0 - energy * s0
        beta = h1 - energy * s1
        beta_dag = h1.T.conj() - energy * s1.T.conj()
        
        # Initial guess for surface Green's function
        g_s = linalg.inv(alpha)
        
        for iteration in range(self.max_iterations):
            # Calculate new effective Hamiltonian
            g_beta = g_s @ beta
            g_beta_dag = g_s @ beta_dag
            
            # Update alpha (self-energy correction)
            alpha_new = alpha - beta_dag @ g_beta - beta @ g_beta_dag
            
            # Update beta (renormalized coupling)
            beta_new = -beta @ g_beta
            beta_dag_new = -beta_dag @ g_beta_dag
            
            # Calculate new surface Green's function
            g_s_new = linalg.inv(alpha_new)
            
            # Check convergence
            diff = linalg.norm(g_s_new - g_s) / linalg.norm(g_s)
            
            if diff < self.tolerance:
                return g_s_new
            
            alpha, beta, beta_dag, g_s = alpha_new, beta_new, beta_dag_new, g_s_new
        
        warnings.warn(f"Surface GF did not converge in {self.max_iterations} iterations")
        return g_s
    
    def _recursive_method(self, energy: complex,
                         h0: np.ndarray, h1: np.ndarray,
                         s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
        """
        Recursive method for surface Green's function.
        Simpler but may be slower than decimation for large systems.
        """
        n = h0.shape[0]
        
        # Initialize
        epsilon = h0 - energy * s0
        alpha = h1 - energy * s1
        alpha_dag = h1.T.conj() - energy * s1.T.conj()
        
        g_s = linalg.inv(epsilon)
        
        for i in range(self.max_iterations):
            g_s_old = g_s.copy()
            
            # Self-energy from semi-infinite chain
            sigma = alpha_dag @ g_s @ alpha
            
            # Update surface Green's function
            g_s = linalg.inv(epsilon - sigma)
            
            # Check convergence
            diff = linalg.norm(g_s - g_s_old) / linalg.norm(g_s)
            if diff < self.tolerance:
                break
        
        return g_s


class SelfEnergyCalculator:
    """
    Calculator for electrode self-energies.
    
    Self-energy accounts for the coupling between the scattering region
    and the infinite electrodes.
    """
    
    def __init__(self):
        self.surface_calculator = SurfaceGFCalculator()
    
    def calculate_self_energy(self, electrode: ElectrodeConfig,
                             energy: complex,
                             coupling_h: csr_matrix,
                             coupling_s: csr_matrix) -> np.ndarray:
        """
        Calculate self-energy: Σ = (H - ES)† g_s (H - ES)
        
        Args:
            electrode: Electrode configuration with surface GF
            energy: Complex energy
            coupling_h: Hamiltonian coupling matrix
            coupling_s: Overlap coupling matrix
            
        Returns:
            Self-energy matrix
        """
        # Calculate or retrieve surface Green's function
        if electrode.surface_gf is None:
            if electrode.h0 is None or electrode.s0 is None:
                raise ValueError("Electrode matrices not initialized")
            
            electrode.surface_gf = self.surface_calculator.calculate_surface_gf(
                energy, electrode.h0.toarray(), electrode.h1.toarray(),
                electrode.s0.toarray(), electrode.s1.toarray()
            )
        
        g_s = electrode.surface_gf
        
        # Calculate (H - ES)
        coupling = coupling_h - energy * coupling_s
        
        # Self-energy: Σ = coupling† @ g_s @ coupling
        sigma = coupling.T.conj() @ g_s @ coupling
        
        return sigma
    
    def calculate_broadening(self, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate broadening function: Γ = i(Σ - Σ†)
        
        The broadening is related to the density of states in the electrode.
        """
        return 1j * (sigma - sigma.T.conj())


class NEGFSystem:
    """
    Non-Equilibrium Green's Function system for transport calculations.
    
    This implements the full NEGF formalism for calculating electronic
    transport properties.
    """
    
    def __init__(self, structure: TransportStructure):
        self.structure = structure
        self.self_energy_calc = SelfEnergyCalculator()
        
        # Cache for computed quantities
        self._cached_energies = {}
        self._green_functions = {}
        self._transmissions = {}
    
    def calculate_green_function(self, energy: complex) -> np.ndarray:
        """
        Calculate retarded Green's function:
        G^R = [(E + iη)S - H - Σ_L - Σ_R]^{-1}
        """
        # Check cache
        e_key = round(energy.real, 8)
        if e_key in self._green_functions:
            return self._green_functions[e_key]
        
        nS = self.structure.num_scatter_orbitals
        
        # Construct Hamiltonian with self-energies
        h_eff = self.structure.h_scatter.toarray().copy()
        s_scatter = self.structure.s_scatter.toarray()
        
        # Add self-energies from electrodes
        sigma_L = self.self_energy_calc.calculate_self_energy(
            self.structure.left_electrode, energy,
            self.structure.h_lc, self.structure.s_lc
        )
        
        sigma_R = self.self_energy_calc.calculate_self_energy(
            self.structure.right_electrode, energy,
            self.structure.h_cr, self.structure.s_cr
        )
        
        # Effective Hamiltonian: (E+iη)S - H - Σ_L - Σ_R
        g_inv = energy * s_scatter - h_eff - sigma_L - sigma_R
        
        # Calculate Green's function
        g_R = linalg.inv(g_inv)
        
        # Cache result
        self._green_functions[e_key] = g_R
        
        return g_R
    
    def calculate_transmission(self, energy: float, 
                              eta: float = 1e-6) -> float:
        """
        Calculate transmission coefficient T(E) using Caroli formula:
        T(E) = Tr[Γ_L G^R Γ_R G^A]
        
        Args:
            energy: Real energy value
            eta: Small imaginary part for numerical stability
            
        Returns:
            Transmission coefficient (0 to number of channels)
        """
        # Check cache
        e_key = round(energy, 8)
        if e_key in self._transmissions:
            return self._transmissions[e_key]
        
        z_energy = energy + 1j * eta
        
        # Calculate Green's functions
        g_R = self.calculate_green_function(z_energy)
        g_A = g_R.T.conj()
        
        # Calculate broadenings
        sigma_L = self.self_energy_calc.calculate_self_energy(
            self.structure.left_electrode, z_energy,
            self.structure.h_lc, self.structure.s_lc
        )
        sigma_R = self.self_energy_calc.calculate_self_energy(
            self.structure.right_electrode, z_energy,
            self.structure.h_cr, self.structure.s_cr
        )
        
        gamma_L = self.self_energy_calc.calculate_broadening(sigma_L)
        gamma_R = self.self_energy_calc.calculate_broadening(sigma_R)
        
        # Transmission: T = Tr[Γ_L G^R Γ_R G^A]
        t_matrix = gamma_L @ g_R @ gamma_R @ g_A
        transmission = np.real(np.trace(t_matrix))
        
        # Cache result
        self._transmissions[e_key] = transmission
        
        return transmission
    
    def calculate_ldos(self, energy: float, eta: float = 1e-6) -> np.ndarray:
        """
        Calculate Local Density of States at given energy.
        
        LDOS(E, r) = -1/π Im[G^R(E, r, r)]
        """
        z_energy = energy + 1j * eta
        g_R = self.calculate_green_function(z_energy)
        
        # Diagonal elements give LDOS at each orbital
        ldos = -np.imag(np.diag(g_R)) / np.pi
        
        return ldos
    
    def calculate_spectral_function(self, energy: float, 
                                    eta: float = 1e-6) -> np.ndarray:
        """
        Calculate spectral function A(E) = G^R Γ G^A.
        """
        z_energy = energy + 1j * eta
        g_R = self.calculate_green_function(z_energy)
        g_A = g_R.T.conj()
        
        # Calculate total broadening
        sigma_L = self.self_energy_calc.calculate_self_energy(
            self.structure.left_electrode, z_energy,
            self.structure.h_lc, self.structure.s_lc
        )
        sigma_R = self.self_energy_calc.calculate_self_energy(
            self.structure.right_electrode, z_energy,
            self.structure.h_cr, self.structure.s_cr
        )
        
        gamma_L = self.self_energy_calc.calculate_broadening(sigma_L)
        gamma_R = self.self_energy_calc.calculate_broadening(sigma_R)
        gamma_total = gamma_L + gamma_R
        
        # Spectral function
        A = g_R @ gamma_total @ g_A
        
        return A


class IVCalculator:
    """
    Calculator for I-V characteristics using Landauer-Buttiker formula.
    """
    
    def __init__(self, negf_system: NEGFSystem):
        self.negf = negf_system
        self.kB = 8.617333e-5  # Boltzmann constant in eV/K
    
    def fermi_distribution(self, E: float, mu: float, 
                          T: float) -> float:
        """
        Fermi-Dirac distribution.
        
        Args:
            E: Energy
            mu: Chemical potential
            T: Temperature in Kelvin
        """
        if T == 0:
            return 1.0 if E < mu else 0.0
        
        kT = self.kB * T
        return 1.0 / (np.exp((E - mu) / kT) + 1.0)
    
    def calculate_current(self, bias_voltage: float,
                         energy_range: Tuple[float, float],
                         num_points: int = 1000,
                         temperature: float = 300.0) -> float:
        """
        Calculate current at given bias voltage using Landauer formula:
        
        I(V) = (2e/h) ∫ T(E) [f_L(E) - f_R(E)] dE
        
        Args:
            bias_voltage: Applied bias in Volts
            energy_range: (E_min, E_max) for integration
            num_points: Number of integration points
            temperature: Temperature in Kelvin
            
        Returns:
            Current in Amperes
        """
        # Physical constants
        e_charge = 1.602e-19  # C
        h_planck = 4.136e-15  # eV*s
        G0 = 2 * e_charge**2 / h_planck  # Quantum of conductance in A/V
        
        # Chemical potentials
        mu_L = bias_voltage / 2  # Left electrode
        mu_R = -bias_voltage / 2  # Right electrode
        
        # Integration grid
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        de = energies[1] - energies[0]
        
        current_integral = 0.0
        
        for E in energies:
            # Calculate transmission
            T_E = self.negf.calculate_transmission(E)
            
            # Fermi distributions
            f_L = self.fermi_distribution(E, mu_L, temperature)
            f_R = self.fermi_distribution(E, mu_R, temperature)
            
            # Integrand
            current_integral += T_E * (f_L - f_R) * de
        
        # Current in Amperes: I = (2e/h) ∫ T(E)[f_L - f_R] dE
        current = (2 * e_charge / h_planck) * current_integral
        
        return current
    
    def calculate_iv_curve(self, bias_range: np.ndarray,
                          energy_range: Tuple[float, float],
                          num_points: int = 1000,
                          temperature: float = 300.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate full I-V curve.
        
        Args:
            bias_range: Array of bias voltages
            energy_range: (E_min, E_max) for integration
            num_points: Number of integration points
            temperature: Temperature in Kelvin
            
        Returns:
            (bias_voltages, currents)
        """
        currents = []
        
        for V in bias_range:
            I = self.calculate_current(V, energy_range, num_points, temperature)
            currents.append(I)
        
        return bias_range, np.array(currents)
    
    def calculate_differential_conductance(self, bias_voltage: float,
                                          energy_range: Tuple[float, float],
                                          delta_V: float = 0.001,
                                          **kwargs) -> float:
        """
        Calculate differential conductance dI/dV at given bias.
        """
        I_plus = self.calculate_current(bias_voltage + delta_V/2, 
                                        energy_range, **kwargs)
        I_minus = self.calculate_current(bias_voltage - delta_V/2, 
                                         energy_range, **kwargs)
        
        return (I_plus - I_minus) / delta_V


class SiestaInterface:
    """
    Interface to SIESTA/TranSIESTA for reading/writing input/output files.
    """
    
    def __init__(self, siesta_path: Optional[str] = None):
        self.siesta_path = siesta_path or "siesta"
        self.transiesta_path = siesta_path or "transiesta"
    
    def read_tshs_file(self, filename: str) -> Dict:
        """
        Read TranSIESTA TSHS (TranSIESTA Hamiltonian and Structure) file.
        
        TSHS files contain the Hamiltonian and overlap matrices in a
        format optimized for transport calculations.
        """
        # This is a simplified implementation
        # Real implementation would use proper binary parsing
        
        data = {
            'filename': filename,
            'hamiltonian': None,
            'overlap': None,
            'coordinates': None,
            'species': None
        }
        
        # TODO: Implement actual TSHS binary format parsing
        # For now, return placeholder
        warnings.warn("TSHS file reading not fully implemented")
        
        return data
    
    def write_transiesta_input(self, structure: TransportStructure,
                               filename: str = "transiesta.fdf",
                               **kwargs) -> str:
        """
        Generate TranSIESTA input file (FDF format).
        """
        lines = [
            "# TranSIESTA input file generated by dftlammps",
            "",
            "# General settings",
            "SystemName transport_calculation",
            "SystemLabel trans",
            "",
            "# DFT settings",
            "XC.functional GGA",
            "XC.authors PBE",
            "MeshCutoff 300 Ry",
            "",
            "# TranSIESTA specific",
            "TS.Voltage 0.0 eV",
            "TS.Elecs.Bulk true",
            "TS.Elecs.DM.Update cross-terms",
            "",
            "# Left electrode",
            "%block TS.Elec.Left",
            "  HS trans_Left.TSHS",
            "  semi-inf-direction -a3",
            "  electrode-position 1",
            "%endblock TS.Elec.Left",
            "",
            "# Right electrode", 
            "%block TS.Elec.Right",
            "  HS trans_Right.TSHS",
            "  semi-inf-direction +a3",
            "  electrode-position end -1",
            "%endblock TS.Elec.Right",
            "",
            "# Electronic structure",
            "%block TS.ChemPot.Left",
            "  mu V/2",
            "  contour.eq",
            "    begin",
            "      C-1",
            "      C-2",
            "      C-3",
            "      C-4",
            "    end",
            "%endblock TS.ChemPot.Left",
            "",
            "# k-grid",
            "%block kgrid.MonkhorstPack",
            "  3 3 100 0 0 0",
            "%endblock kgrid.MonkhorstPack",
        ]
        
        content = "\n".join(lines)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(content)
        
        return content
    
    def parse_transiesta_output(self, output_file: str) -> Dict:
        """
        Parse TranSIESTA output file for transmission and other results.
        """
        results = {
            'transmission': [],
            'current': None,
            'fermi_level': None,
            'converged': False
        }
        
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    if "Transmission" in line:
                        # Parse transmission data
                        pass
                    elif "Total current" in line:
                        # Parse current
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                results['current'] = float(parts[-2])
                            except ValueError:
                                pass
                    elif "SCF converged" in line:
                        results['converged'] = True
        except FileNotFoundError:
            warnings.warn(f"Output file {output_file} not found")
        
        return results


class TransmissionAnalyzer:
    """
    Analyzer for transmission coefficients and related quantities.
    """
    
    def __init__(self, energies: np.ndarray, 
                 transmissions: np.ndarray):
        self.energies = energies
        self.transmissions = transmissions
    
    def find_resonances(self, threshold: float = 0.5) -> List[Dict]:
        """
        Find transmission resonances (peaks).
        
        Returns:
            List of resonance dictionaries with 'energy', 'height', 'width'
        """
        resonances = []
        
        # Find local maxima
        for i in range(1, len(self.transmissions) - 1):
            if (self.transmissions[i] > self.transmissions[i-1] and
                self.transmissions[i] > self.transmissions[i+1] and
                self.transmissions[i] > threshold):
                
                # Estimate width at half maximum
                peak_height = self.transmissions[i]
                half_max = peak_height / 2
                
                # Find left half-max point
                left_idx = i
                while left_idx > 0 and self.transmissions[left_idx] > half_max:
                    left_idx -= 1
                
                # Find right half-max point
                right_idx = i
                while right_idx < len(self.transmissions) - 1 and \
                      self.transmissions[right_idx] > half_max:
                    right_idx += 1
                
                width = self.energies[right_idx] - self.energies[left_idx]
                
                resonances.append({
                    'energy': self.energies[i],
                    'height': peak_height,
                    'width': width,
                    'index': i
                })
        
        return resonances
    
    def calculate_conductance(self, fermi_level: float = 0.0) -> float:
        """
        Calculate zero-bias conductance in units of G0 = 2e²/h.
        """
        # Interpolate transmission at Fermi level
        conductance = np.interp(fermi_level, self.energies, self.transmissions)
        return conductance
    
    def calculate_seebeck_coefficient(self, fermi_level: float = 0.0,
                                     delta_E: float = 0.1) -> float:
        """
        Estimate Seebeck coefficient from transmission slope.
        
        S ≈ -(π²/3e)(k_B T) (∂lnT/∂E)|_Ef
        """
        # Calculate derivative at Fermi level
        idx = np.argmin(np.abs(self.energies - fermi_level))
        
        # Finite difference derivative
        dT_dE = (self.transmissions[idx+1] - self.transmissions[idx-1]) / \
                (self.energies[idx+1] - self.energies[idx-1])
        
        T_Ef = self.transmissions[idx]
        
        if T_Ef > 1e-10:
            dlnT_dE = dT_dE / T_Ef
        else:
            dlnT_dE = 0
        
        # Seebeck coefficient (simplified formula)
        kB = 8.617e-5  # eV/K
        T = 300  # Room temperature
        
        S = - (np.pi**2 / 3) * (kB * T) * dlnT_dE / 1.602e-19  # V/K
        
        return S


def example_molecular_junction():
    """
    Example: Calculate transmission through a molecular junction.
    """
    print("=" * 60)
    print("Example: Molecular Junction Transport Calculation")
    print("=" * 60)
    
    # Create simple electrode configurations
    n_orbital = 10
    
    # Left electrode (metallic)
    h0_L = np.diag([-2.0] * n_orbital) + 0.5 * (np.diag([1.0] * (n_orbital-1), 1) + 
                                                  np.diag([1.0] * (n_orbital-1), -1))
    h1_L = 0.3 * np.eye(n_orbital)
    s0_L = np.eye(n_orbital)
    s1_L = 0.1 * np.eye(n_orbital)
    
    left_elec = ElectrodeConfig(
        name="Left_Au",
        electrode_type=ElectrodeType.LEFT,
        num_atoms=5,
        fermi_level=0.0,
        num_orbitals=n_orbital,
        h0=csr_matrix(h0_L),
        h1=csr_matrix(h1_L),
        s0=csr_matrix(s0_L),
        s1=csr_matrix(s1_L)
    )
    
    # Right electrode
    right_elec = ElectrodeConfig(
        name="Right_Au",
        electrode_type=ElectrodeType.RIGHT,
        num_atoms=5,
        fermi_level=0.0,
        num_orbitals=n_orbital,
        h0=csr_matrix(h0_L),
        h1=csr_matrix(h1_L),
        s0=csr_matrix(s0_L),
        s1=csr_matrix(s1_L)
    )
    
    # Scattering region (molecule + contact atoms)
    n_scatter = 20
    
    # Simple model: molecule with HOMO-LUMO gap
    h_scatter = np.diag([-1.5] * 10 + [2.0] * 10)  # HOMO-LUMO gap
    # Coupling between sites
    for i in range(n_scatter - 1):
        h_scatter[i, i+1] = 0.5
        h_scatter[i+1, i] = 0.5
    
    s_scatter = np.eye(n_scatter)
    
    # Coupling to electrodes
    h_lc = np.zeros((n_orbital, n_scatter))
    h_lc[:, :n_orbital] = 0.4 * np.eye(n_orbital)
    
    h_cr = np.zeros((n_scatter, n_orbital))
    h_cr[-n_orbital:, :] = 0.4 * np.eye(n_orbital)
    
    # Create transport structure
    structure = TransportStructure(
        left_electrode=left_elec,
        right_electrode=right_elec,
        num_scatter_atoms=10,
        num_scatter_orbitals=n_scatter,
        h_scatter=csr_matrix(h_scatter),
        s_scatter=csr_matrix(s_scatter),
        h_lc=csr_matrix(h_lc),
        h_cr=csr_matrix(h_cr),
        s_lc=csr_matrix(0.1 * h_lc),
        s_cr=csr_matrix(0.1 * h_cr)
    )
    
    # Create NEGF system
    negf = NEGFSystem(structure)
    
    # Calculate transmission
    energies = np.linspace(-3, 3, 200)
    transmissions = []
    
    print("\nCalculating transmission...")
    for E in energies:
        T = negf.calculate_transmission(E)
        transmissions.append(T)
    
    transmissions = np.array(transmissions)
    
    # Analyze results
    analyzer = TransmissionAnalyzer(energies, transmissions)
    resonances = analyzer.find_resonances()
    conductance = analyzer.calculate_conductance()
    
    print(f"\nResults:")
    print(f"  Zero-bias conductance: {conductance:.4f} G₀")
    print(f"  Number of resonances: {len(resonances)}")
    
    if resonances:
        print(f"\n  Resonances:")
        for res in resonances[:3]:  # Show first 3
            print(f"    E = {res['energy']:.3f} eV, "
                  f"T = {res['height']:.3f}, "
                  f"width = {res['width']:.3f} eV")
    
    # Calculate I-V curve
    print("\nCalculating I-V curve...")
    iv_calc = IVCalculator(negf)
    
    bias_range = np.linspace(0, 2.0, 21)
    biases, currents = iv_calc.calculate_iv_curve(
        bias_range, (-3, 3), num_points=500
    )
    
    print(f"\nI-V characteristics (first 5 points):")
    for i in range(min(5, len(biases))):
        print(f"  V = {biases[i]:.2f} V, I = {currents[i]*1e6:.3f} μA")
    
    return {
        'energies': energies,
        'transmissions': transmissions,
        'biases': biases,
        'currents': currents,
        'resonances': resonances
    }


if __name__ == "__main__":
    results = example_molecular_junction()
    
    print("\n" + "=" * 60)
    print("TranSIESTA Interface Module - Test Complete")
    print("=" * 60)
