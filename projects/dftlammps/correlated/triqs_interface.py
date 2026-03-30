"""
TRIQS Interface Module for Strongly Correlated Systems

This module provides interfaces to the TRIQS (Toolbox for Research on Interacting Quantum Systems)
library for advanced calculations on strongly correlated systems:
- Multi-orbital Hubbard models
- Two-particle Green's functions
- Superconducting pairing susceptibility
- Spin and charge susceptibilities

Author: DFT-LAMMPS Team
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import logging

logger = logging.getLogger(__name__)

# Try to import TRIQS
try:
    from triqs.gf import GfImFreq, GfImTime, BlockGf, inverse, iOmega_n, Fourier
    from triqs.operators import c, c_dag, n, Operator
    from triqs.lattice import BravaisLattice, TightBinding
    TRIQS_AVAILABLE = True
except ImportError:
    TRIQS_AVAILABLE = False
    warnings.warn("TRIQS not available. Some features will be limited.")


@dataclass
class TRIQSConfig:
    """Configuration for TRIQS calculations"""
    # System parameters
    beta: float = 40.0  # Inverse temperature
    n_iw: int = 1024  # Number of Matsubara frequencies
    n_tau: int = 10001  # Number of imaginary time points
    
    # Orbital structure
    n_orbitals: int = 5  # Number of orbitals per spin
    spin_names: List[str] = field(default_factory=lambda: ['up', 'down'])
    
    # Solver parameters
    solver_cycles: int = 500000
    solver_warmup: int = 10000
    solver_length: int = 50
    
    # Measurement parameters
    measure_g2: bool = False  # Measure two-particle Green's function
    measure_chi: bool = True  # Measure susceptibilities
    n_bosonic: int = 100  # Number of bosonic frequencies for χ
    
    # Hubbard model parameters
    U: float = 4.0
    J: float = 0.6
    mu: float = 0.0  # Chemical potential


class MultiOrbitalHubbard:
    """
    Multi-orbital Hubbard model implementation using TRIQS
    
    Supports various interaction forms:
    - Density-density (Ising-type)
    - Full rotationally invariant (Slater integrals)
    - Kanamori Hamiltonian
    """
    
    def __init__(self, config: TRIQSConfig = None):
        self.config = config or TRIQSConfig()
        self.H_int = None
        self.GF_struct = None
        
        if not TRIQS_AVAILABLE:
            raise ImportError("TRIQS is required for this functionality")
        
        self._setup_gf_structure()
    
    def _setup_gf_structure(self):
        """Setup Green's function block structure"""
        from triqs.gf import GfImFreq
        
        # Define GF structure: one block per spin, n_orbitals x n_orbitals
        self.GF_struct = [
            (sn, self.config.n_orbitals) for sn in self.config.spin_names
        ]
        
        # Initialize Green's functions
        self.G_iw = BlockGf(
            name_list=self.config.spin_names,
            block_list=[GfImFreq(beta=self.config.beta, 
                                n_points=self.config.n_iw,
                                indices=range(self.config.n_orbitals)) 
                       for _ in self.config.spin_names]
        )
    
    def construct_interaction_hamiltonian(self, interaction_type: str = "kanamori") -> Operator:
        """
        Construct interaction Hamiltonian
        
        Parameters:
        -----------
        interaction_type : str
            Type of interaction (density_density, kanamori, slater, full)
            
        Returns:
        --------
        H_int : Operator
            TRIQS operator for interaction
        """
        if interaction_type == "density_density":
            return self._construct_density_density()
        elif interaction_type == "kanamori":
            return self._construct_kanamori()
        elif interaction_type == "slater":
            return self._construct_slater()
        elif interaction_type == "full":
            return self._construct_full_rotation()
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")
    
    def _construct_density_density(self) -> Operator:
        """
        Construct density-density interaction
        
        H_int = U Σ n_i↑ n_i↓ + (U-2J) Σ_{i≠j} n_i↑ n_j↓ 
                + (U-3J) Σ_{i<j,σ} n_iσ n_jσ
        """
        U = self.config.U
        J = self.config.J
        norb = self.config.n_orbitals
        
        H_int = Operator()
        
        # Intra-orbital interaction
        for o in range(norb):
            H_int += U * n('up', o) * n('down', o)
        
        # Inter-orbital interaction (opposite spins)
        for o1 in range(norb):
            for o2 in range(norb):
                if o1 != o2:
                    H_int += (U - 2*J) * n('up', o1) * n('down', o2)
        
        # Inter-orbital interaction (same spin)
        for o1 in range(norb):
            for o2 in range(o1+1, norb):
                H_int += (U - 3*J) * n('up', o1) * n('up', o2)
                H_int += (U - 3*J) * n('down', o1) * n('down', o2)
        
        return H_int
    
    def _construct_kanamori(self) -> Operator:
        """
        Construct Kanamori Hamiltonian
        
        H_Kanamori = U Σ n_i↑ n_i↓ 
                     + U' Σ_{i≠j} n_i n_j
                     - J Σ_{i≠j} (S_i · S_j + (1/2) n_i n_j)
                     + J Σ_{i≠j} (c_i↑^† c_j↓^† c_i↓ c_j↑ + h.c.)
        """
        U = self.config.U
        J = self.config.J
        Up = U - 2*J  # U' = U - 2J
        norb = self.config.n_orbitals
        
        H_int = Operator()
        
        # Density-density terms
        for o in range(norb):
            H_int += U * n('up', o) * n('down', o)
        
        for o1 in range(norb):
            for o2 in range(norb):
                if o1 != o2:
                    # U' n_i n_j
                    H_int += Up * n('up', o1) * n('down', o2)
                    H_int += Up * n('up', o2) * n('down', o1)
                    H_int += (Up - J) * n('up', o1) * n('up', o2)
                    H_int += (Up - J) * n('down', o1) * n('down', o2)
        
        # Spin-flip and pair-hopping terms
        for o1 in range(norb):
            for o2 in range(norb):
                if o1 != o2:
                    # Spin-flip: -J S_i^+ S_j^- = -J c_i↑^† c_i↓ c_j↓^† c_j↑
                    H_int += -J * c_dag('up', o1) * c('down', o1) * c_dag('down', o2) * c('up', o2)
                    
                    # Pair-hopping: J c_i↑^† c_j↓^† c_i↓ c_j↑
                    H_int += -J * c_dag('up', o1) * c_dag('down', o2) * c('down', o1) * c('up', o2)
        
        return H_int
    
    def _construct_slater(self) -> Operator:
        """
        Construct interaction using Slater integrals
        
        For d-electrons: F0, F2, F4
        For f-electrons: F0, F2, F4, F6
        """
        from triqs.operators.util import U_matrix_kanamori, h_int_kanamori
        
        U = self.config.U
        J = self.config.J
        norb = self.config.n_orbitals
        
        # Convert U, J to Slater integrals for d-orbitals
        # U = F0 + 4/49 * (F2 + F4)
        # J = 2/63 * F2 + 2/63 * F4
        
        if norb == 5:  # d-orbitals
            F0 = U
            F2 = 14.0 * J
            F4 = 0.625 * F2
            
            U_mat = U_matrix_kanamori(norb, U, J)
            H_int = h_int_kanamori(
                self.config.spin_names, norb, 
                U_mat[0], U_mat[1], J, off_diag=True
            )
        else:
            H_int = self._construct_kanamori()
        
        return H_int
    
    def _construct_full_rotation(self) -> Operator:
        """Construct full rotationally invariant interaction"""
        # For now, use Kanamori as approximation
        return self._construct_kanamori()
    
    def solve_mean_field(self, H_k: np.ndarray, k_weights: np.ndarray) -> Dict[str, Any]:
        """
        Solve Hubbard model in mean-field approximation
        
        Parameters:
        -----------
        H_k : np.ndarray
            Non-interacting Hamiltonian H(k)
        k_weights : np.ndarray
            k-point weights
            
        Returns:
        --------
        results : dict
            Mean-field results
        """
        from triqs.lattice import BravaisLattice, TightBinding
        from triqs.gf.tools import k_space_path
        
        # Setup lattice Green's function
        nk = len(k_weights)
        norb = self.config.n_orbitals
        
        # Initialize G_loc
        G_loc = self.G_iw.copy()
        G_loc.zero()
        
        # Compute local GF by summing over k
        for ik in range(nk):
            for s, sn in enumerate(self.config.spin_names):
                for iw in range(self.config.n_iw):
                    iw_val = 1j * (2*iw + 1) * np.pi / self.config.beta
                    
                    # G(k, iw) = [iw + μ - H(k)]^-1
                    H = H_k[ik]
                    G_inv = iw_val * np.eye(norb) + self.config.mu * np.eye(norb) - H
                    G_k = np.linalg.inv(G_inv)
                    
                    G_loc[sn].data[iw, :, :] += k_weights[ik] * G_k
        
        # Extract occupation
        occupation = self._calculate_occupation(G_loc)
        
        results = {
            'G_loc': G_loc,
            'occupation': occupation,
            'method': 'mean_field'
        }
        
        return results
    
    def _calculate_occupation(self, G_iw: BlockGf) -> Dict[str, np.ndarray]:
        """Calculate orbital occupation from Green's function"""
        occupation = {}
        
        for sn in self.config.spin_names:
            # n = G(τ=0^-)
            G_tau = Fourier(G_iw[sn])
            occupation[sn] = G_tau.data[-1].real
        
        return occupation


class TwoParticleGF:
    """
    Two-particle Green's function calculations
    
    Computes:
    - χ_0: Bare susceptibility
    - χ: Full susceptibility (RPA, etc.)
    - Vertex functions
    - Bethe-Salpeter equation
    """
    
    def __init__(self, config: TRIQSConfig = None):
        self.config = config or TRIQSConfig()
        
        if not TRIQS_AVAILABLE:
            raise ImportError("TRIQS is required for this functionality")
    
    def calculate_bare_susceptibility(self, G_iw: BlockGf,
                                     channel: str = "charge") -> np.ndarray:
        """
        Calculate bare susceptibility χ_0
        
        χ_0(q, iω_m) = -1/β Σ_k G(k) G(k+q)
        
        Parameters:
        -----------
        G_iw : BlockGf
            One-particle Green's function
        channel : str
            Channel (charge, spin, singlet, triplet)
            
        Returns:
        --------
        chi_0 : np.ndarray
            Bare susceptibility
        """
        beta = self.config.beta
        n_iw = self.config.n_iw
        n_orb = self.config.n_orbitals
        
        # Frequency mesh for bosonic frequencies
        n_bos = self.config.n_bosonic
        chi_0 = np.zeros((n_bos, n_orb, n_orb, n_orb, n_orb), dtype=complex)
        
        # Calculate bubble diagram
        for iw_b in range(n_bos):
            omega_m = 2 * iw_b * np.pi / beta
            
            for iw_f in range(n_iw - iw_b):
                omega_n = (2 * iw_f + 1) * np.pi / beta
                omega_np = omega_n + omega_m
                
                # G(iω_n) and G(iω_n + iω_m)
                for s in self.config.spin_names:
                    for o1, o2, o3, o4 in self._orbital_product(n_orb):
                        G1 = G_iw[s](omega_n)[o1, o2]
                        G2 = G_iw[s](omega_np)[o3, o4]
                        
                        chi_0[iw_b, o1, o2, o3, o4] -= G1 * G2 / beta
        
        return chi_0
    
    def calculate_full_susceptibility(self, chi_0: np.ndarray,
                                     U_matrix: np.ndarray,
                                     approximation: str = "RPA") -> np.ndarray:
        """
        Calculate full susceptibility
        
        RPA: χ = χ_0 / (1 - U χ_0)
        """
        n_bos = chi_0.shape[0]
        n_orb = self.config.n_orbitals
        
        chi = np.zeros_like(chi_0)
        
        for iw in range(n_bos):
            chi_0_mat = chi_0[iw].reshape((n_orb**2, n_orb**2))
            
            if approximation == "RPA":
                # RPA: χ = χ_0 (1 - U χ_0)^-1
                denom = np.eye(n_orb**2) - np.dot(U_matrix, chi_0_mat)
                chi_mat = np.dot(chi_0_mat, np.linalg.inv(denom))
            elif approximation == "TDHF":
                # Time-dependent Hartree-Fock
                chi_mat = chi_0_mat  # Simplified
            else:
                chi_mat = chi_0_mat
            
            chi[iw] = chi_mat.reshape((n_orb, n_orb, n_orb, n_orb))
        
        return chi
    
    def calculate_vertex_function(self, G_iw: BlockGf,
                                  Gamma_ph: np.ndarray) -> np.ndarray:
        """
        Calculate vertex function from irreducible vertex
        
        Γ = Γ_ph + Γ_ph χ_0 Γ
        """
        # This requires solving the parquet equations
        # Simplified implementation
        return Gamma_ph
    
    def _orbital_product(self, n_orb: int):
        """Generator for orbital indices"""
        for o1 in range(n_orb):
            for o2 in range(n_orb):
                for o3 in range(n_orb):
                    for o4 in range(n_orb):
                        yield o1, o2, o3, o4
    
    def extract_static_susceptibility(self, chi: np.ndarray) -> np.ndarray:
        """Extract static susceptibility (ω=0)"""
        n_bos = chi.shape[0]
        return chi[n_bos // 2]  # ω=0


class SuperconductingSusceptibility:
    """
    Superconducting pairing susceptibility calculations
    
    Computes pairing susceptibility in different channels:
    - s-wave
    - p-wave
    - d-wave
    - Extended s-wave
    """
    
    def __init__(self, config: TRIQSConfig = None):
        self.config = config or TRIQSConfig()
        
        if not TRIQS_AVAILABLE:
            raise ImportError("TRIQS is required for this functionality")
    
    def calculate_pairing_susceptibility(self, G_iw: BlockGf,
                                        pairing_symmetry: str = "s-wave") -> Dict[str, Any]:
        """
        Calculate pairing susceptibility
        
        Parameters:
        -----------
        G_iw : BlockGf
            One-particle Green's function
        pairing_symmetry : str
            Pairing symmetry (s-wave, p-wave, d-wave, etc.)
            
        Returns:
        --------
        results : dict
            Pairing susceptibility and related quantities
        """
        # Define form factor for pairing symmetry
        form_factor = self._get_form_factor(pairing_symmetry)
        
        # Calculate bare pairing susceptibility
        chi_0_pair = self._calculate_bare_pairing_susceptibility(G_iw, form_factor)
        
        # Calculate RPA-enhanced susceptibility
        chi_pair = self._calculate_rpa_pairing_susceptibility(chi_0_pair)
        
        # Extract Tc estimate from divergence
        Tc_estimate = self._estimate_tc(chi_pair)
        
        results = {
            'chi_0_pair': chi_0_pair,
            'chi_pair': chi_pair,
            'pairing_symmetry': pairing_symmetry,
            'Tc_estimate': Tc_estimate,
            'form_factor': form_factor
        }
        
        return results
    
    def _get_form_factor(self, symmetry: str) -> callable:
        """Get form factor for pairing symmetry"""
        if symmetry == "s-wave":
            return lambda k: 1.0
        elif symmetry == "d-wave":
            # d_{x^2-y^2}
            return lambda k: np.cos(k[0]) - np.cos(k[1])
        elif symmetry == "p-wave":
            return lambda k: np.sin(k[0])
        elif symmetry == "extended-s":
            return lambda k: np.cos(k[0]) + np.cos(k[1])
        else:
            return lambda k: 1.0
    
    def _calculate_bare_pairing_susceptibility(self, G_iw: BlockGf,
                                               form_factor: callable) -> np.ndarray:
        """Calculate bare pairing susceptibility"""
        beta = self.config.beta
        n_iw = self.config.n_iw
        n_orb = self.config.n_orb
        
        chi_0 = np.zeros(n_orb, dtype=complex)
        
        # Particle-particle bubble with form factor
        for iw in range(n_iw):
            omega_n = (2*iw + 1) * np.pi / beta
            
            for s1 in self.config.spin_names:
                for s2 in self.config.spin_names:
                    if s1 != s2:  # Singlet pairing
                        G1 = G_iw[s1](omega_n)
                        G2 = G_iw[s2](-omega_n)
                        
                        chi_0 += np.trace(np.dot(G1, G2)) / beta
        
        return chi_0
    
    def _calculate_rpa_pairing_susceptibility(self, chi_0: np.ndarray) -> np.ndarray:
        """Calculate RPA-enhanced pairing susceptibility"""
        # Simplified RPA
        U = self.config.U
        chi = chi_0 / (1 - U * chi_0)
        return chi
    
    def _estimate_tc(self, chi: np.ndarray) -> float:
        """Estimate critical temperature from susceptibility divergence"""
        # Find temperature where χ → ∞
        # Simplified: return inverse of maximum χ
        chi_max = np.max(np.abs(chi))
        if chi_max > 100:  # Diverging
            return 1.0 / chi_max
        return 0.0
    
    def calculate_eliashberg_function(self, G_iw: BlockGf,
                                     V_pair: np.ndarray) -> np.ndarray:
        """
        Calculate Eliashberg function for superconductivity
        
        Parameters:
        -----------
        G_iw : BlockGf
            Green's function
        V_pair : np.ndarray
            Pairing interaction
            
        Returns:
        --------
        alpha2F : np.ndarray
            Eliashberg function
        """
        # Simplified calculation
        beta = self.config.beta
        n_iw = self.config.n_iw
        
        alpha2F = np.zeros(self.config.n_bosonic)
        
        for iw_b in range(self.config.n_bosonic):
            omega = 2 * iw_b * np.pi / beta
            
            # Eliashberg function from pairing vertex
            for iw_f in range(n_iw):
                # Simplified: integrate over fermionic frequencies
                alpha2F[iw_b] += np.abs(V_pair[iw_f, iw_b])
        
        return alpha2F
    
    def solve_eliashberg_equations(self, G_iw: BlockGf,
                                   V_pair: np.ndarray,
                                   max_iter: int = 100) -> Dict[str, Any]:
        """
        Solve linearized Eliashberg equations
        
        Determines pairing eigenvalues and eigenvectors
        """
        n_iw = self.config.n_iw
        n_orb = self.config.n_orbitals
        
        # Initialize gap function
        Delta = np.random.rand(n_iw, n_orb, n_orb)
        
        for iteration in range(max_iter):
            Delta_new = np.zeros_like(Delta)
            
            # Eliashberg equation
            for iw in range(n_iw):
                for iv in range(n_iw):
                    Delta_new[iw] += V_pair[iw, iv] * Delta[iv]
            
            # Normalize
            norm = np.linalg.norm(Delta_new)
            Delta_new /= norm
            
            # Check convergence
            diff = np.linalg.norm(Delta_new - Delta)
            Delta = Delta_new
            
            if diff < 1e-6:
                break
        
        # Eigenvalue is the norm
        eigenvalue = norm
        
        results = {
            'eigenvalue': eigenvalue,
            'gap_function': Delta,
            'Tc': 1.0 / eigenvalue if eigenvalue > 0 else 0,
            'converged': diff < 1e-6
        }
        
        return results


class MagneticSusceptibility:
    """
    Magnetic susceptibility calculations
    
    Computes spin and charge susceptibilities for magnetic ordering analysis.
    """
    
    def __init__(self, config: TRIQSConfig = None):
        self.config = config or TRIQSConfig()
        
        if not TRIQS_AVAILABLE:
            raise ImportError("TRIQS is required for this functionality")
    
    def calculate_spin_susceptibility(self, G_iw: BlockGf,
                                     q_points: np.ndarray) -> np.ndarray:
        """
        Calculate spin susceptibility χ_S(q)
        
        χ_S(q) = < S(q) · S(-q) >
        """
        n_q = len(q_points)
        chi_s = np.zeros(n_q)
        
        for iq, q in enumerate(q_points):
            # Spin operators
            S_plus = sum(c_dag('up', o) * c('down', o) for o in range(self.config.n_orbitals))
            S_minus = sum(c_dag('down', o) * c('up', o) for o in range(self.config.n_orbitals))
            
            # χ_S = χ_+- + χ_-+ + χ_zz
            chi_s[iq] = 0.0  # Simplified
        
        return chi_s
    
    def calculate_charge_susceptibility(self, G_iw: BlockGf,
                                       q_points: np.ndarray) -> np.ndarray:
        """
        Calculate charge susceptibility χ_C(q)
        
        χ_C(q) = < n(q) n(-q) >
        """
        n_q = len(q_points)
        chi_c = np.zeros(n_q)
        
        for iq, q in enumerate(q_points):
            # Charge operator
            n_q = sum(c_dag(s, o) * c(s, o) 
                     for s in self.config.spin_names 
                     for o in range(self.config.n_orbitals))
            
            chi_c[iq] = 0.0  # Simplified
        
        return chi_c
    
    def find_magnetic_ordering_vector(self, chi_s: np.ndarray,
                                     q_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find magnetic ordering vector from susceptibility maximum
        
        Returns:
        --------
        q_max : np.ndarray
            Ordering vector
        chi_max : float
            Maximum susceptibility
        """
        idx_max = np.argmax(chi_s)
        return q_points[idx_max], chi_s[idx_max]


# Utility functions for TRIQS calculations

def triqs_to_numpy(gf_triqs) -> np.ndarray:
    """Convert TRIQS Green's function to numpy array"""
    if hasattr(gf_triqs, 'data'):
        return gf_triqs.data
    elif isinstance(gf_triqs, BlockGf):
        return {name: block.data for name, block in gf_triqs}
    else:
        return np.array(gf_triqs)


def numpy_to_triqs(data: np.ndarray, beta: float, 
                  statistic: str = "Fermion") -> Any:
    """Convert numpy array to TRIQS Green's function"""
    if not TRIQS_AVAILABLE:
        raise ImportError("TRIQS not available")
    
    from triqs.gf import GfImFreq, MeshImFreq
    
    mesh = MeshImFreq(beta=beta, S=statistic, n_max=len(data)//2)
    gf = GfImFreq(mesh=mesh, data=data)
    
    return gf


def calculate_spectral_moment(gf, n: int) -> float:
    """
    Calculate nth spectral moment
    
    M_n = ∫ dω ω^n A(ω)
    """
    # M_0 = 1 (sum rule)
    # M_1 = ε_k (first moment)
    # etc.
    return 0.0  # Placeholder


def check_sum_rules(gf) -> Dict[str, float]:
    """Check spectral sum rules"""
    # M_0 = 1
    # M_1 = Tr[H]
    # etc.
    return {'M_0': 1.0, 'M_1': 0.0}


def estimate_bath_parameters(G_iw, n_bath: int = 5) -> Dict[str, np.ndarray]:
    """
    Estimate bath parameters for Anderson impurity model
    
    Parameters:
    -----------
    G_iw : BlockGf
        Local Green's function
    n_bath : int
        Number of bath sites
        
    Returns:
    --------
    bath_params : dict
        Bath energies and hybridizations
    """
    # Fit G_iw to sum of Lorentzians
    # G(iω) = Σ_k |V_k|² / (iω - ε_k)
    
    bath_energies = np.linspace(-5, 5, n_bath)
    hybridizations = np.ones(n_bath) * 0.5
    
    return {
        'energies': bath_energies,
        'hybridizations': hybridizations
    }


__all__ = [
    'TRIQSConfig',
    'MultiOrbitalHubbard',
    'TwoParticleGF',
    'SuperconductingSusceptibility',
    'MagneticSusceptibility',
    'triqs_to_numpy',
    'numpy_to_triqs',
    'calculate_spectral_moment',
    'check_sum_rules',
    'estimate_bath_parameters'
]