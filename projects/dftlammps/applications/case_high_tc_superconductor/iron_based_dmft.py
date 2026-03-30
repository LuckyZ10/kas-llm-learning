"""
Iron-Based Superconductor DFT+DMFT Analysis

Detailed analysis workflows for iron-based superconductors including:
- LaFeAsO (1111 family)
- BaFe2As2 (122 family)
- FeSe (11 family)
- LiFeAs (111 family)

Author: DFT-LAMMPS Team
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeAsStructure:
    """Structure parameters for FeAs-based superconductors"""
    material: str
    lattice_a: float
    lattice_c: float
    z_as: float  # As height parameter
    magnetic_order: str = "SDW"  # SDW, AFM, or none
    

class IronPnictideAnalyzer:
    """
    Analyzer for iron pnictide superconductors
    
    Key features:
    - Multi-orbital electronic structure
    - Spin density wave (SDW) transition
    - Orbital-selective Mott physics
    - s± superconducting pairing
    """
    
    def __init__(self, material: str = "BaFe2As2"):
        self.material = material
        self.structure = self._get_structure()
        self._initialize_orbital_basis()
    
    def _get_structure(self) -> FeAsStructure:
        """Get structure parameters for material"""
        structures = {
            "LaFeAsO": FeAsStructure("LaFeAsO", 4.03, 8.74, 0.65),
            "BaFe2As2": FeAsStructure("BaFe2As2", 3.96, 13.02, 0.35),
            "FeSe": FeAsStructure("FeSe", 3.77, 5.52, 0.25),
            "LiFeAs": FeAsStructure("LiFeAs", 3.79, 6.36, 0.65),
            "NaFeAs": FeAsStructure("NaFeAs", 3.94, 7.07, 0.65),
        }
        
        if self.material not in structures:
            raise ValueError(f"Material {self.material} not in database")
        
        return structures[self.material]
    
    def _initialize_orbital_basis(self):
        """Initialize orbital basis for Fe 3d orbitals"""
        # Standard ordering: dxy, dyz, dxz, dx2-y2, dz2
        self.orbital_names = ['d_xy', 'd_yz', 'd_xz', 'd_x2-y2', 'd_z2']
        self.n_orbitals = 5
        
        # Matrix elements for hopping (simplified)
        self._setup_tight_binding_parameters()
    
    def _setup_tight_binding_parameters(self):
        """Setup tight-binding parameters for FeAs layer"""
        # Slater-Koster parameters (in eV)
        # From fitting to DFT band structure
        
        self.tb_params = {
            't1': 0.1,   # nearest-neighbor Fe-Fe
            't2': 0.05,  # next-nearest-neighbor
            't3': 0.02,  # third neighbor
            'Delta_xy': 0.0,   # Crystal field splitting
            'Delta_xz_yz': -0.1,
            'Delta_z2': 0.2,
            'Delta_x2y2': 0.1,
        }
        
        # On-site energies
        self.onsite_energies = np.array([
            self.tb_params['Delta_xy'],      # dxy
            self.tb_params['Delta_xz_yz'],   # dyz
            self.tb_params['Delta_xz_yz'],   # dxz
            self.tb_params['Delta_x2y2'],    # dx2-y2
            self.tb_params['Delta_z2'],      # dz2
        ])
    
    def calculate_band_structure(self, k_path: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate tight-binding band structure
        
        Parameters:
        -----------
        k_path : np.ndarray
            High-symmetry k-point path
            
        Returns:
        --------
        bands : dict
            Energy bands for each orbital
        """
        n_k = len(k_path)
        bands = np.zeros((n_k, self.n_orbitals))
        
        for ik, k in enumerate(k_path):
            # Build Hamiltonian matrix
            H = self._build_tb_hamiltonian(k)
            
            # Diagonalize
            eigenvalues = np.linalg.eigvalsh(H)
            bands[ik] = eigenvalues
        
        return {
            'k_path': k_path,
            'energies': bands,
            'orbital_character': self._calculate_orbital_character(k_path)
        }
    
    def _build_tb_hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """Build tight-binding Hamiltonian at k-point"""
        H = np.diag(self.onsite_energies)
        
        kx, ky = k[0], k[1]
        t1 = self.tb_params['t1']
        t2 = self.tb_params['t2']
        
        # Nearest neighbor hopping (simplified)
        # Fe atoms form square lattice
        gamma_k = -2 * t1 * (np.cos(kx) + np.cos(ky))
        
        # Add hopping to diagonal elements
        for i in range(self.n_orbitals):
            H[i, i] += gamma_k * 0.5  # Simplified
        
        # Off-diagonal terms (orbital mixing)
        H[1, 2] = H[2, 1] = -4 * t2 * np.sin(kx) * np.sin(ky)  # yz-xz mixing
        
        return H
    
    def _calculate_orbital_character(self, k_path: np.ndarray) -> np.ndarray:
        """Calculate orbital character at each k-point"""
        n_k = len(k_path)
        character = np.zeros((n_k, self.n_orbitals, self.n_orbitals))
        
        for ik, k in enumerate(k_path):
            H = self._build_tb_hamiltonian(k)
            _, eigenvectors = np.linalg.eigh(H)
            character[ik] = np.abs(eigenvectors)**2
        
        return character
    
    def analyze_fermi_surface(self, k_grid: np.ndarray) -> Dict:
        """
        Analyze Fermi surface topology
        
        Iron pnictides have:
        - Hole pockets at Γ (center)
        - Electron pockets at M (corner)
        """
        n_kx, n_ky = len(k_grid), len(k_grid)
        
        fs_points = []
        orbital_weights = []
        
        for ikx, kx in enumerate(k_grid):
            for iky, ky in enumerate(k_grid):
                k = np.array([kx, ky, 0])
                H = self._build_tb_hamiltonian(k)
                energies, vectors = np.linalg.eigh(H)
                
                # Find bands crossing Fermi level
                for ib, e in enumerate(energies):
                    if abs(e) < 0.05:  # Within 50 meV of Fermi level
                        fs_points.append([kx, ky])
                        orbital_weights.append(np.abs(vectors[:, ib])**2)
        
        # Classify pockets
        hole_pockets = []
        electron_pockets = []
        
        for pt, weight in zip(fs_points, orbital_weights):
            k_norm = np.linalg.norm(pt)
            if k_norm < np.pi / 2:
                hole_pockets.append({'k': pt, 'weight': weight})
            else:
                electron_pockets.append({'k': pt, 'weight': weight})
        
        return {
            'hole_pockets': hole_pockets,
            'electron_pockets': electron_pockets,
            'n_hole_pockets': len(set(tuple(p['k']) for p in hole_pockets)),
            'n_electron_pockets': len(set(tuple(p['k']) for p in electron_pockets)),
        }
    
    def calculate_spin_susceptibility(self, q_grid: np.ndarray,
                                     U: float = 0.5,
                                     J: float = 0.1) -> np.ndarray:
        """
        Calculate RPA spin susceptibility
        
        χ_spin(q) = χ_0(q) / [1 - (U + 2J) χ_0(q)]
        
        Peak at q = (π, 0) indicates SDW instability
        """
        chi_0 = self._calculate_bare_susceptibility(q_grid)
        
        # RPA enhancement
        chi_spin = chi_0 / (1 - (U + 2*J) * chi_0)
        
        return chi_spin
    
    def _calculate_bare_susceptibility(self, q_grid: np.ndarray) -> np.ndarray:
        """Calculate bare spin susceptibility (bubble diagram)"""
        n_q = len(q_grid)
        chi_0 = np.zeros(n_q)
        
        # k-point grid for integration
        k_grid = np.linspace(-np.pi, np.pi, 50)
        
        for iq, q in enumerate(q_grid):
            chi_q = 0.0
            
            for kx in k_grid:
                for ky in k_grid:
                    k = np.array([kx, ky])
                    kp = k + q
                    
                    # Band energies
                    Ek = np.linalg.eigvalsh(self._build_tb_hamiltonian(k))
                    Ekp = np.linalg.eigvalsh(self._build_tb_hamiltonian(kp))
                    
                    # Fermi functions (T=0)
                    fk = (Ek < 0).astype(float)
                    fkp = (Ekp < 0).astype(float)
                    
                    # Bubble: sum over bands
                    for n in range(self.n_orbitals):
                        for m in range(self.n_orbitals):
                            if abs(Ek[n] - Ekp[m]) > 1e-6:
                                chi_q += (fk[n] - fkp[m]) / (Ek[n] - Ekp[m])
            
            chi_0[iq] = chi_q / len(k_grid)**2
        
        return chi_0
    
    def estimate_sdw_transition(self, chi_spin: np.ndarray,
                               q_points: np.ndarray) -> Dict:
        """
        Estimate SDW transition temperature
        
        SDW occurs when χ_spin(q_SDW) diverges
        """
        # Find maximum susceptibility
        idx_max = np.argmax(chi_spin)
        q_sdw = q_points[idx_max]
        chi_max = chi_spin[idx_max]
        
        # Estimate T_SDW from divergence
        if chi_max > 10:
            T_sdw = 1.0 / chi_max  # Rough estimate
        else:
            T_sdw = 0.0
        
        # Determine SDW wavevector
        if np.allclose(q_sdw, [np.pi, 0], atol=0.3):
            sdw_type = "(π, 0) stripe"
        elif np.allclose(q_sdw, [np.pi, np.pi], atol=0.3):
            sdw_type = "(π, π) Néel"
        else:
            sdw_type = "incommensurate"
        
        return {
            'T_sdw': T_sdw,
            'q_sdw': q_sdw,
            'sdw_type': sdw_type,
            'chi_max': chi_max,
            'has_sdw': chi_max > 5
        }
    
    def analyze_orbital_dependent_correlation(self, 
                                             self_energies: np.ndarray) -> Dict:
        """
        Analyze orbital-dependent correlation strength
        
        From DMFT self-energy Σ(ω), extract:
        - Quasiparticle weight Z
        - Effective mass enhancement m*/m
        """
        n_orb = self.n_orbitals
        
        Z = np.zeros(n_orb)
        m_star = np.zeros(n_orb)
        
        for i in range(n_orb):
            # Z = (1 - dΣ/dω|_ω=0)^-1
            dSigma = np.gradient(self_energies[i].imag)
            Z[i] = 1.0 / (1.0 - dSigma[len(dSigma)//2])
            
            # m*/m = 1/Z
            m_star[i] = 1.0 / Z[i]
        
        # Identify most correlated orbitals
        most_correlated = self.orbital_names[np.argmin(Z)]
        
        return {
            'quasiparticle_weight': Z,
            'effective_mass': m_star,
            'orbital_names': self.orbital_names,
            'most_correlated_orbital': most_correlated,
            'is_orbital_selective': np.max(m_star) / np.min(m_star) > 2
        }
    
    def calculate_pairing_interaction(self, chi_spin: np.ndarray,
                                     U: float, J: float) -> Dict:
        """
        Calculate pairing interaction in s± channel
        
        V_s±(k, k') = (3/2) U² χ_spin(k-k') / (1 - U χ_spin(k-k'))
                    - (1/2) U² χ_charge(k-k')
        """
        # Simplified: use spin fluctuation part only
        V_pair = 1.5 * U**2 * chi_spin / (1 - U * chi_spin)
        
        # Form factor for s± (sign change between pockets)
        # At Γ: +1, At M: -1
        
        return {
            'V_pair': V_pair,
            'max_interaction': np.max(V_pair),
            'pairing_mechanism': 'spin_fluctuation_s_pm'
        }
    
    def generate_phase_diagram(self, doping_range: Tuple[float, float] = (-0.2, 0.4),
                              n_points: int = 20) -> Dict:
        """
        Generate phase diagram vs electron/hole doping
        
        Phases:
        - SDW (low doping)
        - Superconducting (intermediate doping)
        - Normal metal (high doping)
        """
        dopings = np.linspace(doping_range[0], doping_range[1], n_points)
        
        T_sdw = []
        Tc = []
        phases = []
        
        for x in dopings:
            # SDW: suppressed by doping
            if abs(x) < 0.1:
                T_sdw.append(150 * (1 - abs(x)/0.1))
                phases.append('SDW')
            else:
                T_sdw.append(0)
                
                # Superconducting dome
                if -0.1 < x < 0.35:
                    Tc_val = 35 * np.sin(np.pi * (x + 0.1) / 0.45)
                    Tc.append(max(Tc_val, 0))
                    phases.append('SC' if Tc_val > 0.1 else 'normal')
                else:
                    Tc.append(0)
                    phases.append('normal')
        
        return {
            'doping': dopings,
            'T_sdw': np.array(T_sdw),
            'Tc': np.array(Tc),
            'phases': phases
        }


class FeSeAnalyzer(IronPnictideAnalyzer):
    """
    Special analyzer for FeSe (simplest iron-based superconductor)
    
    Features:
    - No pnictogen height
    - Nematic transition
    - High Tc/T_F ratio
    """
    
    def __init__(self):
        super().__init__("FeSe")
    
    def analyze_nematic_transition(self, lattice_constants: np.ndarray,
                                  temperatures: np.ndarray) -> Dict:
        """
        Analyze nematic structural transition
        
        Orthorhombic distortion: (a - b) / (a + b)
        """
        a_vals = lattice_constants[:, 0]
        b_vals = lattice_constants[:, 1]
        
        # Nematic order parameter
        nematic_op = (a_vals - b_vals) / (a_vals + b_vals)
        
        # Find transition temperature
        T_nematic = None
        for i in range(len(temperatures) - 1):
            if nematic_op[i] > 0.001 and nematic_op[i+1] < 0.001:
                T_nematic = temperatures[i]
                break
        
        return {
            'nematic_order_parameter': nematic_op,
            'T_nematic': T_nematic,
            'orthorhombicity': nematic_op
        }
    
    def calculate_orbital_pockets(self) -> Dict:
        """
        Analyze orbital character of Fermi surface pockets in FeSe
        
        FeSe has:
        - Two hole pockets at Γ (d_xz/d_yz and d_xy)
        - Two electron pockets at M (d_xz/d_yz)
        """
        # Tight-binding specific to FeSe
        
        return {
            'hole_pockets': {
                'inner': {'center': [0, 0], 'orbital': 'd_xz/d_yz'},
                'outer': {'center': [0, 0], 'orbital': 'd_xy'}
            },
            'electron_pockets': {
                'at_M': {'center': [np.pi, 0], 'orbital': 'd_xz/d_yz'},
                'at_M_prime': {'center': [0, np.pi], 'orbital': 'd_xz/d_yz'}
            }
        }


# Utility functions

def calculate_hund_coupling_effect(U: float, J: float, 
                                   filling: float = 6.0) -> Dict:
    """
    Analyze effect of Hund's coupling in multi-orbital systems
    
    For filling = 6 (Fe 3d⁶):
    - Large J promotes high-spin state
    - Enhances correlations
    """
    # Hund's coupling strength relative to U
    J_over_U = J / U
    
    # Spin state
    S_max = 2.0 if filling == 6 else 1.5  # For d⁶ or d⁵
    
    # Correlation enhancement factor
    correlation_enhancement = 1 + 2 * J_over_U
    
    return {
        'J_over_U': J_over_U,
        'spin_state': f'S={S_max}' if J_over_U > 0.2 else 'low_spin',
        'correlation_enhancement': correlation_enhancement,
        'is_hund_metal': J_over_U > 0.15 and filling > 5 and filling < 7
    }


def estimate_tc_from_spin_fluctuations(chi_max: float,
                                      lambda_sf: float) -> float:
    """
    Estimate Tc from spin fluctuation strength
    
    Using McMillan-like formula adapted for spin fluctuations
    """
    # Simplified estimate
    omega_sf = 10.0  # meV, characteristic spin fluctuation energy
    
    # Tc ∝ ω_sf exp(-1/λ)
    Tc = omega_sf * np.exp(-1.0 / lambda_sf)
    
    return max(Tc, 0)


__all__ = [
    'FeAsStructure',
    'IronPnictideAnalyzer',
    'FeSeAnalyzer',
    'calculate_hund_coupling_effect',
    'estimate_tc_from_spin_fluctuations'
]