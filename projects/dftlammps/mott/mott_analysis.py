"""
Mott Insulator Analysis Module

This module provides tools for analyzing Mott insulating behavior, including:
- Gap opening/closing criteria
- Metal-insulator transition detection
- Charge and spin order analysis
- Phase diagrams

Author: DFT-LAMMPS Team
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


@dataclass
class MottAnalysisConfig:
    """Configuration for Mott insulator analysis"""
    # Gap analysis parameters
    gap_threshold: float = 0.01  # eV, minimum gap for insulator
    occupation_tol: float = 1e-3  # Tolerance for integer occupation
    
    # MIT detection
    mit_criterion: str = "gap"  # gap, resistivity, or correlation_length
    mit_smoothing: int = 5  # Points for smoothing
    
    # Order parameters
    order_tol: float = 1e-4  # Tolerance for order detection
    
    # Phase diagram
    u_range: Tuple[float, float] = (0.0, 10.0)
    t_range: Tuple[float, float] = (0.01, 1.0)
    n_points: int = 50


class GapAnalyzer:
    """
    Analyze electronic gap in correlated systems
    
    Detects:
    - Direct and indirect gaps
    - Gap closure points
    - Band narrowing effects
    """
    
    def __init__(self, config: MottAnalysisConfig = None):
        self.config = config or MottAnalysisConfig()
        
    def calculate_gap(self, eigenvalues: np.ndarray, k_points: np.ndarray,
                     occupations: np.ndarray = None, 
                     temperature: float = 0.0) -> Dict[str, float]:
        """
        Calculate electronic gap
        
        Parameters:
        -----------
        eigenvalues : np.ndarray
            Eigenvalues at each k-point (nk, nbands)
        k_points : np.ndarray
            k-point coordinates
        occupations : np.ndarray, optional
            Occupation numbers
        temperature : float
            Temperature for Fermi-Dirac smearing
            
        Returns:
        --------
        gap_info : dict
            Dictionary with gap information
        """
        nk, nbands = eigenvalues.shape
        
        # Find Fermi energy
        if occupations is not None:
            E_Fermi = self._find_fermi_energy(eigenvalues, occupations)
        else:
            E_Fermi = self._find_fermi_energy_at_t(eigenvalues, temperature)
        
        # Separate occupied and unoccupied states
        occupied_max = np.max(eigenvalues[eigenvalues < E_Fermi])
        unoccupied_min = np.min(eigenvalues[eigenvalues > E_Fermi])
        
        # Direct gap at each k-point
        gaps_direct = []
        for ik in range(nk):
            occ = eigenvalues[ik][eigenvalues[ik] < E_Fermi]
            unocc = eigenvalues[ik][eigenvalues[ik] > E_Fermi]
            
            if len(occ) > 0 and len(unocc) > 0:
                gap = np.min(unocc) - np.max(occ)
                gaps_direct.append(gap)
        
        # Indirect gap
        gap_indirect = unoccupied_min - occupied_max
        
        # Minimum direct gap
        gap_direct_min = np.min(gaps_direct) if gaps_direct else gap_indirect
        
        results = {
            'E_Fermi': E_Fermi,
            'gap_indirect': gap_indirect,
            'gap_direct_min': gap_direct_min,
            'gap_direct_max': np.max(gaps_direct) if gaps_direct else gap_indirect,
            'VBM': occupied_max,  # Valence band maximum
            'CBM': unoccupied_min,  # Conduction band minimum
            'is_insulator': gap_indirect > self.config.gap_threshold
        }
        
        return results
    
    def _find_fermi_energy(self, eigenvalues: np.ndarray, 
                          occupations: np.ndarray) -> float:
        """Find Fermi energy from occupations"""
        # Flatten and sort
        all_eig = eigenvalues.flatten()
        all_occ = occupations.flatten()
        
        # Sort by energy
        idx = np.argsort(all_eig)
        all_eig = all_eig[idx]
        all_occ = all_occ[idx]
        
        # Find where occupation changes
        for i in range(len(all_occ) - 1):
            if all_occ[i] > 0.5 and all_occ[i+1] < 0.5:
                return (all_eig[i] + all_eig[i+1]) / 2
        
        return all_eig[len(all_eig)//2]
    
    def _find_fermi_energy_at_t(self, eigenvalues: np.ndarray,
                               temperature: float) -> float:
        """Find Fermi energy at finite temperature"""
        all_eig = np.sort(eigenvalues.flatten())
        
        # Find energy with occupation = 0.5
        def obj(E_F):
            occ = self._fermi_dirac(all_eig, E_F, temperature)
            return (np.sum(occ) - len(all_eig)/2)**2
        
        result = minimize_scalar(obj, bounds=(all_eig[0], all_eig[-1]))
        return result.x
    
    def _fermi_dirac(self, E: np.ndarray, E_F: float, T: float) -> np.ndarray:
        """Fermi-Dirac distribution"""
        if T == 0:
            return (E <= E_F).astype(float)
        kB = 8.617333e-5
        return 1.0 / (np.exp((E - E_F) / (kB * T)) + 1.0)
    
    def track_gap_closure(self, gaps: np.ndarray, 
                         control_parameter: np.ndarray) -> Dict[str, Any]:
        """
        Track gap closure as function of control parameter
        
        Parameters:
        -----------
        gaps : np.ndarray
            Gap values
        control_parameter : np.ndarray
            Control parameter values (e.g., U/t, pressure, doping)
            
        Returns:
        --------
        results : dict
            Gap closure information
        """
        # Find where gap crosses threshold
        threshold = self.config.gap_threshold
        
        crossings = []
        for i in range(len(gaps) - 1):
            if (gaps[i] - threshold) * (gaps[i+1] - threshold) < 0:
                # Linear interpolation for crossing point
                alpha = (threshold - gaps[i]) / (gaps[i+1] - gaps[i])
                crossing = control_parameter[i] + alpha * (control_parameter[i+1] - control_parameter[i])
                crossings.append(crossing)
        
        # Fit gap near closure
        if len(crossings) > 0:
            # Power law fit: gap ∝ |x - x_c|^ν
            x_c = crossings[0]
            
            # Find data near critical point
            mask = np.abs(control_parameter - x_c) < 2.0
            x_fit = control_parameter[mask]
            gap_fit = gaps[mask]
            
            # Fit gap = A |x - x_c|^ν
            log_gap = np.log(np.abs(gap_fit) + 1e-10)
            log_x = np.log(np.abs(x_fit - x_c) + 1e-10)
            
            # Linear fit
            coeffs = np.polyfit(log_x, log_gap, 1)
            nu = coeffs[0]
        else:
            x_c = None
            nu = None
        
        results = {
            'gap_closures': crossings,
            'critical_point': x_c,
            'critical_exponent': nu,
            'min_gap': np.min(gaps),
            'max_gap': np.max(gaps)
        }
        
        return results
    
    def analyze_gap_renormalization(self, gap_dft: float, gap_dmft: float,
                                   U: float, W: float) -> Dict[str, float]:
        """
        Analyze gap renormalization from DFT to DMFT
        
        For Mott insulators: gap_dmft >> gap_dft
        """
        renormalization = gap_dmft / gap_dft if gap_dft > 0 else np.inf
        
        # Estimate correlation strength U/W
        correlation_ratio = U / W
        
        results = {
            'gap_dft': gap_dft,
            'gap_dmft': gap_dmft,
            'renormalization': renormalization,
            'correlation_ratio': correlation_ratio,
            'is_strongly_correlated': renormalization > 2.0
        }
        
        return results


class MetalInsulatorTransition:
    """
    Metal-Insulator Transition (MIT) analyzer
    
    Detects and characterizes MIT using various criteria:
    - Gap opening/closing
    - Resistivity divergence
    - Correlation length divergence
    - Spectral weight transfer
    """
    
    def __init__(self, config: MottAnalysisConfig = None):
        self.config = config or MottAnalysisConfig()
        self.transition_history = []
    
    def detect_mit_gap_criterion(self, gaps: np.ndarray,
                                  parameter_values: np.ndarray) -> Dict[str, Any]:
        """
        Detect MIT using gap criterion
        
        MIT occurs when gap opens/closes
        """
        threshold = self.config.gap_threshold
        
        # Find transition points
        transition_points = []
        for i in range(len(gaps) - 1):
            if (gaps[i] - threshold) * (gaps[i+1] - threshold) < 0:
                # Interpolate
                alpha = (threshold - gaps[i]) / (gaps[i+1] - gaps[i])
                p_trans = parameter_values[i] + alpha * (parameter_values[i+1] - parameter_values[i])
                transition_points.append(p_trans)
        
        # Determine phases
        phases = []
        for i, gap in enumerate(gaps):
            phases.append('insulator' if gap > threshold else 'metal')
        
        results = {
            'transition_points': transition_points,
            'phases': phases,
            'parameter_values': parameter_values,
            'gaps': gaps,
            'criterion': 'gap'
        }
        
        return results
    
    def detect_mit_resistivity_criterion(self, resistivity: np.ndarray,
                                         parameter_values: np.ndarray) -> Dict[str, Any]:
        """
        Detect MIT using resistivity criterion
        
        MIT occurs when resistivity diverges or shows activation behavior
        """
        # Smooth resistivity
        from scipy.ndimage import uniform_filter1d
        rho_smooth = uniform_filter1d(resistivity, size=self.config.mit_smoothing)
        
        # Find maximum slope (indicative of transition)
        dlog_rho = np.gradient(np.log(rho_smooth + 1e-10))
        d_param = np.gradient(parameter_values)
        slope = dlog_rho / (d_param + 1e-10)
        
        # Transition at maximum slope
        idx_trans = np.argmax(np.abs(slope))
        p_trans = parameter_values[idx_trans]
        
        # Check for activation behavior in insulating phase
        # ρ = ρ_0 exp(Δ/T)
        
        results = {
            'transition_point': p_trans,
            'max_slope': slope[idx_trans],
            'susceptibility': slope,
            'criterion': 'resistivity'
        }
        
        return results
    
    def detect_mit_correlation_length(self, correlation_lengths: np.ndarray,
                                     parameter_values: np.ndarray) -> Dict[str, Any]:
        """
        Detect MIT using correlation length divergence
        
        In Mott transition, ξ diverges at critical point
        """
        # Fit ξ ∝ |x - x_c|^-ν
        # Use maximum as estimate of transition
        
        xi = correlation_lengths
        idx_max = np.argmax(xi)
        p_trans = parameter_values[idx_max]
        
        # Fit critical behavior
        mask = xi > 0.5 * np.max(xi)
        if np.sum(mask) > 2:
            log_xi = np.log(xi[mask])
            log_dx = np.log(np.abs(parameter_values[mask] - p_trans) + 1e-10)
            
            coeffs = np.polyfit(log_dx, log_xi, 1)
            nu = -coeffs[0]
        else:
            nu = None
        
        results = {
            'transition_point': p_trans,
            'correlation_length_max': xi[idx_max],
            'critical_exponent': nu,
            'criterion': 'correlation_length'
        }
        
        return results
    
    def analyze_spectral_weight_transfer(self, 
                                         spectral_functions: List[np.ndarray],
                                         omega: np.ndarray,
                                         parameter_values: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spectral weight transfer across MIT
        
        In Mott transition:
        - Low-frequency weight suppressed
        - Weight transferred to Hubbard bands
        """
        n_param = len(parameter_values)
        
        # Calculate spectral weight in different regions
        weight_low = np.zeros(n_param)  # |ω| < 1 eV
        weight_mid = np.zeros(n_param)  # 1 < |ω| < U/2
        weight_high = np.zeros(n_param)  # |ω| > U/2
        
        for i, A_w in enumerate(spectral_functions):
            dw = omega[1] - omega[0]
            
            mask_low = np.abs(omega) < 1.0
            mask_mid = (np.abs(omega) >= 1.0) & (np.abs(omega) < 3.0)
            mask_high = np.abs(omega) >= 3.0
            
            weight_low[i] = np.sum(A_w[mask_low]) * dw
            weight_mid[i] = np.sum(A_w[mask_mid]) * dw
            weight_high[i] = np.sum(A_w[mask_high]) * dw
        
        results = {
            'weight_low_freq': weight_low,
            'weight_mid_freq': weight_mid,
            'weight_high_freq': weight_high,
            'total_weight': weight_low + weight_mid + weight_high,
            'parameter_values': parameter_values
        }
        
        return results
    
    def construct_phase_diagram(self, U_values: np.ndarray,
                               T_values: np.ndarray,
                               gap_data: np.ndarray) -> Dict[str, Any]:
        """
        Construct U-T phase diagram for Mott transition
        
        Parameters:
        -----------
        U_values : np.ndarray
            Hubbard U values
        T_values : np.ndarray
            Temperature values
        gap_data : np.ndarray
            2D array of gap values (n_U, n_T)
            
        Returns:
        --------
        phase_diagram : dict
            Phase diagram information
        """
        n_U, n_T = gap_data.shape
        
        # Identify phases
        phase_mask = gap_data > self.config.gap_threshold
        
        # Find phase boundary
        boundary_U = []
        boundary_T = []
        
        for i_T in range(n_T):
            for i_U in range(n_U - 1):
                if phase_mask[i_U, i_T] != phase_mask[i_U+1, i_T]:
                    # Linear interpolation
                    alpha = (self.config.gap_threshold - gap_data[i_U, i_T]) / \
                           (gap_data[i_U+1, i_T] - gap_data[i_U, i_T])
                    U_bound = U_values[i_U] + alpha * (U_values[i_U+1] - U_values[i_U])
                    boundary_U.append(U_bound)
                    boundary_T.append(T_values[i_T])
        
        results = {
            'U_values': U_values,
            'T_values': T_values,
            'gap_data': gap_data,
            'phase_mask': phase_mask,
            'boundary_U': np.array(boundary_U),
            'boundary_T': np.array(boundary_T),
            'critical_endpoint': self._find_critical_endpoint(boundary_U, boundary_T)
        }
        
        return results
    
    def _find_critical_endpoint(self, boundary_U: np.ndarray,
                               boundary_T: np.ndarray) -> Tuple[float, float]:
        """Find critical endpoint of first-order transition line"""
        if len(boundary_U) == 0:
            return None, None
        
        # Maximum T on phase boundary
        idx_max = np.argmax(boundary_T)
        return boundary_U[idx_max], boundary_T[idx_max]


class OrderParameterAnalyzer:
    """
    Analyze charge and spin order in Mott insulators
    
    Detects:
    - Charge density waves (CDW)
    - Spin density waves (SDW)
    - Antiferromagnetic order
    - Stripe phases
    """
    
    def __init__(self, config: MottAnalysisConfig = None):
        self.config = config or MottAnalysisConfig()
    
    def analyze_charge_order(self, charge_density: np.ndarray,
                            lattice_vectors: np.ndarray,
                            q_points: np.ndarray) -> Dict[str, Any]:
        """
        Analyze charge order from density distribution
        
        Parameters:
        -----------
        charge_density : np.ndarray
            Charge density on lattice sites
        lattice_vectors : np.ndarray
            Real-space lattice vectors
        q_points : np.ndarray
            q-points for Fourier analysis
            
        Returns:
        --------
        order_info : dict
            Charge order parameters
        """
        # Fourier transform of charge density
        rho_q = self._fourier_transform(charge_density, lattice_vectors, q_points)
        rho_q_abs = np.abs(rho_q)
        
        # Find dominant modulation
        idx_max = np.argmax(rho_q_abs)
        q_cdw = q_points[idx_max]
        amplitude_cdw = rho_q_abs[idx_max]
        
        # Calculate structure factor
        structure_factor = np.abs(rho_q)**2
        
        results = {
            'q_cdw': q_cdw,
            'amplitude_cdw': amplitude_cdw,
            'rho_q': rho_q,
            'structure_factor': structure_factor,
            'has_cdw': amplitude_cdw > self.config.order_tol
        }
        
        return results
    
    def analyze_spin_order(self, spin_density: np.ndarray,
                          lattice_vectors: np.ndarray,
                          q_points: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spin order from spin density distribution
        
        Parameters:
        -----------
        spin_density : np.ndarray
            Spin density (S_z) on lattice sites
        lattice_vectors : np.ndarray
            Real-space lattice vectors
        q_points : np.ndarray
            q-points for Fourier analysis
            
        Returns:
        --------
        order_info : dict
            Spin order parameters
        """
        # Fourier transform
        S_q = self._fourier_transform(spin_density, lattice_vectors, q_points)
        S_q_abs = np.abs(S_q)
        
        # Find dominant modulation
        idx_max = np.argmax(S_q_abs)
        q_sdw = q_points[idx_max]
        amplitude_sdw = S_q_abs[idx_max]
        
        # Determine order type
        order_type = self._classify_spin_order(q_sdw)
        
        # Staggered magnetization
        m_staggered = np.abs(np.mean(spin_density * np.exp(1j * np.dot(
            self._get_lattice_positions(lattice_vectors), q_sdw))))
        
        results = {
            'q_sdw': q_sdw,
            'amplitude_sdw': amplitude_sdw,
            'S_q': S_q,
            'order_type': order_type,
            'staggered_magnetization': m_staggered,
            'has_sdw': amplitude_sdw > self.config.order_tol
        }
        
        return results
    
    def _fourier_transform(self, real_space_data: np.ndarray,
                          lattice_vectors: np.ndarray,
                          q_points: np.ndarray) -> np.ndarray:
        """Perform lattice Fourier transform"""
        positions = self._get_lattice_positions(lattice_vectors)
        
        ft_result = np.zeros(len(q_points), dtype=complex)
        
        for iq, q in enumerate(q_points):
            ft_result[iq] = np.sum(real_space_data * np.exp(1j * np.dot(positions, q)))
        
        return ft_result / len(real_space_data)
    
    def _get_lattice_positions(self, lattice_vectors: np.ndarray) -> np.ndarray:
        """Generate lattice positions"""
        # Simplified: assume cubic lattice
        n_sites = len(lattice_vectors)
        positions = np.zeros((n_sites, 3))
        
        for i in range(n_sites):
            positions[i] = lattice_vectors[i]
        
        return positions
    
    def _classify_spin_order(self, q_vector: np.ndarray) -> str:
        """Classify spin order from q-vector"""
        q_norm = np.linalg.norm(q_vector)
        
        # Check for common orderings
        if np.allclose(q_vector, [np.pi, np.pi, np.pi], atol=0.1):
            return "G-type_AFM"
        elif np.allclose(q_vector, [np.pi, np.pi, 0], atol=0.1):
            return "C-type_AFM"
        elif np.allclose(q_vector, [np.pi, 0, 0], atol=0.1):
            return "A-type_AFM"
        elif np.allclose(q_vector, [np.pi/2, np.pi/2, 0], atol=0.1):
            return "stripe"
        elif np.allclose(q_vector, [0, 0, 0], atol=0.1):
            return "ferromagnetic"
        else:
            return "incommensurate"
    
    def calculate_structure_factor(self, spin_correlations: np.ndarray,
                                  q_points: np.ndarray) -> np.ndarray:
        """
        Calculate magnetic structure factor
        
        S(q) = (1/N) Σ_{i,j} e^{-iq·(r_i-r_j)} <S_i·S_j>
        """
        n_sites = spin_correlations.shape[0]
        S_q = np.zeros(len(q_points))
        
        for iq, q in enumerate(q_points):
            for i in range(n_sites):
                for j in range(n_sites):
                    phase = np.exp(-1j * np.dot(q, self._r_ij(i, j)))
                    S_q[iq] += np.real(phase * spin_correlations[i, j])
        
        return S_q / n_sites
    
    def _r_ij(self, i: int, j: int) -> np.ndarray:
        """Calculate displacement between sites i and j"""
        # Simplified
        return np.array([i - j, 0, 0])
    
    def analyze_order_parameter_temperature(self, 
                                            order_parameters: np.ndarray,
                                            temperatures: np.ndarray) -> Dict[str, Any]:
        """
        Analyze order parameter vs temperature
        
        Determine transition temperature T_N or T_c
        """
        # Find where order parameter vanishes
        threshold = self.config.order_tol
        
        transition_temp = None
        for i in range(len(order_parameters) - 1):
            if (order_parameters[i] - threshold) * (order_parameters[i+1] - threshold) < 0:
                # Linear interpolation
                alpha = (threshold - order_parameters[i]) / (order_parameters[i+1] - order_parameters[i])
                transition_temp = temperatures[i] + alpha * (temperatures[i+1] - temperatures[i])
                break
        
        # Fit critical behavior near T_N
        if transition_temp is not None:
            mask = temperatures < transition_temp
            T_reduced = 1 - temperatures[mask] / transition_temp
            m = order_parameters[mask]
            
            # Mean-field: m ∝ (T_N - T)^(1/2)
            # 2D Ising: m ∝ (T_N - T)^(1/8)
            
            log_m = np.log(m + 1e-10)
            log_T = np.log(T_reduced + 1e-10)
            
            if len(log_T) > 1:
                beta_exp = np.polyfit(log_T, log_m, 1)[0]
            else:
                beta_exp = None
        else:
            beta_exp = None
        
        results = {
            'transition_temperature': transition_temp,
            'critical_exponent_beta': beta_exp,
            'order_parameters': order_parameters,
            'temperatures': temperatures
        }
        
        return results


# Utility functions for Mott analysis

def estimate_mott_gap(U: float, t: float, n: float = 1.0) -> float:
    """
    Estimate Mott gap from Hubbard model parameters
    
    For half-filled Hubbard model at strong coupling:
    Δ ≈ U - 12t + O(t²/U)
    """
    if n == 1.0:  # Half-filled
        return max(U - 12*t, 0)
    else:
        return max(U - 12*t * np.abs(2*n - 1), 0)


def calculate_u_critical_2d(t: float) -> float:
    """
    Estimate critical U for 2D Mott transition
    
    From QMC: U_c/t ≈ 6-7
    """
    return 6.5 * t


def calculate_u_critical_3d(t: float) -> float:
    """
    Estimate critical U for 3D Mott transition
    
    From DMFT: U_c/t ≈ 9.3
    """
    return 9.3 * t


def analyze_quasiparticle_residue(spectral_function: np.ndarray,
                                  omega: np.ndarray,
                                  omega_range: Tuple[float, float] = (-1, 1)) -> float:
    """
    Extract quasiparticle residue Z from spectral function
    
    Z = (1 - dΣ/dω)^-1 ≈ spectral weight near ω=0
    """
    mask = (omega >= omega_range[0]) & (omega <= omega_range[1])
    weight = np.trapz(spectral_function[mask], omega[mask])
    
    return weight


def estimate_brinkman_rice_z(U: float, U_c: float) -> float:
    """
    Estimate quasiparticle weight from Brinkman-Rice theory
    
    Z = (1 - U/U_c) for U < U_c
    """
    if U < U_c:
        return 1 - U/U_c
    else:
        return 0.0


def check_luttinger_theorem(occupation: np.ndarray, 
                           k_fermi_surface: np.ndarray) -> bool:
    """
    Check Luttinger theorem (volume enclosed by Fermi surface)
    
    Should hold for Fermi liquids, violated in Mott insulators
    """
    # Simplified check
    total_occupation = np.sum(occupation)
    luttinger_volume = np.sum(k_fermi_surface) / len(k_fermi_surface)
    
    # Check if occupation matches Luttinger volume
    return np.abs(total_occupation - luttinger_volume) < 0.01


__all__ = [
    'MottAnalysisConfig',
    'GapAnalyzer',
    'MetalInsulatorTransition',
    'OrderParameterAnalyzer',
    'estimate_mott_gap',
    'calculate_u_critical_2d',
    'calculate_u_critical_3d',
    'analyze_quasiparticle_residue',
    'estimate_brinkman_rice_z',
    'check_luttinger_theorem'
]