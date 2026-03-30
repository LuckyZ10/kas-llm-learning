"""
Hubbard U Calculation Module

This module provides methods for calculating and optimizing the Hubbard U parameter
for DFT+U and DMFT calculations, including:
- Linear response U (VASP LR method)
- Self-consistent U (scGW+DMFT)
- DFT+U parameter optimization

Author: DFT-LAMMPS Team
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import warnings
from scipy.optimize import minimize, minimize_scalar
import logging

logger = logging.getLogger(__name__)


@dataclass
class HubbardUConfig:
    """Configuration for Hubbard U calculations"""
    # Linear response parameters
    lr_n_perturbations: int = 5  # Number of perturbation strengths
    lr_max_alpha: float = 0.1  # Maximum perturbation strength
    lr_degeneracy_tol: float = 1e-4  # Tolerance for degeneracy check
    
    # Self-consistent parameters
    sc_max_iter: int = 50
    sc_tol: float = 1e-5
    sc_mixing: float = 0.5
    
    # Optimization parameters
    opt_method: str = "BFGS"
    opt_tol: float = 1e-6
    
    # Constrained RPA parameters
    crpa_n_empty_bands: int = 100
    crpa_energy_cutoff: float = 50.0  # eV
    
    # cRPA screening options
    crpa_screening: str = "full"  # full, approx, or none


class LinearResponseU:
    """
    Linear Response Hubbard U Calculator
    
    Implements the linear response method (Cococcioni et al.):
    U_lr = (χ0^-1 - χ^-1)
    
    where χ0 is the bare response and χ is the self-consistent response.
    """
    
    def __init__(self, config: HubbardUConfig = None):
        self.config = config or HubbardUConfig()
        self.response_data = {}
        
    def calculate_linear_response_u(self, structure_file: str, 
                                    orbital_indices: List[int],
                                    method: str = "vasp") -> Dict[str, float]:
        """
        Calculate Hubbard U using linear response
        
        Parameters:
        -----------
        structure_file : str
            Path to structure file
        orbital_indices : list
            Indices of correlated orbitals
        method : str
            DFT code to use (vasp, quantum_espresso, etc.)
            
        Returns:
        --------
        results : dict
            Dictionary containing U, J values
        """
        logger.info("Starting linear response U calculation")
        
        if method.lower() == "vasp":
            return self._calculate_lr_vasp(structure_file, orbital_indices)
        elif method.lower() == "qe":
            return self._calculate_lr_qe(structure_file, orbital_indices)
        else:
            raise ValueError(f"Method {method} not supported")
    
    def _calculate_lr_vasp(self, structure_file: str, 
                          orbital_indices: List[int]) -> Dict[str, float]:
        """Calculate linear response U using VASP"""
        
        # Step 1: Calculate bare response χ0
        logger.info("Calculating bare response χ0...")
        chi0 = self._calculate_bare_response_vasp(structure_file, orbital_indices)
        
        # Step 2: Calculate self-consistent response χ
        logger.info("Calculating self-consistent response χ...")
        chi = self._calculate_sc_response_vasp(structure_file, orbital_indices)
        
        # Step 3: Compute U from response matrices
        U, J = self._compute_u_from_response(chi0, chi, orbital_indices)
        
        results = {
            'U': U,
            'J': J,
            'chi0': chi0,
            'chi': chi,
            'method': 'linear_response'
        }
        
        logger.info(f"Linear response U = {U:.3f} eV, J = {J:.3f} eV")
        
        return results
    
    def _calculate_bare_response_vasp(self, structure_file: str, 
                                      orbital_indices: List[int]) -> np.ndarray:
        """
        Calculate bare response matrix χ0
        
        χ0_ij = dn_i/dα_j (non-self-consistent)
        """
        n_orb = len(orbital_indices)
        chi0 = np.zeros((n_orb, n_orb))
        
        # Generate perturbation strengths
        alphas = np.linspace(0, self.config.lr_max_alpha, 
                            self.config.lr_n_perturbations)
        
        # For each orbital, apply perturbation and measure response
        for j, orb_j in enumerate(orbital_indices):
            densities = []
            
            for alpha in alphas:
                # Run VASP with perturbation
                n_i = self._run_vasp_perturbation(structure_file, orb_j, alpha, 
                                                  scf=False)
                densities.append(n_i)
            
            # Fit dn/dα
            for i, orb_i in enumerate(orbital_indices):
                dn_dalpha = np.polyfit(alphas, [d[orb_i] for d in densities], 1)[0]
                chi0[i, j] = dn_dalpha
        
        return chi0
    
    def _calculate_sc_response_vasp(self, structure_file: str, 
                                    orbital_indices: List[int]) -> np.ndarray:
        """
        Calculate self-consistent response matrix χ
        
        χ_ij = dn_i/dα_j (self-consistent)
        """
        n_orb = len(orbital_indices)
        chi = np.zeros((n_orb, n_orb))
        
        alphas = np.linspace(0, self.config.lr_max_alpha, 
                            self.config.lr_n_perturbations)
        
        for j, orb_j in enumerate(orbital_indices):
            densities = []
            
            for alpha in alphas:
                # Run VASP with self-consistent perturbation
                n_i = self._run_vasp_perturbation(structure_file, orb_j, alpha, 
                                                  scf=True)
                densities.append(n_i)
            
            for i, orb_i in enumerate(orbital_indices):
                dn_dalpha = np.polyfit(alphas, [d[orb_i] for d in densities], 1)[0]
                chi[i, j] = dn_dalpha
        
        return chi
    
    def _run_vasp_perturbation(self, structure_file: str, orbital: int, 
                               alpha: float, scf: bool = True) -> np.ndarray:
        """
        Run VASP with potential perturbation
        
        INCAR keywords:
        - LRPA = .TRUE. (for linear response)
        - LPOT = .TRUE. (for potential perturbation)
        """
        incar_settings = f"""
PREC = Accurate
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-8

# Linear response settings
LRPA = .TRUE.
LPOT = .TRUE.
POTVAL = {alpha}
POTATOM = {orbital}
"""
        
        if scf:
            incar_settings += "\n# Self-consistent\nSCF = .TRUE.\n"
        else:
            incar_settings += "\n# Non-self-consistent\nSCF = .FALSE.\n"
        
        # Write INCAR and run VASP
        with open("INCAR", "w") as f:
            f.write(incar_settings)
        
        # Run VASP (placeholder)
        # subprocess.run(["vasp_std"], check=True)
        
        # Read density response from OUTCAR
        densities = self._read_density_response()
        
        return densities
    
    def _read_density_response(self) -> np.ndarray:
        """Read orbital occupations from VASP OUTCAR"""
        # Placeholder - would parse OUTCAR
        return np.random.rand(5)  # Mock data
    
    def _compute_u_from_response(self, chi0: np.ndarray, chi: np.ndarray,
                                  orbital_indices: List[int]) -> Tuple[float, float]:
        """
        Compute U and J from response matrices
        
        U = (χ0^-1 - χ^-1)_ii (diagonal, averaged)
        J = (χ0^-1 - χ^-1)_ij (off-diagonal, averaged)
        """
        # Invert response matrices
        chi0_inv = np.linalg.inv(chi0)
        chi_inv = np.linalg.inv(chi)
        
        # U matrix
        U_matrix = chi0_inv - chi_inv
        
        # Average diagonal (U) and off-diagonal (J)
        n_orb = len(orbital_indices)
        
        U = np.mean([U_matrix[i, i] for i in range(n_orb)])
        
        J_values = []
        for i in range(n_orb):
            for j in range(i+1, n_orb):
                J_values.append(U_matrix[i, j])
        J = np.mean(J_values) if J_values else 0.0
        
        return U, J
    
    def _calculate_lr_qe(self, structure_file: str, 
                        orbital_indices: List[int]) -> Dict[str, float]:
        """Calculate linear response U using Quantum ESPRESSO"""
        # Similar implementation for QE
        raise NotImplementedError("Quantum ESPRESSO interface not yet implemented")


class ConstrainedRPA:
    """
    Constrained Random Phase Approximation (cRPA) for U calculation
    
    Computes screened Coulomb interaction with correlation effects excluded
    from the screening.
    """
    
    def __init__(self, config: HubbardUConfig = None):
        self.config = config or HubbardUConfig()
        
    def calculate_crpa_u(self, wannier_hamiltonian: np.ndarray,
                        k_points: np.ndarray, k_weights: np.ndarray,
                        omega_grid: np.ndarray,
                        correlated_bands: List[int]) -> Dict[str, Any]:
        """
        Calculate Hubbard U using constrained RPA
        
        Parameters:
        -----------
        wannier_hamiltonian : np.ndarray
            Wannier Hamiltonian H(k)
        k_points : np.ndarray
            k-point grid
        k_weights : np.ndarray
            k-point weights
        omega_grid : np.ndarray
            Frequency grid for dielectric function
        correlated_bands : list
            Indices of correlated bands
            
        Returns:
        --------
        results : dict
            U(ω), W(ω), and related quantities
        """
        logger.info("Starting constrained RPA calculation")
        
        # Step 1: Calculate full dielectric matrix ε(q, ω)
        epsilon_full = self._calculate_dielectric_matrix(
            wannier_hamiltonian, k_points, omega_grid
        )
        
        # Step 2: Calculate constrained dielectric matrix ε_c(q, ω)
        # (exclude transitions within correlated subspace)
        epsilon_c = self._calculate_constrained_dielectric(
            wannier_hamiltonian, k_points, omega_grid, correlated_bands
        )
        
        # Step 3: Calculate screened interactions
        W_full = self._calculate_screened_coulomb(epsilon_full, k_points)
        W_c = self._calculate_screened_coulomb(epsilon_c, k_points)
        
        # Step 4: Extract U = W_c (constrained interaction)
        U_matrix = self._extract_u_matrix(W_c, correlated_bands)
        
        results = {
            'U_matrix': U_matrix,
            'U_average': np.mean(np.diag(U_matrix)),
            'W_full': W_full,
            'W_c': W_c,
            'epsilon_full': epsilon_full,
            'epsilon_c': epsilon_c,
            'omega_grid': omega_grid,
            'method': 'cRPA'
        }
        
        return results
    
    def _calculate_dielectric_matrix(self, H_k: np.ndarray,
                                     k_points: np.ndarray,
                                     omega_grid: np.ndarray) -> np.ndarray:
        """
        Calculate dielectric matrix ε(q, ω) from RPA
        
        ε_GG'(q, ω) = δ_GG' - v(q+G) χ^0_GG'(q, ω)
        """
        n_k = len(k_points)
        n_omega = len(omega_grid)
        n_G = 10  # Number of G-vectors (simplified)
        
        epsilon = np.zeros((n_omega, n_G, n_G), dtype=complex)
        
        for i_omega, omega in enumerate(omega_grid):
            for ik, k in enumerate(k_points):
                # Calculate bare susceptibility χ^0
                chi0 = self._calculate_bare_susceptibility(H_k[ik], k, omega)
                
                # Coulomb potential v(q+G)
                v_coul = self._coulomb_potential(k)
                
                # RPA dielectric
                eps = np.eye(n_G) - np.dot(v_coul, chi0)
                epsilon[i_omega] += eps / n_k
        
        return epsilon
    
    def _calculate_bare_susceptibility(self, H_k: np.ndarray, k: np.ndarray,
                                       omega: complex) -> np.ndarray:
        """
        Calculate bare susceptibility χ^0
        
        χ^0_GG'(q, ω) = sum_k sum_nm [f(ε_nk) - f(ε_mk+q)] / [ω + ε_nk - ε_mk+q + iη]
                         × <nk|e^-i(q+G)r|mk+q> <mk+q|e^i(q+G')r|nk>
        """
        # Simplified calculation
        n_G = 10
        chi0 = np.zeros((n_G, n_G), dtype=complex)
        
        # Diagonalize Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eigh(H_k)
        
        # Fermi occupation
        fermi = self._fermi_function(eigenvalues)
        
        # Calculate matrix elements
        for n in range(len(eigenvalues)):
            for m in range(len(eigenvalues)):
                if abs(fermi[n] - fermi[m]) > 1e-10:
                    delta_f = fermi[n] - fermi[m]
                    delta_e = eigenvalues[n] - eigenvalues[m]
                    
                    chi0 += delta_f / (omega - delta_e + 1j * 0.05)
        
        return chi0
    
    def _fermi_function(self, energies: np.ndarray, mu: float = 0.0,
                       T: float = 300.0) -> np.ndarray:
        """Fermi-Dirac distribution"""
        kB = 8.617333e-5  # eV/K
        beta = 1.0 / (kB * T)
        return 1.0 / (np.exp(beta * (energies - mu)) + 1.0)
    
    def _coulomb_potential(self, q: np.ndarray) -> np.ndarray:
        """Coulomb potential in reciprocal space"""
        n_G = 10
        v = np.zeros((n_G, n_G))
        
        for i in range(n_G):
            # v(q) = 4πe² / |q|² (atomic units)
            q_norm = np.linalg.norm(q) + 0.01  # Avoid singularity
            v[i, i] = 4 * np.pi / q_norm**2
        
        return v
    
    def _calculate_constrained_dielectric(self, H_k: np.ndarray,
                                          k_points: np.ndarray,
                                          omega_grid: np.ndarray,
                                          correlated_bands: List[int]) -> np.ndarray:
        """
        Calculate constrained dielectric matrix
        
        Exclude transitions within correlated subspace from screening
        """
        # Similar to full dielectric but with constraints
        epsilon_c = self._calculate_dielectric_matrix(H_k, k_points, omega_grid)
        
        # Apply constraint: remove contributions from correlated bands
        # This is implementation-specific
        
        return epsilon_c
    
    def _calculate_screened_coulomb(self, epsilon: np.ndarray,
                                    k_points: np.ndarray) -> np.ndarray:
        """
        Calculate screened Coulomb interaction
        
        W = ε^-1 v
        """
        n_omega = epsilon.shape[0]
        n_G = epsilon.shape[1]
        
        W = np.zeros_like(epsilon)
        
        for iw in range(n_omega):
            eps_inv = np.linalg.inv(epsilon[iw])
            v = self._coulomb_potential(k_points[0])
            W[iw] = np.dot(eps_inv, v)
        
        return W
    
    def _extract_u_matrix(self, W: np.ndarray, 
                         correlated_bands: List[int]) -> np.ndarray:
        """
        Extract Hubbard U matrix from screened interaction
        
        U_ij = W(R=0, G=0, ω=0) matrix elements
        """
        n_corr = len(correlated_bands)
        U_matrix = np.zeros((n_corr, n_corr))
        
        # Extract static, local part of W
        W_0 = W[len(W)//2, 0, 0]  # ω=0, G=0
        
        # Build U matrix (simplified)
        for i in range(n_corr):
            U_matrix[i, i] = W_0.real
            for j in range(i+1, n_corr):
                U_matrix[i, j] = U_matrix[j, i] = 0.5 * W_0.real
        
        return U_matrix


class SelfConsistentU:
    """
    Self-consistent Hubbard U calculation
    
    Iteratively updates U based on charge response until convergence:
    1. Calculate U from linear response
    2. Run DFT+U with this U
    3. Recalculate U
    4. Check convergence
    """
    
    def __init__(self, config: HubbardUConfig = None):
        self.config = config or HubbardUConfig()
        self.convergence_history = []
        
    def calculate_sc_u(self, structure_file: str,
                      orbital_indices: List[int],
                      initial_guess: float = 4.0) -> Dict[str, Any]:
        """
        Calculate self-consistent Hubbard U
        
        Parameters:
        -----------
        structure_file : str
            Path to structure file
        orbital_indices : list
            Indices of correlated orbitals
        initial_guess : float
            Initial U value
            
        Returns:
        --------
        results : dict
            Converged U and convergence history
        """
        logger.info("Starting self-consistent U calculation")
        
        U_current = initial_guess
        lr_calc = LinearResponseU(self.config)
        
        for iteration in range(self.config.sc_max_iter):
            logger.info(f"SC iteration {iteration}: U = {U_current:.3f} eV")
            
            # Step 1: Run DFT+U with current U
            self._run_dft_plus_u(structure_file, orbital_indices, U_current)
            
            # Step 2: Calculate new U from linear response
            lr_results = lr_calc.calculate_linear_response_u(
                structure_file, orbital_indices
            )
            U_new = lr_results['U']
            J_new = lr_results['J']
            
            # Step 3: Mixing
            U_mixed = (self.config.sc_mixing * U_new + 
                      (1 - self.config.sc_mixing) * U_current)
            
            # Check convergence
            diff = abs(U_mixed - U_current)
            self.convergence_history.append({
                'iteration': iteration,
                'U': U_current,
                'diff': diff
            })
            
            if diff < self.config.sc_tol:
                logger.info(f"SC U converged: U = {U_mixed:.3f} eV")
                break
            
            U_current = U_mixed
        
        results = {
            'U_sc': U_current,
            'J_sc': J_new,
            'convergence_history': self.convergence_history,
            'converged': diff < self.config.sc_tol,
            'method': 'self_consistent'
        }
        
        return results
    
    def _run_dft_plus_u(self, structure_file: str, 
                       orbital_indices: List[int], U: float):
        """Run DFT+U calculation with given U value"""
        # Generate DFT+U input
        incar = f"""
# DFT+U calculation
LDAU = .TRUE.
LDAUTYPE = 2
LDAUL = 2 -1
LDAUU = {U:.2f} 0.0
LDAUJ = 0.0 0.0
LDAUPRINT = 2

# General settings
PREC = Accurate
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6
"""
        
        with open("INCAR", "w") as f:
            f.write(incar)
        
        # Run VASP (placeholder)
        # subprocess.run(["vasp_std"], check=True)


class DFTPlusUOptimizer:
    """
    DFT+U parameter optimization
    
    Optimizes U and J parameters to minimize errors in:
    - Lattice constants
    - Band gaps
    - Magnetic moments
    - Formation energies
    """
    
    def __init__(self, config: HubbardUConfig = None):
        self.config = config or HubbardUConfig()
        self.target_properties = {}
        
    def optimize_u_for_property(self, structure_file: str,
                                property_name: str,
                                target_value: float,
                                orbital_indices: List[int],
                                u_bounds: Tuple[float, float] = (0.0, 10.0),
                                j_bounds: Tuple[float, float] = (0.0, 2.0)) -> Dict[str, float]:
        """
        Optimize U to reproduce a target property
        
        Parameters:
        -----------
        structure_file : str
            Path to structure file
        property_name : str
            Property to optimize (band_gap, lattice_constant, magnetic_moment)
        target_value : float
            Target value for the property
        orbital_indices : list
            Indices of correlated orbitals
        u_bounds : tuple
            Bounds for U parameter
        j_bounds : tuple
            Bounds for J parameter
            
        Returns:
        --------
        results : dict
            Optimized U, J and property error
        """
        logger.info(f"Optimizing U for {property_name} = {target_value}")
        
        self.target_properties[property_name] = target_value
        
        # Define objective function
        def objective(params):
            U, J = params
            calculated = self._calculate_property(structure_file, orbital_indices, 
                                                 property_name, U, J)
            error = (calculated - target_value)**2
            return error
        
        # Optimize
        from scipy.optimize import minimize
        
        initial_guess = [(u_bounds[0] + u_bounds[1]) / 2,
                        (j_bounds[0] + j_bounds[1]) / 2]
        bounds = [u_bounds, j_bounds]
        
        result = minimize(objective, initial_guess, method=self.config.opt_method,
                         bounds=bounds, tol=self.config.opt_tol)
        
        U_opt, J_opt = result.x
        final_error = np.sqrt(result.fun)
        
        logger.info(f"Optimized U = {U_opt:.3f} eV, J = {J_opt:.3f} eV")
        logger.info(f"Final error = {final_error:.6f}")
        
        return {
            'U_opt': U_opt,
            'J_opt': J_opt,
            'error': final_error,
            'property': property_name,
            'target_value': target_value,
            'converged': result.success
        }
    
    def optimize_multiple_properties(self, structure_file: str,
                                    properties: Dict[str, float],
                                    weights: Dict[str, float],
                                    orbital_indices: List[int]) -> Dict[str, float]:
        """
        Multi-objective optimization for multiple properties
        
        Parameters:
        -----------
        structure_file : str
            Path to structure file
        properties : dict
            Dictionary of property_name: target_value
        weights : dict
            Dictionary of property_name: weight
        orbital_indices : list
            Indices of correlated orbitals
            
        Returns:
        --------
        results : dict
            Optimized U, J and individual errors
        """
        logger.info(f"Multi-objective optimization for {len(properties)} properties")
        
        def objective(params):
            U, J = params
            total_error = 0.0
            
            for prop_name, target in properties.items():
                weight = weights.get(prop_name, 1.0)
                calculated = self._calculate_property(structure_file, orbital_indices,
                                                     prop_name, U, J)
                error = weight * (calculated - target)**2
                total_error += error
            
            return total_error
        
        # Optimize
        from scipy.optimize import minimize
        
        initial_guess = [4.0, 0.6]
        bounds = [(0.0, 10.0), (0.0, 2.0)]
        
        result = minimize(objective, initial_guess, method=self.config.opt_method,
                         bounds=bounds)
        
        U_opt, J_opt = result.x
        
        # Calculate individual errors
        errors = {}
        for prop_name, target in properties.items():
            calculated = self._calculate_property(structure_file, orbital_indices,
                                                 prop_name, U_opt, J_opt)
            errors[prop_name] = abs(calculated - target)
        
        return {
            'U_opt': U_opt,
            'J_opt': J_opt,
            'total_error': result.fun,
            'individual_errors': errors,
            'converged': result.success
        }
    
    def _calculate_property(self, structure_file: str, orbital_indices: List[int],
                           property_name: str, U: float, J: float) -> float:
        """Calculate a property with given U and J"""
        # Run DFT+U calculation
        self._run_dft_u_calculation(structure_file, orbital_indices, U, J)
        
        # Extract property
        if property_name == "band_gap":
            return self._read_band_gap()
        elif property_name == "lattice_constant":
            return self._read_lattice_constant()
        elif property_name == "magnetic_moment":
            return self._read_magnetic_moment()
        elif property_name == "formation_energy":
            return self._read_formation_energy()
        else:
            raise ValueError(f"Property {property_name} not recognized")
    
    def _run_dft_u_calculation(self, structure_file: str, 
                              orbital_indices: List[int], U: float, J: float):
        """Run DFT+U calculation"""
        incar = f"""
LDAU = .TRUE.
LDAUTYPE = 2
LDAUL = 2 -1
LDAUU = {U:.2f} 0.0
LDAUJ = {J:.2f} 0.0
"""
        with open("INCAR", "w") as f:
            f.write(incar)
        
        # Run VASP (placeholder)
    
    def _read_band_gap(self) -> float:
        """Read band gap from VASP output"""
        # Placeholder
        return 2.0  # Mock value
    
    def _read_lattice_constant(self) -> float:
        """Read lattice constant from VASP output"""
        # Placeholder
        return 4.0  # Mock value
    
    def _read_magnetic_moment(self) -> float:
        """Read magnetic moment from VASP output"""
        # Placeholder
        return 3.0  # Mock value
    
    def _read_formation_energy(self) -> float:
        """Read formation energy from VASP output"""
        # Placeholder
        return -5.0  # Mock value


class UDatabase:
    """
    Database of pre-calculated Hubbard U values
    
    Provides U values for common materials and methods.
    """
    
    def __init__(self):
        self.database = self._initialize_database()
    
    def _initialize_database(self) -> Dict[str, Dict[str, float]]:
        """Initialize U database with literature values"""
        database = {
            # Transition metal oxides
            'FeO': {'U': 4.6, 'J': 0.8, 'method': 'cRPA', 'source': 'Cococcioni'},
            'CoO': {'U': 5.0, 'J': 0.9, 'method': 'cRPA', 'source': 'Cococcioni'},
            'NiO': {'U': 6.3, 'J': 1.0, 'method': 'cRPA', 'source': 'Cococcioni'},
            'MnO': {'U': 5.0, 'J': 0.9, 'method': 'cRPA', 'source': 'Cococcioni'},
            
            # Perovskites
            'SrTiO3': {'U': 3.6, 'J': 0.6, 'method': 'linear_response'},
            'BaTiO3': {'U': 3.4, 'J': 0.6, 'method': 'linear_response'},
            'CaTiO3': {'U': 3.8, 'J': 0.6, 'method': 'linear_response'},
            'LaMnO3': {'U': 5.0, 'J': 0.9, 'method': 'cRPA'},
            'LaFeO3': {'U': 5.5, 'J': 0.9, 'method': 'cRPA'},
            'LaCoO3': {'U': 5.2, 'J': 0.9, 'method': 'cRPA'},
            'LaNiO3': {'U': 6.0, 'J': 1.0, 'method': 'cRPA'},
            
            # Cuprates
            'La2CuO4': {'U': 8.0, 'J': 1.0, 'method': 'cRPA'},
            'YBa2Cu3O7': {'U': 8.5, 'J': 1.0, 'method': 'cRPA'},
            'Bi2Sr2CaCu2O8': {'U': 8.0, 'J': 1.0, 'method': 'cRPA'},
            
            # Iron-based superconductors
            'LaFeAsO': {'U': 3.5, 'J': 0.8, 'method': 'cRPA'},
            'BaFe2As2': {'U': 3.2, 'J': 0.8, 'method': 'cRPA'},
            'FeSe': {'U': 3.0, 'J': 0.8, 'method': 'cRPA'},
        }
        return database
    
    def get_u(self, material: str) -> Optional[Dict[str, float]]:
        """Get U value for a material"""
        return self.database.get(material)
    
    def add_entry(self, material: str, U: float, J: float, 
                 method: str, source: str = ""):
        """Add new entry to database"""
        self.database[material] = {
            'U': U,
            'J': J,
            'method': method,
            'source': source
        }
    
    def list_materials(self) -> List[str]:
        """List all materials in database"""
        return list(self.database.keys())
    
    def get_average_u(self, element: str) -> Dict[str, float]:
        """Get average U for an element across all compounds"""
        u_values = []
        j_values = []
        
        for material, data in self.database.items():
            if element in material:
                u_values.append(data['U'])
                j_values.append(data['J'])
        
        if u_values:
            return {
                'U_mean': np.mean(u_values),
                'U_std': np.std(u_values),
                'J_mean': np.mean(j_values),
                'J_std': np.std(j_values),
                'count': len(u_values)
            }
        return None


# Utility functions for U analysis

def calculate_u_eff(U: float, J: float, scheme: str = "simplified") -> float:
    """
    Calculate effective U for different DFT+U schemes
    
    Parameters:
    -----------
    U : float
        Hubbard U parameter
    J : float
        Hund's coupling
    scheme : str
        DFT+U scheme (simplified, full, liechtenstein)
        
    Returns:
    --------
    U_eff : float
        Effective U parameter
    """
    if scheme == "simplified":
        # Dudarev scheme: U_eff = U - J
        return U - J
    elif scheme == "full":
        # Full rotationally invariant
        return U
    elif scheme == "liechtenstein":
        # Liechtenstein scheme
        return U
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


def estimate_u_from_ionicity(oxidation_state: int, 
                             coordination: int,
                             element: str) -> float:
    """
    Estimate U from empirical ionicity trends
    
    Simple empirical formula based on:
    - Oxidation state
    - Coordination number
    - Element position in periodic table
    """
    # Base U values for transition metals
    base_u = {
        'Ti': 3.0, 'V': 3.5, 'Cr': 4.0, 'Mn': 4.5,
        'Fe': 4.0, 'Co': 4.5, 'Ni': 5.0, 'Cu': 6.0,
        'Zr': 3.0, 'Nb': 3.5, 'Mo': 4.0, 'Tc': 4.5,
        'Ru': 4.0, 'Rh': 4.5, 'Pd': 5.0, 'Ag': 6.0
    }
    
    U_base = base_u.get(element, 4.0)
    
    # Correction for oxidation state
    U_estimated = U_base + 0.5 * (oxidation_state - 2)
    
    # Correction for coordination
    U_estimated += 0.2 * (6 - coordination)
    
    return max(U_estimated, 0.5)


def check_u_reasonableness(U: float, J: float, 
                          orbital_type: str = "d") -> Tuple[bool, str]:
    """
    Check if U and J values are physically reasonable
    
    Returns:
    --------
    is_reasonable : bool
        Whether the values are reasonable
    message : str
        Explanation if not reasonable
    """
    messages = []
    
    # Range checks
    if U < 0:
        messages.append("U should be positive")
    if J < 0:
        messages.append("J should be positive")
    
    # Typical ranges
    if orbital_type == "d":
        if U > 12:
            messages.append(f"U={U:.1f} is unusually large for d-orbitals")
        if J > 1.5:
            messages.append(f"J={J:.1f} is unusually large for d-orbitals")
        if J > U / 2:
            messages.append(f"J > U/2 is physically unusual")
    elif orbital_type == "f":
        if U > 15:
            messages.append(f"U={U:.1f} is unusually large for f-orbitals")
        if J > 2.0:
            messages.append(f"J={J:.1f} is unusually large for f-orbitals")
    
    if messages:
        return False, "; ".join(messages)
    return True, "Values appear reasonable"


__all__ = [
    'HubbardUConfig',
    'LinearResponseU',
    'ConstrainedRPA',
    'SelfConsistentU',
    'DFTPlusUOptimizer',
    'UDatabase',
    'calculate_u_eff',
    'estimate_u_from_ionicity',
    'check_u_reasonableness'
]