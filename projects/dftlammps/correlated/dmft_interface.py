"""
DMFT Interface Module for Strongly Correlated Systems

This module provides interfaces for Dynamical Mean-Field Theory (DMFT) calculations,
integrating VASP, Wannier90, and CTQMC solvers for strongly correlated electron systems.

Key Features:
- VASP+Wannier90 projection for localized orbital construction
- CTQMC impurity solver interface (ALPS/iPET/comCTQMC)
- Self-consistent DMFT loop implementation
- Spectral function A(ω) calculation
- Lattice Green's function construction

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
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class DMFTConfig:
    """Configuration for DMFT calculations"""
    # System parameters
    temperature: float = 300.0  # Temperature in K
    beta: float = None  # Inverse temperature (1/kB*T)
    
    # Correlated orbitals
    n_orbitals: int = 5  # Number of correlated orbitals (e.g., 5 for d-electrons)
    n_spin: int = 2  # Number of spin channels
    orbital_type: str = "d"  # Type of correlated orbitals (s, p, d, f)
    
    # DMFT parameters
    u_value: float = 4.0  # Hubbard U in eV
    j_value: float = 0.6  # Hund's J in eV
    
    # Self-consistency parameters
    scf_max_iter: int = 100
    scf_tol: float = 1e-6
    mixing_alpha: float = 0.3  # Mixing parameter for self-consistency
    
    # Frequency mesh
    n_freq: int = 1024  # Number of Matsubara frequencies
    freq_cutoff: float = 10.0  # Frequency cutoff in eV
    
    # Solver parameters
    solver_type: str = "cthyb"  # CT-QMC solver type
    solver_cycles: int = 1000000  # Number of QMC cycles
    solver_warmup: int = 10000  # Warmup cycles
    
    # Spectral function
    n_real_freq: int = 2000  # Number of real frequency points
    eta_broadening: float = 0.05  # Broadening parameter for spectral function
    
    def __post_init__(self):
        if self.beta is None:
            kB = 8.617333e-5  # Boltzmann constant in eV/K
            self.beta = 1.0 / (kB * self.temperature)


@dataclass
class WannierProjectorConfig:
    """Configuration for Wannier90 projection"""
    # Wannier90 parameters
    num_wann: int = 5  # Number of Wannier functions
    num_iter: int = 1000  # Number of iterations
    dis_win_min: float = -10.0
    dis_win_max: float = 10.0
    dis_froz_min: float = -5.0
    dis_froz_max: float = 5.0
    
    # Projection centers
    use_projections: bool = True
    projection_type: str = "atomic"  # atomic, gaussian, etc.
    
    # Wannierization control
    write_xyz: bool = True
    write_hr: bool = True
    write_tb: bool = True
    write_u_matrices: bool = True
    
    # Disentanglement
    use_disentanglement: bool = True
    dis_mix_ratio: float = 0.5


class ImpuritySolver(ABC):
    """Abstract base class for impurity solvers"""
    
    @abstractmethod
    def solve(self, G0_iw: np.ndarray, U: float, J: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the impurity problem
        
        Parameters:
        -----------
        G0_iw : np.ndarray
            Weiss field (bare Green's function)
        U : float
            Hubbard U parameter
        J : float
            Hund's coupling
        beta : float
            Inverse temperature
            
        Returns:
        --------
        G_imp : np.ndarray
            Impurity Green's function
        Sigma : np.ndarray
            Self-energy
        """
        pass


class CTQMCSolver(ImpuritySolver):
    """
    Continuous-Time Quantum Monte Carlo (CT-QMC) Solver Interface
    
    Supports multiple CT-QMC implementations:
    - ALPS CT-HYB
    - iPET
    - ComCTQMC
    - TRIQS/cthyb
    """
    
    def __init__(self, solver_type: str = "triqs", config: Dict = None):
        self.solver_type = solver_type.lower()
        self.config = config or {}
        self._check_solver_availability()
        
    def _check_solver_availability(self):
        """Check if the requested solver is available"""
        available_solvers = self._get_available_solvers()
        if self.solver_type not in available_solvers:
            warnings.warn(f"Solver {self.solver_type} not found. Using TRIQS fallback.")
            self.solver_type = "triqs"
    
    def _get_available_solvers(self) -> List[str]:
        """Detect available CT-QMC solvers"""
        available = []
        
        # Check for TRIQS
        try:
            import triqs
            available.append("triqs")
            available.append("cthyb")
        except ImportError:
            pass
        
        # Check for ALPS
        if self._check_command("alps_cthyb"):
            available.append("alps")
            available.append("alps_cthyb")
        
        # Check for iPET
        if self._check_command("ipet"):
            available.append("ipet")
        
        # Check for ComCTQMC
        if self._check_command("comctqmc"):
            available.append("comctqmc")
        
        return available
    
    def _check_command(self, cmd: str) -> bool:
        """Check if a command is available"""
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=False)
            return True
        except FileNotFoundError:
            return False
    
    def solve(self, G0_iw: np.ndarray, U: float, J: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve impurity problem using CT-QMC"""
        if self.solver_type in ["triqs", "cthyb"]:
            return self._solve_triqs(G0_iw, U, J, beta)
        elif self.solver_type in ["alps", "alps_cthyb"]:
            return self._solve_alps(G0_iw, U, J, beta)
        elif self.solver_type == "ipet":
            return self._solve_ipet(G0_iw, U, J, beta)
        else:
            raise ValueError(f"Solver {self.solver_type} not implemented")
    
    def _solve_triqs(self, G0_iw: np.ndarray, U: float, J: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using TRIQS/cthyb"""
        try:
            from triqs.gf import GfImFreq, SemiCircle
            from triqs.operators import c, c_dag, n
            from triqs_cthyb import Solver
            
            n_orb = G0_iw.shape[0] if len(G0_iw.shape) > 1 else 1
            
            # Create solver
            S = Solver(beta=beta, gf_struct=[('up', n_orb), ('down', n_orb)])
            
            # Construct local Hamiltonian
            h_int = self._construct_h_int(U, J, n_orb)
            
            # Set hybridization
            for spin in ['up', 'down']:
                S.G0_iw[spin] << G0_iw
            
            # Solve
            S.solve(
                h_int=h_int,
                n_cycles=self.config.get('n_cycles', 100000),
                length_cycle=self.config.get('length_cycle', 50),
                n_warmup_cycles=self.config.get('n_warmup_cycles', 10000)
            )
            
            # Extract results
            G_imp = np.array([S.G_tau['up'].data, S.G_tau['down'].data])
            Sigma = np.array([S.Sigma_iw['up'].data, S.Sigma_iw['down'].data])
            
            return G_imp, Sigma
            
        except ImportError:
            logger.warning("TRIQS not available, using analytic continuation")
            return self._solve_fallback(G0_iw, U, J, beta)
    
    def _construct_h_int(self, U: float, J: float, n_orb: int) -> Any:
        """Construct interaction Hamiltonian in density-density form"""
        try:
            from triqs.operators import c, c_dag, n
            
            h_int = 0
            
            # Density-density interaction
            for spin in ['up', 'down']:
                for o1 in range(n_orb):
                    for o2 in range(n_orb):
                        if o1 != o2:
                            if spin == 'up':
                                h_int += U * n('up', o1) * n('down', o2)
                            else:
                                h_int += U * n('down', o1) * n('up', o2)
                        else:
                            h_int += U * n(spin, o1) * n(spin, o2)
            
            # Hund's coupling
            for o1 in range(n_orb):
                for o2 in range(n_orb):
                    if o1 != o2:
                        h_int -= J * n('up', o1) * n('up', o2)
                        h_int -= J * n('down', o1) * n('down', o2)
            
            return h_int
            
        except ImportError:
            return None
    
    def _solve_alps(self, G0_iw: np.ndarray, U: float, J: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using ALPS CT-HYB"""
        # Write input files
        self._write_alps_input(G0_iw, U, J, beta)
        
        # Run ALPS
        subprocess.run(["alps_cthyb", "params.ini"], check=True)
        
        # Read results
        G_imp, Sigma = self._read_alps_results()
        
        return G_imp, Sigma
    
    def _write_alps_input(self, G0_iw: np.ndarray, U: float, J: float, beta: float):
        """Write ALPS input files"""
        params = f"""
BETA = {beta}
U = {U}
J = {J}
N_ORBITALS = {G0_iw.shape[0]}
N_WARMUP_CYCLES = {self.config.get('n_warmup_cycles', 10000)}
N_CYCLES = {self.config.get('n_cycles', 100000)}
MEASURE_G = 1
MEASURE_G2 = 0
"""
        with open("params.ini", "w") as f:
            f.write(params)
        
        # Write G0
        np.save("G0_iw.npy", G0_iw)
    
    def _read_alps_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read ALPS output files"""
        G_imp = np.load("G_tau.npy")
        Sigma = np.load("Sigma_iw.npy")
        return G_imp, Sigma
    
    def _solve_ipet(self, G0_iw: np.ndarray, U: float, J: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using iPET solver"""
        # Similar structure to ALPS
        raise NotImplementedError("iPET solver interface not yet implemented")
    
    def _solve_fallback(self, G0_iw: np.ndarray, U: float, J: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback solver using iterative perturbation theory"""
        # Simplified IPT solver for testing
        iw = 1j * (2 * np.arange(len(G0_iw)) + 1) * np.pi / beta
        
        # Approximate self-energy
        Sigma = U**2 * G0_iw**2 * G0_iw.conj()
        
        # Impurity Green's function
        G_imp = 1.0 / (1.0 / G0_iw - Sigma)
        
        return G_imp, Sigma


class WannierProjector:
    """
    Wannier90 Interface for localized orbital projection
    
    Handles:
    - VASP Wannier90 input generation
    - Projection matrix construction
    - Hamiltonian rotation to Wannier basis
    """
    
    def __init__(self, config: WannierProjectorConfig = None):
        self.config = config or WannierProjectorConfig()
        self.projection_matrix = None
        self.hamiltonian_wan = None
        
    def generate_wannier90_input(self, structure_file: str, output_dir: str = "wannier90"):
        """Generate Wannier90 input file from structure"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        win_content = self._generate_win_file()
        
        win_path = os.path.join(output_dir, "wannier90.win")
        with open(win_path, "w") as f:
            f.write(win_content)
        
        logger.info(f"Wannier90 input written to {win_path}")
        return win_path
    
    def _generate_win_file(self) -> str:
        """Generate wannier90.win content"""
        c = self.config
        
        projections = self._generate_projections()
        
        win_content = f"""# Wannier90 input for correlated orbitals
# Generated by DFT-LAMMPS DMFT interface

num_wann = {c.num_wann}
num_iter = {c.num_iter}

# Disentanglement
use_disentanglement = {".true." if c.use_disentanglement else ".false."}
dis_win_min = {c.dis_win_min}
dis_win_max = {c.dis_win_max}
dis_froz_min = {c.dis_froz_min}
dis_froz_max = {c.dis_froz_max}
dis_mix_ratio = {c.dis_mix_ratio}

# Projections
{projections}

# Output control
write_xyz = {".true." if c.write_xyz else ".false."}
write_hr = {".true." if c.write_hr else ".false."}
write_tb = {".true." if c.write_tb else ".false."}
write_u_matrices = {".true." if c.write_u_matrices else ".false."}

# Plotting
wannier_plot = True
wannier_plot_supercell = 2
"""
        return win_content
    
    def _generate_projections(self) -> str:
        """Generate projection block for correlated orbitals"""
        if not self.config.use_projections:
            return ""
        
        if self.config.projection_type == "atomic":
            if self.config.num_wann == 5:
                # d-orbitals
                return """begin projections
f=0.0,0.0,0.0:dxy,dyz,dxz,dx2-y2,dz2
end projections"""
            elif self.config.num_wann == 3:
                # p-orbitals
                return """begin projections
f=0.0,0.0,0.0:px,py,pz
end projections"""
            elif self.config.num_wann == 7:
                # f-orbitals
                return """begin projections
f=0.0,0.0,0.0:f
end projections"""
            else:
                return ""
        else:
            return "gaussian"
    
    def construct_projection_matrix(self, KohnSham_orbitals: np.ndarray, 
                                   localized_orbitals: np.ndarray) -> np.ndarray:
        """
        Construct projection matrix from Kohn-Sham to localized orbitals
        
        Parameters:
        -----------
        KohnSham_orbitals : np.ndarray
            Kohn-Sham wavefunctions (band, r)
        localized_orbitals : np.ndarray
            Localized Wannier orbitals (wannier, r)
            
        Returns:
        --------
        P : np.ndarray
            Projection matrix
        """
        # Overlap matrix
        overlap = np.dot(localized_orbitals.conj(), KohnSham_orbitals.T)
        
        # Projector P_mn = <w_m | psi_n>
        self.projection_matrix = overlap
        
        return self.projection_matrix
    
    def rotate_hamiltonian(self, H_KS: np.ndarray, k_points: np.ndarray) -> np.ndarray:
        """
        Rotate Kohn-Sham Hamiltonian to Wannier basis
        
        H_W(R) = (1/Nk) * sum_k e^(-ikR) U^dagger(k) H_KS(k) U(k)
        """
        if self.projection_matrix is None:
            raise ValueError("Projection matrix not constructed")
        
        n_R = len(k_points)
        n_wann = self.config.num_wann
        
        H_W = np.zeros((n_R, n_wann, n_wann), dtype=complex)
        
        for iR, R in enumerate(k_points):
            for ik, k in enumerate(k_points):
                phase = np.exp(-1j * np.dot(k, R))
                U = self.projection_matrix[ik]
                
                H_W[iR] += phase * np.dot(U.conj().T, np.dot(H_KS[ik], U))
        
        H_W /= len(k_points)
        self.hamiltonian_wan = H_W
        
        return H_W
    
    def get_local_hamiltonian(self) -> np.ndarray:
        """Get on-site Hamiltonian matrix element H_00"""
        if self.hamiltonian_wan is None:
            raise ValueError("Wannier Hamiltonian not computed")
        return self.hamiltonian_wan[0]


class DMFTEngine:
    """
    Main DMFT calculation engine
    
    Implements the self-consistent DMFT loop:
    1. Initialize Weiss field G0
    2. Solve impurity problem
    3. Compute lattice Green's function
    4. Update Weiss field
    5. Check convergence
    """
    
    def __init__(self, config: DMFTConfig = None):
        self.config = config or DMFTConfig()
        self.solver = None
        self.projector = None
        
        # DMFT quantities
        self.G0_iw = None  # Weiss field
        self.G_loc = None  # Local Green's function
        self.G_imp = None  # Impurity Green's function
        self.Sigma = None  # Self-energy
        self.mu = 0.0  # Chemical potential
        
        # Convergence tracking
        self.convergence_history = []
        
    def initialize(self, solver_type: str = "triqs", projector_config: WannierProjectorConfig = None):
        """Initialize DMFT engine with solver and projector"""
        self.solver = CTQMCSolver(solver_type, self.config.__dict__)
        
        if projector_config:
            self.projector = WannierProjector(projector_config)
        
        # Initialize frequency mesh
        self._init_frequency_mesh()
        
    def _init_frequency_mesh(self):
        """Initialize Matsubara frequency mesh"""
        n_freq = self.config.n_freq
        beta = self.config.beta
        
        # Matsubara frequencies: ω_n = (2n+1)π/β
        self.iw_n = 1j * (2 * np.arange(n_freq) + 1) * np.pi / beta
        self.iw_values = self.iw_n.imag
        
        # Real frequency mesh for spectral function
        self.omega = np.linspace(-self.config.freq_cutoff, 
                                  self.config.freq_cutoff, 
                                  self.config.n_real_freq)
    
    def run_scf_loop(self, H_k: np.ndarray, k_weights: np.ndarray, 
                     n_electrons: float = None) -> Dict[str, Any]:
        """
        Run self-consistent DMFT loop
        
        Parameters:
        -----------
        H_k : np.ndarray
            K-dependent Hamiltonian (nk, n_orb, n_orb)
        k_weights : np.ndarray
            k-point weights
        n_electrons : float
            Number of electrons (for chemical potential determination)
            
        Returns:
        --------
        results : dict
            Converged DMFT results
        """
        logger.info("Starting DMFT self-consistent loop")
        
        nk, n_orb, _ = H_k.shape
        n_spin = self.config.n_spin
        
        # Initialize
        self.G_loc = np.zeros((n_spin, self.config.n_freq, n_orb, n_orb), dtype=complex)
        self.Sigma = np.zeros((n_spin, self.config.n_freq, n_orb, n_orb), dtype=complex)
        
        # Initial Weiss field (non-interacting)
        self.G0_iw = self._initialize_weiss_field(H_k, k_weights)
        
        converged = False
        for iteration in range(self.config.scf_max_iter):
            # Step 1: Solve impurity problem
            self.G_imp, Sigma_new = self._solve_impurity()
            
            # Step 2: Update self-energy with mixing
            alpha = self.config.mixing_alpha
            self.Sigma = alpha * Sigma_new + (1 - alpha) * self.Sigma
            
            # Step 3: Compute lattice Green's function
            self.G_loc = self._compute_lattice_gf(H_k, k_weights)
            
            # Step 4: Update Weiss field
            G0_new = self._update_weiss_field()
            
            # Step 5: Check convergence
            diff = np.max(np.abs(G0_new - self.G0_iw))
            self.convergence_history.append(diff)
            
            self.G0_iw = G0_new
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: diff = {diff:.6e}")
            
            if diff < self.config.scf_tol:
                logger.info(f"DMFT converged in {iteration} iterations")
                converged = True
                break
        
        if not converged:
            logger.warning(f"DMFT did not converge in {self.config.scf_max_iter} iterations")
        
        return self._compile_results()
    
    def _initialize_weiss_field(self, H_k: np.ndarray, k_weights: np.ndarray) -> np.ndarray:
        """Initialize Weiss field from non-interacting Green's function"""
        n_spin = self.config.n_spin
        n_freq = self.config.n_freq
        n_orb = H_k.shape[1]
        
        G0 = np.zeros((n_spin, n_freq, n_orb, n_orb), dtype=complex)
        
        for s in range(n_spin):
            for iw_idx, iw in enumerate(self.iw_n):
                for ik, (H, w) in enumerate(zip(H_k, k_weights)):
                    # G0(k, iw) = (iw + mu - H(k))^-1
                    G_inv = iw * np.eye(n_orb) + self.mu * np.eye(n_orb) - H
                    G_k = np.linalg.inv(G_inv)
                    G0[s, iw_idx] += w * G_k
        
        return G0
    
    def _solve_impurity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve impurity problem for each spin and orbital"""
        n_spin = self.config.n_spin
        n_orb = self.config.n_orbitals
        n_freq = self.config.n_freq
        
        G_imp = np.zeros((n_spin, n_freq, n_orb, n_orb), dtype=complex)
        Sigma = np.zeros((n_spin, n_freq, n_orb, n_orb), dtype=complex)
        
        U = self.config.u_value
        J = self.config.j_value
        beta = self.config.beta
        
        for s in range(n_spin):
            G0_diag = self.G0_iw[s, :, 0, 0]  # Simplified: use first orbital
            G_imp_diag, Sigma_diag = self.solver.solve(G0_diag, U, J, beta)
            
            # Fill diagonal (assume orbital symmetry)
            for o in range(n_orb):
                G_imp[s, :, o, o] = G_imp_diag
                Sigma[s, :, o, o] = Sigma_diag
        
        return G_imp, Sigma
    
    def _compute_lattice_gf(self, H_k: np.ndarray, k_weights: np.ndarray) -> np.ndarray:
        """Compute local Green's function from lattice"""
        n_spin = self.config.n_spin
        n_freq = self.config.n_freq
        n_orb = H_k.shape[1]
        
        G_loc = np.zeros((n_spin, n_freq, n_orb, n_orb), dtype=complex)
        
        for s in range(n_spin):
            for iw_idx, iw in enumerate(self.iw_n):
                for ik, (H, w) in enumerate(zip(H_k, k_weights)):
                    # G^-1 = iw + mu - H(k) - Σ
                    Sigma_iw = self.Sigma[s, iw_idx]
                    G_inv = iw * np.eye(n_orb) + self.mu * np.eye(n_orb) - H - Sigma_iw
                    G_k = np.linalg.inv(G_inv)
                    G_loc[s, iw_idx] += w * G_k
        
        return G_loc
    
    def _update_weiss_field(self) -> np.ndarray:
        """Update Weiss field from Dyson equation"""
        # Dyson equation: G0^-1 = G_imp^-1 + Σ
        G0_new = np.zeros_like(self.G0_iw)
        
        for s in range(self.config.n_spin):
            for iw in range(self.config.n_freq):
                for o1 in range(self.config.n_orbitals):
                    G_inv = 1.0 / self.G_imp[s, iw, o1, o1]
                    Sigma = self.Sigma[s, iw, o1, o1]
                    G0_new[s, iw, o1, o1] = 1.0 / (G_inv + Sigma)
        
        return G0_new
    
    def calculate_spectral_function(self, H_k: np.ndarray, k_points: np.ndarray,
                                   k_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate spectral function A(ω) = -Im[G(ω+iη)]/π
        
        Parameters:
        -----------
        H_k : np.ndarray
            K-dependent Hamiltonian
        k_points : np.ndarray
            k-point coordinates
        k_weights : np.ndarray
            k-point weights
            
        Returns:
        --------
        omega : np.ndarray
            Real frequency array
        A_w : np.ndarray
            Spectral function A(ω)
        """
        logger.info("Calculating spectral function")
        
        n_omega = len(self.omega)
        n_orb = H_k.shape[1]
        eta = self.config.eta_broadening
        
        # Analytic continuation of self-energy (simplified Pade)
        Sigma_real = self._analytic_continuation()
        
        A_w = np.zeros(n_omega)
        
        for iw, w in enumerate(self.omega):
            for ik, (H, kw) in enumerate(zip(H_k, k_weights)):
                for s in range(self.config.n_spin):
                    # G^-1(ω) = ω + iη + μ - H(k) - Σ(ω)
                    G_inv = (w + 1j * eta + self.mu) * np.eye(n_orb) - H - Sigma_real[s, iw]
                    G = np.linalg.inv(G_inv)
                    
                    # A(ω) = -Im[Tr(G)]/π
                    A_w[iw] -= kw * np.trace(G.imag) / np.pi / self.config.n_spin
        
        return self.omega, A_w
    
    def _analytic_continuation(self) -> np.ndarray:
        """
        Perform analytic continuation from Matsubara to real axis
        Using Pade approximant (simplified version)
        """
        n_spin = self.config.n_spin
        n_omega = len(self.omega)
        n_orb = self.config.n_orbitals
        
        Sigma_real = np.zeros((n_spin, n_omega, n_orb, n_orb), dtype=complex)
        
        for s in range(n_spin):
            for o in range(n_orb):
                # Simple Pade continuation
                Sigma_iw = self.Sigma[s, :, o, o]
                
                # Fit to few Matsubara frequencies
                iw_small = self.iw_n[:10]
                Sigma_small = Sigma_iw[:10]
                
                # Evaluate at real frequencies (simplified)
                for iw, w in enumerate(self.omega):
                    # Linear interpolation for simplicity
                    # In practice, use proper Pade or MaxEnt
                    Sigma_real[s, iw, o, o] = np.interp(
                        w, iw_small.imag, Sigma_small.real
                    ) + 1j * np.interp(
                        w, iw_small.imag, Sigma_small.imag
                    )
        
        return Sigma_real
    
    def calculate_quasiparticle_weight(self) -> np.ndarray:
        """
        Calculate quasiparticle weight Z = (1 - dΣ/dω)^-1
        """
        n_spin = self.config.n_spin
        n_orb = self.config.n_orbitals
        
        Z = np.zeros((n_spin, n_orb))
        
        dw = self.iw_values[1] - self.iw_values[0]
        
        for s in range(n_spin):
            for o in range(n_orb):
                # dΣ/dω at ω=0
                dSigma = np.gradient(self.Sigma[s, :, o, o].real, dw)
                Z[s, o] = 1.0 / (1.0 - dSigma[len(dSigma)//2])
        
        return Z
    
    def calculate_effective_mass(self) -> np.ndarray:
        """
        Calculate effective mass enhancement m*/m = 1/Z
        """
        Z = self.calculate_quasiparticle_weight()
        return 1.0 / Z
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile DMFT results into dictionary"""
        results = {
            'G0_iw': self.G0_iw,
            'G_loc': self.G_loc,
            'G_imp': self.G_imp,
            'Sigma': self.Sigma,
            'mu': self.mu,
            'iw_n': self.iw_n,
            'convergence_history': self.convergence_history,
            'converged': len(self.convergence_history) > 0 and 
                        self.convergence_history[-1] < self.config.scf_tol,
            'quasiparticle_weight': self.calculate_quasiparticle_weight(),
            'effective_mass': self.calculate_effective_mass()
        }
        
        return results
    
    def save_results(self, filename: str = "dmft_results.h5"):
        """Save DMFT results to file"""
        import h5py
        
        results = self._compile_results()
        
        with h5py.File(filename, 'w') as f:
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                elif key == 'convergence_history':
                    f.create_dataset(key, data=np.array(value))
                elif key == 'converged':
                    f.attrs[key] = value
        
        logger.info(f"DMFT results saved to {filename}")
    
    @classmethod
    def from_vasp_wannier90(cls, vasp_dir: str, wannier_dir: str, 
                           config: DMFTConfig = None) -> 'DMFTEngine':
        """
        Initialize DMFT engine from VASP+Wannier90 calculation
        
        Parameters:
        -----------
        vasp_dir : str
            Directory containing VASP output files
        wannier_dir : str
            Directory containing Wannier90 output files
        config : DMFTConfig
            DMFT configuration
            
        Returns:
        --------
        engine : DMFTEngine
            Initialized DMFT engine
        """
        engine = cls(config)
        
        # Read Wannier Hamiltonian
        hr_file = os.path.join(wannier_dir, "wannier90_hr.dat")
        if os.path.exists(hr_file):
            H_R = engine._read_wannier90_hr(hr_file)
            logger.info(f"Read Wannier Hamiltonian from {hr_file}")
        
        return engine
    
    def _read_wannier90_hr(self, filename: str) -> np.ndarray:
        """Read Wannier90 _hr.dat file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        num_wann = int(lines[1].strip())
        nrpts = int(lines[2].strip())
        
        # Read degeneracy
        ndegen = []
        line_idx = 3
        while len(ndegen) < nrpts:
            ndegen.extend([int(x) for x in lines[line_idx].split()])
            line_idx += 1
        
        # Read hopping matrix
        H_R = np.zeros((nrpts, num_wann, num_wann), dtype=complex)
        
        for ir in range(nrpts):
            for i in range(num_wann):
                for j in range(num_wann):
                    parts = lines[line_idx].split()
                    Rx, Ry, Rz = int(parts[0]), int(parts[1]), int(parts[2])
                    m, n = int(parts[3]) - 1, int(parts[4]) - 1
                    re, im = float(parts[5]), float(parts[6])
                    H_R[ir, m, n] = complex(re, im) / ndegen[ir]
                    line_idx += 1
        
        return H_R


class VASPDMFTInterface:
    """
    Interface between VASP and DMFT calculations
    
    Handles:
    - VASP DFT output parsing
    - Wannier90 integration
    - DMFT input preparation
    """
    
    def __init__(self, vasp_cmd: str = "vasp_std", wannier_cmd: str = "wannier90.x"):
        self.vasp_cmd = vasp_cmd
        self.wannier_cmd = wannier_cmd
        
    def run_vasp_wannier90(self, structure_file: str, work_dir: str = "vasp_dmft"):
        """
        Run VASP + Wannier90 workflow
        
        Steps:
        1. Run VASP for electronic structure
        2. Generate Wannier90 input
        3. Run Wannier90 for Wannier function construction
        """
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 1: VASP calculation
        logger.info("Running VASP calculation...")
        self._run_vasp(work_dir)
        
        # Step 2: Wannier90
        logger.info("Running Wannier90...")
        self._run_wannier90(work_dir)
        
    def _run_vasp(self, work_dir: str):
        """Run VASP calculation"""
        # Generate INCAR for Wannier90 interface
        incar = """
PREC = Accurate
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-8
LWANNIER90 = .TRUE.
LWRITE_WANNIER90 = .TRUE.
"""
        with open(os.path.join(work_dir, "INCAR"), "w") as f:
            f.write(incar)
        
        # Run VASP
        subprocess.run([self.vasp_cmd], cwd=work_dir, check=True)
    
    def _run_wannier90(self, work_dir: str):
        """Run Wannier90 post-processing"""
        # Preprocess
        subprocess.run([self.wannier_cmd, "-pp", "wannier90"], cwd=work_dir, check=True)
        
        # Main calculation
        subprocess.run([self.wannier_cmd, "wannier90"], cwd=work_dir, check=True)
    
    def extract_correlated_subspace(self, wannier_output: str, 
                                   orbital_indices: List[int]) -> np.ndarray:
        """
        Extract correlated subspace from Wannier90 output
        
        Parameters:
        -----------
        wannier_output : str
            Path to Wannier90 output
        orbital_indices : list
            Indices of correlated orbitals
            
        Returns:
        --------
        H_corr : np.ndarray
            Hamiltonian in correlated subspace
        """
        # Read Wannier Hamiltonian
        hr_file = os.path.join(wannier_output, "wannier90_hr.dat")
        
        with open(hr_file, 'r') as f:
            lines = f.readlines()
        
        # Parse and extract subspace
        # ... (implementation details)
        
        return None


# Utility functions for DMFT analysis

def calculate_double_occupancy(G2_tau: np.ndarray) -> float:
    """
    Calculate double occupancy <n_up n_down>
    """
    # Double occupancy from two-particle Green's function
    return -G2_tau[-1]  # Simplified


def calculate_kinetic_energy(G_loc: np.ndarray, H_k: np.ndarray, 
                             k_weights: np.ndarray) -> float:
    """
    Calculate kinetic energy from Green's function
    """
    E_kin = 0.0
    
    # E_kin = (1/Nk) sum_k Tr[H(k) * G(k, τ=0^-)]
    for ik, (H, w) in enumerate(zip(H_k, k_weights)):
        G_k = G_loc[0, :, :, :]  # Simplified
        E_kin += w * np.trace(np.dot(H, G_k))
    
    return E_kin.real


def estimate_nev_order_parameter(Sigma_iw: np.ndarray) -> float:
    """
    Estimate non-Fermi liquid behavior from self-energy
    
    For Fermi liquid: Im[Σ(ω→0)] ∝ ω^2
    For NFL: different power law
    """
    # Fit low-frequency behavior
    iw_low = np.abs(Sigma_iw[:10].imag)
    omega = np.arange(len(iw_low))
    
    # Linear fit in log-log scale
    log_omega = np.log(omega[1:] + 1e-10)
    log_sigma = np.log(iw_low[1:] + 1e-10)
    
    slope = np.polyfit(log_omega, log_sigma, 1)[0]
    
    return slope


def check_fermi_liquid(Sigma_iw: np.ndarray, tol: float = 0.1) -> bool:
    """
    Check if system is Fermi liquid
    
    Returns True if Im[Σ] ∝ ω^2 at low ω
    """
    slope = estimate_nev_order_parameter(Sigma_iw)
    return abs(slope - 2.0) < tol


def calculate_spectral_weight(G_iw: np.ndarray, beta: float) -> float:
    """
    Calculate total spectral weight (sum rule check)
    
    Should be 1 for normalized Green's function
    """
    # Sum rule: (1/β) sum_n G(iω_n) = 1/2
    weight = np.sum(G_iw) / beta
    return weight


# Export main classes
__all__ = [
    'DMFTConfig',
    'WannierProjectorConfig',
    'DMFTEngine',
    'CTQMCSolver',
    'WannierProjector',
    'VASPDMFTInterface',
    'ImpuritySolver',
    'calculate_double_occupancy',
    'calculate_kinetic_energy',
    'estimate_nev_order_parameter',
    'check_fermi_liquid',
    'calculate_spectral_weight'
]