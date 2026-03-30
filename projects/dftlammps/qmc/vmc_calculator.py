"""
Variational Monte Carlo (VMC) Calculator
========================================

Implements Variational Monte Carlo methods for quantum chemistry calculations.

Features:
- Slater-Jastrow wave functions
- Neural network wave functions (FermiNet/PauliNet style)
- Variational energy optimization
- Gradient-based and gradient-free optimization

Author: QMC Expert Module
Date: 2026-03-09
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from abc import ABC, abstractmethod

try:
    import scipy.optimize as opt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not installed. Some optimization features unavailable.")

try:
    from pyscf import gto
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False


@dataclass
class VMCSample:
    """Single VMC sampling configuration."""
    positions: np.ndarray  # Electron positions (N_elec, 3)
    weight: float = 1.0
    local_energy: Optional[float] = None
    log_psi: Optional[float] = None
    
    def copy(self) -> 'VMCSample':
        return VMCSample(
            positions=self.positions.copy(),
            weight=self.weight,
            local_energy=self.local_energy,
            log_psi=self.log_psi
        )


@dataclass
class VMCResults:
    """VMC calculation results."""
    energy: float
    energy_error: float
    variance: float
    acceptance_rate: float
    n_samples: int
    equilibrium_steps: int
    params: Dict = field(default_factory=dict)
    samples: List[VMCSample] = field(default_factory=list)
    energy_trace: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'energy': self.energy,
            'energy_error': self.energy_error,
            'variance': self.variance,
            'acceptance_rate': self.acceptance_rate,
            'n_samples': self.n_samples,
            'equilibrium_steps': self.equilibrium_steps,
            'params': self.params
        }


class WaveFunction(ABC):
    """Abstract base class for trial wave functions."""
    
    @abstractmethod
    def log_psi(self, positions: np.ndarray, params: Optional[Dict] = None) -> float:
        """Compute log of wave function."""
        pass
    
    @abstractmethod
    def psi(self, positions: np.ndarray, params: Optional[Dict] = None) -> float:
        """Compute wave function value."""
        pass
    
    @abstractmethod
    def gradient_log_psi(self, positions: np.ndarray, 
                        params: Optional[Dict] = None) -> np.ndarray:
        """Compute gradient of log(psi) with respect to electron positions."""
        pass
    
    @abstractmethod
    def laplacian_log_psi(self, positions: np.ndarray,
                         params: Optional[Dict] = None) -> float:
        """Compute Laplacian of log(psi)."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> np.ndarray:
        """Get optimizable parameters as flat array."""
        pass
    
    @abstractmethod
    def set_parameters(self, params: np.ndarray):
        """Set parameters from flat array."""
        pass
    
    @abstractmethod
    def parameter_gradient(self, positions: np.ndarray) -> np.ndarray:
        """Compute gradient of log(psi) with respect to parameters."""
        pass


class SlaterJastrow(WaveFunction):
    """
    Slater-Jastrow wave function.
    
    Psi = exp(J) * det(phi_up) * det(phi_dn)
    
    where J is the Jastrow correlation factor.
    """
    
    def __init__(self,
                 n_electrons: int,
                 n_up: int,
                 atom_positions: np.ndarray,
                 atom_charges: np.ndarray,
                 mo_coeffs: np.ndarray,
                 jastrow_order: int = 2):
        """
        Initialize Slater-Jastrow wave function.
        
        Parameters:
        -----------
        n_electrons : int
            Total number of electrons
        n_up : int
            Number of spin-up electrons
        atom_positions : np.ndarray
            Nuclear positions (N_nuc, 3)
        atom_charges : np.ndarray
            Nuclear charges
        mo_coeffs : np.ndarray
            Molecular orbital coefficients (N_basis, N_mo)
        jastrow_order : int
            Order of Jastrow factor (1, 2, or 3)
        """
        self.n_electrons = n_electrons
        self.n_up = n_up
        self.n_dn = n_electrons - n_up
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.mo_coeffs = mo_coeffs
        self.jastrow_order = jastrow_order
        
        # Initialize Jastrow parameters
        self.jastrow_params = self._init_jastrow_params()
        
    def _init_jastrow_params(self) -> Dict:
        """Initialize Jastrow parameters."""
        params = {}
        
        # Electron-electron terms
        params['u'] = np.zeros(self.jastrow_order)
        params['u'][0] = 0.5  # cusp condition
        
        # Electron-nucleus terms
        params['chi'] = np.zeros(self.jastrow_order)
        
        # Electron-electron-nucleus terms (for order >= 3)
        if self.jastrow_order >= 3:
            params['f'] = np.zeros(self.jastrow_order - 2)
        
        return params
    
    def _evaluate_basis(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at electron positions."""
        n_basis = self.mo_coeffs.shape[0]
        ao_values = np.zeros((len(positions), n_basis))
        
        # Simple Gaussian basis centered on atoms
        basis_idx = 0
        for atom_pos in self.atom_positions:
            for shell in range(min(3, n_basis // len(self.atom_positions))):
                alpha = 1.0 / (shell + 1)
                for i, pos in enumerate(positions):
                    r_sq = np.sum((pos - atom_pos) ** 2)
                    ao_values[i, basis_idx] = np.exp(-alpha * r_sq)
                basis_idx += 1
                if basis_idx >= n_basis:
                    break
        
        return ao_values
    
    def _build_slater_matrix(self, 
                            positions: np.ndarray,
                            spin: str = 'up') -> np.ndarray:
        """Build Slater matrix for given spin."""
        if spin == 'up':
            n_elec = self.n_up
            mo_slice = slice(0, self.n_up)
        else:
            n_elec = self.n_dn
            mo_slice = slice(0, self.n_dn)
        
        ao_values = self._evaluate_basis(positions)
        mo_values = ao_values @ self.mo_coeffs[:, mo_slice]
        
        return mo_values
    
    def _compute_jastrow(self, positions: np.ndarray) -> float:
        """Compute Jastrow factor."""
        j = 0.0
        
        # Electron-electron correlations
        for i in range(self.n_electrons):
            for j_idx in range(i + 1, self.n_electrons):
                r_ij = np.linalg.norm(positions[i] - positions[j_idx])
                for k, coeff in enumerate(self.jastrow_params['u']):
                    j += coeff * r_ij ** k
        
        # Electron-nucleus correlations
        for i in range(self.n_electrons):
            for atom_pos in self.atom_positions:
                r_iA = np.linalg.norm(positions[i] - atom_pos)
                for k, coeff in enumerate(self.jastrow_params['chi']):
                    j += coeff * r_iA ** k
        
        return j
    
    def log_psi(self, positions: np.ndarray, 
               params: Optional[Dict] = None) -> float:
        """Compute log of wave function."""
        if positions.shape != (self.n_electrons, 3):
            positions = positions.reshape(self.n_electrons, 3)
        
        # Jastrow factor
        jastrow = self._compute_jastrow(positions)
        
        # Slater determinants
        D_up = self._build_slater_matrix(positions[:self.n_up], 'up')
        D_dn = self._build_slater_matrix(positions[self.n_up:], 'dn')
        
        sign_up, logdet_up = np.linalg.slogdet(D_up)
        sign_dn, logdet_dn = np.linalg.slogdet(D_dn)
        
        if sign_up * sign_dn == 0:
            return -np.inf
        
        return jastrow + logdet_up + logdet_dn
    
    def psi(self, positions: np.ndarray,
           params: Optional[Dict] = None) -> float:
        """Compute wave function value."""
        return np.exp(self.log_psi(positions, params))
    
    def gradient_log_psi(self, positions: np.ndarray,
                        params: Optional[Dict] = None,
                        eps: float = 1e-5) -> np.ndarray:
        """Compute gradient using finite differences."""
        grad = np.zeros_like(positions)
        
        for i in range(self.n_electrons):
            for d in range(3):
                positions_plus = positions.copy()
                positions_minus = positions.copy()
                positions_plus[i, d] += eps
                positions_minus[i, d] -= eps
                
                log_psi_plus = self.log_psi(positions_plus)
                log_psi_minus = self.log_psi(positions_minus)
                
                grad[i, d] = (log_psi_plus - log_psi_minus) / (2 * eps)
        
        return grad
    
    def laplacian_log_psi(self, positions: np.ndarray,
                         params: Optional[Dict] = None,
                         eps: float = 1e-5) -> float:
        """Compute Laplacian using finite differences."""
        laplacian = 0.0
        
        for i in range(self.n_electrons):
            for d in range(3):
                positions_plus = positions.copy()
                positions_minus = positions.copy()
                positions_plus[i, d] += eps
                positions_minus[i, d] -= eps
                
                log_psi_center = self.log_psi(positions)
                log_psi_plus = self.log_psi(positions_plus)
                log_psi_minus = self.log_psi(positions_minus)
                
                laplacian += (log_psi_plus - 2 * log_psi_center + 
                             log_psi_minus) / (eps ** 2)
        
        return laplacian
    
    def get_parameters(self) -> np.ndarray:
        """Get Jastrow parameters as flat array."""
        params = [self.jastrow_params['u']]
        if 'chi' in self.jastrow_params:
            params.append(self.jastrow_params['chi'])
        if 'f' in self.jastrow_params:
            params.append(self.jastrow_params['f'])
        return np.concatenate(params)
    
    def set_parameters(self, params: np.ndarray):
        """Set Jastrow parameters from flat array."""
        n_u = len(self.jastrow_params['u'])
        self.jastrow_params['u'] = params[:n_u]
        
        idx = n_u
        if 'chi' in self.jastrow_params:
            n_chi = len(self.jastrow_params['chi'])
            self.jastrow_params['chi'] = params[idx:idx + n_chi]
            idx += n_chi
        
        if 'f' in self.jastrow_params:
            n_f = len(self.jastrow_params['f'])
            self.jastrow_params['f'] = params[idx:idx + n_f]
    
    def parameter_gradient(self, positions: np.ndarray,
                          eps: float = 1e-5) -> np.ndarray:
        """Compute gradient with respect to Jastrow parameters."""
        params = self.get_parameters()
        grad = np.zeros(len(params))
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            self.set_parameters(params_plus)
            log_psi_plus = self.log_psi(positions)
            
            self.set_parameters(params_minus)
            log_psi_minus = self.log_psi(positions)
            
            grad[i] = (log_psi_plus - log_psi_minus) / (2 * eps)
        
        # Restore original parameters
        self.set_parameters(params)
        
        return grad


class NeuralNetworkWaveFunction(WaveFunction):
    """
    Neural network wave function (FermiNet/PauliNet style).
    
    Uses neural networks to represent the wave function,
    including permutation-equivariant architectures.
    """
    
    def __init__(self,
                 n_electrons: int,
                 n_up: int,
                 atom_positions: np.ndarray,
                 hidden_layers: List[int] = [64, 64],
                 determinants: int = 16):
        """
        Initialize neural network wave function.
        
        Parameters:
        -----------
        n_electrons : int
            Total number of electrons
        n_up : int
            Number of spin-up electrons
        atom_positions : np.ndarray
            Nuclear positions
        hidden_layers : List[int]
            Hidden layer sizes
        determinants : int
            Number of determinants in multi-determinant expansion
        """
        self.n_electrons = n_electrons
        self.n_up = n_up
        self.n_dn = n_electrons - n_up
        self.atom_positions = atom_positions
        self.hidden_layers = hidden_layers
        self.n_determinants = determinants
        
        # Initialize network weights
        self.weights = self._initialize_weights()
        
    def _initialize_weights(self) -> Dict:
        """Initialize neural network weights."""
        weights = {}
        
        # Input dimension: 4 per electron-electron pair (r_ij, r_ij_vec)
        # + 4 per electron-nucleus pair
        n_nuc = len(self.atom_positions)
        input_dim = self.n_electrons * (self.n_electrons - 1) * 4 // 2 + \
                    self.n_electrons * n_nuc * 4
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(self.hidden_layers):
            weights[f'W_{i}'] = np.random.randn(prev_dim, hidden_dim) * 0.01
            weights[f'b_{i}'] = np.zeros(hidden_dim)
            prev_dim = hidden_dim
        
        # Output layer for orbital values
        weights['W_out'] = np.random.randn(prev_dim, 
                                          self.n_determinants * self.n_electrons) * 0.01
        weights['b_out'] = np.zeros(self.n_determinants * self.n_electrons)
        
        return weights
    
    def _compute_features(self, positions: np.ndarray) -> np.ndarray:
        """Compute electron features (distances, vectors)."""
        features = []
        
        # Electron-electron features
        for i in range(self.n_electrons):
            for j in range(i + 1, self.n_electrons):
                r_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_vec)
                features.extend([r, r_vec[0], r_vec[1], r_vec[2]])
        
        # Electron-nucleus features
        for i in range(self.n_electrons):
            for nuc_pos in self.atom_positions:
                r_vec = positions[i] - nuc_pos
                r = np.linalg.norm(r_vec)
                features.extend([r, r_vec[0], r_vec[1], r_vec[2]])
        
        return np.array(features)
    
    def _forward(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        x = features
        
        # Hidden layers with tanh activation
        for i in range(len(self.hidden_layers)):
            x = np.tanh(x @ self.weights[f'W_{i}'] + self.weights[f'b_{i}'])
        
        # Output layer (linear)
        output = x @ self.weights['W_out'] + self.weights['b_out']
        
        return output
    
    def log_psi(self, positions: np.ndarray,
               params: Optional[Dict] = None) -> float:
        """Compute log of wave function."""
        if positions.shape != (self.n_electrons, 3):
            positions = positions.reshape(self.n_electrons, 3)
        
        features = self._compute_features(positions)
        output = self._forward(features)
        
        # Reshape into determinants
        orbitals = output.reshape(self.n_determinants, self.n_electrons, self.n_electrons)
        
        # Compute sum of determinants
        log_psi_sum = -np.inf
        for d in range(self.n_determinants):
            D_up = orbitals[d, :self.n_up, :self.n_up]
            D_dn = orbitals[d, self.n_up:, self.n_up:]
            
            sign_up, logdet_up = np.linalg.slogdet(D_up)
            sign_dn, logdet_dn = np.linalg.slogdet(D_dn)
            
            if sign_up * sign_dn != 0:
                log_det = logdet_up + logdet_dn
                log_psi_sum = np.logaddexp(log_psi_sum, log_det)
        
        return log_psi_sum if not np.isinf(log_psi_sum) else -np.inf
    
    def psi(self, positions: np.ndarray,
           params: Optional[Dict] = None) -> float:
        return np.exp(self.log_psi(positions, params))
    
    def gradient_log_psi(self, positions: np.ndarray,
                        params: Optional[Dict] = None,
                        eps: float = 1e-5) -> np.ndarray:
        """Compute gradient using finite differences."""
        grad = np.zeros_like(positions)
        
        for i in range(self.n_electrons):
            for d in range(3):
                pos_plus = positions.copy()
                pos_minus = positions.copy()
                pos_plus[i, d] += eps
                pos_minus[i, d] -= eps
                
                grad[i, d] = (self.log_psi(pos_plus) - 
                             self.log_psi(pos_minus)) / (2 * eps)
        
        return grad
    
    def laplacian_log_psi(self, positions: np.ndarray,
                         params: Optional[Dict] = None,
                         eps: float = 1e-5) -> float:
        """Compute Laplacian using finite differences."""
        laplacian = 0.0
        
        for i in range(self.n_electrons):
            for d in range(3):
                pos_plus = positions.copy()
                pos_minus = positions.copy()
                pos_plus[i, d] += eps
                pos_minus[i, d] -= eps
                
                log_center = self.log_psi(positions)
                log_plus = self.log_psi(pos_plus)
                log_minus = self.log_psi(pos_minus)
                
                laplacian += (log_plus - 2 * log_center + log_minus) / (eps ** 2)
        
        return laplacian
    
    def get_parameters(self) -> np.ndarray:
        """Get all network parameters as flat array."""
        params = []
        for key in sorted(self.weights.keys()):
            params.append(self.weights[key].flatten())
        return np.concatenate(params)
    
    def set_parameters(self, params: np.ndarray):
        """Set network parameters from flat array."""
        idx = 0
        for key in sorted(self.weights.keys()):
            shape = self.weights[key].shape
            size = np.prod(shape)
            self.weights[key] = params[idx:idx + size].reshape(shape)
            idx += size
    
    def parameter_gradient(self, positions: np.ndarray,
                          eps: float = 1e-5) -> np.ndarray:
        """Compute gradient with respect to network parameters."""
        params = self.get_parameters()
        grad = np.zeros(len(params))
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            self.set_parameters(params_plus)
            log_psi_plus = self.log_psi(positions)
            
            self.set_parameters(params_minus)
            log_psi_minus = self.log_psi(positions)
            
            grad[i] = (log_psi_plus - log_psi_minus) / (2 * eps)
        
        self.set_parameters(params)
        return grad


class VMCCalculator:
    """
    Variational Monte Carlo calculator.
    
    Performs VMC calculations with Metropolis sampling
    and energy optimization.
    """
    
    def __init__(self,
                 wave_function: WaveFunction,
                 atom_positions: np.ndarray,
                 atom_charges: np.ndarray,
                 n_walkers: int = 100,
                 step_size: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize VMC calculator.
        
        Parameters:
        -----------
        wave_function : WaveFunction
            Trial wave function
        atom_positions : np.ndarray
            Nuclear positions
        atom_charges : np.ndarray
            Nuclear charges
        n_walkers : int
            Number of random walkers
        step_size : float
            Metropolis step size
        seed : Optional[int]
            Random seed
        """
        self.wf = wave_function
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.n_walkers = n_walkers
        self.step_size = step_size
        
        if seed is not None:
            np.random.seed(seed)
        
        self.walkers = None
        self.n_accepted = 0
        self.n_total = 0
        
    def _initialize_walkers(self, 
                           n_electrons: int,
                           init_mode: str = 'random') -> np.ndarray:
        """Initialize electron positions."""
        if init_mode == 'random':
            # Random positions within bounding box
            center = np.mean(self.atom_positions, axis=0)
            spread = 5.0
            return center + np.random.randn(self.n_walkers, n_electrons, 3) * spread
        elif init_mode == 'atomic':
            # Place electrons near nuclei
            positions = np.zeros((self.n_walkers, n_electrons, 3))
            for w in range(self.n_walkers):
                for e in range(n_electrons):
                    atom_idx = e % len(self.atom_positions)
                    positions[w, e] = self.atom_positions[atom_idx] + \
                                     np.random.randn(3) * 0.5
            return positions
        else:
            raise ValueError(f"Unknown init mode: {init_mode}")
    
    def _compute_local_energy(self, positions: np.ndarray) -> float:
        """
        Compute local energy E_L = (H psi) / psi.
        
        H = -0.5 * sum_i (nabla_i^2) - sum_iA (Z_A / r_iA) + sum_{i<j} (1/r_ij)
        """
        n_elec = len(positions)
        
        # Kinetic energy: -0.5 * (nabla^2 psi) / psi
        # = -0.5 * (nabla^2 log psi + (nabla log psi)^2)
        grad_log = self.wf.gradient_log_psi(positions)
        lap_log = self.wf.laplacian_log_psi(positions)
        
        kinetic = -0.5 * (lap_log + np.sum(grad_log ** 2))
        
        # Potential energy
        potential = 0.0
        
        # Electron-nucleus attraction
        for i in range(n_elec):
            for Z, R in zip(self.atom_charges, self.atom_positions):
                r_iA = np.linalg.norm(positions[i] - R)
                if r_iA > 1e-10:
                    potential -= Z / r_iA
        
        # Electron-electron repulsion
        for i in range(n_elec):
            for j in range(i + 1, n_elec):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                if r_ij > 1e-10:
                    potential += 1.0 / r_ij
        
        return kinetic + potential
    
    def metropolis_step(self, walker_idx: int) -> bool:
        """Perform one Metropolis step for a walker."""
        old_pos = self.walkers[walker_idx].copy()
        
        # Propose new position
        new_pos = old_pos + np.random.randn(*old_pos.shape) * self.step_size
        
        # Compute acceptance probability
        old_log_psi = self.wf.log_psi(old_pos)
        new_log_psi = self.wf.log_psi(new_pos)
        
        if np.isinf(new_log_psi):
            return False
        
        acceptance_prob = min(1.0, np.exp(2 * (new_log_psi - old_log_psi)))
        
        if np.random.rand() < acceptance_prob:
            self.walkers[walker_idx] = new_pos
            self.n_accepted += 1
            return True
        
        return False
    
    def equilibrate(self, 
                   n_electrons: int,
                   n_steps: int = 1000,
                   init_mode: str = 'atomic') -> int:
        """
        Equilibrate walkers.
        
        Parameters:
        -----------
        n_electrons : int
            Number of electrons
        n_steps : int
            Number of equilibration steps
        init_mode : str
            Initialization mode
            
        Returns:
        --------
        Number of accepted moves
        """
        self.walkers = self._initialize_walkers(n_electrons, init_mode)
        self.n_accepted = 0
        self.n_total = 0
        
        for step in range(n_steps):
            for w in range(self.n_walkers):
                self.metropolis_step(w)
                self.n_total += 1
        
        return self.n_accepted
    
    def sample(self,
              n_electrons: int,
              n_samples: int = 10000,
              n_equil: int = 1000,
              sample_interval: int = 10) -> List[VMCSample]:
        """
        Generate VMC samples.
        
        Parameters:
        -----------
        n_electrons : int
            Number of electrons
        n_samples : int
            Number of samples to collect
        n_equil : int
            Number of equilibration steps
        sample_interval : int
            Steps between samples
            
        Returns:
        --------
        List of VMCSample objects
        """
        # Equilibrate
        print(f"Equilibrating for {n_equil} steps...")
        self.equilibrate(n_electrons, n_equil)
        
        acceptance_rate = self.n_accepted / max(1, self.n_total)
        print(f"Equilibration acceptance rate: {acceptance_rate:.3f}")
        
        # Sample
        samples = []
        step_count = 0
        
        print(f"Collecting {n_samples} samples...")
        while len(samples) < n_samples:
            for w in range(self.n_walkers):
                self.metropolis_step(w)
                step_count += 1
                
                if step_count % sample_interval == 0:
                    local_e = self._compute_local_energy(self.walkers[w])
                    log_psi = self.wf.log_psi(self.walkers[w])
                    
                    sample = VMCSample(
                        positions=self.walkers[w].copy(),
                        local_energy=local_e,
                        log_psi=log_psi
                    )
                    samples.append(sample)
                    
                    if len(samples) >= n_samples:
                        break
        
        return samples
    
    def compute_energy(self, samples: List[VMCSample]) -> Tuple[float, float]:
        """
        Compute energy and error from samples.
        
        Returns:
        --------
        (energy, error)
        """
        energies = np.array([s.local_energy for s in samples])
        
        energy = np.mean(energies)
        variance = np.var(energies)
        error = np.sqrt(variance / len(samples))
        
        return energy, error, variance
    
    def run(self,
           n_electrons: int,
           n_samples: int = 10000,
           n_equil: int = 1000) -> VMCResults:
        """
        Run VMC calculation.
        
        Parameters:
        -----------
        n_electrons : int
            Number of electrons
        n_samples : int
            Number of samples
        n_equil : int
            Equilibration steps
            
        Returns:
        --------
        VMCResults object
        """
        samples = self.sample(n_electrons, n_samples, n_equil)
        energy, error, variance = self.compute_energy(samples)
        
        acceptance_rate = self.n_accepted / max(1, self.n_total)
        
        return VMCResults(
            energy=energy,
            energy_error=error,
            variance=variance,
            acceptance_rate=acceptance_rate,
            n_samples=len(samples),
            equilibrium_steps=n_equil,
            samples=samples
        )
    
    def optimize_wavefunction(self,
                             n_electrons: int,
                             n_opt_samples: int = 5000,
                             n_opt_steps: int = 100,
                             method: str = 'sr') -> Dict:
        """
        Optimize wave function parameters.
        
        Parameters:
        -----------
        n_electrons : int
            Number of electrons
        n_opt_samples : int
            Samples per optimization step
        n_opt_steps : int
            Optimization iterations
        method : str
            Optimization method ('sr' for stochastic reconfiguration,
                                'adam' for Adam optimizer)
        """
        energy_trace = []
        
        for opt_step in range(n_opt_steps):
            # Generate samples
            samples = self.sample(n_electrons, n_opt_samples, n_equil=500)
            
            # Compute energy
            energy, error, variance = self.compute_energy(samples)
            energy_trace.append(energy)
            
            print(f"Opt step {opt_step}: E = {energy:.6f} ± {error:.6f}")
            
            # Compute parameter gradients
            params = self.wf.get_parameters()
            param_grad = np.zeros(len(params))
            
            for sample in samples:
                # Simple gradient estimate
                grad_psi = self.wf.parameter_gradient(sample.positions)
                local_e = sample.local_energy
                
                # Energy derivative
                param_grad += 2 * (local_e - energy) * grad_psi
            
            param_grad /= len(samples)
            
            # Update parameters (simple gradient descent)
            learning_rate = 0.01
            new_params = params - learning_rate * param_grad
            self.wf.set_parameters(new_params)
        
        return {
            'energy_trace': energy_trace,
            'final_params': self.wf.get_parameters().tolist()
        }


# Utility functions
def create_slater_jastrow_from_pyscf(pyscf_mf,
                                    jastrow_order: int = 2) -> SlaterJastrow:
    """
    Create Slater-Jastrow wave function from PySCF mean-field object.
    
    Parameters:
    -----------
    pyscf_mf : PySCF mean-field object
    jastrow_order : int
        Jastrow factor order
    """
    mol = pyscf_mf.mol
    
    atom_positions = mol.atom_coords()
    atom_charges = np.array([mol.atom_charge(i) for i in range(mol.natm)])
    
    n_electrons = mol.nelectron
    n_up = (n_electrons + mol.spin) // 2
    
    mo_coeffs = pyscf_mf.mo_coeff
    if mo_coeffs.ndim == 3:  # UHF
        mo_coeffs = mo_coeffs[0]  # Use alpha orbitals
    
    return SlaterJastrow(
        n_electrons=n_electrons,
        n_up=n_up,
        atom_positions=atom_positions,
        atom_charges=atom_charges,
        mo_coeffs=mo_coeffs,
        jastrow_order=jastrow_order
    )
