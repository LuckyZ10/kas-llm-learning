"""
Diffusion Monte Carlo (DMC) Calculator
======================================

Implements Diffusion Monte Carlo methods for quantum chemistry calculations.

Features:
- Fixed-node approximation
- Projection energy calculation
- Reweighting and branching algorithms
- Population control

Author: QMC Expert Module
Date: 2026-03-09
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from copy import deepcopy


@dataclass
class DMCWalker:
    """Single DMC walker/particle."""
    positions: np.ndarray  # Electron positions (N_elec, 3)
    weight: float = 1.0
    age: int = 0
    local_energy: Optional[float] = None
    branch_weight: float = 1.0
    
    def copy(self) -> 'DMCWalker':
        return DMCWalker(
            positions=self.positions.copy(),
            weight=self.weight,
            age=self.age,
            local_energy=self.local_energy,
            branch_weight=self.branch_weight
        )


@dataclass
class DMCResults:
    """DMC calculation results."""
    energy: float
    energy_error: float
    energy_trial: float
    variance: float
    acceptance_rate: float
    n_walkers_avg: float
    n_walkers_final: int
    population_control_factor: float
    n_steps: int
    time_step: float
    projected_energies: List[float] = field(default_factory=list)
    mixed_energies: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'energy': self.energy,
            'energy_error': self.energy_error,
            'energy_trial': self.energy_trial,
            'variance': self.variance,
            'acceptance_rate': self.acceptance_rate,
            'n_walkers_avg': self.n_walkers_avg,
            'n_walkers_final': self.n_walkers_final,
            'population_control_factor': self.population_control_factor,
            'n_steps': self.n_steps,
            'time_step': self.time_step,
            'projected_energies': self.projected_energies,
            'mixed_energies': self.mixed_energies
        }


class TrialWaveFunction:
    """
    Trial wave function for fixed-node DMC.
    
    This is typically a Slater-Jastrow wave function
    or a neural network wave function.
    """
    
    def __init__(self,
                 log_psi_fn: Callable[[np.ndarray], float],
                 grad_log_psi_fn: Callable[[np.ndarray], np.ndarray],
                 lap_log_psi_fn: Callable[[np.ndarray], float]):
        """
        Initialize trial wave function.
        
        Parameters:
        -----------
        log_psi_fn : Callable
            Function to compute log(psi)
        grad_log_psi_fn : Callable
            Function to compute gradient of log(psi)
        lap_log_psi_fn : Callable
            Function to compute Laplacian of log(psi)
        """
        self.log_psi = log_psi_fn
        self.grad_log_psi = grad_log_psi_fn
        self.lap_log_psi = lap_log_psi_fn
    
    def drift_velocity(self, positions: np.ndarray) -> np.ndarray:
        """Compute drift velocity: v_D = grad(log|psi|)."""
        return self.grad_log_psi(positions)
    
    def local_kinetic_energy(self, positions: np.ndarray) -> float:
        """Compute local kinetic energy."""
        grad_log = self.grad_log_psi(positions)
        lap_log = self.lap_log_psi(positions)
        # T = -0.5 * (nabla^2 psi) / psi = -0.5 * (lap_log + grad_log^2)
        return -0.5 * (lap_log + np.sum(grad_log ** 2))


class DMCCalculator:
    """
    Diffusion Monte Carlo calculator.
    
    Implements the fixed-node DMC algorithm with:
    - Importance sampling
    - Branching/reweighting
    - Population control
    """
    
    def __init__(self,
                 trial_wf: TrialWaveFunction,
                 atom_positions: np.ndarray,
                 atom_charges: np.ndarray,
                 n_walkers_initial: int = 1000,
                 time_step: float = 0.01,
                 target_walkers: Optional[int] = None,
                 population_control_freq: int = 1,
                 seed: Optional[int] = None):
        """
        Initialize DMC calculator.
        
        Parameters:
        -----------
        trial_wf : TrialWaveFunction
            Trial wave function for importance sampling
        atom_positions : np.ndarray
            Nuclear positions (N_nuc, 3)
        atom_charges : np.ndarray
            Nuclear charges
        n_walkers_initial : int
            Initial number of walkers
        time_step : float
            DMC time step (tau)
        target_walkers : Optional[int]
            Target number of walkers for population control
        population_control_freq : int
            Frequency of population control steps
        seed : Optional[int]
            Random seed
        """
        self.trial_wf = trial_wf
        self.atom_positions = atom_positions
        self.atom_charges = atom_charges
        self.n_walkers_initial = n_walkers_initial
        self.time_step = time_step
        self.target_walkers = target_walkers or n_walkers_initial
        self.population_control_freq = population_control_freq
        
        if seed is not None:
            np.random.seed(seed)
        
        self.walkers: List[DMCWalker] = []
        self.trial_energy = 0.0
        
        # Statistics
        self.n_accepted = 0
        self.n_total = 0
        
    def _initialize_walkers(self,
                           n_electrons: int,
                           init_positions: Optional[np.ndarray] = None) -> List[DMCWalker]:
        """Initialize walker population."""
        walkers = []
        
        if init_positions is not None:
            # Use provided positions
            for i in range(self.n_walkers_initial):
                walker = DMCWalker(
                    positions=init_positions.copy(),
                    weight=1.0
                )
                walkers.append(walker)
        else:
            # Random initialization near nuclei
            for i in range(self.n_walkers_initial):
                positions = np.zeros((n_electrons, 3))
                for e in range(n_electrons):
                    atom_idx = e % len(self.atom_positions)
                    positions[e] = self.atom_positions[atom_idx] + \
                                  np.random.randn(3) * 0.5
                
                walker = DMCWalker(positions=positions, weight=1.0)
                walkers.append(walker)
        
        return walkers
    
    def _compute_local_energy(self, positions: np.ndarray) -> float:
        """
        Compute local energy.
        
        E_L = T_L + V_ne + V_ee + V_nn
        """
        n_elec = len(positions)
        
        # Kinetic energy from trial wave function
        kinetic = self.trial_wf.local_kinetic_energy(positions)
        
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
    
    def _diffusion_step(self, walker: DMCWalker) -> DMCWalker:
        """
        Perform one diffusion step with importance sampling.
        
        Uses the Green's function:
        G(R' <- R) = exp(-(R' - R - v_D*tau)^2 / (2*tau))
        """
        old_pos = walker.positions.copy()
        
        # Compute drift velocity
        drift = self.trial_wf.drift_velocity(old_pos) * self.time_step
        
        # Propose new position: R' = R + drift + random displacement
        new_pos = old_pos + drift + \
                 np.random.randn(*old_pos.shape) * np.sqrt(self.time_step)
        
        # Compute acceptance probability (fixed-node approximation)
        old_sign = np.sign(self.trial_wf.log_psi(old_pos))
        new_sign = np.sign(self.trial_wf.log_psi(new_pos))
        
        # Fixed-node constraint: reject moves that cross nodes
        if new_sign != old_sign:
            self.n_total += 1
            walker.age += 1
            return walker
        
        # Compute acceptance probability for importance sampling
        new_drift = self.trial_wf.drift_velocity(new_pos) * self.time_step
        
        # Forward and backward probabilities
        forward_prob = -np.sum((new_pos - old_pos - drift) ** 2) / (2 * self.time_step)
        backward_prob = -np.sum((old_pos - new_pos - new_drift) ** 2) / (2 * self.time_step)
        
        acceptance_prob = min(1.0, np.exp(backward_prob - forward_prob))
        
        self.n_total += 1
        
        if np.random.rand() < acceptance_prob:
            self.n_accepted += 1
            walker.positions = new_pos
            walker.age = 0
        else:
            walker.age += 1
        
        return walker
    
    def _compute_branching_weight(self,
                                  walker: DMCWalker,
                                  eref: float) -> float:
        """
        Compute branching weight.
        
        w_branch = exp(-tau * (E_L - E_ref))
        """
        local_e = self._compute_local_energy(walker.positions)
        walker.local_energy = local_e
        
        # Weight factor
        weight_factor = np.exp(-self.time_step * (local_e - eref))
        
        # Limit weight to prevent explosion
        weight_factor = np.clip(weight_factor, 0.1, 10.0)
        
        return weight_factor
    
    def _branching(self, walkers: List[DMCWalker]) -> List[DMCWalker]:
        """
        Perform branching/reweighting step.
        
        Creates or destroys walkers based on their weights.
        """
        new_walkers = []
        
        for walker in walkers:
            # Expected number of copies
            n_copies = int(walker.weight + np.random.rand())
            n_copies = min(n_copies, 3)  # Limit maximum copies
            
            if n_copies > 0:
                for _ in range(n_copies):
                    new_walker = walker.copy()
                    new_walker.weight = 1.0  # Reset weight
                    new_walkers.append(new_walker)
        
        return new_walkers
    
    def _population_control(self,
                           walkers: List[DMCWalker],
                           eref: float) -> Tuple[List[DMCWalker], float]:
        """
        Control walker population.
        
        Returns:
        --------
        (controlled_walkers, adjusted_eref)
        """
        current_n = len(walkers)
        
        # Adjust reference energy
        adjustment = 0.1 * np.log(self.target_walkers / max(1, current_n))
        eref_new = eref + adjustment
        
        # Scale weights if population too large/small
        if current_n > 2 * self.target_walkers:
            # Kill random walkers
            keep_indices = np.random.choice(current_n, 
                                          self.target_walkers, 
                                          replace=False)
            walkers = [walkers[i] for i in keep_indices]
        elif current_n < 0.5 * self.target_walkers:
            # Duplicate random walkers
            n_duplicate = self.target_walkers - current_n
            duplicate_indices = np.random.choice(current_n, n_duplicate, replace=True)
            for idx in duplicate_indices:
                walkers.append(walkers[idx].copy())
        
        return walkers, eref_new
    
    def _mixed_estimate_energy(self, walkers: List[DMCWalker]) -> float:
        """
        Compute mixed estimate of energy.
        
        E_mixed = <psi_T | H | phi> / <psi_T | phi>
                ≈ sum(w_i * E_L(R_i)) / sum(w_i)
        """
        total_weight = sum(w.weight for w in walkers)
        weighted_energy = sum(w.weight * (w.local_energy or 0) 
                             for w in walkers)
        
        return weighted_energy / max(total_weight, 1e-10)
    
    def _growth_estimate_energy(self,
                               walkers_old: List[DMCWalker],
                               walkers_new: List[DMCWalker],
                               eref: float) -> float:
        """
        Compute growth estimate of energy.
        
        E_growth = E_ref - (1/tau) * log(N_new / N_old)
        """
        n_old = len(walkers_old)
        n_new = len(walkers_new)
        
        if n_old > 0 and n_new > 0:
            return eref - np.log(n_new / n_old) / self.time_step
        return eref
    
    def equilibrate(self,
                   n_electrons: int,
                   n_steps: int = 1000,
                   init_positions: Optional[np.ndarray] = None) -> float:
        """
        Equilibrate DMC walkers.
        
        Parameters:
        -----------
        n_electrons : int
            Number of electrons
        n_steps : int
            Number of equilibration steps
        init_positions : Optional[np.ndarray]
            Initial electron positions
            
        Returns:
        --------
        Estimated trial energy
        """
        # Initialize walkers
        self.walkers = self._initialize_walkers(n_electrons, init_positions)
        
        # Initial trial energy estimate
        for walker in self.walkers:
            walker.local_energy = self._compute_local_energy(walker.positions)
        
        self.trial_energy = np.mean([w.local_energy for w in self.walkers])
        
        print(f"Starting DMC equilibration ({n_steps} steps)...")
        
        eref = self.trial_energy
        
        for step in range(n_steps):
            # Diffusion steps
            new_walkers = []
            for walker in self.walkers:
                new_walker = self._diffusion_step(walker)
                new_walkers.append(new_walker)
            
            self.walkers = new_walkers
            
            # Compute weights and energies
            for walker in self.walkers:
                weight_factor = self._compute_branching_weight(walker, eref)
                walker.weight *= weight_factor
            
            # Branching
            self.walkers = self._branching(self.walkers)
            
            # Population control
            if step % self.population_control_freq == 0:
                self.walkers, eref = self._population_control(self.walkers, eref)
            
            # Update trial energy
            if step > n_steps // 2:
                self.trial_energy = 0.9 * self.trial_energy + \
                                   0.1 * self._mixed_estimate_energy(self.walkers)
            
            if step % 100 == 0:
                print(f"  Step {step}: n_walkers = {len(self.walkers)}, "
                      f"E_ref = {eref:.6f}")
        
        return self.trial_energy
    
    def run(self,
           n_electrons: int,
           n_steps: int = 10000,
           n_equil: int = 1000,
           init_positions: Optional[np.ndarray] = None) -> DMCResults:
        """
        Run DMC calculation.
        
        Parameters:
        -----------
        n_electrons : int
            Number of electrons
        n_steps : int
            Number of production steps
        n_equil : int
            Number of equilibration steps
        init_positions : Optional[np.ndarray]
            Initial positions
            
        Returns:
        --------
        DMCResults object
        """
        # Equilibrate
        eref = self.equilibrate(n_electrons, n_equil, init_positions)
        
        print(f"\nStarting DMC production ({n_steps} steps)...")
        
        # Production run
        mixed_energies = []
        projected_energies = []
        walker_counts = []
        
        for step in range(n_steps):
            # Store old walkers for growth estimate
            walkers_old = [w.copy() for w in self.walkers]
            
            # Diffusion steps
            new_walkers = []
            for walker in self.walkers:
                new_walker = self._diffusion_step(walker)
                new_walkers.append(new_walker)
            
            self.walkers = new_walkers
            
            # Compute weights
            for walker in self.walkers:
                weight_factor = self._compute_branching_weight(walker, eref)
                walker.weight *= weight_factor
            
            # Branching
            self.walkers = self._branching(self.walkers)
            
            # Population control
            if step % self.population_control_freq == 0:
                self.walkers, eref = self._population_control(self.walkers, eref)
            
            # Record energies
            mixed_e = self._mixed_estimate_energy(self.walkers)
            growth_e = self._growth_estimate_energy(walkers_old, self.walkers, eref)
            
            mixed_energies.append(mixed_e)
            projected_energies.append(growth_e)
            walker_counts.append(len(self.walkers))
            
            if step % 500 == 0:
                avg_e = np.mean(mixed_energies[-100:]) if len(mixed_energies) >= 100 \
                       else np.mean(mixed_energies)
                print(f"  Step {step}: n_walkers = {len(self.walkers)}, "
                      f"E_mixed = {avg_e:.6f}")
        
        # Compute final statistics
        # Discard first 20% for thermalization
        burn_in = len(mixed_energies) // 5
        mixed_energies_trimmed = mixed_energies[burn_in:]
        
        energy_mean = np.mean(mixed_energies_trimmed)
        energy_var = np.var(mixed_energies_trimmed)
        energy_error = np.sqrt(energy_var / len(mixed_energies_trimmed))
        
        acceptance_rate = self.n_accepted / max(1, self.n_total)
        
        return DMCResults(
            energy=energy_mean,
            energy_error=energy_error,
            energy_trial=self.trial_energy,
            variance=energy_var,
            acceptance_rate=acceptance_rate,
            n_walkers_avg=np.mean(walker_counts),
            n_walkers_final=len(self.walkers),
            population_control_factor=self.target_walkers / max(1, np.mean(walker_counts)),
            n_steps=n_steps,
            time_step=self.time_step,
            projected_energies=projected_energies,
            mixed_energies=mixed_energies
        )
    
    def run_reweighted_dmc(self,
                          n_electrons: int,
                          n_steps: int = 10000,
                          n_equil: int = 1000,
                          init_positions: Optional[np.ndarray] = None) -> DMCResults:
        """
        Run DMC with pure reweighting (no branching).
        
        This is useful for studying systematic biases.
        """
        # Equilibrate
        eref = self.equilibrate(n_electrons, n_equil, init_positions)
        
        print(f"\nStarting reweighted DMC production ({n_steps} steps)...")
        
        mixed_energies = []
        walker_counts = []
        
        for step in range(n_steps):
            # Diffusion steps
            for walker in self.walkers:
                self._diffusion_step(walker)
            
            # Update weights
            for walker in self.walkers:
                weight_factor = self._compute_branching_weight(walker, eref)
                walker.weight *= weight_factor
            
            # Renormalize weights periodically
            if step % 10 == 0:
                total_weight = sum(w.weight for w in self.walkers)
                for walker in self.walkers:
                    walker.weight *= len(self.walkers) / total_weight
            
            # Record energy
            mixed_e = self._mixed_estimate_energy(self.walkers)
            mixed_energies.append(mixed_e)
            walker_counts.append(len(self.walkers))
        
        # Statistics
        burn_in = len(mixed_energies) // 5
        mixed_energies_trimmed = mixed_energies[burn_in:]
        
        energy_mean = np.mean(mixed_energies_trimmed)
        energy_error = np.sqrt(np.var(mixed_energies_trimmed) / len(mixed_energies_trimmed))
        
        return DMCResults(
            energy=energy_mean,
            energy_error=energy_error,
            energy_trial=self.trial_energy,
            variance=np.var(mixed_energies_trimmed),
            acceptance_rate=self.n_accepted / max(1, self.n_total),
            n_walkers_avg=np.mean(walker_counts),
            n_walkers_final=len(self.walkers),
            population_control_factor=1.0,
            n_steps=n_steps,
            time_step=self.time_step,
            mixed_energies=mixed_energies,
            projected_energies=[]
        )


def create_trial_wf_from_vmc(vmc_wf) -> TrialWaveFunction:
    """
    Create trial wave function from VMC wave function.
    
    Parameters:
    -----------
    vmc_wf : WaveFunction
        VMC wave function object
    """
    return TrialWaveFunction(
        log_psi_fn=lambda r: vmc_wf.log_psi(r),
        grad_log_psi_fn=lambda r: vmc_wf.gradient_log_psi(r),
        lap_log_psi_fn=lambda r: vmc_wf.laplacian_log_psi(r)
    )


def extrapolate_dmc_energy(dmc_energy: float,
                          vmc_energy: float,
                          method: str = 'linear') -> float:
    """
    Extrapolate to zero time-step error.
    
    E(0) = 2*E_DMC - E_VMC (linear extrapolation)
    
    Parameters:
    -----------
    dmc_energy : float
        DMC energy
    vmc_energy : float
        VMC energy with same trial wave function
    method : str
        Extrapolation method
    """
    if method == 'linear':
        return 2 * dmc_energy - vmc_energy
    elif method == 'simple':
        return dmc_energy
    else:
        raise ValueError(f"Unknown extrapolation method: {method}")


# Time-step error analysis
def analyze_time_step_error(atom_positions: np.ndarray,
                           atom_charges: np.ndarray,
                           trial_wf: TrialWaveFunction,
                           n_electrons: int,
                           time_steps: List[float] = None) -> Dict:
    """
    Analyze time-step error by running DMC with different time steps.
    
    Parameters:
    -----------
    atom_positions : np.ndarray
        Nuclear positions
    atom_charges : np.ndarray
        Nuclear charges
    trial_wf : TrialWaveFunction
        Trial wave function
    n_electrons : int
        Number of electrons
    time_steps : List[float]
        List of time steps to test
        
    Returns:
    --------
    Dict with results for each time step
    """
    if time_steps is None:
        time_steps = [0.001, 0.005, 0.01, 0.02]
    
    results = {}
    
    for tau in time_steps:
        print(f"\nRunning DMC with time step {tau}")
        
        dmc = DMCCalculator(
            trial_wf=trial_wf,
            atom_positions=atom_positions,
            atom_charges=atom_charges,
            time_step=tau,
            n_walkers_initial=500
        )
        
        result = dmc.run(n_electrons, n_steps=5000, n_equil=500)
        
        results[tau] = {
            'energy': result.energy,
            'error': result.energy_error,
            'variance': result.variance
        }
        
        print(f"  Energy: {result.energy:.6f} ± {result.energy_error:.6f}")
    
    # Fit to E(tau) = E_0 + a*tau + b*tau^2
    taus = np.array(list(results.keys()))
    energies = np.array([results[t]['energy'] for t in taus])
    
    # Linear fit
    p = np.polyfit(taus, energies, 1)
    e_extrapolated = p[1]  # Intercept at tau = 0
    
    results['extrapolated'] = {
        'energy': e_extrapolated,
        'fit_coefficients': p.tolist()
    }
    
    return results
