"""
Quantum Monte Carlo Dynamics
============================

Implements QMC-based dynamics methods including:
- Path Integral Monte Carlo (PIMC)
- Finite temperature properties
- Quantum dynamics simulations

Author: QMC Expert Module
Date: 2026-03-09
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings


@dataclass
class PIMCBead:
    """Single bead in path integral."""
    positions: np.ndarray  # Positions of all particles
    potential: float = 0.0
    kinetic: float = 0.0
    
    def copy(self) -> 'PIMCBead':
        return PIMCBead(
            positions=self.positions.copy(),
            potential=self.potential,
            kinetic=self.kinetic
        )


@dataclass
class PIMCResults:
    """PIMC calculation results."""
    temperature: float
    beta: float
    n_beads: int
    energy: float
    energy_error: float
    potential_energy: float
    kinetic_energy: float
    heat_capacity: float
    acceptance_rate: float
    
    def to_dict(self) -> Dict:
        return {
            'temperature': self.temperature,
            'beta': self.beta,
            'n_beads': self.n_beads,
            'energy': self.energy,
            'energy_error': self.energy_error,
            'potential_energy': self.potential_energy,
            'kinetic_energy': self.kinetic_energy,
            'heat_capacity': self.heat_capacity,
            'acceptance_rate': self.acceptance_rate
        }


class PotentialEnergySurface:
    """
    Potential energy surface for nuclear dynamics.
    
    Can be constructed from electronic structure calculations
    or using analytical forms.
    """
    
    def __init__(self,
                 potential_fn: Callable[[np.ndarray], float],
                 gradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Initialize PES.
        
        Parameters:
        -----------
        potential_fn : Callable
            Function that computes potential energy given positions
        gradient_fn : Optional[Callable]
            Function that computes gradient of potential
        """
        self.potential_fn = potential_fn
        self.gradient_fn = gradient_fn
    
    def potential(self, positions: np.ndarray) -> float:
        """Compute potential energy."""
        return self.potential_fn(positions)
    
    def gradient(self, positions: np.ndarray) -> np.ndarray:
        """Compute gradient."""
        if self.gradient_fn is not None:
            return self.gradient_fn(positions)
        
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros_like(positions)
        
        for i in range(positions.shape[0]):
            for d in range(positions.shape[1]):
                pos_plus = positions.copy()
                pos_minus = positions.copy()
                pos_plus[i, d] += eps
                pos_minus[i, d] -= eps
                
                grad[i, d] = (self.potential(pos_plus) - 
                             self.potential(pos_minus)) / (2 * eps)
        
        return grad
    
    @classmethod
    def harmonic_oscillator(cls, 
                           masses: np.ndarray,
                           omega: float = 1.0) -> 'PotentialEnergySurface':
        """Create harmonic oscillator PES."""
        def potential(positions):
            return 0.5 * omega**2 * np.sum(masses[:, None] * positions**2)
        
        def gradient(positions):
            return omega**2 * masses[:, None] * positions
        
        return cls(potential, gradient)
    
    @classmethod
    def lennard_jones(cls,
                     epsilon: float = 1.0,
                     sigma: float = 1.0) -> 'PotentialEnergySurface':
        """Create Lennard-Jones PES."""
        def potential(positions):
            n_atoms = len(positions)
            energy = 0.0
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r = np.linalg.norm(positions[i] - positions[j])
                    if r > 0:
                        sr6 = (sigma / r) ** 6
                        energy += 4 * epsilon * sr6 * (sr6 - 1)
            return energy
        
        return cls(potential)


class PathIntegralMonteCarlo:
    """
    Path Integral Monte Carlo for finite temperature quantum systems.
    
    Uses the path integral formulation:
    Z = ∫ D[x(τ)] exp(-S_E[x(τ)])
    
    where S_E is the Euclidean action.
    """
    
    def __init__(self,
                 n_atoms: int,
                 masses: np.ndarray,
                 temperature: float,
                 pes: PotentialEnergySurface,
                 n_beads: int = 32,
                 seed: Optional[int] = None):
        """
        Initialize PIMC calculator.
        
        Parameters:
        -----------
        n_atoms : int
            Number of atoms
        masses : np.ndarray
            Atomic masses (n_atoms,)
        temperature : float
            Temperature in energy units (k_B T)
        pes : PotentialEnergySurface
            Potential energy surface
        n_beads : int
            Number of path integral beads (Trotter number)
        seed : Optional[int]
            Random seed
        """
        self.n_atoms = n_atoms
        self.masses = masses
        self.temperature = temperature
        self.beta = 1.0 / temperature  # 1/(k_B T)
        self.pes = pes
        self.n_beads = n_beads
        
        if seed is not None:
            np.random.seed(seed)
        
        # Path: array of beads
        self.path: List[PIMCBead] = []
        self._initialize_path()
        
        # Derived quantities
        self.tau = self.beta / n_beads  # Imaginary time step
        self.omega_n = 1.0 / self.tau  # Matsubara frequency
        
    def _initialize_path(self):
        """Initialize path with random configurations."""
        self.path = []
        for _ in range(self.n_beads):
            positions = np.random.randn(self.n_atoms, 3)
            bead = PIMCBead(positions=positions)
            bead.potential = self.pes.potential(positions)
            self.path.append(bead)
    
    def _compute_kinetic_action(self, bead_i: int, bead_j: int) -> float:
        """
        Compute kinetic action between two beads.
        
        S_kin = (m / 2*tau) * (R_i - R_j)^2
        """
        pos_i = self.path[bead_i].positions
        pos_j = self.path[bead_j].positions
        
        diff = pos_i - pos_j
        kinetic = 0.5 * np.sum(self.masses[:, None] * diff**2) / self.tau
        
        return kinetic
    
    def _compute_total_action(self) -> float:
        """
        Compute total Euclidean action.
        
        S = sum_beads [ (m/2*tau) * (R_i - R_{i+1})^2 + tau * V(R_i) ]
        """
        action = 0.0
        
        for i in range(self.n_beads):
            j = (i + 1) % self.n_beads  # Periodic boundary conditions
            
            # Kinetic term
            action += self._compute_kinetic_action(i, j)
            
            # Potential term
            action += self.tau * self.path[i].potential
        
        return action
    
    def _metropolis_step_bead(self, bead_idx: int, step_size: float) -> bool:
        """
        Perform Metropolis move for single bead.
        
        Parameters:
        -----------
        bead_idx : int
            Index of bead to move
        step_size : float
            Maximum displacement
            
        Returns:
        --------
        True if move accepted
        """
        old_bead = self.path[bead_idx].copy()
        
        # Propose new position
        new_positions = old_bead.positions + np.random.randn(*old_bead.positions.shape) * step_size
        new_potential = self.pes.potential(new_positions)
        
        # Compute action change
        prev_idx = (bead_idx - 1) % self.n_beads
        next_idx = (bead_idx + 1) % self.n_beads
        
        # Old action contribution
        old_action = (self._compute_kinetic_action(prev_idx, bead_idx) +
                     self._compute_kinetic_action(bead_idx, next_idx) +
                     self.tau * old_bead.potential)
        
        # New action contribution
        old_pos = self.path[bead_idx].positions
        self.path[bead_idx].positions = new_positions
        self.path[bead_idx].potential = new_potential
        
        new_action = (self._compute_kinetic_action(prev_idx, bead_idx) +
                     self._compute_kinetic_action(bead_idx, next_idx) +
                     self.tau * new_potential)
        
        # Metropolis acceptance
        delta_action = new_action - old_action
        
        if np.random.rand() < np.exp(-delta_action):
            return True
        else:
            # Reject move
            self.path[bead_idx] = old_bead
            return False
    
    def _center_of_mass_move(self, step_size: float) -> bool:
        """Move center of mass of all beads."""
        # Compute center of mass
        com = np.mean([bead.positions for bead in self.path], axis=0)
        
        # Propose displacement
        displacement = np.random.randn(*com.shape) * step_size
        
        # Accept/reject based on potential change
        old_action = sum(bead.potential for bead in self.path) * self.tau
        
        for bead in self.path:
            bead.positions += displacement
            bead.potential = self.pes.potential(bead.positions)
        
        new_action = sum(bead.potential for bead in self.path) * self.tau
        
        if np.random.rand() < np.exp(-(new_action - old_action)):
            return True
        else:
            # Reject
            for bead in self.path:
                bead.positions -= displacement
                bead.potential = self.pes.potential(bead.positions)
            return False
    
    def _staging_move(self, bead_idx: int, stage_length: int = 4) -> bool:
        """
        Perform staging move for improved sampling.
        
        This is more efficient for harmonic degrees of freedom.
        """
        if stage_length >= self.n_beads:
            return False
        
        # Select staging region
        start_idx = bead_idx
        end_idx = (bead_idx + stage_length) % self.n_beads
        
        # Save old positions
        old_positions = [self.path[(start_idx + i) % self.n_beads].positions.copy()
                        for i in range(stage_length - 1)]
        
        # Generate new positions using free particle sampling
        for i in range(1, stage_length):
            idx = (start_idx + i) % self.n_beads
            
            # Mean position from neighbors
            prev_pos = self.path[(idx - 1) % self.n_beads].positions
            next_pos = self.path[(idx + 1) % self.n_beads].positions if i < stage_length - 1 \
                      else self.path[end_idx].positions
            
            # Sample from Gaussian distribution
            mean = 0.5 * (prev_pos + next_pos)
            sigma = np.sqrt(self.tau / (2 * self.masses[:, None]))
            
            self.path[idx].positions = mean + np.random.randn(*mean.shape) * sigma
            self.path[idx].potential = self.pes.potential(self.path[idx].positions)
        
        # Accept/reject
        old_action = 0.0
        new_action = 0.0
        
        for i in range(stage_length):
            idx = (start_idx + i) % self.n_beads
            next_idx = (idx + 1) % self.n_beads
            
            old_action += (self._compute_kinetic_action(idx, next_idx) +
                          self.tau * old_positions[i % (stage_length - 1)] if 
                          i > 0 and i < stage_length - 1 else 0)
            new_action += (self._compute_kinetic_action(idx, next_idx) +
                          self.tau * self.path[idx].potential)
        
        if np.random.rand() < np.exp(-(new_action - old_action)):
            return True
        else:
            # Restore old positions
            for i in range(1, stage_length):
                idx = (start_idx + i) % self.n_beads
                self.path[idx].positions = old_positions[i - 1]
                self.path[idx].potential = self.pes.potential(old_positions[i - 1])
            return False
    
    def _thermostat(self) -> None:
        """Apply thermostat to maintain temperature."""
        # Simple velocity rescaling (not used in PIMC, kept for compatibility)
        pass
    
    def run(self,
           n_steps: int = 10000,
           n_equil: int = 1000,
           bead_step_size: float = 0.1,
           com_step_size: float = 0.5) -> PIMCResults:
        """
        Run PIMC simulation.
        
        Parameters:
        -----------
        n_steps : int
            Number of production steps
        n_equil : int
            Number of equilibration steps
        bead_step_size : float
            Maximum bead displacement
        com_step_size : float
            Maximum center-of-mass displacement
            
        Returns:
        --------
        PIMCResults object
        """
        print(f"Running PIMC: T = {self.temperature}, P = {self.n_beads} beads")
        
        # Equilibration
        print(f"Equilibrating for {n_equil} steps...")
        n_accepted = 0
        n_total = 0
        
        for step in range(n_equil):
            # Single bead moves
            for i in range(self.n_beads):
                if self._metropolis_step_bead(i, bead_step_size):
                    n_accepted += 1
                n_total += 1
            
            # Center of mass move
            if step % 10 == 0:
                if self._center_of_mass_move(com_step_size):
                    n_accepted += 1
                n_total += 1
        
        # Production
        print(f"Production run for {n_steps} steps...")
        
        energies = []
        potential_energies = []
        kinetic_energies = []
        
        for step in range(n_steps):
            # MC steps
            for i in range(self.n_beads):
                if self._metropolis_step_bead(i, bead_step_size):
                    n_accepted += 1
                n_total += 1
            
            if step % 10 == 0:
                if self._center_of_mass_move(com_step_size):
                    n_accepted += 1
                n_total += 1
            
            # Measurements
            if step % 10 == 0:
                # Thermodynamic estimator for potential energy
                v_avg = np.mean([bead.potential for bead in self.path])
                potential_energies.append(v_avg)
                
                # Virial estimator for kinetic energy
                k_virial = 0.0
                for i, bead in enumerate(self.path):
                    grad = self.pes.gradient(bead.positions)
                    k_virial += np.sum(bead.positions * grad)
                k_virial /= (2 * self.n_beads)
                k_virial = 1.5 * self.n_atoms * self.temperature - k_virial
                kinetic_energies.append(k_virial)
                
                # Total energy
                energies.append(v_avg + k_virial)
            
            if step % 1000 == 0 and step > 0:
                avg_e = np.mean(energies[-100:])
                print(f"  Step {step}: E = {avg_e:.6f}")
        
        # Compute statistics
        energy_mean = np.mean(energies)
        energy_error = np.std(energies) / np.sqrt(len(energies))
        
        # Heat capacity (from energy fluctuations)
        heat_capacity = (np.std(energies) ** 2) / (self.temperature ** 2)
        
        acceptance_rate = n_accepted / max(1, n_total)
        
        return PIMCResults(
            temperature=self.temperature,
            beta=self.beta,
            n_beads=self.n_beads,
            energy=energy_mean,
            energy_error=energy_error,
            potential_energy=np.mean(potential_energies),
            kinetic_energy=np.mean(kinetic_energies),
            heat_capacity=heat_capacity,
            acceptance_rate=acceptance_rate
        )
    
    def compute_radial_distribution(self,
                                   n_bins: int = 100,
                                   r_max: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radial distribution function g(r).
        
        Returns:
        --------
        (r_bins, g_r) : Radial distances and g(r) values
        """
        dr = r_max / n_bins
        r_bins = np.linspace(0, r_max, n_bins + 1)
        g_r = np.zeros(n_bins)
        
        # Collect distances from all beads
        for bead in self.path:
            for i in range(self.n_atoms):
                for j in range(i + 1, self.n_atoms):
                    r = np.linalg.norm(bead.positions[i] - bead.positions[j])
                    if r < r_max:
                        bin_idx = int(r / dr)
                        if bin_idx < n_bins:
                            g_r[bin_idx] += 1
        
        # Normalize
        volume = (4/3) * np.pi * r_max**3
        rho = self.n_atoms / volume
        
        for i in range(n_bins):
            shell_volume = 4 * np.pi * r_bins[i]**2 * dr
            g_r[i] /= (self.n_beads * self.n_atoms * (self.n_atoms - 1) / 2 * 
                      shell_volume * rho)
        
        return r_bins[:-1], g_r
    
    def get_instantaneous_positions(self) -> np.ndarray:
        """Get positions from all beads (for analysis)."""
        return np.array([bead.positions for bead in self.path])
    
    def get_centroid(self) -> np.ndarray:
        """Get centroid (classical-like position)."""
        return np.mean([bead.positions for bead in self.path], axis=0)


class QuantumDynamics:
    """
    Real-time quantum dynamics using surface hopping
    or other methods combined with QMC potentials.
    """
    
    def __init__(self,
                 n_atoms: int,
                 masses: np.ndarray,
                 pes_list: List[PotentialEnergySurface],
                 initial_state: int = 0):
        """
        Initialize quantum dynamics.
        
        Parameters:
        -----------
        n_atoms : int
            Number of atoms
        masses : np.ndarray
            Atomic masses
        pes_list : List[PotentialEnergySurface]
            List of PES for different electronic states
        initial_state : int
            Initial electronic state
        """
        self.n_atoms = n_atoms
        self.masses = masses
        self.pes_list = pes_list
        self.n_states = len(pes_list)
        self.current_state = initial_state
        
        self.positions = np.zeros((n_atoms, 3))
        self.velocities = np.zeros((n_atoms, 3))
        self.forces = np.zeros((n_atoms, 3))
        
    def initialize_from_pimc(self, pimc: PathIntegralMonteCarlo):
        """Initialize positions from PIMC centroid."""
        self.positions = pimc.get_centroid()
    
    def compute_forces(self, state: Optional[int] = None) -> np.ndarray:
        """Compute forces on current state."""
        state = state or self.current_state
        return -self.pes_list[state].gradient(self.positions)
    
    def velocity_verlet_step(self, dt: float):
        """One step of velocity Verlet integration."""
        # Half-step velocity
        self.velocities += 0.5 * dt * self.forces / self.masses[:, None]
        
        # Full step position
        self.positions += dt * self.velocities
        
        # Update forces
        self.forces = self.compute_forces()
        
        # Half-step velocity
        self.velocities += 0.5 * dt * self.forces / self.masses[:, None]
    
    def run_surface_hopping(self,
                           n_steps: int,
                           dt: float,
                           temperature: float = 300.0) -> Dict:
        """
        Run surface hopping dynamics.
        
        This is a simplified implementation of Tully's fewest switches
        surface hopping.
        
        Parameters:
        -----------
        n_steps : int
            Number of steps
        dt : float
            Time step
        temperature : float
            Temperature for hopping probabilities
            
        Returns:
        --------
        Dict with trajectory data
        """
        trajectory = []
        state_trajectory = []
        
        # Initialize forces
        self.forces = self.compute_forces()
        
        for step in range(n_steps):
            # Record state
            state_trajectory.append(self.current_state)
            trajectory.append(self.positions.copy())
            
            # MD step on current surface
            self.velocity_verlet_step(dt)
            
            # Attempt hop (simplified)
            if self.n_states > 1:
                # Compute energy gap
                e_current = self.pes_list[self.current_state].potential(self.positions)
                
                # Simple hopping criterion (thermal activation)
                for other_state in range(self.n_states):
                    if other_state != self.current_state:
                        e_other = self.pes_list[other_state].potential(self.positions)
                        delta_e = e_other - e_current
                        
                        # Boltzmann factor for uphill hops
                        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / temperature):
                            self.current_state = other_state
                            self.forces = self.compute_forces()
                            break
        
        return {
            'trajectory': np.array(trajectory),
            'states': np.array(state_trajectory),
            'positions': np.array(trajectory)
        }


def calculate_thermodynamic_properties(temperatures: List[float],
                                       n_atoms: int,
                                       masses: np.ndarray,
                                       pes: PotentialEnergySurface,
                                       n_beads: int = 32) -> Dict:
    """
    Calculate thermodynamic properties over a temperature range.
    
    Parameters:
    -----------
    temperatures : List[float]
        List of temperatures to sample
    n_atoms : int
        Number of atoms
    masses : np.ndarray
        Atomic masses
    pes : PotentialEnergySurface
        Potential energy surface
    n_beads : int
        Number of beads for PIMC
        
    Returns:
    --------
    Dict with thermodynamic properties at each temperature
    """
    results = {
        'temperatures': temperatures,
        'energies': [],
        'heat_capacities': [],
        'free_energies': []
    }
    
    for T in temperatures:
        print(f"\nCalculating properties at T = {T}")
        
        pimc = PathIntegralMonteCarlo(
            n_atoms=n_atoms,
            masses=masses,
            temperature=T,
            pes=pes,
            n_beads=n_beads
        )
        
        result = pimc.run(n_steps=5000, n_equil=500)
        
        results['energies'].append(result.energy)
        results['heat_capacities'].append(result.heat_capacity)
        
        # Free energy estimate (simplified)
        # F = E - T*S (entropy not directly computed here)
        results['free_energies'].append(result.energy)
        
        print(f"  E = {result.energy:.6f}, Cv = {result.heat_capacity:.6f}")
    
    return results
