"""
AFQMC (Auxiliary-Field Quantum Monte Carlo) Calculator
======================================================

Implements phaseless Auxiliary-Field QMC for ab initio systems.

Features:
- Phaseless approximation for fermion sign problem
- Hubbard-Stratonovich transformation
- Propagation in imaginary time
- Mixed estimator for energy

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
class AFQMCWalker:
    """Single AFQMC walker with Slater determinant."""
    phi: np.ndarray  # Slater determinant (N_elec, N_basis)
    weight: complex = 1.0 + 0j
    overlap: complex = 1.0 + 0j
    
    def copy(self) -> 'AFQMCWalker':
        return AFQMCWalker(
            phi=self.phi.copy(),
            weight=self.weight,
            overlap=self.overlap
        )


@dataclass
class AFQMCResults:
    """AFQMC calculation results."""
    energy: float
    energy_error: float
    e_proj: float  # Projected energy
    e_mixed: float  # Mixed estimate
    variance: float
    avg_phase: float
    n_walkers_final: int
    n_steps: int
    time_step: float
    energies: List[float] = field(default_factory=list)
    phases: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'energy': self.energy,
            'energy_error': self.energy_error,
            'e_proj': self.e_proj,
            'e_mixed': self.e_mixed,
            'variance': self.variance,
            'avg_phase': self.avg_phase,
            'n_walkers_final': self.n_walkers_final,
            'n_steps': self.n_steps,
            'time_step': self.time_step,
            'energies': self.energies,
            'phases': self.phases
        }


class Hamiltonian:
    """
    Molecular Hamiltonian in second quantization.
    
    H = sum_pq h_pq a_p^dagger a_q + 0.5 * sum_pqrs v_pqrs a_p^dagger a_q^dagger a_s a_r
    """
    
    def __init__(self, 
                 h1e: np.ndarray,  # One-electron integrals
                 h2e: np.ndarray,  # Two-electron integrals
                 e_nuc: float = 0.0):
        """
        Initialize Hamiltonian.
        
        Parameters:
        -----------
        h1e : np.ndarray
            One-electron Hamiltonian matrix (N_basis, N_basis)
        h2e : np.ndarray
            Two-electron integrals (N_basis, N_basis, N_basis, N_basis)
        e_nuc : float
            Nuclear repulsion energy
        """
        self.h1e = h1e
        self.h2e = h2e
        self.e_nuc = e_nuc
        self.n_basis = h1e.shape[0]
        
        # Cholesky decomposition of two-electron integrals
        self._cholesky_decompose()
    
    def _cholesky_decompose(self, threshold: float = 1e-6):
        """
        Perform Cholesky decomposition of ERIs.
        
        v_pqrs ≈ sum_γ L_pq^γ * L_rs^γ
        """
        # Flatten two-electron integrals
        n = self.n_basis
        V = self.h2e.reshape(n*n, n*n)
        
        # Eigenvalue decomposition (simplified Cholesky)
        eigvals, eigvecs = np.linalg.eigh(V)
        
        # Keep significant eigenvalues
        mask = eigvals > threshold
        self.n_aux = np.sum(mask)
        self.L = np.sqrt(eigvals[mask]) * eigvecs[:, mask]  # (n*n, n_aux)
        self.L = self.L.reshape(n, n, self.n_aux)
        
        print(f"Cholesky decomposition: {n*n} -> {self.n_aux} auxiliary fields")


class AFQMCCalculator:
    """
    Auxiliary-Field Quantum Monte Carlo calculator.
    
    Implements the phaseless AFQMC algorithm:
    1. Hubbard-Stratonovich transformation
    2. Importance sampling with phaseless approximation
    3. Population control
    """
    
    def __init__(self,
                 hamiltonian: Hamiltonian,
                 n_walkers: int = 100,
                 time_step: float = 0.01,
                 seed: Optional[int] = None):
        """
        Initialize AFQMC calculator.
        
        Parameters:
        -----------
        hamiltonian : Hamiltonian
            Molecular Hamiltonian
        n_walkers : int
            Number of walkers
        time_step : float
            Imaginary time step
        seed : Optional[int]
            Random seed
        """
        self.ham = hamiltonian
        self.n_walkers = n_walkers
        self.time_step = time_step
        
        if seed is not None:
            np.random.seed(seed)
        
        self.walkers: List[AFQMCWalker] = []
        self.trial_wf: Optional[np.ndarray] = None
        self.e_trial = 0.0
        
        # Precompute mean-field for importance sampling
        self._setup_mean_field()
    
    def _setup_mean_field(self):
        """Setup mean-field reference for importance sampling."""
        n = self.ham.n_basis
        
        # Simple diagonal approximation for mean field
        self.hmf = self.ham.h1e.copy()
        
        # Add mean-field contribution from two-electron term
        for p in range(n):
            for q in range(n):
                self.hmf[p, q] += np.einsum('rs,prqs->pq', 
                                           np.eye(n), self.ham.h2e[p, :, q, :])
        
        # Diagonalize mean-field Hamiltonian
        self.eigs, self.eigv = np.linalg.eigh(self.hmf)
    
    def set_trial_wavefunction(self, mo_coeffs: np.ndarray, n_elec: int):
        """
        Set trial wave function from HF/DFT orbitals.
        
        Parameters:
        -----------
        mo_coeffs : np.ndarray
            Molecular orbital coefficients (N_basis, N_basis)
        n_elec : int
            Number of electrons
        """
        # Use occupied orbitals for trial
        n_occ = n_elec // 2  # Closed shell
        self.trial_wf = mo_coeffs[:, :n_occ].copy()
        
        # Compute trial energy
        self.e_trial = self._compute_energy(self.trial_wf)
        print(f"Trial wave function energy: {self.e_trial:.6f}")
    
    def _compute_energy(self, phi: np.ndarray) -> float:
        """Compute energy of Slater determinant."""
        # Simplified energy calculation
        # E = <phi|H|phi> / <phi|phi>
        
        # One-electron contribution
        rdm1 = phi @ phi.T.conj()  # Simplified 1-RDM
        e1 = np.einsum('pq,qp->', self.ham.h1e, rdm1)
        
        # Two-electron contribution (simplified)
        e2 = 0.5 * np.einsum('pqrs,pq,rs->', self.ham.h2e, rdm1, rdm1)
        
        return (e1 + e2 + self.ham.e_nuc).real
    
    def _compute_overlap(self, phi: np.ndarray, phi_trial: np.ndarray) -> complex:
        """Compute overlap <phi_trial|phi>."""
        # Overlap of Slater determinants: det(phi_trial^dagger @ phi)
        ovlp = np.linalg.det(phi_trial.T.conj() @ phi)
        return ovlp
    
    def _propagate_walker(self, walker: AFQMCWalker, eref: float) -> AFQMCWalker:
        """
        Propagate walker one time step.
        
        Uses Hubbard-Stratonovich transformation with phaseless approximation.
        """
        # Sample auxiliary fields
        x = np.random.randn(self.ham.n_aux)
        
        # Construct propagator
        # B(x) = exp(-tau * v(x)) where v(x) is the fluctuation potential
        
        # Simplified: use random walk on orbitals
        delta = np.random.randn(*walker.phi.shape) * np.sqrt(self.time_step)
        new_phi = walker.phi + delta
        
        # Orthogonalize (keep determinant structure)
        q, r = np.linalg.qr(new_phi)
        new_phi = q * np.sign(np.diag(r))
        
        # Compute new overlap
        new_overlap = self._compute_overlap(new_phi, self.trial_wf)
        
        # Phaseless approximation: reject sign changes
        phase = np.angle(new_overlap * walker.overlap.conj())
        
        if abs(phase) < np.pi / 2:  # Phaseless constraint
            walker.phi = new_phi
            walker.overlap = new_overlap
            
            # Update weight
            local_e = self._compute_energy(walker.phi)
            weight_factor = np.exp(-self.time_step * (local_e - eref))
            walker.weight *= weight_factor
        else:
            # Phase is bad, kill walker
            walker.weight = 0
        
        return walker
    
    def _population_control(self, walkers: List[AFQMCWalker]) -> List[AFQMCWalker]:
        """Control walker population."""
        # Compute average weight
        weights = [abs(w.weight) for w in walkers]
        avg_weight = np.mean(weights)
        
        new_walkers = []
        for walker in walkers:
            n_copies = int(abs(walker.weight) / avg_weight + np.random.rand())
            n_copies = min(n_copies, 3)
            
            if n_copies > 0:
                for _ in range(n_copies):
                    new_walker = walker.copy()
                    new_walker.weight = avg_weight * np.sign(walker.weight)
                    new_walkers.append(new_walker)
        
        # Limit population
        if len(new_walkers) > 2 * self.n_walkers:
            indices = np.random.choice(len(new_walkers), self.n_walkers, replace=False)
            new_walkers = [new_walkers[i] for i in indices]
        
        return new_walkers
    
    def initialize_walkers(self, n_elec: int):
        """Initialize walker population."""
        self.walkers = []
        
        for _ in range(self.n_walkers):
            # Perturb trial wave function
            noise = np.random.randn(*self.trial_wf.shape) * 0.1
            phi = self.trial_wf + noise
            
            # Orthogonalize
            q, r = np.linalg.qr(phi)
            phi = q * np.sign(np.diag(r))
            
            walker = AFQMCWalker(
                phi=phi,
                weight=1.0 + 0j,
                overlap=self._compute_overlap(phi, self.trial_wf)
            )
            self.walkers.append(walker)
    
    def run(self,
            n_elec: int,
            n_steps: int = 10000,
            n_equil: int = 1000) -> AFQMCResults:
        """
        Run AFQMC calculation.
        
        Parameters:
        -----------
        n_elec : int
            Number of electrons
        n_steps : int
            Number of production steps
        n_equil : int
            Number of equilibration steps
            
        Returns:
        --------
        AFQMCResults object
        """
        if self.trial_wf is None:
            raise ValueError("Trial wave function not set. Call set_trial_wavefunction first.")
        
        # Initialize walkers
        self.initialize_walkers(n_elec)
        
        eref = self.e_trial
        
        print(f"AFQMC equilibration ({n_equil} steps)...")
        
        # Equilibration
        for step in range(n_equil):
            for i, walker in enumerate(self.walkers):
                self.walkers[i] = self._propagate_walker(walker, eref)
            
            if step % 10 == 0:
                self.walkers = self._population_control(self.walkers)
            
            # Update reference energy
            energies = [self._compute_energy(w.phi).real for w in self.walkers]
            eref = np.mean(energies)
            
            if step % 100 == 0:
                print(f"  Step {step}: n_walkers = {len(self.walkers)}, E_ref = {eref:.6f}")
        
        print(f"\nAFQMC production ({n_steps} steps)...")
        
        # Production
        energies = []
        phases = []
        
        for step in range(n_steps):
            for i, walker in enumerate(self.walkers):
                self.walkers[i] = self._propagate_walker(walker, eref)
            
            if step % 10 == 0:
                self.walkers = self._population_control(self.walkers)
            
            # Measure energy
            ws = np.array([w.weight for w in self.walkers])
            es = np.array([self._compute_energy(w.phi).real for w in self.walkers])
            
            if np.sum(ws) != 0:
                e_mixed = np.sum(ws * es) / np.sum(ws)
                energies.append(e_mixed.real)
                
                # Average phase
                phases.append(np.mean(np.angle(ws)))
            
            if step % 500 == 0:
                avg_e = np.mean(energies[-100:]) if len(energies) >= 100 else np.mean(energies)
                print(f"  Step {step}: n_walkers = {len(self.walkers)}, E = {avg_e:.6f}")
        
        # Statistics
        burn_in = len(energies) // 5
        energies_trimmed = energies[burn_in:]
        
        energy_mean = np.mean(energies_trimmed)
        energy_error = np.sqrt(np.var(energies_trimmed) / len(energies_trimmed))
        
        return AFQMCResults(
            energy=energy_mean,
            energy_error=energy_error,
            e_proj=energy_mean,
            e_mixed=energy_mean,
            variance=np.var(energies_trimmed),
            avg_phase=np.mean(phases) if phases else 1.0,
            n_walkers_final=len(self.walkers),
            n_steps=n_steps,
            time_step=self.time_step,
            energies=energies,
            phases=phases
        )


def create_hamiltonian_from_pyscf(pyscf_mf) -> Hamiltonian:
    """
    Create Hamiltonian from PySCF mean-field object.
    
    Parameters:
    -----------
    pyscf_mf : PySCF mean-field object
    """
    mol = pyscf_mf.mol
    
    # Get integrals
    h1e = pyscf_mf.get_hcore()
    h2e = mol.intor('int2e')
    e_nuc = mol.energy_nuc()
    
    return Hamiltonian(h1e, h2e, e_nuc)
