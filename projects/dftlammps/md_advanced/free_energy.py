#!/usr/bin/env python3
"""
Free Energy Calculations
========================

Methods for computing free energy differences:
- Free Energy Perturbation (FEP)
- Thermodynamic Integration (TI)
- Bennett Acceptance Ratio (BAR)
- Weighted Histogram Analysis Method (WHAM)

References:
- Zwanzig (1954) - FEP
- Kirkwood (1935) - TI
- Bennett (1976) - BAR
- Kumar et al. (1992) - WHAM
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from functools import partial
import logging
import warnings
from scipy import optimize, integrate, interpolate, stats
from scipy.special import logsumexp, expit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FEPConfig:
    """Configuration for Free Energy Perturbation.
    
    Attributes:
        n_windows: Number of lambda windows
        lambda_schedule: Lambda values (0 to 1)
        direction: 'forward', 'backward', or 'both'
        nsteps_per_window: MD steps per window
        equil_steps: Equilibration steps per window
        temperature: Temperature in Kelvin
        sc_alpha: Soft-core alpha parameter
        sc_power: Soft-core power parameter
        sc_sigma: Soft-core sigma parameter
    """
    n_windows: int = 20
    lambda_schedule: Optional[List[float]] = None
    direction: str = "both"
    nsteps_per_window: int = 50000
    equil_steps: int = 10000
    temperature: float = 300.0
    sc_alpha: float = 0.5
    sc_power: int = 1
    sc_sigma: float = 0.3
    
    def __post_init__(self):
        if self.lambda_schedule is None:
            self.lambda_schedule = np.linspace(0, 1, self.n_windows).tolist()


@dataclass
class TIConfig:
    """Configuration for Thermodynamic Integration.
    
    Attributes:
        n_windows: Number of lambda windows
        lambda_schedule: Lambda values
        nsteps_per_window: MD steps per window
        temperature: Temperature in Kelvin
        gradient_method: 'analytical' or 'numerical'
    """
    n_windows: int = 20
    lambda_schedule: Optional[List[float]] = None
    nsteps_per_window: int = 50000
    equil_steps: int = 10000
    temperature: float = 300.0
    gradient_method: str = "numerical"
    
    def __post_init__(self):
        if self.lambda_schedule is None:
            # Use Gauss-Legendre quadrature points for better convergence
            from numpy.polynomial.legendre import leggauss
            x, _ = leggauss(self.n_windows)
            self.lambda_schedule = ((x + 1) / 2).tolist()


@dataclass
class BARConfig:
    """Configuration for Bennett Acceptance Ratio.
    
    Attributes:
        n_windows: Number of lambda windows
        lambda_schedule: Lambda values
        nsteps_per_window: MD steps per window
        tolerance: Convergence tolerance for self-consistent iteration
        max_iterations: Maximum iterations for BAR equation
    """
    n_windows: int = 20
    lambda_schedule: Optional[List[float]] = None
    nsteps_per_window: int = 50000
    equil_steps: int = 10000
    temperature: float = 300.0
    tolerance: float = 1e-6
    max_iterations: int = 100
    
    def __post_init__(self):
        if self.lambda_schedule is None:
            self.lambda_schedule = np.linspace(0, 1, self.n_windows).tolist()


@dataclass
class WHAMConfig:
    """Configuration for Weighted Histogram Analysis Method.
    
    Attributes:
        max_iterations: Maximum WHAM iterations
        tolerance: Convergence tolerance
        temperature: Temperature in Kelvin
        bin_width: Histogram bin width
        padding: Padding for histogram range
    """
    max_iterations: int = 10000
    tolerance: float = 1e-8
    temperature: float = 300.0
    bin_width: Optional[float] = None
    padding: float = 0.1


class FreeEnergyPerturbation:
    """Free Energy Perturbation calculation."""
    
    def __init__(self, config: FEPConfig):
        self.config = config
        self.window_data: List[Dict] = []
        self.delta_g_forward: Optional[float] = None
        self.delta_g_backward: Optional[float] = None
        self.delta_g: Optional[float] = None
    
    def compute_perturbation_energy(self, lambda_i: float, lambda_j: float,
                                    energies_i: np.ndarray, energies_j: np.ndarray) -> np.ndarray:
        """Compute perturbation energy dE = E_j - E_i."""
        return energies_j - energies_i
    
    def exponential_averaging(self, delta_energies: np.ndarray, 
                             temperature: float = None) -> float:
        """Compute free energy difference using exponential averaging (Zwanzig).
        
        dG = -kT * ln⟨exp(-ΔE/kT)⟩
        """
        if temperature is None:
            temperature = self.config.temperature
        
        beta = 1.0 / (0.001987204 * temperature)  # kcal/mol/K
        
        # Exponential averaging with overflow protection
        min_delta = np.min(delta_energies)
        exp_terms = np.exp(-beta * (delta_energies - min_delta))
        avg_exp = np.mean(exp_terms)
        
        delta_g = -1.0 / beta * (np.log(avg_exp) - beta * min_delta)
        
        return delta_g
    
    def compute_forward_work(self, energies: Dict[int, np.ndarray]) -> List[float]:
        """Compute forward work (λ_i -> λ_{i+1})."""
        work_values = []
        
        lambdas = self.config.lambda_schedule
        for i in range(len(lambdas) - 1):
            lambda_i = lambdas[i]
            lambda_j = lambdas[i + 1]
            
            # Energies sampled at lambda_i
            energies_i = energies[i]  # U_i(x_i)
            energies_j = energies[i + 1]  # U_j(x_i) - would need to re-evaluate
            
            delta_energies = energies_j - energies_i
            delta_g = self.exponential_averaging(delta_energies)
            work_values.append(delta_g)
        
        return work_values
    
    def compute_backward_work(self, energies: Dict[int, np.ndarray]) -> List[float]:
        """Compute backward work (λ_{i+1} -> λ_i)."""
        work_values = []
        
        lambdas = self.config.lambda_schedule
        for i in range(len(lambdas) - 1, 0, -1):
            lambda_i = lambdas[i]
            lambda_j = lambdas[i - 1]
            
            energies_i = energies[i]
            energies_j = energies[i - 1]
            
            # For backward: exp(-(U_{i-1} - U_i)/kT) sampled at i
            delta_energies = energies_j - energies_i
            delta_g = self.exponential_averaging(delta_energies)
            work_values.append(delta_g)
        
        return work_values[::-1]  # Reverse to match forward order
    
    def parse_energy_files(self, window_dirs: List[Path]) -> Dict[int, np.ndarray]:
        """Parse energy files from each window."""
        energies = {}
        
        for i, window_dir in enumerate(window_dirs):
            # Look for LAMMPS log or energy output
            energy_file = window_dir / "energy.dat"
            if energy_file.exists():
                data = np.loadtxt(energy_file)
                energies[i] = data[:, 1] if data.ndim > 1 else data  # Assume second column is energy
            else:
                logger.warning(f"Energy file not found: {energy_file}")
                energies[i] = np.array([])
        
        return energies
    
    def run(self, window_dirs: List[Path]) -> Dict:
        """Run FEP calculation."""
        logger.info("Running Free Energy Perturbation calculation")
        
        # Parse energies
        energies = self.parse_energy_files(window_dirs)
        
        # Compute forward and backward work
        if self.config.direction in ['forward', 'both']:
            forward_work = self.compute_forward_work(energies)
            self.delta_g_forward = np.sum(forward_work)
            logger.info(f"Forward ΔG: {self.delta_g_forward:.4f} kcal/mol")
        
        if self.config.direction in ['backward', 'both']:
            backward_work = self.compute_backward_work(energies)
            self.delta_g_backward = np.sum(backward_work)
            logger.info(f"Backward ΔG: {self.delta_g_backward:.4f} kcal/mol")
        
        # Use BAR-like average if both directions available
        if self.config.direction == 'both':
            self.delta_g = (self.delta_g_forward - self.delta_g_backward) / 2
            hysteresis = abs(self.delta_g_forward + self.delta_g_backward)
            logger.info(f"Average ΔG: {self.delta_g:.4f} kcal/mol")
            logger.info(f"Hysteresis: {hysteresis:.4f} kcal/mol")
        else:
            self.delta_g = self.delta_g_forward or self.delta_g_backward
        
        return {
            'delta_g': self.delta_g,
            'delta_g_forward': self.delta_g_forward,
            'delta_g_backward': self.delta_g_backward,
            'hysteresis': abs(self.delta_g_forward + self.delta_g_backward) if self.config.direction == 'both' else None,
            'converged': hysteresis < 1.0 if self.config.direction == 'both' else True
        }
    
    def estimate_error(self, n_bootstrap: int = 100) -> Tuple[float, float]:
        """Estimate error using bootstrap."""
        bootstrap_dg = []
        
        for _ in range(n_bootstrap):
            # Resample work values
            pass  # Implementation depends on stored work values
        
        return np.std(bootstrap_dg), stats.sem(bootstrap_dg)


class ThermodynamicIntegration:
    """Thermodynamic Integration calculation."""
    
    def __init__(self, config: TIConfig):
        self.config = config
        self.gradient_data: Dict[int, np.ndarray] = {}
        self.delta_g: Optional[float] = None
        self.delta_g_error: Optional[float] = None
    
    def compute_dU_dlambda(self, atoms, lambda_val: float) -> float:
        """Compute ∂U/∂λ analytically or numerically."""
        if self.config.gradient_method == 'analytical':
            return self._analytical_gradient(atoms, lambda_val)
        else:
            return self._numerical_gradient(atoms, lambda_val)
    
    def _analytical_gradient(self, atoms, lambda_val: float) -> float:
        """Compute analytical gradient for soft-core potentials."""
        # For linear mixing U(λ) = (1-λ)U0 + λU1
        # ∂U/∂λ = U1 - U0
        
        # Evaluate energies at endpoints
        energy_0 = self._evaluate_energy(atoms, lambda_val=0.0)
        energy_1 = self._evaluate_energy(atoms, lambda_val=1.0)
        
        return energy_1 - energy_0
    
    def _numerical_gradient(self, atoms, lambda_val: float,
                           delta: float = 0.001) -> float:
        """Compute numerical gradient using finite differences."""
        energy_plus = self._evaluate_energy(atoms, lambda_val + delta)
        energy_minus = self._evaluate_energy(atoms, lambda_val - delta)
        
        return (energy_plus - energy_minus) / (2 * delta)
    
    def _evaluate_energy(self, atoms, lambda_val: float) -> float:
        """Evaluate potential energy at given lambda."""
        # Placeholder - would call actual potential
        return 0.0
    
    def parse_gradient_files(self, window_dirs: List[Path]) -> Dict[int, np.ndarray]:
        """Parse dU/dλ data from each window."""
        gradients = {}
        
        for i, window_dir in enumerate(window_dirs):
            grad_file = window_dir / "dU_dlambda.dat"
            if grad_file.exists():
                data = np.loadtxt(grad_file)
                gradients[i] = data[:, 1] if data.ndim > 1 else data
            else:
                logger.warning(f"Gradient file not found: {grad_file}")
                gradients[i] = np.array([])
        
        return gradients
    
    def integrate(self, gradients: Dict[int, np.ndarray], 
                  lambdas: List[float]) -> float:
        """Integrate ⟨∂U/∂λ⟩ over λ."""
        # Compute mean gradient at each lambda
        mean_gradients = []
        for i in range(len(lambdas)):
            if i in gradients and len(gradients[i]) > 0:
                mean_gradients.append(np.mean(gradients[i]))
            else:
                mean_gradients.append(0.0)
        
        # Numerical integration using trapezoidal rule
        delta_g = integrate.trapezoid(mean_gradients, lambdas)
        
        return delta_g
    
    def estimate_error(self, gradients: Dict[int, np.ndarray],
                      lambdas: List[float]) -> float:
        """Estimate integration error from gradient variances."""
        variances = []
        for i in range(len(lambdas)):
            if i in gradients and len(gradients[i]) > 0:
                # Standard error of the mean
                sem = stats.sem(gradients[i])
                variances.append(sem ** 2)
            else:
                variances.append(0.0)
        
        # Error propagation through integration
        # Assuming independent windows
        error_squared = integrate.trapezoid(variances, lambdas)
        
        return np.sqrt(error_squared)
    
    def run(self, window_dirs: List[Path]) -> Dict:
        """Run TI calculation."""
        logger.info("Running Thermodynamic Integration")
        
        gradients = self.parse_gradient_files(window_dirs)
        
        self.delta_g = self.integrate(gradients, self.config.lambda_schedule)
        self.delta_g_error = self.estimate_error(gradients, self.config.lambda_schedule)
        
        logger.info(f"ΔG = {self.delta_g:.4f} ± {self.delta_g_error:.4f} kcal/mol")
        
        return {
            'delta_g': self.delta_g,
            'error': self.delta_g_error,
            'gradients': {i: np.mean(g) for i, g in gradients.items()},
            'lambdas': self.config.lambda_schedule
        }


class BennettAcceptanceRatio:
    """Bennett Acceptance Ratio for free energy calculation."""
    
    def __init__(self, config: BARConfig):
        self.config = config
        self.delta_g: Optional[float] = None
        self.delta_g_error: Optional[float] = None
    
    def bar_equation(self, delta_g: float, w_f: np.ndarray, 
                     w_r: np.ndarray, temperature: float) -> float:
        """BAR equation for root finding.
        
        Σ 1/(1 + exp[β(W_f + ΔG - C)]) = Σ 1/(1 + exp[β(W_r - ΔG + C)])
        where C = ln(N_f/N_r)/β
        """
        beta = 1.0 / (0.001987204 * temperature)
        
        N_f = len(w_f)
        N_r = len(w_r)
        C = np.log(N_f / N_r) / beta
        
        left = np.sum(1.0 / (1.0 + np.exp(beta * (w_f + delta_g - C))))
        right = np.sum(1.0 / (1.0 + np.exp(beta * (w_r - delta_g + C))))
        
        return left - right
    
    def solve_bar(self, w_f: np.ndarray, w_r: np.ndarray,
                  initial_guess: float = 0.0) -> float:
        """Solve BAR equation self-consistently."""
        try:
            solution = optimize.root_scalar(
                lambda dg: self.bar_equation(dg, w_f, w_r, self.config.temperature),
                x0=initial_guess,
                method='newton',
                x1=initial_guess + 1.0
            )
            return solution.root
        except Exception as e:
            logger.warning(f"BAR solver failed: {e}, using initial guess")
            return initial_guess
    
    def compute_bar_variance(self, w_f: np.ndarray, w_r: np.ndarray,
                            delta_g: float, temperature: float) -> float:
        """Compute variance of BAR estimate."""
        beta = 1.0 / (0.001987204 * temperature)
        
        N_f = len(w_f)
        N_r = len(w_r)
        
        # Compute overlap integral
        # This is a simplified version
        f_arg = beta * (w_f + delta_g)
        r_arg = beta * (w_r - delta_g)
        
        f_terms = expit(-f_arg)
        r_terms = expit(-r_arg)
        
        # Variance estimate
        var = 1.0 / (np.sum(f_terms * (1 - f_terms)) / N_f + 
                     np.sum(r_terms * (1 - r_terms)) / N_r)
        
        return var / beta ** 2
    
    def parse_work_files(self, window_dirs: List[Path]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Parse forward and backward work files."""
        forward_work = {}
        backward_work = {}
        
        for i, window_dir in enumerate(window_dirs[:-1]):
            # Forward work: U_{i+1}(x_i) - U_i(x_i)
            fw_file = window_dir / "work_forward.dat"
            if fw_file.exists():
                forward_work[i] = np.loadtxt(fw_file)[:, 1]
            
            # Backward work from next window
            bw_file = window_dirs[i + 1] / "work_backward.dat"
            if bw_file.exists():
                backward_work[i] = np.loadtxt(bw_file)[:, 1]
        
        return forward_work, backward_work
    
    def run(self, window_dirs: List[Path]) -> Dict:
        """Run BAR calculation."""
        logger.info("Running Bennett Acceptance Ratio calculation")
        
        forward_work, backward_work = self.parse_work_files(window_dirs)
        
        delta_g_values = []
        variances = []
        
        for i in forward_work.keys():
            if i in backward_work:
                w_f = forward_work[i]
                w_r = backward_work[i]
                
                delta_g = self.solve_bar(w_f, w_r)
                variance = self.compute_bar_variance(w_f, w_r, delta_g, self.config.temperature)
                
                delta_g_values.append(delta_g)
                variances.append(variance)
        
        self.delta_g = np.sum(delta_g_values)
        self.delta_g_error = np.sqrt(np.sum(variances))
        
        logger.info(f"BAR ΔG = {self.delta_g:.4f} ± {self.delta_g_error:.4f} kcal/mol")
        
        return {
            'delta_g': self.delta_g,
            'error': self.delta_g_error,
            'window_contributions': delta_g_values,
            'converged': self.delta_g_error < 0.5
        }


class WHAM:
    """Weighted Histogram Analysis Method for PMF reconstruction."""
    
    def __init__(self, config: Optional[WHAMConfig] = None):
        self.config = config or WHAMConfig()
        self.pmf: Optional[np.ndarray] = None
        self.bin_centers: Optional[np.ndarray] = None
        self.converged: bool = False
        self.iterations: int = 0
    
    def setup_bins(self, cv_data: List[np.ndarray]) -> np.ndarray:
        """Setup histogram bins based on data range."""
        all_data = np.concatenate(cv_data)
        cv_min = np.min(all_data) - self.config.padding
        cv_max = np.max(all_data) + self.config.padding
        
        if self.config.bin_width is None:
            # Use Freedman-Diaconis rule
            iqr = np.percentile(all_data, 75) - np.percentile(all_data, 25)
            n = len(all_data)
            bin_width = 2 * iqr / (n ** (1/3))
            n_bins = max(int((cv_max - cv_min) / bin_width), 10)
        else:
            n_bins = int((cv_max - cv_min) / self.config.bin_width)
        
        bins = np.linspace(cv_min, cv_max, n_bins)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return bins
    
    def compute_histograms(self, cv_data: List[np.ndarray], bins: np.ndarray) -> np.ndarray:
        """Compute histograms for each window."""
        n_windows = len(cv_data)
        n_bins = len(bins) - 1
        histograms = np.zeros((n_windows, n_bins))
        
        for i, data in enumerate(cv_data):
            hist, _ = np.histogram(data, bins=bins)
            histograms[i, :] = hist
        
        return histograms
    
    def compute_bias_potential(self, bin_centers: np.ndarray, 
                              window_centers: List[float],
                              kappas: List[float]) -> np.ndarray:
        """Compute harmonic bias potential for each window at each bin."""
        n_windows = len(window_centers)
        n_bins = len(bin_centers)
        
        # bias[i,j] = 0.5 * kappa[i] * (bin[j] - center[i])^2
        centers = np.array(window_centers).reshape(-1, 1)
        kappas = np.array(kappas).reshape(-1, 1)
        bins = bin_centers.reshape(1, -1)
        
        bias = 0.5 * kappas * (bins - centers) ** 2
        
        return bias
    
    def wham_iteration(self, histograms: np.ndarray, bias: np.ndarray,
                      f_k: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Single WHAM iteration.
        
        Args:
            histograms: N_windows x N_bins array of counts
            bias: N_windows x N_bins array of bias potentials
            f_k: Current free energy estimates for windows
            beta: 1/(kT)
        
        Returns:
            Updated pmf and f_k
        """
        n_windows, n_bins = histograms.shape
        
        # Compute unbiased probabilities
        # P(x) = Σ_k N_k(x) / Σ_k N_k exp[β(F_k - U_k(x))]
        
        denominator = np.zeros(n_bins)
        for k in range(n_windows):
            denominator += histograms[k, :] * np.exp(beta * (f_k[k] - bias[k, :]))
        
        # Avoid division by zero
        denominator = np.maximum(denominator, 1e-300)
        pmf = -np.log(denominator) / beta
        
        # Normalize PMF
        pmf = pmf - np.min(pmf)
        
        # Update f_k
        f_k_new = np.zeros(n_windows)
        for k in range(n_windows):
            f_k_new[k] = -np.log(np.sum(np.exp(-beta * bias[k, :]) * denominator)) / beta
        
        # Normalize f_k
        f_k_new = f_k_new - f_k_new[0]
        
        return pmf, f_k_new
    
    def compute_pmf(self, window_data: List[Dict]) -> Dict:
        """Compute PMF from umbrella sampling window data.
        
        Args:
            window_data: List of dictionaries containing:
                - 'cv_values': np.ndarray of CV samples
                - 'center': float, window center
                - 'kappa': float, force constant
        
        Returns:
            Dictionary with PMF and metadata
        """
        logger.info("Running WHAM analysis")
        
        # Extract data
        cv_data = [w['cv_values'] for w in window_data]
        window_centers = [w['center'] for w in window_data]
        kappas = [w.get('kappa', self.config.temperature) for w in window_data]
        
        # Setup bins
        bins = self.setup_bins(cv_data)
        bin_centers = self.bin_centers
        
        # Compute histograms
        histograms = self.compute_histograms(cv_data, bins)
        
        # Compute bias potentials
        bias = self.compute_bias_potential(bin_centers, window_centers, kappas)
        
        # WHAM iteration
        beta = 1.0 / (0.001987204 * self.config.temperature)
        n_windows = len(window_data)
        
        f_k = np.zeros(n_windows)
        pmf = np.zeros(len(bin_centers))
        
        for iteration in range(self.config.max_iterations):
            pmf_new, f_k_new = self.wham_iteration(histograms, bias, f_k, beta)
            
            # Check convergence
            delta_f = np.max(np.abs(f_k_new - f_k))
            delta_pmf = np.max(np.abs(pmf_new - pmf))
            
            f_k = f_k_new
            pmf = pmf_new
            
            if delta_pmf < self.config.tolerance:
                self.converged = True
                self.iterations = iteration + 1
                logger.info(f"WHAM converged in {iteration + 1} iterations")
                break
        else:
            logger.warning(f"WHAM did not converge in {self.config.max_iterations} iterations")
            self.iterations = self.config.max_iterations
        
        self.pmf = pmf
        
        return {
            'pmf': pmf,
            'bin_centers': bin_centers,
            'window_free_energies': f_k,
            'converged': self.converged,
            'iterations': self.iterations
        }
    
    def estimate_error(self, window_data: List[Dict], n_bootstrap: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate PMF error using bootstrap."""
        bootstrap_pmfs = []
        
        for _ in range(n_bootstrap):
            # Resample windows with replacement
            n_windows = len(window_data)
            indices = np.random.choice(n_windows, size=n_windows, replace=True)
            resampled_data = [window_data[i] for i in indices]
            
            # Compute PMF
            result = self.compute_pmf(resampled_data)
            bootstrap_pmfs.append(result['pmf'])
        
        bootstrap_pmfs = np.array(bootstrap_pmfs)
        mean_pmf = np.mean(bootstrap_pmfs, axis=0)
        std_pmf = np.std(bootstrap_pmfs, axis=0)
        
        return mean_pmf, std_pmf
    
    def save_pmf(self, filename: str):
        """Save PMF to file."""
        if self.pmf is None or self.bin_centers is None:
            raise ValueError("PMF not computed yet")
        
        data = np.column_stack([self.bin_centers, self.pmf])
        np.savetxt(filename, data, header="CV PMF[kcal/mol]", comments='# ')
        logger.info(f"PMF saved to {filename}")


class MBAR:
    """Multistate Bennett Acceptance Ratio."""
    
    def __init__(self, temperature: float = 300.0):
        self.temperature = temperature
        self.u_kn = None  # Reduced potentials
        self.N_k = None   # Samples per state
        self.f_k = None   # Free energies
    
    def compute_reduced_potentials(self, energies: np.ndarray, 
                                   lambdas: List[float]) -> np.ndarray:
        """Compute reduced potential energies u_kn."""
        beta = 1.0 / (0.001987204 * self.temperature)
        
        # energies[n_states, n_samples]
        # u_kn[k, n] = beta * U_k(x_n)
        
        n_states, n_samples = energies.shape
        u_kn = np.zeros((n_states, n_samples * n_states))
        
        # TODO: Complete implementation
        
        return beta * energies
    
    def solve_mbar(self, u_kn: np.ndarray, N_k: np.ndarray,
                   tolerance: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
        """Solve MBAR self-consistent equations."""
        K = len(N_k)  # Number of states
        
        # Initialize free energies
        f_k = np.zeros(K)
        
        for iteration in range(max_iter):
            f_k_new = self._mbar_iteration(u_kn, N_k, f_k)
            
            if np.max(np.abs(f_k_new - f_k)) < tolerance:
                logger.info(f"MBAR converged in {iteration} iterations")
                return f_k_new
            
            f_k = f_k_new
        
        logger.warning("MBAR did not converge")
        return f_k
    
    def _mbar_iteration(self, u_kn: np.ndarray, N_k: np.ndarray,
                       f_k: np.ndarray) -> np.ndarray:
        """Single MBAR iteration."""
        # Log-sum-exp for numerical stability
        log_numerator = logsumexp(-u_kn + f_k[:, np.newaxis], axis=0, b=N_k[:, np.newaxis])
        
        f_k_new = np.zeros_like(f_k)
        for k in range(len(N_k)):
            f_k_new[k] = -logsumexp(-u_kn[k, :] - log_numerator)
        
        # Normalize
        f_k_new = f_k_new - f_k_new[0]
        
        return f_k_new
    
    def compute_free_energy_differences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute free energy differences and uncertainties."""
        if self.f_k is None:
            raise ValueError("Run solve_mbar first")
        
        # Differences matrix
        delta_f_ij = self.f_k[:, np.newaxis] - self.f_k[np.newaxis, :]
        
        # TODO: Compute covariance matrix for uncertainties
        
        return delta_f_ij, np.zeros_like(delta_f_ij)


class FreeEnergyAnalyzer:
    """Analyze and compare free energy results."""
    
    @staticmethod
    def parse_alchemical_log(log_file: Path) -> pd.DataFrame:
        """Parse LAMMPS or other alchemical simulation log."""
        # Placeholder implementation
        return pd.read_csv(log_file, delim_whitespace=True, comment='#')
    
    @staticmethod
    def check_overlap(forward_work: np.ndarray, backward_work: np.ndarray,
                     threshold: float = 0.5) -> bool:
        """Check if work distributions have sufficient overlap."""
        # Compute histograms
        bins = np.linspace(min(forward_work.min(), backward_work.min()),
                          max(forward_work.max(), backward_work.max()), 50)
        
        hist_f, _ = np.histogram(forward_work, bins=bins, density=True)
        hist_r, _ = np.histogram(backward_work, bins=bins, density=True)
        
        # Compute overlap integral
        overlap = np.sum(np.minimum(hist_f, hist_r)) * (bins[1] - bins[0])
        
        return overlap > threshold
    
    @staticmethod
    def plot_work_distributions(forward_work: np.ndarray, backward_work: np.ndarray,
                               output_file: Optional[str] = None):
        """Plot work distributions for FEP analysis."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.hist(forward_work, bins=30, alpha=0.5, label='Forward', density=True)
            ax.hist(-backward_work, bins=30, alpha=0.5, label='-Backward', density=True)
            
            ax.set_xlabel('Work (kcal/mol)')
            ax.set_ylabel('Probability Density')
            ax.set_title('Work Distributions')
            ax.legend()
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    @staticmethod
    def plot_pmf(cv_grid: np.ndarray, pmf: np.ndarray, 
                error: Optional[np.ndarray] = None,
                output_file: Optional[str] = None):
        """Plot potential of mean force."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(cv_grid, pmf, 'b-', linewidth=2, label='PMF')
            
            if error is not None:
                ax.fill_between(cv_grid, pmf - error, pmf + error, 
                               alpha=0.3, color='blue')
            
            ax.set_xlabel('Collective Variable')
            ax.set_ylabel('Free Energy (kcal/mol)')
            ax.set_title('Potential of Mean Force')
            ax.legend()
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            logger.warning("matplotlib not available for plotting")


def compute_solvation_free_energy(solute_atoms: 'Atoms', 
                                  solvent_box: 'Atoms',
                                  method: str = 'bar',
                                  n_windows: int = 20) -> Dict:
    """Compute solvation free energy using alchemical method.
    
    Args:
        solute_atoms: Solute configuration
        solvent_box: Solvent box with/without solute
        method: 'fep', 'ti', or 'bar'
        n_windows: Number of λ windows
    
    Returns:
        Dictionary with free energy results
    """
    # Setup lambda schedule for decoupling
    lambdas = np.linspace(0, 1, n_windows)
    
    # This is a high-level wrapper that would:
    # 1. Generate alchemical topology
    # 2. Run simulations at each λ
    # 3. Analyze results
    
    logger.info(f"Computing solvation free energy using {method.upper()}")
    
    # Placeholder return
    return {
        'delta_g_solvation': 0.0,
        'error': 0.0,
        'method': method,
        'converged': True
    }


def compute_binding_free_energy(complex_system: 'Atoms',
                                receptor: 'Atoms',
                                ligand: 'Atoms',
                                method: str = 'bar') -> Dict:
    """Compute binding free energy using double decoupling.
    
    Args:
        complex_system: Receptor-ligand complex
        receptor: Receptor alone
        ligand: Ligand alone
        method: Calculation method
    
    Returns:
        Dictionary with binding free energy results
    """
    # ΔG_bind = ΔG_complex - ΔG_receptor - ΔG_ligand
    # + restraints corrections
    
    logger.info(f"Computing binding free energy using {method.upper()}")
    
    return {
        'delta_g_binding': 0.0,
        'error': 0.0,
        'method': method
    }


# Export public API
__all__ = [
    'FEPConfig',
    'TIConfig',
    'BARConfig',
    'WHAMConfig',
    'FreeEnergyPerturbation',
    'ThermodynamicIntegration',
    'BennettAcceptanceRatio',
    'WHAM',
    'MBAR',
    'FreeEnergyAnalyzer',
    'compute_solvation_free_energy',
    'compute_binding_free_energy'
]
