"""
QMC Analysis and Utilities Module
==================================

Provides analysis tools for QMC calculations:
- Statistical analysis (blocking, reblocking)
- Error estimation
- Convergence analysis
- Data visualization utilities

Author: QMC Expert Module
Date: 2026-03-09
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import warnings


@dataclass
class StatisticalSummary:
    """Statistical summary of QMC data."""
    mean: float
    std_error: float
    variance: float
    autocorrelation_time: float
    effective_samples: int
    convergence_ratio: float


def blocking_analysis(data: np.ndarray, 
                     min_block_size: int = 1,
                     max_block_size: Optional[int] = None) -> Dict:
    """
    Perform blocking analysis for error estimation.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data (e.g., local energies)
    min_block_size : int
        Minimum block size
    max_block_size : int
        Maximum block size (default: len(data)//10)
        
    Returns:
    --------
    Dict with blocking analysis results
    """
    n = len(data)
    if max_block_size is None:
        max_block_size = n // 10
    
    block_sizes = []
    std_errors = []
    variances = []
    
    for block_size in range(min_block_size, max_block_size + 1):
        n_blocks = n // block_size
        if n_blocks < 2:
            break
        
        # Compute block means
        blocks = data[:n_blocks * block_size].reshape(n_blocks, block_size)
        block_means = np.mean(blocks, axis=1)
        
        # Block variance
        block_var = np.var(block_means, ddof=1)
        
        # Standard error
        std_error = np.sqrt(block_var / n_blocks)
        
        block_sizes.append(block_size)
        std_errors.append(std_error)
        variances.append(block_var)
    
    # Find plateau
    if len(std_errors) > 5:
        # Use last 30% as estimate of true error
        plateau_start = int(0.7 * len(std_errors))
        true_error = np.mean(std_errors[plateau_start:])
    else:
        true_error = std_errors[-1] if std_errors else np.std(data) / np.sqrt(n)
    
    return {
        'block_sizes': block_sizes,
        'std_errors': std_errors,
        'variances': variances,
        'recommended_error': true_error,
        'naive_error': np.std(data) / np.sqrt(n)
    }


def reblocking_analysis(data: np.ndarray, 
                       n_reblocks: int = 10) -> StatisticalSummary:
    """
    Perform reblocking analysis (Flyvbjerg-Petersen method).
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    n_reblocks : int
        Number of reblocking iterations
        
    Returns:
    --------
    StatisticalSummary object
    """
    n = len(data)
    
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    
    # Compute autocorrelation time
    c0 = variance
    
    # Reblocking
    current_data = data.copy()
    block_variances = [variance]
    
    for _ in range(n_reblocks):
        if len(current_data) < 4:
            break
        
        # Combine adjacent pairs
        n_pairs = len(current_data) // 2
        current_data = 0.5 * (current_data[:2*n_pairs:2] + current_data[1:2*n_pairs:2])
        
        block_var = np.var(current_data, ddof=1)
        block_variances.append(block_var)
    
    # Estimate autocorrelation time
    if len(block_variances) > 1:
        # Find where variance saturates
        tau = 0.5 * (block_variances[-1] / block_variances[0] - 1)
        tau = max(tau, 0.5)
    else:
        tau = 0.5
    
    effective_samples = n / (2 * tau)
    std_error = np.sqrt(variance / effective_samples)
    
    # Convergence ratio
    convergence_ratio = block_variances[-1] / block_variances[0] if block_variances else 1.0
    
    return StatisticalSummary(
        mean=mean,
        std_error=std_error,
        variance=variance,
        autocorrelation_time=tau,
        effective_samples=int(effective_samples),
        convergence_ratio=convergence_ratio
    )


def compute_autocorrelation(data: np.ndarray, 
                           max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation function.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    max_lag : int
        Maximum lag to compute
        
    Returns:
    --------
        Array of autocorrelation values
    """
    n = len(data)
    if max_lag is None:
        max_lag = n // 4
    
    # Normalize data
    data_normalized = data - np.mean(data)
    
    # Compute autocorrelation
    autocorr = np.correlate(data_normalized, data_normalized, mode='full')
    autocorr = autocorr[n-1:n-1+max_lag]
    autocorr = autocorr / autocorr[0]
    
    return autocorr


def estimate_integrated_autocorrelation_time(data: np.ndarray,
                                              cutoff: float = 0.05) -> float:
    """
    Estimate integrated autocorrelation time.
    
    tau_int = 0.5 + sum_{t=1}^infty rho(t)
    
    where rho(t) is the normalized autocorrelation.
    """
    autocorr = compute_autocorrelation(data)
    
    # Find cutoff where autocorrelation becomes small
    significant = np.where(autocorr > cutoff)[0]
    if len(significant) > 0:
        max_lag = significant[-1] + 1
    else:
        max_lag = len(autocorr) // 4
    
    tau_int = 0.5 + np.sum(autocorr[1:max_lag])
    
    return max(tau_int, 0.5)


def jackknife_analysis(data: np.ndarray, 
                      statistic_fn: Optional[callable] = None) -> Dict:
    """
    Perform jackknife resampling for error estimation.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array
    statistic_fn : callable
        Function to compute statistic (default: np.mean)
        
    Returns:
    --------
    Dict with jackknife results
    """
    if statistic_fn is None:
        statistic_fn = np.mean
    
    n = len(data)
    
    # Full sample statistic
    full_stat = statistic_fn(data)
    
    # Jackknife samples
    jack_stats = []
    for i in range(n):
        jack_sample = np.delete(data, i)
        jack_stats.append(statistic_fn(jack_sample))
    
    jack_stats = np.array(jack_stats)
    
    # Jackknife variance
    jack_mean = np.mean(jack_stats)
    jack_var = (n - 1) / n * np.sum((jack_stats - jack_mean) ** 2)
    jack_error = np.sqrt(jack_var)
    
    # Bias estimate
    bias = (n - 1) * (jack_mean - full_stat)
    
    return {
        'full_statistic': full_stat,
        'jackknife_mean': jack_mean,
        'jackknife_error': jack_error,
        'bias': bias,
        'jackknife_samples': jack_stats
    }


def bootstrap_analysis(data: np.ndarray,
                      statistic_fn: Optional[callable] = None,
                      n_bootstrap: int = 1000,
                      confidence: float = 0.95) -> Dict:
    """
    Perform bootstrap resampling for error estimation.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array
    statistic_fn : callable
        Function to compute statistic
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level for intervals
        
    Returns:
    --------
    Dict with bootstrap results
    """
    if statistic_fn is None:
        statistic_fn = np.mean
    
    n = len(data)
    
    # Full sample
    full_stat = statistic_fn(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Statistics
    boot_mean = np.mean(bootstrap_stats)
    boot_std = np.std(bootstrap_stats, ddof=1)
    
    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    # Bias
    bias = boot_mean - full_stat
    
    return {
        'full_statistic': full_stat,
        'bootstrap_mean': boot_mean,
        'bootstrap_std': boot_std,
        'bias': bias,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence': confidence,
        'bootstrap_samples': bootstrap_stats
    }


def analyze_monte_carlo_data(energies: np.ndarray,
                             weights: Optional[np.ndarray] = None,
                             equilibration_fraction: float = 0.2) -> Dict:
    """
    Comprehensive analysis of Monte Carlo energy data.
    
    Parameters:
    -----------
    energies : np.ndarray
        Local energies from Monte Carlo
    weights : np.ndarray
        Walker weights (for weighted averages)
    equilibration_fraction : float
        Fraction of data to discard as equilibration
        
    Returns:
    --------
    Dict with comprehensive analysis
    """
    # Discard equilibration
    n_equil = int(equilibration_fraction * len(energies))
    data = energies[n_equil:]
    
    if weights is not None:
        w = weights[n_equil:]
        w = w / np.sum(w)
        mean = np.sum(w * data)
        variance = np.sum(w * (data - mean) ** 2)
    else:
        mean = np.mean(data)
        variance = np.var(data, ddof=1)
        w = None
    
    # Blocking analysis
    blocking = blocking_analysis(data)
    
    # Reblocking
    reblock = reblocking_analysis(data)
    
    # Autocorrelation time
    tau = estimate_integrated_autocorrelation_time(data)
    
    # Bootstrap for comparison
    boot = bootstrap_analysis(data)
    
    return {
        'mean': mean,
        'variance': variance,
        'std_deviation': np.sqrt(variance),
        'std_error_blocking': blocking['recommended_error'],
        'std_error_reblocking': reblock.std_error,
        'std_error_bootstrap': boot['bootstrap_std'],
        'autocorrelation_time': tau,
        'effective_samples': len(data) / (2 * tau),
        'blocking_analysis': blocking,
        'reblocking_analysis': reblock,
        'bootstrap_analysis': boot,
        'n_total_samples': len(energies),
        'n_used_samples': len(data)
    }


def convergence_test(data: np.ndarray,
                    window_fraction: float = 0.1) -> Dict:
    """
    Test for convergence by comparing early and late averages.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    window_fraction : float
        Fraction of data for comparison windows
        
    Returns:
    --------
    Dict with convergence metrics
    """
    n = len(data)
    window_size = int(window_fraction * n)
    
    # Early and late averages
    early_mean = np.mean(data[:window_size])
    early_var = np.var(data[:window_size], ddof=1)
    
    late_mean = np.mean(data[-window_size:])
    late_var = np.var(data[-window_size:], ddof=1)
    
    # Difference
    diff = late_mean - early_mean
    diff_error = np.sqrt(early_var / window_size + late_var / window_size)
    
    # Cramer-Rao bound
    cramer_rao = diff / diff_error if diff_error > 0 else 0
    
    # Converged if difference is less than 2 sigma
    is_converged = abs(diff) < 2 * diff_error
    
    return {
        'early_mean': early_mean,
        'early_variance': early_var,
        'late_mean': late_mean,
        'late_variance': late_var,
        'difference': diff,
        'difference_error': diff_error,
        'cramer_rao': cramer_rao,
        'is_converged': is_converged,
        'convergence_criterion': 2.0
    }


def extrapolate_to_zero_time_step(energies: List[float],
                                  time_steps: List[float],
                                  orders: List[int] = [1, 2]) -> Dict:
    """
    Extrapolate energies to zero time step.
    
    Fits E(tau) = E_0 + a*tau + b*tau^2 + ...
    
    Parameters:
    -----------
    energies : List[float]
        Energies at different time steps
    time_steps : List[float]
        Time step values
    orders : List[int]
        Polynomial orders to try
        
    Returns:
    --------
    Dict with extrapolation results
    """
    energies = np.array(energies)
    taus = np.array(time_steps)
    
    results = {}
    
    for order in orders:
        if len(energies) <= order:
            continue
        
        # Polynomial fit
        coeffs = np.polyfit(taus, energies, order)
        e_extrapolated = coeffs[-1]  # Constant term
        
        # Standard error estimate
        fit_vals = np.polyval(coeffs, taus)
        residuals = energies - fit_vals
        mse = np.mean(residuals ** 2)
        
        results[f'order_{order}'] = {
            'energy_extrapolated': e_extrapolated,
            'fit_coefficients': coeffs.tolist(),
            'residual_mse': mse,
            'fit_energies': fit_vals.tolist()
        }
    
    return results


def save_analysis_results(results: Dict, output_file: str):
    """Save analysis results to JSON file."""
    # Convert numpy arrays to lists
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, StatisticalSummary):
            return convert_to_serializable(obj.__dict__)
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_analysis_results(input_file: str) -> Dict:
    """Load analysis results from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)
