"""
QMC Example: Statistical Analysis Demo
=======================================

Demonstrates statistical analysis tools for QMC data.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from qmc_analysis import (
    blocking_analysis,
    reblocking_analysis,
    compute_autocorrelation,
    estimate_integrated_autocorrelation_time,
    bootstrap_analysis,
    convergence_test,
    analyze_monte_carlo_data
)


def generate_correlated_data(n=10000, tau=5.0, true_mean=0.0, true_var=1.0):
    """
    Generate correlated Gaussian data with specified autocorrelation time.
    
    Parameters:
    -----------
    n : int
        Number of samples
    tau : float
        Autocorrelation time
    true_mean : float
        True mean of distribution
    true_var : float
        True variance of distribution
    """
    # AR(1) process: x_t = phi * x_{t-1} + epsilon
    phi = np.exp(-1/tau)
    
    data = np.zeros(n)
    data[0] = np.random.randn() * np.sqrt(true_var)
    
    for i in range(1, n):
        data[i] = phi * data[i-1] + np.random.randn() * np.sqrt(true_var * (1 - phi**2))
    
    return data + true_mean


def main():
    print("="*70)
    print("STATISTICAL ANALYSIS DEMO")
    print("="*70)
    print()
    
    # Generate test data
    print("Generating correlated test data...")
    print("  True mean: -1.174")
    print("  True autocorrelation time: 10.0")
    print()
    
    np.random.seed(42)
    true_mean = -1.174
    data = generate_correlated_data(n=10000, tau=10.0, true_mean=true_mean, true_var=0.1)
    
    # Basic statistics
    print("Basic Statistics")
    print("-" * 40)
    print(f"Sample mean:     {np.mean(data):.6f}")
    print(f"Sample std:      {np.std(data):.6f}")
    print(f"Naive error:     {np.std(data)/np.sqrt(len(data)):.6f}")
    print()
    
    # Blocking analysis
    print("Blocking Analysis")
    print("-" * 40)
    
    block_result = blocking_analysis(data, min_block_size=1, max_block_size=200)
    
    print(f"Recommended error: {block_result['recommended_error']:.6f}")
    print(f"Naive error:       {block_result['naive_error']:.6f}")
    print(f"Error inflation:   {block_result['recommended_error']/block_result['naive_error']:.2f}x")
    print()
    
    # Reblocking
    print("Reblocking Analysis")
    print("-" * 40)
    
    reblock = reblocking_analysis(data, n_reblocks=10)
    
    print(f"Mean:                  {reblock.mean:.6f}")
    print(f"Standard error:        {reblock.std_error:.6f}")
    print(f"Autocorrelation time:  {reblock.autocorrelation_time:.2f}")
    print(f"Effective samples:     {reblock.effective_samples}")
    print()
    
    # Autocorrelation
    print("Autocorrelation Function")
    print("-" * 40)
    
    autocorr = compute_autocorrelation(data, max_lag=100)
    tau_est = estimate_integrated_autocorrelation_time(data)
    
    print(f"Estimated tau_int: {tau_est:.2f}")
    print(f"First 10 values: {autocorr[:10]}")
    print()
    
    # Bootstrap
    print("Bootstrap Analysis")
    print("-" * 40)
    
    boot = bootstrap_analysis(data, n_bootstrap=1000, confidence=0.95)
    
    print(f"Bootstrap mean:  {boot['bootstrap_mean']:.6f}")
    print(f"Bootstrap std:   {boot['bootstrap_std']:.6f}")
    print(f"95% CI: [{boot['ci_lower']:.6f}, {boot['ci_upper']:.6f}]")
    print(f"Bias: {boot['bias']:.6f}")
    print()
    
    # Convergence test
    print("Convergence Test")
    print("-" * 40)
    
    conv = convergence_test(data, window_fraction=0.1)
    
    print(f"Early mean:  {conv['early_mean']:.6f}")
    print(f"Late mean:   {conv['late_mean']:.6f}")
    print(f"Difference:  {conv['difference']:.6f} ± {conv['difference_error']:.6f}")
    print(f"Converged:   {conv['is_converged']}")
    print()
    
    # Full analysis
    print("Comprehensive Analysis")
    print("-" * 40)
    
    analysis = analyze_monte_carlo_data(data, equilibration_fraction=0.1)
    
    print(f"Mean:                {analysis['mean']:.6f}")
    print(f"Variance:            {analysis['variance']:.6f}")
    print(f"Std (blocking):      {analysis['std_error_blocking']:.6f}")
    print(f"Std (reblocking):    {analysis['std_error_reblocking']:.6f}")
    print(f"Std (bootstrap):     {analysis['std_error_bootstrap']:.6f}")
    print(f"Autocorrelation:     {analysis['autocorrelation_time']:.2f}")
    print(f"Effective samples:   {analysis['effective_samples']:.0f}")
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print(f"True mean:       {true_mean:.6f}")
    print(f"Estimated mean:  {analysis['mean']:.6f}")
    print(f"Error estimate:  {analysis['std_error_blocking']:.6f}")
    print(f"Actual error:    {abs(analysis['mean'] - true_mean):.6f}")
    print()
    print(f"The actual error should be comparable to the error estimate")
    print(f"if the statistical analysis is correct.")
    print()


if __name__ == '__main__':
    main()
