#!/usr/bin/env python3
"""
Time Correlation Analysis
=========================

Time correlation functions for transport properties:
- Velocity autocorrelation (VAC)
- Stress autocorrelation (SAC) for viscosity
- Dipole autocorrelation for dielectric properties
- Shear stress correlations

References:
- Green-Kubo relations
- Einstein relations
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
import logging
from scipy import integrate, signal, fftpack
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation function analysis.
    
    Attributes:
        max_lag: Maximum lag time for correlation (in steps)
        window_function: 'hann', 'hamming', 'blackman', or None
        normalize: Normalize correlation functions
        block_size: Size of blocks for error estimation
        n_blocks: Number of blocks for averaging
    """
    max_lag: int = 10000
    window_function: Optional[str] = 'hann'
    normalize: bool = True
    block_size: int = 1000
    n_blocks: int = 10


class TimeCorrelationAnalyzer:
    """Compute time correlation functions."""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
    
    def compute_autocorrelation_fft(self, signal_data: np.ndarray,
                                   max_lag: Optional[int] = None) -> np.ndarray:
        """Compute autocorrelation using FFT for efficiency.
        
        Uses Wiener-Khinchin theorem: C(t) = IFFT(|FFT(A)|²)
        """
        if max_lag is None:
            max_lag = self.config.max_lag
        
        n = len(signal_data)
        
        # Zero-pad to next power of 2
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        
        # FFT
        f = fftpack.fft(signal_data, n=n_fft)
        
        # Power spectrum
        power = np.abs(f) ** 2
        
        # IFFT to get autocorrelation
        corr = fftpack.ifft(power).real
        
        # Normalize
        corr = corr[:min(max_lag, n)] / np.arange(n, n - min(max_lag, n), -1)
        
        if self.config.normalize:
            corr = corr / corr[0]
        
        return corr
    
    def compute_autocorrelation_direct(self, signal_data: np.ndarray,
                                      max_lag: Optional[int] = None) -> np.ndarray:
        """Compute autocorrelation using direct method."""
        if max_lag is None:
            max_lag = self.config.max_lag
        
        n = len(signal_data)
        max_lag = min(max_lag, n // 2)
        
        corr = np.zeros(max_lag)
        
        for lag in range(max_lag):
            if lag == 0:
                corr[lag] = np.mean(signal_data ** 2)
            else:
                corr[lag] = np.mean(signal_data[:-lag] * signal_data[lag:])
        
        if self.config.normalize:
            corr = corr / corr[0]
        
        return corr
    
    def apply_window(self, corr: np.ndarray) -> np.ndarray:
        """Apply window function to correlation."""
        if self.config.window_function is None:
            return corr
        
        n = len(corr)
        
        if self.config.window_function == 'hann':
            window = np.hanning(2 * n)[:n]
        elif self.config.window_function == 'hamming':
            window = np.hamming(2 * n)[:n]
        elif self.config.window_function == 'blackman':
            window = np.blackman(2 * n)[:n]
        else:
            return corr
        
        return corr * window


class VelocityAutocorrelation:
    """Velocity autocorrelation function and diffusion."""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.analyzer = TimeCorrelationAnalyzer(config)
    
    def compute_vacf(self, velocities: np.ndarray,
                    max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity autocorrelation function.
        
        Args:
            velocities: (n_frames, n_atoms, 3) array
        
        Returns:
            (lag_times, vacf)
        """
        if max_lag is None:
            max_lag = self.config.max_lag
        
        n_frames = velocities.shape[0]
        max_lag = min(max_lag, n_frames // 2)
        
        # Average over atoms and dimensions
        vacf = np.zeros(max_lag)
        
        for atom in range(velocities.shape[1]):
            for dim in range(3):
                v = velocities[:, atom, dim]
                vacf += self.analyzer.compute_autocorrelation_fft(v, max_lag)
        
        vacf /= (velocities.shape[1] * 3)
        
        lag_times = np.arange(max_lag)
        
        return lag_times, vacf
    
    def compute_diffusion_coefficient(self, vacf: np.ndarray,
                                     timestep: float = 1.0,
                                     method: str = 'green-kubo') -> float:
        """Compute diffusion coefficient from VACF.
        
        D = (1/3) ∫₀^∞ ⟨v(0)·v(t)⟩ dt
        """
        if method == 'green-kubo':
            # Integrate VACF
            D = integrate.trapezoid(vacf, dx=timestep) / 3.0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return D
    
    def compute_power_spectrum(self, vacf: np.ndarray,
                              timestep: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute vibrational density of states from VACF."""
        # Apply window
        vacf_windowed = self.analyzer.apply_window(vacf)
        
        # FFT
        spectrum = np.abs(fftpack.fft(vacf_windowed))
        frequencies = fftpack.fftfreq(len(vacf), d=timestep)
        
        # Keep positive frequencies
        mask = frequencies >= 0
        
        return frequencies[mask], spectrum[mask]
    
    def analyze_vibrational_modes(self, vacf: np.ndarray,
                                 timestep: float = 1.0,
                                 peak_threshold: float = 0.1) -> Dict:
        """Identify vibrational modes from VACF spectrum."""
        frequencies, spectrum = self.compute_power_spectrum(vacf, timestep)
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(spectrum, height=peak_threshold * spectrum.max())
        
        peak_frequencies = frequencies[peaks]
        peak_intensities = spectrum[peaks]
        
        return {
            'frequencies': frequencies,
            'spectrum': spectrum,
            'peak_frequencies': peak_frequencies,
            'peak_intensities': peak_intensities,
            'n_peaks': len(peaks)
        }


class StressAutocorrelation:
    """Stress autocorrelation for viscosity calculation."""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.analyzer = TimeCorrelationAnalyzer(config)
    
    def compute_sacf(self, stress_tensor: np.ndarray,
                    max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute stress autocorrelation function.
        
        Args:
            stress_tensor: (n_frames, 6) array of [σxx, σyy, σzz, σxy, σxz, σyz]
        
        Returns:
            (lag_times, sacf)
        """
        if max_lag is None:
            max_lag = self.config.max_lag
        
        # Use off-diagonal components for shear viscosity
        # sacf = ⟨σxy(0)σxy(t)⟩ + ⟨σxz(0)σxz(t)⟩ + ⟨σyz(0)σyz(t)⟩
        
        sacf = np.zeros(max_lag)
        
        # Off-diagonal components are indices 3, 4, 5
        for i in [3, 4, 5]:
            component = stress_tensor[:, i]
            sacf += self.analyzer.compute_autocorrelation_fft(component, max_lag)
        
        sacf /= 3.0
        
        lag_times = np.arange(max_lag)
        
        return lag_times, sacf
    
    def compute_viscosity(self, sacf: np.ndarray,
                         volume: float,
                         temperature: float,
                         timestep: float = 1.0) -> float:
        """Compute shear viscosity from SACF using Green-Kubo.
        
        η = (V / k_B T) ∫₀^∞ ⟨σxy(0)σxy(t)⟩ dt
        """
        kB = 8.617e-5  # eV/K
        
        # Integrate SACF
        integral = integrate.trapezoid(sacf, dx=timestep)
        
        # Viscosity (convert to appropriate units)
        eta = (volume * integral) / (kB * temperature)
        
        # Convert to Pa·s (rough conversion from metal units)
        eta_pas = eta * 1.602e-1  # Conversion factor
        
        return eta_pas
    
    def estimate_viscosity_error(self, stress_tensor: np.ndarray,
                                volume: float,
                                temperature: float,
                                timestep: float = 1.0) -> Tuple[float, float]:
        """Estimate viscosity error using block averaging."""
        n_frames = len(stress_tensor)
        block_size = self.config.block_size
        n_blocks = min(self.config.n_blocks, n_frames // block_size)
        
        viscosities = []
        
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            
            if end > n_frames:
                break
            
            _, sacf = self.compute_sacf(stress_tensor[start:end])
            eta = self.compute_viscosity(sacf, volume, temperature, timestep)
            viscosities.append(eta)
        
        mean_eta = np.mean(viscosities)
        std_eta = np.std(viscosities, ddof=1) / np.sqrt(len(viscosities))
        
        return mean_eta, std_eta


class DipoleAutocorrelation:
    """Dipole autocorrelation for dielectric properties."""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.analyzer = TimeCorrelationAnalyzer(config)
    
    def compute_dacf(self, dipoles: np.ndarray,
                    max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dipole autocorrelation function.
        
        Args:
            dipoles: (n_frames, 3) array of dipole moments
        
        Returns:
            (lag_times, dacf)
        """
        if max_lag is None:
            max_lag = self.config.max_lag
        
        # Compute correlation for each component and average
        dacf = np.zeros(max_lag)
        
        for dim in range(3):
            dacf += self.analyzer.compute_autocorrelation_fft(dipoles[:, dim], max_lag)
        
        dacf /= 3.0
        
        lag_times = np.arange(max_lag)
        
        return lag_times, dacf
    
    def compute_dielectric_constant(self, dacf: np.ndarray,
                                   volume: float,
                                   temperature: float,
                                   dipole_magnitude: float,
                                   timestep: float = 1.0) -> float:
        """Compute dielectric constant from DACF.
        
        Uses fluctuation formula:
        ε = 1 + (4π / 3VkT) ⟨M²⟩
        """
        kB = 8.617e-5  # eV/K
        
        # Long-time limit of DACF gives ⟨M²⟩/3
        m_squared = 3 * np.mean(dacf[-len(dacf)//10:]) * dipole_magnitude ** 2
        
        epsilon = 1 + (4 * np.pi * m_squared) / (3 * volume * kB * temperature)
        
        return epsilon
    
    def compute_relaxation_time(self, dacf: np.ndarray,
                               timestep: float = 1.0) -> float:
        """Compute dielectric relaxation time."""
        # Integral of normalized DACF
        if dacf[0] > 0:
            normalized_dacf = dacf / dacf[0]
        else:
            return 0.0
        
        tau = integrate.trapezoid(normalized_dacf, dx=timestep)
        
        return tau


class IntermediateScattering:
    """Intermediate scattering function analysis."""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
    
    def compute_fs(self, positions: np.ndarray,
                  q_vector: np.ndarray,
                  max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute self-intermediate scattering function.
        
        F_s(q, t) = (1/N) Σᵢ ⟨exp[iq·(rᵢ(t) - rᵢ(0))]⟩
        """
        if max_lag is None:
            max_lag = self.config.max_lag
        
        n_frames, n_atoms, _ = positions.shape
        max_lag = min(max_lag, n_frames - 1)
        
        fs = np.zeros(max_lag, dtype=complex)
        
        for lag in range(max_lag):
            for t0 in range(0, n_frames - lag, max(1, lag)):
                dr = positions[t0 + lag] - positions[t0]
                phase = np.dot(dr, q_vector)
                fs[lag] += np.mean(np.exp(1j * phase))
            
            fs[lag] /= max(1, (n_frames - lag) // max(1, lag))
        
        lag_times = np.arange(max_lag)
        
        return lag_times, np.abs(fs)
    
    def compute_alpha_relaxation(self, fs: np.ndarray,
                                lag_times: np.ndarray,
                                q_value: float) -> Dict:
        """Analyze alpha relaxation from F_s(q,t)."""
        # Fit to Kohlrausch-Williams-Watts function
        # F_s(t) = A exp[-(t/τ)^β]
        
        def kww(t, A, tau, beta):
            return A * np.exp(-(t / tau) ** beta)
        
        try:
            popt, _ = curve_fit(kww, lag_times, fs, 
                               p0=[fs[0], len(fs)/10, 0.5],
                               bounds=([0, 0, 0], [1, np.inf, 1]))
            
            A, tau, beta = popt
            
            return {
                'amplitude': A,
                'relaxation_time': tau,
                'kww_exponent': beta,
                'fitted': True
            }
        except:
            return {
                'amplitude': fs[0],
                'relaxation_time': np.nan,
                'kww_exponent': np.nan,
                'fitted': False
            }


class CorrelationAnalysisSuite:
    """Unified correlation analysis interface."""
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self.vacf_analyzer = VelocityAutocorrelation(self.config)
        self.sacf_analyzer = StressAutocorrelation(self.config)
        self.dacf_analyzer = DipoleAutocorrelation(self.config)
        self.fs_analyzer = IntermediateScattering(self.config)
    
    def full_analysis(self, trajectory_data: Dict[str, Any]) -> Dict:
        """Perform complete correlation analysis.
        
        Args:
            trajectory_data: Dictionary containing:
                - 'velocities': (n_frames, n_atoms, 3)
                - 'stress': (n_frames, 6)
                - 'positions': (n_frames, n_atoms, 3)
                - 'volume': float
                - 'temperature': float
                - 'timestep': float
        """
        results = {}
        
        # VACF analysis
        if 'velocities' in trajectory_data:
            lag, vacf = self.vacf_analyzer.compute_vacf(trajectory_data['velocities'])
            D = self.vacf_analyzer.compute_diffusion_coefficient(
                vacf, trajectory_data.get('timestep', 1.0)
            )
            
            vibrational = self.vacf_analyzer.analyze_vibrational_modes(
                vacf, trajectory_data.get('timestep', 1.0)
            )
            
            results['vacf'] = {
                'lag_times': lag,
                'vacf': vacf,
                'diffusion_coefficient': D,
                'vibrational_spectrum': vibrational
            }
        
        # SACF analysis
        if 'stress' in trajectory_data:
            lag, sacf = self.sacf_analyzer.compute_sacf(trajectory_data['stress'])
            
            volume = trajectory_data.get('volume', 1.0)
            temperature = trajectory_data.get('temperature', 300.0)
            timestep = trajectory_data.get('timestep', 1.0)
            
            eta, eta_err = self.sacf_analyzer.estimate_viscosity_error(
                trajectory_data['stress'], volume, temperature, timestep
            )
            
            results['sacf'] = {
                'lag_times': lag,
                'sacf': sacf,
                'viscosity_pa_s': eta,
                'viscosity_error': eta_err
            }
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Save analysis results."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        logger.info(f"Correlation analysis saved to {output_file}")


# Export public API
__all__ = [
    'CorrelationConfig',
    'TimeCorrelationAnalyzer',
    'VelocityAutocorrelation',
    'StressAutocorrelation',
    'DipoleAutocorrelation',
    'IntermediateScattering',
    'CorrelationAnalysisSuite'
]
