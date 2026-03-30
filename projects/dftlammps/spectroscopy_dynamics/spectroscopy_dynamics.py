"""
Spectroscopy Dynamics Module
============================

Simulation and analysis of time-resolved spectroscopic signals:
- Ultrafast absorption spectroscopy
- Transient absorption (pump-probe)
- Time-resolved photoelectron spectroscopy (TRPES)

Integrates with non-adiabatic dynamics for complete description
of ultrafast processes.

Author: dftlammps development team
"""

import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import warnings

# Optional imports
try:
    import scipy
    from scipy import signal, integrate, interpolate, optimize
    from scipy.fft import fft, fftfreq, ifft
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class LaserPulse:
    """
    Represents a laser pulse for pump-probe spectroscopy.
    
    Attributes:
        wavelength: Central wavelength in nm
        energy: Photon energy in eV
        fwhm: Full width at half maximum in fs
        intensity: Peak intensity in W/cm²
        chirp: Linear chirp parameter (fs^-2)
        polarization: Polarization vector
    """
    
    wavelength: float = 800.0  # nm
    energy: float = None  # eV (calculated from wavelength)
    fwhm: float = 50.0  # fs
    intensity: float = 1e12  # W/cm²
    chirp: float = 0.0  # fs^-2
    polarization: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    
    def __post_init__(self):
        if self.energy is None:
            # E(eV) = 1240 / λ(nm)
            self.energy = 1240.0 / self.wavelength
        
        # Normalize polarization
        self.polarization = self.polarization / np.linalg.norm(self.polarization)
    
    def get_envelope(self, times: np.ndarray, t0: float = 0.0) -> np.ndarray:
        """
        Get pulse electric field envelope.
        
        Parameters
        ----------
        times : np.ndarray
            Time array in fs
        t0 : float
            Pulse center time in fs
            
        Returns
        -------
        np.ndarray : Electric field envelope
        """
        # Gaussian envelope
        sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))
        envelope = np.exp(-(times - t0)**2 / (2 * sigma**2))
        
        return envelope
    
    def get_electric_field(self, times: np.ndarray, t0: float = 0.0) -> np.ndarray:
        """
        Get time-dependent electric field including carrier wave.
        
        Parameters
        ----------
        times : np.ndarray
            Time array in fs
        t0 : float
            Pulse center time
            
        Returns
        -------
        np.ndarray : Electric field
        """
        envelope = self.get_envelope(times, t0)
        
        # Carrier frequency (fs^-1)
        omega = self.energy / 0.6582119  # Convert eV to fs^-1
        
        # Chirp
        chirp_phase = 0.5 * self.chirp * (times - t0)**2
        
        # Electric field
        E_field = envelope * np.cos(omega * (times - t0) + chirp_phase)
        
        return E_field
    
    def get_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Get pulse frequency spectrum.
        
        Parameters
        ----------
        frequencies : np.ndarray
            Frequency array in fs^-1
            
        Returns
        -------
        np.ndarray : Spectral intensity
        """
        # Fourier transform of Gaussian envelope
        sigma = self.fwhm / (2 * np.sqrt(2 * np.log(2)))
        omega0 = self.energy / 0.6582119
        
        spectrum = np.exp(-(frequencies - omega0)**2 * sigma**2 / 2)
        
        return spectrum


@dataclass
class ElectronicTransition:
    """
    Represents an electronic transition between states.
    
    Attributes:
        initial_state: Initial state index
        final_state: Final state index
        energy: Transition energy in eV
        dipole_moment: Transition dipole moment in Debye
        oscillator_strength: Oscillator strength
        linewidth: Homogeneous linewidth in eV
    """
    
    initial_state: int
    final_state: int
    energy: float  # eV
    dipole_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    oscillator_strength: float = 0.0
    linewidth: float = 0.1  # eV
    
    @property
    def transition_dipole_magnitude(self) -> float:
        """Get magnitude of transition dipole moment."""
        return np.linalg.norm(self.dipole_moment)
    
    def get_lineshape(self, energies: np.ndarray, 
                     lineshape_type: str = "gaussian") -> np.ndarray:
        """
        Calculate absorption/emission lineshape.
        
        Parameters
        ----------
        energies : np.ndarray
            Energy array in eV
        lineshape_type : str
            Type of lineshape ('gaussian', 'lorentzian')
            
        Returns
        -------
        np.ndarray : Lineshape
        """
        if lineshape_type == "gaussian":
            sigma = self.linewidth / (2 * np.sqrt(2 * np.log(2)))
            lineshape = np.exp(-(energies - self.energy)**2 / (2 * sigma**2))
        elif lineshape_type == "lorentzian":
            gamma = self.linewidth / 2
            lineshape = gamma**2 / ((energies - self.energy)**2 + gamma**2)
        else:
            raise ValueError(f"Unknown lineshape type: {lineshape_type}")
        
        return lineshape


class UltrafastAbsorption:
    """
    Simulator for ultrafast absorption spectroscopy.
    
    Calculates time-dependent absorption spectra including:
    - Ground state bleach
    - Stimulated emission
    - Excited state absorption
    - Coherent oscillations
    """
    
    def __init__(self, 
                 pump_pulse: Optional[LaserPulse] = None,
                 probe_pulse: Optional[LaserPulse] = None):
        self.pump_pulse = pump_pulse or LaserPulse()
        self.probe_pulse = probe_pulse or LaserPulse()
        
        self.transitions = []
        self.population_evolution = None
        
    def add_transition(self, transition: ElectronicTransition):
        """Add an electronic transition."""
        self.transitions.append(transition)
    
    def calculate_linear_absorption(self,
                                    energies: np.ndarray,
                                    temperature: float = 300.0) -> np.ndarray:
        """
        Calculate linear absorption spectrum.
        
        Parameters
        ----------
        energies : np.ndarray
            Energy array in eV
        temperature : float
            Temperature in K
            
        Returns
        -------
        np.ndarray : Absorption spectrum (molar extinction coefficient)
        """
        absorption = np.zeros_like(energies)
        
        for trans in self.transitions:
            if trans.initial_state == 0:  # Ground state absorption
                # Thermal population
                kT = 8.617e-5 * temperature  # eV
                population = 1.0  # Ground state
                
                # Oscillator strength contribution
                f = trans.oscillator_strength
                
                # Lineshape
                lineshape = trans.get_lineshape(energies)
                
                # Add to absorption
                absorption += f * population * lineshape
        
        return absorption
    
    def calculate_transient_absorption(self,
                                       delay_times: np.ndarray,
                                       probe_energies: np.ndarray,
                                       state_populations: np.ndarray) -> np.ndarray:
        """
        Calculate transient absorption spectrum ΔA(ω, τ).
        
        ΔA = A_excited - A_ground
        
        Parameters
        ----------
        delay_times : np.ndarray
            Pump-probe delay times in fs
        probe_energies : np.ndarray
            Probe photon energies in eV
        state_populations : np.ndarray
            Population evolution [ntimes, nstates]
            
        Returns
        -------
        np.ndarray : Transient absorption [ntimes, nenergies]
        """
        n_delays = len(delay_times)
        n_energies = len(probe_energies)
        
        delta_A = np.zeros((n_delays, n_energies))
        
        # Ground state absorption
        ground_abs = self.calculate_linear_absorption(probe_energies)
        
        for i, t in enumerate(delay_times):
            # Find closest population state
            pop_idx = min(i, len(state_populations) - 1)
            populations = state_populations[pop_idx]
            
            # Calculate excited state absorption
            excited_abs = np.zeros(n_energies)
            
            for trans in self.transitions:
                # Ground state bleach (negative contribution)
                if trans.initial_state == 0 and len(populations) > trans.final_state:
                    bleach = populations[trans.final_state] * trans.oscillator_strength
                    lineshape = trans.get_lineshape(probe_energies)
                    excited_abs -= bleach * lineshape
                
                # Excited state absorption
                if trans.initial_state > 0 and trans.initial_state < len(populations):
                    esa = populations[trans.initial_state] * trans.oscillator_strength
                    lineshape = trans.get_lineshape(probe_energies)
                    excited_abs += esa * lineshape
                
                # Stimulated emission (negative at emission energy)
                if trans.final_state == 0 and trans.initial_state < len(populations):
                    se = populations[trans.initial_state] * trans.oscillator_strength
                    lineshape = trans.get_lineshape(probe_energies)
                    excited_abs -= se * lineshape * 0.5
            
            # Transient absorption
            delta_A[i] = excited_abs - ground_abs * (np.sum(populations[1:]) if len(populations) > 1 else 0)
        
        return delta_A
    
    def calculate_2d_spectrum(self,
                             pump_energies: np.ndarray,
                             probe_energies: np.ndarray,
                             delay_time: float,
                             state_populations: np.ndarray) -> np.ndarray:
        """
        Calculate 2D electronic spectrum.
        
        Parameters
        ----------
        pump_energies : np.ndarray
            Pump photon energies in eV
        probe_energies : np.ndarray
            Probe photon energies in eV
        delay_time : float
            Population time in fs
        state_populations : np.ndarray
            State populations
            
        Returns
        -------
        np.ndarray : 2D spectrum [npump, nprobe]
        """
        npump = len(pump_energies)
        nprobe = len(probe_energies)
        
        spectrum_2d = np.zeros((npump, nprobe))
        
        for i, E_pump in enumerate(pump_energies):
            # Calculate which states are pumped
            pumped_states = []
            for trans in self.transitions:
                if trans.initial_state == 0:
                    if abs(trans.energy - E_pump) < 0.2:  # Within bandwidth
                        pumped_states.append(trans.final_state)
            
            # Calculate probe spectrum for these populations
            pop_idx = min(int(delay_time / 10), len(state_populations) - 1)
            populations = state_populations[pop_idx]
            
            for j, E_probe in enumerate(probe_energies):
                signal = 0.0
                
                for trans in self.transitions:
                    if abs(trans.energy - E_probe) < 0.1:
                        if trans.initial_state == 0:
                            # Ground state bleach
                            signal -= populations[trans.final_state] * trans.oscillator_strength
                        elif trans.initial_state < len(populations):
                            # ESA or SE
                            signal += populations[trans.initial_state] * trans.oscillator_strength * 0.5
                
                spectrum_2d[i, j] = signal
        
        return spectrum_2d


class TimeResolvedPhotoelectronSpectroscopy:
    """
    Simulator for time-resolved photoelectron spectroscopy (TRPES).
    
    Calculates photoelectron spectra as a function of pump-probe delay.
    """
    
    def __init__(self,
                 ionization_potential: float = 6.0,
                 probe_photon_energy: float = 6.2):
        """
        Initialize TRPES simulator.
        
        Parameters
        ----------
        ionization_potential : float
            Ionization potential in eV
        probe_photon_energy : float
            Probe photon energy in eV
        """
        self.ionization_potential = ionization_potential
        self.probe_photon_energy = probe_photon_energy
        
        self.states = []
        self.ionization_cross_sections = {}
        
    def add_electronic_state(self, 
                            state_index: int,
                            energy: float,
                            ionization_cross_section: float = 1.0):
        """
        Add an electronic state.
        
        Parameters
        ----------
        state_index : int
            State index
        energy : float
            State energy relative to ground state in eV
        ionization_cross_section : float
            Relative ionization cross section
        """
        self.states.append({
            'index': state_index,
            'energy': energy,
            'cross_section': ionization_cross_section
        })
        self.ionization_cross_sections[state_index] = ionization_cross_section
    
    def calculate_kinetic_energy(self, state_energy: float) -> float:
        """
        Calculate photoelectron kinetic energy.
        
        E_kin = hν_probe - IP - E_state
        
        Parameters
        ----------
        state_energy : float
            Electronic state energy in eV
            
        Returns
        -------
        float : Kinetic energy in eV
        """
        return self.probe_photon_energy - self.ionization_potential - state_energy
    
    def calculate_trpes_spectrum(self,
                                 delay_times: np.ndarray,
                                 kinetic_energies: np.ndarray,
                                 state_populations: np.ndarray,
                                 energy_resolution: float = 0.1) -> np.ndarray:
        """
        Calculate time-resolved photoelectron spectrum.
        
        Parameters
        ----------
        delay_times : np.ndarray
            Pump-probe delay times in fs
        kinetic_energies : np.ndarray
            Photoelectron kinetic energies in eV
        state_populations : np.ndarray
            State population evolution [ntimes, nstates]
        energy_resolution : float
            Energy resolution in eV
            
        Returns
        -------
        np.ndarray : TRPES spectrum [ntimes, nenergies]
        """
        n_delays = len(delay_times)
        n_energies = len(kinetic_energies)
        
        spectrum = np.zeros((n_delays, n_energies))
        
        for i, t in enumerate(delay_times):
            pop_idx = min(i, len(state_populations) - 1)
            populations = state_populations[pop_idx]
            
            for state in self.states:
                state_idx = state['index']
                if state_idx >= len(populations):
                    continue
                
                pop = populations[state_idx]
                cross_section = state['cross_section']
                
                # Kinetic energy for this state
                E_kin = self.calculate_kinetic_energy(state['energy'])
                
                # Gaussian broadening
                sigma = energy_resolution / (2 * np.sqrt(2 * np.log(2)))
                intensity = pop * cross_section * np.exp(
                    -(kinetic_energies - E_kin)**2 / (2 * sigma**2)
                )
                
                spectrum[i] += intensity
        
        return spectrum
    
    def analyze_wavepacket_motion(self,
                                  trpes_data: np.ndarray,
                                  delay_times: np.ndarray,
                                  kinetic_energies: np.ndarray) -> Dict:
        """
        Analyze wavepacket motion from TRPES data.
        
        Parameters
        ----------
        trpes_data : np.ndarray
            TRPES spectrum [ntimes, nenergies]
        delay_times : np.ndarray
            Time array in fs
        kinetic_energies : np.ndarray
            Energy array in eV
            
        Returns
        -------
        Dict with wavepacket analysis
        """
        # Find peak positions as function of time
        peak_positions = np.zeros(len(delay_times))
        
        for i in range(len(delay_times)):
            spectrum = trpes_data[i]
            if np.max(spectrum) > 0:
                peak_positions[i] = kinetic_energies[np.argmax(spectrum)]
            else:
                peak_positions[i] = np.nan
        
        # Fit oscillations
        valid = ~np.isnan(peak_positions)
        if np.sum(valid) > 10 and SCIPY_AVAILABLE:
            # Try to fit damped oscillation
            def damped_oscillation(t, A, omega, gamma, phi, offset):
                return A * np.exp(-gamma * t) * np.cos(omega * t + phi) + offset
            
            try:
                popt, _ = optimize.curve_fit(
                    damped_oscillation, 
                    delay_times[valid], 
                    peak_positions[valid],
                    p0=[0.1, 0.05, 0.001, 0, np.mean(peak_positions[valid])]
                )
                
                oscillation_freq = popt[1] * 1000 / (2 * np.pi)  # THz
                damping_time = 1 / popt[2] if popt[2] > 0 else np.inf
                
                analysis = {
                    'peak_positions': peak_positions,
                    'oscillation_frequency_THz': oscillation_freq,
                    'damping_time_fs': damping_time,
                    'amplitude_eV': abs(popt[0]),
                    'fit_parameters': popt
                }
            except:
                analysis = {
                    'peak_positions': peak_positions,
                    'oscillation_frequency_THz': None,
                    'damping_time_fs': None
                }
        else:
            analysis = {
                'peak_positions': peak_positions,
                'oscillation_frequency_THz': None,
                'damping_time_fs': None
            }
        
        return analysis


class VibrationalCoherenceAnalysis:
    """
    Analysis of vibrational coherences in spectroscopic data.
    """
    
    def __init__(self):
        self.frequencies = None
        self.amplitudes = None
        
    def extract_coherences(self,
                          signal: np.ndarray,
                          times: np.ndarray,
                          window: Optional[np.ndarray] = None) -> Dict:
        """
        Extract vibrational coherences from time-domain signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Time-dependent signal (e.g., absorption change)
        times : np.ndarray
            Time array in fs
        window : np.ndarray, optional
            Window function for FFT
            
        Returns
        -------
        Dict with coherence analysis
        """
        dt = times[1] - times[0]
        
        # Detrend signal
        signal_detrended = signal - np.mean(signal)
        
        # Apply window
        if window is None:
            window = np.hanning(len(signal))
        
        signal_windowed = signal_detrended * window
        
        # FFT
        if SCIPY_AVAILABLE:
            # Frequency axis in THz
            freqs = fftfreq(len(signal), dt * 1e-15) * 1e-12
            fft_values = fft(signal_windowed)
            
            # Only positive frequencies
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            amplitudes = np.abs(fft_values[pos_mask])
            
            # Find peaks
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(amplitudes, height=np.max(amplitudes) * 0.1)
            
            peak_freqs = freqs[peaks]
            peak_amps = amplitudes[peaks]
            
            # Sort by amplitude
            sort_idx = np.argsort(peak_amps)[::-1]
            
            coherence_data = {
                'frequencies_THz': freqs,
                'amplitudes': amplitudes,
                'peak_frequencies_THz': peak_freqs[sort_idx][:10],
                'peak_amplitudes': peak_amps[sort_idx][:10],
                'dominant_frequency_THz': peak_freqs[sort_idx[0]] if len(sort_idx) > 0 else None
            }
        else:
            coherence_data = {
                'frequencies_THz': None,
                'amplitudes': None,
                'peak_frequencies_THz': [],
                'peak_amplitudes': [],
                'dominant_frequency_THz': None
            }
        
        return coherence_data
    
    def wavelet_analysis(self,
                        signal: np.ndarray,
                        times: np.ndarray,
                        frequencies: np.ndarray) -> np.ndarray:
        """
        Perform continuous wavelet transform for time-frequency analysis.
        
        Parameters
        ----------
        signal : np.ndarray
            Time-domain signal
        times : np.ndarray
            Time array in fs
        frequencies : np.ndarray
            Frequencies to analyze in THz
            
        Returns
        -------
        np.ndarray : Wavelet coefficients [nfreqs, ntimes]
        """
        if not SCIPY_AVAILABLE:
            return np.zeros((len(frequencies), len(times)))
        
        from scipy.signal import morlet2, cwt
        
        dt = times[1] - times[0]
        widths = 1.0 / (frequencies * 1e12 * dt * 1e-15)  # Scale factors
        
        wavelet = lambda M, s: morlet2(M, s, w=5)
        cwt_matrix = cwt(signal, wavelet, widths)
        
        return np.abs(cwt_matrix)


class SpectroscopyDynamicsWorkflow:
    """
    Complete workflow for simulating time-resolved spectroscopy.
    """
    
    def __init__(self):
        self.ultrafast_abs = UltrafastAbsorption()
        self.trpes = None
        self.coherence_analyzer = VibrationalCoherenceAnalysis()
        
        self.pump_pulse = LaserPulse()
        self.probe_pulse = LaserPulse(wavelength=400.0, fwhm=20.0)
        
        self.results = {}
        
    def setup_pump_pulse(self, **kwargs):
        """Setup pump pulse parameters."""
        self.pump_pulse = LaserPulse(**kwargs)
        self.ultrafast_abs.pump_pulse = self.pump_pulse
    
    def setup_probe_pulse(self, **kwargs):
        """Setup probe pulse parameters."""
        self.probe_pulse = LaserPulse(**kwargs)
        self.ultrafast_abs.probe_pulse = self.probe_pulse
    
    def setup_trpes(self, ionization_potential: float, probe_energy: float):
        """Setup TRPES parameters."""
        self.trpes = TimeResolvedPhotoelectronSpectroscopy(
            ionization_potential, probe_energy
        )
    
    def run_transient_absorption(self,
                                  state_populations: np.ndarray,
                                  delay_times: np.ndarray,
                                  probe_energies: np.ndarray) -> np.ndarray:
        """
        Run transient absorption simulation.
        
        Parameters
        ----------
        state_populations : np.ndarray
            State populations [ntimes, nstates]
        delay_times : np.ndarray
            Pump-probe delay times in fs
        probe_energies : np.ndarray
            Probe photon energies in eV
            
        Returns
        -------
        np.ndarray : Transient absorption spectrum
        """
        delta_A = self.ultrafast_abs.calculate_transient_absorption(
            delay_times, probe_energies, state_populations
        )
        
        self.results['transient_absorption'] = {
            'delay_times': delay_times,
            'probe_energies': probe_energies,
            'delta_A': delta_A
        }
        
        return delta_A
    
    def run_trpes(self,
                 state_populations: np.ndarray,
                 delay_times: np.ndarray,
                 kinetic_energies: np.ndarray) -> np.ndarray:
        """
        Run TRPES simulation.
        
        Parameters
        ----------
        state_populations : np.ndarray
            State populations
        delay_times : np.ndarray
            Pump-probe delays in fs
        kinetic_energies : np.ndarray
            Kinetic energy range in eV
            
        Returns
        -------
        np.ndarray : TRPES spectrum
        """
        if self.trpes is None:
            raise ValueError("TRPES not setup. Call setup_trpes() first.")
        
        spectrum = self.trpes.calculate_trpes_spectrum(
            delay_times, kinetic_energies, state_populations
        )
        
        self.results['trpes'] = {
            'delay_times': delay_times,
            'kinetic_energies': kinetic_energies,
            'spectrum': spectrum
        }
        
        # Analyze wavepacket motion
        wavepacket_analysis = self.trpes.analyze_wavepacket_motion(
            spectrum, delay_times, kinetic_energies
        )
        self.results['wavepacket_analysis'] = wavepacket_analysis
        
        return spectrum
    
    def analyze_vibrational_coherences(self,
                                      signal: np.ndarray,
                                      times: np.ndarray) -> Dict:
        """
        Analyze vibrational coherences in spectroscopic signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Time-dependent signal at specific wavelength
        times : np.ndarray
            Time array
            
        Returns
        -------
        Dict with coherence analysis
        """
        coherence_data = self.coherence_analyzer.extract_coherences(signal, times)
        
        self.results['vibrational_coherences'] = coherence_data
        
        return coherence_data
    
    def visualize_spectra(self, output_dir: str = "./spectroscopy_results"):
        """Generate visualizations of spectroscopic data."""
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot transient absorption
        if 'transient_absorption' in self.results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            data = self.results['transient_absorption']
            delay_times = data['delay_times']
            probe_energies = data['probe_energies']
            delta_A = data['delta_A']
            
            # 2D plot
            ax = axes[0]
            E_grid, T_grid = np.meshgrid(probe_energies, delay_times)
            
            vmax = np.max(np.abs(delta_A))
            im = ax.contourf(E_grid, T_grid, delta_A, levels=50, 
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_xlabel('Probe Energy (eV)')
            ax.set_ylabel('Delay Time (fs)')
            ax.set_title('Transient Absorption')
            plt.colorbar(im, ax=ax, label='ΔA (a.u.)')
            
            # Time traces at selected energies
            ax = axes[1]
            selected_energies = [probe_energies[len(probe_energies)//4],
                               probe_energies[len(probe_energies)//2],
                               probe_energies[3*len(probe_energies)//4]]
            
            for E in selected_energies:
                idx = np.argmin(np.abs(probe_energies - E))
                ax.plot(delay_times, delta_A[:, idx], label=f'{E:.2f} eV')
            
            ax.set_xlabel('Delay Time (fs)')
            ax.set_ylabel('ΔA (a.u.)')
            ax.set_title('Time Traces')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/transient_absorption.png", dpi=150)
            plt.close()
        
        # Plot TRPES
        if 'trpes' in self.results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            data = self.results['trpes']
            delay_times = data['delay_times']
            kinetic_energies = data['kinetic_energies']
            spectrum = data['spectrum']
            
            # 2D plot
            ax = axes[0]
            E_grid, T_grid = np.meshgrid(kinetic_energies, delay_times)
            im = ax.contourf(E_grid, T_grid, spectrum, levels=50, cmap='hot')
            ax.set_xlabel('Kinetic Energy (eV)')
            ax.set_ylabel('Delay Time (fs)')
            ax.set_title('TRPES')
            plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
            
            # Wavepacket analysis
            ax = axes[1]
            if 'wavepacket_analysis' in self.results:
                analysis = self.results['wavepacket_analysis']
                peak_pos = analysis['peak_positions']
                valid = ~np.isnan(peak_pos)
                
                ax.plot(delay_times[valid], peak_pos[valid], 'bo-', label='Peak position')
                
                if analysis.get('oscillation_frequency_THz'):
                    ax.set_title(f"Wavepacket Motion ({analysis['oscillation_frequency_THz']:.1f} THz)")
                else:
                    ax.set_title('Wavepacket Motion')
                
                ax.set_xlabel('Delay Time (fs)')
                ax.set_ylabel('Kinetic Energy (eV)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/trpes.png", dpi=150)
            plt.close()
        
        # Plot vibrational coherences
        if 'vibrational_coherences' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data = self.results['vibrational_coherences']
            if data['frequencies_THz'] is not None:
                ax.plot(data['frequencies_THz'], data['amplitudes'], 'b-')
                
                # Mark peaks
                for freq, amp in zip(data['peak_frequencies_THz'][:5], 
                                    data['peak_amplitudes'][:5]):
                    ax.axvline(freq, color='r', linestyle='--', alpha=0.5)
                    ax.text(freq, amp, f'{freq:.1f} THz', rotation=90, 
                           ha='right', va='bottom')
                
                ax.set_xlabel('Frequency (THz)')
                ax.set_ylabel('Amplitude (a.u.)')
                ax.set_title('Vibrational Coherences')
                ax.set_xlim(0, 100)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/vibrational_coherences.png", dpi=150)
            plt.close()
        
        logger.info(f"Spectroscopy visualizations saved to {output_dir}")
    
    def generate_report(self) -> str:
        """Generate comprehensive spectroscopy report."""
        
        report = []
        report.append("=" * 60)
        report.append("SPECTROSCOPY DYNAMICS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        if 'transient_absorption' in self.results:
            report.append("TRANSIENT ABSORPTION")
            report.append("-" * 40)
            delta_A = self.results['transient_absorption']['delta_A']
            report.append(f"Signal range: {np.min(delta_A):.3f} to {np.max(delta_A):.3f}")
            report.append(f"Time window: {self.results['transient_absorption']['delay_times'][-1]:.0f} fs")
            report.append("")
        
        if 'trpes' in self.results:
            report.append("TIME-RESOLVED PHOTOELECTRON SPECTROSCOPY")
            report.append("-" * 40)
            if 'wavepacket_analysis' in self.results:
                analysis = self.results['wavepacket_analysis']
                if analysis.get('oscillation_frequency_THz'):
                    report.append(f"Oscillation frequency: {analysis['oscillation_frequency_THz']:.1f} THz")
                    report.append(f"Damping time: {analysis['damping_time_fs']:.0f} fs")
            report.append("")
        
        if 'vibrational_coherences' in self.results:
            report.append("VIBRATIONAL COHERENCES")
            report.append("-" * 40)
            data = self.results['vibrational_coherences']
            if data.get('dominant_frequency_THz'):
                report.append(f"Dominant frequency: {data['dominant_frequency_THz']:.1f} THz")
                report.append(f"Peak frequencies: {', '.join([f'{f:.1f}' for f in data['peak_frequencies_THz'][:3]])} THz")
            report.append("")
        
        report.append("=" * 60)
        
        return '\n'.join(report)


def demo_spectroscopy_dynamics():
    """Demonstrate spectroscopy dynamics simulation."""
    
    print("=" * 70)
    print("SPECTROSCOPY DYNAMICS DEMONSTRATION")
    print("=" * 70)
    
    # Create workflow
    workflow = SpectroscopyDynamicsWorkflow()
    
    # Setup pulses
    print("\n1. Setting up laser pulses...")
    workflow.setup_pump_pulse(wavelength=400.0, fwhm=30.0, intensity=1e13)
    workflow.setup_probe_pulse(wavelength=800.0, fwhm=20.0)
    
    # Add electronic transitions
    print("2. Adding electronic transitions...")
    workflow.ultrafast_abs.add_transition(ElectronicTransition(
        initial_state=0, final_state=1, energy=2.5, 
        oscillator_strength=1.0, linewidth=0.1
    ))
    workflow.ultrafast_abs.add_transition(ElectronicTransition(
        initial_state=0, final_state=2, energy=3.0,
        oscillator_strength=0.5, linewidth=0.15
    ))
    workflow.ultrafast_abs.add_transition(ElectronicTransition(
        initial_state=1, final_state=3, energy=4.0,
        oscillator_strength=0.8, linewidth=0.2
    ))
    
    # Generate synthetic population dynamics
    print("3. Generating synthetic population dynamics...")
    times = np.linspace(0, 1000, 500)
    nstates = 4
    populations = np.zeros((len(times), nstates))
    populations[:, 0] = np.exp(-times / 100)  # Ground state bleach recovery
    populations[:, 1] = np.exp(-times / 200) * (1 - np.exp(-times / 50))  # S1
    populations[:, 2] = 0.3 * (1 - np.exp(-times / 300))  # S2
    populations[:, 3] = 0.2 * (1 - np.exp(-times / 400))  # Higher state
    
    # Run transient absorption
    print("4. Running transient absorption simulation...")
    delay_times = np.linspace(0, 1000, 100)
    probe_energies = np.linspace(1.5, 4.5, 200)
    delta_A = workflow.run_transient_absorption(
        populations, delay_times, probe_energies
    )
    print(f"   ΔA range: {np.min(delta_A):.3f} to {np.max(delta_A):.3f}")
    
    # Setup and run TRPES
    print("5. Running TRPES simulation...")
    workflow.setup_trpes(ionization_potential=6.0, probe_energy=6.2)
    workflow.trpes.add_electronic_state(0, 0.0, 1.0)
    workflow.trpes.add_electronic_state(1, 2.5, 0.8)
    workflow.trpes.add_electronic_state(2, 3.0, 0.6)
    
    kinetic_energies = np.linspace(0, 3, 150)
    trpes_spectrum = workflow.run_trpes(populations, delay_times, kinetic_energies)
    
    if workflow.results.get('wavepacket_analysis'):
        analysis = workflow.results['wavepacket_analysis']
        if analysis.get('oscillation_frequency_THz'):
            print(f"   Wavepacket frequency: {analysis['oscillation_frequency_THz']:.1f} THz")
    
    # Analyze vibrational coherences
    print("6. Analyzing vibrational coherences...")
    # Synthetic signal with oscillations
    signal = np.exp(-times / 500) * np.cos(times * 0.05) + \
             0.5 * np.exp(-times / 300) * np.cos(times * 0.08)
    
    coherence_data = workflow.analyze_vibrational_coherences(signal, times)
    if coherence_data.get('dominant_frequency_THz'):
        print(f"   Dominant coherence: {coherence_data['dominant_frequency_THz']:.1f} THz")
    
    # Generate report
    print("\n" + "=" * 70)
    print(workflow.generate_report())
    
    # Visualizations
    if MATPLOTLIB_AVAILABLE:
        print("\n7. Generating visualizations...")
        workflow.visualize_spectra()
        print("   Done! Check spectroscopy_results/ directory")
    
    return workflow


if __name__ == "__main__":
    demo_spectroscopy_dynamics()
