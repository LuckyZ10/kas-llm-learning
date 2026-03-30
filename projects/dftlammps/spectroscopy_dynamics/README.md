# Spectroscopy Dynamics Module

Time-resolved spectroscopy simulation for ultrafast processes.

## Overview

This module simulates and analyzes time-resolved spectroscopic signals from non-adiabatic dynamics simulations, including:

- Ultrafast absorption spectroscopy
- Transient absorption (pump-probe)
- Time-resolved photoelectron spectroscopy (TRPES)
- 2D electronic spectroscopy

## Classes

### LaserPulse

Defines laser pulse properties for pump-probe experiments.

```python
from dftlammps.spectroscopy_dynamics import LaserPulse

# Create pump pulse
pump = LaserPulse(
    wavelength=400.0,  # nm
    fwhm=30.0,  # fs
    intensity=1e13,  # W/cm²
    chirp=0.0  # fs⁻²
)

# Get electric field
times = np.linspace(-100, 100, 1000)  # fs
E_field = pump.get_electric_field(times, t0=0)
```

### UltrafastAbsorption

Simulates transient absorption spectra.

```python
from dftlammps.spectroscopy_dynamics import UltrafastAbsorption, ElectronicTransition

# Setup simulator
sim = UltrafastAbsorption(
    pump_pulse=pump,
    probe_pulse=probe
)

# Add transitions
sim.add_transition(ElectronicTransition(
    initial_state=0,
    final_state=1,
    energy=2.5,  # eV
    oscillator_strength=1.0,
    linewidth=0.1  # eV
))

# Calculate transient absorption
delta_A = sim.calculate_transient_absorption(
    delay_times=np.linspace(0, 1000, 100),  # fs
    probe_energies=np.linspace(1.5, 4.0, 200),  # eV
    state_populations=populations  # From dynamics
)
```

### TimeResolvedPhotoelectronSpectroscopy

Simulates TRPES for tracking wavepacket motion.

```python
from dftlammps.spectroscopy_dynamics import TimeResolvedPhotoelectronSpectroscopy

# Setup TRPES
trpes = TimeResolvedPhotoelectronSpectroscopy(
    ionization_potential=6.0,  # eV
    probe_photon_energy=6.2  # eV
)

# Add electronic states
trpes.add_electronic_state(
    state_index=0,
    energy=0.0,
    ionization_cross_section=1.0
)

# Calculate spectrum
spectrum = trpes.calculate_trpes_spectrum(
    delay_times,
    kinetic_energies,
    state_populations
)
```

### VibrationalCoherenceAnalysis

Extracts vibrational coherences from time-domain signals.

```python
from dftlammps.spectroscopy_dynamics import VibrationalCoherenceAnalysis

analyzer = VibrationalCoherenceAnalysis()

# Extract frequencies
coherence_data = analyzer.extract_coherences(
    signal=delta_A[:, energy_index],  # Time trace
    times=delay_times
)

print(f"Dominant frequency: {coherence_data['dominant_frequency_THz']:.1f} THz")
```

### SpectroscopyDynamicsWorkflow

Complete workflow for spectroscopy simulations.

```python
from dftlammps.spectroscopy_dynamics import SpectroscopyDynamicsWorkflow

# Create workflow
workflow = SpectroscopyDynamicsWorkflow()

# Setup pulses
workflow.setup_pump_pulse(wavelength=400.0, fwhm=30.0)
workflow.setup_probe_pulse(wavelength=800.0, fwhm=20.0)

# Run transient absorption
delta_A = workflow.run_transient_absorption(
    state_populations=populations,
    delay_times=delay_times,
    probe_energies=probe_energies
)

# Analyze coherences
coherence_data = workflow.analyze_vibrational_coherences(
    signal=delta_A[:, idx],
    times=delay_times
)

# Generate visualizations
workflow.visualize_spectra(output_dir='./results')
```

## Signal Interpretation

### Transient Absorption

$$\Delta A(\omega, \tau) = A_{\text{excited}}(\omega, \tau) - A_{\text{ground}}(\omega)$$

Features:
- **Ground state bleach (GSB)**: Negative signal at absorption energies
- **Stimulated emission (SE)**: Negative signal at emission energies
- **Excited state absorption (ESA)**: Positive signal at higher energies

### TRPES

$$E_{\text{kin}} = h\nu_{\text{probe}} - IP - E_{\text{state}}$$

Advantages:
- Direct probe of electronic state energies
- No dark states
- Element and orbital specificity

### Vibrational Coherences

Oscillations in transient signals indicate wavepacket motion:

$$S(t) = \sum_j A_j \cos(\omega_j t + \phi_j) e^{-t/\tau_j}$$

## Example: Complete Analysis

```python
import numpy as np
from dftlammps.spectroscopy_dynamics import (
    SpectroscopyDynamicsWorkflow,
    LaserPulse,
    ElectronicTransition
)

# Initialize
workflow = SpectroscopyDynamicsWorkflow()

# Setup laser pulses
workflow.setup_pump_pulse(wavelength=400.0, fwhm=30.0)
workflow.setup_probe_pulse(wavelength=800.0, fwhm=20.0)

# Define electronic transitions
for i, (E, f) in enumerate(zip([2.5, 3.0, 4.0], [1.0, 0.5, 0.8])):
    workflow.ultrafast_abs.add_transition(ElectronicTransition(
        initial_state=0, final_state=i+1,
        energy=E, oscillator_strength=f
    ))

# Simulate with synthetic population data
delay_times = np.linspace(0, 1000, 100)
probe_energies = np.linspace(1.5, 4.5, 200)
populations = np.random.rand(100, 4)  # From dynamics

delta_A = workflow.run_transient_absorption(
    populations, delay_times, probe_energies
)

# Visualize
workflow.visualize_spectra()

# Generate report
print(workflow.generate_report())
```

## References

1. Hamm, P.; Zanni, M. *Concepts and Methods of 2D Infrared Spectroscopy*. Cambridge University Press (2011)
2. Stolow, A.; Jonas, D. M. "Multidimensional Snapshots of Chemical Dynamics" Science 305, 1575 (2004)
3. Schmitt-Rink, S.; Mukamel, S. "Theory of Transient Excitonic Optical Nonlinearities" J. Lumin. 30, 123 (1985)

## See Also

- `nonadiabatic`: Non-adiabatic dynamics module
- `electronic_structure`: Electronic structure calculations
