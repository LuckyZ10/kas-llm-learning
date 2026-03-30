# QMC Module Documentation

## Overview

The QMC (Quantum Monte Carlo) module provides a comprehensive framework for performing quantum Monte Carlo calculations within the DFT-LAMMPS integration environment.

### Supported Methods

1. **Variational Monte Carlo (VMC)**
   - Slater-Jastrow wave functions
   - Neural network wave functions (FermiNet/PauliNet style)
   - Wave function optimization

2. **Diffusion Monte Carlo (DMC)**
   - Fixed-node approximation
   - Importance sampling
   - Population control
   - Time-step extrapolation

3. **Auxiliary-Field QMC (AFQMC)**
   - Phaseless approximation
   - Hubbard-Stratonovich transformation

## Installation

### Requirements

- Python >= 3.8
- NumPy
- SciPy (optional, for optimization)
- PySCF (optional, for wave function preparation)

### Setup

```bash
# Install required dependencies
pip install numpy scipy

# Install PySCF for full functionality
pip install pyscf
```

## Quick Start

### Basic VMC Calculation

```python
from dftlammps.qmc import PySCFQMCInterface, VMCCalculator
import numpy as np

# Define molecule
coordinates = np.array([
    [0.0, 0.0, 0.0],
    [0.74, 0.0, 0.0]
])

# Create interface
qmc = PySCFQMCInterface(
    atom_symbols=['H', 'H'],
    coordinates=coordinates,
    basis='cc-pVTZ'
)

# Run HF for trial wave function
hf_result = qmc.run_hf()

# Create VMC calculator
from dftlammps.qmc import create_slater_jastrow_from_pyscf

slater = create_slater_jastrow_from_pyscf(qmc.mf)
vmc_calc = VMCCalculator(
    wave_function=slater,
    atom_positions=coordinates,
    atom_charges=np.array([1, 1]),
    n_walkers=100,
    step_size=0.2
)

# Run VMC
result = vmc_calc.run(n_electrons=2, n_samples=10000, n_equil=1000)
print(f"VMC Energy: {result.energy:.6f} ± {result.energy_error:.6f} Ha")
```

### DMC Calculation

```python
from dftlammps.qmc import DMCCalculator, create_trial_wf_from_vmc

# Create trial wave function from VMC
trial_wf = create_trial_wf_from_vmc(slater)

# Run DMC
dmc_calc = DMCCalculator(
    trial_wf=trial_wf,
    atom_positions=coordinates,
    atom_charges=np.array([1, 1]),
    n_walkers_initial=500,
    time_step=0.01
)

dmc_result = dmc_calc.run(n_electrons=2, n_steps=10000, n_equil=1000)
print(f"DMC Energy: {dmc_result.energy:.6f} ± {dmc_result.energy_error:.6f} Ha")
```

## Module Structure

```
qmc/
├── __init__.py                   # Module initialization
├── pyscf_qmc_interface.py        # PySCF interface (763 lines)
├── vmc_calculator.py             # VMC implementation (913 lines)
├── dmc_calculator.py             # DMC implementation (642 lines)
├── afqmc_calculator.py           # AFQMC implementation (405 lines)
├── qmc_analysis.py               # Statistical analysis tools (418 lines)
├── case_qmc_benchmark/           # Benchmark calculations
│   └── benchmark.py              # Small molecule benchmarks (373 lines)
├── case_qmc_solid/               # Solid state calculations
│   └── solid_calculations.py     # Periodic systems (317 lines)
└── case_qmc_catalysis/           # Catalysis applications
    └── catalysis_calculations.py # Surface reactions (313 lines)
```

## API Reference

### PySCF-QMC Interface

#### `PySCFQMCInterface`

Main interface between PySCF and QMC calculations.

**Parameters:**
- `atom_symbols`: List of atomic symbols
- `coordinates`: Atomic coordinates (N, 3)
- `basis`: Basis set name (default: 'cc-pVTZ')
- `charge`: Total charge (default: 0)
- `spin`: Spin multiplicity (default: 0)
- `periodic`: Use periodic boundary conditions (default: False)
- `cell`: Unit cell vectors for periodic systems

**Methods:**
- `run_hf()`: Run Hartree-Fock calculation
- `run_dft(xc)`: Run DFT calculation
- `run_casscf(ncas, nelecas)`: Run CASSCF for multireference systems
- `run_mp2()`: Run MP2 calculation
- `run_ccsd(with_t)`: Run CCSD(T) calculation
- `get_slater_determinant()`: Get Slater determinant wave function
- `export_to_casino(output_dir)`: Export to CASINO format
- `export_to_qwalk(output_dir)`: Export to QWalk format

### VMC Calculator

#### `VMCCalculator`

Variational Monte Carlo calculator with Metropolis sampling.

**Parameters:**
- `wave_function`: Trial wave function object
- `atom_positions`: Nuclear positions
- `atom_charges`: Nuclear charges
- `n_walkers`: Number of random walkers
- `step_size`: Metropolis step size
- `seed`: Random seed

**Methods:**
- `run(n_electrons, n_samples, n_equil)`: Run VMC calculation
- `sample(n_electrons, n_samples, n_equil)`: Generate samples
- `compute_energy(samples)`: Compute energy from samples
- `optimize_wavefunction(...)`: Optimize wave function parameters

### DMC Calculator

#### `DMCCalculator`

Diffusion Monte Carlo calculator with fixed-node approximation.

**Parameters:**
- `trial_wf`: Trial wave function
- `atom_positions`: Nuclear positions
- `atom_charges`: Nuclear charges
- `n_walkers_initial`: Initial number of walkers
- `time_step`: DMC time step (tau)
- `target_walkers`: Target walkers for population control

**Methods:**
- `run(n_electrons, n_steps, n_equil)`: Run DMC calculation
- `equilibrate(n_electrons, n_steps)`: Equilibrate walkers

## Statistical Analysis

### Blocking Analysis

```python
from dftlammps.qmc import blocking_analysis, reblocking_analysis

# Perform blocking analysis
results = blocking_analysis(energies, min_block_size=1)
print(f"Recommended error: {results['recommended_error']}")

# Reblocking analysis
summary = reblocking_analysis(energies)
print(f"Mean: {summary.mean:.6f} ± {summary.std_error:.6f}")
```

### Time-Step Extrapolation

```python
from dftlammps.qmc import analyze_time_step_error

# Run DMC with different time steps
results = analyze_time_step_error(
    atom_positions=coords,
    atom_charges=charges,
    trial_wf=trial_wf,
    n_electrons=2,
    time_steps=[0.001, 0.005, 0.01, 0.02]
)

print(f"Extrapolated energy: {results['extrapolated']['energy']:.6f} Ha")
```

## Application Examples

### Benchmark Calculations

```bash
# Run all benchmarks
cd case_qmc_benchmark
python benchmark.py --system all

# Run specific system
python benchmark.py --system H2 --vmc-samples 10000
```

### Solid State Calculations

```bash
# Run hydrogen chain
cd case_qmc_solid
python solid_calculations.py --calculation chain

# Run all solid calculations
python solid_calculations.py --calculation all
```

### Catalysis Calculations

```bash
# Run catalysis examples
cd case_qmc_catalysis
python catalysis_calculations.py --calculation all
```

## Best Practices

### Wave Function Quality

1. **Trial Wave Functions**
   - Use high-quality HF/DFT orbitals
   - Include Jastrow factors for electron correlation
   - Consider multi-determinant expansions for strongly correlated systems

2. **Jastrow Factors**
   - Start with electron-electron terms
   - Add electron-nucleus terms for better accuracy
   - Optimize parameters using variance minimization

### Sampling

1. **Equilibration**
   - Always discard initial samples (equilibration)
   - Monitor acceptance rate (target: 50-70%)
   - Check convergence using multiple blocks

2. **Sample Size**
   - Use blocking analysis to estimate errors
   - Aim for statistical errors < 1 mHa for chemical accuracy
   - Consider autocorrelation when estimating effective samples

### DMC Specific

1. **Time Step**
   - Start with small time steps (0.001-0.01)
   - Extrapolate to zero time step
   - Check stability of population

2. **Fixed-Node Approximation**
   - Use best available trial wave function
   - Consider orbital optimization
   - Test with different trial functions

## Troubleshooting

### Common Issues

**Low Acceptance Rate**
- Decrease step size
- Check wave function normalization
- Verify electron initialization

**Large Variance**
- Improve trial wave function
- Add higher-order Jastrow terms
- Check for node crossing in DMC

**Population Explosion/Collapse**
- Adjust population control frequency
- Tune reference energy
- Check branching weights

### Performance Tips

1. **Parallelization**: The code can be extended with MPI for parallel walkers
2. **GPU Acceleration**: Neural network wave functions benefit from GPU
3. **Basis Sets**: Use localized basis sets for better scaling

## References

1. Foulkes, Mitas, Needs, Rajagopal. "Quantum Monte Carlo simulations of solids." Rev. Mod. Phys. 73, 33 (2001)
2. Austin, Zubarev, Lester. "Quantum Monte Carlo and related approaches." Chem. Rev. 112, 263 (2012)
3. Spencer, Blunt, Foulkes. "Projector quantum Monte Carlo with density matrix embedding theory." J. Chem. Phys. 151, 014107 (2019)

## License

This module is part of the DFT-LAMMPS integration package.
