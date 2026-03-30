# DFT-LAMMPS Strongly Correlated Systems Module

## Overview

This module provides comprehensive tools for studying strongly correlated electron systems using DFT+U and Dynamical Mean-Field Theory (DMFT) methods.

## Module Structure

```
dftlammps/
├── correlated/              # Core DMFT and Hubbard U functionality
│   ├── dmft_interface.py    # DMFT implementation with VASP+Wannier90
│   ├── hubbard_u.py         # Hubbard U calculation methods
│   └── triqs_interface.py   # TRIQS interface for advanced calculations
│
├── mott/                    # Mott insulator analysis tools
│   └── mott_analysis.py     # Gap analysis, MIT detection, order parameters
│
└── applications/            # Application case studies
    ├── case_high_tc_superconductor/  # Cuprates and iron-based SC
    ├── case_mott_insulator/          # NiO, CoO analysis
    └── case_correlated_catalyst/     # TMO catalysis
```

## Installation

### Prerequisites

- Python 3.8+
- NumPy, SciPy
- (Optional) TRIQS for advanced many-body calculations
- (Optional) VASP for DFT calculations
- (Optional) Wannier90 for Wannier function construction

### Installation

```bash
pip install dftlammps
```

## Quick Start

### DMFT Calculation

```python
from dftlammps.correlated import DMFTEngine, DMFTConfig

# Setup configuration
config = DMFTConfig(
    temperature=300.0,
    u_value=4.0,
    j_value=0.6,
    n_orbitals=5
)

# Initialize DMFT engine
dmft = DMFTEngine(config)
dmft.initialize(solver_type="triqs")

# Run self-consistent loop
results = dmft.run_scf_loop(H_k, k_weights)

# Calculate spectral function
omega, A_w = dmft.calculate_spectral_function(H_k, k_points, k_weights)
```

### Hubbard U Calculation

```python
from dftlammps.correlated import LinearResponseU

# Calculate U using linear response
u_calc = LinearResponseU()
results = u_calc.calculate_linear_response_u("POSCAR", [0,1,2,3,4])

print(f"U = {results['U']:.2f} eV")
print(f"J = {results['J']:.2f} eV")
```

### Mott Insulator Analysis

```python
from dftlammps.mott import GapAnalyzer, MetalInsulatorTransition

# Analyze electronic gap
analyzer = GapAnalyzer()
gap_info = analyzer.calculate_gap(eigenvalues, k_points)

# Detect metal-insulator transition
mit = MetalInsulatorTransition()
mit_results = mit.detect_mit_gap_criterion(gaps, U_values)
```

## Case Studies

### 1. High-Tc Superconductors

#### Cuprates

```python
from dftlammps.applications.case_high_tc_superconductor import CuprateDFTDMFT, CuprateConfig

config = CuprateConfig(
    material="La2CuO4",
    U_cu=8.0,
    hole_doping=0.15
)

cuprate = CuprateDFTDMFT(config)
cuprate.generate_dft_input()

# Analyze d-wave pairing
pairing = cuprate.analyze_d_wave_pairing(chi_spin, q_points)
```

#### Iron-Based Superconductors

```python
from dftlammps.applications.case_high_tc_superconductor import IronPnictideAnalyzer

fe_as = IronPnictideAnalyzer("BaFe2As2")
fs = fe_as.analyze_fermi_surface(k_grid)
chi = fe_as.calculate_spin_susceptibility(q_grid)
```

### 2. Mott Insulators

```python
from dftlammps.applications.case_mott_insulator import MottInsulatorWorkflow

workflow = MottInsulatorWorkflow("NiO")
results = workflow.run_complete_analysis()
workflow.generate_dft_input()
```

### 3. Correlated Catalysts

```python
from dftlammps.applications.case_correlated_catalyst import Co3O4Catalyst

catalyst = Co3O4Catalyst()
surface = catalyst.setup_surface_structure(n_layers=4)

oer = catalyst.calculate_oer_mechanism()
print(f"Overpotential: {oer['overpotential_V']:.2f} V")
```

## Features

### DMFT Implementation

- Self-consistent DMFT loop
- CT-QMC impurity solvers (TRIQS, ALPS, iPET)
- Wannier90 interface for localized orbitals
- Spectral function calculation
- Real-frequency analytical continuation

### Hubbard U Methods

- Linear response method (Cococcioni)
- Constrained RPA
- Self-consistent U
- DFT+U parameter optimization
- Database of literature values

### TRIQS Interface

- Multi-orbital Hubbard models
- Two-particle Green's functions
- Superconducting pairing susceptibility
- Magnetic and charge susceptibilities

### Mott Insulator Analysis

- Gap opening/closing criteria
- Metal-insulator transition detection
- Charge and spin order parameters
- Phase diagram construction

## References

### DMFT
1. Georges et al., Rev. Mod. Phys. 68, 13 (1996)
2. Kotliar et al., Rev. Mod. Phys. 78, 865 (2006)

### Hubbard U
1. Cococcioni & de Gironcoli, PRB 71, 035105 (2005)
2. Aryasetiawan et al., PRB 70, 195104 (2004)

### Superconductivity
1. Scalapino, Rev. Mod. Phys. 84, 1383 (2012)
2. Hirschfeld et al., Rep. Prog. Phys. 74, 124508 (2011)

## License

MIT License - See LICENSE file for details

## Citation

If you use this module in your research, please cite:

```
DFT-LAMMPS: A Python Framework for Strongly Correlated Materials
[Your citation here]
```