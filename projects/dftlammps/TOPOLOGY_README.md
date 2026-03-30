# DFTLammps Topology Module

## Overview

The Topology Module extends DFTLammps with comprehensive tools for topological materials calculations, including Z2 invariants, Berry phases, Weyl points, and surface states.

## Module Structure

```
dftlammps/
├── topology/                          # Topological invariant calculations
│   ├── z2pack_interface.py           # Z2Pack interface (~830 lines)
│   ├── wannier_tools_interface.py    # WannierTools interface (~1050 lines)
│   ├── berry_phase.py                # Berry phase & AHC (~820 lines)
│   └── __init__.py
├── weyl/                              # Weyl semimetal analysis
│   ├── weyl_semimetal.py             # Weyl point calculations (~920 lines)
│   └── __init__.py
└── applications/
    ├── case_topological_insulator/   # Bi2Se3/Bi2Te3 case study (~530 lines)
    ├── case_weyl_semimetal/          # TaAs case study (~560 lines)
    └── case_quantum_anomalous_hall/  # QAHE case study (~600 lines)
```

**Total: ~5300 lines of code**

## Features

### 1. Z2Pack Interface (`z2pack_interface.py`)

**Capabilities:**
- VASP wavefunction extraction
- Wilson loop calculations
- Z2 invariant determination (time-reversal symmetric/asymmetric)
- Chern number calculations
- Topological phase classification

**Key Classes:**
- `Z2PackConfig`: Configuration for Z2Pack calculations
- `VASPWavefunctionExtractor`: Extract and process VASP wavefunctions
- `Z2VASPInterface`: Interface between Z2Pack and VASP
- `ChernNumberCalculator`: Calculate Chern numbers
- `TopologicalClassifier`: Classify topological phases

**Example Usage:**
```python
from dftlammps.topology import Z2PackConfig, Z2VASPInterface

config = Z2PackConfig(
    num_bands=40,
    num_wann=20,
    surface="kz-surface",
    time_reversal_symmetric=True
)

interface = Z2VASPInterface("./Bi2Se3", config)
result = interface.calculate_z2_invariant()

print(f"Strong Z2 index: {result.z2_index}")
print(f"Weak indices: {result.z2_indices[1:]}")
print(f"Chern number: {result.chern_number}")
```

### 2. WannierTools Interface (`wannier_tools_interface.py`)

**Capabilities:**
- Wannier90 Hamiltonian construction
- Surface state calculations
- Band inversion identification
- Weyl point search
- Fermi arc analysis

**Key Classes:**
- `WannierToolsConfig`: Configuration for WannierTools
- `Wannier90HamiltonianBuilder`: Build tight-binding Hamiltonians
- `WannierToolsCalculator`: Calculate surface states and Weyl points
- `BandInversionAnalyzer`: Analyze band inversions at TRIM points

**Example Usage:**
```python
from dftlammps.topology import WannierToolsCalculator, WannierToolsConfig

config = WannierToolsConfig(
    num_layers=10,
    calculate_fermi_arc=True
)

calculator = WannierToolsCalculator("wannier90_hr.dat", config)
surface_result = calculator.calculate_surface_states()
weyl_result = calculator.search_weyl_points()
```

### 3. Berry Phase Module (`berry_phase.py`)

**Capabilities:**
- Electric polarization calculation (VASP LCALCPOL)
- Berry curvature in k-space
- Anomalous Hall conductivity (AHC)
- Chern number from Berry curvature
- Born effective charges

**Key Classes:**
- `PolarizationCalculator`: Calculate electric polarization
- `BerryCurvatureCalculator`: Calculate Berry curvature
- `AnomalousHallConductivityCalculator`: Calculate AHC

**Example Usage:**
```python
from dftlammps.topology import (
    BerryCurvatureCalculator, 
    AnomalousHallConductivityCalculator
)

# Berry curvature
berry_calc = BerryCurvatureCalculator("./calculation")
curvature = berry_calc.calculate_berry_curvature(k_mesh=(30, 30, 1))
print(f"Chern number: {curvature.chern_numbers}")

# Anomalous Hall conductivity
ahc_calc = AnomalousHallConductivityCalculator("./calculation")
ahc = ahc_calc.calculate_ahc(temperature=0.0)
print(f"σ_xy = {ahc.get_hall_conductivity('xy'):.2e} S/cm")
```

### 4. Weyl Semimetal Module (`weyl/weyl_semimetal.py`)

**Capabilities:**
- Weyl point location and classification (Type I / Type II)
- Chirality calculation via Berry curvature integration
- Fermi arc surface state calculation
- Magnetotransport and chiral anomaly analysis

**Key Classes:**
- `WeylPointLocator`: Locate Weyl points in BZ
- `ChiralityCalculator`: Calculate Weyl point chirality
- `FermiArcCalculator`: Calculate Fermi arcs
- `MagnetotransportCalculator`: Calculate transport properties

**Example Usage:**
```python
from dftlammps.weyl import (
    WeylSemimetalConfig, 
    WeylPointLocator,
    analyze_weyl_semimetal
)

config = WeylSemimetalConfig(
    k_mesh_fine=(50, 50, 50),
    gap_threshold=0.001
)

locator = WeylPointLocator("./TaAs", config)
weyl_points = locator.search_weyl_points()

for wp in weyl_points:
    print(f"k = {wp.k_point}, C = {wp.chirality}, E = {wp.energy:.3f} eV")
```

## Application Cases

### 1. Topological Insulators: Bi2Se3 / Bi2Te3

**File:** `applications/case_topological_insulator/bi2se3_analysis.py`

**Features:**
- Structure generation for Bi2Se3/Bi2Te3
- VASP input with SOC
- Z2 invariant calculation
- Surface state analysis

**Quick Start:**
```python
from dftlammps.applications.case_topological_insulator import analyze_bi2se3

results = analyze_bi2se3("./Bi2Se3")
# Expected: Z2 = 1 (strong topological insulator)
```

### 2. Weyl Semimetals: TaAs Family

**File:** `applications/case_weyl_semimetal/taas_analysis.py`

**Features:**
- Structure generation for TaAs, TaP, NbAs, NbP
- Weyl point search (24 points in TaAs)
- Chirality determination
- Fermi arc calculation
- Chiral anomaly analysis

**Quick Start:**
```python
from dftlammps.applications.case_weyl_semimetal import analyze_taas

results = analyze_taas("./TaAs")
# Expected: 24 Weyl points (12 pairs)
```

### 3. Quantum Anomalous Hall Effect

**File:** `applications/case_quantum_anomalous_hall/qahe_analysis.py`

**Features:**
- Magnetic doping structure generation
- Cr/V-doped (Bi,Sb)2Te3 workflows
- Chern number calculation
- Quantized Hall conductivity verification

**Quick Start:**
```python
from dftlammps.applications.case_quantum_anomalous_hall import (
    analyze_cr_doped_bi2te3
)

results = analyze_cr_doped_bi2te3(concentration=0.08)
# Expected: C = 1, σ_xy = e²/h
```

## Workflow Integration

### VASP + Z2Pack Workflow

```python
from dftlammps.topology import (
    Z2VASPInterface, 
    Wannier90HamiltonianBuilder,
    calculate_z2_index
)

# Step 1: Run VASP with LWANNIER90 = .TRUE.
# Step 2: Generate Wannier90 Hamiltonian
builder = Wannier90HamiltonianBuilder("./vasp_output")
builder.run_wannier90("./wannier")

# Step 3: Calculate Z2 invariant
z2 = calculate_z2_index("./wannier", surface="kz-surface")
```

### WannierTools Workflow

```python
from dftlammps.topology import WannierToolsCalculator

# Use Wannier90 output
calc = WannierToolsCalculator("wannier90_hr.dat")

# Surface states
surface = calc.calculate_surface_states()

# Weyl points
weyl = calc.search_weyl_points()
```

## Dependencies

**Required:**
- numpy
- scipy
- pymatgen
- ase

**Optional:**
- z2pack (for Z2 calculations)
- WannierTools (for surface states)
- matplotlib (for plotting)

## References

### Topological Insulators
- Zhang et al., Nature Phys. 5, 438 (2009) - Bi2Se3 prediction
- Xia et al., Nature Phys. 5, 398 (2009) - ARPES observation
- Chen et al., Science 325, 178 (2009) - Bi2Te3

### Weyl Semimetals
- Wan et al., PRL 107, 127601 (2011) - Theoretical foundation
- Weng et al., PRX 5, 011029 (2015) - TaAs prediction
- Xu et al., Science 349, 613 (2015) - TaAs observation

### Quantum Anomalous Hall Effect
- Yu et al., Science 329, 61 (2010) - Theoretical prediction
- Chang et al., Science 340, 167 (2013) - First observation
- Chang et al., Nature Mater. 14, 473 (2015) - Higher temperature

### Methods
- Soluyanov et al., PRB 83, 235401 (2011) - Wannier charge centers
- Yu et al., PRB 84, 075119 (2011) - Equivalent Berry phases
- King-Smith and Vanderbilt, PRB 47, 1651 (1993) - Modern polarization theory
- Xiao et al., RMP 82, 1959 (2010) - Berry phase in solids

## License

MIT License - See LICENSE file for details.
