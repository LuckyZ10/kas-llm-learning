# Phonon Calculation Workflow and Convergence Testing

## Table of Contents
1. [Pre-calculation Setup](#setup)
2. [Step-by-Step Calculation Workflow](#workflow)
3. [Convergence Testing Protocol](#convergence)
4. [Validation Checks](#validation)
5. [Troubleshooting](#troubleshooting)

---

## 1. Pre-calculation Setup <a name="setup"></a>

### 1.1 Required Software

- **DFT Code**: VASP, Quantum ESPRESSO, or ABINIT
- **Phonon Code**: Phonopy (required), Phono3py (for thermal conductivity)
- **Python Environment**: NumPy, SciPy, Matplotlib
- **Structure Tools**: Pymatgen, ASE (optional but recommended)

### 1.2 Input Structure Requirements

```python
# Checklist before starting phonon calculation
structure_checks = {
    "fully_relaxed": True,           # Geometry optimized
    "forces_converged": True,        # Forces < 1e-5 eV/Å
    "pressure_converged": True,      # Stress < 0.1 kB
    "symmetry_ana": True,            # Symmetry analysis done
    "primitive_cell": True,          # Use primitive cell if possible
}
```

### 1.3 Pseudopotential Considerations

For accurate phonon calculations:
- Use consistent pseudopotentials across all calculations
- Include semi-core states for heavy elements
- Test pseudopotential convergence

---

## 2. Step-by-Step Calculation Workflow <a name="workflow"></a>

### Phase 1: Force Constant Calculation

#### Step 1.1: Determine Supercell Size

```python
from dftlammps.phonon import PhonopyInterface, PhononConfig
import numpy as np

# Strategy: Start with 2x2x2, check convergence
config = PhononConfig(
    structure_path="POSCAR",
    supercell_matrix=np.diag([2, 2, 2]),  # Start here
    displacement_distance=0.01,
    output_dir="./phonon_calc"
)

phonon = PhonopyInterface(config)
```

**Guidelines for Supercell Size:**

| Material Type | Recommended Supercell | Notes |
|--------------|----------------------|-------|
| Simple metals | 3×3×3 or 4×4×4 | Long-range interactions |
| Covalent solids | 2×2×2 or 3×3×3 | Directional bonds |
| Ionic compounds | 2×2×2 | Often sufficient |
| Molecular crystals | 2×2×2 | Van der Waals |
| Low-κ materials | 3×3×3 or larger | Complex phonons |

#### Step 1.2: Generate Displacements

```python
# Generate displacement structures
displacements = phonon.create_displacements(
    structure="POSCAR",
    distance=0.01,           # Å - typical value
    is_plusminus='auto',     # Generate both + and - displacements
    is_diagonal=True,        # Include diagonal displacements
    is_trigonal=False        # Usually not needed
)

print(f"Generated {len(displacements)} displacement structures")
```

**Displacement Convergence:**

Test different displacement magnitudes:
- Too small: Numerical noise
- Too large: Breaks harmonic approximation
- Typical range: 0.005 - 0.02 Å
- Recommended: 0.01 Å for most systems

#### Step 1.3: Run DFT Calculations

**VASP Settings for Forces:**

```bash
# INCAR for force calculations
PREC = Accurate
ENCUT = 1.3 * ENMAX  # 30% higher than default
EDIFF = 1E-8         # Tight convergence
NSW = 0              # Single point
ISMEAR = 0           # Gaussian smearing (insulators)
# or
ISMEAR = -5          # Tetrahedron (metals with dense k-mesh)
SIGMA = 0.05         # Small smearing
IBRION = -1          # No relaxation
```

**Important Notes:**
- Use same ENCUT for all displacement calculations
- Maintain consistent k-point grid
- Keep high symmetry if present
- Save WAVECAR to speed up subsequent calculations

#### Step 1.4: Extract Force Constants

```python
# From VASP finite differences (IBRION=5)
force_constants = phonon.calculate_force_constants_vasp(
    outcar_paths=["OUTCAR-001", "OUTCAR-002", ...],
    disp_yaml_path="disp.yaml"
)

# Or from VASP DFPT (IBRION=6,7,8)
force_constants = phonon.calculate_force_constants_vasp(
    vasprun_paths=["vasprun.xml"]
)
```

---

### Phase 2: Phonon Calculation

#### Step 2.1: Calculate Band Structure

```python
# Calculate along high-symmetry path
band_structure = phonon.calculate_band_structure(
    path="GXMGRX",           # Example for FCC
    npoints=101,
    with_eigenvectors=True
)

# Plot
phonon.plot_band_structure(
    unit='THz',
    save_path="band_structure.png"
)
```

**High-Symmetry Paths:**

| Lattice | Path String | Points |
|---------|-------------|--------|
| FCC | Γ-X-W-K-Γ-L-U-W-L-K | 8 |
| BCC | Γ-H-N-Γ-P-H|P-N | 6 |
| Hexagonal | Γ-M-K-Γ-A-L-H-A|L-M|K-H | 9 |
| Simple Cubic | Γ-X-M-Γ-R-X|M-R | 6 |

#### Step 2.2: Calculate DOS

```python
# Calculate DOS
dos = phonon.calculate_dos(
    mesh=(20, 20, 20),       # Converge this!
    t_max=1000.0
)

# PDOS
pdos = phonon.calculate_pdos(
    mesh=(20, 20, 20),
    legendre_delta=0.1
)

phonon.plot_dos(save_path="dos.png")
```

#### Step 2.3: Calculate Thermodynamic Properties

```python
from dftlammps.phonon import ThermalPropertyCalculator

thermal_calc = ThermalPropertyCalculator()
thermal_results = thermal_calc.calculate_from_phonopy(
    phonopy=phonon.phonopy,
    temperatures=np.arange(0, 1001, 10)
)

# Save results
thermal_results.save("thermal_properties.npz")
thermal_calc.plot_thermal_properties(
    save_path="thermal_properties.png"
)
```

---

### Phase 3: Advanced Calculations (Optional)

#### Step 3.1: Quasi-Harmonic Approximation

```python
from dftlammps.phonon import QHACalculator

qha_calc = QHACalculator()

# Prepare structures at different volumes
structures = qha_calc.prepare_volume_expansions(
    structure=relaxed_structure,
    n_volumes=7,
    volume_range=(0.94, 1.06)
)

# Calculate for each volume
# ... run phonon for each volume ...

# Run QHA
qha_results = qha_calc.run_qha(
    volumes=volumes,
    electronic_energies=energies,
    phonopy_objects=phonopy_at_volumes
)
```

#### Step 3.2: Thermal Conductivity (Phono3py)

```python
from dftlammps.phonon import LatticeThermalConductivity, ThermalConductivityConfig

config = ThermalConductivityConfig(
    mesh=(11, 11, 11),       # Odd mesh required
    temperatures=np.arange(300, 1001, 100),
    method=ConductivityMethod.RTA,
    include_isotope=True,
    output_dir="./kappa"
)

kappa_calc = LatticeThermalConductivity(config)

# Generate displacements for 3rd order
kappa_calc.create_displacements(structure, displacement_distance=0.03)

# After DFT calculations, set force constants
kappa_calc.set_force_constants(fc2_path, fc3_path)

# Calculate thermal conductivity
results = kappa_calc.run_thermal_conductivity_rta()

kappa_calc.plot_kappa_vs_temperature(
    save_path="thermal_conductivity.png"
)
```

---

## 3. Convergence Testing Protocol <a name="convergence"></a>

### 3.1 DFT Convergence

#### Plane Wave Cutoff Convergence

```python
cutoff_tests = [300, 400, 500, 600, 700]  # eV
convergence_data = []

for encut in cutoff_tests:
    energy = run_scf(encut=encut)
    convergence_data.append((encut, energy))

# Plot convergence
# Look for plateau in energy vs cutoff
```

**Criterion:** Energy difference < 1 meV/atom between successive cutoffs

#### k-point Convergence

```python
k_mesh_tests = [(4,4,4), (6,6,6), (8,8,8), (10,10,10)]

for k_mesh in k_mesh_tests:
    energy = run_scf(k_mesh=k_mesh)
    # Track energy and forces
```

**Criterion:** Energy difference < 1 meV/atom

### 3.2 Supercell Convergence

Test phonon frequencies with increasing supercell size:

```python
supercell_tests = [
    np.diag([1, 1, 1]),  # Primitive
    np.diag([2, 2, 2]),  # 2x2x2
    np.diag([3, 3, 3]),  # 3x3x3
    np.diag([4, 4, 4]),  # 4x4x4
]

results = []
for sc in supercell_tests:
    config = PhononConfig(supercell_matrix=sc)
    phonon = PhonopyInterface(config)
    # ... calculate frequencies at Γ
    gamma_freqs = phonon.phonopy.get_frequencies([0,0,0])
    results.append((sc[0,0], gamma_freqs))
```

**Criterion:** Optical mode frequencies converged to < 0.1 THz

### 3.3 Mesh Convergence for DOS

```python
mesh_tests = [
    (10, 10, 10),
    (15, 15, 15),
    (20, 20, 20),
    (25, 25, 25),
    (30, 30, 30),
]

for mesh in mesh_tests:
    dos = phonon.calculate_dos(mesh=mesh)
    free_energy = calculate_free_energy(dos)
    print(f"Mesh {mesh}: F = {free_energy:.4f} meV/atom")
```

**Convergence Criteria:**

| Property | Mesh Size | Convergence |
|----------|-----------|-------------|
| Free Energy (0K) | 20×20×20 | < 1 meV/atom |
| Entropy (300K) | 20×20×20 | < 0.1 J/mol/K |
| Heat Capacity | 20×20×20 | < 1% |
| Thermal Conductivity | 11×11×11 | < 10% |

### 3.4 Displacement Convergence

```python
displacement_tests = [0.005, 0.01, 0.015, 0.02, 0.03]

for disp in displacement_tests:
    phonon.create_displacements(distance=disp)
    # ... calculate force constants ...
    freqs = get_gamma_point_frequencies()
    print(f"Disp {disp}: freq = {freqs}")
```

**Criterion:** Frequencies stable to < 0.05 THz

---

## 4. Validation Checks <a name="validation"></a>

### 4.1 Acoustic Sum Rule

The acoustic sum rule requires that the sum of force constants for each atom equals zero:

```python
# Check acoustic sum rule
fc = phonon.phonopy.force_constants
n_atoms = len(fc)

for i in range(n_atoms):
    sum_fc = np.sum(fc[i], axis=0)
    deviation = np.linalg.norm(sum_fc)
    print(f"Atom {i}: deviation = {deviation:.6f}")
    if deviation > 1e-3:
        print("  WARNING: Acoustic sum rule violated!")
```

**Action:** If violated, apply acoustic sum rule correction in Phonopy.

### 4.2 Dynamical Stability

Check for imaginary frequencies (negative eigenvalues):

```python
is_stable, imag_modes = phonon.check_dynamical_stability(threshold=-0.1)

if not is_stable:
    print(f"WARNING: {len(imag_modes)} imaginary modes found")
    print(f"Frequencies: {imag_modes} THz")
    print("Structure may be dynamically unstable!")
```

### 4.3 High-Temperature Limit

Verify heat capacity approaches Dulong-Petit limit:

```python
# High-temperature heat capacity should approach 3N*k_B
n_atoms = phonon.phonopy.unitcell.get_number_of_atoms()
cv_limit = 3 * n_atoms * 8.314  # J/mol/K

temps = np.array([800, 900, 1000, 1100, 1200])
cv_values = thermal_results.heat_capacity_v[-5:]

for T, cv in zip(temps, cv_values):
    ratio = cv / cv_limit
    print(f"T={T}K: Cv/Cv_limit = {ratio:.3f}")
```

### 4.4 Compare with Experimental Data

```python
# Load experimental data
exp_freqs = load_experimental_phonon_frequencies()
calc_freqs = phonon.phonopy.get_frequencies(exp_qpoints)

# Calculate RMS error
rms_error = np.sqrt(np.mean((calc_freqs - exp_freqs)**2))
print(f"RMS error vs experiment: {rms_error:.2f} THz")
```

**Expected Accuracy:**
- LDA/GGA: ~5-10% error in phonon frequencies
- Including anharmonicity: ~2-5% error

---

## 5. Troubleshooting <a name="troubleshooting"></a>

### Common Issues and Solutions

#### Issue 1: Imaginary Frequencies

**Symptoms:** Negative frequencies in the band structure or DOS

**Causes:**
- Structure not at energy minimum
- Insufficient supercell size
- Numerical issues with forces

**Solutions:**
```python
# 1. Re-relax structure with tighter criteria
# 2. Increase supercell size
# 3. Check force convergence
# 4. Use smaller displacement distance

# Check forces
if max_forces > 1e-4:
    print("Structure not fully relaxed!")
    print("Re-run relaxation with tighter EDIFFG")
```

#### Issue 2: Noisy Force Constants

**Symptoms:** Irregular phonon dispersion, spurious crossings

**Solutions:**
- Increase plane wave cutoff
- Use denser k-point grid
- Check for numerical noise in forces
- Apply symmetry constraints

#### Issue 3: Slow Convergence of DOS

**Symptoms:** DOS changes significantly with mesh size

**Solutions:**
- Use tetrahedron method instead of smearing
- Increase q-point mesh
- Check for van Hove singularities

#### Issue 4: Thermal Conductivity Too High

**Symptoms:** Calculated κ much larger than experiment

**Possible Causes:**
- Missing scattering mechanisms
- Insufficient q-point mesh for Phono3py
- Isotope scattering not included

**Solutions:**
```python
config = ThermalConductivityConfig(
    mesh=(15, 15, 15),  # Increase mesh
    include_isotope=True,  # Include isotope scattering
    include_boundary=True,  # Add boundary scattering
    boundary_size=1e6  # Grain size in Angstrom
)
```

---

## Quick Reference

### File Organization

```
phonon_calculation/
├── 1_relaxation/
│   ├── POSCAR
│   ├── INCAR
│   └── KPOINTS
├── 2_displacements/
│   ├── disp.yaml
│   ├── POSCAR-{001..NNN}
│   ├── run_calculations.sh
│   └── collect_forces.py
├── 3_phonon/
│   ├── force_constants.hdf5
│   ├── band_structure.pdf
│   ├── dos.pdf
│   └── thermal_properties.pdf
└── 4_analysis/
    ├── convergence_tests.pdf
    └── comparison_with_experiment.pdf
```

### Key Parameters Summary

| Parameter | Typical Value | Converged? |
|-----------|---------------|------------|
| ENCUT | 1.3 × ENMAX | Check energy |
| k-mesh | Dense (6×6×6 min) | Check energy |
| Supercell | 2×2×2 to 4×4×4 | Check frequencies |
| Displacement | 0.01 Å | Check force linearity |
| q-mesh (DOS) | 20×20×20 | Check free energy |
| q-mesh (κ) | 11×11×11 odd | Check κ convergence |

---

*Document version: 1.0*
*Last updated: 2026-03-09*
