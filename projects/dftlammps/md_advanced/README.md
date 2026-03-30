# Advanced Molecular Dynamics Module

This module provides comprehensive tools for advanced molecular dynamics simulations including enhanced sampling, free energy calculations, rare event analysis, and reaction mechanism studies.

## Overview

### Module Structure

```
dftlammps/md_advanced/
├── enhanced_sampling.py      # Umbrella sampling, Metadynamics, REMD, TAD
├── free_energy.py            # FEP, TI, BAR, WHAM, MBAR
├── rare_events.py            # NEB, String Method, Dimer Method, TST
└── reaction_analysis.py      # Path search, rate constants, KMC

dftlammps/md_analysis_advanced/
├── dynamic_heterogeneity.py  # Glass transition analysis
├── structural_analysis.py    # Ring/Voronoi/Bond order analysis
└── correlation_analysis.py   # Time correlation functions

dftlammps/mlip_training/
├── mace_training.py          # MACE equivariant potential training
├── chgnet_training.py        # CHGNet GNN training
├── orb_training.py           # Orb potential training
└── __init__.py               # Unified MLIP interface

dftlammps/applications/
├── case_ion_migration/       # Ion migration barriers (NEB+ML)
├── case_phase_transition/    # Phase transitions (Metadynamics)
└── case_catalytic_mechanism/ # Catalytic reactions (Path search)
```

## Enhanced Sampling Methods

### Umbrella Sampling

Compute potentials of mean force (PMF) along reaction coordinates:

```python
from dftlammps.md_advanced import UmbrellaSampling, UmbrellaSamplingConfig

config = UmbrellaSamplingConfig(
    collective_variables=[
        {'name': 'distance', 'type': 'distance', 'group1': [0], 'group2': [1]}
    ],
    reaction_path=[[2.0 + 0.1*i] for i in range(20)],
    kappa=50.0,
    n_windows=20,
    temperature=300.0,
    nsteps_per_window=50000
)

us = UmbrellaSampling(config)
results = us.run_all_windows(atoms, parallel=True)
pmf = us.analyze_windows()
```

### Metadynamics

Explore free energy surfaces with adaptive bias:

```python
from dftlammps.md_advanced import Metadynamics, MetadynamicsConfig

config = MetadynamicsConfig(
    collective_variables=[
        {'name': 'cv1', 'type': 'distance', 'group1': [0], 'group2': [1]},
        {'name': 'cv2', 'type': 'angle', 'group1': [0], 'group2': [1], 'group3': [2]}
    ],
    gaussian_height=1.2,  # kcal/mol
    gaussian_width=0.2,
    hill_frequency=500,
    well_tempered=True,
    bias_factor=10.0,
    nsteps=1000000
)

mtd = Metadynamics(config)
mtd.run_lammps_with_plumed(atoms)

# Reconstruct FES
grid, fes = mtd.reweight_free_energy()
```

### Replica Exchange MD

Overcome energy barriers through temperature replica exchange:

```python
from dftlammps.md_advanced import REMD, REMDConfig

config = REMDConfig(
    n_replicas=8,
    t_min=300.0,
    t_max=800.0,
    exchange_frequency=1000,
    nsteps=100000
)

remd = REMD(config)
result = remd.run(atoms)

# Analyze acceptance rates
rates = remd.compute_acceptance_rates()
```

## Free Energy Calculations

### Thermodynamic Integration

```python
from dftlammps.md_advanced import ThermodynamicIntegration, TIConfig

config = TIConfig(
    n_windows=20,
    lambda_schedule=np.linspace(0, 1, 20),
    nsteps_per_window=50000
)

ti = ThermodynamicIntegration(config)
result = ti.run(window_dirs)
print(f"ΔG = {result['delta_g']:.4f} ± {result['error']:.4f} kcal/mol")
```

### Bennett Acceptance Ratio

```python
from dftlammps.md_advanced import BennettAcceptanceRatio, BARConfig

config = BARConfig(n_windows=20, tolerance=1e-6)
bar = BennettAcceptanceRatio(config)
result = bar.run(window_dirs)

print(f"BAR ΔG = {result['delta_g']:.4f} ± {result['error']:.4f} kcal/mol")
```

### WHAM

Weighted Histogram Analysis for umbrella sampling:

```python
from dftlammps.md_advanced import WHAM, WHAMConfig

config = WHAMConfig(tolerance=1e-8, temperature=300.0)
wham = WHAM(config)

window_data = [
    {'cv_values': np.loadtxt(f'window_{i}/cv.dat'),
     'center': centers[i],
     'kappa': 50.0}
    for i in range(n_windows)
]

result = wham.compute_pmf(window_data)
wham.save_pmf('pmf.dat')
```

## Rare Event Methods

### Nudged Elastic Band

Find minimum energy paths and transition states:

```python
from dftlammps.md_advanced import NEB, NEBConfig

config = NEBConfig(
    n_images=10,
    k_spring=1.0,
    climb=True,
    fmax=0.05,
    neb_method='ase'  # or 'lammps', 'vasp'
)

neb = NEB(config)
result = neb.run(initial_atoms, final_atoms)

print(f"Barrier: {result['barrier']:.3f} eV")
print(f"TS index: {result['ts_index']}")
```

### String Method

```python
from dftlammps.md_advanced import StringMethod, StringMethodConfig

config = StringMethodConfig(
    n_images=20,
    dt=0.1,
    smoothing_factor=0.1,
    max_steps=1000
)

sm = StringMethod(config)
result = sm.run(initial_atoms, final_atoms)
```

### Dimer Method

Find saddle points without knowledge of final state:

```python
from dftlammps.md_advanced import DimerMethod, DimerConfig

config = DimerConfig(
    dimer_distance=0.01,
    rotation_threshold=1e-4,
    max_steps=500
)

dimer = DimerMethod(config)
result = dimer.run(initial_atoms)

ts_candidate = result['ts_candidate']
mode_direction = result['mode_direction']
```

## Reaction Analysis

### Automatic Path Search

```python
from dftlammps.md_advanced import ReactionPathSearcher, ReactionPathSearchConfig

config = ReactionPathSearchConfig(
    search_method='neb',
    n_initials=10,
    barrier_threshold=3.0,
    max_paths=5
)

searcher = ReactionPathSearcher(config)
paths = searcher.search_all_paths(reactant, product)
```

### Rate Constant Calculation

```python
from dftlammps.md_advanced import RateConstantCalculator, RateConstantConfig

config = RateConstantConfig(
    method='tst',
    temperature_range=(300, 1000),
    include_tunneling=True,
    tunneling_method='wigner'
)

calculator = RateConstantCalculator(config)
rates = calculator.compute_rates_vs_temperature(barrier, ts_analysis)
arrhenius_params = calculator.fit_arrhenius(rates)
```

### Kinetic Monte Carlo

```python
from dftlammps.md_advanced import KMCPreprocessor, KMCConfig

config = KMCConfig(
    max_events=1000,
    hop_distance=3.5,
    temperature=300.0,
    simulation_time=1.0
)

kmc = KMCPreprocessor(config)
events = kmc.identify_diffusion_events(atoms, mobile_species=['Li'])
kmc.generate_kmc_input('kmc_input.json')
```

## Advanced Analysis

### Dynamic Heterogeneity

```python
from dftlammps.md_analysis_advanced import DynamicHeterogeneityAnalyzer

analyzer = DynamicHeterogeneityAnalyzer(config)
results = analyzer.analyze_dynamics(trajectory, box=cell)

# Access results
msd = results['msd']
alpha2 = results['non_gaussian']
chi4 = results['chi4']
mobile_fraction = results['mobile_particles']['fraction']
```

### Structural Analysis

```python
from dftlammps.md_analysis_advanced import StructuralAnalyzer

analyzer = StructuralAnalyzer()
results = analyzer.full_analysis(positions, symbols=symbols)

# Ring statistics
ring_stats = results['rings']

# Voronoi analysis
voronoi = results['voronoi']
ico_fraction = voronoi['fraction_icosahedral']

# Bond-orientational order
boo = results['boo']
structure_type = boo['structure_type']
```

### Time Correlation Functions

```python
from dftlammps.md_analysis_advanced import CorrelationAnalysisSuite

suite = CorrelationAnalysisSuite()

# VACF and diffusion
lag, vacf = suite.vacf_analyzer.compute_vacf(velocities)
D = suite.vacf_analyzer.compute_diffusion_coefficient(vacf, timestep=1.0)

# Stress autocorrelation and viscosity
lag, sacf = suite.sacf_analyzer.compute_sacf(stress_tensor)
eta = suite.sacf_analyzer.compute_viscosity(sacf, volume, temperature)
```

## MLIP Training

### Unified Interface

```python
from dftlammps.mlip_training import (
    UnifiedMLIPTrainer, UnifiedTrainingConfig, MLIPType
)

config = UnifiedTrainingConfig(
    mlip_type=MLIPType.MACE,
    train_data='train.extxyz',
    valid_data='valid.extxyz',
    max_epochs=1000,
    cutoff=5.0
)

trainer = UnifiedMLIPTrainer(config)
result = trainer.train()

# Load trained model
from dftlammps.mlip_training import load_model
mlip = load_model('mace', model_path='mace_model.pt')
```

### MACE Training

```python
from dftlammps.mlip_training.mace_training import (
    MACETrainer, MACERunConfig,
    MACEDataConfig, MACEArchitectureConfig, MACETrainingConfig
)

data_config = MACEDataConfig(
    train_file='train.extxyz',
    valid_file='valid.extxyz',
    cutoff=5.0
)

arch_config = MACEArchitectureConfig(
    hidden_irreps='128x0e + 128x1o',
    num_interactions=2
)

train_config = MACETrainingConfig(
    max_num_epochs=1000,
    batch_size=5,
    lr=0.01
)

run_config = MACERunConfig(data_config, arch_config, train_config)
trainer = MACETrainer(run_config)
result = trainer.run_training()
```

## Application Cases

### Ion Migration

```python
from dftlammps.applications.case_ion_migration import (
    IonMigrationNEB, IonMigrationConfig
)

config = IonMigrationConfig(
    ion_symbol='Li',
    mlip_type='chgnet',
    n_images=7,
    temperature=300
)

migration = IonMigrationNEB(config)
result = migration.run_full_analysis(structure, start_site=0, end_site=1)

print(f"Barrier: {result['migration_analysis']['barrier_ev']:.3f} eV")
print(f"D: {result['migration_analysis']['diffusion_coefficient']:.2e} cm²/s")
```

### Phase Transitions

```python
from dftlammps.applications.case_phase_transition import (
    PhaseTransitionAnalyzer, PhaseTransitionConfig
)

config = PhaseTransitionConfig(
    initial_phase='solid.xyz',
    transition_type='solid_liquid',
    sampling_method='metadynamics',
    temperature=1800
)

analyzer = PhaseTransitionAnalyzer(config)
result = analyzer.run_full_analysis()

fes = result['fes_analysis']
print(f"Barrier: {fes['barrier']:.3f} eV")
```

### Catalytic Reactions

```python
from dftlammps.applications.case_catalytic_mechanism import (
    CatalyticReactionConfig, CatalystBuilder, ReactionPathwayAnalyzer
)

config = CatalyticReactionConfig(
    catalyst_type='Pt',
    reaction_type='co_oxidation',
    reaction_temperature=500
)

builder = CatalystBuilder(config)
analyzer = ReactionPathwayAnalyzer(config)

surface = builder.build_surface()
result = analyzer.compute_barrier_neb(initial_state, final_state)
```

## References

1. Torrie & Valleau (1977) - Umbrella Sampling
2. Laio & Parrinello (2002) - Metadynamics
3. Sugita & Okamoto (1999) - REMD
4. Voter (1997) - Hyperdynamics/TAD
5. Bennett (1976) - BAR
6. Kumar et al. (1992) - WHAM
7. Mills, Jónsson & Schenter (1995) - NEB
8. Henkelman & Jónsson (1999) - Dimer Method
9. E, Ren & Vanden-Eijnden (2002) - String Method
10. Batatia et al. (2022) - MACE
11. Deng et al. (2023) - CHGNet

## Installation Requirements

```bash
# Core dependencies
pip install ase numpy scipy pandas scikit-learn

# For enhanced sampling with PLUMED
pip install plumed

# For MLIP training
pip install mace chgnet orb-models

# For analysis
pip install networkx
```

## License

MIT License - See LICENSE file for details
