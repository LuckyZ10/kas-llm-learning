# DFT+LAMMPS API Reference

Complete API documentation for the `dftlammps` package.

## Table of Contents

- [Package Overview](#package-overview)
- [Core Module](#core-module)
  - [DFT Bridge](#dft-bridge)
  - [ML Potential](#ml-potential)
  - [MD Simulation](#md-simulation)
  - [Workflow](#workflow)
- [HPC Module](#hpc-module)
- [Applications Module](#applications-module)
- [Utils Module](#utils-module)

---

## Package Overview

```python
import dftlammps

# Check version
print(dftlammps.__version__)  # '1.0.0'

# Access main components
from dftlammps import IntegratedMaterialsWorkflow, HPCScheduler
```

---

## Core Module

### DFT Bridge

The `dft_bridge` module provides interfaces between DFT codes (VASP, Quantum ESPRESSO) and LAMMPS.

#### `VASPParserConfig`

Configuration class for VASP output parsing.

```python
from dftlammps.core import VASPParserConfig

config = VASPParserConfig(
    extract_energy=True,
    extract_forces=True,
    extract_stress=True,
    extract_positions=True,
    filter_unconverged=True,
    energy_threshold=100.0  # eV/atom
)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `extract_energy` | bool | True | Extract total energies |
| `extract_forces` | bool | True | Extract atomic forces |
| `extract_stress` | bool | True | Extract stress tensors |
| `extract_positions` | bool | True | Extract atomic positions |
| `extract_velocities` | bool | False | Extract velocities |
| `extract_magmom` | bool | False | Extract magnetic moments |
| `filter_unconverged` | bool | True | Filter unconverged calculations |
| `energy_threshold` | float | 100.0 | Energy threshold for outlier filtering |

#### `ForceFieldConfig`

Configuration for force field fitting.

```python
from dftlammps.core import ForceFieldConfig

config = ForceFieldConfig(
    ff_type="buckingham",  # buckingham, morse, lj, eam, snap, nnp
    elements=["Li", "S", "P"],
    cutoff=6.0,  # Angstrom
    fit_method="least_squares"
)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `ff_type` | str | "buckingham" | Force field type |
| `elements` | List[str] | [] | List of element symbols |
| `cutoff` | float | 6.0 | Cutoff distance in Angstrom |
| `charge_dict` | Dict[str, float] | {} | Element charges |
| `fit_method` | str | "least_squares" | Fitting method |
| `regularization` | float | 0.01 | Regularization strength |
| `max_iterations` | int | 1000 | Maximum fitting iterations |

#### `LAMMPSInputConfig`

Configuration for LAMMPS input generation.

```python
from dftlammps.core import LAMMPSInputConfig

config = LAMMPSInputConfig(
    units="metal",
    atom_style="atomic",
    pair_style="buck/coul/long",
    ensemble="nvt",
    temperature=300.0,
    timestep=1.0,
    nsteps=100000
)
```

#### `VASPDataExtractor`

Extracts data from VASP output files.

```python
from dftlammps.core import VASPDataExtractor, VASPParserConfig

extractor = VASPDataExtractor(config=VASPParserConfig())

# Extract from OUTCAR
data = extractor.extract_from_outcar("path/to/OUTCAR")

# Extract from multiple calculations
all_data = extractor.extract_from_directory("path/to/calculations/")

# Access extracted data
print(data['energies'])      # List of total energies
print(data['forces'])        # List of force arrays
print(data['structures'])    # List of ASE Atoms objects
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `extract_from_outcar` | `(outcar_path: str) -> Dict` | Extract data from single OUTCAR |
| `extract_from_directory` | `(directory: str, pattern: str = "OUTCAR") -> List[Dict]` | Batch extraction |
| `to_deepmd` | `(output_dir: str) -> None` | Export to DeepMD format |
| `to_nep` | `(output_file: str) -> None` | Export to NEP format |

#### `ForceFieldFitter`

Fits classical force field parameters to DFT data.

```python
from dftlammps.core import ForceFieldFitter, ForceFieldConfig

fitter = ForceFieldFitter(config=ForceFieldConfig(ff_type="buckingham"))

# Add training data
fitter.add_training_data(structures, energies, forces)

# Fit parameters
result = fitter.fit()

# Evaluate fit
rmse_energy = fitter.evaluate_energy_rmse()
rmse_force = fitter.evaluate_force_rmse()

# Export potential file
fitter.export_lammps_potential("potential.mod")
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_training_data` | `(structures, energies, forces) -> None` | Add training data |
| `fit` | `() -> Dict` | Perform parameter fitting |
| `predict` | `(structure: Atoms) -> Tuple[float, np.ndarray]` | Predict energy and forces |
| `evaluate_energy_rmse` | `() -> float` | Compute energy RMSE |
| `evaluate_force_rmse` | `() -> float` | Compute force RMSE |
| `export_lammps_potential` | `(filename: str) -> None` | Export LAMMPS potential |

#### `LAMMPSInputGenerator`

Generates LAMMPS input files.

```python
from dftlammps.core import LAMMPSInputGenerator, LAMMPSInputConfig
from ase import Atoms

generator = LAMMPSInputGenerator(config=LAMMPSInputConfig())

# Generate input for structure
atoms = Atoms(...)
input_file = generator.generate(atoms, output_file="in.lammps")
```

#### `DFTToLAMMPSBridge`

Main bridge class connecting DFT and LAMMPS workflows.

```python
from dftlammps.core import DFTToLAMMPSBridge

bridge = DFTToLAMMPSBridge(
    dft_code="vasp",
    potential_type="buckingham"
)

# Full workflow: DFT → Force Field → LAMMPS
bridge.run_workflow(
    vasp_dir="dft_calculations/",
    output_dir="lammps_setup/"
)
```

---

### ML Potential

The `ml_potential` module provides interfaces for training neural network potentials.

#### `NEPDataConfig`

Configuration for NEP training data preparation.

```python
from dftlammps.core import NEPDataConfig

config = NEPDataConfig(
    vasp_outcars=["OUTCAR1", "OUTCAR2"],
    energy_threshold=50.0,
    force_threshold=50.0,
    train_ratio=0.9,
    type_map=["Li", "S", "P"]
)
```

#### `NEPModelConfig`

Configuration for NEP model architecture.

```python
from dftlammps.core import NEPModelConfig

config = NEPModelConfig(
    model_type=0,  # 0=PES, 1=dipole, 2=polarizability
    type_list=["Li", "S", "P"],
    version=4,  # NEP version
    cutoff_radial=6.0,
    cutoff_angular=4.0,
    n_max_radial=4,
    n_max_angular=4,
    basis_size_radial=8,
    basis_size_angular=8,
    neuron=50,
    batch_size=1000,
    learning_rate=0.001
)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | int | 0 | Model type (0=PES, 1=dipole, 2=polarizability) |
| `type_list` | List[str] | [] | Element type list |
| `version` | int | 4 | NEP version (2, 3, or 4) |
| `cutoff_radial` | float | 6.0 | Radial cutoff in Angstrom |
| `cutoff_angular` | float | 4.0 | Angular cutoff in Angstrom |
| `n_max_radial` | int | 4 | Radial descriptor max order |
| `n_max_angular` | int | 4 | Angular descriptor max order |
| `basis_size_radial` | int | 8 | Radial basis size |
| `basis_size_angular` | int | 8 | Angular basis size |
| `l_max_3body` | int | 4 | 3-body max angular momentum |
| `l_max_4body` | int | 0 | 4-body max angular momentum |
| `l_max_5body` | int | 0 | 5-body max angular momentum |
| `neuron` | int | 50 | Number of neurons in hidden layer |
| `batch_size` | int | 1000 | Training batch size |
| `learning_rate` | float | 0.001 | Initial learning rate |
| `max_steps` | int | 1000000 | Maximum training steps |

#### `NEPTrainingPipeline`

Complete pipeline for NEP model training.

```python
from dftlammps.core import NEPTrainingPipeline, NEPDataConfig, NEPModelConfig

# Initialize pipeline
pipeline = NEPTrainingPipeline(
    data_config=NEPDataConfig(),
    model_config=NEPModelConfig(),
    working_dir="nep_training/"
)

# Prepare training data
pipeline.prepare_data(vasp_outcars=["OUTCAR1", "OUTCAR2"])

# Generate NEP input
pipeline.generate_nep_in()

# Run training
pipeline.train(gpu_id=0)

# Validate model
results = pipeline.validate()

# Export to LAMMPS
pipeline.export_for_lammps("nep_potential/")
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `prepare_data` | `(vasp_outcars: List[str]) -> None` | Prepare training data from VASP |
| `generate_nep_in` | `() -> str` | Generate nep.in input file |
| `train` | `(gpu_id: int = 0) -> None` | Run NEP training |
| `validate` | `() -> Dict` | Validate trained model |
| `export_for_lammps` | `(output_dir: str) -> None` | Export for LAMMPS usage |
| `get_loss_history` | `() -> pd.DataFrame` | Get training loss history |

---

### MD Simulation

The `md_simulation` module provides molecular dynamics simulation capabilities.

#### `MDConfig`

Configuration for MD simulations.

```python
from dftlammps.core import MDConfig

config = MDConfig(
    ensemble="nvt",           # nve, nvt, npt
    temperature=300.0,        # K
    pressure=None,            # atm (for NPT)
    timestep=1.0,             # fs
    nsteps=100000,
    nsteps_equil=10000,
    pair_style="deepmd",      # deepmd, snap, tersoff, etc.
    potential_file="graph.pb",
    working_dir="./md_run",
    nprocs=4
)
```

#### `MDTrajectory`

Container for MD trajectory data.

```python
from dftlammps.core import MDTrajectory

trajectory = MDTrajectory()

# Access trajectory data
positions = trajectory.positions      # List of position arrays
velocities = trajectory.velocities   # List of velocity arrays
energies = trajectory.energies       # Dict of energy components
temperatures = trajectory.temperatures
pressures = trajectory.pressures
time = trajectory.time
```

#### `MDSimulationRunner`

Runner for MD simulations using LAMMPS.

```python
from dftlammps.core import MDSimulationRunner, MDConfig
from ase import Atoms

# Setup and run simulation
runner = MDSimulationRunner(config=MDConfig())
atoms = Atoms(...)  # Initial structure
trajectory = runner.run(atoms, potential_file="graph.pb")
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `(atoms: Atoms, potential_file: str) -> MDTrajectory` | Run simulation |

#### `MDTrajectoryAnalyzer`

Analyzer for MD trajectory data.

```python
from dftlammps.core import MDTrajectoryAnalyzer

analyzer = MDTrajectoryAnalyzer(trajectory)

# Compute radial distribution function
r, g_r = analyzer.compute_rdf(bin_width=0.1, r_max=10.0)

# Compute mean square displacement
time, msd = analyzer.compute_msd(atom_indices=[0, 1, 2])

# Compute diffusion coefficient
D = analyzer.compute_diffusion_coefficient(atom_indices=[0, 1, 2])

# Compute ionic conductivity
sigma = analyzer.compute_ionic_conductivity(
    D=D,
    temperature=300.0,
    concentration=1e22,  # cm^-3
    charge=1.0
)

# Get summary statistics
summary = analyzer.get_summary()
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute_rdf` | `(bin_width: float, r_max: float) -> Tuple[np.ndarray, np.ndarray]` | Radial distribution function |
| `compute_msd` | `(atom_indices: Optional[List[int]]) -> Tuple[np.ndarray, np.ndarray]` | Mean square displacement |
| `compute_diffusion_coefficient` | `(atom_indices, fit_range) -> float` | Diffusion coefficient in cm²/s |
| `compute_ionic_conductivity` | `(D, temperature, concentration, charge) -> float` | Ionic conductivity in S/cm |
| `get_summary` | `() -> Dict` | Analysis summary |

---

### Workflow

The `workflow` module provides end-to-end workflow management.

#### `IntegratedWorkflowConfig`

Configuration for integrated materials workflow.

```python
from dftlammps.core import (
    IntegratedWorkflowConfig,
    MaterialsProjectConfig,
    DFTStageConfig,
    MLPotentialConfig,
    MDStageConfig,
    AnalysisConfig
)

config = IntegratedWorkflowConfig(
    workflow_name="my_workflow",
    working_dir="./workflow_output",
    mp_config=MaterialsProjectConfig(api_key="your_key"),
    dft_config=DFTStageConfig(code="vasp", functional="PBE"),
    ml_config=MLPotentialConfig(framework="deepmd"),
    md_config=MDStageConfig(temperatures=[300, 500, 700]),
    analysis_config=AnalysisConfig(compute_diffusion=True)
)
```

#### `IntegratedMaterialsWorkflow`

Main workflow class for end-to-end materials simulation.

```python
from dftlammps.core import IntegratedMaterialsWorkflow, IntegratedWorkflowConfig

# Create workflow
workflow = IntegratedMaterialsWorkflow(
    config=IntegratedWorkflowConfig()
)

# Run full workflow
results = workflow.run()

# Or run specific stages
workflow.fetch_structures(query={"formula": "Li3PS4"})
workflow.run_dft_calculations()
workflow.train_ml_potential()
workflow.run_md_simulations()
results = workflow.analyze_results()

# Save/Load workflow state
workflow.save_checkpoint("checkpoint.pkl")
workflow.load_checkpoint("checkpoint.pkl")

# Generate report
workflow.generate_report("workflow_report.html")
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `() -> Dict` | Run complete workflow |
| `fetch_structures` | `(query: Dict) -> List[Atoms]` | Fetch structures from Materials Project |
| `run_dft_calculations` | `() -> None` | Run DFT calculations |
| `train_ml_potential` | `() -> None` | Train ML potential |
| `run_md_simulations` | `() -> None` | Run MD simulations |
| `analyze_results` | `() -> Dict` | Analyze and compile results |
| `save_checkpoint` | `(path: str) -> None` | Save workflow state |
| `load_checkpoint` | `(path: str) -> None` | Load workflow state |
| `generate_report` | `(output_file: str) -> None` | Generate HTML report |

---

## HPC Module

### Scheduler

The `scheduler` module provides interfaces to HPC job schedulers.

#### `SchedulerType`

Enumeration of supported schedulers.

```python
from dftlammps.hpc import SchedulerType

SchedulerType.SLURM   # Slurm Workload Manager
SchedulerType.PBS     # Portable Batch System
SchedulerType.LSF     # Load Sharing Facility
SchedulerType.LOCAL   # Local execution
```

#### `JobStatus`

Enumeration of job statuses.

```python
from dftlammps.hpc import JobStatus

JobStatus.PENDING     # Job is queued
JobStatus.RUNNING     # Job is running
JobStatus.COMPLETED   # Job completed successfully
JobStatus.FAILED      # Job failed
JobStatus.CANCELLED   # Job was cancelled
JobStatus.TIMEOUT     # Job timed out
JobStatus.UNKNOWN     # Status unknown
```

#### `ResourceRequest`

Resource request specification.

```python
from dftlammps.hpc import ResourceRequest

resources = ResourceRequest(
    num_nodes=2,
    num_cores_per_node=32,
    num_gpus=4,
    gpu_type="a100",
    memory_gb=128,
    walltime_hours=48.0,
    partition="gpu",
    modules=["cuda", "gcc"],
    conda_env="mlpot"
)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_slurm` | `() -> str` | Convert to Slurm directives |
| `to_pbs` | `() -> str` | Convert to PBS directives |
| `to_lsf` | `() -> str` | Convert to LSF directives |

#### `JobSpec`

Complete job specification.

```python
from dftlammps.hpc import JobSpec, ResourceRequest

job = JobSpec(
    name="dft_calculation",
    executable="python",
    arguments=["run_dft.py", "--config", "config.yaml"],
    resources=ResourceRequest(num_nodes=2, num_cores_per_node=32),
    working_dir="./calculation",
    environment_vars={"OMP_NUM_THREADS": "4"},
    dependencies=[]
)
```

#### `HPCScheduler`

Base class for HPC schedulers.

```python
from dftlammps.hpc import HPCScheduler, JobSpec

# Auto-detect scheduler
scheduler = HPCScheduler.auto_detect()

# Submit job
job_id = scheduler.submit(job_spec)

# Check status
status = scheduler.get_job_status(job_id)

# Wait for completion
scheduler.wait_for_job(job_id)

# Cancel job
scheduler.cancel_job(job_id)

# List jobs
jobs = scheduler.list_jobs()

# Get queue info
queue_info = scheduler.get_queue_info()
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `auto_detect` | `() -> HPCScheduler` | Auto-detect scheduler type |
| `submit` | `(job: JobSpec) -> str` | Submit job to queue |
| `submit_batch` | `(jobs: List[JobSpec]) -> List[str]` | Submit multiple jobs |
| `get_job_status` | `(job_id: str) -> JobStatus` | Get job status |
| `cancel_job` | `(job_id: str) -> bool` | Cancel a job |
| `wait_for_job` | `(job_id: str, poll_interval: int = 60) -> JobStatus` | Wait for job completion |
| `list_jobs` | `(user: Optional[str]) -> List[JobInfo]` | List jobs |
| `get_queue_info` | `() -> Dict` | Get queue information |
| `estimate_wait_time` | `(resources: ResourceRequest) -> timedelta` | Estimate wait time |

#### `ParallelOptimizer`

Parallel optimization using HPC resources.

```python
from dftlammps.hpc import ParallelOptimizer

optimizer = ParallelOptimizer(scheduler=scheduler)

# Run parallel optimization
results = optimizer.optimize(
    objective_function=my_function,
    parameter_space=param_space,
    n_workers=10,
    max_iterations=100
)
```

---

## Applications Module

### Screening

The `screening` module provides high-throughput screening capabilities.

#### `BatteryScreeningConfig`

Configuration for battery material screening.

```python
from dftlammps.applications import BatteryScreeningConfig

config = BatteryScreeningConfig(
    ion_type="Li",            # Li or Na
    max_entries=1000,
    featurize=True,
    ml_screening=True,
    dft_validation=True
)
```

#### `FeatureEngineer`

Feature engineering for materials screening.

```python
from dftlammps.applications import FeatureEngineer

featurizer = FeatureEngineer()

# Extract structural features
features = featurizer.extract_features(structures)

# Compute SOAP descriptors
soap_features = featurizer.compute_soap(structures, rcut=6.0, nmax=8, lmax=6)

# Get feature names
feature_names = featurizer.get_feature_names()
```

#### `PerformancePredictor`

ML-based performance prediction.

```python
from dftlammps.applications import PerformancePredictor

predictor = PerformancePredictor(model_type="xgboost")

# Train model
predictor.train(features, targets)

# Make predictions
predictions = predictor.predict(new_features)

# Evaluate model
metrics = predictor.evaluate(test_features, test_targets)
```

#### `BatteryScreeningPipeline`

Complete screening pipeline for battery materials.

```python
from dftlammps.applications import BatteryScreeningPipeline, BatteryScreeningConfig

# Initialize pipeline
pipeline = BatteryScreeningPipeline(config=BatteryScreeningConfig())

# Fetch candidate structures
pipeline.fetch_candidates(formula="Li*x*")

# Run screening
pipeline.run_screening()

# Get top candidates
top_candidates = pipeline.get_top_candidates(n=10)

# Export results
pipeline.export_results("screening_results.csv")
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `fetch_candidates` | `(formula: str) -> List[Structure]` | Fetch candidate structures |
| `run_screening` | `() -> None` | Run screening workflow |
| `get_top_candidates` | `(n: int) -> pd.DataFrame` | Get top N candidates |
| `export_results` | `(filename: str) -> None` | Export results to file |

---

## Utils Module

### Checkpoint

The `checkpoint` module provides workflow checkpointing.

#### `CheckpointManager`

Manages workflow checkpoints for fault tolerance.

```python
from dftlammps.utils import CheckpointManager

manager = CheckpointManager(checkpoint_dir="./checkpoints")

# Save checkpoint
manager.save(state={"step": 10, "data": data}, name="workflow_step_10")

# Load checkpoint
state = manager.load(name="workflow_step_10")

# List checkpoints
checkpoints = manager.list_checkpoints()

# Clean old checkpoints
manager.clean_old_checkpoints(keep_last=5)
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `save` | `(state: Dict, name: str) -> str` | Save checkpoint |
| `load` | `(name: str) -> Dict` | Load checkpoint |
| `list_checkpoints` | `() -> List[str]` | List available checkpoints |
| `clean_old_checkpoints` | `(keep_last: int) -> None` | Clean old checkpoints |
| `exists` | `(name: str) -> bool` | Check if checkpoint exists |

### Monitoring

The `monitoring` module provides monitoring capabilities.

#### `MonitoringDashboard`

Real-time monitoring dashboard for workflows.

```python
from dftlammps.utils import MonitoringDashboard

dashboard = MonitoringDashboard(port=8080)

# Start dashboard
dashboard.start()

# Update metrics
dashboard.update_metric("energy", -123.45)
dashboard.update_metric("step", 100)

# Add plot data
dashboard.add_plot_data("loss", x=100, y=0.01)

# Stop dashboard
dashboard.stop()
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `start` | `() -> None` | Start dashboard server |
| `stop` | `() -> None` | Stop dashboard server |
| `update_metric` | `(name: str, value: float) -> None` | Update metric value |
| `add_plot_data` | `(plot_name: str, x: float, y: float) -> None` | Add plot data point |
| `log_event` | `(message: str, level: str = "info") -> None` | Log event |

---

## Data Types

### Common Type Aliases

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ase import Atoms

# Structure representations
StructureType = Union[Atoms, 'Structure']  # ASE Atoms or Pymatgen Structure

# Energy/Force data
EnergyArray = np.ndarray  # Shape: (n_structures,)
ForceArray = np.ndarray   # Shape: (n_structures, n_atoms, 3)
StressArray = np.ndarray  # Shape: (n_structures, 6)

# Configuration types
ConfigDict = Dict[str, any]
```

---

## Exceptions

```python
class DFTLAMMPSException(Exception):
    """Base exception for dftlammps package."""
    pass

class DFTCalculationError(DFTLAMMPSException):
    """Raised when DFT calculation fails."""
    pass

class MLPotentialError(DFTLAMMPSException):
    """Raised when ML potential training fails."""
    pass

class MDSimulationError(DFTLAMMPSException):
    """Raised when MD simulation fails."""
    pass

class SchedulerError(DFTLAMMPSException):
    """Raised when job scheduling fails."""
    pass
```

---

## Constants

```python
# Physical constants
BOLTZMANN_KB = 8.617333262e-5  # eV/K
EV_TO_KJ_MOL = 96.485  # Conversion factor
ANGSTROM_TO_BOHR = 1.88973  # Length conversion

# Default values
DEFAULT_DFT_ENCUT = 520  # eV
DEFAULT_DFT_KDENSITY = 0.25  # k-points per Å^-1
DEFAULT_MD_TIMESTEP = 1.0  # fs
DEFAULT_MD_TEMPERATURE = 300.0  # K
DEFAULT_CUTOFF = 6.0  # Å
```
