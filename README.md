# DFT+LAMMPS Integration Package

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python package for integrating **DFT (Density Functional Theory)** calculations with **classical molecular dynamics simulations** using LAMMPS, enhanced with **machine learning potentials**.

## Features

### Core Capabilities

- **DFT Bridge**: Seamless integration between VASP/Quantum ESPRESSO and LAMMPS
- **ML Potentials**: Support for DeepMD, NEP, and custom neural network potentials
- **MD Simulations**: High-performance molecular dynamics with various ensembles
- **Workflow Orchestration**: End-to-end automation from structure to analysis

### HPC Integration

- Multi-scheduler support (Slurm, PBS, LSF)
- Automatic resource optimization
- Parallel job execution and dependency management

### Applications

- **Battery Materials**: Solid electrolyte screening and ionic conductivity prediction
- **Catalyst Design**: Surface reaction modeling
- **Perovskite Systems**: Phase stability analysis

## Installation

### Quick Install

```bash
pip install dftlammps
```

### Development Install

```bash
git clone https://github.com/example/dftlammps.git
cd dftlammps
pip install -e ".[dev,all]"
```

### Optional Dependencies

```bash
# DeepMD support
pip install dftlammps[deepmd]

# Materials screening
pip install dftlammps[screening]

# Visualization
pip install dftlammps[viz]

# Everything
pip install dftlammps[all]
```

## Quick Start

### 1. DFT to LAMMPS Bridge

```python
from dftlammps import DFTToLAMMPSBridge

# Initialize bridge
bridge = DFTToLAMMPSBridge(
    dft_code="vasp",
    potential_type="buckingham"
)

# Convert DFT data to LAMMPS potential
bridge.run_workflow(
    vasp_dir="dft_calculations/",
    output_dir="lammps_setup/"
)
```

### 2. ML Potential Training (NEP)

```python
from dftlammps import NEPTrainingPipeline, NEPDataConfig, NEPModelConfig

# Setup training
pipeline = NEPTrainingPipeline(
    data_config=NEPDataConfig(type_map=["Li", "S", "P"]),
    model_config=NEPModelConfig(version=4, neuron=50),
    working_dir="nep_training/"
)

# Train model
pipeline.prepare_data(vasp_outcars=["OUTCAR1", "OUTCAR2"])
pipeline.train(gpu_id=0)
```

### 3. Molecular Dynamics Simulation

```python
from dftlammps import MDSimulationRunner, MDConfig
from ase import Atoms

# Setup MD
config = MDConfig(
    ensemble="nvt",
    temperature=300.0,
    pair_style="deepmd",
    potential_file="graph.pb"
)

# Run simulation
runner = MDSimulationRunner(config)
atoms = Atoms(...)  # Your structure
trajectory = runner.run(atoms)
```

### 4. HPC Job Submission

```python
from dftlammps import HPCScheduler, ResourceRequest, JobSpec

# Auto-detect scheduler
scheduler = HPCScheduler.auto_detect()

# Define resources
resources = ResourceRequest(
    num_nodes=2,
    num_cores_per_node=32,
    walltime_hours=48.0,
    partition="gpu"
)

# Create job
job = JobSpec(
    name="dft_calculation",
    executable="python",
    arguments=["run.py"],
    resources=resources
)

# Submit
job_id = scheduler.submit(job)
scheduler.wait_for_job(job_id)
```

### 5. Battery Screening Pipeline

```python
from dftlammps import BatteryScreeningPipeline, BatteryScreeningConfig

# Initialize screening
pipeline = BatteryScreeningPipeline(
    config=BatteryScreeningConfig(ion_type="Li")
)

# Run screening
pipeline.fetch_candidates(formula="Li*x*")
pipeline.run_screening()
top_candidates = pipeline.get_top_candidates(n=10)
```

### 6. Complete Workflow

```python
from dftlammps import IntegratedMaterialsWorkflow, IntegratedWorkflowConfig

# Configure workflow
config = IntegratedWorkflowConfig(
    workflow_name="Li3PS4_study",
    working_dir="./Li3PS4_workflow",
    dft_config={"code": "vasp", "functional": "PBE"},
    ml_config={"framework": "nep"},
    md_config={"temperatures": [300, 500, 700]}
)

# Run complete workflow
workflow = IntegratedMaterialsWorkflow(config)
results = workflow.run()
```

## Documentation

- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Examples](examples/) - Example scripts and tutorials

## Package Structure

```
dftlammps/
├── __init__.py              # Package entry point
├── core/                    # Core modules
│   ├── dft_bridge.py        # DFT/LAMMPS interface
│   ├── ml_potential.py      # ML potential training
│   ├── md_simulation.py     # MD simulation runner
│   └── workflow.py          # Integrated workflows
├── hpc/                     # HPC modules
│   └── scheduler.py         # Job scheduling
├── applications/            # Application cases
│   └── screening.py         # Materials screening
└── utils/                   # Utilities
    ├── checkpoint.py        # Checkpoint management
    └── monitoring.py        # Monitoring dashboard
```

## Requirements

### Core Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- ASE >= 3.22.0
- Pymatgen >= 2022.0.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0

### Optional Dependencies

- **DeepMD**: `deepmd-kit`, `dpdata`
- **Screening**: `matminer`, `dscribe`, `mp-api`
- **Visualization**: `ovito`, `nglview`, `plotly`

See [requirements.txt](requirements.txt) for complete list.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dftlammps --cov-report=html

# Run specific test file
pytest tests/test_core/test_dft_bridge.py -v
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{dftlammps,
  title = {DFT+LAMMPS: Integration of DFT and Classical MD},
  author = {DFT+LAMMPS Integration Team},
  year = {2026},
  url = {https://github.com/example/dftlammps}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/example/dftlammps/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/dftlammps/discussions)
- **Email**: dftlammps@example.com

## Acknowledgments

- ASE (Atomic Simulation Environment)
- Pymatgen
- LAMMPS
- GPUMD (NEP)
- DeepMD-kit

---

**Note**: This is a research software package. Please validate results for production use.
