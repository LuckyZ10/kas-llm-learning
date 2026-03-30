"""
DFTLAMMPS Multiscale Coupling Module
=====================================

A comprehensive framework for multiscale molecular simulations,
from quantum mechanics to coarse-grained models.

## Features

### 1. QM/MM Coupling
- Additive QM/MM coupling schemes
- Support for VASP (QM) + LAMMPS (MM)
- Electrostatic and mechanical embedding
- Link atom methods for covalent boundaries

### 2. Machine Learning Coarse-Graining
- Learnable CG mappings using neural networks
- Force matching for CG force field parameterization
- Autoencoder-based mapping discovery

### 3. Graph Neural Networks
- E(3)-equivariant message passing
- Simultaneous atomistic and CG modeling
- Cross-scale attention mechanisms

### 4. Validation Tools
- Energy conservation checks
- Force consistency validation
- Thermodynamic consistency analysis
- Structure comparison tools

## Quick Start

```python
from dftlammps.multiscale_coupling import VASPLAMMPSCoupling

# Set up QM/MM simulation
qmmm = VASPLAMMPSCoupling(
    vasp_cmd='vasp_std',
    lammps_cmd='lmp'
)

# Define regions
qmmm.set_regions(qm_mask, mm_mask)

# Run calculation
results = qmmm.calculate(positions, elements, mm_types)
```

## Examples

See the `examples/` directory for:
- `ex1_qmmm_water.py`: QM/MM simulation of water
- `ex2_ml_coarse_graining.py`: ML coarse-graining
- `ex3_gnn_force_field.py`: GNN force field training
- `ex4_validation.py`: Cross-scale validation
- `ex5_multiscale_gnn.py`: Multiscale GNN

## Installation

```bash
# Install required dependencies
pip install numpy scipy scikit-learn

# Optional: PyTorch for ML features
pip install torch

# Optional: ASE for structure manipulation
pip install ase
```

## Citation

If you use this module in your research, please cite:

```
@software{dftlammps_multiscale,
  title = {DFTLAMMPS Multiscale Coupling Module},
  year = {2025}
}
```

## License

MIT License

## Authors

DFTLAMMPS Team
