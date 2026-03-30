# Physics-AI Module for DFT-LAMMPS Pipeline

## Overview

This module integrates physics constraints deeply into AI models for molecular dynamics simulations and potential energy surface fitting. It implements state-of-the-art physics-informed machine learning techniques including PINNs, Neural Operators (DeepONet, FNO), Physics-Informed GNNs, and Symbolic Regression.

## Features

### Core Components

- **Physics Constraint Layer**: Pluggable layer that enforces physical laws in any neural network
- **Conservation Laws**: Implementation of energy, momentum, mass, and angular momentum conservation
- **Adaptive Weighting**: Automatically balance multiple physics constraints during training

### Neural Network Models

#### 1. Physics-Informed Neural Networks (PINNs)
- General PDE solver using automatic differentiation
- SIREN (Sinusoidal Representation Networks) activation
- Adaptive weighting for loss terms
- Domain decomposition for complex geometries
- Built-in PDE formulations (Burgers, Navier-Stokes, Heat, Schrödinger, Poisson)

#### 2. Deep Operator Networks (DeepONet)
- Learns solution operators between infinite-dimensional function spaces
- Physics-informed training mode
- Separable DeepONet for dimensionality reduction
- Multi-output and attention variants

#### 3. Fourier Neural Operators (FNO)
- Resolution-invariant PDE solutions
- 1D, 2D, and 3D implementations
- Physics-informed FNO
- Multi-scale and adaptive variants

#### 4. Physics-Informed Graph Neural Networks
- E(n) Equivariant GNN (EGNN)
- Momentum-conserving message passing
- Hamiltonian GNN
- Equivariant Transformer

### Symbolic Regression

- **Multiple Backends**: PySR, gplearn, AI Feynman
- **Conservation Law Discovery**: Automatically discover conserved quantities
- **Expression Validation**: Verify discovered equations against physics constraints
- **Export**: Convert to LaTeX, Python functions

### Physics Law Validator

- **Conservation Tests**: Energy, momentum, angular momentum
- **Symmetry Tests**: Translational, rotational, time-reversal
- **Physical Constraints**: Positivity, action-reaction, boundedness
- **Trajectory Validation**: Check long-term stability

### MD Potential Fitting

- Multi-fidelity training (DFT + experimental data)
- Physics constraint enforcement
- Active learning for data selection
- Export to LAMMPS format
- Uncertainty quantification

## Installation

```bash
# Core dependencies
pip install torch numpy scipy matplotlib

# For symbolic regression
pip install pysr gplearn

# For GNN models
pip install torch-geometric

# Optional dependencies
pip install sympy scikit-learn
```

## Quick Start

### PINN for PDEs

```python
from dftlammps.physics_ai.models import PhysicsInformedNN

# Define PDE
def heat_equation(x, u, derivatives, alpha=1.0):
    u_t = derivatives['first'][:, 0, 0]
    u_xx = derivatives['second'][:, 0, 1, 1]
    return u_t - alpha * u_xx

# Create model
model = PhysicsInformedNN(
    input_dim=2,  # (t, x)
    output_dim=1,  # u(t, x)
    hidden_dims=[64, 64, 64],
    pde_fn=heat_equation
)

# Train
losses = model.compute_loss(
    x_data=x_data, u_data=u_data,
    x_collocation=x_collocation
)
```

### DeepONet for Solution Operators

```python
from dftlammps.physics_ai.models import DeepONet

model = DeepONet(
    branch_input_dim=100,  # Sensor points
    trunk_input_dim=1,     # Evaluation coordinate
    output_dim=1,
    branch_hidden_dims=[128, 128],
    trunk_hidden_dims=[128, 128]
)

# Predict
output = model(u_values, y_coordinates)
```

### Physics-Informed GNN for MD

```python
from dftlammps.physics_ai.models import PhysicsInformedGNN

model = PhysicsInformedGNN(
    node_dim=5,      # Number of atom types
    hidden_dim=128,
    n_layers=4,
    output_type='both'  # Energy and forces
)

# Forward pass
output = model(node_attr=atom_types, pos=positions)
energy = output['energy']
forces = output['forces']
```

### Symbolic Regression

```python
from dftlammps.physics_ai.symbolic import SymbolicRegressionEngine

# Create engine
engine = SymbolicRegressionEngine(backend='pysr')

# Fit
best_expr = engine.fit(X, y, variable_names=['x', 'v'])

print(f"Discovered: {best_expr.expression}")
print(f"LaTeX: {engine.to_latex()}")
```

### Physics Validation

```python
from dftlammps.physics_ai.validators import PhysicsLawValidator

validator = PhysicsLawValidator(tolerance=1e-6)

# Validate model
results = validator.validate_model(model, test_data)
validator.print_report()

# Validate trajectory
trajectory_results = validator.validate_trajectory(trajectory, dt=0.001)
```

### MD Potential Fitting

```python
from dftlammps.physics_ai.integration import MDPotentialFitter

# Create fitter
fitter = MDPotentialFitter(
    model_type='egnn',
    physics_constraints=['energy', 'force']
)

# Create and train model
fitter.create_model(n_atom_types=5)
fitter.fit(train_data, val_data, n_epochs=100)

# Predict
predictions = fitter.predict(positions, atom_types)

# Export to LAMMPS
fitter.export_to_lammps('potential.pt')
```

## Examples

See `examples/` directory for complete examples:

- `example_pinn_harmonic_oscillator.py`: PINN for 1D harmonic oscillator
- `example_deeponet.py`: DeepONet for solution operators
- `example_fno.py`: FNO for 2D Darcy flow
- `example_physics_validator.py`: Physics validation examples
- `example_md_potential.py`: MD potential fitting

## Project Structure

```
dftlammps/physics_ai/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── physics_layer.py      # Physics constraint layer
│   └── conservation.py        # Conservation laws
├── models/
│   ├── __init__.py
│   ├── pinns.py              # Physics-Informed Neural Networks
│   ├── deeponet.py           # Deep Operator Networks
│   ├── fno.py                # Fourier Neural Operators
│   └── physics_gnn.py        # Physics-Informed GNNs
├── symbolic/
│   ├── __init__.py
│   └── regression_engine.py  # Symbolic regression
├── validators/
│   ├── __init__.py
│   └── physics_validator.py  # Physics law validation
├── integration/
│   ├── __init__.py
│   └── md_potential.py       # MD potential fitting
├── tests/
│   └── test_physics_ai.py
└── examples/
    ├── example_pinn_harmonic_oscillator.py
    ├── example_deeponet.py
    ├── example_fno.py
    ├── example_physics_validator.py
    └── example_md_potential.py
```

## Code Statistics

- Total lines: ~3,500
- Core physics layer: ~400 lines
- Conservation laws: ~600 lines
- PINNs: ~600 lines
- DeepONet: ~500 lines
- FNO: ~500 lines
- Physics GNN: ~500 lines
- Symbolic regression: ~500 lines
- Physics validator: ~450 lines
- MD integration: ~500 lines

## References

### PINNs
- Raissi et al., "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations", JCP 2019
- Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020

### Neural Operators
- Lu et al., "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators", Nature MI 2021
- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021

### Physics-Informed GNNs
- Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021
- Batzner et al., "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials", Nature Communications 2022

### Symbolic Regression
- Cranmer et al., "Discovering Symbolic Models from Deep Learning with Inductive Biases", NeurIPS 2020
- Udrescu & Tegmark, "AI Feynman: A physics-inspired method for symbolic regression", Science Advances 2020

## License

MIT License
