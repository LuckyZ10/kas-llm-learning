# DFT-LAMMPS Generative AI Module

A comprehensive module for generative modeling of materials using state-of-the-art diffusion models, flow matching, and consistency models.

## Overview

This module provides:

1. **Diffusion Transformers (DiT/ADiT)** - Scalable transformer-based diffusion models for crystals
2. **Riemannian Flow Matching** - Fast generation via continuous normalizing flows
3. **Consistency Models** - Single-step/few-step generation
4. **Conditional Generation** - Property-guided inverse design
5. **Joint Molecular-Crystal Generation** - Unified model for molecules and crystals

## Installation

```bash
# Install dftlammps with generative AI support
pip install dftlammps[generative]

# Or install dependencies manually
pip install torch pymatgen torch_geometric
```

## Quick Start

### 1. Basic Crystal Generation

```python
from dftlammps.generative_ai import CrystalDiT, CrystalDiTConfig
from dftlammps.generative_ai.pretrained import load_model

# Load pretrained model
model = load_model("crystal_dit_base")

# Generate crystals
generated = model.generate(
    batch_size=10,
    num_atoms=20,
    num_steps=100
)

# Convert to pymatgen structures
from dftlammps.generative_ai.data import tensors_to_structure

for i in range(len(generated['atom_types'])):
    structure = tensors_to_structure({
        "atom_types": generated["atom_types"][i],
        "frac_coords": generated["frac_coords"][i],
        "lattice": generated["lattice"][i]
    })
    structure.to(filename=f"generated_{i}.cif")
```

### 2. Conditional Generation

```python
from dftlammps.generative_ai import ConditionalDiffusion, ConditionalConfig

# Create conditional model
cond_config = ConditionalConfig(
    num_properties=3,
    property_names=["band_gap", "formation_energy", "bulk_modulus"],
    use_cfg=True,
    guidance_scale=2.0
)

model = ConditionalDiffusion(base_model, cond_config)

# Generate with target properties
target_properties = torch.tensor([[2.0, -3.0, 50.0]])  # eV, eV/atom, GPa
generated = model.generate(
    batch_size=10,
    num_atoms=20,
    properties=target_properties.expand(10, -1)
)
```

### 3. Fast Generation with Flow Matching

```python
from dftlammps.generative_ai import CrystalFlow, FlowMatchingConfig
from dftlammps.generative_ai.utils import FlowSampler

config = FlowMatchingConfig(num_steps=50)  # Fewer steps than diffusion
model = CrystalFlow(config)

sampler = FlowSampler(model, num_steps=50, method="euler")
generated = sampler.sample(batch_size=10, num_atoms=20)
```

### 4. Inverse Design

```python
from dftlammps.generative_ai.integration import InverseDesignPipeline

# Create pipeline
pipeline = InverseDesignPipeline(
    generative_model=model,
    property_predictors={
        "band_gap": band_gap_predictor,
        "formation_energy": formation_energy_predictor
    }
)

# Design with target properties
designed = pipeline.design(
    target_properties={
        "band_gap": 2.5,
        "formation_energy": -4.0
    }
)
```

## Model Architecture

### CrystalDiT

Diffusion Transformer for crystal generation:

```python
from dftlammps.generative_ai import CrystalDiT, CrystalDiTConfig

config = CrystalDiTConfig(
    latent_dim=512,
    num_layers=12,
    num_heads=8,
    max_atoms=100,
    num_timesteps=1000
)

model = CrystalDiT(config)
```

Key features:
- Transformer-based denoising with adaptive layer norm
- Separate heads for atom types, fractional coordinates, and lattice
- SE(3) equivariance through data augmentation
- Optional property conditioning

### Riemannian Flow Matching

Flow matching on manifolds for faster generation:

```python
from dftlammps.generative_ai import RiemannianFlowMatcher, FlowMatchingConfig

config = FlowMatchingConfig(
    hidden_dim=256,
    num_layers=6,
    num_steps=50  # ODE integration steps
)

model = RiemannianFlowMatcher(config)
```

Key features:
- 3x faster than diffusion with comparable quality
- Straight-line probability paths
- Riemannian geometry for periodic boundary conditions

### Consistency Models

Single-step/few-step generation:

```python
from dftlammps.generative_ai import ConsistencyCrystalModel, ConsistencyConfig

config = ConsistencyConfig(
    hidden_dim=256,
    num_layers=6,
    num_discretization_steps=50
)

model = ConsistencyCrystalModel(config)

# Single-step generation
generated = model.generate(batch_size=10, num_atoms=20, num_steps=1)
```

## Training

### Train a Diffusion Model

```python
from dftlammps.generative_ai.training import DiffusionTrainer
from dftlammps.generative_ai.data import MPDataset, collate_crystal_batch
from torch.utils.data import DataLoader

# Load dataset
dataset = MPDataset(
    api_key="YOUR_MP_API_KEY",
    chemsys="Li-P-S",
    nsites=(5, 50)
)

# DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_crystal_batch
)

# Create trainer
trainer = DiffusionTrainer(
    model=model,
    train_loader=train_loader,
    config={
        "max_epochs": 100,
        "learning_rate": 1e-4,
        "save_every": 10
    }
)

# Train
trainer.train()
```

### Train with Conditional Generation

```python
from dftlammps.generative_ai.training import DiffusionTrainer

# The trainer handles classifier-free guidance training automatically
trainer = DiffusionTrainer(
    model=conditional_model,
    train_loader=train_loader,
    config={
        "max_epochs": 100,
        "use_cfg": True,
        "cfg_dropout": 0.1
    }
)
```

## Evaluation

### Compute Generation Metrics

```python
from dftlammps.generative_ai.utils import CrystalMetrics

metrics = CrystalMetrics()

# Compute all metrics
results = metrics.compute_all(
    generated=generated_structures,
    reference=reference_structures
)

print(f"Validity: {results['validity_strict']:.3f}")
print(f"Uniqueness: {results['uniqueness']:.3f}")
print(f"Novelty: {results['novelty']:.3f}")
print(f"Match rate: {results['match_rate']:.3f}")
```

## Integration with Screening

```python
from dftlammps.generative_ai.integration import GenerativeScreening

# Create screening interface
screening = GenerativeScreening(
    generative_model=model,
    property_predictors={
        "band_gap": m3gnet_predictor,
        "ionic_conductivity": diffusion_predictor
    }
)

# Generate for screening
candidates = screening.generate_for_screening(
    num_structures=1000,
    target_properties={
        "band_gap": (1.5, 2.5),
        "ionic_conductivity": (> 0.1)
    }
)
```

## Pretrained Models

Available pretrained models:

| Model | Description | Size | Parameters |
|-------|-------------|------|------------|
| `crystal_dit_base` | Base CrystalDiT on MP-20 | 150MB | 50M |
| `crystal_dit_large` | Large CrystalDiT | 600MB | 450M |
| `flow_mm_mp20` | FlowMM on MP-20 | 200MB | 80M |
| `consistency_fast` | 1-step consistency | 150MB | 50M |
| `conditional_bandgap` | Band gap targeting | 150MB | 50M |
| `joint_mol_crystal` | Joint generator | 180MB | 60M |

Download and use:

```python
from dftlammps.generative_ai.pretrained import load_model, list_models

# List available models
print(list_models())

# Load model
model = load_model("crystal_dit_base")
```

## Examples

See `examples/` directory:

- `conditional_generation.py` - Conditional generation with property targets
- `flow_matching_demo.py` - Fast generation with flow matching
- `inverse_design_demo.py` - Property-targeted inverse design

Run examples:

```bash
cd dftlammps/generative_ai/examples
python conditional_generation.py
python flow_matching_demo.py
python inverse_design_demo.py
```

## Architecture Details

### DiT Block

The DiT (Diffusion Transformer) block uses adaptive layer normalization conditioned on timestep and properties:

```
x_out = x + gate_msa * Attention(AdaLN(x, c))
x_out = x + gate_mlp * MLP(AdaLN(x_out, c))
```

Where `c` is the conditioning vector (timestep + property embeddings).

### Flow Matching

Flow matching learns a vector field that transforms noise to data:

```
dx/dt = v(x, t)
```

The training objective is:

```
L = E_t ||v_θ(x_t, t) - u_t(x_t|x_1)||^2
```

Where `u_t` is the conditional vector field for the straight-line path.

### Consistency Models

Consistency models learn to map any point on the diffusion trajectory to the origin:

```
f(x_t, t) = x_0
```

This enables single-step generation by directly predicting `x_0` from noise.

## Citation

If you use this module, please cite:

```bibtex
@software{dftlammps_generative,
  title={DFT-LAMMPS Generative AI Module},
  author={DFT-LAMMPS Team},
  year={2026}
}

@inproceedings{peebles2023scalable,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  booktitle={ICCV},
  year={2023}
}

@inproceedings{lipman2023flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and Chen, Ricky TQ and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matt},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{miller2024flowmm,
  title={FlowMM: Generating Materials with Riemannian Flow Matching},
  author={Miller, Brandon K and Chen, Ricky TQ and Sriram, Anuroop and Wood, Benjamin M},
  booktitle={ICML},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This module implements ideas from:
- DiT (Peebles & Xie, 2023)
- ADiT (Joshi et al., 2025)
- FlowMM (Miller et al., 2024)
- CrystalFlow (Luo et al., 2025)
- Consistency Models (Song et al., 2023)
- CDVAE (Xie et al., 2022)
- MatterGen (Zeni et al., 2025)
