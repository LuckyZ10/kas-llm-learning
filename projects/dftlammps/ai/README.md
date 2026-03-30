# AI Materials Discovery Module Documentation

## Overview

The `dftlammps.ai` module provides state-of-the-art AI/ML capabilities for autonomous materials discovery. It integrates generative models, graph neural networks for property prediction, Bayesian optimization, and active learning workflows with the existing DFT+MD pipeline.

## Module Structure

```
dftlammps/ai/
├── __init__.py                 # Module exports
├── generative_models.py        # Structure generation (CDVAE, DiffCSP, MatterGen)
├── property_predictor.py       # GNN property prediction (CGCNN, MegNet, ALIGNN)
├── bayesian_optimizer.py       # Bayesian optimization for materials discovery
└── active_discovery.py         # Active learning workflows
```

## Quick Start

### 1. Structure Generation

```python
from dftlammps.ai import StructureGenerator, GenerativeModelConfig

# Configure generator
config = GenerativeModelConfig(
    model_type='cdvae',
    latent_dim=256,
    hidden_dim=512,
    num_structures=100
)

# Create generator
generator = StructureGenerator('cdvae', config)

# Generate structures for a target composition
structures = generator.generate(
    num_structures=100,
    target_composition='Li3PS4',
    num_atoms_range=(10, 50)
)

print(f"Generated {len(structures)} structures")
```

### 2. Property Prediction

```python
from dftlammps.ai import PropertyPredictor, PropertyPredictorConfig

# Create predictor
config = PropertyPredictorConfig(
    model_type='cgcnn',
    hidden_dim=128,
    num_layers=3,
    output_dim=1
)
predictor = PropertyPredictor('cgcnn', config)

# Train on known data
predictor.train(train_structures, train_targets, num_epochs=200)

# Predict properties for new structures
predictions = predictor.predict(new_structures)
```

### 3. Bayesian Optimization

```python
from dftlammps.ai import BayesianOptimizer, BayesianOptimizerConfig

# Configure optimizer
config = BayesianOptimizerConfig(
    num_init_samples=10,
    num_iterations=50,
    acquisition_type='ei',
    num_candidates=100
)

# Create optimizer
optimizer = BayesianOptimizer(
    config=config,
    structure_generator=generator,
    property_predictor=predictor
)

# Run optimization
result = optimizer.optimize(
    composition_space=['Li3PS4', 'Li2S', 'Li3N', 'Li7P3S11']
)

print(f"Best value found: {result.best_y:.4f}")
print(f"Best composition: {result.best_structure.composition}")
```

### 4. Active Discovery Workflow

```python
from dftlammps.ai import ActiveDiscovery, ActiveDiscoveryConfig, SamplingStrategy

# Configure discovery
config = ActiveDiscoveryConfig(
    num_iterations=30,
    samples_per_iteration=5,
    sampling_strategy=SamplingStrategy.HYBRID,
    retrain_frequency=3
)

# Create discovery workflow
discovery = ActiveDiscovery(
    config=config,
    structure_generator=generator,
    property_predictor=predictor,
    dft_evaluator=dft_function  # Your DFT evaluation function
)

# Run discovery
result = discovery.run(composition_space=['Li3PS4', 'Li2S', 'Li7P3S11'])

# Get top discoveries
top_structures = result.get_best_structures(10)
```

## Generative Models

### CDVAE (Crystal Diffusion Variational Autoencoder)

CDVAE learns a continuous latent space of crystal structures using:
- Graph neural network encoder
- Diffusion-based decoder for structure generation
- Handles periodic boundary conditions natively

```python
config = GenerativeModelConfig(model_type='cdvae')
generator = StructureGenerator('cdvae', config)
```

### DiffCSP (Diffusion Model for Crystal Structure Prediction)

DiffCSP generates structures conditioned on composition:
- Composition-conditional generation
- Stable structure prediction
- Good for known chemical systems

```python
config = GenerativeModelConfig(model_type='diffcsp')
generator = StructureGenerator('diffcsp', config)
```

### MatterGen

MatterGen uses a transformer-based architecture:
- Self-attention over atomic positions
- Iterative refinement
- Property-conditioned generation

```python
config = GenerativeModelConfig(model_type='mattergen')
generator = StructureGenerator('mattergen', config)
```

## Property Predictors

### CGCNN (Crystal Graph Convolutional Neural Network)

Reference: Xie & Grossman, Phys. Rev. Lett. 2018

```python
config = PropertyPredictorConfig(model_type='cgcnn')
predictor = PropertyPredictor('cgcnn', config)
```

Features:
- Graph convolutions on crystal graphs
- RBF expansion of interatomic distances
- Mean and max pooling for readout

### MegNet (MatErials Graph Network)

Reference: Chen et al., Phys. Rev. Materials 2019

```python
config = PropertyPredictorConfig(model_type='megnet')
predictor = PropertyPredictor('megnet', config)
```

Features:
- Global, node, and edge updates
- Set2Set readout for global features
- Multi-fidelity learning

### ALIGNN (Atomistic Line Graph Neural Network)

Reference: Choudhary & DeCost, npj Comput. Mater. 2021

```python
config = PropertyPredictorConfig(model_type='alignn')
predictor = PropertyPredictor('alignn', config)
```

Features:
- Line graph representation
- Angle features between bonds
- State-of-the-art accuracy on Materials Project

## Pretrained Models

### Loading Pretrained Universal Potentials

```python
from dftlammps.ai import load_pretrained_predictor

# Load M3GNet
m3gnet = load_pretrained_predictor('m3gnet')

# Load CHGNet
chgnet = load_pretrained_predictor('chgnet')

# Predict energy
energy = m3gnet.predict_energy(structure)
forces = m3gnet.predict_forces(structure)
```

Supported models:
- M3GNet: Universal potential for materials
- CHGNet: Universal potential with charge information
- ORB: Orbital-based representation
- Equiformer V2: Equivariant transformer

## Bayesian Optimization

### Acquisition Functions

- **Expected Improvement (EI)**: Balances exploration and exploitation
- **Upper Confidence Bound (UCB)**: Tunable exploration parameter
- **Probability of Improvement (PI)**: Probability of improvement over best

```python
from dftlammps.ai import ExpectedImprovement, UpperConfidenceBound

# Use EI
config = BayesianOptimizerConfig(acquisition_type='ei', xi=0.01)

# Use UCB
config = BayesianOptimizerConfig(acquisition_type='ucb', beta_ucb=2.0)
```

### Multi-Objective Optimization

```python
from dftlammps.ai import MultiObjectiveBayesianOptimizer

# Define multiple objectives
def objective1(structure):
    return ionic_conductivity(structure)

def objective2(structure):
    return -band_gap(structure)  # Negative for maximization

optimizer = MultiObjectiveBayesianOptimizer(config)
result = optimizer.optimize([objective1, objective2])

# Get Pareto front
pareto_front = result.pareto_front
```

## Active Learning

### Sampling Strategies

```python
from dftlammps.ai import SamplingStrategy

# Uncertainty sampling
config = ActiveDiscoveryConfig(sampling_strategy=SamplingStrategy.UNCERTAINTY)

# Diversity sampling
config = ActiveDiscoveryConfig(sampling_strategy=SamplingStrategy.DIVERSITY)

# Hybrid (recommended)
config = ActiveDiscoveryConfig(
    sampling_strategy=SamplingStrategy.HYBRID,
    uncertainty_weight=0.4,
    diversity_weight=0.3,
    greedy_weight=0.3
)
```

### Uncertainty Estimation Methods

```python
from dftlammps.ai import EnsembleUncertainty, MCDropoutUncertainty, QueryByCommittee

# Ensemble uncertainty
estimator = EnsembleUncertainty(num_models=5)

# MC Dropout
estimator = MCDropoutUncertainty(num_iterations=20)

# Query by Committee
estimator = QueryByCommittee(model_types=['cgcnn', 'megnet', 'alignn'])
```

## Application Examples

### Case 1: Solid Electrolyte Discovery

```python
from dftlammps.applications.ai_discovery import run_solid_electrolyte_discovery

results = run_solid_electrolyte_discovery(
    target_ion='Li',
    num_iterations=30,
    output_dir='./li_electrolyte_discovery'
)

# Access top discoveries
for discovery in results['top_discoveries'][:5]:
    print(f"{discovery['composition']}: {discovery['predicted_value']:.4f}")
```

### Case 2: High-Entropy Alloy Catalyst Design

```python
from dftlammps.applications.ai_discovery import run_hea_catalyst_discovery

results = run_hea_catalyst_discovery(
    target_reaction='ORR',  # Oxygen Reduction Reaction
    num_iterations=30,
    output_dir='./hea_catalyst_discovery'
)
```

## Integration with DFT+MD Pipeline

### Using AI for Pre-screening

```python
from dftlammps.ai import integrate_with_high_throughput_screening

# Pre-screen candidates with AI
selected = integrate_with_high_throughput_screening(
    screening_pipeline=screening,
    active_discovery=discovery,
    selection_ratio=0.1  # Select top 10%
)

# Run DFT only on selected structures
for structure in selected:
    dft_result = run_dft(structure)
```

### Continuous Learning

```python
# Start with initial model
predictor = PropertyPredictor('cgcnn')
predictor.train(initial_data, initial_targets)

# Iterative improvement
for iteration in range(10):
    # Generate candidates
    candidates = generator.generate(100)
    
    # Predict with current model
    predictions = predictor.predict(candidates)
    uncertainties = uncertainty_estimator.estimate(candidates, predictor)
    
    # Select for DFT
    selected = select_by_uncertainty(candidates, uncertainties, n=5)
    
    # Run DFT
    dft_results = [run_dft(s) for s in selected]
    
    # Update training data
    train_data.extend(selected)
    train_targets.extend(dft_results)
    
    # Retrain model
    predictor.train(train_data, train_targets)
```

## Configuration Reference

### GenerativeModelConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_type | str | 'cdvae' | Model type: cdvae, diffcsp, mattergen |
| latent_dim | int | 256 | Latent space dimension |
| hidden_dim | int | 512 | Hidden layer dimension |
| num_layers | int | 6 | Number of graph layers |
| num_heads | int | 8 | Number of attention heads |
| max_atoms | int | 100 | Maximum atoms per structure |
| cutoff_distance | float | 8.0 | Graph construction cutoff (Å) |

### PropertyPredictorConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_type | str | 'cgcnn' | Model: cgcnn, megnet, alignn, transformer |
| hidden_dim | int | 128 | Hidden layer dimension |
| num_layers | int | 3 | Number of layers |
| output_dim | int | 1 | Number of target properties |
| cutoff_distance | float | 8.0 | Graph cutoff (Å) |
| learning_rate | float | 1e-3 | Learning rate |

### BayesianOptimizerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| surrogate_type | str | 'gp' | Surrogate: gp, deep_gp |
| acquisition_type | str | 'ei' | Acquisition: ei, ucb, pi |
| num_init_samples | int | 10 | Initial random samples |
| num_iterations | int | 50 | Optimization iterations |
| batch_size | int | 1 | Parallel evaluations |
| num_candidates | int | 100 | Candidates per iteration |

### ActiveDiscoveryConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| sampling_strategy | SamplingStrategy | HYBRID | Sampling strategy |
| num_iterations | int | 20 | Discovery iterations |
| samples_per_iteration | int | 5 | Samples per iteration |
| retrain_frequency | int | 3 | Model retrain interval |
| uncertainty_method | str | 'ensemble' | Uncertainty: ensemble, mcdropout |

## Advanced Usage

### Custom Objective Functions

```python
def custom_objective(structure):
    # Your custom evaluation
    property1 = predictor1.predict([structure])[0]
    property2 = predictor2.predict([structure])[0]
    
    # Combine properties
    return property1 * 0.6 + property2 * 0.4

optimizer = BayesianOptimizer(
    config=config,
    structure_generator=generator,
    property_predictor=predictor
)

result = optimizer.optimize(objective_function=custom_objective)
```

### Custom Structure Featurization

```python
class CustomFeatureExtractor:
    def extract(self, structure):
        # Your custom features
        features = []
        features.append(len(structure.atomic_numbers))
        features.append(np.mean(structure.atomic_numbers))
        # ... more features
        return np.array(features)

discovery = ActiveDiscovery(
    config=config,
    structure_generator=generator,
    property_predictor=predictor,
    feature_extractor=CustomFeatureExtractor()
)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size or num_candidates
2. **Poor Convergence**: Increase num_init_samples or adjust learning rate
3. **Slow Generation**: Use smaller num_structures or simpler models
4. **Import Errors**: Install optional dependencies: `pip install gpytorch matgl chgnet`

### Performance Tips

- Use GPU for training and inference
- Cache structure graphs to avoid recomputation
- Use batched predictions for large candidate pools
- Pre-train on large datasets before fine-tuning

## References

1. Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Phys. Rev. Lett., 120(14), 145301.

2. Chen, C., Ye, W., Zuo, Y., Zheng, C., & Ong, S. P. (2019). Graph networks as a universal machine learning framework for molecules and crystals. Chemistry of Materials, 31(9), 3564-3572.

3. Choudhary, K., & DeCost, B. (2021). Atomistic line graph neural network for improved materials property predictions. npj Computational Materials, 7(1), 185.

4. Xie, Y., et al. (2022). Crystal diffusion variational autoencoder for periodic material generation. ICLR 2022.

5. Zeni, C., et al. (2023). A generative model for inorganic materials design. Nature, 624(7990), 102-108.
