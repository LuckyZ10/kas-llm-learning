"""
AI Module for Materials Discovery
=================================

This module provides state-of-the-art AI/ML capabilities for materials
discovery, integrating with the DFT+LAMMPS workflow.

Submodules:
- generative_models: CDVAE, DiffCSP, MatterGen for structure generation
- property_predictor: CGCNN, MegNet, ALIGNN for property prediction
- bayesian_optimizer: Bayesian optimization for materials discovery
- active_discovery: Active learning workflows

Example Usage:
    from dftlammps.ai import StructureGenerator, PropertyPredictor
    from dftlammps.ai import BayesianOptimizer, ActiveDiscovery
    
    # Generate structures
    generator = StructureGenerator('cdvae')
    structures = generator.generate(num_structures=100)
    
    # Predict properties
    predictor = PropertyPredictor('cgcnn')
    predictions = predictor.predict(structures)
    
    # Run Bayesian optimization
    optimizer = BayesianOptimizer()
    result = optimizer.optimize(composition_space=['Li3PS4', 'Li2S'])

Author: DFT+LAMMPS AI Team
"""

# Import main classes from submodules
from .generative_models import (
    StructureGenerator,
    CrystalStructure,
    GenerativeModelConfig,
    CDVAE,
    DiffCSP,
    MatterGen,
    load_pretrained_model,
    generate_structures_for_screening,
)

from .property_predictor import (
    PropertyPredictor,
    PropertyPredictorConfig,
    MaterialGraph,
    CGCNN,
    MegNet,
    ALIGNN,
    TransformerModel,
    PretrainedModelLoader,
    load_pretrained_predictor,
    AtomFeatureEncoder,
)

from .bayesian_optimizer import (
    BayesianOptimizer,
    BayesianOptimizerConfig,
    OptimizationResult,
    GaussianProcessSurrogate,
    ExpectedImprovement,
    UpperConfidenceBound,
    optimize_materials,
    batch_bayesian_optimization,
)

from .active_discovery import (
    ActiveDiscovery,
    ActiveDiscoveryConfig,
    ActiveDiscoveryResult,
    DiscoveryIteration,
    SamplingStrategy,
    UncertaintyEstimator,
    EnsembleUncertainty,
    MCDropoutUncertainty,
    QueryByCommittee,
    DiversitySampler,
    ActiveDiscoveryPipeline,
    run_active_discovery_for_battery_materials,
    run_active_discovery_for_catalysts,
    integrate_with_high_throughput_screening,
)

# Version
__version__ = "1.0.0"

__all__ = [
    # Generative Models
    "StructureGenerator",
    "CrystalStructure",
    "GenerativeModelConfig",
    "CDVAE",
    "DiffCSP",
    "MatterGen",
    "load_pretrained_model",
    "generate_structures_for_screening",
    
    # Property Predictors
    "PropertyPredictor",
    "PropertyPredictorConfig",
    "MaterialGraph",
    "CGCNN",
    "MegNet",
    "ALIGNN",
    "TransformerModel",
    "PretrainedModelLoader",
    "load_pretrained_predictor",
    "AtomFeatureEncoder",
    
    # Bayesian Optimization
    "BayesianOptimizer",
    "BayesianOptimizerConfig",
    "OptimizationResult",
    "GaussianProcessSurrogate",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "optimize_materials",
    "batch_bayesian_optimization",
    
    # Active Discovery
    "ActiveDiscovery",
    "ActiveDiscoveryConfig",
    "ActiveDiscoveryResult",
    "DiscoveryIteration",
    "SamplingStrategy",
    "UncertaintyEstimator",
    "EnsembleUncertainty",
    "MCDropoutUncertainty",
    "QueryByCommittee",
    "DiversitySampler",
    "ActiveDiscoveryPipeline",
    "run_active_discovery_for_battery_materials",
    "run_active_discovery_for_catalysts",
    "integrate_with_high_throughput_screening",
]
