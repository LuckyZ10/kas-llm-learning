#!/usr/bin/env python3
"""
AI Materials Discovery Examples
================================

This script demonstrates the usage of the dftlammps.ai module for
materials discovery tasks.

Run with: python ai_examples.py
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, '/root/.openclaw/workspace')

from dftlammps.ai import (
    StructureGenerator, CrystalStructure, GenerativeModelConfig,
    PropertyPredictor, PropertyPredictorConfig,
    BayesianOptimizer, BayesianOptimizerConfig,
    ActiveDiscovery, ActiveDiscoveryConfig, SamplingStrategy,
    ExpectedImprovement, UpperConfidenceBound,
    load_pretrained_predictor
)


def example_1_basic_generation():
    """Example 1: Basic structure generation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Structure Generation")
    print("=" * 70)
    
    # Create generator configuration
    config = GenerativeModelConfig(
        model_type='random',  # Use 'random' for demo (no training needed)
        num_structures=10,
        max_atoms=30
    )
    
    # Create structure generator
    generator = StructureGenerator('random', config)
    
    # Generate structures
    print("\nGenerating 10 random structures...")
    structures = generator.generate(num_structures=10)
    
    print(f"Generated {len(structures)} structures")
    for i, struct in enumerate(structures[:5]):
        print(f"  Structure {i+1}: {len(struct.atomic_numbers)} atoms, "
              f"composition={struct.composition}")
    
    return structures


def example_2_property_prediction(structures):
    """Example 2: Property prediction with CGCNN."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Property Prediction with CGCNN")
    print("=" * 70)
    
    # Create predictor
    config = PropertyPredictorConfig(
        model_type='cgcnn',
        hidden_dim=64,
        num_layers=2,
        output_dim=1
    )
    predictor = PropertyPredictor('cgcnn', config)
    
    print(f"\nPredictor created: {predictor.model_type}")
    print(f"Model parameters: {sum(p.numel() for p in predictor.model.parameters()):,}")
    
    # For demo, we use random predictions
    # In practice, you would train the model first:
    # predictor.train(train_structures, train_targets, num_epochs=200)
    
    print("\nMaking predictions...")
    predictions = predictor.predict(structures[:5])
    
    print(f"Predictions for 5 structures:")
    for i, pred in enumerate(predictions):
        print(f"  Structure {i+1}: {pred[0]:.4f}")
    
    return predictor


def example_3_bayesian_optimization():
    """Example 3: Bayesian optimization for materials."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Bayesian Optimization")
    print("=" * 70)
    
    # Create optimizer configuration
    config = BayesianOptimizerConfig(
        num_init_samples=5,
        num_iterations=10,
        acquisition_type='ei',
        num_candidates=20,
        batch_size=1
    )
    
    # Create components
    gen_config = GenerativeModelConfig(model_type='random')
    generator = StructureGenerator('random', gen_config)
    
    pred_config = PropertyPredictorConfig(model_type='cgcnn')
    predictor = PropertyPredictor('cgcnn', pred_config)
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        config=config,
        structure_generator=generator,
        property_predictor=predictor
    )
    
    print(f"\nOptimizer created with:")
    print(f"  Initial samples: {config.num_init_samples}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Acquisition: {config.acquisition_type}")
    
    # Run optimization (with mock DFT)
    print("\nRunning optimization (reduced for demo)...")
    result = optimizer.optimize(
        composition_space=['Li3PS4', 'Li2S', 'Na2S'],
        seed_structures=None
    )
    
    print(f"\nOptimization complete!")
    print(f"Best value found: {result.best_y:.4f}")
    if result.best_structure:
        print(f"Best structure composition: {result.best_structure.composition}")
    print(f"Total iterations: {result.num_iterations}")
    
    return result


def example_4_active_learning():
    """Example 4: Active learning workflow."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Active Learning Workflow")
    print("=" * 70)
    
    # Create configuration
    config = ActiveDiscoveryConfig(
        num_iterations=5,
        samples_per_iteration=3,
        sampling_strategy=SamplingStrategy.HYBRID,
        uncertainty_weight=0.4,
        diversity_weight=0.3,
        greedy_weight=0.3,
        retrain_frequency=2
    )
    
    # Create components
    generator = StructureGenerator('random', GenerativeModelConfig())
    predictor = PropertyPredictor('cgcnn', PropertyPredictorConfig())
    
    # Create discovery workflow
    discovery = ActiveDiscovery(
        config=config,
        structure_generator=generator,
        property_predictor=predictor
    )
    
    print(f"\nActive discovery created with:")
    print(f"  Strategy: {config.sampling_strategy.value}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Samples/iteration: {config.samples_per_iteration}")
    
    # Run discovery (reduced for demo)
    print("\nRunning active discovery...")
    result = discovery.run(
        composition_space=['Li3PS4', 'Li2S', 'Na2S'],
        num_candidates_per_iteration=15
    )
    
    print(f"\nDiscovery complete!")
    print(f"Total structures explored: {len(result.all_structures)}")
    print(f"Total DFT calculations: {result.total_dft_calculations}")
    
    # Show top discoveries
    print("\nTop 3 discoveries:")
    for struct, value in result.get_best_structures(3):
        print(f"  {struct.composition}: {value:.4f}")
    
    return result


def example_5_custom_acquisition():
    """Example 5: Custom acquisition functions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Custom Acquisition Functions")
    print("=" * 70)
    
    # Expected Improvement
    ei = ExpectedImprovement(xi=0.01)
    print("\nExpected Improvement (EI):")
    print("  xi (exploration): 0.01")
    print("  Use case: Balanced exploration/exploitation")
    
    # Upper Confidence Bound
    ucb = UpperConfidenceBound(beta=2.0)
    print("\nUpper Confidence Bound (UCB):")
    print("  beta (exploration): 2.0")
    print("  Use case: Tunable exploration")
    
    print("\nTo use in optimizer:")
    print("  config = BayesianOptimizerConfig(acquisition_type='ei')")
    print("  # or")
    print("  config = BayesianOptimizerConfig(acquisition_type='ucb', beta_ucb=2.0)")


def example_6_multi_objective():
    """Example 6: Multi-objective optimization."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Multi-Objective Optimization")
    print("=" * 70)
    
    print("""
For multi-objective optimization, use MultiObjectiveBayesianOptimizer:

    from dftlammps.ai import MultiObjectiveBayesianOptimizer
    
    # Define objectives
    def conductivity(structure):
        return predictor.predict([structure])[0][0]
    
    def stability(structure):
        return -predictor.predict([structure])[0][1]  # Negative for max
    
    # Create optimizer
    optimizer = MultiObjectiveBayesianOptimizer(config)
    
    # Optimize both objectives
    result = optimizer.optimize([conductivity, stability])
    
    # Get Pareto front
    pareto_front = result.pareto_front
    print(f"Pareto front shape: {pareto_front.shape}")
    """)


def example_7_integration_with_dft():
    """Example 7: Integration with DFT workflow."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Integration with DFT Pipeline")
    print("=" * 70)
    
    print("""
To integrate with DFT calculations:

    from dftlammps.ai import ActiveDiscovery
    
    # Define DFT evaluator
    def dft_evaluator(structure):
        # Your DFT calculation here
        # Convert structure to DFT input
        # Run calculation
        # Return property value
        
        # Example with VASP:
        from dftlammps.core.dft_bridge import DFTToLAMMPSBridge
        
        bridge = DFTToLAMMPSBridge()
        result = bridge.run_dft(structure)
        
        return result.energy
    
    # Create discovery with DFT
    discovery = ActiveDiscovery(
        config=config,
        structure_generator=generator,
        property_predictor=predictor,
        dft_evaluator=dft_evaluator
    )
    
    # Run discovery with DFT validation
    result = discovery.run(composition_space=['Li3PS4', 'Li2S'])
    """)


def example_8_case_studies():
    """Example 8: Running case studies."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Application Case Studies")
    print("=" * 70)
    
    print("""
Run complete case studies:

    from dftlammps.applications.ai_discovery import (
        run_solid_electrolyte_discovery,
        run_hea_catalyst_discovery
    )
    
    # Case 1: Solid Electrolyte Discovery
    results = run_solid_electrolyte_discovery(
        target_ion='Li',
        num_iterations=30,
        output_dir='./li_electrolyte_results'
    )
    
    # Case 2: HEA Catalyst Discovery
    results = run_hea_catalyst_discovery(
        target_reaction='ORR',
        num_iterations=30,
        output_dir='./hea_catalyst_results'
    )
    
    # Access results
    for discovery in results['top_discoveries'][:5]:
        print(f"{discovery['composition']}: {discovery['predicted_value']:.4f}")
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DFT+LAMMPS AI MODULE - USAGE EXAMPLES")
    print("=" * 70)
    
    try:
        # Example 1: Structure Generation
        structures = example_1_basic_generation()
        
        # Example 2: Property Prediction
        predictor = example_2_property_prediction(structures)
        
        # Example 3: Bayesian Optimization
        result_bo = example_3_bayesian_optimization()
        
        # Example 4: Active Learning
        result_al = example_4_active_learning()
        
        # Example 5: Custom Acquisition
        example_5_custom_acquisition()
        
        # Example 6: Multi-Objective
        example_6_multi_objective()
        
        # Example 7: DFT Integration
        example_7_integration_with_dft()
        
        # Example 8: Case Studies
        example_8_case_studies()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
