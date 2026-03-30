#!/usr/bin/env python3
"""
World Model and Internal Simulation Examples
=============================================

Comprehensive examples demonstrating:
1. Material World Model usage
2. Imagination Engine (counterfactual, hypothetical, creative)
3. Model Predictive Control for synthesis planning
4. Internal Simulation (fast physics, dreams, mental simulation)

Run with: python world_model_examples.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dftlammps.world_model import (
    MaterialWorldModel,
    MaterialState,
    MaterialAction,
    ActionType,
    Transition,
    WorldModelConfig,
    ImaginationEngine,
    ModelPredictiveController,
    MPCConfig,
    TrajectoryCost,
    OptimizationMethod,
    MaterialImaginationCases,
    SynthesisPathPlanner
)

from dftlammps.internal_sim import (
    FastPhysicsSimulator,
    SimulationConfig,
    AbstractRepresentationLearner,
    RepresentationConfig,
    DreamGenerator,
    DreamConfig,
    MentalSimulationEngine
)


def example_1_world_model_training():
    """Example 1: Training a world model on synthetic data"""
    print("="*60)
    print("Example 1: World Model Training")
    print("="*60)
    
    # Create synthetic training data
    def create_synthetic_dataset(n_samples=500):
        transitions = []
        
        for i in range(n_samples):
            # Random material state
            state = MaterialState(
                state_id=f"state_{i}",
                timestamp=i * 0.1,
                temperature=np.random.uniform(300, 1000),
                pressure=np.random.uniform(1, 10),
                total_energy=np.random.randn() * 5 - 10,
                potential_energy=np.random.randn() * 3 - 8,
                kinetic_energy=np.random.randn() * 2,
                features=np.random.randn(20)
            )
            
            # Random action
            action = MaterialAction(
                action_id=f"action_{i}",
                action_type=np.random.choice(list(ActionType)),
                magnitude=np.random.randn(),
                duration=np.random.uniform(10, 100)
            )
            
            # Simulate next state (simple dynamics)
            delta = 0.1 * state.to_vector() + 0.05 * action.to_vector()[:20]
            next_features = state.to_vector() + delta + 0.01 * np.random.randn(20)
            
            next_state = MaterialState(
                state_id=f"state_{i+1}",
                timestamp=state.timestamp + 0.1,
                temperature=state.temperature * (1 + 0.01 * action.magnitude),
                pressure=state.pressure * (1 + 0.005 * action.magnitude),
                total_energy=state.total_energy + np.random.randn() * 0.1,
                potential_energy=state.potential_energy + np.random.randn() * 0.05,
                kinetic_energy=state.kinetic_energy + np.random.randn() * 0.05,
                features=next_features
            )
            
            reward = -abs(next_state.total_energy - (-10.0))
            done = next_state.temperature > 2000
            
            transitions.append(Transition(
                state=state, action=action, next_state=next_state,
                reward=reward, done=done
            ))
        
        return transitions
    
    print("Creating synthetic training data...")
    transitions = create_synthetic_dataset(500)
    print(f"Created {len(transitions)} transition samples")
    
    # Create and train world model
    config = WorldModelConfig(
        state_dim=20,
        action_dim=10,
        latent_dim=16,
        hidden_dims=[128, 128, 64],
        model_type='ensemble',
        ensemble_size=3,
        num_epochs=20,
        batch_size=32,
        learning_rate=1e-3
    )
    
    print(f"\nInitializing world model...")
    print(f"  Model type: {config.model_type}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Ensemble size: {config.ensemble_size}")
    
    world_model = MaterialWorldModel(config)
    model_info = world_model.get_model_info()
    print(f"  Total parameters: {model_info['total_parameters']:,}")
    
    print(f"\nTraining world model...")
    history = world_model.train(transitions, verbose=True)
    
    print(f"\nTraining completed!")
    print(f"  Initial loss: {history['train_loss'][0]:.4f}")
    print(f"  Final loss: {history['train_loss'][-1]:.4f}")
    
    # Test prediction
    test_state = transitions[0].state
    test_action = transitions[0].action
    
    result = world_model.predict(test_state, test_action, return_uncertainty=True)
    next_state, reward, done, info = result
    
    print(f"\nPrediction test:")
    print(f"  Current temperature: {test_state.temperature:.2f} K")
    print(f"  Predicted temperature: {next_state.temperature:.2f} K")
    print(f"  Predicted reward: {reward:.4f}")
    print(f"  Uncertainty: {np.mean(info['state_uncertainty']):.4f}")
    
    # Multi-step rollout
    actions = [transitions[i].action for i in range(10)]
    trajectory = world_model.rollout(test_state, actions)
    
    print(f"\nMulti-step rollout:")
    print(f"  Trajectory length: {trajectory['length']}")
    print(f"  Total reward: {trajectory['total_reward']:.4f}")
    temps = [s.temperature for s in trajectory['states']]
    print(f"  Temperature trajectory: {temps[:5]}...")
    
    return world_model, transitions


def example_2_imagination_engine(world_model, transitions):
    """Example 2: Using the Imagination Engine"""
    print("\n" + "="*60)
    print("Example 2: Imagination Engine")
    print("="*60)
    
    engine = ImaginationEngine(world_model)
    base_state = transitions[0].state
    
    # 2.1 Counterfactual simulation
    print("\n2.1 Counterfactual Simulation")
    print("-" * 40)
    
    what_if_result = engine.what_if(
        base_state=base_state,
        intervention='heat',
        intervention_params={'magnitude': 500},
        compare_to_factual=True
    )
    
    print(f"Counterfactual: 'What if we heated the material by 500K?'")
    print(f"  Expected final temp: {what_if_result['summary']['expected_final_temp']:.2f} K")
    print(f"  Success probability: {what_if_result['summary']['success_probability']:.2%}")
    print(f"  Key insights:")
    for insight in what_if_result['summary']['key_insights']:
        print(f"    - {insight}")
    
    # 2.2 Hypothetical scenario
    print("\n2.2 Hypothetical Scenario")
    print("-" * 40)
    
    scenario_result = engine.imagine(
        query_type='hypothetical',
        template_name='extreme_temperature',
        base_state=base_state,
        target_temp=2000
    )
    
    print(f"Hypothetical: 'Extreme temperature at 2000K'")
    print(f"  Number of trajectories: {len(scenario_result.trajectories)}")
    print(f"  Expected reward: {scenario_result.expected_reward:.4f}")
    print(f"  Success probability: {scenario_result.success_probability:.2%}")
    
    # 2.3 Creative design
    print("\n2.3 Creative Design")
    print("-" * 40)
    
    designs = engine.generate_creative_designs(
        seed_materials=[base_state],
        target_properties={'temperature': (600, 800)},
        num_designs=5
    )
    
    print(f"Generated {len(designs)} creative designs")
    for i, design in enumerate(designs[:3]):
        print(f"  Design {i+1}: fitness={np.mean(design['fitness']):.4f}, "
              f"novelty={design['novelty_score']:.4f}")
    
    # 2.4 Application cases
    print("\n2.4 Application Cases")
    print("-" * 40)
    
    cases = MaterialImaginationCases(engine)
    
    # Defect engineering case
    defect_result = cases.case_defect_engineering(
        pristine_material=base_state,
        target_property='ionic_conductivity'
    )
    
    print(f"Defect Engineering:")
    print(f"  Optimal defect concentration: {defect_result['optimal_defect_concentration']:.3f}")
    print(f"  Predicted performance improvement: {defect_result['predicted_performance']:.2f}x")
    
    # Phase stability case
    phase_result = cases.case_phase_stability(
        material=base_state,
        temperature_range=(300, 1500),
        pressure_range=(0.1, 50)
    )
    
    print(f"\nPhase Stability:")
    print(f"  Found {len(phase_result['stable_conditions'])} stable conditions")
    if phase_result['optimal_condition']:
        opt = phase_result['optimal_condition']
        print(f"  Optimal: T={opt['temperature']:.1f}K, P={opt['pressure']:.1f}bar")
    
    return engine


def example_3_model_predictive_control(world_model):
    """Example 3: Model Predictive Control"""
    print("\n" + "="*60)
    print("Example 3: Model Predictive Control")
    print("="*60)
    
    # Create MPC
    mpc_config = MPCConfig(
        horizon=15,
        control_dim=5,
        optimization_method=OptimizationMethod.CEM,
        num_iterations=50,
        num_samples=500,
        elite_fraction=0.1
    )
    
    mpc = ModelPredictiveController(world_model, mpc_config)
    
    # Define cost function
    def terminal_cost(state):
        # Target: temperature around 800K, low energy
        temp_cost = (state.temperature - 800) ** 2 / 10000
        energy_cost = (state.total_energy + 10) ** 2
        return temp_cost + energy_cost
    
    cost = TrajectoryCost(terminal_cost_fn=terminal_cost)
    mpc.set_cost_function(cost)
    
    print(f"MPC Configuration:")
    print(f"  Horizon: {mpc_config.horizon}")
    print(f"  Optimizer: {mpc_config.optimization_method.name}")
    print(f"  Samples per iteration: {mpc_config.num_samples}")
    
    # Initial state
    initial_state = MaterialState(
        state_id="initial",
        temperature=500,
        pressure=1.0,
        total_energy=-5.0,
        features=np.random.randn(20)
    )
    
    print(f"\nInitial state:")
    print(f"  Temperature: {initial_state.temperature:.2f} K")
    print(f"  Pressure: {initial_state.pressure:.2f} bar")
    print(f"  Energy: {initial_state.total_energy:.4f} eV")
    
    # Single step control
    print(f"\nComputing optimal control...")
    action, info = mpc.compute_optimal_control(initial_state)
    
    print(f"Optimal action:")
    print(f"  Type: {action.action_type.name}")
    print(f"  Magnitude: {action.magnitude:.4f}")
    print(f"  Optimal cost: {info['optimal_cost']:.4f}")
    
    # Run control loop
    print(f"\nRunning control loop for 20 steps...")
    result = mpc.run_control_loop(initial_state, num_steps=20)
    
    print(f"Control loop results:")
    print(f"  Total cost: {result['total_cost']:.4f}")
    print(f"  Final temperature: {result['final_state'].temperature:.2f} K")
    print(f"  Final energy: {result['final_state'].total_energy:.4f} eV")
    
    # Synthesis path planning
    print(f"\nSynthesis Path Planning:")
    planner = SynthesisPathPlanner(mpc)
    
    path_result = planner.plan_path(
        initial_material=initial_state,
        target_properties={'temperature': (750, 850)},
        max_steps=20
    )
    
    print(f"  Success rate: {path_result['success_rate']:.1%}")
    print(f"  Number of steps: {len(path_result['recommended_steps'])}")
    
    return mpc


def example_4_fast_simulation():
    """Example 4: Fast Physics Simulator"""
    print("\n" + "="*60)
    print("Example 4: Fast Physics Simulator")
    print("="*60)
    
    config = SimulationConfig(
        state_dim=20,
        hidden_dim=128,
        num_layers=3,
        use_attention=True
    )
    
    simulator = FastPhysicsSimulator(config)
    
    print(f"Simulator Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Use attention: {config.use_attention}")
    print(f"  Total parameters: {sum(p.numel() for p in simulator.parameters()):,}")
    
    # Test forward pass
    batch_size = 16
    state = torch.randn(batch_size, config.state_dim)
    action = torch.randn(batch_size, 5)
    
    print(f"\nTesting forward pass...")
    result = simulator(state, action, return_uncertainty=True)
    
    print(f"  Input shape: {state.shape}")
    print(f"  Next state shape: {result['next_state'].shape}")
    print(f"  Energy shape: {result['energy'].shape}")
    print(f"  Uncertainty shape: {result['uncertainty'].shape}")
    
    # Test trajectory simulation
    print(f"\nSimulating trajectory...")
    traj = simulator.simulate_trajectory(state[:1], num_steps=100)
    print(f"  Trajectory shape: {traj.shape}")
    
    # Benchmark
    print(f"\nBenchmarking (CPU)...")
    stats = simulator.benchmark(batch_size=32, num_steps=100, device='cpu')
    
    print(f"  Steps per second: {stats['steps_per_second']:,.0f}")
    print(f"  Time per step: {stats['time_per_step_ms']:.3f} ms")
    print(f"  Speedup over MD: {stats['speedup_factor']:.0f}x")
    
    return simulator


def example_5_representation_learning():
    """Example 5: Abstract Representation Learning"""
    print("\n" + "="*60)
    print("Example 5: Abstract Representation Learning")
    print("="*60)
    
    config = RepresentationConfig(
        input_dim=20,
        abstract_dim=16,
        num_levels=3,
        use_vq=True
    )
    
    learner = AbstractRepresentationLearner(config)
    
    print(f"Representation Configuration:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Abstract dim: {config.abstract_dim}")
    print(f"  Num levels: {config.num_levels}")
    print(f"  Use VQ: {config.use_vq}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.input_dim)
    
    print(f"\nTesting forward pass...")
    result = learner(x, use_quantization=True)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {result['reconstruction'].shape}")
    print(f"  Most abstract shape: {result['most_abstract'].shape}")
    print(f"  Reconstruction loss: {result['recon_loss'].item():.4f}")
    print(f"  VQ loss: {result['vq_loss'].item():.4f}")
    
    # Test compression
    print(f"\nCompression test:")
    compressed = learner.compress(x, level=-1)
    print(f"  Original: {x.shape}")
    print(f"  Compressed: {compressed.shape}")
    print(f"  Compression ratio: {learner.get_compression_ratio():.2f}x")
    
    # Test interpolation
    print(f"\nInterpolation test:")
    z1 = learner.compress(x[:1], level=-1).squeeze()
    z2 = learner.compress(x[1:2], level=-1).squeeze()
    interpolated = learner.interpolate(z1, z2, num_steps=5)
    print(f"  Interpolated shape: {interpolated.shape}")
    
    return learner


def example_6_dream_generation():
    """Example 6: Dream Generation and Mental Simulation"""
    print("\n" + "="*60)
    print("Example 6: Dream Generation and Mental Simulation")
    print("="*60)
    
    config = DreamConfig(
        latent_dim=32,
        hidden_dim=128,
        length=30,
        num_dreams=5
    )
    
    generator = DreamGenerator(config)
    
    print(f"Dream Generator Configuration:")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Dream length: {config.length}")
    print(f"  Temperature: {config.temperature}")
    
    # Generate free dream
    print(f"\nGenerating free dream...")
    dream = generator.generate_dream()
    
    print(f"  Latent states: {dream['latent_states'].shape}")
    print(f"  Generated states: {dream['generated_states'].shape}")
    print(f"  Average value: {dream['values'].mean().item():.4f}")
    print(f"  Average novelty: {dream['novelties'].mean().item():.4f}")
    
    # Guided dream
    print(f"\nGenerating guided dream...")
    def objective_fn(state):
        # Prefer states with smaller norm
        return -torch.norm(state, dim=-1, keepdim=True)
    
    guided_dreams = generator.guided_dream(
        objective_fn,
        num_attempts=50,
        top_k=3
    )
    
    print(f"  Best dream score: {guided_dreams[0]['score']:.4f}")
    print(f"  Best dream value: {guided_dreams[0]['avg_value']:.4f}")
    
    # Creative exploration
    print(f"\nCreative exploration...")
    discoveries = generator.creative_exploration(
        seed_states=[torch.randn(config.latent_dim) for _ in range(5)],
        exploration_budget=50,
        novelty_threshold=0.6
    )
    
    print(f"  Found {len(discoveries)} novel discoveries")
    for i, disc in enumerate(discoveries[:3]):
        print(f"    Discovery {i+1}: novelty={disc['novelty']:.3f}, value={disc['value']:.3f}")
    
    # Mental simulation
    print(f"\nMental Simulation Engine...")
    engine = MentalSimulationEngine(generator)
    
    plan = torch.randn(10, config.latent_dim)
    initial = torch.randn(config.latent_dim)
    
    rehearsal = engine.mental_rehearsal(plan, initial, num_variations=5)
    
    print(f"  Mental rehearsal:")
    print(f"    Success rate: {rehearsal['success_rate']:.1%}")
    print(f"    Average value: {rehearsal['average_value']:.4f}")
    
    # Future projection
    futures = engine.future_projection(initial, num_steps=20, num_scenarios=3)
    
    print(f"  Future projections:")
    for f in futures:
        print(f"    Scenario {f['scenario_id']}: value={f['final_value']:.3f}, "
              f"diversity={f['path_diversity']:.3f}")
    
    return generator, engine


def run_all_examples():
    """Run all examples in sequence"""
    print("\n" + "#"*60)
    print("# World Model and Internal Simulation Examples")
    print("#"*60)
    
    # Example 1: World Model
    world_model, transitions = example_1_world_model_training()
    
    # Example 2: Imagination Engine
    engine = example_2_imagination_engine(world_model, transitions)
    
    # Example 3: MPC
    mpc = example_3_model_predictive_control(world_model)
    
    # Example 4: Fast Simulator
    simulator = example_4_fast_simulation()
    
    # Example 5: Representation Learning
    learner = example_5_representation_learning()
    
    # Example 6: Dream Generation
    generator, mental_engine = example_6_dream_generation()
    
    print("\n" + "#"*60)
    print("# All examples completed successfully!")
    print("#"*60)


if __name__ == "__main__":
    run_all_examples()
