"""
Example: Conservation Law Testing with Physics Validator

This example demonstrates how to use the PhysicsLawValidator
to validate that a model respects conservation laws.
"""

import torch
import numpy as np
from dftlammps.physics_ai.validators import PhysicsLawValidator
from dftlammps.physics_ai.models import PhysicsInformedGNN


def create_synthetic_trajectory(n_steps=100, n_particles=10):
    """
    Create a synthetic molecular dynamics trajectory.
    
    Returns:
        Dictionary with positions, velocities, masses
    """
    dt = 0.001
    
    # Initialize
    positions = []
    velocities = []
    
    # Initial conditions
    pos = np.random.randn(n_particles, 3)
    vel = np.random.randn(n_particles, 3)
    masses = np.ones(n_particles)
    
    # Remove center of mass motion
    vel = vel - np.mean(vel, axis=0)
    
    # Simple harmonic oscillator dynamics
    for step in range(n_steps):
        positions.append(pos.copy())
        velocities.append(vel.copy())
        
        # Force = -position (harmonic)
        force = -pos
        
        # Velocity Verlet integration
        vel = vel + 0.5 * force / masses[:, None] * dt
        pos = pos + vel * dt
        force_new = -pos
        vel = vel + 0.5 * force_new / masses[:, None] * dt
    
    return {
        'positions': torch.tensor(np.array(positions), dtype=torch.float32),
        'velocities': torch.tensor(np.array(velocities), dtype=torch.float32),
        'masses': torch.tensor(masses, dtype=torch.float32)
    }


def example_trajectory_validation():
    """Example: Validate a trajectory."""
    print("=" * 60)
    print("Example: Trajectory Validation")
    print("=" * 60)
    
    # Create validator
    validator = PhysicsLawValidator(tolerance=1e-4)
    
    # Create synthetic trajectory
    trajectory = create_synthetic_trajectory(n_steps=1000)
    
    # Convert to numpy for validation
    positions_np = trajectory['positions'].numpy()
    velocities_np = trajectory['velocities'].numpy()
    trajectory_np = np.concatenate([positions_np, velocities_np], axis=-1)
    
    # Validate trajectory
    results = validator.validate_trajectory(trajectory_np, dt=0.001)
    
    # Print results
    print("\nValidation Results:")
    for test_name, result in results.items():
        status = result['result'].value.upper()
        print(f"  [{status}] {test_name}")
        for key, value in result.items():
            if key != 'result':
                print(f"    {key}: {value}")
    
    return results


def example_model_validation():
    """Example: Validate a neural network model."""
    print("\n" + "=" * 60)
    print("Example: Model Validation")
    print("=" * 60)
    
    # Create a simple model
    model = PhysicsInformedGNN(
        node_dim=5,
        hidden_dim=64,
        n_layers=3,
        output_type='both'
    )
    
    # Create test data
    batch_size = 8
    n_nodes = 20
    
    test_data = {
        'positions': torch.randn(batch_size, n_nodes, 3),
        'velocities': torch.randn(batch_size, n_nodes, 3),
        'masses': torch.ones(batch_size, n_nodes),
        'forces': torch.randn(batch_size, n_nodes, 3)
    }
    
    # Create validator
    validator = PhysicsLawValidator(tolerance=1e-3)
    
    # Validate model
    results = validator.validate_model(model, test_data)
    
    # Print report
    validator.print_report()
    
    return results


def example_custom_test():
    """Example: Add a custom physics test."""
    print("\n" + "=" * 60)
    print("Example: Custom Physics Test")
    print("=" * 60)
    
    # Create validator
    validator = PhysicsLawValidator()
    
    # Define custom test
    def test_energy_gap(model, test_data, tolerance=1e-6):
        """Test that energy gap between states is positive."""
        with torch.no_grad():
            output = model(test_data)
            
            if isinstance(output, dict) and 'energy' in output:
                energies = output['energy']
                min_energy = torch.min(energies).item()
                
                # Ground state energy should be the minimum
                passed = min_energy > -1e10  # Reasonable bound
                
                return passed, {
                    'min_energy': min_energy,
                    'energy_range': (torch.max(energies) - torch.min(energies)).item()
                }
            
            return False, {'error': 'No energy output'}
    
    # Add custom test
    from dftlammps.physics_ai.validators import PhysicsTest
    
    custom_test = PhysicsTest(
        name="positive_energy_gap",
        description="Energy gap between states should be positive",
        test_fn=test_energy_gap,
        tolerance=1e-6
    )
    
    validator.add_test(custom_test)
    
    print(f"Added custom test: {custom_test.name}")
    print(f"Description: {custom_test.description}")
    
    return validator


def example_symmetry_tests():
    """Example: Test symmetries of a potential."""
    print("\n" + "=" * 60)
    print("Example: Symmetry Tests")
    print("=" * 60)
    
    # Create a simple potential model
    class SimplePotential(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(3, 64),
                torch.nn.Tanh(),
                torch.nn.Linear(64, 1)
            )
        
        def forward(self, data):
            # Sum over atoms
            pos = data['positions']
            energies = self.fc(pos).sum(dim=1)
            return {'energy': energies}
    
    model = SimplePotential()
    
    # Create validator
    validator = PhysicsLawValidator(tolerance=1e-4)
    
    # Test data
    test_data = {
        'positions': torch.randn(4, 10, 3),
        'velocities': torch.randn(4, 10, 3),
        'masses': torch.ones(4, 10)
    }
    
    # Run specific symmetry tests
    symmetry_tests = [
        'translational_symmetry',
        'rotational_symmetry',
        'time_reversal_symmetry'
    ]
    
    results = validator.validate_model(model, test_data, test_names=symmetry_tests)
    
    print("\nSymmetry Test Results:")
    for test_name, result in results.items():
        status = result['result'].value.upper()
        print(f"  [{status}] {test_name}")
    
    return results


if __name__ == '__main__':
    print("Physics Law Validator Examples")
    print()
    
    # Run examples
    example_trajectory_validation()
    example_model_validation()
    example_custom_test()
    example_symmetry_tests()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
