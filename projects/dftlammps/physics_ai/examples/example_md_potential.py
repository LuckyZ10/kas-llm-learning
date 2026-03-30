"""
Example: MD Potential Energy Surface Fitting

This example demonstrates fitting a neural network potential for
molecular dynamics using physics constraints.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dftlammps.physics_ai.integration import MDPotentialFitter


def generate_synthetic_md_data(n_configs=500, n_atoms=10):
    """
    Generate synthetic MD data for training.
    
    Uses a simple harmonic potential:
    V = 0.5 * k * sum((r_ij - r_eq)^2)
    """
    np.random.seed(42)
    
    positions_list = []
    energies_list = []
    forces_list = []
    atom_types_list = []
    
    k = 1.0  # Spring constant
    r_eq = 1.5  # Equilibrium distance
    
    for _ in range(n_configs):
        # Random positions
        pos = np.random.randn(n_atoms, 3) * 0.5
        
        # Compute energy (harmonic potential)
        energy = 0.0
        forces = np.zeros((n_atoms, 3))
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_ij = pos[j] - pos[i]
                dist = np.linalg.norm(r_ij)
                
                # Potential
                dr = dist - r_eq
                energy += 0.5 * k * dr ** 2
                
                # Force
                force_mag = k * dr / (dist + 1e-8)
                force_vec = force_mag * r_ij / (dist + 1e-8)
                
                forces[i] -= force_vec
                forces[j] += force_vec
        
        positions_list.append(pos)
        energies_list.append(energy)
        forces_list.append(forces)
        
        # Random atom types (0 or 1)
        atom_types_list.append(np.random.randint(0, 2, n_atoms))
    
    return {
        'positions': np.array(positions_list),
        'energies': np.array(energies_list),
        'forces': np.array(forces_list),
        'atom_types': np.array(atom_types_list)
    }


def example_train_potential():
    """Example: Train a neural network potential."""
    print("=" * 60)
    print("Example: Training Neural Network Potential")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating synthetic MD data...")
    data = generate_synthetic_md_data(n_configs=1000, n_atoms=10)
    
    print(f"Dataset size: {len(data['energies'])} configurations")
    print(f"Number of atoms: {data['positions'].shape[1]}")
    print(f"Energy range: [{data['energies'].min():.2f}, {data['energies'].max():.2f}]")
    
    # Split data
    n_train = 800
    train_data_raw = {
        'positions': data['positions'][:n_train],
        'energies': data['energies'][:n_train],
        'forces': data['forces'][:n_train],
        'atom_types': data['atom_types'][:n_train]
    }
    val_data_raw = {
        'positions': data['positions'][n_train:],
        'energies': data['energies'][n_train:],
        'forces': data['forces'][n_train:],
        'atom_types': data['atom_types'][n_train:]
    }
    
    # Create fitter
    print("\nInitializing potential fitter...")
    fitter = MDPotentialFitter(
        model_type='egnn',
        model_config={
            'hidden_dim': 64,
            'n_layers': 3,
            'cutoff': 3.0
        },
        physics_constraints=['energy', 'force'],
        device='cpu'
    )
    
    # Create model
    fitter.create_model(n_atom_types=2, n_features=64)
    
    # Preprocess data
    print("Preprocessing data...")
    train_data = fitter.preprocess_data(
        train_data_raw['positions'],
        train_data_raw['energies'],
        train_data_raw['forces'],
        train_data_raw['atom_types'],
        normalize=True
    )
    
    val_data = fitter.preprocess_data(
        val_data_raw['positions'],
        val_data_raw['energies'],
        val_data_raw['forces'],
        val_data_raw['atom_types'],
        normalize=True
    )
    
    # Train
    print("\nTraining model...")
    history = fitter.fit(
        train_data=train_data,
        val_data=val_data,
        n_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
        patience=20
    )
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['energy_mae'], label='Energy MAE')
    axes[1].plot(history['force_mae'], label='Force MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Validation MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('md_potential_training.png', dpi=150)
    plt.close()
    
    print("\nTraining history saved to md_potential_training.png")
    
    return fitter, val_data_raw


def example_predict_and_evaluate(fitter, val_data):
    """Example: Make predictions and evaluate."""
    print("\n" + "=" * 60)
    print("Example: Predictions and Evaluation")
    print("=" * 60)
    
    # Make predictions
    predictions = fitter.predict(
        val_data['positions'],
        val_data['atom_types']
    )
    
    # Compute errors
    energy_error = np.mean(np.abs(predictions['energy'] - val_data['energies']))
    force_error = np.mean(np.abs(predictions['forces'] - val_data['forces']))
    
    print(f"\nValidation Errors:")
    print(f"  Energy MAE: {energy_error:.6f}")
    print(f"  Force MAE: {force_error:.6f}")
    
    # Plot parity plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy parity
    axes[0].scatter(val_data['energies'], predictions['energy'], alpha=0.5)
    axes[0].plot(
        [val_data['energies'].min(), val_data['energies'].max()],
        [val_data['energies'].min(), val_data['energies'].max()],
        'r--', label='Perfect'
    )
    axes[0].set_xlabel('True Energy')
    axes[0].set_ylabel('Predicted Energy')
    axes[0].set_title(f'Energy Parity (MAE={energy_error:.4f})')
    axes[0].legend()
    axes[0].grid(True)
    
    # Force parity
    true_forces_flat = val_data['forces'].flatten()
    pred_forces_flat = predictions['forces'].flatten()
    axes[1].scatter(true_forces_flat, pred_forces_flat, alpha=0.3, s=1)
    axes[1].plot(
        [true_forces_flat.min(), true_forces_flat.max()],
        [true_forces_flat.min(), true_forces_flat.max()],
        'r--', label='Perfect'
    )
    axes[1].set_xlabel('True Force')
    axes[1].set_ylabel('Predicted Force')
    axes[1].set_title(f'Force Parity (MAE={force_error:.4f})')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('md_potential_parity.png', dpi=150)
    plt.close()
    
    print("Parity plots saved to md_potential_parity.png")


def example_active_learning(fitter):
    """Example: Active learning for data selection."""
    print("\n" + "=" * 60)
    print("Example: Active Learning")
    print("=" * 60)
    
    # Generate pool of unlabeled configurations
    pool_data = generate_synthetic_md_data(n_configs=1000, n_atoms=10)
    
    # Select configurations using different methods
    methods = ['uncertainty', 'diversity', 'forces']
    
    for method in methods:
        indices = fitter.active_learning_selection(
            pool_data,
            n_select=50,
            method=method
        )
        
        print(f"\nMethod: {method}")
        print(f"  Selected {len(indices)} configurations")
        print(f"  Indices: {indices[:10]}...")


def example_model_export(fitter):
    """Example: Export model for LAMMPS."""
    print("\n" + "=" * 60)
    print("Example: Model Export")
    print("=" * 60)
    
    # Save checkpoint
    fitter.save_checkpoint('md_potential_checkpoint.pt')
    print("Saved checkpoint to md_potential_checkpoint.pt")
    
    # Export to LAMMPS format (TorchScript)
    try:
        fitter.export_to_lammps('md_potential_lammps.pt')
    except Exception as e:
        print(f"Export note: {e}")


if __name__ == '__main__':
    print("MD Potential Energy Surface Fitting Examples")
    print()
    
    # Train potential
    fitter, val_data = example_train_potential()
    
    # Evaluate
    example_predict_and_evaluate(fitter, val_data)
    
    # Active learning
    example_active_learning(fitter)
    
    # Export
    example_model_export(fitter)
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
