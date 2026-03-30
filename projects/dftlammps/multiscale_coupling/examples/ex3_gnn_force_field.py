#!/usr/bin/env python3
"""
Example 3: Graph Neural Network for CG Force Field

This example demonstrates training a GNN to predict coarse-grained
forces from CG configurations.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dftlammps.multiscale_coupling import CGGNN
from dftlammps.multiscale_coupling.gnn_models import build_graph, train_gnn


def generate_cg_training_data(n_samples: int = 100,
                              n_beads: int = 10) -> tuple:
    """
    Generate training data for CG GNN.
    
    Args:
        n_samples: Number of training samples
        n_beads: Number of CG beads
        
    Returns:
        graphs, forces
    """
    np.random.seed(42)
    
    graphs = []
    forces_list = []
    
    for i in range(n_samples):
        # Random CG configuration
        positions = np.random.randn(n_beads, 3) * 5.0
        
        # Simple bead types
        bead_types = ['A'] * (n_beads // 2) + ['B'] * (n_beads - n_beads // 2)
        
        # Build graph
        graph = build_graph(positions, bead_types, cutoff=8.0)
        
        # Generate mock forces (harmonic spring-like)
        forces = np.zeros((n_beads, 3))
        for j in range(n_beads):
            # Spring force toward origin
            k = 0.1
            forces[j] = -k * positions[j]
            
            # Lennard-Jones-like interactions
            for k_idx in range(j + 1, n_beads):
                r_vec = positions[k_idx] - positions[j]
                r = np.linalg.norm(r_vec)
                if r < 8.0 and r > 0.1:
                    # Simplified LJ force
                    f_mag = 24 * (2 / r**13 - 1 / r**7)
                    f_vec = f_mag * r_vec / r
                    forces[j] -= f_vec
                    forces[k_idx] += f_vec
        
        # Add noise
        forces += np.random.randn(n_beads, 3) * 0.01
        
        graphs.append(graph)
        forces_list.append(forces)
    
    return graphs, forces_list


def main():
    """Run GNN training example."""
    print("=" * 60)
    print("GNN Force Field Training Example")
    print("=" * 60)
    
    # Generate training data
    print("\nGenerating training data...")
    train_graphs, train_forces = generate_cg_training_data(
        n_samples=50, n_beads=8
    )
    val_graphs, val_forces = generate_cg_training_data(
        n_samples=20, n_beads=8
    )
    
    print(f"Training samples: {len(train_graphs)}")
    print(f"Validation samples: {len(val_graphs)}")
    
    # Get feature dimensions from first graph
    first_graph = train_graphs[0]
    n_node_features = first_graph.nodes.shape[1]
    
    print(f"Node features: {n_node_features}")
    print(f"Edge features: {first_graph.edge_features.shape[1]}")
    
    # Create model
    print("\nInitializing CG-GNN model...")
    model = CGGNN(
        n_node_features=n_node_features,
        n_edge_features=4,
        hidden_dim=32,
        n_layers=3,
        cutoff=8.0
    )
    
    print(f"Model layers: {model.n_layers}")
    print(f"Hidden dimension: {model.hidden_dim}")
    
    # Train model
    print("\nTraining model...")
    history = train_gnn(
        model=model,
        train_graphs=train_graphs,
        train_forces=train_forces,
        val_graphs=val_graphs,
        val_forces=val_forces,
        n_epochs=20,
        learning_rate=0.01
    )
    
    # Evaluate
    print("\nEvaluating model...")
    test_graphs, test_forces = generate_cg_training_data(
        n_samples=10, n_beads=8
    )
    
    total_mse = 0.0
    for graph, true_forces in zip(test_graphs, test_forces):
        pred_forces = model.forward(graph)
        mse = np.mean((pred_forces - true_forces) ** 2)
        total_mse += mse
    
    avg_mse = total_mse / len(test_graphs)
    rmse = np.sqrt(avg_mse)
    
    print(f"\nTest Results:")
    print(f"  MSE: {avg_mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    # Compare force magnitudes
    true_mags = [np.linalg.norm(f) for f in test_forces]
    pred_mags = [np.linalg.norm(model.forward(g)) for g in test_graphs]
    
    print(f"  True force magnitude (mean): {np.mean(true_mags):.4f}")
    print(f"  Pred force magnitude (mean): {np.mean(pred_mags):.4f}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
