#!/usr/bin/env python3
"""
Example 5: Multiscale GNN for Simultaneous Atom/CG Modeling

This example demonstrates the MultiscaleGNN that models both
atomistic and coarse-grained scales simultaneously.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dftlammps.multiscale_coupling import MultiscaleGNN
from dftlammps.multiscale_coupling.gnn_models import build_graph


def generate_multiscale_system():
    """Generate a simple multiscale system."""
    np.random.seed(42)
    
    # Atomistic system: 20 atoms
    n_atoms = 20
    atom_positions = np.random.randn(n_atoms, 3) * 3.0
    atom_elements = ['C'] * 10 + ['H'] * 10
    
    # CG system: 5 beads (4 atoms per bead)
    n_cg = 5
    cg_positions = np.random.randn(n_cg, 3) * 3.0
    cg_types = ['BEAD'] * n_cg
    
    # Mapping: 4 atoms -> 1 bead
    atom_to_cg = np.repeat(np.arange(n_cg), 4)
    
    return (atom_positions, atom_elements, 
            cg_positions, cg_types, atom_to_cg)


def main():
    """Run multiscale GNN example."""
    print("=" * 60)
    print("Multiscale GNN Example")
    print("=" * 60)
    
    # Generate system
    print("\nGenerating multiscale system...")
    (atom_pos, atom_elem, cg_pos, cg_types, atom_to_cg) = generate_multiscale_system()
    
    print(f"Atomistic scale: {len(atom_pos)} atoms")
    print(f"CG scale: {len(cg_pos)} beads")
    print(f"Compression ratio: {len(atom_pos) / len(cg_pos):.1f}x")
    
    # Initialize model
    print("\nInitializing MultiscaleGNN...")
    model = MultiscaleGNN(
        atom_features=2,  # C, H
        cg_features=1,    # BEAD
        hidden_dim=32,
        n_atom_layers=3,
        n_cg_layers=2
    )
    
    # Make predictions
    print("\nRunning multiscale prediction...")
    results = model.predict_multiscale(
        atom_positions=atom_pos,
        atom_elements=atom_elem,
        cg_positions=cg_pos,
        cg_types=cg_types,
        atom_to_cg_mapping=atom_to_cg
    )
    
    print("\nResults:")
    print(f"  Atom forces shape: {results['atom_forces'].shape}")
    print(f"  CG forces shape: {results['cg_forces'].shape}")
    print(f"  Atom force magnitude: {np.linalg.norm(results['atom_forces']):.4f}")
    print(f"  CG force magnitude: {np.linalg.norm(results['cg_forces']):.4f}")
    
    # Cross-scale attention
    print("\nApplying cross-scale attention...")
    
    # Get atom features
    atom_graph = build_graph(atom_pos, atom_elem)
    h_atom = atom_graph.nodes
    
    # Get CG features
    cg_graph = build_graph(cg_pos, cg_types, cutoff=15.0)
    h_cg = cg_graph.nodes
    
    # Apply cross-attention
    updated_cg = model.cross_scale_attention(
        atom_features=h_atom,
        cg_features=h_cg,
        atom_to_cg_mapping=atom_to_cg
    )
    
    print(f"  CG features updated with atom-scale information")
    print(f"  Updated CG features shape: {updated_cg.shape}")
    
    # Verify conservation properties
    print("\nChecking conservation properties...")
    
    # Total force should be conserved (approximately)
    total_atom_force = np.sum(results['atom_forces'], axis=0)
    total_cg_force = np.sum(results['cg_forces'], axis=0)
    
    print(f"  Total atom force: {total_atom_force}")
    print(f"  Total CG force: {total_cg_force}")
    print(f"  Force conservation error: {np.linalg.norm(total_atom_force - total_cg_force):.6f}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
