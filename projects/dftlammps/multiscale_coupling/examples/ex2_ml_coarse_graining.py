#!/usr/bin/env python3
"""
Example 2: Machine Learning Coarse-Graining of Polymer

This example demonstrates how to use ML-based coarse-graining
to map an atomistic polymer trajectory to a CG representation.
"""
import numpy as np
import sys
import os

# Add workspace to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dftlammps.multiscale_coupling import CoarseGrainer, MLCGMapping
from dftlammps.multiscale_coupling.ml_cg import ForceMatcher


def generate_polymer_trajectory(n_monomers: int = 20,
                                n_frames: int = 100) -> tuple:
    """
    Generate a mock polymer trajectory.
    
    Args:
        n_monomers: Number of monomers
        n_frames: Number of frames
        
    Returns:
        positions, forces, atom_types
    """
    np.random.seed(42)
    
    # Each monomer: CH2-CH2 (4 atoms)
    atoms_per_monomer = 4
    n_atoms = n_monomers * atoms_per_monomer
    
    # Atom types
    atom_types = []
    for i in range(n_monomers):
        atom_types.extend(['C', 'H', 'H', 'H'])  # CH3-like
    
    # Generate trajectory (random walk polymer)
    positions = []
    forces = []
    
    for frame in range(n_frames):
        frame_pos = []
        
        # Start at origin
        current_pos = np.array([0.0, 0.0, 0.0])
        
        for monomer in range(n_monomers):
            # Add some randomness
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # Carbon position
            if monomer > 0:
                current_pos = current_pos + direction * 1.54  # C-C bond
            
            frame_pos.append(current_pos)
            
            # Hydrogens (tetrahedral)
            h1 = current_pos + np.array([0.6, 0.6, 0.3])
            h2 = current_pos + np.array([-0.6, 0.6, -0.3])
            h3 = current_pos + np.array([0.0, -0.8, 0.5])
            
            frame_pos.extend([h1, h2, h3])
        
        positions.append(frame_pos)
        
        # Mock forces (random with some structure)
        frame_forces = np.random.randn(n_atoms, 3) * 0.1
        forces.append(frame_forces)
    
    return np.array(positions), np.array(forces), atom_types


def main():
    """Run coarse-graining example."""
    print("=" * 60)
    print("ML Coarse-Graining Example: Polymer")
    print("=" * 60)
    
    # Generate data
    print("\nGenerating polymer trajectory...")
    positions, forces, atom_types = generate_polymer_trajectory(
        n_monomers=10, n_frames=50
    )
    
    print(f"Trajectory: {len(positions)} frames, {len(atom_types)} atoms")
    print(f"Atom types: {len(set(atom_types))} unique")
    
    # Define CG mapping: each monomer -> one bead
    n_beads = 10
    print(f"\nCoarse-graining to {n_beads} beads...")
    
    # Create mapping: 4 atoms per bead
    atom_to_bead = np.repeat(np.arange(n_beads), 4)
    
    # Use centroid-based coarse-graining
    from dftlammps.multiscale_coupling.ml_cg import CentroidCoarseGrainer
    
    cg = CentroidCoarseGrainer(n_beads=n_beads, predefined_mapping=atom_to_bead)
    
    mapping = cg.fit([positions], atom_types)
    
    print(f"Mapping created: {mapping.n_atoms} atoms -> {mapping.n_beads} beads")
    
    # Transform trajectory
    print("\nTransforming trajectory...")
    cg_trajectory = cg.transform(positions)
    
    print(f"CG trajectory shape: {cg_trajectory.shape}")
    
    # Compute CG forces via force matching
    print("\nComputing CG reference forces...")
    fm = ForceMatcher(mapping)
    cg_forces = fm.compute_reference_forces(positions, forces)
    
    print(f"CG forces shape: {cg_forces.shape}")
    print(f"Force magnitude range: [{np.min(np.linalg.norm(cg_forces, axis=2)):.4f}, "
          f"{np.max(np.linalg.norm(cg_forces, axis=2)):.4f}]")
    
    # Optimize CG force field
    print("\nOptimizing CG force field...")
    cg_ff_params = fm.optimize_force_field(
        cg_trajectory, cg_forces, force_field_type='spline'
    )
    
    print(f"Force field type: {cg_ff_params['type']}")
    print(f"Cutoff: {cg_ff_params['cutoff']} Å")
    
    # Save results
    print("\nSaving results...")
    mapping.save('polymer_cg_mapping')
    print("Mapping saved to polymer_cg_mapping.npz")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Original system: {len(atom_types)} atoms")
    print(f"  CG system: {n_beads} beads")
    print(f"  Compression ratio: {len(atom_types) / n_beads:.1f}x")
    print("=" * 60)


if __name__ == '__main__':
    main()
