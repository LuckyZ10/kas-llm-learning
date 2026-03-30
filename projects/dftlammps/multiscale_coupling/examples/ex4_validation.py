#!/usr/bin/env python3
"""
Example 4: Cross-Scale Validation

This example demonstrates validation tools for checking consistency
between atomistic and coarse-grained simulations.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from dftlammps.multiscale_coupling import CrossScaleValidator
from dftlammps.multiscale_coupling.validation import EnergyConsistencyCheck
from dftlammps.multiscale_coupling.ml_cg import CGMapping


def generate_mock_data() -> tuple:
    """
    Generate mock data for validation.
    
    Returns:
        atom_data, cg_data, mapping
    """
    np.random.seed(42)
    
    n_frames = 100
    n_atoms = 40
    n_cg = 10
    
    # Atomistic data
    atom_positions = np.random.randn(n_frames, n_atoms, 3) * 5.0
    atom_forces = np.random.randn(n_frames, n_atoms, 3) * 0.5
    atom_energies = np.cumsum(np.random.randn(n_frames) * 0.01) - 100.0
    
    # CG data
    cg_positions = np.random.randn(n_frames, n_cg, 3) * 5.0
    cg_forces = np.random.randn(n_frames, n_cg, 3) * 2.0
    cg_energies = atom_energies + np.random.randn(n_frames) * 0.5
    
    # Mapping
    atom_to_bead = np.repeat(np.arange(n_cg), 4)
    
    mapping = CGMapping(
        atom_to_bead=atom_to_bead,
        bead_positions=cg_positions[0],
        bead_types=['A'] * n_cg,
        n_beads=n_cg,
        n_atoms=n_atoms
    )
    
    atom_data = {
        'positions': atom_positions[0],  # Use first frame
        'forces': atom_forces[0],  # Use first frame
        'energies': atom_energies,
        'times': np.arange(n_frames) * 0.001
    }
    
    cg_data = {
        'positions': cg_positions[0],  # Use first frame
        'forces': cg_forces[0],  # Use first frame
        'energies': cg_energies
    }
    
    return atom_data, cg_data, mapping


def main():
    """Run validation example."""
    print("=" * 60)
    print("Cross-Scale Validation Example")
    print("=" * 60)
    
    # Generate mock data
    print("\nGenerating mock simulation data...")
    atom_data, cg_data, mapping = generate_mock_data()
    
    print(f"Atomistic data:")
    print(f"  Frames: {len(atom_data['energies'])}")
    print(f"  Atoms: {atom_data['positions'].shape[1]}")
    
    print(f"CG data:")
    print(f"  Frames: {len(cg_data['energies'])}")
    print(f"  Beads: {cg_data['positions'].shape[1]}")
    
    # Run validations
    print("\nRunning cross-scale validations...")
    validator = CrossScaleValidator(tolerance=1e-3)
    
    results = validator.run_all_validations(atom_data, cg_data, mapping)
    
    print(f"\nValidation Results ({len(results)} tests):")
    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.test_name}: {result.message}")
        print(f"      Score: {result.score:.6f}")
    
    # Generate report
    print("\n" + validator.generate_report())
    
    # Energy consistency check
    print("\n" + "=" * 60)
    print("Additional Energy Consistency Checks")
    print("=" * 60)
    
    ecc = EnergyConsistencyCheck()
    
    # QM/MM partitioning check
    qm_energy = -45.6
    mm_energy = -78.9
    coupling_energy = -0.9
    total_energy = qm_energy + mm_energy + coupling_energy
    
    result = ecc.check_qm_mm_partitioning(
        total_energy, qm_energy, mm_energy, coupling_energy
    )
    
    print(f"\nQM/MM Energy Partitioning:")
    print(f"  Total: {result.details['total']:.4f} eV")
    print(f"  QM: {result.details['qm']:.4f} eV")
    print(f"  MM: {result.details['mm']:.4f} eV")
    print(f"  Coupling: {result.details['coupling']:.4f} eV")
    print(f"  Expected: {result.details['expected']:.4f} eV")
    print(f"  Status: {'PASS' if result.passed else 'FAIL'}")
    
    # Trajectory comparison
    print("\nTrajectory Comparison:")
    from dftlammps.multiscale_coupling.validation import compare_trajectories
    
    # Generate proper trajectory data
    ref_traj = np.random.randn(50, 10, 3)  # 50 frames, 10 atoms
    test_traj = ref_traj + np.random.randn(*ref_traj.shape) * 0.1  # Add noise
    
    comparison = compare_trajectories(ref_traj, test_traj, align=True)
    
    print(f"  Mean RMSD: {comparison['mean_rmsd']:.4f} Å")
    print(f"  Max RMSD: {comparison['max_rmsd']:.4f} Å")
    print(f"  Std RMSD: {comparison['std_rmsd']:.4f} Å")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
