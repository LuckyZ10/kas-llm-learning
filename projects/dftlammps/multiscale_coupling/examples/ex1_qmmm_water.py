#!/usr/bin/env python3
"""
Example 1: QM/MM simulation of water cluster

This example demonstrates how to set up and run a QM/MM calculation
on a water cluster where the inner shell is treated with QM (VASP)
and the outer shell with MM (LAMMPS).
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dftlammps.multiscale_coupling import VASPLAMMPSCoupling
from dftlammps.multiscale_coupling.utils import AtomSelection, UnitConverter


def create_water_cluster(n_inner: int = 10, n_outer: int = 40) -> tuple:
    """
    Create a water cluster with inner and outer shells.
    
    Args:
        n_inner: Number of water molecules in QM region
        n_outer: Number of water molecules in MM region
        
    Returns:
        positions, elements, qm_mask, mm_mask
    """
    np.random.seed(42)
    
    total_molecules = n_inner + n_outer
    n_atoms = total_molecules * 3  # 3 atoms per water
    
    # Create positions (simple cubic for demonstration)
    positions = []
    elements = []
    
    # Inner shell (clustered near center)
    for i in range(n_inner):
        center = np.random.randn(3) * 3.0  # Tight cluster
        # O atom
        positions.append(center)
        elements.append('O')
        # H atoms
        positions.append(center + [0.96, 0, 0])
        elements.append('H')
        positions.append(center + [-0.24, 0.93, 0])
        elements.append('H')
    
    # Outer shell
    for i in range(n_outer):
        center = np.random.randn(3) * 8.0 + [0, 0, 15]  # Extended
        positions.append(center)
        elements.append('O')
        positions.append(center + [0.96, 0, 0])
        elements.append('H')
        positions.append(center + [-0.24, 0.93, 0])
        elements.append('H')
    
    positions = np.array(positions)
    
    # Define QM/MM regions
    qm_mask = np.zeros(n_atoms, dtype=bool)
    mm_mask = np.zeros(n_atoms, dtype=bool)
    
    qm_atoms = n_inner * 3
    qm_mask[:qm_atoms] = True
    mm_mask[qm_atoms:] = True
    
    return positions, elements, qm_mask, mm_mask


def main():
    """Run QM/MM example."""
    print("=" * 60)
    print("QM/MM Example: Water Cluster")
    print("=" * 60)
    
    # Create system
    positions, elements, qm_mask, mm_mask = create_water_cluster(
        n_inner=5, n_outer=20
    )
    
    print(f"\nSystem size: {len(positions)} atoms")
    print(f"  QM region: {np.sum(qm_mask)} atoms ({np.sum(qm_mask)//3} waters)")
    print(f"  MM region: {np.sum(mm_mask)} atoms ({np.sum(mm_mask)//3} waters)")
    
    # Initialize QM/MM coupling
    # Note: This requires VASP and LAMMPS to be installed
    print("\nInitializing QM/MM interface...")
    try:
        qmmm = VASPLAMMPSCoupling(
            vasp_cmd='vasp_std',
            lammps_cmd='lmp',
            work_dir='./qmmm_water_example',
            embedding='electrostatic'
        )
        
        qmmm.set_regions(qm_mask, mm_mask)
        
        # Define MM types (simplified)
        mm_types = ['OW', 'HW', 'HW'] * 20  # Water types
        
        # Run calculation
        print("Running QM/MM calculation...")
        results = qmmm.calculate(
            positions=positions,
            qm_elements=elements,
            mm_types=mm_types
        )
        
        print("\nResults:")
        print(f"  Total energy: {results['total_energy']:.4f} eV")
        print(f"  QM energy: {results['qm_energy']:.4f} eV")
        print(f"  MM energy: {results['mm_energy']:.4f} eV")
        print(f"  Coupling energy: {results['coupling_energy']:.4f} eV")
        
    except Exception as e:
        print(f"\nNote: This example requires VASP and LAMMPS to be installed.")
        print(f"Error: {e}")
        print("\nMock results for demonstration:")
        print("  Total energy: -125.4567 eV (mock)")
        print("  QM energy: -45.6789 eV (mock)")
        print("  MM energy: -78.9012 eV (mock)")
        print("  Coupling energy: -0.8766 eV (mock)")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
