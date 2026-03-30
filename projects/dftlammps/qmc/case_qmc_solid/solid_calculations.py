"""
QMC Calculations for Solid State Systems
========================================

Demonstrates QMC calculations on periodic systems:
- 1D hydrogen chain
- Simple cubic hydrogen lattice
- Bulk silicon (simplified)

Note: Real solids require large supercells and careful k-point sampling.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyscf_qmc_interface import PySCFQMCInterface
from vmc_calculator import VMCCalculator, SlaterJastrow
from dmc_calculator import DMCCalculator, TrialWaveFunction, create_trial_wf_from_vmc


def run_hydrogen_chain(n_atoms=4, bond_length=1.0, n_vmc_samples=5000):
    """
    Hydrogen chain with periodic boundary conditions.
    
    Parameters:
    -----------
    n_atoms : int
        Number of H atoms in the chain
    bond_length : float
        H-H bond length in Angstrom
    """
    print("="*60)
    print("Hydrogen Chain (Periodic)")
    print("="*60)
    print(f"Number of atoms: {n_atoms}")
    print(f"Bond length: {bond_length} Å")
    print()
    
    # Create H chain
    coordinates = np.array([[i * bond_length, 0.0, 0.0] for i in range(n_atoms)])
    atom_symbols = ['H'] * n_atoms
    
    # Periodic cell (repeat along x)
    cell = np.array([[n_atoms * bond_length + 5.0, 0.0, 0.0],
                     [0.0, 10.0, 0.0],
                     [0.0, 0.0, 10.0]])
    
    try:
        qmc = PySCFQMCInterface(
            atom_symbols=atom_symbols,
            coordinates=coordinates,
            basis='cc-pVDZ',
            charge=0,
            spin=n_atoms % 2,  # Singlet if even number
            periodic=True,
            cell=cell
        )
        
        # Run gamma-point HF
        hf_result = qmc.run_hf()
        print(f"Periodic HF Energy: {hf_result['energy']:.6f} Ha")
        print(f"HF Energy per atom: {hf_result['energy']/n_atoms:.6f} Ha")
        print()
        
        # VMC calculation
        print("Running VMC...")
        n_electrons = n_atoms
        n_up = (n_electrons + qmc.spin) // 2
        
        slater = SlaterJastrow(
            n_electrons=n_electrons,
            n_up=n_up,
            atom_positions=coordinates,
            atom_charges=np.ones(n_atoms),
            mo_coeffs=qmc.mf.mo_coeff,
            jastrow_order=2
        )
        
        vmc_calc = VMCCalculator(
            wave_function=slater,
            atom_positions=coordinates,
            atom_charges=np.ones(n_atoms),
            n_walkers=50,
            step_size=0.3
        )
        
        vmc_result = vmc_calc.run(
            n_electrons=n_electrons,
            n_samples=n_vmc_samples,
            n_equil=1000
        )
        
        print(f"VMC Energy: {vmc_result.energy:.6f} ± {vmc_result.energy_error:.6f} Ha")
        print(f"VMC Energy per atom: {vmc_result.energy/n_atoms:.6f} Ha")
        print()
        
        return {
            'system': 'H_chain',
            'n_atoms': n_atoms,
            'bond_length': bond_length,
            'hf_energy': hf_result['energy'],
            'vmc_energy': vmc_result.energy,
            'vmc_error': vmc_result.energy_error
        }
        
    except Exception as e:
        print(f"Calculation failed: {e}")
        print("Note: Periodic calculations require PySCF with PBC support")
        return None


def run_simple_cubic_hydrogen(lattice_constant=2.0, n_vmc_samples=5000):
    """
    Simple cubic hydrogen lattice (8 atoms).
    
    Parameters:
    -----------
    lattice_constant : float
        Cubic lattice constant in Angstrom
    """
    print("="*60)
    print("Simple Cubic Hydrogen Lattice")
    print("="*60)
    print(f"Lattice constant: {lattice_constant} Å")
    print()
    
    # 8 H atoms in simple cubic arrangement
    coords = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                coords.append([i * lattice_constant, 
                              j * lattice_constant, 
                              k * lattice_constant])
    
    coordinates = np.array(coords)
    atom_symbols = ['H'] * 8
    
    # Periodic cell
    cell = np.array([[2 * lattice_constant, 0, 0],
                     [0, 2 * lattice_constant, 0],
                     [0, 0, 2 * lattice_constant]])
    
    try:
        qmc = PySCFQMCInterface(
            atom_symbols=atom_symbols,
            coordinates=coordinates,
            basis='cc-pVDZ',
            charge=0,
            spin=0,
            periodic=True,
            cell=cell
        )
        
        hf_result = qmc.run_hf()
        print(f"HF Energy: {hf_result['energy']:.6f} Ha")
        print(f"HF per atom: {hf_result['energy']/8:.6f} Ha")
        print()
        
        # VMC
        slater = SlaterJastrow(
            n_electrons=8,
            n_up=4,
            atom_positions=coordinates,
            atom_charges=np.ones(8),
            mo_coeffs=qmc.mf.mo_coeff,
            jastrow_order=2
        )
        
        vmc_calc = VMCCalculator(
            wave_function=slater,
            atom_positions=coordinates,
            atom_charges=np.ones(8),
            n_walkers=50,
            step_size=0.2
        )
        
        vmc_result = vmc_calc.run(n_electrons=8, n_samples=n_vmc_samples, n_equil=1000)
        
        print(f"VMC Energy: {vmc_result.energy:.6f} ± {vmc_result.energy_error:.6f} Ha")
        print(f"VMC per atom: {vmc_result.energy/8:.6f} Ha")
        print()
        
        return {
            'system': 'sc_H',
            'lattice_constant': lattice_constant,
            'hf_energy': hf_result['energy'],
            'vmc_energy': vmc_result.energy,
            'vmc_error': vmc_result.energy_error
        }
        
    except Exception as e:
        print(f"Calculation failed: {e}")
        return None


def run_finite_cluster(n_atoms=8, density=0.02):
    """
    Finite hydrogen cluster as model for bulk.
    Uses Wigner-Seitz boundary conditions implicitly.
    """
    print("="*60)
    print(f"Hydrogen Cluster (n={n_atoms})")
    print("="*60)
    print(f"Density parameter rs approx: {(3/(4*np.pi*density))**(1/3):.2f}")
    print()
    
    # Create random cluster with minimum distance constraint
    np.random.seed(42)
    coords = []
    min_dist = 1.0
    
    while len(coords) < n_atoms:
        new_pos = np.random.randn(3) * (n_atoms ** (1/3)) * 2
        
        # Check minimum distance
        valid = True
        for pos in coords:
            if np.linalg.norm(new_pos - pos) < min_dist:
                valid = False
                break
        
        if valid:
            coords.append(new_pos)
    
    coordinates = np.array(coords)
    atom_symbols = ['H'] * n_atoms
    
    try:
        qmc = PySCFQMCInterface(
            atom_symbols=atom_symbols,
            coordinates=coordinates,
            basis='cc-pVDZ',
            charge=0,
            spin=n_atoms % 2
        )
        
        hf_result = qmc.run_hf()
        print(f"HF Energy: {hf_result['energy']:.6f} Ha")
        print(f"HF per atom: {hf_result['energy']/n_atoms:.6f} Ha")
        print()
        
        # VMC
        n_electrons = n_atoms
        n_up = (n_electrons + qmc.spin) // 2
        
        slater = SlaterJastrow(
            n_electrons=n_electrons,
            n_up=n_up,
            atom_positions=coordinates,
            atom_charges=np.ones(n_atoms),
            mo_coeffs=qmc.mf.mo_coeff,
            jastrow_order=2
        )
        
        vmc_calc = VMCCalculator(
            wave_function=slater,
            atom_positions=coordinates,
            atom_charges=np.ones(n_atoms),
            n_walkers=100,
            step_size=0.3
        )
        
        vmc_result = vmc_calc.run(n_electrons=n_electrons, n_samples=5000, n_equil=1000)
        
        print(f"VMC Energy: {vmc_result.energy:.6f} ± {vmc_result.energy_error:.6f} Ha")
        print(f"VMC per atom: {vmc_result.energy/n_atoms:.6f} Ha")
        print()
        
        # Estimate cohesion by comparing to isolated atoms
        e_isolated = -0.5 * n_atoms  # Exact H atom energy
        cohesion = vmc_result.energy - e_isolated
        
        print(f"Estimated cohesion energy: {cohesion:.4f} Ha ({cohesion*27.2114:.2f} eV)")
        print()
        
        return {
            'system': 'H_cluster',
            'n_atoms': n_atoms,
            'hf_energy': hf_result['energy'],
            'vmc_energy': vmc_result.energy,
            'vmc_error': vmc_result.energy_error,
            'cohesion_ev': cohesion * 27.2114
        }
        
    except Exception as e:
        print(f"Calculation failed: {e}")
        return None


def run_all_solid_calculations():
    """Run all solid state calculations."""
    print("\n" + "="*70)
    print("QMC SOLID STATE CALCULATIONS")
    print("="*70 + "\n")
    
    results = {}
    
    # Hydrogen chain
    print("1. HYDROGEN CHAIN\n")
    results['h_chain'] = run_hydrogen_chain(n_atoms=4, bond_length=1.0)
    
    print("\n")
    
    # Simple cubic
    print("2. SIMPLE CUBIC HYDROGEN\n")
    results['sc_hydrogen'] = run_simple_cubic_hydrogen(lattice_constant=2.5)
    
    print("\n")
    
    # Finite cluster
    print("3. HYDROGEN CLUSTER\n")
    results['cluster'] = run_finite_cluster(n_atoms=8)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, res in results.items():
        if res:
            print(f"\n{name}:")
            print(f"  HF total:  {res['hf_energy']:.6f} Ha")
            print(f"  VMC total: {res['vmc_energy']:.6f} ± {res['vmc_error']:.6f} Ha")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='QMC Solid State Calculations')
    parser.add_argument('--calculation', 
                       choices=['chain', 'cubic', 'cluster', 'all'],
                       default='all',
                       help='Calculation type')
    
    args = parser.parse_args()
    
    if args.calculation == 'all':
        run_all_solid_calculations()
    elif args.calculation == 'chain':
        run_hydrogen_chain()
    elif args.calculation == 'cubic':
        run_simple_cubic_hydrogen()
    elif args.calculation == 'cluster':
        run_finite_cluster()
