"""
QMC Calculations for Catalysis
===============================

QMC applications in catalysis:
- Hydrogen dissociation on metal surface (simplified model)
- CO adsorption on surface
- Reaction barriers

Note: Full catalysis calculations require large periodic cells.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyscf_qmc_interface import PySCFQMCInterface
from vmc_calculator import VMCCalculator, SlaterJastrow, create_slater_jastrow_from_pyscf
from dmc_calculator import DMCCalculator, create_trial_wf_from_vmc


def run_h2_dissociation_barrier():
    """
    Calculate H2 dissociation barrier as a simple catalysis model.
    Models H2 approaching a surface (simplified as 4 dummy atoms).
    """
    print("="*60)
    print("H2 Dissociation Barrier (Simplified Model)")
    print("="*60)
    print()
    
    # Parameters
    bond_lengths = np.linspace(0.74, 3.0, 8)  # From equilibrium to dissociated
    surface_distance = 2.0  # Distance from surface
    
    results = []
    
    for i, r_hh in enumerate(bond_lengths):
        print(f"\nGeometry {i+1}/{len(bond_lengths)}: R_HH = {r_hh:.2f} Å")
        
        # H2 above surface (simplified with 4 dummy surface atoms)
        coordinates = np.array([
            [0.0, 0.0, surface_distance],           # H1
            [r_hh, 0.0, surface_distance],          # H2
            [-2.0, -2.0, 0.0],                      # Surface atom
            [2.0, -2.0, 0.0],                       # Surface atom
            [-2.0, 2.0, 0.0],                       # Surface atom
            [2.0, 2.0, 0.0]                         # Surface atom
        ])
        
        atom_symbols = ['H', 'H', 'He', 'He', 'He', 'He']  # He as dummy surface
        atom_charges = np.array([1, 1, 2, 2, 2, 2])
        
        try:
            qmc = PySCFQMCInterface(
                atom_symbols=atom_symbols,
                coordinates=coordinates,
                basis='cc-pVDZ',
                charge=0,
                spin=0
            )
            
            # HF calculation
            hf_result = qmc.run_hf()
            
            # VMC calculation
            slater = create_slater_jastrow_from_pyscf(qmc.mf, jastrow_order=2)
            
            vmc_calc = VMCCalculator(
                wave_function=slater,
                atom_positions=coordinates,
                atom_charges=atom_charges,
                n_walkers=50,
                step_size=0.2
            )
            
            vmc_result = vmc_calc.run(
                n_electrons=2,  # Only H2 electrons active
                n_samples=3000,
                n_equil=500
            )
            
            print(f"  HF:  {hf_result['energy']:.6f} Ha")
            print(f"  VMC: {vmc_result.energy:.6f} ± {vmc_result.energy_error:.6f} Ha")
            
            results.append({
                'r_hh': r_hh,
                'hf_energy': hf_result['energy'],
                'vmc_energy': vmc_result.energy,
                'vmc_error': vmc_result.energy_error
            })
            
        except Exception as e:
            print(f"  Calculation failed: {e}")
    
    # Find barrier
    if results:
        hf_energies = [r['hf_energy'] for r in results]
        vmc_energies = [r['vmc_energy'] for r in results]
        
        hf_barrier = max(hf_energies) - min(hf_energies)
        # Find VMC barrier relative to minimum
        vmc_min_idx = np.argmin(vmc_energies)
        vmc_max = max(vmc_energies)
        vmc_barrier = vmc_max - vmc_energies[vmc_min_idx]
        
        print(f"\n{'='*60}")
        print("BARRIER SUMMARY")
        print(f"{'='*60}")
        print(f"HF barrier:  {hf_barrier*27.2114:.3f} eV")
        print(f"VMC barrier: {vmc_barrier*27.2114:.3f} eV")
        print()
    
    return results


def run_co_adsorption():
    """
    CO adsorption on a metal cluster (simplified model).
    Uses a small metal cluster to mimic surface adsorption.
    """
    print("="*60)
    print("CO Adsorption (Simplified Model)")
    print("="*60)
    print()
    
    # Metal cluster (Li4 as simple model for metal)
    metal_cluster = np.array([
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    
    # CO molecule
    r_co = 1.13  # CO bond length
    co_molecule = np.array([
        [0.0, 0.0, 2.5],      # C above surface
        [0.0, 0.0, 2.5 + r_co]  # O
    ])
    
    # Combined system
    coordinates = np.vstack([metal_cluster, co_molecule])
    atom_symbols = ['Li', 'Li', 'Li', 'Li', 'C', 'O']
    atom_charges = np.array([3, 3, 3, 3, 6, 8])
    
    print("System: Li4 + CO (simplified adsorption model)")
    print()
    
    # Test different CO heights
    heights = [2.0, 2.5, 3.0, 3.5, 4.0]
    results = []
    
    for height in heights:
        print(f"CO height: {height} Å")
        
        # Update coordinates
        coords = np.vstack([metal_cluster, 
                          np.array([[0.0, 0.0, height], 
                                   [0.0, 0.0, height + r_co]])])
        
        try:
            qmc = PySCFQMCInterface(
                atom_symbols=atom_symbols,
                coordinates=coords,
                basis='cc-pVDZ',
                charge=0,
                spin=0
            )
            
            hf_result = qmc.run_hf()
            
            print(f"  HF Energy: {hf_result['energy']:.6f} Ha")
            
            results.append({
                'height': height,
                'hf_energy': hf_result['energy']
            })
            
        except Exception as e:
            print(f"  Calculation failed: {e}")
        print()
    
    # Calculate adsorption energy
    if len(results) >= 2:
        e_adsorbed = min(r['hf_energy'] for r in results)
        e_far = max(r['hf_energy'] for r in results)
        
        # Reference energies (isolated)
        e_li4 = -29.0  # Approximate
        e_co = -112.8  # Approximate (Hartree)
        
        adsorption_energy = e_adsorbed - e_far
        
        print(f"{'='*60}")
        print("ADSORPTION SUMMARY")
        print(f"{'='*60}")
        print(f"Adsorption energy estimate: {adsorption_energy*27.2114:.3f} eV")
        print(f"(Negative = favorable adsorption)")
        print()
    
    return results


def run_reaction_pathway():
    """
    Calculate reaction pathway using the nudged elastic band (NEB) concept.
    Simplified: just calculate energies along interpolated path.
    """
    print("="*60)
    print("Reaction Pathway: H2 Dissociation")
    print("="*60)
    print()
    
    # Initial state: H2 molecule
    r_initial = np.array([
        [0.0, 0.0, 2.0],
        [0.74, 0.0, 2.0]
    ])
    
    # Final state: Dissociated H atoms
    r_final = np.array([
        [-1.5, 0.0, 2.0],
        [1.5, 0.0, 2.0]
    ])
    
    # Interpolate path
    n_images = 5
    
    print(f"Computing {n_images} images along reaction path...")
    print()
    
    results = []
    
    for i in range(n_images):
        alpha = i / (n_images - 1)
        coords = (1 - alpha) * r_initial + alpha * r_final
        
        print(f"Image {i+1}/{n_images} (alpha={alpha:.2f})")
        
        try:
            qmc = PySCFQMCInterface(
                atom_symbols=['H', 'H'],
                coordinates=coords,
                basis='cc-pVTZ',
                charge=0,
                spin=0
            )
            
            hf_result = qmc.run_hf()
            
            # VMC
            slater = create_slater_jastrow_from_pyscf(qmc.mf, jastrow_order=2)
            vmc_calc = VMCCalculator(
                wave_function=slater,
                atom_positions=coords,
                atom_charges=np.array([1, 1]),
                n_walkers=50,
                step_size=0.3
            )
            
            vmc_result = vmc_calc.run(n_electrons=2, n_samples=2000, n_equil=500)
            
            print(f"  HF:  {hf_result['energy']:.6f} Ha")
            print(f"  VMC: {vmc_result.energy:.6f} ± {vmc_result.energy_error:.6f} Ha")
            
            results.append({
                'image': i,
                'alpha': alpha,
                'hf_energy': hf_result['energy'],
                'vmc_energy': vmc_result.energy,
                'vmc_error': vmc_result.energy_error
            })
            
        except Exception as e:
            print(f"  Calculation failed: {e}")
        print()
    
    # Summary
    if results:
        print(f"{'='*60}")
        print("REACTION PATHWAY SUMMARY")
        print(f"{'='*60}")
        
        hf_energies = [r['hf_energy'] for r in results]
        vmc_energies = [r['vmc_energy'] for r in results]
        
        hf_barrier = max(hf_energies) - hf_energies[0]
        vmc_barrier = max(vmc_energies) - vmc_energies[0]
        
        print(f"HF barrier:  {hf_barrier*27.2114:.3f} eV")
        print(f"VMC barrier: {vmc_barrier*27.2114:.3f} eV")
        print()
    
    return results


def run_all_catalysis_calculations():
    """Run all catalysis calculations."""
    print("\n" + "="*70)
    print("QMC CATALYSIS CALCULATIONS")
    print("="*70 + "\n")
    
    all_results = {}
    
    # H2 dissociation
    print("1. H2 DISSOCIATION BARRIER\n")
    all_results['h2_dissociation'] = run_h2_dissociation_barrier()
    
    print("\n")
    
    # CO adsorption
    print("2. CO ADSORPTION\n")
    all_results['co_adsorption'] = run_co_adsorption()
    
    print("\n")
    
    # Reaction pathway
    print("3. REACTION PATHWAY\n")
    all_results['reaction_pathway'] = run_reaction_pathway()
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='QMC Catalysis Calculations')
    parser.add_argument('--calculation',
                       choices=['dissociation', 'adsorption', 'pathway', 'all'],
                       default='all',
                       help='Calculation type')
    
    args = parser.parse_args()
    
    if args.calculation == 'all':
        run_all_catalysis_calculations()
    elif args.calculation == 'dissociation':
        run_h2_dissociation_barrier()
    elif args.calculation == 'adsorption':
        run_co_adsorption()
    elif args.calculation == 'pathway':
        run_reaction_pathway()
