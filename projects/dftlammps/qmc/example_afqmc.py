"""
QMC Example: Be Atom with AFQMC
================================

Demonstrates AFQMC calculation on the Beryllium atom.
AFQMC provides an alternative to DMC for certain systems.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pyscf_qmc_interface import PySCFQMCInterface
from afqmc_calculator import AFQMCCalculator, create_hamiltonian_from_pyscf


def main():
    print("="*70)
    print("AFQMC EXAMPLE: Be ATOM")
    print("="*70)
    print()
    
    # Create Be atom
    print("System: Be atom")
    print("Exact energy reference: -14.6674 Ha")
    print()
    
    qmc = PySCFQMCInterface(
        atom_symbols=['Be'],
        coordinates=np.array([[0.0, 0.0, 0.0]]),
        basis='cc-pVDZ',
        charge=0,
        spin=0
    )
    
    # Run HF
    hf_result = qmc.run_hf()
    print(f"HF Energy: {hf_result['energy']:.6f} Ha")
    print()
    
    # Create Hamiltonian
    print("Building Hamiltonian...")
    hamiltonian = create_hamiltonian_from_pyscf(qmc.mf)
    print()
    
    # Run AFQMC
    print("Running AFQMC...")
    afqmc = AFQMCCalculator(
        hamiltonian=hamiltonian,
        n_walkers=100,
        time_step=0.01,
        seed=42
    )
    
    # Set trial wave function
    afqmc.set_trial_wavefunction(qmc.mf.mo_coeff, n_elec=4)
    
    result = afqmc.run(n_elec=4, n_steps=5000, n_equil=1000)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"AFQMC Energy: {result.energy:.6f} ± {result.energy_error:.6f} Ha")
    print(f"Average phase: {result.avg_phase:.4f}")
    print(f"Final walkers: {result.n_walkers_final}")
    print()
    print(f"Exact: -14.6674 Ha")
    print(f"Error: {abs(result.energy - (-14.6674)):.4f} Ha")
    print()
    
    return result


if __name__ == '__main__':
    main()
