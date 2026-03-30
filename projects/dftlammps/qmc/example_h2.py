"""
QMC Example: H2 Molecule
========================

Complete workflow for H2 molecule QMC calculation.
Demonstrates:
1. Wave function preparation with PySCF
2. VMC calculation with Slater-Jastrow ansatz
3. DMC calculation with fixed-node approximation
4. Statistical analysis
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pyscf_qmc_interface import PySCFQMCInterface
from vmc_calculator import VMCCalculator, create_slater_jastrow_from_pyscf
from dmc_calculator import DMCCalculator, create_trial_wf_from_vmc
from qmc_analysis import analyze_monte_carlo_data, blocking_analysis


def main():
    print("="*70)
    print("QMC EXAMPLE: H2 MOLECULE")
    print("="*70)
    print()
    
    # Step 1: Define system
    print("Step 1: System Setup")
    print("-" * 40)
    
    bond_length = 0.74  # Angstrom (equilibrium)
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [bond_length, 0.0, 0.0]
    ])
    
    print(f"H2 molecule")
    print(f"Bond length: {bond_length} Å")
    print()
    
    # Step 2: Create PySCF interface and run HF
    print("Step 2: Hartree-Fock Calculation")
    print("-" * 40)
    
    qmc = PySCFQMCInterface(
        atom_symbols=['H', 'H'],
        coordinates=coordinates,
        basis='cc-pVTZ',
        charge=0,
        spin=0
    )
    
    hf_result = qmc.run_hf()
    print(f"HF Energy: {hf_result['energy']:.6f} Ha")
    print(f"Converged: {hf_result['converged']}")
    print()
    
    # Step 3: Run DFT for comparison
    print("Step 3: DFT Calculation (PBE)")
    print("-" * 40)
    
    dft_result = qmc.run_dft(xc='PBE')
    print(f"PBE Energy: {dft_result['energy']:.6f} Ha")
    print()
    
    # Step 4: VMC calculation
    print("Step 4: VMC Calculation")
    print("-" * 40)
    
    # Create Slater-Jastrow wave function
    slater = create_slater_jastrow_from_pyscf(qmc.mf, jastrow_order=2)
    
    vmc_calc = VMCCalculator(
        wave_function=slater,
        atom_positions=coordinates,
        atom_charges=np.array([1, 1]),
        n_walkers=100,
        step_size=0.2,
        seed=42
    )
    
    print("Running VMC sampling...")
    vmc_result = vmc_calc.run(
        n_electrons=2,
        n_samples=10000,
        n_equil=1000
    )
    
    print(f"VMC Energy:     {vmc_result.energy:.6f} Ha")
    print(f"Standard Error: {vmc_result.energy_error:.6f} Ha")
    print(f"Variance:       {vmc_result.variance:.6f}")
    print(f"Acceptance:     {vmc_result.acceptance_rate:.3f}")
    print()
    
    # Statistical analysis of VMC
    print("VMC Statistical Analysis:")
    energies = np.array([s.local_energy for s in vmc_result.samples])
    vmc_analysis = analyze_monte_carlo_data(energies, equilibration_fraction=0.2)
    
    print(f"  Blocking error:  {vmc_analysis['std_error_blocking']:.6f} Ha")
    print(f"  Autocorr. time:  {vmc_analysis['autocorrelation_time']:.2f}")
    print(f"  Effective N:     {vmc_analysis['effective_samples']:.0f}")
    print()
    
    # Step 5: Optimize wave function (optional)
    print("Step 5: Wave Function Optimization")
    print("-" * 40)
    
    print("Running 20 optimization steps...")
    opt_result = vmc_calc.optimize_wavefunction(
        n_electrons=2,
        n_opt_samples=2000,
        n_opt_steps=20
    )
    
    # Re-run VMC with optimized parameters
    vmc_result_opt = vmc_calc.run(
        n_electrons=2,
        n_samples=10000,
        n_equil=500
    )
    
    print(f"VMC (optimized): {vmc_result_opt.energy:.6f} Ha")
    print(f"Improvement:     {vmc_result.energy - vmc_result_opt.energy:.6f} Ha")
    print()
    
    # Step 6: DMC calculation
    print("Step 6: DMC Calculation")
    print("-" * 40)
    
    # Create trial wave function from optimized VMC
    trial_wf = create_trial_wf_from_vmc(slater)
    
    dmc_calc = DMCCalculator(
        trial_wf=trial_wf,
        atom_positions=coordinates,
        atom_charges=np.array([1, 1]),
        n_walkers_initial=500,
        time_step=0.01,
        target_walkers=500,
        seed=42
    )
    
    print("Running DMC...")
    dmc_result = dmc_calc.run(
        n_electrons=2,
        n_steps=10000,
        n_equil=1000
    )
    
    print(f"DMC Energy:      {dmc_result.energy:.6f} Ha")
    print(f"Standard Error:  {dmc_result.energy_error:.6f} Ha")
    print(f"Trial Energy:    {dmc_result.energy_trial:.6f} Ha")
    print(f"Acceptance:      {dmc_result.acceptance_rate:.3f}")
    print(f"Avg Walkers:     {dmc_result.n_walkers_avg:.0f}")
    print()
    
    # Step 7: Time-step extrapolation
    print("Step 7: Time-Step Extrapolation (Brief)")
    print("-" * 40)
    
    print("Testing 3 time steps for extrapolation...")
    tau_values = [0.005, 0.01, 0.02]
    tau_energies = []
    
    for tau in tau_values:
        dmc_tau = DMCCalculator(
            trial_wf=trial_wf,
            atom_positions=coordinates,
            atom_charges=np.array([1, 1]),
            n_walkers_initial=300,
            time_step=tau
        )
        
        result = dmc_tau.run(n_electrons=2, n_steps=3000, n_equil=500)
        tau_energies.append(result.energy)
        print(f"  tau={tau}: E={result.energy:.6f} Ha")
    
    # Linear fit to tau=0
    p = np.polyfit(tau_values, tau_energies, 1)
    e_extrapolated = p[1]
    
    print(f"Extrapolated (tau=0): {e_extrapolated:.6f} Ha")
    print()
    
    # Final summary
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()
    print(f"Method          Energy (Ha)      Error (Ha)")
    print("-" * 45)
    print(f"HF              {hf_result['energy']:10.6f}      -")
    print(f"PBE             {dft_result['energy']:10.6f}      -")
    print(f"VMC (initial)   {vmc_result.energy:10.6f}      {vmc_result.energy_error:.6f}")
    print(f"VMC (optimized) {vmc_result_opt.energy:10.6f}      {vmc_result_opt.energy_error:.6f}")
    print(f"DMC             {dmc_result.energy:10.6f}      {dmc_result.energy_error:.6f}")
    print(f"DMC (extrap.)   {e_extrapolated:10.6f}      -")
    print()
    print("Reference CCSD(T)/CBS: ~-1.174 Ha")
    print()
    
    # Save results
    results_dict = {
        'system': 'H2',
        'bond_length': bond_length,
        'hf_energy': hf_result['energy'],
        'pbe_energy': dft_result['energy'],
        'vmc_energy': vmc_result.energy,
        'vmc_error': vmc_result.energy_error,
        'vmc_optimized': vmc_result_opt.energy,
        'dmc_energy': dmc_result.energy,
        'dmc_error': dmc_result.energy_error,
        'dmc_extrapolated': e_extrapolated
    }
    
    print(f"Results saved to h2_qmc_results.npy")
    np.save('h2_qmc_results.npy', results_dict)
    
    return results_dict


if __name__ == '__main__':
    main()
