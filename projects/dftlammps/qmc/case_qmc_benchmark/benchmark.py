"""
QMC Benchmark Calculations
===========================

Benchmark QMC calculations on standard test systems:
- H atom (exact solution)
- H2 molecule
- He atom
- LiH molecule
- Be atom

Compares VMC and DMC results with exact/HF/CCSD(T) values.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyscf_qmc_interface import PySCFQMCInterface
from vmc_calculator import VMCCalculator, SlaterJastrow, create_slater_jastrow_from_pyscf
from dmc_calculator import DMCCalculator, TrialWaveFunction, create_trial_wf_from_vmc
from qmc_analysis import analyze_monte_carlo_data, save_analysis_results


def run_h_atom_benchmark(n_vmc_samples=5000, n_dmc_steps=5000):
    """
    Hydrogen atom benchmark.
    Exact energy: -0.5 Ha
    """
    print("="*60)
    print("H Atom Benchmark")
    print("="*60)
    print(f"Exact energy: -0.5 Ha")
    print()
    
    # Create H atom
    qmc = PySCFQMCInterface(
        atom_symbols=['H'],
        coordinates=np.array([[0.0, 0.0, 0.0]]),
        basis='cc-pVTZ',
        charge=0,
        spin=1
    )
    
    # Run HF
    hf_result = qmc.run_hf()
    print(f"HF Energy: {hf_result['energy']:.6f} Ha")
    print(f"Exact: -0.500000 Ha")
    print()
    
    # VMC calculation
    print("Running VMC...")
    slater = create_slater_jastrow_from_pyscf(qmc.mf, jastrow_order=1)
    
    vmc_calc = VMCCalculator(
        wave_function=slater,
        atom_positions=qmc.coordinates,
        atom_charges=np.array([1]),
        n_walkers=50,
        step_size=0.5
    )
    
    vmc_result = vmc_calc.run(
        n_electrons=1,
        n_samples=n_vmc_samples,
        n_equil=500
    )
    
    print(f"VMC Energy: {vmc_result.energy:.6f} ± {vmc_result.energy_error:.6f} Ha")
    print(f"VMC Variance: {vmc_result.variance:.6f}")
    print()
    
    # DMC calculation
    print("Running DMC...")
    trial_wf = create_trial_wf_from_vmc(slater)
    
    dmc_calc = DMCCalculator(
        trial_wf=trial_wf,
        atom_positions=qmc.coordinates,
        atom_charges=np.array([1]),
        n_walkers_initial=200,
        time_step=0.01
    )
    
    dmc_result = dmc_calc.run(
        n_electrons=1,
        n_steps=n_dmc_steps,
        n_equil=500
    )
    
    print(f"DMC Energy: {dmc_result.energy:.6f} ± {dmc_result.energy_error:.6f} Ha")
    print()
    
    results = {
        'system': 'H_atom',
        'exact': -0.5,
        'hf': hf_result['energy'],
        'vmc': {
            'energy': vmc_result.energy,
            'error': vmc_result.energy_error,
            'variance': vmc_result.variance
        },
        'dmc': {
            'energy': dmc_result.energy,
            'error': dmc_result.energy_error,
            'variance': dmc_result.variance
        }
    }
    
    return results


def run_h2_benchmark(bond_length=0.74, n_vmc_samples=8000, n_dmc_steps=8000):
    """
    H2 molecule benchmark at equilibrium geometry.
    
    Parameters:
    -----------
    bond_length : float
        H-H bond length in Angstrom (default: 0.74)
    """
    print("="*60)
    print("H2 Molecule Benchmark")
    print("="*60)
    print(f"Bond length: {bond_length} Å")
    print(f"Reference (CCSD(T)/CBS): ~-1.174 Ha")
    print()
    
    # Create H2
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [bond_length, 0.0, 0.0]
    ])
    
    qmc = PySCFQMCInterface(
        atom_symbols=['H', 'H'],
        coordinates=coordinates,
        basis='cc-pVTZ',
        charge=0,
        spin=0
    )
    
    # Run HF
    hf_result = qmc.run_hf()
    print(f"HF Energy: {hf_result['energy']:.6f} Ha")
    
    # Run CCSD
    try:
        ccsd_result = qmc.run_ccsd(with_t=True)
        print(f"CCSD(T) Energy: {ccsd_result['e_tot']:.6f} Ha")
    except:
        print("CCSD(T) calculation failed")
    
    print()
    
    # VMC calculation
    print("Running VMC...")
    slater = create_slater_jastrow_from_pyscf(qmc.mf, jastrow_order=2)
    
    vmc_calc = VMCCalculator(
        wave_function=slater,
        atom_positions=coordinates,
        atom_charges=np.array([1, 1]),
        n_walkers=100,
        step_size=0.2
    )
    
    vmc_result = vmc_calc.run(
        n_electrons=2,
        n_samples=n_vmc_samples,
        n_equil=1000
    )
    
    print(f"VMC Energy: {vmc_result.energy:.6f} ± {vmc_result.energy_error:.6f} Ha")
    print()
    
    # Optimize Jastrow
    print("Optimizing wave function...")
    opt_result = vmc_calc.optimize_wavefunction(
        n_electrons=2,
        n_opt_samples=2000,
        n_opt_steps=20
    )
    
    # Rerun VMC with optimized parameters
    vmc_result_opt = vmc_calc.run(
        n_electrons=2,
        n_samples=n_vmc_samples,
        n_equil=500
    )
    print(f"VMC (optimized): {vmc_result_opt.energy:.6f} ± {vmc_result_opt.energy_error:.6f} Ha")
    print()
    
    # DMC calculation
    print("Running DMC...")
    trial_wf = create_trial_wf_from_vmc(slater)
    
    dmc_calc = DMCCalculator(
        trial_wf=trial_wf,
        atom_positions=coordinates,
        atom_charges=np.array([1, 1]),
        n_walkers_initial=500,
        time_step=0.005
    )
    
    dmc_result = dmc_calc.run(
        n_electrons=2,
        n_steps=n_dmc_steps,
        n_equil=1000
    )
    
    print(f"DMC Energy: {dmc_result.energy:.6f} ± {dmc_result.energy_error:.6f} Ha")
    print()
    
    results = {
        'system': 'H2',
        'bond_length': bond_length,
        'hf': hf_result['energy'],
        'vmc': {
            'energy': vmc_result.energy,
            'error': vmc_result.energy_error,
            'variance': vmc_result.variance
        },
        'vmc_optimized': {
            'energy': vmc_result_opt.energy,
            'error': vmc_result_opt.energy_error,
            'variance': vmc_result_opt.variance
        },
        'dmc': {
            'energy': dmc_result.energy,
            'error': dmc_result.energy_error,
            'variance': dmc_result.variance
        }
    }
    
    return results


def run_he_atom_benchmark(n_vmc_samples=5000, n_dmc_steps=5000):
    """Helium atom benchmark."""
    print("="*60)
    print("He Atom Benchmark")
    print("="*60)
    print(f"Exact energy: -2.9037 Ha")
    print()
    
    qmc = PySCFQMCInterface(
        atom_symbols=['He'],
        coordinates=np.array([[0.0, 0.0, 0.0]]),
        basis='cc-pVTZ',
        charge=0,
        spin=0
    )
    
    hf_result = qmc.run_hf()
    print(f"HF Energy: {hf_result['energy']:.6f} Ha")
    
    # CCSD
    try:
        ccsd_result = qmc.run_ccsd(with_t=True)
        print(f"CCSD(T) Energy: {ccsd_result['e_tot']:.6f} Ha")
    except Exception as e:
        print(f"CCSD failed: {e}")
    print()
    
    # VMC
    slater = create_slater_jastrow_from_pyscf(qmc.mf, jastrow_order=2)
    vmc_calc = VMCCalculator(
        wave_function=slater,
        atom_positions=qmc.coordinates,
        atom_charges=np.array([2]),
        n_walkers=50,
        step_size=0.2
    )
    
    vmc_result = vmc_calc.run(n_electrons=2, n_samples=n_vmc_samples, n_equil=500)
    print(f"VMC Energy: {vmc_result.energy:.6f} ± {vmc_result.energy_error:.6f} Ha")
    print()
    
    # DMC
    trial_wf = create_trial_wf_from_vmc(slater)
    dmc_calc = DMCCalculator(
        trial_wf=trial_wf,
        atom_positions=qmc.coordinates,
        atom_charges=np.array([2]),
        n_walkers_initial=200,
        time_step=0.01
    )
    
    dmc_result = dmc_calc.run(n_electrons=2, n_steps=n_dmc_steps, n_equil=500)
    print(f"DMC Energy: {dmc_result.energy:.6f} ± {dmc_result.energy_error:.6f} Ha")
    print()
    
    return {
        'system': 'He',
        'exact': -2.9037,
        'hf': hf_result['energy'],
        'vmc': {'energy': vmc_result.energy, 'error': vmc_result.energy_error},
        'dmc': {'energy': dmc_result.energy, 'error': dmc_result.energy_error}
    }


def run_all_benchmarks(save_results=True):
    """Run all benchmark calculations."""
    print("\n" + "="*70)
    print("QMC BENCHMARK SUITE")
    print("="*70 + "\n")
    
    all_results = {}
    
    # H atom
    try:
        all_results['H'] = run_h_atom_benchmark()
    except Exception as e:
        print(f"H atom benchmark failed: {e}")
    
    print("\n")
    
    # H2 molecule
    try:
        all_results['H2'] = run_h2_benchmark()
    except Exception as e:
        print(f"H2 benchmark failed: {e}")
    
    print("\n")
    
    # He atom
    try:
        all_results['He'] = run_he_atom_benchmark()
    except Exception as e:
        print(f"He benchmark failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    for system, results in all_results.items():
        print(f"\n{system}:")
        if 'exact' in results:
            print(f"  Exact:   {results['exact']:.6f} Ha")
        print(f"  HF:      {results['hf']:.6f} Ha")
        print(f"  VMC:     {results['vmc']['energy']:.6f} ± {results['vmc']['error']:.6f} Ha")
        print(f"  DMC:     {results['dmc']['energy']:.6f} ± {results['dmc']['error']:.6f} Ha")
    
    if save_results:
        output_file = Path(__file__).parent / 'benchmark_results.json'
        save_analysis_results(all_results, str(output_file))
        print(f"\nResults saved to {output_file}")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='QMC Benchmark Calculations')
    parser.add_argument('--system', choices=['H', 'H2', 'He', 'all'], default='all',
                       help='System to run benchmark on')
    parser.add_argument('--vmc-samples', type=int, default=5000,
                       help='Number of VMC samples')
    parser.add_argument('--dmc-steps', type=int, default=5000,
                       help='Number of DMC steps')
    
    args = parser.parse_args()
    
    if args.system == 'all':
        run_all_benchmarks()
    elif args.system == 'H':
        run_h_atom_benchmark(args.vmc_samples, args.dmc_steps)
    elif args.system == 'H2':
        run_h2_benchmark(n_vmc_samples=args.vmc_samples, n_dmc_steps=args.dmc_steps)
    elif args.system == 'He':
        run_he_atom_benchmark(args.vmc_samples, args.dmc_steps)
