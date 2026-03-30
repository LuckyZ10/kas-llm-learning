#!/usr/bin/env python
"""
Example: DMFT Calculation for NiO Mott Insulator

This script demonstrates a complete workflow for studying
NiO using DFT+DMFT methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import correlated systems modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dftlammps.correlated import (
    DMFTEngine, DMFTConfig, 
    LinearResponseU, HubbardUConfig,
    WannierProjector, WannierProjectorConfig
)

from dftlammps.mott import (
    GapAnalyzer, MetalInsulatorTransition,
    MottAnalysisConfig
)

from dftlammps.applications.case_mott_insulator import (
    NiOAnalyzer, MottInsulatorWorkflow
)


def example_1_hubbard_u_calculation():
    """
    Example 1: Calculate Hubbard U for Ni using linear response
    """
    print("=" * 60)
    print("Example 1: Linear Response Hubbard U Calculation")
    print("=" * 60)
    
    # Setup U calculator
    config = HubbardUConfig(
        lr_n_perturbations=5,
        lr_max_alpha=0.1
    )
    
    u_calc = LinearResponseU(config)
    
    # Note: This requires actual VASP calculations
    # For demonstration, we use mock data
    print("\nSetting up linear response calculation...")
    print("- Number of perturbations: 5")
    print("- Max perturbation strength: 0.1 eV")
    
    # Mock results
    results = {
        'U': 6.3,
        'J': 1.0,
        'method': 'linear_response',
        'chi0': np.random.rand(5, 5),
        'chi': np.random.rand(5, 5)
    }
    
    print(f"\nResults:")
    print(f"  U = {results['U']:.2f} eV")
    print(f"  J = {results['J']:.2f} eV")
    print(f"  U_eff = U - J = {results['U'] - results['J']:.2f} eV")
    
    return results


def example_2_dmft_calculation():
    """
    Example 2: Single-band DMFT for NiO
    """
    print("\n" + "=" * 60)
    print("Example 2: DMFT Calculation")
    print("=" * 60)
    
    # Setup DMFT configuration
    config = DMFTConfig(
        temperature=300.0,
        u_value=6.3,
        j_value=1.0,
        n_orbitals=5,  # Ni 3d
        scf_max_iter=20,
        scf_tol=1e-5
    )
    
    print("\nDMFT Configuration:")
    print(f"  Temperature: {config.temperature} K")
    print(f"  U: {config.u_value} eV")
    print(f"  J: {config.j_value} eV")
    print(f"  Max iterations: {config.scf_max_iter}")
    
    # Initialize DMFT engine
    dmft = DMFTEngine(config)
    
    # Mock Hamiltonian (would come from Wannier90)
    n_k = 10
    H_k = np.array([np.diag([-1.0, -0.5, 0.0, 0.5, 1.0]) 
                   for _ in range(n_k)])
    k_weights = np.ones(n_k) / n_k
    
    print("\nRunning self-consistent DMFT loop...")
    print("  (Using mock data for demonstration)")
    
    # Mock results
    results = {
        'G_loc': np.random.rand(2, 1024, 5, 5),
        'Sigma': np.random.rand(2, 1024, 5, 5),
        'converged': True,
        'convergence_history': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    }
    
    print(f"  Converged: {results['converged']}")
    print(f"  Final diff: {results['convergence_history'][-1]:.2e}")
    
    return results


def example_3_gap_analysis():
    """
    Example 3: Electronic gap analysis
    """
    print("\n" + "=" * 60)
    print("Example 3: Gap Analysis")
    print("=" * 60)
    
    analyzer = GapAnalyzer()
    
    # Mock eigenvalues for NiO (charge-transfer insulator)
    k_points = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
    
    # Valence band (O 2p) and conduction band (Ni 3d)
    eigenvalues = np.array([
        [-4.0, -3.5, 2.0, 3.0],  # Gamma
        [-3.8, -3.3, 2.2, 3.2],  # X
        [-3.9, -3.4, 2.1, 3.1],  # M
    ])
    
    gap_info = analyzer.calculate_gap(eigenvalues, k_points)
    
    print("\nGap Analysis Results:")
    print(f"  Fermi energy: {gap_info['E_Fermi']:.2f} eV")
    print(f"  Indirect gap: {gap_info['gap_indirect']:.2f} eV")
    print(f"  VBM: {gap_info['VBM']:.2f} eV")
    print(f"  CBM: {gap_info['CBM']:.2f} eV")
    print(f"  Is insulator: {gap_info['is_insulator']}")
    
    return gap_info


def example_4_mott_transition():
    """
    Example 4: Metal-Insulator Transition Analysis
    """
    print("\n" + "=" * 60)
    print("Example 4: Metal-Insulator Transition")
    print("=" * 60)
    
    mit = MetalInsulatorTransition()
    
    # Gap as function of U/t
    U_values = np.linspace(0, 12, 25)
    t = 0.5  # eV
    
    # Mock gap data (opens at critical U)
    U_c = 6.0  # Critical U
    gaps = np.maximum(0, U_values - U_c)
    
    results = mit.detect_mit_gap_criterion(gaps, U_values)
    
    print("\nMIT Analysis:")
    print(f"  Critical U: {results['transition_points'][0]:.2f} eV" 
          if results['transition_points'] else "  No transition found")
    print(f"  Phase sequence: {set(results['phases'])}")
    
    # Phase diagram
    phase_diag = mit.construct_phase_diagram(
        U_values, 
        np.linspace(100, 1000, 10),
        np.outer(gaps, np.ones(10))
    )
    
    print(f"  Phase boundary points: {len(phase_diag['boundary_U'])}")
    
    return results


def example_5_nio_complete_workflow():
    """
    Example 5: Complete NiO workflow
    """
    print("\n" + "=" * 60)
    print("Example 5: Complete NiO Analysis Workflow")
    print("=" * 60)
    
    workflow = MottInsulatorWorkflow("NiO")
    
    # Run complete analysis
    print("\nRunning complete analysis...")
    results = workflow.run_complete_analysis()
    
    # Electronic structure
    print("\nElectronic Structure:")
    print(f"  Gap type: {results['electronic']['gap_type']}")
    print(f"  Gap size: {results['electronic']['gap_size']:.2f} eV")
    print(f"  Classification: {results['electronic']['classification']}")
    
    # Magnetic properties
    print("\nMagnetic Properties:")
    print(f"  Structure: {results['magnetic']['magnetic_structure']}")
    print(f"  Neel T (exp): {results['magnetic']['T_Neel_experimental']} K")
    
    # Exchange interactions
    print("\nExchange Interactions:")
    J = results['exchange']['J_superexchange_180']
    print(f"  J_super (180°): {J*1000:.2f} meV")
    print(f"  T_Neel (est): {results['exchange']['T_Neel_estimated']:.0f} K")
    
    # Generate DFT input
    output_dir = "vasp_nio_example"
    workflow.generate_dft_input(output_dir)
    print(f"\nDFT input files generated in: {output_dir}/")
    
    return results


def example_6_triqs_interface():
    """
    Example 6: TRIQS interface for multi-orbital Hubbard model
    """
    print("\n" + "=" * 60)
    print("Example 6: TRIQS Interface (if available)")
    print("=" * 60)
    
    try:
        from dftlammps.correlated import (
            TRIQSConfig, MultiOrbitalHubbard,
            TwoParticleGF, SuperconductingSusceptibility
        )
        
        # Setup multi-orbital Hubbard model
        config = TRIQSConfig(
            beta=40.0,
            n_orbitals=5,
            U=4.0,
            J=0.6
        )
        
        hubbard = MultiOrbitalHubbard(config)
        
        # Construct interaction Hamiltonian
        H_int = hubbard.construct_interaction_hamiltonian("kanamori")
        
        print("\nTRIQS interface available!")
        print(f"  Model: 5-orbital Hubbard with Kanamori interaction")
        print(f"  U = {config.U} eV, J = {config.J} eV")
        print(f"  β = {config.beta}")
        
        # Two-particle Green's function
        chi_calc = TwoParticleGF(config)
        print("  Two-particle GF calculator initialized")
        
        # Superconducting susceptibility
        sc_calc = SuperconductingSusceptibility(config)
        print("  SC susceptibility calculator initialized")
        
    except ImportError:
        print("\nTRIQS not available - skipping this example")
        print("To use TRIQS features, install: pip install triqs")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("DFT-LAMMPS Correlated Systems - Example Script")
    print("=" * 60)
    
    # Run examples
    example_1_hubbard_u_calculation()
    example_2_dmft_calculation()
    example_3_gap_analysis()
    example_4_mott_transition()
    example_5_nio_complete_workflow()
    example_6_triqs_interface()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()