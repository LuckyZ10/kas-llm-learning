"""
Topology Module Examples
=========================

This script demonstrates the usage of the DFTLammps topology module
for topological materials calculations.
"""

import numpy as np

# ============================================================================
# Example 1: Z2 Invariant Calculation for Bi2Se3
# ============================================================================

def example_z2_invariant():
    """Calculate Z2 invariant for Bi2Se3 topological insulator."""
    print("=" * 70)
    print("Example 1: Z2 Invariant Calculation for Bi2Se3")
    print("=" * 70)
    
    from dftlammps.topology import (
        Z2PackConfig, Z2VASPInterface, classify_topological_material
    )
    
    # Configure Z2Pack calculation
    config = Z2PackConfig(
        num_bands=40,
        num_wann=20,
        surface="kz-surface",  # For (111) surface
        num_lines=11,
        pos_tol=0.01,
        gap_tol=0.1,
    )
    
    print("\nConfiguration:")
    print(f"  Number of bands: {config.num_bands}")
    print(f"  Number of Wannier functions: {config.num_wann}")
    print(f"  Surface: {config.surface}")
    
    # Initialize interface (requires VASP output)
    # interface = Z2VASPInterface("./Bi2Se3", config)
    # result = interface.calculate_z2_invariant()
    
    print("\nNote: This requires VASP WAVECAR and Wannier90 output")
    print("Expected result for Bi2Se3: Z2 = 1 (strong topological insulator)")


# ============================================================================
# Example 2: Berry Curvature and Chern Number
# ============================================================================

def example_berry_curvature():
    """Calculate Berry curvature and Chern number."""
    print("\n" + "=" * 70)
    print("Example 2: Berry Curvature and Chern Number")
    print("=" * 70)
    
    from dftlammps.topology import (
        BerryPhaseConfig, BerryCurvatureCalculator, BerryCurvatureMethod
    )
    
    # Configure calculation
    config = BerryPhaseConfig(
        method=BerryCurvatureMethod.FINITE_DIFFERENCE,
        berry_k_mesh=(30, 30, 1),  # 2D mesh for slab
    )
    
    print("\nConfiguration:")
    print(f"  Method: {config.method.value}")
    print(f"  k-mesh: {config.berry_k_mesh}")
    
    # Initialize calculator
    # calculator = BerryCurvatureCalculator("./calculation", config)
    # result = calculator.calculate_berry_curvature()
    
    print("\nNote: This requires VASP WAVECAR output")
    print("Chern number C = (1/2π) ∫∫ Ω_z d²k")


# ============================================================================
# Example 3: Weyl Point Search in TaAs
# ============================================================================

def example_weyl_points():
    """Search for Weyl points in TaAs."""
    print("\n" + "=" * 70)
    print("Example 3: Weyl Point Search in TaAs")
    print("=" * 70)
    
    from dftlammps.weyl import (
        WeylSemimetalConfig, WeylPointLocator, analyze_weyl_semimetal
    )
    
    # Configure Weyl point search
    config = WeylSemimetalConfig(
        search_window=(-0.5, 0.5),  # Energy window around E_F
        k_mesh_fine=(50, 50, 50),
        gap_threshold=0.001,  # eV
    )
    
    print("\nConfiguration:")
    print(f"  Energy window: {config.search_window} eV")
    print(f"  k-mesh: {config.k_mesh_fine}")
    print(f"  Gap threshold: {config.gap_threshold} eV")
    
    # Search for Weyl points
    # locator = WeylPointLocator("./TaAs", config)
    # weyl_points = locator.search_weyl_points()
    
    print("\nExpected for TaAs:")
    print("  - 24 Weyl points (12 pairs)")
    print("  - 12 with chirality +1")
    print("  - 12 with chirality -1")
    print("  - Fermi arcs on (001) surface")
    
    # Alternative: complete analysis
    # results = analyze_weyl_semimetal("./TaAs")


# ============================================================================
# Example 4: Fermi Arc Calculation
# ============================================================================

def example_fermi_arcs():
    """Calculate Fermi arcs on surface of Weyl semimetal."""
    print("\n" + "=" * 70)
    print("Example 4: Fermi Arc Surface States")
    print("=" * 70)
    
    from dftlammps.weyl import FermiArcCalculator, WeylSemimetalConfig
    
    config = WeylSemimetalConfig(
        surface_thickness=10,
        k_mesh_surface=(100, 100),
    )
    
    print("\nConfiguration:")
    print(f"  Surface thickness: {config.surface_thickness} layers")
    print(f"  Surface k-mesh: {config.k_mesh_surface}")
    
    # Calculate Fermi arcs
    # calculator = FermiArcCalculator("./TaAs", config)
    # fermi_arcs = calculator.calculate_fermi_arcs(weyl_points)
    
    print("\nFermi arcs connect Weyl points with opposite chirality")
    print("Expected arc length in TaAs: ~0.1-0.2 Å⁻¹")


# ============================================================================
# Example 5: Quantum Anomalous Hall Effect
# ============================================================================

def example_qahe():
    """Analyze Quantum Anomalous Hall Effect."""
    print("\n" + "=" * 70)
    print("Example 5: Quantum Anomalous Hall Effect")
    print("=" * 70)
    
    from dftlammps.topology import (
        AnomalousHallConductivityCalculator, BerryPhaseConfig
    )
    
    config = BerryPhaseConfig()
    calculator = AnomalousHallConductivityCalculator("./Cr_Bi2Te3", config)
    
    print("\nCalculating anomalous Hall conductivity...")
    # result = calculator.calculate_ahc(temperature=0.0)
    
    print("\nExpected for QAHE:")
    print("  σ_xy = e²/h = 3.87 × 10⁻⁵ S")
    print("  Quantized to integer multiples of e²/h")
    print("  Chern number C = 1 for single edge channel")


# ============================================================================
# Example 6: Complete Workflow - Bi2Se3
# ============================================================================

def example_bi2se3_workflow():
    """Complete workflow for Bi2Se3 topological insulator."""
    print("\n" + "=" * 70)
    print("Example 6: Complete Bi2Se3 Analysis Workflow")
    print("=" * 70)
    
    from dftlammps.applications.case_topological_insulator import (
        Bi2Se3Workflow, TopologicalInsulatorConfig
    )
    
    # Configure workflow
    config = TopologicalInsulatorConfig(
        material="Bi2Se3",
        encut=500,
        num_bands=40,
        calculate_surface_states=True,
    )
    
    workflow = Bi2Se3Workflow(config)
    
    print("\nWorkflow steps:")
    print("  1. Generate crystal structure (R-3m space group)")
    print("  2. Generate VASP input with SOC")
    print("  3. Calculate Z2 invariant")
    print("  4. Calculate surface states")
    print("  5. Verify topological protection")
    
    # Run workflow (requires VASP)
    # results = workflow.run_full_analysis("./Bi2Se3")


# ============================================================================
# Example 7: Complete Workflow - TaAs
# ============================================================================

def example_taas_workflow():
    """Complete workflow for TaAs Weyl semimetal."""
    print("\n" + "=" * 70)
    print("Example 7: Complete TaAs Analysis Workflow")
    print("=" * 70)
    
    from dftlammps.applications.case_weyl_semimetal import (
        TaAsWorkflow, TaAsConfig
    )
    
    config = TaAsConfig(
        material="TaAs",
        encut=600,
        lsorbit=True,
    )
    
    workflow = TaAsWorkflow(config)
    
    print("\nWorkflow steps:")
    print("  1. Generate crystal structure (I4_1md space group)")
    print("  2. Generate VASP input with SOC")
    print("  3. Locate Weyl points (expect 24)")
    print("  4. Calculate chirality for each point")
    print("  5. Calculate Fermi arcs")
    print("  6. Analyze chiral anomaly")
    
    # Run workflow (requires VASP)
    # results = workflow.run_full_analysis("./TaAs")


# ============================================================================
# Example 8: Complete Workflow - QAHE
# ============================================================================

def example_qahe_workflow():
    """Complete workflow for Quantum Anomalous Hall Effect."""
    print("\n" + "=" * 70)
    print("Example 8: QAHE Analysis Workflow")
    print("=" * 70)
    
    from dftlammps.applications.case_quantum_anomalous_hall import (
        QAHEWorkflow, QAHEConfig
    )
    
    config = QAHEConfig(
        base_material="Bi2Te3",
        dopant="Cr",
        doping_concentration=0.08,
        magnetic_order="ferromagnetic",
    )
    
    workflow = QAHEWorkflow(config)
    
    print("\nWorkflow steps:")
    print("  1. Generate Cr-doped Bi2Te3 structure")
    print("  2. Apply ferromagnetic ordering")
    print("  3. Generate VASP input (SOC + magnetism)")
    print("  4. Calculate Chern number")
    print("  5. Calculate anomalous Hall conductivity")
    print("  6. Verify quantization")


# ============================================================================
# Run all examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# DFTLammps Topology Module Examples")
    print("#" * 70)
    
    # Run examples
    example_z2_invariant()
    example_berry_curvature()
    example_weyl_points()
    example_fermi_arcs()
    example_qahe()
    example_bi2se3_workflow()
    example_taas_workflow()
    example_qahe_workflow()
    
    print("\n" + "#" * 70)
    print("# Examples completed!")
    print("# Note: Some examples require VASP/Wannier90 output files")
    print("#" * 70)
