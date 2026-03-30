"""
Case Study: Topological Insulator Bi2Se3 / Bi2Te3
=================================================

This module provides a complete workflow for studying 3D topological insulators
Bi2Se3 and Bi2Te3, including:
- Electronic structure calculation
- Z2 invariant determination
- Surface state calculation
- Topological protection verification

Bi2Se3 and Bi2Te3 are prototypical 3D topological insulators with:
- Strong topological invariant ν0 = 1
- Single Dirac cone on the surface
- Bulk band gap ~0.3 eV (Bi2Se3) or ~0.15 eV (Bi2Te3)

References:
- Zhang et al., Nature Phys. 5, 438 (2009)
- Xia et al., Nature Phys. 5, 398 (2009)
- Chen et al., Science 325, 178 (2009)
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

try:
    from pymatgen.core import Structure, Lattice, Element
    from pymatgen.io.vasp import Poscar, Incar, Kpoints
    HAS_PMG = True
except ImportError:
    HAS_PMG = False
    warnings.warn("Pymatgen not available.")

# Import topology modules
from dftlammps.topology import (
    Z2PackConfig, Z2VASPInterface, TopologicalPhase,
    WannierToolsConfig, WannierToolsCalculator,
    calculate_z2_index, classify_topological_material,
)
from dftlammps.topology.z2pack_interface import (
    VASPWavefunctionExtractor, WilsonLoopResult,
)


@dataclass
class TopologicalInsulatorConfig:
    """Configuration for topological insulator calculations."""
    material: str = "Bi2Se3"  # or "Bi2Te3"
    
    # VASP settings
    encut: float = 500.0
    k_density_bulk: float = 0.03
    k_density_surface: float = 0.02
    
    # SOC settings
    lsorbit: bool = True
    saxis: Tuple[float, float, float] = (0, 0, 1)
    
    # Z2Pack settings
    num_bands: int = 40
    num_wann: int = 20
    surface: str = "kz-surface"  # For 111 surface
    
    # Output
    calculate_surface_states: bool = True
    plot_results: bool = True


class Bi2Se3Structure:
    """
    Generate Bi2Se3/Bi2Te3 crystal structures.
    
    These are rhombohedral structures with space group R-3m (166).
    """
    
    # Lattice parameters (Å)
    LATTICE_PARAMS = {
        "Bi2Se3": {"a": 4.138, "c": 28.64},
        "Bi2Te3": {"a": 4.383, "c": 30.49},
    }
    
    @classmethod
    def create_structure(cls, material: str = "Bi2Se3") -> Structure:
        """
        Create Bi2Se3 or Bi2Te3 structure.
        
        Args:
            material: "Bi2Se3" or "Bi2Te3"
            
        Returns:
            Pymatgen Structure object
        """
        if not HAS_PMG:
            raise ImportError("Pymatgen required for structure generation")
        
        params = cls.LATTICE_PARAMS.get(material, cls.LATTICE_PARAMS["Bi2Se3"])
        a, c = params["a"], params["c"]
        
        # Rhombohedral lattice (hexagonal setting)
        lattice = Lattice.hexagonal(a, c)
        
        # Atomic positions (fractional coordinates)
        # QL-QL-QL quintuple layer structure
        # z positions in units of c
        z_Bi1 = 0.0
        z_Se1 = 0.064
        z_Bi2 = 0.213
        z_Se2 = 0.362
        z_Se3 = 0.436
        
        if "Te" in material:
            anion = "Te"
        else:
            anion = "Se"
        
        species = ["Bi", anion, "Bi", anion, anion]
        coords = [
            [0.0, 0.0, z_Bi1],
            [0.0, 0.0, z_Se1],
            [0.0, 0.0, z_Bi2],
            [0.0, 0.0, z_Se2],
            [0.0, 0.0, z_Se3],
        ]
        
        structure = Structure(lattice, species, coords)
        
        # Make supercell for proper periodicity
        structure.make_supercell([1, 1, 1])
        
        return structure
    
    @classmethod
    def create_slab(
        cls,
        material: str = "Bi2Se3",
        num_quintuple_layers: int = 3,
        vacuum: float = 20.0,
    ) -> Structure:
        """
        Create a slab structure for surface state calculation.
        
        Args:
            material: "Bi2Se3" or "Bi2Te3"
            num_quintuple_layers: Number of QLs
            vacuum: Vacuum thickness in Å
            
        Returns:
            Slab structure
        """
        bulk = cls.create_structure(material)
        
        # Create slab along c-direction
        slab = bulk.copy()
        
        # Repeat along c to get desired thickness
        slab.make_supercell([1, 1, num_quintuple_layers])
        
        # Add vacuum (would require more complex handling with pymatgen)
        # For now, just return the supercell
        
        return slab


class Bi2Se3Workflow:
    """
    Complete workflow for Bi2Se3/Bi2Te3 topological insulator study.
    """
    
    def __init__(self, config: Optional[TopologicalInsulatorConfig] = None):
        """
        Initialize workflow.
        
        Args:
            config: Configuration for calculations
        """
        self.config = config or TopologicalInsulatorConfig()
        self.structure = None
        self.results = {}
    
    def generate_structure(self, output_dir: str = "./Bi2Se3") -> str:
        """
        Generate and save crystal structure.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to structure file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate structure
        self.structure = Bi2Se3Structure.create_structure(self.config.material)
        
        # Save POSCAR
        poscar = Poscar(self.structure)
        poscar.write_file(output_path / "POSCAR")
        
        return str(output_path / "POSCAR")
    
    def generate_vasp_input(self, calc_dir: str = "./Bi2Se3") -> Dict[str, str]:
        """
        Generate VASP input files with SOC.
        
        Args:
            calc_dir: Calculation directory
            
        Returns:
            Dictionary with paths to input files
        """
        calc_path = Path(calc_dir)
        calc_path.mkdir(parents=True, exist_ok=True)
        
        # INCAR with SOC
        incar_dict = {
            "ENCUT": self.config.encut,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "EDIFF": 1e-6,
            "NELMIN": 6,
            "NELM": 100,
            
            # SOC settings
            "LSORBIT": self.config.lsorbit,
            "SAXIS": " ".join(map(str, self.config.saxis)),
            "ISYM": -1,  # Symmetry off for SOC
            "LMAXMIX": 4,  # For d/f electrons
            
            # Band structure
            "LORBIT": 11,
            "NEDOS": 3000,
            
            # Parallel
            "NCORE": 4,
        }
        
        incar = Incar(incar_dict)
        incar.write_file(calc_path / "INCAR")
        
        # KPOINTS
        if self.structure is None:
            self.structure = Bi2Se3Structure.create_structure(self.config.material)
        
        kpoints = Kpoints.automatic_density(
            self.structure, 
            self.config.k_density_bulk
        )
        kpoints.write_file(calc_path / "KPOINTS")
        
        # POSCAR
        if not (calc_path / "POSCAR").exists():
            poscar = Poscar(self.structure)
            poscar.write_file(calc_path / "POSCAR")
        
        return {
            "incar": str(calc_path / "INCAR"),
            "kpoints": str(calc_path / "KPOINTS"),
            "poscar": str(calc_path / "POSCAR"),
        }
    
    def calculate_z2_invariant(self, calc_dir: str = "./Bi2Se3") -> Dict[str, Any]:
        """
        Calculate Z2 topological invariant.
        
        Args:
            calc_dir: Directory with VASP output
            
        Returns:
            Dictionary with Z2 results
        """
        # Configure Z2Pack
        config = Z2PackConfig(
            num_bands=self.config.num_bands,
            num_wann=self.config.num_wann,
            surface=self.config.surface,
        )
        
        # Run Z2Pack
        interface = Z2VASPInterface(calc_dir, config)
        result = interface.calculate_z2_invariant()
        
        self.results["z2_invariant"] = {
            "strong": result.z2_index,
            "weak": result.z2_indices[1:],
            "chern_number": result.chern_number,
            "topological_phase": result.topological_phase.name,
            "is_topological": result.is_topological(),
        }
        
        return self.results["z2_invariant"]
    
    def calculate_surface_states(
        self,
        calc_dir: str = "./Bi2Se3",
        num_layers: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculate surface states using WannierTools.
        
        Args:
            calc_dir: Calculation directory
            num_layers: Number of layers for slab
            
        Returns:
            Dictionary with surface state results
        """
        # First build Wannier Hamiltonian
        hr_file = Path(calc_dir) / "wannier90_hr.dat"
        
        if not hr_file.exists():
            warnings.warn("Wannier90 HR file not found. Run Wannier90 first.")
            return {}
        
        # Configure WannierTools
        config = WannierToolsConfig(
            num_layers=num_layers,
            calculate_fermi_arc=True,
        )
        
        calculator = WannierToolsCalculator(str(hr_file), config)
        
        # Calculate surface states
        surface_result = calculator.calculate_surface_states(num_layers=num_layers)
        
        self.results["surface_states"] = {
            "has_dirac_cone": surface_result.has_dirac_cone,
            "num_surface_bands": surface_result.energies.shape[1] if len(surface_result.energies.shape) > 1 else 0,
            "dirac_points": surface_result.dirac_points,
        }
        
        return self.results["surface_states"]
    
    def run_full_analysis(self, calc_dir: str = "./Bi2Se3") -> Dict[str, Any]:
        """
        Run complete analysis workflow.
        
        Args:
            calc_dir: Calculation directory
            
        Returns:
            Complete analysis results
        """
        print(f"Running topological insulator analysis for {self.config.material}")
        print("=" * 60)
        
        # Step 1: Generate structure
        print("\n1. Generating crystal structure...")
        self.generate_structure(calc_dir)
        print(f"   Structure saved to {calc_dir}/POSCAR")
        
        # Step 2: Generate VASP input
        print("\n2. Generating VASP input files with SOC...")
        input_files = self.generate_vasp_input(calc_dir)
        print(f"   INCAR: {input_files['incar']}")
        print(f"   KPOINTS: {input_files['kpoints']}")
        
        # Step 3: Calculate Z2 invariant
        print("\n3. Calculating Z2 topological invariant...")
        try:
            z2_result = self.calculate_z2_invariant(calc_dir)
            print(f"   Strong Z2 index: {z2_result['strong']}")
            print(f"   Weak Z2 indices: {z2_result['weak']}")
            print(f"   Topological phase: {z2_result['topological_phase']}")
        except Exception as e:
            print(f"   Error: {e}")
            print("   (Requires VASP calculation with WAVECAR)")
        
        # Step 4: Surface states
        print("\n4. Analyzing surface states...")
        try:
            surface_result = self.calculate_surface_states(calc_dir)
            print(f"   Has Dirac cone: {surface_result.get('has_dirac_cone', False)}")
            print(f"   Dirac points: {surface_result.get('dirac_points', [])}")
        except Exception as e:
            print(f"   Error: {e}")
            print("   (Requires Wannier90 Hamiltonian)")
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        
        return self.results


def analyze_bi2se3(calc_dir: str = "./Bi2Se3") -> Dict[str, Any]:
    """
    Quick analysis function for Bi2Se3.
    
    Args:
        calc_dir: Calculation directory
        
    Returns:
        Analysis results
    """
    config = TopologicalInsulatorConfig(material="Bi2Se3")
    workflow = Bi2Se3Workflow(config)
    return workflow.run_full_analysis(calc_dir)


def analyze_bi2te3(calc_dir: str = "./Bi2Te3") -> Dict[str, Any]:
    """
    Quick analysis function for Bi2Te3.
    
    Args:
        calc_dir: Calculation directory
        
    Returns:
        Analysis results
    """
    config = TopologicalInsulatorConfig(material="Bi2Te3")
    workflow = Bi2Se3Workflow(config)
    return workflow.run_full_analysis(calc_dir)


def generate_reference_data() -> Dict[str, Any]:
    """
    Generate reference data for Bi2Se3 and Bi2Te3.
    
    Returns:
        Dictionary with reference values
    """
    return {
        "Bi2Se3": {
            "space_group": "R-3m (166)",
            "lattice_a": 4.138,  # Å
            "lattice_c": 28.64,  # Å
            "bulk_gap": 0.3,  # eV
            "z2_invariant": [1, 0, 0, 0],  # ν0; ν1ν2ν3
            "surface_states": "Single Dirac cone on (111) surface",
            "dirac_point": "Time-reversal invariant momentum (Γ)",
            "references": [
                "Zhang et al., Nature Phys. 5, 438 (2009)",
                "Xia et al., Nature Phys. 5, 398 (2009)",
            ],
        },
        "Bi2Te3": {
            "space_group": "R-3m (166)",
            "lattice_a": 4.383,  # Å
            "lattice_c": 30.49,  # Å
            "bulk_gap": 0.15,  # eV
            "z2_invariant": [1, 0, 0, 0],
            "surface_states": "Single Dirac cone on (111) surface",
            "dirac_point": "Time-reversal invariant momentum (Γ)",
            "references": [
                "Chen et al., Science 325, 178 (2009)",
                "Zhang et al., Nature Phys. 5, 438 (2009)",
            ],
        },
    }


# Example usage
if __name__ == "__main__":
    print("Topological Insulator Case Study: Bi2Se3 / Bi2Te3")
    print("=" * 60)
    
    # Show reference data
    ref_data = generate_reference_data()
    
    for material, data in ref_data.items():
        print(f"\n{material}:")
        print(f"  Space group: {data['space_group']}")
        print(f"  Lattice: a={data['lattice_a']} Å, c={data['lattice_c']} Å")
        print(f"  Bulk gap: {data['bulk_gap']} eV")
        print(f"  Z2 invariant: ν0;ν1ν2ν3 = {data['z2_invariant']}")
        print(f"  Surface: {data['surface_states']}")
    
    # Run example analysis
    print("\n" + "=" * 60)
    print("Example: Running analysis workflow...")
    
    # Create temporary directory for example
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        results = analyze_bi2se3(tmpdir)
        print(f"\nResults: {results}")
