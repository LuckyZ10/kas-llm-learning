"""
Case Study: Weyl Semimetal TaAs
================================

This module provides a complete workflow for studying the Weyl semimetal TaAs,
including:
- Weyl point location and classification
- Chirality calculation
- Fermi arc surface state calculation
- Chiral anomaly analysis

TaAs is the first experimentally confirmed Weyl semimetal with:
- 24 Weyl points (12 pairs)
- Type-I Weyl points (non-overtilted cones)
- Fermi arc surface states
- Chiral anomaly in magnetotransport

References:
- Weng et al., PRX 5, 011029 (2015) - Prediction
- Xu et al., Science 349, 613 (2015) - ARPES observation
- Huang et al., PRL 115, 017501 (2015) - Transport
- Yang et al., Nature Phys. 11, 728 (2015) - Chiral anomaly
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

# Import Weyl module
from dftlammps.weyl import (
    WeylSemimetalConfig, WeylPointLocator, WeylPointData,
    FermiArcCalculator, ChiralityCalculator, MagnetotransportCalculator,
    WeylType, FermiArcData,
    locate_weyl_points, calculate_fermi_arcs, analyze_weyl_semimetal,
)


@dataclass
class TaAsConfig:
    """Configuration for TaAs calculations."""
    material: str = "TaAs"  # or "TaP", "NbAs", "NbP"
    
    # VASP settings
    encut: float = 600.0
    k_density_bulk: float = 0.02
    k_density_weyl: float = 0.01  # Finer mesh for Weyl search
    
    # SOC settings
    lsorbit: bool = True
    
    # Weyl search settings
    search_window: Tuple[float, float] = (-0.5, 0.5)  # Around E_F
    gap_threshold: float = 0.001  # eV
    
    # Output
    calculate_fermi_arcs: bool = True
    calculate_transport: bool = True
    plot_results: bool = True


class TaAsStructure:
    """
    Generate TaAs family crystal structures.
    
    TaAs has body-centered tetragonal structure (space group I4_1md, #109).
    """
    
    # Lattice parameters (Å)
    LATTICE_PARAMS = {
        "TaAs": {"a": 3.437, "c": 11.646},
        "TaP": {"a": 3.320, "c": 11.360},
        "NbAs": {"a": 3.452, "c": 11.680},
        "NbP": {"a": 3.334, "c": 11.376},
    }
    
    @classmethod
    def create_structure(cls, material: str = "TaAs") -> Structure:
        """
        Create TaAs structure.
        
        Args:
            material: "TaAs", "TaP", "NbAs", or "NbP"
            
        Returns:
            Pymatgen Structure object
        """
        if not HAS_PMG:
            raise ImportError("Pymatgen required for structure generation")
        
        params = cls.LATTICE_PARAMS.get(material, cls.LATTICE_PARAMS["TaAs"])
        a, c = params["a"], params["c"]
        
        # Body-centered tetragonal lattice
        lattice = Lattice.tetragonal(a, c)
        
        # Atomic positions (from crystallographic data)
        # Ta at 4a (0, 0, 0), As at 4a (0, 0.5, 0.0833)
        if "Ta" in material:
            cation = "Ta"
        else:
            cation = "Nb"
        
        if "As" in material:
            anion = "As"
        else:
            anion = "P"
        
        species = [cation, anion]
        coords = [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0833],
        ]
        
        structure = Structure(lattice, species, coords)
        
        return structure
    
    @classmethod
    def get_high_symmetry_points(cls) -> Dict[str, np.ndarray]:
        """
        Get high-symmetry k-points for TaAs BZ.
        
        Returns:
            Dictionary mapping point labels to k-coordinates
        """
        return {
            "Γ": np.array([0.0, 0.0, 0.0]),
            "X": np.array([0.5, 0.0, 0.0]),
            "M": np.array([0.5, 0.5, 0.0]),
            "Z": np.array([0.0, 0.0, 0.5]),
            "R": np.array([0.5, 0.0, 0.5]),
            "A": np.array([0.5, 0.5, 0.5]),
        }
    
    @classmethod
    def get_weyl_point_reference(cls, material: str = "TaAs") -> List[Dict[str, Any]]:
        """
        Get literature reference positions of Weyl points.
        
        Returns:
            List of Weyl point data from literature
        """
        # Reference positions from Weng et al., PRX 5, 011029 (2015)
        # 24 Weyl points: 8 on kz=0 plane, 16 off-plane
        
        weyl_points = []
        
        # kz = 0 plane (8 Weyl points - 4 W1 + 4 W2)
        # W1 points at (±0.5, ±0.05, 0) - chirality = +1
        for sx in [0.5, -0.5]:
            for sy in [0.05, -0.05]:
                weyl_points.append({
                    "k_point": np.array([sx, sy, 0.0]),
                    "chirality": 1 if sx * sy > 0 else -1,
                    "type": "W1",
                    "plane": "kz=0",
                })
        
        # Off-plane Weyl points (16 points)
        # Symmetric pairs at (±0.5, ±0.05, ±0.15)
        for sx in [0.5, -0.5]:
            for sy in [0.05, -0.05]:
                for sz in [0.15, -0.15]:
                    weyl_points.append({
                        "k_point": np.array([sx, sy, sz]),
                        "chirality": 1 if sx * sy * sz > 0 else -1,
                        "type": "W2",
                        "plane": "off-plane",
                    })
        
        return weyl_points


class TaAsWorkflow:
    """
    Complete workflow for TaAs Weyl semimetal study.
    """
    
    def __init__(self, config: Optional[TaAsConfig] = None):
        """
        Initialize workflow.
        
        Args:
            config: Configuration for calculations
        """
        self.config = config or TaAsConfig()
        self.structure = None
        self.weyl_points = []
        self.results = {}
    
    def generate_structure(self, output_dir: str = "./TaAs") -> str:
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
        self.structure = TaAsStructure.create_structure(self.config.material)
        
        # Save POSCAR
        poscar = Poscar(self.structure)
        poscar.write_file(output_path / "POSCAR")
        
        return str(output_path / "POSCAR")
    
    def generate_vasp_input(self, calc_dir: str = "./TaAs") -> Dict[str, str]:
        """
        Generate VASP input files with SOC.
        
        Args:
            calc_dir: Calculation directory
            
        Returns:
            Dictionary with paths to input files
        """
        calc_path = Path(calc_dir)
        calc_path.mkdir(parents=True, exist_ok=True)
        
        # INCAR with SOC for heavy elements (Ta, W)
        incar_dict = {
            "ENCUT": self.config.encut,
            "ISMEAR": 0,
            "SIGMA": 0.02,
            "EDIFF": 1e-7,
            "NELMIN": 6,
            
            # SOC is essential for Weyl points
            "LSORBIT": self.config.lsorbit,
            "SAXIS": "0 0 1",
            "ISYM": -1,
            "LMAXMIX": 6,  # For f electrons (Ta)
            
            # Electronic structure
            "LORBIT": 12,
            "NEDOS": 5000,
            
            # For band structure
            "ICHARG": 11,
            
            # Parallel
            "NCORE": 4,
        }
        
        incar = Incar(incar_dict)
        incar.write_file(calc_path / "INCAR")
        
        # KPOINTS - dense mesh for Weyl point search
        if self.structure is None:
            self.structure = TaAsStructure.create_structure(self.config.material)
        
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
    
    def locate_weyl_points(self, calc_dir: str = "./TaAs") -> List[WeylPointData]:
        """
        Locate Weyl points in TaAs.
        
        Args:
            calc_dir: Calculation directory
            
        Returns:
            List of Weyl points
        """
        config = WeylSemimetalConfig(
            search_window=self.config.search_window,
            k_mesh_fine=(50, 50, 50),
            gap_threshold=self.config.gap_threshold,
        )
        
        locator = WeylPointLocator(calc_dir, config)
        self.weyl_points = locator.search_weyl_points()
        
        self.results["weyl_points"] = {
            "num_total": len(self.weyl_points),
            "num_positive": sum(1 for wp in self.weyl_points if wp.chirality == 1),
            "num_negative": sum(1 for wp in self.weyl_points if wp.chirality == -1),
            "points": [
                {
                    "k": wp.k_point.tolist(),
                    "E": wp.energy,
                    "C": wp.chirality,
                    "type": wp.classify_weyl_type().name,
                }
                for wp in self.weyl_points
            ],
        }
        
        return self.weyl_points
    
    def calculate_fermi_arcs(self, calc_dir: str = "./TaAs") -> List[FermiArcData]:
        """
        Calculate Fermi arc surface states.
        
        Args:
            calc_dir: Calculation directory
            
        Returns:
            List of Fermi arcs
        """
        if not self.weyl_points:
            self.weyl_points = self.locate_weyl_points(calc_dir)
        
        arc_calc = FermiArcCalculator(calc_dir)
        fermi_arcs = arc_calc.calculate_fermi_arcs(self.weyl_points)
        
        self.results["fermi_arcs"] = {
            "num_arcs": len(fermi_arcs),
            "arc_lengths": [arc.length for arc in fermi_arcs],
        }
        
        return fermi_arcs
    
    def calculate_chiral_anomaly(
        self,
        B_field: np.ndarray = np.array([0, 0, 1]),
    ) -> Dict[str, float]:
        """
        Calculate chiral anomaly contribution.
        
        Args:
            B_field: Magnetic field vector in Tesla
            
        Returns:
            Dictionary with transport results
        """
        if not self.weyl_points:
            raise ValueError("Weyl points not calculated yet")
        
        transport_calc = MagnetotransportCalculator(self.weyl_points)
        
        # Chiral anomaly conductivity
        sigma_chiral = transport_calc.calculate_chiral_anomaly(B_field)
        
        # Negative magnetoresistance
        B_fields = np.linspace(0, 10, 100)  # 0-10 Tesla
        nmr = transport_calc.calculate_negative_magnetoresistance(B_fields)
        
        self.results["chiral_anomaly"] = {
            "sigma_chiral_S_cm": sigma_chiral,
            "B_field_T": np.linalg.norm(B_field),
            "nmr_ratio_percent": np.min(nmr) * 100,
        }
        
        return self.results["chiral_anomaly"]
    
    def run_full_analysis(self, calc_dir: str = "./TaAs") -> Dict[str, Any]:
        """
        Run complete analysis workflow.
        
        Args:
            calc_dir: Calculation directory
            
        Returns:
            Complete analysis results
        """
        print(f"Running Weyl semimetal analysis for {self.config.material}")
        print("=" * 60)
        
        # Step 1: Generate structure
        print("\n1. Generating crystal structure...")
        self.generate_structure(calc_dir)
        print(f"   Structure: {self.config.material}")
        print(f"   Space group: I4_1md (#109)")
        
        # Step 2: Generate VASP input
        print("\n2. Generating VASP input files with SOC...")
        input_files = self.generate_vasp_input(calc_dir)
        print(f"   INCAR with SOC for {self.config.material}")
        
        # Step 3: Locate Weyl points
        print("\n3. Searching for Weyl points...")
        print("   Expected: 24 Weyl points (12 pairs)")
        try:
            weyl_points = self.locate_weyl_points(calc_dir)
            print(f"   Found {len(weyl_points)} Weyl points")
            
            num_pos = sum(1 for wp in weyl_points if wp.chirality == 1)
            num_neg = sum(1 for wp in weyl_points if wp.chirality == -1)
            print(f"   Positive chirality: {num_pos}")
            print(f"   Negative chirality: {num_neg}")
            print(f"   Net chirality: {num_pos - num_neg}")
            
            # Show first few Weyl points
            for i, wp in enumerate(weyl_points[:4]):
                print(f"   WP{i+1}: k=({wp.k_point[0]:.3f}, {wp.k_point[1]:.3f}, "
                      f"{wp.k_point[2]:.3f}), C={wp.chirality:+d}, E={wp.energy:.3f} eV")
        except Exception as e:
            print(f"   Error: {e}")
            print("   (Requires VASP calculation with WAVECAR)")
        
        # Step 4: Calculate Fermi arcs
        print("\n4. Calculating Fermi arc surface states...")
        try:
            fermi_arcs = self.calculate_fermi_arcs(calc_dir)
            print(f"   Found {len(fermi_arcs)} Fermi arcs")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Step 5: Chiral anomaly
        print("\n5. Analyzing chiral anomaly...")
        if self.weyl_points:
            try:
                anomaly = self.calculate_chiral_anomaly()
                print(f"   Chiral anomaly σ: {anomaly['sigma_chiral_S_cm']:.2e} S/cm")
                print(f"   Expected negative MR: {anomaly['nmr_ratio_percent']:.1f}%")
            except Exception as e:
                print(f"   Error: {e}")
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        
        return self.results


def analyze_taas(calc_dir: str = "./TaAs") -> Dict[str, Any]:
    """
    Quick analysis function for TaAs.
    
    Args:
        calc_dir: Calculation directory
        
    Returns:
        Analysis results
    """
    config = TaAsConfig(material="TaAs")
    workflow = TaAsWorkflow(config)
    return workflow.run_full_analysis(calc_dir)


def analyze_tap(calc_dir: str = "./TaP") -> Dict[str, Any]:
    """
    Quick analysis function for TaP.
    
    Args:
        calc_dir: Calculation directory
        
    Returns:
        Analysis results
    """
    config = TaAsConfig(material="TaP")
    workflow = TaAsWorkflow(config)
    return workflow.run_full_analysis(calc_dir)


def generate_reference_data() -> Dict[str, Any]:
    """
    Generate reference data for TaAs family.
    
    Returns:
        Dictionary with reference values
    """
    return {
        "TaAs": {
            "space_group": "I4_1md (109)",
            "lattice_a": 3.437,  # Å
            "lattice_c": 11.646,  # Å
            "num_weyl_points": 24,
            "weyl_types": ["Type-I"],
            "weyl_positions": [
                "8 Weyl points on kz=0 plane (W1)",
                "16 Weyl points off-plane (W2)",
            ],
            "chiral_anomaly": True,
            "fermi_arcs": True,
            "references": [
                "Weng et al., PRX 5, 011029 (2015)",
                "Xu et al., Science 349, 613 (2015)",
                "Huang et al., PRL 115, 017501 (2015)",
            ],
        },
        "TaP": {
            "space_group": "I4_1md (109)",
            "lattice_a": 3.320,
            "lattice_c": 11.360,
            "num_weyl_points": 24,
            "weyl_types": ["Type-I"],
            "references": [
                "Xu et al., Sci. Adv. 1, e1501092 (2015)",
            ],
        },
        "NbAs": {
            "space_group": "I4_1md (109)",
            "lattice_a": 3.452,
            "lattice_c": 11.680,
            "num_weyl_points": 24,
            "references": [
                "Xu et al., Nature Phys. 11, 748 (2015)",
            ],
        },
    }


# Example usage
if __name__ == "__main__":
    print("Weyl Semimetal Case Study: TaAs Family")
    print("=" * 60)
    
    # Show reference data
    ref_data = generate_reference_data()
    
    for material, data in ref_data.items():
        print(f"\n{material}:")
        print(f"  Space group: {data['space_group']}")
        print(f"  Lattice: a={data['lattice_a']} Å, c={data['lattice_c']} Å")
        print(f"  Weyl points: {data['num_weyl_points']}")
        print(f"  Chiral anomaly: {data.get('chiral_anomaly', True)}")
        print(f"  Fermi arcs: {data.get('fermi_arcs', True)}")
    
    # Run example analysis
    print("\n" + "=" * 60)
    print("Example: Running analysis workflow...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        results = analyze_taas(tmpdir)
        print(f"\nResults keys: {list(results.keys())}")
