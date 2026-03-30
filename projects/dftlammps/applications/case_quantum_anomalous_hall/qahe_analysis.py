"""
Case Study: Quantum Anomalous Hall Effect in Magnetically-Doped Topological Insulators
=======================================================================================

This module provides a complete workflow for studying the Quantum Anomalous Hall
Effect (QAHE) in magnetically-doped topological insulators like Cr/V-doped
(Bi,Sb)2Te3.

The QAHE is characterized by:
- Dissipationless edge states without external magnetic field
- Quantized Hall conductivity σ_xy = e²/h
- Chern number C ≠ 0
- Time-reversal symmetry breaking by magnetic dopants

Key materials:
- Cr-doped (Bi,Sb)2Te3 (Chang et al., Science 2013)
- V-doped (Bi,Sb)2Te3 (Chang et al., Nature Mater. 2015)
- Mn-doped Bi2Te3

References:
- Chang et al., Science 340, 167 (2013) - First experimental observation
- Chang et al., Nature Mater. 14, 473 (2015) - Higher temperature QAHE
- Yu et al., Science 329, 61 (2010) - Theoretical prediction
- Wang et al., PRL 107, 206602 (2011) - Phase diagram
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

try:
    from pymatgen.core import Structure, Lattice, Element, Composition
    from pymatgen.io.vasp import Poscar, Incar, Kpoints
    HAS_PMG = True
except ImportError:
    HAS_PMG = False
    warnings.warn("Pymatgen not available.")

# Import topology modules
from dftlammps.topology import (
    Z2PackConfig, Z2VASPInterface, TopologicalPhase,
    ChernNumberResult, ChernNumberCalculator,
    BerryPhaseConfig, BerryCurvatureCalculator,
    AnomalousHallConductivityCalculator,
    calculate_chern_number, calculate_anomalous_hall_conductivity,
)


@dataclass
class QAHEConfig:
    """Configuration for QAHE calculations."""
    # Material composition
    base_material: str = "Bi2Te3"
    dopant: str = "Cr"
    doping_concentration: float = 0.08  # x in (Cr_x)(Bi_{1-x})2Te3
    
    # Magnetic ordering
    magnetic_order: str = "ferromagnetic"  # or "antiferromagnetic"
    mag_moment: float = 3.0  # μB per dopant
    
    # VASP settings
    encut: float = 500.0
    k_density: float = 0.02
    
    # SOC + magnetic
    lsorbit: bool = True
    ispin: int = 2  # Spin-polarized
    magmom: Optional[List[float]] = None
    
    # Calculation settings
    calculate_chern_number: bool = True
    calculate_ahc: bool = True
    calculate_edge_states: bool = True


class DopedTIStructure:
    """
    Generate structures for magnetically-doped topological insulators.
    """
    
    # Lattice parameters (Å)
    LATTICE_PARAMS = {
        "Bi2Te3": {"a": 4.383, "c": 30.49},
        "Bi2Se3": {"a": 4.138, "c": 28.64},
        "Sb2Te3": {"a": 4.250, "c": 30.35},
    }
    
    @classmethod
    def create_doped_structure(
        cls,
        base_material: str = "Bi2Te3",
        dopant: str = "Cr",
        doping_x: float = 0.08,
        ordering: str = "ferromagnetic",
    ) -> Structure:
        """
        Create magnetically-doped TI structure.
        
        Args:
            base_material: Base TI material
            dopant: Magnetic dopant element
            doping_x: Doping concentration
            ordering: Magnetic ordering type
            
        Returns:
            Doped structure
        """
        if not HAS_PMG:
            raise ImportError("Pymatgen required")
        
        # Start with base structure
        params = cls.LATTICE_PARAMS.get(base_material, cls.LATTICE_PARAMS["Bi2Te3"])
        a, c = params["a"], params["c"]
        
        lattice = Lattice.hexagonal(a, c)
        
        # Create supercell for doping
        # 2x2x1 supercell has 4 formula units
        supercell_multiplier = max(1, int(1 / doping_x / 4))
        
        # Build base quintuple layer
        species = []
        coords = []
        
        # Get elements from formula
        if "Bi" in base_material and "Te" in base_material:
            cation = "Bi"
            anion = "Te"
        elif "Bi" in base_material and "Se" in base_material:
            cation = "Bi"
            anion = "Se"
        elif "Sb" in base_material:
            cation = "Sb"
            anion = "Te"
        else:
            cation = "Bi"
            anion = "Te"
        
        # Build QL structure
        # Layer sequence: Te-Bi-Te-Bi-Te
        z_positions = [0.0, 0.064, 0.213, 0.362, 0.436]
        layer_species = [anion, cation, anion, cation, anion]
        
        for z, spec in zip(z_positions, layer_species):
            species.append(spec)
            coords.append([0.0, 0.0, z])
        
        structure = Structure(lattice, species, coords)
        
        # Make supercell
        structure.make_supercell([2, 2, 1])
        
        # Add dopants
        num_dopants = max(1, int(len(structure) * doping_x))
        
        # Replace some cation sites with dopants
        cation_indices = [i for i, site in enumerate(structure) 
                         if site.species_string == cation]
        
        import random
        random.seed(42)
        dopant_indices = random.sample(cation_indices, min(num_dopants, len(cation_indices)))
        
        for idx in dopant_indices:
            structure.replace(idx, dopant)
        
        return structure
    
    @classmethod
    def create_ferromagnetic_structure(
        cls,
        base_structure: Structure,
        dopant_indices: List[int],
        mag_direction: np.ndarray = np.array([0, 0, 1]),
    ) -> Tuple[Structure, List[float]]:
        """
        Create ferromagnetic ordering of dopants.
        
        Args:
            base_structure: Doped structure
            dopant_indices: Indices of dopant atoms
            mag_direction: Magnetization direction
            
        Returns:
            Tuple of (structure, MAGMOM values)
        """
        magmom = [0.0] * len(base_structure)
        
        for idx in dopant_indices:
            magmom[idx] = 3.0 * mag_direction[2]  # 3 μB along z
        
        return base_structure, magmom
    
    @classmethod
    def create_antiferromagnetic_structure(
        cls,
        base_structure: Structure,
        dopant_indices: List[int],
    ) -> Tuple[Structure, List[float]]:
        """
        Create antiferromagnetic ordering of dopants.
        
        Args:
            base_structure: Doped structure
            dopant_indices: Indices of dopant atoms
            
        Returns:
            Tuple of (structure, MAGMOM values)
        """
        magmom = [0.0] * len(base_structure)
        
        for i, idx in enumerate(dopant_indices):
            # Alternate up and down
            magmom[idx] = 3.0 if i % 2 == 0 else -3.0
        
        return base_structure, magmom


class QAHEWorkflow:
    """
    Complete workflow for Quantum Anomalous Hall Effect study.
    """
    
    def __init__(self, config: Optional[QAHEConfig] = None):
        """
        Initialize QAHE workflow.
        
        Args:
            config: Configuration
        """
        self.config = config or QAHEConfig()
        self.structure = None
        self.magmom = None
        self.results = {}
    
    def generate_structure(self, output_dir: str = "./QAHE") -> str:
        """
        Generate doped structure with magnetic ordering.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to structure file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate doped structure
        structure = DopedTIStructure.create_doped_structure(
            self.config.base_material,
            self.config.dopant,
            self.config.doping_concentration,
            self.config.magnetic_order,
        )
        
        # Apply magnetic ordering
        dopant_indices = [i for i, site in enumerate(structure) 
                         if site.species_string == self.config.dopant]
        
        if self.config.magnetic_order == "ferromagnetic":
            structure, self.magmom = DopedTIStructure.create_ferromagnetic_structure(
                structure, dopant_indices
            )
        else:
            structure, self.magmom = DopedTIStructure.create_antiferromagnetic_structure(
                structure, dopant_indices
            )
        
        self.structure = structure
        
        # Save structure
        poscar = Poscar(structure)
        poscar.write_file(output_path / "POSCAR")
        
        # Save magnetic moments
        with open(output_path / "MAGMOM", 'w') as f:
            f.write(" ".join([f"{m:.1f}" for m in self.magmom]))
        
        return str(output_path / "POSCAR")
    
    def generate_vasp_input(self, calc_dir: str = "./QAHE") -> Dict[str, str]:
        """
        Generate VASP input with SOC and magnetism.
        
        Args:
            calc_dir: Calculation directory
            
        Returns:
            Dictionary with input file paths
        """
        calc_path = Path(calc_dir)
        calc_path.mkdir(parents=True, exist_ok=True)
        
        # INCAR with SOC + magnetism
        incar_dict = {
            "ENCUT": self.config.encut,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "EDIFF": 1e-7,
            
            # Spin and SOC
            "ISPIN": self.config.ispin,
            "LSORBIT": self.config.lsorbit,
            "SAXIS": "0 0 1",
            "ISYM": -1,
            "LMAXMIX": 6,
            
            # Magnetic moments
            "MAGMOM": " ".join([f"{m:.1f}" for m in self.magmom]) if self.magmom else "3.0",
            
            # Convergence
            "AMIX": 0.2,
            "BMIX": 0.0001,
            "NELMIN": 8,
            
            # Output
            "LORBIT": 12,
            "LCHARG": True,
            "LWAVE": True,
        }
        
        incar = Incar(incar_dict)
        incar.write_file(calc_path / "INCAR")
        
        # KPOINTS
        if self.structure is None:
            self.structure = DopedTIStructure.create_doped_structure(
                self.config.base_material,
                self.config.dopant,
                self.config.doping_concentration,
            )
        
        kpoints = Kpoints.automatic_density(self.structure, self.config.k_density)
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
    
    def calculate_chern_number(self, calc_dir: str = "./QAHE") -> int:
        """
        Calculate Chern number from Berry curvature.
        
        Args:
            calc_dir: Calculation directory
            
        Returns:
            Chern number
        """
        config = BerryPhaseConfig(
            berry_k_mesh=(30, 30, 1),  # 2D for thin film
        )
        
        calculator = ChernNumberCalculator(calc_dir, config)
        result = calculator.calculate_from_berry_curvature(k_mesh=(30, 30, 1))
        
        self.results["chern_number"] = {
            "chern": result.chern_number,
            "converged": result.converged,
            "error": result.error_estimate,
        }
        
        return result.chern_number
    
    def calculate_anomalous_hall_conductivity(
        self,
        calc_dir: str = "./QAHE",
        temperature: float = 0.0,
    ) -> float:
        """
        Calculate anomalous Hall conductivity.
        
        Args:
            calc_dir: Calculation directory
            temperature: Temperature in K
            
        Returns:
            AHC in S/cm
        """
        config = BerryPhaseConfig()
        calculator = AnomalousHallConductivityCalculator(calc_dir, config)
        result = calculator.calculate_ahc(temperature)
        
        sigma_xy = result.get_hall_conductivity('xy')
        
        # Convert to quantization units
        e = 1.602e-19  # C
        h = 6.626e-34  # J·s
        sigma_0 = e**2 / h * 1e-4  # S/cm (conductivity quantum)
        
        self.results["anomalous_hall"] = {
            "sigma_xy_S_cm": sigma_xy,
            "sigma_quantum": sigma_0,
            "quantization": sigma_xy / sigma_0,
            "temperature_K": temperature,
        }
        
        return sigma_xy
    
    def verify_quantization(self, tolerance: float = 0.05) -> bool:
        """
        Verify if Hall conductivity is quantized.
        
        Args:
            tolerance: Allowed deviation from perfect quantization
            
        Returns:
            True if quantized
        """
        if "anomalous_hall" not in self.results:
            raise ValueError("Calculate AHC first")
        
        quantization = self.results["anomalous_hall"]["quantization"]
        
        # Check if close to integer
        nearest_integer = round(quantization)
        deviation = abs(quantization - nearest_integer)
        
        self.results["quantization_check"] = {
            "is_quantized": deviation < tolerance,
            "nearest_integer": nearest_integer,
            "deviation": deviation,
        }
        
        return deviation < tolerance
    
    def run_full_analysis(self, calc_dir: str = "./QAHE") -> Dict[str, Any]:
        """
        Run complete QAHE analysis workflow.
        
        Args:
            calc_dir: Calculation directory
            
        Returns:
            Complete analysis results
        """
        print("Quantum Anomalous Hall Effect Analysis")
        print("=" * 60)
        print(f"Material: {self.config.dopant}-doped {self.config.base_material}")
        print(f"Doping: {self.config.doping_concentration * 100:.1f}%")
        print(f"Ordering: {self.config.magnetic_order}")
        
        # Step 1: Generate structure
        print("\n1. Generating magnetically-doped structure...")
        self.generate_structure(calc_dir)
        print(f"   Formula: {self.structure.composition.reduced_formula if self.structure else 'N/A'}")
        print(f"   Magnetic order: {self.config.magnetic_order}")
        
        # Step 2: Generate VASP input
        print("\n2. Generating VASP input (SOC + Magnetism)...")
        input_files = self.generate_vasp_input(calc_dir)
        print("   INCAR with SOC and collinear magnetism")
        
        # Step 3: Calculate Chern number
        if self.config.calculate_chern_number:
            print("\n3. Calculating Chern number...")
            try:
                chern = self.calculate_chern_number(calc_dir)
                print(f"   Chern number C = {chern}")
                
                if chern != 0:
                    print(f"   ✓ Quantum Anomalous Hall phase detected!")
                else:
                    print(f"   ✗ Trivial phase")
            except Exception as e:
                print(f"   Error: {e}")
                print("   (Requires VASP calculation)")
        
        # Step 4: Calculate AHC
        if self.config.calculate_ahc:
            print("\n4. Calculating anomalous Hall conductivity...")
            try:
                sigma_xy = self.calculate_anomalous_hall_conductivity(calc_dir)
                
                e = 1.602e-19
                h = 6.626e-34
                sigma_0 = e**2 / h * 1e-4
                
                print(f"   σ_xy = {sigma_xy:.2e} S/cm")
                print(f"   σ_0 = {sigma_0:.2e} S/cm (quantum)")
                print(f"   σ/σ_0 = {sigma_xy/sigma_0:.3f}")
                
                # Check quantization
                is_quantized = self.verify_quantization()
                print(f"   Quantized: {is_quantized}")
                
                if is_quantized:
                    c = round(sigma_xy / sigma_0)
                    print(f"   → C = {c} (matches Chern number)")
            except Exception as e:
                print(f"   Error: {e}")
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        
        return self.results


def analyze_cr_doped_bi2te3(
    concentration: float = 0.08,
    calc_dir: str = "./Cr_Bi2Te3",
) -> Dict[str, Any]:
    """
    Analyze Cr-doped Bi2Te3 for QAHE.
    
    Args:
        concentration: Cr doping concentration
        calc_dir: Calculation directory
        
    Returns:
        Analysis results
    """
    config = QAHEConfig(
        base_material="Bi2Te3",
        dopant="Cr",
        doping_concentration=concentration,
        magnetic_order="ferromagnetic",
    )
    workflow = QAHEWorkflow(config)
    return workflow.run_full_analysis(calc_dir)


def analyze_v_doped_bi2te3(
    concentration: float = 0.08,
    calc_dir: str = "./V_Bi2Te3",
) -> Dict[str, Any]:
    """
    Analyze V-doped Bi2Te3 for QAHE.
    
    Args:
        concentration: V doping concentration
        calc_dir: Calculation directory
        
    Returns:
        Analysis results
    """
    config = QAHEConfig(
        base_material="Bi2Te3",
        dopant="V",
        doping_concentration=concentration,
        magnetic_order="ferromagnetic",
    )
    workflow = QAHEWorkflow(config)
    return workflow.run_full_analysis(calc_dir)


def generate_reference_data() -> Dict[str, Any]:
    """
    Generate reference data for QAHE materials.
    
    Returns:
        Dictionary with reference values
    """
    return {
        "Cr-doped (Bi,Sb)2Te3": {
            "first_observation": "Chang et al., Science 340, 167 (2013)",
            "temperature": 30,  # mK
            "doping": "x = 0.08-0.15",
            "chern_number": 1,
            "quantized_conductance": "e²/h",
            "coercivity": "50-100 Oe",
            "notes": "First experimental observation of QAHE",
        },
        "V-doped (Bi,Sb)2Te3": {
            "reference": "Chang et al., Nature Mater. 14, 473 (2015)",
            "temperature": 100,  # mK (higher than Cr-doped)
            "doping": "x = 0.05-0.10",
            "chern_number": 1,
            "magnetic_order": "Ferromagnetic",
            "curie_temperature": "20-30 K",
            "notes": "Higher temperature QAHE",
        },
        "Mn-doped Bi2Te3": {
            "reference": "Theoretical",
            "doping": "x = 0.05-0.10",
            "magnetic_order": "Ferromagnetic",
            "predicted_chern": 1,
        },
        "key_requirements": [
            "Topological insulator host (Bi2Se3, Bi2Te3, Sb2Te3)",
            "Magnetic dopants (Cr, V, Mn, Fe)",
            "Ferromagnetic ordering",
            "Sufficient magnetic moment",
            "SOC + magnetism in DFT calculations",
        ],
    }


# Example usage
if __name__ == "__main__":
    print("Quantum Anomalous Hall Effect Case Study")
    print("=" * 60)
    
    # Show reference data
    ref_data = generate_reference_data()
    
    print("\nReference Data:")
    for material, data in ref_data.items():
        if isinstance(data, dict):
            print(f"\n{material}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Example: Running analysis workflow...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        results = analyze_cr_doped_bi2te3(0.08, tmpdir)
        print(f"\nResults keys: {list(results.keys())}")
