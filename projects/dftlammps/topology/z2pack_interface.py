"""
Z2Pack Interface for Topological Invariant Calculations
=========================================================

This module provides interfaces to Z2Pack for calculating topological invariants
including Z2 indices, Wilson loops, and Chern numbers from VASP wavefunctions.

Features:
- VASP wavefunction extraction and processing
- Wilson loop calculations
- Z2 invariant determination (with/without time-reversal symmetry)
- Chern number calculations
- Automated topological classification

References:
- Z2Pack: https://z2pack.greschd.ch
- Soluyanov et al., PRB 83, 235401 (2011)
- Yu et al., PRB 84, 075119 (2011)
"""

import os
import re
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

try:
    import z2pack
    HAS_Z2PACK = True
except ImportError:
    HAS_Z2PACK = False
    warnings.warn("Z2Pack not available. Install with: pip install z2pack")

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar, Kpoints, Incar
    from pymatgen.io.vasp.outputs import Wavecar, Waveder
    HAS_PMG = True
except ImportError:
    HAS_PMG = False
    warnings.warn("Pymatgen not available.")

try:
    import ase
    from ase.io import read, write
    from ase.dft.kpoints import get_special_points, bandpath
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    warnings.warn("ASE not available.")


class TopologicalPhase(Enum):
    """Enumeration of topological phases."""
    TRIVIAL = auto()
    Z2_TI = auto()  # Z2 topological insulator
    CHERN_INSULATOR = auto()
    WEYL_SEMIMETAL = auto()
    DIRAC_SEMIMETAL = auto()
    NODAL_LINE = auto()
    QUANTUM_ANOMALOUS_HALL = auto()
    QUANTUM_SPIN_HALL = auto()
    UNKNOWN = auto()


class SymmetryType(Enum):
    """Symmetry types for Z2 calculation."""
    TIME_REVERSAL = "time_reversal"
    INVERSION = "inversion"
    NONSYMMORPHIC = "nonsymmorphic"
    MAGNETIC = "magnetic"


@dataclass
class Z2PackConfig:
    """Configuration for Z2Pack calculations."""
    # VASP settings
    vasp_executable: str = "vasp_std"
    num_bands: int = 20
    num_wann: Optional[int] = None
    
    # Surface settings
    surface: str = "ky-surface"  # or "kx-surface", "kz-surface"
    num_lines: int = 11
    pos_tol: float = 0.01
    gap_tol: float = 0.1
    move_tol: float = 0.3
    iterator: range = field(default_factory=lambda: range(8, 27, 2))
    
    # Convergence settings
    min_neighbour_dist: float = 1e-5
    
    # Time-reversal settings
    time_reversal_symmetric: bool = True
    symprec: float = 1e-5
    
    # Output settings
    verbose: bool = True
    save_plots: bool = True
    plot_dir: str = "./z2pack_plots"


@dataclass
class WilsonLoopResult:
    """Result of Wilson loop calculation."""
    k_points: np.ndarray  # Shape: (num_k,)
    phases: np.ndarray    # Shape: (num_k, num_bands)
    gaps: np.ndarray      # Shape: (num_k,)
    converged: bool
    z2_index: Optional[int] = None
    chern_number: Optional[int] = None
    
    def get_wcc_centers(self) -> np.ndarray:
        """Get Wannier charge centers (WCC)."""
        return self.phases / (2 * np.pi)


@dataclass
class Z2InvariantResult:
    """Result of Z2 invariant calculation."""
    z2_index: int  # 0 or 1
    z2_indices: List[int]  # For 3D: [ν0; ν1ν2ν3]
    chern_number: int
    gap_at_time_reversal: float
    gap_minimum: float
    converged: bool
    topological_phase: TopologicalPhase
    surface: str
    
    def is_topological(self) -> bool:
        """Check if the system is topological."""
        return self.z2_index == 1 or self.chern_number != 0


@dataclass
class ChernNumberResult:
    """Result of Chern number calculation."""
    chern_number: int
    berry_curvature: np.ndarray
    k_mesh: np.ndarray
    converged: bool
    error_estimate: float


class VASPWavefunctionExtractor:
    """
    Extract and process VASP wavefunctions for topological calculations.
    """
    
    def __init__(self, calc_dir: str = ".", config: Optional[Z2PackConfig] = None):
        """
        Initialize wavefunction extractor.
        
        Args:
            calc_dir: VASP calculation directory
            config: Z2Pack configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or Z2PackConfig()
        self.structure = None
        self.kpoints = None
        
        if not HAS_PMG:
            raise ImportError("Pymatgen required for VASP wavefunction extraction")
    
    def read_structure(self) -> Structure:
        """Read structure from POSCAR/CONTCAR."""
        poscar_path = self.calc_dir / "POSCAR"
        if not poscar_path.exists():
            poscar_path = self.calc_dir / "CONTCAR"
        
        if not poscar_path.exists():
            raise FileNotFoundError(f"No POSCAR/CONTCAR found in {self.calc_dir}")
        
        self.structure = Poscar.from_file(poscar_path).structure
        return self.structure
    
    def read_kpoints(self) -> np.ndarray:
        """Read k-points from KPOINTS file."""
        kpoints_path = self.calc_dir / "KPOINTS"
        if not kpoints_path.exists():
            raise FileNotFoundError(f"No KPOINTS found in {self.calc_dir}")
        
        kpoints = Kpoints.from_file(kpoints_path)
        self.kpoints = kpoints.kpts
        return np.array(self.kpoints)
    
    def read_wavecar(self) -> Optional[Wavecar]:
        """Read WAVECAR file."""
        wavecar_path = self.calc_dir / "WAVECAR"
        if not wavecar_path.exists():
            warnings.warn(f"No WAVECAR found in {self.calc_dir}")
            return None
        
        try:
            return Wavecar(str(wavecar_path))
        except Exception as e:
            warnings.warn(f"Error reading WAVECAR: {e}")
            return None
    
    def prepare_for_z2pack(self, output_dir: str = "./z2pack_input") -> str:
        """
        Prepare VASP input files for Z2Pack calculation.
        
        Args:
            output_dir: Output directory for Z2Pack input
            
        Returns:
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy necessary files
        required_files = ["POSCAR", "POTCAR"]
        for fname in required_files:
            src = self.calc_dir / fname
            if src.exists():
                import shutil
                shutil.copy2(src, output_path / fname)
        
        # Generate INCAR for Wannier90 interface
        incar_dict = {
            "ISTART": 1,
            "LWAVE": True,
            "LCHARG": True,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "ENCUT": 500,
            "NBANDS": self.config.num_bands,
            "LORBIT": 11,
        }
        
        # Add Wannier90 tags if needed
        if self.config.num_wann:
            incar_dict.update({
                "LWANNIER90": True,
                "NUM_WANN": self.config.num_wann,
            })
        
        incar = Incar(incar_dict)
        incar.write_file(output_path / "INCAR")
        
        return str(output_path)


class Z2VASPInterface:
    """
    Interface between Z2Pack and VASP for calculating topological invariants.
    """
    
    def __init__(self, calc_dir: str = ".", config: Optional[Z2PackConfig] = None):
        """
        Initialize Z2Pack-VASP interface.
        
        Args:
            calc_dir: VASP calculation directory
            config: Z2Pack configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or Z2PackConfig()
        self.extractor = VASPWavefunctionExtractor(calc_dir, config)
        
        if not HAS_Z2PACK:
            raise ImportError("Z2Pack required. Install with: pip install z2pack")
    
    def create_surface(self, surface_type: str = "ky-surface") -> callable:
        """
        Create a surface function for Z2Pack calculation.
        
        Args:
            surface_type: Type of surface ("kx-surface", "ky-surface", "kz-surface")
            
        Returns:
            Surface function for Z2Pack
        """
        # Get reciprocal lattice
        structure = self.extractor.read_structure()
        reciprocal_lattice = structure.lattice.reciprocal_lattice
        
        if surface_type == "ky-surface":
            # Surface perpendicular to ky (varying kx, kz)
            def surface(s, t):
                return [t, s, 0]  # (kx, ky, kz)
        elif surface_type == "kx-surface":
            # Surface perpendicular to kx (varying ky, kz)
            def surface(s, t):
                return [s, t, 0]
        elif surface_type == "kz-surface":
            # Surface perpendicular to kz (varying kx, ky)
            def surface(s, t):
                return [t, 0, s]
        else:
            raise ValueError(f"Unknown surface type: {surface_type}")
        
        return surface
    
    def create_system(self, input_dir: str) -> z2pack.fp.System:
        """
        Create a Z2Pack system from VASP input.
        
        Args:
            input_dir: Directory containing VASP input files
            
        Returns:
            Z2Pack system object
        """
        input_path = Path(input_dir)
        
        # Create Z2Pack VASP system
        system = z2pack.fp.System(
            input_files=[str(input_path / "POSCAR"),
                        str(input_path / "INCAR"),
                        str(input_path / "POTCAR")],
            kpt_fct=z2pack.fp.kpts.vasp,
            kpt_path=input_path / "KPOINTS",
            command=f"{self.config.vasp_executable} > z2pack_output.log 2>&1",
            executable=self.config.vasp_executable,
            mmn_path=input_path / "wannier90.mmn",
        )
        
        return system
    
    def calculate_wilson_loop(
        self,
        surface: Optional[callable] = None,
        num_lines: Optional[int] = None,
    ) -> WilsonLoopResult:
        """
        Calculate Wilson loop for a given surface.
        
        Args:
            surface: Surface function (if None, use config default)
            num_lines: Number of k-lines (if None, use config default)
            
        Returns:
            Wilson loop result
        """
        # Prepare input
        input_dir = self.extractor.prepare_for_z2pack()
        
        # Create system
        system = self.create_system(input_dir)
        
        # Create surface
        if surface is None:
            surface = self.create_surface(self.config.surface)
        
        if num_lines is None:
            num_lines = self.config.num_lines
        
        # Run Z2Pack calculation
        result = z2pack.surface.run(
            system=system,
            surface=surface,
            save_file=Path(input_dir) / "wilson_loop.json",
            load=True,
            num_lines=num_lines,
            pos_tol=self.config.pos_tol,
            gap_tol=self.config.gap_tol,
            move_tol=self.config.move_tol,
            iterator=self.config.iterator,
            min_neighbour_dist=self.config.min_neighbour_dist,
        )
        
        # Extract Wilson loop data
        k_points = np.array([line.t for line in result.lines])
        phases = np.array([line.pol for line in result.lines])
        gaps = np.array([line.gap for line in result.lines])
        
        # Check convergence
        converged = z2pack.surface.check_convergence(
            result,
            pos_tol=self.config.pos_tol,
            gap_tol=self.config.gap_tol,
            move_tol=self.config.move_tol,
        )
        
        return WilsonLoopResult(
            k_points=k_points,
            phases=phases,
            gaps=gaps,
            converged=converged,
        )
    
    def calculate_z2_invariant(
        self,
        surfaces: Optional[List[str]] = None,
    ) -> Z2InvariantResult:
        """
        Calculate Z2 invariant (time-reversal symmetric case).
        
        For 3D materials, calculates all 4 Z2 invariants: ν0; ν1ν2ν3
        
        Args:
            surfaces: List of surfaces to calculate (if None, calculate all)
            
        Returns:
            Z2 invariant result
        """
        if surfaces is None:
            surfaces = ["ky-surface", "kx-surface", "kz-surface"]
        
        z2_indices = []
        chern_numbers = []
        gap_mins = []
        
        for surface_name in surfaces:
            # Calculate Wilson loop for this surface
            surface_func = self.create_surface(surface_name)
            wilson_result = self.calculate_wilson_loop(
                surface=surface_func,
                num_lines=self.config.num_lines,
            )
            
            # Calculate Z2 index from Wilson loop
            z2 = self._compute_z2_from_wilson(wilson_result)
            z2_indices.append(z2)
            
            # Calculate Chern number (should be 0 for time-reversal symmetric)
            chern = self._compute_chern_from_wilson(wilson_result)
            chern_numbers.append(chern)
            
            gap_mins.append(np.min(wilson_result.gaps))
        
        # Determine 3D Z2 invariants
        if len(z2_indices) >= 3:
            # Strong index ν0
            nu_0 = z2_indices[0] if z2_indices[0] == z2_indices[1] == z2_indices[2] else 0
            # Weak indices
            nu_1 = z2_indices[1] ^ z2_indices[2]
            nu_2 = z2_indices[0] ^ z2_indices[2]
            nu_3 = z2_indices[0] ^ z2_indices[1]
            z2_indices_3d = [nu_0, nu_1, nu_2, nu_3]
        else:
            z2_indices_3d = [z2_indices[0]] if z2_indices else [0]
        
        # Determine topological phase
        phase = self._determine_phase(z2_indices_3d, chern_numbers)
        
        return Z2InvariantResult(
            z2_index=z2_indices_3d[0],
            z2_indices=z2_indices_3d,
            chern_number=sum(chern_numbers) // len(chern_numbers) if chern_numbers else 0,
            gap_at_time_reversal=min(gap_mins) if gap_mins else 0.0,
            gap_minimum=min(gap_mins) if gap_mins else 0.0,
            converged=all([wilson_result.converged for wilson_result in [wilson_result]]),
            topological_phase=phase,
            surface=surfaces[0],
        )
    
    def _compute_z2_from_wilson(self, wilson_result: WilsonLoopResult) -> int:
        """
        Compute Z2 index from Wilson loop phases.
        
        Uses the parity method: count crossings of WCC at 0.5
        """
        wcc_centers = wilson_result.get_wcc_centers()
        
        # For each k-point, count number of WCC above 0.5
        count_above = np.sum(wcc_centers > 0.5, axis=1)
        
        # Z2 index is the parity of the number of swaps
        # Simplified: check if there's an odd number of WCC crossings
        z2 = int(np.sum(count_above) % 2)
        
        return z2
    
    def _compute_chern_from_wilson(self, wilson_result: WilsonLoopResult) -> int:
        """
        Compute Chern number from Wilson loop phases.
        
        Chern number is the total winding of WCC across the BZ.
        """
        wcc_centers = wilson_result.get_wcc_centers()
        
        # Calculate total winding
        total_winding = 0
        for i in range(len(wcc_centers) - 1):
            delta = wcc_centers[i+1] - wcc_centers[i]
            # Handle periodic boundary conditions
            delta = np.mod(delta + 0.5, 1) - 0.5
            total_winding += np.sum(delta)
        
        chern = int(round(total_winding))
        return chern
    
    def _determine_phase(
        self,
        z2_indices: List[int],
        chern_numbers: List[int],
    ) -> TopologicalPhase:
        """Determine topological phase from invariants."""
        if z2_indices[0] == 1:
            if self.config.time_reversal_symmetric:
                return TopologicalPhase.Z2_TI
            else:
                return TopologicalPhase.QUANTUM_ANOMALOUS_HALL
        elif any(c != 0 for c in chern_numbers):
            return TopologicalPhase.CHERN_INSULATOR
        else:
            return TopologicalPhase.TRIVIAL
    
    def plot_wilson_loop(self, wilson_result: WilsonLoopResult, save_path: Optional[str] = None):
        """
        Plot Wilson loop results.
        
        Args:
            wilson_result: Wilson loop result to plot
            save_path: Path to save plot (if None, display)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib required for plotting")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        
        # Plot WCC evolution
        ax1 = axes[0]
        wcc = wilson_result.get_wcc_centers()
        for i in range(wcc.shape[1]):
            ax1.plot(wilson_result.k_points, wcc[:, i], 'b-', linewidth=0.5)
        ax1.set_xlabel('k (fractional)')
        ax1.set_ylabel('WCC (fractional)')
        ax1.set_title('Wilson Loop / Wannier Charge Centers')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Plot gaps
        ax2 = axes[1]
        ax2.plot(wilson_result.k_points, wilson_result.gaps, 'r-', linewidth=1)
        ax2.axhline(y=self.config.gap_tol, color='k', linestyle='--', label=f'gap_tol={self.config.gap_tol}')
        ax2.set_xlabel('k (fractional)')
        ax2.set_ylabel('Gap (eV)')
        ax2.set_title('Direct Gap')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class ChernNumberCalculator:
    """
    Calculate Chern numbers using various methods.
    """
    
    def __init__(self, calc_dir: str = ".", config: Optional[Z2PackConfig] = None):
        """
        Initialize Chern number calculator.
        
        Args:
            calc_dir: VASP calculation directory
            config: Z2Pack configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or Z2PackConfig()
    
    def calculate_from_berry_curvature(
        self,
        k_mesh: Tuple[int, int, int] = (10, 10, 1),
    ) -> ChernNumberResult:
        """
        Calculate Chern number by integrating Berry curvature.
        
        Args:
            k_mesh: k-point mesh for integration
            
        Returns:
            Chern number result
        """
        # Generate k-mesh
        kx = np.linspace(0, 1, k_mesh[0], endpoint=False)
        ky = np.linspace(0, 1, k_mesh[1], endpoint=False)
        kz = np.linspace(0, 1, k_mesh[2], endpoint=False)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        berry_curvature = np.zeros_like(KX)
        
        # Calculate Berry curvature at each k-point
        for i in range(k_mesh[0]):
            for j in range(k_mesh[1]):
                for k in range(k_mesh[2]):
                    k_point = np.array([KX[i,j,k], KY[i,j,k], KZ[i,j,k]])
                    berry_curvature[i,j,k] = self._compute_berry_curvature(k_point)
        
        # Integrate to get Chern number
        chern = np.sum(berry_curvature) / (2 * np.pi)
        chern_int = int(round(chern))
        
        error = abs(chern - chern_int)
        
        return ChernNumberResult(
            chern_number=chern_int,
            berry_curvature=berry_curvature,
            k_mesh=np.stack([KX, KY, KZ], axis=-1),
            converged=error < 0.1,
            error_estimate=error,
        )
    
    def _compute_berry_curvature(self, k_point: np.ndarray) -> float:
        """
        Compute Berry curvature at a k-point.
        
        This is a placeholder - in practice, this would compute from
        wavefunction overlaps or use VASP's LCALCPOL output.
        """
        # Placeholder implementation
        # Real implementation would read from WAVECAR or use finite differences
        return 0.0
    
    def calculate_from_wilson_loop(
        self,
        plane_normal: np.ndarray = np.array([0, 0, 1]),
    ) -> int:
        """
        Calculate Chern number from Wilson loop on a plane.
        
        Args:
            plane_normal: Normal vector to the plane
            
        Returns:
            Chern number
        """
        # Use Z2Pack interface
        interface = Z2VASPInterface(str(self.calc_dir), self.config)
        
        # Create surface perpendicular to plane_normal
        surface_name = self._get_surface_name(plane_normal)
        surface = interface.create_surface(surface_name)
        
        wilson_result = interface.calculate_wilson_loop(surface=surface)
        
        return interface._compute_chern_from_wilson(wilson_result)
    
    def _get_surface_name(self, plane_normal: np.ndarray) -> str:
        """Convert plane normal to surface name."""
        normal = plane_normal / np.linalg.norm(plane_normal)
        
        if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
            return "kz-surface"
        elif np.allclose(normal, [0, 1, 0]) or np.allclose(normal, [0, -1, 0]):
            return "ky-surface"
        elif np.allclose(normal, [1, 0, 0]) or np.allclose(normal, [-1, 0, 0]):
            return "kx-surface"
        else:
            raise ValueError("Arbitrary plane normals not yet supported")


class TopologicalClassifier:
    """
    Classify materials by their topological properties.
    """
    
    def __init__(self, config: Optional[Z2PackConfig] = None):
        """
        Initialize topological classifier.
        
        Args:
            config: Z2Pack configuration
        """
        self.config = config or Z2PackConfig()
    
    def classify_material(
        self,
        calc_dir: str,
        has_time_reversal: bool = True,
        has_inversion: bool = True,
    ) -> Dict[str, Any]:
        """
        Classify a material's topological phase.
        
        Args:
            calc_dir: VASP calculation directory
            has_time_reversal: Whether system has time-reversal symmetry
            has_inversion: Whether system has inversion symmetry
            
        Returns:
            Dictionary with classification results
        """
        results = {
            "calc_dir": calc_dir,
            "symmetries": {
                "time_reversal": has_time_reversal,
                "inversion": has_inversion,
            },
        }
        
        # Calculate invariants
        interface = Z2VASPInterface(calc_dir, self.config)
        
        if has_time_reversal:
            # Calculate Z2 invariants
            z2_result = interface.calculate_z2_invariant()
            results["z2_invariant"] = {
                "strong": z2_result.z2_index,
                "weak": z2_result.z2_indices[1:],
                "chern": z2_result.chern_number,
            }
            results["topological_phase"] = z2_result.topological_phase.name
        else:
            # Calculate Chern numbers only
            chern_calc = ChernNumberCalculator(calc_dir, self.config)
            chern = chern_calc.calculate_from_wilson_loop()
            results["chern_number"] = chern
            results["topological_phase"] = (
                TopologicalPhase.CHERN_INSULATOR.name if chern != 0 
                else TopologicalPhase.TRIVIAL.name
            )
        
        return results
    
    def check_band_inversion(
        self,
        band_structure: np.ndarray,
        reference_bands: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Check for band inversion at time-reversal invariant momenta (TRIM).
        
        Args:
            band_structure: Band energies at TRIM points
            reference_bands: Reference (atomic limit) band energies
            
        Returns:
            Dictionary with band inversion analysis
        """
        if reference_bands is None:
            # Use first TRIM as reference if not provided
            reference_bands = band_structure[0]
        
        inversions = []
        for i, bands in enumerate(band_structure):
            # Check for band crossing/inversion
            # Simplified: check if band ordering differs from reference
            ref_order = np.argsort(reference_bands)
            current_order = np.argsort(bands)
            
            if not np.array_equal(ref_order, current_order):
                inversions.append(i)
        
        return {
            "num_inversions": len(inversions),
            "inversion_points": inversions,
            "has_inversion": len(inversions) > 0,
        }


# Convenience functions

def calculate_z2_index(
    calc_dir: str,
    surface: str = "ky-surface",
    num_lines: int = 11,
) -> int:
    """
    Calculate Z2 index for a material.
    
    Args:
        calc_dir: VASP calculation directory
        surface: Surface for Wilson loop calculation
        num_lines: Number of k-lines
        
    Returns:
        Z2 index (0 or 1)
    """
    config = Z2PackConfig(surface=surface, num_lines=num_lines)
    interface = Z2VASPInterface(calc_dir, config)
    result = interface.calculate_z2_invariant(surfaces=[surface])
    return result.z2_index


def calculate_chern_number(
    calc_dir: str,
    plane_normal: np.ndarray = np.array([0, 0, 1]),
) -> int:
    """
    Calculate Chern number for a material.
    
    Args:
        calc_dir: VASP calculation directory
        plane_normal: Normal to the plane for calculation
        
    Returns:
        Chern number
    """
    config = Z2PackConfig()
    calculator = ChernNumberCalculator(calc_dir, config)
    return calculator.calculate_from_wilson_loop(plane_normal)


def classify_topological_material(calc_dir: str) -> Dict[str, Any]:
    """
    Complete topological classification of a material.
    
    Args:
        calc_dir: VASP calculation directory
        
    Returns:
        Complete classification results
    """
    classifier = TopologicalClassifier()
    return classifier.classify_material(calc_dir)


# Example usage
if __name__ == "__main__":
    # Example: Calculate Z2 index for Bi2Se3
    print("Z2Pack Interface for Topological Calculations")
    print("=" * 50)
    
    # Check if Z2Pack is available
    if HAS_Z2PACK:
        print("✓ Z2Pack available")
    else:
        print("✗ Z2Pack not available. Install with: pip install z2pack")
    
    if HAS_PMG:
        print("✓ Pymatgen available")
    else:
        print("✗ Pymatgen not available")
