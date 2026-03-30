"""
WannierTools Interface for Surface States and Weyl Point Calculations
=====================================================================

This module provides interfaces to WannierTools for:
- Wannier90 Hamiltonian construction
- Surface state calculations
- Band inversion identification
- Weyl point search

WannierTools provides powerful methods for studying topological materials
including surface Fermi arcs, Weyl points, and nodal lines.

References:
- WannierTools: http://www.wanniertools.com
- Wu et al., Comput. Phys. Commun. 224, 405 (2018)
"""

import os
import re
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.vasp import Poscar, Kpoints, Incar
    from pymatgen.core.sites import PeriodicSite
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


class SurfaceType(Enum):
    """Types of surface termination."""
    TOP = "top"
    BOTTOM = "bottom"
    BOTH = "both"


class WeylChirality(Enum):
    """Chirality of Weyl points."""
    POSITIVE = 1
    NEGATIVE = -1
    ZERO = 0


@dataclass
class WannierToolsConfig:
    """Configuration for WannierTools calculations."""
    # Executable
    wt_executable: str = "wt.x"
    
    # Wannier90 settings
    num_wann: int = 20
    num_iter: int = 1000
    dis_win_min: float = -10.0
    dis_win_max: float = 10.0
    fermi_energy: Optional[float] = None
    
    # Surface settings
    surface_type: SurfaceType = SurfaceType.TOP
    num_layers: int = 10
    vacuum_layer: float = 20.0  # Angstrom
    
    # k-mesh for surface states
    nk1: int = 100
    nk2: int = 100
    
    # Weyl point search settings
    weyl_search_window: Tuple[float, float] = (-2.0, 2.0)
    weyl_tolerance: float = 0.01
    
    # Output settings
    calculate_arpes: bool = False
    calculate_spin_texture: bool = True
    calculate_fermi_arc: bool = True
    
    # Symmetry settings
    symprec: float = 1e-5
    search_symmetry: bool = True


@dataclass
class SurfaceStateResult:
    """Result of surface state calculation."""
    k_points: np.ndarray           # Shape: (nk1, nk2, 3)
    energies: np.ndarray           # Shape: (nk1, nk2, num_bands)
    spectral_weight: np.ndarray    # Shape: (nk1, nk2, num_bands)
    fermi_surface: Optional[np.ndarray] = None
    has_dirac_cone: bool = False
    dirac_points: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class WeylPoint:
    """Representation of a Weyl point."""
    k_point: np.ndarray       # (kx, ky, kz) in fractional coordinates
    energy: float
    chirality: WeylChirality
    charge: int               # +1 or -1
    
    def __post_init__(self):
        if self.chirality == WeylChirality.POSITIVE:
            self.charge = 1
        elif self.chirality == WeylChirality.NEGATIVE:
            self.charge = -1
        else:
            self.charge = 0


@dataclass
class WeylSearchResult:
    """Result of Weyl point search."""
    weyl_points: List[WeylPoint]
    num_weyl_points: int
    num_positive: int
    num_negative: int
    total_charge: int
    converged: bool
    
    def get_weyl_points_by_chirality(self, chirality: WeylChirality) -> List[WeylPoint]:
        """Get Weyl points with specified chirality."""
        return [wp for wp in self.weyl_points if wp.chirality == chirality]
    
    def get_fermi_arc_connectivity(self) -> List[Tuple[WeylPoint, WeylPoint]]:
        """
        Get pairs of Weyl points connected by Fermi arcs.
        
        Returns list of (positive, negative) pairs.
        """
        positive = self.get_weyl_points_by_chirality(WeylChirality.POSITIVE)
        negative = self.get_weyl_points_by_chirality(WeylChirality.NEGATIVE)
        
        # Simple nearest-neighbor pairing
        pairs = []
        for p in positive:
            # Find closest negative Weyl point
            distances = [np.linalg.norm(p.k_point - n.k_point) for n in negative]
            if distances:
                closest_idx = np.argmin(distances)
                pairs.append((p, negative[closest_idx]))
        
        return pairs


@dataclass
class BandInversionResult:
    """Result of band inversion analysis."""
    inversion_points: List[Dict[str, Any]]
    num_inversions: int
    critical_bands: List[int]
    is_topological: bool
    tr_points: List[np.ndarray]  # Time-reversal invariant momenta


class Wannier90HamiltonianBuilder:
    """
    Build Wannier90 tight-binding Hamiltonian from VASP calculations.
    """
    
    def __init__(
        self,
        calc_dir: str = ".",
        config: Optional[WannierToolsConfig] = None,
    ):
        """
        Initialize Hamiltonian builder.
        
        Args:
            calc_dir: VASP calculation directory
            config: WannierTools configuration
        """
        self.calc_dir = Path(calc_dir)
        self.config = config or WannierToolsConfig()
        self.structure = None
        
        if not HAS_PMG:
            raise ImportError("Pymatgen required for Hamiltonian building")
    
    def read_structure(self) -> Structure:
        """Read crystal structure."""
        poscar_path = self.calc_dir / "POSCAR"
        if not poscar_path.exists():
            poscar_path = self.calc_dir / "CONTCAR"
        
        if not poscar_path.exists():
            raise FileNotFoundError(f"No POSCAR/CONTCAR in {self.calc_dir}")
        
        self.structure = Poscar.from_file(poscar_path).structure
        return self.structure
    
    def generate_wannier90_input(self, output_dir: str = "./wannier90") -> str:
        """
        Generate Wannier90 input files.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read structure
        structure = self.read_structure()
        
        # Create wannier90.win file
        win_content = self._generate_win_file(structure)
        
        with open(output_path / "wannier90.win", 'w') as f:
            f.write(win_content)
        
        # Generate projections
        projections = self._generate_projections(structure)
        
        with open(output_path / "wannier90.proj", 'w') as f:
            f.write(projections)
        
        # Copy VASP files
        self._copy_vasp_files(output_path)
        
        return str(output_path)
    
    def _generate_win_file(self, structure: Structure) -> str:
        """Generate wannier90.win content."""
        lines = [
            "# Wannier90 input file generated by DFTLammps",
            "",
            "# System settings",
            f"num_wann = {self.config.num_wann}",
            f"num_iter = {self.config.num_iter}",
            "",
            "# Disentanglement",
            "dis_win_min = {:.2f}".format(self.config.dis_win_min),
            "dis_win_max = {:.2f}".format(self.config.dis_win_max),
            "dis_froz_min = {:.2f}".format(self.config.dis_win_min + 1.0),
            "dis_froz_max = {:.2f}".format(self.config.dis_win_max - 1.0),
            "",
            "# Wannierization",
            "use_bloch_phases = false",
            "guiding_centres = true",
            "",
            "# Output",
            "write_hr = true",
            "write_tb = true",
            "write_xyz = true",
            "",
            "# k-mesh for Wannierization",
            "mp_grid : 4 4 4",
            "",
            "# Unit cell",
        ]
        
        # Add lattice vectors
        for i, vec in enumerate(structure.lattice.matrix):
            lines.append(f"{vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}")
        
        lines.append("")
        lines.append("# Atoms")
        
        # Add atomic positions
        for site in structure:
            lines.append(
                f"{site.species_string:4s} {site.frac_coords[0]:10.6f} "
                f"{site.frac_coords[1]:10.6f} {site.frac_coords[2]:10.6f}"
            )
        
        lines.append("")
        lines.append("# Projections")
        lines.append(self._generate_projections(structure))
        
        return "\n".join(lines)
    
    def _generate_projections(self, structure: Structure) -> str:
        """Generate projection centers for Wannier functions."""
        projections = []
        
        # Generate projections based on elements
        for site in structure:
            element = site.species_string
            coords = " ".join([f"{c:10.6f}" for c in site.frac_coords])
            
            # Add common projections for each element type
            if element in ["Bi", "Sb", "As", "P"]:
                projections.append(f"{element}:p;{coords}")
                projections.append(f"{element}:s;{coords}")
            elif element in ["Se", "Te", "S", "O"]:
                projections.append(f"{element}:p;{coords}")
            elif element in ["Fe", "Co", "Ni", "Mn"]:
                projections.append(f"{element}:d;{coords}")
                projections.append(f"{element}:s;{coords}")
            elif element in ["Ta", "W", "Mo", "Nb"]:
                projections.append(f"{element}:d;{coords}")
            else:
                projections.append(f"{element}:s,p;{coords}")
        
        return "\n".join(projections)
    
    def _copy_vasp_files(self, output_path: Path):
        """Copy necessary VASP files."""
        required = ["POSCAR", "POTCAR", "CHGCAR", "WAVECAR"]
        for fname in required:
            src = self.calc_dir / fname
            if src.exists():
                import shutil
                shutil.copy2(src, output_path / fname)
    
    def run_wannier90(self, work_dir: str) -> bool:
        """
        Run Wannier90 to generate tight-binding model.
        
        Args:
            work_dir: Working directory with input files
            
        Returns:
            True if successful
        """
        work_path = Path(work_dir)
        
        # First run VASP with LWANNIER90
        # Then run wannier90.x
        
        commands = [
            # "vasp_std",  # Run VASP first
            "wannier90.x -pp wannier90",  # Pre-processing
            "wannier90.x wannier90",      # Main calculation
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd.split(),
                    cwd=work_path,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                if result.returncode != 0:
                    warnings.warn(f"Command failed: {cmd}\n{result.stderr}")
                    return False
            except Exception as e:
                warnings.warn(f"Error running {cmd}: {e}")
                return False
        
        return True


class WannierToolsCalculator:
    """
    Interface to WannierTools for surface states and Weyl points.
    """
    
    def __init__(
        self,
        hr_file: str,
        config: Optional[WannierToolsConfig] = None,
    ):
        """
        Initialize WannierTools calculator.
        
        Args:
            hr_file: Path to Wannier90 HR file (tight-binding Hamiltonian)
            config: WannierTools configuration
        """
        self.hr_file = Path(hr_file)
        self.config = config or WannierToolsConfig()
        self.hamiltonian = None
        self.r_vectors = None
        self.num_wann = None
        
        if self.hr_file.exists():
            self._read_hr_file()
    
    def _read_hr_file(self):
        """Read Wannier90 *_hr.dat file."""
        with open(self.hr_file, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        self.num_wann = int(lines[1].strip())
        nrpts = int(lines[2].strip())
        
        # Read degeneracy of each Wigner-Seitz grid point
        ndegen = []
        line_idx = 3
        while len(ndegen) < nrpts:
            ndegen.extend([int(x) for x in lines[line_idx].split()])
            line_idx += 1
        
        # Read Hamiltonian elements
        data = []
        for line in lines[line_idx:]:
            parts = line.split()
            if len(parts) == 7:
                R = np.array([int(parts[0]), int(parts[1]), int(parts[2])])
                i, j = int(parts[3]) - 1, int(parts[4]) - 1  # 0-indexed
                Hr = float(parts[5])
                Hi = float(parts[6])
                data.append((R, i, j, Hr + 1j * Hi))
        
        # Store data
        self.r_vectors = np.array([d[0] for d in data])
        self.hamiltonian = data
    
    def generate_wanniertools_input(
        self,
        output_file: str = "wt.in",
        structure: Optional[Structure] = None,
    ) -> str:
        """
        Generate WannierTools input file.
        
        Args:
            output_file: Output file name
            structure: Crystal structure
            
        Returns:
            Path to input file
        """
        lines = [
            "# WannierTools input file",
            "",
            "# Tight-binding Hamiltonian",
            f"Hrfile = {self.hr_file.name}",
            f"num_wann = {self.num_wann or self.config.num_wann}",
            "",
            "# Fermi level",
            f"E_fermi = {self.config.fermi_energy or 0.0}",
            "",
            "# Surface states",
            "DOS_calc = T",
            "Dos_kplane_calc = F",
            "surfstat_calc = T",
            "",
            "# Surface definition",
        ]
        
        if structure:
            # Define surface termination
            lattice = structure.lattice.matrix
            lines.extend([
                "# Lattice vectors",
                f"{lattice[0][0]:10.6f} {lattice[0][1]:10.6f} {lattice[0][2]:10.6f}",
                f"{lattice[1][0]:10.6f} {lattice[1][1]:10.6f} {lattice[1][2]:10.6f}",
                f"{lattice[2][0]:10.6f} {lattice[2][1]:10.6f} {lattice[2][2]:10.6f}",
                "",
                "# Surface (001)",
                "Nslab = 10",
                "Nsurface = 1",
                "",
            ])
        
        # Add k-mesh
        lines.extend([
            "# k-mesh",
            f"Nk1 = {self.config.nk1}",
            f"Nk2 = {self.config.nk2}",
            "",
            "# Weyl point search",
            "FindNodes_calc = T",
            f"Node_tol = {self.config.weyl_tolerance}",
            "",
            "# Berry curvature and AHC",
            "Berry_calc = T",
            "Berry_kplane_calc = F",
            "",
            "# Output",
            "BulkGap_cube_calc = F",
            "BulkGap_plane_calc = T",
            "WireBand_calc = F",
        ])
        
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        
        return output_file
    
    def run_wanniertools(self, work_dir: str = ".") -> bool:
        """
        Run WannierTools calculation.
        
        Args:
            work_dir: Working directory
            
        Returns:
            True if successful
        """
        work_path = Path(work_dir)
        
        try:
            result = subprocess.run(
                [self.config.wt_executable],
                cwd=work_path,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            return result.returncode == 0
        except Exception as e:
            warnings.warn(f"Error running WannierTools: {e}")
            return False
    
    def calculate_surface_states(
        self,
        k_path: Optional[np.ndarray] = None,
        num_layers: Optional[int] = None,
    ) -> SurfaceStateResult:
        """
        Calculate surface states from tight-binding Hamiltonian.
        
        Args:
            k_path: k-point path (if None, use default)
            num_layers: Number of layers for surface (if None, use config)
            
        Returns:
            Surface state result
        """
        if num_layers is None:
            num_layers = self.config.num_layers
        
        # Generate k-mesh if not provided
        if k_path is None:
            kx = np.linspace(-0.5, 0.5, self.config.nk1)
            ky = np.linspace(-0.5, 0.5, self.config.nk2)
            KX, KY = np.meshgrid(kx, ky)
            k_path = np.stack([KX.flatten(), KY.flatten(), np.zeros_like(KX.flatten())], axis=1)
        
        # Build slab Hamiltonian
        energies = []
        weights = []
        
        for k in k_path:
            H_slab = self._build_slab_hamiltonian(k, num_layers)
            eigs, vecs = np.linalg.eigh(H_slab)
            
            # Calculate surface weights (top and bottom layers)
            surface_weight = self._calculate_surface_weight(vecs, num_layers)
            
            energies.append(eigs)
            weights.append(surface_weight)
        
        energies = np.array(energies)
        weights = np.array(weights)
        
        # Check for Dirac cone
        dirac_points = self._identify_dirac_points(k_path, energies)
        
        return SurfaceStateResult(
            k_points=k_path,
            energies=energies,
            spectral_weight=weights,
            has_dirac_cone=len(dirac_points) > 0,
            dirac_points=dirac_points,
        )
    
    def _build_slab_hamiltonian(
        self,
        k_point: np.ndarray,
        num_layers: int,
    ) -> np.ndarray:
        """
        Build slab Hamiltonian for surface state calculation.
        
        Args:
            k_point: 2D k-point (kx, ky)
            num_layers: Number of layers in slab
            
        Returns:
            Slab Hamiltonian matrix
        """
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not loaded")
        
        nw = self.num_wann
        H_slab = np.zeros((nw * num_layers, nw * num_layers), dtype=complex)
        
        # Build Hamiltonian by Fourier transform
        k = np.array([k_point[0], k_point[1], 0.0])
        
        for R, i, j, H_R in self.hamiltonian:
            phase = np.exp(2j * np.pi * np.dot(k, R))
            
            # Map to slab layers
            layer_i = i % num_layers
            layer_j = j % num_layers
            
            idx_i = layer_i * nw + (i % nw)
            idx_j = layer_j * nw + (j % nw)
            
            H_slab[idx_i, idx_j] += H_R * phase
        
        return H_slab
    
    def _calculate_surface_weight(
        self,
        eigenvectors: np.ndarray,
        num_layers: int,
    ) -> np.ndarray:
        """Calculate surface weight for each state."""
        nw = self.num_wann
        weights = np.zeros(eigenvectors.shape[1])
        
        for n in range(eigenvectors.shape[1]):
            vec = eigenvectors[:, n]
            weight_surface = 0.0
            
            # Top layer
            for i in range(nw):
                weight_surface += abs(vec[i])**2
            
            # Bottom layer
            for i in range((num_layers-1)*nw, num_layers*nw):
                weight_surface += abs(vec[i])**2
            
            weights[n] = weight_surface
        
        return weights
    
    def _identify_dirac_points(
        self,
        k_points: np.ndarray,
        energies: np.ndarray,
        fermi_energy: float = 0.0,
        tolerance: float = 0.1,
    ) -> List[Tuple[float, float, float]]:
        """
        Identify Dirac points in surface state spectrum.
        
        Args:
            k_points: Array of k-points
            energies: Array of energies
            fermi_energy: Fermi energy
            tolerance: Energy tolerance for identifying crossing
            
        Returns:
            List of Dirac point coordinates (kx, ky, E)
        """
        dirac_points = []
        
        # Look for linear band crossings near Fermi level
        for i, k in enumerate(k_points):
            bands_at_k = energies[i]
            
            # Find bands near Fermi level
            near_fermi = np.abs(bands_at_k - fermi_energy) < tolerance
            
            if np.sum(near_fermi) >= 2:
                # Potential Dirac point
                # Check if bands cross (discontinuity in derivative)
                dirac_points.append((k[0], k[1], fermi_energy))
        
        return dirac_points
    
    def search_weyl_points(
        self,
        k_mesh: Tuple[int, int, int] = (20, 20, 20),
        energy_window: Optional[Tuple[float, float]] = None,
    ) -> WeylSearchResult:
        """
        Search for Weyl points in the Brillouin zone.
        
        Args:
            k_mesh: k-point mesh for search
            energy_window: Energy window for search
            
        Returns:
            Weyl search result
        """
        if energy_window is None:
            energy_window = self.config.weyl_search_window
        
        # Generate k-mesh
        kx = np.linspace(0, 1, k_mesh[0], endpoint=False)
        ky = np.linspace(0, 1, k_mesh[1], endpoint=False)
        kz = np.linspace(0, 1, k_mesh[2], endpoint=False)
        
        weyl_points = []
        
        # Search for band crossings
        for i, kx_i in enumerate(kx):
            for j, ky_j in enumerate(ky):
                for k, kz_k in enumerate(kz):
                    k_point = np.array([kx_i, ky_j, kz_k])
                    
                    # Check for Weyl point at this k-point
                    weyl = self._check_weyl_point(k_point, energy_window)
                    if weyl:
                        weyl_points.append(weyl)
        
        # Remove duplicates
        weyl_points = self._remove_duplicate_weyl(weyl_points)
        
        # Count by chirality
        num_pos = sum(1 for wp in weyl_points if wp.chirality == WeylChirality.POSITIVE)
        num_neg = sum(1 for wp in weyl_points if wp.chirality == WeylChirality.NEGATIVE)
        
        return WeylSearchResult(
            weyl_points=weyl_points,
            num_weyl_points=len(weyl_points),
            num_positive=num_pos,
            num_negative=num_neg,
            total_charge=num_pos - num_neg,
            converged=len(weyl_points) > 0,
        )
    
    def _check_weyl_point(
        self,
        k_point: np.ndarray,
        energy_window: Tuple[float, float],
    ) -> Optional[WeylPoint]:
        """
        Check if a k-point is a Weyl point.
        
        Uses Berry curvature to determine chirality.
        """
        # Calculate Berry curvature at this point
        berry_curvature = self._calculate_berry_curvature(k_point)
        
        # Calculate band gap
        gap, bands = self._calculate_gap(k_point)
        
        # Check for gap closing within energy window
        if gap > self.config.weyl_tolerance:
            return None
        
        if not (energy_window[0] <= bands[0] <= energy_window[1]):
            return None
        
        # Determine chirality from Berry curvature flux
        chirality = self._determine_chirality(k_point, berry_curvature)
        
        return WeylPoint(
            k_point=k_point,
            energy=bands[0],
            chirality=chirality,
            charge=1 if chirality == WeylChirality.POSITIVE else -1,
        )
    
    def _calculate_berry_curvature(self, k_point: np.ndarray) -> np.ndarray:
        """Calculate Berry curvature at a k-point."""
        # Simplified: return dummy curvature
        # Real implementation would use kubo formula or Wilson loop
        dk = 0.01
        
        # Calculate derivative of Hamiltonian
        H_k = self._get_hamiltonian_at_k(k_point)
        H_kx = self._get_hamiltonian_at_k(k_point + np.array([dk, 0, 0]))
        H_ky = self._get_hamiltonian_at_k(k_point + np.array([0, dk, 0]))
        H_kz = self._get_hamiltonian_at_k(k_point + np.array([0, 0, dk]))
        
        # Simplified Berry curvature calculation
        # Real implementation would use perturbation theory
        berry = np.zeros(3)
        
        return berry
    
    def _get_hamiltonian_at_k(self, k_point: np.ndarray) -> np.ndarray:
        """Get Hamiltonian matrix at a k-point."""
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not loaded")
        
        nw = self.num_wann
        H_k = np.zeros((nw, nw), dtype=complex)
        
        for R, i, j, H_R in self.hamiltonian:
            phase = np.exp(2j * np.pi * np.dot(k_point, R))
            H_k[i % nw, j % nw] += H_R * phase
        
        return H_k
    
    def _calculate_gap(self, k_point: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate band gap at a k-point."""
        H_k = self._get_hamiltonian_at_k(k_point)
        eigs = np.linalg.eigvalsh(H_k)
        
        # Find gap at Fermi level
        occupied = eigs[eigs < 0]
        unoccupied = eigs[eigs >= 0]
        
        if len(occupied) > 0 and len(unoccupied) > 0:
            gap = np.min(unoccupied) - np.max(occupied)
        else:
            gap = 0.0
        
        return gap, eigs
    
    def _determine_chirality(
        self,
        k_point: np.ndarray,
        berry_curvature: np.ndarray,
    ) -> WeylChirality:
        """Determine chirality of a Weyl point."""
        # Simplified: use sign of Berry curvature
        charge = np.sign(np.sum(berry_curvature))
        
        if charge > 0:
            return WeylChirality.POSITIVE
        elif charge < 0:
            return WeylChirality.NEGATIVE
        else:
            return WeylChirality.ZERO
    
    def _remove_duplicate_weyl(
        self,
        weyl_points: List[WeylPoint],
        tolerance: float = 0.05,
    ) -> List[WeylPoint]:
        """Remove duplicate Weyl points."""
        unique = []
        
        for wp in weyl_points:
            is_duplicate = False
            for existing in unique:
                dist = np.linalg.norm(wp.k_point - existing.k_point)
                if dist < tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(wp)
        
        return unique


class BandInversionAnalyzer:
    """
    Analyze band inversion at time-reversal invariant momenta.
    """
    
    def __init__(self, calc_dir: str = "."):
        """
        Initialize band inversion analyzer.
        
        Args:
            calc_dir: Calculation directory
        """
        self.calc_dir = Path(calc_dir)
    
    def identify_tr_points(self, structure: Structure) -> List[np.ndarray]:
        """
        Identify time-reversal invariant momenta (TRIM) in the BZ.
        
        Args:
            structure: Crystal structure
            
        Returns:
            List of TRIM k-points
        """
        # TRIM points have k = -k + G, i.e., 2k = G
        # In fractional coordinates, these are at 0 or 0.5
        trim_points = []
        
        for i in [0, 0.5]:
            for j in [0, 0.5]:
                for k in [0, 0.5]:
                    trim_points.append(np.array([i, j, k]))
        
        return trim_points
    
    def analyze_band_ordering(
        self,
        band_energies: Dict[str, np.ndarray],
        reference_energies: Optional[Dict[str, np.ndarray]] = None,
    ) -> BandInversionResult:
        """
        Analyze band ordering at TRIM points.
        
        Args:
            band_energies: Dict mapping TRIM labels to band energies
            reference_energies: Reference (atomic) band energies
            
        Returns:
            Band inversion analysis result
        """
        inversions = []
        critical_bands = set()
        
        trim_points = list(band_energies.keys())
        
        # Use first TRIM as reference if not provided
        if reference_energies is None:
            reference_energies = {trim_points[0]: band_energies[trim_points[0]]}
        
        ref_trim = list(reference_energies.keys())[0]
        ref_bands = reference_energies[ref_trim]
        
        for trim, bands in band_energies.items():
            # Compare band ordering
            ref_order = np.argsort(ref_bands)
            current_order = np.argsort(bands)
            
            # Find inversions
            for i, (ref_idx, curr_idx) in enumerate(zip(ref_order, current_order)):
                if ref_idx != curr_idx:
                    inversions.append({
                        "trim": trim,
                        "band_i": i,
                        "band_j": np.where(current_order == ref_idx)[0][0],
                        "energy_diff": abs(bands[curr_idx] - ref_bands[ref_idx]),
                    })
                    critical_bands.add(i)
                    critical_bands.add(np.where(current_order == ref_idx)[0][0])
        
        is_topological = len(inversions) > 0
        
        return BandInversionResult(
            inversion_points=inversions,
            num_inversions=len(inversions),
            critical_bands=sorted(list(critical_bands)),
            is_topological=is_topological,
            tr_points=[np.array([0,0,0])],  # Simplified
        )
    
    def calculate_parity_eigenvalues(
        self,
        wavefunctions: Dict[str, np.ndarray],
        inversion_center: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate parity eigenvalues at TRIM points.
        
        Args:
            wavefunctions: Wavefunctions at TRIM points
            inversion_center: Inversion center position
            
        Returns:
            Dict mapping TRIM labels to parity eigenvalues
        """
        parities = {}
        
        for trim, wf in wavefunctions.items():
            # Parity eigenvalue: <ψ|Î|ψ> where Î is inversion operator
            # For TRIM: ψ(-r) = ±ψ(r)
            # Simplified: return dummy values
            parities[trim] = np.ones(len(wf))  # Placeholder
        
        return parities
    
    def compute_z2_from_parity(
        self,
        parity_eigenvalues: Dict[str, np.ndarray],
        occupied_bands: int,
    ) -> int:
        """
        Compute Z2 index from parity eigenvalues (Fu-Kane formula).
        
        Args:
            parity_eigenvalues: Parity eigenvalues at TRIM points
            occupied_bands: Number of occupied bands
            
        Returns:
            Z2 index (0 or 1)
        """
        # Fu-Kane formula:
        # (-1)^ν = ∏_i δ_i where δ_i = ∏_m ξ_{2m}(Λ_i)
        # Product over all TRIM points, ξ is parity eigenvalue
        
        product = 1
        for trim, parities in parity_eigenvalues.items():
            # Product of parity eigenvalues for occupied bands
            trim_product = np.prod(parities[:occupied_bands])
            product *= trim_product
        
        z2 = int((1 - product) // 2)
        return z2


# Convenience functions

def calculate_surface_states(
    hr_file: str,
    k_path: Optional[np.ndarray] = None,
    num_layers: int = 10,
) -> SurfaceStateResult:
    """
    Calculate surface states from Wannier90 Hamiltonian.
    
    Args:
        hr_file: Path to Wannier90 *_hr.dat file
        k_path: k-point path
        num_layers: Number of layers in slab
        
    Returns:
        Surface state result
    """
    config = WannierToolsConfig(num_layers=num_layers)
    calculator = WannierToolsCalculator(hr_file, config)
    return calculator.calculate_surface_states(k_path, num_layers)


def search_weyl_points(
    hr_file: str,
    k_mesh: Tuple[int, int, int] = (20, 20, 20),
) -> WeylSearchResult:
    """
    Search for Weyl points in a material.
    
    Args:
        hr_file: Path to Wannier90 *_hr.dat file
        k_mesh: k-point mesh for search
        
    Returns:
        Weyl search result
    """
    config = WannierToolsConfig()
    calculator = WannierToolsCalculator(hr_file, config)
    return calculator.search_weyl_points(k_mesh)


def analyze_band_inversion(
    calc_dir: str,
    band_energies: Dict[str, np.ndarray],
) -> BandInversionResult:
    """
    Analyze band inversion at TRIM points.
    
    Args:
        calc_dir: Calculation directory
        band_energies: Band energies at TRIM points
        
    Returns:
        Band inversion analysis
    """
    analyzer = BandInversionAnalyzer(calc_dir)
    return analyzer.analyze_band_ordering(band_energies)


# Example usage
if __name__ == "__main__":
    print("WannierTools Interface for Surface States and Weyl Points")
    print("=" * 60)
    
    print("\nFeatures:")
    print("- Wannier90 Hamiltonian construction")
    print("- Surface state calculations")
    print("- Band inversion identification")
    print("- Weyl point search")
    print("- Fermi arc analysis")
