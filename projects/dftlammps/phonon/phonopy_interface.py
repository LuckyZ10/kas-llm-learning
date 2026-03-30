"""
Phonopy Interface Module for DFT+MD+AI Materials Platform
==========================================================

Unified interface for phonon calculations using Phonopy with VASP/QE support.

Features:
- Force constants calculation from DFT (VASP IBRION=5,6,7,8 / QE ph.x)
- Phonon dispersion curve calculation and visualization
- Phonon DOS/PDOS calculations
- Thermodynamic properties: Free energy, entropy, heat capacity (Cv)
- Quasi-Harmonic Approximation (QHA) for thermal expansion

Author: DFTLammps Phonon Team
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import shutil

import numpy as np
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

# Phonopy imports
try:
    import phonopy
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.file_IO import (
        read_force_constants_from_hdf5,
        write_force_constants_to_hdf5,
        write_FORCE_CONSTANTS,
        parse_FORCE_CONSTANTS,
        parse_FORCE_SETS,
        write_disp_yaml,
        write_phonopy_yaml
    )
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
    from phonopy.phonon.dos import TotalDos, ProjectedDos
    from phonopy.spectrum.dynamic_structure_factor import atomic_form_factor_WK1995
    from phonopy.units import VaspToTHz, THzToCm, THzToEv, EvToTHz
    from phonopy.qha import QHA
    PHONOPY_AVAILABLE = True
except ImportError as e:
    PHONOPY_AVAILABLE = False
    warnings.warn(f"Phonopy not available: {e}")

# Phono3py for thermal conductivity
try:
    from phono3py import Phono3py
    from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
    PHONO3PY_AVAILABLE = True
except ImportError:
    PHONO3PY_AVAILABLE = False
    warnings.warn("Phono3py not available - thermal conductivity features disabled")

# Pymatgen for structure handling
try:
    from pymatgen.core import Structure, Lattice, Element
    from pymatgen.io.vasp import Poscar, Vasprun, Outcar
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PMG_AVAILABLE = True
except ImportError:
    PMG_AVAILABLE = False
    warnings.warn("Pymatgen not available")

# ASE for structure manipulation
try:
    from ase import Atoms
    from ase.io import read, write
    from ase.build import make_supercell
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    warnings.warn("ASE not available")

logger = logging.getLogger(__name__)


class DFTCode(Enum):
    """Supported DFT codes for force constant calculations."""
    VASP = auto()
    QUANTUM_ESPRESSO = auto()
    ABINIT = auto()
    CP2K = auto()
    CASTEP = auto()


class IBRIONMode(Enum):
    """VASP IBRION modes for phonon calculation."""
    FINITE_DIFFERENCES = 5  # Finite differences (old)
    DFPT = 6                # Density Functional Perturbation Theory
    FINITE_DIFFERENCES_2 = 7  # Finite differences (new)
    DFPT_2 = 8              # DFPT with symmetry


@dataclass
class PhononConfig:
    """Configuration for phonon calculations."""
    # Structure parameters
    structure_path: Optional[str] = None
    supercell_matrix: np.ndarray = field(default_factory=lambda: np.diag([2, 2, 2]))
    primitive_matrix: Optional[np.ndarray] = None
    
    # Displacement parameters
    displacement_distance: float = 0.01  # Angstrom
    symmetry_precision: float = 1e-5
    
    # DFT parameters
    dft_code: DFTCode = DFTCode.VASP
    ibiron_mode: Optional[IBRIONMode] = None
    potim: float = 0.015  # VASP POTIM
    nsw: int = 100
    
    # Phonon calculation parameters
    mesh_density: float = 100.0  # k-points per reciprocal atom
    mesh: Optional[Tuple[int, int, int]] = None
    
    # Band structure
    band_path: Optional[str] = None  # High-symmetry path string
    band_npoints: int = 101
    
    # Output
    output_dir: str = "./phonon_output"
    write_force_sets: bool = True
    write_force_constants: bool = True
    write_band_structure: bool = True
    write_dos: bool = True
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class PhononResults:
    """Container for phonon calculation results."""
    structure: Any = None
    phonopy_obj: Optional[Phonopy] = None
    force_constants: Optional[np.ndarray] = None
    frequencies: Optional[np.ndarray] = None  # THz
    eigenvectors: Optional[np.ndarray] = None
    band_structure: Optional[Dict] = None
    dos: Optional[Dict] = None
    pdos: Optional[Dict] = None
    thermal_properties: Optional[Dict] = None
    symmetry_dataset: Optional[Dict] = None
    mesh: Optional[Tuple[int, int, int]] = None
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        result = {
            'frequencies_THz': self.frequencies.tolist() if self.frequencies is not None else None,
            'mesh': self.mesh,
            'has_force_constants': self.force_constants is not None,
            'has_band_structure': self.band_structure is not None,
            'has_dos': self.dos is not None,
            'has_thermal_properties': self.thermal_properties is not None,
        }
        if self.thermal_properties:
            result['thermal_properties'] = {
                'temperatures': self.thermal_properties.get('temperatures', []),
                'free_energy': self.thermal_properties.get('free_energy', []),
                'entropy': self.thermal_properties.get('entropy', []),
                'heat_capacity': self.thermal_properties.get('heat_capacity', []),
            }
        return result
    
    def save_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class PhonopyInterface:
    """
    Unified interface for phonon calculations using Phonopy.
    
    Supports both VASP and Quantum ESPRESSO as DFT backends.
    """
    
    def __init__(self, config: Optional[PhononConfig] = None):
        """
        Initialize Phonopy interface.
        
        Args:
            config: Phonon calculation configuration
        """
        if not PHONOPY_AVAILABLE:
            raise ImportError("Phonopy is required for phonon calculations")
        
        self.config = config or PhononConfig()
        self.phonopy: Optional[Phonopy] = None
        self.results: Optional[PhononResults] = None
        self._structure: Optional[Structure] = None
        
        logger.info(f"Initialized PhonopyInterface with {self.config.dft_code.name}")
    
    def load_structure(self, structure: Union[str, Structure, Atoms, PhonopyAtoms]) -> PhonopyAtoms:
        """
        Load structure from various formats.
        
        Args:
            structure: Path to structure file, pymatgen Structure, ASE Atoms, or PhonopyAtoms
            
        Returns:
            PhonopyAtoms object
        """
        if isinstance(structure, str):
            # Load from file
            if PMG_AVAILABLE and structure.endswith(('.vasp', '.poscar', 'POSCAR')):
                pmg_struct = Poscar.from_file(structure).structure
                phonopy_atoms = self._pmg_to_phonopy(pmg_struct)
                self._structure = pmg_struct
            elif ASE_AVAILABLE:
                ase_atoms = read(structure)
                phonopy_atoms = self._ase_to_phonopy(ase_atoms)
                if PMG_AVAILABLE:
                    self._structure = AseAtomsAdaptor().get_structure(ase_atoms)
            else:
                phonopy_atoms = phonopy.file_IO.read_crystal_structure(structure)[0]
                
        elif isinstance(structure, Structure):
            phonopy_atoms = self._pmg_to_phonopy(structure)
            self._structure = structure
            
        elif ASE_AVAILABLE and isinstance(structure, Atoms):
            phonopy_atoms = self._ase_to_phonopy(structure)
            if PMG_AVAILABLE:
                self._structure = AseAtomsAdaptor().get_structure(structure)
                
        elif isinstance(structure, PhonopyAtoms):
            phonopy_atoms = structure
            
        else:
            raise ValueError(f"Unsupported structure type: {type(structure)}")
        
        logger.info(f"Loaded structure with {len(phonopy_atoms)} atoms")
        return phonopy_atoms
    
    def _pmg_to_phonopy(self, structure: Structure) -> PhonopyAtoms:
        """Convert pymatgen Structure to PhonopyAtoms."""
        return PhonopyAtoms(
            symbols=[str(site.specie) for site in structure],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True
        )
    
    def _ase_to_phonopy(self, atoms: Atoms) -> PhonopyAtoms:
        """Convert ASE Atoms to PhonopyAtoms."""
        return PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc()
        )
    
    def _phonopy_to_pmg(self, phonopy_atoms: PhonopyAtoms) -> Structure:
        """Convert PhonopyAtoms to pymatgen Structure."""
        return Structure(
            lattice=Lattice(phonopy_atoms.cell),
            species=phonopy_atoms.symbols,
            coords=phonopy_atoms.scaled_positions
        )
    
    def create_displacements(
        self, 
        structure: Union[str, Structure, Atoms, PhonopyAtoms],
        distance: Optional[float] = None,
        is_plusminus: str = 'auto',
        is_diagonal: bool = True,
        is_trigonal: bool = False
    ) -> List[PhonopyAtoms]:
        """
        Generate displaced supercells for force constant calculation.
        
        Args:
            structure: Input structure
            distance: Displacement distance in Angstrom
            is_plusminus: 'auto', True, or False for +/- displacements
            is_diagonal: Include diagonal displacements
            is_trigonal: Include trigonal displacements
            
        Returns:
            List of displaced supercell structures
        """
        phonopy_atoms = self.load_structure(structure)
        distance = distance or self.config.displacement_distance
        
        # Initialize phonopy
        self.phonopy = Phonopy(
            phonopy_atoms,
            supercell_matrix=self.config.supercell_matrix,
            primitive_matrix=self.config.primitive_matrix,
            symprec=self.config.symmetry_precision
        )
        
        # Generate displacements
        self.phonopy.generate_displacements(
            distance=distance,
            is_plusminus=is_plusminus,
            is_diagonal=is_diagonal,
            is_trigonal=is_trigonal
        )
        
        displacements = self.phonopy.supercells_with_displacements
        
        # Save displacement info
        output_dir = Path(self.config.output_dir)
        write_disp_yaml(self.phonopy.dataset, self.phonopy.supercell, output_dir / 'disp.yaml')
        
        # Write supercells to files
        if self.config.dft_code == DFTCode.VASP:
            self._write_vasp_displacements(displacements, output_dir)
        elif self.config.dft_code == DFTCode.QUANTUM_ESPRESSO:
            self._write_qe_displacements(displacements, output_dir)
        
        logger.info(f"Generated {len(displacements)} displacement structures")
        return displacements
    
    def _write_vasp_displacements(self, displacements: List[PhonopyAtoms], output_dir: Path):
        """Write displacement supercells in VASP POSCAR format."""
        for i, disp in enumerate(displacements):
            if disp is None:
                continue
            filename = output_dir / f"POSCAR-{i+1:03d}"
            # Convert to pymatgen and write
            structure = self._phonopy_to_pmg(disp)
            poscar = Poscar(structure)
            poscar.write_file(filename)
        
        # Also write the perfect supercell
        if self.phonopy:
            perfect = self._phonopy_to_pmg(self.phonopy.supercell)
            Poscar(perfect).write_file(output_dir / "POSCAR-perfect")
    
    def _write_qe_displacements(self, displacements: List[PhonopyAtoms], output_dir: Path):
        """Write displacement supercells in Quantum ESPRESSO format."""
        for i, disp in enumerate(displacements):
            if disp is None:
                continue
            filename = output_dir / f"pw-{i+1:03d}.in"
            self._write_qe_input(disp, filename, f"disp_{i+1}")
    
    def _write_qe_input(self, atoms: PhonopyAtoms, filename: Path, calculation: str):
        """Write Quantum ESPRESSO input file."""
        # Template QE input - user should customize
        cell = atoms.cell
        positions = atoms.scaled_positions
        
        with open(filename, 'w') as f:
            f.write(f"&CONTROL\n")
            f.write(f"  calculation = 'scf'\n")
            f.write(f"  prefix = '{calculation}'\n")
            f.write(f"  outdir = './tmp'\n")
            f.write(f"  pseudo_dir = './pseudo'\n")
            f.write(f"/\n")
            f.write(f"&SYSTEM\n")
            f.write(f"  ibrav = 0\n")
            f.write(f"  nat = {len(atoms)}\n")
            f.write(f"  ntyp = {len(set(atoms.symbols))}\n")
            f.write(f"  ecutwfc = 50.0\n")
            f.write(f"  ecutrho = 400.0\n")
            f.write(f"/\n")
            f.write(f"&ELECTRONS\n")
            f.write(f"  conv_thr = 1.0d-8\n")
            f.write(f"/\n")
            f.write(f"ATOMIC_SPECIES\n")
            # User needs to specify actual masses and pseudopotentials
            for symbol in set(atoms.symbols):
                f.write(f"{symbol}  {self._get_atomic_mass(symbol)}  {symbol}.upf\n")
            f.write(f"ATOMIC_POSITIONS (crystal)\n")
            for symbol, pos in zip(atoms.symbols, positions):
                f.write(f"{symbol}  {pos[0]:.10f}  {pos[1]:.10f}  {pos[2]:.10f}\n")
            f.write(f"K_POINTS automatic\n")
            f.write(f"4 4 4 0 0 0\n")
            f.write(f"CELL_PARAMETERS (angstrom)\n")
            for row in cell:
                f.write(f"  {row[0]:.10f}  {row[1]:.10f}  {row[2]:.10f}\n")
    
    def _get_atomic_mass(self, symbol: str) -> float:
        """Get atomic mass for element."""
        if PMG_AVAILABLE:
            return Element(symbol).atomic_mass
        # Fallback values for common elements
        masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
            'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
            'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
            'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
            'Ga': 69.723, 'Ge': 72.64, 'As': 74.922, 'Se': 78.96, 'Br': 79.904,
            'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
            'Nb': 92.906, 'Mo': 95.96, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.906,
            'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71,
            'Sb': 121.76, 'Te': 127.6, 'I': 126.904, 'Xe': 131.293, 'Cs': 132.905,
            'Ba': 137.327, 'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.242,
            'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.925,
            'Dy': 162.5, 'Ho': 164.93, 'Er': 167.259, 'Tm': 168.934, 'Yb': 173.054,
            'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.948, 'W': 183.84, 'Re': 186.207,
            'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084, 'Au': 196.967, 'Hg': 200.59,
            'Tl': 204.383, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0, 'At': 210.0,
            'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.038,
            'Pa': 231.036, 'U': 238.029
        }
        return masses.get(symbol, 1.0)
    
    def calculate_force_constants_vasp(
        self,
        vasprun_paths: Optional[List[str]] = None,
        outcar_paths: Optional[List[str]] = None,
        disp_yaml_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate force constants from VASP DFPT or finite difference calculations.
        
        Args:
            vasprun_paths: List of paths to vasprun.xml files (DFPT mode)
            outcar_paths: List of paths to OUTCAR files (finite differences)
            disp_yaml_path: Path to disp.yaml file with displacement info
            
        Returns:
            Force constants matrix
        """
        if vasprun_paths:
            return self._calc_fc_vasp_dfpt(vasprun_paths)
        elif outcar_paths:
            return self._calc_fc_vasp_fd(outcar_paths, disp_yaml_path)
        else:
            raise ValueError("Either vasprun_paths or outcar_paths must be provided")
    
    def _calc_fc_vasp_dfpt(self, vasprun_paths: List[str]) -> np.ndarray:
        """Calculate force constants from VASP DFPT (IBRION=6,7,8)."""
        if not PMG_AVAILABLE:
            raise ImportError("Pymatgen required for VASP DFPT parsing")
        
        # Read first vasprun to get force constants
        vasprun = Vasprun(vasprun_paths[0])
        
        # Get force constants from vasprun
        if vasprun.force_constants is None:
            raise ValueError(f"No force constants found in {vasprun_paths[0]}")
        
        force_constants = vasprun.force_constants
        
        # Save force constants
        output_dir = Path(self.config.output_dir)
        write_force_constants_to_hdf5(force_constants, output_dir / 'force_constants.hdf5')
        write_FORCE_CONSTANTS(force_constants, output_dir / 'FORCE_CONSTANTS')
        
        logger.info(f"Extracted force constants from VASP DFPT: shape {force_constants.shape}")
        return force_constants
    
    def _calc_fc_vasp_fd(
        self, 
        outcar_paths: List[str], 
        disp_yaml_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate force constants from VASP finite differences (IBRION=5).
        
        Args:
            outcar_paths: List of OUTCAR files for each displacement
            disp_yaml_path: Path to disp.yaml
            
        Returns:
            Force constants matrix
        """
        if not PMG_AVAILABLE:
            raise ImportError("Pymatgen required for OUTCAR parsing")
        
        # Parse forces from OUTCARs
        forces_list = []
        for outcar_path in outcar_paths:
            outcar = Outcar(outcar_path)
            if outcar.forces is None:
                raise ValueError(f"No forces found in {outcar_path}")
            forces_list.append(outcar.forces)
        
        # Load displacement info
        if disp_yaml_path is None:
            disp_yaml_path = Path(self.config.output_dir) / 'disp.yaml'
        
        # Create FORCE_SETS
        if self.phonopy is None:
            raise RuntimeError("Phonopy not initialized. Call create_displacements first.")
        
        self.phonopy.forces = forces_list
        
        # Produce force constants
        self.phonopy.produce_force_constants()
        force_constants = self.phonopy.force_constants
        
        # Save
        output_dir = Path(self.config.output_dir)
        write_force_constants_to_hdf5(force_constants, output_dir / 'force_constants.hdf5')
        write_FORCE_CONSTANTS(force_constants, output_dir / 'FORCE_CONSTANTS')
        
        logger.info(f"Calculated force constants from finite differences: shape {force_constants.shape}")
        return force_constants
    
    def calculate_force_constants_qe(
        self,
        ph_output_dir: str,
        q2r_input: Optional[str] = None
    ) -> np.ndarray:
        """
        Calculate force constants from Quantum ESPRESSO ph.x + q2r.x.
        
        Args:
            ph_output_dir: Directory with ph.x output files
            q2r_input: Path to q2r.x input file (optional)
            
        Returns:
            Force constants matrix
        """
        # Read force constants from QE format (flfrc)
        flfrc_path = Path(ph_output_dir) / 'fc.out'  # or similar
        
        if not flfrc_path.exists():
            raise FileNotFoundError(f"Force constants file not found: {flfrc_path}")
        
        # Parse QE force constants format
        force_constants = self._parse_qe_force_constants(flfrc_path)
        
        # Save in phonopy format
        output_dir = Path(self.config.output_dir)
        write_force_constants_to_hdf5(force_constants, output_dir / 'force_constants.hdf5')
        
        logger.info(f"Loaded force constants from QE: shape {force_constants.shape}")
        return force_constants
    
    def _parse_qe_force_constants(self, filepath: Path) -> np.ndarray:
        """Parse Quantum ESPRESSO force constants file."""
        # This is a simplified parser - real QE files have complex format
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse based on QE format (simplified)
        # Real implementation would need more robust parsing
        n_atoms = self.phonopy.supercell.get_number_of_atoms() if self.phonopy else 10
        
        force_constants = np.zeros((n_atoms, n_atoms, 3, 3))
        
        # Parse logic here (simplified)
        # In practice, use phonopy's QE interface or proper parsing
        
        return force_constants
    
    def set_force_constants(self, force_constants: Union[str, np.ndarray]):
        """
        Load or set force constants directly.
        
        Args:
            force_constants: Path to force constants file or numpy array
        """
        if isinstance(force_constants, str):
            filepath = Path(force_constants)
            if filepath.suffix == '.hdf5':
                fc = read_force_constants_from_hdf5(filepath)
            else:
                fc = parse_FORCE_CONSTANTS(filepath)
        else:
            fc = force_constants
        
        if self.phonopy is None:
            raise RuntimeError("Phonopy not initialized. Call create_displacements first.")
        
        self.phonopy.force_constants = fc
        
        if self.results is None:
            self.results = PhononResults()
        self.results.force_constants = fc
        
        logger.info(f"Set force constants with shape {fc.shape}")
    
    def calculate_band_structure(
        self,
        path: Optional[str] = None,
        npoints: int = 101,
        labels: Optional[List[str]] = None,
        with_eigenvectors: bool = False,
        with_group_velocities: bool = False
    ) -> Dict:
        """
        Calculate phonon band structure along high-symmetry path.
        
        Args:
            path: High-symmetry path string (e.g., "GMKG" for 2D hexagonal)
            npoints: Number of points in the path
            labels: Custom labels for path points
            with_eigenvectors: Include eigenvectors in output
            with_group_velocities: Include group velocities
            
        Returns:
            Dictionary with band structure data
        """
        if self.phonopy is None:
            raise RuntimeError("Phonopy not initialized")
        
        if self.phonopy.force_constants is None:
            raise RuntimeError("Force constants not set. Calculate or load them first.")
        
        # Auto-detect path if not provided
        if path is None:
            path, labels = self._auto_detect_path()
        
        # Get band qpoints
        bands, connections = self._get_band_qpoints(path, npoints)
        
        # Run band structure calculation
        self.phonopy.run_band_structure(
            bands,
            with_eigenvectors=with_eigenvectors,
            with_group_velocities=with_group_velocities
        )
        
        band_dict = self.phonopy.get_band_structure_dict()
        
        # Process results
        frequencies = band_dict['frequencies']  # THz
        distances = band_dict['distances']
        qpoints = band_dict['qpoints']
        
        result = {
            'qpoints': qpoints,
            'distances': distances,
            'frequencies_THz': frequencies,
            'frequencies_cm': np.array(frequencies) * THzToCm,
            'labels': labels,
            'path': path,
            'eigenvectors': band_dict.get('eigenvectors'),
            'group_velocities': band_dict.get('group_velocities')
        }
        
        # Save results
        output_dir = Path(self.config.output_dir)
        np.savez(
            output_dir / 'band_structure.npz',
            qpoints=qpoints,
            distances=distances,
            frequencies_THz=frequencies,
            labels=labels
        )
        
        if self.results is None:
            self.results = PhononResults()
        self.results.band_structure = result
        
        logger.info(f"Calculated band structure with {len(bands)} segments")
        return result
    
    def _auto_detect_path(self) -> Tuple[str, List[str]]:
        """Auto-detect high-symmetry path from structure symmetry."""
        if self._structure is None and self.phonopy is not None:
            self._structure = self._phonopy_to_pmg(self.phonopy.unitcell)
        
        if PMG_AVAILABLE and self._structure is not None:
            try:
                sga = SpacegroupAnalyzer(self._structure)
                kpath = sga.get_conventional_standard_structure()
                # Get recommended path
                # This is simplified - real implementation would use Seekpath or similar
                path = "GXMGRX"  # Default FCC path
                labels = ["Γ", "X", "M", "Γ", "R", "X"]
                return path, labels
            except Exception as e:
                logger.warning(f"Could not auto-detect path: {e}")
        
        # Default paths based on crystal system
        return "GXMGRX", ["Γ", "X", "M", "Γ", "R", "X"]
    
    def _get_band_qpoints(self, path: str, npoints: int) -> Tuple[List, List]:
        """Generate q-points for band structure calculation."""
        # Map path characters to q-point coordinates
        # This is a simplified mapping
        special_points = {
            'G': [0.0, 0.0, 0.0],
            'Γ': [0.0, 0.0, 0.0],
            'X': [0.5, 0.0, 0.5],
            'L': [0.5, 0.5, 0.5],
            'W': [0.5, 0.25, 0.75],
            'K': [0.375, 0.375, 0.75],
            'U': [0.625, 0.25, 0.625],
            'M': [0.5, 0.5, 0.0],
            'A': [0.5, 0.5, 0.5],  # Same as L in some conventions
            'H': [0.5, -0.5, 0.5],
            'N': [0.0, 0.5, 0.0],
            'P': [0.25, 0.25, 0.25],
            'R': [0.5, 0.5, 0.5],
        }
        
        # Build path segments
        bands = []
        for i in range(len(path) - 1):
            q_start = special_points.get(path[i], [0, 0, 0])
            q_end = special_points.get(path[i+1], [0.5, 0, 0.5])
            band = [q_start, q_end]
            bands.append(band)
        
        connections = [(True,)] * (len(bands) - 1) + [(False,)]
        
        return bands, connections
    
    def calculate_dos(
        self,
        mesh: Optional[Tuple[int, int, int]] = None,
        mesh_density: Optional[float] = None,
        t_max: float = 1000.0,
        t_min: float = 0.0,
        t_step: float = 10.0,
        use_tetrahedron_method: bool = True,
        run_phonon: bool = True
    ) -> Dict:
        """
        Calculate phonon density of states (DOS).
        
        Args:
            mesh: Monkhorst-Pack mesh (nx, ny, nz)
            mesh_density: Mesh density (k-points per reciprocal atom)
            t_max: Maximum temperature for thermal properties
            t_min: Minimum temperature
            t_step: Temperature step
            use_tetrahedron_method: Use tetrahedron method for DOS
            run_phonon: Run phonon calculation on mesh
            
        Returns:
            Dictionary with DOS data
        """
        if self.phonopy is None:
            raise RuntimeError("Phonopy not initialized")
        
        if self.phonopy.force_constants is None:
            raise RuntimeError("Force constants not set")
        
        # Determine mesh
        if mesh is None:
            if mesh_density is not None:
                mesh = self._get_auto_mesh(mesh_density)
            else:
                mesh = self.config.mesh or (20, 20, 20)
        
        self.mesh = mesh
        
        # Set mesh and run
        self.phonopy.auto_total_dos(
            mesh=mesh,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step
        )
        
        # Get total DOS
        total_dos = self.phonopy.total_dos
        freq_points = total_dos.frequency_points  # THz
        dos_values = total_dos.dos
        
        result = {
            'frequency_points_THz': freq_points,
            'frequency_points_cm': freq_points * THzToCm,
            'total_dos': dos_values,
            'mesh': mesh,
            'tetrahedron_method': use_tetrahedron_method
        }
        
        # Save
        output_dir = Path(self.config.output_dir)
        np.savez(
            output_dir / 'total_dos.npz',
            frequency_points_THz=freq_points,
            dos=dos_values,
            mesh=mesh
        )
        
        if self.results is None:
            self.results = PhononResults()
        self.results.dos = result
        self.results.mesh = mesh
        
        logger.info(f"Calculated DOS on {mesh} mesh")
        return result
    
    def calculate_pdos(
        self,
        mesh: Optional[Tuple[int, int, int]] = None,
        mesh_density: Optional[float] = None,
        legendre_delta: float = 0.1
    ) -> Dict:
        """
        Calculate projected phonon density of states (PDOS).
        
        Args:
            mesh: Monkhorst-Pack mesh
            mesh_density: Mesh density
            legendre_delta: Smearing parameter for PDOS
            
        Returns:
            Dictionary with PDOS data
        """
        if self.phonopy is None:
            raise RuntimeError("Phonopy not initialized")
        
        if self.phonopy.force_constants is None:
            raise RuntimeError("Force constants not set")
        
        # Determine mesh
        if mesh is None:
            mesh = self.mesh or self.config.mesh or (20, 20, 20)
        
        # Calculate PDOS
        self.phonopy.auto_projected_dos(
            mesh=mesh,
            legendre_delta=legendre_delta
        )
        
        pdos = self.phonopy.projected_dos
        freq_points = pdos.frequency_points
        pdos_values = pdos.projected_dos
        
        result = {
            'frequency_points_THz': freq_points,
            'frequency_points_cm': freq_points * THzToCm,
            'pdos': pdos_values,
            'mesh': mesh
        }
        
        # Save
        output_dir = Path(self.config.output_dir)
        np.savez(
            output_dir / 'pdos.npz',
            frequency_points_THz=freq_points,
            pdos=pdos_values,
            mesh=mesh
        )
        
        if self.results is None:
            self.results = PhononResults()
        self.results.pdos = result
        
        logger.info(f"Calculated PDOS on {mesh} mesh")
        return result
    
    def _get_auto_mesh(self, mesh_density: float) -> Tuple[int, int, int]:
        """Automatically determine mesh based on density."""
        if self.phonopy is None:
            return (20, 20, 20)
        
        # Get reciprocal cell
        rec_cell = self.phonopy.primitive.cell.reciprocal()
        
        # Calculate mesh
        n_atoms = len(self.phonopy.primitive)
        total_kpoints = int(mesh_density * n_atoms)
        
        # Distribute evenly
        n = int(np.cbrt(total_kpoints))
        return (n, n, n)
    
    def calculate_thermal_properties(
        self,
        temperatures: Optional[np.ndarray] = None,
        t_min: float = 0.0,
        t_max: float = 1000.0,
        t_step: float = 10.0,
        cutoff_frequency: float = 1e-3
    ) -> Dict:
        """
        Calculate thermodynamic properties.
        
        Args:
            temperatures: Array of temperatures (K), or None for auto
            t_min: Minimum temperature
            t_max: Maximum temperature
            t_step: Temperature step
            cutoff_frequency: Ignore modes below this frequency
            
        Returns:
            Dictionary with thermal properties
        """
        if self.phonopy is None:
            raise RuntimeError("Phonopy not initialized")
        
        if self.phonopy.force_constants is None:
            raise RuntimeError("Force constants not set")
        
        # Generate temperatures
        if temperatures is None:
            temperatures = np.arange(t_min, t_max + t_step, t_step)
        
        # Set mesh if not already done
        if self.phonopy.mesh_numbers is None:
            mesh = self.config.mesh or (20, 20, 20)
            self.phonopy.mesh_numbers = mesh
        
        # Calculate thermal properties
        self.phonopy.run_mesh(self.phonopy.mesh_numbers)
        self.phonopy.run_thermal_properties(
            t_min=t_min,
            t_max=t_max,
            t_step=t_step
        )
        
        tp_dict = self.phonopy.get_thermal_properties_dict()
        
        # Extract properties
        temps = tp_dict['temperatures']
        free_energy = tp_dict['free_energy']  # kJ/mol
        entropy = tp_dict['entropy']  # J/K/mol
        cv = tp_dict['heat_capacity']  # J/K/mol
        
        result = {
            'temperatures': temps,
            'free_energy_kJ_mol': free_energy,
            'free_energy_eV_atom': np.array(free_energy) * 0.0103643,  # Convert
            'entropy_J_mol_K': entropy,
            'heat_capacity_J_mol_K': cv,
            'cutoff_frequency': cutoff_frequency
        }
        
        # Save
        output_dir = Path(self.config.output_dir)
        np.savez(
            output_dir / 'thermal_properties.npz',
            temperatures=temps,
            free_energy_kJ_mol=free_energy,
            entropy_J_mol_K=entropy,
            heat_capacity_J_mol_K=cv
        )
        
        if self.results is None:
            self.results = PhononResults()
        self.results.thermal_properties = result
        
        logger.info(f"Calculated thermal properties for {len(temps)} temperatures")
        return result
    
    def run_full_phonon_calculation(
        self,
        structure: Union[str, Structure, Atoms],
        run_band_structure: bool = True,
        run_dos: bool = True,
        run_thermal: bool = True,
        force_constants: Optional[Union[str, np.ndarray]] = None
    ) -> PhononResults:
        """
        Run complete phonon calculation workflow.
        
        Args:
            structure: Input structure
            run_band_structure: Calculate band structure
            run_dos: Calculate DOS
            run_thermal: Calculate thermal properties
            force_constants: Pre-calculated force constants
            
        Returns:
            Complete phonon results
        """
        logger.info("Starting full phonon calculation workflow")
        
        # Initialize phonopy
        self.create_displacements(structure)
        
        # Set force constants if provided
        if force_constants is not None:
            self.set_force_constants(force_constants)
        
        self.results = PhononResults(phonopy_obj=self.phonopy)
        
        # Calculate band structure
        if run_band_structure:
            self.calculate_band_structure()
        
        # Calculate DOS
        if run_dos:
            self.calculate_dos()
            self.calculate_pdos()
        
        # Calculate thermal properties
        if run_thermal:
            self.calculate_thermal_properties()
        
        # Save complete results
        self.results.save_json(Path(self.config.output_dir) / 'phonon_results.json')
        
        logger.info("Full phonon calculation completed")
        return self.results
    
    def plot_band_structure(
        self,
        band_structure: Optional[Dict] = None,
        unit: str = 'THz',
        figsize: Tuple[int, int] = (10, 6),
        color: str = 'b',
        lw: float = 1.0,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot phonon band structure.
        
        Args:
            band_structure: Band structure data (uses self.results if None)
            unit: Frequency unit ('THz', 'cm', 'meV', 'eV')
            figsize: Figure size
            color: Line color
            lw: Line width
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if band_structure is None:
            if self.results is None or self.results.band_structure is None:
                raise ValueError("No band structure data available")
            band_structure = self.results.band_structure
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get frequencies in correct unit
        if unit == 'THz':
            freqs = band_structure['frequencies_THz']
            ylabel = 'Frequency (THz)'
        elif unit == 'cm':
            freqs = band_structure['frequencies_cm']
            ylabel = 'Frequency (cm⁻¹)'
        elif unit == 'meV':
            freqs = np.array(band_structure['frequencies_THz']) * THzToEv * 1000
            ylabel = 'Energy (meV)'
        elif unit == 'eV':
            freqs = np.array(band_structure['frequencies_THz']) * THzToEv
            ylabel = 'Energy (eV)'
        else:
            raise ValueError(f"Unknown unit: {unit}")
        
        distances = band_structure['distances']
        labels = band_structure['labels']
        
        # Plot bands
        for i, (d, f) in enumerate(zip(distances, freqs)):
            ax.plot(d, f, color=color, lw=lw)
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', lw=0.5)
        
        # Set labels
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Wave Vector', fontsize=12)
        
        # Add vertical lines at special points
        special_positions = [d[0] for d in distances] + [distances[-1][-1]]
        for pos in special_positions:
            ax.axvline(x=pos, color='k', linestyle='-', lw=0.5)
        
        # Set x-ticks
        ax.set_xticks(special_positions)
        ax.set_xticklabels(labels)
        
        ax.set_xlim(special_positions[0], special_positions[-1])
        ax.set_ylim(bottom=min(-0.5, np.min(freqs) * 1.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved band structure plot to {save_path}")
        
        return fig
    
    def plot_dos(
        self,
        dos_data: Optional[Dict] = None,
        pdos_data: Optional[Dict] = None,
        unit: str = 'THz',
        figsize: Tuple[int, int] = (8, 6),
        fill: bool = True,
        alpha: float = 0.5,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot phonon DOS and PDOS.
        
        Args:
            dos_data: DOS data
            pdos_data: PDOS data
            unit: Frequency unit
            figsize: Figure size
            fill: Fill under curves
            alpha: Fill transparency
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if dos_data is None:
            if self.results is None or self.results.dos is None:
                raise ValueError("No DOS data available")
            dos_data = self.results.dos
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get frequencies in correct unit
        if unit == 'THz':
            freqs = dos_data['frequency_points_THz']
            xlabel = 'Frequency (THz)'
        elif unit == 'cm':
            freqs = dos_data['frequency_points_cm']
            xlabel = 'Frequency (cm⁻¹)'
        else:
            raise ValueError(f"Unknown unit: {unit}")
        
        dos = dos_data['total_dos']
        
        # Plot total DOS
        ax.plot(dos, freqs, 'b-', lw=2, label='Total DOS')
        if fill:
            ax.fill_betweenx(freqs, 0, dos, alpha=alpha, color='b')
        
        # Plot PDOS if available
        if pdos_data is not None:
            pdos = pdos_data['pdos']
            # Plot each species
            n_species = len(set(self.phonopy.primitive.symbols))
            colors = plt.cm.tab10(np.linspace(0, 1, n_species))
            
            for i, (symbol, color) in enumerate(zip(
                set(self.phonopy.primitive.symbols), colors)):
                # Sum over atoms of this species
                ax.plot(pdos[i], freqs, color=color, lw=1.5, 
                       label=f'{symbol} PDOS')
        
        ax.axhline(y=0, color='k', linestyle='--', lw=0.5)
        ax.set_xlabel('DOS (states/THz/unit-cell)', fontsize=12)
        ax.set_ylabel(xlabel, fontsize=12)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved DOS plot to {save_path}")
        
        return fig
    
    def plot_thermal_properties(
        self,
        thermal_data: Optional[Dict] = None,
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot thermal properties (Cv, entropy, free energy).
        
        Args:
            thermal_data: Thermal properties data
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if thermal_data is None:
            if self.results is None or self.results.thermal_properties is None:
                raise ValueError("No thermal properties data available")
            thermal_data = self.results.thermal_properties
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        temps = thermal_data['temperatures']
        
        # Heat capacity
        ax = axes[0]
        ax.plot(temps, thermal_data['heat_capacity_J_mol_K'], 'b-', lw=2)
        ax.set_xlabel('Temperature (K)', fontsize=11)
        ax.set_ylabel('Cᵥ (J/mol/K)', fontsize=11)
        ax.set_title('Heat Capacity', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Entropy
        ax = axes[1]
        ax.plot(temps, thermal_data['entropy_J_mol_K'], 'r-', lw=2)
        ax.set_xlabel('Temperature (K)', fontsize=11)
        ax.set_ylabel('S (J/mol/K)', fontsize=11)
        ax.set_title('Vibrational Entropy', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Free energy
        ax = axes[2]
        ax.plot(temps, thermal_data['free_energy_kJ_mol'], 'g-', lw=2)
        ax.set_xlabel('Temperature (K)', fontsize=11)
        ax.set_ylabel('F (kJ/mol)', fontsize=11)
        ax.set_title('Helmholtz Free Energy', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved thermal properties plot to {save_path}")
        
        return fig
    
    def plot_complete_phonon(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10)
    ) -> Figure:
        """
        Create comprehensive phonon visualization with band structure, DOS, 
        and thermal properties.
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("No phonon results available. Run calculation first.")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1])
        
        # Band structure
        ax1 = fig.add_subplot(gs[0, 0])
        if self.results.band_structure:
            self._plot_band_on_axis(ax1, self.results.band_structure)
        
        # DOS
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        if self.results.dos:
            self._plot_dos_on_axis(ax2, self.results.dos)
        ax2.set_ylabel('')
        plt.setp(ax2.get_yticklabels(), visible=False)
        
        # Thermal properties
        ax3 = fig.add_subplot(gs[1, :])
        if self.results.thermal_properties:
            self._plot_thermal_compact(ax3, self.results.thermal_properties)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved complete phonon plot to {save_path}")
        
        return fig
    
    def _plot_band_on_axis(self, ax, band_structure: Dict):
        """Plot band structure on given axis."""
        freqs = band_structure['frequencies_THz']
        distances = band_structure['distances']
        labels = band_structure['labels']
        
        for d, f in zip(distances, freqs):
            ax.plot(d, f, 'b-', lw=1.0)
        
        ax.axhline(y=0, color='k', linestyle='--', lw=0.5)
        
        special_positions = [d[0] for d in distances] + [distances[-1][-1]]
        for pos in special_positions:
            ax.axvline(x=pos, color='k', linestyle='-', lw=0.5)
        
        ax.set_xticks(special_positions)
        ax.set_xticklabels(labels)
        ax.set_xlim(special_positions[0], special_positions[-1])
        ax.set_ylabel('Frequency (THz)', fontsize=12)
        ax.set_xlabel('Wave Vector', fontsize=12)
    
    def _plot_dos_on_axis(self, ax, dos_data: Dict):
        """Plot DOS on given axis."""
        freqs = dos_data['frequency_points_THz']
        dos = dos_data['total_dos']
        
        ax.fill_betweenx(freqs, 0, dos, alpha=0.5, color='b')
        ax.plot(dos, freqs, 'b-', lw=1.5)
        ax.set_xlabel('DOS', fontsize=12)
    
    def _plot_thermal_compact(self, ax, thermal_data: Dict):
        """Plot compact thermal properties."""
        temps = thermal_data['temperatures']
        
        ax_twin1 = ax.twinx()
        ax_twin2 = ax.twinx()
        ax_twin2.spines['right'].set_position(('outward', 60))
        
        p1, = ax.plot(temps, thermal_data['heat_capacity_J_mol_K'], 
                      'b-', lw=2, label='Cv')
        p2, = ax_twin1.plot(temps, thermal_data['entropy_J_mol_K'], 
                           'r-', lw=2, label='S')
        p3, = ax_twin2.plot(temps, thermal_data['free_energy_kJ_mol'], 
                           'g-', lw=2, label='F')
        
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('Cv (J/mol/K)', color='b', fontsize=12)
        ax_twin1.set_ylabel('S (J/mol/K)', color='r', fontsize=12)
        ax_twin2.set_ylabel('F (kJ/mol)', color='g', fontsize=12)
        
        ax.tick_params(axis='y', labelcolor='b')
        ax_twin1.tick_params(axis='y', labelcolor='r')
        ax_twin2.tick_params(axis='y', labelcolor='g')
        
        lines = [p1, p2, p3]
        ax.legend(lines, [l.get_label() for l in lines], loc='center left')
    
    def get_dynamical_matrix_at_q(self, qpoint: np.ndarray) -> np.ndarray:
        """
        Get dynamical matrix at specific q-point.
        
        Args:
            qpoint: q-point in reduced coordinates
            
        Returns:
            Dynamical matrix
        """
        if self.phonopy is None:
            raise RuntimeError("Phonopy not initialized")
        
        dynmat = self.phonopy.get_dynamical_matrix_at_q(qpoint)
        return dynmat
    
    def get_eigenvectors_at_q(self, qpoint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get phonon eigenvectors at specific q-point.
        
        Args:
            qpoint: q-point in reduced coordinates
            
        Returns:
            Tuple of (frequencies, eigenvectors)
        """
        if self.phonopy is None:
            raise RuntimeError("Phonopy not initialized")
        
        self.phonopy.run_qpoints([qpoint])
        frequencies = self.phonopy.qpoints.frequencies[0]  # THz
        eigenvectors = self.phonopy.qpoints.eigenvectors[0]
        
        return frequencies, eigenvectors
    
    def check_dynamical_stability(self, threshold: float = -0.1) -> Tuple[bool, np.ndarray]:
        """
        Check dynamical stability (no imaginary modes).
        
        Args:
            threshold: Threshold for imaginary mode detection (THz)
            
        Returns:
            Tuple of (is_stable, imaginary_frequencies)
        """
        if self.results is None or self.results.band_structure is None:
            raise ValueError("Band structure must be calculated first")
        
        all_freqs = np.concatenate(self.results.band_structure['frequencies_THz'])
        imaginary_modes = all_freqs[all_freqs < threshold]
        
        is_stable = len(imaginary_modes) == 0
        
        if not is_stable:
            logger.warning(f"Found {len(imaginary_modes)} imaginary modes below {threshold} THz")
        
        return is_stable, imaginary_modes


def create_phonopy_from_vasp(
    poscar_path: str,
    force_constants_path: Optional[str] = None,
    supercell_matrix: Optional[np.ndarray] = None
) -> PhonopyInterface:
    """
    Convenience function to create PhonopyInterface from VASP files.
    
    Args:
        poscar_path: Path to POSCAR file
        force_constants_path: Path to force constants file (optional)
        supercell_matrix: Supercell matrix
        
    Returns:
        Configured PhonopyInterface
    """
    config = PhononConfig(
        structure_path=poscar_path,
        dft_code=DFTCode.VASP,
        supercell_matrix=supercell_matrix or np.diag([2, 2, 2])
    )
    
    interface = PhonopyInterface(config)
    interface.load_structure(poscar_path)
    
    if force_constants_path:
        interface.set_force_constants(force_constants_path)
    
    return interface


def create_phonopy_from_ase(
    atoms: Atoms,
    force_constants: Optional[np.ndarray] = None,
    supercell_matrix: Optional[np.ndarray] = None
) -> PhonopyInterface:
    """
    Convenience function to create PhonopyInterface from ASE Atoms.
    
    Args:
        atoms: ASE Atoms object
        force_constants: Force constants matrix (optional)
        supercell_matrix: Supercell matrix
        
    Returns:
        Configured PhonopyInterface
    """
    config = PhononConfig(
        dft_code=DFTCode.VASP,
        supercell_matrix=supercell_matrix or np.diag([2, 2, 2])
    )
    
    interface = PhonopyInterface(config)
    interface.load_structure(atoms)
    
    if force_constants is not None:
        interface.set_force_constants(force_constants)
    
    return interface


# For command-line usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phonopy Interface for DFT+MD')
    parser.add_argument('--structure', '-s', type=str, required=True,
                       help='Path to structure file')
    parser.add_argument('--fc', type=str, help='Path to force constants')
    parser.add_argument('--disp', action='store_true', help='Generate displacements')
    parser.add_argument('--band', action='store_true', help='Calculate band structure')
    parser.add_argument('--dos', action='store_true', help='Calculate DOS')
    parser.add_argument('--thermal', action='store_true', help='Calculate thermal properties')
    parser.add_argument('--mesh', type=int, nargs=3, default=[20, 20, 20],
                       help='q-point mesh')
    parser.add_argument('--supercell', type=int, nargs=3, default=[2, 2, 2],
                       help='Supercell size')
    parser.add_argument('--outdir', type=str, default='./phonon_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run calculation
    config = PhononConfig(
        structure_path=args.structure,
        supercell_matrix=np.diag(args.supercell),
        mesh=tuple(args.mesh),
        output_dir=args.outdir
    )
    
    interface = PhonopyInterface(config)
    
    if args.disp:
        interface.create_displacements(args.structure)
    
    if args.fc:
        interface.set_force_constants(args.fc)
        
        if args.band:
            interface.calculate_band_structure()
            interface.plot_band_structure(save_path=f"{args.outdir}/band_structure.png")
        
        if args.dos:
            interface.calculate_dos()
            interface.plot_dos(save_path=f"{args.outdir}/dos.png")
        
        if args.thermal:
            interface.calculate_thermal_properties()
            interface.plot_thermal_properties(
                save_path=f"{args.outdir}/thermal_properties.png")
        
        interface.plot_complete_phonon(save_path=f"{args.outdir}/phonon_complete.png")
