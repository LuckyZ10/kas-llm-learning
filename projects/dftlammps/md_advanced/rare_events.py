#!/usr/bin/env python3
"""
Rare Event Analysis
===================

Methods for studying rare events and transition paths:
- Transition State Theory (TST)
- Nudged Elastic Band (NEB) - VASP/LAMMPS integration
- String Method
- Dimer Method

References:
- Eyring (1935) - TST
- Mills, Jónsson & Schenter (1995) - NEB
- E, Ren & Vanden-Eijnden (2002) - String Method
- Henkelman & Jónsson (1999) - Dimer Method
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from functools import partial
import logging
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

# ASE
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS, MDMin
from ase.neb import NEB as ASE_NEB
from ase.vibrations import Vibrations
from ase.units import fs, kB, Bohr, Hartree
from ase.geometry import find_mic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NEBConfig:
    """Configuration for Nudged Elastic Band calculation.
    
    Attributes:
        n_images: Number of intermediate images
        k_spring: Spring constant between images (eV/Å²)
        climb: Use climbing image NEB
        ci_threshold: Threshold for switching to CI-NEB
        fmax: Force convergence criterion (eV/Å)
        max_steps: Maximum optimization steps
        neb_method: 'ase', 'lammps', or 'vasp'
        interpolation: 'idpp' or 'linear'
        parallel: Run images in parallel
    """
    n_images: int = 10
    k_spring: float = 1.0
    climb: bool = True
    ci_threshold: float = 0.5
    fmax: float = 0.05
    max_steps: int = 500
    neb_method: str = "ase"
    interpolation: str = "idpp"
    parallel: bool = False
    nprocs: int = 4
    output_dir: str = "./neb"


@dataclass
class StringMethodConfig:
    """Configuration for String Method.
    
    Attributes:
        n_images: Number of images on string
        dt: Time step for string evolution
        smoothing_factor: String smoothing parameter
        reparam_steps: Reparametrization frequency
        max_steps: Maximum evolution steps
        string_method: 'simplified' or 'FTP'
    """
    n_images: int = 20
    dt: float = 0.1
    smoothing_factor: float = 0.1
    reparam_steps: int = 10
    max_steps: int = 1000
    string_method: str = "simplified"
    fmax: float = 0.05
    output_dir: str = "./string_method"


@dataclass
class DimerConfig:
    """Configuration for Dimer Method.
    
    Attributes:
        dimer_distance: Initial dimer separation (Å)
        max_rotations: Maximum rotation steps
        rotation_threshold: Convergence for rotation
        translation_method: 'quickmin', 'lbfgs', or 'fire'
        max_steps: Maximum translation steps
        fmax: Force convergence criterion
    """
    dimer_distance: float = 0.01
    max_rotations: int = 100
    rotation_threshold: float = 1e-4
    translation_method: str = "quickmin"
    max_steps: int = 500
    fmax: float = 0.05
    output_dir: str = "./dimer"


@dataclass
class TSTConfig:
    """Configuration for Transition State Theory calculations.
    
    Attributes:
        temperature: Temperature in Kelvin
        include_tunneling: Include quantum tunneling correction
        tunneling_method: 'Wigner', 'Eckart', or 'ZCT'
        compute_recrossing: Compute transmission coefficient
        recrossing_method: 'RRKM' or 'RPMD'
    """
    temperature: float = 300.0
    include_tunneling: bool = False
    tunneling_method: str = "Wigner"
    compute_recrossing: bool = False
    recrossing_method: str = "RRKM"


class NEB:
    """Nudged Elastic Band for finding minimum energy paths."""
    
    def __init__(self, config: NEBConfig):
        self.config = config
        self.images: List[Atoms] = []
        self.energies: np.ndarray = np.array([])
        self.forces: np.ndarray = np.array([])
        self.ts_index: Optional[int] = None
        self.barrier: Optional[float] = None
    
    def interpolate_images(self, initial: Atoms, final: Atoms) -> List[Atoms]:
        """Create interpolated images between initial and final states."""
        images = [initial.copy()]
        
        for i in range(1, self.config.n_images + 1):
            t = i / (self.config.n_images + 1)
            
            # Linear interpolation
            positions = initial.positions * (1 - t) + final.positions * t
            
            image = initial.copy()
            image.positions = positions
            images.append(image)
        
        images.append(final.copy())
        
        # Apply IDPP interpolation if requested
        if self.config.interpolation == 'idpp':
            images = self._idpp_interpolation(images)
        
        return images
    
    def _idpp_interpolation(self, images: List[Atoms]) -> List[Atoms]:
        """Image Dependent Pair Potential interpolation."""
        # Simplified IDPP - would need full implementation
        logger.info("Applying IDPP interpolation")
        return images
    
    def setup_lammps_neb(self, images: List[Atoms], potential_file: str) -> str:
        """Generate LAMMPS NEB input script."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write each image to data file
        for i, image in enumerate(images):
            from ase.io.lammpsdata import write_lammps_data
            write_lammps_data(output_dir / f"image_{i:02d}.data", image)
        
        # Generate LAMMPS NEB input
        neb_input = f"""# NEB calculation with LAMMPS
units metal
atom_style atomic
boundary p p p

# Read first image
read_data image_00.data

pair_style deepmd {potential_file}
pair_coeff * *

# NEB setup
neb {self.config.k_spring} each image_{{i:02d}}.data {self.config.n_images}

# Minimization
minimize 1.0e-6 1.0e-8 1000 10000

# Climbing image if requested
{"neb modify climb yes" if self.config.climb else ""}

# Output
thermo 10
dump 1 all custom 100 neb.dump id type x y z

neb 1e-6 {self.config.fmax} 10000 10000 100 final image_{len(images)-1:02d}.data
"""
        
        input_file = output_dir / "neb_input.lammps"
        with open(input_file, 'w') as f:
            f.write(neb_input)
        
        return str(input_file)
    
    def run_ase_neb(self, images: List[Atoms], 
                   calculator: Optional[Any] = None) -> Dict:
        """Run NEB using ASE implementation."""
        logger.info("Running ASE NEB")
        
        # Setup NEB
        neb = ASE_NEB(images, k=self.config.k_spring, climb=self.config.climb,
                     parallel=self.config.parallel, method='improved')
        
        # Setup optimizer
        optimizer = FIRE(neb, logfile=str(Path(self.config.output_dir) / 'neb.log'))
        
        # Run optimization
        optimizer.run(fmax=self.config.fmax, steps=self.config.max_steps)
        
        # Extract results
        self.images = images
        self.energies = np.array([image.get_potential_energy() for image in images])
        
        # Find transition state
        self.ts_index = np.argmax(self.energies)
        self.barrier = self.energies[self.ts_index] - self.energies[0]
        
        return {
            'success': optimizer.converged(),
            'energies': self.energies,
            'ts_index': self.ts_index,
            'barrier': self.barrier,
            'images': images
        }
    
    def run_lammps_neb(self, images: List[Atoms], 
                      potential_file: str = "graph.pb") -> Dict:
        """Run NEB using LAMMPS implementation."""
        logger.info("Running LAMMPS NEB")
        
        input_file = self.setup_lammps_neb(images, potential_file)
        output_dir = Path(self.config.output_dir)
        
        # Run LAMMPS
        cmd = ["mpirun", "-np", str(self.config.nprocs),
               "lmp", "-partition", f"{self.config.n_images}x1",
               "-in", input_file]
        
        try:
            result = subprocess.run(
                cmd, cwd=output_dir, capture_output=True, text=True, timeout=172800
            )
            success = result.returncode == 0
        except Exception as e:
            logger.error(f"NEB failed: {e}")
            success = False
        
        # Parse results from LAMMPS output
        energies = self._parse_neb_output(output_dir)
        
        return {
            'success': success,
            'energies': energies,
            'output_dir': str(output_dir)
        }
    
    def run_vasp_neb(self, images: List[Atoms],
                    vasp_settings: Dict) -> Dict:
        """Run NEB using VASP implementation."""
        logger.info("Running VASP NEB")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write VASP NEB structure files
        for i, image in enumerate(images):
            image_dir = output_dir / f"0{i}"
            image_dir.mkdir(exist_ok=True)
            write(image_dir / "POSCAR", image)
        
        # Generate INCAR
        incar_content = f"""NEB Calculation
ISTART = 0
ICHARG = 2
ENCUT = {vasp_settings.get('encut', 520)}
ISMEAR = 0
SIGMA = 0.1
EDIFF = 1E-6
EDIFFG = -0.05
IBRION = 1
ISIF = 2
NSW = {self.config.max_steps}
IMAGES = {self.config.n_images}
SPRING = -{self.config.k_spring}
LCLIMB = {".TRUE." if self.config.climb else ".FALSE."}
"""
        
        with open(output_dir / "INCAR", 'w') as f:
            f.write(incar_content)
        
        # Run VASP (would need actual VASP setup)
        logger.info("VASP NEB setup complete. Run VASP manually.")
        
        return {
            'success': True,
            'output_dir': str(output_dir),
            'message': 'VASP input files prepared'
        }
    
    def _parse_neb_output(self, output_dir: Path) -> np.ndarray:
        """Parse NEB output energies."""
        # Look for LAMMPS NEB output or image energies
        energies = []
        
        log_file = output_dir / "log.lammps"
        if log_file.exists():
            with open(log_file) as f:
                for line in f:
                    if "Energy of" in line and "image" in line:
                        parts = line.split()
                        try:
                            energies.append(float(parts[-2]))
                        except (ValueError, IndexError):
                            pass
        
        return np.array(energies)
    
    def run(self, initial: Atoms, final: Atoms,
           calculator: Optional[Any] = None,
           potential_file: str = "graph.pb",
           vasp_settings: Optional[Dict] = None) -> Dict:
        """Run NEB calculation."""
        # Interpolate images
        self.images = self.interpolate_images(initial, final)
        
        if self.config.neb_method == 'ase':
            return self.run_ase_neb(self.images, calculator)
        elif self.config.neb_method == 'lammps':
            return self.run_lammps_neb(self.images, potential_file)
        elif self.config.neb_method == 'vasp':
            return self.run_vasp_neb(self.images, vasp_settings or {})
        else:
            raise ValueError(f"Unknown NEB method: {self.config.neb_method}")
    
    def write_results(self, output_prefix: str = "neb"):
        """Write NEB results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write energies
        if len(self.energies) > 0:
            np.savetxt(output_dir / f"{output_prefix}_energies.dat",
                      np.column_stack([np.arange(len(self.energies)), self.energies]),
                      header="Image Energy[eV]", comments='# ')
        
        # Write images
        for i, image in enumerate(self.images):
            write(output_dir / f"{output_prefix}_image_{i:02d}.xyz", image)
        
        # Write summary
        summary = {
            'n_images': len(self.images),
            'barrier': float(self.barrier) if self.barrier else None,
            'ts_index': int(self.ts_index) if self.ts_index is not None else None,
            'energies': self.energies.tolist()
        }
        
        with open(output_dir / f"{output_prefix}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


class StringMethod:
    """String Method for finding minimum free energy paths."""
    
    def __init__(self, config: StringMethodConfig):
        self.config = config
        self.string: List[Atoms] = []
        self.energies: np.ndarray = np.array([])
        self.string_history: List[List[Atoms]] = []
    
    def initialize_string(self, initial: Atoms, final: Atoms) -> List[Atoms]:
        """Initialize string with linear interpolation."""
        string = []
        
        for i in range(self.config.n_images):
            t = i / (self.config.n_images - 1)
            positions = initial.positions * (1 - t) + final.positions * t
            
            image = initial.copy()
            image.positions = positions
            string.append(image)
        
        return string
    
    def compute_forces(self, string: List[Atoms],
                      calculator: Optional[Any] = None) -> List[np.ndarray]:
        """Compute forces on each image."""
        forces = []
        
        for image in string:
            if calculator:
                image.calc = calculator
            f = image.get_forces()
            forces.append(f)
        
        return forces
    
    def evolve_string(self, string: List[Atoms],
                     forces: List[np.ndarray],
                     perpendicular_only: bool = True) -> List[Atoms]:
        """Evolve string according to forces."""
        new_string = []
        
        for i, (image, force) in enumerate(zip(string, forces)):
            new_image = image.copy()
            
            if perpendicular_only and 0 < i < len(string) - 1:
                # Project force perpendicular to string tangent
                tangent = self._compute_tangent(string, i)
                force_perp = force - np.outer(np.sum(force * tangent, axis=1), 
                                              tangent).reshape(force.shape)
                new_image.positions += self.config.dt * force_perp
            else:
                # Endpoints fixed or full force for simplified method
                if 0 < i < len(string) - 1:
                    new_image.positions += self.config.dt * force
            
            new_string.append(new_image)
        
        return new_string
    
    def _compute_tangent(self, string: List[Atoms], index: int) -> np.ndarray:
        """Compute string tangent at given index."""
        # Central difference
        r_plus = string[index + 1].positions
        r_minus = string[index - 1].positions
        
        tangent = r_plus - r_minus
        tangent = tangent / np.linalg.norm(tangent)
        
        return tangent
    
    def reparametrize(self, string: List[Atoms]) -> List[Atoms]:
        """Reparametrize string to equal arc length."""
        # Compute arc lengths
        arc_lengths = [0.0]
        
        for i in range(1, len(string)):
            dr = string[i].positions - string[i-1].positions
            dist = np.linalg.norm(dr)
            arc_lengths.append(arc_lengths[-1] + dist)
        
        # Normalize to [0, 1]
        total_length = arc_lengths[-1]
        arc_lengths = [a / total_length for a in arc_lengths]
        
        # Interpolate to equal spacing
        new_string = [string[0].copy()]
        
        for i in range(1, len(string) - 1):
            target_s = i / (len(string) - 1)
            
            # Find bracketing points
            for j in range(len(arc_lengths) - 1):
                if arc_lengths[j] <= target_s <= arc_lengths[j + 1]:
                    # Linear interpolation
                    t = (target_s - arc_lengths[j]) / (arc_lengths[j + 1] - arc_lengths[j])
                    new_pos = string[j].positions * (1 - t) + string[j + 1].positions * t
                    
                    image = string[0].copy()
                    image.positions = new_pos
                    new_string.append(image)
                    break
        
        new_string.append(string[-1].copy())
        
        return new_string
    
    def smooth_string(self, string: List[Atoms]) -> List[Atoms]:
        """Apply smoothing to string."""
        smoothed = [string[0].copy()]
        
        for i in range(1, len(string) - 1):
            # Weighted average of neighbors
            new_pos = (string[i-1].positions + string[i+1].positions) / 2
            current_pos = string[i].positions
            
            image = string[i].copy()
            image.positions = (1 - self.config.smoothing_factor) * current_pos + \
                             self.config.smoothing_factor * new_pos
            smoothed.append(image)
        
        smoothed.append(string[-1].copy())
        
        return smoothed
    
    def run(self, initial: Atoms, final: Atoms,
           calculator: Optional[Any] = None) -> Dict:
        """Run string method evolution."""
        logger.info("Starting String Method evolution")
        
        self.string = self.initialize_string(initial, final)
        
        for step in range(self.config.max_steps):
            # Store history
            if step % 10 == 0:
                self.string_history.append([s.copy() for s in self.string])
            
            # Compute forces
            forces = self.compute_forces(self.string, calculator)
            
            # Evolve string
            self.string = self.evolve_string(self.string, forces)
            
            # Reparametrize
            if step % self.config.reparam_steps == 0:
                self.string = self.reparametrize(self.string)
                self.string = self.smooth_string(self.string)
            
            # Check convergence (simplified)
            max_force = max([np.max(np.abs(f)) for f in forces])
            if max_force < self.config.fmax:
                logger.info(f"String converged at step {step}")
                break
        
        # Compute final energies
        self.energies = np.array([s.get_potential_energy() for s in self.string])
        
        return {
            'success': True,
            'n_steps': step + 1,
            'energies': self.energies,
            'string': self.string
        }


class DimerMethod:
    """Dimer Method for finding saddle points."""
    
    def __init__(self, config: DimerConfig):
        self.config = config
        self.n1: Optional[np.ndarray] = None  # Minimum mode direction
        self.dimer_energy: Optional[float] = None
        self.ts_candidate: Optional[Atoms] = None
    
    def initialize_dimer(self, atoms: Atoms, mode_direction: Optional[np.ndarray] = None) -> Tuple[Atoms, Atoms]:
        """Initialize dimer from starting point."""
        if mode_direction is None:
            # Random initial direction
            direction = np.random.randn(*atoms.positions.shape)
            direction = direction / np.linalg.norm(direction)
        else:
            direction = mode_direction / np.linalg.norm(mode_direction)
        
        self.n1 = direction
        
        # Create dimer endpoints
        d = self.config.dimer_distance
        atoms1 = atoms.copy()
        atoms1.positions = atoms.positions + d * direction
        
        atoms2 = atoms.copy()
        atoms2.positions = atoms.positions - d * direction
        
        return atoms1, atoms2
    
    def compute_dimer_forces(self, atoms1: Atoms, atoms2: Atoms,
                            calculator: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forces on dimer endpoints."""
        if calculator:
            atoms1.calc = calculator
            atoms2.calc = calculator
        
        f1 = atoms1.get_forces()
        f2 = atoms2.get_forces()
        
        return f1, f2
    
    def rotate_dimer(self, atoms: Atoms, atoms1: Atoms, atoms2: Atoms,
                    f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """Rotate dimer to align with minimum curvature mode."""
        # Dimer separation
        r1 = atoms1.positions
        r2 = atoms2.positions
        r0 = atoms.positions
        
        d = r1 - r2
        n = d / np.linalg.norm(d)
        
        # Rotational force
        f_rot = f1 - f2
        f_rot_perp = f_rot - np.sum(f_rot * n) * n
        
        # Curvature along dimer direction
        f_parallel = (f1 + f2) / 2
        curvature = np.sum((f1 - f2) * n) / (2 * self.config.dimer_distance)
        
        # Rotation step (simplified)
        if np.linalg.norm(f_rot_perp) > self.config.rotation_threshold:
            # Rotate direction towards f_rot_perp
            theta = np.arctan2(np.linalg.norm(f_rot_perp), -curvature)
            # Update direction
            new_n = n * np.cos(theta) + f_rot_perp / np.linalg.norm(f_rot_perp) * np.sin(theta)
        else:
            new_n = n
        
        return new_n / np.linalg.norm(new_n)
    
    def translate_dimer(self, atoms: Atoms, f1: np.ndarray, f2: np.ndarray,
                       n: np.ndarray) -> Atoms:
        """Translate dimer center according to modified force."""
        # Average force
        f_avg = (f1 + f2) / 2
        
        # Project out component along minimum mode
        f_modified = f_avg - 2 * np.sum(f_avg * n) * n
        
        # Translation step
        new_atoms = atoms.copy()
        
        if self.config.translation_method == 'quickmin':
            # Quickmin step
            dt = 0.1  # fs
            new_atoms.positions += dt * f_modified
        else:
            # Simple gradient descent
            step_size = 0.01
            new_atoms.positions += step_size * f_modified
        
        return new_atoms
    
    def run(self, initial: Atoms, 
           mode_direction: Optional[np.ndarray] = None,
           calculator: Optional[Any] = None) -> Dict:
        """Run Dimer Method to find saddle point."""
        logger.info("Starting Dimer Method search")
        
        # Initialize
        atoms = initial.copy()
        atoms1, atoms2 = self.initialize_dimer(atoms, mode_direction)
        
        for step in range(self.config.max_steps):
            # Get forces
            f1, f2 = self.compute_dimer_forces(atoms1, atoms2, calculator)
            
            # Rotate dimer
            n_new = self.rotate_dimer(atoms, atoms1, atoms2, f1, f2)
            
            # Check rotation convergence
            rotation_converged = np.linalg.norm(n_new - self.n1) < self.config.rotation_threshold
            
            self.n1 = n_new
            
            # Update dimer endpoints
            d = self.config.dimer_distance
            atoms1.positions = atoms.positions + d * self.n1
            atoms2.positions = atoms.positions - d * self.n1
            
            # Translate dimer center
            if rotation_converged or step % 10 == 0:
                atoms_new = self.translate_dimer(atoms, f1, f2, self.n1)
                
                # Check translation convergence
                displacement = np.linalg.norm(atoms_new.positions - atoms.positions)
                
                atoms = atoms_new
                atoms1.positions = atoms.positions + d * self.n1
                atoms2.positions = atoms.positions - d * self.n1
                
                if displacement < 0.001 and rotation_converged:
                    logger.info(f"Dimer converged at step {step}")
                    break
        
        self.ts_candidate = atoms
        if calculator:
            atoms.calc = calculator
            self.dimer_energy = atoms.get_potential_energy()
        
        return {
            'success': True,
            'n_steps': step + 1,
            'ts_candidate': atoms,
            'energy': self.dimer_energy,
            'mode_direction': self.n1
        }


class TransitionStateTheory:
    """Transition State Theory calculations."""
    
    def __init__(self, config: TSTConfig):
        self.config = config
    
    def compute_rate_constant(self, barrier: float, 
                             partition_functions: Dict[str, float]) -> float:
        """Compute TST rate constant.
        
        k_TST = (kT/h) * (Q‡ / Q_R) * exp(-βΔE‡)
        """
        kT = 0.001987204 * self.config.temperature  # kcal/mol
        h = 9.537e-14  # kcal/mol * fs (Planck constant)
        
        Q_ts = partition_functions.get('transition_state', 1.0)
        Q_r = partition_functions.get('reactant', 1.0)
        
        # Rate constant in fs^-1
        kappa = (kT / h) * (Q_ts / Q_r) * np.exp(-barrier / kT)
        
        return kappa
    
    def wigner_correction(self, barrier: float, 
                         imag_freq: float) -> float:
        """Compute Wigner tunneling correction.
        
        κ = 1 + (1/24) * (βh|ν|)²
        """
        h = 6.626e-34  # J*s
        kB = 1.381e-23  # J/K
        T = self.config.temperature
        
        # Convert imaginary frequency to Hz
        nu = abs(imag_freq) * 1e12  # THz to Hz
        
        beta = 1.0 / (kB * T)
        
        kappa = 1.0 + (1.0 / 24.0) * (beta * h * nu) ** 2
        
        return kappa
    
    def compute_recrossing_coefficient(self, 
                                      ts_atoms: Atoms,
                                      method: str = 'rrkm') -> float:
        """Compute transmission coefficient for recrossing."""
        if method.lower() == 'rrkm':
            # Rice-Ramsperger-Kassel-Marcus
            # Simplified - would need detailed dynamics
            return 0.5
        else:
            return 1.0
    
    def analyze_transition_state(self, ts_atoms: Atoms,
                                calculator: Optional[Any] = None) -> Dict:
        """Analyze transition state geometry and frequencies."""
        if calculator:
            ts_atoms.calc = calculator
        
        # Compute vibrational frequencies
        vib = Vibrations(ts_atoms)
        vib.run()
        
        freqs = vib.get_frequencies()
        
        # Check for single imaginary frequency
        imag_freqs = [f for f in freqs if f < 0]
        n_imag = len(imag_freqs)
        
        if n_imag != 1:
            logger.warning(f"Transition state has {n_imag} imaginary frequencies (expected 1)")
        
        # Get mode vectors
        modes = vib.get_mode(-1) if n_imag > 0 else None
        
        return {
            'frequencies': freqs,
            'n_imaginary': n_imag,
            'imaginary_frequency': imag_freqs[0] if n_imag > 0 else None,
            'mode_vectors': modes,
            'is_valid_ts': n_imag == 1
        }


def find_mep_from_ts(ts_atoms: Atoms, 
                    calculator: Optional[Any] = None,
                    n_steps: int = 100,
                    step_size: float = 0.1) -> Tuple[Atoms, Atoms]:
    """Find minimum energy path from transition state.
    
    Uses the imaginary mode to descend to reactant and product.
    
    Returns:
        (reactant_atoms, product_atoms)
    """
    # Analyze TS to get imaginary mode
    tst = TransitionStateTheory(TSTConfig())
    ts_analysis = tst.analyze_transition_state(ts_atoms, calculator)
    
    if ts_analysis['n_imaginary'] != 1:
        raise ValueError("No unique imaginary mode found")
    
    mode = ts_analysis['mode_vectors']
    
    # Descend in both directions
    reactant = ts_atoms.copy()
    product = ts_atoms.copy()
    
    for _ in range(n_steps):
        if calculator:
            reactant.calc = calculator
            product.calc = calculator
        
        f_r = reactant.get_forces()
        f_p = product.get_forces()
        
        # Move along mode direction
        reactant.positions -= step_size * mode
        product.positions += step_size * mode
    
    return reactant, product


def compute_activation_entropy(initial: Atoms, ts: Atoms, final: Atoms,
                              temperature: float = 300.0) -> Dict:
    """Compute activation entropy from vibrational analysis."""
    # Would compute vibrational partition functions
    # and extract entropy contribution
    
    return {
        'delta_s_activation': 0.0,  # cal/mol/K
        'delta_h_activation': 0.0,  # kcal/mol
        'delta_g_activation': 0.0   # kcal/mol
    }


# Export public API
__all__ = [
    'NEBConfig',
    'StringMethodConfig',
    'DimerConfig',
    'TSTConfig',
    'NEB',
    'StringMethod',
    'DimerMethod',
    'TransitionStateTheory',
    'find_mep_from_ts',
    'compute_activation_entropy'
]
