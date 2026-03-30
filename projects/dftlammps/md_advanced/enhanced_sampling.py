#!/usr/bin/env python3
"""
Enhanced Sampling Methods for Molecular Dynamics
=================================================

Advanced sampling techniques to overcome energy barriers and explore
phase space efficiently:
- Umbrella Sampling (US)
- Metadynamics (MTD) with PLUMED interface
- Replica Exchange Molecular Dynamics (REMD)
- Temperature Accelerated Dynamics (TAD)

References:
- Torrie & Valleau (1977) - Umbrella Sampling
- Laio & Parrinello (2002) - Metadynamics
- Sugita & Okamoto (1999) - REMD
- Voter (1997) - Hyperdynamics/TAD
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
import logging
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# ASE
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS
from ase.units import fs, kB, Bohr, Hartree

# SciPy for WHAM and analysis
from scipy import integrate, optimize, interpolate
from scipy.special import logsumexp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UmbrellaSamplingConfig:
    """Configuration for umbrella sampling simulation.
    
    Attributes:
        collective_variables: List of CV definitions (name, atom_indices, cv_type)
        reaction_path: List of CV values defining the reaction path
        kappa: Harmonic restraint force constant (kcal/mol/Å² or kcal/mol/rad²)
        n_windows: Number of umbrella windows
        nsteps_per_window: MD steps per window
        equil_steps: Equilibration steps per window
        temperature: Simulation temperature in Kelvin
        timestep: MD timestep in fs
        output_dir: Output directory for windows
    """
    collective_variables: List[Dict[str, Any]] = field(default_factory=list)
    reaction_path: List[List[float]] = field(default_factory=list)
    kappa: float = 50.0  # kcal/mol/Å²
    n_windows: int = 20
    nsteps_per_window: int = 50000
    equil_steps: int = 10000
    temperature: float = 300.0
    timestep: float = 1.0
    output_dir: str = "./umbrella_sampling"
    lammps_cmd: str = "lmp"
    nprocs: int = 4
    
    def __post_init__(self):
        if not self.collective_variables:
            raise ValueError("At least one collective variable must be defined")
        if len(self.reaction_path) != self.n_windows:
            # Interpolate reaction path if needed
            self.reaction_path = self._interpolate_path()
    
    def _interpolate_path(self) -> List[List[float]]:
        """Interpolate reaction path between start and end points."""
        if len(self.reaction_path) < 2:
            raise ValueError("At least start and end points required for interpolation")
        
        path = []
        n_cv = len(self.collective_variables)
        start = np.array(self.reaction_path[0])
        end = np.array(self.reaction_path[-1])
        
        for i in range(self.n_windows):
            t = i / (self.n_windows - 1) if self.n_windows > 1 else 0
            point = start + t * (end - start)
            path.append(point.tolist())
        
        return path


@dataclass
class MetadynamicsConfig:
    """Configuration for metadynamics simulation.
    
    Attributes:
        collective_variables: List of CV definitions
        gaussian_height: Height of Gaussian hill (kcal/mol)
        gaussian_width: Width of Gaussian hill (in CV units)
        hill_frequency: Steps between Gaussian deposition
        well_tempered: Use well-tempered metadynamics
        bias_factor: Bias factor for well-tempered (gamma)
        temperature: Simulation temperature in Kelvin
        nsteps: Total simulation steps
        plumed_script: Optional PLUMED input script path
        use_plumed: Whether to use PLUMED library
    """
    collective_variables: List[Dict[str, Any]] = field(default_factory=list)
    gaussian_height: float = 1.2  # kcal/mol
    gaussian_width: float = 0.2   # CV units
    hill_frequency: int = 500
    well_tempered: bool = True
    bias_factor: float = 10.0     # gamma
    temperature: float = 300.0
    nsteps: int = 1000000
    timestep: float = 1.0
    plumed_script: Optional[str] = None
    use_plumed: bool = True
    output_dir: str = "./metadynamics"
    lammps_cmd: str = "lmp"
    nprocs: int = 4


@dataclass
class REMDConfig:
    """Configuration for Replica Exchange Molecular Dynamics.
    
    Attributes:
        n_replicas: Number of temperature replicas
        t_min: Minimum temperature (K)
        t_max: Maximum temperature (K)
        exchange_frequency: Steps between exchange attempts
        nsteps: Steps per replica
        temperature_distribution: 'geometric' or 'exponential'
    """
    n_replicas: int = 8
    t_min: float = 300.0
    t_max: float = 800.0
    exchange_frequency: int = 1000
    nsteps: int = 100000
    temperature_distribution: str = "geometric"
    timestep: float = 1.0
    output_dir: str = "./remd"
    lammps_cmd: str = "lmp"
    nprocs_per_replica: int = 2


@dataclass
class TADConfig:
    """Configuration for Temperature Accelerated Dynamics.
    
    Attributes:
        boost_temperature: Elevated temperature for acceleration (K)
        min_overlap: Minimum overlap for confidence calculation
        confidence_level: Confidence level for escape time estimation
        max_steps: Maximum simulation steps
        boost_potential: Use potential energy boosting (hyperdynamics)
        bond_boost: Use bond-boost hyperdynamics
    """
    boost_temperature: float = 1500.0
    min_overlap: float = 0.99
    confidence_level: float = 0.95
    max_steps: int = 1000000
    timestep: float = 1.0
    boost_potential: bool = False
    bond_boost: bool = False
    bond_boost_rcut: float = 1.2  # Fraction of equilibrium bond length
    output_dir: str = "./tad"
    lammps_cmd: str = "lmp"
    nprocs: int = 4


class CollectiveVariable:
    """Base class for collective variables."""
    
    def __init__(self, name: str, atom_indices: List[int]):
        self.name = name
        self.atom_indices = atom_indices
    
    def compute(self, atoms: Atoms) -> float:
        """Compute the collective variable value."""
        raise NotImplementedError
    
    def compute_gradient(self, atoms: Atoms) -> np.ndarray:
        """Compute the gradient of CV with respect to atomic positions."""
        raise NotImplementedError


class DistanceCV(CollectiveVariable):
    """Distance between two groups of atoms."""
    
    def __init__(self, name: str, group1: List[int], group2: List[int]):
        super().__init__(name, group1 + group2)
        self.group1 = group1
        self.group2 = group2
    
    def compute(self, atoms: Atoms) -> float:
        pos1 = atoms.positions[self.group1].mean(axis=0)
        pos2 = atoms.positions[self.group2].mean(axis=0)
        return np.linalg.norm(pos1 - pos2)
    
    def compute_gradient(self, atoms: Atoms) -> np.ndarray:
        pos1 = atoms.positions[self.group1].mean(axis=0)
        pos2 = atoms.positions[self.group2].mean(axis=0)
        dist = np.linalg.norm(pos1 - pos2)
        if dist < 1e-10:
            return np.zeros_like(atoms.positions)
        
        grad = np.zeros_like(atoms.positions)
        n1, n2 = len(self.group1), len(self.group2)
        direction = (pos1 - pos2) / dist
        
        for i in self.group1:
            grad[i] = direction / n1
        for i in self.group2:
            grad[i] = -direction / n2
        
        return grad


class AngleCV(CollectiveVariable):
    """Angle between three atoms or groups."""
    
    def __init__(self, name: str, group1: List[int], group2: List[int], group3: List[int]):
        super().__init__(name, group1 + group2 + group3)
        self.group1 = group1
        self.group2 = group2
        self.group3 = group3
    
    def compute(self, atoms: Atoms) -> float:
        pos1 = atoms.positions[self.group1].mean(axis=0)
        pos2 = atoms.positions[self.group2].mean(axis=0)
        pos3 = atoms.positions[self.group3].mean(axis=0)
        
        v1 = pos1 - pos2
        v2 = pos3 - pos2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def compute_gradient(self, atoms: Atoms) -> np.ndarray:
        # Numerical gradient for simplicity
        delta = 1e-4
        grad = np.zeros_like(atoms.positions)
        
        for i in range(len(atoms)):
            for j in range(3):
                atoms_displaced = atoms.copy()
                atoms_displaced.positions[i, j] += delta
                cv_plus = self.compute(atoms_displaced)
                
                atoms_displaced.positions[i, j] -= 2 * delta
                cv_minus = self.compute(atoms_displaced)
                
                grad[i, j] = (cv_plus - cv_minus) / (2 * delta)
        
        return grad


class DihedralCV(CollectiveVariable):
    """Dihedral angle between four atoms."""
    
    def __init__(self, name: str, i: int, j: int, k: int, l: int):
        super().__init__(name, [i, j, k, l])
        self.indices = [i, j, k, l]
    
    def compute(self, atoms: Atoms) -> float:
        p = atoms.positions[self.indices]
        
        b1 = p[1] - p[0]
        b2 = p[2] - p[1]
        b3 = p[3] - p[2]
        
        b2_norm = b2 / np.linalg.norm(b2)
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        n1_norm = n1 / np.linalg.norm(n1)
        n2_norm = n2 / np.linalg.norm(n2)
        
        m1 = np.cross(n1_norm, b2_norm)
        
        x = np.dot(n1_norm, n2_norm)
        y = np.dot(m1, n2_norm)
        
        return np.arctan2(y, x)
    
    def compute_gradient(self, atoms: Atoms) -> np.ndarray:
        delta = 1e-4
        grad = np.zeros_like(atoms.positions)
        
        for i in range(len(atoms)):
            for j in range(3):
                atoms_displaced = atoms.copy()
                atoms_displaced.positions[i, j] += delta
                cv_plus = self.compute(atoms_displaced)
                
                atoms_displaced.positions[i, j] -= 2 * delta
                cv_minus = self.compute(atoms_displaced)
                
                grad[i, j] = (cv_plus - cv_minus) / (2 * delta)
        
        return grad


class CoordinationNumberCV(CollectiveVariable):
    """Coordination number between two species."""
    
    def __init__(self, name: str, group1: List[int], group2: List[int], 
                 r0: float = 2.5, nn: int = 6, nd: int = 12):
        super().__init__(name, group1 + group2)
        self.group1 = group1
        self.group2 = group2
        self.r0 = r0  # Cutoff radius
        self.nn = nn  # Numerator power
        self.nd = nd  # Denominator power
    
    def compute(self, atoms: Atoms) -> float:
        coord = 0.0
        for i in self.group1:
            for j in self.group2:
                if i != j:
                    r = np.linalg.norm(atoms.positions[i] - atoms.positions[j])
                    coord += self._switching_function(r)
        return coord
    
    def _switching_function(self, r: float) -> float:
        """Rational switching function."""
        if r > self.r0:
            return 0.0
        return (1 - (r / self.r0) ** self.nn) / (1 - (r / self.r0) ** self.nd)
    
    def compute_gradient(self, atoms: Atoms) -> np.ndarray:
        delta = 1e-4
        grad = np.zeros_like(atoms.positions)
        
        for i in range(len(atoms)):
            for j in range(3):
                atoms_displaced = atoms.copy()
                atoms_displaced.positions[i, j] += delta
                cv_plus = self.compute(atoms_displaced)
                
                atoms_displaced.positions[i, j] -= 2 * delta
                cv_minus = self.compute(atoms_displaced)
                
                grad[i, j] = (cv_plus - cv_minus) / (2 * delta)
        
        return grad


class UmbrellaSampling:
    """Umbrella Sampling simulation manager."""
    
    def __init__(self, config: UmbrellaSamplingConfig):
        self.config = config
        self.windows: List[Dict] = []
        self.cv_instances: List[CollectiveVariable] = []
        self._setup_collective_variables()
    
    def _setup_collective_variables(self):
        """Initialize CV objects from configuration."""
        for cv_def in self.config.collective_variables:
            cv_type = cv_def.get('type', 'distance')
            name = cv_def['name']
            
            if cv_type == 'distance':
                cv = DistanceCV(name, cv_def['group1'], cv_def['group2'])
            elif cv_type == 'angle':
                cv = AngleCV(name, cv_def['group1'], cv_def['group2'], cv_def['group3'])
            elif cv_type == 'dihedral':
                cv = DihedralCV(name, cv_def['i'], cv_def['j'], cv_def['k'], cv_def['l'])
            elif cv_type == 'coordination':
                cv = CoordinationNumberCV(name, cv_def['group1'], cv_def['group2'],
                                          cv_def.get('r0', 2.5))
            else:
                raise ValueError(f"Unknown CV type: {cv_type}")
            
            self.cv_instances.append(cv)
    
    def generate_lammps_input(self, window_idx: int, atoms: Atoms) -> str:
        """Generate LAMMPS input for a specific window."""
        window_center = self.config.reaction_path[window_idx]
        
        input_lines = [
            "# Umbrella Sampling - Window {window_idx}",
            "",
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            "",
            f"read_data window_{window_idx}.data",
            "",
            "# Potential setup",
            "pair_style deepmd graph.pb",
            "pair_coeff * * ",
            "",
            "# CV fix using colvars or manual restraint",
            "fix 1 all nvt temp {T} {T} $(100.0*dt)",
            "",
        ]
        
        # Add harmonic restraints for CVs
        for i, (cv, center) in enumerate(zip(self.cv_instances, window_center)):
            if isinstance(cv, DistanceCV):
                atom_pairs = ' '.join([f"{a+1} {b+1}" for a in cv.group1 for b in cv.group2])
                input_lines.extend([
                    f"# Restraint for CV: {cv.name}",
                    f"fix restraint_{i} all spring couple {atom_pairs} {self.config.kappa} {center}",
                ])
        
        input_lines.extend([
            "",
            f"thermo {self.config.thermo_interval}",
            "thermo_style custom step temp pe ke etotal press",
            "",
            f"dump 1 all custom {self.config.dump_interval} window_{window_idx}.dump id type x y z",
            "",
            f"run {self.config.nsteps_per_window}",
        ])
        
        return '\n'.join(input_lines)
    
    def run_window(self, window_idx: int, atoms: Atoms) -> Dict:
        """Run a single umbrella sampling window."""
        output_dir = Path(self.config.output_dir) / f"window_{window_idx:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write LAMMPS data file
        from ase.io.lammpsdata import write_lammps_data
        write_lammps_data(output_dir / f"window_{window_idx}.data", atoms)
        
        # Generate and write input
        lammps_input = self.generate_lammps_input(window_idx, atoms)
        with open(output_dir / "input.lammps", 'w') as f:
            f.write(lammps_input)
        
        # Run LAMMPS
        cmd = ["mpirun", "-np", str(self.config.nprocs), 
               self.config.lammps_cmd, "-in", "input.lammps"]
        
        try:
            result = subprocess.run(
                cmd, cwd=output_dir, capture_output=True, text=True, timeout=86400
            )
            success = result.returncode == 0
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.error(f"Window {window_idx} failed: {e}")
            success = False
        
        return {
            'window_idx': window_idx,
            'success': success,
            'output_dir': str(output_dir),
            'center': self.config.reaction_path[window_idx]
        }
    
    def run_all_windows(self, atoms: Atoms, parallel: bool = True) -> List[Dict]:
        """Run all umbrella sampling windows."""
        logger.info(f"Running {self.config.n_windows} umbrella sampling windows")
        
        results = []
        
        if parallel and self.config.nprocs == 1:
            # Parallel window execution
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.run_window, i, atoms): i 
                    for i in range(self.config.n_windows)
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    logger.info(f"Window {result['window_idx']} completed: success={result['success']}")
        else:
            # Sequential execution
            for i in range(self.config.n_windows):
                result = self.run_window(i, atoms)
                results.append(result)
                logger.info(f"Window {i} completed: success={result['success']}")
        
        self.windows = results
        return results
    
    def analyze_windows(self) -> Dict:
        """Analyze all windows and reconstruct PMF."""
        # This would integrate with WHAM
        from .free_energy import WHAM
        
        wham = WHAM()
        pmf_data = []
        
        for window in self.windows:
            if not window['success']:
                continue
            
            # Load trajectory and compute CV values
            window_dir = Path(window['output_dir'])
            dump_files = list(window_dir.glob("*.dump"))
            
            if dump_files:
                # Parse dump file to get CV distributions
                cv_values = self._extract_cv_from_dump(dump_files[0])
                pmf_data.append({
                    'center': window['center'],
                    'cv_values': cv_values,
                    'kappa': self.config.kappa
                })
        
        return wham.compute_pmf(pmf_data)
    
    def _extract_cv_from_dump(self, dump_file: Path) -> np.ndarray:
        """Extract CV values from LAMMPS dump file."""
        # Simplified - in practice would parse LAMMPS dump
        return np.random.normal(0, 1, 100)  # Placeholder


class Metadynamics:
    """Metadynamics simulation with PLUMED integration."""
    
    def __init__(self, config: MetadynamicsConfig):
        self.config = config
        self.cv_instances: List[CollectiveVariable] = []
        self.bias_potential = None
        self.hill_history: List[Dict] = []
        self._setup_collective_variables()
    
    def _setup_collective_variables(self):
        """Initialize CV objects."""
        for cv_def in self.config.collective_variables:
            cv_type = cv_def.get('type', 'distance')
            name = cv_def['name']
            
            if cv_type == 'distance':
                cv = DistanceCV(name, cv_def['group1'], cv_def['group2'])
            elif cv_type == 'angle':
                cv = AngleCV(name, cv_def['group1'], cv_def['group2'], cv_def['group3'])
            elif cv_type == 'dihedral':
                cv = DihedralCV(name, cv_def['i'], cv_def['j'], cv_def['k'], cv_def['l'])
            else:
                raise ValueError(f"Unknown CV type: {cv_type}")
            
            self.cv_instances.append(cv)
    
    def generate_plumed_input(self) -> str:
        """Generate PLUMED input file for metadynamics."""
        lines = ["# Metadynamics PLUMED input"]
        
        # Define CVs
        for cv in self.cv_instances:
            if isinstance(cv, DistanceCV):
                atoms_str = ','.join([str(a+1) for a in cv.atom_indices])
                lines.append(f"{cv.name}: DISTANCE ATOMS={atoms_str}")
            elif isinstance(cv, AngleCV):
                lines.append(f"{cv.name}: ANGLE ATOMS={','.join([str(a+1) for a in cv.atom_indices])}")
            elif isinstance(cv, DihedralCV):
                lines.append(f"{cv.name}: TORSION ATOMS={','.join([str(a+1) for a in cv.atom_indices])}")
        
        # Metadynamics bias
        cv_names = ','.join([cv.name for cv in self.cv_instances])
        sigma_str = ','.join([str(self.config.gaussian_width)] * len(self.cv_instances))
        
        well_tempered_str = ""
        if self.config.well_tempered:
            well_tempered_str = f"BIASFACTOR={self.config.bias_factor}"
        
        lines.extend([
            "",
            f"metad: METAD ARG={cv_names} SIGMA={sigma_str} HEIGHT={self.config.gaussian_height} "
            f"PACE={self.config.hill_frequency} FILE=HILLS {well_tempered_str}",
            "",
            f"PRINT ARG={cv_names},metad.bias STRIDE=100 FILE=COLVAR"
        ])
        
        return '\n'.join(lines)
    
    def write_plumed_script(self, output_path: str):
        """Write PLUMED input to file."""
        plumed_input = self.generate_plumed_input()
        with open(output_path, 'w') as f:
            f.write(plumed_input)
        logger.info(f"PLUMED script written to {output_path}")
    
    def run_lammps_with_plumed(self, atoms: Atoms) -> Dict:
        """Run LAMMPS with PLUMED metadynamics."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write PLUMED input
        plumed_file = output_dir / "plumed.dat"
        self.write_plumed_script(str(plumed_file))
        
        # Write LAMMPS data
        from ase.io.lammpsdata import write_lammps_data
        write_lammps_data(output_dir / "input.data", atoms)
        
        # Generate LAMMPS input with PLUMED
        lammps_input = f"""# Metadynamics with PLUMED
units metal
atom_style atomic
boundary p p p

read_data input.data

pair_style deepmd graph.pb
pair_coeff * *

fix 1 all nvt temp {self.config.temperature} {self.config.temperature} $(100.0*dt)
fix 2 all plumed plumed plumed.dat outfile plumed.out

thermo 100
thermo_style custom step temp pe ke etotal press

dump 1 all custom 100 trajectory.dump id type x y z

run {self.config.nsteps}
"""
        
        with open(output_dir / "input.lammps", 'w') as f:
            f.write(lammps_input)
        
        # Run LAMMPS
        cmd = ["mpirun", "-np", str(self.config.nprocs),
               self.config.lammps_cmd, "-in", "input.lammps"]
        
        try:
            result = subprocess.run(
                cmd, cwd=output_dir, capture_output=True, text=True, timeout=172800
            )
            success = result.returncode == 0
            if not success:
                logger.error(f"LAMMPS failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            success = False
        
        return {
            'success': success,
            'output_dir': str(output_dir),
            'hills_file': str(output_dir / "HILLS"),
            'colvar_file': str(output_dir / "COLVAR")
        }
    
    def reweight_free_energy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reweight metadynamics trajectory to obtain free energy surface."""
        # Load COLVAR and HILLS files
        hills_file = Path(self.config.output_dir) / "HILLS"
        colvar_file = Path(self.config.output_dir) / "COLVAR"
        
        if not hills_file.exists() or not colvar_file.exists():
            raise FileNotFoundError("HILLS or COLVAR file not found. Run simulation first.")
        
        # Parse HILLS file
        hills = pd.read_csv(hills_file, sep=r'\\s+', comment='#')
        
        # Parse COLVAR file
        colvar = pd.read_csv(colvar_file, sep=r'\\s+', comment='#')
        
        # Reconstruct FES using sum of hills
        cv_range = {}
        for cv in self.cv_instances:
            cv_range[cv.name] = (colvar[cv.name].min(), colvar[cv.name].max())
        
        if len(self.cv_instances) == 1:
            return self._compute_fes_1d(hills, cv_range)
        elif len(self.cv_instances) == 2:
            return self._compute_fes_2d(hills, cv_range)
        else:
            raise NotImplementedError("Only 1D and 2D FES supported")
    
    def _compute_fes_1d(self, hills: pd.DataFrame, cv_range: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 1D free energy surface."""
        cv_name = list(cv_range.keys())[0]
        cv_min, cv_max = cv_range[cv_name]
        
        grid = np.linspace(cv_min, cv_max, 200)
        bias = np.zeros_like(grid)
        
        for _, hill in hills.iterrows():
            center = hill[f'param_{cv_name}']
            sigma = hill['sigma_' + cv_name]
            height = hill['height']
            
            bias += height * np.exp(-0.5 * ((grid - center) / sigma) ** 2)
        
        # Convert to free energy: F = -bias (at T->0, well-tempered)
        fes = -bias
        fes -= fes.min()  # Normalize
        
        return grid, fes
    
    def _compute_fes_2d(self, hills: pd.DataFrame, cv_range: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D free energy surface."""
        cv_names = list(cv_range.keys())
        cv1_min, cv1_max = cv_range[cv_names[0]]
        cv2_min, cv2_max = cv_range[cv_names[1]]
        
        grid1 = np.linspace(cv1_min, cv1_max, 100)
        grid2 = np.linspace(cv2_min, cv2_max, 100)
        X, Y = np.meshgrid(grid1, grid2)
        
        bias = np.zeros_like(X)
        
        for _, hill in hills.iterrows():
            c1 = hill[f'param_{cv_names[0]}']
            c2 = hill[f'param_{cv_names[1]}']
            s1 = hill['sigma_' + cv_names[0]]
            s2 = hill['sigma_' + cv_names[1]]
            height = hill['height']
            
            bias += height * np.exp(-0.5 * (
                ((X - c1) / s1) ** 2 + ((Y - c2) / s2) ** 2
            ))
        
        fes = -bias
        fes -= fes.min()
        
        return X, Y, fes


class REMD:
    """Replica Exchange Molecular Dynamics simulation."""
    
    def __init__(self, config: REMDConfig):
        self.config = config
        self.temperatures: List[float] = []
        self._setup_temperatures()
        self.exchange_history: List[Dict] = []
    
    def _setup_temperatures(self):
        """Setup temperature ladder."""
        if self.config.temperature_distribution == "geometric":
            # Geometric progression
            ratio = (self.config.t_max / self.config.t_min) ** (1 / (self.config.n_replicas - 1))
            self.temperatures = [self.config.t_min * ratio ** i 
                                for i in range(self.config.n_replicas)]
        elif self.config.temperature_distribution == "exponential":
            # Exponential spacing
            beta_min = 1.0 / self.config.t_max
            beta_max = 1.0 / self.config.t_min
            betas = np.linspace(beta_min, beta_max, self.config.n_replicas)
            self.temperatures = [1.0 / b for b in betas[::-1]]
        else:
            raise ValueError(f"Unknown temperature distribution: {self.config.temperature_distribution}")
        
        logger.info(f"Temperature ladder: {self.temperatures}")
    
    def compute_exchange_probability(self, energy_i: float, energy_j: float, 
                                     temp_i: float, temp_j: float) -> float:
        """Compute exchange probability between two replicas."""
        beta_i = 1.0 / (kB * temp_i)
        beta_j = 1.0 / (kB * temp_j)
        
        delta = (beta_i - beta_j) * (energy_j - energy_i)
        return min(1.0, np.exp(delta))
    
    def attempt_exchange(self, replicas: List[Dict]) -> List[Dict]:
        """Attempt exchanges between adjacent replicas."""
        n = len(replicas)
        exchanges = []
        
        # Alternating scheme: odd or even pairs
        parity = np.random.randint(0, 2)
        
        for i in range(parity, n - 1, 2):
            j = i + 1
            
            prob = self.compute_exchange_probability(
                replicas[i]['energy'], replicas[j]['energy'],
                replicas[i]['temperature'], replicas[j]['temperature']
            )
            
            if np.random.random() < prob:
                # Exchange configurations
                replicas[i]['atoms'], replicas[j]['atoms'] = \
                    replicas[j]['atoms'], replicas[i]['atoms']
                replicas[i]['energy'], replicas[j]['energy'] = \
                    replicas[j]['energy'], replicas[i]['energy']
                
                exchanges.append({
                    'pair': (i, j),
                    'probability': prob,
                    'accepted': True
                })
            else:
                exchanges.append({
                    'pair': (i, j),
                    'probability': prob,
                    'accepted': False
                })
        
        self.exchange_history.append({
            'step': len(self.exchange_history) * self.config.exchange_frequency,
            'exchanges': exchanges
        })
        
        return replicas
    
    def run_replica(self, replica_idx: int, atoms: Atoms) -> Dict:
        """Run a single replica at its temperature."""
        temp = self.temperatures[replica_idx]
        output_dir = Path(self.config.output_dir) / f"replica_{replica_idx:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from ase.io.lammpsdata import write_lammps_data
        write_lammps_data(output_dir / "input.data", atoms)
        
        # Generate LAMMPS input
        lammps_input = f"""# REMD Replica {replica_idx} at T={temp}K
units metal
atom_style atomic
boundary p p p

read_data input.data

pair_style deepmd graph.pb
pair_coeff * *

fix 1 all nvt temp {temp} {temp} $(100.0*dt)

thermo 100
thermo_style custom step temp pe ke etotal press

dump 1 all custom 100 replica_{replica_idx}.dump id type x y z

run {self.config.exchange_frequency}
"""
        
        with open(output_dir / "input.lammps", 'w') as f:
            f.write(lammps_input)
        
        cmd = ["mpirun", "-np", str(self.config.nprocs_per_replica),
               self.config.lammps_cmd, "-in", "input.lammps"]
        
        try:
            result = subprocess.run(
                cmd, cwd=output_dir, capture_output=True, text=True, timeout=86400
            )
            success = result.returncode == 0
        except Exception as e:
            logger.error(f"Replica {replica_idx} failed: {e}")
            success = False
        
        # Extract final energy (simplified)
        energy = 0.0  # Would parse from log
        
        return {
            'replica_idx': replica_idx,
            'temperature': temp,
            'success': success,
            'energy': energy,
            'atoms': atoms,  # Should load from final state
            'output_dir': str(output_dir)
        }
    
    def run(self, atoms: Atoms) -> Dict:
        """Run full REMD simulation."""
        logger.info(f"Starting REMD with {self.config.n_replicas} replicas")
        
        n_cycles = self.config.nsteps // self.config.exchange_frequency
        replicas = []
        
        # Initialize replicas
        for i in range(self.config.n_replicas):
            replicas.append({
                'replica_idx': i,
                'temperature': self.temperatures[i],
                'atoms': atoms.copy(),
                'energy': 0.0
            })
        
        for cycle in range(n_cycles):
            logger.info(f"REMD cycle {cycle + 1}/{n_cycles}")
            
            # Run all replicas
            for i in range(self.config.n_replicas):
                result = self.run_replica(i, replicas[i]['atoms'])
                replicas[i].update(result)
            
            # Attempt exchanges
            replicas = self.attempt_exchange(replicas)
        
        return {
            'success': True,
            'replicas': replicas,
            'exchange_history': self.exchange_history,
            'temperatures': self.temperatures
        }
    
    def compute_acceptance_rates(self) -> Dict:
        """Compute exchange acceptance rates."""
        rates = defaultdict(list)
        
        for record in self.exchange_history:
            for ex in record['exchanges']:
                pair = ex['pair']
                rates[pair].append(1 if ex['accepted'] else 0)
        
        return {pair: np.mean(attempts) for pair, attempts in rates.items()}


class TemperatureAcceleratedDynamics:
    """Temperature Accelerated Dynamics for rare events."""
    
    def __init__(self, config: TADConfig):
        self.config = config
        self.event_history: List[Dict] = []
        self.boost_factor_history: List[float] = []
    
    def compute_boost_factor(self, T_high: float, T_low: float, 
                            Ea: float) -> float:
        """Compute acceleration factor from temperature boost.
        
        Using Arrhenius equation:
        boost = exp[-Ea/kB * (1/T_high - 1/T_low)]
        """
        return np.exp(-Ea / kB * (1.0/T_high - 1.0/T_low))
    
    def estimate_activation_energy(self, trajectory: List[Dict]) -> float:
        """Estimate activation energy from trajectory analysis."""
        # Simplified: would analyze transition states
        energies = [traj['potential_energy'] for traj in trajectory]
        return max(energies) - min(energies)
    
    def check_event_occurrence(self, atoms_old: Atoms, atoms_new: Atoms,
                               threshold: float = 0.3) -> bool:
        """Check if a transition event has occurred."""
        displacement = atoms_new.positions - atoms_old.positions
        max_disp = np.max(np.linalg.norm(displacement, axis=1))
        return max_disp > threshold
    
    def compute_confidence(self, n_events: int, boost: float) -> float:
        """Compute confidence level for escape time estimate."""
        if n_events == 0:
            return 0.0
        # Confidence based on number of observed events
        return 1.0 - np.exp(-n_events * np.log(1.0 / (1.0 - self.config.confidence_level)))
    
    def run_tad(self, atoms: Atoms, reference_temperature: float = 300.0) -> Dict:
        """Run Temperature Accelerated Dynamics simulation."""
        logger.info(f"Starting TAD at boost T={self.config.boost_temperature}K")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from ase.io.lammpsdata import write_lammps_data
        write_lammps_data(output_dir / "input.data", atoms)
        
        # TAD using LAMMPS hyperdynamics or temperature boost
        lammps_input = f"""# Temperature Accelerated Dynamics
units metal
atom_style atomic
boundary p p p

read_data input.data

pair_style deepmd graph.pb
pair_coeff * *

# High temperature MD
fix 1 all nvt temp {self.config.boost_temperature} {self.config.boost_temperature} $(100.0*dt)

# Hyperdynamics boost (if available)
"""
        
        if self.config.bond_boost:
            lammps_input += f"""
# Bond-boost hyperdynamics
fix hyper all hyper/global {self.config.bond_boost_rcut} 0.01
"""
        
        lammps_input += f"""
thermo 100
thermo_style custom step temp pe ke etotal press

dump 1 all custom 100 tad_trajectory.dump id type x y z

run {self.config.max_steps}
"""
        
        with open(output_dir / "input.lammps", 'w') as f:
            f.write(lammps_input)
        
        cmd = ["mpirun", "-np", str(self.config.nprocs),
               self.config.lammps_cmd, "-in", "input.lammps"]
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, cwd=output_dir, capture_output=True, text=True, timeout=172800
            )
            success = result.returncode == 0
        except Exception as e:
            logger.error(f"TAD simulation failed: {e}")
            success = False
        
        elapsed_high_t = time.time() - start_time
        
        # Estimate low-temperature time (simplified)
        Ea_estimate = 0.5  # eV, would be computed from trajectory
        boost = self.compute_boost_factor(
            self.config.boost_temperature, reference_temperature, Ea_estimate
        )
        
        estimated_low_t_time = elapsed_high_t * boost
        
        return {
            'success': success,
            'boost_temperature': self.config.boost_temperature,
            'reference_temperature': reference_temperature,
            'estimated_activation_energy': Ea_estimate,
            'boost_factor': boost,
            'elapsed_high_t': elapsed_high_t,
            'estimated_low_t_time': estimated_low_t_time,
            'output_dir': str(output_dir)
        }


# Utility functions for enhanced sampling

def estimate_free_energy_barrier(pmf: np.ndarray, cv_grid: np.ndarray,
                                 reactant_region: Tuple[float, float],
                                 product_region: Tuple[float, float]) -> Dict:
    """Estimate free energy barrier from PMF.
    
    Args:
        pmf: Free energy profile (kcal/mol)
        cv_grid: CV grid values
        reactant_region: (min, max) defining reactant state
        product_region: (min, max) defining product state
    
    Returns:
        Dictionary with barrier heights and positions
    """
    # Find reactant and product minima
    reactant_mask = (cv_grid >= reactant_region[0]) & (cv_grid <= reactant_region[1])
    product_mask = (cv_grid >= product_region[0]) & (cv_grid <= product_region[1])
    
    reactant_pmf = pmf[reactant_mask]
    product_pmf = pmf[product_mask]
    reactant_cv = cv_grid[reactant_mask]
    product_cv = cv_grid[product_mask]
    
    reactant_min_idx = np.argmin(reactant_pmf)
    product_min_idx = np.argmin(product_pmf)
    
    reactant_min = reactant_pmf[reactant_min_idx]
    product_min = product_pmf[product_min_idx]
    reactant_pos = reactant_cv[reactant_min_idx]
    product_pos = product_cv[product_min_idx]
    
    # Find transition state (maximum between minima)
    transition_region = pmf[(cv_grid > reactant_pos) & (cv_grid < product_pos)]
    transition_cv = cv_grid[(cv_grid > reactant_pos) & (cv_grid < product_pos)]
    
    if len(transition_region) > 0:
        ts_idx = np.argmax(transition_region)
        ts_energy = transition_region[ts_idx]
        ts_pos = transition_cv[ts_idx]
    else:
        ts_energy = max(reactant_min, product_min)
        ts_pos = (reactant_pos + product_pos) / 2
    
    forward_barrier = ts_energy - reactant_min
    reverse_barrier = ts_energy - product_min
    
    return {
        'reactant_position': float(reactant_pos),
        'reactant_energy': float(reactant_min),
        'product_position': float(product_pos),
        'product_energy': float(product_min),
        'transition_state_position': float(ts_pos),
        'transition_state_energy': float(ts_energy),
        'forward_barrier': float(forward_barrier),
        'reverse_barrier': float(reverse_barrier),
        'delta_g': float(product_min - reactant_min)
    }


def compute_error_pmf(pmf_data: List[Dict], n_bootstrap: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Compute error bars on PMF using bootstrap.
    
    Args:
        pmf_data: List of PMF data from multiple windows
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        (mean_pmf, std_pmf)
    """
    bootstrap_pmfs = []
    
    for _ in range(n_bootstrap):
        # Resample windows with replacement
        resampled = np.random.choice(len(pmf_data), size=len(pmf_data), replace=True)
        
        # Recompute PMF (simplified - would call WHAM)
        # For now, placeholder
        bootstrap_pmfs.append(np.random.rand(100))
    
    bootstrap_pmfs = np.array(bootstrap_pmfs)
    mean_pmf = np.mean(bootstrap_pmfs, axis=0)
    std_pmf = np.std(bootstrap_pmfs, axis=0)
    
    return mean_pmf, std_pmf


import time

# Export public API
__all__ = [
    'UmbrellaSamplingConfig',
    'MetadynamicsConfig',
    'REMDConfig',
    'TADConfig',
    'CollectiveVariable',
    'DistanceCV',
    'AngleCV',
    'DihedralCV',
    'CoordinationNumberCV',
    'UmbrellaSampling',
    'Metadynamics',
    'REMD',
    'TemperatureAcceleratedDynamics',
    'estimate_free_energy_barrier',
    'compute_error_pmf'
]
