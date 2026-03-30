#!/usr/bin/env python3
"""
MD Simulation Module
====================
Molecular dynamics simulation interface for LAMMPS.

This module provides a high-level interface for running MD simulations
using various potentials (classical, ML-based) through LAMMPS.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import logging
import subprocess
import tempfile
from collections import defaultdict

# ASE
from ase import Atoms
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump_text
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize import BFGS, FIRE, LBFGS
from ase.units import fs, kB, GPa
from ase.neighborlist import NeighborList
from ase.geometry.analysis import Analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MDConfig:
    """Molecular dynamics simulation configuration.
    
    Attributes:
        ensemble: Simulation ensemble type ('nve', 'nvt', 'npt')
        temperature: Target temperature in Kelvin
        pressure: Target pressure in atm (for NPT ensemble)
        timestep: Integration timestep in femtoseconds
        nsteps: Total number of simulation steps
        nsteps_equil: Number of equilibration steps
        thermo_interval: Output interval for thermodynamic properties
        dump_interval: Output interval for trajectory dumps
        pair_style: LAMMPS pair style ('deepmd', 'snap', 'tersoff', 'eam/alloy')
        potential_file: Path to potential file
        working_dir: Working directory for simulation
        nprocs: Number of processors for parallel execution
    """
    ensemble: str = "nvt"
    temperature: float = 300.0
    pressure: Optional[float] = None
    timestep: float = 1.0
    nsteps: int = 100000
    nsteps_equil: int = 10000
    thermo_interval: int = 100
    dump_interval: int = 1000
    pair_style: str = "deepmd"
    potential_file: str = "graph.pb"
    working_dir: str = "./md_run"
    nprocs: int = 4
    lammps_cmd: str = "lmp"


@dataclass
class MDTrajectory:
    """MD trajectory container.
    
    Attributes:
        positions: Atomic positions over time
        velocities: Atomic velocities over time
        forces: Atomic forces over time
        energies: Energy components over time
        temperatures: Temperature over time
        pressures: Pressure over time
        time: Simulation time array
    """
    positions: List[np.ndarray] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    forces: List[np.ndarray] = field(default_factory=list)
    energies: Dict[str, List[float]] = field(default_factory=dict)
    temperatures: List[float] = field(default_factory=list)
    pressures: List[float] = field(default_factory=list)
    time: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert trajectory to dictionary."""
        return {
            'positions': [p.tolist() for p in self.positions],
            'velocities': [v.tolist() for v in self.velocities] if self.velocities else [],
            'forces': [f.tolist() for f in self.forces] if self.forces else [],
            'energies': self.energies,
            'temperatures': self.temperatures,
            'pressures': self.pressures,
            'time': self.time,
        }


class MDSimulationRunner:
    """MD simulation runner for LAMMPS.
    
    This class manages the execution of molecular dynamics simulations
    using LAMMPS with various potential types.
    
    Example:
        >>> config = MDConfig(temperature=300, nsteps=100000)
        >>> runner = MDSimulationRunner(config)
        >>> trajectory = runner.run(atoms, potential_file='graph.pb')
    """
    
    def __init__(self, config: Optional[MDConfig] = None):
        """Initialize MD simulation runner.
        
        Args:
            config: MD configuration object
        """
        self.config = config or MDConfig()
        self.trajectory = MDTrajectory()
        self.logger = logging.getLogger(__name__)
        
    def run(self, atoms: Atoms, potential_file: Optional[str] = None) -> MDTrajectory:
        """Run MD simulation.
        
        Args:
            atoms: Initial atomic structure
            potential_file: Path to potential file (overrides config)
            
        Returns:
            MDTrajectory: Simulation trajectory
        """
        if potential_file:
            self.config.potential_file = potential_file
            
        # Create working directory
        Path(self.config.working_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate LAMMPS input
        input_file = self._generate_input(atoms)
        
        # Run simulation
        self._execute_lammps(input_file)
        
        # Parse output
        trajectory = self._parse_output()
        
        return trajectory
    
    def _generate_input(self, atoms: Atoms) -> str:
        """Generate LAMMPS input file."""
        lines = []
        
        # Header
        lines.extend([
            "# LAMMPS input generated by dftlammps",
            "",
            "units metal",
            "dimension 3",
            "boundary p p p",
            "atom_style atomic",
            "",
        ])
        
        # Write structure
        data_file = Path(self.config.working_dir) / "structure.data"
        write_lammps_data(data_file, atoms, atom_style='atomic')
        lines.append(f"read_data {data_file}")
        lines.append("")
        
        # Potential
        lines.extend([
            f"pair_style {self.config.pair_style} {self.config.potential_file}",
            "pair_coeff * *",
            "",
            "neighbor 2.0 bin",
            "neigh_modify every 10 delay 0 check yes",
            "",
        ])
        
        # Output
        dump_file = Path(self.config.working_dir) / "dump.lammpstrj"
        lines.extend([
            f"thermo {self.config.thermo_interval}",
            "thermo_style custom step temp pe ke etotal press",
            f"dump traj all custom {self.config.dump_interval} {dump_file} id type x y z vx vy vz",
            "",
        ])
        
        # Equilibration
        lines.extend([
            "# Equilibration",
            f"velocity all create {self.config.temperature} 12345",
            f"fix nvt_eq all nvt temp {self.config.temperature} {self.config.temperature} 100.0",
            f"run {self.config.nsteps_equil}",
            "unfix nvt_eq",
            "",
        ])
        
        # Production
        if self.config.ensemble == "nvt":
            lines.extend([
                "# Production NVT",
                f"fix nvt_prod all nvt temp {self.config.temperature} {self.config.temperature} 100.0",
            ])
        elif self.config.ensemble == "npt":
            pressure = self.config.pressure or 1.0
            lines.extend([
                "# Production NPT",
                f"fix npt_prod all npt temp {self.config.temperature} {self.config.temperature} 100.0 "
                f"iso {pressure} {pressure} 1000.0",
            ])
        else:  # nve
            lines.extend([
                "# Production NVE",
                "fix nve_prod all nve",
            ])
        
        lines.extend([
            f"run {self.config.nsteps}",
            "",
            "write_restart final.restart",
        ])
        
        # Write input file
        input_file = Path(self.config.working_dir) / "in.lammps"
        with open(input_file, 'w') as f:
            f.write("\n".join(lines))
        
        return str(input_file)
    
    def _execute_lammps(self, input_file: str):
        """Execute LAMMPS simulation."""
        cmd = f"{self.config.lammps_cmd} -in {input_file}"
        if self.config.nprocs > 1:
            cmd = f"mpirun -np {self.config.nprocs} {cmd}"
        
        self.logger.info(f"Running: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=self.config.working_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"LAMMPS failed: {result.stderr}")
        
        self.logger.info("LAMMPS simulation completed")
    
    def _parse_output(self) -> MDTrajectory:
        """Parse LAMMPS output files."""
        trajectory = MDTrajectory()
        
        # Parse thermo output
        log_file = Path(self.config.working_dir) / "log.lammps"
        if log_file.exists():
            self._parse_log(log_file, trajectory)
        
        # Parse trajectory dump
        dump_file = Path(self.config.working_dir) / "dump.lammpstrj"
        if dump_file.exists():
            self._parse_dump(dump_file, trajectory)
        
        return trajectory
    
    def _parse_log(self, log_file: Path, trajectory: MDTrajectory):
        """Parse LAMMPS log file."""
        # Simplified log parsing
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        in_thermo = False
        step_idx = temp_idx = pe_idx = None
        
        for line in lines:
            if line.startswith("Step"):
                in_thermo = True
                headers = line.split()
                if 'Temp' in headers:
                    temp_idx = headers.index('Temp')
                if 'PotEng' in headers or 'Pot' in headers:
                    pe_idx = headers.index('PotEng') if 'PotEng' in headers else headers.index('Pot')
                continue
            
            if in_thermo and line.strip() and not line.startswith('Loop'):
                parts = line.split()
                if temp_idx and len(parts) > temp_idx:
                    try:
                        trajectory.temperatures.append(float(parts[temp_idx]))
                    except ValueError:
                        pass
            
            if line.startswith('Loop'):
                in_thermo = False
    
    def _parse_dump(self, dump_file: Path, trajectory: MDTrajectory):
        """Parse LAMMPS dump file."""
        try:
            from ase.io.lammpsrun import read_lammps_dump_text
            with open(dump_file, 'r') as f:
                content = f.read()
            
            # Parse step by step
            steps = content.split('ITEM: TIMESTEP')[1:]
            for step in steps:
                lines = step.strip().split('\n')
                timestep = int(lines[0])
                
                # Find number of atoms
                natoms = int(lines[2])
                
                # Parse positions
                positions = []
                for i in range(9, 9 + natoms):
                    parts = lines[i].split()
                    positions.append([float(parts[2]), float(parts[3]), float(parts[4])])
                
                trajectory.positions.append(np.array(positions))
                trajectory.time.append(timestep * self.config.timestep)
        except Exception as e:
            self.logger.warning(f"Failed to parse dump file: {e}")


class MDTrajectoryAnalyzer:
    """MD trajectory analyzer.
    
    Provides analysis tools for MD trajectories including:
    - Radial distribution functions
    - Mean square displacement
    - Diffusion coefficients
    - Structure factor
    """
    
    def __init__(self, trajectory: MDTrajectory):
        """Initialize analyzer with trajectory.
        
        Args:
            trajectory: MD trajectory object
        """
        self.trajectory = trajectory
        self.logger = logging.getLogger(__name__)
    
    def compute_rdf(self, bin_width: float = 0.1, r_max: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute radial distribution function.
        
        Args:
            bin_width: Bin width for histogram in Angstrom
            r_max: Maximum radius for calculation
            
        Returns:
            Tuple of (r, g(r)) arrays
        """
        if not self.trajectory.positions:
            raise ValueError("No positions in trajectory")
        
        r_bins = np.arange(0, r_max, bin_width)
        g_r = np.zeros(len(r_bins) - 1)
        
        for positions in self.trajectory.positions:
            # Compute pairwise distances
            n_atoms = len(positions)
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    r = np.linalg.norm(positions[i] - positions[j])
                    if r < r_max:
                        bin_idx = int(r / bin_width)
                        if bin_idx < len(g_r):
                            g_r[bin_idx] += 2  # Count each pair twice
        
        # Normalize
        rho = n_atoms / (4/3 * np.pi * r_max**3)
        shell_volumes = 4/3 * np.pi * ((r_bins[1:])**3 - (r_bins[:-1])**3)
        g_r = g_r / (len(self.trajectory.positions) * n_atoms * rho * shell_volumes)
        
        return r_bins[:-1], g_r
    
    def compute_msd(self, atom_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean square displacement.
        
        Args:
            atom_indices: Indices of atoms to include (None = all)
            
        Returns:
            Tuple of (time, MSD) arrays
        """
        if not self.trajectory.positions:
            raise ValueError("No positions in trajectory")
        
        positions = np.array(self.trajectory.positions)
        time = np.array(self.trajectory.time)
        
        if atom_indices is not None:
            positions = positions[:, atom_indices, :]
        
        # Compute MSD
        msd = np.zeros(len(positions))
        ref_positions = positions[0]
        
        for i, pos in enumerate(positions):
            displacements = pos - ref_positions
            msd[i] = np.mean(np.sum(displacements**2, axis=1))
        
        return time, msd
    
    def compute_diffusion_coefficient(self, atom_indices: Optional[List[int]] = None,
                                     fit_range: Optional[Tuple[int, int]] = None) -> float:
        """Compute diffusion coefficient from MSD.
        
        Args:
            atom_indices: Indices of atoms to include
            fit_range: (start, end) indices for linear fit
            
        Returns:
            Diffusion coefficient in cm^2/s
        """
        time, msd = self.compute_msd(atom_indices)
        
        if fit_range is None:
            # Use last 50% of data
            start_idx = len(time) // 2
            fit_range = (start_idx, len(time))
        
        t_fit = time[fit_range[0]:fit_range[1]]
        msd_fit = msd[fit_range[0]:fit_range[1]]
        
        # Linear fit: MSD = 6*D*t (3D)
        slope = np.polyfit(t_fit, msd_fit, 1)[0]
        D = slope / 6.0  # Angstrom^2/fs
        
        # Convert to cm^2/s
        D_cm2s = D * 1e-16 / 1e-15  # Angstrom^2/fs to cm^2/s
        
        return D_cm2s
    
    def compute_ionic_conductivity(self, D: float, temperature: float, 
                                   concentration: float, charge: float = 1.0) -> float:
        """Compute ionic conductivity from diffusion coefficient.
        
        Uses Nernst-Einstein relation: sigma = n * q^2 * D / (k_B * T)
        
        Args:
            D: Diffusion coefficient in cm^2/s
            temperature: Temperature in K
            concentration: Carrier concentration in cm^-3
            charge: Ion charge in elementary charge units
            
        Returns:
            Ionic conductivity in S/cm
        """
        k_B = 1.380649e-23  # J/K
        q = charge * 1.602176634e-19  # C
        
        # Nernst-Einstein
        sigma = concentration * q**2 * D / (k_B * temperature)  # S/cm
        
        return sigma
    
    def get_summary(self) -> Dict:
        """Get analysis summary."""
        summary = {
            'n_frames': len(self.trajectory.positions),
            'simulation_time_ps': self.trajectory.time[-1] if self.trajectory.time else 0,
            'average_temperature': np.mean(self.trajectory.temperatures) if self.trajectory.temperatures else 0,
        }
        
        if self.trajectory.temperatures:
            summary['temperature_std'] = np.std(self.trajectory.temperatures)
        
        return summary
