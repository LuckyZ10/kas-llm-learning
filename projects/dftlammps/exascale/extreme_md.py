#!/usr/bin/env python3
"""
extreme_md.py - 极端条件分子动力学模块

支持冲击波模拟、超高压相变和高温熔化模拟。
适用于百万原子体系的极端条件研究。

Author: DFT-LAMMPS Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum, auto
import logging
import time
from collections import deque
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShockMethod(Enum):
    """Shock wave simulation methods"""
    PISTON = "piston"           # Moving piston method
    MOMENTUM_MIRROR = "momentum_mirror"  # Momentum mirror (NEMD)
    BILAYER = "bilayer"         # Bilayer shock
    VELOCITY_RAM = "velocity_ram"  # Velocity ramping
    HUGONIOT = "hugoniot"       # Hugoniot state calculation


class PhaseTransitionMethod(Enum):
    """Phase transition detection methods"""
    LINDEMANN = "lindemann"     # Lindemann criterion
    RDF_ANALYSIS = "rdf"        # Radial distribution function
    CNA = "cna"                 # Common neighbor analysis
    VORONOI = "voronoi"         # Voronoi analysis
    ORDER_PARAMETER = "order"   # Order parameter tracking


@dataclass
class ExtremeMDConfig:
    """Configuration for extreme conditions MD"""
    # Time integration
    timestep: float = 1.0           # fs
    n_steps: int = 10000
    
    # Thermostat/Barostat
    temperature: float = 300.0      # K
    pressure: float = 1.0           # GPa
    target_temperature: float = 300.0
    target_pressure: float = 1.0
    
    # Shock parameters
    shock_method: ShockMethod = ShockMethod.PISTON
    piston_velocity: float = 1.0    # km/s
    shock_direction: Tuple[int, int, int] = (1, 0, 0)
    
    # Phase transition
    phase_detect_method: PhaseTransitionMethod = PhaseTransitionMethod.LINDEMANN
    melting_threshold: float = 0.15  # Lindemann parameter
    
    # Output
    dump_frequency: int = 100
    thermo_frequency: int = 10
    
    # Numerical stability
    max_displacement: float = 0.5   # Angstrom
    energy_drift_tolerance: float = 1e-4


class ShockWaveSimulator:
    """
    Shock wave simulation using various methods
    
    Supports piston, momentum mirror, and bilayer methods for
    generating shock waves in materials.
    """
    
    def __init__(self, config: ExtremeMDConfig):
        self.config = config
        self.piston_position = None
        self.piston_velocity = config.piston_velocity * 1e5  # km/s to Angstrom/fs
        self.shock_direction = np.array(config.shock_direction, dtype=float)
        self.shock_direction /= np.linalg.norm(self.shock_direction)
        
        # Shock front tracking
        self.shock_front_position = None
        self.shock_velocity = None
        self.particle_velocity_history = deque(maxlen=100)
        
        # Hugoniot state variables
        self.P0 = None  # Initial pressure
        self.V0 = None  # Initial volume
        self.E0 = None  # Initial energy
        
    def initialize_shock(self, positions: np.ndarray, box: np.ndarray):
        """Initialize shock simulation setup"""
        self.box = box.copy()
        self.P0 = self.config.pressure
        self.V0 = np.linalg.det(box)
        
        if self.config.shock_method == ShockMethod.PISTON:
            # Set piston at one end of the box
            self.piston_position = np.min(positions @ self.shock_direction)
            logger.info(f"Piston initialized at position {self.piston_position:.2f} A")
            
        elif self.config.shock_method == ShockMethod.MOMENTUM_MIRROR:
            # Set momentum mirror at the shock boundary
            self.mirror_position = np.min(positions @ self.shock_direction)
            logger.info(f"Momentum mirror at position {self.mirror_position:.2f} A")
    
    def apply_piston_boundary(self, positions: np.ndarray, 
                             velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply moving piston boundary condition
        
        Args:
            positions: Atomic positions
            velocities: Atomic velocities
            
        Returns:
            Modified positions and velocities
        """
        # Update piston position
        self.piston_position += self.piston_velocity * self.config.timestep
        
        # Find atoms that would be behind the piston
        positions_proj = positions @ self.shock_direction
        behind_piston = positions_proj < self.piston_position
        
        # Push atoms with piston velocity
        velocities[behind_piston] = self.piston_velocity * self.shock_direction
        
        # Ensure atoms don't cross piston
        positions[behind_piston] = (
            self.piston_position * self.shock_direction + 
            positions[behind_piston] - np.outer(positions_proj[behind_piston], self.shock_direction)
        )
        
        return positions, velocities
    
    def apply_momentum_mirror(self, positions: np.ndarray,
                              velocities: np.ndarray) -> np.ndarray:
        """
        Apply momentum mirror boundary condition (NEMD)
        
        Reflects momenta of atoms crossing the mirror plane,
        effectively creating a shock wave.
        """
        positions_proj = positions @ self.shock_direction
        crossed_mirror = positions_proj < self.mirror_position
        
        # Reflect velocity component normal to mirror
        v_proj = velocities @ self.shock_direction
        velocities[crossed_mirror] -= 2 * np.outer(
            v_proj[crossed_mirror], self.shock_direction
        )
        
        return velocities
    
    def apply_velocity_ramp(self, positions: np.ndarray,
                           velocities: np.ndarray,
                           shock_position: float) -> np.ndarray:
        """
        Apply velocity ramp behind shock front
        
        Creates smooth velocity profile behind the shock.
        """
        positions_proj = positions @ self.shock_direction
        behind_shock = positions_proj < shock_position
        
        # Linear velocity ramp
        distances = shock_position - positions_proj[behind_shock]
        ramp_factor = np.clip(distances / 10.0, 0, 1)  # 10 A ramp width
        
        velocities[behind_shock] += np.outer(
            ramp_factor * self.piston_velocity, self.shock_direction
        )
        
        return velocities
    
    def track_shock_front(self, positions: np.ndarray,
                         velocities: np.ndarray,
                         temperatures: np.ndarray) -> float:
        """
        Track shock front position using temperature gradient
        
        Returns:
            Shock front position along shock direction
        """
        positions_proj = positions @ self.shock_direction
        
        # Sort by position
        sort_idx = np.argsort(positions_proj)
        sorted_pos = positions_proj[sort_idx]
        sorted_temp = temperatures[sort_idx]
        
        # Find maximum temperature gradient
        temp_grad = np.gradient(sorted_temp, sorted_pos)
        shock_idx = np.argmax(np.abs(temp_grad))
        
        self.shock_front_position = sorted_pos[shock_idx]
        
        # Estimate shock velocity from particle velocities
        behind_shock = positions_proj < self.shock_front_position
        if np.any(behind_shock):
            avg_velocity = np.mean(velocities[behind_shock] @ self.shock_direction)
            self.particle_velocity_history.append(avg_velocity)
            
            if len(self.particle_velocity_history) > 10:
                self.shock_velocity = np.mean(self.particle_velocity_history)
        
        return self.shock_front_position
    
    def compute_hugoniot_state(self, positions: np.ndarray,
                              velocities: np.ndarray,
                              forces: np.ndarray,
                              energy: float,
                              pressure_tensor: np.ndarray) -> Dict[str, float]:
        """
        Compute Hugoniot state variables
        
        Uses Rankine-Hugoniot jump conditions to determine
        post-shock state.
        """
        # Current state
        V = np.linalg.det(self.box)
        P = np.trace(pressure_tensor) / 3
        E = energy
        
        # Particle velocity (average behind shock)
        positions_proj = positions @ self.shock_direction
        behind_shock = positions_proj < self.shock_front_position if self.shock_front_position else np.ones(len(positions), dtype=bool)
        up = np.mean(velocities[behind_shock] @ self.shock_direction) if np.any(behind_shock) else 0.0
        
        # Estimate shock velocity
        if self.shock_velocity is None:
            Us = self.piston_velocity
        else:
            Us = self.shock_velocity
        
        # Hugoniot relations
        # Mass: rho0 * Us = rho * (Us - up)
        # Momentum: P - P0 = rho0 * Us * up
        # Energy: E - E0 = 0.5 * (P + P0) * (V0 - V)
        
        hugoniot = {
            'P0': self.P0,
            'V0': self.V0,
            'E0': self.E0,
            'P': P,
            'V': V,
            'E': E,
            'Us': Us,
            'up': up,
            'compression': self.V0 / V if V > 0 else 1.0,
            'particle_velocity': up
        }
        
        return hugoniot


class PhaseTransitionDetector:
    """
    Detect and analyze phase transitions under extreme conditions
    
    Uses various methods including Lindemann criterion,
    RDF analysis, and common neighbor analysis.
    """
    
    def __init__(self, config: ExtremeMDConfig):
        self.config = config
        self.phase_history = []
        self.lindemann_history = []
        self.structure_types = []
        
    def compute_lindemann_parameter(self, positions: np.ndarray,
                                    box: np.ndarray,
                                    n_neighbors: int = 12) -> float:
        """
        Compute Lindemann melting criterion
        
        The Lindemann parameter is the ratio of root-mean-square
        displacement to nearest-neighbor distance. Melting typically
        occurs when this exceeds ~0.15.
        """
        from scipy.spatial import cKDTree
        
        # Build neighbor list
        tree = cKDTree(positions, boxsize=np.diag(box))
        
        # Find nearest neighbors for each atom
        distances, indices = tree.query(positions, k=n_neighbors+1)
        
        # Compute RMS displacement from equilibrium positions
        # Use time averaging (simplified here)
        nn_distances = distances[:, 1:]  # Exclude self
        mean_nn_dist = np.mean(nn_distances)
        
        # RMS displacement (simplified instantaneous version)
        # In practice, this should be time-averaged
        rmsd = np.std(nn_distances) / np.sqrt(2)  # Factor of sqrt(2) for relative displacement
        
        lindemann_param = rmsd / mean_nn_dist
        self.lindemann_history.append(lindemann_param)
        
        return lindemann_param
    
    def compute_radial_distribution_function(self, positions: np.ndarray,
                                             box: np.ndarray,
                                             n_bins: int = 100,
                                             r_max: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radial distribution function g(r)
        
        Used to identify structural changes and phase transitions.
        """
        from scipy.spatial import cKDTree
        
        if r_max is None:
            r_max = np.min(np.diag(box)) / 2
        
        # Build tree
        tree = cKDTree(positions, boxsize=np.diag(box))
        
        # Compute pairwise distances
        distances = tree.sparse_distance_matrix(tree, r_max, output_type='dok_matrix')
        
        # Create histogram
        r_bins = np.linspace(0, r_max, n_bins + 1)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        
        # Get distance values
        dist_values = np.array(list(distances.keys()))
        if len(dist_values) > 0:
            dist_list = [distances[k] for k in dist_values if distances[k] > 0]
            hist, _ = np.histogram(dist_list, bins=r_bins)
        else:
            hist = np.zeros(n_bins)
        
        # Normalize
        dr = r_bins[1] - r_bins[0]
        rho = len(positions) / np.linalg.det(box)
        
        # g(r) normalization
        shell_volumes = 4 * np.pi * r_centers**2 * dr
        normalization = rho * len(positions) * shell_volumes
        g_r = hist / normalization
        
        return r_centers, g_r
    
    def common_neighbor_analysis(self, positions: np.ndarray,
                                  box: np.ndarray,
                                  cutoff: float = 3.5) -> np.ndarray:
        """
        Common Neighbor Analysis (CNA) for structure identification
        
        Returns structure type for each atom:
        0: Unknown/liquid
        1: FCC
        2: BCC
        3: HCP
        """
        from scipy.spatial import cKDTree
        
        tree = cKDTree(positions, boxsize=np.diag(box))
        structure_types = np.zeros(len(positions), dtype=int)
        
        for i, pos in enumerate(positions):
            # Find neighbors within cutoff
            neighbors = tree.query_ball_point(pos, r=cutoff)
            neighbors.remove(i)  # Remove self
            n_neighbors = len(neighbors)
            
            if n_neighbors == 12:
                # Could be FCC or HCP
                # Count common neighbors between pairs
                cn_count = 0
                for j_idx, j in enumerate(neighbors):
                    for k in neighbors[j_idx+1:]:
                        # Check if j and k are neighbors
                        dist = np.linalg.norm(positions[j] - positions[k])
                        if dist < cutoff:
                            cn_count += 1
                
                # FCC: 4 pairs with 2 common neighbors
                # HCP: 3 pairs with 2 common neighbors + 3 pairs with 3 common neighbors
                if cn_count == 24:  # Simplified check
                    structure_types[i] = 1  # FCC
                elif cn_count == 21:
                    structure_types[i] = 3  # HCP
                    
            elif n_neighbors == 14:
                # Could be BCC
                structure_types[i] = 2  # BCC
        
        self.structure_types = structure_types
        return structure_types
    
    def detect_melting(self, lindemann_param: float = None,
                      positions: np.ndarray = None,
                      box: np.ndarray = None) -> Tuple[bool, float]:
        """
        Detect melting using Lindemann criterion
        
        Returns:
            (is_molten, lindemann_parameter)
        """
        if lindemann_param is None:
            if positions is None or box is None:
                raise ValueError("Need positions and box to compute Lindemann parameter")
            lindemann_param = self.compute_lindemann_parameter(positions, box)
        
        is_molten = lindemann_param > self.config.melting_threshold
        
        if is_molten:
            logger.info(f"Melting detected! Lindemann parameter = {lindemann_param:.3f}")
        
        return is_molten, lindemann_param
    
    def detect_phase_transition(self, positions: np.ndarray,
                                box: np.ndarray,
                                temperature: float,
                                pressure: float) -> Dict[str, Any]:
        """
        Comprehensive phase transition detection
        
        Returns dictionary with all phase information.
        """
        # Lindemann parameter
        lindemann = self.compute_lindemann_parameter(positions, box)
        is_molten, _ = self.detect_melting(lindemann)
        
        # Structure analysis
        structure_types = self.common_neighbor_analysis(positions, box)
        fcc_fraction = np.sum(structure_types == 1) / len(structure_types)
        bcc_fraction = np.sum(structure_types == 2) / len(structure_types)
        hcp_fraction = np.sum(structure_types == 3) / len(structure_types)
        liquid_fraction = np.sum(structure_types == 0) / len(structure_types)
        
        # RDF analysis
        r, g_r = self.compute_radial_distribution_function(positions, box)
        
        # Find peaks in g(r) for coordination analysis
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(g_r, height=0.5)
        n_peaks = len(peaks)
        
        phase_info = {
            'lindemann_parameter': lindemann,
            'is_molten': is_molten,
            'fcc_fraction': fcc_fraction,
            'bcc_fraction': bcc_fraction,
            'hcp_fraction': hcp_fraction,
            'liquid_fraction': liquid_fraction,
            'dominant_structure': max(['fcc', 'bcc', 'hcp', 'liquid'],
                                       key=lambda x: locals()[f'{x}_fraction']),
            'rdf_peaks': n_peaks,
            'temperature': temperature,
            'pressure': pressure
        }
        
        self.phase_history.append(phase_info)
        
        return phase_info


class ExtremeMD:
    """
    Main class for extreme conditions molecular dynamics
    
    Integrates shock wave simulation and phase transition detection
    for studying materials under extreme conditions.
    """
    
    def __init__(self, config: ExtremeMDConfig):
        self.config = config
        self.shock_sim = ShockWaveSimulator(config)
        self.phase_detector = PhaseTransitionDetector(config)
        
        # Simulation state
        self.positions = None
        self.velocities = None
        self.forces = None
        self.box = None
        self.masses = None
        self.step = 0
        
        # Output
        self.trajectory = []
        self.thermo_data = []
        
    def initialize(self, positions: np.ndarray, velocities: np.ndarray,
                   masses: np.ndarray, box: np.ndarray):
        """Initialize simulation"""
        self.positions = positions.copy()
        self.velocities = velocities.copy()
        self.masses = masses
        self.box = box.copy()
        self.forces = np.zeros_like(positions)
        
        # Initialize shock simulation
        self.shock_sim.initialize_shock(positions, box)
        
        logger.info(f"Initialized extreme MD with {len(positions)} atoms")
        logger.info(f"Target temperature: {self.config.target_temperature} K")
        logger.info(f"Target pressure: {self.config.target_pressure} GPa")
    
    def compute_forces(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute forces (placeholder - integrate with LAMMPS or other)
        
        In practice, this would call LAMMPS or a force field calculator.
        """
        # Placeholder: simple harmonic forces
        forces = np.zeros_like(positions)
        
        # Add some interatomic forces (simplified)
        from scipy.spatial import cKDTree
        tree = cKDTree(positions, boxsize=np.diag(self.box))
        
        for i, pos in enumerate(positions):
            neighbors = tree.query_ball_point(pos, r=3.0)
            for j in neighbors:
                if i != j:
                    dr = positions[j] - pos
                    # Minimum image convention
                    dr -= self.box @ np.round(np.linalg.solve(self.box, dr))
                    r = np.linalg.norm(dr)
                    if r > 0.1:
                        # Lennard-Jones like repulsion
                        f = 24 * (2/r**13 - 1/r**7) * dr / r
                        forces[i] += f
        
        return forces
    
    def integrate_verlet(self):
        """Velocity Verlet integration"""
        dt = self.config.timestep
        
        # Half-step velocity update
        self.velocities += 0.5 * dt * self.forces / self.masses[:, None]
        
        # Full position update
        self.positions += dt * self.velocities
        
        # Apply boundary conditions
        self.positions = self._apply_pbc(self.positions)
        
        # Apply shock boundary conditions
        if self.config.shock_method == ShockMethod.PISTON:
            self.positions, self.velocities = self.shock_sim.apply_piston_boundary(
                self.positions, self.velocities
            )
        elif self.config.shock_method == ShockMethod.MOMENTUM_MIRROR:
            self.velocities = self.shock_sim.apply_momentum_mirror(
                self.positions, self.velocities
            )
        
        # Compute new forces
        self.forces = self.compute_forces(self.positions)
        
        # Half-step velocity update
        self.velocities += 0.5 * dt * self.forces / self.masses[:, None]
    
    def _apply_pbc(self, positions: np.ndarray) -> np.ndarray:
        """Apply periodic boundary conditions"""
        frac = np.linalg.solve(self.box.T, positions.T).T
        frac = frac % 1.0
        return frac @ self.box.T
    
    def compute_temperature(self) -> float:
        """Compute instantaneous temperature"""
        kinetic_energy = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
        n_dof = 3 * len(self.positions) - 3  # Subtract center of mass
        kB = 8.617333e-5  # eV/K
        return 2 * kinetic_energy / (n_dof * kB) * 1.602e-19 * 1e10  # Convert to K
    
    def compute_pressure(self) -> np.ndarray:
        """Compute pressure tensor"""
        volume = np.linalg.det(self.box)
        
        # Kinetic contribution
        kinetic_tensor = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                kinetic_tensor[i, j] = np.sum(
                    self.masses * self.velocities[:, i] * self.velocities[:, j]
                )
        
        # Virial contribution (simplified)
        virial_tensor = np.zeros((3, 3))
        for i, pos in enumerate(self.positions):
            virial_tensor += np.outer(pos, self.forces[i])
        
        pressure_tensor = (kinetic_tensor + virial_tensor) / volume
        return pressure_tensor
    
    def run(self):
        """Run the simulation"""
        logger.info("Starting extreme conditions MD simulation")
        
        start_time = time.time()
        
        for step in range(self.config.n_steps):
            self.step = step
            
            # Integrate equations of motion
            self.integrate_verlet()
            
            # Compute properties
            if step % self.config.thermo_frequency == 0:
                temp = self.compute_temperature()
                pressure_tensor = self.compute_pressure()
                pressure = np.trace(pressure_tensor) / 3
                
                # Phase detection
                phase_info = self.phase_detector.detect_phase_transition(
                    self.positions, self.box, temp, pressure
                )
                
                # Shock tracking
                shock_pos = self.shock_sim.track_shock_front(
                    self.positions, self.velocities,
                    np.full(len(self.positions), temp)  # Simplified temperature array
                )
                
                # Hugoniot state
                energy = self.compute_total_energy()
                hugoniot = self.shock_sim.compute_hugoniot_state(
                    self.positions, self.velocities, self.forces,
                    energy, pressure_tensor
                )
                
                self.thermo_data.append({
                    'step': step,
                    'temperature': temp,
                    'pressure': pressure,
                    'energy': energy,
                    'shock_position': shock_pos,
                    **hugoniot,
                    **phase_info
                })
                
                if step % (self.config.thermo_frequency * 10) == 0:
                    logger.info(f"Step {step}: T={temp:.1f}K, P={pressure:.2f}GPa, "
                               f"L={phase_info['lindemann_parameter']:.3f}")
            
            # Save trajectory
            if step % self.config.dump_frequency == 0:
                self.trajectory.append({
                    'step': step,
                    'positions': self.positions.copy(),
                    'velocities': self.velocities.copy(),
                    'box': self.box.copy()
                })
        
        elapsed = time.time() - start_time
        logger.info(f"Simulation completed in {elapsed:.1f} seconds")
        logger.info(f"Performance: {self.config.n_steps / elapsed:.1f} steps/second")
    
    def compute_total_energy(self) -> float:
        """Compute total energy"""
        kinetic = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
        potential = 0.0  # Would need proper potential energy calculation
        return kinetic + potential
    
    def get_hugoniot_curve(self) -> List[Dict]:
        """Extract Hugoniot curve from simulation data"""
        hugoniot_data = []
        for data in self.thermo_data:
            hugoniot_data.append({
                'pressure': data['P'],
                'volume': data['V'],
                'energy': data['E'],
                'particle_velocity': data['up'],
                'shock_velocity': data['Us'],
                'compression': data['compression']
            })
        return hugoniot_data


def example_shock_simulation():
    """Example: Shock wave simulation in iron"""
    config = ExtremeMDConfig(
        shock_method=ShockMethod.PISTON,
        piston_velocity=5.0,  # km/s
        shock_direction=(1, 0, 0),
        timestep=0.5,
        n_steps=5000,
        target_temperature=300.0,
        melting_threshold=0.15
    )
    
    # Create BCC iron sample
    a = 2.87  # Lattice constant
    n = 10
    positions = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                positions.append([i*a, j*a, k*a])
                positions.append([i*a+a/2, j*a+a/2, k*a+a/2])
    
    positions = np.array(positions)
    n_atoms = len(positions)
    
    # Initialize velocities (Maxwell-Boltzmann)
    kB = 8.617333e-5
    T = 300.0
    masses = np.full(n_atoms, 55.845)  # Fe mass in amu
    velocities = np.random.normal(0, np.sqrt(kB * T / masses[0]), (n_atoms, 3))
    
    # Center of mass velocity
    velocities -= np.mean(velocities, axis=0)
    
    box = np.eye(3) * n * a
    
    # Run simulation
    md = ExtremeMD(config)
    md.initialize(positions, velocities, masses, box)
    md.run()
    
    # Print results
    hugoniot = md.get_hugoniot_curve()
    print("\nHugoniot states:")
    for i, state in enumerate(hugoniot[::len(hugoniot)//5]):
        print(f"  P={state['pressure']:.2f} GPa, "
              f"up={state['particle_velocity']:.2f} km/s, "
              f"Us={state['shock_velocity']:.2f} km/s")
    
    return md


if __name__ == "__main__":
    example_shock_simulation()
