#!/usr/bin/env python3
"""
applications.py - Extreme conditions application cases

Application cases for extreme-scale simulations:
- Planetary core simulation
- Nuclear materials under extreme conditions
- Asteroid impact simulation

Author: DFT-LAMMPS Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import time
from pathlib import Path

# Import exascale modules
try:
    from .exascale_dft import ExascaleDFT, ExascaleDFTConfig, LinearScalingMethod
    from .extreme_md import ExtremeMD, ExtremeMDConfig, ShockMethod, PhaseTransitionMethod
    from ..machine_limits.parallel_optimization import MillionCoreOptimizer, ParallelConfig
    from ..machine_limits.memory_optimization import MemoryManager, MemoryConfig
    from ..machine_limits.checkpoint_restart import CheckpointManager, CheckpointConfig
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from exascale.exascale_dft import ExascaleDFT, ExascaleDFTConfig, LinearScalingMethod
    from exascale.extreme_md import ExtremeMD, ExtremeMDConfig, ShockMethod, PhaseTransitionMethod
    from machine_limits.parallel_optimization import MillionCoreOptimizer, ParallelConfig
    from machine_limits.memory_optimization import MemoryManager, MemoryConfig
    from machine_limits.checkpoint_restart import CheckpointManager, CheckpointConfig
    from exascale.exascale_dft import ExascaleDFT, ExascaleDFTConfig, LinearScalingMethod
    from exascale.extreme_md import ExtremeMD, ExtremeMDConfig, ShockMethod, PhaseTransitionMethod
    from machine_limits.parallel_optimization import MillionCoreOptimizer, ParallelConfig
    from machine_limits.memory_optimization import MemoryManager, MemoryConfig
    from machine_limits.checkpoint_restart import CheckpointManager, CheckpointConfig

logger = logging.getLogger(__name__)


@dataclass
class PlanetaryCoreConfig:
    """Configuration for planetary core simulation"""
    # Core composition (Earth-like)
    inner_core_fraction: float = 0.16  # Fraction of core radius
    light_element_fraction: float =0.10  # S/Si fraction in outer core
    
    # Physical conditions
    inner_core_temp: float = 6000.0    # K
    outer_core_temp: float = 4000.0    # K
    center_pressure: float = 360.0     # GPa
    
    # Simulation parameters
    n_atoms: int = 1000000
    core_radius_km: float = 3480.0     # Earth core radius


@dataclass
class NuclearMaterialConfig:
    """Configuration for nuclear material simulation"""
    # Material properties
    material: str = "UO2"              # Nuclear fuel material
    enrichment: float = 0.05           # U-235 enrichment
    
    # Extreme conditions
    max_temperature: float = 3000.0    # K (melting point ~3150K)
    max_pressure: float = 50.0         # GPa
    irradiation_dpa: float = 100.0     # Displacements per atom
    
    # Simulation parameters
    n_atoms: int = 500000
    simulation_time_ns: float = 10.0


@dataclass
class AsteroidImpactConfig:
    """Configuration for asteroid impact simulation"""
    # Impactor properties
    impactor_diameter_km: float = 10.0
    impactor_velocity_kms: float = 20.0  # km/s
    impactor_density: float = 3000.0     # kg/m^3
    
    # Target properties
    target_material: str = "basalt"
    target_thickness_km: float = 50.0
    
    # Impact parameters
    impact_angle_degrees: float = 45.0
    gravity_ms2: float = 9.8
    
    # Simulation parameters
    n_atoms: int = 5000000
    simulation_time_s: float = 10.0


class PlanetaryCoreSimulator:
    """
    Planetary core formation and evolution simulator
    
    Simulates iron-nickel core formation with phase transitions
    between liquid outer core and solid inner core.
    """
    
    def __init__(self, config: PlanetaryCoreConfig):
        self.config = config
        
        # Initialize components
        self.dft_config = ExascaleDFTConfig(
            method=LinearScalingMethod.ONETEP,
            localization_radius=6.0,
            max_scf_iterations=50,
            scf_tolerance=1e-5,
            use_gpu=False,
            domain_decomposition=(8, 8, 8)
        )
        
        self.md_config = ExtremeMDConfig(
            shock_method=ShockMethod.PISTON,
            timestep=1.0,
            n_steps=10000,
            target_temperature=config.inner_core_temp,
            target_pressure=config.center_pressure,
            phase_detect_method=PhaseTransitionMethod.LINDEMANN
        )
        
        self.parallel_config = ParallelConfig(
            network_topology="torus",
            torus_dimensions=(16, 16, 64),
            dynamic_load_balance=True
        )
        
        self.checkpoint_config = CheckpointConfig(
            checkpoint_steps=500,
            format=CheckpointFormat.HDF5,
            enable_async=True
        )
        
        # Components
        self.dft = None
        self.md = None
        self.optimizer = None
        self.checkpoint_mgr = None
    
    def create_initial_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create initial core structure
        
        Returns:
            (positions, atomic_numbers, box)
        """
        n = int((self.config.n_atoms / 2) ** (1/3)) + 1
        
        # BCC iron structure for inner core
        a_fe = 2.87  # Angstrom
        positions = []
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len(positions) < self.config.n_atoms // 2:
                        x, y, z = i*a_fe, j*a_fe, k*a_fe
                        positions.append([x, y, z])
                        if len(positions) < self.config.n_atoms // 2:
                            positions.append([x+a_fe/2, y+a_fe/2, z+a_fe/2])
        
        # FCC nickel structure for outer core
        a_ni = 3.52
        offset = n * a_fe
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len(positions) < self.config.n_atoms:
                        x, y, z = offset + i*a_ni, j*a_ni, k*a_ni
                        positions.append([x, y, z])
                        if len(positions) < self.config.n_atoms:
                            positions.append([x+a_ni/2, y+a_ni/2, z])
                        if len(positions) < self.config.n_atoms:
                            positions.append([x+a_ni/2, z, z+a_ni/2])
                        if len(positions) < self.config.n_atoms:
                            positions.append([x, y+a_ni/2, z+a_ni/2])
        
        positions = np.array(positions[:self.config.n_atoms])
        
        # Atomic numbers (Fe=26, Ni=28)
        n_fe = self.config.n_atoms // 2
        atomic_numbers = np.concatenate([
            np.full(n_fe, 26),
            np.full(self.config.n_atoms - n_fe, 28)
        ])
        
        # Simulation box
        box_size = max(offset + n * a_ni, n * a_fe) + 10
        box = np.eye(3) * box_size
        
        return positions, atomic_numbers, box
    
    def apply_temperature_gradient(self, positions: np.ndarray,
                                   box: np.ndarray) -> np.ndarray:
        """
        Apply radial temperature gradient
        
        Args:
            positions: Atomic positions
            box: Simulation box
            
        Returns:
            Temperature at each atom
        """
        center = box.diagonal() / 2
        
        # Calculate radial distance from center
        r = np.linalg.norm(positions - center, axis=1)
        r_max = np.max(r)
        
        # Linear temperature profile
        T_inner = self.config.inner_core_temp
        T_outer = self.config.outer_core_temp
        
        temperatures = T_outer + (T_inner - T_outer) * (1 - r / r_max)
        
        return temperatures
    
    def detect_inner_core_boundary(self, positions: np.ndarray,
                                   temperatures: np.ndarray,
                                   phase_info: Dict) -> float:
        """
        Detect inner core-outer core boundary
        
        Returns:
            Radius of inner core boundary in Angstrom
        """
        # Find where melting occurs
        center = np.mean(positions, axis=0)
        r = np.linalg.norm(positions - center, axis=1)
        
        # Use Lindemann criterion
        is_solid = phase_info.get('lindemann_parameter', 0) < 0.15
        
        if is_solid:
            solid_mask = np.array([is_solid] * len(positions))
            if np.any(solid_mask):
                inner_core_radius = np.max(r[solid_mask])
                return inner_core_radius
        
        # Estimate from temperature
        melting_temp = 5000.0  # Approximate melting temperature
        inner_core_radius = np.max(r[temperatures > melting_temp])
        
        return inner_core_radius
    
    def run(self):
        """Run planetary core simulation"""
        logger.info("Starting planetary core simulation")
        logger.info(f"Target atoms: {self.config.n_atoms}")
        logger.info(f"Temperature range: {self.config.outer_core_temp} - "
                   f"{self.config.inner_core_temp} K")
        
        # Create initial structure
        positions, atomic_numbers, box = self.create_initial_structure()
        
        # Initialize components
        self.optimizer = MillionCoreOptimizer(self.parallel_config)
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_config)
        
        # Run DFT calculation
        logger.info("Running initial DFT calculation...")
        self.dft = ExascaleDFT(self.dft_config)
        self.dft.initialize_system(positions, atomic_numbers, box)
        energy = self.dft.run_scf()
        
        logger.info(f"Initial energy: {energy:.4f} eV")
        
        # Apply temperature gradient
        temperatures = self.apply_temperature_gradient(positions, box)
        
        # Initialize velocities from Maxwell-Boltzmann distribution
        kB = 8.617333e-5
        masses = np.where(atomic_numbers == 26, 55.845, 58.693)
        velocities = np.zeros((len(positions), 3))
        
        for i, T in enumerate(temperatures):
            sigma = np.sqrt(kB * T / masses[i])
            velocities[i] = np.random.normal(0, sigma, 3)
        
        # Run MD simulation
        logger.info("Running MD simulation...")
        self.md = ExtremeMD(self.md_config)
        self.md.initialize(positions, velocities, masses, box)
        self.md.run()
        
        # Analyze results
        final_positions = self.md.positions
        phase_info = self.md.phase_detector.phase_history[-1] if self.md.phase_detector.phase_history else {}
        
        inner_core_radius = self.detect_inner_core_boundary(
            final_positions, temperatures, phase_info
        )
        
        # Convert to km
        inner_core_radius_km = inner_core_radius * 1e-13 * 1e-3  # A to km
        
        results = {
            'initial_energy_ev': energy,
            'final_temperature_K': self.md.compute_temperature(),
            'inner_core_radius_km': inner_core_radius_km,
            'inner_core_fraction': inner_core_radius_km / self.config.core_radius_km,
            'phase_info': phase_info,
            'trajectory': self.md.trajectory,
            'thermo_data': self.md.thermo_data
        }
        
        logger.info("Planetary core simulation completed")
        logger.info(f"Inner core radius: {inner_core_radius_km:.1f} km")
        
        return results


class NuclearMaterialSimulator:
    """
    Nuclear material behavior under reactor conditions
    
    Simulates fuel behavior under extreme temperature,
    pressure, and radiation damage.
    """
    
    def __init__(self, config: NuclearMaterialConfig):
        self.config = config
        
        # Configure for nuclear material
        self.md_config = ExtremeMDConfig(
            timestep=0.5,  # Small timestep for accurate collision cascades
            n_steps=50000,
            target_temperature=300.0,
            target_pressure=0.1,
            shock_method=ShockMethod.VELOCITY_RAM,
            phase_detect_method=PhaseTransitionMethod.CNA,
            melting_threshold=0.12  # Lower threshold for oxides
        )
        
        self.checkpoint_config = CheckpointConfig(
            checkpoint_steps=1000,
            enable_fault_tolerance=True
        )
        
        self.md = None
        self.checkpoint_mgr = None
        
        # Defect tracking
        self.defects_history = []
    
    def create_uo2_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create UO2 fluorite structure
        
        Returns:
            (positions, atomic_numbers, box)
        """
        # UO2 fluorite structure
        a = 5.47  # Lattice constant
        
        # Number of unit cells
        n_cells = int((self.config.n_atoms / 12) ** (1/3)) + 1
        
        positions = []
        atomic_numbers = []
        
        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    base = np.array([i, j, k]) * a
                    
                    # U positions (fcc)
                    u_sites = [
                        [0, 0, 0],
                        [0.5, 0.5, 0],
                        [0.5, 0, 0.5],
                        [0, 0.5, 0.5]
                    ]
                    
                    for site in u_sites:
                        if len(positions) < self.config.n_atoms:
                            positions.append(base + np.array(site) * a)
                            atomic_numbers.append(92)  # U
                    
                    # O positions
                    o_sites = [
                        [0.25, 0.25, 0.25],
                        [0.75, 0.75, 0.25],
                        [0.75, 0.25, 0.75],
                        [0.25, 0.75, 0.75],
                        [0.75, 0.25, 0.25],
                        [0.25, 0.75, 0.25],
                        [0.25, 0.25, 0.75],
                        [0.75, 0.75, 0.75]
                    ]
                    
                    for site in o_sites:
                        if len(positions) < self.config.n_atoms:
                            positions.append(base + np.array(site) * a)
                            atomic_numbers.append(8)  # O
        
        positions = np.array(positions[:self.config.n_atoms])
        atomic_numbers = np.array(atomic_numbers[:self.config.n_atoms])
        
        box = np.eye(3) * n_cells * a
        
        return positions, atomic_numbers, box
    
    def introduce_radiation_damage(self, positions: np.ndarray,
                                   atomic_numbers: np.ndarray,
                                   box: np.ndarray,
                                   dpa: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Introduce radiation damage (displacement cascades)
        
        Args:
            positions: Atomic positions
            atomic_numbers: Atomic numbers
            box: Simulation box
            dpa: Displacements per atom
            
        Returns:
            (modified_positions, pkas) - Modified positions and primary knock-on atoms
        """
        n_displacements = int(len(positions) * dpa / 100)  # Scale down for initial structure
        
        # Select PKAs (Primary Knock-on Atoms)
        pka_indices = np.random.choice(len(positions), 
                                      size=min(n_displacements, len(positions)//10),
                                      replace=False)
        
        # Assign high kinetic energy to PKAs (typical recoil energy 1-100 keV)
        velocities = np.zeros((len(positions), 3))
        
        for pka in pka_indices:
            # Random direction
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Energy ~10 keV -> velocity
            E_pka = 10000  # eV
            mass = 238.0 if atomic_numbers[pka] == 92 else 16.0  # amu
            
            # v = sqrt(2E/m)
            v = np.sqrt(2 * E_pka / (mass * 1.66e-27)) * 1e-5  # to Angstrom/fs
            velocities[pka] = direction * v
        
        return velocities, pka_indices
    
    def detect_defects(self, positions: np.ndarray,
                      reference_positions: np.ndarray,
                      atomic_numbers: np.ndarray,
                      box: np.ndarray) -> Dict:
        """
        Detect point defects (vacancies and interstitials)
        
        Returns:
            Defect statistics
        """
        from scipy.spatial import cKDTree
        
        # Build reference tree
        tree = cKDTree(reference_positions, boxsize=np.diag(box))
        
        # Find vacancies (reference sites without nearby atoms)
        distances, _ = tree.query(positions, k=1)
        vacancy_threshold = 0.5  # Angstrom
        
        n_vacancies = np.sum(distances > vacancy_threshold)
        
        # Find interstitials (atoms not near reference sites)
        tree_new = cKDTree(positions, boxsize=np.diag(box))
        distances_ref, _ = tree_new.query(reference_positions, k=1)
        n_interstitials = np.sum(distances_ref > vacancy_threshold)
        
        return {
            'n_vacancies': int(n_vacancies),
            'n_interstitials': int(n_interstitials),
            'vacancy_concentration': n_vacancies / len(reference_positions),
            'interstitial_concentration': n_interstitials / len(positions)
        }
    
    def run_temperature_ramp(self, positions: np.ndarray,
                            velocities: np.ndarray,
                            masses: np.ndarray,
                            box: np.ndarray) -> Dict:
        """Run temperature ramp to simulate accident conditions"""
        results = []
        
        # Temperature ramp from 300K to melting
        temperatures = np.linspace(300, self.config.max_temperature, 20)
        
        for T_target in temperatures:
            logger.info(f"Running at T = {T_target:.0f} K")
            
            # Adjust velocities to target temperature
            current_T = self.md.compute_temperature() if self.md else 300
            scale = np.sqrt(T_target / max(current_T, 10))
            velocities *= scale
            
            # Run short simulation
            self.md.config.n_steps = 1000
            self.md.config.target_temperature = T_target
            self.md.run()
            
            # Record state
            phase_info = self.md.phase_detector.detect_phase_transition(
                self.md.positions, box, T_target,
                self.md.compute_pressure()
            )
            
            results.append({
                'temperature': T_target,
                'pressure': self.md.compute_pressure(),
                'phase': phase_info
            })
            
            # Check for melting
            if phase_info.get('is_molten', False):
                logger.info(f"Melting detected at {T_target:.0f} K")
                break
        
        return {'temperature_ramp': results}
    
    def run(self):
        """Run nuclear material simulation"""
        logger.info(f"Starting nuclear material simulation: {self.config.material}")
        
        # Create structure
        positions, atomic_numbers, box = self.create_uo2_structure()
        reference_positions = positions.copy()
        
        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_config)
        
        # Initialize MD
        n_atoms = len(positions)
        masses = np.where(atomic_numbers == 92, 238.03,  # U
                         np.where(atomic_numbers == 8, 16.00, 14.01))  # O, N
        
        velocities = np.random.randn(n_atoms, 3) * 0.01
        
        self.md = ExtremeMD(self.md_config)
        self.md.initialize(positions, velocities, masses, box)
        
        # Run temperature ramp
        logger.info("Running temperature ramp...")
        ramp_results = self.run_temperature_ramp(positions, velocities, masses, box)
        
        # Introduce radiation damage
        logger.info("Introducing radiation damage...")
        damaged_velocities, pka_indices = self.introduce_radiation_damage(
            self.md.positions, atomic_numbers, box, self.config.irradiation_dpa
        )
        
        self.md.velocities = damaged_velocities
        self.md.config.n_steps = 10000
        self.md.run()
        
        # Detect defects
        defects = self.detect_defects(
            self.md.positions, reference_positions, atomic_numbers, box
        )
        
        results = {
            'material': self.config.material,
            'temperature_ramp': ramp_results,
            'defects': defects,
            'n_pka': len(pka_indices),
            'final_temperature': self.md.compute_temperature(),
            'final_pressure': self.md.compute_pressure(),
            'trajectory': self.md.trajectory
        }
        
        logger.info("Nuclear material simulation completed")
        logger.info(f"Defects: {defects['n_vacancies']} vacancies, "
                   f"{defects['n_interstitials']} interstitials")
        
        return results


class AsteroidImpactSimulator:
    """
    Asteroid impact simulation
    
    Simulates hypervelocity impact of asteroid on planetary surface
    with shock wave propagation and crater formation.
    """
    
    def __init__(self, config: AsteroidImpactConfig):
        self.config = config
        
        # Configure for impact simulation
        self.md_config = ExtremeMDConfig(
            timestep=0.1,  # Very small timestep for shock
            n_steps=100000,
            shock_method=ShockMethod.BILAYER,
            piston_velocity=self.config.impactor_velocity_kms,
            target_temperature=300.0,
            target_pressure=0.0001,  # Atmospheric
            phase_detect_method=PhaseTransitionMethod.RDF_ANALYSIS
        )
        
        self.parallel_config = ParallelConfig(
            network_topology="mesh",
            dynamic_load_balance=True
        )
        
        self.md = None
        self.optimizer = None
    
    def create_impact_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create target and impactor system
        
        Returns:
            (positions, atomic_numbers, box)
        """
        # Target: crystalline rock (simplified as SiO2)
        a_sio2 = 5.0  # Approximate
        
        # Target dimensions
        target_size = self.config.target_thickness_km * 1e8 / 2  # to Angstrom, half for box
        n_target = int(self.config.n_atoms * 0.9)  # 90% target
        
        n_cells = int((n_target / 9) ** (1/3)) + 1  # 3 Si + 6 O per unit
        
        positions = []
        atomic_numbers = []
        
        # Create target
        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    base = np.array([i, j, k]) * a_sio2
                    
                    # Si sites
                    for _ in range(3):
                        if len(positions) < n_target:
                            positions.append(base + np.random.rand(3) * a_sio2 * 0.5)
                            atomic_numbers.append(14)  # Si
                    
                    # O sites
                    for _ in range(6):
                        if len(positions) < n_target:
                            positions.append(base + np.random.rand(3) * a_sio2 * 0.5 + a_sio2/2)
                            atomic_numbers.append(8)  # O
        
        target_positions = np.array(positions[:n_target])
        target_z_max = np.max(target_positions[:, 2])
        
        # Create impactor (iron-nickel asteroid)
        n_impactor = self.config.n_atoms - n_target
        
        # Spherical impactor
        radius = (self.config.impactor_diameter_km / 2) * 1e8  # to Angstrom
        impactor_center = np.array([
            np.mean(target_positions[:, 0]),
            np.mean(target_positions[:, 1]),
            target_z_max + radius + 10  # Just above target
        ])
        
        impactor_positions = []
        n_impactor_cells = int((n_impactor / 2) ** (1/3)) + 1
        a_fe = 2.87
        
        for i in range(n_impactor_cells):
            for j in range(n_impactor_cells):
                for k in range(n_impactor_cells):
                    if len(impactor_positions) < n_impactor:
                        pos = impactor_center + np.array([
                            (i - n_impactor_cells/2) * a_fe,
                            (j - n_impactor_cells/2) * a_fe,
                            (k - n_impactor_cells/2) * a_fe
                        ])
                        
                        # Check if inside sphere
                        if np.linalg.norm(pos - impactor_center) < radius:
                            impactor_positions.append(pos)
                            if len(impactor_positions) < n_impactor:
                                impactor_positions.append(pos + np.array([a_fe/2, a_fe/2, a_fe/2]))
        
        impactor_positions = np.array(impactor_positions[:n_impactor])
        
        # Assign impactor atomic numbers (Fe=26, Ni=28)
        n_fe = len(impactor_positions) // 2
        impactor_numbers = np.concatenate([
            np.full(n_fe, 26),
            np.full(len(impactor_positions) - n_fe, 28)
        ])
        
        # Combine
        all_positions = np.vstack([target_positions, impactor_positions])
        all_numbers = np.concatenate([atomic_numbers[:n_target], impactor_numbers])
        
        # Box
        box_size = max(
            np.max(all_positions[:, 0]),
            np.max(all_positions[:, 1]),
            np.max(all_positions[:, 2])
        ) + 50
        box = np.eye(3) * box_size
        
        return all_positions, all_numbers, box
    
    def set_impactor_velocity(self, velocities: np.ndarray,
                             atomic_numbers: np.ndarray,
                             positions: np.ndarray,
                             box: np.ndarray) -> np.ndarray:
        """
        Set impactor velocity towards target
        
        Args:
            velocities: Velocity array to modify
            atomic_numbers: Atomic numbers
            positions: Atomic positions
            box: Simulation box
            
        Returns:
            Modified velocities
        """
        # Identify impactor (Fe/Ni)
        impactor_mask = (atomic_numbers == 26) | (atomic_numbers == 28)
        
        # Velocity direction (downward with impact angle)
        angle_rad = np.radians(self.config.impact_angle_degrees)
        velocity_direction = np.array([
            np.sin(angle_rad),
            0,
            -np.cos(angle_rad)
        ])
        
        # Convert velocity to simulation units
        v_kms = self.config.impactor_velocity_kms
        v_angstrom_fs = v_kms * 1e5  # km/s to Angstrom/fs
        
        velocities[impactor_mask] = velocity_direction * v_angstrom_fs
        
        return velocities
    
    def analyze_crater(self, final_positions: np.ndarray,
                      initial_positions: np.ndarray,
                      box: np.ndarray) -> Dict:
        """
        Analyze crater formation
        
        Returns:
            Crater statistics
        """
        # Surface was initially at approximately max z of lower half
        surface_z = np.percentile(initial_positions[:, 2], 50)
        
        # Find crater (region where material is removed)
        displaced = final_positions[:, 2] < surface_z - 5
        
        if np.sum(displaced) < 10:
            return {'crater_formed': False}
        
        crater_positions = final_positions[displaced]
        
        # Crater dimensions
        crater_center_xy = np.mean(crater_positions[:, :2], axis=0)
        crater_radius_xy = np.std(crater_positions[:, :2])
        crater_depth = surface_z - np.min(crater_positions[:, 2])
        crater_volume = len(crater_positions) * 30  # Approximate atomic volume
        
        return {
            'crater_formed': True,
            'center_xy': crater_center_xy,
            'radius_xy_A': crater_radius_xy,
            'depth_A': crater_depth,
            'volume_A3': crater_volume,
            'radius_m': crater_radius_xy * 1e-10,
            'depth_m': crater_depth * 1e-10,
            'volume_m3': crater_volume * 1e-30
        }
    
    def compute_shock_pressure(self, hugoniot_data: List[Dict]) -> Dict:
        """
        Extract shock pressure from Hugoniot states
        
        Returns:
            Shock pressure statistics
        """
        if not hugoniot_data:
            return {'peak_pressure': 0}
        
        pressures = [h['P'] for h in hugoniot_data if 'P' in h]
        
        if not pressures:
            return {'peak_pressure': 0}
        
        return {
            'peak_pressure_GPa': max(pressures),
            'average_pressure_GPa': np.mean(pressures),
            'final_pressure_GPa': pressures[-1]
        }
    
    def run(self):
        """Run asteroid impact simulation"""
        logger.info("Starting asteroid impact simulation")
        logger.info(f"Impactor: {self.config.impactor_diameter_km} km at "
                   f"{self.config.impactor_velocity_kms} km/s")
        
        # Create system
        positions, atomic_numbers, box = self.create_impact_system()
        initial_positions = positions.copy()
        
        logger.info(f"Total atoms: {len(positions)}")
        
        # Initialize
        self.optimizer = MillionCoreOptimizer(self.parallel_config)
        
        masses = np.where(
            atomic_numbers == 14, 28.09,  # Si
            np.where(atomic_numbers == 8, 16.00,  # O
                    np.where(atomic_numbers == 26, 55.85, 58.69))  # Fe, Ni
        )
        
        # Initial velocities
        velocities = np.random.randn(len(positions), 3) * 0.01
        
        # Set impactor velocity
        velocities = self.set_impactor_velocity(
            velocities, atomic_numbers, positions, box
        )
        
        # Run simulation
        self.md = ExtremeMD(self.md_config)
        self.md.initialize(positions, velocities, masses, box)
        
        logger.info("Running impact simulation...")
        self.md.run()
        
        # Analyze results
        crater_info = self.analyze_crater(
            self.md.positions, initial_positions, box
        )
        
        shock_info = self.compute_shock_pressure(
            self.md.get_hugoniot_curve()
        )
        
        results = {
            'impactor_diameter_km': self.config.impactor_diameter_km,
            'impactor_velocity_kms': self.config.impactor_velocity_kms,
            'impact_angle_degrees': self.config.impact_angle_degrees,
            'crater': crater_info,
            'shock': shock_info,
            'final_temperature_K': self.md.compute_temperature(),
            'trajectory': self.md.trajectory,
            'hugoniot': self.md.get_hugoniot_curve()
        }
        
        logger.info("Asteroid impact simulation completed")
        if crater_info.get('crater_formed'):
            logger.info(f"Crater radius: {crater_info['radius_m']:.1f} m")
            logger.info(f"Crater depth: {crater_info['depth_m']:.1f} m")
        
        return results


def run_all_applications():
    """Run all example applications"""
    results = {}
    
    # 1. Planetary core simulation (reduced scale for demo)
    print("="*60)
    print("1. PLANETARY CORE SIMULATION")
    print("="*60)
    
    core_config = PlanetaryCoreConfig(
        n_atoms=10000,  # Scaled down for demo
        inner_core_temp=6000.0,
        outer_core_temp=4000.0,
        center_pressure=360.0
    )
    
    core_sim = PlanetaryCoreSimulator(core_config)
    results['planetary_core'] = core_sim.run()
    
    # 2. Nuclear material simulation (reduced scale for demo)
    print("\n" + "="*60)
    print("2. NUCLEAR MATERIAL SIMULATION")
    print("="*60)
    
    nuclear_config = NuclearMaterialConfig(
        n_atoms=5000,  # Scaled down for demo
        max_temperature=2000.0,
        irradiation_dpa=10.0
    )
    
    nuclear_sim = NuclearMaterialSimulator(nuclear_config)
    results['nuclear_material'] = nuclear_sim.run()
    
    # 3. Asteroid impact simulation (reduced scale for demo)
    print("\n" + "="*60)
    print("3. ASTEROID IMPACT SIMULATION")
    print("="*60)
    
    impact_config = AsteroidImpactConfig(
        n_atoms=10000,  # Scaled down for demo
        impactor_diameter_km=0.5,  # Smaller for demo
        impactor_velocity_kms=15.0
    )
    
    impact_sim = AsteroidImpactSimulator(impact_config)
    results['asteroid_impact'] = impact_sim.run()
    
    print("\n" + "="*60)
    print("ALL SIMULATIONS COMPLETED")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = run_all_applications()
    
    # Print summary
    print("\nRESULTS SUMMARY:")
    print(f"Planetary core inner core fraction: "
          f"{results['planetary_core'].get('inner_core_fraction', 0):.3f}")
    print(f"Nuclear material defects: "
          f"{results['nuclear_material']['defects']['n_vacancies']} vacancies")
    print(f"Asteroid impact crater radius: "
          f"{results['asteroid_impact']['crater'].get('radius_m', 0):.1f} m")
