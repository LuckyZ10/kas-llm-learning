#!/usr/bin/env python3
"""
Application Examples for Adaptive Multiscale Methods

This module provides practical examples of using the adaptive multiscale
framework for real-world simulations including:
- Crack propagation
- Catalytic reactions
- Defect evolution

Author: DFTLammps Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# Import adaptive multiscale modules
try:
    from dftlammps.adaptive_multiscale.resolution_adapter import (
        AdaptiveResolutionManager, ResolutionLevel, create_default_manager
    )
    from dftlammps.adaptive_multiscale.error_estimators import (
        EnsembleErrorEstimator, AdaptiveSamplingTrigger, UncertaintyEstimate
    )
    from dftlammps.adaptive_multiscale.coupling_controller import (
        CouplingController, QMRegion, BoundaryType
    )
    from dftlammps.smart_sampling import (
        SmartSamplingManager, RareEvent, EventType
    )
except ModuleNotFoundError:
    # Direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from dftlammps.adaptive_multiscale.resolution_adapter import (
        AdaptiveResolutionManager, ResolutionLevel, create_default_manager
    )
    from dftlammps.adaptive_multiscale.error_estimators import (
        EnsembleErrorEstimator, AdaptiveSamplingTrigger, UncertaintyEstimate
    )
    from dftlammps.adaptive_multiscale.coupling_controller import (
        CouplingController, QMRegion, BoundaryType
    )
    from dftlammps.smart_sampling import (
        SmartSamplingManager, RareEvent, EventType
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Adaptive Crack Propagation Simulation
# =============================================================================

@dataclass
class CrackSimulationConfig:
    """Configuration for crack propagation simulation."""
    system_size: Tuple[int, int, int] = (100, 100, 10)
    crack_tip_position: np.ndarray = None
    initial_crack_length: float = 20.0
    strain_rate: float = 1e-3
    
    def __post_init__(self):
        if self.crack_tip_position is None:
            self.crack_tip_position = np.array([50.0, 50.0, 5.0])


class CrackTipTracker:
    """Tracks crack tip position and surrounding high-stress region."""
    
    def __init__(self, initial_position: np.ndarray, tracking_radius: float = 15.0):
        self.position = initial_position.copy()
        self.tracking_radius = tracking_radius
        self.position_history: List[np.ndarray] = [initial_position.copy()]
        self.stress_history: List[float] = []
        
    def update(self, positions: np.ndarray, stresses: np.ndarray) -> np.ndarray:
        """Update crack tip position based on stress field."""
        # Find atoms with highest stress near current tip
        distances = np.linalg.norm(positions - self.position, axis=1)
        nearby_mask = distances < self.tracking_radius * 2
        
        if np.any(nearby_mask):
            nearby_stresses = stresses[nearby_mask]
            nearby_positions = positions[nearby_mask]
            
            # New tip at maximum stress location
            max_stress_idx = np.argmax(nearby_stresses)
            self.position = nearby_positions[max_stress_idx]
            
        self.position_history.append(self.position.copy())
        self.stress_history.append(np.max(stresses) if len(stresses) > 0 else 0.0)
        
        return self.position
    
    def get_high_stress_atoms(self, positions: np.ndarray, 
                             threshold: float = 0.7) -> List[int]:
        """Get indices of atoms in high-stress region."""
        distances = np.linalg.norm(positions - self.position, axis=1)
        
        # High stress region around crack tip
        high_stress_indices = np.where(distances < self.tracking_radius)[0]
        
        return high_stress_indices.tolist()


class AdaptiveCrackSimulation:
    """
    Adaptive multiscale simulation of crack propagation.
    
    Uses high-resolution DFT near the crack tip and ML/forcefield
    in the bulk material.
    """
    
    def __init__(self, config: CrackSimulationConfig):
        self.config = config
        
        # Initialize components
        self.resolution_manager = create_default_manager(
            target_accuracy=0.05,
            budget_ms=5000.0
        )
        
        self.coupling_controller = CouplingController()
        self.sampling_manager = SmartSamplingManager()
        
        # Crack tracking
        self.crack_tracker = CrackTipTracker(config.crack_tip_position)
        
        # Simulation state
        self.step_count = 0
        self.energy_history: List[float] = []
        self.qm_region_sizes: List[int] = []
        
    def initialize(self, positions: np.ndarray, atomic_numbers: np.ndarray) -> None:
        """Initialize simulation with crack tip QM region."""
        # Define initial QM region around crack tip
        crack_tip_atoms = self.crack_tracker.get_high_stress_atoms(positions)
        
        # Initialize coupling controller
        self.coupling_controller.initialize(positions, atomic_numbers, crack_tip_atoms)
        
        logger.info(f"Initialized crack simulation with {len(crack_tip_atoms)} QM atoms")
        
    def step(self,
             positions: np.ndarray,
             atomic_numbers: np.ndarray,
             stresses: np.ndarray,
             forces_qm: Optional[np.ndarray] = None,
             forces_ml: Optional[np.ndarray] = None,
             forces_mm: Optional[np.ndarray] = None,
             energy: float = 0.0) -> Dict[str, Any]:
        """
        Execute one adaptive simulation step.
        
        Returns:
            Dictionary with simulation results
        """
        self.step_count += 1
        
        # Update crack tip position
        crack_tip = self.crack_tracker.update(positions, stresses)
        
        # Get high-stress atoms for QM region
        high_stress_atoms = self.crack_tracker.get_high_stress_atoms(positions)
        
        # Compute uncertainties for adaptive sampling
        uncertainties = np.random.rand(len(positions)) * 0.1  # Placeholder
        uncertainties[high_stress_atoms] *= 3  # Higher uncertainty near crack
        
        # Adaptive resolution step
        resolution_result = self.resolution_manager.step(
            positions=positions,
            forces_ml=forces_ml if forces_ml is not None else np.zeros_like(positions),
            uncertainty=np.mean(uncertainties[high_stress_atoms]) if high_stress_atoms else 0.01,
            system_context={'geometry_changed': self.step_count % 10 == 0}
        )
        
        # QM/MM coupling step
        forces_qm = forces_qm if forces_qm is not None else np.zeros_like(positions)
        forces_ml = forces_ml if forces_ml is not None else np.zeros_like(positions)
        forces_mm = forces_mm if forces_mm is not None else np.zeros_like(positions)
        
        coupling_result = self.coupling_controller.step(
            positions, atomic_numbers,
            forces_qm, forces_mm, forces_ml,
            uncertainties
        )
        
        # Detect rare events (bond breaking at crack tip)
        events = self.sampling_manager.process_trajectory_frame(
            self.step_count, positions, atomic_numbers, energy, coupling_result['mixed_forces']
        )
        
        # Record history
        self.energy_history.append(energy)
        self.qm_region_sizes.append(len(coupling_result['qm_region']))
        
        return {
            'step': self.step_count,
            'crack_tip': crack_tip,
            'qm_region': coupling_result['qm_region'],
            'mixed_forces': coupling_result['mixed_forces'],
            'resolution': resolution_result,
            'events': events,
            'n_qm_atoms': len(coupling_result['qm_region'])
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary."""
        return {
            'total_steps': self.step_count,
            'crack_tip_path': np.array(self.crack_tracker.position_history).tolist(),
            'average_qm_size': np.mean(self.qm_region_sizes) if self.qm_region_sizes else 0,
            'resolution_summary': self.resolution_manager.get_summary(),
            'coupling_summary': self.coupling_controller.get_summary(),
            'sampling_summary': self.sampling_manager.get_summary()
        }


def run_crack_example():
    """Run example crack propagation simulation."""
    print("=" * 60)
    print("Example: Adaptive Crack Propagation Simulation")
    print("=" * 60)
    
    # Create configuration
    config = CrackSimulationConfig(
        system_size=(100, 100, 10),
        initial_crack_length=20.0,
        strain_rate=1e-3
    )
    
    # Initialize simulation
    simulation = AdaptiveCrackSimulation(config)
    
    # Create test system
    np.random.seed(42)
    n_atoms = 1000
    positions = np.random.randn(n_atoms, 3) * 20
    atomic_numbers = np.random.choice([14, 8], n_atoms)  # Si/O system
    
    # Initialize
    simulation.initialize(positions, atomic_numbers)
    
    print(f"\nInitial system: {n_atoms} atoms")
    print(f"Crack tip: {config.crack_tip_position}")
    
    # Run simulation steps
    print("\nRunning adaptive simulation...")
    for step in range(20):
        # Generate fake data
        stresses = np.random.rand(n_atoms) * 0.1
        # Higher stress near crack tip
        distances = np.linalg.norm(positions - config.crack_tip_position, axis=1)
        stresses += 0.5 * np.exp(-distances / 10.0)
        
        forces = np.random.randn(n_atoms, 3) * 0.01
        energy = np.random.randn() * 0.1
        
        result = simulation.step(
            positions, atomic_numbers, stresses,
            forces_qm=forces, forces_ml=forces * 1.1, forces_mm=forces * 1.2,
            energy=energy
        )
        
        if step % 5 == 0:
            print(f"  Step {step}: QM atoms={result['n_qm_atoms']}, "
                  f"Crack tip=[{result['crack_tip'][0]:.1f}, "
                  f"{result['crack_tip'][1]:.1f}]")
        
        # Move atoms slightly (simulate deformation)
        positions += np.random.randn(n_atoms, 3) * 0.01
    
    # Summary
    print("\n--- Simulation Summary ---")
    summary = simulation.get_summary()
    print(f"Total steps: {summary['total_steps']}")
    print(f"Average QM region size: {summary['average_qm_size']:.1f} atoms")
    print(f"Crack tip displacement: {np.linalg.norm(
        np.array(summary['crack_tip_path'][-1]) - np.array(summary['crack_tip_path'][0])
    ):.2f} Å")
    
    return simulation


# =============================================================================
# Example 2: Smart Catalytic Reaction Path
# =============================================================================

@dataclass
class CatalyticReactionConfig:
    """Configuration for catalytic reaction simulation."""
    catalyst_type: str = "Pt"
    surface_plane: str = "111"
    adsorbate: str = "CO"
    temperature: float = 500.0
    reaction_coordinate_resolution: float = 0.1


class ReactionPathExplorer:
    """
    Explores catalytic reaction pathways with adaptive resolution.
    
    Automatically identifies reaction intermediates and transition states,
    applying high-accuracy methods at critical points.
    """
    
    def __init__(self, config: CatalyticReactionConfig):
        self.config = config
        
        # Initialize adaptive components
        self.resolution_manager = create_default_manager(
            target_accuracy=0.02,  # Higher accuracy for chemistry
            budget_ms=10000.0
        )
        
        self.sampling_manager = SmartSamplingManager(temperature=config.temperature)
        self.coupling_controller = CouplingController()
        
        # Reaction tracking
        self.intermediates: List[Dict] = []
        self.transition_states: List[Dict] = []
        self.current_path: List[np.ndarray] = []
        
        self.reaction_coordinate = 0.0
        self.step_count = 0
        
    def identify_reactive_region(self, 
                                 positions: np.ndarray,
                                 atomic_numbers: np.ndarray) -> List[int]:
        """Identify reactive region around adsorbate."""
        # Find adsorbate atoms (typically light atoms on surface)
        adsorbate_mask = atomic_numbers <= 8  # H, C, N, O
        adsorbate_indices = np.where(adsorbate_mask)[0]
        
        if len(adsorbate_indices) == 0:
            return []
        
        # Find surface atoms near adsorbate
        surface_mask = ~adsorbate_mask
        surface_indices = np.where(surface_mask)[0]
        
        reactive_indices = list(adsorbate_indices)
        
        # Add nearby surface atoms
        for ads_idx in adsorbate_indices:
            ads_pos = positions[ads_idx]
            distances = np.linalg.norm(positions[surface_indices] - ads_pos, axis=1)
            nearby_surface = surface_indices[distances < 5.0]  # 5 Å cutoff
            reactive_indices.extend(nearby_surface.tolist())
        
        return list(set(reactive_indices))
    
    def compute_reaction_coordinate(self, positions: np.ndarray) -> float:
        """Compute reaction coordinate for current configuration."""
        # Example: distance between two key atoms
        # In real implementation, would use proper reaction coordinate
        
        if len(self.current_path) > 0:
            # RMSD from initial state
            rmsd = np.sqrt(np.mean((positions - self.current_path[0])**2))
            return rmsd
        
        return 0.0
    
    def step(self,
             positions: np.ndarray,
             atomic_numbers: np.ndarray,
             energy: float,
             forces: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Execute one reaction exploration step."""
        self.step_count += 1
        
        # Track reaction path
        self.current_path.append(positions.copy())
        
        # Compute reaction coordinate
        self.reaction_coordinate = self.compute_reaction_coordinate(positions)
        
        # Identify reactive region
        reactive_atoms = self.identify_reactive_region(positions, atomic_numbers)
        
        # High uncertainty in reactive region
        uncertainties = np.ones(len(positions)) * 0.05
        uncertainties[reactive_atoms] = 0.3
        
        # Detect events
        events = self.sampling_manager.process_trajectory_frame(
            self.step_count, positions, atomic_numbers, energy, forces
        )
        
        # Check for transition state
        is_ts = self._check_transition_state(energy)
        
        if is_ts:
            self.transition_states.append({
                'step': self.step_count,
                'positions': positions.copy(),
                'energy': energy,
                'reaction_coordinate': self.reaction_coordinate
            })
            # Use high accuracy at TS
            self.resolution_manager.step(
                positions, forces if forces is not None else np.zeros_like(positions),
                uncertainty=0.5,  # Force high resolution
                system_context={'geometry_changed': True}
            )
        
        # Check for intermediate
        is_intermediate = self._check_intermediate(energy)
        if is_intermediate:
            self.intermediates.append({
                'step': self.step_count,
                'positions': positions.copy(),
                'energy': energy
            })
        
        return {
            'step': self.step_count,
            'reaction_coordinate': self.reaction_coordinate,
            'reactive_atoms': reactive_atoms,
            'is_transition_state': is_ts,
            'is_intermediate': is_intermediate,
            'events': events,
            'n_reactive': len(reactive_atoms)
        }
    
    def _check_transition_state(self, energy: float) -> bool:
        """Check if current configuration is a transition state."""
        # Simplified: check for local energy maximum
        if len(self.intermediates) >= 2:
            energies = [self.intermediates[-2]['energy'],
                       energy,
                       self.intermediates[-1]['energy']]
            return energies[1] > energies[0] and energies[1] > energies[2]
        return False
    
    def _check_intermediate(self, energy: float) -> bool:
        """Check if current configuration is a stable intermediate."""
        # Simplified: check for local energy minimum
        if len(self.intermediates) >= 1:
            prev_energy = self.intermediates[-1]['energy']
            return energy < prev_energy - 0.1  # Energy drop
        return False
    
    def get_reaction_profile(self) -> Dict[str, Any]:
        """Get computed reaction profile."""
        return {
            'n_steps': self.step_count,
            'n_intermediates': len(self.intermediates),
            'n_transition_states': len(self.transition_states),
            'intermediate_energies': [i['energy'] for i in self.intermediates],
            'ts_energies': [ts['energy'] for ts in self.transition_states],
            'reaction_coordinates': [i['reaction_coordinate'] for i in self.intermediates]
        }


def run_catalysis_example():
    """Run example catalytic reaction simulation."""
    print("\n" + "=" * 60)
    print("Example: Smart Catalytic Reaction Path")
    print("=" * 60)
    
    # Create configuration
    config = CatalyticReactionConfig(
        catalyst_type="Pt",
        surface_plane="111",
        adsorbate="CO",
        temperature=500.0
    )
    
    # Initialize explorer
    explorer = ReactionPathExplorer(config)
    
    # Create test system (Pt surface with adsorbate)
    np.random.seed(123)
    n_atoms = 200
    positions = np.random.randn(n_atoms, 3) * 10
    # Create slab-like structure
    positions[:, 2] = np.abs(positions[:, 2])  # z >= 0
    atomic_numbers = np.array([78] * 180 + [6, 8] * 10)  # Pt + CO
    
    print(f"\nSystem: {np.sum(atomic_numbers == 78)} Pt atoms + "
          f"{np.sum(atomic_numbers == 6)} C + {np.sum(atomic_numbers == 8)} O")
    
    # Run exploration
    print("\nExploring reaction path...")
    for step in range(30):
        # Simulate reaction coordinate progression
        t = step / 30.0
        # Move adsorbate
        positions[-20:] += np.array([0.1 * np.sin(t * np.pi), 
                                     0.05 * np.cos(t * np.pi), 
                                     0.02]) 
        
        energy = np.sin(t * np.pi) * 2.0 + np.random.randn() * 0.05
        forces = np.random.randn(n_atoms, 3) * 0.01
        
        result = explorer.step(positions, atomic_numbers, energy, forces)
        
        if result['is_transition_state']:
            print(f"  Step {step}: ** Transition State detected ** "
                  f"(RC={result['reaction_coordinate']:.2f})")
        elif result['is_intermediate']:
            print(f"  Step {step}: Intermediate found "
                  f"(RC={result['reaction_coordinate']:.2f})")
        elif step % 10 == 0:
            print(f"  Step {step}: Exploring... "
                  f"(RC={result['reaction_coordinate']:.2f})")
    
    # Summary
    print("\n--- Reaction Profile ---")
    profile = explorer.get_reaction_profile()
    print(f"Intermediates found: {profile['n_intermediates']}")
    print(f"Transition states found: {profile['n_transition_states']}")
    if profile['intermediate_energies']:
        print(f"Energy range: {min(profile['intermediate_energies']):.3f} - "
              f"{max(profile['intermediate_energies']):.3f} eV")
    
    return explorer


# =============================================================================
# Example 3: Automatic Defect Evolution Tracking
# =============================================================================

@dataclass
class DefectTrackingConfig:
    """Configuration for defect evolution tracking."""
    material_type: str = "Si"
    defect_types: List[str] = None
    temperature_range: Tuple[float, float] = (300.0, 1000.0)
    annealing_schedule: str = "linear"
    
    def __post_init__(self):
        if self.defect_types is None:
            self.defect_types = ["vacancy", "interstitial", "dislocation"]


class DefectTracker:
    """
    Tracks defect evolution with adaptive resolution.
    
    Automatically detects defects, tracks their movement,
    and applies appropriate resolution levels.
    """
    
    def __init__(self, config: DefectTrackingConfig):
        self.config = config
        
        self.resolution_manager = create_default_manager(
            target_accuracy=0.03,
            budget_ms=8000.0
        )
        
        self.sampling_manager = SmartSamplingManager()
        self.coupling_controller = CouplingController()
        
        # Defect tracking
        self.detected_defects: List[Dict] = []
        self.defect_trajectories: Dict[int, List[np.ndarray]] = {}
        self.defect_types: Dict[int, str] = {}
        
        self.step_count = 0
        self.temperature = config.temperature_range[0]
        
    def detect_defects(self,
                      positions: np.ndarray,
                      atomic_numbers: np.ndarray,
                      reference_positions: Optional[np.ndarray] = None) -> List[Dict]:
        """Detect defects by comparing to perfect crystal."""
        defects = []
        
        if reference_positions is None:
            # Simple detection: atoms far from regular positions
            # In practice, would use proper crystal structure analysis
            return defects
        
        # Find displaced atoms
        displacements = np.linalg.norm(positions - reference_positions, axis=1)
        threshold = 0.5  # Å
        
        displaced_atoms = np.where(displacements > threshold)[0]
        
        # Group into defects
        if len(displaced_atoms) > 0:
            # Simple clustering
            defect_center = np.mean(positions[displaced_atoms], axis=0)
            defect_type = self._classify_defect(positions, atomic_numbers, displaced_atoms)
            
            defects.append({
                'id': len(self.detected_defects),
                'center': defect_center,
                'atoms': displaced_atoms.tolist(),
                'type': defect_type,
                'size': len(displaced_atoms)
            })
        
        return defects
    
    def _classify_defect(self,
                        positions: np.ndarray,
                        atomic_numbers: np.ndarray,
                        defect_atoms: np.ndarray) -> str:
        """Classify defect type."""
        n_atoms = len(defect_atoms)
        
        if n_atoms < 3:
            return "vacancy_cluster"
        elif n_atoms > 10:
            return "dislocation"
        else:
            return "interstitial_cluster"
    
    def track_defects(self, defects: List[Dict]) -> None:
        """Track defect movement across steps."""
        for defect in defects:
            defect_id = defect['id']
            
            if defect_id not in self.defect_trajectories:
                self.defect_trajectories[defect_id] = []
                self.defect_types[defect_id] = defect['type']
            
            self.defect_trajectories[defect_id].append(defect['center'])
        
        self.detected_defects.extend(defects)
    
    def step(self,
             positions: np.ndarray,
             atomic_numbers: np.ndarray,
             reference_positions: Optional[np.ndarray] = None,
             energy: float = 0.0,
             forces: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Execute one defect tracking step."""
        self.step_count += 1
        
        # Update temperature
        T_min, T_max = self.config.temperature_range
        self.temperature = T_min + (T_max - T_min) * (self.step_count / 1000)
        
        # Detect defects
        defects = self.detect_defects(positions, atomic_numbers, reference_positions)
        
        # Track defects
        self.track_defects(defects)
        
        # Identify high-uncertainty regions (defects + surroundings)
        high_uncertainty_atoms = set()
        for defect in defects:
            high_uncertainty_atoms.update(defect['atoms'])
            # Add neighbors
            for atom_idx in defect['atoms']:
                atom_pos = positions[atom_idx]
                distances = np.linalg.norm(positions - atom_pos, axis=1)
                neighbors = np.where(distances < 5.0)[0]
                high_uncertainty_atoms.update(neighbors.tolist())
        
        # Compute uncertainties
        uncertainties = np.ones(len(positions)) * 0.03
        uncertainties[list(high_uncertainty_atoms)] = 0.2
        
        # Adaptive resolution
        resolution_result = self.resolution_manager.step(
            positions,
            forces if forces is not None else np.zeros_like(positions),
            uncertainty=np.mean(uncertainties[list(high_uncertainty_atoms)]) if high_uncertainty_atoms else 0.03,
            system_context={'geometry_changed': len(defects) > 0}
        )
        
        # Detect rare events
        events = self.sampling_manager.process_trajectory_frame(
            self.step_count, positions, atomic_numbers, energy, forces
        )
        
        return {
            'step': self.step_count,
            'temperature': self.temperature,
            'n_defects': len(defects),
            'defects': defects,
            'high_uncertainty_atoms': list(high_uncertainty_atoms),
            'events': events,
            'resolution': resolution_result
        }
    
    def get_defect_summary(self) -> Dict[str, Any]:
        """Get summary of defect evolution."""
        return {
            'total_steps': self.step_count,
            'total_defects_detected': len(self.detected_defects),
            'defect_types': {tid: ttype for tid, ttype in self.defect_types.items()},
            'trajectories': {tid: len(traj) for tid, traj in self.defect_trajectories.items()},
            'final_temperature': self.temperature
        }


def run_defect_example():
    """Run example defect evolution simulation."""
    print("\n" + "=" * 60)
    print("Example: Automatic Defect Evolution Tracking")
    print("=" * 60)
    
    # Create configuration
    config = DefectTrackingConfig(
        material_type="Si",
        defect_types=["vacancy", "interstitial"],
        temperature_range=(300.0, 800.0)
    )
    
    # Initialize tracker
    tracker = DefectTracker(config)
    
    # Create test system
    np.random.seed(456)
    n_atoms = 500
    
    # Create crystal-like structure with defect
    x = np.linspace(0, 20, 10)
    y = np.linspace(0, 20, 10)
    z = np.linspace(0, 10, 5)
    xx, yy, zz = np.meshgrid(x, y, z)
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    positions = positions[:n_atoms]
    
    # Create vacancy by removing some atoms
    positions = np.delete(positions, slice(100, 110), axis=0)
    n_atoms = len(positions)
    
    atomic_numbers = np.full(n_atoms, 14)  # Silicon
    reference_positions = positions.copy()
    
    print(f"\nSystem: {n_atoms} Si atoms with initial defects")
    
    # Run tracking
    print("\nTracking defect evolution...")
    for step in range(25):
        # Simulate defect migration
        if step > 5:
            # Move atoms near defect
            positions[90:100] += np.random.randn(10, 3) * 0.1
        
        energy = np.random.randn() * 0.05 + step * 0.01
        forces = np.random.randn(n_atoms, 3) * 0.01
        
        result = tracker.step(positions, atomic_numbers, reference_positions, 
                             energy, forces)
        
        if result['n_defects'] > 0:
            for defect in result['defects']:
                print(f"  Step {step}: {defect['type']} detected "
                      f"({defect['size']} atoms, T={result['temperature']:.0f}K)")
        elif step % 5 == 0:
            print(f"  Step {step}: Monitoring... (T={result['temperature']:.0f}K)")
    
    # Summary
    print("\n--- Defect Evolution Summary ---")
    summary = tracker.get_defect_summary()
    print(f"Total defects detected: {summary['total_defects_detected']}")
    print(f"Final temperature: {summary['final_temperature']:.0f}K")
    if summary['defect_types']:
        print(f"Defect types: {set(summary['defect_types'].values())}")
    
    return tracker


# =============================================================================
# Main execution
# =============================================================================

def main():
    """Run all example simulations."""
    print("\n" + "=" * 60)
    print("DFTLammps Adaptive Multiscale - Application Examples")
    print("=" * 60)
    
    # Run crack propagation example
    crack_sim = run_crack_example()
    
    # Run catalysis example
    catalysis_sim = run_catalysis_example()
    
    # Run defect tracking example
    defect_sim = run_defect_example()
    
    # Final summary
    print("\n" + "=" * 60)
    print("All Examples Completed Successfully!")
    print("=" * 60)
    
    return {
        'crack': crack_sim,
        'catalysis': catalysis_sim,
        'defect': defect_sim
    }


if __name__ == "__main__":
    results = main()
