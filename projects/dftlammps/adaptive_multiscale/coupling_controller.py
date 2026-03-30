#!/usr/bin/env python3
"""
Coupling Controller Module - QM/MM Boundary Management and Load Balancing

This module handles automatic QM/MM boundary optimization, information
passing between resolution levels, and computational load balancing.

Author: DFTLammps Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    """Types of QM/MM boundaries."""
    FIXED = auto()        # Fixed geometry-based boundary
    ADAPTIVE = auto()     # Dynamically adjusting boundary
    BUFFER = auto()       # Buffer zone approach
    ONION = auto()        # Multi-layer onion model
    HOTSPOT = auto()      # Reaction hotspot tracking


class CouplingScheme(Enum):
    """QM/MM coupling schemes."""
    MECHANICAL = auto()   # Mechanical embedding
    ELECTROSTATIC = auto()  # Electrostatic embedding
    POLARIZABLE = auto()  # Polarizable embedding
    CAP = auto()          # Cap atom approach


@dataclass
class QMRegion:
    """Definition of a QM region."""
    atom_indices: Set[int]
    charge: int = 0
    multiplicity: int = 1
    description: str = ""
    
    # Buffer layers
    buffer_indices: Set[int] = field(default_factory=set)
    link_atom_indices: Set[int] = field(default_factory=set)
    
    def __len__(self) -> int:
        return len(self.atom_indices)
    
    def get_all_indices(self) -> Set[int]:
        """Get all indices including buffer."""
        return self.atom_indices | self.buffer_indices | self.link_atom_indices


@dataclass
class BoundaryMetrics:
    """Metrics for boundary quality assessment."""
    n_qm_atoms: int = 0
    n_mm_atoms: int = 0
    n_buffer_atoms: int = 0
    n_link_atoms: int = 0
    
    # Quality metrics
    boundary_energy: float = 0.0
    force_continuity: float = 0.0
    charge_neutrality_error: float = 0.0
    
    # Performance
    qm_calculation_time: float = 0.0
    mm_calculation_time: float = 0.0
    communication_overhead: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_qm_atoms': self.n_qm_atoms,
            'n_mm_atoms': self.n_mm_atoms,
            'n_buffer_atoms': self.n_buffer_atoms,
            'n_link_atoms': self.n_link_atoms,
            'boundary_energy': self.boundary_energy,
            'force_continuity': self.force_continuity,
            'qm_calculation_time': self.qm_calculation_time,
            'mm_calculation_time': self.mm_calculation_time
        }


class BoundaryOptimizer(ABC):
    """Abstract base class for boundary optimizers."""
    
    @abstractmethod
    def optimize(self,
                 positions: np.ndarray,
                 atomic_numbers: np.ndarray,
                 current_qm_region: QMRegion,
                 error_indicators: Optional[np.ndarray] = None) -> QMRegion:
        """Optimize QM/MM boundary."""
        pass
    
    @abstractmethod
    def evaluate_quality(self,
                        positions: np.ndarray,
                        atomic_numbers: np.ndarray,
                        qm_region: QMRegion) -> BoundaryMetrics:
        """Evaluate quality of current boundary."""
        pass


class AdaptiveBoundaryOptimizer(BoundaryOptimizer):
    """
    Adaptive boundary optimizer that adjusts QM region based on
    local chemistry and error indicators.
    """
    
    def __init__(self,
                 min_qm_radius: float = 4.0,
                 max_qm_radius: float = 12.0,
                 buffer_thickness: float = 2.0,
                 error_threshold: float = 0.1,
                 growth_rate: float = 0.5,
                 shrink_rate: float = 0.3):
        """
        Initialize adaptive boundary optimizer.
        
        Args:
            min_qm_radius: Minimum QM region radius (Å)
            max_qm_radius: Maximum QM region radius (Å)
            buffer_thickness: Thickness of buffer layer (Å)
            error_threshold: Error threshold for region expansion
            growth_rate: Rate of region growth
            shrink_rate: Rate of region shrinkage
        """
        self.min_radius = min_qm_radius
        self.max_radius = max_qm_radius
        self.buffer_thickness = buffer_thickness
        self.error_threshold = error_threshold
        self.growth_rate = growth_rate
        self.shrink_rate = shrink_rate
        
        self.current_radius = min_qm_radius
        self.radius_history: deque = deque(maxlen=100)
        
    def optimize(self,
                 positions: np.ndarray,
                 atomic_numbers: np.ndarray,
                 current_qm_region: Optional[QMRegion] = None,
                 error_indicators: Optional[np.ndarray] = None) -> QMRegion:
        """
        Optimize QM region based on error indicators and geometry.
        
        Args:
            positions: Atomic positions (N, 3)
            atomic_numbers: Atomic numbers (N,)
            current_qm_region: Current QM region (if any)
            error_indicators: Per-atom error estimates (N,)
            
        Returns:
            Optimized QM region
        """
        n_atoms = len(positions)
        
        # Find high-error atoms as QM seeds
        if error_indicators is not None and len(error_indicators) == n_atoms:
            high_error_mask = error_indicators > self.error_threshold
            seed_indices = set(np.where(high_error_mask)[0])
        else:
            # Default: no specific seeds
            seed_indices = set()
        
        # Add existing QM region if provided
        if current_qm_region is not None:
            seed_indices.update(current_qm_region.atom_indices)
        
        # If no seeds, use geometric center
        if not seed_indices:
            center = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - center, axis=1)
            seed_indices = {int(np.argmin(distances))}
        
        # Build QM region around seeds
        qm_indices = self._grow_region(positions, seed_indices, error_indicators)
        
        # Define buffer layer
        buffer_indices = self._define_buffer_layer(positions, qm_indices)
        
        # Define link atoms
        link_indices = self._define_link_atoms(positions, atomic_numbers, qm_indices, buffer_indices)
        
        # Compute charge and multiplicity
        charge, multiplicity = self._compute_electronic_state(atomic_numbers, qm_indices)
        
        return QMRegion(
            atom_indices=qm_indices,
            charge=charge,
            multiplicity=multiplicity,
            buffer_indices=buffer_indices,
            link_atom_indices=link_indices,
            description=f"Adaptive region with {len(qm_indices)} QM atoms"
        )
    
    def _grow_region(self,
                     positions: np.ndarray,
                     seed_indices: Set[int],
                     error_indicators: Optional[np.ndarray]) -> Set[int]:
        """Grow QM region from seeds."""
        n_atoms = len(positions)
        region = set(seed_indices)
        
        # Build KD-tree for efficient neighbor search
        tree = cKDTree(positions)
        
        # Iterative growth
        max_iterations = 10
        for iteration in range(max_iterations):
            # Find atoms within current radius of any region atom
            region_list = list(region)
            if not region_list:
                break
                
            # Query neighbors
            neighbors = tree.query_ball_point(positions[list(region)], r=self.current_radius)
            
            # Add all neighbors to region
            new_region = region.copy()
            for neigh_list in neighbors:
                new_region.update(neigh_list)
            
            # Check for convergence
            if len(new_region) == len(region):
                break
            
            region = new_region
            
            # Expand radius if high errors at boundary
            if error_indicators is not None:
                boundary_atoms = self._get_boundary_atoms(positions, region)
                if len(boundary_atoms) > 0:
                    boundary_errors = error_indicators[list(boundary_atoms)]
                    if np.max(boundary_errors) > self.error_threshold:
                        self.current_radius = min(self.max_radius, 
                                                 self.current_radius * (1 + self.growth_rate))
        
        # Apply bounds
        if len(region) < 10:
            # Too small - expand
            self.current_radius = min(self.max_radius, self.current_radius * 1.2)
        elif len(region) > 500:
            # Too large - shrink
            self.current_radius = max(self.min_radius, self.current_radius * 0.8)
        
        self.radius_history.append(self.current_radius)
        
        return region
    
    def _get_boundary_atoms(self, positions: np.ndarray, region: Set[int]) -> Set[int]:
        """Identify atoms at the boundary of the QM region."""
        boundary = set()
        region_list = list(region)
        tree = cKDTree(positions[region_list])
        
        for idx in region:
            # Check if atom has neighbors outside region
            neighbors = tree.query_ball_point(positions[idx], r=self.current_radius)
            if len(neighbors) < len(region_list):
                boundary.add(idx)
        
        return boundary
    
    def _define_buffer_layer(self,
                            positions: np.ndarray,
                            qm_indices: Set[int]) -> Set[int]:
        """Define buffer layer around QM region."""
        if not qm_indices:
            return set()
        
        qm_list = list(qm_indices)
        tree = cKDTree(positions[qm_list])
        
        buffer = set()
        outer_radius = self.current_radius + self.buffer_thickness
        
        for i, pos in enumerate(positions):
            if i in qm_indices:
                continue
            
            dist, _ = tree.query(pos, k=1)
            if dist <= outer_radius and dist > self.current_radius:
                buffer.add(i)
        
        return buffer
    
    def _define_link_atoms(self,
                          positions: np.ndarray,
                          atomic_numbers: np.ndarray,
                          qm_indices: Set[int],
                          buffer_indices: Set[int]) -> Set[int]:
        """Define link atoms at QM/MM boundary."""
        link_atoms = set()
        
        # Find bonds crossing the boundary
        bonds = self._find_bonds_across_boundary(positions, atomic_numbers, 
                                                  qm_indices, buffer_indices)
        
        # For each crossing bond, add link atom
        for qm_idx, mm_idx in bonds:
            # Place link atom along bond
            link_atoms.add(mm_idx)  # Simplified - would compute position
        
        return link_atoms
    
    def _find_bonds_across_boundary(self,
                                   positions: np.ndarray,
                                   atomic_numbers: np.ndarray,
                                   qm_indices: Set[int],
                                   buffer_indices: Set[int]) -> List[Tuple[int, int]]:
        """Find chemical bonds crossing the QM/MM boundary."""
        bonds = []
        
        # Simple distance-based bond detection
        qm_list = list(qm_indices)
        tree = cKDTree(positions)
        
        for qm_idx in qm_list:
            # Find neighbors within typical bond length
            neighbors = tree.query_ball_point(positions[qm_idx], r=2.0)
            
            for neighbor in neighbors:
                if neighbor not in qm_indices and neighbor in buffer_indices:
                    # Check if valid bond
                    if self._is_valid_bond(atomic_numbers[qm_idx], atomic_numbers[neighbor],
                                          positions[qm_idx], positions[neighbor]):
                        bonds.append((qm_idx, neighbor))
        
        return bonds
    
    def _is_valid_bond(self, z1: int, z2: int, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if atoms form a valid chemical bond."""
        distance = np.linalg.norm(pos1 - pos2)
        
        # Approximate covalent radii
        radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 14: 1.11, 15: 1.07, 16: 1.05}
        r1 = radii.get(z1, 1.0)
        r2 = radii.get(z2, 1.0)
        
        # Bond if within 1.2x sum of covalent radii
        return distance <= 1.2 * (r1 + r2)
    
    def _compute_electronic_state(self,
                                  atomic_numbers: np.ndarray,
                                  qm_indices: Set[int]) -> Tuple[int, int]:
        """Compute charge and multiplicity for QM region."""
        qm_list = list(qm_indices)
        total_protons = sum(atomic_numbers[i] for i in qm_list)
        
        # Assume neutral, singlet (would use proper chemistry for real cases)
        charge = 0
        
        # Simple multiplicity rule: even electrons -> singlet, odd -> doublet
        n_electrons = total_protons  # Neutral assumption
        multiplicity = 1 if n_electrons % 2 == 0 else 2
        
        return charge, multiplicity
    
    def evaluate_quality(self,
                        positions: np.ndarray,
                        atomic_numbers: np.ndarray,
                        qm_region: QMRegion) -> BoundaryMetrics:
        """Evaluate quality of QM/MM boundary."""
        metrics = BoundaryMetrics()
        
        metrics.n_qm_atoms = len(qm_region.atom_indices)
        metrics.n_buffer_atoms = len(qm_region.buffer_indices)
        metrics.n_link_atoms = len(qm_region.link_atom_indices)
        metrics.n_mm_atoms = len(positions) - metrics.n_qm_atoms - metrics.n_buffer_atoms
        
        # Evaluate force continuity across boundary
        # This would use actual forces in a real implementation
        metrics.force_continuity = 0.95  # Placeholder
        
        return metrics


class InformationCoordinator:
    """
    Coordinates information passing between different resolution levels.
    
    Handles force mixing, energy interpolation, and smooth transitions.
    """
    
    def __init__(self,
                 mixing_function: str = 'linear',
                 transition_width: float = 1.0):
        """
        Initialize information coordinator.
        
        Args:
            mixing_function: Type of mixing ('linear', 'spline', 'switch')
            transition_width: Width of transition region (Å)
        """
        self.mixing_function = mixing_function
        self.transition_width = transition_width
        
        self.mixing_history: deque = deque(maxlen=100)
        
    def mix_forces(self,
                   positions: np.ndarray,
                   qm_region: QMRegion,
                   forces_qm: np.ndarray,
                   forces_mm: np.ndarray,
                   forces_ml: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Mix forces from different resolution levels.
        
        Args:
            positions: Atomic positions
            qm_region: QM region definition
            forces_qm: Forces from QM calculation
            forces_mm: Forces from MM calculation
            forces_ml: Forces from ML potential (optional)
            
        Returns:
            Mixed forces for all atoms
        """
        n_atoms = len(positions)
        mixed_forces = np.zeros((n_atoms, 3))
        
        qm_indices = list(qm_region.atom_indices)
        buffer_indices = list(qm_region.buffer_indices)
        all_qm = qm_indices + buffer_indices
        
        # QM atoms: use QM forces
        mixed_forces[qm_indices] = forces_qm[qm_indices]
        
        # Buffer atoms: mix QM and MM forces
        for idx in buffer_indices:
            mixing_factor = self._compute_mixing_factor(positions, idx, qm_region)
            
            if forces_ml is not None:
                # Three-way mixing: QM, ML, MM
                f_qm = forces_qm[idx] if idx < len(forces_qm) else forces_ml[idx]
                f_ml = forces_ml[idx]
                f_mm = forces_mm[idx]
                
                # Weight by proximity to QM region
                mixed_forces[idx] = mixing_factor * f_qm + \
                                   (1 - mixing_factor) * 0.5 * (f_ml + f_mm)
            else:
                # Two-way mixing
                mixed_forces[idx] = mixing_factor * forces_qm[idx] + \
                                   (1 - mixing_factor) * forces_mm[idx]
        
        # MM atoms: use ML/MM forces
        mm_mask = np.ones(n_atoms, dtype=bool)
        mm_mask[all_qm] = False
        mm_indices = np.where(mm_mask)[0]
        
        if forces_ml is not None:
            mixed_forces[mm_indices] = forces_ml[mm_indices]
        else:
            mixed_forces[mm_indices] = forces_mm[mm_indices]
        
        self.mixing_history.append({
            'qm_weight': len(qm_indices) / n_atoms,
            'buffer_weight': len(buffer_indices) / n_atoms
        })
        
        return mixed_forces
    
    def _compute_mixing_factor(self,
                              positions: np.ndarray,
                              atom_idx: int,
                              qm_region: QMRegion) -> float:
        """Compute mixing factor for buffer atom."""
        qm_positions = positions[list(qm_region.atom_indices)]
        atom_pos = positions[atom_idx]
        
        # Distance to nearest QM atom
        distances = np.linalg.norm(qm_positions - atom_pos, axis=1)
        min_dist = np.min(distances)
        
        # Smooth switching function
        if self.mixing_function == 'linear':
            factor = max(0, 1 - min_dist / self.transition_width)
        elif self.mixing_function == 'spline':
            x = min_dist / self.transition_width
            factor = max(0, 1 - 3*x**2 + 2*x**3) if x < 1 else 0
        else:  # switch
            factor = 0.5 * (1 + np.tanh((self.transition_width - min_dist) / 0.5))
        
        return factor
    
    def interpolate_energy(self,
                          energy_qm: float,
                          energy_mm: float,
                          energy_ml: Optional[float] = None,
                          qm_size: int = 0,
                          total_size: int = 1) -> float:
        """
        Interpolate energy from different resolution levels.
        
        Args:
            energy_qm: QM energy contribution
            energy_mm: MM energy contribution
            energy_ml: ML energy (optional)
            qm_size: Number of QM atoms
            total_size: Total number of atoms
            
        Returns:
            Interpolated total energy
        """
        if energy_ml is not None:
            # Use ML as baseline, QM for correction
            qm_correction = energy_qm - energy_ml
            return energy_ml + qm_correction
        
        # Simple weighted combination
        qm_weight = qm_size / total_size
        return qm_weight * energy_qm + (1 - qm_weight) * energy_mm
    
    def smooth_transition(self,
                         old_region: QMRegion,
                         new_region: QMRegion,
                         positions: np.ndarray,
                         n_interpolation_steps: int = 5) -> List[QMRegion]:
        """
        Create smooth transition between QM regions.
        
        Returns list of intermediate regions for gradual transition.
        """
        intermediate_regions = []
        
        old_set = old_region.atom_indices
        new_set = new_region.atom_indices
        
        # Atoms leaving QM
        leaving = old_set - new_set
        # Atoms entering QM
        entering = new_set - old_set
        
        # Create gradual transition
        for step in range(1, n_interpolation_steps):
            frac = step / n_interpolation_steps
            
            # Keep atoms that stay, plus fraction of entering/leaving
            current_set = (old_set & new_set) | \
                         set(list(leaving)[:int(len(leaving) * (1 - frac))]) | \
                         set(list(entering)[:int(len(entering) * frac)])
            
            intermediate_regions.append(QMRegion(
                atom_indices=current_set,
                charge=old_region.charge,
                multiplicity=old_region.multiplicity,
                description=f"Transition step {step}"
            ))
        
        intermediate_regions.append(new_region)
        
        return intermediate_regions


class LoadBalancer:
    """
    Balances computational load across resolution levels and processors.
    """
    
    def __init__(self,
                 n_processors: int = 1,
                 target_load_imbalance: float = 0.1,
                 load_check_interval: int = 10):
        """
        Initialize load balancer.
        
        Args:
            n_processors: Number of available processors
            target_load_imbalance: Target load imbalance tolerance
            load_check_interval: Steps between load checks
        """
        self.n_processors = n_processors
        self.target_imbalance = target_load_imbalance
        self.check_interval = load_check_interval
        
        self.load_history: deque = deque(maxlen=100)
        self.timing_data: Dict[str, List[float]] = defaultdict(list)
        
    def estimate_optimal_partition(self,
                                  system_size: int,
                                  qm_region: QMRegion) -> Dict[str, int]:
        """
        Estimate optimal processor partition for QM/MM.
        
        Returns dictionary with processor allocation.
        """
        n_qm = len(qm_region.atom_indices)
        n_buffer = len(qm_region.buffer_indices)
        n_mm = system_size - n_qm - n_buffer
        
        # Estimate relative costs (QM much more expensive)
        qm_cost_per_atom = 100.0
        buffer_cost_per_atom = 50.0
        mm_cost_per_atom = 1.0
        
        total_cost = (n_qm * qm_cost_per_atom + 
                     n_buffer * buffer_cost_per_atom + 
                     n_mm * mm_cost_per_atom)
        
        # Allocate processors proportionally
        if self.n_processors > 1:
            qm_procs = max(1, int(self.n_processors * n_qm * qm_cost_per_atom / total_cost))
            buffer_procs = max(1, int(self.n_processors * n_buffer * buffer_cost_per_atom / total_cost))
            mm_procs = max(1, self.n_processors - qm_procs - buffer_procs)
        else:
            qm_procs = buffer_procs = mm_procs = 1
        
        return {
            'qm_processors': qm_procs,
            'buffer_processors': buffer_procs,
            'mm_processors': mm_procs,
            'total_processors': qm_procs + buffer_procs + mm_procs
        }
    
    def analyze_load_balance(self,
                            qm_time: float,
                            mm_time: float,
                            ml_time: float = 0.0) -> Dict[str, Any]:
        """Analyze load balance across components."""
        times = [qm_time, mm_time, ml_time]
        max_time = max(times)
        min_time = min(t for t in times if t > 0)
        
        imbalance = (max_time - min_time) / max_time if max_time > 0 else 0
        
        analysis = {
            'imbalance_ratio': imbalance,
            'is_balanced': imbalance <= self.target_imbalance,
            'bottleneck': 'qm' if qm_time == max_time else \
                         'mm' if mm_time == max_time else 'ml',
            'times': {'qm': qm_time, 'mm': mm_time, 'ml': ml_time},
            'efficiency': sum(times) / (len(times) * max_time) if max_time > 0 else 1.0
        }
        
        self.load_history.append(analysis)
        
        return analysis
    
    def suggest_rebalancing(self) -> Optional[Dict[str, Any]]:
        """Suggest rebalancing actions based on history."""
        if len(self.load_history) < self.check_interval:
            return None
        
        recent = list(self.load_history)[-self.check_interval:]
        avg_imbalance = np.mean([r['imbalance_ratio'] for r in recent])
        
        if avg_imbalance <= self.target_imbalance:
            return None
        
        # Identify consistent bottleneck
        bottlenecks = [r['bottleneck'] for r in recent]
        common_bottleneck = max(set(bottlenecks), key=bottlenecks.count)
        
        suggestions = {
            'bottleneck': common_bottleneck,
            'avg_imbalance': avg_imbalance,
            'actions': []
        }
        
        if common_bottleneck == 'qm':
            suggestions['actions'] = [
                'Increase QM processors',
                'Reduce QM region size',
                'Use lower level of theory for QM'
            ]
        elif common_bottleneck == 'mm':
            suggestions['actions'] = [
                'Increase MM processors',
                'Use smaller time step for MM',
                'Optimize neighbor lists'
            ]
        else:
            suggestions['actions'] = [
                'Optimize ML model inference',
                'Use batched predictions',
                'Consider model quantization'
            ]
        
        return suggestions
    
    def record_timing(self, component: str, elapsed_time: float) -> None:
        """Record timing for a component."""
        self.timing_data[component].append(elapsed_time)
        
        # Keep only recent data
        if len(self.timing_data[component]) > 100:
            self.timing_data[component] = self.timing_data[component][-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {}
        
        for component, times in self.timing_data.items():
            if times:
                summary[component] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_calls': len(times)
                }
        
        return summary


class CouplingController:
    """
    Main controller for QM/MM coupling and multiscale information management.
    """
    
    def __init__(self,
                 boundary_optimizer: Optional[BoundaryOptimizer] = None,
                 info_coordinator: Optional[InformationCoordinator] = None,
                 load_balancer: Optional[LoadBalancer] = None):
        """
        Initialize coupling controller.
        
        Args:
            boundary_optimizer: Boundary optimizer instance
            info_coordinator: Information coordinator instance
            load_balancer: Load balancer instance
        """
        self.boundary_optimizer = boundary_optimizer or AdaptiveBoundaryOptimizer()
        self.info_coordinator = info_coordinator or InformationCoordinator()
        self.load_balancer = load_balancer or LoadBalancer()
        
        self.current_qm_region: Optional[QMRegion] = None
        self.step_count = 0
        
        self.boundary_history: deque = deque(maxlen=100)
        self.metrics_history: List[BoundaryMetrics] = []
        
    def initialize(self,
                   positions: np.ndarray,
                   atomic_numbers: np.ndarray,
                   seed_indices: Optional[List[int]] = None) -> QMRegion:
        """Initialize QM/MM coupling."""
        # Create a minimal QMRegion from seed indices for initialization
        if seed_indices:
            seed_set = set(seed_indices)
            initial_region = QMRegion(
                atom_indices=seed_set,
                charge=0,
                multiplicity=1,
                description="Initial seed region"
            )
        else:
            initial_region = None
        
        self.current_qm_region = self.boundary_optimizer.optimize(
            positions, atomic_numbers, initial_region
        )
        
        logger.info(f"Initialized QM region with {len(self.current_qm_region)} atoms")
        
        return self.current_qm_region
    
    def step(self,
             positions: np.ndarray,
             atomic_numbers: np.ndarray,
             forces_qm: np.ndarray,
             forces_mm: np.ndarray,
             forces_ml: Optional[np.ndarray] = None,
             error_indicators: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Execute one coupling step.
        
        Returns dictionary with mixed forces and region information.
        """
        self.step_count += 1
        
        # Update QM region if needed
        if self.step_count % 10 == 0 or error_indicators is not None:
            new_region = self.boundary_optimizer.optimize(
                positions, atomic_numbers, self.current_qm_region, error_indicators
            )
            
            if new_region.atom_indices != self.current_qm_region.atom_indices:
                # Smooth transition
                transition_regions = self.info_coordinator.smooth_transition(
                    self.current_qm_region, new_region, positions
                )
                
                self.boundary_history.append({
                    'step': self.step_count,
                    'old_size': len(self.current_qm_region),
                    'new_size': len(new_region),
                    'transition_steps': len(transition_regions)
                })
                
                self.current_qm_region = new_region
        
        # Mix forces
        mixed_forces = self.info_coordinator.mix_forces(
            positions, self.current_qm_region,
            forces_qm, forces_mm, forces_ml
        )
        
        # Evaluate boundary quality
        metrics = self.boundary_optimizer.evaluate_quality(
            positions, atomic_numbers, self.current_qm_region
        )
        self.metrics_history.append(metrics)
        
        return {
            'mixed_forces': mixed_forces,
            'qm_region': self.current_qm_region,
            'metrics': metrics,
            'step': self.step_count
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of coupling performance."""
        return {
            'total_steps': self.step_count,
            'current_qm_size': len(self.current_qm_region) if self.current_qm_region else 0,
            'boundary_changes': len(self.boundary_history),
            'load_balance': self.load_balancer.get_performance_summary(),
            'recent_metrics': self.metrics_history[-10:] if self.metrics_history else []
        }


# Example usage
if __name__ == "__main__":
    print("=== Coupling Controller Demo ===\n")
    
    # Create test system
    np.random.seed(42)
    n_atoms = 200
    positions = np.random.randn(n_atoms, 3) * 10
    atomic_numbers = np.random.choice([1, 6, 7, 8], n_atoms)
    
    # Create controller
    controller = CouplingController()
    
    # Initialize
    qm_region = controller.initialize(positions, atomic_numbers)
    print(f"Initial QM region: {len(qm_region)} atoms")
    print(f"Charge: {qm_region.charge}, Multiplicity: {qm_region.multiplicity}")
    
    # Simulate steps
    for step in range(5):
        # Generate fake forces
        forces_qm = np.random.randn(n_atoms, 3) * 0.1
        forces_mm = np.random.randn(n_atoms, 3) * 0.2
        forces_ml = np.random.randn(n_atoms, 3) * 0.15
        
        # Simulate error indicators (higher near some atoms)
        error_indicators = np.random.rand(n_atoms) * 0.2
        
        result = controller.step(positions, atomic_numbers, 
                                forces_qm, forces_mm, forces_ml,
                                error_indicators)
        
        print(f"Step {step+1}: QM size={len(result['qm_region'])}, "
              f"Mixed forces shape={result['mixed_forces'].shape}")
    
    print("\n--- Summary ---")
    summary = controller.get_summary()
    print(f"Total steps: {summary['total_steps']}")
    print(f"Boundary changes: {summary['boundary_changes']}")
