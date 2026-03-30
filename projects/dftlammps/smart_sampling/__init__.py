#!/usr/bin/env python3
"""
Smart Sampling Module - Adaptive Importance Sampling and Rare Event Detection

This module provides intelligent sampling strategies for molecular dynamics,
including adaptive importance sampling, rare event detection, and
parallel efficiency optimization.

Author: DFTLammps Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
from scipy import stats
from scipy.special import logsumexp
import heapq

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Available sampling strategies."""
    UNIFORM = auto()
    IMPORTANCE = auto()
    BOLTZMANN = auto()
    METADYNAMICS = auto()
    TEMPERED = auto()
    WANG_LANDAU = auto()


class EventType(Enum):
    """Types of rare events to detect."""
    BOND_BREAKING = auto()
    BOND_FORMING = auto()
    PHASE_TRANSITION = auto()
    DIFFUSION_HOP = auto()
    REACTION = auto()
    DEFECT_MIGRATION = auto()
    NUCLEATION = auto()


@dataclass
class SamplingWeights:
    """Weights for adaptive importance sampling."""
    position_weights: np.ndarray
    energy_weights: np.ndarray
    force_weights: np.ndarray
    total_weights: np.ndarray
    
    effective_sample_size: float = 0.0
    entropy: float = 0.0
    
    @classmethod
    def uniform(cls, n_samples: int) -> 'SamplingWeights':
        """Create uniform weights."""
        weights = np.ones(n_samples) / n_samples
        return cls(
            position_weights=weights.copy(),
            energy_weights=weights.copy(),
            force_weights=weights.copy(),
            total_weights=weights.copy(),
            effective_sample_size=float(n_samples),
            entropy=np.log(n_samples)
        )


@dataclass
class RareEvent:
    """Detected rare event information."""
    event_type: EventType
    timestep: int
    atom_indices: List[int]
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    reaction_coordinate: float
    energy_barrier: Optional[float] = None
    probability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.name,
            'timestep': self.timestep,
            'atom_indices': self.atom_indices,
            'reaction_coordinate': self.reaction_coordinate,
            'energy_barrier': self.energy_barrier,
            'probability': self.probability
        }


class BaseSampler(ABC):
    """Abstract base class for samplers."""
    
    @abstractmethod
    def sample(self, 
               positions: np.ndarray,
               energies: np.ndarray,
               **kwargs) -> np.ndarray:
        """Sample configurations based on strategy."""
        pass
    
    @abstractmethod
    def update(self, 
               selected_indices: np.ndarray,
               dft_energies: np.ndarray) -> None:
        """Update sampler with DFT results."""
        pass


class AdaptiveImportanceSampler(BaseSampler):
    """
    Adaptive importance sampler that dynamically adjusts sampling weights
    based on model uncertainty and configuration novelty.
    """
    
    def __init__(self,
                 temperature: float = 300.0,
                 uncertainty_weight: float = 0.4,
                 energy_weight: float = 0.3,
                 novelty_weight: float = 0.3,
                 adaptivity_rate: float = 0.1):
        """
        Initialize adaptive importance sampler.
        
        Args:
            temperature: Simulation temperature (K)
            uncertainty_weight: Weight for uncertainty in importance
            energy_weight: Weight for energy in importance
            novelty_weight: Weight for configuration novelty
            adaptivity_rate: Rate of weight adaptation
        """
        self.temperature = temperature
        self.kT = temperature * 8.617333e-5  # Boltzmann constant in eV/K
        
        self.w_unc = uncertainty_weight
        self.w_energy = energy_weight
        self.w_novelty = novelty_weight
        self.adaptivity_rate = adaptivity_rate
        
        self.weights_history: deque = deque(maxlen=100)
        self.selected_history: deque = deque(maxlen=1000)
        
        # Kernel density for novelty estimation
        self.density_estimator = None
        self.reference_structures: List[np.ndarray] = []
        
    def sample(self,
               positions: np.ndarray,
               energies: np.ndarray,
               uncertainties: Optional[np.ndarray] = None,
               n_samples: int = 10,
               method: str = 'systematic') -> np.ndarray:
        """
        Sample configurations using adaptive importance sampling.
        
        Args:
            positions: Configuration positions (N_configs, N_atoms, 3)
            energies: ML predicted energies (N_configs,)
            uncertainties: Uncertainty estimates (N_configs,)
            n_samples: Number of samples to select
            method: Sampling method ('systematic', 'residual', 'multinomial')
            
        Returns:
            Indices of selected configurations
        """
        n_configs = len(positions)
        
        # Compute importance weights
        if uncertainties is None:
            uncertainties = np.ones(n_configs) * 0.1
        
        # Normalize uncertainties
        unc_normalized = uncertainties / (np.max(uncertainties) + 1e-10)
        
        # Energy-based weights (Boltzmann-like, but flat for high-energy)
        energy_weights = self._compute_energy_weights(energies)
        
        # Novelty weights
        novelty_weights = self._compute_novelty_weights(positions)
        
        # Combined importance weights
        importance = (self.w_unc * unc_normalized + 
                     self.w_energy * energy_weights +
                     self.w_novelty * novelty_weights)
        
        # Normalize to probabilities
        probs = importance / (np.sum(importance) + 1e-10)
        
        # Ensure valid probabilities
        probs = np.maximum(probs, 1e-10)
        probs = probs / np.sum(probs)
        
        # Sample indices
        if method == 'systematic':
            indices = self._systematic_sample(probs, n_samples)
        elif method == 'residual':
            indices = self._residual_sample(probs, n_samples)
        else:  # multinomial
            indices = np.random.choice(n_configs, size=n_samples, p=probs, replace=False)
        
        # Record selection
        self.selected_history.extend(indices.tolist())
        
        # Compute effective sample size
        ess = 1.0 / np.sum(probs**2)
        
        self.weights_history.append({
            'probs': probs,
            'ess': ess,
            'entropy': -np.sum(probs * np.log(probs + 1e-10))
        })
        
        return indices
    
    def _compute_energy_weights(self, energies: np.ndarray) -> np.ndarray:
        """Compute energy-based importance weights."""
        # Shift to zero minimum
        e_shifted = energies - np.min(energies)
        
        # Boltzmann-like weights, but flattened at high energy
        weights = np.exp(-e_shifted / self.kT)
        weights = np.maximum(weights, 0.01 * np.max(weights))  # Floor
        
        return weights / np.sum(weights)
    
    def _compute_novelty_weights(self, positions: np.ndarray) -> np.ndarray:
        """Compute novelty-based importance weights."""
        n_configs = len(positions)
        
        if len(self.reference_structures) < 5:
            # Not enough reference data - uniform novelty
            return np.ones(n_configs) / n_configs
        
        novelty = np.zeros(n_configs)
        
        for i, pos in enumerate(positions):
            # Compute distance to nearest reference structure
            min_dist = float('inf')
            
            for ref_pos in self.reference_structures[-50:]:  # Recent references
                if ref_pos.shape == pos.shape:
                    # RMSD-like distance
                    dist = np.sqrt(np.mean((pos - ref_pos)**2))
                    min_dist = min(min_dist, dist)
            
            # Higher novelty for larger distances
            novelty[i] = min_dist
        
        # Normalize and convert to weight
        if np.max(novelty) > 0:
            novelty = novelty / np.max(novelty)
        
        return novelty / (np.sum(novelty) + 1e-10)
    
    def _systematic_sample(self, probs: np.ndarray, n_samples: int) -> np.ndarray:
        """Systematic resampling for lower variance."""
        n = len(probs)
        
        # Cumulative distribution
        cumsum = np.cumsum(probs)
        
        # Systematic sampling
        u = np.random.uniform(0, 1.0 / n_samples)
        indices = []
        
        j = 0
        for i in range(n_samples):
            while cumsum[j] < u + i / n_samples:
                j += 1
                if j >= n:
                    j = n - 1
                    break
            indices.append(j)
        
        return np.array(indices)
    
    def _residual_sample(self, probs: np.ndarray, n_samples: int) -> np.ndarray:
        """Residual resampling."""
        n = len(probs)
        
        # Deterministic part
        counts = np.floor(probs * n_samples).astype(int)
        
        indices = []
        for i, count in enumerate(counts):
            indices.extend([i] * count)
        
        # Residual part
        remaining = n_samples - len(indices)
        if remaining > 0:
            residuals = probs * n_samples - counts
            residual_probs = residuals / np.sum(residuals)
            extra_indices = np.random.choice(n, size=remaining, p=residual_probs)
            indices.extend(extra_indices.tolist())
        
        return np.array(indices)
    
    def update(self,
               selected_indices: np.ndarray,
               dft_energies: np.ndarray) -> None:
        """Update sampler with DFT results."""
        # Add selected structures to reference
        # (Would need actual positions here)
        
        # Adapt weights based on prediction errors
        if len(dft_energies) > 0:
            # If DFT energies vary significantly, increase energy weight
            energy_std = np.std(dft_energies)
            if energy_std > self.kT:
                self.w_energy = min(0.6, self.w_energy + self.adaptivity_rate)
                self.w_unc = max(0.2, self.w_unc - self.adaptivity_rate / 2)
                self.w_novelty = 1.0 - self.w_energy - self.w_unc
    
    def add_reference_structure(self, positions: np.ndarray) -> None:
        """Add a reference structure for novelty estimation."""
        self.reference_structures.append(positions.copy())
        
        # Limit reference size
        if len(self.reference_structures) > 500:
            self.reference_structures = self.reference_structures[-400:]
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        if not self.weights_history:
            return {'n_samples': 0}
        
        recent = list(self.weights_history)[-50:]
        
        return {
            'n_samples': len(self.selected_history),
            'mean_ess': np.mean([w['ess'] for w in recent]),
            'mean_entropy': np.mean([w['entropy'] for w in recent]),
            'current_weights': {
                'uncertainty': self.w_unc,
                'energy': self.w_energy,
                'novelty': self.w_novelty
            },
            'n_reference_structures': len(self.reference_structures)
        }


class RareEventDetector:
    """
    Detects rare events in molecular dynamics trajectories.
    
    Uses multiple detection strategies including:
    - Bond topology analysis
    - Collective variable monitoring
    - Transition state identification
    """
    
    def __init__(self,
                 bond_threshold: float = 2.0,
                 detection_buffer: int = 10,
                 min_event_separation: int = 50):
        """
        Initialize rare event detector.
        
        Args:
            bond_threshold: Distance threshold for bonds (Å)
            detection_buffer: Frames to keep for analysis
            min_event_separation: Minimum steps between events
        """
        self.bond_threshold = bond_threshold
        self.detection_buffer = detection_buffer
        self.min_separation = min_event_separation
        
        self.trajectory_buffer: deque = deque(maxlen=detection_buffer)
        self.topology_buffer: deque = deque(maxlen=detection_buffer)
        
        self.detected_events: List[RareEvent] = []
        self.last_event_step = -min_event_separation
        
        # Collective variables
        self.cv_history: deque = deque(maxlen=1000)
        
    def process_frame(self,
                     timestep: int,
                     positions: np.ndarray,
                     atomic_numbers: np.ndarray,
                     energy: float,
                     forces: Optional[np.ndarray] = None) -> List[RareEvent]:
        """
        Process a trajectory frame and detect events.
        
        Returns:
            List of detected events
        """
        # Compute current topology
        topology = self._compute_topology(positions, atomic_numbers)
        
        # Store in buffer
        self.trajectory_buffer.append({
            'timestep': timestep,
            'positions': positions.copy(),
            'energy': energy,
            'forces': forces
        })
        self.topology_buffer.append(topology)
        
        # Compute collective variables
        cv_value = self._compute_collective_variable(positions, atomic_numbers)
        self.cv_history.append(cv_value)
        
        # Detect events
        events = []
        
        # Check for bond changes
        bond_events = self._detect_bond_events(timestep)
        events.extend(bond_events)
        
        # Check for transition states
        ts_events = self._detect_transition_states(timestep)
        events.extend(ts_events)
        
        # Check for rare configurations
        rare_events = self._detect_rare_configurations(timestep, positions)
        events.extend(rare_events)
        
        # Record events
        for event in events:
            if timestep - self.last_event_step >= self.min_separation:
                self.detected_events.append(event)
                self.last_event_step = timestep
        
        return events
    
    def _compute_topology(self, 
                         positions: np.ndarray,
                         atomic_numbers: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Compute molecular topology as bond graph."""
        n_atoms = len(positions)
        topology = {}
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                
                # Check if bond exists
                if self._is_bond(atomic_numbers[i], atomic_numbers[j], dist):
                    topology[(i, j)] = dist
        
        return topology
    
    def _is_bond(self, z1: int, z2: int, distance: float) -> bool:
        """Check if two atoms form a bond."""
        # Covalent radii (approximate)
        radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 
                14: 1.11, 15: 1.07, 16: 1.05, 26: 1.32}
        
        r1 = radii.get(z1, 1.0)
        r2 = radii.get(z2, 1.0)
        
        return distance <= 1.3 * (r1 + r2)
    
    def _detect_bond_events(self, timestep: int) -> List[RareEvent]:
        """Detect bond breaking/forming events."""
        events = []
        
        if len(self.topology_buffer) < 2:
            return events
        
        current_topo = self.topology_buffer[-1]
        previous_topo = self.topology_buffer[-2]
        
        # Check for broken bonds
        for bond in previous_topo:
            if bond not in current_topo:
                events.append(RareEvent(
                    event_type=EventType.BOND_BREAKING,
                    timestep=timestep,
                    atom_indices=list(bond),
                    initial_state={'bond_length': previous_topo[bond]},
                    final_state={'bond_length': None},
                    reaction_coordinate=previous_topo[bond],
                    probability=0.5
                ))
        
        # Check for formed bonds
        for bond in current_topo:
            if bond not in previous_topo:
                events.append(RareEvent(
                    event_type=EventType.BOND_FORMING,
                    timestep=timestep,
                    atom_indices=list(bond),
                    initial_state={'bond_length': None},
                    final_state={'bond_length': current_topo[bond]},
                    reaction_coordinate=current_topo[bond],
                    probability=0.5
                ))
        
        return events
    
    def _detect_transition_states(self, timestep: int) -> List[RareEvent]:
        """Detect transition states from trajectory."""
        events = []
        
        if len(self.trajectory_buffer) < 3:
            return events
        
        # Look for energy maxima
        energies = [f['energy'] for f in self.trajectory_buffer]
        
        if len(energies) >= 3:
            # Check for local maximum
            if energies[-2] > energies[-3] and energies[-2] > energies[-1]:
                # Potential transition state
                barrier = energies[-2] - min(energies[-3], energies[-1])
                
                events.append(RareEvent(
                    event_type=EventType.REACTION,
                    timestep=timestep - 1,
                    atom_indices=[],
                    initial_state={'energy': energies[-3]},
                    final_state={'energy': energies[-1]},
                    reaction_coordinate=0.5,
                    energy_barrier=barrier,
                    probability=np.exp(-barrier / 0.025)  # kT at room temp
                ))
        
        return events
    
    def _detect_rare_configurations(self, 
                                   timestep: int,
                                   positions: np.ndarray) -> List[RareEvent]:
        """Detect rare configurations using statistical analysis."""
        events = []
        
        # This would use more sophisticated analysis in practice
        # For now, placeholder for rare configuration detection
        
        return events
    
    def _compute_collective_variable(self, 
                                    positions: np.ndarray,
                                    atomic_numbers: np.ndarray) -> float:
        """Compute a collective variable for the configuration."""
        # Example: average coordination number
        n_atoms = len(positions)
        total_coordination = 0
        
        for i in range(n_atoms):
            coordination = 0
            for j in range(n_atoms):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    # Smooth cutoff for coordination
                    coordination += 0.5 * (1 + np.tanh((3.0 - dist) / 0.5))
            total_coordination += coordination
        
        return total_coordination / n_atoms
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected events."""
        if not self.detected_events:
            return {'n_events': 0}
        
        event_types = [e.event_type.name for e in self.detected_events]
        
        return {
            'n_events': len(self.detected_events),
            'event_type_distribution': {t: event_types.count(t) for t in set(event_types)},
            'average_barrier': np.mean([e.energy_barrier for e in self.detected_events 
                                       if e.energy_barrier is not None]),
            'event_times': [e.timestep for e in self.detected_events]
        }


class ParallelEfficiencyOptimizer:
    """
    Optimizes parallel efficiency for multi-scale simulations.
    """
    
    def __init__(self,
                 n_workers: int = 4,
                 batch_size: int = 32,
                 load_balance_threshold: float = 0.2):
        """
        Initialize parallel efficiency optimizer.
        
        Args:
            n_workers: Number of parallel workers
            batch_size: Default batch size for processing
            load_balance_threshold: Load imbalance tolerance
        """
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.balance_threshold = load_balance_threshold
        
        self.worker_loads: Dict[int, float] = {i: 0.0 for i in range(n_workers)}
        self.task_history: deque = deque(maxlen=1000)
        
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
    def optimize_task_distribution(self,
                                  tasks: List[Dict[str, Any]]) -> List[List[Dict]]:
        """
        Optimize distribution of tasks across workers.
        
        Args:
            tasks: List of task dictionaries with 'estimated_cost'
            
        Returns:
            List of task assignments for each worker
        """
        n_tasks = len(tasks)
        
        # Sort tasks by estimated cost (largest first for better packing)
        sorted_tasks = sorted(enumerate(tasks), 
                            key=lambda x: x[1].get('estimated_cost', 1.0),
                            reverse=True)
        
        # Greedy assignment using min-heap
        worker_queues = [(0.0, i, []) for i in range(self.n_workers)]
        heapq.heapify(worker_queues)
        
        for original_idx, task in sorted_tasks:
            load, worker_id, queue = heapq.heappop(worker_queues)
            queue.append(task)
            new_load = load + task.get('estimated_cost', 1.0)
            heapq.heappush(worker_queues, (new_load, worker_id, queue))
        
        # Extract assignments
        assignments = [[] for _ in range(self.n_workers)]
        for _, worker_id, queue in worker_queues:
            assignments[worker_id] = queue
        
        return assignments
    
    def optimize_batch_size(self,
                           recent_times: List[float],
                           target_time: float = 1.0) -> int:
        """
        Dynamically adjust batch size based on performance.
        
        Args:
            recent_times: Recent processing times
            target_time: Target time per batch (seconds)
            
        Returns:
            Optimized batch size
        """
        if not recent_times:
            return self.batch_size
        
        mean_time = np.mean(recent_times)
        
        # Adjust batch size to hit target time
        if mean_time > 0:
            optimal_batch = int(self.batch_size * target_time / mean_time)
            # Limit change rate
            max_change = int(self.batch_size * 0.2)
            optimal_batch = max(self.batch_size - max_change,
                              min(self.batch_size + max_change, optimal_batch))
        else:
            optimal_batch = self.batch_size
        
        # Keep within reasonable bounds
        optimal_batch = max(4, min(256, optimal_batch))
        
        self.batch_size = optimal_batch
        return optimal_batch
    
    def analyze_parallel_efficiency(self,
                                   worker_times: List[float]) -> Dict[str, Any]:
        """
        Analyze parallel efficiency.
        
        Args:
            worker_times: Processing times for each worker
            
        Returns:
            Efficiency analysis
        """
        if not worker_times or len(worker_times) < 2:
            return {'efficiency': 1.0}
        
        max_time = max(worker_times)
        total_time = sum(worker_times)
        
        # Parallel efficiency
        efficiency = total_time / (len(worker_times) * max_time) if max_time > 0 else 1.0
        
        # Load imbalance
        mean_time = np.mean(worker_times)
        std_time = np.std(worker_times)
        imbalance = std_time / mean_time if mean_time > 0 else 0.0
        
        analysis = {
            'efficiency': efficiency,
            'load_imbalance': imbalance,
            'is_balanced': imbalance <= self.balance_threshold,
            'max_time': max_time,
            'min_time': min(worker_times),
            'mean_time': mean_time
        }
        
        self.performance_metrics['efficiency'].append(efficiency)
        self.performance_metrics['imbalance'].append(imbalance)
        
        return analysis
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on performance history."""
        suggestions = []
        
        if len(self.performance_metrics['efficiency']) < 10:
            return suggestions
        
        recent_eff = list(self.performance_metrics['efficiency'])[-10:]
        avg_efficiency = np.mean(recent_eff)
        
        if avg_efficiency < 0.7:
            suggestions.append("Consider reducing number of workers")
            suggestions.append("Increase batch size to reduce overhead")
        
        recent_imb = list(self.performance_metrics['imbalance'])[-10:]
        avg_imbalance = np.mean(recent_imb)
        
        if avg_imbalance > self.balance_threshold:
            suggestions.append("Improve load balancing with better task estimation")
            suggestions.append("Consider dynamic task stealing")
        
        return suggestions
    
    def record_task_completion(self,
                              worker_id: int,
                              task_cost: float,
                              actual_time: float) -> None:
        """Record completed task for performance tracking."""
        self.task_history.append({
            'worker_id': worker_id,
            'estimated_cost': task_cost,
            'actual_time': actual_time,
            'error': abs(task_cost - actual_time) / (actual_time + 1e-6)
        })
        
        self.worker_loads[worker_id] += actual_time
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'n_workers': self.n_workers,
            'current_batch_size': self.batch_size,
            'worker_loads': self.worker_loads.copy(),
            'n_tasks_completed': len(self.task_history)
        }
        
        for metric, values in self.performance_metrics.items():
            if values:
                summary[f'{metric}_history'] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'recent': list(values)[-10:]
                }
        
        return summary


class SmartSamplingManager:
    """
    Main manager for smart sampling strategies.
    
    Orchestrates adaptive importance sampling, rare event detection,
    and parallel efficiency optimization.
    """
    
    def __init__(self,
                 sampler: Optional[BaseSampler] = None,
                 event_detector: Optional[RareEventDetector] = None,
                 parallel_optimizer: Optional[ParallelEfficiencyOptimizer] = None,
                 temperature: float = 300.0):
        """
        Initialize smart sampling manager.
        
        Args:
            sampler: Adaptive importance sampler
            event_detector: Rare event detector
            parallel_optimizer: Parallel efficiency optimizer
            temperature: Simulation temperature
        """
        self.sampler = sampler or AdaptiveImportanceSampler(temperature=temperature)
        self.event_detector = event_detector or RareEventDetector()
        self.parallel_optimizer = parallel_optimizer or ParallelEfficiencyOptimizer()
        
        self.temperature = temperature
        self.step_count = 0
        
        self.sampling_history: deque = deque(maxlen=1000)
        
    def sample_configurations(self,
                             positions_batch: np.ndarray,
                             energies: np.ndarray,
                             uncertainties: Optional[np.ndarray] = None,
                             n_samples: int = 10) -> np.ndarray:
        """
        Sample configurations for DFT evaluation.
        
        Returns:
            Indices of selected configurations
        """
        indices = self.sampler.sample(positions_batch, energies, uncertainties, n_samples)
        
        self.sampling_history.append({
            'step': self.step_count,
            'n_candidates': len(positions_batch),
            'n_selected': len(indices),
            'selected_indices': indices.tolist()
        })
        
        return indices
    
    def process_trajectory_frame(self,
                                timestep: int,
                                positions: np.ndarray,
                                atomic_numbers: np.ndarray,
                                energy: float,
                                forces: Optional[np.ndarray] = None) -> List[RareEvent]:
        """Process trajectory frame and detect events."""
        self.step_count = timestep
        
        events = self.event_detector.process_frame(
            timestep, positions, atomic_numbers, energy, forces
        )
        
        return events
    
    def optimize_parallel_execution(self,
                                   tasks: List[Dict[str, Any]],
                                   worker_times: Optional[List[float]] = None) -> Dict[str, Any]:
        """Optimize parallel task execution."""
        # Distribute tasks
        assignments = self.parallel_optimizer.optimize_task_distribution(tasks)
        
        # Analyze efficiency if times available
        efficiency = None
        if worker_times:
            efficiency = self.parallel_optimizer.analyze_parallel_efficiency(worker_times)
        
        # Get optimization suggestions
        suggestions = self.parallel_optimizer.suggest_optimizations()
        
        return {
            'task_assignments': assignments,
            'efficiency_analysis': efficiency,
            'optimization_suggestions': suggestions
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary."""
        return {
            'step_count': self.step_count,
            'sampling_stats': self.sampler.get_sampling_statistics(),
            'event_stats': self.event_detector.get_event_statistics(),
            'parallel_stats': self.parallel_optimizer.get_performance_summary()
        }


# Example usage
if __name__ == "__main__":
    print("=== Smart Sampling Module Demo ===\n")
    
    np.random.seed(42)
    
    # Create manager
    manager = SmartSamplingManager()
    
    # Simulate sampling
    print("--- Adaptive Importance Sampling ---")
    for step in range(5):
        n_configs = 100
        positions_batch = np.random.randn(n_configs, 50, 3) * 5
        energies = np.random.randn(n_configs) * 0.5
        uncertainties = np.random.rand(n_configs) * 0.2
        
        indices = manager.sample_configurations(
            positions_batch, energies, uncertainties, n_samples=10
        )
        
        print(f"Step {step+1}: Selected {len(indices)} configs from {n_configs}")
    
    print("\n--- Rare Event Detection ---")
    # Simulate trajectory with bond breaking
    n_atoms = 20
    for step in range(20):
        # Simulate atoms moving
        positions = np.random.randn(n_atoms, 3) * 5
        positions[0] = positions[1] + np.array([0.1, 0, 0]) if step < 10 else positions[1] + np.array([5.0, 0, 0])
        
        atomic_numbers = np.ones(n_atoms, dtype=int)
        energy = np.random.randn() * 0.1
        
        events = manager.process_trajectory_frame(
            step, positions, atomic_numbers, energy
        )
        
        if events:
            for event in events:
                print(f"Step {step}: Detected {event.event_type.name} "
                      f"for atoms {event.atom_indices}")
    
    print("\n--- Summary ---")
    summary = manager.get_summary()
    print(f"Sampling: {summary['sampling_stats']}")
    print(f"Events: {summary['event_stats']}")
