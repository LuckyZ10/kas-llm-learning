#!/usr/bin/env python3
"""
Resolution Adapter Module - Automatic Multiscale Resolution Switching

This module provides intelligent switching between DFT and ML potentials
based on error estimates, computational cost, and accuracy requirements.

Author: DFTLammps Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time
import json
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResolutionLevel(Enum):
    """Enumeration of available resolution levels."""
    DFT_HIGH = auto()      # High-accuracy DFT (e.g., HSE06)
    DFT_STANDARD = auto()  # Standard DFT (e.g., PBE)
    ML_DENSE = auto()      # Dense neural network potential
    ML_STANDARD = auto()   # Standard ML potential
    ML_FAST = auto()       # Fast ML potential (lightweight model)
    MM_CLASSICAL = auto()  # Classical force field


class SwitchTrigger(Enum):
    """Reasons for resolution switching."""
    ERROR_THRESHOLD = auto()
    COST_BUDGET = auto()
    USER_REQUEST = auto()
    ADAPTIVE_SAMPLING = auto()
    RARE_EVENT = auto()
    BOND_BREAKING = auto()
    GEOMETRY_CHANGE = auto()
    TEMPERATURE_SPIKE = auto()


@dataclass
class ComputationalMetrics:
    """Metrics for computational performance tracking."""
    wall_time: float = 0.0
    cpu_time: float = 0.0
    memory_usage_mb: float = 0.0
    energy_evaluations: int = 0
    force_evaluations: int = 0
    scf_cycles: int = 0
    parallel_efficiency: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'wall_time': self.wall_time,
            'cpu_time': self.cpu_time,
            'memory_usage_mb': self.memory_usage_mb,
            'energy_evaluations': self.energy_evaluations,
            'force_evaluations': self.force_evaluations,
            'scf_cycles': self.scf_cycles,
            'parallel_efficiency': self.parallel_efficiency
        }


@dataclass
class AccuracyMetrics:
    """Metrics for accuracy assessment."""
    energy_rmse: float = 0.0
    force_rmse: float = 0.0
    stress_rmse: float = 0.0
    max_force_error: float = 0.0
    confidence_score: float = 1.0
    uncertainty_quantile: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'energy_rmse': self.energy_rmse,
            'force_rmse': self.force_rmse,
            'stress_rmse': self.stress_rmse,
            'max_force_error': self.max_force_error,
            'confidence_score': self.confidence_score,
            'uncertainty_quantile': self.uncertainty_quantile
        }


@dataclass
class ResolutionState:
    """Complete state information for a resolution level."""
    level: ResolutionLevel
    metrics: ComputationalMetrics = field(default_factory=ComputationalMetrics)
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    cost_per_atom_ms: float = 0.0
    accuracy_rating: float = 0.0
    active: bool = False
    last_used: float = 0.0
    usage_count: int = 0
    
    def compute_efficiency_score(self) -> float:
        """Compute efficiency score (accuracy per unit cost)."""
        if self.cost_per_atom_ms <= 0:
            return 0.0
        return self.accuracy_rating / self.cost_per_atom_ms


class CostAccuracyTradeoff:
    """
    Manages the tradeoff between computational cost and accuracy.
    
    Uses Pareto frontier analysis to find optimal resolution levels
    for given constraints.
    """
    
    def __init__(self, target_accuracy: float = 0.001, 
                 budget_ms_per_step: float = 1000.0):
        """
        Initialize tradeoff manager.
        
        Args:
            target_accuracy: Target accuracy (eV/atom for energy)
            budget_ms_per_step: Computational budget per step (ms)
        """
        self.target_accuracy = target_accuracy
        self.budget_ms_per_step = budget_ms_per_step
        self.resolution_states: Dict[ResolutionLevel, ResolutionState] = {}
        self.history: deque = deque(maxlen=1000)
        self.adaptive_weights = {'accuracy': 0.5, 'cost': 0.5}
        
    def register_resolution(self, state: ResolutionState) -> None:
        """Register a resolution level with its performance metrics."""
        self.resolution_states[state.level] = state
        logger.info(f"Registered resolution {state.level.name} "
                   f"(cost={state.cost_per_atom_ms:.2f}ms, "
                   f"accuracy={state.accuracy_rating:.4f})")
    
    def update_metrics(self, level: ResolutionLevel, 
                      metrics: ComputationalMetrics,
                      accuracy: AccuracyMetrics) -> None:
        """Update metrics for a resolution level."""
        if level in self.resolution_states:
            state = self.resolution_states[level]
            state.metrics = metrics
            state.accuracy = accuracy
            state.last_used = time.time()
            state.usage_count += 1
            
            # Update cost estimate with exponential moving average
            alpha = 0.3
            current_cost = metrics.wall_time / max(metrics.energy_evaluations, 1)
            state.cost_per_atom_ms = (alpha * current_cost + 
                                     (1 - alpha) * state.cost_per_atom_ms)
            
            # Update accuracy rating
            state.accuracy_rating = self._compute_accuracy_rating(accuracy)
            
            self.history.append({
                'timestamp': time.time(),
                'level': level,
                'cost': state.cost_per_atom_ms,
                'accuracy': state.accuracy_rating
            })
    
    def _compute_accuracy_rating(self, accuracy: AccuracyMetrics) -> float:
        """Compute normalized accuracy rating from accuracy metrics."""
        # Weighted combination of accuracy metrics
        energy_score = max(0, 1.0 - accuracy.energy_rmse / 0.1)
        force_score = max(0, 1.0 - accuracy.force_rmse / 1.0)
        confidence_score = accuracy.confidence_score
        
        return 0.4 * energy_score + 0.4 * force_score + 0.2 * confidence_score
    
    def find_pareto_optimal(self) -> List[ResolutionLevel]:
        """Find Pareto-optimal resolution levels."""
        states = list(self.resolution_states.values())
        pareto_optimal = []
        
        for i, state_i in enumerate(states):
            dominated = False
            for j, state_j in enumerate(states):
                if i != j:
                    # state_j dominates state_i if it's better in at least one
                    # objective and not worse in any other
                    better_cost = state_j.cost_per_atom_ms <= state_i.cost_per_atom_ms
                    better_accuracy = state_j.accuracy_rating >= state_i.accuracy_rating
                    strictly_better = (state_j.cost_per_atom_ms < state_i.cost_per_atom_ms or 
                                     state_j.accuracy_rating > state_i.accuracy_rating)
                    
                    if better_cost and better_accuracy and strictly_better:
                        dominated = True
                        break
            
            if not dominated:
                pareto_optimal.append(state_i.level)
        
        return pareto_optimal
    
    def select_optimal_resolution(self, 
                                 accuracy_requirement: Optional[float] = None,
                                 cost_constraint: Optional[float] = None,
                                 system_size: int = 100) -> ResolutionLevel:
        """
        Select optimal resolution based on constraints.
        
        Args:
            accuracy_requirement: Required accuracy (if None, use target)
            cost_constraint: Maximum cost per step (if None, use budget)
            system_size: Number of atoms in system
            
        Returns:
            Optimal resolution level
        """
        accuracy_req = accuracy_requirement or self.target_accuracy
        cost_limit = cost_constraint or self.budget_ms_per_step
        
        # Calculate effective cost per step
        effective_costs = {
            level: state.cost_per_atom_ms * system_size
            for level, state in self.resolution_states.items()
        }
        
        # Score each resolution
        scores = {}
        for level, state in self.resolution_states.items():
            # Accuracy component: how well it meets requirement
            accuracy_diff = state.accuracy_rating - accuracy_req
            accuracy_score = 1.0 / (1.0 + np.exp(-10 * accuracy_diff))  # Sigmoid
            
            # Cost component: how well it fits budget
            cost_ratio = effective_costs[level] / cost_limit
            cost_score = 1.0 / (1.0 + cost_ratio)
            
            # Combined score with adaptive weights
            combined_score = (self.adaptive_weights['accuracy'] * accuracy_score +
                            self.adaptive_weights['cost'] * cost_score)
            
            # Penalty for insufficient accuracy
            if accuracy_diff < 0:
                combined_score *= 0.5
            
            scores[level] = combined_score
        
        # Select best scoring resolution
        if scores:
            best_level = max(scores, key=scores.get)
            return best_level
        
        # Fallback to standard ML
        return ResolutionLevel.ML_STANDARD
    
    def update_adaptive_weights(self, recent_errors: List[float]) -> None:
        """Update weighting based on recent performance."""
        if not recent_errors:
            return
        
        mean_error = np.mean(recent_errors)
        
        # If errors are high, prioritize accuracy
        if mean_error > self.target_accuracy * 2:
            self.adaptive_weights['accuracy'] = min(0.8, 
                self.adaptive_weights['accuracy'] + 0.1)
            self.adaptive_weights['cost'] = 1.0 - self.adaptive_weights['accuracy']
        # If errors are low, can prioritize cost
        elif mean_error < self.target_accuracy * 0.5:
            self.adaptive_weights['cost'] = min(0.8,
                self.adaptive_weights['cost'] + 0.1)
            self.adaptive_weights['accuracy'] = 1.0 - self.adaptive_weights['cost']


class ResolutionSwitcher:
    """
    Manages dynamic switching between resolution levels.
    
    Implements hysteresis to prevent rapid oscillation between levels.
    """
    
    def __init__(self, 
                 tradeoff_manager: CostAccuracyTradeoff,
                 hysteresis_threshold: float = 0.15,
                 min_steps_at_level: int = 5,
                 max_history: int = 100):
        """
        Initialize resolution switcher.
        
        Args:
            tradeoff_manager: Cost-accuracy tradeoff manager
            hysteresis_threshold: Minimum improvement to switch (fraction)
            min_steps_at_level: Minimum steps before allowing switch
            max_history: Maximum history to maintain
        """
        self.tradeoff = tradeoff_manager
        self.hysteresis = hysteresis_threshold
        self.min_steps = min_steps_at_level
        
        self.current_level: Optional[ResolutionLevel] = None
        self.steps_at_current = 0
        self.switch_history: deque = deque(maxlen=max_history)
        
        self.error_estimator: Optional[Any] = None
        self.callbacks: List[Callable] = []
        
    def set_error_estimator(self, estimator: Any) -> None:
        """Set the error estimator for confidence assessment."""
        self.error_estimator = estimator
    
    def register_callback(self, callback: Callable[[ResolutionLevel, 
                                                    ResolutionLevel, 
                                                    SwitchTrigger], None]) -> None:
        """Register callback for switch events."""
        self.callbacks.append(callback)
    
    def initialize(self, default_level: ResolutionLevel = ResolutionLevel.ML_STANDARD,
                   system_size: int = 100) -> None:
        """Initialize with a default resolution level."""
        self.current_level = default_level
        self.steps_at_current = 0
        
        if self.current_level in self.tradeoff.resolution_states:
            self.tradeoff.resolution_states[self.current_level].active = True
        
        logger.info(f"Initialized resolution switcher at {default_level.name}")
    
    def evaluate_switch(self, 
                       current_error: float,
                       system_size: int = 100,
                       geometry_changed: bool = False) -> Tuple[ResolutionLevel, bool]:
        """
        Evaluate whether to switch resolution level.
        
        Args:
            current_error: Current error estimate
            system_size: Number of atoms
            geometry_changed: Whether significant geometry change occurred
            
        Returns:
            Tuple of (recommended_level, should_switch)
        """
        if self.current_level is None:
            self.initialize(system_size=system_size)
        
        self.steps_at_current += 1
        
        # Get current state
        current_state = self.tradeoff.resolution_states.get(self.current_level)
        if current_state is None:
            return self.current_level, False
        
        # Determine if switch is needed
        trigger = None
        
        # Check error threshold
        if current_error > self.tradeoff.target_accuracy * 2:
            trigger = SwitchTrigger.ERROR_THRESHOLD
        
        # Check geometry changes
        if geometry_changed:
            trigger = SwitchTrigger.GEOMETRY_CHANGE
        
        # If no trigger, stay at current level
        if trigger is None and self.steps_at_current >= self.min_steps:
            # Check if we can downgrade for efficiency
            optimal = self.tradeoff.select_optimal_resolution(
                system_size=system_size
            )
            if optimal != self.current_level:
                # Check hysteresis
                optimal_state = self.tradeoff.resolution_states[optimal]
                current_efficiency = current_state.compute_efficiency_score()
                optimal_efficiency = optimal_state.compute_efficiency_score()
                
                improvement = (optimal_efficiency - current_efficiency) / \
                             (current_efficiency + 1e-10)
                
                if improvement > self.hysteresis:
                    return optimal, True
        
        # If triggered to upgrade
        if trigger is not None:
            # Find higher resolution levels
            higher_levels = [l for l in ResolutionLevel 
                           if self._is_higher_resolution(l, self.current_level)]
            
            if higher_levels and self.steps_at_current >= self.min_steps:
                # Select best higher level
                best_level = min(higher_levels, 
                                key=lambda l: self.tradeoff.resolution_states[l].cost_per_atom_ms)
                return best_level, True
        
        return self.current_level, False
    
    def _is_higher_resolution(self, level1: ResolutionLevel, 
                             level2: ResolutionLevel) -> bool:
        """Check if level1 is higher resolution than level2."""
        hierarchy = [
            ResolutionLevel.MM_CLASSICAL,
            ResolutionLevel.ML_FAST,
            ResolutionLevel.ML_STANDARD,
            ResolutionLevel.ML_DENSE,
            ResolutionLevel.DFT_STANDARD,
            ResolutionLevel.DFT_HIGH
        ]
        
        try:
            idx1 = hierarchy.index(level1)
            idx2 = hierarchy.index(level2)
            return idx1 > idx2
        except ValueError:
            return False
    
    def execute_switch(self, new_level: ResolutionLevel, 
                      trigger: SwitchTrigger) -> None:
        """Execute resolution switch."""
        if self.current_level == new_level:
            return
        
        old_level = self.current_level
        
        # Update states
        if old_level in self.tradeoff.resolution_states:
            self.tradeoff.resolution_states[old_level].active = False
        
        if new_level in self.tradeoff.resolution_states:
            self.tradeoff.resolution_states[new_level].active = True
        
        # Record switch
        self.switch_history.append({
            'timestamp': time.time(),
            'from': old_level,
            'to': new_level,
            'trigger': trigger,
            'steps_at_previous': self.steps_at_current
        })
        
        self.current_level = new_level
        self.steps_at_current = 0
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(old_level, new_level, trigger)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
        
        logger.info(f"Switched from {old_level.name} to {new_level.name} "
                   f"(trigger: {trigger.name})")
    
    def get_switch_statistics(self) -> Dict[str, Any]:
        """Get statistics about resolution switching."""
        if not self.switch_history:
            return {'total_switches': 0}
        
        switches = list(self.switch_history)
        triggers = [s['trigger'].name for s in switches]
        
        return {
            'total_switches': len(switches),
            'average_steps_per_level': np.mean([s['steps_at_previous'] 
                                               for s in switches]),
            'trigger_distribution': {t: triggers.count(t) for t in set(triggers)},
            'current_level': self.current_level.name if self.current_level else None,
            'steps_at_current': self.steps_at_current
        }


class AdaptiveResolutionManager:
    """
    Main manager for adaptive multiscale resolution.
    
    Orchestrates resolution switching, error estimation, and
    cost-accuracy tradeoffs in an integrated framework.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adaptive resolution manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.tradeoff = CostAccuracyTradeoff(
            target_accuracy=self.config.get('target_accuracy', 0.001),
            budget_ms_per_step=self.config.get('budget_ms', 1000.0)
        )
        
        self.switcher = ResolutionSwitcher(
            tradeoff_manager=self.tradeoff,
            hysteresis_threshold=self.config.get('hysteresis', 0.15),
            min_steps_at_level=self.config.get('min_steps', 5)
        )
        
        # State tracking
        self.step_count = 0
        self.error_history: deque = deque(maxlen=100)
        self.performance_log: List[Dict] = []
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Callbacks for integration
        self.pre_step_callbacks: List[Callable] = []
        self.post_step_callbacks: List[Callable] = []
        
    def register_resolution_level(self, level: ResolutionLevel,
                                   cost_per_atom_ms: float,
                                   accuracy_rating: float) -> None:
        """Register a resolution level with estimated performance."""
        state = ResolutionState(
            level=level,
            cost_per_atom_ms=cost_per_atom_ms,
            accuracy_rating=accuracy_rating
        )
        self.tradeoff.register_resolution(state)
    
    def add_pre_step_callback(self, callback: Callable) -> None:
        """Add callback to run before each step."""
        self.pre_step_callbacks.append(callback)
    
    def add_post_step_callback(self, callback: Callable) -> None:
        """Add callback to run after each step."""
        self.post_step_callbacks.append(callback)
    
    def step(self, 
             positions: np.ndarray,
             forces_ml: np.ndarray,
             uncertainty: float,
             system_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute one adaptive resolution step.
        
        Args:
            positions: Atomic positions (N, 3)
            forces_ml: ML-predicted forces (N, 3)
            uncertainty: Uncertainty estimate from ML model
            system_context: Additional system information
            
        Returns:
            Dictionary with step results and recommendations
        """
        with self._lock:
            self.step_count += 1
            
            # Run pre-step callbacks
            for callback in self.pre_step_callbacks:
                callback()
            
            # Evaluate current situation
            system_size = len(positions)
            context = system_context or {}
            geometry_changed = context.get('geometry_changed', False)
            
            # Store error in history
            self.error_history.append(uncertainty)
            
            # Update adaptive weights periodically
            if self.step_count % 10 == 0:
                self.tradeoff.update_adaptive_weights(list(self.error_history))
            
            # Evaluate switch
            recommended_level, should_switch = self.switcher.evaluate_switch(
                current_error=uncertainty,
                system_size=system_size,
                geometry_changed=geometry_changed
            )
            
            result = {
                'step': self.step_count,
                'current_level': self.switcher.current_level,
                'recommended_level': recommended_level,
                'should_switch': should_switch,
                'uncertainty': uncertainty,
                'system_size': system_size
            }
            
            # Execute switch if needed
            if should_switch:
                trigger = SwitchTrigger.ERROR_THRESHOLD if uncertainty > \
                    self.tradeoff.target_accuracy * 2 else SwitchTrigger.ADAPTIVE_SAMPLING
                self.switcher.execute_switch(recommended_level, trigger)
                result['switch_executed'] = True
                result['switch_trigger'] = trigger
            else:
                result['switch_executed'] = False
            
            # Run post-step callbacks
            for callback in self.post_step_callbacks:
                callback(result)
            
            # Log performance
            self.performance_log.append({
                'step': self.step_count,
                'timestamp': time.time(),
                'level': self.switcher.current_level.name if self.switcher.current_level else None,
                'uncertainty': uncertainty,
                'system_size': system_size
            })
            
            return result
    
    def request_dft_evaluation(self, 
                              positions: np.ndarray,
                              subset_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Request DFT evaluation for specific atoms.
        
        Args:
            positions: Full atomic positions
            subset_indices: Indices to evaluate with DFT (None = all)
            
        Returns:
            DFT evaluation request specification
        """
        if subset_indices is None:
            # Automatically select high-uncertainty atoms
            if self.error_estimator is not None:
                subset_indices = self.error_estimator.select_high_uncertainty_atoms(
                    positions, n_atoms=min(50, len(positions) // 10)
                )
            else:
                subset_indices = list(range(min(50, len(positions))))
        
        return {
            'method': 'DFT',
            'level': ResolutionLevel.DFT_STANDARD,
            'indices': subset_indices,
            'positions': positions[subset_indices],
            'priority': 'high' if self.error_history and 
                       self.error_history[-1] > self.tradeoff.target_accuracy * 3 else 'normal'
        }
    
    def set_error_estimator(self, estimator: Any) -> None:
        """Set the error estimator."""
        self.error_estimator = estimator
        self.switcher.set_error_estimator(estimator)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive resolution performance."""
        return {
            'total_steps': self.step_count,
            'switch_statistics': self.switcher.get_switch_statistics(),
            'current_weights': self.tradeoff.adaptive_weights,
            'error_history_stats': {
                'mean': np.mean(self.error_history) if self.error_history else 0,
                'max': np.max(self.error_history) if self.error_history else 0,
                'recent_mean': np.mean(list(self.error_history)[-10:]) \
                              if len(self.error_history) >= 10 else \
                              np.mean(self.error_history) if self.error_history else 0
            },
            'pareto_optimal_levels': [l.name for l in self.tradeoff.find_pareto_optimal()]
        }
    
    def save_state(self, filepath: str) -> None:
        """Save manager state to file."""
        state = {
            'config': self.config,
            'step_count': self.step_count,
            'switch_statistics': self.switcher.get_switch_statistics(),
            'performance_log': self.performance_log[-100:],  # Last 100 entries
            'error_history': list(self.error_history),
            'resolution_states': {
                level.name: {
                    'cost_per_atom_ms': state.cost_per_atom_ms,
                    'accuracy_rating': state.accuracy_rating,
                    'usage_count': state.usage_count
                }
                for level, state in self.tradeoff.resolution_states.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved adaptive resolution state to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load manager state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config = state.get('config', self.config)
        self.step_count = state.get('step_count', 0)
        self.error_history = deque(state.get('error_history', []), maxlen=100)
        self.performance_log = state.get('performance_log', [])
        
        # Restore resolution states
        for level_name, level_state in state.get('resolution_states', {}).items():
            try:
                level = ResolutionLevel[level_name]
                if level in self.tradeoff.resolution_states:
                    self.tradeoff.resolution_states[level].cost_per_atom_ms = \
                        level_state['cost_per_atom_ms']
                    self.tradeoff.resolution_states[level].accuracy_rating = \
                        level_state['accuracy_rating']
                    self.tradeoff.resolution_states[level].usage_count = \
                        level_state['usage_count']
            except KeyError:
                pass
        
        logger.info(f"Loaded adaptive resolution state from {filepath}")


def create_default_manager(target_accuracy: float = 0.001,
                          budget_ms: float = 1000.0) -> AdaptiveResolutionManager:
    """
    Create a default adaptive resolution manager with standard settings.
    
    Args:
        target_accuracy: Target accuracy in eV/atom
        budget_ms: Computational budget per step in milliseconds
        
    Returns:
        Configured AdaptiveResolutionManager
    """
    manager = AdaptiveResolutionManager({
        'target_accuracy': target_accuracy,
        'budget_ms': budget_ms,
        'hysteresis': 0.15,
        'min_steps': 5
    })
    
    # Register default resolution levels with typical costs
    # Costs are in ms per atom
    manager.register_resolution_level(
        ResolutionLevel.MM_CLASSICAL, 
        cost_per_atom_ms=0.01, 
        accuracy_rating=0.3
    )
    manager.register_resolution_level(
        ResolutionLevel.ML_FAST,
        cost_per_atom_ms=0.1,
        accuracy_rating=0.6
    )
    manager.register_resolution_level(
        ResolutionLevel.ML_STANDARD,
        cost_per_atom_ms=1.0,
        accuracy_rating=0.85
    )
    manager.register_resolution_level(
        ResolutionLevel.ML_DENSE,
        cost_per_atom_ms=5.0,
        accuracy_rating=0.95
    )
    manager.register_resolution_level(
        ResolutionLevel.DFT_STANDARD,
        cost_per_atom_ms=100.0,
        accuracy_rating=0.99
    )
    manager.register_resolution_level(
        ResolutionLevel.DFT_HIGH,
        cost_per_atom_ms=500.0,
        accuracy_rating=0.999
    )
    
    manager.switcher.initialize(default_level=ResolutionLevel.ML_STANDARD)
    
    return manager


# Example usage and testing
if __name__ == "__main__":
    # Create manager
    manager = create_default_manager(target_accuracy=0.05, budget_ms=2000.0)
    
    print("=== Adaptive Resolution Manager Demo ===\n")
    
    # Simulate MD steps
    np.random.seed(42)
    n_atoms = 200
    
    for step in range(20):
        # Generate fake positions and forces
        positions = np.random.randn(n_atoms, 3) * 5
        forces_ml = np.random.randn(n_atoms, 3) * 0.1
        
        # Simulate varying uncertainty
        base_uncertainty = 0.02 + 0.03 * np.sin(step * 0.3)
        uncertainty = max(0.001, base_uncertainty + np.random.randn() * 0.005)
        
        # Run adaptive step
        result = manager.step(
            positions=positions,
            forces_ml=forces_ml,
            uncertainty=uncertainty,
            system_context={'geometry_changed': step == 10}
        )
        
        print(f"Step {step+1}: Level={result['current_level'].name if result['current_level'] else None}, "
              f"Uncertainty={uncertainty:.4f}, Switch={result['should_switch']}")
        
        if result['should_switch']:
            print(f"  -> Switched to {result['recommended_level'].name}")
    
    print("\n=== Summary ===")
    summary = manager.get_summary()
    print(f"Total steps: {summary['total_steps']}")
    print(f"Switches: {summary['switch_statistics']['total_switches']}")
    print(f"Pareto-optimal levels: {summary['pareto_optimal_levels']}")
