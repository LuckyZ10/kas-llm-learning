#!/usr/bin/env python3
"""
Error Estimators Module - ML Potential Confidence and Uncertainty Quantification

This module provides comprehensive error estimation capabilities for ML potentials,
including ensemble-based uncertainty, gradient-based sensitivity analysis,
and Bayesian neural network approaches.

Author: DFTLammps Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """Container for comprehensive uncertainty information."""
    total_uncertainty: float
    aleatoric_uncertainty: float  # Data noise
    epistemic_uncertainty: float  # Model uncertainty
    
    # Per-atom uncertainties
    per_atom_energy: Optional[np.ndarray] = None
    per_atom_forces: Optional[np.ndarray] = None
    
    # Component breakdown
    ensemble_variance: float = 0.0
    gradient_sensitivity: float = 0.0
    distance_to_training: float = 0.0
    
    # Confidence metrics
    confidence_score: float = 0.0  # 0-1 scale
    reliability_diagnostic: Dict[str, float] = field(default_factory=dict)
    
    def is_reliable(self, threshold: float = 0.5) -> bool:
        """Check if estimate is reliable based on confidence."""
        return self.confidence_score >= threshold
    
    def requires_dft(self, threshold: float = 0.1) -> bool:
        """Determine if DFT validation is required."""
        return self.total_uncertainty > threshold


class BaseErrorEstimator(ABC):
    """Abstract base class for error estimators."""
    
    @abstractmethod
    def estimate(self, 
                 positions: np.ndarray,
                 atomic_numbers: np.ndarray,
                 **kwargs) -> UncertaintyEstimate:
        """Estimate uncertainty for given configuration."""
        pass
    
    @abstractmethod
    def update(self, 
               positions: np.ndarray,
               atomic_numbers: np.ndarray,
               dft_energy: float,
               dft_forces: np.ndarray) -> None:
        """Update estimator with new DFT reference data."""
        pass
    
    def batch_estimate(self,
                      positions_list: List[np.ndarray],
                      atomic_numbers_list: List[np.ndarray]) -> List[UncertaintyEstimate]:
        """Estimate uncertainty for multiple configurations."""
        return [self.estimate(pos, atoms) 
                for pos, atoms in zip(positions_list, atomic_numbers_list)]


class EnsembleErrorEstimator(BaseErrorEstimator):
    """
    Error estimator using deep ensemble of ML models.
    
    Based on the principle that disagreement between ensemble members
    indicates regions of high uncertainty.
    """
    
    def __init__(self, 
                 ensemble_models: List[Any],
                 temperature: float = 300.0,
                 energy_weight: float = 0.3,
                 force_weight: float = 0.7):
        """
        Initialize ensemble error estimator.
        
        Args:
            ensemble_models: List of ML model instances
            temperature: System temperature for scaling (K)
            energy_weight: Weight for energy uncertainty
            force_weight: Weight for force uncertainty
        """
        self.models = ensemble_models
        self.n_models = len(ensemble_models)
        self.temperature = temperature
        self.kT = temperature * 8.617333e-5  # Boltzmann constant in eV/K
        
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        
        # Calibration parameters
        self.energy_bias = 0.0
        self.energy_scale = 1.0
        self.force_scale = 1.0
        
        # Training data reference for distance-based correction
        self.training_structures: List[np.ndarray] = []
        self.training_energies: List[float] = []
        
        # Performance tracking
        self.prediction_history: deque = deque(maxlen=1000)
        
    def _get_ensemble_predictions(self, 
                                  positions: np.ndarray,
                                  atomic_numbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from all ensemble members."""
        energies = []
        forces_list = []
        
        for model in self.models:
            try:
                energy, forces = model.predict(positions, atomic_numbers)
                energies.append(energy)
                forces_list.append(forces)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue
        
        if not energies:
            raise RuntimeError("All ensemble models failed to predict")
        
        return np.array(energies), np.array(forces_list)
    
    def estimate(self,
                 positions: np.ndarray,
                 atomic_numbers: np.ndarray,
                 **kwargs) -> UncertaintyEstimate:
        """
        Estimate uncertainty using ensemble disagreement.
        
        Returns:
            UncertaintyEstimate with comprehensive uncertainty metrics
        """
        # Get ensemble predictions
        energies, forces = self._get_ensemble_predictions(positions, atomic_numbers)
        
        n_atoms = len(atomic_numbers)
        
        # Energy statistics
        mean_energy = np.mean(energies)
        energy_variance = np.var(energies)
        energy_std = np.std(energies)
        
        # Force statistics
        mean_forces = np.mean(forces, axis=0)  # Average over ensemble
        force_variance = np.var(forces, axis=0)  # Variance over ensemble
        force_magnitude_variance = np.mean(np.sum(force_variance, axis=1))
        
        # Per-atom force uncertainty
        per_atom_force_std = np.sqrt(np.sum(force_variance, axis=1))
        
        # Compute components
        # Epistemic: ensemble disagreement
        epistemic_energy = energy_variance
        epistemic_force = force_magnitude_variance
        
        # Estimate aleatoric from training data variance
        aleatoric_energy = self._estimate_aleatoric_energy(positions, atomic_numbers)
        aleatoric_force = self._estimate_aleatoric_force(positions, atomic_numbers)
        
        # Distance to training
        distance_factor = self._compute_distance_factor(positions)
        
        # Combine uncertainties
        total_energy_unc = np.sqrt(epistemic_energy + aleatoric_energy**2)
        total_force_unc = np.sqrt(epistemic_force + aleatoric_force**2)
        
        # Weighted combination
        total_uncertainty = (self.energy_weight * total_energy_unc / max(n_atoms * self.kT, 1e-6) +
                           self.force_weight * total_force_unc / max(np.mean(np.abs(mean_forces)) + 1e-6, 1e-6))
        
        # Confidence score (sigmoid of negative uncertainty)
        confidence = 1.0 / (1.0 + np.exp(5 * (total_uncertainty - 0.5)))
        
        # Reliability diagnostics
        diagnostics = {
            'ensemble_agreement': 1.0 - min(1.0, energy_std / (abs(mean_energy) + 1.0)),
            'force_consistency': 1.0 - min(1.0, np.mean(per_atom_force_std) / 
                                          (np.mean(np.linalg.norm(mean_forces, axis=1)) + 1.0)),
            'out_of_distribution': distance_factor
        }
        
        return UncertaintyEstimate(
            total_uncertainty=float(total_uncertainty),
            aleatoric_uncertainty=float(aleatoric_energy),
            epistemic_uncertainty=float(epistemic_energy),
            per_atom_energy=np.full(n_atoms, total_energy_unc / n_atoms),
            per_atom_forces=per_atom_force_std,
            ensemble_variance=float(energy_variance),
            gradient_sensitivity=0.0,  # Computed separately if needed
            distance_to_training=float(distance_factor),
            confidence_score=float(confidence),
            reliability_diagnostic=diagnostics
        )
    
    def _estimate_aleatoric_energy(self, 
                                   positions: np.ndarray,
                                   atomic_numbers: np.ndarray) -> float:
        """Estimate aleatoric uncertainty from training data residuals."""
        if not self.training_energies:
            return 0.01  # Default prior
        
        # Simple estimate based on training variance
        return np.std(self.training_energies) / max(len(atomic_numbers), 1)
    
    def _estimate_aleatoric_force(self,
                                  positions: np.ndarray,
                                  atomic_numbers: np.ndarray) -> float:
        """Estimate aleatoric force uncertainty."""
        # Default estimate: forces are typically noisier
        return 0.05  # eV/Å
    
    def _compute_distance_factor(self, positions: np.ndarray) -> float:
        """Compute distance factor to training structures."""
        if not self.training_structures:
            return 1.0  # Maximum uncertainty if no training data
        
        # Compute minimum distance to any training structure
        min_distances = []
        for train_pos in self.training_structures[-100:]:  # Last 100 structures
            if train_pos.shape == positions.shape:
                # Use root-mean-square deviation
                rmsd = np.sqrt(np.mean((positions - train_pos)**2))
                min_distances.append(rmsd)
        
        if not min_distances:
            return 1.0
        
        min_dist = min(min_distances)
        # Exponential decay of uncertainty with distance
        return 1.0 - np.exp(-min_dist / 0.5)  # 0.5 Å characteristic length
    
    def update(self,
               positions: np.ndarray,
               atomic_numbers: np.ndarray,
               dft_energy: float,
               dft_forces: np.ndarray) -> None:
        """Update with new DFT reference."""
        self.training_structures.append(positions.copy())
        self.training_energies.append(dft_energy)
        
        # Keep only recent structures
        if len(self.training_structures) > 1000:
            self.training_structures = self.training_structures[-1000:]
            self.training_energies = self.training_energies[-1000:]
        
        # Calibrate uncertainty estimates
        self._calibrate(positions, atomic_numbers, dft_energy, dft_forces)
    
    def _calibrate(self,
                   positions: np.ndarray,
                   atomic_numbers: np.ndarray,
                   dft_energy: float,
                   dft_forces: np.ndarray) -> None:
        """Calibrate uncertainty estimates based on DFT comparison."""
        energies, forces = self._get_ensemble_predictions(positions, atomic_numbers)
        
        mean_energy = np.mean(energies)
        mean_forces = np.mean(forces, axis=0)
        
        # Update bias and scale
        energy_error = abs(mean_energy - dft_energy)
        energy_unc = np.std(energies)
        
        if energy_unc > 0:
            # Adjust scale to match observed error
            target_scale = energy_error / energy_unc
            self.energy_scale = 0.9 * self.energy_scale + 0.1 * target_scale
        
        # Force calibration
        force_errors = np.linalg.norm(mean_forces - dft_forces, axis=1)
        mean_force_error = np.mean(force_errors)
        
        self.prediction_history.append({
            'energy_error': energy_error,
            'energy_unc': energy_unc,
            'force_error': mean_force_error
        })
    
    def select_high_uncertainty_atoms(self,
                                     positions: np.ndarray,
                                     n_atoms: int = 10) -> List[int]:
        """Select atoms with highest uncertainty for DFT evaluation."""
        # Estimate per-atom uncertainties
        uncertainty = self.estimate(positions, np.ones(len(positions)))
        
        if uncertainty.per_atom_forces is not None:
            atom_uncertainties = uncertainty.per_atom_forces
        else:
            # Fallback: use position-based heuristic
            atom_uncertainties = np.random.rand(len(positions))
        
        # Select top n_atoms with highest uncertainty
        top_indices = np.argsort(atom_uncertainties)[-n_atoms:]
        return top_indices.tolist()


class GradientSensitivityEstimator(BaseErrorEstimator):
    """
    Error estimator based on sensitivity to input perturbations.
    
    Uses finite differences to assess how rapidly predictions change
    with respect to atomic positions.
    """
    
    def __init__(self,
                 model: Any,
                 perturbation_scale: float = 0.01,
                 n_perturbations: int = 10):
        """
        Initialize gradient sensitivity estimator.
        
        Args:
            model: ML model for predictions
            perturbation_scale: Magnitude of position perturbations (Å)
            n_perturbations: Number of perturbation samples
        """
        self.model = model
        self.perturbation_scale = perturbation_scale
        self.n_perturbations = n_perturbations
        
        self.sensitivity_history: deque = deque(maxlen=100)
        
    def estimate(self,
                 positions: np.ndarray,
                 atomic_numbers: np.ndarray,
                 **kwargs) -> UncertaintyEstimate:
        """Estimate uncertainty using gradient sensitivity."""
        # Get reference prediction
        ref_energy, ref_forces = self.model.predict(positions, atomic_numbers)
        
        n_atoms = len(atomic_numbers)
        
        # Generate perturbed structures
        energy_changes = []
        force_changes = []
        
        for _ in range(self.n_perturbations):
            # Random perturbation
            perturbation = np.random.randn(*positions.shape) * self.perturbation_scale
            perturbed_pos = positions + perturbation
            
            # Predict on perturbed structure
            try:
                pert_energy, pert_forces = self.model.predict(perturbed_pos, atomic_numbers)
                
                energy_changes.append(abs(pert_energy - ref_energy))
                force_changes.append(np.linalg.norm(pert_forces - ref_forces, axis=1))
            except Exception as e:
                logger.warning(f"Perturbation prediction failed: {e}")
                continue
        
        if not energy_changes:
            # All perturbations failed
            return UncertaintyEstimate(
                total_uncertainty=1.0,
                aleatoric_uncertainty=0.0,
                epistemic_uncertainty=1.0,
                confidence_score=0.0
            )
        
        # Compute sensitivity metrics
        mean_energy_sensitivity = np.mean(energy_changes) / self.perturbation_scale
        mean_force_sensitivity = np.mean(force_changes) / self.perturbation_scale
        
        # Per-atom sensitivity
        per_atom_sensitivity = np.mean(force_changes, axis=0) / self.perturbation_scale
        
        # Normalize by reference values
        normalized_energy_sens = mean_energy_sensitivity / (abs(ref_energy) / n_atoms + 0.1)
        normalized_force_sens = mean_force_sensitivity / (np.mean(np.abs(ref_forces)) + 0.1)
        
        # Combine into total uncertainty
        total_unc = 0.3 * normalized_energy_sens + 0.7 * normalized_force_sens
        
        # Confidence based on sensitivity
        confidence = 1.0 / (1.0 + total_unc)
        
        self.sensitivity_history.append(total_unc)
        
        return UncertaintyEstimate(
            total_uncertainty=float(total_unc),
            aleatoric_uncertainty=0.0,
            epistemic_uncertainty=float(total_unc),
            per_atom_forces=per_atom_sensitivity,
            gradient_sensitivity=float(total_unc),
            confidence_score=float(confidence),
            reliability_diagnostic={
                'energy_sensitivity': float(normalized_energy_sens),
                'force_sensitivity': float(normalized_force_sens),
                'perturbation_samples': len(energy_changes)
            }
        )
    
    def update(self, *args, **kwargs) -> None:
        """Gradient estimator doesn't require updates."""
        pass


class BayesianNNEstimator(BaseErrorEstimator):
    """
    Bayesian Neural Network-based uncertainty estimator.
    
    Uses variational inference or MC dropout for uncertainty quantification.
    """
    
    def __init__(self,
                 model: Any,
                 n_mcmc_samples: int = 100,
                 dropout_rate: float = 0.1):
        """
        Initialize Bayesian NN estimator.
        
        Args:
            model: Bayesian neural network model
            n_mcmc_samples: Number of MCMC/dropout samples
            dropout_rate: Dropout rate for MC dropout
        """
        self.model = model
        self.n_mcmc_samples = n_mcmc_samples
        self.dropout_rate = dropout_rate
        
        self.posterior_samples: List[Dict] = []
        
    def estimate(self,
                 positions: np.ndarray,
                 atomic_numbers: np.ndarray,
                 **kwargs) -> UncertaintyEstimate:
        """Estimate uncertainty using Bayesian inference."""
        # Collect samples with dropout
        energies = []
        forces = []
        
        for _ in range(self.n_mcmc_samples):
            # Enable dropout at test time (MC dropout)
            energy, force = self.model.predict_with_dropout(
                positions, atomic_numbers, dropout_rate=self.dropout_rate
            )
            energies.append(energy)
            forces.append(force)
        
        energies = np.array(energies)
        forces = np.array(forces)
        
        # Compute posterior statistics
        mean_energy = np.mean(energies)
        var_energy = np.var(energies)
        
        mean_forces = np.mean(forces, axis=0)
        var_forces = np.var(forces, axis=0)
        
        # Estimate aleatoric vs epistemic
        # Aleatoric: inherent noise (use low-quantile variance)
        aleatoric_energy = np.percentile(np.abs(energies - mean_energy), 25)**2
        epistemic_energy = max(0, var_energy - aleatoric_energy)
        
        # Total uncertainty
        total_unc = np.sqrt(var_energy) / (abs(mean_energy) + 1.0)
        
        # Confidence score
        confidence = np.exp(-total_unc)
        
        return UncertaintyEstimate(
            total_uncertainty=float(total_unc),
            aleatoric_uncertainty=float(np.sqrt(aleatoric_energy)),
            epistemic_uncertainty=float(np.sqrt(epistemic_energy)),
            per_atom_forces=np.sqrt(np.sum(var_forces, axis=1)),
            confidence_score=float(confidence),
            reliability_diagnostic={
                'posterior_samples': self.n_mcmc_samples,
                'effective_sample_size': self._compute_ess(energies)
            }
        )
    
    def _compute_ess(self, samples: np.ndarray) -> float:
        """Compute effective sample size from MCMC samples."""
        # Simple ESS estimate using autocorrelation
        if len(samples) < 2:
            return float(len(samples))
        
        # Compute autocorrelation at lag 1
        autocorr = np.corrcoef(samples[:-1], samples[1:])[0, 1]
        
        if autocorr >= 1.0:
            return 1.0
        
        ess = len(samples) / (1 + 2 * autocorr / (1 - autocorr))
        return float(ess)
    
    def update(self, *args, **kwargs) -> None:
        """Bayesian estimator updates through model training."""
        pass


class AdaptiveSamplingTrigger:
    """
    Intelligent trigger for adaptive sampling decisions.
    
    Determines when to perform DFT calculations based on
    uncertainty estimates and sampling strategy.
    """
    
    def __init__(self,
                 error_estimator: BaseErrorEstimator,
                 uncertainty_threshold: float = 0.1,
                 min_steps_between_dft: int = 10,
                 max_dft_per_100_steps: int = 20,
                 exploration_factor: float = 0.2):
        """
        Initialize adaptive sampling trigger.
        
        Args:
            error_estimator: Error estimator instance
            uncertainty_threshold: Uncertainty threshold for DFT trigger
            min_steps_between_dft: Minimum steps between DFT calculations
            max_dft_per_100_steps: Maximum DFT calls per 100 steps
            exploration_factor: Fraction of DFT for exploration vs exploitation
        """
        self.error_estimator = error_estimator
        self.uncertainty_threshold = uncertainty_threshold
        self.min_steps = min_steps_between_dft
        self.max_dft_rate = max_dft_per_100_steps / 100.0
        self.exploration_factor = exploration_factor
        
        self.step_count = 0
        self.last_dft_step = -self.min_steps
        self.dft_count_window: deque = deque(maxlen=100)
        
        # Trigger history
        self.trigger_history: List[Dict] = []
        
    def should_trigger_dft(self,
                          positions: np.ndarray,
                          atomic_numbers: np.ndarray,
                          step_type: str = 'dynamics') -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if DFT calculation should be triggered.
        
        Returns:
            Tuple of (should_trigger, trigger_info)
        """
        self.step_count += 1
        
        trigger_info = {
            'triggered': False,
            'reason': None,
            'uncertainty': None,
            'priority': 'normal'
        }
        
        # Check minimum steps
        if self.step_count - self.last_dft_step < self.min_steps:
            return False, trigger_info
        
        # Check rate limit
        recent_dft_rate = sum(self.dft_count_window) / max(len(self.dft_count_window), 1)
        if recent_dft_rate >= self.max_dft_rate:
            return False, trigger_info
        
        # Get uncertainty estimate
        uncertainty = self.error_estimator.estimate(positions, atomic_numbers)
        trigger_info['uncertainty'] = uncertainty
        
        # Check uncertainty threshold
        if uncertainty.total_uncertainty > self.uncertainty_threshold:
            trigger_info['triggered'] = True
            trigger_info['reason'] = 'high_uncertainty'
            trigger_info['priority'] = 'high' if uncertainty.total_uncertainty > \
                self.uncertainty_threshold * 2 else 'normal'
            
            self._record_trigger(trigger_info)
            return True, trigger_info
        
        # Exploration: occasional random DFT for diverse sampling
        if np.random.random() < self.exploration_factor * self.max_dft_rate:
            trigger_info['triggered'] = True
            trigger_info['reason'] = 'exploration'
            trigger_info['priority'] = 'low'
            
            self._record_trigger(trigger_info)
            return True, trigger_info
        
        return False, trigger_info
    
    def _record_trigger(self, trigger_info: Dict) -> None:
        """Record trigger event."""
        self.last_dft_step = self.step_count
        self.dft_count_window.append(1)
        
        self.trigger_history.append({
            'step': self.step_count,
            'timestamp': trigger_info,
            'uncertainty': trigger_info['uncertainty'].total_uncertainty if \
                          trigger_info['uncertainty'] else None
        })
    
    def update_with_dft_result(self,
                              positions: np.ndarray,
                              atomic_numbers: np.ndarray,
                              dft_energy: float,
                              dft_forces: np.ndarray) -> None:
        """Update error estimator with new DFT result."""
        self.error_estimator.update(positions, atomic_numbers, dft_energy, dft_forces)
    
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get statistics about sampling decisions."""
        if not self.trigger_history:
            return {'total_triggers': 0}
        
        reasons = [t['timestamp']['reason'] for t in self.trigger_history]
        
        return {
            'total_triggers': len(self.trigger_history),
            'trigger_reasons': {r: reasons.count(r) for r in set(reasons)},
            'average_uncertainty': np.mean([t['uncertainty'] for t in self.trigger_history 
                                           if t['uncertainty'] is not None]),
            'dft_rate': sum(self.dft_count_window) / max(len(self.dft_count_window), 1),
            'steps_since_last_dft': self.step_count - self.last_dft_step
        }


class CompositeErrorEstimator(BaseErrorEstimator):
    """
    Combines multiple error estimators for robust uncertainty quantification.
    """
    
    def __init__(self,
                 estimators: List[BaseErrorEstimator],
                 weights: Optional[List[float]] = None,
                 aggregation_method: str = 'weighted_average'):
        """
        Initialize composite estimator.
        
        Args:
            estimators: List of error estimators
            weights: Weights for each estimator (normalized)
            aggregation_method: How to combine estimates
        """
        self.estimators = estimators
        self.n_estimators = len(estimators)
        
        if weights is None:
            weights = [1.0 / self.n_estimators] * self.n_estimators
        
        self.weights = np.array(weights) / sum(weights)
        self.aggregation_method = aggregation_method
        
        # Performance tracking for adaptive weighting
        self.performance_history: Dict[int, deque] = {
            i: deque(maxlen=50) for i in range(self.n_estimators)
        }
        
    def estimate(self,
                 positions: np.ndarray,
                 atomic_numbers: np.ndarray,
                 **kwargs) -> UncertaintyEstimate:
        """Combine estimates from all estimators."""
        estimates = []
        
        for estimator in self.estimators:
            try:
                est = estimator.estimate(positions, atomic_numbers, **kwargs)
                estimates.append(est)
            except Exception as e:
                logger.warning(f"Estimator failed: {e}")
                continue
        
        if not estimates:
            # All failed - return maximum uncertainty
            return UncertaintyEstimate(
                total_uncertainty=1.0,
                aleatoric_uncertainty=0.5,
                epistemic_uncertainty=0.5,
                confidence_score=0.0,
                reliability_diagnostic={'all_estimators_failed': True}
            )
        
        # Combine estimates
        if self.aggregation_method == 'weighted_average':
            return self._weighted_average(estimates)
        elif self.aggregation_method == 'conservative':
            return self._conservative_combine(estimates)
        elif self.aggregation_method == 'voting':
            return self._voting_combine(estimates)
        else:
            return self._weighted_average(estimates)
    
    def _weighted_average(self, estimates: List[UncertaintyEstimate]) -> UncertaintyEstimate:
        """Weighted average of estimates."""
        n = len(estimates)
        weights = self.weights[:n] / sum(self.weights[:n])
        
        total_unc = sum(w * e.total_uncertainty for w, e in zip(weights, estimates))
        aleatoric = sum(w * e.aleatoric_uncertainty for w, e in zip(weights, estimates))
        epistemic = sum(w * e.epistemic_uncertainty for w, e in zip(weights, estimates))
        confidence = sum(w * e.confidence_score for w, e in zip(weights, estimates))
        
        return UncertaintyEstimate(
            total_uncertainty=float(total_unc),
            aleatoric_uncertainty=float(aleatoric),
            epistemic_uncertainty=float(epistemic),
            confidence_score=float(confidence),
            reliability_diagnostic={'n_estimators': n, 'method': 'weighted_average'}
        )
    
    def _conservative_combine(self, estimates: List[UncertaintyEstimate]) -> UncertaintyEstimate:
        """Take maximum uncertainty (conservative approach)."""
        return UncertaintyEstimate(
            total_uncertainty=max(e.total_uncertainty for e in estimates),
            aleatoric_uncertainty=max(e.aleatoric_uncertainty for e in estimates),
            epistemic_uncertainty=max(e.epistemic_uncertainty for e in estimates),
            confidence_score=min(e.confidence_score for e in estimates),
            reliability_diagnostic={'n_estimators': len(estimates), 'method': 'conservative'}
        )
    
    def _voting_combine(self, estimates: List[UncertaintyEstimate]) -> UncertaintyEstimate:
        """Majority voting on reliability."""
        reliable_count = sum(1 for e in estimates if e.is_reliable())
        
        # Weight by reliability consensus
        if reliable_count >= len(estimates) / 2:
            return self._weighted_average(estimates)
        else:
            # Conservative if majority unreliable
            return self._conservative_combine(estimates)
    
    def update(self, *args, **kwargs) -> None:
        """Update all component estimators."""
        for estimator in self.estimators:
            try:
                estimator.update(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Component update failed: {e}")
    
    def adapt_weights(self, ground_truth_errors: List[float]) -> None:
        """Adapt weights based on recent performance."""
        if len(ground_truth_errors) != self.n_estimators:
            return
        
        # Lower error -> higher weight
        inverse_errors = [1.0 / (e + 1e-6) for e in ground_truth_errors]
        self.weights = np.array(inverse_errors) / sum(inverse_errors)


def create_default_estimator(model: Any = None,
                             ensemble: Optional[List[Any]] = None) -> BaseErrorEstimator:
    """
    Create a default error estimator configuration.
    
    Args:
        model: Primary ML model
        ensemble: Ensemble of models (if available)
        
    Returns:
        Configured error estimator
    """
    estimators = []
    
    # Add ensemble estimator if available
    if ensemble and len(ensemble) >= 2:
        estimators.append(EnsembleErrorEstimator(ensemble))
    
    # Add gradient sensitivity estimator
    if model is not None:
        estimators.append(GradientSensitivityEstimator(model))
    
    if len(estimators) > 1:
        return CompositeErrorEstimator(estimators)
    elif estimators:
        return estimators[0]
    else:
        # Return dummy estimator
        return EnsembleErrorEstimator([])


# Example usage
if __name__ == "__main__":
    print("=== Error Estimators Module Demo ===\n")
    
    # Create a mock model
    class MockModel:
        def predict(self, positions, atomic_numbers):
            n = len(positions)
            energy = np.random.randn() * 0.1 * n
            forces = np.random.randn(n, 3) * 0.05
            return energy, forces
        
        def predict_with_dropout(self, positions, atomic_numbers, dropout_rate=0.1):
            return self.predict(positions, atomic_numbers)
    
    # Create mock ensemble
    mock_ensemble = [MockModel() for _ in range(5)]
    
    # Create ensemble estimator
    estimator = EnsembleErrorEstimator(mock_ensemble)
    
    # Test estimation
    positions = np.random.randn(50, 3) * 5
    atomic_numbers = np.ones(50, dtype=int)
    
    uncertainty = estimator.estimate(positions, atomic_numbers)
    
    print(f"Total uncertainty: {uncertainty.total_uncertainty:.4f}")
    print(f"Confidence score: {uncertainty.confidence_score:.4f}")
    print(f"Epistemic uncertainty: {uncertainty.epistemic_uncertainty:.4f}")
    print(f"Reliable: {uncertainty.is_reliable()}")
    print(f"Requires DFT: {uncertainty.requires_dft()}")
    
    # Test adaptive sampling trigger
    print("\n--- Adaptive Sampling Trigger ---")
    trigger = AdaptiveSamplingTrigger(estimator)
    
    for i in range(15):
        should_trigger, info = trigger.should_trigger_dft(positions, atomic_numbers)
        if should_trigger:
            print(f"Step {i+1}: DFT triggered ({info['reason']})")
            # Simulate DFT update
            estimator.update(positions, atomic_numbers, 
                           np.random.randn() * 10, 
                           np.random.randn(50, 3) * 0.1)
        else:
            print(f"Step {i+1}: No trigger (unc={info['uncertainty'].total_uncertainty:.4f})")
    
    print("\n--- Sampling Statistics ---")
    stats = trigger.get_sampling_statistics()
    print(f"Total triggers: {stats['total_triggers']}")
    print(f"Trigger reasons: {stats.get('trigger_reasons', {})}")
