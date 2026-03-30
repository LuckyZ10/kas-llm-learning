"""
Federated Materials Discovery Module
=====================================

This module implements privacy-preserving federated learning for 
materials discovery across multiple institutions.

Features:
- Distributed high-throughput screening
- Privacy-preserving data sharing
- Cross-institutional model collaboration
- Federated Bayesian optimization
- Secure multi-party computation for property prediction

Author: DFT-LAMMPS Team
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Callable, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import copy
import json
import logging
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import secrets


logger = logging.getLogger(__name__)


class DiscoveryStrategy(Enum):
    """Strategies for federated materials discovery."""
    FEDERATED_BO = "federated_bo"  # Federated Bayesian Optimization
    FEDERATED_EA = "federated_ea"  # Federated Evolutionary Algorithm
    FEDERATED_RL = "federated_rl"  # Federated Reinforcement Learning
    SECURE_MT = "secure_mt"  # Secure Multi-Task Learning


@dataclass
class MaterialCandidate:
    """
    Represents a candidate material structure.
    
    Attributes:
        composition: Chemical composition (e.g., "LiFePO4")
        structure: Crystal structure information
        features: Feature vector representation
        predicted_properties: Dictionary of predicted properties
        uncertainty: Prediction uncertainty
        source_institution: Institution that proposed this candidate
        privacy_level: Privacy sensitivity level
    """
    composition: str
    structure: Optional[Dict] = None
    features: Optional[np.ndarray] = None
    predicted_properties: Dict[str, float] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    source_institution: str = ""
    privacy_level: str = "public"  # public, internal, confidential
    candidate_id: str = ""
    acquisition_score: float = 0.0
    
    def __post_init__(self):
        if not self.candidate_id:
            self.candidate_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique candidate ID."""
        data = f"{self.composition}_{self.source_institution}_{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'candidate_id': self.candidate_id,
            'composition': self.composition,
            'structure': self.structure,
            'features': self.features.tolist() if self.features is not None else None,
            'predicted_properties': self.predicted_properties,
            'uncertainty': self.uncertainty,
            'source_institution': self.source_institution,
            'privacy_level': self.privacy_level,
            'acquisition_score': self.acquisition_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MaterialCandidate':
        """Create from dictionary."""
        candidate = cls(
            composition=data['composition'],
            structure=data.get('structure'),
            features=np.array(data['features']) if data.get('features') else None,
            source_institution=data.get('source_institution', ''),
            privacy_level=data.get('privacy_level', 'public')
        )
        candidate.candidate_id = data.get('candidate_id', candidate._generate_id())
        candidate.predicted_properties = data.get('predicted_properties', {})
        candidate.uncertainty = data.get('uncertainty', {})
        candidate.acquisition_score = data.get('acquisition_score', 0.0)
        return candidate


@dataclass
class FederatedDiscoveryConfig:
    """Configuration for federated materials discovery."""
    
    # Discovery parameters
    num_candidates: int = 1000
    num_iterations: int = 50
    batch_size: int = 10
    
    # Strategy
    strategy: DiscoveryStrategy = DiscoveryStrategy.FEDERATED_BO
    
    # Bayesian optimization parameters
    exploration_factor: float = 0.1
    acquisition_function: str = "ei"  # ei, ucb, pi
    
    # Privacy parameters
    use_dp: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    
    # Secure computation
    use_mpc: bool = True
    num_parties: int = 3
    
    # Collaboration
    share_candidates: bool = True
    candidate_anonymization: bool = True
    min_institutions_per_candidate: int = 2
    
    # Screening
    target_properties: List[str] = field(default_factory=lambda: ['band_gap', 'formation_energy'])
    property_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Evaluation
    num_evaluations_per_candidate: int = 3


class SecureGaussianProcess:
    """
    Privacy-preserving Gaussian Process for materials property prediction.
    
    Implements secure multi-party computation for GP regression where
    training data remains private to each institution.
    """
    
    def __init__(self, kernel: str = "rbf", noise_level: float = 1e-5):
        self.kernel_type = kernel
        self.noise_level = noise_level
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        
    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix.
        
        Args:
            x1: First set of points (n1, d)
            x2: Second set of points (n2, d)
            
        Returns:
            Kernel matrix (n1, n2)
        """
        if self.kernel_type == "rbf":
            # RBF kernel
            dist_sq = (
                np.sum(x1**2, axis=1).reshape(-1, 1) +
                np.sum(x2**2, axis=1) -
                2 * np.dot(x1, x2.T)
            )
            return np.exp(-0.5 * dist_sq)
        elif self.kernel_type == "matern52":
            # Matérn 5/2 kernel
            dist = np.sqrt(
                np.sum(x1**2, axis=1).reshape(-1, 1) +
                np.sum(x2**2, axis=1) -
                2 * np.dot(x1, x2.T) + 1e-10
            )
            sqrt5_dist = np.sqrt(5) * dist
            return (1 + sqrt5_dist + 5 * dist**2 / 3) * np.exp(-sqrt5_dist)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Gaussian Process.
        
        Args:
            X: Training features (n, d)
            y: Training targets (n,)
        """
        self.X_train = X
        self.y_train = y
        
        # Compute kernel matrix
        K = self.kernel(X, X)
        K += self.noise_level * np.eye(len(X))
        
        # Cholesky decomposition for stability
        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.K_inv = np.linalg.pinv(K)
            self.alpha = self.K_inv @ y
    
    def predict(self, X_test: np.ndarray, return_std: bool = True
               ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.
        
        Args:
            X_test: Test features (m, d)
            return_std: Whether to return standard deviation
            
        Returns:
            Predictions and optionally standard deviations
        """
        if self.X_train is None:
            raise ValueError("Model not fitted yet")
        
        # Compute kernel between test and train
        K_s = self.kernel(X_test, self.X_train)
        
        # Mean prediction
        mu = K_s @ self.alpha
        
        if not return_std:
            return mu
        
        # Compute variance
        K_ss = self.kernel(X_test, X_test)
        v = np.linalg.solve(
            np.linalg.cholesky(self.kernel(self.X_train, self.X_train) + 
                             self.noise_level * np.eye(len(self.X_train))),
            K_s.T
        )
        var = np.diag(K_ss) - np.sum(v**2, axis=0)
        std = np.sqrt(np.maximum(var, 0))
        
        return mu, std
    
    def secure_predict(self, X_test: np.ndarray, 
                      institutions: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Secure multi-party prediction where each institution contributes
        to the prediction without revealing their data.
        
        Args:
            X_test: Test features
            institutions: List of participating institutions
            
        Returns:
            Secure aggregated prediction and uncertainty
        """
        # Simulate secure aggregation
        # In real implementation, use MPC protocol
        predictions = []
        uncertainties = []
        
        for _ in institutions:
            pred, unc = self.predict(X_test, return_std=True)
            # Add differential privacy noise
            pred_noisy = pred + np.random.normal(0, 0.01, size=pred.shape)
            unc_noisy = unc + np.random.normal(0, 0.005, size=unc.shape)
            predictions.append(pred_noisy)
            uncertainties.append(unc_noisy)
        
        # Aggregate predictions
        mean_pred = np.mean(predictions, axis=0)
        mean_unc = np.mean(uncertainties, axis=0)
        
        return mean_pred, mean_unc


class PrivacyPreservingSampler:
    """
    Privacy-preserving sampling for candidate generation.
    
    Ensures that sensitive material compositions are not leaked while
    still enabling effective collaborative discovery.
    """
    
    def __init__(self, epsilon: float = 1.0, mechanism: str = "exponential"):
        self.epsilon = epsilon
        self.mechanism = mechanism
        self.sensitivity = 1.0
        
    def exponential_mechanism(self, scores: np.ndarray, 
                              candidates: List[MaterialCandidate]) -> MaterialCandidate:
        """
        Exponential mechanism for differentially private selection.
        
        Args:
            scores: Utility scores for each candidate
            candidates: List of candidate materials
            
        Returns:
            Selected candidate with DP guarantee
        """
        # Compute probabilities
        exp_scores = np.exp(self.epsilon * scores / (2 * self.sensitivity))
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Sample
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected_idx]
    
    def laplace_mechanism(self, value: float, sensitivity: float = None) -> float:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            value: Original value
            sensitivity: Sensitivity of the query
            
        Returns:
            Noisy value
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        
        noise = np.random.laplace(0, sensitivity / self.epsilon)
        return value + noise
    
    def anonymize_candidate(self, candidate: MaterialCandidate,
                           k: int = 5) -> MaterialCandidate:
        """
        k-anonymize candidate by generalizing composition.
        
        Args:
            candidate: Original candidate
            k: Anonymity parameter
            
        Returns:
            Anonymized candidate
        """
        # Create anonymized version
        anon_candidate = copy.deepcopy(candidate)
        
        # Generalize composition (simplified - real implementation would use
        # proper k-anonymity algorithms)
        composition = candidate.composition
        
        # Group elements into classes
        element_classes = {
            'alkali': ['Li', 'Na', 'K', 'Rb', 'Cs'],
            'alkaline': ['Be', 'Mg', 'Ca', 'Sr', 'Ba'],
            'transition': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 
                          'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                          'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re',
                          'Os', 'Ir', 'Pt', 'Au', 'Hg'],
            'chalcogen': ['O', 'S', 'Se', 'Te'],
            'pnictogen': ['N', 'P', 'As', 'Sb', 'Bi'],
            'halogen': ['F', 'Cl', 'Br', 'I'],
        }
        
        # Replace specific elements with their class
        for element_class, elements in element_classes.items():
            for element in elements:
                if element in composition:
                    composition = composition.replace(element, f"[{element_class}]")
        
        anon_candidate.composition = composition
        anon_candidate.privacy_level = "anonymized"
        
        return anon_candidate
    
    def generate_synthetic_candidates(self, n: int, 
                                      feature_dim: int = 100) -> List[MaterialCandidate]:
        """
        Generate synthetic candidates for privacy protection.
        
        Args:
            n: Number of synthetic candidates
            feature_dim: Feature dimension
            
        Returns:
            List of synthetic candidates
        """
        synthetic = []
        
        for i in range(n):
            features = np.random.randn(feature_dim)
            
            candidate = MaterialCandidate(
                composition=f"Synthetic_{i}",
                features=features,
                source_institution="synthetic",
                privacy_level="synthetic"
            )
            synthetic.append(candidate)
        
        return synthetic


class FederatedDiscoveryCoordinator:
    """
    Central coordinator for federated materials discovery.
    
    Manages the discovery process across multiple institutions while
    preserving privacy and enabling secure collaboration.
    """
    
    def __init__(self, config: FederatedDiscoveryConfig):
        self.config = config
        self.institutions: Dict[str, 'DiscoveryClient'] = {}
        self.candidate_pool: List[MaterialCandidate] = []
        self.evaluated_candidates: List[MaterialCandidate] = []
        self.iteration = 0
        
        # Gaussian Process for each property
        self.gp_models: Dict[str, SecureGaussianProcess] = {}
        
        # Privacy mechanisms
        self.privacy_sampler = PrivacyPreservingSampler(
            epsilon=config.epsilon
        )
        
        # History
        self.history = {
            'iterations': [],
            'best_scores': [],
            'discovered_materials': [],
            'privacy_budget_spent': []
        }
        
    def register_institution(self, institution_id: str, 
                            client: 'DiscoveryClient') -> None:
        """
        Register a participating institution.
        
        Args:
            institution_id: Unique institution identifier
            client: Discovery client instance
        """
        self.institutions[institution_id] = client
        logger.info(f"Registered institution: {institution_id}")
        
    def initialize_candidate_pool(self, 
                                  initial_candidates: List[MaterialCandidate] = None):
        """
        Initialize pool of candidate materials.
        
        Args:
            initial_candidates: Optional initial candidates
        """
        if initial_candidates:
            self.candidate_pool = initial_candidates
        else:
            # Generate initial candidates
            self.candidate_pool = self._generate_initial_candidates()
        
        logger.info(f"Initialized candidate pool with {len(self.candidate_pool)} candidates")
    
    def _generate_initial_candidates(self) -> List[MaterialCandidate]:
        """Generate initial candidate materials."""
        candidates = []
        
        # Common material compositions for demonstration
        compositions = [
            "LiFePO4", "LiCoO2", "LiMn2O4", "LiNiMnCoO2",
            "NaFePO4", "Na3V2(PO4)3", "NaNiO2",
            "MgSiO3", "CaTiO3", "BaTiO3", "SrTiO3",
            "ZnO", "TiO2", "SiO2", "Al2O3",
            "CuInGaSe2", "CdTe", "GaAs", "InP",
            "NiFe", "CoFe", "FeCr", "TiAl",
            "MoS2", "WS2", "BN", "Gr", "C3N4"
        ]
        
        for i, comp in enumerate(compositions):
            features = np.random.randn(100)  # Placeholder features
            
            candidate = MaterialCandidate(
                composition=comp,
                features=features,
                source_institution=np.random.choice(list(self.institutions.keys()) or ["default"]),
                privacy_level=np.random.choice(["public", "internal"])
            )
            candidates.append(candidate)
        
        # Add synthetic candidates for privacy
        if self.config.use_dp:
            synthetic = self.privacy_sampler.generate_synthetic_candidates(50)
            candidates.extend(synthetic)
        
        return candidates
    
    def federated_screening(self) -> List[MaterialCandidate]:
        """
        Perform privacy-preserving federated screening of candidates.
        
        Each institution evaluates candidates locally and contributes
        to the global ranking without revealing their evaluation criteria.
        
        Returns:
            Ranked list of candidates
        """
        logger.info("Starting federated screening...")
        
        # Collect scores from each institution
        institution_scores = defaultdict(dict)
        
        for inst_id, client in self.institutions.items():
            for candidate in self.candidate_pool:
                # Local evaluation
                score = client.evaluate_candidate(candidate)
                
                # Add differential privacy noise
                if self.config.use_dp:
                    score = self.privacy_sampler.laplace_mechanism(score)
                
                institution_scores[inst_id][candidate.candidate_id] = score
        
        # Aggregate scores securely
        aggregated_scores = {}
        for candidate in self.candidate_pool:
            scores = [
                institution_scores[inst_id].get(candidate.candidate_id, 0)
                for inst_id in self.institutions
            ]
            
            # Secure aggregation (simplified)
            aggregated_scores[candidate.candidate_id] = np.mean(scores)
        
        # Rank candidates
        for candidate in self.candidate_pool:
            candidate.acquisition_score = aggregated_scores.get(
                candidate.candidate_id, 0
            )
        
        ranked = sorted(self.candidate_pool, 
                       key=lambda x: x.acquisition_score, 
                       reverse=True)
        
        logger.info(f"Federated screening completed. Top score: {ranked[0].acquisition_score:.4f}")
        
        return ranked
    
    def secure_property_prediction(self, 
                                   candidate: MaterialCandidate) -> Dict[str, float]:
        """
        Predict material properties using secure multi-party computation.
        
        Args:
            candidate: Material candidate to evaluate
            
        Returns:
            Dictionary of predicted properties
        """
        predictions = {}
        
        for prop in self.config.target_properties:
            # Initialize GP if not exists
            if prop not in self.gp_models:
                self.gp_models[prop] = SecureGaussianProcess(kernel="matern52")
            
            # Get features
            if candidate.features is not None:
                X = candidate.features.reshape(1, -1)
                
                # Secure prediction across institutions
                pred, unc = self.gp_models[prop].secure_predict(
                    X,
                    list(self.institutions.keys())
                )
                
                predictions[prop] = float(pred[0])
                candidate.uncertainty[prop] = float(unc[0])
            else:
                # Random prediction for demonstration
                predictions[prop] = np.random.uniform(-5, 5)
                candidate.uncertainty[prop] = np.random.uniform(0.1, 1.0)
        
        return predictions
    
    def federated_bayesian_optimization(self) -> MaterialCandidate:
        """
        Perform federated Bayesian optimization for candidate selection.
        
        Uses secure multi-party computation to compute acquisition functions
        without revealing institution-specific models.
        
        Returns:
            Selected candidate
        """
        logger.info(f"Starting Bayesian optimization iteration {self.iteration}")
        
        # Compute acquisition scores
        acquisition_scores = []
        
        for candidate in self.candidate_pool:
            # Predict properties
            predictions = self.secure_property_prediction(candidate)
            candidate.predicted_properties = predictions
            
            # Compute acquisition function
            score = self._compute_acquisition(candidate)
            acquisition_scores.append(score)
        
        acquisition_scores = np.array(acquisition_scores)
        
        # Privacy-preserving selection
        if self.config.use_dp:
            selected = self.privacy_sampler.exponential_mechanism(
                acquisition_scores,
                self.candidate_pool
            )
        else:
            selected = self.candidate_pool[np.argmax(acquisition_scores)]
        
        logger.info(f"Selected candidate: {selected.composition}")
        
        return selected
    
    def _compute_acquisition(self, candidate: MaterialCandidate) -> float:
        """
        Compute acquisition function value.
        
        Args:
            candidate: Material candidate
            
        Returns:
            Acquisition score
        """
        # Multi-objective acquisition
        scores = []
        
        for prop in self.config.target_properties:
            if prop not in candidate.predicted_properties:
                continue
            
            mu = candidate.predicted_properties[prop]
            sigma = candidate.uncertainty.get(prop, 1.0)
            
            if self.config.acquisition_function == "ei":
                # Expected Improvement
                if not self.evaluated_candidates:
                    score = mu + self.config.exploration_factor * sigma
                else:
                    best_so_far = max(
                        c.predicted_properties.get(prop, -np.inf)
                        for c in self.evaluated_candidates
                    )
                    improvement = mu - best_so_far - self.config.exploration_factor
                    z = improvement / (sigma + 1e-10)
                    score = improvement * (0.5 * (1 + np.sign(z))) + sigma * np.exp(-z**2/2) / np.sqrt(2*np.pi)
                    
            elif self.config.acquisition_function == "ucb":
                # Upper Confidence Bound
                score = mu + self.config.exploration_factor * sigma
                
            elif self.config.acquisition_function == "pi":
                # Probability of Improvement
                if not self.evaluated_candidates:
                    score = 0.5
                else:
                    best_so_far = max(
                        c.predicted_properties.get(prop, -np.inf)
                        for c in self.evaluated_candidates
                    )
                    z = (mu - best_so_far - self.config.exploration_factor) / (sigma + 1e-10)
                    score = 0.5 * (1 + np.sign(z) * np.minimum(np.abs(z), 1))
            else:
                score = mu
            
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def update_models(self, evaluated_candidate: MaterialCandidate) -> None:
        """
        Update GP models with newly evaluated candidate.
        
        Args:
            evaluated_candidate: Candidate with true property values
        """
        for prop in self.config.target_properties:
            if prop in evaluated_candidate.predicted_properties:
                # Update GP with new data point
                if prop in self.gp_models:
                    # In practice, this would trigger federated model update
                    pass
        
        self.evaluated_candidates.append(evaluated_candidate)
    
    def run_discovery(self, num_iterations: Optional[int] = None) -> List[MaterialCandidate]:
        """
        Run federated materials discovery process.
        
        Args:
            num_iterations: Number of iterations (defaults to config)
            
        Returns:
            List of discovered materials
        """
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        logger.info("=" * 60)
        logger.info("Starting Federated Materials Discovery")
        logger.info("=" * 60)
        
        # Initialize
        self.initialize_candidate_pool()
        
        discovered = []
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            
            # Federated screening
            ranked_candidates = self.federated_screening()
            
            # Bayesian optimization
            selected = self.federated_bayesian_optimization()
            
            # Simulate evaluation (in practice, this would be DFT/Experiment)
            # Add true property values
            for prop in self.config.target_properties:
                # Simulate ground truth with noise
                true_value = selected.predicted_properties.get(prop, 0)
                true_value += np.random.normal(0, selected.uncertainty.get(prop, 0.1))
                selected.predicted_properties[prop] = true_value
            
            # Update models
            self.update_models(selected)
            
            # Add to discovered
            discovered.append(selected)
            
            # Remove from pool
            self.candidate_pool = [c for c in self.candidate_pool 
                                  if c.candidate_id != selected.candidate_id]
            
            # Generate new candidates
            new_candidates = self._generate_new_candidates(iteration)
            self.candidate_pool.extend(new_candidates)
            
            # Record history
            best_score = max(c.acquisition_score for c in discovered)
            self.history['iterations'].append(iteration)
            self.history['best_scores'].append(best_score)
            self.history['discovered_materials'].append(selected.composition)
            self.history['privacy_budget_spent'].append(
                iteration * self.config.epsilon / num_iterations
            )
            
            logger.info(f"Iteration {iteration}: Discovered {selected.composition}, "
                       f"Score: {selected.acquisition_score:.4f}")
        
        logger.info("=" * 60)
        logger.info("Federated Discovery Completed")
        logger.info(f"Total materials discovered: {len(discovered)}")
        logger.info("=" * 60)
        
        return discovered
    
    def _generate_new_candidates(self, iteration: int) -> List[MaterialCandidate]:
        """Generate new candidate materials based on current knowledge."""
        new_candidates = []
        
        # Generate variations of top candidates
        if self.evaluated_candidates:
            top_candidates = sorted(self.evaluated_candidates, 
                                   key=lambda x: x.acquisition_score,
                                   reverse=True)[:5]
            
            for candidate in top_candidates:
                # Create variations (simplified)
                for i in range(3):
                    features = candidate.features + np.random.randn(100) * 0.1 if candidate.features is not None else np.random.randn(100)
                    
                    new_candidate = MaterialCandidate(
                        composition=f"{candidate.composition}_var{i}",
                        features=features,
                        source_institution="generated",
                        privacy_level="public"
                    )
                    new_candidates.append(new_candidate)
        
        return new_candidates
    
    def export_results(self, filepath: str) -> None:
        """Export discovery results to file."""
        results = {
            'config': {
                'num_iterations': self.config.num_iterations,
                'num_candidates': self.config.num_candidates,
                'strategy': self.config.strategy.value,
                'use_dp': self.config.use_dp,
                'epsilon': self.config.epsilon
            },
            'discovered_materials': [
                c.to_dict() for c in self.evaluated_candidates
            ],
            'history': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")


class DiscoveryClient:
    """
    Client for participating institution in federated discovery.
    
    Each institution maintains their private data and evaluation criteria
    while contributing to the collaborative discovery process.
    """
    
    def __init__(self, institution_id: str, institution_name: str,
                 config: FederatedDiscoveryConfig):
        self.institution_id = institution_id
        self.institution_name = institution_name
        self.config = config
        
        # Local database
        self.local_materials: List[MaterialCandidate] = []
        self.evaluation_history: List[Dict] = []
        
        # Local model
        self.local_surrogate = None
        
        # Privacy budget
        self.privacy_spent = 0.0
        
    def add_local_materials(self, materials: List[MaterialCandidate]) -> None:
        """Add materials to local private database."""
        for mat in materials:
            mat.source_institution = self.institution_id
        self.local_materials.extend(materials)
        
    def evaluate_candidate(self, candidate: MaterialCandidate) -> float:
        """
        Evaluate a candidate material using local criteria.
        
        Args:
            candidate: Material to evaluate
            
        Returns:
            Evaluation score
        """
        # Institution-specific evaluation logic
        score = 0.0
        
        # Property target matching
        for prop, target_range in self.config.property_ranges.items():
            if prop in candidate.predicted_properties:
                value = candidate.predicted_properties[prop]
                lower, upper = target_range
                if lower <= value <= upper:
                    score += 1.0
                else:
                    # Distance from range
                    dist = min(abs(value - lower), abs(value - upper))
                    score += max(0, 1.0 - dist / abs(upper - lower))
        
        # Novelty bonus
        if candidate.source_institution != self.institution_id:
            score += 0.1  # Bonus for cross-institution collaboration
        
        # Composition preference (example: prefer Li-based materials)
        if "Li" in candidate.composition:
            score += 0.2
        
        return score
    
    def propose_candidates(self, n: int = 10) -> List[MaterialCandidate]:
        """
        Propose new candidates from local knowledge.
        
        Args:
            n: Number of candidates to propose
            
        Returns:
            List of proposed candidates
        """
        # Select from local materials with privacy consideration
        if len(self.local_materials) <= n:
            proposed = self.local_materials
        else:
            proposed = np.random.choice(self.local_materials, n, replace=False).tolist()
        
        # Anonymize if required
        if self.config.candidate_anonymization:
            sampler = PrivacyPreservingSampler(epsilon=self.config.epsilon)
            proposed = [sampler.anonymize_candidate(c) for c in proposed]
        
        return proposed
    
    def local_train_surrogate(self) -> None:
        """Train local surrogate model for property prediction."""
        # Placeholder for local model training
        pass
    
    def share_encrypted_update(self) -> Dict:
        """
        Share encrypted model update with coordinator.
        
        Returns:
            Encrypted update dictionary
        """
        # Placeholder for secure sharing
        return {}


class CrossInstitutionalCollaboration:
    """
    Manages cross-institutional collaboration protocols.
    
    Implements secure protocols for sharing insights while maintaining
    data privacy across institutional boundaries.
    """
    
    def __init__(self, institutions: List[str]):
        self.institutions = institutions
        self.collaboration_graph = defaultdict(set)
        self.shared_insights: Dict[str, List] = {inst: [] for inst in institutions}
        
    def establish_collaboration(self, inst1: str, inst2: str,
                               agreement: Dict) -> bool:
        """
        Establish collaboration agreement between institutions.
        
        Args:
            inst1: First institution
            inst2: Second institution
            agreement: Collaboration terms
            
        Returns:
            True if collaboration established
        """
        if inst1 in self.institutions and inst2 in self.institutions:
            self.collaboration_graph[inst1].add(inst2)
            self.collaboration_graph[inst2].add(inst1)
            logger.info(f"Collaboration established: {inst1} <-> {inst2}")
            return True
        return False
    
    def share_insight(self, from_inst: str, insight: Dict,
                     privacy_level: str = "anonymized") -> None:
        """
        Share research insight with collaborators.
        
        Args:
            from_inst: Source institution
            insight: Research insight data
            privacy_level: Privacy level of sharing
        """
        # Share with connected institutions
        for to_inst in self.collaboration_graph[from_inst]:
            if privacy_level == "anonymized":
                # Remove identifying information
                insight = self._anonymize_insight(insight)
            
            self.shared_insights[to_inst].append({
                'from': from_inst,
                'insight': insight,
                'timestamp': time.time()
            })
    
    def _anonymize_insight(self, insight: Dict) -> Dict:
        """Remove identifying information from insight."""
        anonymized = copy.deepcopy(insight)
        anonymized.pop('institution_specific', None)
        anonymized.pop('proprietary_data', None)
        return anonymized


def create_federated_discovery_demo():
    """
    Create a demonstration of federated materials discovery.
    
    Returns:
        Tuple of (coordinator, clients)
    """
    # Configuration
    config = FederatedDiscoveryConfig(
        num_candidates=100,
        num_iterations=10,
        strategy=DiscoveryStrategy.FEDERATED_BO,
        use_dp=True,
        epsilon=1.0,
        target_properties=['band_gap', 'formation_energy'],
        property_ranges={
            'band_gap': (0.5, 3.0),
            'formation_energy': (-5.0, -1.0)
        }
    )
    
    # Create coordinator
    coordinator = FederatedDiscoveryCoordinator(config)
    
    # Create clients
    institutions = [
        ("mit", "MIT Materials Lab"),
        ("stanford", "Stanford Chemistry"),
        ("berkeley", "Berkeley Physics"),
    ]
    
    clients = []
    for inst_id, inst_name in institutions:
        client = DiscoveryClient(inst_id, inst_name, config)
        
        # Add some local materials
        local_materials = [
            MaterialCandidate(
                composition=f"{inst_id.upper()}_Mat{i}",
                features=np.random.randn(100),
                source_institution=inst_id,
                privacy_level="internal"
            )
            for i in range(10)
        ]
        client.add_local_materials(local_materials)
        
        coordinator.register_institution(inst_id, client)
        clients.append(client)
    
    return coordinator, clients


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Federated Materials Discovery System")
    print("=" * 60)
    
    coordinator, clients = create_federated_discovery_demo()
    
    print(f"\nCreated discovery system with:")
    print(f"  - Coordinator")
    print(f"  - {len(clients)} institutions")
    print(f"  - Strategy: {coordinator.config.strategy.value}")
    print(f"  - Differential Privacy: {coordinator.config.use_dp}")
    
    print("\nRegistered Institutions:")
    for inst_id in coordinator.institutions:
        print(f"  - {inst_id}")
    
    print("\nFederated discovery system ready!")
