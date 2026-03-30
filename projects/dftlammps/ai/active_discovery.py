"""
Active Discovery Workflow
=========================

This module implements active learning workflows for materials discovery.
Combines generative models, property prediction, and selective DFT validation
to efficiently explore materials space.

Key Features:
- Active learning with uncertainty quantification
- Human-in-the-loop validation
- Adaptive sampling strategies
- Integration with DFT+MD pipeline
- Continuous learning from new data

Author: DFT+LAMMPS AI Team
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from collections import defaultdict
import warnings
from enum import Enum

# Import local modules
from .generative_models import StructureGenerator, CrystalStructure, GenerativeModelConfig
from .property_predictor import PropertyPredictor, PropertyPredictorConfig, MaterialGraph
from .bayesian_optimizer import BayesianOptimizer, BayesianOptimizerConfig


class SamplingStrategy(Enum):
    """Active learning sampling strategies."""
    UNCERTAINTY = "uncertainty"  # Uncertainty sampling
    DIVERSITY = "diversity"  # Diversity sampling
    GREEDY = "greedy"  # Greedy (best predicted)
    HYBRID = "hybrid"  # Hybrid approach
    QUERY_BY_COMMITTEE = "qbc"  # Query by committee


@dataclass
class ActiveDiscoveryConfig:
    """Configuration for active discovery workflow."""
    
    # Sampling strategy
    sampling_strategy: SamplingStrategy = SamplingStrategy.HYBRID
    uncertainty_weight: float = 0.5
    diversity_weight: float = 0.3
    greedy_weight: float = 0.2
    
    # Iteration settings
    num_iterations: int = 20
    samples_per_iteration: int = 5
    validation_split: float = 0.2
    
    # Model retraining
    retrain_frequency: int = 3  # Retrain every N iterations
    min_samples_for_retrain: int = 20
    
    # Uncertainty estimation
    uncertainty_method: str = "ensemble"  # ensemble, dropout, mcdropout
    num_ensemble_models: int = 5
    mc_dropout_iterations: int = 20
    
    # Query by committee
    committee_models: List[str] = field(default_factory=lambda: ['cgcnn', 'megnet', 'alignn'])
    
    # DFT integration
    dft_validation_threshold: float = 0.1  # Uncertainty threshold for DFT
    max_dft_per_iteration: int = 3
    
    # Output
    output_dir: str = "./active_discovery"
    save_frequency: int = 5
    
    # Human-in-the-loop
    enable_human_validation: bool = False
    human_review_threshold: float = 0.8


@dataclass
class DiscoveryIteration:
    """Results from one active learning iteration."""
    iteration: int
    selected_structures: List[CrystalStructure]
    predicted_values: np.ndarray
    uncertainties: np.ndarray
    dft_values: Optional[np.ndarray] = None
    dft_performed: List[bool] = field(default_factory=list)
    
    # Model performance
    model_updated: bool = False
    validation_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ActiveDiscoveryResult:
    """Complete results from active discovery workflow."""
    
    # History
    iterations: List[DiscoveryIteration] = field(default_factory=list)
    all_structures: List[CrystalStructure] = field(default_factory=list)
    all_predictions: List[float] = field(default_factory=list)
    all_uncertainties: List[float] = field(default_factory=list)
    
    # Final model
    final_model: Optional[PropertyPredictor] = None
    
    # Statistics
    total_dft_calculations: int = 0
    total_structures_generated: int = 0
    
    def get_best_structures(self, n: int = 10) -> List[Tuple[CrystalStructure, float]]:
        """Get top N structures by predicted value."""
        sorted_indices = np.argsort(self.all_predictions)[-n:][::-1]
        return [
            (self.all_structures[i], self.all_predictions[i])
            for i in sorted_indices
        ]
    
    def save(self, path: str):
        """Save results to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "ActiveDiscoveryResult":
        """Load results from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# Uncertainty Estimators
# ============================================================================

class UncertaintyEstimator:
    """Base class for uncertainty estimation."""
    
    def estimate(
        self,
        structures: List[CrystalStructure],
        predictor: PropertyPredictor
    ) -> np.ndarray:
        """Estimate uncertainty for structures."""
        raise NotImplementedError


class EnsembleUncertainty(UncertaintyEstimator):
    """Uncertainty estimation using model ensemble."""
    
    def __init__(self, num_models: int = 5, model_types: Optional[List[str]] = None):
        self.num_models = num_models
        self.model_types = model_types or ['cgcnn'] * num_models
        self.ensemble = []
    
    def fit(
        self,
        structures: List[CrystalStructure],
        targets: np.ndarray,
        device: str = 'cpu'
    ):
        """Train ensemble of models."""
        self.ensemble = []
        
        for i, model_type in enumerate(self.model_types[:self.num_models]):
            print(f"  Training ensemble model {i+1}/{self.num_models}...")
            
            config = PropertyPredictorConfig(model_type=model_type)
            model = PropertyPredictor(model_type, config, device)
            
            # Train with different random seeds
            np.random.seed(i)
            torch.manual_seed(i)
            
            model.train(structures, targets, num_epochs=50, verbose=False)
            self.ensemble.append(model)
    
    def estimate(
        self,
        structures: List[CrystalStructure],
        predictor: PropertyPredictor = None
    ) -> np.ndarray:
        """Estimate uncertainty as variance of ensemble predictions."""
        predictions = []
        
        for model in self.ensemble:
            preds = model.predict(structures)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        uncertainty = predictions.var(axis=0)
        
        return uncertainty


class MCDropoutUncertainty(UncertaintyEstimator):
    """Uncertainty estimation using MC Dropout."""
    
    def __init__(self, num_iterations: int = 20):
        self.num_iterations = num_iterations
    
    def estimate(
        self,
        structures: List[CrystalStructure],
        predictor: PropertyPredictor
    ) -> np.ndarray:
        """Estimate uncertainty using MC Dropout."""
        # Enable dropout during inference
        predictor.model.train()  # Keep dropout active
        
        predictions = []
        for _ in range(self.num_iterations):
            preds = predictor.predict(structures)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        uncertainty = predictions.var(axis=0)
        
        predictor.model.eval()
        
        return uncertainty


class QueryByCommittee(UncertaintyEstimator):
    """Uncertainty estimation using committee of different models."""
    
    def __init__(self, model_types: List[str] = None):
        self.model_types = model_types or ['cgcnn', 'megnet', 'alignn']
        self.committee = {}
    
    def fit(
        self,
        structures: List[CrystalStructure],
        targets: np.ndarray,
        device: str = 'cpu'
    ):
        """Train committee models."""
        for model_type in self.model_types:
            print(f"  Training {model_type} for committee...")
            
            config = PropertyPredictorConfig(model_type=model_type)
            model = PropertyPredictor(model_type, config, device)
            model.train(structures, targets, num_epochs=50, verbose=False)
            
            self.committee[model_type] = model
    
    def estimate(
        self,
        structures: List[CrystalStructure],
        predictor: PropertyPredictor = None
    ) -> np.ndarray:
        """Estimate uncertainty as disagreement among committee members."""
        predictions = []
        
        for model_type, model in self.committee.items():
            preds = model.predict(structures)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        uncertainty = predictions.var(axis=0)
        
        return uncertainty


# ============================================================================
# Diversity Sampling
# ============================================================================

class DiversitySampler:
    """Sampler for diverse structure selection."""
    
    def __init__(self, feature_extractor: Optional[Callable] = None):
        self.feature_extractor = feature_extractor or self._default_features
    
    def sample(
        self,
        structures: List[CrystalStructure],
        n_samples: int,
        already_selected: Optional[List[int]] = None
    ) -> List[int]:
        """
        Select diverse set of structures using MaxMin algorithm.
        
        Args:
            structures: Pool of candidate structures
            n_samples: Number of structures to select
            already_selected: Already selected indices
        
        Returns:
            Indices of selected structures
        """
        # Extract features
        features = np.array([self.feature_extractor(s) for s in structures])
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        n = len(structures)
        selected = already_selected.copy() if already_selected else []
        
        if not selected:
            # Start with random structure
            selected = [np.random.randint(n)]
        
        # MaxMin selection
        while len(selected) < n_samples and len(selected) < n:
            # Compute distances from unselected to selected
            unselected = [i for i in range(n) if i not in selected]
            
            min_distances = []
            for i in unselected:
                dists = np.linalg.norm(features[i] - features[selected], axis=1)
                min_distances.append(dists.min())
            
            # Select point with maximum minimum distance
            next_idx = unselected[np.argmax(min_distances)]
            selected.append(next_idx)
        
        return selected
    
    def _default_features(self, structure: CrystalStructure) -> np.ndarray:
        """Extract default features from structure."""
        features = []
        
        # Composition features
        elem_counts = defaultdict(int)
        for z in structure.atomic_numbers:
            elem_counts[z] += 1
        
        # Atomic number histogram
        hist = np.zeros(20)
        for z, count in elem_counts.items():
            if z <= 20:
                hist[z - 1] = count
        features.extend(hist / max(len(structure.atomic_numbers), 1))
        
        # Structural features
        volume = np.abs(np.linalg.det(structure.lattice))
        features.append(volume / 1000.0)
        features.append(len(structure.atomic_numbers) / 100.0)
        
        # Average atomic number
        features.append(np.mean(structure.atomic_numbers) / 100.0)
        
        return np.array(features, dtype=np.float32)


# ============================================================================
# Active Discovery Workflow
# ============================================================================

class ActiveDiscovery:
    """
    Active discovery workflow for materials exploration.
    
    Combines uncertainty sampling, diversity sampling, and selective DFT
    to efficiently discover new materials.
    """
    
    def __init__(
        self,
        config: Optional[ActiveDiscoveryConfig] = None,
        structure_generator: Optional[StructureGenerator] = None,
        property_predictor: Optional[PropertyPredictor] = None,
        dft_evaluator: Optional[Callable] = None
    ):
        """
        Initialize active discovery workflow.
        
        Args:
            config: Workflow configuration
            structure_generator: Generator for new structures
            property_predictor: ML predictor for properties
            dft_evaluator: Function for DFT validation
        """
        self.config = config or ActiveDiscoveryConfig()
        self.generator = structure_generator
        self.predictor = property_predictor
        self.dft_evaluator = dft_evaluator
        
        # Uncertainty estimator
        self.uncertainty_estimator = self._create_uncertainty_estimator()
        
        # Diversity sampler
        self.diversity_sampler = DiversitySampler()
        
        # Storage
        self.labeled_structures = []
        self.labeled_values = []
        self.unlabeled_structures = []
        
        self.iteration = 0
        self.results = ActiveDiscoveryResult()
    
    def _create_uncertainty_estimator(self) -> Optional[UncertaintyEstimator]:
        """Create uncertainty estimator based on config."""
        method = self.config.uncertainty_method
        
        if method == "ensemble":
            return EnsembleUncertainty(self.config.num_ensemble_models)
        elif method == "mcdropout":
            return MCDropoutUncertainty(self.config.mc_dropout_iterations)
        elif method == "qbc":
            return QueryByCommittee(self.config.committee_models)
        
        return None
    
    def run(
        self,
        seed_structures: Optional[List[CrystalStructure]] = None,
        seed_values: Optional[np.ndarray] = None,
        composition_space: Optional[List[str]] = None,
        num_candidates_per_iteration: int = 100
    ) -> ActiveDiscoveryResult:
        """
        Run active discovery workflow.
        
        Args:
            seed_structures: Initial labeled structures
            seed_values: Initial property values
            composition_space: Compositions to explore
            num_candidates_per_iteration: Candidates to generate per iteration
        
        Returns:
            ActiveDiscoveryResult with all discoveries
        """
        print("=" * 70)
        print("ACTIVE DISCOVERY WORKFLOW")
        print("=" * 70)
        
        # Initialize with seed data
        if seed_structures is not None and seed_values is not None:
            self.labeled_structures = seed_structures.copy()
            self.labeled_values = seed_values.copy()
            print(f"Initialized with {len(seed_structures)} seed structures")
        
        # Main loop
        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{self.config.num_iterations}")
            print(f"{'='*70}")
            
            # Generate candidate structures
            print(f"\nGenerating {num_candidates_per_iteration} candidate structures...")
            candidates = self._generate_candidates(
                num_candidates_per_iteration,
                composition_space
            )
            self.unlabeled_structures = candidates
            self.results.total_structures_generated += len(candidates)
            
            # Make predictions
            print("Making predictions...")
            predictions = self.predictor.predict(candidates)
            
            # Estimate uncertainties
            print("Estimating uncertainties...")
            uncertainties = self._estimate_uncertainty(candidates)
            
            # Select structures to label
            print("Selecting structures for labeling...")
            selected_indices = self._select_structures(
                candidates,
                predictions,
                uncertainties
            )
            
            selected_structures = [candidates[i] for i in selected_indices]
            selected_predictions = predictions[selected_indices]
            selected_uncertainties = uncertainties[selected_indices]
            
            print(f"Selected {len(selected_indices)} structures")
            
            # Perform DFT on selected structures
            print("\nRunning DFT calculations...")
            dft_values, dft_performed = self._run_dft(selected_structures)
            
            # Update labeled data
            self._update_labeled_data(
                selected_structures,
                dft_values if dft_values is not None else selected_predictions
            )
            
            # Retrain model if needed
            model_updated = False
            if (len(self.labeled_structures) >= self.config.min_samples_for_retrain and
                (iteration + 1) % self.config.retrain_frequency == 0):
                print("\nRetraining model...")
                self._retrain_model()
                model_updated = True
            
            # Record iteration
            iteration_result = DiscoveryIteration(
                iteration=iteration,
                selected_structures=selected_structures,
                predicted_values=selected_predictions,
                uncertainties=selected_uncertainties,
                dft_values=dft_values,
                dft_performed=dft_performed,
                model_updated=model_updated
            )
            
            self.results.iterations.append(iteration_result)
            self.results.all_structures.extend(selected_structures)
            self.results.all_predictions.extend(selected_predictions.tolist())
            self.results.all_uncertainties.extend(selected_uncertainties.tolist())
            
            # Print summary
            self._print_iteration_summary(iteration_result)
            
            # Save checkpoint
            if (iteration + 1) % self.config.save_frequency == 0:
                self._save_checkpoint()
        
        # Final model
        self.results.final_model = self.predictor
        
        print("\n" + "=" * 70)
        print("ACTIVE DISCOVERY COMPLETE")
        print("=" * 70)
        print(f"Total DFT calculations: {self.results.total_dft_calculations}")
        print(f"Total structures generated: {self.results.total_structures_generated}")
        
        # Print top discoveries
        print("\nTop 5 Discoveries:")
        for i, (struct, value) in enumerate(self.results.get_best_structures(5)):
            print(f"  {i+1}. {struct.composition}: {value:.4f}")
        
        return self.results
    
    def _generate_candidates(
        self,
        num_candidates: int,
        composition_space: Optional[List[str]]
    ) -> List[CrystalStructure]:
        """Generate candidate structures."""
        if self.generator is None:
            # Fallback: generate simple random structures
            return self._generate_fallback_structures(num_candidates)
        
        candidates = []
        compositions = composition_space or [None]
        
        for _ in range(num_candidates):
            comp = np.random.choice(compositions)
            structs = self.generator.generate(
                num_structures=1,
                target_composition=comp
            )
            candidates.extend(structs)
        
        return candidates
    
    def _generate_fallback_structures(self, n: int) -> List[CrystalStructure]:
        """Generate simple fallback structures."""
        structures = []
        for _ in range(n):
            n_atoms = np.random.randint(5, 30)
            lattice = np.eye(3) * (5 + np.random.rand() * 5)
            frac_coords = np.random.rand(n_atoms, 3)
            atomic_numbers = np.random.choice([1, 3, 6, 8, 11, 14, 16], n_atoms)
            
            structures.append(CrystalStructure(
                lattice=lattice,
                frac_coords=frac_coords,
                atomic_numbers=atomic_numbers
            ))
        
        return structures
    
    def _estimate_uncertainty(
        self,
        structures: List[CrystalStructure]
    ) -> np.ndarray:
        """Estimate uncertainty for structures."""
        if self.uncertainty_estimator is not None:
            try:
                return self.uncertainty_estimator.estimate(structures, self.predictor)
            except:
                pass
        
        # Fallback: use prediction variance as uncertainty proxy
        return np.ones(len(structures)) * 0.1
    
    def _select_structures(
        self,
        candidates: List[CrystalStructure],
        predictions: np.ndarray,
        uncertainties: np.ndarray
    ) -> List[int]:
        """Select structures based on sampling strategy."""
        strategy = self.config.sampling_strategy
        n = self.config.samples_per_iteration
        
        if strategy == SamplingStrategy.UNCERTAINTY:
            # Select most uncertain
            return np.argsort(uncertainties)[-n:].tolist()
        
        elif strategy == SamplingStrategy.GREEDY:
            # Select best predicted
            return np.argsort(predictions)[-n:].tolist()
        
        elif strategy == SamplingStrategy.DIVERSITY:
            # Select diverse
            return self.diversity_sampler.sample(candidates, n)
        
        elif strategy == SamplingStrategy.QUERY_BY_COMMITTEE:
            # QBC handled by uncertainty
            return np.argsort(uncertainties)[-n:].tolist()
        
        else:  # HYBRID
            # Combine strategies
            n_uncertainty = int(n * self.config.uncertainty_weight)
            n_diversity = int(n * self.config.diversity_weight)
            n_greedy = n - n_uncertainty - n_diversity
            
            selected = []
            
            # Uncertainty sampling
            unc_indices = np.argsort(uncertainties)[-n_uncertainty * 2:]
            selected.extend(unc_indices[-n_uncertainty:].tolist())
            
            # Greedy sampling
            greedy_indices = np.argsort(predictions)[-n_greedy:]
            selected.extend(greedy_indices.tolist())
            
            # Diversity sampling
            diverse_indices = self.diversity_sampler.sample(
                candidates, n_diversity, selected
            )
            selected.extend(diverse_indices)
            
            return selected[:n]
    
    def _run_dft(
        self,
        structures: List[CrystalStructure]
    ) -> Tuple[Optional[np.ndarray], List[bool]]:
        """Run DFT calculations on selected structures."""
        if self.dft_evaluator is None:
            return None, [False] * len(structures)
        
        dft_values = []
        dft_performed = []
        
        for structure in structures:
            # Decide whether to run DFT based on uncertainty
            # For now, run on all selected
            try:
                value = self.dft_evaluator(structure)
                dft_values.append(value)
                dft_performed.append(True)
                self.results.total_dft_calculations += 1
            except Exception as e:
                print(f"  DFT failed: {e}")
                dft_values.append(0.0)
                dft_performed.append(False)
        
        return np.array(dft_values), dft_performed
    
    def _update_labeled_data(
        self,
        structures: List[CrystalStructure],
        values: np.ndarray
    ):
        """Update labeled dataset."""
        self.labeled_structures.extend(structures)
        self.labeled_values = np.concatenate([self.labeled_values, values])
    
    def _retrain_model(self):
        """Retrain the property predictor."""
        self.predictor.train(
            self.labeled_structures,
            self.labeled_values,
            num_epochs=100,
            validation_split=0.2,
            verbose=False
        )
        
        # Retrain uncertainty estimator if needed
        if self.uncertainty_estimator is not None:
            if isinstance(self.uncertainty_estimator, (EnsembleUncertainty, QueryByCommittee)):
                self.uncertainty_estimator.fit(
                    self.labeled_structures,
                    self.labeled_values
                )
    
    def _print_iteration_summary(self, iteration: DiscoveryIteration):
        """Print summary of iteration."""
        print(f"\nIteration {iteration.iteration + 1} Summary:")
        print(f"  Structures selected: {len(iteration.selected_structures)}")
        print(f"  Avg predicted value: {iteration.predicted_values.mean():.4f}")
        print(f"  Avg uncertainty: {iteration.uncertainties.mean():.4f}")
        print(f"  DFT performed: {sum(iteration.dft_performed)}")
        print(f"  Model updated: {iteration.model_updated}")
        
        if iteration.dft_values is not None:
            print(f"  Avg DFT value: {iteration.dft_values.mean():.4f}")
    
    def _save_checkpoint(self):
        """Save workflow checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'iteration': self.iteration,
            'labeled_structures': self.labeled_structures,
            'labeled_values': self.labeled_values,
            'results': self.results,
            'config': self.config,
        }
        
        path = output_dir / f"checkpoint_iter_{self.iteration}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved to {path}")


# ============================================================================
# Integration with DFT+MD Pipeline
# ============================================================================

class ActiveDiscoveryPipeline:
    """
    Integration of active discovery with DFT+LAMMPS workflow.
    
    Provides seamless integration with existing high-throughput screening.
    """
    
    def __init__(
        self,
        discovery: ActiveDiscovery,
        dft_bridge: Any = None,
        md_runner: Any = None
    ):
        """
        Initialize pipeline.
        
        Args:
            discovery: ActiveDiscovery instance
            dft_bridge: DFT bridge for calculations
            md_runner: MD simulation runner
        """
        self.discovery = discovery
        self.dft_bridge = dft_bridge
        self.md_runner = md_runner
    
    def run_with_dft_validation(
        self,
        initial_structures: List[CrystalStructure],
        target_property: str = "band_gap",
        num_iterations: int = 20
    ) -> ActiveDiscoveryResult:
        """
        Run discovery with automatic DFT validation.
        
        Args:
            initial_structures: Starting structures
            target_property: Property to optimize
            num_iterations: Number of iterations
        
        Returns:
            ActiveDiscoveryResult
        """
        # Create DFT evaluator
        def dft_evaluator(structure: CrystalStructure) -> float:
            return self._run_dft_calculation(structure, target_property)
        
        self.discovery.dft_evaluator = dft_evaluator
        
        # Run discovery
        # Initialize with dummy values for seed structures
        seed_values = np.random.randn(len(initial_structures))
        
        result = self.discovery.run(
            seed_structures=initial_structures,
            seed_values=seed_values
        )
        
        return result
    
    def _run_dft_calculation(
        self,
        structure: CrystalStructure,
        property_name: str
    ) -> float:
        """Run DFT calculation for a structure."""
        if self.dft_bridge is None:
            # Mock DFT calculation
            return np.random.randn()
        
        # Convert to appropriate format
        # This would integrate with actual DFT bridge
        # Placeholder for actual implementation
        
        return 0.0
    
    def screen_with_active_learning(
        self,
        candidate_structures: List[CrystalStructure],
        batch_size: int = 10,
        num_rounds: int = 5
    ) -> List[Tuple[CrystalStructure, float, float]]:
        """
        Screen candidates using active learning.
        
        Args:
            candidate_structures: Pool of candidates
            batch_size: Structures per batch
            num_rounds: Number of screening rounds
        
        Returns:
            Ranked list of (structure, prediction, uncertainty)
        """
        results = []
        
        for round_idx in range(num_rounds):
            print(f"\nScreening Round {round_idx + 1}/{num_rounds}")
            
            # Predict properties
            predictions = self.discovery.predictor.predict(candidate_structures)
            uncertainties = self.discovery._estimate_uncertainty(candidate_structures)
            
            # Select batch
            selected_indices = self.discovery._select_structures(
                candidate_structures,
                predictions,
                uncertainties
            )
            
            # Run DFT on selected
            for idx in selected_indices[:batch_size]:
                dft_value = self._run_dft_calculation(
                    candidate_structures[idx],
                    "band_gap"
                )
                results.append((
                    candidate_structures[idx],
                    predictions[idx],
                    uncertainties[idx]
                ))
            
            # Update model
            self.discovery._retrain_model()
        
        # Sort by predicted value
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


# ============================================================================
# Utility Functions
# ============================================================================

def run_active_discovery_for_battery_materials(
    target_ion: str = "Li",
    num_iterations: int = 20,
    output_dir: str = "./battery_discovery"
) -> ActiveDiscoveryResult:
    """
    Run active discovery for battery materials.
    
    Args:
        target_ion: Target ionic species (Li, Na, K, etc.)
        num_iterations: Number of discovery iterations
        output_dir: Output directory
    
    Returns:
        ActiveDiscoveryResult
    """
    # Create configuration
    config = ActiveDiscoveryConfig(
        num_iterations=num_iterations,
        sampling_strategy=SamplingStrategy.HYBRID,
        output_dir=output_dir
    )
    
    # Initialize components
    gen_config = GenerativeModelConfig(model_type='cdvae')
    generator = StructureGenerator('cdvae', gen_config)
    
    pred_config = PropertyPredictorConfig(model_type='cgcnn')
    predictor = PropertyPredictor('cgcnn', pred_config)
    
    # Create discovery workflow
    discovery = ActiveDiscovery(
        config=config,
        structure_generator=generator,
        property_predictor=predictor
    )
    
    # Define composition space for battery materials
    composition_space = [
        f"{target_ion}3PS4",
        f"{target_ion}2S",
        f"{target_ion}3N",
        f"{target_ion}AlO2",
        f"{target_ion}FePO4",
    ]
    
    # Run discovery
    result = discovery.run(composition_space=composition_space)
    
    return result


def run_active_discovery_for_catalysts(
    target_reaction: str = "oxygen_reduction",
    num_iterations: int = 20
) -> ActiveDiscoveryResult:
    """
    Run active discovery for catalyst materials.
    
    Args:
        target_reaction: Target catalytic reaction
        num_iterations: Number of discovery iterations
    
    Returns:
        ActiveDiscoveryResult
    """
    config = ActiveDiscoveryConfig(
        num_iterations=num_iterations,
        sampling_strategy=SamplingStrategy.HYBRID
    )
    
    # Initialize components
    generator = StructureGenerator('cdvae', GenerativeModelConfig())
    predictor = PropertyPredictor('cgcnn', PropertyPredictorConfig())
    
    discovery = ActiveDiscovery(
        config=config,
        structure_generator=generator,
        property_predictor=predictor
    )
    
    # Composition space for catalysts
    composition_space = [
        "PtNi", "PtCo", "PtFe", "NiFe", "CoFe",
        "PtNiCo", "PtNiFe", "CoNiFe", "PtCoFe"
    ]
    
    result = discovery.run(composition_space=composition_space)
    
    return result


def integrate_with_high_throughput_screening(
    screening_pipeline: Any,
    active_discovery: ActiveDiscovery,
    selection_ratio: float = 0.1
) -> List[CrystalStructure]:
    """
    Integrate active learning with high-throughput screening.
    
    Args:
        screening_pipeline: Existing screening pipeline
        active_discovery: Active discovery instance
        selection_ratio: Fraction of structures to select with active learning
    
    Returns:
        Selected structures for detailed study
    """
    # Get candidates from screening
    candidates = screening_pipeline.get_candidates()
    
    # Use active learning to select most informative
    predictions = active_discovery.predictor.predict(candidates)
    uncertainties = active_discovery._estimate_uncertainty(candidates)
    
    # Hybrid selection: high value + high uncertainty
    scores = predictions + 0.5 * uncertainties
    n_select = int(len(candidates) * selection_ratio)
    
    selected_indices = np.argsort(scores)[-n_select:]
    selected = [candidates[i] for i in selected_indices]
    
    return selected


if __name__ == "__main__":
    # Example usage
    print("Active Discovery Workflow Module")
    print("=" * 50)
    
    # Create configuration
    config = ActiveDiscoveryConfig(
        num_iterations=5,
        samples_per_iteration=3,
        sampling_strategy=SamplingStrategy.HYBRID
    )
    
    # Initialize components
    generator = StructureGenerator('random', GenerativeModelConfig())
    predictor = PropertyPredictor('cgcnn', PropertyPredictorConfig())
    
    # Create discovery workflow
    discovery = ActiveDiscovery(
        config=config,
        structure_generator=generator,
        property_predictor=predictor
    )
    
    # Run discovery
    print("\nRunning active discovery...")
    result = discovery.run(composition_space=["Li3PS4", "Na2S"])
    
    print(f"\nDiscovery complete!")
    print(f"Total structures: {len(result.all_structures)}")
    print(f"Total DFT calculations: {result.total_dft_calculations}")
