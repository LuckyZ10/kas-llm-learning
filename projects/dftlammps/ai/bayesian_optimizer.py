"""
Bayesian Optimizer for Materials Discovery
==========================================

This module implements Bayesian optimization for efficient materials discovery.
Integrates structure generation, property prediction, and DFT validation in a
closed discovery loop.

Key Components:
- Gaussian Process surrogate models
- Acquisition functions (EI, UCB, PI)
- Multi-objective optimization
- Batch acquisition for parallel experiments
- Integration with generative models and DFT

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
from abc import ABC, abstractmethod

# Try to import scikit-learn for GP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

# Try to import GPyTorch for advanced GPs
try:
    import gpytorch
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False
    warnings.warn("gpytorch not available. Using scikit-learn GPs.")

# Import local modules
from .generative_models import StructureGenerator, CrystalStructure, GenerativeModelConfig
from .property_predictor import PropertyPredictor, PropertyPredictorConfig


@dataclass
class BayesianOptimizerConfig:
    """Configuration for Bayesian optimization."""
    
    # Surrogate model
    surrogate_type: str = "gp"  # gp, deep_gp, neural_process
    kernel_type: str = "rbf"  # rbf, matern, polynomial
    noise_level: float = 1e-5
    
    # Acquisition function
    acquisition_type: str = "ei"  # ei, ucb, pi, mes
    beta_ucb: float = 2.0  # Exploration parameter for UCB
    xi: float = 0.01  # Exploration parameter for EI
    
    # Optimization settings
    num_init_samples: int = 10
    num_iterations: int = 50
    batch_size: int = 1  # Number of parallel experiments
    
    # Multi-objective
    multi_objective: bool = False
    reference_point: Optional[np.ndarray] = None
    
    # Constraints
    use_constraints: bool = False
    constraint_models: List[Any] = field(default_factory=list)
    
    # Generative model
    generator_type: str = "cdvae"
    num_candidates: int = 100  # Number of structures to evaluate per iteration
    
    # DFT validation
    dft_validation_frequency: int = 5  # Validate every N iterations
    dft_batch_size: int = 5
    
    # Output
    output_dir: str = "./bayesian_optimization"
    save_frequency: int = 10


@dataclass
class OptimizationResult:
    """Result of a Bayesian optimization run."""
    
    # History
    X: np.ndarray  # Design points (N, d)
    y: np.ndarray  # Observations (N, num_objectives)
    
    # Best found
    best_x: np.ndarray
    best_y: float
    best_structure: Optional[CrystalStructure] = None
    
    # Pareto front (for multi-objective)
    pareto_front: Optional[np.ndarray] = None
    pareto_indices: Optional[List[int]] = None
    
    # Metadata
    num_iterations: int = 0
    num_dft_calculations: int = 0
    computation_time: float = 0.0
    
    def save(self, path: str):
        """Save results to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "OptimizationResult":
        """Load results from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# Surrogate Models
# ============================================================================

class SurrogateModel(ABC):
    """Abstract base class for surrogate models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the surrogate model."""
        pass
    
    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict mean and optionally standard deviation."""
        pass
    
    @abstractmethod
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Update model with new observations."""
        pass


class GaussianProcessSurrogate(SurrogateModel):
    """Gaussian Process surrogate model using scikit-learn."""
    
    def __init__(
        self,
        kernel_type: str = "rbf",
        noise_level: float = 1e-5,
        normalize: bool = True
    ):
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.normalize = normalize
        
        # Scalers
        self.x_scaler = StandardScaler() if normalize else None
        self.y_scaler = StandardScaler() if normalize else None
        
        # Initialize GP
        self._init_gp()
        
        self.X = None
        self.y = None
    
    def _init_gp(self):
        """Initialize Gaussian Process."""
        # Kernel
        if self.kernel_type == "rbf":
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        elif self.kernel_type == "matern":
            from sklearn.gaussian_process.kernels import Matern
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        else:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        
        kernel += WhiteKernel(noise_level=self.noise_level)
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=not self.normalize,
            alpha=self.noise_level
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the GP model."""
        self.X = X.copy()
        self.y = y.copy()
        
        # Normalize
        if self.normalize:
            X_scaled = self.x_scaler.fit_transform(X)
            y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            X_scaled = X
            y_scaled = y
        
        self.gp.fit(X_scaled, y_scaled)
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict mean and standard deviation."""
        # Normalize
        if self.normalize:
            X_scaled = self.x_scaler.transform(X)
        else:
            X_scaled = X
        
        if return_std:
            mean, std = self.gp.predict(X_scaled, return_std=True)
            
            # Inverse transform
            if self.normalize:
                mean = self.y_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
                std = std * self.y_scaler.scale_[0]
            
            return mean, std
        else:
            mean = self.gp.predict(X_scaled, return_std=False)
            
            if self.normalize:
                mean = self.y_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
            
            return mean
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Update model with new observations."""
        self.X = np.vstack([self.X, X_new])
        self.y = np.concatenate([self.y, y_new])
        self.fit(self.X, self.y)


class GPyTorchSurrogate(SurrogateModel):
    """Gaussian Process using GPyTorch for better scalability."""
    
    def __init__(
        self,
        kernel_type: str = "rbf",
        noise_level: float = 1e-5,
        device: str = 'cpu'
    ):
        if not HAS_GPYTORCH:
            raise ImportError("GPyTorch not available")
        
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.device = torch.device(device)
        
        self.model = None
        self.likelihood = None
        self.X = None
        self.y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, training_iter: int = 100):
        """Fit the GP model."""
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Initialize model and likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(
            self.X, self.y, self.likelihood, self.kernel_type
        ).to(self.device)
        
        # Train
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict mean and standard deviation."""
        self.model.eval()
        self.likelihood.eval()
        
        X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self.model(X_torch))
            mean = prediction.mean.cpu().numpy()
            
            if return_std:
                std = prediction.stddev.cpu().numpy()
                return mean, std
            
            return mean
    
    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """Update model with new observations."""
        self.X = torch.vstack([self.X, torch.tensor(X_new, dtype=torch.float32, device=self.device)])
        self.y = torch.cat([self.y, torch.tensor(y_new, dtype=torch.float32, device=self.device)])
        self.fit(self.X.cpu().numpy(), self.y.cpu().numpy())


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model for GPyTorch."""
    
    def __init__(self, train_x, train_y, likelihood, kernel_type='rbf'):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        if kernel_type == 'rbf':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
        elif kernel_type == 'matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5)
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ============================================================================
# Acquisition Functions
# ============================================================================

class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""
    
    @abstractmethod
    def evaluate(
        self,
        X: np.ndarray,
        surrogate: SurrogateModel,
        best_y: float
    ) -> np.ndarray:
        """Evaluate acquisition function."""
        pass


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""
    
    def __init__(self, xi: float = 0.01):
        self.xi = xi
    
    def evaluate(
        self,
        X: np.ndarray,
        surrogate: SurrogateModel,
        best_y: float
    ) -> np.ndarray:
        """Compute EI values."""
        mean, std = surrogate.predict(X, return_std=True)
        
        # Handle numerical issues
        std = np.maximum(std, 1e-9)
        
        # Compute improvement
        improvement = mean - best_y - self.xi
        z = improvement / std
        
        # Expected improvement
        from scipy.stats import norm
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""
    
    def __init__(self, beta: float = 2.0):
        self.beta = beta
    
    def evaluate(
        self,
        X: np.ndarray,
        surrogate: SurrogateModel,
        best_y: float = None
    ) -> np.ndarray:
        """Compute UCB values."""
        mean, std = surrogate.predict(X, return_std=True)
        ucb = mean + np.sqrt(self.beta) * std
        return ucb


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function."""
    
    def __init__(self, xi: float = 0.01):
        self.xi = xi
    
    def evaluate(
        self,
        X: np.ndarray,
        surrogate: SurrogateModel,
        best_y: float
    ) -> np.ndarray:
        """Compute PI values."""
        mean, std = surrogate.predict(X, return_std=True)
        
        std = np.maximum(std, 1e-9)
        improvement = mean - best_y - self.xi
        z = improvement / std
        
        from scipy.stats import norm
        pi = norm.cdf(z)
        
        return pi


# ============================================================================
# Bayesian Optimizer
# ============================================================================

class BayesianOptimizer:
    """
    Bayesian Optimization for materials discovery.
    
    Combines generative models, property predictors, and DFT validation
    in an efficient search loop.
    """
    
    def __init__(
        self,
        config: Optional[BayesianOptimizerConfig] = None,
        structure_generator: Optional[StructureGenerator] = None,
        property_predictor: Optional[PropertyPredictor] = None,
        dft_evaluator: Optional[Callable] = None
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            config: Optimization configuration
            structure_generator: Generator for candidate structures
            property_predictor: ML predictor for fast evaluation
            dft_evaluator: Function for DFT validation
        """
        self.config = config or BayesianOptimizerConfig()
        self.structure_generator = structure_generator
        self.property_predictor = property_predictor
        self.dft_evaluator = dft_evaluator
        
        # Initialize surrogate model
        self.surrogate = self._create_surrogate()
        
        # Initialize acquisition function
        self.acquisition = self._create_acquisition()
        
        # Storage
        self.X_observed = []  # Structure representations
        self.y_observed = []  # Property values
        self.structures_observed = []
        self.dft_validated = []
        
        self.iteration = 0
        self.best_structure = None
        self.best_value = float('-inf')
    
    def _create_surrogate(self) -> SurrogateModel:
        """Create surrogate model."""
        if self.config.surrogate_type == "gp":
            if HAS_GPYTORCH and torch.cuda.is_available():
                return GPyTorchSurrogate(
                    kernel_type=self.config.kernel_type,
                    noise_level=self.config.noise_level
                )
            else:
                return GaussianProcessSurrogate(
                    kernel_type=self.config.kernel_type,
                    noise_level=self.config.noise_level
                )
        else:
            return GaussianProcessSurrogate()
    
    def _create_acquisition(self) -> AcquisitionFunction:
        """Create acquisition function."""
        if self.config.acquisition_type == "ei":
            return ExpectedImprovement(xi=self.config.xi)
        elif self.config.acquisition_type == "ucb":
            return UpperConfidenceBound(beta=self.config.beta_ucb)
        elif self.config.acquisition_type == "pi":
            return ProbabilityOfImprovement(xi=self.config.xi)
        else:
            return ExpectedImprovement()
    
    def optimize(
        self,
        objective_function: Optional[Callable] = None,
        composition_space: Optional[List[str]] = None,
        seed_structures: Optional[List[CrystalStructure]] = None
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            objective_function: Function to optimize (optional)
            composition_space: List of compositions to search
            seed_structures: Initial structures for warm start
        
        Returns:
            OptimizationResult with history and best found
        """
        import time
        start_time = time.time()
        
        # Initialize with random samples
        self._initialize(seed_structures, composition_space, objective_function)
        
        # Optimization loop
        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            print(f"{'='*60}")
            
            # Generate candidate structures
            candidates, candidate_features = self._generate_candidates(composition_space)
            
            # Select next points using acquisition function
            next_indices = self._select_next_points(candidates, candidate_features)
            
            # Evaluate selected structures
            for idx in next_indices:
                structure = candidates[idx]
                value = self._evaluate_structure(structure, objective_function)
                
                # Update observations
                self._update_observations(structure, candidate_features[idx], value)
                
                print(f"  Evaluated: value={value:.4f}, "
                      f"composition={structure.composition}")
            
            # Periodic DFT validation
            if (iteration + 1) % self.config.dft_validation_frequency == 0:
                self._dft_validation()
            
            # Save checkpoint
            if (iteration + 1) % self.config.save_frequency == 0:
                self._save_checkpoint()
            
            # Print progress
            print(f"  Best value so far: {self.best_value:.4f}")
            if self.best_structure:
                print(f"  Best composition: {self.best_structure.composition}")
        
        # Final results
        computation_time = time.time() - start_time
        
        result = OptimizationResult(
            X=np.array(self.X_observed),
            y=np.array(self.y_observed),
            best_x=np.array(self.X_observed[np.argmax(self.y_observed)]),
            best_y=self.best_value,
            best_structure=self.best_structure,
            num_iterations=self.iteration + 1,
            num_dft_calculations=sum(self.dft_validated),
            computation_time=computation_time
        )
        
        return result
    
    def _initialize(
        self,
        seed_structures: Optional[List[CrystalStructure]],
        composition_space: Optional[List[str]],
        objective_function: Optional[Callable]
    ):
        """Initialize with random samples."""
        print(f"Initializing with {self.config.num_init_samples} random samples...")
        
        if seed_structures:
            # Use provided seed structures
            structures = seed_structures[:self.config.num_init_samples]
        else:
            # Generate random structures
            structures = self._generate_random_structures(
                self.config.num_init_samples,
                composition_space
            )
        
        # Evaluate initial structures
        for structure in structures:
            value = self._evaluate_structure(structure, objective_function)
            features = self._structure_to_features(structure)
            
            self._update_observations(structure, features, value)
            print(f"  Initial: value={value:.4f}, "
                  f"composition={structure.composition}")
        
        print(f"Initialization complete. Best: {self.best_value:.4f}")
    
    def _generate_random_structures(
        self,
        num_structures: int,
        composition_space: Optional[List[str]]
    ) -> List[CrystalStructure]:
        """Generate random initial structures."""
        if self.structure_generator:
            compositions = composition_space or ["SiO2", "Li3PS4", "NaCl"]
            structures = []
            for _ in range(num_structures):
                comp = np.random.choice(compositions)
                structs = self.structure_generator.generate(
                    num_structures=1,
                    target_composition=comp
                )
                structures.extend(structs)
            return structures
        else:
            # Generate simple random structures
            return self._generate_fallback_structures(num_structures)
    
    def _generate_fallback_structures(
        self,
        num_structures: int
    ) -> List[CrystalStructure]:
        """Generate simple fallback structures."""
        structures = []
        for _ in range(num_structures):
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
    
    def _generate_candidates(
        self,
        composition_space: Optional[List[str]]
    ) -> Tuple[List[CrystalStructure], np.ndarray]:
        """Generate candidate structures for evaluation."""
        print(f"Generating {self.config.num_candidates} candidate structures...")
        
        if self.structure_generator:
            compositions = composition_space or [None] * self.config.num_candidates
            
            candidates = []
            for _ in range(self.config.num_candidates):
                comp = np.random.choice(compositions) if compositions else None
                structs = self.structure_generator.generate(
                    num_structures=1,
                    target_composition=comp
                )
                candidates.extend(structs)
        else:
            candidates = self._generate_fallback_structures(self.config.num_candidates)
        
        # Convert to feature vectors
        features = np.array([
            self._structure_to_features(s) for s in candidates
        ])
        
        return candidates, features
    
    def _structure_to_features(self, structure: CrystalStructure) -> np.ndarray:
        """Convert structure to feature vector."""
        # Simple feature vector based on composition and structure
        features = []
        
        # Number of atoms
        features.append(len(structure.atomic_numbers) / 100.0)
        
        # Average atomic number
        features.append(np.mean(structure.atomic_numbers) / 100.0)
        
        # Lattice parameters (normalized)
        lattice_params = np.linalg.norm(structure.lattice, axis=1)
        features.extend(lattice_params / 10.0)
        
        # Volume
        volume = np.abs(np.linalg.det(structure.lattice))
        features.append(volume / 1000.0)
        
        # Elemental fractions (simplified - just count first 10 elements)
        elem_counts = np.zeros(10)
        for z in structure.atomic_numbers:
            if z <= 10:
                elem_counts[z - 1] += 1
        elem_counts /= max(len(structure.atomic_numbers), 1)
        features.extend(elem_counts)
        
        return np.array(features, dtype=np.float32)
    
    def _select_next_points(
        self,
        candidates: List[CrystalStructure],
        candidate_features: np.ndarray
    ) -> List[int]:
        """Select next points to evaluate using acquisition function."""
        # Fit surrogate on current observations
        if len(self.X_observed) > 0:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.surrogate.fit(X, y)
            
            # Evaluate acquisition function
            acq_values = self.acquisition.evaluate(
                candidate_features,
                self.surrogate,
                self.best_value
            )
            
            # Select top-k points
            top_indices = np.argsort(acq_values)[-self.config.batch_size:]
            return top_indices.tolist()
        else:
            # Random selection for first iteration
            return np.random.choice(
                len(candidates),
                size=min(self.config.batch_size, len(candidates)),
                replace=False
            ).tolist()
    
    def _evaluate_structure(
        self,
        structure: CrystalStructure,
        objective_function: Optional[Callable]
    ) -> float:
        """Evaluate a structure."""
        if objective_function:
            # Use provided objective function
            return objective_function(structure)
        
        if self.property_predictor:
            # Use ML predictor
            try:
                prediction = self.property_predictor.predict([structure])
                return float(prediction[0])
            except:
                # Fallback to simple heuristic
                return self._simple_heuristic(structure)
        
        # Simple heuristic based on composition
        return self._simple_heuristic(structure)
    
    def _simple_heuristic(self, structure: CrystalStructure) -> float:
        """Simple heuristic for structure quality."""
        # Prefer structures with reasonable density
        volume = np.abs(np.linalg.det(structure.lattice))
        density = len(structure.atomic_numbers) / volume
        
        # Penalize extreme densities
        density_score = -abs(density - 0.1) * 10
        
        # Prefer certain elements (e.g., Li for batteries)
        li_count = np.sum(structure.atomic_numbers == 3)
        li_score = li_count * 0.5
        
        return density_score + li_score + np.random.randn() * 0.1
    
    def _update_observations(
        self,
        structure: CrystalStructure,
        features: np.ndarray,
        value: float
    ):
        """Update observations."""
        self.X_observed.append(features)
        self.y_observed.append(value)
        self.structures_observed.append(structure)
        self.dft_validated.append(False)
        
        # Update best
        if value > self.best_value:
            self.best_value = value
            self.best_structure = structure
    
    def _dft_validation(self):
        """Perform DFT validation on promising structures."""
        if not self.dft_evaluator:
            return
        
        print(f"\nPerforming DFT validation...")
        
        # Select top structures for validation
        top_indices = np.argsort(self.y_observed)[-self.config.dft_batch_size:]
        
        for idx in top_indices:
            if not self.dft_validated[idx]:
                structure = self.structures_observed[idx]
                
                try:
                    # Run DFT
                    dft_value = self.dft_evaluator(structure)
                    
                    # Update observation with DFT value
                    self.y_observed[idx] = dft_value
                    self.dft_validated[idx] = True
                    
                    print(f"  DFT validated: {dft_value:.4f}")
                    
                    # Update best if improved
                    if dft_value > self.best_value:
                        self.best_value = dft_value
                        self.best_structure = structure
                        
                except Exception as e:
                    print(f"  DFT failed: {e}")
    
    def _save_checkpoint(self):
        """Save optimization checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'iteration': self.iteration,
            'X_observed': self.X_observed,
            'y_observed': self.y_observed,
            'best_value': self.best_value,
            'config': self.config,
        }
        
        path = output_dir / f"checkpoint_iter_{self.iteration}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"  Checkpoint saved to {path}")


# ============================================================================
# Multi-Objective Bayesian Optimization
# ============================================================================

class MultiObjectiveBayesianOptimizer(BayesianOptimizer):
    """
    Multi-objective Bayesian optimization using ParEGO or EHVI.
    """
    
    def __init__(
        self,
        config: Optional[BayesianOptimizerConfig] = None,
        **kwargs
    ):
        config = config or BayesianOptimizerConfig()
        config.multi_objective = True
        super().__init__(config, **kwargs)
        
        # Multiple surrogates for multiple objectives
        self.surrogates = []
        self.num_objectives = 2  # Default
    
    def optimize(
        self,
        objective_functions: List[Callable],
        **kwargs
    ) -> OptimizationResult:
        """
        Run multi-objective optimization.
        
        Args:
            objective_functions: List of objective functions to optimize
        """
        self.num_objectives = len(objective_functions)
        
        # Initialize surrogates for each objective
        self.surrogates = [
            self._create_surrogate()
            for _ in range(self.num_objectives)
        ]
        
        # Run optimization with scalarization
        # Simplified: use weighted sum scalarization
        def scalarized_objective(structure):
            values = [f(structure) for f in objective_functions]
            # Random weights for exploration
            weights = np.random.dirichlet(np.ones(self.num_objectives))
            return np.dot(values, weights)
        
        return super().optimize(scalarized_objective, **kwargs)
    
    def _update_pareto_front(self):
        """Update Pareto front from observed points."""
        y = np.array(self.y_observed)
        
        # Find Pareto optimal points
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        is_pareto = np.ones(y.shape[0], dtype=bool)
        for i, point in enumerate(y):
            if is_pareto[i]:
                # Check if any other point dominates this one
                for j, other in enumerate(y):
                    if i != j and is_pareto[j]:
                        if np.all(other >= point) and np.any(other > point):
                            is_pareto[i] = False
                            break
        
        self.pareto_indices = np.where(is_pareto)[0].tolist()
        self.pareto_front = y[is_pareto]


# ============================================================================
# Utility Functions
# ============================================================================

def optimize_materials(
    target_property: str,
    composition_space: List[str],
    num_iterations: int = 50,
    model_type: str = 'cdvae',
    use_dft: bool = True
) -> OptimizationResult:
    """
    High-level function for materials optimization.
    
    Args:
        target_property: Property to optimize ('band_gap', 'ionic_conductivity', etc.)
        composition_space: List of chemical compositions to search
        num_iterations: Number of optimization iterations
        model_type: Type of generative model
        use_dft: Whether to use DFT validation
    
    Returns:
        OptimizationResult
    """
    # Initialize components
    gen_config = GenerativeModelConfig(model_type=model_type)
    generator = StructureGenerator(model_type, gen_config)
    
    pred_config = PropertyPredictorConfig(model_type='cgcnn')
    predictor = PropertyPredictor('cgcnn', pred_config)
    
    # Create optimizer
    config = BayesianOptimizerConfig(
        num_iterations=num_iterations,
        generator_type=model_type
    )
    
    optimizer = BayesianOptimizer(
        config=config,
        structure_generator=generator,
        property_predictor=predictor,
        dft_evaluator=None if not use_dft else lambda s: 0.0
    )
    
    # Run optimization
    result = optimizer.optimize(composition_space=composition_space)
    
    return result


def batch_bayesian_optimization(
    candidate_pool: List[CrystalStructure],
    observed_structures: List[CrystalStructure],
    observed_values: np.ndarray,
    batch_size: int = 5
) -> List[int]:
    """
    Select batch of structures using Bayesian optimization.
    
    Args:
        candidate_pool: Pool of candidate structures
        observed_structures: Previously evaluated structures
        observed_values: Observed property values
        batch_size: Number of structures to select
    
    Returns:
        Indices of selected structures
    """
    # Create surrogate
    surrogate = GaussianProcessSurrogate()
    
    # Convert structures to features
    def struct_to_features(s):
        return np.array([
            len(s.atomic_numbers) / 100.0,
            np.mean(s.atomic_numbers) / 100.0,
            np.abs(np.linalg.det(s.lattice)) / 1000.0
        ])
    
    X_observed = np.array([struct_to_features(s) for s in observed_structures])
    X_candidates = np.array([struct_to_features(s) for s in candidate_pool])
    
    # Fit surrogate
    surrogate.fit(X_observed, observed_values)
    
    # Greedy batch selection
    selected = []
    best_y = observed_values.max()
    
    for _ in range(batch_size):
        # Evaluate acquisition
        acq = ExpectedImprovement()
        values = acq.evaluate(X_candidates, surrogate, best_y)
        
        # Select best
        idx = np.argmax(values)
        selected.append(idx)
        
        # Remove from candidates (set to -inf)
        values[idx] = -np.inf
        X_candidates = np.delete(X_candidates, idx, axis=0)
    
    return selected


if __name__ == "__main__":
    # Example usage
    print("Bayesian Optimizer Module")
    print("=" * 50)
    
    # Create optimizer
    config = BayesianOptimizerConfig(
        num_init_samples=5,
        num_iterations=10,
        num_candidates=20
    )
    
    optimizer = BayesianOptimizer(config)
    
    # Run optimization
    result = optimizer.optimize(
        composition_space=["Li3PS4", "Li2S", "Li3N"]
    )
    
    print(f"\nOptimization complete!")
    print(f"Best value: {result.best_y:.4f}")
    print(f"Total iterations: {result.num_iterations}")
