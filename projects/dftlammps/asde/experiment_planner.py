"""
Experiment Planner Module

Implements active learning and Bayesian optimization for selecting
optimal experiments to test hypotheses efficiently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol, TypeVar
import uuid

import numpy as np
from numpy.typing import NDArray

from .hypothesis_generator import Hypothesis


T = TypeVar('T')
Array = NDArray[np.float64]


class VariableType(Enum):
    """Types of experimental variables."""
    CONTINUOUS = auto()
    DISCRETE = auto()
    CATEGORICAL = auto()
    BOOLEAN = auto()


@dataclass
class ExperimentalVariable:
    """Defines an experimental variable (parameter)."""
    name: str
    var_type: VariableType
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    categories: Optional[list[str]] = None
    default_value: Optional[Any] = None
    unit: Optional[str] = None
    description: str = ""
    
    def __post_init__(self) -> None:
        """Validate variable definition."""
        if self.var_type == VariableType.CONTINUOUS:
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("Continuous variables require bounds")
        elif self.var_type == VariableType.CATEGORICAL:
            if not self.categories:
                raise ValueError("Categorical variables require categories")


@dataclass
class ExperimentalCondition:
    """A specific set of experimental parameters."""
    id: str
    values: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    predicted_outcome: Optional[float] = None
    uncertainty: Optional[float] = None
    acquisition_score: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentalCondition:
        """Create from dictionary."""
        return cls(
            id=data.get('id', str(uuid.uuid4())[:8]),
            values=data.get('values', {}),
            metadata=data.get('metadata', {}),
            predicted_outcome=data.get('predicted_outcome'),
            uncertainty=data.get('uncertainty'),
            acquisition_score=data.get('acquisition_score')
        )


@dataclass
class ExperimentResult:
    """Result of running an experiment."""
    condition_id: str
    values: dict[str, Any]  # The condition that was tested
    outcome: float
    outcome_name: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "values": self.values,
            "outcome": self.outcome,
            "outcome_name": self.outcome_name,
            "success": self.success,
            "metadata": self.metadata
        }


class SurrogateModel(ABC):
    """Abstract base class for surrogate models in Bayesian optimization."""
    
    @abstractmethod
    def fit(self, X: Array, y: Array) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: Array, return_std: bool = True) -> tuple[Array, Optional[Array]]:
        """Predict mean and optionally standard deviation."""
        pass
    
    @abstractmethod
    def update(self, X_new: Array, y_new: Array) -> None:
        """Update model with new observations."""
        pass


class GaussianProcessSurrogate(SurrogateModel):
    """
    Gaussian Process surrogate model with RBF kernel.
    
    Simple implementation suitable for small-scale optimization.
    """
    
    def __init__(
        self,
        length_scale: float = 1.0,
        noise_level: float = 1e-5,
        signal_variance: float = 1.0
    ) -> None:
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.signal_variance = signal_variance
        
        self.X_train: Optional[Array] = None
        self.y_train: Optional[Array] = None
        self.K_inv: Optional[Array] = None
        self.alpha: Optional[Array] = None
    
    def _rbf_kernel(self, X1: Array, X2: Array) -> Array:
        """RBF (squared exponential) kernel."""
        sq_dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1) +
            np.sum(X2**2, axis=1) -
            2 * np.dot(X1, X2.T)
        )
        return self.signal_variance * np.exp(-0.5 * sq_dists / self.length_scale**2)
    
    def fit(self, X: Array, y: Array) -> None:
        """Fit GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Compute kernel matrix
        K = self._rbf_kernel(X, X)
        K += self.noise_level * np.eye(len(X))
        
        # Store inverse for prediction
        self.K_inv = np.linalg.inv(K)
        self.alpha = self.K_inv @ y
    
    def predict(self, X: Array, return_std: bool = True) -> tuple[Array, Optional[Array]]:
        """Predict mean and standard deviation."""
        if self.X_train is None or self.alpha is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Compute kernel with training data
        K_s = self._rbf_kernel(X, self.X_train)
        
        # Mean prediction
        mu = K_s @ self.alpha
        
        if not return_std:
            return mu, None
        
        # Variance prediction
        K_ss = self._rbf_kernel(X, X)
        v = self.K_inv @ K_s.T
        var = np.diag(K_ss) - np.sum(K_s.T * v, axis=0)
        std = np.sqrt(np.maximum(var, 0))
        
        return mu, std
    
    def update(self, X_new: Array, y_new: Array) -> None:
        """Update with new observations."""
        if self.X_train is None:
            self.fit(X_new, y_new)
        else:
            X = np.vstack([self.X_train, X_new])
            y = np.concatenate([self.y_train, y_new])
            self.fit(X, y)


class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""
    
    @abstractmethod
    def compute(
        self,
        mu: Array,
        sigma: Array,
        y_best: float
    ) -> Array:
        """Compute acquisition values."""
        pass


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement acquisition function.
    
    EI(x) = E[max(0, f(x) - f(x+))]
    """
    
    def __init__(self, xi: float = 0.01) -> None:
        self.xi = xi  # Exploration parameter
    
    def compute(self, mu: Array, sigma: Array, y_best: float) -> Array:
        """Compute expected improvement."""
        with np.errstate(divide='warn'):
            imp = mu - y_best - self.xi
            Z = imp / (sigma + 1e-9)
            
            from scipy.stats import norm
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-9] = 0
        
        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.
    
    UCB(x) = mu(x) + kappa * sigma(x)
    """
    
    def __init__(self, kappa: float = 2.0) -> None:
        self.kappa = kappa  # Exploration parameter
    
    def compute(self, mu: Array, sigma: Array, y_best: float) -> Array:
        """Compute UCB."""
        return mu + self.kappa * sigma


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement acquisition function.
    
    PI(x) = P(f(x) > f(x+) + xi)
    """
    
    def __init__(self, xi: float = 0.01) -> None:
        self.xi = xi
    
    def compute(self, mu: Array, sigma: Array, y_best: float) -> Array:
        """Compute PI."""
        from scipy.stats import norm
        
        with np.errstate(divide='warn'):
            Z = (mu - y_best - self.xi) / (sigma + 1e-9)
            pi = norm.cdf(Z)
        
        return pi


class ExperimentPlanner:
    """
    Active learning experiment planner using Bayesian optimization.
    """
    
    def __init__(
        self,
        variables: list[ExperimentalVariable],
        outcome_name: str,
        surrogate: Optional[SurrogateModel] = None,
        acquisition: Optional[AcquisitionFunction] = None,
        maximize: bool = True,
        random_seed: Optional[int] = None
    ) -> None:
        self.variables = {v.name: v for v in variables}
        self.outcome_name = outcome_name
        self.surrogate = surrogate or GaussianProcessSurrogate()
        self.acquisition = acquisition or ExpectedImprovement()
        self.maximize = maximize
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.history: list[ExperimentResult] = []
        self.candidates: list[ExperimentalCondition] = []
        self._continuous_vars: list[str] = []
        self._var_order: list[str] = []
        
        self._identify_variable_types()
    
    def _identify_variable_types(self) -> None:
        """Identify continuous variables for optimization."""
        self._continuous_vars = [
            name for name, var in self.variables.items()
            if var.var_type == VariableType.CONTINUOUS
        ]
        self._var_order = list(self.variables.keys())
    
    def suggest_initial_points(
        self,
        n_points: int = 5,
        method: str = "latin_hypercube"
    ) -> list[ExperimentalCondition]:
        """
        Suggest initial experimental conditions.
        
        Methods:
            - random: Random sampling
            - grid: Regular grid
            - latin_hypercube: Latin hypercube sampling
        """
        conditions: list[ExperimentalCondition] = []
        
        if method == "random":
            for i in range(n_points):
                values = self._sample_random()
                conditions.append(ExperimentalCondition(
                    id=f"init_{i}",
                    values=values
                ))
        
        elif method == "grid":
            # Create grid for continuous variables
            if self._continuous_vars:
                n_per_var = int(np.ceil(n_points ** (1 / len(self._continuous_vars))))
                grids = []
                
                for var_name in self._continuous_vars:
                    var = self.variables[var_name]
                    grid = np.linspace(var.lower_bound, var.upper_bound, n_per_var)
                    grids.append(grid)
                
                # Cartesian product
                from itertools import product
                for i, point in enumerate(product(*grids)):
                    if i >= n_points:
                        break
                    values = self._sample_random()  # For non-continuous vars
                    for j, var_name in enumerate(self._continuous_vars):
                        values[var_name] = point[j]
                    conditions.append(ExperimentalCondition(
                        id=f"init_{i}",
                        values=values
                    ))
            else:
                return self.suggest_initial_points(n_points, "random")
        
        elif method == "latin_hypercube":
            # Latin hypercube sampling for continuous vars
            if self._continuous_vars:
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=len(self._continuous_vars), seed=self.random_seed)
                sample = sampler.random(n=n_points)
                
                for i, point in enumerate(sample):
                    values = self._sample_random()  # For non-continuous vars
                    for j, var_name in enumerate(self._continuous_vars):
                        var = self.variables[var_name]
                        values[var_name] = var.lower_bound + point[j] * (
                            var.upper_bound - var.lower_bound
                        )
                    conditions.append(ExperimentalCondition(
                        id=f"init_{i}",
                        values=values
                    ))
            else:
                return self.suggest_initial_points(n_points, "random")
        
        return conditions
    
    def _sample_random(self) -> dict[str, Any]:
        """Sample random values for all variables."""
        values: dict[str, Any] = {}
        
        for name, var in self.variables.items():
            if var.var_type == VariableType.CONTINUOUS:
                values[name] = np.random.uniform(var.lower_bound, var.upper_bound)
            elif var.var_type == VariableType.DISCRETE:
                values[name] = int(np.random.uniform(
                    var.lower_bound or 0,
                    var.upper_bound or 10
                ))
            elif var.var_type == VariableType.CATEGORICAL:
                values[name] = np.random.choice(var.categories or ["default"])
            elif var.var_type == VariableType.BOOLEAN:
                values[name] = bool(np.random.randint(0, 2))
        
        return values
    
    def update_with_result(self, result: ExperimentResult) -> None:
        """Update planner with experimental result."""
        self.history.append(result)
        
        # Update surrogate model
        X = self._conditions_to_array([result])
        y = np.array([result.outcome])
        
        if len(self.history) == 1:
            self.surrogate.fit(X, y)
        else:
            self.surrogate.update(X, y)
    
    def suggest_next_experiment(
        self,
        n_candidates: int = 1000,
        batch_size: int = 1
    ) -> list[ExperimentalCondition]:
        """
        Suggest next experiment(s) using acquisition function.
        
        Args:
            n_candidates: Number of candidate points to evaluate
            batch_size: Number of experiments to suggest
        """
        if len(self.history) == 0:
            return self.suggest_initial_points(batch_size)
        
        # Generate candidate points
        candidates = self._generate_candidates(n_candidates)
        
        # Evaluate acquisition function
        X_candidates = self._conditions_to_array(candidates)
        mu, sigma = self.surrogate.predict(X_candidates, return_std=True)
        
        # Get best observed value
        y_observed = np.array([r.outcome for r in self.history])
        y_best = np.max(y_observed) if self.maximize else np.min(y_observed)
        
        # Compute acquisition scores
        acquisition_values = self.acquisition.compute(mu, sigma, y_best)
        
        if not self.maximize:
            acquisition_values = -acquisition_values
        
        # Select top candidates
        top_indices = np.argsort(acquisition_values)[-batch_size:][::-1]
        
        selected: list[ExperimentalCondition] = []
        for idx in top_indices:
            candidate = candidates[idx]
            candidate.predicted_outcome = float(mu[idx])
            candidate.uncertainty = float(sigma[idx])
            candidate.acquisition_score = float(acquisition_values[idx])
            selected.append(candidate)
        
        return selected
    
    def _generate_candidates(self, n: int) -> list[ExperimentalCondition]:
        """Generate candidate experimental conditions."""
        candidates: list[ExperimentalCondition] = []
        
        for i in range(n):
            values = self._sample_random()
            candidates.append(ExperimentalCondition(
                id=f"cand_{i}",
                values=values
            ))
        
        return candidates
    
    def _conditions_to_array(self, conditions: list[ExperimentResult | ExperimentalCondition]) -> Array:
        """Convert conditions to numpy array for model."""
        # Handle continuous variables only for now
        arrays = []
        
        for cond in conditions:
            row = []
            for var_name in self._continuous_vars:
                value = cond.values.get(var_name, 0)
                row.append(float(value))
            arrays.append(row)
        
        return np.array(arrays)
    
    def get_best_observed(self) -> Optional[ExperimentResult]:
        """Get the best experimental result so far."""
        if not self.history:
            return None
        
        if self.maximize:
            return max(self.history, key=lambda r: r.outcome)
        else:
            return min(self.history, key=lambda r: r.outcome)
    
    def get_convergence_stats(self) -> dict[str, Any]:
        """Get statistics on optimization convergence."""
        if len(self.history) < 2:
            return {"n_experiments": len(self.history)}
        
        outcomes = np.array([r.outcome for r in self.history])
        
        # Track best over time
        if self.maximize:
            best_over_time = np.maximum.accumulate(outcomes)
        else:
            best_over_time = np.minimum.accumulate(outcomes)
        
        # Calculate improvement rate
        improvements = np.diff(best_over_time)
        n_improvements = np.sum(improvements != 0)
        
        return {
            "n_experiments": len(self.history),
            "best_outcome": float(best_over_time[-1]),
            "mean_outcome": float(np.mean(outcomes)),
            "std_outcome": float(np.std(outcomes)),
            "n_improvements": int(n_improvements),
            "improvement_rate": float(n_improvements / len(improvements)),
            "converged": n_improvements == 0 and len(self.history) > 10
        }
    
    def plan_for_hypothesis(
        self,
        hypothesis: Hypothesis,
        n_experiments: int = 10
    ) -> list[ExperimentalCondition]:
        """
        Create an experimental plan to test a specific hypothesis.
        
        This designs experiments that can validate or refute the hypothesis.
        """
        plan: list[ExperimentalCondition] = []
        
        # Parse hypothesis for relevant variables
        # For now, use a generic approach
        
        # Initial experiments to establish baseline
        initial = self.suggest_initial_points(min(3, n_experiments))
        plan.extend(initial)
        
        # Remaining experiments from predictions
        remaining = n_experiments - len(initial)
        
        for prediction in hypothesis.testable_predictions[:remaining]:
            # Create condition targeting the prediction
            values = self._sample_random()
            # Adjust values based on prediction context
            # This is a simplified approach
            
            plan.append(ExperimentalCondition(
                id=f"hypo_{len(plan)}",
                values=values,
                metadata={"prediction": prediction, "hypothesis_id": hypothesis.id}
            ))
        
        return plan[:n_experiments]


class HypothesisTester:
    """
    Designs experiments to test specific hypotheses.
    """
    
    def __init__(self, planner: ExperimentPlanner) -> None:
        self.planner = planner
    
    def design_validation_experiments(
        self,
        hypothesis: Hypothesis,
        n_experiments: int = 5
    ) -> list[ExperimentalCondition]:
        """Design experiments to validate a hypothesis."""
        conditions: list[ExperimentalCondition] = []
        
        # For each prediction, create test conditions
        for i, prediction in enumerate(hypothesis.testable_predictions[:n_experiments]):
            values = self.planner._sample_random()
            
            condition = ExperimentalCondition(
                id=f"val_{hypothesis.id}_{i}",
                values=values,
                metadata={
                    "hypothesis_id": hypothesis.id,
                    "prediction": prediction,
                    "test_type": "validation"
                }
            )
            conditions.append(condition)
        
        # Add negative control
        if len(conditions) < n_experiments:
            values = self.planner._sample_random()
            conditions.append(ExperimentalCondition(
                id=f"val_{hypothesis.id}_ctrl",
                values=values,
                metadata={
                    "hypothesis_id": hypothesis.id,
                    "test_type": "negative_control"
                }
            ))
        
        return conditions
    
    def design_falsification_experiments(
        self,
        hypothesis: Hypothesis,
        n_experiments: int = 3
    ) -> list[ExperimentalCondition]:
        """Design experiments that could falsify the hypothesis."""
        conditions: list[ExperimentalCondition] = []
        
        # Create conditions that test boundary cases
        for i in range(n_experiments):
            values = self.planner._sample_random()
            
            condition = ExperimentalCondition(
                id=f"fals_{hypothesis.id}_{i}",
                values=values,
                metadata={
                    "hypothesis_id": hypothesis.id,
                    "test_type": "falsification",
                    "boundary_test": True
                }
            )
            conditions.append(condition)
        
        return conditions


def demo():
    """Demo experiment planning."""
    # Define experimental variables
    variables = [
        ExperimentalVariable(
            name="temperature",
            var_type=VariableType.CONTINUOUS,
            lower_bound=300,
            upper_bound=800,
            unit="K",
            description="Reaction temperature"
        ),
        ExperimentalVariable(
            name="pressure",
            var_type=VariableType.CONTINUOUS,
            lower_bound=1,
            upper_bound=100,
            unit="atm",
            description="Reaction pressure"
        ),
        ExperimentalVariable(
            name="catalyst",
            var_type=VariableType.CATEGORICAL,
            categories=["Pt", "Pd", "Ni", "Cu"],
            description="Catalyst material"
        ),
    ]
    
    # Create planner
    planner = ExperimentPlanner(
        variables=variables,
        outcome_name="conversion_rate",
        maximize=True,
        random_seed=42
    )
    
    print("=== Initial Experimental Plan ===")
    initial = planner.suggest_initial_points(n_points=5, method="latin_hypercube")
    for cond in initial:
        print(f"  {cond.id}: T={cond.values['temperature']:.1f}K, "
              f"P={cond.values['pressure']:.1f}atm, "
              f"catalyst={cond.values['catalyst']}")
    
    # Simulate some results
    print("\n=== Simulating Results ===")
    for cond in initial:
        # Simulate a noisy response
        outcome = (
            -0.001 * (cond.values['temperature'] - 550)**2 +
            -0.01 * (cond.values['pressure'] - 50)**2 +
            np.random.normal(0, 5)
        )
        
        result = ExperimentResult(
            condition_id=cond.id,
            values=cond.values,
            outcome=outcome,
            outcome_name="conversion_rate"
        )
        planner.update_with_result(result)
        print(f"  {cond.id}: outcome = {outcome:.2f}")
    
    # Get next experiment
    print("\n=== Suggesting Next Experiment ===")
    next_exps = planner.suggest_next_experiment(batch_size=1)
    for exp in next_exps:
        print(f"  {exp.id}: T={exp.values['temperature']:.1f}K, "
              f"P={exp.values['pressure']:.1f}atm")
        print(f"    Predicted: {exp.predicted_outcome:.2f} ± {exp.uncertainty:.2f}")
        print(f"    Acquisition score: {exp.acquisition_score:.4f}")
    
    # Convergence stats
    print("\n=== Convergence Statistics ===")
    stats = planner.get_convergence_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
