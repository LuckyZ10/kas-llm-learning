"""
Symbolic Regression Engine

Implementation of symbolic regression for discovering physical laws
from data. Supports multiple backends: PySR, gplearn, and custom algorithms.

Reference:
    - Cranmer et al., "Discovering Symbolic Models from Deep Learning with 
      Inductive Biases", NeurIPS 2020 (PySR)
    - Schmidt & Lipson, "Distilling Free-Form Natural Laws from Experimental 
      Data", Science 2009 (Eureqa)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Union, Any
from abc import ABC, abstractmethod
import warnings


class SymbolicExpression:
    """
    Represents a discovered symbolic expression.
    
    Attributes:
        expression: String representation of the expression
        sympy_expr: SymPy expression object
        variables: List of variable names
        parameters: Fitted parameters
        score: Quality score (lower is better)
        complexity: Expression complexity
    """
    
    def __init__(
        self,
        expression: str,
        sympy_expr=None,
        variables: Optional[List[str]] = None,
        parameters: Optional[Dict[str, float]] = None,
        score: Optional[float] = None,
        complexity: Optional[int] = None,
        mse: Optional[float] = None
    ):
        self.expression = expression
        self.sympy_expr = sympy_expr
        self.variables = variables or []
        self.parameters = parameters or {}
        self.score = score
        self.complexity = complexity
        self.mse = mse
    
    def __repr__(self) -> str:
        return f"SymbolicExpression({self.expression}, score={self.score:.4f})"
    
    def evaluate(self, **kwargs) -> np.ndarray:
        """
        Evaluate expression with given variable values.
        
        Args:
            **kwargs: Variable name -> value mapping
            
        Returns:
            Evaluated expression values
        """
        if self.sympy_expr is not None:
            try:
                import sympy as sp
                subs = {sp.Symbol(var): val for var, val in kwargs.items()}
                result = self.sympy_expr.evalf(subs=subs)
                return np.array(result, dtype=float)
            except:
                pass
        
        # Fallback: evaluate string expression
        local_dict = {**kwargs, **np.__dict__}
        return eval(self.expression, {"__builtins__": {}}, local_dict)


class SymbolicRegressionBackend(ABC):
    """Abstract base class for symbolic regression backends."""
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> List[SymbolicExpression]:
        """Fit symbolic regression and return candidate expressions."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best expression."""
        pass


class PySRBackend(SymbolicRegressionBackend):
    """
    PySR backend for symbolic regression.
    
    PySR uses evolutionary algorithms and combines genetic programming
    with machine learning techniques.
    
    Reference: https://github.com/MilesCranmer/PySR
    """
    
    def __init__(
        self,
        niterations: int = 100,
        population_size: int = 50,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        maxsize: int = 20,
        complexity_of_operators: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, int]] = None,
        parsimony: float = 0.0032,
        **kwargs
    ):
        """
        Initialize PySR backend.
        
        Args:
            niterations: Number of iterations
            population_size: Population size for genetic programming
            binary_operators: Binary operators (e.g., ['+', '-', '*', '/'])
            unary_operators: Unary operators (e.g., ['sin', 'cos', 'exp', 'log'])
            maxsize: Maximum expression size
            complexity_of_operators: Complexity weights for operators
            constraints: Operator constraints (e.g., {'exp': 4})
            parsimony: Parsimony coefficient for complexity penalty
            **kwargs: Additional PySR arguments
        """
        self.niterations = niterations
        self.population_size = population_size
        self.binary_operators = binary_operators or ['+', '-', '*', '/']
        self.unary_operators = unary_operators or ['sin', 'cos', 'exp', 'log', 'sqrt']
        self.maxsize = maxsize
        self.complexity_of_operators = complexity_of_operators
        self.constraints = constraints
        self.parsimony = parsimony
        self.kwargs = kwargs
        
        self.model = None
        self.expressions = []
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> List[SymbolicExpression]:
        """
        Fit symbolic regression using PySR.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            variable_names: Names of input variables
            
        Returns:
            List of discovered expressions
        """
        try:
            from pysr import PySRRegressor
        except ImportError:
            raise ImportError(
                "PySR not installed. Install with: pip install pysr"
            )
        
        # Create model
        self.model = PySRRegressor(
            niterations=self.niterations,
            population_size=self.population_size,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            maxsize=self.maxsize,
            complexity_of_operators=self.complexity_of_operators,
            constraints=self.constraints,
            parsimony=self.parsimony,
            **self.kwargs
        )
        
        # Fit
        self.model.fit(X, y)
        
        # Extract expressions
        self.expressions = self._extract_expressions(variable_names)
        
        return self.expressions
    
    def _extract_expressions(
        self,
        variable_names: Optional[List[str]] = None
    ) -> List[SymbolicExpression]:
        """Extract expressions from PySR model."""
        expressions = []
        
        if hasattr(self.model, 'equations_'):
            for _, row in self.model.equations_.iterrows():
                expr = SymbolicExpression(
                    expression=str(row['equation']),
                    sympy_expr=row.get('sympy_format'),
                    variables=variable_names,
                    complexity=row.get('complexity'),
                    score=row.get('loss'),
                    mse=row.get('mse')
                )
                expressions.append(expr)
        
        return expressions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best expression."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X)


class GPLearnBackend(SymbolicRegressionBackend):
    """
    gplearn backend for symbolic regression.
    
    Uses genetic programming with scikit-learn compatible API.
    
    Reference: https://github.com/trevorstephens/gplearn
    """
    
    def __init__(
        self,
        population_size: int = 1000,
        generations: int = 20,
        tournament_size: int = 20,
        stopping_criteria: float = 0.0,
        const_range: Tuple[float, float] = (-1.0, 1.0),
        init_depth: Tuple[int, int] = (2, 6),
        init_method: str = 'half and half',
        function_set: Optional[List[str]] = None,
        metric: str = 'mean absolute error',
        parsimony_coefficient: float = 0.01,
        p_crossover: float = 0.9,
        p_subtree_mutation: float = 0.01,
        p_hoist_mutation: float = 0.01,
        p_point_mutation: float = 0.01,
        **kwargs
    ):
        """Initialize gplearn backend."""
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set or [
            'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv',
            'max', 'min', 'sin', 'cos'
        ]
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.kwargs = kwargs
        
        self.model = None
        self.expressions = []
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> List[SymbolicExpression]:
        """Fit symbolic regression using gplearn."""
        try:
            from gplearn.genetic import SymbolicRegressor
        except ImportError:
            raise ImportError(
                "gplearn not installed. Install with: pip install gplearn"
            )
        
        # Create model
        self.model = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=self.tournament_size,
            stopping_criteria=self.stopping_criteria,
            const_range=self.const_range,
            init_depth=self.init_depth,
            init_method=self.init_method,
            function_set=self.function_set,
            metric=self.metric,
            parsimony_coefficient=self.parsimony_coefficient,
            p_crossover=self.p_crossover,
            p_subtree_mutation=self.p_subtree_mutation,
            p_hoist_mutation=self.p_hoist_mutation,
            p_point_mutation=self.p_point_mutation,
            **self.kwargs
        )
        
        # Fit
        self.model.fit(X, y)
        
        # Extract expression
        expr_str = str(self.model._program)
        
        # Convert to readable form
        if variable_names:
            for i, name in enumerate(variable_names):
                expr_str = expr_str.replace(f'X{i}', name)
        
        expr = SymbolicExpression(
            expression=expr_str,
            variables=variable_names,
            complexity=self.model._program.length_
        )
        
        self.expressions = [expr]
        return self.expressions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best expression."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X)


class AIFeynmanBackend(SymbolicRegressionBackend):
    """
    AI Feynman backend for symbolic regression.
    
    Uses physics-inspired approach combining neural networks with
    symbolic techniques.
    
    Reference: Udrescu & Tegmark, "AI Feynman: A physics-inspired method 
    for symbolic regression", Science Advances 2020
    """
    
    def __init__(
        self,
        BF_try_time: float = 60.0,
        polyfit_deg: int = 4,
        NN_epochs: int = 4000,
        **kwargs
    ):
        """Initialize AI Feynman backend."""
        self.BF_try_time = BF_try_time
        self.polyfit_deg = polyfit_deg
        self.NN_epochs = NN_epochs
        self.kwargs = kwargs
        
        self.model = None
        self.expressions = []
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> List[SymbolicExpression]:
        """Fit symbolic regression using AI Feynman."""
        try:
            from aifeynman import run_aifeynman
        except ImportError:
            raise ImportError(
                "aifeynman not installed. Install with: pip install aifeynman"
            )
        
        # Save data to file (AI Feynman requires file input)
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            data_path = f.name
            for i in range(len(y)):
                row = list(X[i]) + [y[i]]
                f.write(' '.join(map(str, row)) + '\n')
        
        try:
            # Run AI Feynman
            run_aifeynman(
                pathdir="",
                filename=data_path,
                BF_try_time=self.BF_try_time,
                polyfit_deg=self.polyfit_deg,
                NN_epochs=self.NN_epochs,
                **self.kwargs
            )
            
            # Extract results
            # This would parse the output files
            expressions = []
            
        finally:
            os.unlink(data_path)
        
        return expressions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best expression."""
        raise NotImplementedError("AI Feynman prediction not implemented")


class SymbolicRegressionEngine:
    """
    Unified interface for symbolic regression.
    
    Supports multiple backends and provides cross-validation,
    ensembling, and result validation.
    """
    
    def __init__(
        self,
        backend: str = 'pysr',
        backend_kwargs: Optional[Dict[str, Any]] = None,
        ensemble: bool = False,
        ensemble_backends: Optional[List[str]] = None,
        cv_folds: int = 5
    ):
        """
        Initialize symbolic regression engine.
        
        Args:
            backend: Backend to use ('pysr', 'gplearn', 'aifeynman')
            backend_kwargs: Keyword arguments for backend
            ensemble: Whether to ensemble multiple backends
            ensemble_backends: List of backends for ensemble
            cv_folds: Number of cross-validation folds
        """
        self.backend_name = backend
        self.backend_kwargs = backend_kwargs or {}
        self.ensemble = ensemble
        self.ensemble_backends = ensemble_backends or ['pysr', 'gplearn']
        self.cv_folds = cv_folds
        
        self.backend = None
        self.ensemble_models = []
        self.best_expression = None
        self.expressions = []
        
        if not ensemble:
            self.backend = self._create_backend(backend, self.backend_kwargs)
    
    def _create_backend(
        self,
        name: str,
        kwargs: Dict[str, Any]
    ) -> SymbolicRegressionBackend:
        """Create backend instance."""
        if name == 'pysr':
            return PySRBackend(**kwargs)
        elif name == 'gplearn':
            return GPLearnBackend(**kwargs)
        elif name == 'aifeynman':
            return AIFeynmanBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {name}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
        physics_constraints: Optional[Dict[str, Any]] = None
    ) -> SymbolicExpression:
        """
        Fit symbolic regression.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Target values [n_samples]
            variable_names: Names of input variables
            physics_constraints: Physics-based constraints
            
        Returns:
            Best symbolic expression
        """
        if self.ensemble:
            return self._fit_ensemble(X, y, variable_names, physics_constraints)
        else:
            expressions = self.backend.fit(X, y, variable_names)
            self.expressions = expressions
            if expressions:
                self.best_expression = min(expressions, key=lambda e: e.score or float('inf'))
            return self.best_expression
    
    def _fit_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        variable_names: Optional[List[str]] = None,
        physics_constraints: Optional[Dict[str, Any]] = None
    ) -> SymbolicExpression:
        """Fit ensemble of symbolic regression models."""
        all_expressions = []
        
        for backend_name in self.ensemble_backends:
            try:
                backend = self._create_backend(backend_name, {})
                expressions = backend.fit(X, y, variable_names)
                all_expressions.extend(expressions)
                self.ensemble_models.append(backend)
            except Exception as e:
                warnings.warn(f"Backend {backend_name} failed: {e}")
        
        # Select best expression
        if all_expressions:
            self.expressions = all_expressions
            self.best_expression = min(
                all_expressions,
                key=lambda e: e.score or float('inf')
            )
        
        return self.best_expression
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best expression."""
        if self.best_expression is None:
            raise ValueError("Model not fitted")
        
        if self.ensemble and self.ensemble_models:
            # Average predictions from ensemble
            predictions = []
            for model in self.ensemble_models:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except:
                    pass
            return np.mean(predictions, axis=0)
        else:
            return self.backend.predict(X)
    
    def get_equations(self, n: int = 10) -> List[SymbolicExpression]:
        """Get top n equations."""
        sorted_exprs = sorted(
            self.expressions,
            key=lambda e: e.score or float('inf')
        )
        return sorted_exprs[:n]
    
    def validate_with_physics(
        self,
        expression: SymbolicExpression,
        physics_tests: List[Callable]
    ) -> Dict[str, bool]:
        """
        Validate expression against physics constraints.
        
        Args:
            expression: Symbolic expression to validate
            physics_tests: List of physics test functions
            
        Returns:
            Dictionary of test results
        """
        results = {}
        for test in physics_tests:
            try:
                results[test.__name__] = test(expression)
            except Exception as e:
                results[test.__name__] = False
        return results
    
    def to_latex(self, expression: Optional[SymbolicExpression] = None) -> str:
        """Convert expression to LaTeX."""
        expr = expression or self.best_expression
        
        if expr is None:
            return ""
        
        if expr.sympy_expr is not None:
            try:
                import sympy as sp
                return sp.latex(expr.sympy_expr)
            except:
                pass
        
        return expr.expression
    
    def to_python_function(
        self,
        expression: Optional[SymbolicExpression] = None
    ) -> Callable:
        """
        Convert expression to Python function.
        
        Args:
            expression: Symbolic expression
            
        Returns:
            Python function
        """
        expr = expression or self.best_expression
        
        if expr is None:
            raise ValueError("No expression available")
        
        def func(**kwargs):
            return expr.evaluate(**kwargs)
        
        return func
    
    def discover_conservation_law(
        self,
        trajectories: List[np.ndarray],
        variables: List[str],
        candidate_terms: Optional[List[str]] = None
    ) -> SymbolicExpression:
        """
        Discover conservation laws from trajectory data.
        
        Args:
            trajectories: List of trajectory arrays
            variables: Variable names
            candidate_terms: Candidate terms for conservation law
            
        Returns:
            Discovered conservation law
        """
        # Compute time derivatives
        # Look for quantities that are conserved (dC/dt ≈ 0)
        
        conserved_candidates = []
        
        for traj in trajectories:
            # Compute various combinations
            # This is a simplified version - full implementation would
            # search over expression trees
            
            # Example: check if kinetic + potential is conserved
            if 'v' in variables and 'x' in variables:
                # E = 0.5*v^2 + V(x)
                pass
        
        # Fit symbolic regression to conserved quantity
        # Return best expression
        
        return self.best_expression
