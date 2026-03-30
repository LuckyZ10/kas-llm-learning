"""
Decision Engine Module
======================

Intelligent decision-making system for method selection (DFT vs ML vs QMC),
accuracy-cost trade-off optimization, and automatic convergence judgment.

Author: DFT+LAMMPS Automation Team
"""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Protocol,
)
from collections import defaultdict

import numpy as np
from scipy import optimize, stats

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class CalculationMethod(Enum):
    """Available calculation methods."""
    DFT = "dft"
    ML_POTENTIAL = "ml_potential"
    QMC = "qmc"
    SEMI_EMPIRICAL = "semi_empirical"
    FORCE_FIELD = "force_field"
    MULTISCALE = "multiscale"


class DFTFunctional(Enum):
    """DFT exchange-correlation functionals by accuracy level."""
    LDA = "lda"              # Fastest, lowest accuracy
    GGA_PBE = "pbe"          # Standard
    GGA_PBEsol = "pbesol"    # For solids
    META_GGA = "scan"        # Higher accuracy
    HYBRID_PBE0 = "pbe0"     # High accuracy
    HYBRID_HSE06 = "hse06"   # High accuracy, lower cost than PBE0
    GW = "gw"                # Very high accuracy
    RPA = "rpa"              # Highest accuracy, highest cost


class QMCMethod(Enum):
    """Quantum Monte Carlo methods."""
    VMC = "vmc"              # Variational MC
    DMC = "dmc"              # Diffusion MC
    AFQMC = "afqmc"          # Auxiliary-field QMC
    LQMC = "lqmc"            # Lattice QMC


class PropertyType(Enum):
    """Types of material properties to calculate."""
    TOTAL_ENERGY = "total_energy"
    BAND_GAP = "band_gap"
    LATTICE_CONSTANT = "lattice_constant"
    BULK_MODULUS = "bulk_modulus"
    ELASTIC_CONSTANTS = "elastic_constants"
    PHONON_SPECTRUM = "phonon_spectrum"
    FORMATION_ENERGY = "formation_energy"
    BARRIER_ENERGY = "barrier_energy"
    OPTICAL_SPECTRUM = "optical_spectrum"
    MAGNETIC_MOMENT = "magnetic_moment"
    IONIC_CONDUCTIVITY = "ionic_conductivity"
    DIFFUSION_COEFFICIENT = "diffusion_coefficient"
    DEFECT_ENERGY = "defect_energy"


class AccuracyRequirement(Enum):
    """Accuracy requirements for calculations."""
    ROUGH = 0.1              # 10% accuracy - screening
    MODERATE = 0.05          # 5% accuracy - preliminary
    HIGH = 0.01              # 1% accuracy - production
    VERY_HIGH = 0.001        # 0.1% accuracy - benchmark


class ConvergenceCriterion(Enum):
    """Types of convergence criteria."""
    ENERGY = "energy"
    FORCE = "force"
    STRESS = "stress"
    GEOMETRY = "geometry"
    ELECTRONIC = "electronic"
    IONIC = "ionic"
    CUSTOM = "custom"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MethodCharacteristics:
    """Characteristics of a calculation method."""
    method: CalculationMethod
    functional: Optional[str] = None
    
    # Accuracy metrics (relative to experiment, lower is better)
    accuracy_energy: float = 0.01           # eV/atom
    accuracy_forces: float = 0.05           # eV/Angstrom
    accuracy_lattice: float = 0.01          # Angstrom
    accuracy_band_gap: float = 0.2          # eV
    
    # Computational cost (relative to LDA)
    relative_cost_cpu: float = 1.0
    relative_cost_memory: float = 1.0
    relative_cost_time: float = 1.0
    
    # Scaling with system size
    scaling_factor: float = 3.0             # O(N^3) for DFT
    
    # Applicability
    max_atoms: int = 1000
    requires_periodic: bool = True
    supports_magnetic: bool = True
    supports_spin_orbit: bool = False


@dataclass
class CostEstimate:
    """Estimated computational cost."""
    cpu_hours: float = 0.0
    memory_gb: float = 0.0
    wall_time_hours: float = 0.0
    gpu_hours: float = 0.0
    
    # Financial cost (if applicable)
    estimated_cost_usd: float = 0.0
    
    def __add__(self, other: CostEstimate) -> CostEstimate:
        return CostEstimate(
            cpu_hours=self.cpu_hours + other.cpu_hours,
            memory_gb=max(self.memory_gb, other.memory_gb),
            wall_time_hours=max(self.wall_time_hours, other.wall_time_hours),
            gpu_hours=self.gpu_hours + other.gpu_hours,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )
    
    @property
    def total_cost_score(self) -> float:
        """Calculate a single cost score for comparison."""
        return (
            self.cpu_hours * 1.0 +
            self.memory_gb * 0.1 +
            self.gpu_hours * 2.0 +
            self.estimated_cost_usd * 0.01
        )


@dataclass
class AccuracyEstimate:
    """Estimated accuracy for a property."""
    property_type: PropertyType
    expected_rmse: float = 0.0              # Root mean squared error
    expected_mae: float = 0.0               # Mean absolute error
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    confidence_level: float = 0.95
    
    def meets_requirement(self, requirement: AccuracyRequirement) -> bool:
        """Check if accuracy meets the requirement."""
        return self.expected_mae <= requirement.value


@dataclass
class SystemCharacteristics:
    """Characteristics of the system being studied."""
    n_atoms: int = 1
    n_electrons: int = 1
    n_species: int = 1
    is_periodic: bool = True
    is_magnetic: bool = False
    has_spin_orbit: bool = False
    cell_volume: float = 100.0              # Angstrom^3
    density: float = 1.0                    # g/cm^3
    
    # Electronic properties
    band_gap: Optional[float] = None        # eV
    is_metal: bool = False
    is_strongly_correlated: bool = False
    
    # Structural complexity
    max_coordination: int = 8
    disorder_parameter: float = 0.0


@dataclass
class DecisionConfig:
    """Configuration for decision making."""
    # Accuracy requirements
    default_accuracy: AccuracyRequirement = AccuracyRequirement.MODERATE
    property_requirements: Dict[PropertyType, AccuracyRequirement] = field(default_factory=dict)
    
    # Cost constraints
    max_cpu_hours: Optional[float] = None
    max_wall_time_hours: Optional[float] = None
    max_cost_usd: Optional[float] = None
    
    # Method preferences
    prefer_speed: bool = False
    prefer_accuracy: bool = False
    allowed_methods: Optional[List[CalculationMethod]] = None
    forbidden_methods: List[CalculationMethod] = field(default_factory=list)
    
    # Convergence settings
    convergence_tolerance: float = 1e-6
    max_iterations: int = 100
    
    # Adaptive settings
    enable_adaptive_refinement: bool = True
    refinement_threshold: float = 0.5


@dataclass
class ConvergenceState:
    """Current convergence state."""
    criterion: ConvergenceCriterion
    current_value: float = 0.0
    previous_value: float = 0.0
    target_tolerance: float = 1e-6
    iteration: int = 0
    history: List[float] = field(default_factory=list)
    
    @property
    def is_converged(self) -> bool:
        """Check if convergence criterion is met."""
        if not self.history:
            return False
        
        # Check absolute change
        if len(self.history) >= 2:
            change = abs(self.history[-1] - self.history[-2])
            if change < self.target_tolerance:
                return True
        
        # Check trend over last few iterations
        if len(self.history) >= 3:
            recent = self.history[-3:]
            if np.std(recent) < self.target_tolerance * 0.1:
                return True
        
        return False
    
    @property
    def convergence_rate(self) -> float:
        """Calculate convergence rate (0-1, higher is better)."""
        if len(self.history) < 2:
            return 0.0
        
        # Exponential fit to convergence
        x = np.arange(len(self.history))
        y = np.array(self.history)
        
        try:
            # Fit exponential decay
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, _ = optimize.curve_fit(exp_decay, x, y, maxfev=5000)
            # b is the decay rate
            return min(1.0, popt[1] / (popt[1] + 0.01))
        except:
            # Fall back to simple ratio
            if len(y) >= 2 and y[-2] != 0:
                ratio = abs(y[-1] - y[-2]) / abs(y[1] - y[0]) if len(y) > 1 else 1.0
                return max(0.0, 1.0 - ratio)
            return 0.0


@dataclass
class MethodRecommendation:
    """Recommended calculation method with justification."""
    method: CalculationMethod
    functional: Optional[str] = None
    basis_set: Optional[str] = None
    
    estimated_cost: CostEstimate = field(default_factory=CostEstimate)
    estimated_accuracy: Dict[PropertyType, AccuracyEstimate] = field(default_factory=dict)
    
    confidence_score: float = 0.0           # 0-1 confidence in recommendation
    justification: List[str] = field(default_factory=list)
    
    # Alternative methods if recommendation fails
    alternatives: List[CalculationMethod] = field(default_factory=list)


# =============================================================================
# Method Database
# =============================================================================

class MethodDatabase:
    """Database of method characteristics and performance data."""
    
    def __init__(self):
        self._methods: Dict[Tuple[CalculationMethod, Optional[str]], MethodCharacteristics] = {}
        self._performance_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._init_default_methods()
    
    def _init_default_methods(self) -> None:
        """Initialize default method characteristics."""
        # DFT methods
        self._methods[(CalculationMethod.DFT, "lda")] = MethodCharacteristics(
            method=CalculationMethod.DFT,
            functional="lda",
            accuracy_energy=0.5,
            accuracy_forces=0.2,
            accuracy_lattice=0.05,
            accuracy_band_gap=1.0,
            relative_cost_cpu=1.0,
            relative_cost_time=1.0,
            scaling_factor=3.0,
        )
        
        self._methods[(CalculationMethod.DFT, "pbe")] = MethodCharacteristics(
            method=CalculationMethod.DFT,
            functional="pbe",
            accuracy_energy=0.1,
            accuracy_forces=0.05,
            accuracy_lattice=0.02,
            accuracy_band_gap=0.5,
            relative_cost_cpu=1.2,
            relative_cost_time=1.2,
            scaling_factor=3.0,
        )
        
        self._methods[(CalculationMethod.DFT, "scan")] = MethodCharacteristics(
            method=CalculationMethod.DFT,
            functional="scan",
            accuracy_energy=0.05,
            accuracy_forces=0.03,
            accuracy_lattice=0.01,
            accuracy_band_gap=0.3,
            relative_cost_cpu=2.0,
            relative_cost_time=2.5,
            scaling_factor=3.0,
        )
        
        self._methods[(CalculationMethod.DFT, "pbe0")] = MethodCharacteristics(
            method=CalculationMethod.DFT,
            functional="pbe0",
            accuracy_energy=0.03,
            accuracy_forces=0.02,
            accuracy_lattice=0.005,
            accuracy_band_gap=0.15,
            relative_cost_cpu=10.0,
            relative_cost_time=15.0,
            scaling_factor=3.0,
        )
        
        self._methods[(CalculationMethod.DFT, "hse06")] = MethodCharacteristics(
            method=CalculationMethod.DFT,
            functional="hse06",
            accuracy_energy=0.03,
            accuracy_forces=0.02,
            accuracy_lattice=0.005,
            accuracy_band_gap=0.12,
            relative_cost_cpu=5.0,
            relative_cost_time=8.0,
            scaling_factor=3.0,
        )
        
        # ML potentials
        self._methods[(CalculationMethod.ML_POTENTIAL, "mace")] = MethodCharacteristics(
            method=CalculationMethod.ML_POTENTIAL,
            functional="mace",
            accuracy_energy=0.005,
            accuracy_forces=0.05,
            accuracy_lattice=0.01,
            accuracy_band_gap=0.5,
            relative_cost_cpu=0.1,
            relative_cost_time=0.01,
            scaling_factor=1.0,
            max_atoms=100000,
        )
        
        self._methods[(CalculationMethod.ML_POTENTIAL, "chgnet")] = MethodCharacteristics(
            method=CalculationMethod.ML_POTENTIAL,
            functional="chgnet",
            accuracy_energy=0.008,
            accuracy_forces=0.06,
            accuracy_lattice=0.015,
            accuracy_band_gap=0.6,
            relative_cost_cpu=0.05,
            relative_cost_time=0.005,
            scaling_factor=1.0,
            max_atoms=100000,
        )
        
        # QMC
        self._methods[(CalculationMethod.QMC, "dmc")] = MethodCharacteristics(
            method=CalculationMethod.QMC,
            functional="dmc",
            accuracy_energy=0.001,
            accuracy_forces=0.01,
            accuracy_lattice=0.001,
            accuracy_band_gap=0.05,
            relative_cost_cpu=1000.0,
            relative_cost_time=5000.0,
            scaling_factor=3.0,
            max_atoms=100,
        )
    
    def get_method(self, method: CalculationMethod, 
                   functional: Optional[str] = None) -> Optional[MethodCharacteristics]:
        """Get method characteristics."""
        return self._methods.get((method, functional))
    
    def add_method(self, characteristics: MethodCharacteristics) -> None:
        """Add or update method characteristics."""
        key = (characteristics.method, characteristics.functional)
        self._methods[key] = characteristics
    
    def record_performance(self, method: CalculationMethod, 
                          functional: Optional[str],
                          data: Dict[str, Any]) -> None:
        """Record actual performance data."""
        key = f"{method.value}_{functional or 'default'}"
        self._performance_data[key].append({
            'timestamp': data.get('timestamp'),
            'system_size': data.get('n_atoms', 0),
            'actual_time': data.get('actual_time', 0),
            'actual_accuracy': data.get('actual_accuracy', 0),
            'property': data.get('property_type', 'unknown'),
        })
    
    def get_average_performance(self, method: CalculationMethod,
                                functional: Optional[str] = None) -> Dict[str, float]:
        """Get average performance statistics."""
        key = f"{method.value}_{functional or 'default'}"
        data = self._performance_data[key]
        
        if not data:
            return {}
        
        return {
            'avg_time': np.mean([d['actual_time'] for d in data]),
            'std_time': np.std([d['actual_time'] for d in data]),
            'avg_accuracy': np.mean([d['actual_accuracy'] for d in data]),
            'n_samples': len(data),
        }


# =============================================================================
# Cost Estimator
# =============================================================================

class CostEstimator:
    """Estimates computational cost for different methods."""
    
    def __init__(self, method_db: MethodDatabase):
        self.method_db = method_db
        
        # Cost calibration factors
        self._cpu_hour_cost = 0.05            # USD per CPU hour
        self._gpu_hour_cost = 0.5             # USD per GPU hour
    
    def estimate(self, 
                 method: CalculationMethod,
                 system: SystemCharacteristics,
                 functional: Optional[str] = None,
                 n_calculations: int = 1) -> CostEstimate:
        """
        Estimate computational cost for a calculation.
        
        Args:
            method: Calculation method
            system: System characteristics
            functional: Specific functional (for DFT)
            n_calculations: Number of calculations to perform
            
        Returns:
            Cost estimate
        """
        method_char = self.method_db.get_method(method, functional)
        if not method_char:
            logger.warning(f"Unknown method {method}, using default estimate")
            method_char = MethodCharacteristics(method=method)
        
        # Base cost calculation using scaling
        n = system.n_atoms
        scaling = method_char.scaling_factor
        
        # Calculate base operations
        if scaling <= 1.0:
            base_ops = n
        else:
            base_ops = n ** scaling
        
        # Adjust for system complexity
        complexity_factor = (
            1.0 +
            0.2 * (system.n_species - 1) +
            0.3 * int(system.is_magnetic) +
            0.5 * int(system.has_spin_orbit) +
            0.3 * system.disorder_parameter
        )
        
        # Calculate costs
        base_cpu_hours = base_ops * method_char.relative_cost_cpu * complexity_factor / 1e6
        base_time_hours = base_ops * method_char.relative_cost_time * complexity_factor / 1e6
        
        memory_gb = (
            method_char.relative_cost_memory * 
            system.n_atoms * 0.01 * 
            (1 + 0.5 * system.n_electrons / system.n_atoms)
        )
        
        gpu_hours = base_cpu_hours * 0.1 if method == CalculationMethod.ML_POTENTIAL else 0.0
        
        # Apply number of calculations
        total_cpu = base_cpu_hours * n_calculations
        total_time = base_time_hours * n_calculations
        total_gpu = gpu_hours * n_calculations
        
        # Financial cost
        cost_usd = (
            total_cpu * self._cpu_hour_cost +
            total_gpu * self._gpu_hour_cost
        )
        
        return CostEstimate(
            cpu_hours=total_cpu,
            memory_gb=memory_gb,
            wall_time_hours=total_time,
            gpu_hours=total_gpu,
            estimated_cost_usd=cost_usd,
        )
    
    def estimate_workflow(self,
                         steps: List[Tuple[CalculationMethod, SystemCharacteristics, Optional[str]]]
                         ) -> CostEstimate:
        """Estimate cost for a multi-step workflow."""
        total = CostEstimate()
        for method, system, functional in steps:
            total = total + self.estimate(method, system, functional)
        return total


# =============================================================================
# Accuracy Estimator
# =============================================================================

class AccuracyEstimator:
    """Estimates accuracy for different methods and properties."""
    
    def __init__(self, method_db: MethodDatabase):
        self.method_db = method_db
        
        # Accuracy multipliers for different property types
        self._property_multipliers: Dict[PropertyType, Dict[str, float]] = {
            PropertyType.TOTAL_ENERGY: {'energy': 1.0, 'forces': 1.0},
            PropertyType.BAND_GAP: {'energy': 2.0, 'forces': 0.0},
            PropertyType.LATTICE_CONSTANT: {'energy': 0.5, 'forces': 0.0},
            PropertyType.BULK_MODULUS: {'energy': 1.5, 'forces': 0.5},
            PropertyType.ELASTIC_CONSTANTS: {'energy': 1.0, 'forces': 1.0},
            PropertyType.PHONON_SPECTRUM: {'energy': 0.5, 'forces': 2.0},
            PropertyType.FORMATION_ENERGY: {'energy': 1.5, 'forces': 0.0},
            PropertyType.BARRIER_ENERGY: {'energy': 2.0, 'forces': 1.0},
            PropertyType.IONIC_CONDUCTIVITY: {'energy': 0.5, 'forces': 2.0},
        }
    
    def estimate(self,
                 method: CalculationMethod,
                 property_type: PropertyType,
                 system: SystemCharacteristics,
                 functional: Optional[str] = None) -> AccuracyEstimate:
        """
        Estimate accuracy for a specific property calculation.
        
        Args:
            method: Calculation method
            property_type: Type of property to calculate
            system: System characteristics
            functional: Specific functional
            
        Returns:
            Accuracy estimate
        """
        method_char = self.method_db.get_method(method, functional)
        if not method_char:
            return AccuracyEstimate(property_type=property_type, expected_mae=1.0)
        
        # Get multipliers for this property
        multipliers = self._property_multipliers.get(property_type, {'energy': 1.0, 'forces': 1.0})
        
        # Base accuracy
        base_accuracy = method_char.accuracy_energy * multipliers.get('energy', 1.0)
        base_accuracy += method_char.accuracy_forces * multipliers.get('forces', 0.0)
        
        # Adjust for system characteristics
        system_factor = 1.0
        if system.is_strongly_correlated:
            system_factor *= 2.0 if method != CalculationMethod.QMC else 1.2
        if system.has_spin_orbit:
            system_factor *= 1.5 if not method_char.supports_spin_orbit else 1.0
        if system.is_magnetic and not method_char.supports_magnetic:
            system_factor *= 1.3
        
        # Adjust for system size (finite size effects)
        size_factor = 1.0 + 0.01 * (1000 - min(1000, system.n_atoms)) / 100
        
        expected_mae = base_accuracy * system_factor * size_factor
        expected_rmse = expected_mae * 1.2  # RMSE typically higher than MAE
        
        # Confidence interval
        confidence = 0.95
        margin = expected_mae * 0.5
        ci_lower = max(0, expected_mae - margin)
        ci_upper = expected_mae + margin
        
        return AccuracyEstimate(
            property_type=property_type,
            expected_rmse=expected_rmse,
            expected_mae=expected_mae,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=confidence,
        )
    
    def estimate_all_properties(self,
                                method: CalculationMethod,
                                system: SystemCharacteristics,
                                functional: Optional[str] = None
                                ) -> Dict[PropertyType, AccuracyEstimate]:
        """Estimate accuracy for all property types."""
        return {
            prop: self.estimate(method, prop, system, functional)
            for prop in PropertyType
        }


# =============================================================================
# Convergence Monitor
# =============================================================================

class ConvergenceMonitor:
    """Monitors and judges convergence of calculations."""
    
    def __init__(self):
        self._states: Dict[str, ConvergenceState] = {}
        self._convergence_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def register(self, 
                 calculation_id: str,
                 criterion: ConvergenceCriterion,
                 target_tolerance: float = 1e-6) -> ConvergenceState:
        """Register a new calculation for convergence monitoring."""
        state = ConvergenceState(
            criterion=criterion,
            target_tolerance=target_tolerance,
        )
        self._states[calculation_id] = state
        return state
    
    def update(self, calculation_id: str, value: float) -> ConvergenceState:
        """Update convergence state with a new value."""
        if calculation_id not in self._states:
            raise ValueError(f"Unknown calculation: {calculation_id}")
        
        state = self._states[calculation_id]
        state.previous_value = state.current_value
        state.current_value = value
        state.history.append(value)
        state.iteration += 1
        
        # Record history
        self._convergence_history[calculation_id].append({
            'iteration': state.iteration,
            'value': value,
            'timestamp': time.time(),
        })
        
        return state
    
    def check_convergence(self, calculation_id: str) -> Tuple[bool, ConvergenceState]:
        """Check if calculation has converged."""
        state = self._states.get(calculation_id)
        if not state:
            return False, None
        
        return state.is_converged, state
    
    def predict_iterations_to_converge(self, calculation_id: str) -> Optional[int]:
        """Predict remaining iterations needed for convergence."""
        state = self._states.get(calculation_id)
        if not state or len(state.history) < 3:
            return None
        
        # Fit exponential decay model
        x = np.arange(len(state.history))
        y = np.array(state.history)
        
        try:
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, _ = optimize.curve_fit(exp_decay, x, y, maxfev=5000)
            a, b, c = popt
            
            # Find when value approaches asymptote
            if b > 0:
                threshold = c + state.target_tolerance
                remaining = np.log((threshold - c) / a) / (-b) - len(y)
                return max(0, int(remaining))
        except:
            pass
        
        return None
    
    def detect_divergence(self, calculation_id: str,
                         window_size: int = 5) -> bool:
        """Detect if calculation is diverging."""
        state = self._states.get(calculation_id)
        if not state or len(state.history) < window_size:
            return False
        
        recent = state.history[-window_size:]
        
        # Check if values are consistently increasing
        diffs = np.diff(recent)
        if np.all(diffs > 0) and np.mean(diffs) > state.target_tolerance:
            return True
        
        # Check for oscillation with increasing amplitude
        if len(recent) >= 4:
            amplitudes = [abs(recent[i] - recent[i+1]) for i in range(len(recent)-1)]
            if np.all(np.diff(amplitudes) > 0):
                return True
        
        return False
    
    def get_recommendation(self, calculation_id: str) -> Dict[str, Any]:
        """Get recommendations for improving convergence."""
        state = self._states.get(calculation_id)
        if not state:
            return {'error': 'Unknown calculation'}
        
        recommendations = []
        
        if state.is_converged:
            return {'status': 'converged', 'recommendations': []}
        
        # Check convergence rate
        rate = state.convergence_rate
        if rate < 0.1:
            recommendations.append({
                'type': 'slow_convergence',
                'message': 'Convergence is very slow. Consider changing algorithm or initial guess.',
                'priority': 'high',
            })
        elif rate < 0.3:
            recommendations.append({
                'type': 'moderate_slow',
                'message': 'Convergence could be improved. Consider mixing parameters.',
                'priority': 'medium',
            })
        
        # Check for divergence
        if self.detect_divergence(calculation_id):
            recommendations.append({
                'type': 'divergence',
                'message': 'Calculation appears to be diverging. Restart with different settings.',
                'priority': 'critical',
            })
        
        # Predict remaining time
        remaining = self.predict_iterations_to_converge(calculation_id)
        if remaining and remaining > 100:
            recommendations.append({
                'type': 'long_convergence',
                'message': f'Estimated {remaining} iterations remaining. Consider approximations.',
                'priority': 'medium',
            })
        
        return {
            'status': 'running',
            'convergence_rate': rate,
            'iterations': state.iteration,
            'predicted_remaining': remaining,
            'recommendations': recommendations,
        }


# =============================================================================
# Decision Engine
# =============================================================================

class DecisionEngine:
    """
    Main decision engine for method selection and optimization.
    
    Provides intelligent recommendations for calculation methods based on
    accuracy requirements, cost constraints, and system characteristics.
    """
    
    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()
        
        # Components
        self.method_db = MethodDatabase()
        self.cost_estimator = CostEstimator(self.method_db)
        self.accuracy_estimator = AccuracyEstimator(self.method_db)
        self.convergence_monitor = ConvergenceMonitor()
        
        # Decision history for learning
        self._decision_history: List[Dict[str, Any]] = []
    
    def select_method(
        self,
        system: SystemCharacteristics,
        properties: List[PropertyType],
        accuracy_requirement: Optional[AccuracyRequirement] = None,
        cost_budget: Optional[CostEstimate] = None,
    ) -> MethodRecommendation:
        """
        Select the optimal calculation method.
        
        Args:
            system: System characteristics
            properties: Properties to calculate
            accuracy_requirement: Required accuracy level
            cost_budget: Maximum cost budget
            
        Returns:
            Method recommendation with justification
        """
        accuracy_req = accuracy_requirement or self.config.default_accuracy
        
        # Get candidate methods
        candidates = self._get_candidate_methods(system)
        
        # Score each candidate
        scored_methods = []
        for method, functional in candidates:
            score, justification = self._score_method(
                method, functional, system, properties, 
                accuracy_req, cost_budget
            )
            scored_methods.append((score, method, functional, justification))
        
        # Sort by score (higher is better)
        scored_methods.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_methods:
            return MethodRecommendation(
                method=CalculationMethod.DFT,
                justification=["No valid methods found, defaulting to DFT"]
            )
        
        # Get top recommendation
        top_score, top_method, top_functional, top_justification = scored_methods[0]
        
        # Calculate estimates
        estimated_cost = self.cost_estimator.estimate(
            top_method, system, top_functional
        )
        estimated_accuracy = {
            prop: self.accuracy_estimator.estimate(top_method, prop, system, top_functional)
            for prop in properties
        }
        
        # Get alternatives
        alternatives = [
            m for _, m, _, _ in scored_methods[1:3]
        ]
        
        # Confidence score based on data quality
        confidence = self._calculate_confidence(top_method, top_functional, system)
        
        recommendation = MethodRecommendation(
            method=top_method,
            functional=top_functional,
            estimated_cost=estimated_cost,
            estimated_accuracy=estimated_accuracy,
            confidence_score=confidence,
            justification=top_justification,
            alternatives=alternatives,
        )
        
        # Record decision
        self._decision_history.append({
            'timestamp': time.time(),
            'system': system,
            'properties': properties,
            'recommendation': recommendation,
        })
        
        return recommendation
    
    def _get_candidate_methods(
        self, 
        system: SystemCharacteristics
    ) -> List[Tuple[CalculationMethod, Optional[str]]]:
        """Get list of candidate methods for the system."""
        candidates = []
        
        # Define method variants to consider
        method_variants = [
            (CalculationMethod.DFT, "lda"),
            (CalculationMethod.DFT, "pbe"),
            (CalculationMethod.DFT, "scan"),
            (CalculationMethod.DFT, "pbe0"),
            (CalculationMethod.DFT, "hse06"),
            (CalculationMethod.ML_POTENTIAL, "mace"),
            (CalculationMethod.ML_POTENTIAL, "chgnet"),
            (CalculationMethod.QMC, "dmc"),
        ]
        
        for method, functional in method_variants:
            # Check if method is allowed
            if self.config.allowed_methods and method not in self.config.allowed_methods:
                continue
            if method in self.config.forbidden_methods:
                continue
            
            # Check system constraints
            method_char = self.method_db.get_method(method, functional)
            if not method_char:
                continue
            
            if system.n_atoms > method_char.max_atoms:
                continue
            if not system.is_periodic and method_char.requires_periodic:
                continue
            if system.is_magnetic and not method_char.supports_magnetic:
                continue
            if system.has_spin_orbit and not method_char.supports_spin_orbit:
                continue
            
            candidates.append((method, functional))
        
        return candidates
    
    def _score_method(
        self,
        method: CalculationMethod,
        functional: Optional[str],
        system: SystemCharacteristics,
        properties: List[PropertyType],
        accuracy_req: AccuracyRequirement,
        cost_budget: Optional[CostEstimate],
    ) -> Tuple[float, List[str]]:
        """
        Score a method based on multiple criteria.
        
        Returns:
            (score, justification_list)
        """
        score = 1.0
        justification = []
        
        # Estimate accuracy
        accuracies = [
            self.accuracy_estimator.estimate(method, prop, system, functional)
            for prop in properties
        ]
        
        # Check if accuracy meets requirement
        meets_accuracy = all(acc.meets_requirement(accuracy_req) for acc in accuracies)
        if meets_accuracy:
            score += 2.0
            justification.append(f"Meets {accuracy_req.name} accuracy requirement")
        else:
            score -= 1.0
            justification.append("Does not meet accuracy requirement")
        
        # Estimate cost
        cost = self.cost_estimator.estimate(method, system, functional)
        
        # Check cost budget
        if cost_budget:
            if cost.total_cost_score <= cost_budget.total_cost_score * 1.2:
                score += 1.0
                justification.append("Within cost budget")
            else:
                score -= 0.5
                justification.append("Exceeds cost budget")
        
        # Cost efficiency bonus
        avg_accuracy = np.mean([acc.expected_mae for acc in accuracies])
        efficiency = 1.0 / (avg_accuracy * cost.total_cost_score + 1e-6)
        score += np.log1p(efficiency) * 0.5
        
        # System-specific bonuses
        if method == CalculationMethod.ML_POTENTIAL:
            if system.n_atoms > 1000:
                score += 1.0
                justification.append("ML potential suitable for large systems")
        
        if method == CalculationMethod.QMC:
            if system.is_strongly_correlated:
                score += 1.5
                justification.append("QMC recommended for strongly correlated systems")
            else:
                score -= 0.5
                justification.append("QMC may be overkill for non-correlated systems")
        
        if method == CalculationMethod.DFT:
            if functional in ["hse06", "pbe0"]:
                if PropertyType.BAND_GAP in properties:
                    score += 0.5
                    justification.append("Hybrid functional good for band gaps")
        
        # Preference adjustments
        if self.config.prefer_speed:
            if cost.wall_time_hours < 1.0:
                score += 0.5
        
        if self.config.prefer_accuracy:
            if avg_accuracy < 0.01:
                score += 0.5
        
        return score, justification
    
    def _calculate_confidence(
        self,
        method: CalculationMethod,
        functional: Optional[str],
        system: SystemCharacteristics
    ) -> float:
        """Calculate confidence score for the recommendation."""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on available performance data
        perf_data = self.method_db.get_average_performance(method, functional)
        if perf_data:
            n_samples = perf_data.get('n_samples', 0)
            if n_samples > 100:
                confidence += 0.2
            elif n_samples > 10:
                confidence += 0.1
        
        # Adjust for system similarity to known systems
        if system.n_atoms < 100:
            confidence += 0.05  # More confidence for small systems
        
        return min(1.0, confidence)
    
    def optimize_accuracy_cost_tradeoff(
        self,
        system: SystemCharacteristics,
        properties: List[PropertyType],
        accuracy_targets: Dict[PropertyType, float],
        cost_constraint: Optional[CostEstimate] = None,
    ) -> List[MethodRecommendation]:
        """
        Find Pareto-optimal methods balancing accuracy and cost.
        
        Args:
            system: System characteristics
            properties: Properties to calculate
            accuracy_targets: Target accuracy for each property
            cost_constraint: Optional cost constraint
            
        Returns:
            List of Pareto-optimal recommendations
        """
        candidates = self._get_candidate_methods(system)
        recommendations = []
        
        for method, functional in candidates:
            cost = self.cost_estimator.estimate(method, system, functional)
            
            if cost_constraint and cost.total_cost_score > cost_constraint.total_cost_score:
                continue
            
            accuracies = {
                prop: self.accuracy_estimator.estimate(method, prop, system, functional)
                for prop in properties
            }
            
            # Check if meets all targets
            meets_targets = all(
                accuracies[prop].expected_mae <= target
                for prop, target in accuracy_targets.items()
            )
            
            if meets_targets:
                rec = MethodRecommendation(
                    method=method,
                    functional=functional,
                    estimated_cost=cost,
                    estimated_accuracy=accuracies,
                )
                recommendations.append(rec)
        
        # Filter to Pareto frontier
        pareto_optimal = []
        for i, rec1 in enumerate(recommendations):
            dominated = False
            for j, rec2 in enumerate(recommendations):
                if i != j:
                    # Check if rec2 dominates rec1
                    if (rec2.estimated_cost.total_cost_score <= rec1.estimated_cost.total_cost_score and
                        all(rec2.estimated_accuracy[p].expected_mae <= rec1.estimated_accuracy[p].expected_mae
                            for p in properties) and
                        (rec2.estimated_cost.total_cost_score < rec1.estimated_cost.total_cost_score or
                         any(rec2.estimated_accuracy[p].expected_mae < rec1.estimated_accuracy[p].expected_mae
                             for p in properties))):
                        dominated = True
                        break
            
            if not dominated:
                pareto_optimal.append(rec1)
        
        # Sort by cost
        pareto_optimal.sort(key=lambda r: r.estimated_cost.total_cost_score)
        
        return pareto_optimal
    
    def judge_convergence(
        self,
        calculation_id: str,
        current_value: float
    ) -> Dict[str, Any]:
        """
        Judge whether a calculation has converged.
        
        Args:
            calculation_id: Unique calculation identifier
            current_value: Current convergence metric value
            
        Returns:
            Dictionary with convergence status and recommendations
        """
        state = self.convergence_monitor.update(calculation_id, current_value)
        is_converged, _ = self.convergence_monitor.check_convergence(calculation_id)
        
        result = {
            'calculation_id': calculation_id,
            'is_converged': is_converged,
            'current_value': current_value,
            'iteration': state.iteration,
        }
        
        if is_converged:
            result['message'] = f"Calculation converged after {state.iteration} iterations"
            result['final_value'] = current_value
        else:
            # Get recommendations
            recommendation = self.convergence_monitor.get_recommendation(calculation_id)
            result.update(recommendation)
        
        return result
    
    def suggest_adaptive_refinement(
        self,
        current_method: CalculationMethod,
        current_results: Dict[str, Any],
        target_accuracy: AccuracyRequirement,
    ) -> Optional[MethodRecommendation]:
        """
        Suggest method refinement if current method is insufficient.
        
        Args:
            current_method: Currently used method
            current_results: Results from current calculation
            target_accuracy: Target accuracy requirement
            
        Returns:
            Recommendation for refined method, or None if current is sufficient
        """
        # Check if current accuracy is sufficient
        current_accuracy = current_results.get('accuracy', 0.1)
        
        if current_accuracy <= target_accuracy.value:
            return None  # Current method is sufficient
        
        # Suggest higher-level method
        refinement_map = {
            CalculationMethod.FORCE_FIELD: CalculationMethod.ML_POTENTIAL,
            CalculationMethod.SEMI_EMPIRICAL: CalculationMethod.DFT,
            CalculationMethod.ML_POTENTIAL: CalculationMethod.DFT,
            CalculationMethod.DFT: CalculationMethod.QMC,
        }
        
        suggested_method = refinement_map.get(current_method)
        if not suggested_method:
            return None
        
        # Create recommendation
        return MethodRecommendation(
            method=suggested_method,
            justification=[
                f"Current method {current_method.value} accuracy ({current_accuracy:.4f}) "
                f"does not meet target ({target_accuracy.value:.4f})",
                f"Suggested refinement to {suggested_method.value}",
            ]
        )
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about past decisions."""
        if not self._decision_history:
            return {'error': 'No decision history available'}
        
        method_counts = defaultdict(int)
        accuracy_meeting = 0
        
        for decision in self._decision_history:
            rec = decision['recommendation']
            method_counts[rec.method.value] += 1
            
            # Check if accuracy requirements were met (would need actual results)
            # This is a placeholder
        
        return {
            'total_decisions': len(self._decision_history),
            'method_distribution': dict(method_counts),
            'most_common_method': max(method_counts.items(), key=lambda x: x[1]),
        }


# =============================================================================
# Utility Functions
# =============================================================================

import time


def quick_method_select(
    n_atoms: int,
    property_type: str,
    required_accuracy: str = "moderate",
    max_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Quick method selection without full system setup.
    
    Args:
        n_atoms: Number of atoms in system
        property_type: Type of property to calculate
        required_accuracy: Required accuracy level
        max_hours: Maximum wall time in hours
        
    Returns:
        Recommendation dictionary
    """
    engine = DecisionEngine()
    
    system = SystemCharacteristics(n_atoms=n_atoms)
    
    try:
        prop = PropertyType(property_type)
    except ValueError:
        prop = PropertyType.TOTAL_ENERGY
    
    try:
        acc_req = AccuracyRequirement[required_accuracy.upper()]
    except KeyError:
        acc_req = AccuracyRequirement.MODERATE
    
    cost_budget = None
    if max_hours:
        cost_budget = CostEstimate(wall_time_hours=max_hours)
    
    recommendation = engine.select_method(
        system=system,
        properties=[prop],
        accuracy_requirement=acc_req,
        cost_budget=cost_budget,
    )
    
    return {
        'method': recommendation.method.value,
        'functional': recommendation.functional,
        'estimated_cost_hours': recommendation.estimated_cost.wall_time_hours,
        'estimated_accuracy': recommendation.estimated_accuracy.get(prop, AccuracyEstimate(prop)).expected_mae,
        'confidence': recommendation.confidence_score,
        'justification': recommendation.justification,
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create decision engine
    engine = DecisionEngine()
    
    # Define system
    system = SystemCharacteristics(
        n_atoms=64,
        n_electrons=512,
        n_species=3,
        is_periodic=True,
        is_magnetic=False,
        band_gap=2.5,
        is_strongly_correlated=False,
    )
    
    # Select method for band gap calculation
    recommendation = engine.select_method(
        system=system,
        properties=[PropertyType.BAND_GAP, PropertyType.TOTAL_ENERGY],
        accuracy_requirement=AccuracyRequirement.HIGH,
    )
    
    print(f"\nRecommended method: {recommendation.method.value}")
    print(f"Functional: {recommendation.functional}")
    print(f"Estimated cost: {recommendation.estimated_cost.wall_time_hours:.2f} hours")
    print(f"Confidence: {recommendation.confidence_score:.2f}")
    print(f"Justification:")
    for j in recommendation.justification:
        print(f"  - {j}")
    
    # Test convergence monitoring
    calc_id = "test_calculation"
    engine.convergence_monitor.register(calc_id, ConvergenceCriterion.ENERGY, 1e-5)
    
    # Simulate convergence
    for i in range(20):
        value = 1.0 * np.exp(-0.3 * i) + 1e-6 * np.random.random()
        result = engine.judge_convergence(calc_id, value)
        
        if result['is_converged']:
            print(f"\nConverged at iteration {i}")
            break
    
    # Quick select example
    print("\n" + "="*50)
    print("Quick method selection:")
    quick_rec = quick_method_select(
        n_atoms=100,
        property_type="band_gap",
        required_accuracy="high",
        max_hours=24,
    )
    print(quick_rec)
