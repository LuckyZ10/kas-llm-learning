"""
World Model Module
==================

Material world modeling with predictive dynamics and imagination.

Components:
- MaterialWorldModel: Core world model for state transition prediction
- ImaginationEngine: Counterfactual simulation and creative design
- ModelPredictiveControl: Optimal control and constraint satisfaction
"""

from .material_world_model import (
    MaterialWorldModel,
    MaterialState,
    MaterialAction,
    ActionType,
    StateType,
    Transition,
    WorldModelConfig,
    DynamicsModel,
    EnsembleDynamicsModel,
    RecurrentDynamicsModel,
    MultiFidelityWorldModel,
    create_synthetic_transitions
)

from .imagination_engine import (
    ImaginationEngine,
    CounterfactualSimulator,
    HypotheticalScenarioGenerator,
    CreativeDesignSpace,
    CounterfactualQuery,
    HypotheticalScenario,
    ImaginedOutcome,
    ScenarioType,
    DesignStrategy,
    MaterialImaginationCases
)

from .model_predictive_control import (
    ModelPredictiveController,
    MultiObjectiveMPC,
    AdaptiveMPC,
    SynthesisPathPlanner,
    RealTimeSynthesisController,
    CrossEntropyOptimizer,
    MPPIOptimizer,
    GradientMPCOptimizer,
    GeneticOptimizer,
    ConstraintHandler,
    TrajectoryCost,
    ControlConstraint,
    MPCConfig,
    OptimizationMethod,
    ConstraintType
)

__all__ = [
    # World Model
    'MaterialWorldModel',
    'MaterialState',
    'MaterialAction',
    'ActionType',
    'StateType',
    'Transition',
    'WorldModelConfig',
    'DynamicsModel',
    'EnsembleDynamicsModel',
    'RecurrentDynamicsModel',
    'MultiFidelityWorldModel',
    'create_synthetic_transitions',
    
    # Imagination Engine
    'ImaginationEngine',
    'CounterfactualSimulator',
    'HypotheticalScenarioGenerator',
    'CreativeDesignSpace',
    'CounterfactualQuery',
    'HypotheticalScenario',
    'ImaginedOutcome',
    'ScenarioType',
    'DesignStrategy',
    'MaterialImaginationCases',
    
    # MPC
    'ModelPredictiveController',
    'MultiObjectiveMPC',
    'AdaptiveMPC',
    'SynthesisPathPlanner',
    'RealTimeSynthesisController',
    'CrossEntropyOptimizer',
    'MPPIOptimizer',
    'GradientMPCOptimizer',
    'GeneticOptimizer',
    'ConstraintHandler',
    'TrajectoryCost',
    'ControlConstraint',
    'MPCConfig',
    'OptimizationMethod',
    'ConstraintType'
]

__version__ = '1.0.0'
