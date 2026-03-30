"""
DFTLammps Adaptive Multiscale Module

This module provides automatic multiscale coupling and intelligent resolution
selection for molecular dynamics simulations.

Key Components:
- resolution_adapter: Automatic DFT/ML potential switching with cost-accuracy tradeoffs
- error_estimators: ML potential confidence assessment and uncertainty quantification
- coupling_controller: QM/MM boundary optimization and information passing

Usage:
    from dftlammps.adaptive_multiscale import (
        AdaptiveResolutionManager,
        EnsembleErrorEstimator,
        CouplingController
    )
    
    # Create manager
    manager = AdaptiveResolutionManager(config)
    
    # Run adaptive step
    result = manager.step(positions, forces_ml, uncertainty)
"""

from .resolution_adapter import (
    ResolutionLevel,
    ResolutionState,
    ResolutionSwitcher,
    CostAccuracyTradeoff,
    AdaptiveResolutionManager,
    create_default_manager,
    ComputationalMetrics,
    AccuracyMetrics,
    SwitchTrigger
)

from .error_estimators import (
    BaseErrorEstimator,
    UncertaintyEstimate,
    EnsembleErrorEstimator,
    GradientSensitivityEstimator,
    BayesianNNEstimator,
    AdaptiveSamplingTrigger,
    CompositeErrorEstimator,
    create_default_estimator
)

from .coupling_controller import (
    QMRegion,
    BoundaryType,
    CouplingScheme,
    BoundaryMetrics,
    BoundaryOptimizer,
    AdaptiveBoundaryOptimizer,
    InformationCoordinator,
    LoadBalancer,
    CouplingController
)

__version__ = "1.0.0"
__author__ = "DFTLammps Team"

__all__ = [
    # Resolution Adapter
    'ResolutionLevel',
    'ResolutionState',
    'ResolutionSwitcher',
    'CostAccuracyTradeoff',
    'AdaptiveResolutionManager',
    'create_default_manager',
    'ComputationalMetrics',
    'AccuracyMetrics',
    'SwitchTrigger',
    
    # Error Estimators
    'BaseErrorEstimator',
    'UncertaintyEstimate',
    'EnsembleErrorEstimator',
    'GradientSensitivityEstimator',
    'BayesianNNEstimator',
    'AdaptiveSamplingTrigger',
    'CompositeErrorEstimator',
    'create_default_estimator',
    
    # Coupling Controller
    'QMRegion',
    'BoundaryType',
    'CouplingScheme',
    'BoundaryMetrics',
    'BoundaryOptimizer',
    'AdaptiveBoundaryOptimizer',
    'InformationCoordinator',
    'LoadBalancer',
    'CouplingController',
]
