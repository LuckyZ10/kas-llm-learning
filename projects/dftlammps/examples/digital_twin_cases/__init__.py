"""
Digital Twin Application Cases
数字孪生应用案例

This module provides example applications of the digital twin system:
- Battery health monitoring and lifetime prediction
- Structural material fatigue and crack growth prediction
- Catalyst deactivation monitoring and regeneration optimization

Each case demonstrates the integration of:
- Core digital twin functionality
- Sensor fusion and data processing
- Predictive modeling and maintenance
- Real-time simulation capabilities
"""

from .battery_health_twin import BatteryDigitalTwin, BatteryState, simulate_battery_degradation
from .structural_lifetime_prediction import (
    StructuralDigitalTwin,
    StructuralState,
    FatigueLifePredictor,
    CrackGrowthPredictor,
    simulate_fatigue_loading
)
from .catalyst_deactivation_monitoring import (
    CatalystDigitalTwin,
    CatalystState,
    DeactivationKinetics,
    MechanismIdentifier,
    RegenerationOptimizer,
    simulate_catalyst_deactivation
)

__all__ = [
    # Battery case
    'BatteryDigitalTwin',
    'BatteryState',
    'simulate_battery_degradation',
    
    # Structural case
    'StructuralDigitalTwin',
    'StructuralState',
    'FatigueLifePredictor',
    'CrackGrowthPredictor',
    'simulate_fatigue_loading',
    
    # Catalyst case
    'CatalystDigitalTwin',
    'CatalystState',
    'DeactivationKinetics',
    'MechanismIdentifier',
    'RegenerationOptimizer',
    'simulate_catalyst_deactivation'
]
