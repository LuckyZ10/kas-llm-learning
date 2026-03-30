"""
Real-time Simulation Module for Materials Science
材料科学实时模拟模块

This module provides real-time simulation capabilities:
- Reduced Order Models (ROM)
- Proper Orthogonal Decomposition (POD)
- Dynamic Mode Decomposition (DMD)
- Neural Network-based ROM (Autoencoder, DeepONet)
- Online learning and adaptation
- Edge computing deployment

Modules:
    rom_simulator: Core ROM implementations
    online_learning: Online adaptation algorithms
    edge_deployment: Edge computing optimization

Example:
    >>> from dftlammps.realtime_sim import RealtimeSimulator, ROMMethod
    >>> simulator = RealtimeSimulator(state_dim=1000, rom_method=ROMMethod.POD, n_modes=10)
    >>> simulator.train_rom(training_data)
    >>> results = simulator.run_simulation(duration=10.0)
"""

from .rom_simulator import (
    RealtimeSimulator,
    ReducedOrderModel,
    ProperOrthogonalDecomposition,
    DynamicModeDecomposition,
    AutoencoderROM,
    DeepONetROM,
    OnlineLearner,
    EdgeDeployment,
    ROMMethod,
    ROMState
)

__version__ = "1.0.0"

__all__ = [
    # Main classes
    'RealtimeSimulator',
    'ReducedOrderModel',
    'OnlineLearner',
    'EdgeDeployment',
    
    # ROM methods
    'ProperOrthogonalDecomposition',
    'DynamicModeDecomposition',
    'AutoencoderROM',
    'DeepONetROM',
    
    # Enums and dataclasses
    'ROMMethod',
    'ROMState'
]
