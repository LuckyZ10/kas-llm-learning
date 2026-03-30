"""
Lab Automation Module for DFT-LAMMPS Research Platform

This module provides comprehensive laboratory automation capabilities including:
- Equipment abstraction layer for robotics and instruments
- Synthesis planners for powder and thin-film materials
- Characterization data parsers (XRD, SEM, electrochemical)
- Closed-loop feedback control systems
- LIMS integration
- ROS2 interface for robot control

Author: DFT-LAMMPS Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DFT-LAMMPS Team"

# Core imports
from .equipment import (
    BaseEquipment,
    RobotArm,
    SynthesisEquipment,
    CharacterizationInstrument,
    EquipmentManager
)

from .synthesis import (
    SynthesisPlanner,
    PowderSynthesizer,
    ThinFilmDepositor,
    SynthesisRecipe,
    ProcessParameters
)

from .characterization import (
    XRDParser,
    SEMParser,
    ElectrochemicalParser,
    DataAggregator,
    CharacterizationWorkflow
)

from .control import (
    FeedbackController,
    PIDController,
    MPCController,
    ControlLoop,
    OptimizationEngine
)

from .lims import (
    LIMSClient,
    SampleTracker,
    DataUploader,
    ExperimentLogger
)

__all__ = [
    # Equipment
    'BaseEquipment',
    'RobotArm', 
    'SynthesisEquipment',
    'CharacterizationInstrument',
    'EquipmentManager',
    # Synthesis
    'SynthesisPlanner',
    'PowderSynthesizer',
    'ThinFilmDepositor',
    'SynthesisRecipe',
    'ProcessParameters',
    # Characterization
    'XRDParser',
    'SEMParser',
    'ElectrochemicalParser',
    'DataAggregator',
    'CharacterizationWorkflow',
    # Control
    'FeedbackController',
    'PIDController',
    'MPCController',
    'ControlLoop',
    'OptimizationEngine',
    # LIMS
    'LIMSClient',
    'SampleTracker',
    'DataUploader',
    'ExperimentLogger',
]
