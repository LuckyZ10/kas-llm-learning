"""
DFT-LAMMPS 实验自动化模块

提供实验室自动化和合成规划功能
"""

from .lab_automation import (
    ExperimentStatus,
    RobotCommand,
    MaterialSpec,
    SynthesisParameter,
    RobotInstruction,
    ExperimentResult,
    RobotInterface,
    SimulatedRobot,
    UR5Robot,
    OT2Robot,
    SynthesisProtocol,
    ProtocolGenerator,
    ExperimentRunner,
    AutonomousLab,
    ExperimentQueue,
    DataReader,
    FileDataReader,
    InstrumentDataReader,
    DatabaseConnector,
    create_lab,
    create_simulated_robot,
    create_protocol_generator
)

from .synthesis_planning import (
    SynthesisDifficulty,
    ReactionType,
    ChemicalCompound,
    Precursor,
    SynthesisStep,
    SynthesisRoute,
    SynthesisPredictor,
    KnowledgeBasedPredictor,
    MLPredictor,
    SynthesisPlanner,
    PrecursorOptimizer,
    create_planner,
    predict_synthesis_feasibility,
    plan_synthesis_route
)

__all__ = [
    # Enums
    'ExperimentStatus',
    'RobotCommand',
    'SynthesisDifficulty',
    'ReactionType',
    
    # Data Classes
    'MaterialSpec',
    'SynthesisParameter',
    'RobotInstruction',
    'ExperimentResult',
    'ChemicalCompound',
    'Precursor',
    'SynthesisStep',
    'SynthesisRoute',
    
    # Robot Interfaces
    'RobotInterface',
    'SimulatedRobot',
    'UR5Robot',
    'OT2Robot',
    
    # Automation
    'SynthesisProtocol',
    'ProtocolGenerator',
    'ExperimentRunner',
    'AutonomousLab',
    'ExperimentQueue',
    
    # Data Reading
    'DataReader',
    'FileDataReader',
    'InstrumentDataReader',
    'DatabaseConnector',
    
    # Planning
    'SynthesisPredictor',
    'KnowledgeBasedPredictor',
    'MLPredictor',
    'SynthesisPlanner',
    'PrecursorOptimizer',
    
    # Factory Functions
    'create_lab',
    'create_simulated_robot',
    'create_protocol_generator',
    'create_planner',
    
    # Utility Functions
    'predict_synthesis_feasibility',
    'plan_synthesis_route'
]
