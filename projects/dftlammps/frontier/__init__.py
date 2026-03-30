"""
DFT-LAMMPS Frontier Methods

2024-2025材料计算和AI领域最新进展的前沿方法实现。

Modules:
    - diffusion_materials: 扩散模型生成晶体结构 (CDVAE, DiffCSP)
    - flow_matching: 流匹配模型 (比扩散模型更快)
    - llm_materials_design: 大语言模型材料设计
    - mace_integration: MACE等变图神经网络
    - alignn_wrapper: ALIGNN原子线图神经网络
    - dimenet_plus_plus: DimeNet++方向消息传递
    - pinns_for_pde: PINNs求解材料PDE
    - neural_operators: 神经算子 (FNO, DeepONet)
    - physics_informed_ml: 物理约束ML势
    - self_driving_lab: 自动驾驶实验室
    - robotic_synthesis: 机器人合成规划
    - closed_loop_discovery: 闭环发现系统
"""

__version__ = "0.1.0"

# 生成式AI材料设计
from .diffusion_materials import (
    CDVAE, DiffCSP, CrystalGenerator, CrystalFeatures,
    evaluate_structure_quality
)
from .flow_matching import (
    RiemannianFlowMatching, CrystalFlowMatching, FlowSample
)
from .llm_materials_design import (
    MaterialsLLMInterface, StructureToTextEncoder,
    TextToStructureDecoder, LLMMaterialsAgent
)

# 图神经网络新材料
from .mace_integration import (
    MACE, AtomicData, MACEActiveLearner, mace_md_simulation
)
from .alignn_wrapper import (
    ALIGNN, ALIGNNForceField, GraphData, train_alignn
)
from .dimenet_plus_plus import (
    DimeNetPlusPlus, DimeNetPlusPlusPeriodic, train_dimenet_pp
)

# 物理信息神经网络
from .pinns_for_pde import (
    PINN, PhaseFieldPINN, DiffusionPINN, ElasticityPINN,
    SirenNetwork, FourierFeatureEncoding, train_pinn
)
from .neural_operators import (
    FNO2d, FNO3d, DeepONet, GNO, MultiscaleFNO,
    SpectralConv2d, SpectralConv3d
)
from .physics_informed_ml import (
    EnergyConservingPotential, PhysicsInformedLoss,
    PhysicsConstraints, ChargeConservingPotential
)

# 自动化实验
from .self_driving_lab import (
    SelfDrivingLab, SynthesisPlanner, SynthesisParameters,
    CharacterizationData, ExperimentResult
)
from .robotic_synthesis import (
    RoboticSynthesisPlanner, SynthesisPlan, ChemicalStep,
    RoboticScheduler, ReactionNetwork
)
from .closed_loop_discovery import (
    ClosedLoopDiscovery, BayesianOptimizer, FeedbackAnalyzer,
    ComputationResult, SynthesisResult
)

__all__ = [
    # Diffusion Models
    'CDVAE', 'DiffCSP', 'CrystalGenerator', 'CrystalFeatures',
    # Flow Matching
    'RiemannianFlowMatching', 'CrystalFlowMatching', 'FlowSample',
    # LLM Materials Design
    'MaterialsLLMInterface', 'StructureToTextEncoder', 'LLMMaterialsAgent',
    # GNN Models
    'MACE', 'AtomicData', 'MACEActiveLearner',
    'ALIGNN', 'ALIGNNForceField', 'GraphData',
    'DimeNetPlusPlus', 'DimeNetPlusPlusPeriodic',
    # PINNs
    'PINN', 'PhaseFieldPINN', 'DiffusionPINN', 'ElasticityPINN',
    'SirenNetwork', 'FourierFeatureEncoding',
    # Neural Operators
    'FNO2d', 'FNO3d', 'DeepONet', 'GNO', 'MultiscaleFNO',
    # Physics Informed ML
    'EnergyConservingPotential', 'PhysicsInformedLoss', 'PhysicsConstraints',
    # Self-Driving Lab
    'SelfDrivingLab', 'SynthesisPlanner', 'SynthesisParameters',
    # Robotic Synthesis
    'RoboticSynthesisPlanner', 'SynthesisPlan', 'ChemicalStep',
    # Closed-Loop Discovery
    'ClosedLoopDiscovery', 'BayesianOptimizer', 'FeedbackAnalyzer',
]
