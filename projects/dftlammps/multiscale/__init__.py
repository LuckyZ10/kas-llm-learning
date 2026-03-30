"""
dftlammps.multiscale
====================
多尺度耦合模块

本模块提供从原子尺度到连续尺度的耦合能力：

1. phase_field.py: 相场模拟接口
   - 枝晶生长模拟
   - 相分离动力学
   - 与MD的耦合

2. continuum.py: 连续介质力学
   - 有限元分析
   - 热传导分析
   - 耦合热-力问题

3. parameter_passing.py: 跨尺度参数传递
   - 从DFT提取弹性常数
   - 从MD提取扩散系数
   - 自动格式转换

示例用法:
---------
>>> from dftlammps.multiscale import PhaseFieldWorkflow, ContinuumWorkflow
>>> from dftlammps.multiscale import ParameterPassingWorkflow

# 相场模拟
>>> pf_workflow = PhaseFieldWorkflow()
>>> pf_workflow.extract_parameters_from_md("md_trajectory.xyz")
>>> pf_workflow.setup_dendrite_simulation()
>>> pf_workflow.run_simulation()

# 连续介质分析
>>> cont_workflow = ContinuumWorkflow()
>>> cont_workflow.setup_material(elastic_props={'C11': 100, 'C12': 50})
>>> cont_workflow.generate_mesh(FEMConfig(nx=50, ny=50))
>>> results = cont_workflow.run_mechanics_analysis(mechanics_config)

# 跨尺度参数传递
>>> param_workflow = ParameterPassingWorkflow()
>>> params = param_workflow.run_full_workflow(
...     dft_structure="POSCAR",
...     md_trajectory="md.xyz"
... )
"""

from .phase_field import (
    PhaseFieldConfig,
    DendriteConfig,
    SpinodalConfig,
    PhaseFieldSolver,
    DendriteGrowthSolver,
    SpinodalDecompositionSolver,
    MDtoPhaseFieldExtractor,
    PRISMSPFInterface,
    MOOSEInterface,
    PhaseFieldWorkflow,
)

from .continuum import (
    ElasticProperties,
    ThermalProperties,
    MaterialModel,
    FEMConfig,
    MechanicsConfig,
    ThermalConfig,
    FEMMesh,
    ThermalSolver,
    MechanicsSolver,
    CoupledThermalMechanicsSolver,
    ContinuumWorkflow,
)

from .parameter_passing import (
    ElasticConstants,
    TransportProperties,
    InterfaceProperties,
    PhaseFieldParameters,
    MultiScaleParameters,
    DFTParameterExtractor,
    MDParameterExtractor,
    ParameterConverter,
    ParameterValidator,
    ParameterPassingWorkflow,
)

__all__ = [
    # Phase Field
    'PhaseFieldConfig',
    'DendriteConfig',
    'SpinodalConfig',
    'PhaseFieldSolver',
    'DendriteGrowthSolver',
    'SpinodalDecompositionSolver',
    'MDtoPhaseFieldExtractor',
    'PRISMSPFInterface',
    'MOOSEInterface',
    'PhaseFieldWorkflow',
    
    # Continuum
    'ElasticProperties',
    'ThermalProperties',
    'MaterialModel',
    'FEMConfig',
    'MechanicsConfig',
    'ThermalConfig',
    'FEMMesh',
    'ThermalSolver',
    'MechanicsSolver',
    'CoupledThermalMechanicsSolver',
    'ContinuumWorkflow',
    
    # Parameter Passing
    'ElasticConstants',
    'TransportProperties',
    'InterfaceProperties',
    'PhaseFieldParameters',
    'MultiScaleParameters',
    'DFTParameterExtractor',
    'MDParameterExtractor',
    'ParameterConverter',
    'ParameterValidator',
    'ParameterPassingWorkflow',
]
