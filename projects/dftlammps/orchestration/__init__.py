"""
DFT-LAMMPS Orchestration System
===============================

模块编排与乐高式组合系统

提供模块注册、能力图谱、工作流组合等核心功能。

Modules:
    module_registry: 全局模块注册中心
    capability_graph: 能力图谱系统
    workflow_composer: 工作流组合器
    topic_template: 课题模板
    cross_module_bridge: 跨模块桥接

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from .module_registry import (
    ModuleRegistry,
    Capability,
    CapabilityType,
    ModuleInterface,
    ModuleVersion,
    DependencyResolver,
    module,
    capability,
    register_module,
    get_registry
)

from .capability_graph import (
    CapabilityGraph,
    CapabilityNode,
    CapabilityEdge,
    CapabilityPath,
    NodeType,
    EdgeType,
    get_global_graph,
    build_graph_from_registry
)

from .workflow_composer import (
    Workflow,
    WorkflowStep,
    WorkflowType,
    StepStatus,
    WorkflowComposer,
    DeclarativeComposer,
    CodeBasedComposer,
    DragDropComposer,
    SmartComposer,
    WorkflowExecutor,
    create_workflow,
    load_workflow,
    save_workflow
)

from .topic_template import (
    TopicTemplate,
    TopicTemplateManager,
    ResearchTopic,
    get_template_manager,
    create_battery_workflow,
    create_catalyst_workflow,
    create_photovoltaic_workflow,
    create_alloy_workflow
)

from .cross_module_bridge import (
    CrossModuleBridge,
    DataConverter,
    StructureConverter,
    EnergyConverter,
    ForceConverter,
    Adapter,
    get_global_bridge,
    convert_data,
    adapt_output
)

__version__ = "1.0.0"
__all__ = [
    # Module Registry
    'ModuleRegistry',
    'Capability',
    'CapabilityType',
    'ModuleInterface',
    'ModuleVersion',
    'DependencyResolver',
    'module',
    'capability',
    'register_module',
    'get_registry',
    
    # Capability Graph
    'CapabilityGraph',
    'CapabilityNode',
    'CapabilityEdge',
    'CapabilityPath',
    'NodeType',
    'EdgeType',
    'get_global_graph',
    'build_graph_from_registry',
    
    # Workflow Composer
    'Workflow',
    'WorkflowStep',
    'WorkflowType',
    'StepStatus',
    'WorkflowComposer',
    'DeclarativeComposer',
    'CodeBasedComposer',
    'DragDropComposer',
    'SmartComposer',
    'WorkflowExecutor',
    'create_workflow',
    'load_workflow',
    'save_workflow',
    
    # Topic Templates
    'TopicTemplate',
    'TopicTemplateManager',
    'ResearchTopic',
    'get_template_manager',
    'create_battery_workflow',
    'create_catalyst_workflow',
    'create_photovoltaic_workflow',
    'create_alloy_workflow',
    
    # Cross Module Bridge
    'CrossModuleBridge',
    'DataConverter',
    'StructureConverter',
    'EnergyConverter',
    'ForceConverter',
    'Adapter',
    'get_global_bridge',
    'convert_data',
    'adapt_output'
]