"""
DFT-LAMMPS Integration Layer
============================

集成层模块

提供统一数据模型、事件总线、状态管理和资源调度。

Modules:
    unified_data_model: 统一数据模型
    event_bus: 事件总线
    state_manager: 状态管理
    resource_scheduler: 资源调度

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from .unified_data_model import (
    DataCategory,
    DataQuality,
    PropertyType,
    UnifiedDataModel,
    StructureData,
    PropertyData,
    CalculationResultData,
    SimulationTrajectoryData,
    DataRepository,
    create_structure_data,
    get_repository
)

from .event_bus import (
    EventBus,
    Event,
    EventType,
    EventPriority,
    EventSubscription,
    event_handler,
    publish_event,
    subscribe_to,
    EventLogger,
    EventReplayer
)

from .state_manager import (
    StateManager,
    WorkflowState,
    CheckpointStrategy,
    Checkpoint,
    ExecutionRecord,
    WorkflowRecovery,
    get_state_manager,
    create_checkpoint,
    resume_workflow
)

from .resource_scheduler import (
    ResourceScheduler,
    ResourcePool,
    Resource,
    CPUResource,
    GPUResource,
    MemoryResource,
    LicenseResource,
    ComputeTask,
    TaskPriority,
    TaskStatus,
    ResourceRequirement,
    get_scheduler,
    submit_module_task
)

__all__ = [
    # Unified Data Model
    'DataCategory',
    'DataQuality',
    'PropertyType',
    'UnifiedDataModel',
    'StructureData',
    'PropertyData',
    'CalculationResultData',
    'SimulationTrajectoryData',
    'DataRepository',
    'create_structure_data',
    'get_repository',
    
    # Event Bus
    'EventBus',
    'Event',
    'EventType',
    'EventPriority',
    'EventSubscription',
    'event_handler',
    'publish_event',
    'subscribe_to',
    'EventLogger',
    'EventReplayer',
    
    # State Manager
    'StateManager',
    'WorkflowState',
    'CheckpointStrategy',
    'Checkpoint',
    'ExecutionRecord',
    'WorkflowRecovery',
    'get_state_manager',
    'create_checkpoint',
    'resume_workflow',
    
    # Resource Scheduler
    'ResourceScheduler',
    'ResourcePool',
    'Resource',
    'CPUResource',
    'GPUResource',
    'MemoryResource',
    'LicenseResource',
    'ComputeTask',
    'TaskPriority',
    'TaskStatus',
    'ResourceRequirement',
    'get_scheduler',
    'submit_module_task'
]