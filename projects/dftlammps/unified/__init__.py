"""
DFT+LAMMPS Unified Architecture
===============================
统一架构包 - 整合所有104个模块的协调框架

核心组件：
- config_system: 统一配置系统
- unified_api: 统一API层
- orchestrator_v2: 超级编排器
- common: 共享工具（错误处理、日志、装饰器）

使用示例：
    from dftlammps.unified import (
        ConfigManager, GlobalConfig,
        UnifiedAPIRouter, ModuleInterface,
        OrchestratorV2, WorkflowBuilder,
        get_logger
    )
"""

__version__ = "2.0.0"
__author__ = "DFT+LAMMPS Team"

# 配置系统
from .config_system import (
    ConfigManager,
    ConfigBuilder,
    GlobalConfig,
    ModuleConfig,
    DFTConfig,
    LAMMPSConfig,
    MLPotentialConfig,
    WorkflowConfig,
    HPCConfig,
    DatabaseConfig,
    LoggingConfig,
    ConfigFormat,
    ConfigSchema,
    ConfigValidationError,
    ConfigNotFoundError,
    load_config,
    load_from_env,
    create_default_config,
    get_config_manager,
    DEFAULT_CONFIG_TEMPLATE
)

# 统一API
from .unified_api import (
    UnifiedAPIRouter,
    ModuleInterface,
    RouteInfo,
    APIRequest,
    APIResponse,
    HTTPMethod,
    APIVersion,
    ResponseStatus,
    Middleware,
    AuthenticationMiddleware,
    RateLimitMiddleware,
    LoggingMiddleware,
    CORSMiddleware,
    ModuleRegistry,
    api_route,
    get, post, put, delete,
    get_router,
    init_api,
    call_api,
    call_api_sync
)

# 编排器
from .orchestrator_v2 import (
    OrchestratorV2,
    WorkflowBuilder,
    Workflow,
    Task,
    TaskResult,
    TaskStatus,
    TaskPriority,
    ExecutionMode,
    ExecutionContext,
    ResourceRequirements,
    SmartScheduler,
    GlobalOptimizer,
    TaskExecutor,
    ModuleTaskExecutor,
    PythonFunctionExecutor,
    ShellCommandExecutor,
    get_orchestrator,
    run_workflow
)

# 通用工具
from .common import (
    # 异常类
    DFTLAMMPSError,
    ConfigurationError,
    ValidationError,
    FileSystemError,
    CalculationError,
    DFTError,
    LAMMPSError,
    MLPotentialError,
    WorkflowError,
    SchedulerError,
    DatabaseError,
    APIError,
    ResourceError,
    TimeoutError,
    ParallelError,
    ErrorCode,
    
    # 日志系统
    LoggerManager,
    ContextAdapter,
    StructuredLogFormatter,
    ColoredFormatter,
    get_logger,
    logger_manager,
    init_logging,
    
    # 装饰器
    retry,
    log_execution,
    handle_errors,
    timing,
    
    # 上下文管理器
    log_context,
    timer,
    suppress_exceptions,
    
    # 辅助函数
    generate_id,
    safe_divide,
    format_duration,
    truncate_string,
    merge_dicts,
    deep_merge
)

# 模块导出列表
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 配置系统
    "ConfigManager",
    "ConfigBuilder", 
    "GlobalConfig",
    "ModuleConfig",
    "DFTConfig",
    "LAMMPSConfig",
    "MLPotentialConfig",
    "WorkflowConfig",
    "HPCConfig",
    "DatabaseConfig",
    "LoggingConfig",
    "ConfigFormat",
    "ConfigSchema",
    "ConfigValidationError",
    "ConfigNotFoundError",
    "load_config",
    "load_from_env",
    "create_default_config",
    "get_config_manager",
    
    # 统一API
    "UnifiedAPIRouter",
    "ModuleInterface",
    "RouteInfo",
    "APIRequest",
    "APIResponse",
    "HTTPMethod",
    "APIVersion",
    "ResponseStatus",
    "Middleware",
    "AuthenticationMiddleware",
    "RateLimitMiddleware",
    "LoggingMiddleware",
    "CORSMiddleware",
    "ModuleRegistry",
    "api_route",
    "get", "post", "put", "delete",
    "get_router",
    "init_api",
    "call_api",
    "call_api_sync",
    
    # 编排器
    "OrchestratorV2",
    "WorkflowBuilder",
    "Workflow",
    "Task",
    "TaskResult",
    "TaskStatus",
    "TaskPriority",
    "ExecutionMode",
    "ExecutionContext",
    "ResourceRequirements",
    "SmartScheduler",
    "GlobalOptimizer",
    "TaskExecutor",
    "ModuleTaskExecutor",
    "PythonFunctionExecutor",
    "ShellCommandExecutor",
    "get_orchestrator",
    "run_workflow",
    
    # 异常类
    "DFTLAMMPSError",
    "ConfigurationError",
    "ValidationError",
    "FileSystemError",
    "CalculationError",
    "DFTError",
    "LAMMPSError",
    "MLPotentialError",
    "WorkflowError",
    "SchedulerError",
    "DatabaseError",
    "APIError",
    "ResourceError",
    "TimeoutError",
    "ParallelError",
    "ErrorCode",
    
    # 日志系统
    "LoggerManager",
    "ContextAdapter",
    "StructuredLogFormatter",
    "ColoredFormatter",
    "get_logger",
    "logger_manager",
    "init_logging",
    
    # 装饰器和工具
    "retry",
    "log_execution",
    "handle_errors",
    "timing",
    "log_context",
    "timer",
    "generate_id",
    "format_duration",
    "merge_dicts",
    "deep_merge"
]


def initialize_unified_system(config_path: str = None) -> tuple:
    """
    初始化统一架构系统
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        (ConfigManager, UnifiedAPIRouter, OrchestratorV2) 元组
    """
    # 初始化日志
    init_logging()
    logger = get_logger("unified.init")
    logger.info("Initializing DFT+LAMMPS Unified Architecture v" + __version__)
    
    # 加载配置
    config_manager = get_config_manager()
    if config_path:
        config_manager.load_config(config_path)
    config_manager.load_from_env()
    
    # 初始化API路由器
    api_router = init_api(config_manager.global_config)
    
    # 初始化编排器
    orchestrator = get_orchestrator(config_manager.global_config)
    
    logger.info("Unified system initialization complete")
    
    return config_manager, api_router, orchestrator


async def quick_start_workflow(tasks_config: list, workflow_name: str = "quick_workflow") -> dict:
    """
    快速启动工作流的便捷函数
    
    Args:
        tasks_config: 任务配置列表
        workflow_name: 工作流名称
        
    Returns:
        工作流执行结果
    """
    from .orchestrator_v2 import WorkflowBuilder, Task, ResourceRequirements
    
    builder = WorkflowBuilder(name=workflow_name)
    
    for task_conf in tasks_config:
        builder.add_task(
            name=task_conf.get("name", "unnamed"),
            module=task_conf.get("module", ""),
            operation=task_conf.get("operation", ""),
            params=task_conf.get("params", {}),
            depends_on=task_conf.get("depends_on"),
            priority=task_conf.get("priority", TaskPriority.NORMAL),
            resources=task_conf.get("resources", ResourceRequirements())
        )
    
    return await builder.execute()
