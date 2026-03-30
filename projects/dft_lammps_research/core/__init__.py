"""
DFT-LAMMPS 多尺度材料研究平台 - 核心模块

使用示例:
    # 方式1: 使用统一配置
    from core import get_config
    config = get_config()
    
    # 方式2: 使用工厂创建计算器
    from core import CalculatorFactory
    calc = CalculatorFactory.create_dft("vasp", config.dft)
    
    # 方式3: 使用便捷函数
    from core import get_logger, get_dft_calculator
    
    logger = get_logger(__name__)
    logger.info("Starting calculation...")
    
    calc = get_dft_calculator("vasp")
    result = calc.calculate(atoms)
"""

__version__ = "2.0.0"

# 配置系统
from .config import (
    get_config,
    reload_config,
    save_config,
    GlobalConfig,
    DFTConfig,
    MDConfig,
    MLPotentialConfig,
    WorkflowConfig,
    ScreeningConfig,
    HardwareConfig,
    LoggingConfig,
)

# 日志系统
from .logging import (
    get_logger,
    initialize_logging,
    log_structured,
    LogContext,
    log_execution,
    log_time,
)

# 抽象基类
from .base import (
    # 数据结构
    CalculationResult,
    Trajectory,
    TrainingData,
    
    # 抽象基类
    DFTCalculator,
    MDSimulator,
    MLPotential,
    Workflow,
    ActiveLearningStrategy,
    
    # 工厂
    CalculatorFactory,
)

# 向后兼容 - 保留原有导入
try:
    from .dft.bridge import UnifiedDFTMDCalculator
except ImportError:
    UnifiedDFTMDCalculator = None

try:
    from .common.workflow_engine import MaterialsWorkflow
except ImportError:
    MaterialsWorkflow = None

# 便捷函数

def get_dft_calculator(code: str = "vasp", config=None):
    """便捷函数：创建DFT计算器"""
    if config is None:
        from .config import get_config as _get_config
        config = _get_config().dft
    
    return CalculatorFactory.create_dft(code, config)


def get_md_simulator(engine: str = "lammps", config=None):
    """便捷函数：创建MD模拟器"""
    if config is None:
        from .config import get_config as _get_config
        config = _get_config().md
    
    return CalculatorFactory.create_md(engine, config)


def get_ml_potential(potential_type: str = "nep", config=None):
    """便捷函数：创建ML势"""
    if config is None:
        from .config import get_config as _get_config
        config = _get_config().ml
    
    return CalculatorFactory.create_ml(potential_type, config)


# 模块元信息
__all__ = [
    # 版本
    "__version__",
    
    # 配置
    "get_config",
    "reload_config",
    "save_config",
    "GlobalConfig",
    "DFTConfig",
    "MDConfig",
    "MLPotentialConfig",
    "WorkflowConfig",
    "ScreeningConfig",
    "HardwareConfig",
    "LoggingConfig",
    
    # 日志
    "get_logger",
    "initialize_logging",
    "log_structured",
    "LogContext",
    "log_execution",
    "log_time",
    
    # 基类
    "CalculationResult",
    "Trajectory",
    "TrainingData",
    "DFTCalculator",
    "MDSimulator",
    "MLPotential",
    "Workflow",
    "ActiveLearningStrategy",
    "CalculatorFactory",
    
    # 便捷函数
    "get_dft_calculator",
    "get_md_simulator",
    "get_ml_potential",
    
    # 向后兼容
    "UnifiedDFTMDCalculator",
    "MaterialsWorkflow",
]
