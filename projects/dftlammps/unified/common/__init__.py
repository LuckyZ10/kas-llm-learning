"""
DFT+LAMMPS Common Utilities
===========================
统一错误处理、日志系统和通用工具

功能：
1. 统一异常层次结构
2. 结构化日志系统
3. 装饰器工具
4. 通用辅助函数
"""

import sys
import traceback
import functools
import logging
import logging.handlers
from typing import Dict, List, Optional, Any, Callable, Type, Union
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
import json
import uuid
import time
from contextlib import contextmanager


# =============================================================================
# 异常类层次结构
# =============================================================================

class DFTLAMMPSError(Exception):
    """DFT+LAMMPS 基础异常类"""
    
    def __init__(self, message: str, error_code: str = "",
                 details: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now().isoformat()
        self.traceback_str = traceback.format_exc() if sys.exc_info()[0] else None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
            "traceback": self.traceback_str,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return "\n".join(parts)


class ConfigurationError(DFTLAMMPSError):
    """配置错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)


class ValidationError(DFTLAMMPSError):
    """验证错误"""
    def __init__(self, message: str, field: str = "", **kwargs):
        details = kwargs.get('details', {})
        details['field'] = field
        kwargs['details'] = details
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)


class FileSystemError(DFTLAMMPSError):
    """文件系统错误"""
    def __init__(self, message: str, path: str = "", **kwargs):
        details = kwargs.get('details', {})
        details['path'] = path
        kwargs['details'] = details
        super().__init__(message, error_code="FILESYSTEM_ERROR", **kwargs)


class CalculationError(DFTLAMMPSError):
    """计算错误"""
    def __init__(self, message: str, calculator: str = "", **kwargs):
        details = kwargs.get('details', {})
        details['calculator'] = calculator
        kwargs['details'] = details
        super().__init__(message, error_code="CALCULATION_ERROR", **kwargs)


class DFTError(CalculationError):
    """DFT计算错误"""
    def __init__(self, message: str, code: str = "vasp", **kwargs):
        super().__init__(message, calculator=code, **kwargs)
        self.error_code = f"DFT_ERROR_{code.upper()}"


class LAMMPSError(CalculationError):
    """LAMMPS模拟错误"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, calculator="lammps", **kwargs)
        self.error_code = "LAMMPS_ERROR"


class MLPotentialError(CalculationError):
    """ML势错误"""
    def __init__(self, message: str, framework: str = "", **kwargs):
        details = kwargs.get('details', {})
        details['framework'] = framework
        kwargs['details'] = details
        super().__init__(message, calculator="ml_potential", **kwargs)
        self.error_code = f"MLP_ERROR_{framework.upper()}" if framework else "MLP_ERROR"


class WorkflowError(DFTLAMMPSError):
    """工作流错误"""
    def __init__(self, message: str, step: str = "", **kwargs):
        details = kwargs.get('details', {})
        details['step'] = step
        kwargs['details'] = details
        super().__init__(message, error_code="WORKFLOW_ERROR", **kwargs)


class SchedulerError(DFTLAMMPSError):
    """调度器错误"""
    def __init__(self, message: str, scheduler: str = "", **kwargs):
        details = kwargs.get('details', {})
        details['scheduler'] = scheduler
        kwargs['details'] = details
        super().__init__(message, error_code="SCHEDULER_ERROR", **kwargs)


class DatabaseError(DFTLAMMPSError):
    """数据库错误"""
    def __init__(self, message: str, operation: str = "", **kwargs):
        details = kwargs.get('details', {})
        details['operation'] = operation
        kwargs['details'] = details
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)


class APIError(DFTLAMMPSError):
    """API错误"""
    def __init__(self, message: str, status_code: int = 500, **kwargs):
        details = kwargs.get('details', {})
        details['status_code'] = status_code
        kwargs['details'] = details
        super().__init__(message, error_code=f"API_ERROR_{status_code}", **kwargs)


class ResourceError(DFTLAMMPSError):
    """资源错误"""
    def __init__(self, message: str, resource_type: str = "", **kwargs):
        details = kwargs.get('details', {})
        details['resource_type'] = resource_type
        kwargs['details'] = details
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)


class TimeoutError(DFTLAMMPSError):
    """超时错误"""
    def __init__(self, message: str, timeout_seconds: float = 0, **kwargs):
        details = kwargs.get('details', {})
        details['timeout_seconds'] = timeout_seconds
        kwargs['details'] = details
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)


class ParallelError(DFTLAMMPSError):
    """并行计算错误"""
    def __init__(self, message: str, task_id: str = "", **kwargs):
        details = kwargs.get('details', {})
        details['task_id'] = task_id
        kwargs['details'] = details
        super().__init__(message, error_code="PARALLEL_ERROR", **kwargs)


# =============================================================================
# 错误代码枚举
# =============================================================================

class ErrorCode(Enum):
    """标准错误代码"""
    # 通用错误 (1-99)
    UNKNOWN_ERROR = "E001"
    NOT_IMPLEMENTED = "E002"
    INVALID_ARGUMENT = "E003"
    MISSING_DEPENDENCY = "E004"
    
    # 配置错误 (100-199)
    CONFIG_NOT_FOUND = "E100"
    CONFIG_INVALID = "E101"
    CONFIG_VERSION_MISMATCH = "E102"
    
    # 文件错误 (200-299)
    FILE_NOT_FOUND = "E200"
    FILE_PERMISSION_DENIED = "E201"
    FILE_CORRUPTED = "E202"
    DIRECTORY_NOT_FOUND = "E203"
    
    # 计算错误 (300-399)
    CALCULATION_FAILED = "E300"
    CALCULATION_NOT_CONVERGED = "E301"
    CALCULATION_TIMEOUT = "E302"
    INSUFFICIENT_RESOURCES = "E303"
    
    # DFT特定错误 (400-499)
    DFT_SCF_NOT_CONVERGED = "E400"
    DFT_GEOMETRY_NOT_CONVERGED = "E401"
    DFT_INVALID_INPUT = "E402"
    DFT_OUTPUT_CORRUPTED = "E403"
    
    # LAMMPS特定错误 (500-599)
    LAMMPS_INIT_FAILED = "E500"
    LAMMPS_RUN_FAILED = "E501"
    LAMMPS_INVALID_INPUT = "E502"
    LAMMPS_CRASHED = "E503"
    
    # ML势错误 (600-699)
    MLP_TRAINING_FAILED = "E600"
    MLP_INVALID_DATA = "E601"
    MLP_INFERENCE_FAILED = "E602"
    MLP_MODEL_NOT_FOUND = "E603"
    
    # 工作流错误 (700-799)
    WORKFLOW_STEP_FAILED = "E700"
    WORKFLOW_INVALID_DEPENDENCY = "E701"
    WORKFLOW_CIRCULAR_DEPENDENCY = "E702"
    
    # 网络/通信错误 (800-899)
    NETWORK_ERROR = "E800"
    API_RATE_LIMIT = "E801"
    AUTHENTICATION_FAILED = "E802"


# =============================================================================
# 结构化日志系统
# =============================================================================

class StructuredLogFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化为JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }
        
        # 添加额外字段
        if hasattr(record, 'task_id'):
            log_data['task_id'] = record.task_id
        if hasattr(record, 'workflow_id'):
            log_data['workflow_id'] = record.workflow_id
        if hasattr(record, 'component'):
            log_data['component'] = record.component
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 添加额外属性
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith('_'):
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True):
        super().__init__(fmt or '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


class ContextAdapter(logging.LoggerAdapter):
    """带上下文的日志适配器"""
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Any) -> tuple:
        """处理日志消息"""
        kwargs['extra'] = {**(kwargs.get('extra', {})), **self.extra}
        return msg, kwargs
    
    def with_context(self, **kwargs) -> 'ContextAdapter':
        """添加上下文"""
        new_extra = {**self.extra, **kwargs}
        return ContextAdapter(self.logger, new_extra)


class LoggerManager:
    """日志管理器"""
    
    _instance: Optional['LoggerManager'] = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.default_level = logging.INFO
        self.formatter: Optional[logging.Formatter] = None
        self.handlers: List[logging.Handler] = []
    
    def setup(self, 
             level: Union[str, int] = logging.INFO,
             log_file: Optional[str] = None,
             structured: bool = False,
             colored: bool = True,
             max_bytes: int = 10*1024*1024,
             backup_count: int = 5) -> None:
        """
        设置日志系统
        
        Args:
            level: 日志级别
            log_file: 日志文件路径
            structured: 是否使用结构化JSON格式
            colored: 是否使用彩色输出
            max_bytes: 单个日志文件最大大小
            backup_count: 保留的备份文件数量
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        self.default_level = level
        
        # 根日志器设置
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # 清除现有处理器
        root_logger.handlers.clear()
        self.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        if structured:
            console_handler.setFormatter(StructuredLogFormatter())
        elif colored:
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        root_logger.addHandler(console_handler)
        self.handlers.append(console_handler)
        
        # 文件处理器
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            if structured:
                file_handler.setFormatter(StructuredLogFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
                ))
            root_logger.addHandler(file_handler)
            self.handlers.append(file_handler)
    
    def get_logger(self, name: str, 
                   component: Optional[str] = None,
                   task_id: Optional[str] = None) -> ContextAdapter:
        """
        获取日志器
        
        Args:
            name: 日志器名称
            component: 组件标识
            task_id: 任务ID
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.default_level)
            self._loggers[name] = logger
        
        extra = {}
        if component:
            extra['component'] = component
        if task_id:
            extra['task_id'] = task_id
        
        return ContextAdapter(self._loggers[name], extra)
    
    def set_level(self, level: Union[str, int], logger_name: Optional[str] = None) -> None:
        """设置日志级别"""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        if logger_name:
            logging.getLogger(logger_name).setLevel(level)
        else:
            logging.getLogger().setLevel(level)
            for handler in self.handlers:
                handler.setLevel(level)


# 全局日志管理器
logger_manager = LoggerManager()


def get_logger(name: str, **kwargs) -> ContextAdapter:
    """获取日志器"""
    return logger_manager.get_logger(name, **kwargs)


# =============================================================================
# 装饰器工具
# =============================================================================

def retry(max_attempts: int = 3, 
          delay: float = 1.0,
          backoff: float = 2.0,
          exceptions: Tuple[Type[Exception], ...] = (Exception,),
          on_retry: Optional[Callable[[Exception, int], None]] = None):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟（秒）
        backoff: 延迟增长因子
        exceptions: 捕获的异常类型
        on_retry: 重试回调函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def log_execution(logger_name: Optional[str] = None,
                 level: int = logging.DEBUG,
                 log_args: bool = True,
                 log_result: bool = False):
    """
    执行日志装饰器
    
    Args:
        logger_name: 日志器名称
        level: 日志级别
        log_args: 是否记录参数
        log_result: 是否记录返回值
    """
    def decorator(func: Callable) -> Callable:
        logger = get_logger(logger_name or func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            
            # 记录开始
            if log_args:
                logger.log(level, f"[START] {func_name} args={args}, kwargs={kwargs}")
            else:
                logger.log(level, f"[START] {func_name}")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                if log_result:
                    logger.log(level, f"[END] {func_name} duration={duration:.2f}ms result={result}")
                else:
                    logger.log(level, f"[END] {func_name} duration={duration:.2f}ms")
                
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(f"[ERROR] {func_name} duration={duration:.2f}ms error={e}")
                raise
        
        return wrapper
    return decorator


def handle_errors(default_return: Any = None,
                 exceptions: Tuple[Type[Exception], ...] = (Exception,),
                 reraise: bool = False,
                 logger_name: Optional[str] = None):
    """
    错误处理装饰器
    
    Args:
        default_return: 异常时的默认返回值
        exceptions: 捕获的异常类型
        reraise: 是否重新抛出异常
        logger_name: 日志器名称
    """
    def decorator(func: Callable) -> Callable:
        logger = get_logger(logger_name or func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.error(f"Error in {func.__qualname__}: {e}", exc_info=True)
                if reraise:
                    raise
                return default_return
        
        return wrapper
    return decorator


def timing(func: Callable) -> Callable:
    """计时装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"⏱️  {func.__qualname__} took {duration:.4f}s")
        return result
    return wrapper


# =============================================================================
# 上下文管理器
# =============================================================================

@contextmanager
def log_context(operation: str, logger: Optional[logging.Logger] = None):
    """日志上下文管理器"""
    log = logger or logging.getLogger(__name__)
    start_time = time.time()
    log.info(f"[BEGIN] {operation}")
    
    try:
        yield
        duration = (time.time() - start_time) * 1000
        log.info(f"[SUCCESS] {operation} completed in {duration:.2f}ms")
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log.error(f"[FAILED] {operation} failed after {duration:.2f}ms: {e}")
        raise


@contextmanager
def timer(name: str = "Operation", print_fn: Callable[[str], None] = print):
    """计时上下文管理器"""
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    print_fn(f"⏱️  {name}: {duration:.4f}s")


@contextmanager
def suppress_exceptions(*exceptions: Type[Exception], default: Any = None):
    """异常抑制上下文管理器"""
    try:
        yield
    except exceptions:
        pass
    return default


# =============================================================================
# 辅助函数
# =============================================================================

def generate_id(prefix: str = "") -> str:
    """生成唯一ID"""
    unique_id = uuid.uuid4().hex[:12]
    return f"{prefix}_{unique_id}" if prefix else unique_id


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法"""
    return a / b if b != 0 else default


def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断字符串"""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个字典"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# 初始化默认日志配置
# =============================================================================

def init_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """初始化日志系统"""
    logger_manager.setup(level=level, log_file=log_file, colored=True)


# 模块初始化
init_logging()
