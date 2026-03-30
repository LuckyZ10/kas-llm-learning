"""
DFT-LAMMPS 统一日志系统
========================

统一的日志配置和管理，替代分散的logging.basicConfig调用。

Usage:
    from core.logging import get_logger
    
    logger = get_logger("my_module")
    logger.info("Processing started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


# =============================================================================
# 日志格式化器
# =============================================================================

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
    
    def format(self, record: logging.LogRecord) -> str:
        # 保存原始levelname
        original_levelname = record.levelname
        
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        result = super().format(record)
        
        # 恢复原始levelname
        record.levelname = original_levelname
        
        return result


class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器 - 便于日志分析"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False)


# =============================================================================
# 日志管理器
# =============================================================================

class LoggingManager:
    """日志管理器 - 统一管理所有日志配置"""
    
    _initialized: bool = False
    _loggers: Dict[str, logging.Logger] = {}
    _default_level: int = logging.INFO
    _default_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _log_dir: Optional[Path] = None
    
    @classmethod
    def initialize(
        cls,
        level: int = logging.INFO,
        log_dir: Optional[Path] = None,
        console: bool = True,
        file: Optional[str] = None,
        json_format: bool = False,
        colors: bool = True
    ) -> None:
        """初始化全局日志配置
        
        此函数应该在程序入口处调用一次，替代所有的logging.basicConfig
        
        Args:
            level: 日志级别
            log_dir: 日志文件目录
            console: 是否输出到控制台
            file: 日志文件名（相对于log_dir）
            json_format: 是否使用JSON格式
            colors: 控制台是否使用颜色
        """
        if cls._initialized:
            return
        
        cls._default_level = level
        cls._log_dir = Path(log_dir) if log_dir else None
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # 清除现有处理器
        root_logger.handlers = []
        
        # 创建格式化器
        if json_format:
            formatter = JSONFormatter()
        elif colors:
            formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            formatter = logging.Formatter(cls._default_format)
        
        # 控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # 文件处理器
        if file and cls._log_dir:
            cls._log_dir.mkdir(parents=True, exist_ok=True)
            file_path = cls._log_dir / file
            
            # JSON格式文件使用不同的格式化器
            if json_format:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(cls._default_format)
            
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # 设置第三方库日志级别
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        cls._initialized = True
        root_logger.info(f"Logging initialized: level={logging.getLevelName(level)}, dir={log_dir}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取命名日志记录器
        
        替代 logging.getLogger()，确保配置已初始化
        """
        if not cls._initialized:
            # 使用默认配置自动初始化
            cls.initialize()
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def set_level(cls, level: int, logger_name: Optional[str] = None) -> None:
        """设置日志级别
        
        Args:
            level: 新的日志级别
            logger_name: 特定日志记录器名称，None表示所有
        """
        if logger_name:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
        else:
            logging.getLogger().setLevel(level)
            cls._default_level = level
    
    @classmethod
    def add_file_handler(
        cls,
        filename: str,
        level: Optional[int] = None,
        json_format: bool = False
    ) -> None:
        """添加文件处理器"""
        if not cls._log_dir:
            raise RuntimeError("Log directory not set")
        
        file_path = cls._log_dir / filename
        
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(cls._default_format)
        
        handler = logging.FileHandler(file_path)
        handler.setLevel(level or cls._default_level)
        handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(handler)
    
    @classmethod
    def shutdown(cls) -> None:
        """关闭日志系统"""
        logging.shutdown()
        cls._initialized = False
        cls._loggers.clear()


# =============================================================================
# 便捷函数
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """获取日志记录器
    
    Usage:
        from core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return LoggingManager.get_logger(name)


def initialize_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    file: Optional[str] = None,
    json_format: bool = False,
    colors: bool = True
) -> None:
    """初始化日志系统（便捷函数）
    
    应该在程序入口处调用一次
    
    Args:
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_dir: 日志目录
        console: 是否输出到控制台
        file: 日志文件名
        json_format: 是否使用JSON格式
        colors: 是否使用颜色
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    numeric_level = level_map.get(level.upper(), logging.INFO)
    
    LoggingManager.initialize(
        level=numeric_level,
        log_dir=Path(log_dir) if log_dir else None,
        console=console,
        file=file,
        json_format=json_format,
        colors=colors
    )


def log_structured(logger: logging.Logger, level: int, message: str, **kwargs) -> None:
    """记录结构化日志
    
    可以附加额外的结构化数据
    
    Usage:
        log_structured(logger, logging.INFO, "Calculation complete", 
                      structure_id="mp-123", energy=-100.5)
    """
    extra = {'extra_data': kwargs}
    logger.log(level, message, extra=extra)


# =============================================================================
# 上下文管理器
# =============================================================================

class LogContext:
    """日志上下文管理器 - 临时修改日志级别"""
    
    def __init__(self, logger_name: Optional[str] = None, level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self.previous_level: Optional[int] = None
    
    def __enter__(self):
        logger = logging.getLogger(self.logger_name or '')
        self.previous_level = logger.level
        logger.setLevel(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name or '')
        logger.setLevel(self.previous_level)


# =============================================================================
# 装饰器
# =============================================================================

def log_execution(logger_name: Optional[str] = None):
    """记录函数执行的装饰器
    
    Usage:
        @log_execution()
        def my_function():
            pass
    """
    def decorator(func):
        logger = get_logger(logger_name or func.__module__)
        
        def wrapper(*args, **kwargs):
            logger.debug(f"Executing {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Failed {func.__name__}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def log_time(logger_name: Optional[str] = None, level: int = logging.INFO):
    """记录函数执行时间的装饰器
    
    Usage:
        @log_time()
        def expensive_function():
            pass
    """
    import time
    
    def decorator(func):
        logger = get_logger(logger_name or func.__module__)
        
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                logger.log(level, f"{func.__name__} took {elapsed:.3f}s")
        
        return wrapper
    return decorator


# =============================================================================
# 兼容性函数（替换旧的分散配置）
# =============================================================================

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """兼容旧代码的日志设置函数
    
    将旧的分散配置统一到 LoggingManager
    """
    # 确保全局初始化
    if not LoggingManager._initialized:
        LoggingManager.initialize(level=level)
    
    logger = LoggingManager.get_logger(name)
    
    # 添加文件处理器（如果指定）
    if log_file and LoggingManager._log_dir:
        handler = logging.FileHandler(LoggingManager._log_dir / log_file)
        handler.setLevel(level)
        formatter = logging.Formatter(format_string or LoggingManager._default_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# 为了兼容直接调用 logging.basicConfig 的代码
_original_basic_config = logging.basicConfig

def _patched_basic_config(**kwargs):
    """拦截 logging.basicConfig 调用，重定向到 LoggingManager"""
    # 提取参数
    level = kwargs.get('level', logging.WARNING)
    format_str = kwargs.get('format')
    filename = kwargs.get('filename')
    filemode = kwargs.get('filemode', 'a')
    handlers = kwargs.get('handlers')
    
    # 转换为 LoggingManager 参数
    initialize_logging(
        level=logging.getLevelName(level) if isinstance(level, int) else 'INFO',
        file=filename,
        console=handlers is None or any(isinstance(h, logging.StreamHandler) for h in handlers) if handlers else True
    )

# 可选：打补丁（如果确定要拦截所有 basicConfig 调用）
# logging.basicConfig = _patched_basic_config


# =============================================================================
# 默认初始化
# =============================================================================

# 模块加载时自动初始化（使用默认配置）
LoggingManager.initialize()
