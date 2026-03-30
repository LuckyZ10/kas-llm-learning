"""
DFT+LAMMPS Unified Configuration System
=======================================
统一配置管理系统 - 支持YAML/JSON配置、环境变量和配置验证

功能：
1. 多格式配置支持 (YAML/JSON)
2. 环境变量覆盖
3. 配置验证与类型检查
4. 配置版本管理
5. 模块化配置加载
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic, Callable
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from enum import Enum
import logging
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigFormat(Enum):
    """支持的配置格式"""
    YAML = "yaml"
    JSON = "json"
    AUTO = "auto"


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigNotFoundError(Exception):
    """配置文件未找到错误"""
    pass


@dataclass
class ConfigSchema:
    """配置模式定义"""
    key: str
    type: type
    required: bool = True
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    
    def validate(self, value: Any) -> bool:
        """验证值是否符合模式"""
        if value is None and self.required:
            return False
        if value is not None and not isinstance(value, self.type):
            try:
                self.type(value)
            except (ValueError, TypeError):
                return False
        if self.validator and value is not None:
            return self.validator(value)
        return True


@dataclass
class ModuleConfig:
    """模块配置基类"""
    name: str = ""
    enabled: bool = True
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DFTConfig(ModuleConfig):
    """DFT计算配置"""
    name: str = "dft"
    calculator: str = "vasp"  # vasp, espresso, abinit
    pp_path: str = ""
    kpoints_density: float = 0.2
    encut: float = 520.0
    ediff: float = 1e-6
    nelm: int = 100
    ismear: int = 0
    sigma: float = 0.05
    ibrion: int = 2
    nsw: int = 100
    ediffg: float = -0.01
    lwave: bool = False
    lcharg: bool = False
    lreal: str = "Auto"
    ncore: int = 4
    kpar: int = 1
    
    # VASPsol 配置
    use_vaspsol: bool = False
    lsol: bool = False
    lambda_vaspsol: float = 1.0
    tau: float = 0.0


@dataclass
class LAMMPSConfig(ModuleConfig):
    """LAMMPS模拟配置"""
    name: str = "lammps"
    pair_style: str = "deepmd"
    potential_file: str = ""
    timestep: float = 0.001
    temperature: float = 300.0
    pressure: float = 1.0
    nsteps: int = 100000
    thermo_interval: int = 100
    dump_interval: int = 1000
    ensemble: str = "npt"  # nve, nvt, npt
    
    # 高级设置
    use_gpu: bool = False
    gpu_devices: List[int] = field(default_factory=list)
    mpi_processes: int = 4
    omp_threads: int = 1


@dataclass
class MLPotentialConfig(ModuleConfig):
    """机器学习势配置"""
    name: str = "ml_potential"
    framework: str = "deepmd"  # deepmd, nequip, mace, chgnet, orb
    descriptor_type: str = "se_e2_a"
    rcut: float = 6.0
    rcut_smth: float = 0.5
    sel: List[int] = field(default_factory=lambda: [50, 50])
    neuron: List[int] = field(default_factory=lambda: [25, 50, 100])
    axis_neuron: int = 16
    
    # 训练参数
    batch_size: int = 4
    num_steps: int = 1000000
    learning_rate: float = 0.001
    decay_steps: int = 5000
    decay_rate: float = 0.95
    start_lr: float = 0.001
    stop_lr: float = 1e-8
    
    # 数据集
    training_systems: List[str] = field(default_factory=list)
    validation_systems: List[str] = field(default_factory=list)
    numb_steps: int = 1000000


@dataclass
class WorkflowConfig(ModuleConfig):
    """工作流配置"""
    name: str = "workflow"
    max_iterations: int = 10
    convergence_threshold: float = 1e-5
    checkpoint_interval: int = 100
    auto_restart: bool = True
    backup_enabled: bool = True
    parallel_jobs: int = 4
    
    # 调度策略
    scheduler: str = "local"  # local, slurm, pbs, lsf
    queue_name: str = ""
    walltime: str = "24:00:00"
    memory_per_node: str = "64GB"


@dataclass
class HPCConfig(ModuleConfig):
    """HPC资源配置"""
    name: str = "hpc"
    cluster_type: str = "local"  # local, slurm, pbs, lsf, aws, azure
    max_nodes: int = 10
    cores_per_node: int = 64
    gpus_per_node: int = 4
    
    # 调度策略
    priority: str = "normal"  # low, normal, high, urgent
    preemption: bool = False
    
    # 存储
    scratch_path: str = "/tmp"
    shared_fs: str = "/shared"
    
    # 监控
    monitoring_enabled: bool = True
    metrics_interval: int = 60


@dataclass
class DatabaseConfig(ModuleConfig):
    """数据库配置"""
    name: str = "database"
    backend: str = "sqlite"  # sqlite, postgresql, mongodb
    host: str = "localhost"
    port: int = 5432
    database: str = "dftlammps"
    username: str = ""
    password: str = ""
    
    # SQLite 特定
    db_path: str = "./dftlammps.db"
    
    # 连接池
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30


@dataclass
class LoggingConfig(ModuleConfig):
    """日志配置"""
    name: str = "logging"
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    log_file: str = "dftlammps.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    
    # 远程日志
    remote_logging: bool = False
    remote_host: str = ""
    remote_port: int = 514


@dataclass
class GlobalConfig:
    """全局配置"""
    project_name: str = "DFT+LAMMPS Project"
    project_version: str = "2.0.0"
    debug_mode: bool = False
    
    # 子模块配置
    dft: DFTConfig = field(default_factory=DFTConfig)
    lammps: LAMMPSConfig = field(default_factory=LAMMPSConfig)
    ml_potential: MLPotentialConfig = field(default_factory=MLPotentialConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    hpc: HPCConfig = field(default_factory=HPCConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 自定义模块配置存储
    custom_modules: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ConfigManager:
    """
    统一配置管理器
    
    功能：
    - 加载YAML/JSON配置文件
    - 环境变量覆盖
    - 配置验证
    - 配置合并
    """
    
    ENV_PREFIX = "DFTLAMMPS_"
    
    # 配置版本，用于兼容性检查
    CONFIG_VERSION = "2.0.0"
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.global_config = GlobalConfig()
        self._schema_registry: Dict[str, List[ConfigSchema]] = {}
        self._loaded_configs: List[Path] = []
        
    def register_schema(self, module: str, schemas: List[ConfigSchema]) -> None:
        """注册配置模式"""
        self._schema_registry[module] = schemas
        logger.debug(f"Registered schema for module: {module}")
        
    def load_config(self, path: Union[str, Path], 
                    format: ConfigFormat = ConfigFormat.AUTO) -> GlobalConfig:
        """
        加载配置文件
        
        Args:
            path: 配置文件路径
            format: 配置格式
            
        Returns:
            GlobalConfig: 全局配置对象
        """
        path = Path(path)
        
        if not path.exists():
            raise ConfigNotFoundError(f"Configuration file not found: {path}")
        
        # 自动检测格式
        if format == ConfigFormat.AUTO:
            format = self._detect_format(path)
        
        # 加载配置
        with open(path, 'r', encoding='utf-8') as f:
            if format == ConfigFormat.YAML:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # 合并到全局配置
        self._merge_config(data)
        self._loaded_configs.append(path)
        
        logger.info(f"Loaded configuration from {path}")
        return self.global_config
    
    def load_from_env(self) -> GlobalConfig:
        """
        从环境变量加载配置
        
        环境变量格式: DFTLAMMPS_<SECTION>_<KEY>
        例如: DFTLAMMPS_DFT_ENCUT=520
        """
        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                parts = key[len(self.ENV_PREFIX):].lower().split('_')
                if len(parts) >= 2:
                    section = parts[0]
                    config_key = '_'.join(parts[1:])
                    self._set_config_value(section, config_key, value)
        
        logger.info("Loaded configuration from environment variables")
        return self.global_config
    
    def save_config(self, path: Union[str, Path], 
                   format: ConfigFormat = ConfigFormat.YAML) -> None:
        """保存配置到文件"""
        path = Path(path)
        data = self._config_to_dict(self.global_config)
        
        # 添加版本信息
        data['_version'] = self.CONFIG_VERSION
        data['_generated'] = self._get_timestamp()
        
        with open(path, 'w', encoding='utf-8') as f:
            if format == ConfigFormat.YAML:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved configuration to {path}")
    
    def validate_config(self, module: Optional[str] = None) -> List[str]:
        """
        验证配置
        
        Args:
            module: 指定模块验证，None则验证所有
            
        Returns:
            List[str]: 错误信息列表
        """
        errors = []
        
        modules_to_validate = [module] if module else list(self._schema_registry.keys())
        
        for mod in modules_to_validate:
            if mod not in self._schema_registry:
                continue
                
            schemas = self._schema_registry[mod]
            config_data = self._get_module_config(mod)
            
            for schema in schemas:
                value = config_data.get(schema.key, schema.default)
                if not schema.validate(value):
                    errors.append(
                        f"[{mod}] Invalid value for '{schema.key}': {value} "
                        f"(expected {schema.type.__name__})"
                    )
        
        return errors
    
    def get_config(self, module: Optional[str] = None) -> Union[GlobalConfig, ModuleConfig]:
        """获取配置"""
        if module is None:
            return self.global_config
        
        return getattr(self.global_config, module, None)
    
    def set_config(self, module: str, key: str, value: Any) -> None:
        """设置配置值"""
        self._set_config_value(module, key, value)
    
    def create_default_config(self, path: Union[str, Path]) -> None:
        """创建默认配置文件"""
        config = GlobalConfig()
        config.project_name = "My DFT+LAMMPS Project"
        
        # 设置合理的默认值
        config.dft.encut = 520.0
        config.dft.kpoints_density = 0.2
        config.lammps.timestep = 0.001
        config.ml_potential.rcut = 6.0
        config.workflow.max_iterations = 10
        config.hpc.max_nodes = 4
        
        self.global_config = config
        self.save_config(path, ConfigFormat.YAML)
    
    def _detect_format(self, path: Path) -> ConfigFormat:
        """检测配置文件格式"""
        suffix = path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix == '.json':
            return ConfigFormat.JSON
        else:
            # 默认尝试YAML
            return ConfigFormat.YAML
    
    def _merge_config(self, data: Dict[str, Any]) -> None:
        """合并配置数据"""
        for key, value in data.items():
            if key.startswith('_'):
                continue  # 跳过元数据
                
            if hasattr(self.global_config, key):
                if isinstance(value, dict):
                    current = getattr(self.global_config, key)
                    if is_dataclass(current):
                        for sub_key, sub_value in value.items():
                            if hasattr(current, sub_key):
                                setattr(current, sub_key, sub_value)
                    else:
                        setattr(self.global_config, key, value)
                else:
                    setattr(self.global_config, key, value)
            else:
                # 存储到自定义模块
                self.global_config.custom_modules[key] = value
    
    def _set_config_value(self, section: str, key: str, value: str) -> None:
        """设置配置值（支持嵌套）"""
        try:
            # 尝试转换为适当的类型
            value = self._convert_value(value)
            
            if hasattr(self.global_config, section):
                section_obj = getattr(self.global_config, section)
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                    logger.debug(f"Set config: {section}.{key} = {value}")
        except Exception as e:
            logger.warning(f"Failed to set config {section}.{key}: {e}")
    
    def _convert_value(self, value: str) -> Any:
        """转换字符串值为适当的类型"""
        # 尝试布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # 尝试整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 尝试浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 尝试列表（逗号分隔）
        if ',' in value:
            return [self._convert_value(v.strip()) for v in value.split(',')]
        
        return value
    
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        if is_dataclass(config):
            result = {}
            for key, value in asdict(config).items():
                result[key] = self._config_to_dict(value)
            return result
        elif isinstance(config, (list, tuple)):
            return [self._config_to_dict(item) for item in config]
        elif isinstance(config, dict):
            return {k: self._config_to_dict(v) for k, v in config.items()}
        else:
            return config
    
    def _get_module_config(self, module: str) -> Dict[str, Any]:
        """获取模块配置字典"""
        if hasattr(self.global_config, module):
            config = getattr(self.global_config, module)
            return asdict(config) if is_dataclass(config) else {}
        return self.global_config.custom_modules.get(module, {})
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_loaded_configs(self) -> List[Path]:
        """获取已加载的配置文件列表"""
        return self._loaded_configs.copy()


class ConfigBuilder:
    """配置构建器 - 链式API创建配置"""
    
    def __init__(self):
        self.config = GlobalConfig()
    
    def with_project(self, name: str, version: str = "1.0.0") -> 'ConfigBuilder':
        """设置项目信息"""
        self.config.project_name = name
        self.config.project_version = version
        return self
    
    def with_dft(self, calculator: str = "vasp", encut: float = 520.0,
                kpoints: float = 0.2, **kwargs) -> 'ConfigBuilder':
        """配置DFT设置"""
        self.config.dft.calculator = calculator
        self.config.dft.encut = encut
        self.config.dft.kpoints_density = kpoints
        for key, value in kwargs.items():
            if hasattr(self.config.dft, key):
                setattr(self.config.dft, key, value)
        return self
    
    def with_lammps(self, pair_style: str = "deepmd", timestep: float = 0.001,
                   temperature: float = 300.0, **kwargs) -> 'ConfigBuilder':
        """配置LAMMPS设置"""
        self.config.lammps.pair_style = pair_style
        self.config.lammps.timestep = timestep
        self.config.lammps.temperature = temperature
        for key, value in kwargs.items():
            if hasattr(self.config.lammps, key):
                setattr(self.config.lammps, key, value)
        return self
    
    def with_ml_potential(self, framework: str = "deepmd", rcut: float = 6.0,
                         **kwargs) -> 'ConfigBuilder':
        """配置ML势设置"""
        self.config.ml_potential.framework = framework
        self.config.ml_potential.rcut = rcut
        for key, value in kwargs.items():
            if hasattr(self.config.ml_potential, key):
                setattr(self.config.ml_potential, key, value)
        return self
    
    def with_hpc(self, cluster_type: str = "local", max_nodes: int = 4,
                **kwargs) -> 'ConfigBuilder':
        """配置HPC设置"""
        self.config.hpc.cluster_type = cluster_type
        self.config.hpc.max_nodes = max_nodes
        for key, value in kwargs.items():
            if hasattr(self.config.hpc, key):
                setattr(self.config.hpc, key, value)
        return self
    
    def with_logging(self, level: str = "INFO", file: str = "dftlammps.log",
                    **kwargs) -> 'ConfigBuilder':
        """配置日志设置"""
        self.config.logging.level = level
        self.config.logging.log_file = file
        for key, value in kwargs.items():
            if hasattr(self.config.logging, key):
                setattr(self.config.logging, key, value)
        return self
    
    def build(self) -> GlobalConfig:
        """构建配置"""
        return self.config


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(path: Union[str, Path], **kwargs) -> GlobalConfig:
    """便捷函数：加载配置"""
    return get_config_manager().load_config(path, **kwargs)


def load_from_env() -> GlobalConfig:
    """便捷函数：从环境变量加载"""
    return get_config_manager().load_from_env()


def create_default_config(path: Union[str, Path]) -> None:
    """便捷函数：创建默认配置"""
    get_config_manager().create_default_config(path)


# 预定义配置模板
DEFAULT_CONFIG_TEMPLATE = """
# DFT+LAMMPS 配置文件

project_name: "My DFT+LAMMPS Project"
project_version: "1.0.0"
debug_mode: false

# DFT 计算配置
dft:
  enabled: true
  version: "1.0.0"
  calculator: "vasp"
  pp_path: ""
  kpoints_density: 0.2
  encut: 520.0
  ediff: 1.0e-6
  nelm: 100
  ismear: 0
  sigma: 0.05
  ibrion: 2
  nsw: 100
  ediffg: -0.01
  lwave: false
  lcharg: false
  lreal: "Auto"
  ncore: 4
  kpar: 1

# LAMMPS 模拟配置
lammps:
  enabled: true
  version: "1.0.0"
  pair_style: "deepmd"
  potential_file: ""
  timestep: 0.001
  temperature: 300.0
  pressure: 1.0
  nsteps: 100000
  thermo_interval: 100
  dump_interval: 1000
  ensemble: "npt"
  use_gpu: false
  mpi_processes: 4
  omp_threads: 1

# 机器学习势配置
ml_potential:
  enabled: true
  version: "1.0.0"
  framework: "deepmd"
  descriptor_type: "se_e2_a"
  rcut: 6.0
  rcut_smth: 0.5
  sel: [50, 50]
  neuron: [25, 50, 100]
  axis_neuron: 16
  batch_size: 4
  num_steps: 1000000
  learning_rate: 0.001
  numb_steps: 1000000

# 工作流配置
workflow:
  enabled: true
  version: "1.0.0"
  max_iterations: 10
  convergence_threshold: 1.0e-5
  checkpoint_interval: 100
  auto_restart: true
  backup_enabled: true
  parallel_jobs: 4
  scheduler: "local"
  queue_name: ""
  walltime: "24:00:00"

# HPC 资源配置
hpc:
  enabled: true
  version: "1.0.0"
  cluster_type: "local"
  max_nodes: 4
  cores_per_node: 64
  gpus_per_node: 4
  priority: "normal"
  preemption: false
  scratch_path: "/tmp"
  shared_fs: "/shared"
  monitoring_enabled: true
  metrics_interval: 60

# 数据库配置
database:
  enabled: true
  version: "1.0.0"
  backend: "sqlite"
  host: "localhost"
  port: 5432
  database: "dftlammps"
  username: ""
  password: ""
  db_path: "./dftlammps.db"
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30

# 日志配置
logging:
  enabled: true
  version: "1.0.0"
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_enabled: true
  log_file: "dftlammps.log"
  max_bytes: 10485760
  backup_count: 5
  remote_logging: false
  remote_host: ""
  remote_port: 514
"""


if __name__ == "__main__":
    # 测试配置系统
    logging.basicConfig(level=logging.DEBUG)
    
    # 使用构建器创建配置
    builder = ConfigBuilder()
    config = (builder
        .with_project("Test Project", "1.0.0")
        .with_dft(calculator="vasp", encut=600.0)
        .with_lammps(timestep=0.0005, temperature=500.0)
        .with_ml_potential(framework="mace")
        .with_hpc(cluster_type="slurm", max_nodes=8)
        .build())
    
    print(f"Project: {config.project_name}")
    print(f"DFT ENCUT: {config.dft.encut}")
    print(f"LAMMPS Timestep: {config.lammps.timestep}")
    print(f"ML Framework: {config.ml_potential.framework}")
