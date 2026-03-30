"""
DFT-LAMMPS 统一配置系统
============================

使用 OmegaConf 管理所有配置，支持：
1. 层次化配置继承
2. 配置文件覆盖
3. 环境变量注入
4. 配置验证

Usage:
    from core.config import get_config, DFTConfig, MDConfig
    
    config = get_config()
    print(config.dft.vasp.encut)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import os

from omegaconf import OmegaConf, MISSING, DictConfig


# =============================================================================
# 基础配置类
# =============================================================================

@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True
    json_format: bool = False


@dataclass
class HardwareConfig:
    """硬件资源配置"""
    ncores: int = 1
    ngpus: int = 0
    memory_gb: float = 4.0
    use_mpi: bool = False
    use_openmp: bool = False


@dataclass
class DatabaseConfig:
    """数据库配置"""
    type: str = "sqlite"  # sqlite, postgresql, mongodb
    url: Optional[str] = None
    name: str = "dft_lammps.db"
    echo: bool = False


# =============================================================================
# DFT 配置
# =============================================================================

@dataclass
class VaspConfig:
    """VASP 特定配置"""
    encut: float = 520.0
    ediff: float = 1e-6
    ediffg: float = -0.01
    ismear: int = 0
    sigma: float = 0.05
    ibrion: int = 2
    nsw: int = 100
    isif: int = 3
    lreal: str = "Auto"
    lwave: bool = True
    lcharg: bool = True
    lorbit: int = 11
    nelm: int = 100
    nelmin: int = 4
    
    # 并行设置
    npar: Optional[int] = None
    kpar: int = 1
    
    # 路径
    vasp_cmd: str = "vasp_std"
    vasp_gam_cmd: str = "vasp_gam"


@dataclass
class QuantumEspressoConfig:
    """Quantum ESPRESSO 特定配置"""
    pw_cmd: str = "pw.x"
    pseudo_dir: str = "./pseudo"
    calculation: str = "scf"
    ecutwfc: float = 50.0
    ecutrho: float = 200.0
    conv_thr: float = 1e-8
    mixing_beta: float = 0.7
    electron_maxstep: int = 100


@dataclass
class DFTConfig:
    """DFT 计算通用配置"""
    code: str = "vasp"  # vasp, espresso, abacus
    functional: str = "PBE"
    kpoints_density: float = 0.25
    spin_polarized: bool = False
    
    # 子配置
    vasp: VaspConfig = field(default_factory=VaspConfig)
    espresso: QuantumEspressoConfig = field(default_factory=QuantumEspressoConfig)
    
    # 硬件
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


# =============================================================================
# MD 配置
# =============================================================================

@dataclass
class LammpsConfig:
    """LAMMPS 特定配置"""
    lmp_cmd: str = "lmp"
    pair_style: str = "ne"
    pair_coeff: str = "* * potential nep"
    atom_style: str = "atomic"
    units: str = "metal"
    boundary: str = "p p p"
    timestep: float = 1.0  # fs
    
    # 邻居列表
    neigh_modify: str = "every 1 delay 0 check yes"
    neighbor: str = "2.0 bin"
    
    # 热浴
    temperature: float = 300.0  # K
    pressure: Optional[float] = None  # bar
    thermostat: str = "nose-hoover"
    barostat: Optional[str] = None
    tdamp: float = 100.0
    pdamp: float = 1000.0


@dataclass
class MDConfig:
    """分子动力学通用配置"""
    engine: str = "lammps"
    ensemble: str = "nvt"  # nvt, npt, nve
    nsteps: int = 10000
    
    # 输出
    dump_freq: int = 100
    thermo_freq: int = 10
    
    # 子配置
    lammps: LammpsConfig = field(default_factory=LammpsConfig)
    
    # 硬件
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


# =============================================================================
# ML 势配置
# =============================================================================

@dataclass
class NEPConfig:
    """NEP 势训练配置"""
    version: int = 4
    type: int = 0  # 0=radial, 1=full
    
    # 网络结构
    neuron: int = 30
    layer: int = 3
    batch_size: int = 1000
    learning_rate: float = 0.001
    
    # 训练参数
    num_epochs: int = 10000
    force_weight: float = 1.0
    virial_weight: float = 0.1
    energy_weight: float = 1.0
    
    # 描述符
    cutoff: float = 6.0
    n_max: int = 4
    l_max: int = 4
    
    # 优化器
    optimizer: str = "adam"
    lr_decay: float = 0.95
    stop_lr: float = 1e-7


@dataclass
class DeepMDConfig:
    """DeepMD-kit 配置"""
    descriptor_type: str = "se_e2_a"
    rcut: float = 6.0
    rcut_smth: float = 0.5
    sel: List[int] = field(default_factory=lambda: [46, 92])
    neuron: List[int] = field(default_factory=lambda: [25, 50, 100])
    axis_neuron: int = 16
    
    # 训练
    numb_steps: int = 1000000
    batch_size: str = "auto"
    learning_rate: float = 0.001
    start_lr: float = 0.001
    stop_lr: float = 3.51e-8


@dataclass
class MLPotentialConfig:
    """机器学习势通用配置"""
    type: str = "nep"  # nep, deepmd, mace, chgnet
    
    # 数据
    data_dir: str = "./training_data"
    train_ratio: float = 0.9
    validation_ratio: float = 0.05
    
    # 子配置
    nep: NEPConfig = field(default_factory=NEPConfig)
    deepmd: DeepMDConfig = field(default_factory=DeepMDConfig)
    
    # 硬件
    hardware: HardwareConfig = field(default_factory=lambda: HardwareConfig(
        ncores=4, ngpus=1, use_mpi=True
    ))


# =============================================================================
# 工作流配置
# =============================================================================

@dataclass
class MaterialsProjectConfig:
    """Materials Project API 配置"""
    api_key: Optional[str] = None
    base_url: str = "https://api.materialsproject.org"
    chunk_size: int = 1000
    max_entries: int = 100


@dataclass
class ScreeningConfig:
    """高通量筛选配置"""
    # 筛选标准
    max_atoms: int = 100
    max_energy_above_hull: float = 0.1  # eV/atom
    min_band_gap: Optional[float] = None
    max_band_gap: Optional[float] = None
    
    # 批次
    batch_size: int = 10
    max_concurrent: int = 5
    
    # Materials Project
    mp: MaterialsProjectConfig = field(default_factory=MaterialsProjectConfig)


@dataclass
class WorkflowConfig:
    """工作流通用配置"""
    name: str = "default_workflow"
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    save_intermediate: bool = True
    
    # 阶段控制
    stages: List[str] = field(default_factory=lambda: [
        "structure_preparation",
        "dft_calculation", 
        "ml_training",
        "md_simulation",
        "analysis"
    ])


# =============================================================================
# 根配置
# =============================================================================

@dataclass
class GlobalConfig:
    """全局配置根对象"""
    
    # 元信息
    project_name: str = "dft_lammps_research"
    version: str = "2.0.0"
    debug: bool = False
    
    # 子系统配置
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # 计算模块
    dft: DFTConfig = field(default_factory=DFTConfig)
    md: MDConfig = field(default_factory=MDConfig)
    ml: MLPotentialConfig = field(default_factory=MLPotentialConfig)
    
    # 工作流
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    
    # 硬件默认值
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


# =============================================================================
# 配置加载与管理
# =============================================================================

class ConfigManager:
    """配置管理器单例"""
    
    _instance: Optional[DictConfig] = None
    _config_path: Optional[Path] = None
    
    @classmethod
    def get_config(cls, config_path: Optional[Union[str, Path]] = None) -> DictConfig:
        """获取配置实例
        
        Args:
            config_path: 配置文件路径，如果不指定则使用默认值
            
        Returns:
            OmegaConf DictConfig 对象
        """
        if cls._instance is None:
            # 创建默认配置
            schema = OmegaConf.structured(GlobalConfig)
            
            # 从文件加载（如果提供）
            if config_path:
                file_config = OmegaConf.load(config_path)
                schema = OmegaConf.merge(schema, file_config)
            
            # 从环境变量加载
            env_config = cls._load_from_env()
            if env_config:
                schema = OmegaConf.merge(schema, env_config)
            
            cls._instance = schema
            cls._config_path = Path(config_path) if config_path else None
        
        return cls._instance
    
    @classmethod
    def _load_from_env(cls) -> Optional[DictConfig]:
        """从环境变量加载配置"""
        env_vars = {}
        
        # 映射环境变量到配置路径
        mappings = {
            "DFT_CODE": "dft.code",
            "DFT_VASP_ENCUT": "dft.vasp.encut",
            "DFT_FUNCTIONAL": "dft.functional",
            "ML_TYPE": "ml.type",
            "ML_NEP_NEURON": "ml.nep.neuron",
            "MP_API_KEY": "screening.mp.api_key",
            "NCORES": "hardware.ncores",
            "NGPUS": "hardware.ngpus",
        }
        
        for env_key, conf_path in mappings.items():
            value = os.getenv(env_key)
            if value is not None:
                # 尝试转换为数字
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # 保持字符串
                
                env_vars[conf_path] = value
        
        if env_vars:
            return OmegaConf.create(env_vars)
        return None
    
    @classmethod
    def reload(cls, config_path: Optional[Union[str, Path]] = None) -> DictConfig:
        """重新加载配置"""
        cls._instance = None
        return cls.get_config(config_path)
    
    @classmethod
    def save(cls, path: Union[str, Path], resolve: bool = True) -> None:
        """保存当前配置到文件"""
        if cls._instance is None:
            raise RuntimeError("Config not loaded yet")
        
        config_to_save = OmegaConf.to_container(cls._instance, resolve=resolve)
        OmegaConf.save(config_to_save, path)


# =============================================================================
# 便捷函数
# =============================================================================

def get_config(config_path: Optional[Union[str, Path]] = None) -> DictConfig:
    """获取全局配置
    
    Usage:
        config = get_config()
        print(config.dft.vasp.encut)
        
        # 或从文件加载
        config = get_config("config.yaml")
    """
    return ConfigManager.get_config(config_path)


def reload_config(config_path: Optional[Union[str, Path]] = None) -> DictConfig:
    """重新加载配置"""
    return ConfigManager.reload(config_path)


def save_config(path: Union[str, Path], resolve: bool = True) -> None:
    """保存配置到文件"""
    ConfigManager.save(path, resolve)


# 兼容性函数 - 用于替换原有的分散配置
def get_dft_config() -> DFTConfig:
    """获取DFT配置（兼容旧代码）"""
    return get_config().dft


def get_md_config() -> MDConfig:
    """获取MD配置（兼容旧代码）"""
    return get_config().md


def get_ml_config() -> MLPotentialConfig:
    """获取ML配置（兼容旧代码）"""
    return get_config().ml


# =============================================================================
# 示例配置
# =============================================================================

EXAMPLE_CONFIG_YAML = """
# DFT-LAMMPS 全局配置文件示例

project_name: "my_materials_project"
debug: false

# 日志配置
logging:
  level: INFO
  file: logs/workflow.log
  json_format: false

# 硬件配置
hardware:
  ncores: 32
  ngpus: 2
  use_mpi: true

# DFT 配置
dft:
  code: vasp
  functional: PBE
  
  vasp:
    encut: 600
    ediff: 1.0e-7
    kpar: 4
    npar: 4

# MD 配置
md:
  engine: lammps
  ensemble: npt
  nsteps: 100000
  
  lammps:
    timestep: 1.0
    temperature: 300
    pressure: 1.0
    barostat: nose-hoover

# ML 势配置
ml:
  type: nep
  
  nep:
    neuron: 50
    layer: 4
    num_epochs: 20000
    learning_rate: 0.001

# 工作流配置
workflow:
  output_dir: ./results
  save_intermediate: true
  
screening:
  batch_size: 20
  mp:
    api_key: ${oc.env:MP_API_KEY,null}
    max_entries: 500
"""


if __name__ == "__main__":
    # 测试配置系统
    config = get_config()
    print("Default VASP ENCUT:", config.dft.vasp.encut)
    print("Default NEP Neuron:", config.ml.nep.neuron)
    
    # 转换为YAML查看
    print("\n--- Default Config as YAML ---")
    print(OmegaConf.to_yaml(config))
