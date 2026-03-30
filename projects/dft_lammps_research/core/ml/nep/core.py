"""
nep_training/core.py
====================
NEP训练核心配置与数据类
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """精度模式"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    MIXED = "mixed"


class TrainingState(Enum):
    """训练状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    EVALUATING = "evaluating"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class NEPDataConfig:
    """
    NEP数据配置
    
    支持多种数据源格式和高级数据预处理
    """
    # 输入数据源
    vasp_outcars: List[str] = field(default_factory=list)
    dft_trajectories: List[str] = field(default_factory=list)
    existing_xyz: Optional[str] = None
    deepmd_data: Optional[str] = None  # DeepMD格式数据
    ase_db: Optional[str] = None  # ASE数据库
    
    # 数据增强
    augment_data: bool = True
    rotation_augment: bool = True
    translation_augment: bool = False
    noise_augment: bool = True
    noise_std: float = 0.01  # 位置噪声标准差 (Å)
    
    # 数据处理
    energy_threshold: float = 50.0  # eV/atom, 异常值过滤
    force_threshold: float = 50.0  # eV/Å, 异常力过滤
    min_force: float = 0.001  # 最小力值
    max_force: float = 100.0  # 最大力值
    normalize_energy: bool = True  # 能量归一化
    energy_reference: Optional[Dict[str, float]] = None  # 元素参考能量
    
    # 数据集分割
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    stratified_split: bool = True  # 按结构类型分层分割
    
    # 元素映射
    type_map: List[str] = field(default_factory=list)
    
    # 缓存
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    def validate(self) -> bool:
        """验证配置有效性"""
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        assert len(self.type_map) > 0, "type_map must not be empty"
        return True


@dataclass  
class NEPModelConfig:
    """
    NEP模型配置 (支持NEP 2/3/4)
    
    参考 GPUMD 官方文档配置参数
    """
    # 模型类型
    model_type: int = 0  # 0=PES, 1=dipole, 2=polarizability
    
    # 元素类型
    type_list: List[str] = field(default_factory=list)
    
    # NEP版本
    version: int = 4  # NEP版本 (2, 3, or 4)
    
    # 描述符参数 - 径向
    cutoff_radial: float = 6.0  # 径向截断 (Å)
    n_max_radial: int = 4  # 径向基函数数
    basis_size_radial: int = 8  # 径向基大小 (NEP4)
    
    # 描述符参数 - 角向
    cutoff_angular: float = 4.0  # 角向截断 (Å)
    n_max_angular: int = 4  # 角向基函数数
    basis_size_angular: int = 8  # 角向基大小 (NEP4)
    
    # 多体相互作用
    l_max_3body: int = 4  # 3-body最大角动量
    l_max_4body: int = 0  # 4-body最大角动量 (0=禁用)
    l_max_5body: int = 0  # 5-body最大角动量 (0=禁用)
    
    # 神经网络参数
    neuron: int = 30  # 隐藏层神经元数
    hidden_layers: int = 2  # 隐藏层数
    activation: str = "tanh"  # 激活函数
    
    # 优化参数 - SNES
    population_size: int = 50  # SNES种群大小
    maximum_generation: int = 100000  # 最大迭代代数
    batch_size: int = 1000  # 批量大小
    
    # 学习率参数 (高级训练)
    initial_lr: float = 0.1
    min_lr: float = 1e-6
    lr_decay_steps: int = 10000
    lr_decay_rate: float = 0.95
    
    # 正则化
    l2_regularization: float = 0.0
    dropout_rate: float = 0.0
    
    # 权重初始化
    weight_init: str = "xavier"
    
    def to_nep_in_dict(self) -> Dict[str, Any]:
        """转换为nep.in参数字典"""
        return {
            'type': ' '.join(self.type_list),
            'model_type': self.model_type if self.model_type != 0 else None,
            'version': self.version,
            'cutoff': f"{self.cutoff_radial} {self.cutoff_angular}",
            'n_max': f"{self.n_max_radial} {self.n_max_angular}",
            'basis_size': f"{self.basis_size_radial} {self.basis_size_angular}" if self.version >= 4 else None,
            'l_max': f"{self.l_max_3body} {self.l_max_4body} {self.l_max_5body}".rstrip(),
            'neuron': self.neuron,
            'population': self.population_size,
            'generation': self.maximum_generation,
            'batch': self.batch_size,
        }
    
    def save(self, path: str):
        """保存配置到JSON"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "NEPModelConfig":
        """从JSON加载配置"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class DistributedConfig:
    """
    分布式训练配置
    
    支持多GPU并行训练
    """
    enabled: bool = False
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    num_workers: int = 4
    
    def __post_init__(self):
        # 从环境变量读取分布式配置
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        if "MASTER_ADDR" in os.environ:
            self.master_addr = os.environ["MASTER_ADDR"]
        if "MASTER_PORT" in os.environ:
            self.master_port = os.environ["MASTER_PORT"]


@dataclass
class NEPTrainingConfig:
    """
    NEP训练配置 (增强版)
    
    整合高级训练功能
    """
    # 路径
    gpumd_path: str = "/path/to/gpumd"
    working_dir: str = "./nep_training"
    output_dir: str = "./nep_output"
    checkpoint_dir: str = "./nep_checkpoints"
    
    # 文件
    train_xyz: str = "train.xyz"
    val_xyz: str = "val.xyz"
    test_xyz: str = "test.xyz"
    nep_in: str = "nep.in"
    
    # 计算设置
    use_gpu: bool = True
    gpu_id: int = 0
    
    # 多精度训练
    precision: PrecisionMode = PrecisionMode.FP32
    
    # 分布式训练
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    # 高级训练策略
    use_lr_scheduler: bool = True
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-5
    
    # 模型集成
    use_ensemble: bool = False
    ensemble_size: int = 4
    ensemble_bootstrap: bool = True
    
    # 检查点
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1000  # 每N代保存
    keep_best_only: bool = False
    
    # 验证
    validate_frequency: int = 100  # 每N代验证
    
    # 重启选项
    restart: bool = False
    checkpoint_path: Optional[str] = None
    
    # 日志
    log_frequency: int = 10
    log_level: str = "INFO"
    
    # 性能优化
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    
    def __post_init__(self):
        """创建必要目录"""
        for path in [self.working_dir, self.output_dir, self.checkpoint_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class NEPCheckpoint:
    """
    NEP训练检查点
    
    支持完整训练状态保存与恢复
    """
    generation: int
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    best_loss: float
    train_loss_history: List[float]
    val_loss_history: List[float]
    metrics: Dict[str, float]
    timestamp: str
    config: NEPTrainingConfig
    
    def save(self, path: str):
        """保存检查点"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "NEPCheckpoint":
        """加载检查点"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


# 预设配置
NEP_PRESETS = {
    "fast": {
        "version": 3,
        "n_max_radial": 4,
        "n_max_angular": 4,
        "l_max_3body": 4,
        "neuron": 10,
        "population_size": 30,
        "maximum_generation": 10000,
        "description": "快速训练，适合小数据集和初步测试"
    },
    "balanced": {
        "version": 4,
        "n_max_radial": 6,
        "n_max_angular": 6,
        "basis_size_radial": 8,
        "basis_size_angular": 8,
        "l_max_3body": 4,
        "neuron": 30,
        "population_size": 50,
        "maximum_generation": 100000,
        "description": "平衡配置，适合大多数应用场景"
    },
    "accurate": {
        "version": 4,
        "n_max_radial": 6,
        "n_max_angular": 6,
        "basis_size_radial": 12,
        "basis_size_angular": 12,
        "l_max_3body": 6,
        "neuron": 50,
        "population_size": 100,
        "maximum_generation": 1000000,
        "description": "高精度配置，适合复杂系统和最终模型"
    },
    "light": {
        "version": 3,
        "n_max_radial": 4,
        "n_max_angular": 2,
        "l_max_3body": 2,
        "neuron": 5,
        "population_size": 20,
        "maximum_generation": 50000,
        "description": "轻量级模型，推理速度快，适合在线MD"
    },
    "transfer": {
        "version": 4,
        "n_max_radial": 6,
        "n_max_angular": 6,
        "basis_size_radial": 8,
        "basis_size_angular": 8,
        "l_max_3body": 4,
        "neuron": 30,
        "population_size": 20,  # 迁移学习使用小种群
        "maximum_generation": 20000,
        "description": "迁移学习配置，基于预训练模型微调"
    }
}


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """获取预设配置"""
    if preset_name not in NEP_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(NEP_PRESETS.keys())}")
    return NEP_PRESETS[preset_name].copy()
