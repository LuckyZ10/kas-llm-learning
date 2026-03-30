#!/usr/bin/env python3
"""
advanced_potentials/__init__.py
==============================
先进机器学习势能统一接口

整合以下ML势：
- MACE: 高阶等变消息传递势
- CHGNet: 晶体哈密顿图神经网络势
- Orb: 轨道表示超快推理势
- DeePMD/NEP: 现有接口兼容

提供统一的API用于：
- 模型加载
- 能量/力/应力预测
- MD模拟
- 结构优化
- 高通量筛选

作者: ML Potential Integration Expert
日期: 2026-03-09
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator

# 导入各ML势接口
from .mace_interface import (
    MACECalculator, MACEWorkflow, MACEConfig, MACEMDConfig,
    MACEDatasetPreparer, MACEActiveLearning
)

from .chgnet_interface import (
    CHGNetASECalculator, CHGNetWorkflow, CHGNetConfig,
    CHGNetPrediction, MagneticProperties
)

from .orb_interface import (
    OrbASECalculator, OrbWorkflow, OrbConfig, OrbMDConfig,
    OrbBatchPredictor, OrbBenchmarkResult
)

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================

class MLPotentialType(Enum):
    """ML势类型枚举"""
    MACE = "mace"
    CHGNET = "chgnet"
    ORB = "orb"
    DEEPMD = "deepmd"
    NEP = "nep"


class MLPotentialCapability(Enum):
    """ML势能力枚举"""
    ENERGY = "energy"
    FORCES = "forces"
    STRESS = "stress"
    MAGMOM = "magmom"
    CHARGE = "charge"
    FAST_INFERENCE = "fast_inference"


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class UnifiedMLPotentialConfig:
    """统一的ML势配置"""
    potential_type: MLPotentialType = MLPotentialType.MACE
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    
    # 计算设置
    device: str = "cuda"  # cuda, cpu
    precision: str = "float32"  # float32, float64
    
    # 性能设置
    batch_size: int = 32
    use_cache: bool = True
    
    # 特定势的参数
    mace_config: Optional[MACEConfig] = None
    chgnet_config: Optional[CHGNetConfig] = None
    orb_config: Optional[OrbConfig] = None


@dataclass
class MLPredictionResult:
    """ML预测结果"""
    energy: float = 0.0
    forces: Optional[Any] = None
    stress: Optional[Any] = None
    magmom: Optional[Any] = None
    
    # 元数据
    potential_type: str = ""
    inference_time: float = 0.0
    success: bool = True
    error_message: str = ""


@dataclass
class MLPotentialInfo:
    """ML势信息"""
    name: str
    version: str
    capabilities: List[MLPotentialCapability]
    training_data_sources: List[str]
    typical_accuracy: Dict[str, float]  # e.g., {"energy": 0.01, "forces": 0.05}
    inference_speed: str  # "slow", "medium", "fast", "ultrafast"
    recommended_use_cases: List[str]


# =============================================================================
# ML势能力数据库
# =============================================================================

ML_POTENTIAL_DATABASE = {
    MLPotentialType.MACE: MLPotentialInfo(
        name="MACE",
        version="2024.1",
        capabilities=[
            MLPotentialCapability.ENERGY,
            MLPotentialCapability.FORCES,
            MLPotentialCapability.STRESS
        ],
        training_data_sources=["Materials Project", "Alexandria", "Custom"],
        typical_accuracy={"energy": 0.01, "forces": 0.05},
        inference_speed="medium",
        recommended_use_cases=[
            "High-accuracy MD simulations",
            "Materials property prediction",
            "Active learning"
        ]
    ),
    
    MLPotentialType.CHGNET: MLPotentialInfo(
        name="CHGNet",
        version="0.3.0",
        capabilities=[
            MLPotentialCapability.ENERGY,
            MLPotentialCapability.FORCES,
            MLPotentialCapability.STRESS,
            MLPotentialCapability.MAGMOM
        ],
        training_data_sources=["Materials Project"],
        typical_accuracy={"energy": 0.02, "forces": 0.07, "magmom": 0.1},
        inference_speed="fast",
        recommended_use_cases=[
            "Magnetic materials",
            "High-throughput screening",
            "Structure relaxation"
        ]
    ),
    
    MLPotentialType.ORB: MLPotentialInfo(
        name="Orb",
        version="v2",
        capabilities=[
            MLPotentialCapability.ENERGY,
            MLPotentialCapability.FORCES,
            MLPotentialCapability.STRESS,
            MLPotentialCapability.FAST_INFERENCE
        ],
        training_data_sources=["Materials Project", "OMAT"],
        typical_accuracy={"energy": 0.015, "forces": 0.06},
        inference_speed="ultrafast",
        recommended_use_cases=[
            "Large-scale MD simulations",
            "Real-time simulations",
            "High-throughput screening",
            "Online learning"
        ]
    ),
    
    MLPotentialType.DEEPMD: MLPotentialInfo(
        name="DeePMD",
        version="2.x",
        capabilities=[
            MLPotentialCapability.ENERGY,
            MLPotentialCapability.FORCES,
            MLPotentialCapability.STRESS
        ],
        training_data_sources=["Custom DFT"],
        typical_accuracy={"energy": 0.005, "forces": 0.03},
        inference_speed="medium",
        recommended_use_cases=[
            "Custom material systems",
            "Specialized potentials",
            "Transfer learning"
        ]
    ),
    
    MLPotentialType.NEP: MLPotentialInfo(
        name="NEP",
        version="4.0",
        capabilities=[
            MLPotentialCapability.ENERGY,
            MLPotentialCapability.FORCES,
            MLPotentialCapability.STRESS
        ],
        training_data_sources=["Custom DFT"],
        typical_accuracy={"energy": 0.008, "forces": 0.04},
        inference_speed="fast",
        recommended_use_cases=[
            "GPU-accelerated MD",
            "Large systems",
            "Long-time simulations"
        ]
    )
}


# =============================================================================
# 统一ML势计算器
# =============================================================================

class UnifiedMLPotentialCalculator(Calculator):
    """
    统一的ML势计算器
    
    提供统一的接口访问不同的ML势
    """
    
    implemented_properties = ['energy', 'forces', 'stress', 'magmom', 'free_energy']
    
    def __init__(self, config: UnifiedMLPotentialConfig, **kwargs):
        """
        初始化统一ML势计算器
        
        Args:
            config: ML势配置
        """
        super().__init__(**kwargs)
        
        self.config = config
        self.potential_type = config.potential_type
        self._calculator = None
        self._workflow = None
        
        # 初始化底层计算器
        self._init_calculator()
    
    def _init_calculator(self):
        """初始化底层计算器"""
        cfg = self.config
        
        if self.potential_type == MLPotentialType.MACE:
            mace_cfg = cfg.mace_config or MACEConfig(
                model_type=cfg.model_name or "medium",
                device=cfg.device,
                default_dtype=cfg.precision
            )
            
            if cfg.model_path:
                mace_cfg.model_path = cfg.model_path
            
            self._workflow = MACEWorkflow(mace_cfg)
            self._calculator = self._workflow.setup_calculator()
            
        elif self.potential_type == MLPotentialType.CHGNET:
            chgnet_cfg = cfg.chgnet_config or CHGNetConfig(
                use_device=cfg.device,
                model_name=cfg.model_name or "0.3.0"
            )
            
            if cfg.model_path:
                chgnet_cfg.model_path = cfg.model_path
            
            self._workflow = CHGNetWorkflow(chgnet_cfg)
            self._calculator = self._workflow.setup()
            
        elif self.potential_type == MLPotentialType.ORB:
            orb_cfg = cfg.orb_config or OrbConfig(
                device=cfg.device,
                precision=cfg.precision
            )
            
            if cfg.model_path:
                orb_cfg.model_path = cfg.model_path
            
            self._workflow = OrbWorkflow(orb_cfg)
            self._calculator = self._workflow.setup()
            
        else:
            raise ValueError(f"Unsupported potential type: {self.potential_type}")
        
        logger.info(f"Initialized {self.potential_type.value} calculator")
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        """执行计算"""
        super().calculate(atoms, properties, system_changes)
        
        if atoms is None:
            atoms = self.atoms
        
        # 使用底层计算器
        atoms.calc = self._calculator
        
        if 'energy' in properties:
            self.results['energy'] = atoms.get_potential_energy()
            self.results['free_energy'] = self.results['energy']
        
        if 'forces' in properties:
            self.results['forces'] = atoms.get_forces()
        
        if 'stress' in properties:
            try:
                self.results['stress'] = atoms.get_stress()
            except:
                self.results['stress'] = None
        
        if 'magmom' in properties and self.potential_type == MLPotentialType.CHGNET:
            try:
                mag_props = self._calculator.get_magmom_prediction(atoms)
                self.results['magmom'] = mag_props.atomic_magmoms
                self.results['total_magmom'] = mag_props.total_magmom
            except:
                pass
    
    def get_capabilities(self) -> List[MLPotentialCapability]:
        """获取当前势的能力列表"""
        info = ML_POTENTIAL_DATABASE.get(self.potential_type)
        return info.capabilities if info else []
    
    def supports(self, capability: MLPotentialCapability) -> bool:
        """检查是否支持特定能力"""
        return capability in self.get_capabilities()


# =============================================================================
# ML势选择器
# =============================================================================

class MLPotentialSelector:
    """
    ML势选择器
    
    根据任务需求推荐最佳ML势
    """
    
    @staticmethod
    def recommend(use_case: str, 
                  priority_capabilities: Optional[List[MLPotentialCapability]] = None,
                  speed_preference: str = "balanced") -> MLPotentialType:
        """
        推荐最适合的ML势
        
        Args:
            use_case: 使用场景描述
            priority_capabilities: 必需的能力
            speed_preference: "accuracy", "balanced", "speed"
        
        Returns:
            推荐的ML势类型
        """
        use_case_lower = use_case.lower()
        
        # 基于使用场景推荐
        if "magnetic" in use_case_lower or "magmom" in use_case_lower:
            return MLPotentialType.CHGNET
        
        if "large scale" in use_case_lower or "real time" in use_case_lower:
            return MLPotentialType.ORB
        
        if "high accuracy" in use_case_lower or "active learning" in use_case_lower:
            return MLPotentialType.MACE
        
        # 基于速度偏好
        if speed_preference == "speed":
            return MLPotentialType.ORB
        elif speed_preference == "accuracy":
            return MLPotentialType.MACE
        
        # 基于能力需求
        if priority_capabilities:
            if MLPotentialCapability.MAGMOM in priority_capabilities:
                return MLPotentialType.CHGNET
            if MLPotentialCapability.FAST_INFERENCE in priority_capabilities:
                return MLPotentialType.ORB
        
        # 默认推荐
        return MLPotentialType.MACE
    
    @staticmethod
    def compare_potentials(potential_types: Optional[List[MLLPotentialType]] = None) -> Dict:
        """
        比较不同ML势的特点
        
        Returns:
            比较信息字典
        """
        if potential_types is None:
            potential_types = list(MLPotentialType)
        
        comparison = {}
        
        for pt in potential_types:
            info = ML_POTENTIAL_DATABASE.get(pt)
            if info:
                comparison[pt.value] = {
                    'capabilities': [c.value for c in info.capabilities],
                    'accuracy': info.typical_accuracy,
                    'speed': info.inference_speed,
                    'use_cases': info.recommended_use_cases
                }
        
        return comparison
    
    @staticmethod
    def get_potential_info(potential_type: MLPotentialType) -> Optional[MLPotentialInfo]:
        """获取ML势详细信息"""
        return ML_POTENTIAL_DATABASE.get(potential_type)


# =============================================================================
# 统一工作流
# =============================================================================

class UnifiedMLPotentialWorkflow:
    """
    统一的ML势工作流
    
    提供跨ML势的统一工作流程
    """
    
    def __init__(self, config: UnifiedMLPotentialConfig):
        self.config = config
        self.calculator = UnifiedMLPotentialCalculator(config)
        self.workflow = self.calculator._workflow
    
    def predict(self, atoms: Atoms) -> MLPredictionResult:
        """
        预测单个结构
        """
        import time
        
        start_time = time.time()
        
        try:
            atoms.calc = self.calculator._calculator
            
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            
            stress = None
            try:
                stress = atoms.get_stress()
            except:
                pass
            
            magmom = None
            if self.config.potential_type == MLPotentialType.CHGNET:
                try:
                    mag_props = self.calculator._calculator.get_magmom_prediction(atoms)
                    magmom = mag_props.atomic_magmoms
                except:
                    pass
            
            inference_time = time.time() - start_time
            
            return MLPredictionResult(
                energy=energy,
                forces=forces,
                stress=stress,
                magmom=magmom,
                potential_type=self.config.potential_type.value,
                inference_time=inference_time,
                success=True
            )
        
        except Exception as e:
            return MLPredictionResult(
                potential_type=self.config.potential_type.value,
                success=False,
                error_message=str(e)
            )
    
    def batch_predict(self, structures: List[Atoms]) -> List[MLPredictionResult]:
        """
        批量预测
        """
        results = []
        
        for atoms in structures:
            result = self.predict(atoms)
            results.append(result)
        
        return results
    
    def relax_structure(self, 
                       atoms: Atoms,
                       fmax: float = 0.01,
                       max_steps: int = 500) -> Atoms:
        """
        结构弛豫
        """
        if hasattr(self.workflow, 'relax_structure'):
            return self.workflow.relax_structure(atoms, fmax=fmax)
        elif hasattr(self.workflow, 'relax'):
            return self.workflow.relax(atoms, fmax=fmax, max_steps=max_steps)
        else:
            # 通用弛豫
            from ase.optimize import LBFGS
            atoms = atoms.copy()
            atoms.calc = self.calculator._calculator
            opt = LBFGS(atoms)
            opt.run(fmax=fmax, steps=max_steps)
            return atoms
    
    def run_md(self,
              initial_structure: Union[str, Atoms],
              temperature: float = 300.0,
              n_steps: int = 10000,
              timestep: float = 1.0) -> Any:
        """
        运行MD模拟
        """
        if self.config.potential_type == MLPotentialType.MACE:
            from .mace_interface import MACEMDConfig
            md_config = MACEMDConfig(
                temperature=temperature,
                n_steps=n_steps,
                timestep=timestep
            )
            return self.workflow.run_md_simulation(initial_structure, md_config)
        
        elif self.config.potential_type == MLPotentialType.ORB:
            from .orb_interface import OrbMDConfig
            md_config = OrbMDConfig(
                temperature=temperature,
                n_steps=n_steps,
                timestep=timestep
            )
            return self.workflow.run_md(initial_structure, md_config)
        
        else:
            raise NotImplementedError(f"MD not implemented for {self.config.potential_type}")


# =============================================================================
# 便捷函数
# =============================================================================

def load_ml_potential(potential_type: Union[str, MLPotentialType],
                     model_path: Optional[str] = None,
                     **kwargs) -> UnifiedMLPotentialCalculator:
    """
    便捷函数：加载ML势
    
    Args:
        potential_type: ML势类型 ("mace", "chgnet", "orb", etc.)
        model_path: 模型文件路径
        **kwargs: 其他参数
    
    Returns:
        UnifiedMLPotentialCalculator
    """
    if isinstance(potential_type, str):
        potential_type = MLPotentialType(potential_type.lower())
    
    config = UnifiedMLPotentialConfig(
        potential_type=potential_type,
        model_path=model_path,
        **kwargs
    )
    
    return UnifiedMLPotentialCalculator(config)


def get_default_potential(use_case: str = "general") -> MLPotentialType:
    """
    获取默认ML势
    
    Args:
        use_case: 使用场景
    
    Returns:
        推荐的ML势类型
    """
    return MLPotentialSelector.recommend(use_case)


# =============================================================================
# 导出列表
# =============================================================================

__all__ = [
    # 枚举
    'MLPotentialType',
    'MLPotentialCapability',
    
    # 数据类
    'UnifiedMLPotentialConfig',
    'MLPredictionResult',
    'MLPotentialInfo',
    
    # 主要类
    'UnifiedMLPotentialCalculator',
    'UnifiedMLPotentialWorkflow',
    'MLPotentialSelector',
    
    # 便捷函数
    'load_ml_potential',
    'get_default_potential',
    
    # MACE
    'MACECalculator',
    'MACEWorkflow',
    'MACEConfig',
    'MACEMDConfig',
    'MACEDatasetPreparer',
    'MACEActiveLearning',
    
    # CHGNet
    'CHGNetASECalculator',
    'CHGNetWorkflow',
    'CHGNetConfig',
    'CHGNetPrediction',
    'MagneticProperties',
    
    # Orb
    'OrbASECalculator',
    'OrbWorkflow',
    'OrbConfig',
    'OrbMDConfig',
    'OrbBatchPredictor',
    'OrbBenchmarkResult',
    
    # 数据库
    'ML_POTENTIAL_DATABASE',
]
