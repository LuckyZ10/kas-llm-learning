"""
DFT-LAMMPS 抽象基类系统
=========================

定义所有计算模块的统一接口，确保：
1. 一致的API设计
2. 可扩展的架构
3. 类型安全

Usage:
    from core.base import DFTCalculator, MDSimulator, MLPotential
    
    # 所有实现都遵循相同接口
    calc = VaspCalculator(config)  # 或 EspressoCalculator
    energy, forces = calc.calculate(atoms)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


# =============================================================================
# 基础数据结构
# =============================================================================

@dataclass
class CalculationResult:
    """计算结果统一数据结构"""
    energy: float
    forces: Optional[np.ndarray] = None
    stress: Optional[np.ndarray] = None
    
    # 元数据
    success: bool = True
    error_message: Optional[str] = None
    computation_time: float = 0.0
    
    # 额外属性（DFT特有）
    eigenvalues: Optional[np.ndarray] = None
    fermi_level: Optional[float] = None
    band_gap: Optional[float] = None
    
    # 额外属性（MD特有）
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    
    def __post_init__(self):
        if self.forces is not None:
            self.forces = np.array(self.forces)
        if self.stress is not None:
            self.stress = np.array(self.stress)


@dataclass
class Trajectory:
    """MD轨迹统一数据结构"""
    atoms_list: List[Atoms]
    energies: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    pressures: List[float] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.atoms_list)
    
    def __iter__(self) -> Iterator[Atoms]:
        return iter(self.atoms_list)
    
    def __getitem__(self, idx: int) -> Atoms:
        return self.atoms_list[idx]
    
    @property
    def final_structure(self) -> Atoms:
        """返回最终结构"""
        return self.atoms_list[-1] if self.atoms_list else Atoms()
    
    def get_average_temperature(self) -> float:
        """计算平均温度"""
        return np.mean(self.temperatures) if self.temperatures else 0.0
    
    def get_average_pressure(self) -> float:
        """计算平均压力"""
        return np.mean(self.pressures) if self.pressures else 0.0


@dataclass
class TrainingData:
    """训练数据统一数据结构"""
    structures: List[Atoms]
    energies: List[float]
    forces: Optional[List[np.ndarray]] = None
    
    def __len__(self) -> int:
        return len(self.structures)
    
    def split(self, train_ratio: float = 0.8) -> Tuple["TrainingData", "TrainingData"]:
        """分割训练集和验证集"""
        n = len(self)
        n_train = int(n * train_ratio)
        
        train_data = TrainingData(
            structures=self.structures[:n_train],
            energies=self.energies[:n_train],
            forces=self.forces[:n_train] if self.forces else None
        )
        
        val_data = TrainingData(
            structures=self.structures[n_train:],
            energies=self.energies[n_train:],
            forces=self.forces[n_train:] if self.forces else None
        )
        
        return train_data, val_data


# =============================================================================
# DFT 计算器基类
# =============================================================================

class DFTCalculator(ABC):
    """DFT计算器抽象基类
    
    所有DFT代码实现（VASP、QE、ABACUS）必须继承此类
    """
    
    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._calculator = None
        self._working_dir: Optional[Path] = None
    
    @property
    @abstractmethod
    def code_name(self) -> str:
        """返回DFT代码名称（如 'vasp', 'espresso'）"""
        pass
    
    @abstractmethod
    def setup(self, atoms: Atoms, working_dir: Union[str, Path]) -> None:
        """设置计算环境
        
        Args:
            atoms: 输入结构
            working_dir: 工作目录
        """
        pass
    
    @abstractmethod
    def calculate(self, atoms: Atoms) -> CalculationResult:
        """执行DFT计算
        
        Args:
            atoms: 输入结构
            
        Returns:
            CalculationResult 包含能量、力、应力等
        """
        pass
    
    @abstractmethod
    def relax_structure(self, atoms: Atoms, fmax: float = 0.01, max_steps: int = 200) -> Tuple[Atoms, CalculationResult]:
        """结构优化
        
        Args:
            atoms: 输入结构
            fmax: 力收敛标准 (eV/Å)
            max_steps: 最大优化步数
            
        Returns:
            (优化后的结构, 计算结果)
        """
        pass
    
    @abstractmethod
    def read_results(self, working_dir: Union[str, Path]) -> CalculationResult:
        """从输出目录读取结果
        
        Args:
            working_dir: 包含输出文件的目录
            
        Returns:
            CalculationResult
        """
        pass
    
    def get_ase_calculator(self) -> Calculator:
        """返回ASE Calculator实例"""
        return self._calculator
    
    def get_band_structure(self, atoms: Atoms, kpath: Optional[Any] = None) -> Dict[str, Any]:
        """计算能带结构（可选实现）"""
        raise NotImplementedError(f"{self.code_name} does not support band structure calculation")
    
    def get_dos(self, atoms: Atoms) -> Dict[str, Any]:
        """计算态密度（可选实现）"""
        raise NotImplementedError(f"{self.code_name} does not support DOS calculation")


# =============================================================================
# MD 模拟器基类
# =============================================================================

class MDSimulator(ABC):
    """分子动力学模拟器抽象基类
    
    所有MD引擎实现（LAMMPS、GROMACS等）必须继承此类
    """
    
    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._trajectory: Optional[Trajectory] = None
    
    @property
    @abstractmethod
    def engine_name(self) -> str:
        """返回MD引擎名称"""
        pass
    
    @abstractmethod
    def setup(self, atoms: Atoms, potential: Any, working_dir: Union[str, Path]) -> None:
        """设置模拟环境
        
        Args:
            atoms: 初始结构
            potential: 势函数（可以是MLPotential或传统势）
            working_dir: 工作目录
        """
        pass
    
    @abstractmethod
    def run_nvt(self, temperature: float, nsteps: int, timestep: float = 1.0) -> Trajectory:
        """NVT系综模拟
        
        Args:
            temperature: 温度 (K)
            nsteps: 模拟步数
            timestep: 时间步长 (fs)
            
        Returns:
            Trajectory 轨迹对象
        """
        pass
    
    @abstractmethod
    def run_npt(self, temperature: float, pressure: float, nsteps: int, timestep: float = 1.0) -> Trajectory:
        """NPT系综模拟
        
        Args:
            temperature: 温度 (K)
            pressure: 压力 (bar)
            nsteps: 模拟步数
            timestep: 时间步长 (fs)
            
        Returns:
            Trajectory 轨迹对象
        """
        pass
    
    @abstractmethod
    def run_nve(self, nsteps: int, timestep: float = 1.0) -> Trajectory:
        """NVE系综模拟（微正则系综）
        
        Args:
            nsteps: 模拟步数
            timestep: 时间步长 (fs)
            
        Returns:
            Trajectory 轨迹对象
        """
        pass
    
    def get_trajectory(self) -> Optional[Trajectory]:
        """获取上次模拟的轨迹"""
        return self._trajectory
    
    def compute_rdf(self, rmax: float = 10.0, nbins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """计算径向分布函数（可选实现）"""
        raise NotImplementedError(f"{self.engine_name} does not support RDF calculation")
    
    def compute_msd(self, atom_indices: Optional[List[int]] = None) -> np.ndarray:
        """计算均方位移（可选实现）"""
        raise NotImplementedError(f"{self.engine_name} does not support MSD calculation")


# =============================================================================
# ML 势基类
# =============================================================================

class MLPotential(ABC):
    """机器学习势抽象基类
    
    所有ML势实现（NEP、DeepMD、MACE等）必须继承此类
    """
    
    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._model = None
        self._is_trained = False
    
    @property
    @abstractmethod
    def potential_type(self) -> str:
        """返回势函数类型"""
        pass
    
    @abstractmethod
    def train(self, data: TrainingData, validation_data: Optional[TrainingData] = None) -> Dict[str, Any]:
        """训练模型
        
        Args:
            data: 训练数据
            validation_data: 验证数据（可选）
            
        Returns:
            训练历史（损失曲线等）
        """
        pass
    
    @abstractmethod
    def predict(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        """预测能量和力
        
        Args:
            atoms: 输入结构
            
        Returns:
            (能量 eV, 力 eV/Å)
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """保存模型到文件"""
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """从文件加载模型"""
        pass
    
    def get_ase_calculator(self) -> Calculator:
        """返回ASE Calculator实例（可选实现）"""
        raise NotImplementedError(f"{self.potential_type} does not provide ASE calculator interface")
    
    def fine_tune(self, new_data: TrainingData, epochs: int = 100) -> None:
        """在预训练模型上微调（可选实现）"""
        raise NotImplementedError(f"{self.potential_type} does not support fine-tuning")
    
    @property
    def is_trained(self) -> bool:
        """检查模型是否已训练"""
        return self._is_trained


# =============================================================================
# 工作流基类
# =============================================================================

class Workflow(ABC):
    """工作流抽象基类
    
    所有材料计算工作流必须继承此类
    """
    
    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, Any] = {}
        self._completed_stages: List[str] = []
    
    @property
    @abstractmethod
    def workflow_name(self) -> str:
        """返回工作流名称"""
        pass
    
    @abstractmethod
    def run(self, input_data: Any) -> Dict[str, Any]:
        """执行完整工作流
        
        Args:
            input_data: 输入数据（结构列表、配置等）
            
        Returns:
            工作流结果字典
        """
        pass
    
    @abstractmethod
    def get_stages(self) -> List[str]:
        """返回工作流阶段列表"""
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """获取当前结果"""
        return self._results
    
    def get_completed_stages(self) -> List[str]:
        """获取已完成阶段"""
        return self._completed_stages
    
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """保存工作流状态（可选实现）"""
        import pickle
        checkpoint = {
            'results': self._results,
            'completed_stages': self._completed_stages,
            'timestamp': datetime.now().isoformat()
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """加载工作流状态（可选实现）"""
        import pickle
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        self._results = checkpoint['results']
        self._completed_stages = checkpoint['completed_stages']


# =============================================================================
# 主动学习基类
# =============================================================================

class ActiveLearningStrategy(ABC):
    """主动学习策略抽象基类"""
    
    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def select_samples(
        self, 
        candidates: List[Atoms], 
        model: MLPotential,
        n_samples: int
    ) -> List[int]:
        """从候选结构中选择样本进行DFT计算
        
        Args:
            candidates: 候选结构列表
            model: 当前ML势模型
            n_samples: 需要选择的样本数
            
        Returns:
            选中样本的索引列表
        """
        pass
    
    @abstractmethod
    def compute_uncertainty(self, atoms: Atoms, model: MLPotential) -> float:
        """计算单个结构的不确定性
        
        Args:
            atoms: 输入结构
            model: ML势模型
            
        Returns:
            不确定性分数（越高越需要计算）
        """
        pass


# =============================================================================
# 工厂模式
# =============================================================================

class CalculatorFactory:
    """计算器工厂 - 根据配置创建实例"""
    
    _dft_calculators: Dict[str, type] = {}
    _md_simulators: Dict[str, type] = {}
    _ml_potentials: Dict[str, type] = {}
    
    @classmethod
    def register_dft(cls, name: str, calculator_class: type):
        """注册DFT计算器"""
        cls._dft_calculators[name] = calculator_class
    
    @classmethod
    def register_md(cls, name: str, simulator_class: type):
        """注册MD模拟器"""
        cls._md_simulators[name] = simulator_class
    
    @classmethod
    def register_ml(cls, name: str, potential_class: type):
        """注册ML势"""
        cls._ml_potentials[name] = potential_class
    
    @classmethod
    def create_dft(cls, name: str, config: Optional[Any] = None) -> DFTCalculator:
        """创建DFT计算器实例"""
        if name not in cls._dft_calculators:
            raise ValueError(f"Unknown DFT code: {name}. Available: {list(cls._dft_calculators.keys())}")
        return cls._dft_calculators[name](config)
    
    @classmethod
    def create_md(cls, name: str, config: Optional[Any] = None) -> MDSimulator:
        """创建MD模拟器实例"""
        if name not in cls._md_simulators:
            raise ValueError(f"Unknown MD engine: {name}. Available: {list(cls._md_simulators.keys())}")
        return cls._md_simulators[name](config)
    
    @classmethod
    def create_ml(cls, name: str, config: Optional[Any] = None) -> MLPotential:
        """创建ML势实例"""
        if name not in cls._ml_potentials:
            raise ValueError(f"Unknown ML potential: {name}. Available: {list(cls._ml_potentials.keys())}")
        return cls._ml_potentials[name](config)


# =============================================================================
# 示例实现（参考）
# =============================================================================

class ExampleVaspCalculator(DFTCalculator):
    """VASP实现示例 - 展示如何使用基类"""
    
    @property
    def code_name(self) -> str:
        return "vasp"
    
    def setup(self, atoms: Atoms, working_dir: Union[str, Path]) -> None:
        self._working_dir = Path(working_dir)
        self._working_dir.mkdir(parents=True, exist_ok=True)
        # 写INCAR、KPOINTS、POSCAR等
    
    def calculate(self, atoms: Atoms) -> CalculationResult:
        # 调用VASP执行计算
        # 解析结果
        return CalculationResult(energy=0.0, success=True)
    
    def relax_structure(self, atoms: Atoms, fmax: float = 0.01, max_steps: int = 200) -> Tuple[Atoms, CalculationResult]:
        # 结构优化逻辑
        return atoms, CalculationResult(energy=0.0)
    
    def read_results(self, working_dir: Union[str, Path]) -> CalculationResult:
        # 读取OUTCAR
        return CalculationResult(energy=0.0)


# 注册到工厂
# CalculatorFactory.register_dft("vasp", ExampleVaspCalculator)
