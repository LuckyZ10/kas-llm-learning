#!/usr/bin/env python3
"""
DFT/MD耦合模块 - RL与第一性原理计算的集成

包含:
- DFTCoupling: DFT计算接口
- MLCoupling: ML势集成
- ActiveLearningCoupling: 主动学习循环
- HumanInTheLoop: 人机协作优化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class DFTResult:
    """DFT计算结果"""
    energy: float
    forces: Optional[np.ndarray] = None
    stress: Optional[np.ndarray] = None
    band_gap: Optional[float] = None
    converged: bool = True
    calculation_time: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class MDResult:
    """MD模拟结果"""
    trajectory: List[Any]
    energies: List[float]
    temperatures: List[float]
    stable: bool = True
    calculation_time: float = 0.0


class CouplingInterface(ABC):
    """耦合接口基类"""
    
    @abstractmethod
    def calculate_energy(self, structure: Any) -> float:
        """计算能量"""
        pass
    
    @abstractmethod
    def calculate_forces(self, structure: Any) -> np.ndarray:
        """计算力"""
        pass


class DFTCoupling(CouplingInterface):
    """
    DFT计算耦合
    
    集成VASP、Quantum ESPRESSO等DFT计算引擎。
    """
    
    def __init__(
        self,
        calculator: str = 'vasp',
        calculator_params: Optional[Dict] = None,
        max_retries: int = 3,
        timeout: float = 3600.0
    ):
        self.calculator = calculator
        self.calculator_params = calculator_params or {}
        self.max_retries = max_retries
        self.timeout = timeout
        
        # 计算统计
        self.n_calculations = 0
        self.total_calculation_time = 0.0
        self.cache = {}
    
    def calculate_energy(self, structure: Any) -> float:
        """计算DFT能量"""
        result = self.run_dft(structure, properties=['energy'])
        return result.energy if result.converged else float('inf')
    
    def calculate_forces(self, structure: Any) -> np.ndarray:
        """计算DFT力"""
        result = self.run_dft(structure, properties=['forces'])
        return result.forces if result.forces is not None else np.zeros((len(structure.elements), 3))
    
    def run_dft(
        self,
        structure: Any,
        properties: List[str] = None,
        kpoints: Optional[np.ndarray] = None
    ) -> DFTResult:
        """
        运行DFT计算
        
        Args:
            structure: 晶体结构
            properties: 要计算的性质
            kpoints: K点网格
            
        Returns:
            DFT计算结果
        """
        import time
        start_time = time.time()
        
        properties = properties or ['energy']
        
        # 检查缓存
        cache_key = self._get_cache_key(structure, properties)
        if cache_key in self.cache:
            logger.info("Using cached DFT result")
            return self.cache[cache_key]
        
        logger.info(f"Running DFT calculation ({self.calculator})...")
        
        # 生成输入文件
        if self.calculator == 'vasp':
            result = self._run_vasp(structure, properties, kpoints)
        elif self.calculator == 'qe':
            result = self._run_quantum_espresso(structure, properties, kpoints)
        elif self.calculator == 'abacus':
            result = self._run_abacus(structure, properties, kpoints)
        else:
            raise ValueError(f"Unsupported calculator: {self.calculator}")
        
        calculation_time = time.time() - start_time
        result.calculation_time = calculation_time
        
        self.n_calculations += 1
        self.total_calculation_time += calculation_time
        
        # 缓存结果
        if result.converged:
            self.cache[cache_key] = result
        
        logger.info(f"DFT calculation completed in {calculation_time:.1f}s, "
                   f"energy={result.energy:.4f} eV")
        
        return result
    
    def _get_cache_key(self, structure: Any, properties: List[str]) -> str:
        """生成缓存键"""
        composition = structure.get_composition()
        comp_str = '_'.join(f"{elem}{count}" for elem, count in sorted(composition.items()))
        props_str = '_'.join(sorted(properties))
        return f"{comp_str}_{props_str}"
    
    def _run_vasp(
        self,
        structure: Any,
        properties: List[str],
        kpoints: Optional[np.ndarray]
    ) -> DFTResult:
        """运行VASP计算"""
        # 简化实现 - 实际应调用VASP
        # 这里返回模拟结果
        
        # 模拟能量计算 (基于元素电负性差异的简化估算)
        energy = self._estimate_energy(structure)
        
        return DFTResult(
            energy=energy,
            converged=True,
            metadata={'calculator': 'vasp'}
        )
    
    def _run_quantum_espresso(
        self,
        structure: Any,
        properties: List[str],
        kpoints: Optional[np.ndarray]
    ) -> DFTResult:
        """运行Quantum ESPRESSO计算"""
        energy = self._estimate_energy(structure)
        
        return DFTResult(
            energy=energy,
            converged=True,
            metadata={'calculator': 'qe'}
        )
    
    def _run_abacus(
        self,
        structure: Any,
        properties: List[str],
        kpoints: Optional[np.ndarray]
    ) -> DFTResult:
        """运行ABACUS计算"""
        energy = self._estimate_energy(structure)
        
        return DFTResult(
            energy=energy,
            converged=True,
            metadata={'calculator': 'abacus'}
        )
    
    def _estimate_energy(self, structure: Any) -> float:
        """估算能量 (用于模拟)"""
        # 简化估算
        composition = structure.get_composition()
        
        # 基于原子数的基准能量
        n_atoms = sum(composition.values())
        base_energy = -5.0 * n_atoms  # 约-5eV/atom
        
        # 修正项
        correction = np.random.normal(0, 0.1)  # 随机噪声
        
        return base_energy + correction
    
    def get_stats(self) -> Dict[str, Any]:
        """获取计算统计"""
        return {
            'n_calculations': self.n_calculations,
            'total_time': self.total_calculation_time,
            'avg_time': self.total_calculation_time / max(1, self.n_calculations),
            'cache_size': len(self.cache)
        }


class MLCoupling(CouplingInterface):
    """
    ML势耦合
    
    集成NEP、MTP、GAP等机器学习势函数。
    """
    
    def __init__(
        self,
        potential_type: str = 'nep',
        model_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.potential_type = potential_type
        self.model_path = model_path
        self.device = device
        
        # 加载模型
        self.model = self._load_model()
        
        # 统计
        self.n_predictions = 0
        self.total_prediction_time = 0.0
    
    def _load_model(self):
        """加载ML势模型"""
        logger.info(f"Loading {self.potential_type} model from {self.model_path}")
        
        # 简化实现
        # 实际应加载真实的ML势模型
        return None
    
    def calculate_energy(self, structure: Any) -> float:
        """使用ML势计算能量"""
        import time
        start_time = time.time()
        
        # 使用ML势预测
        energy = self._predict(structure, 'energy')
        
        self.n_predictions += 1
        self.total_prediction_time += time.time() - start_time
        
        return energy
    
    def calculate_forces(self, structure: Any) -> np.ndarray:
        """使用ML势计算力"""
        import time
        start_time = time.time()
        
        forces = self._predict(structure, 'forces')
        
        self.n_predictions += 1
        self.total_prediction_time += time.time() - start_time
        
        return forces
    
    def _predict(self, structure: Any, property_name: str):
        """ML预测"""
        # 简化实现 - 实际应调用ML势模型
        
        if property_name == 'energy':
            # 返回基于DFT的近似值
            n_atoms = len(structure.elements)
            return -5.0 * n_atoms + np.random.normal(0, 0.05)
        
        elif property_name == 'forces':
            n_atoms = len(structure.elements)
            return np.random.normal(0, 0.01, (n_atoms, 3))
        
        return 0.0
    
    def run_md(
        self,
        structure: Any,
        temperature: float = 300.0,
        n_steps: int = 1000,
        timestep: float = 1.0  # fs
    ) -> MDResult:
        """
        运行MD模拟
        
        Args:
            structure: 初始结构
            temperature: 温度 (K)
            n_steps: 模拟步数
            timestep: 时间步长 (fs)
            
        Returns:
            MD模拟结果
        """
        logger.info(f"Running MD simulation: {n_steps} steps at {temperature}K")
        
        # 简化实现 - 实际应运行真实MD
        trajectory = [structure]
        energies = [self.calculate_energy(structure)]
        temperatures = [temperature]
        
        return MDResult(
            trajectory=trajectory,
            energies=energies,
            temperatures=temperatures,
            stable=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预测统计"""
        return {
            'n_predictions': self.n_predictions,
            'total_time': self.total_prediction_time,
            'avg_time': self.total_prediction_time / max(1, self.n_predictions),
            'speedup_vs_dft': 1000.0  # 假设比DFT快1000倍
        }


class ActiveLearningCoupling:
    """
    主动学习耦合
    
    结合ML势快速预测和DFT精确计算，
    通过主动学习不断改进ML势。
    """
    
    def __init__(
        self,
        dft_coupling: DFTCoupling,
        ml_coupling: MLCoupling,
        uncertainty_threshold: float = 0.1,
        max_dft_per_iteration: int = 10
    ):
        self.dft_coupling = dft_coupling
        self.ml_coupling = ml_coupling
        self.uncertainty_threshold = uncertainty_threshold
        self.max_dft_per_iteration = max_dft_per_iteration
        
        # 训练数据
        self.training_structures = []
        self.training_energies = []
        self.training_forces = []
        
        # 统计
        self.n_dft_calls = 0
        self.n_ml_calls = 0
    
    def calculate_energy(
        self,
        structure: Any,
        uncertainty: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        智能选择DFT或ML计算能量
        
        Returns:
            (能量, 计算方法)
        """
        # 基于不确定性决定使用DFT还是ML
        if uncertainty is None:
            # 估算不确定性 (简化)
            uncertainty = self._estimate_uncertainty(structure)
        
        if uncertainty > self.uncertainty_threshold and self.n_dft_calls < self.max_dft_per_iteration:
            # 使用DFT
            energy = self.dft_coupling.calculate_energy(structure)
            method = 'dft'
            self.n_dft_calls += 1
            
            # 添加到训练数据
            self.training_structures.append(structure)
            self.training_energies.append(energy)
        else:
            # 使用ML
            energy = self.ml_coupling.calculate_energy(structure)
            method = 'ml'
            self.n_ml_calls += 1
        
        return energy, method
    
    def _estimate_uncertainty(self, structure: Any) -> float:
        """估算预测不确定性"""
        # 简化实现：基于与训练数据的距离
        if not self.training_structures:
            return 1.0  # 高不确定性
        
        # 计算与最近训练样本的距离
        min_distance = float('inf')
        
        for train_struct in self.training_structures:
            distance = self._structure_distance(structure, train_struct)
            min_distance = min(min_distance, distance)
        
        # 距离越远，不确定性越高
        uncertainty = min(1.0, min_distance / 5.0)
        
        return uncertainty
    
    def _structure_distance(self, struct1: Any, struct2: Any) -> float:
        """计算结构距离"""
        # 简化：基于组成差异
        comp1 = struct1.get_composition()
        comp2 = struct2.get_composition()
        
        all_elements = set(comp1.keys()) | set(comp2.keys())
        
        distance = 0.0
        for elem in all_elements:
            frac1 = comp1.get(elem, 0) / sum(comp1.values()) if comp1 else 0
            frac2 = comp2.get(elem, 0) / sum(comp2.values()) if comp2 else 0
            distance += (frac1 - frac2) ** 2
        
        return np.sqrt(distance)
    
    def retrain_model(self):
        """使用新数据重新训练ML势"""
        logger.info(f"Retraining ML model with {len(self.training_structures)} structures")
        
        # 实际应调用训练流程
        # 这里简化处理
        self.n_dft_calls = 0  # 重置计数
        
        return {'n_train': len(self.training_structures)}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            'n_dft_calls': self.n_dft_calls,
            'n_ml_calls': self.n_ml_calls,
            'n_training_structures': len(self.training_structures),
            'dft_ratio': self.n_dft_calls / max(1, self.n_dft_calls + self.n_ml_calls)
        }


class HumanInTheLoop:
    """
    人机协作优化
    
    允许人类专家在优化过程中提供反馈和指导。
    """
    
    def __init__(self, feedback_callback: Optional[Callable] = None):
        self.feedback_callback = feedback_callback
        self.human_feedback = []
        self.preferred_structures = []
    
    def request_feedback(self, structure: Any, predicted_reward: float) -> float:
        """
        请求人类反馈
        
        Args:
            structure: 待评估的结构
            predicted_reward: RL预测的奖励
            
        Returns:
            人类修正的奖励
        """
        if self.feedback_callback is None:
            return predicted_reward
        
        # 调用反馈回调
        human_reward = self.feedback_callback(structure, predicted_reward)
        
        # 记录反馈
        self.human_feedback.append({
            'structure': structure,
            'predicted': predicted_reward,
            'human': human_reward
        })
        
        return human_reward
    
    def add_preference(self, structure1: Any, structure2: Any, prefer_first: bool):
        """
        添加人类偏好
        
        Args:
            structure1: 第一个结构
            structure2: 第二个结构
            prefer_first: 是否更喜欢第一个
        """
        self.preferred_structures.append({
            'better': structure1 if prefer_first else structure2,
            'worse': structure2 if prefer_first else structure1
        })
    
    def get_preference_model(self) -> Optional[Callable]:
        """
        从人类偏好学习奖励模型
        
        Returns:
            学习到的奖励函数
        """
        if len(self.preferred_structures) < 10:
            return None
        
        # 简化实现 - 实际应训练奖励模型
        def learned_reward(structure):
            return 0.0
        
        return learned_reward
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """获取反馈统计"""
        if not self.human_feedback:
            return {}
        
        disagreements = sum(
            1 for f in self.human_feedback
            if (f['predicted'] > 0) != (f['human'] > 0)
        )
        
        return {
            'n_feedback': len(self.human_feedback),
            'n_preferences': len(self.preferred_structures),
            'disagreement_rate': disagreements / len(self.human_feedback)
        }
