#!/usr/bin/env python3
"""
集成模块 - Integration

将主动学习V2模块集成到现有的ML势训练工作流中。
提供与DP-GEN、DeePMD-kit、NEP等工具的接口。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
from pathlib import Path
import logging
import json
import os
import sys

# 添加父目录到路径以导入现有模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from ase import Atoms
from ase.io import read, write
import numpy as np

logger = logging.getLogger(__name__)


class MLPotentialTrainer:
    """
    ML势训练器接口
    
    统一的ML势训练接口，支持DeePMD-kit、NEP、SNAP等多种势函数。
    """
    
    def __init__(
        self,
        potential_type: str = 'deepmd',
        config: Optional[Dict] = None,
        work_dir: str = './ml_training'
    ):
        self.potential_type = potential_type.lower()
        self.config = config or {}
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.is_trained = False
    
    def train(
        self,
        training_data: List[Atoms],
        validation_data: Optional[List[Atoms]] = None,
        **kwargs
    ) -> str:
        """
        训练ML势模型
        
        Returns:
            模型路径
        """
        if self.potential_type == 'deepmd':
            return self._train_deepmd(training_data, validation_data, **kwargs)
        elif self.potential_type == 'nep':
            return self._train_nep(training_data, validation_data, **kwargs)
        elif self.potential_type == 'snip':
            return self._train_snap(training_data, validation_data, **kwargs)
        else:
            raise ValueError(f"Unsupported potential type: {self.potential_type}")
    
    def _train_deepmd(
        self,
        training_data: List[Atoms],
        validation_data: Optional[List[Atoms]] = None,
        **kwargs
    ) -> str:
        """训练DeePMD势"""
        logger.info(f"Training DeePMD potential with {len(training_data)} structures")
        
        # 保存数据为DeePMD格式
        data_dir = self.work_dir / 'data'
        data_dir.mkdir(exist_ok=True)
        
        self._save_deepmd_format(training_data, data_dir / 'training')
        if validation_data:
            self._save_deepmd_format(validation_data, data_dir / 'validation')
        
        # 这里应该调用DeePMD训练
        # 简化版本，仅返回路径
        model_path = self.work_dir / 'graph.pb'
        logger.info(f"DeePMD model would be saved to {model_path}")
        
        return str(model_path)
    
    def _train_nep(
        self,
        training_data: List[Atoms],
        validation_data: Optional[List[Atoms]] = None,
        **kwargs
    ) -> str:
        """训练NEP势"""
        logger.info(f"Training NEP potential with {len(training_data)} structures")
        model_path = self.work_dir / 'nep.txt'
        return str(model_path)
    
    def _train_snap(
        self,
        training_data: List[Atoms],
        validation_data: Optional[List[Atoms]] = None,
        **kwargs
    ) -> str:
        """训练SNAP势"""
        logger.info(f"Training SNAP potential with {len(training_data)} structures")
        model_path = self.work_dir / 'Snapcoeff'
        return str(model_path)
    
    def _save_deepmd_format(self, structures: List[Atoms], output_dir: Path):
        """保存为DeePMD格式"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用dpdata或自定义格式
        for i, atoms in enumerate(structures):
            subdir = output_dir / f'sys-{i:04d}'
            subdir.mkdir(exist_ok=True)
            write(subdir / 'POSCAR', atoms)
    
    def predict(self, structures: List[Atoms]) -> Dict[str, np.ndarray]:
        """
        预测能量和力
        
        Returns:
            {'energies': array, 'forces': array, 'virials': array}
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        n_structures = len(structures)
        
        # 占位符预测
        return {
            'energies': np.random.randn(n_structures),
            'forces': [np.random.randn(len(atoms), 3) for atoms in structures],
            'virials': [np.random.randn(3, 3) for _ in structures]
        }
    
    def compute_uncertainty(self, structures: List[Atoms]) -> np.ndarray:
        """计算预测不确定性"""
        predictions = self.predict(structures)
        # 简化的不确定性估计
        return np.abs(predictions['energies']) * 0.1


class DFTInterface:
    """
    DFT计算接口
    
    统一接口支持VASP、Quantum ESPRESSO、ABACUS等DFT软件。
    """
    
    def __init__(
        self,
        calculator: str = 'vasp',
        config: Optional[Dict] = None,
        work_dir: str = './dft_calculations',
        use_hpc: bool = False
    ):
        self.calculator = calculator.lower()
        self.config = config or {}
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.use_hpc = use_hpc
        
        # 成本估算 (相对单位)
        self.cost_per_calculation = self._estimate_cost()
    
    def _estimate_cost(self) -> float:
        """估算每次DFT计算的成本"""
        costs = {
            'vasp': 100.0,
            'qe': 80.0,
            'abacus': 60.0,
            'cp2k': 70.0
        }
        return costs.get(self.calculator, 100.0)
    
    def calculate(
        self,
        structures: List[Atoms],
        properties: List[str] = None
    ) -> List[Dict]:
        """
        执行DFT计算
        
        Args:
            structures: 待计算结构
            properties: 要计算的属性 ['energy', 'forces', 'stress', 'virial']
        
        Returns:
            计算结果列表
        """
        properties = properties or ['energy', 'forces']
        results = []
        
        logger.info(f"Running DFT calculations for {len(structures)} structures")
        
        for i, atoms in enumerate(structures):
            result = self._run_single_calculation(atoms, i, properties)
            results.append(result)
        
        return results
    
    def _run_single_calculation(
        self,
        atoms: Atoms,
        index: int,
        properties: List[str]
    ) -> Dict:
        """运行单个DFT计算"""
        calc_dir = self.work_dir / f'calc_{index:04d}'
        calc_dir.mkdir(exist_ok=True)
        
        # 保存结构
        write(calc_dir / 'POSCAR', atoms)
        
        # 生成输入文件
        if self.calculator == 'vasp':
            self._generate_vasp_input(calc_dir, properties)
        elif self.calculator == 'qe':
            self._generate_qe_input(calc_dir, atoms, properties)
        
        # 模拟计算结果
        result = {
            'energy': np.random.randn(),  # eV
            'forces': np.random.randn(len(atoms), 3),  # eV/Ang
            'stress': np.random.randn(3, 3),
            'virial': np.random.randn(3, 3),
            'converged': True,
            'calculation_time': np.random.randint(300, 3600),
            'cost': self.cost_per_calculation
        }
        
        return result
    
    def _generate_vasp_input(self, calc_dir: Path, properties: List[str]):
        """生成VASP输入文件"""
        incar_content = """
SYSTEM = DFT Calculation
ENCUT = 500
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6
NSW = 0
IBRION = -1
"""
        (calc_dir / 'INCAR').write_text(incar_content)
    
    def _generate_qe_input(
        self,
        calc_dir: Path,
        atoms: Atoms,
        properties: List[str]
    ):
        """生成Quantum ESPRESSO输入文件"""
        # 简化版本
        pass


class ActiveLearningV2Workflow:
    """
    主动学习V2工作流
    
    整合先进的主动学习策略、自适应采样器和ML势训练。
    这是集成到现有系统的主入口。
    """
    
    def __init__(
        self,
        ml_trainer: Optional[MLPotentialTrainer] = None,
        dft_interface: Optional[DFTInterface] = None,
        strategy: Optional[Any] = None,
        adaptive_sampler: Optional[Any] = None,
        config: Optional[Dict] = None,
        work_dir: str = './active_learning_v2'
    ):
        self.config = config or {}
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.ml_trainer = ml_trainer or MLPotentialTrainer()
        self.dft_interface = dft_interface or DFTInterface()
        
        # 初始化策略
        if adaptive_sampler is not None:
            self.sampler = adaptive_sampler
        elif strategy is not None:
            self.sampler = strategy
        else:
            # 默认使用自适应采样器
            from ..adaptive import AdaptiveSampler
            self.sampler = AdaptiveSampler()
        
        # 数据存储
        self.labeled_structures: List[Atoms] = []
        self.labeled_energies: List[float] = []
        self.labeled_forces: List[np.ndarray] = []
        self.unlabeled_structures: List[Atoms] = []
        
        # 历史记录
        self.history = []
        self.iteration = 0
        
        # 性能追踪
        self.total_dft_calls = 0
        self.total_cost = 0.0
    
    def initialize(
        self,
        initial_structures: List[Atoms],
        initial_data: Optional[List[Dict]] = None
    ):
        """
        初始化工作流
        
        Args:
            initial_structures: 初始结构池
            initial_data: 初始标注数据 (可选)
        """
        logger.info("Initializing Active Learning V2 Workflow")
        
        if initial_data is not None:
            # 使用已有标注数据
            self.labeled_structures = initial_structures[:len(initial_data)]
            for data in initial_data:
                self.labeled_energies.append(data.get('energy', 0.0))
                self.labeled_forces.append(data.get('forces', np.zeros((len(self.labeled_structures[-1]), 3))))
        else:
            # 需要对初始结构进行DFT计算
            logger.info(f"Running initial DFT calculations for {len(initial_structures)} structures")
            results = self.dft_interface.calculate(initial_structures)
            
            self.labeled_structures = initial_structures
            for result in results:
                self.labeled_energies.append(result['energy'])
                self.labeled_forces.append(result['forces'])
                self.total_cost += result['cost']
            
            self.total_dft_calls += len(initial_structures)
        
        # 初始训练
        self._train_model()
        
        logger.info("Initialization completed")
    
    def run_iteration(self) -> Dict:
        """
        运行一个主动学习迭代
        
        Returns:
            迭代结果统计
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Active Learning Iteration {self.iteration}")
        logger.info(f"{'='*60}")
        
        # 1. 探索：生成候选结构
        candidate_structures = self._explore_structures()
        
        # 2. 特征化 (简化：使用能量和原子位置)
        X_candidates = self._featurize_structures(candidate_structures)
        X_labeled = self._featurize_structures(self.labeled_structures)
        y_labeled = np.array(self.labeled_energies)
        
        # 3. 选择：使用策略选择最有价值的结构
        selected_indices, selection_metadata = self.sampler.sample(
            X_unlabeled=X_candidates,
            X_labeled=X_labeled,
            y_labeled=y_labeled,
            model=self.ml_trainer
        )
        
        selected_structures = [candidate_structures[i] for i in selected_indices]
        n_selected = len(selected_structures)
        
        logger.info(f"Selected {n_selected} structures for DFT calculation")
        
        # 4. 标注：DFT计算
        if n_selected > 0:
            dft_results = self.dft_interface.calculate(selected_structures)
            
            # 添加到训练集
            for struct, result in zip(selected_structures, dft_results):
                self.labeled_structures.append(struct)
                self.labeled_energies.append(result['energy'])
                self.labeled_forces.append(result['forces'])
                self.total_cost += result['cost']
            
            self.total_dft_calls += n_selected
            
            # 5. 重训练
            self._train_model()
        
        # 6. 评估
        metrics = self._evaluate()
        
        # 7. 记录性能
        if hasattr(self.sampler, 'record_performance'):
            from ..adaptive import PerformanceMetrics
            self.sampler.record_performance(
                iteration=self.iteration,
                dft_calls=n_selected,
                total_cost=self.total_cost,
                **metrics
            )
        
        # 8. 保存迭代信息
        iteration_info = {
            'iteration': self.iteration,
            'n_selected': n_selected,
            'total_labeled': len(self.labeled_structures),
            'total_cost': self.total_cost,
            'metrics': metrics,
            'selection_strategy': selection_metadata.get('strategy', 'unknown'),
            'selection_reason': selection_metadata.get('recommendation_reason', '')
        }
        self.history.append(iteration_info)
        
        self.iteration += 1
        
        return iteration_info
    
    def run(
        self,
        max_iterations: int = 100,
        convergence_patience: int = 5
    ) -> Dict:
        """
        运行完整主动学习循环
        
        Args:
            max_iterations: 最大迭代次数
            convergence_patience: 收敛耐心值
        
        Returns:
            最终结果统计
        """
        logger.info("Starting Active Learning V2 Workflow")
        
        for iteration in range(max_iterations):
            # 运行迭代
            iteration_info = self.run_iteration()
            
            # 检查收敛
            if hasattr(self.sampler, 'check_convergence'):
                should_stop, reason = self.sampler.check_convergence()
                if should_stop:
                    logger.info(f"Convergence detected: {reason}")
                    break
            
            # 简单收敛检查
            if iteration >= convergence_patience:
                recent_costs = [h['total_cost'] for h in self.history[-convergence_patience:]]
                if all(c == recent_costs[0] for c in recent_costs):
                    logger.info("Convergence: no new structures selected")
                    break
        
        return self.get_summary()
    
    def _explore_structures(self) -> List[Atoms]:
        """探索结构空间，生成候选结构"""
        # 简化版本：从已标注结构进行微小扰动
        candidates = []
        
        for struct in self.labeled_structures[-10:]:  # 使用最近的10个结构
            for _ in range(5):  # 每个结构生成5个扰动
                perturbed = struct.copy()
                positions = perturbed.get_positions()
                # 添加随机扰动
                noise = np.random.randn(*positions.shape) * 0.1  # 0.1 Angstrom
                perturbed.set_positions(positions + noise)
                candidates.append(perturbed)
        
        return candidates
    
    def _featurize_structures(self, structures: List[Atoms]) -> np.ndarray:
        """
        将结构转换为特征向量
        
        简化版本：使用原子位置和数量
        """
        features = []
        
        for atoms in structures:
            # 使用原子数量、平均位置、位置方差等简单特征
            positions = atoms.get_positions()
            feat = [
                len(atoms),
                np.mean(positions),
                np.std(positions),
                np.mean(np.linalg.norm(positions, axis=1)),
            ]
            features.append(feat)
        
        return np.array(features)
    
    def _train_model(self):
        """训练ML势模型"""
        logger.info(f"Training model with {len(self.labeled_structures)} structures")
        
        # 分割训练集和验证集
        n_total = len(self.labeled_structures)
        n_train = int(0.9 * n_total)
        
        indices = np.random.permutation(n_total)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_structures = [self.labeled_structures[i] for i in train_idx]
        val_structures = [self.labeled_structures[i] for i in val_idx]
        
        self.ml_trainer.train(train_structures, val_structures)
    
    def _evaluate(self) -> Dict:
        """评估当前模型性能"""
        # 简化版本
        return {
            'mean_energy': float(np.mean(self.labeled_energies)),
            'std_energy': float(np.std(self.labeled_energies)),
            'n_structures': len(self.labeled_structures)
        }
    
    def get_summary(self) -> Dict:
        """获取工作流摘要"""
        return {
            'total_iterations': self.iteration,
            'total_structures': len(self.labeled_structures),
            'total_dft_calls': self.total_dft_calls,
            'total_cost': self.total_cost,
            'history': self.history,
            'sampler_status': self.sampler.get_status() if hasattr(self.sampler, 'get_status') else {}
        }
    
    def save_results(self, output_dir: Optional[str] = None):
        """保存结果"""
        output_dir = Path(output_dir) if output_dir else self.work_dir / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存历史
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 保存标注数据
        write(output_dir / 'labeled_structures.traj', self.labeled_structures)
        
        # 保存摘要
        summary = self.get_summary()
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")


def create_active_learning_workflow(
    potential_type: str = 'deepmd',
    dft_calculator: str = 'vasp',
    strategy_name: str = 'adaptive',
    work_dir: str = './active_learning_v2',
    config: Optional[Dict] = None
) -> ActiveLearningV2Workflow:
    """
    创建主动学习V2工作流的工厂函数
    
    Args:
        potential_type: ML势类型 ('deepmd', 'nep', 'snap')
        dft_calculator: DFT计算器 ('vasp', 'qe', 'abacus')
        strategy_name: 策略名称 ('bayesian', 'dpp', 'evidential', 'multifidelity', 'adaptive')
        work_dir: 工作目录
        config: 额外配置
    
    Returns:
        ActiveLearningV2Workflow实例
    """
    config = config or {}
    
    # 创建组件
    ml_trainer = MLPotentialTrainer(
        potential_type=potential_type,
        config=config.get('ml_config'),
        work_dir=f"{work_dir}/ml_training"
    )
    
    dft_interface = DFTInterface(
        calculator=dft_calculator,
        config=config.get('dft_config'),
        work_dir=f"{work_dir}/dft_calculations"
    )
    
    # 创建策略
    from .. import strategies, adaptive
    
    if strategy_name == 'bayesian':
        strategy = strategies.BayesianOptimizationStrategy()
    elif strategy_name == 'dpp':
        strategy = strategies.DPPDiversityStrategy()
    elif strategy_name == 'evidential':
        strategy = strategies.EvidentialLearningStrategy()
    elif strategy_name == 'multifidelity':
        strategy = strategies.MultiFidelityStrategy()
    elif strategy_name == 'adaptive':
        strategy = adaptive.AdaptiveSampler()
    else:
        strategy = adaptive.AdaptiveSampler()
    
    # 创建工作流
    workflow = ActiveLearningV2Workflow(
        ml_trainer=ml_trainer,
        dft_interface=dft_interface,
        strategy=strategy if not isinstance(strategy, adaptive.AdaptiveSampler) else None,
        adaptive_sampler=strategy if isinstance(strategy, adaptive.AdaptiveSampler) else None,
        config=config,
        work_dir=work_dir
    )
    
    return workflow
