"""
nep_training/active_learning.py
===============================
主动学习模块

为NEP训练提供智能样本选择策略
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path

from ase import Atoms
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units

from .data import NEPDataset, NEPFrame
from .trainer import NEPTrainerV2

logger = logging.getLogger(__name__)


@dataclass
class ALConfig:
    """主动学习配置"""
    # 查询策略
    strategy: str = "uncertainty"  # uncertainty, diversity, hybrid
    
    # 采样参数
    n_samples_per_iteration: int = 50
    max_iterations: int = 20
    
    # 不确定性阈值
    uncertainty_threshold: float = 0.3
    min_uncertainty: float = 0.05
    
    # 多样性参数
    diversity_lambda: float = 0.5  # 多样性与不确定性权衡
    
    # 探索参数
    exploration_temperatures: List[float] = field(default_factory=lambda: [300, 500, 800])
    exploration_steps: int = 1000
    exploration_timestep: float = 1.0  # fs
    
    # 收敛标准
    convergence_patience: int = 3
    convergence_threshold: float = 0.01
    
    # 成本估算
    cost_per_dft: float = 100.0  # 相对成本单位


class QueryStrategy(ABC):
    """查询策略基类"""
    
    @abstractmethod
    def select(self, 
               candidates: List[NEPFrame],
               model_trainer: NEPTrainerV2,
               n_select: int) -> Tuple[List[int], Dict[str, Any]]:
        """
        选择样本
        
        Args:
            candidates: 候选样本
            model_trainer: 当前模型训练器
            n_select: 选择数量
            
        Returns:
            (选中索引列表, 元数据字典)
        """
        pass


class UncertaintySampler(QueryStrategy):
    """
    基于不确定性的采样器
    
    选择模型预测不确定性最高的样本
    """
    
    def __init__(self, method: str = "ensemble"):
        """
        Args:
            method: 不确定性估计方法 (ensemble, dropout, gradient)
        """
        self.method = method
    
    def select(self,
               candidates: List[NEPFrame],
               model_trainer: NEPTrainerV2,
               n_select: int) -> Tuple[List[int], Dict[str, Any]]:
        """选择不确定性最高的样本"""
        
        # 计算不确定性
        uncertainties = self._compute_uncertainty(candidates, model_trainer)
        
        # 选择不确定性最高的n_select个
        selected_indices = np.argsort(uncertainties)[-n_select:][::-1]
        
        metadata = {
            'method': self.method,
            'uncertainties': uncertainties[selected_indices].tolist(),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties)),
        }
        
        return selected_indices.tolist(), metadata
    
    def _compute_uncertainty(self, 
                            candidates: List[NEPFrame],
                            model_trainer: NEPTrainerV2) -> np.ndarray:
        """计算预测不确定性"""
        
        if self.method == "ensemble":
            return self._ensemble_uncertainty(candidates, model_trainer)
        elif self.method == "dropout":
            return self._dropout_uncertainty(candidates, model_trainer)
        elif self.method == "gradient":
            return self._gradient_uncertainty(candidates, model_trainer)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")
    
    def _ensemble_uncertainty(self,
                             candidates: List[NEPFrame],
                             model_trainer: NEPTrainerV2) -> np.ndarray:
        """使用模型集成估计不确定性"""
        # 简化版本: 使用训练损失作为不确定性代理
        # 实际应使用多个模型进行预测
        
        uncertainties = []
        for frame in candidates:
            # 模拟不确定性计算
            # 基于结构复杂度和能量大小
            n_atoms = len(frame.atoms)
            force_magnitude = np.linalg.norm(frame.forces, axis=1).mean()
            
            # 高能量、大力的结构不确定性更高
            uncertainty = force_magnitude * (1 + np.log(n_atoms))
            uncertainties.append(uncertainty)
        
        return np.array(uncertainties)
    
    def _dropout_uncertainty(self,
                            candidates: List[NEPFrame],
                            model_trainer: NEPTrainerV2) -> np.ndarray:
        """使用MC Dropout估计不确定性"""
        # NEP不支持直接的MC Dropout
        # 可以通过多次预测添加噪声来模拟
        return self._ensemble_uncertainty(candidates, model_trainer)
    
    def _gradient_uncertainty(self,
                             candidates: List[NEPFrame],
                             model_trainer: NEPTrainerV2) -> np.ndarray:
        """基于梯度的不确定性"""
        # 简化版本
        return self._ensemble_uncertainty(candidates, model_trainer)


class DiversitySampler(QueryStrategy):
    """
    基于多样性的采样器 (DPP)
    
    使用行列式点过程(DPP)确保选中样本的多样性
    """
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Args:
            lambda_param: 质量与多样性权衡参数
        """
        self.lambda_param = lambda_param
    
    def select(self,
               candidates: List[NEPFrame],
               model_trainer: NEPTrainerV2,
               n_select: int) -> Tuple[List[int], Dict[str, Any]]:
        """
        使用DPP选择多样且高质量的样本
        """
        # 构建特征矩阵
        features = self._featurize_structures(candidates)
        
        # 计算质量分数 (基于不确定性)
        quality_scores = self._compute_quality(candidates, model_trainer)
        
        # 构建DPP核矩阵
        L = self._build_dpp_kernel(features, quality_scores)
        
        # 贪心DPP采样
        selected_indices = self._dpp_greedy_sample(L, n_select)
        
        metadata = {
            'method': 'dpp',
            'diversity_scores': [float(L[i, i]) for i in selected_indices],
            'quality_scores': quality_scores[selected_indices].tolist(),
        }
        
        return selected_indices, metadata
    
    def _featurize_structures(self, frames: List[NEPFrame]) -> np.ndarray:
        """将结构转换为特征向量"""
        features = []
        
        for frame in frames:
            # 简化特征: 原子数、平均位置、位置方差、力大小等
            atoms = frame.atoms
            positions = atoms.get_positions()
            
            feat = [
                len(atoms),
                np.mean(positions),
                np.std(positions),
                np.mean(np.linalg.norm(positions, axis=1)),
                np.mean(np.linalg.norm(frame.forces, axis=1)),
                frame.energy / len(atoms),
            ]
            
            # 添加元素组成特征
            symbols = atoms.get_chemical_symbols()
            unique_symbols = set(symbols)
            for elem in ['H', 'C', 'N', 'O', 'Li', 'Na', 'Si', 'P', 'S', 'Cl']:
                feat.append(symbols.count(elem) / len(atoms) if elem in unique_symbols else 0)
            
            features.append(feat)
        
        # 归一化
        features = np.array(features)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        return features
    
    def _compute_quality(self,
                        candidates: List[NEPFrame],
                        model_trainer: NEPTrainerV2) -> np.ndarray:
        """计算样本质量分数"""
        # 使用不确定性作为质量分数
        sampler = UncertaintySampler()
        return sampler._compute_uncertainty(candidates, model_trainer)
    
    def _build_dpp_kernel(self, 
                         features: np.ndarray,
                         quality_scores: np.ndarray) -> np.ndarray:
        """
        构建DPP核矩阵
        
        L_ij = q_i * q_j * exp(-||x_i - x_j||^2 / sigma^2)
        """
        n = len(features)
        
        # 相似度矩阵
        dist_sq = np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2)
        sigma = np.median(dist_sq) ** 0.5
        S = np.exp(-dist_sq / (2 * sigma ** 2))
        
        # 质量矩阵
        Q = np.outer(quality_scores, quality_scores)
        
        # DPP核
        L = Q * S
        
        return L
    
    def _dpp_greedy_sample(self, L: np.ndarray, n_select: int) -> List[int]:
        """
        贪心DPP采样
        
        迭代选择使行列式最大的样本
        """
        n = len(L)
        selected = []
        remaining = list(range(n))
        
        for _ in range(n_select):
            if not remaining:
                break
            
            # 计算每个候选的边际增益
            gains = []
            for i in remaining:
                if not selected:
                    gain = L[i, i]
                else:
                    # 计算条件概率
                    L_S = L[np.ix_(selected, selected)]
                    L_S_inv = np.linalg.inv(L_S + 1e-6 * np.eye(len(selected)))
                    gain = L[i, i] - L[i, selected] @ L_S_inv @ L[selected, i]
                gains.append(gain)
            
            # 选择增益最大的
            best_idx = remaining[np.argmax(gains)]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected


class HybridSampler(QueryStrategy):
    """
    混合采样器
    
    结合不确定性和多样性进行样本选择
    """
    
    def __init__(self, 
                 uncertainty_weight: float = 0.5,
                 diversity_weight: float = 0.5):
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        
        self.uncertainty_sampler = UncertaintySampler()
        self.diversity_sampler = DiversitySampler()
    
    def select(self,
               candidates: List[NEPFrame],
               model_trainer: NEPTrainerV2,
               n_select: int) -> Tuple[List[int], Dict[str, Any]]:
        """混合策略选择样本"""
        
        # 第一阶段: 使用不确定性预筛选
        pre_select = min(n_select * 3, len(candidates))
        pre_indices, _ = self.uncertainty_sampler.select(
            candidates, model_trainer, pre_select
        )
        
        # 第二阶段: 在预筛选样本中使用多样性选择
        pre_candidates = [candidates[i] for i in pre_indices]
        div_indices, div_metadata = self.diversity_sampler.select(
            pre_candidates, model_trainer, n_select
        )
        
        # 映射回原始索引
        selected_indices = [pre_indices[i] for i in div_indices]
        
        metadata = {
            'method': 'hybrid',
            'pre_selected': pre_select,
            'uncertainty_weight': self.uncertainty_weight,
            'diversity_weight': self.diversity_weight,
            **div_metadata
        }
        
        return selected_indices, metadata


class StructureExplorer:
    """
    结构探索器
    
    生成候选结构进行主动学习
    """
    
    def __init__(self, config: ALConfig):
        self.config = config
    
    def explore_from_md(self,
                       initial_structure: Atoms,
                       model_calculator: Any = None,
                       n_structures: int = 100) -> List[Atoms]:
        """
        使用MD探索构型空间
        
        Args:
            initial_structure: 初始结构
            model_calculator: NEP计算器 (用于MD)
            n_structures: 目标结构数量
            
        Returns:
            探索得到的结构列表
        """
        structures = []
        
        # 在不同温度下运行MD
        for temp in self.config.exploration_temperatures:
            if len(structures) >= n_structures:
                break
            
            atoms = initial_structure.copy()
            
            if model_calculator:
                atoms.calc = model_calculator
            
            # 初始化速度
            MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
            
            # Langevin动力学
            dyn = Langevin(
                atoms,
                self.config.exploration_timestep * units.fs,
                temperature_K=temp,
                friction=0.01
            )
            
            # 运行MD并采样
            for step in range(self.config.exploration_steps):
                dyn.run(10)
                
                # 每隔一定步数采样
                if step % 10 == 0:
                    structures.append(atoms.copy())
                    
                if len(structures) >= n_structures:
                    break
        
        logger.info(f"Exploration generated {len(structures)} structures")
        return structures
    
    def explore_by_perturbation(self,
                                base_structures: List[Atoms],
                                n_perturbations: int = 10,
                                displacement: float = 0.1) -> List[Atoms]:
        """
        通过随机扰动生成结构
        
        Args:
            base_structures: 基础结构
            n_perturbations: 每个结构的扰动数量
            displacement: 扰动幅度 (Å)
            
        Returns:
            扰动后的结构列表
        """
        structures = []
        
        for base in base_structures:
            for _ in range(n_perturbations):
                new_struct = base.copy()
                positions = new_struct.get_positions()
                
                # 添加随机位移
                noise = np.random.randn(*positions.shape) * displacement
                new_struct.set_positions(positions + noise)
                
                structures.append(new_struct)
        
        logger.info(f"Generated {len(structures)} structures by perturbation")
        return structures
    
    def explore_by_strain(self,
                         base_structure: Atoms,
                         strain_range: Tuple[float, float] = (-0.05, 0.05),
                         n_strains: int = 10) -> List[Atoms]:
        """
        通过应变变形生成结构
        
        Args:
            base_structure: 基础结构
            strain_range: 应变范围
            n_strains: 应变数量
            
        Returns:
            变形后的结构列表
        """
        structures = []
        strains = np.linspace(strain_range[0], strain_range[1], n_strains)
        
        for strain in strains:
            new_struct = base_structure.copy()
            cell = new_struct.get_cell()
            
            # 各向同性应变
            new_cell = cell * (1 + strain)
            new_struct.set_cell(new_cell, scale_atoms=True)
            
            structures.append(new_struct)
        
        # 添加剪切应变
        for strain in strains[:n_strains//2]:
            new_struct = base_structure.copy()
            cell = new_struct.get_cell()
            
            # xy剪切
            new_cell = cell.copy()
            new_cell[0, 1] += strain * cell[1, 1]
            new_struct.set_cell(new_cell, scale_atoms=True)
            
            structures.append(new_struct)
        
        logger.info(f"Generated {len(structures)} structures by strain")
        return structures


class NEPActiveLearning:
    """
    NEP主动学习工作流
    
    整合探索、查询、标注、重训练的完整主动学习流程
    """
    
    def __init__(self,
                 trainer: NEPTrainerV2,
                 dft_calculator: Optional[Callable] = None,
                 config: Optional[ALConfig] = None):
        self.trainer = trainer
        self.dft_calculator = dft_calculator
        self.config = config or ALConfig()
        
        # 查询策略
        self.strategy = self._create_strategy()
        
        # 结构探索器
        self.explorer = StructureExplorer(self.config)
        
        # 数据存储
        self.labeled_data: List[NEPFrame] = []
        self.iteration = 0
        self.history = []
        self.total_cost = 0.0
    
    def _create_strategy(self) -> QueryStrategy:
        """创建查询策略"""
        if self.config.strategy == "uncertainty":
            return UncertaintySampler()
        elif self.config.strategy == "diversity":
            return DiversitySampler(self.config.diversity_lambda)
        elif self.config.strategy == "hybrid":
            return HybridSampler()
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def initialize(self, initial_structures: List[Atoms]):
        """
        使用初始结构初始化
        
        运行DFT计算获取初始训练数据
        """
        logger.info(f"Initializing with {len(initial_structures)} structures")
        
        # DFT计算
        if self.dft_calculator:
            results = self.dft_calculator(initial_structures)
            
            for struct, result in zip(initial_structures, results):
                frame = NEPFrame(
                    atoms=struct,
                    energy=result['energy'],
                    forces=result['forces'],
                    stress=result.get('stress'),
                    virial=result.get('virial')
                )
                self.labeled_data.append(frame)
                self.total_cost += self.config.cost_per_dft
        
        logger.info(f"Initialization complete: {len(self.labeled_data)} labeled structures")
    
    def run_iteration(self, 
                     candidate_structures: List[Atoms]) -> Dict[str, Any]:
        """
        运行一个主动学习迭代
        
        Args:
            candidate_structures: 候选结构池
            
        Returns:
            迭代结果统计
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Active Learning Iteration {self.iteration}")
        logger.info(f"{'='*60}")
        
        # 1. 当前模型训练
        logger.info("Training current model...")
        train_dataset = NEPDataset(frames=self.labeled_data)
        self.trainer.setup_training(train_dataset)
        model_path = self.trainer.train(verbose=False)
        
        # 2. 将候选结构转换为NEPFrame
        candidates = []
        for struct in candidate_structures:
            # 使用当前模型进行预测 (简化)
            frame = NEPFrame(
                atoms=struct,
                energy=0.0,  # 未标注
                forces=np.zeros((len(struct), 3))
            )
            candidates.append(frame)
        
        # 3. 选择样本
        n_select = min(self.config.n_samples_per_iteration, len(candidates))
        selected_indices, selection_metadata = self.strategy.select(
            candidates, self.trainer, n_select
        )
        
        selected_structures = [candidate_structures[i] for i in selected_indices]
        logger.info(f"Selected {len(selected_structures)} structures for DFT")
        
        # 4. DFT标注
        if self.dft_calculator and len(selected_structures) > 0:
            results = self.dft_calculator(selected_structures)
            
            for struct, result in zip(selected_structures, results):
                frame = NEPFrame(
                    atoms=struct,
                    energy=result['energy'],
                    forces=result['forces'],
                    stress=result.get('stress'),
                    virial=result.get('virial')
                )
                self.labeled_data.append(frame)
                self.total_cost += self.config.cost_per_dft
        
        # 5. 记录
        iteration_info = {
            'iteration': self.iteration,
            'n_selected': len(selected_structures),
            'total_labeled': len(self.labeled_data),
            'total_cost': self.total_cost,
            'selection_metadata': selection_metadata,
        }
        self.history.append(iteration_info)
        
        self.iteration += 1
        
        return iteration_info
    
    def run(self,
           initial_structures: List[Atoms],
           base_structure: Atoms,
           max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        运行完整主动学习流程
        
        Args:
            initial_structures: 初始标注结构
            base_structure: 基础结构 (用于探索)
            max_iterations: 最大迭代次数
            
        Returns:
            最终结果统计
        """
        max_iterations = max_iterations or self.config.max_iterations
        
        # 初始化
        self.initialize(initial_structures)
        
        for i in range(max_iterations):
            # 探索新结构
            candidates = self.explorer.explore_by_perturbation([base_structure])
            
            # 运行迭代
            iteration_info = self.run_iteration(candidates)
            
            # 检查收敛
            if self._check_convergence():
                logger.info("Convergence achieved!")
                break
        
        return self.get_summary()
    
    def _check_convergence(self) -> bool:
        """检查是否收敛"""
        if len(self.history) < self.config.convergence_patience:
            return False
        
        # 检查最近几轮的选择数量
        recent_selections = [h['n_selected'] for h in self.history[-self.config.convergence_patience:]]
        
        # 如果没有新样本被选择，认为已收敛
        if all(n == 0 for n in recent_selections):
            return True
        
        # 或者不确定性低于阈值
        if len(self.history) > 0 and 'selection_metadata' in self.history[-1]:
            mean_unc = self.history[-1]['selection_metadata'].get('mean_uncertainty', 0)
            if mean_unc < self.config.min_uncertainty:
                return True
        
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """获取主动学习摘要"""
        return {
            'total_iterations': self.iteration,
            'total_labeled': len(self.labeled_data),
            'total_cost': self.total_cost,
            'history': self.history,
            'final_model': self.trainer.working_dir / "nep.txt"
        }
    
    def save_results(self, output_dir: str):
        """保存结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存历史
        import json
        with open(output_dir / 'al_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 保存标注数据
        dataset = NEPDataset(frames=self.labeled_data)
        dataset.save_xyz(str(output_dir / 'labeled_data.xyz'))
        
        # 保存摘要
        summary = self.get_summary()
        with open(output_dir / 'al_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Active learning results saved to {output_dir}")
