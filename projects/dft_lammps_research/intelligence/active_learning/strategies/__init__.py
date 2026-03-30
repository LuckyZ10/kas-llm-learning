#!/usr/bin/env python3
"""
主动学习策略模块 - Active Learning Strategies

实现5种先进的主动学习策略：
1. BayesianOptimizationStrategy - 贝叶斯优化策略
2. DPPDiversityStrategy - DPP多样性策略
3. MultiFidelityStrategy - 多保真度策略
4. EvidentialLearningStrategy - 证据学习策略
5. AdaptiveHybridStrategy - 自适应混合策略
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from collections import defaultdict
import logging
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """策略配置基类"""
    batch_size: int = 10
    random_state: Optional[int] = None
    verbose: bool = True


@dataclass
class SelectionResult:
    """选择结果"""
    selected_indices: np.ndarray  # 选中的样本索引
    selected_scores: np.ndarray   # 选择分数
    acquisition_values: Optional[np.ndarray] = None  # 采集函数值
    diversity_scores: Optional[np.ndarray] = None    # 多样性分数
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self):
        return len(self.selected_indices)


class ActiveLearningStrategy(ABC):
    """主动学习策略基类"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.iteration = 0
        self.history = []
        
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
    
    @abstractmethod
    def select(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray] = None,
        y_labeled: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> SelectionResult:
        """
        从未标注数据中选择样本
        
        Args:
            X_unlabeled: 未标注特征 (n_unlabeled, n_features)
            X_labeled: 已标注特征 (n_labeled, n_features)
            y_labeled: 已标注标签 (n_labeled,)
            model: 当前训练好的模型
            
        Returns:
            SelectionResult
        """
        pass
    
    def update(self, selected_indices: np.ndarray, feedback: Optional[Dict] = None):
        """更新策略状态（在选择后调用）"""
        self.iteration += 1
        self.history.append({
            'iteration': self.iteration,
            'selected_indices': selected_indices,
            'feedback': feedback
        })
    
    def get_name(self) -> str:
        """获取策略名称"""
        return self.__class__.__name__


# ==============================================================================
# 策略1: 贝叶斯优化策略
# ==============================================================================

class BayesianOptimizationStrategy(ActiveLearningStrategy):
    """
    贝叶斯优化主动学习策略
    
    使用高斯过程建模目标函数，通过采集函数选择下一个最有信息量的样本。
    特别适用于昂贵的DFT计算，能有效平衡探索与利用。
    
    采集函数选项:
    - UCB (Upper Confidence Bound): 上置信界
    - EI (Expected Improvement): 期望改进
    - PI (Probability of Improvement): 改进概率
    - BALD (Bayesian Active Learning by Disagreement): 贝叶斯主动学习分歧
    
    Reference: Frazier, "A Tutorial on Bayesian Optimization", arXiv 2018
    
    Attributes:
        acquisition: 采集函数类型
        beta_ucb: UCB参数 (探索-利用权衡)
        xi_ei: EI参数
        use_batch: 是否使用批量BO
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        acquisition: str = 'ucb',
        beta_ucb: float = 2.0,
        xi_ei: float = 0.01,
        kernel: Optional[Any] = None,
        use_batch: bool = False,
        local_penalty: float = 1.0
    ):
        super().__init__(config)
        self.acquisition = acquisition.lower()
        self.beta_ucb = beta_ucb
        self.xi_ei = xi_ei
        self.kernel = kernel
        self.use_batch = use_batch
        self.local_penalty = local_penalty
        self.gp = None
    
    def select(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray] = None,
        y_labeled: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> SelectionResult:
        """
        使用贝叶斯优化选择样本
        """
        if X_labeled is None or y_labeled is None or len(X_labeled) < 2:
            # 冷启动：随机选择
            logger.info("Cold start: random selection")
            indices = np.random.choice(
                len(X_unlabeled),
                min(self.config.batch_size, len(X_unlabeled)),
                replace=False
            )
            return SelectionResult(
                selected_indices=indices,
                selected_scores=np.ones(len(indices)) / len(indices)
            )
        
        # 训练高斯过程
        self._fit_gp(X_labeled, y_labeled)
        
        if self.use_batch:
            return self._batch_selection(X_unlabeled, X_labeled)
        else:
            return self._sequential_selection(X_unlabeled, X_labeled)
    
    def _fit_gp(self, X: np.ndarray, y: np.ndarray):
        """拟合高斯过程模型"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
        
        if self.kernel is None:
            self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        self.gp.fit(X, y)
        logger.debug(f"GP fitted with kernel: {self.gp.kernel_}")
    
    def _sequential_selection(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: np.ndarray
    ) -> SelectionResult:
        """顺序选择"""
        selected_indices = []
        selected_scores = []
        acquisition_values = []
        
        candidates = X_unlabeled.copy()
        candidate_indices = np.arange(len(X_unlabeled))
        
        for _ in range(min(self.config.batch_size, len(X_unlabeled))):
            # 计算采集函数值
            acq_values = self._compute_acquisition(candidates)
            
            # 选择最大值
            best_idx = np.argmax(acq_values)
            selected_indices.append(candidate_indices[best_idx])
            selected_scores.append(acq_values[best_idx])
            acquisition_values.append(acq_values[best_idx])
            
            # 从候选集中移除
            candidates = np.delete(candidates, best_idx, axis=0)
            candidate_indices = np.delete(candidate_indices, best_idx)
            
            if len(candidates) == 0:
                break
        
        return SelectionResult(
            selected_indices=np.array(selected_indices),
            selected_scores=np.array(selected_scores),
            acquisition_values=np.array(acquisition_values)
        )
    
    def _batch_selection(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: np.ndarray
    ) -> SelectionResult:
        """
        批量选择 (使用局部惩罚避免重复)
        
        Reference: Gonzalez et al. "Batch Bayesian Optimization via Local Penalization", AISTATS 2016
        """
        selected_indices = []
        acquisition_values = []
        
        # 计算初始采集函数值
        acq_values = self._compute_acquisition(X_unlabeled)
        
        for _ in range(min(self.config.batch_size, len(X_unlabeled))):
            # 选择最大值
            available_mask = np.ones(len(X_unlabeled), dtype=bool)
            available_mask[selected_indices] = False
            
            if not np.any(available_mask):
                break
            
            best_idx = np.argmax(acq_values * available_mask - (~available_mask) * 1e10)
            selected_indices.append(best_idx)
            acquisition_values.append(acq_values[best_idx])
            
            # 局部惩罚：降低附近点的采集函数值
            distances = cdist([X_unlabeled[best_idx]], X_unlabeled).squeeze()
            penalty = np.exp(-self.local_penalty * distances)
            acq_values = acq_values * (1 - penalty)
        
        return SelectionResult(
            selected_indices=np.array(selected_indices),
            selected_scores=np.array(acquisition_values),
            acquisition_values=np.array(acquisition_values)
        )
    
    def _compute_acquisition(self, X: np.ndarray) -> np.ndarray:
        """计算采集函数值"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if self.acquisition == 'ucb':
            # Upper Confidence Bound
            return mu + self.beta_ucb * sigma
        
        elif self.acquisition == 'lcb':
            # Lower Confidence Bound (for minimization)
            return -(mu - self.beta_ucb * sigma)
        
        elif self.acquisition == 'ei':
            # Expected Improvement
            if hasattr(self.gp, 'y_train_'):
                y_best = np.min(self.gp.y_train_)
            else:
                y_best = 0
            
            with np.errstate(divide='warn'):
                imp = y_best - mu - self.xi_ei
                Z = imp / (sigma + 1e-10)
                ei = imp * (0.5 * (1 + np.sign(Z)))  # 简化版EI
                ei = imp * np.minimum(1, np.maximum(0, 0.5 + 0.5 * Z))  # 近似
                
            return ei
        
        elif self.acquisition == 'pi':
            # Probability of Improvement
            if hasattr(self.gp, 'y_train_'):
                y_best = np.min(self.gp.y_train_)
            else:
                y_best = 0
            
            with np.errstate(divide='warn'):
                Z = (y_best - mu - self.xi_ei) / (sigma + 1e-10)
            from scipy.stats import norm
            pi = norm.cdf(Z)
            return pi
        
        elif self.acquisition == 'bald':
            # BALD: 选择预测方差最大的点
            return sigma ** 2
        
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")


# ==============================================================================
# 策略2: DPP多样性策略
# ==============================================================================

class DPPDiversityStrategy(ActiveLearningStrategy):
    """
    DPP (Determinantal Point Process) 多样性感知批量选择策略
    
    使用DPP在批量选择中平衡样本的信息量和多样性。
    高质量且彼此不相似的样本更有可能被选中。
    
    核心思想:
    L_ij = q_i * phi_i^T * phi_j * q_j
    - q_i: 样本i的质量分数 (不确定性)
    - phi_i: 样本i的特征表示
    - det(L_S): 子集S的概率正比于其体积
    
    Reference: Kulesza & Taskar, "Determinantal Point Processes for Machine Learning", 2012
    
    Attributes:
        quality_fn: 质量函数 ('uncertainty', 'random', 'distance')
        diversity_weight: 多样性权重
        use_greedy: 是否使用贪心算法 (更快)
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        quality_fn: str = 'uncertainty',
        diversity_weight: float = 1.0,
        quality_weight: float = 1.0,
        use_greedy: bool = True,
        kernel_type: str = 'rbf',
        gamma: float = 1.0
    ):
        super().__init__(config)
        self.quality_fn = quality_fn
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
        self.use_greedy = use_greedy
        self.kernel_type = kernel_type
        self.gamma = gamma
    
    def select(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray] = None,
        y_labeled: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> SelectionResult:
        """
        使用DPP选择多样且有信息量的样本
        """
        n_samples = len(X_unlabeled)
        batch_size = min(self.config.batch_size, n_samples)
        
        # 计算质量分数
        quality_scores = self._compute_quality_scores(
            X_unlabeled, X_labeled, y_labeled, model
        )
        
        # 构建相似度矩阵
        similarity_matrix = self._compute_similarity(X_unlabeled)
        
        # 构建L-ensemble矩阵
        L = self._build_l_ensemble(quality_scores, similarity_matrix)
        
        if self.use_greedy:
            selected_indices, diversity_scores = self._greedy_map_inference(L, batch_size)
        else:
            selected_indices, diversity_scores = self._exact_sampling(L, batch_size)
        
        return SelectionResult(
            selected_indices=selected_indices,
            selected_scores=quality_scores[selected_indices],
            acquisition_values=quality_scores[selected_indices],
            diversity_scores=diversity_scores,
            metadata={'kernel_type': self.kernel_type}
        )
    
    def _compute_quality_scores(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray],
        y_labeled: Optional[np.ndarray],
        model: Optional[Any]
    ) -> np.ndarray:
        """计算样本质量分数"""
        if self.quality_fn == 'uncertainty':
            if model is not None and hasattr(model, 'predict'):
                # 使用模型不确定性
                try:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_unlabeled)
                        # 熵作为不确定性
                        scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                    elif hasattr(model, 'predict'):
                        # 使用距离最近样本的距离作为不确定性代理
                        if X_labeled is not None:
                            distances = cdist(X_unlabeled, X_labeled)
                            scores = np.min(distances, axis=1)
                        else:
                            scores = np.random.random(len(X_unlabeled))
                    else:
                        scores = np.random.random(len(X_unlabeled))
                except:
                    scores = np.random.random(len(X_unlabeled))
            else:
                scores = np.random.random(len(X_unlabeled))
        
        elif self.quality_fn == 'distance':
            # 使用密度估计 (距离越稀疏分数越高)
            if X_labeled is not None:
                distances = cdist(X_unlabeled, X_labeled)
                scores = np.min(distances, axis=1)
            else:
                scores = np.ones(len(X_unlabeled))
        
        elif self.quality_fn == 'random':
            scores = np.random.random(len(X_unlabeled))
        
        else:
            scores = np.random.random(len(X_unlabeled))
        
        return scores
    
    def _compute_similarity(self, X: np.ndarray) -> np.ndarray:
        """计算样本间的相似度矩阵"""
        if self.kernel_type == 'rbf':
            # RBF核
            distances = cdist(X, X, 'euclidean')
            similarity = np.exp(-self.gamma * distances ** 2)
        elif self.kernel_type == 'linear':
            similarity = X @ X.T
        elif self.kernel_type == 'cosine':
            similarity = 1 - cdist(X, X, 'cosine')
            similarity = np.nan_to_num(similarity, nan=0.0)
        else:
            similarity = np.exp(-self.gamma * cdist(X, X, 'euclidean') ** 2)
        
        return similarity
    
    def _build_l_ensemble(
        self,
        quality_scores: np.ndarray,
        similarity_matrix: np.ndarray
    ) -> np.ndarray:
        """
        构建L-ensemble矩阵
        
        L_ij = q_i * S_ij * q_j
        其中 q 是质量分数，S 是相似度矩阵
        """
        q = np.sqrt(quality_scores * self.quality_weight)
        L = q[:, None] * similarity_matrix * q[None, :]
        # 添加正则化确保正定性
        L += 1e-6 * np.eye(len(L))
        return L
    
    def _greedy_map_inference(
        self,
        L: np.ndarray,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        贪心MAP推断算法
        
        参考: Kulesza & Taskar, 2012
        """
        n = len(L)
        selected = []
        diversity_scores = []
        
        # 使用Cholesky分解提高效率
        try:
            L_chol = np.linalg.cholesky(L)
        except:
            # 如果L不是正定矩阵，添加正则化
            L = L + 1e-5 * np.eye(n)
            L_chol = np.linalg.cholesky(L)
        
        C = np.zeros((n, batch_size))
        d = np.diag(L).copy()
        
        for i in range(batch_size):
            # 选择使边际增益最大的元素
            if i == 0:
                j = np.argmax(d)
            else:
                # 确保选择未选过的
                available_mask = np.ones(n, dtype=bool)
                available_mask[selected] = False
                d_masked = np.where(available_mask, d, -np.inf)
                j = np.argmax(d_masked)
            
            if j in selected:
                # 如果已选过，选择下一个最佳
                available_mask = np.ones(n, dtype=bool)
                available_mask[selected] = False
                available_indices = np.where(available_mask)[0]
                if len(available_indices) == 0:
                    break
                j = available_indices[0]
            
            selected.append(j)
            diversity_scores.append(d[j])
            
            # 更新Cholesky因子
            if i < batch_size - 1 and len(selected) < n:
                C[j, i] = np.sqrt(max(d[j], 0))
                remaining = [k for k in range(n) if k not in selected]
                
                for k in remaining:
                    if i == 0:
                        C[k, i] = L[k, j] / (C[j, i] + 1e-10)
                    else:
                        C[k, i] = (L[k, j] - np.sum(C[k, :i] * C[j, :i])) / (C[j, i] + 1e-10)
                    d[k] -= C[k, i] ** 2
                    d[k] = max(d[k], 0)  # 确保非负
        
        return np.array(selected), np.array(diversity_scores)
    
    def _exact_sampling(self, L: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """精确采样 (对于小规模数据)"""
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # 从k-DPP采样
        selected = []
        probs = eigenvalues / (eigenvalues + 1)
        
        # 选择要保留的特征向量
        J = np.where(np.random.random(len(probs)) < probs)[0]
        
        V = eigenvectors[:, J]
        
        while len(selected) < batch_size and V.shape[1] > 0:
            # 计算选择概率
            probs = np.sum(V ** 2, axis=1)
            probs = np.maximum(probs, 0) / np.sum(np.maximum(probs, 0))
            
            # 采样
            i = np.random.choice(len(probs), p=probs)
            selected.append(i)
            
            # 正交化
            if V.shape[1] > 1:
                v_i = V[i, :]
                V = V - np.outer(V @ v_i, v_i) / (v_i @ v_i + 1e-10)
                # QR分解保持正交性
                V, _ = np.linalg.qr(V)
        
        diversity_scores = np.ones(len(selected))  # 简化
        return np.array(selected[:batch_size]), diversity_scores


# ==============================================================================
# 策略3: 多保真度策略
# ==============================================================================

class MultiFidelityStrategy(ActiveLearningStrategy):
    """
    多保真度主动学习策略
    
    利用不同保真度的计算资源（如经典力场 < ML势 < DFT < CCSD(T)）
    高效构建数据集。高保真度数据昂贵但准确，低保真度数据便宜但近似。
    
    策略:
    1. 使用低保真度模型快速探索
    2. 识别需要高保真度计算的区域
    3. 使用delta learning (Δ-learning) 结合不同保真度
    
    Reference: Ghosh et al. "Active learning of molecular data for task-specific objectives", J. Chem. Phys. 2025
    
    Attributes:
        fidelity_levels: 保真度级别配置
        fidelity_costs: 各级别计算成本
        current_fidelity: 当前使用的保真度级别
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        fidelity_levels: Optional[List[Dict]] = None,
        fidelity_costs: Optional[List[float]] = None,
        strategy: str = 'adaptive_threshold',
        delta_learning: bool = True
    ):
        super().__init__(config)
        
        self.fidelity_levels = fidelity_levels or [
            {'name': 'low', 'method': 'classical_ff'},
            {'name': 'medium', 'method': 'ml_potential'},
            {'name': 'high', 'method': 'dft_pbe'},
            {'name': 'highest', 'method': 'dft_hse'},
        ]
        self.fidelity_costs = fidelity_costs or [1.0, 10.0, 1000.0, 10000.0]
        self.strategy = strategy
        self.delta_learning = delta_learning
        
        self.current_fidelity = 0  # 从最低保真度开始
        self.fidelity_data = defaultdict(list)  # 各级别的数据
        self.delta_models = {}  # Delta学习模型
    
    def select(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray] = None,
        y_labeled: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> SelectionResult:
        """
        多保真度选择策略
        """
        # 确定当前应该使用的保真度级别
        target_fidelity = self._determine_target_fidelity(X_labeled, y_labeled, model)
        
        if self.strategy == 'simple':
            return self._simple_selection(X_unlabeled, target_fidelity)
        elif self.strategy == 'adaptive_threshold':
            return self._adaptive_threshold_selection(X_unlabeled, X_labeled, y_labeled, target_fidelity)
        elif self.strategy == 'information_gain':
            return self._information_gain_selection(X_unlabeled, X_labeled, y_labeled, model, target_fidelity)
        else:
            return self._simple_selection(X_unlabeled, target_fidelity)
    
    def _determine_target_fidelity(
        self,
        X_labeled: Optional[np.ndarray],
        y_labeled: Optional[np.ndarray],
        model: Optional[Any]
    ) -> int:
        """确定目标保真度级别"""
        # 根据当前数据量和模型性能决定
        if X_labeled is None or len(X_labeled) < 10:
            return 0  # 从低保真度开始
        elif len(X_labeled) < 100:
            return 1
        elif len(X_labeled) < 500:
            return 2
        else:
            return 2  # 主要在DFT级别
    
    def _simple_selection(
        self,
        X_unlabeled: np.ndarray,
        target_fidelity: int
    ) -> SelectionResult:
        """简单随机选择"""
        n_select = min(self.config.batch_size, len(X_unlabeled))
        indices = np.random.choice(len(X_unlabeled), n_select, replace=False)
        
        return SelectionResult(
            selected_indices=indices,
            selected_scores=np.ones(n_select),
            metadata={'target_fidelity': target_fidelity}
        )
    
    def _adaptive_threshold_selection(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray],
        y_labeled: Optional[np.ndarray],
        target_fidelity: int
    ) -> SelectionResult:
        """
        自适应阈值选择
        
        如果与当前训练集差异大，选择高保真度
        如果相似，使用低保真度
        """
        if X_labeled is None or len(X_labeled) == 0:
            # 冷启动：均匀采样
            indices = np.random.choice(
                len(X_unlabeled),
                min(self.config.batch_size, len(X_unlabeled)),
                replace=False
            )
            return SelectionResult(
                selected_indices=indices,
                selected_scores=np.ones(len(indices)),
                metadata={'target_fidelity': target_fidelity}
            )
        
        # 计算与训练集的最小距离
        distances = cdist(X_unlabeled, X_labeled)
        min_distances = np.min(distances, axis=1)
        
        # 根据距离确定需要的保真度
        # 距离大 = 新颖区域 = 需要高保真度
        threshold = np.percentile(min_distances, 50)
        
        high_fidelity_mask = min_distances > threshold
        high_fidelity_indices = np.where(high_fidelity_mask)[0]
        low_fidelity_indices = np.where(~high_fidelity_mask)[0]
        
        # 优先选择高保真度样本
        n_high = min(self.config.batch_size // 2, len(high_fidelity_indices))
        n_low = self.config.batch_size - n_high
        
        selected = []
        fidelity_map = {}
        
        if n_high > 0:
            high_selected = np.random.choice(high_fidelity_indices, n_high, replace=False)
            selected.extend(high_selected)
            for idx in high_selected:
                fidelity_map[idx] = target_fidelity
        
        if n_low > 0 and len(low_fidelity_indices) > 0:
            n_low = min(n_low, len(low_fidelity_indices))
            low_selected = np.random.choice(low_fidelity_indices, n_low, replace=False)
            selected.extend(low_selected)
            for idx in low_selected:
                fidelity_map[idx] = max(0, target_fidelity - 1)
        
        scores = min_distances[selected] if len(selected) > 0 else np.ones(len(selected))
        
        return SelectionResult(
            selected_indices=np.array(selected),
            selected_scores=scores,
            metadata={
                'target_fidelity': target_fidelity,
                'fidelity_map': fidelity_map
            }
        )
    
    def _information_gain_selection(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray],
        y_labeled: Optional[np.ndarray],
        model: Optional[Any],
        target_fidelity: int
    ) -> SelectionResult:
        """
        基于信息增益的选择
        
        选择能够最大化信息增益的样本和保真度组合
        """
        n_samples = len(X_unlabeled)
        batch_size = min(self.config.batch_size, n_samples)
        
        # 计算信息增益 (简化版本：不确定性 / 成本)
        if model is not None and hasattr(model, 'predict'):
            try:
                # 使用预测方差作为信息代理
                predictions = []
                for _ in range(10):  # 使用dropout或集成
                    pred = model.predict(X_unlabeled)
                    predictions.append(pred)
                uncertainty = np.var(predictions, axis=0)
                if len(uncertainty.shape) > 1:
                    uncertainty = np.mean(uncertainty, axis=-1)
            except:
                uncertainty = np.ones(n_samples)
        else:
            uncertainty = np.ones(n_samples)
        
        # 计算信息增益/成本比
        cost = self.fidelity_costs[target_fidelity]
        info_gain_ratio = uncertainty / (cost + 1e-10)
        
        # 选择top-k
        selected_indices = np.argsort(info_gain_ratio)[-batch_size:][::-1]
        
        return SelectionResult(
            selected_indices=selected_indices,
            selected_scores=uncertainty[selected_indices],
            acquisition_values=info_gain_ratio[selected_indices],
            metadata={'target_fidelity': target_fidelity}
        )
    
    def get_fidelity_for_sample(self, sample_idx: int, result: SelectionResult) -> int:
        """获取特定样本应该使用的保真度级别"""
        if 'fidelity_map' in result.metadata:
            return result.metadata['fidelity_map'].get(sample_idx, result.metadata.get('target_fidelity', 0))
        return result.metadata.get('target_fidelity', 0)


# ==============================================================================
# 策略4: 证据学习策略
# ==============================================================================

class EvidentialLearningStrategy(ActiveLearningStrategy):
    """
    证据学习主动学习策略
    
    利用证据学习网络同时估计认知不确定性和偶然不确定性，
    优先选择认知不确定性高的样本（模型不确定但数据一致）。
    
    优势:
    1. 单次前向传播即可获得不确定性
    2. 明确区分认知不确定性和偶然不确定性
    3. 避免重复采样噪声大的区域
    
    Reference: Amini et al. "Deep Evidential Regression", NeurIPS 2020
    
    Attributes:
        use_epistemic: 是否使用认知不确定性
        use_evidence: 是否考虑证据强度
        min_evidence_threshold: 最小证据阈值
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        use_epistemic: bool = True,
        use_evidence: bool = True,
        min_evidence_threshold: float = 0.1,
        uncertainty_weight: float = 1.0,
        evidence_weight: float = 0.5
    ):
        super().__init__(config)
        self.use_epistemic = use_epistemic
        self.use_evidence = use_evidence
        self.min_evidence_threshold = min_evidence_threshold
        self.uncertainty_weight = uncertainty_weight
        self.evidence_weight = evidence_weight
        self.evidential_model = None
    
    def select(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray] = None,
        y_labeled: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> SelectionResult:
        """
        使用证据学习不确定性选择样本
        """
        n_samples = len(X_unlabeled)
        batch_size = min(self.config.batch_size, n_samples)
        
        # 获取不确定性估计
        uncertainties, evidence = self._compute_evidential_uncertainty(
            X_unlabeled, X_labeled, y_labeled
        )
        
        # 计算选择分数
        if self.use_epistemic and self.use_evidence:
            # 平衡不确定性和证据
            # 高不确定性 + 低证据 = 需要探索
            selection_score = (
                self.uncertainty_weight * uncertainties -
                self.evidence_weight * evidence
            )
        elif self.use_epistemic:
            selection_score = uncertainties
        else:
            selection_score = -evidence  # 选择证据弱的区域
        
        # 排除证据太强的区域（已经学得很好）
        if self.use_evidence:
            mask = evidence < 1 - self.min_evidence_threshold
            selection_score = selection_score * mask - (~mask) * 1e10
        
        # 选择top-k
        selected_indices = np.argsort(selection_score)[-batch_size:][::-1]
        
        return SelectionResult(
            selected_indices=selected_indices,
            selected_scores=uncertainties[selected_indices],
            acquisition_values=selection_score[selected_indices],
            metadata={
                'epistemic_uncertainty': uncertainties[selected_indices],
                'evidence': evidence[selected_indices]
            }
        )
    
    def _compute_evidential_uncertainty(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray],
        y_labeled: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算证据学习不确定性
        
        Returns:
            (uncertainties, evidence)
        """
        try:
            import torch
            from ..uncertainty import EvidentialUncertainty
        except ImportError:
            logger.warning("PyTorch not available, using fallback uncertainty")
            return self._fallback_uncertainty(X_unlabeled, X_labeled)
        
        # 训练或更新证据学习模型
        if self.evidential_model is None or X_labeled is not None:
            self.evidential_model = EvidentialUncertainty(
                input_dim=X_unlabeled.shape[1],
                n_epochs=200,
                random_state=self.config.random_state
            )
            
            if X_labeled is not None and y_labeled is not None:
                self.evidential_model.fit(X_labeled, y_labeled)
        
        # 预测不确定性
        if self.evidential_model.is_fitted:
            result = self.evidential_model.quantify(X_unlabeled)
            uncertainties = result.epistemic_uncertainty
            evidence = result.confidence  # 使用置信度作为证据度量
        else:
            uncertainties, evidence = self._fallback_uncertainty(X_unlabeled, X_labeled)
        
        return uncertainties, evidence
    
    def _fallback_uncertainty(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """当PyTorch不可用时使用的备用不确定性估计"""
        n_samples = len(X_unlabeled)
        
        if X_labeled is not None and len(X_labeled) > 0:
            # 使用距离作为不确定性代理
            distances = cdist(X_unlabeled, X_labeled)
            uncertainties = np.min(distances, axis=1)
            # 归一化
            uncertainties = uncertainties / (np.max(uncertainties) + 1e-10)
        else:
            uncertainties = np.random.random(n_samples)
        
        # 证据 = 1 - 不确定性
        evidence = 1 - uncertainties
        
        return uncertainties, evidence


# ==============================================================================
# 策略5: 自适应混合策略
# ==============================================================================

class AdaptiveHybridStrategy(ActiveLearningStrategy):
    """
    自适应混合主动学习策略
    
    动态组合多种策略，根据当前训练状态自动选择最优策略或策略组合。
    
    策略选择机制:
    1. 早期: 探索为主 (贝叶斯优化、多样性)
    2. 中期: 利用为主 (不确定性、证据学习)
    3. 后期: 精细调整 (多保真度、局部优化)
    
    也可以基于元学习动态调整策略权重。
    
    Attributes:
        strategies: 策略列表
        strategy_weights: 策略权重
        selection_mode: 选择模式 ('weighted', 'best', 'adaptive')
        performance_window: 性能评估窗口大小
    """
    
    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        strategies: Optional[List[ActiveLearningStrategy]] = None,
        strategy_weights: Optional[np.ndarray] = None,
        selection_mode: str = 'adaptive',
        performance_window: int = 3,
        exploration_ratio: float = 0.3
    ):
        super().__init__(config)
        
        self.strategies = strategies or self._default_strategies()
        n_strategies = len(self.strategies)
        
        if strategy_weights is None:
            self.strategy_weights = np.ones(n_strategies) / n_strategies
        else:
            self.strategy_weights = np.array(strategy_weights)
        
        self.selection_mode = selection_mode
        self.performance_window = performance_window
        self.exploration_ratio = exploration_ratio
        
        # 性能追踪
        self.strategy_performance = defaultdict(list)
        self.strategy_usage_count = defaultdict(int)
    
    def _default_strategies(self) -> List[ActiveLearningStrategy]:
        """创建默认策略组合"""
        return [
            BayesianOptimizationStrategy(StrategyConfig(batch_size=self.config.batch_size)),
            DPPDiversityStrategy(StrategyConfig(batch_size=self.config.batch_size)),
            EvidentialLearningStrategy(StrategyConfig(batch_size=self.config.batch_size)),
        ]
    
    def select(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray] = None,
        y_labeled: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> SelectionResult:
        """
        自适应混合选择
        """
        n_samples = len(X_unlabeled)
        batch_size = min(self.config.batch_size, n_samples)
        
        # 根据训练阶段调整策略权重
        if self.selection_mode == 'adaptive':
            weights = self._compute_adaptive_weights(X_labeled, y_labeled)
        else:
            weights = self.strategy_weights
        
        # 选择使用哪个策略
        if self.selection_mode == 'best':
            # 选择性能最好的策略
            best_strategy_idx = self._get_best_strategy()
            selected_strategy = self.strategies[best_strategy_idx]
            result = selected_strategy.select(X_unlabeled, X_labeled, y_labeled, model, **kwargs)
            result.metadata['strategy_used'] = best_strategy_idx
            result.metadata['strategy_name'] = selected_strategy.get_name()
        
        else:
            # 加权组合多个策略的结果
            result = self._weighted_selection(
                X_unlabeled, X_labeled, y_labeled, model, weights, **kwargs
            )
        
        # 记录策略使用
        self._record_strategy_usage(result.metadata.get('strategy_used', 0), result)
        
        return result
    
    def _compute_adaptive_weights(
        self,
        X_labeled: Optional[np.ndarray],
        y_labeled: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        根据当前训练状态计算自适应权重
        
        训练阶段:
        - 早期 (n < 50): 强调探索 (贝叶斯优化、多样性)
        - 中期 (50 <= n < 500): 平衡探索与利用
        - 后期 (n >= 500): 强调利用 (不确定性、精细调整)
        """
        n_labeled = len(X_labeled) if X_labeled is not None else 0
        n_strategies = len(self.strategies)
        
        if n_labeled < 50:
            # 早期：探索为主
            weights = np.array([0.5, 0.4, 0.1, 0.0, 0.0][:n_strategies])
        elif n_labeled < 200:
            # 早期到中期
            weights = np.array([0.3, 0.3, 0.3, 0.1, 0.0][:n_strategies])
        elif n_labeled < 500:
            # 中期
            weights = np.array([0.2, 0.2, 0.4, 0.2, 0.0][:n_strategies])
        else:
            # 后期：利用为主
            weights = np.array([0.1, 0.1, 0.4, 0.4, 0.0][:n_strategies])
        
        # 归一化
        weights = weights / np.sum(weights)
        return weights[:n_strategies]
    
    def _get_best_strategy(self) -> int:
        """获取历史性能最好的策略"""
        if not self.strategy_performance:
            return 0
        
        avg_performance = []
        for i in range(len(self.strategies)):
            if len(self.strategy_performance[i]) > 0:
                # 使用最近的performance_window个结果
                recent_perf = self.strategy_performance[i][-self.performance_window:]
                avg_performance.append(np.mean(recent_perf))
            else:
                avg_performance.append(0)
        
        return int(np.argmax(avg_performance))
    
    def _weighted_selection(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: Optional[np.ndarray],
        y_labeled: Optional[np.ndarray],
        model: Optional[Any],
        weights: np.ndarray,
        **kwargs
    ) -> SelectionResult:
        """
        加权组合多个策略的选择结果
        """
        # 收集所有策略的选择
        all_scores = np.zeros(len(X_unlabeled))
        strategy_votes = defaultdict(int)
        
        for i, (strategy, weight) in enumerate(zip(self.strategies, weights)):
            if weight <= 0:
                continue
            
            try:
                result = strategy.select(
                    X_unlabeled, X_labeled, y_labeled, model, **kwargs
                )
                
                # 将选择分数加到总分中
                scores = np.zeros(len(X_unlabeled))
                if result.acquisition_values is not None:
                    values = result.acquisition_values
                    # 归一化
                    values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
                    scores[result.selected_indices] = values
                else:
                    scores[result.selected_indices] = result.selected_scores
                
                all_scores += weight * scores
                strategy_votes[strategy.get_name()] += len(result)
                
            except Exception as e:
                logger.warning(f"Strategy {strategy.get_name()} failed: {e}")
                continue
        
        # 选择top-k
        batch_size = min(self.config.batch_size, len(X_unlabeled))
        selected_indices = np.argsort(all_scores)[-batch_size:][::-1]
        
        return SelectionResult(
            selected_indices=selected_indices,
            selected_scores=all_scores[selected_indices],
            acquisition_values=all_scores[selected_indices],
            metadata={
                'weights': weights.tolist(),
                'strategy_votes': dict(strategy_votes),
                'combined_scores': True
            }
        )
    
    def _record_strategy_usage(self, strategy_idx: int, result: SelectionResult):
        """记录策略使用情况"""
        self.strategy_usage_count[strategy_idx] += 1
        
        # 这里可以添加性能评估逻辑
        # 例如：如果result中包含反馈信息，可以更新performance
    
    def update_performance(self, strategy_idx: int, performance: float):
        """
        更新策略性能记录
        
        Args:
            strategy_idx: 策略索引
            performance: 性能分数 (如准确性提升、不确定性减少等)
        """
        self.strategy_performance[strategy_idx].append(performance)
        
        # 根据性能调整权重
        if self.selection_mode == 'adaptive':
            self._adjust_weights()
    
    def _adjust_weights(self):
        """根据性能历史调整权重"""
        if len(self.strategy_performance) < len(self.strategies):
            return
        
        # 计算近期平均性能
        recent_perfs = []
        for i in range(len(self.strategies)):
            if len(self.strategy_performance[i]) >= 2:
                perf = np.mean(self.strategy_performance[i][-self.performance_window:])
                recent_perfs.append(perf)
            else:
                recent_perfs.append(0)
        
        # Softmax权重更新
        exp_perfs = np.exp(np.array(recent_perfs) - np.max(recent_perfs))
        new_weights = exp_perfs / np.sum(exp_perfs)
        
        # 平滑过渡
        self.strategy_weights = 0.7 * self.strategy_weights + 0.3 * new_weights
        self.strategy_weights /= np.sum(self.strategy_weights)
    
    def get_strategy_stats(self) -> Dict:
        """获取策略使用统计"""
        return {
            'usage_count': dict(self.strategy_usage_count),
            'current_weights': self.strategy_weights.tolist(),
            'performance_history': dict(self.strategy_performance)
        }
