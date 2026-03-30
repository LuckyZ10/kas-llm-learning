#!/usr/bin/env python3
"""
不确定性量化模块 - Uncertainty Quantification

实现多种先进的不确定性量化方法：
1. 集成不确定性 (Ensemble Uncertainty) - Query by Committee
2. MC Dropout不确定性 (Monte Carlo Dropout)
3. 证据学习不确定性 (Evidential Learning)
4. 贝叶斯高斯过程不确定性 (Bayesian GP)

参考:
- Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles", NeurIPS 2017
- Gal & Ghahramani "Dropout as a Bayesian Approximation", ICML 2016
- Amini et al. "Deep Evidential Regression", NeurIPS 2020
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """不确定性量化结果"""
    total_uncertainty: np.ndarray  # 总不确定性
    epistemic_uncertainty: np.ndarray  # 认知不确定性 (模型相关)
    aleatoric_uncertainty: np.ndarray  # 偶然不确定性 (数据相关)
    predictions: np.ndarray  # 预测值
    confidence: np.ndarray  # 置信度 (1 - uncertainty)
    
    def get_top_uncertain_indices(self, n: int = 10) -> np.ndarray:
        """获取不确定性最高的n个样本索引"""
        return np.argsort(self.total_uncertainty)[-n:][::-1]
    
    def get_uncertainty_stats(self) -> Dict[str, float]:
        """获取不确定性统计信息"""
        return {
            'mean_total': float(np.mean(self.total_uncertainty)),
            'std_total': float(np.std(self.total_uncertainty)),
            'max_total': float(np.max(self.total_uncertainty)),
            'mean_epistemic': float(np.mean(self.epistemic_uncertainty)),
            'mean_aleatoric': float(np.mean(self.aleatoric_uncertainty)),
        }


class UncertaintyQuantifier(ABC):
    """不确定性量化器基类"""
    
    @abstractmethod
    def quantify(self, X: np.ndarray, **kwargs) -> UncertaintyResult:
        """
        量化输入数据的不确定性
        
        Args:
            X: 输入特征矩阵 (n_samples, n_features)
            
        Returns:
            UncertaintyResult: 不确定性结果
        """
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs):
        """拟合模型"""
        pass


class EnsembleUncertainty(UncertaintyQuantifier):
    """
    集成不确定性量化器 (Query by Committee)
    
    训练多个模型，使用模型间的分歧作为不确定性度量。
    这是主动学习中最常用的不确定性量化方法之一。
    
    Attributes:
        n_models: 集成模型数量
        models: 模型列表
        disagreement_method: 分歧计算方法 ('std', 'var', 'entropy', 'mutual_info')
    """
    
    def __init__(
        self,
        n_models: int = 5,
        model_factory: Optional[Callable] = None,
        disagreement_method: str = 'std',
        random_state: Optional[int] = None
    ):
        self.n_models = n_models
        self.model_factory = model_factory
        self.disagreement_method = disagreement_method
        self.random_state = random_state
        self.models = []
        self.is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        训练模型集成
        
        Args:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,) 或 (n_samples, n_outputs)
        """
        logger.info(f"Training ensemble with {self.n_models} models...")
        self.models = []
        
        for i in range(self.n_models):
            logger.debug(f"Training model {i+1}/{self.n_models}")
            
            # 使用Bootstrap采样创建不同的训练集
            n_samples = len(X)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # 创建并训练模型
            if self.model_factory is not None:
                model = self.model_factory(seed=i)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=i)
            
            model.fit(X_boot, y_boot)
            self.models.append(model)
        
        self.is_fitted = True
        logger.info("Ensemble training completed")
        return self
    
    def quantify(self, X: np.ndarray, **kwargs) -> UncertaintyResult:
        """
        计算集成不确定性
        
        Args:
            X: 输入特征
            
        Returns:
            UncertaintyResult
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # 收集所有模型的预测
        predictions = np.array([model.predict(X) for model in self.models])
        # shape: (n_models, n_samples) or (n_models, n_samples, n_outputs)
        
        # 计算预测均值
        pred_mean = np.mean(predictions, axis=0)
        
        # 根据分歧方法计算不确定性
        if self.disagreement_method == 'std':
            uncertainty = np.std(predictions, axis=0)
            if len(uncertainty.shape) > 1:
                uncertainty = np.mean(uncertainty, axis=-1)
        elif self.disagreement_method == 'var':
            uncertainty = np.var(predictions, axis=0)
            if len(uncertainty.shape) > 1:
                uncertainty = np.mean(uncertainty, axis=-1)
        elif self.disagreement_method == 'entropy':
            # 对于分类任务使用熵
            pred_probs = np.mean(predictions, axis=0)
            uncertainty = -np.sum(pred_probs * np.log(pred_probs + 1e-10), axis=-1)
        else:
            uncertainty = np.std(predictions, axis=0)
            if len(uncertainty.shape) > 1:
                uncertainty = np.mean(uncertainty, axis=-1)
        
        # 认知不确定性 = 模型分歧
        epistemic = uncertainty
        # 偶然不确定性 = 假设为0 (集成方法不区分两种不确定性)
        aleatoric = np.zeros_like(uncertainty)
        
        return UncertaintyResult(
            total_uncertainty=uncertainty,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            predictions=pred_mean,
            confidence=1.0 / (1.0 + uncertainty)
        )
    
    def compute_committee_disagreement(self, X: np.ndarray) -> np.ndarray:
        """
        计算委员会分歧 (用于主动学习选择)
        
        Returns:
            disagreement scores: (n_samples,)
        """
        result = self.quantify(X)
        return result.epistemic_uncertainty


class MCDropoutUncertainty(UncertaintyQuantifier):
    """
    MC Dropout不确定性量化器
    
    利用dropout在推理时模拟贝叶斯推断，通过多次前向传播
    估计预测不确定性。计算效率高，无需训练多个模型。
    
    Reference: Gal & Ghahramani "Dropout as a Bayesian Approximation", ICML 2016
    
    Attributes:
        n_forward_passes: 前向传播次数
        dropout_rate: Dropout比率
    """
    
    def __init__(
        self,
        base_model: Any = None,
        n_forward_passes: int = 50,
        dropout_rate: float = 0.1,
        random_state: Optional[int] = None
    ):
        self.base_model = base_model
        self.n_forward_passes = n_forward_passes
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """训练基础模型"""
        logger.info("Training base model for MC Dropout...")
        
        if self.base_model is None:
            # 使用简单的神经网络
            try:
                import torch
                self.base_model = self._create_pytorch_model(X.shape[1], y.shape[1] if len(y.shape) > 1 else 1)
                self._train_pytorch(X, y)
            except ImportError:
                from sklearn.neural_network import MLPRegressor
                self.base_model = MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation='relu',
                    solver='adam',
                    max_iter=1000,
                    random_state=self.random_state
                )
                self.base_model.fit(X, y)
        else:
            self.base_model.fit(X, y)
        
        self.is_fitted = True
        logger.info("Base model training completed")
        return self
    
    def _create_pytorch_model(self, input_dim: int, output_dim: int):
        """创建PyTorch模型"""
        import torch
        import torch.nn as nn
        
        class MCDropoutNet(nn.Module):
            def __init__(self, input_dim, output_dim, dropout_rate):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, output_dim)
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                return self.fc3(x)
        
        return MCDropoutNet(input_dim, output_dim, self.dropout_rate)
    
    def _train_pytorch(self, X: np.ndarray, y: np.ndarray):
        """训练PyTorch模型"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1) if len(y.shape) == 1 else y)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.base_model.parameters(), lr=0.001)
        
        self.base_model.train()
        for epoch in range(500):
            optimizer.zero_grad()
            outputs = self.base_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
    
    def quantify(self, X: np.ndarray, **kwargs) -> UncertaintyResult:
        """
        使用MC Dropout估计不确定性
        
        Args:
            X: 输入特征
            
        Returns:
            UncertaintyResult
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # 多次前向传播，保持dropout开启
        predictions = []
        
        try:
            import torch
            self.base_model.train()  # 保持train模式以保持dropout
            X_tensor = torch.FloatTensor(X)
            
            with torch.no_grad():
                for _ in range(self.n_forward_passes):
                    pred = self.base_model(X_tensor).numpy()
                    predictions.append(pred)
        except:
            # 使用sklearn模型时模拟dropout
            predictions = []
            for _ in range(self.n_forward_passes):
                # 添加噪声模拟dropout
                X_noisy = X * (np.random.random(X.shape) > self.dropout_rate)
                pred = self.base_model.predict(X_noisy)
                predictions.append(pred)
        
        predictions = np.array(predictions)  # (n_passes, n_samples, n_outputs)
        
        # 预测均值
        pred_mean = np.mean(predictions, axis=0)
        
        # 预测方差 = 认知不确定性 + 偶然不确定性
        pred_variance = np.var(predictions, axis=0)
        if len(pred_variance.shape) > 1:
            pred_variance = np.mean(pred_variance, axis=-1)
        
        # 估计偶然不确定性 (噪声方差的均值)
        aleatoric = np.mean(np.mean((predictions - pred_mean) ** 2, axis=0), axis=-1) if len(predictions.shape) > 2 else np.zeros(len(X))
        
        # 认知不确定性 = 总方差 - 偶然不确定性
        epistemic = pred_variance - aleatoric
        epistemic = np.maximum(epistemic, 0)  # 确保非负
        
        return UncertaintyResult(
            total_uncertainty=pred_variance,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            predictions=pred_mean.squeeze(),
            confidence=1.0 / (1.0 + pred_variance)
        )


class EvidentialUncertainty(UncertaintyQuantifier):
    """
    证据学习不确定性量化器 (Evidential Deep Learning)
    
    使用主观逻辑理论直接预测分布的参数，无需多次采样。
    能够区分认知不确定性和偶然不确定性，且单次前向传播即可。
    
    Reference: Amini et al. "Deep Evidential Regression", NeurIPS 2020
    
    Attributes:
        n_evidence: 证据数量
        lambda_reg: 正则化参数
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        n_evidence: int = 4,
        lambda_reg: float = 0.001,
        n_epochs: int = 500,
        learning_rate: float = 0.001,
        random_state: Optional[int] = None
    ):
        self.input_dim = input_dim
        self.n_evidence = n_evidence
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        训练证据学习模型
        
        预测正态-逆伽马分布的参数 (gamma, nu, alpha, beta)
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch is required for EvidentialUncertainty")
        
        logger.info("Training Evidential Neural Network...")
        
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # 创建证据学习网络
        self.model = self._create_evidential_model()
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1) if len(y.shape) == 1 else y)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            
            gamma, nu, alpha, beta = self.model(X_tensor)
            
            # NIG负对数似然损失
            loss = self._nig_nll_loss(y_tensor, gamma, nu, alpha, beta)
            # 添加正则化
            loss += self.lambda_reg * self._nig_reg_loss(y_tensor, gamma, nu, alpha, beta)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                logger.debug(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss.item():.4f}")
        
        self.is_fitted = True
        logger.info("Evidential model training completed")
        return self
    
    def _create_evidential_model(self):
        """创建证据学习神经网络"""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        class EvidentialNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                
                # 输出NIG分布的4个参数
                self.fc_gamma = nn.Linear(64, 1)  # 均值
                self.fc_nu = nn.Linear(64, 1)     # 自由度
                self.fc_alpha = nn.Linear(64, 1)  # 形状参数
                self.fc_beta = nn.Linear(64, 1)   # 尺度参数
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                
                gamma = self.fc_gamma(x)
                nu = F.softplus(self.fc_nu(x) + 1)  # > 0
                alpha = F.softplus(self.fc_alpha(x) + 1) + 1  # > 1
                beta = F.softplus(self.fc_beta(x) + 1e-10)  # > 0
                
                return gamma, nu, alpha, beta
        
        return EvidentialNet(self.input_dim)
    
    def _nig_nll_loss(self, y, gamma, nu, alpha, beta):
        """NIG负对数似然损失"""
        import torch
        
        # 计算误差
        error = y - gamma
        
        # NLL
        nll = (torch.log(torch.sqrt(torch.pi / nu)) 
               - alpha * torch.log(2 * beta)
               + (alpha + 0.5) * torch.log(nu * error**2 + 2 * beta)
               + torch.lgamma(alpha)
               - torch.lgamma(alpha + 0.5))
        
        return nll.mean()
    
    def _nig_reg_loss(self, y, gamma, nu, alpha, beta):
        """NIG正则化损失"""
        import torch
        
        error = torch.abs(y - gamma)
        reg = error * (2 * nu + alpha)
        return reg.mean()
    
    def quantify(self, X: np.ndarray, **kwargs) -> UncertaintyResult:
        """
        计算证据学习不确定性
        
        Returns:
            UncertaintyResult with epistemic and aleatoric uncertainty
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        import torch
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(X_tensor)
        
        gamma = gamma.numpy().squeeze()
        nu = nu.numpy().squeeze()
        alpha = alpha.numpy().squeeze()
        beta = beta.numpy().squeeze()
        
        # 认知不确定性 (epistemic) = variance / (nu - 2) for nu > 2
        epistemic = beta / (nu * (alpha - 1))
        
        # 偶然不确定性 (aleatoric) = beta / (alpha - 1)
        aleatoric = beta / (alpha - 1)
        
        # 总不确定性
        total = epistemic + aleatoric
        
        return UncertaintyResult(
            total_uncertainty=total,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            predictions=gamma,
            confidence=nu / (nu + 1)  # 证据支持度
        )


class BayesianGPUncertainty(UncertaintyQuantifier):
    """
    贝叶斯高斯过程不确定性量化器
    
    使用高斯过程作为概率模型，天然提供不确定性估计。
    适用于小规模数据集，计算复杂度为O(N^3)。
    
    对于大规模数据，可以使用稀疏GP或变分GP。
    """
    
    def __init__(
        self,
        kernel: Optional[Any] = None,
        alpha: float = 1e-10,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 10,
        random_state: Optional[int] = None
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.gp = None
        self.is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """训练高斯过程模型"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
        
        logger.info("Training Gaussian Process model...")
        
        if self.kernel is None:
            # 默认核函数: RBF + 白噪声
            self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
                noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)
            )
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state
        )
        
        self.gp.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"GP training completed. Kernel: {self.gp.kernel_}")
        return self
    
    def quantify(self, X: np.ndarray, return_cov: bool = False, **kwargs) -> UncertaintyResult:
        """
        计算GP不确定性
        
        GP天然提供预测均值和方差。
        认知不确定性 = 预测方差 (远离训练数据时增大)
        偶然不确定性 = 噪声方差 (来自WhiteKernel)
        
        Args:
            X: 输入特征
            return_cov: 是否返回协方差矩阵
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if return_cov:
            pred_mean, pred_cov = self.gp.predict(X, return_cov=True)
            pred_std = np.sqrt(np.diag(pred_cov))
        else:
            pred_mean, pred_std = self.gp.predict(X, return_std=True)
        
        # 总不确定性
        total_variance = pred_std ** 2
        
        # 从核函数中提取噪声方差作为偶然不确定性
        if hasattr(self.gp.kernel_, 'k2'):
            noise_kernel = self.gp.kernel_.k2
            if hasattr(noise_kernel, 'noise_level'):
                aleatoric = np.full(len(X), noise_kernel.noise_level)
            else:
                aleatoric = np.zeros(len(X))
        else:
            aleatoric = np.zeros(len(X))
        
        # 认知不确定性 = 总方差 - 偶然不确定性
        epistemic = total_variance - aleatoric
        epistemic = np.maximum(epistemic, 0)
        
        return UncertaintyResult(
            total_uncertainty=total_variance,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            predictions=pred_mean if len(pred_mean.shape) == 1 else pred_mean.squeeze(),
            confidence=1.0 / (1.0 + total_variance)
        )
    
    def optimize_acquisition(
        self,
        acquisition_fn: Callable,
        bounds: np.ndarray,
        n_candidates: int = 1000,
        n_restarts: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        优化采集函数 (用于贝叶斯优化)
        
        Args:
            acquisition_fn: 采集函数
            bounds: 参数边界 (n_dims, 2)
            n_candidates: 随机采样候选点数量
            n_restarts: 局部优化重启次数
            
        Returns:
            (best_x, best_value)
        """
        from scipy.optimize import minimize
        
        n_dims = bounds.shape[0]
        
        # 随机采样候选点
        random_candidates = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(n_candidates, n_dims)
        )
        
        # 评估采集函数
        acq_values = acquisition_fn(random_candidates, self)
        
        # 选择最好的n_restarts个点进行局部优化
        best_indices = np.argsort(acq_values)[-n_restarts:]
        
        best_x = None
        best_value = -np.inf
        
        for idx in best_indices:
            x0 = random_candidates[idx]
            result = minimize(
                lambda x: -acquisition_fn(x.reshape(1, -1), self),
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if -result.fun > best_value:
                best_value = -result.fun
                best_x = result.x
        
        return best_x, best_value


def compare_uncertainty_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    methods: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    比较不同不确定性量化方法的性能
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        methods: 要比较的方法列表
        
    Returns:
        各方法的评估结果
    """
    if methods is None:
        methods = ['ensemble', 'mc_dropout', 'evidential', 'bayesian_gp']
    
    results = {}
    
    for method in methods:
        logger.info(f"Evaluating {method}...")
        
        try:
            if method == 'ensemble':
                model = EnsembleUncertainty(n_models=5)
            elif method == 'mc_dropout':
                model = MCDropoutUncertainty(n_forward_passes=50)
            elif method == 'evidential':
                model = EvidentialUncertainty(n_epochs=300)
            elif method == 'bayesian_gp':
                model = BayesianGPUncertainty()
            else:
                continue
            
            model.fit(X_train, y_train)
            uq_result = model.quantify(X_test)
            
            # 计算预测误差
            mse = np.mean((uq_result.predictions - y_test) ** 2)
            mae = np.mean(np.abs(uq_result.predictions - y_test))
            
            # 计算不确定性校准
            coverage = np.mean(
                (y_test >= uq_result.predictions - 2 * np.sqrt(uq_result.total_uncertainty)) &
                (y_test <= uq_result.predictions + 2 * np.sqrt(uq_result.total_uncertainty))
            )
            
            results[method] = {
                'mse': float(mse),
                'mae': float(mae),
                'uncertainty_stats': uq_result.get_uncertainty_stats(),
                'coverage_2sigma': float(coverage),
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {method}: {e}")
            results[method] = {'error': str(e)}
    
    return results
