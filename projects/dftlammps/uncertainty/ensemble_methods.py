"""
集合方法模块 - Ensemble Methods for Uncertainty Quantification

本模块实现基于集合的不确定性量化方法：
- 深度集合 (Deep Ensembles)
- 快照集成 (Snapshot Ensembles)
- 加权集合 (Weighted Ensembles)
- 多样性度量与聚合

作者: Causal AI Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class EnsemblePrediction:
    """集合预测结果"""
    mean: np.ndarray
    variance: np.ndarray
    member_predictions: np.ndarray  # [n_members, n_samples, output_dim]
    disagreement: np.ndarray  # 模型间分歧
    confidence: np.ndarray  # 置信度
    
    def prediction_interval(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """计算预测区间"""
        from scipy.stats import norm
        alpha = 1 - confidence
        z = norm.ppf(1 - alpha/2)
        std = np.sqrt(self.variance)
        lower = self.mean - z * std
        upper = self.mean + z * std
        return lower, upper
    
    def consensus_score(self) -> float:
        """计算一致性分数"""
        # 基于方差的共识
        return 1.0 / (1.0 + np.mean(self.disagreement))


class DeepEnsemble:
    """
    深度集合
    
    训练多个神经网络并聚合预测以估计不确定性
    """
    
    def __init__(self,
                 model_builder: Callable,
                 n_members: int = 5,
                 bootstrap: bool = True,
                 diversity_weight: float = 0.0):
        """
        初始化深度集合
        
        Args:
            model_builder: 模型构建函数，返回新模型实例
            n_members: 集合成员数
            bootstrap: 是否使用自助采样
            diversity_weight: 多样性损失权重
        """
        self.model_builder = model_builder
        self.n_members = n_members
        self.bootstrap = bootstrap
        self.diversity_weight = diversity_weight
        
        self.members: List[Any] = []
        self.histories: List[Dict] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray,
           epochs: int = 100,
           batch_size: int = 32,
           learning_rate: float = 1e-3,
           verbose: bool = False) -> 'DeepEnsemble':
        """
        训练深度集合
        
        Args:
            X: 训练数据
            y: 目标值
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            verbose: 是否打印进度
            
        Returns:
            self
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DeepEnsemble")
        
        self.members = []
        self.histories = []
        
        n_samples = len(X)
        
        for i in range(self.n_members):
            if verbose:
                print(f"\nTraining ensemble member {i+1}/{self.n_members}")
            
            # 构建新模型
            model = self.model_builder()
            
            # 准备数据
            if self.bootstrap:
                # 自助采样
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_train = X[indices]
                y_train = y[indices]
            else:
                X_train = X
                y_train = y
            
            # 添加噪声以增加多样性
            if self.diversity_weight > 0:
                noise_scale = 0.01 * np.std(X_train)
                X_train = X_train + np.random.normal(0, noise_scale, X_train.shape)
            
            # 训练模型
            history = self._train_member(model, X_train, y_train,
                                        epochs, batch_size, learning_rate,
                                        verbose and i == 0)
            
            self.members.append(model)
            self.histories.append(history)
        
        return self
    
    def _train_member(self, model, X, y, epochs, batch_size, lr, verbose):
        """训练单个成员"""
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y.reshape(-1, 1) if len(y.shape) == 1 else y)
        
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = torch.mean((y_pred - y_batch) ** 2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        集合预测
        
        Args:
            X: 输入数据
            
        Returns:
            集合预测结果
        """
        if not self.members:
            raise ValueError("Ensemble not fitted yet")
        
        X_t = torch.FloatTensor(X)
        
        # 收集所有成员预测
        predictions = []
        for model in self.members:
            model.eval()
            with torch.no_grad():
                pred = model(X_t).numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)  # [n_members, n_samples, output_dim]
        
        # 计算统计量
        mean_pred = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        disagreement = np.std(predictions, axis=0)
        
        # 置信度（基于预测方差的倒数）
        confidence = 1.0 / (1.0 + variance)
        
        return EnsemblePrediction(
            mean=mean_pred,
            variance=variance,
            member_predictions=predictions,
            disagreement=disagreement,
            confidence=confidence
        )
    
    def epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """计算认知不确定性（模型分歧）"""
        pred = self.predict(X)
        return pred.disagreement
    
    def aleatoric_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """估计偶然不确定性（平均预测方差）"""
        # 简化的估计：假设每个模型输出方差
        # 这里使用成员间的方差作为总不确定性的代理
        pred = self.predict(X)
        return pred.variance * 0.5  # 简化假设
    
    def diversity_metrics(self) -> Dict[str, float]:
        """
        计算集合多样性指标
        
        Returns:
            多样性指标字典
        """
        if not self.histories:
            return {}
        
        # 预测相关性多样性
        # 使用最终损失值的差异作为多样性度量
        final_losses = [h['loss'][-1] for h in self.histories]
        
        metrics = {
            'loss_variance': np.var(final_losses),
            'loss_range': np.max(final_losses) - np.min(final_losses),
            'mean_loss': np.mean(final_losses)
        }
        
        return metrics


class SnapshotEnsemble:
    """
    快照集成
    
    使用循环学习率训练单个模型，保存多个快照
    """
    
    def __init__(self,
                 model: nn.Module if HAS_TORCH else object,
                 n_snapshots: int = 5,
                 n_epochs_per_cycle: int = 50):
        """
        初始化快照集成
        
        Args:
            model: 神经网络模型
            n_snapshots: 快照数量
            n_epochs_per_cycle: 每个周期的epoch数
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for SnapshotEnsemble")
        
        self.base_model = model
        self.n_snapshots = n_snapshots
        self.n_epochs_per_cycle = n_epochs_per_cycle
        
        self.snapshots: List[nn.Module] = []
        self.history: Dict = {'loss': []}
        
    def fit(self, X: np.ndarray, y: np.ndarray,
           batch_size: int = 32,
           lr_max: float = 0.1,
           verbose: bool = False) -> 'SnapshotEnsemble':
        """
        训练快照集成
        
        Args:
            X: 训练数据
            y: 目标值
            batch_size: 批次大小
            lr_max: 最大学习率
            verbose: 是否打印进度
            
        Returns:
            self
        """
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y.reshape(-1, 1) if len(y.shape) == 1 else y)
        
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.SGD(self.base_model.parameters(), lr=lr_max, momentum=0.9)
        
        total_epochs = self.n_epochs_per_cycle * self.n_snapshots
        
        self.snapshots = []
        
        for epoch in range(total_epochs):
            # 计算循环学习率
            t = (epoch % self.n_epochs_per_cycle) / self.n_epochs_per_cycle
            lr = lr_max / 2 * (np.cos(np.pi * t) + 1)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # 训练
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = self.base_model(X_batch)
                loss = torch.mean((y_pred - y_batch) ** 2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.history['loss'].append(avg_loss)
            
            # 保存快照
            if (epoch + 1) % self.n_epochs_per_cycle == 0:
                # 保存模型副本
                snapshot = type(self.base_model)(
                    *[getattr(self.base_model, attr) for attr in ['input_dim', 'hidden_dims', 'output_dim'] 
                      if hasattr(self.base_model, attr)]
                ) if hasattr(self.base_model, 'input_dim') else type(self.base_model)()
                
                snapshot.load_state_dict(self.base_model.state_dict())
                self.snapshots.append(snapshot)
                
                if verbose:
                    print(f"Snapshot {len(self.snapshots)}/{self.n_snapshots} saved at epoch {epoch+1}")
        
        return self
    
    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        集合预测
        
        Args:
            X: 输入数据
            
        Returns:
            集合预测结果
        """
        if not self.snapshots:
            raise ValueError("Snapshot ensemble not fitted yet")
        
        X_t = torch.FloatTensor(X)
        
        predictions = []
        for model in self.snapshots:
            model.eval()
            with torch.no_grad():
                pred = model(X_t).numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        disagreement = np.std(predictions, axis=0)
        confidence = 1.0 / (1.0 + variance)
        
        return EnsemblePrediction(
            mean=mean_pred,
            variance=variance,
            member_predictions=predictions,
            disagreement=disagreement,
            confidence=confidence
        )


class WeightedEnsemble:
    """
    加权集合
    
    根据成员性能加权聚合预测
    """
    
    def __init__(self,
                 models: List[Any],
                 meta_learner: str = 'ridge'):
        """
        初始化加权集合
        
        Args:
            models: 基础模型列表
            meta_learner: 元学习器类型 ('ridge', 'average', 'stacking')
        """
        self.models = models
        self.meta_learner_type = meta_learner
        self.meta_learner = None
        self.weights: np.ndarray = None
        
    def fit(self, X: np.ndarray, y: np.ndarray,
           X_val: np.ndarray = None,
           y_val: np.ndarray = None) -> 'WeightedEnsemble':
        """
        拟合加权集合
        
        Args:
            X: 训练数据
            y: 目标值
            X_val: 验证数据（用于元学习器）
            y_val: 验证标签
            
        Returns:
            self
        """
        # 如果没有验证集，使用训练集
        if X_val is None:
            X_val = X
            y_val = y
        
        # 获取每个模型的预测
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X_val)
                predictions.append(pred.flatten())
        
        pred_matrix = np.column_stack(predictions)
        
        if self.meta_learner_type == 'ridge':
            if HAS_SKLEARN:
                self.meta_learner = Ridge(alpha=1.0)
                self.meta_learner.fit(pred_matrix, y_val)
                self.weights = self.meta_learner.coef_
            else:
                # 简化的岭回归
                self.weights = self._ridge_solve(pred_matrix, y_val)
        
        elif self.meta_learner_type == 'average':
            # 简单平均
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        elif self.meta_learner_type == 'stacking':
            # 使用stacking学习权重
            self.weights = self._learn_stacking_weights(pred_matrix, y_val)
        
        # 归一化权重
        self.weights = np.maximum(self.weights, 0)  # 非负权重
        self.weights = self.weights / np.sum(self.weights)
        
        return self
    
    def _ridge_solve(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """简化的岭回归求解"""
        n_features = X.shape[1]
        return np.linalg.solve(X.T @ X + alpha * np.eye(n_features), X.T @ y)
    
    def _learn_stacking_weights(self, predictions: np.ndarray, 
                                y: np.ndarray) -> np.ndarray:
        """学习stacking权重"""
        # 使用负MSE作为权重
        weights = []
        for i in range(predictions.shape[1]):
            mse = np.mean((predictions[:, i] - y) ** 2)
            weights.append(1.0 / (mse + 1e-10))
        
        return np.array(weights)
    
    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        加权预测
        
        Args:
            X: 输入数据
            
        Returns:
            加权预测结果
        """
        # 获取所有预测
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred.flatten())
        
        pred_matrix = np.column_stack(predictions)
        
        # 加权平均
        if self.meta_learner is not None:
            mean_pred = self.meta_learner.predict(pred_matrix)
        else:
            mean_pred = pred_matrix @ self.weights
        
        # 计算加权方差
        weighted_var = np.average(
            (pred_matrix - mean_pred.reshape(-1, 1)) ** 2,
            axis=1,
            weights=self.weights
        )
        
        return EnsemblePrediction(
            mean=mean_pred.reshape(-1, 1),
            variance=weighted_var.reshape(-1, 1),
            member_predictions=pred_matrix.reshape(len(X), len(self.models), 1),
            disagreement=np.std(pred_matrix, axis=1).reshape(-1, 1),
            confidence=1.0 / (1.0 + weighted_var.reshape(-1, 1))
        )
    
    def get_weights(self) -> Dict[str, float]:
        """获取模型权重"""
        return {f"model_{i}": w for i, w in enumerate(self.weights)}


class BootstrapAggregator:
    """
    Bootstrap聚合 (Bagging)
    
    使用自助采样的传统集成方法
    """
    
    def __init__(self,
                 base_model_class: type,
                 n_estimators: int = 10,
                 max_samples: float = 1.0,
                 bootstrap: bool = True):
        """
        初始化Bagging
        
        Args:
            base_model_class: 基础模型类
            n_estimators: 估计器数量
            max_samples: 每个估计器的样本比例
            bootstrap: 是否使用自助采样
        """
        self.base_model_class = base_model_class
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        
        self.estimators: List[Any] = []
        self.indices: List[np.ndarray] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'BootstrapAggregator':
        """
        拟合Bagging
        
        Args:
            X: 训练数据
            y: 目标值
            **fit_params: 传递给基础模型的参数
            
        Returns:
            self
        """
        n_samples = len(X)
        n_subsample = int(n_samples * self.max_samples)
        
        self.estimators = []
        self.indices = []
        
        for i in range(self.n_estimators):
            # 采样
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_subsample, replace=True)
            else:
                indices = np.random.choice(n_samples, n_subsample, replace=False)
            
            self.indices.append(indices)
            
            # 训练基础模型
            X_subset = X[indices]
            y_subset = y[indices]
            
            estimator = self.base_model_class()
            estimator.fit(X_subset, y_subset, **fit_params)
            self.estimators.append(estimator)
        
        return self
    
    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果
        """
        predictions = []
        for estimator in self.estimators:
            pred = estimator.predict(X)
            predictions.append(pred.flatten())
        
        pred_matrix = np.column_stack(predictions)
        
        mean_pred = np.mean(pred_matrix, axis=1)
        variance = np.var(pred_matrix, axis=1)
        
        return EnsemblePrediction(
            mean=mean_pred.reshape(-1, 1),
            variance=variance.reshape(-1, 1),
            member_predictions=pred_matrix.reshape(len(X), len(self.estimators), 1),
            disagreement=np.std(pred_matrix, axis=1).reshape(-1, 1),
            confidence=1.0 / (1.0 + variance.reshape(-1, 1))
        )
    
    def oob_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算袋外预测
        
        Args:
            X: 训练数据
            y: 目标值
            
        Returns:
            袋外预测
        """
        n_samples = len(X)
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)
        
        for estimator, indices in zip(self.estimators, self.indices):
            # 找到袋外样本
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[indices] = False
            
            if np.any(oob_mask):
                X_oob = X[oob_mask]
                pred = estimator.predict(X_oob)
                oob_predictions[oob_mask] += pred.flatten()
                oob_counts[oob_mask] += 1
        
        # 平均
        oob_predictions[oob_counts > 0] /= oob_counts[oob_counts > 0]
        
        return oob_predictions


class EnsembleCalibration:
    """
    集合校准
    
    校准集合预测的概率估计
    """
    
    def __init__(self, method: str = 'temperature'):
        """
        初始化校准器
        
        Args:
            method: 校准方法 ('temperature', 'isotonic', 'platt')
        """
        self.method = method
        self.calibration_model = None
        self.temperature = 1.0
        
    def fit(self, ensemble_pred: EnsemblePrediction, y_true: np.ndarray):
        """
        拟合校准器
        
        Args:
            ensemble_pred: 集合预测
            y_true: 真实值
        """
        if self.method == 'temperature':
            self._fit_temperature_scaling(ensemble_pred, y_true)
        elif self.method == 'isotonic':
            self._fit_isotonic_regression(ensemble_pred, y_true)
        
    def _fit_temperature_scaling(self, ensemble_pred: EnsemblePrediction, 
                                 y_true: np.ndarray):
        """温度缩放"""
        # 优化温度参数
        def nll_loss(T):
            variance = ensemble_pred.variance * T
            nll = 0.5 * np.log(2 * np.pi * variance) + \
                  0.5 * (y_true - ensemble_pred.mean.flatten()) ** 2 / variance
            return np.mean(nll)
        
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
    
    def _fit_isotonic_regression(self, ensemble_pred: EnsemblePrediction,
                                 y_true: np.ndarray):
        """保序回归"""
        if HAS_SKLEARN:
            from sklearn.isotonic import IsotonicRegression
            
            # 使用预测方差作为不确定性度量
            uncertainty = np.sqrt(ensemble_pred.variance.flatten())
            residual = np.abs(y_true - ensemble_pred.mean.flatten())
            
            self.calibration_model = IsotonicRegression(out_of_bounds='clip')
            self.calibration_model.fit(uncertainty, residual)
        
    def calibrate(self, ensemble_pred: EnsemblePrediction) -> EnsemblePrediction:
        """
        校准预测
        
        Args:
            ensemble_pred: 原始预测
            
        Returns:
            校准后的预测
        """
        if self.method == 'temperature':
            calibrated_var = ensemble_pred.variance * self.temperature
            return EnsemblePrediction(
                mean=ensemble_pred.mean,
                variance=calibrated_var,
                member_predictions=ensemble_pred.member_predictions,
                disagreement=ensemble_pred.disagreement,
                confidence=1.0 / (1.0 + calibrated_var)
            )
        
        elif self.method == 'isotonic' and self.calibration_model is not None:
            uncertainty = np.sqrt(ensemble_pred.variance.flatten())
            calibrated_uncertainty = self.calibration_model.predict(uncertainty)
            calibrated_var = calibrated_uncertainty.reshape(ensemble_pred.variance.shape) ** 2
            
            return EnsemblePrediction(
                mean=ensemble_pred.mean,
                variance=calibrated_var,
                member_predictions=ensemble_pred.member_predictions,
                disagreement=ensemble_pred.disagreement,
                confidence=1.0 / (1.0 + calibrated_var)
            )
        
        return ensemble_pred


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("集合方法示例")
    print("=" * 60)
    
    if not HAS_TORCH:
        print("PyTorch not available, skipping examples")
        return
    
    # 生成数据
    np.random.seed(42)
    n_train = 200
    X_train = np.sort(np.random.uniform(-3, 3, n_train)).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + 0.1 * np.random.randn(n_train)
    
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_test = np.sin(X_test).flatten()
    
    print(f"\n训练样本数: {n_train}")
    
    # 定义模型构建函数
    def build_model():
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        return SimpleNet()
    
    # 示例1: 深度集合
    print("\n" + "-" * 40)
    print("1. 深度集合")
    print("-" * 40)
    
    deep_ensemble = DeepEnsemble(
        model_builder=build_model,
        n_members=5,
        bootstrap=True,
        diversity_weight=0.01
    )
    
    deep_ensemble.fit(X_train, y_train, epochs=300, verbose=False)
    pred_ensemble = deep_ensemble.predict(X_test)
    
    mse = np.mean((pred_ensemble.mean.flatten() - y_test) ** 2)
    print(f"测试MSE: {mse:.6f}")
    print(f"平均预测方差: {np.mean(pred_ensemble.variance):.6f}")
    print(f"多样性指标: {deep_ensemble.diversity_metrics()}")
    
    # 示例2: 快照集成
    print("\n" + "-" * 40)
    print("2. 快照集成")
    print("-" * 40)
    
    base_model = build_model()
    snapshot = SnapshotEnsemble(
        base_model,
        n_snapshots=3,
        n_epochs_per_cycle=100
    )
    
    snapshot.fit(X_train, y_train, verbose=False)
    pred_snapshot = snapshot.predict(X_test)
    
    mse_snap = np.mean((pred_snapshot.mean.flatten() - y_test) ** 2)
    print(f"测试MSE: {mse_snap:.6f}")
    print(f"平均预测方差: {np.mean(pred_snapshot.variance):.6f}")
    
    # 示例3: 加权集合
    print("\n" + "-" * 40)
    print("3. 加权集合")
    print("-" * 40)
    
    # 创建几个简单模型
    models = [build_model() for _ in range(3)]
    
    # 简单训练每个模型
    for i, model in enumerate(models):
        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train.reshape(-1, 1))
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(200):
            optimizer.zero_grad()
            y_pred = model(X_t)
            loss = torch.mean((y_pred - y_t) ** 2)
            loss.backward()
            optimizer.step()
    
    weighted = WeightedEnsemble(models, meta_learner='ridge')
    weighted.fit(X_train, y_train)
    pred_weighted = weighted.predict(X_test)
    
    mse_weighted = np.mean((pred_weighted.mean.flatten() - y_test) ** 2)
    print(f"测试MSE: {mse_weighted:.6f}")
    print(f"模型权重: {weighted.get_weights()}")
    
    # 示例4: Bootstrap聚合
    print("\n" + "-" * 40)
    print("4. Bootstrap聚合")
    print("-" * 40)
    
    if HAS_SKLEARN:
        bagging = BootstrapAggregator(
            base_model_class=RandomForestRegressor,
            n_estimators=5,
            max_samples=0.8
        )
        bagging.fit(X_train, y_train, n_estimators=10)
        pred_bagging = bagging.predict(X_test)
        
        mse_bag = np.mean((pred_bagging.mean.flatten() - y_test) ** 2)
        print(f"测试MSE: {mse_bag:.6f}")
        print(f"平均预测方差: {np.mean(pred_bagging.variance):.6f}")
    else:
        print("sklearn not available for bagging example")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
