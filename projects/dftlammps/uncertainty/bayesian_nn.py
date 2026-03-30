"""
贝叶斯神经网络模块 - Bayesian Neural Networks

本模块实现不确定性量化的贝叶斯方法：
- 变分推断贝叶斯神经网络
- 蒙特卡洛Dropout
- 贝叶斯优化

作者: Causal AI Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class UncertaintyEstimate:
    """不确定性估计结果"""
    mean: np.ndarray
    variance: np.ndarray
    epistemic: np.ndarray = field(default_factory=lambda: np.array([]))
    aleatoric: np.ndarray = field(default_factory=lambda: np.array([]))
    credible_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    
    def prediction_interval(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """获取预测区间"""
        alpha = 1 - confidence
        z = norm.ppf(1 - alpha/2)
        std = np.sqrt(self.variance)
        lower = self.mean - z * std
        upper = self.mean + z * std
        return lower, upper


class BayesianLinear(nn.Module if HAS_TORCH else object):
    """
    贝叶斯线性层
    
    使用变分推断近似后验分布
    """
    
    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 1.0):
        super().__init__() if HAS_TORCH else None
        if not HAS_TORCH:
            return
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # 权重后验的变分参数 (mu, rho)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        
        # 偏置后验的变分参数
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))
        
    @property
    def weight_sigma(self):
        """权重标准差"""
        return torch.log1p(torch.exp(self.weight_rho))
    
    @property
    def bias_sigma(self):
        """偏置标准差"""
        return torch.log1p(torch.exp(self.bias_rho))
    
    def forward(self, x, sample: bool = True):
        """前向传播"""
        if sample:
            # 从后验采样
            weight_epsilon = torch.randn_like(self.weight_sigma)
            bias_epsilon = torch.randn_like(self.bias_sigma)
            
            weight = self.weight_mu + self.weight_sigma * weight_epsilon
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
        else:
            # 使用均值（预测模式）
            weight = self.weight_mu
            bias = self.bias_mu
        
        return torch.nn.functional.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """计算KL散度（后验vs先验）"""
        # KL(q(w)||p(w)) 对于高斯分布
        kl_weight = self._kl_gaussian(
            self.weight_mu, self.weight_sigma,
            torch.zeros_like(self.weight_mu),
            torch.ones_like(self.weight_sigma) * self.prior_sigma
        )
        
        kl_bias = self._kl_gaussian(
            self.bias_mu, self.bias_sigma,
            torch.zeros_like(self.bias_mu),
            torch.ones_like(self.bias_sigma) * self.prior_sigma
        )
        
        return kl_weight + kl_bias
    
    def _kl_gaussian(self, mu_q, sigma_q, mu_p, sigma_p):
        """两个高斯分布之间的KL散度"""
        return torch.sum(
            torch.log(sigma_p / sigma_q) +
            (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) -
            0.5
        )


class BayesianNeuralNetwork(nn.Module if HAS_TORCH else object):
    """
    贝叶斯神经网络
    
    使用变分推断进行不确定性估计
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 prior_sigma: float = 1.0,
                 activation: str = 'tanh'):
        """
        初始化BNN
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度
            prior_sigma: 先验标准差
            activation: 激活函数
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for BayesianNeuralNetwork")
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建贝叶斯网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim, prior_sigma))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(BayesianLinear(prev_dim, output_dim, prior_sigma))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, sample: bool = True):
        """前向传播"""
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                x = layer(x, sample=sample)
            else:
                x = layer(x)
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """计算总KL散度"""
        kl = 0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        return kl
    
    def predict_with_uncertainty(self, X: np.ndarray,
                                  n_samples: int = 100) -> UncertaintyEstimate:
        """
        带不确定性的预测
        
        Args:
            X: 输入数据
            n_samples: 蒙特卡洛采样数
            
        Returns:
            不确定性估计
        """
        self.eval()
        
        X_t = torch.FloatTensor(X)
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(X_t, sample=True)
                predictions.append(pred.numpy())
        
        predictions = np.array(predictions)  # [n_samples, n_data, output_dim]
        
        # 计算统计量
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        
        # 分解不确定性
        # 认知不确定性 (模型不确定性)
        epistemic = np.var(predictions, axis=0)
        
        # 偶然不确定性 (数据噪声) - 简化估计
        aleatoric = np.mean(predictions**2, axis=0) - mean**2 - epistemic
        aleatoric = np.maximum(aleatoric, 0)
        
        # 可信区间
        credible_intervals = {}
        for conf in [0.9, 0.95, 0.99]:
            alpha = 1 - conf
            lower = np.percentile(predictions, 100*alpha/2, axis=0)
            upper = np.percentile(predictions, 100*(1-alpha/2), axis=0)
            credible_intervals[f"{int(conf*100)}%"] = (lower, upper)
        
        return UncertaintyEstimate(
            mean=mean,
            variance=variance,
            epistemic=epistemic,
            aleatoric=aleatoric,
            credible_intervals=credible_intervals
        )


class BNNTrainer:
    """
    贝叶斯神经网络训练器
    
    使用变分推断训练BNN
    """
    
    def __init__(self, model: BayesianNeuralNetwork,
                 learning_rate: float = 1e-3,
                 beta: float = 1.0):
        """
        初始化训练器
        
        Args:
            model: BNN模型
            learning_rate: 学习率
            beta: KL项权重
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.beta = beta
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.history = {'loss': [], 'kl': [], 'nll': []}
        
    def train(self, X: np.ndarray, y: np.ndarray,
             epochs: int = 1000,
             batch_size: int = 32,
             verbose: bool = True) -> Dict:
        """
        训练BNN
        
        Args:
            X: 训练数据
            y: 目标值
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印进度
            
        Returns:
            训练历史
        """
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y.reshape(-1, 1) if len(y.shape) == 1 else y)
        
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        n_batches = len(dataloader)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_nll = 0.0
            
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                
                # 前向传播（采样）
                y_pred = self.model(X_batch, sample=True)
                
                # 负对数似然
                nll = torch.mean((y_pred - y_batch) ** 2)
                
                # KL散度
                kl = self.model.kl_divergence() / n_batches
                
                # 总损失 (ELBO = -E[log p(y|w)] + KL(q(w)||p(w)))
                loss = nll + self.beta * kl
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_kl += kl.item()
                epoch_nll += nll.item()
            
            # 记录历史
            self.history['loss'].append(epoch_loss / len(dataloader))
            self.history['kl'].append(epoch_kl)
            self.history['nll'].append(epoch_nll / len(dataloader))
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss = {self.history['loss'][-1]:.4f}, "
                      f"KL = {self.history['kl'][-1]:.4f}, "
                      f"NLL = {self.history['nll'][-1]:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray, n_samples: int = 100) -> UncertaintyEstimate:
        """预测"""
        return self.model.predict_with_uncertainty(X, n_samples)


class MCDropoutNetwork(nn.Module if HAS_TORCH else object):
    """
    蒙特卡洛Dropout网络
    
    使用Dropout进行近似贝叶斯推断
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        初始化MC Dropout网络
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度
            dropout_rate: Dropout比率
            activation: 激活函数
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MCDropoutNetwork")
        
        super().__init__()
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        """前向传播"""
        return self.network(x)
    
    def predict_with_uncertainty(self, X: np.ndarray,
                                  n_samples: int = 100) -> UncertaintyEstimate:
        """
        带不确定性的预测
        
        即使在eval模式下也保持dropout开启
        """
        self.train()  # 保持dropout开启
        
        X_t = torch.FloatTensor(X)
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(X_t)
                predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        epistemic = variance
        
        # 可信区间
        credible_intervals = {}
        for conf in [0.9, 0.95]:
            alpha = 1 - conf
            lower = np.percentile(predictions, 100*alpha/2, axis=0)
            upper = np.percentile(predictions, 100*(1-alpha/2), axis=0)
            credible_intervals[f"{int(conf*100)}%"] = (lower, upper)
        
        return UncertaintyEstimate(
            mean=mean,
            variance=variance,
            epistemic=epistemic,
            credible_intervals=credible_intervals
        )


class GaussianProcessApproximation:
    """
    高斯过程近似
    
    使用随机特征近似高斯过程
    """
    
    def __init__(self, n_features: int = 100,
                 kernel: str = 'rbf',
                 length_scale: float = 1.0,
                 noise_level: float = 0.1):
        """
        初始化GP近似
        
        Args:
            n_features: 随机特征数
            kernel: 核函数类型
            length_scale: 长度尺度
            noise_level: 噪声水平
        """
        self.n_features = n_features
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise_level = noise_level
        
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        
    def _sample_random_features(self, input_dim: int):
        """采样随机特征"""
        if self.kernel == 'rbf':
            # RFF近似RBF核
            self.random_weights = np.random.normal(
                0, 1/self.length_scale, (input_dim, self.n_features)
            )
            self.random_bias = np.random.uniform(0, 2*np.pi, self.n_features)
        
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """应用随机特征变换"""
        features = X @ self.random_weights + self.random_bias
        return np.sqrt(2.0 / self.n_features) * np.cos(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合GP近似
        
        Args:
            X: 训练数据
            y: 目标值
        """
        n_samples, n_features = X.shape
        
        # 采样随机特征
        self._sample_random_features(n_features)
        
        # 特征变换
        phi = self._transform(X)
        
        # 贝叶斯线性回归
        # 后验: beta ~ N(mu, Sigma)
        # Sigma = (phi^T phi / noise + I)^{-1}
        # mu = Sigma phi^T y / noise
        
        Sigma_inv = (phi.T @ phi) / self.noise_level + np.eye(self.n_features)
        self.Sigma = np.linalg.inv(Sigma_inv)
        self.beta = (self.Sigma @ phi.T @ y.flatten()) / self.noise_level
        
        self.X_train = X
        self.y_train = y
        
        return self
    
    def predict(self, X: np.ndarray) -> UncertaintyEstimate:
        """
        带不确定性的预测
        
        Args:
            X: 输入数据
            
        Returns:
            不确定性估计
        """
        phi = self._transform(X)
        
        # 预测均值
        mean = phi @ self.beta
        
        # 预测方差
        # Var[y*] = noise + phi*^T Sigma phi*
        var_noise = self.noise_level * np.ones(len(X))
        var_epistemic = np.sum(phi @ self.Sigma * phi, axis=1)
        variance = var_noise + var_epistemic
        
        # 可信区间
        credible_intervals = {}
        for conf in [0.9, 0.95]:
            z = norm.ppf((1 + conf) / 2)
            lower = mean - z * np.sqrt(variance)
            upper = mean + z * np.sqrt(variance)
            credible_intervals[f"{int(conf*100)}%"] = (lower, upper)
        
        return UncertaintyEstimate(
            mean=mean,
            variance=variance,
            epistemic=var_epistemic,
            aleatoric=var_noise,
            credible_intervals=credible_intervals
        )


class BayesianOptimization:
    """
    贝叶斯优化
    
    使用高斯过程进行高效的全局优化
    """
    
    def __init__(self,
                 objective: Callable,
                 bounds: List[Tuple[float, float]],
                 acquisition: str = 'ei',
                 xi: float = 0.01,
                 n_initial: int = 5):
        """
        初始化贝叶斯优化
        
        Args:
            objective: 目标函数
            bounds: 参数边界 [(min, max), ...]
            acquisition: 采集函数 ('ei', 'ucb', 'pi')
            xi: 探索参数
            n_initial: 初始采样点数
        """
        self.objective = objective
        self.bounds = np.array(bounds)
        self.acquisition = acquisition
        self.xi = xi
        self.n_initial = n_initial
        
        self.X_observed = []
        self.y_observed = []
        self.gp = None
        
    def _initialize(self):
        """初始化观测点"""
        # 拉丁超立方采样或随机采样
        X_init = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1],
            size=(self.n_initial, len(self.bounds))
        )
        
        for x in X_init:
            y = self.objective(x)
            self.X_observed.append(x)
            self.y_observed.append(y)
    
    def _fit_gp(self):
        """拟合高斯过程"""
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # 使用简单的GP近似
        self.gp = GaussianProcessApproximation(
            n_features=min(100, len(y) * 2),
            length_scale=1.0,
            noise_level=0.01
        )
        self.gp.fit(X, y)
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """计算采集函数值"""
        if self.gp is None:
            return np.zeros(len(X))
        
        pred = self.gp.predict(X)
        mean = pred.mean
        std = np.sqrt(pred.variance)
        
        y_best = np.min(self.y_observed)  # 假设是最小化问题
        
        if self.acquisition == 'ei':  # 期望改进
            with np.errstate(divide='warn'):
                improvement = y_best - mean - self.xi
                Z = improvement / (std + 1e-10)
                ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
                ei[std < 1e-10] = 0
            return -ei  # 最小化问题
        
        elif self.acquisition == 'ucb':  # 上置信界
            kappa = 2.0
            return mean - kappa * std
        
        elif self.acquisition == 'pi':  # 改进概率
            improvement = y_best - mean - self.xi
            return -norm.cdf(improvement / (std + 1e-10))
        
        return mean
    
    def _propose_next(self) -> np.ndarray:
        """提议下一个采样点"""
        # 优化采集函数
        def neg_acquisition(x):
            return self._acquisition_function(x.reshape(1, -1))[0]
        
        # 多起点优化
        best_x = None
        best_acq = np.inf
        
        for _ in range(10):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            
            if HAS_SCIPY:
                result = minimize(
                    neg_acquisition, x0,
                    bounds=self.bounds,
                    method='L-BFGS-B'
                )
                if result.fun < best_acq:
                    best_acq = result.fun
                    best_x = result.x
            else:
                # 简单的梯度下降
                x = x0.copy()
                lr = 0.01
                for _ in range(100):
                    # 数值梯度
                    grad = np.zeros_like(x)
                    for i in range(len(x)):
                        x_plus = x.copy()
                        x_plus[i] += 1e-5
                        grad[i] = (neg_acquisition(x_plus) - neg_acquisition(x)) / 1e-5
                    x -= lr * grad
                    x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
                
                acq = neg_acquisition(x)
                if acq < best_acq:
                    best_acq = acq
                    best_x = x
        
        return best_x
    
    def optimize(self, n_iterations: int = 20,
                verbose: bool = True) -> Dict:
        """
        执行贝叶斯优化
        
        Args:
            n_iterations: 优化迭代次数
            verbose: 是否打印进度
            
        Returns:
            优化结果
        """
        # 初始化
        if len(self.X_observed) == 0:
            self._initialize()
        
        for i in range(n_iterations):
            # 拟合GP
            self._fit_gp()
            
            # 提议下一个点
            x_next = self._propose_next()
            
            # 评估
            y_next = self.objective(x_next)
            
            # 记录
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)
            
            if verbose:
                print(f"Iteration {i+1}/{n_iterations}: "
                      f"y = {y_next:.4f}, x = {x_next}")
        
        # 找到最佳点
        best_idx = np.argmin(self.y_observed)
        
        return {
            'x_opt': self.X_observed[best_idx],
            'y_opt': self.y_observed[best_idx],
            'X_history': np.array(self.X_observed),
            'y_history': np.array(self.y_observed)
        }


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("贝叶斯神经网络示例")
    print("=" * 60)
    
    if not HAS_TORCH:
        print("PyTorch not available, skipping examples")
        return
    
    # 生成数据
    np.random.seed(42)
    n_train = 100
    X_train = np.sort(np.random.uniform(-3, 3, n_train)).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + 0.1 * np.random.randn(n_train)
    
    X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_test = np.sin(X_test).flatten()
    
    print(f"\n训练样本数: {n_train}")
    
    # 示例1: 变分推断BNN
    print("\n" + "-" * 40)
    print("1. 变分推断贝叶斯神经网络")
    print("-" * 40)
    
    bnn = BayesianNeuralNetwork(
        input_dim=1,
        output_dim=1,
        hidden_dims=[32, 32],
        prior_sigma=1.0
    )
    
    trainer = BNNTrainer(bnn, learning_rate=1e-3, beta=0.1)
    trainer.train(X_train, y_train, epochs=500, verbose=False)
    
    # 预测
    pred = bnn.predict_with_uncertainty(X_test, n_samples=100)
    
    # 评估
    mse = np.mean((pred.mean.flatten() - y_test) ** 2)
    print(f"测试MSE: {mse:.6f}")
    print(f"平均不确定性: {np.mean(np.sqrt(pred.variance)):.4f}")
    
    # 示例2: MC Dropout
    print("\n" + "-" * 40)
    print("2. 蒙特卡洛Dropout")
    print("-" * 40)
    
    mc_model = MCDropoutNetwork(
        input_dim=1,
        output_dim=1,
        hidden_dims=[32, 32],
        dropout_rate=0.1
    )
    
    # 简单训练（标准训练）
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train.reshape(-1, 1))
    
    optimizer = optim.Adam(mc_model.parameters(), lr=1e-3)
    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = mc_model(X_t)
        loss = torch.mean((y_pred - y_t) ** 2)
        loss.backward()
        optimizer.step()
    
    # 预测
    mc_pred = mc_model.predict_with_uncertainty(X_test, n_samples=100)
    
    mse_mc = np.mean((mc_pred.mean.flatten() - y_test) ** 2)
    print(f"测试MSE: {mse_mc:.6f}")
    print(f"平均不确定性: {np.mean(np.sqrt(mc_pred.variance)):.4f}")
    
    # 示例3: GP近似
    print("\n" + "-" * 40)
    print("3. 高斯过程近似")
    print("-" * 40)
    
    gp = GaussianProcessApproximation(
        n_features=50,
        length_scale=0.5,
        noise_level=0.01
    )
    gp.fit(X_train, y_train)
    
    gp_pred = gp.predict(X_test)
    
    mse_gp = np.mean((gp_pred.mean.flatten() - y_test) ** 2)
    print(f"测试MSE: {mse_gp:.6f}")
    print(f"平均不确定性: {np.mean(np.sqrt(gp_pred.variance)):.4f}")
    
    # 示例4: 贝叶斯优化
    print("\n" + "-" * 40)
    print("4. 贝叶斯优化")
    print("-" * 40)
    
    # 定义目标函数
    def objective(x):
        # 简单的测试函数
        return np.sin(3*x[0]) * np.cos(3*x[1]) + 0.1 * (x[0]**2 + x[1]**2)
    
    bounds = [(-2, 2), (-2, 2)]
    
    bo = BayesianOptimization(
        objective, bounds,
        acquisition='ei',
        n_initial=5
    )
    
    result = bo.optimize(n_iterations=15, verbose=False)
    
    print(f"最优解: x = {result['x_opt']}")
    print(f"最优值: f(x) = {result['y_opt']:.6f}")
    print(f"总评估次数: {len(result['y_history'])}")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
