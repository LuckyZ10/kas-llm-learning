"""
Real-time Simulation Module
实时模拟模块

提供降阶模型(ROM)、在线学习更新、边缘计算部署功能
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from scipy.sparse.linalg import svds, eigsh
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Union, Any
from collections import deque
from abc import ABC, abstractmethod
import warnings
import time
import json


class ROMMethod(Enum):
    """降阶模型方法枚举"""
    POD = "pod"                    # Proper Orthogonal Decomposition
    PCA = "pca"                    # Principal Component Analysis
    AUTOENCODER = "autoencoder"    # Neural Autoencoder
    DMD = "dmd"                    # Dynamic Mode Decomposition
    GPOD = "gpod"                  # Goal-oriented POD
    RBF = "rbf"                    # Radial Basis Function
    GNN = "gnn"                    # Graph Neural Network
    DEEP_ONET = "deeponet"         # DeepONet


@dataclass
class ROMState:
    """降阶模型状态"""
    reduced_state: np.ndarray
    full_state: Optional[np.ndarray] = None
    reconstruction_error: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class ProperOrthogonalDecomposition:
    """本征正交分解(POD)降阶模型"""
    
    def __init__(self, n_modes: int = 10):
        self.n_modes = n_modes
        self.modes: Optional[np.ndarray] = None
        self.singular_values: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.is_trained = False
        
        # 快照矩阵
        self.snapshots: List[np.ndarray] = []
    
    def add_snapshot(self, state: np.ndarray) -> None:
        """添加快照"""
        self.snapshots.append(state.copy())
    
    def train(self, snapshots: Optional[np.ndarray] = None) -> None:
        """训练POD模型"""
        if snapshots is None:
            if len(self.snapshots) == 0:
                raise ValueError("No snapshots available")
            snapshots = np.array(self.snapshots)
        
        # 计算均值
        self.mean = np.mean(snapshots, axis=0)
        
        # 中心化
        centered = snapshots - self.mean
        
        # SVD分解
        if centered.shape[0] >= centered.shape[1]:
            # 直接SVD
            U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
            self.modes = U[:, :self.n_modes]
            self.singular_values = S[:self.n_modes]
        else:
            # 使用特征值分解
            correlation = centered @ centered.T
            eigenvalues, eigenvectors = eigsh(correlation, k=min(self.n_modes, correlation.shape[0]-1))
            idx = np.argsort(eigenvalues)[::-1]
            self.modes = eigenvectors[:, idx]
            self.singular_values = np.sqrt(np.abs(eigenvalues[idx]))
        
        self.is_trained = True
        
        # 计算能量捕获率
        total_energy = np.sum(S if 'S' in dir() else self.singular_values ** 2)
        captured_energy = np.sum(self.singular_values ** 2)
        self.energy_ratio = captured_energy / total_energy if total_energy > 0 else 1.0
    
    def reduce(self, state: np.ndarray) -> np.ndarray:
        """降维"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        centered = state - self.mean
        return self.modes.T @ centered
    
    def reconstruct(self, reduced_state: np.ndarray) -> np.ndarray:
        """重构"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        return self.mean + self.modes @ reduced_state
    
    def get_mode_energy(self) -> np.ndarray:
        """获取各模态能量"""
        if self.singular_values is None:
            return np.array([])
        return self.singular_values ** 2 / np.sum(self.singular_values ** 2)
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        np.savez(
            filepath,
            modes=self.modes,
            singular_values=self.singular_values,
            mean=self.mean,
            n_modes=self.n_modes,
            energy_ratio=self.energy_ratio
        )
    
    def load(self, filepath: str) -> None:
        """加载模型"""
        data = np.load(filepath)
        self.modes = data['modes']
        self.singular_values = data['singular_values']
        self.mean = data['mean']
        self.n_modes = int(data['n_modes'])
        self.energy_ratio = float(data['energy_ratio'])
        self.is_trained = True


class DynamicModeDecomposition:
    """动态模态分解"""
    
    def __init__(self, n_modes: int = 10, dt: float = 1.0):
        self.n_modes = n_modes
        self.dt = dt
        self.modes: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.amplitudes: Optional[np.ndarray] = None
        self.is_trained = False
    
    def train(self, snapshots: np.ndarray) -> None:
        """训练DMD模型"""
        # 构造快照矩阵
        X = snapshots[:-1].T  # 状态矩阵
        Y = snapshots[1:].T   # 推进后的状态矩阵
        
        # SVD分解
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        
        # 截断
        r = min(self.n_modes, len(S))
        Ur = U[:, :r]
        Sr = np.diag(S[:r])
        Vr = Vh[:r, :].T
        
        # 近似A矩阵
        A_tilde = Ur.T @ Y @ Vr @ np.linalg.inv(Sr)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
        
        # DMD模态
        self.modes = Y @ Vr @ np.linalg.inv(Sr) @ eigenvectors
        self.eigenvalues = eigenvalues
        self.amplitudes = np.linalg.lstsq(self.modes, X[:, 0], rcond=None)[0]
        
        self.is_trained = True
    
    def predict(self, n_steps: int, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """预测未来状态"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        predictions = []
        
        for k in range(n_steps):
            # DMD预测
            state = self.modes @ (self.eigenvalues ** k * self.amplitudes)
            predictions.append(state.real)
        
        return np.array(predictions)
    
    def get_frequency(self) -> np.ndarray:
        """获取频率"""
        if self.eigenvalues is None:
            return np.array([])
        return np.angle(self.eigenvalues) / (2 * np.pi * self.dt)
    
    def get_growth_rate(self) -> np.ndarray:
        """获取增长率"""
        if self.eigenvalues is None:
            return np.array([])
        return np.log(np.abs(self.eigenvalues)) / self.dt


class AutoencoderROM(nn.Module):
    """自编码器降阶模型"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dims: Optional[List[int]] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # 编码器
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_latent_representation(self, x: np.ndarray) -> np.ndarray:
        """获取潜在表示"""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            z = self.encode(x_tensor)
            return z.numpy()


class DeepONetROM(nn.Module):
    """DeepONet降阶模型"""
    
    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        output_dim: int,
        branch_width: int = 128,
        branch_depth: int = 4,
        trunk_width: int = 128,
        trunk_depth: int = 4
    ):
        super().__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.output_dim = output_dim
        
        # Branch网络（处理输入函数）
        branch_layers = []
        in_dim = branch_dim
        for _ in range(branch_depth):
            branch_layers.extend([
                nn.Linear(in_dim, branch_width),
                nn.ReLU()
            ])
            in_dim = branch_width
        branch_layers.append(nn.Linear(in_dim, output_dim))
        self.branch_net = nn.Sequential(*branch_layers)
        
        # Trunk网络（处理位置）
        trunk_layers = []
        in_dim = trunk_dim
        for _ in range(trunk_depth):
            trunk_layers.extend([
                nn.Linear(in_dim, trunk_width),
                nn.ReLU()
            ])
            in_dim = trunk_width
        trunk_layers.append(nn.Linear(in_dim, output_dim))
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # 偏置
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            branch_input: 分支网络输入 [batch_size, branch_dim]
            trunk_input: 主干网络输入 [batch_size, trunk_dim]
        
        Returns:
            输出 [batch_size, 1]
        """
        branch_out = self.branch_net(branch_input)
        trunk_out = self.trunk_net(trunk_input)
        
        # 点积
        output = torch.sum(branch_out * trunk_out, dim=-1, keepdim=True) + self.bias
        return output


class OnlineLearner:
    """在线学习更新模块"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        update_threshold: float = 0.1,
        buffer_size: int = 1000
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.update_threshold = update_threshold
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
        # 数据缓冲区
        self.buffer: deque = deque(maxlen=buffer_size)
        
        # 统计
        self.update_count = 0
        self.last_loss = float('inf')
        
        # 指数移动平均
        self.ema_alpha = 0.99
        self.ema_loss = None
    
    def add_sample(self, input_data: np.ndarray, target: np.ndarray) -> None:
        """添加样本"""
        self.buffer.append({
            'input': input_data,
            'target': target,
            'timestamp': time.time()
        })
    
    def should_update(self) -> bool:
        """判断是否应该更新"""
        if len(self.buffer) < 10:
            return False
        
        if self.ema_loss is None:
            return True
        
        # 基于误差趋势判断
        recent_samples = list(self.buffer)[-10:]
        self.model.eval()
        with torch.no_grad():
            errors = []
            for sample in recent_samples:
                input_tensor = torch.tensor(sample['input'], dtype=torch.float32).unsqueeze(0)
                target_tensor = torch.tensor(sample['target'], dtype=torch.float32).unsqueeze(0)
                
                if hasattr(self.model, 'forward'):
                    output = self.model(input_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    error = torch.mean((output - target_tensor) ** 2).item()
                    errors.append(error)
        
        current_loss = np.mean(errors)
        
        return current_loss > self.update_threshold * self.ema_loss
    
    def update(self, batch_size: int = 32, epochs: int = 1) -> Dict[str, float]:
        """执行在线更新"""
        if len(self.buffer) < batch_size:
            return {'status': 'insufficient_data'}
        
        self.model.train()
        
        # 准备数据
        data = list(self.buffer)
        inputs = np.array([d['input'] for d in data])
        targets = np.array([d['target'] for d in data])
        
        losses = []
        
        for epoch in range(epochs):
            # 随机采样
            indices = np.random.choice(len(data), min(batch_size, len(data)), replace=False)
            
            batch_input = torch.tensor(inputs[indices], dtype=torch.float32)
            batch_target = torch.tensor(targets[indices], dtype=torch.float32)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(batch_input)
            if isinstance(output, tuple):
                output = output[0]
            
            # 损失计算
            loss = nn.MSELoss()(output, batch_target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        
        # 更新EMA
        if self.ema_loss is None:
            self.ema_loss = avg_loss
        else:
            self.ema_loss = self.ema_alpha * self.ema_loss + (1 - self.ema_alpha) * avg_loss
        
        self.update_count += 1
        self.last_loss = avg_loss
        
        return {
            'status': 'updated',
            'loss': avg_loss,
            'ema_loss': self.ema_loss,
            'update_count': self.update_count
        }


class ReducedOrderModel:
    """降阶模型主类"""
    
    def __init__(
        self,
        method: ROMMethod = ROMMethod.POD,
        n_modes: int = 10,
        device: str = 'auto'
    ):
        self.method = method
        self.n_modes = n_modes
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 初始化具体模型
        self.pod_model: Optional[ProperOrthogonalDecomposition] = None
        self.dmd_model: Optional[DynamicModeDecomposition] = None
        self.autoencoder: Optional[AutoencoderROM] = None
        self.deeponet: Optional[DeepONetROM] = None
        
        # 在线学习器
        self.online_learner: Optional[OnlineLearner] = None
        
        # 统计
        self.training_time = 0.0
        self.reconstruction_errors: deque = deque(maxlen=1000)
    
    def train(self, snapshots: np.ndarray, **kwargs) -> Dict[str, Any]:
        """训练降阶模型"""
        start_time = time.time()
        
        if self.method == ROMMethod.POD:
            self.pod_model = ProperOrthogonalDecomposition(self.n_modes)
            self.pod_model.train(snapshots)
            result = {
                'energy_ratio': self.pod_model.energy_ratio,
                'singular_values': self.pod_model.singular_values.tolist()
            }
        
        elif self.method == ROMMethod.DMD:
            dt = kwargs.get('dt', 1.0)
            self.dmd_model = DynamicModeDecomposition(self.n_modes, dt)
            self.dmd_model.train(snapshots)
            result = {
                'frequencies': self.dmd_model.get_frequency().tolist(),
                'growth_rates': self.dmd_model.get_growth_rate().tolist()
            }
        
        elif self.method == ROMMethod.AUTOENCODER:
            input_dim = snapshots.shape[1]
            self.autoencoder = AutoencoderROM(
                input_dim=input_dim,
                latent_dim=self.n_modes,
                hidden_dims=kwargs.get('hidden_dims', [512, 256, 128])
            ).to(self.device)
            
            # 训练自编码器
            result = self._train_autoencoder(snapshots, **kwargs)
            
            # 初始化在线学习器
            self.online_learner = OnlineLearner(self.autoencoder)
        
        elif self.method == ROMMethod.DEEP_ONET:
            branch_dim = kwargs.get('branch_dim', snapshots.shape[1])
            trunk_dim = kwargs.get('trunk_dim', 3)
            output_dim = kwargs.get('output_dim', self.n_modes)
            
            self.deeponet = DeepONetROM(
                branch_dim=branch_dim,
                trunk_dim=trunk_dim,
                output_dim=output_dim
            ).to(self.device)
            
            result = self._train_deeponet(snapshots, **kwargs)
        
        else:
            raise NotImplementedError(f"Method {self.method.value} not implemented")
        
        self.training_time = time.time() - start_time
        result['training_time'] = self.training_time
        
        return result
    
    def _train_autoencoder(
        self,
        snapshots: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3
    ) -> Dict:
        """训练自编码器"""
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=learning_rate
        )
        
        dataset = torch.tensor(snapshots, dtype=torch.float32)
        
        self.autoencoder.train()
        losses = []
        
        for epoch in range(epochs):
            # 随机采样
            indices = torch.randperm(len(dataset))[:batch_size]
            batch = dataset[indices].to(self.device)
            
            optimizer.zero_grad()
            
            recon, latent = self.autoencoder(batch)
            loss = nn.MSELoss()(recon, batch)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        return {
            'final_loss': losses[-1],
            'mean_loss': np.mean(losses)
        }
    
    def _train_deeponet(
        self,
        snapshots: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ) -> Dict:
        """训练DeepONet"""
        optimizer = torch.optim.Adam(
            self.deeponet.parameters(),
            lr=learning_rate
        )
        
        # 假设snapshots包含输入和输出
        # 这里简化处理
        losses = []
        
        for epoch in range(epochs):
            # 模拟训练数据
            branch_input = torch.randn(batch_size, self.deeponet.branch_dim).to(self.device)
            trunk_input = torch.randn(batch_size, self.deeponet.trunk_dim).to(self.device)
            target = torch.randn(batch_size, 1).to(self.device)
            
            optimizer.zero_grad()
            output = self.deeponet(branch_input, trunk_input)
            loss = nn.MSELoss()(output, target)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        return {
            'final_loss': losses[-1],
            'mean_loss': np.mean(losses)
        }
    
    def reduce(self, state: np.ndarray) -> np.ndarray:
        """降维"""
        if self.method == ROMMethod.POD and self.pod_model:
            return self.pod_model.reduce(state)
        elif self.method == ROMMethod.AUTOENCODER and self.autoencoder:
            self.autoencoder.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                latent = self.autoencoder.encode(state_tensor)
                return latent.squeeze().cpu().numpy()
        else:
            raise RuntimeError("Model not trained or method not supported")
    
    def reconstruct(self, reduced_state: np.ndarray) -> np.ndarray:
        """重构"""
        if self.method == ROMMethod.POD and self.pod_model:
            return self.pod_model.reconstruct(reduced_state)
        elif self.method == ROMMethod.AUTOENCODER and self.autoencoder:
            self.autoencoder.eval()
            with torch.no_grad():
                latent_tensor = torch.tensor(reduced_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                recon = self.autoencoder.decode(latent_tensor)
                return recon.squeeze().cpu().numpy()
        else:
            raise RuntimeError("Model not trained or method not supported")
    
    def predict_dynamics(self, n_steps: int, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """预测动态演化"""
        if self.method == ROMMethod.DMD and self.dmd_model:
            return self.dmd_model.predict(n_steps, initial_state)
        else:
            raise NotImplementedError("Dynamics prediction only supported for DMD")
    
    def online_update(self, state: np.ndarray, target: np.ndarray) -> Dict:
        """在线更新"""
        if self.online_learner is None:
            return {'status': 'not_supported'}
        
        # 计算降维表示作为目标
        if self.method == ROMMethod.AUTOENCODER:
            reduced = self.reduce(state)
            self.online_learner.add_sample(state, reduced)
            
            if self.online_learner.should_update():
                return self.online_learner.update()
        
        return {'status': 'no_update_needed'}
    
    def get_compression_ratio(self, original_dim: int) -> float:
        """获取压缩比"""
        return original_dim / self.n_modes
    
    def evaluate_reconstruction(self, test_states: np.ndarray) -> Dict:
        """评估重构质量"""
        errors = []
        
        for state in test_states:
            reduced = self.reduce(state)
            reconstructed = self.reconstruct(reduced)
            error = np.mean((state - reconstructed) ** 2)
            errors.append(error)
            self.reconstruction_errors.append(error)
        
        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'std_error': np.std(errors),
            'rmse': np.sqrt(np.mean(errors))
        }


class EdgeDeployment:
    """边缘计算部署模块"""
    
    def __init__(self, model: ReducedOrderModel):
        self.model = model
        self.is_deployed = False
        self.deployment_config: Dict = {}
        
        # 性能统计
        self.inference_times: deque = deque(maxlen=1000)
        self.memory_usage = 0.0
    
    def optimize_for_edge(self, target_device: str = 'cpu') -> Dict:
        """优化模型用于边缘部署"""
        if self.model.method == ROMMethod.AUTOENCODER and self.model.autoencoder:
            # 量化
            self.model.autoencoder.eval()
            
            # 转换为TorchScript
            example_input = torch.randn(1, self.model.autoencoder.input_dim)
            try:
                scripted = torch.jit.script(self.model.autoencoder)
                self.model.autoencoder = scripted
            except:
                traced = torch.jit.trace(self.model.autoencoder, example_input)
                self.model.autoencoder = traced
            
            # 量化模型
            if target_device == 'cpu':
                self.model.autoencoder = torch.quantization.quantize_dynamic(
                    self.model.autoencoder,
                    {nn.Linear},
                    dtype=torch.qint8
                )
        
        self.deployment_config = {
            'target_device': target_device,
            'quantized': True,
            'optimized': True
        }
        
        return self.deployment_config
    
    def deploy(self, config: Optional[Dict] = None) -> Dict:
        """部署模型"""
        if config:
            self.deployment_config.update(config)
        
        self.is_deployed = True
        
        return {
            'status': 'deployed',
            'config': self.deployment_config,
            'model_type': self.model.method.value
        }
    
    def inference(self, input_data: np.ndarray) -> Dict:
        """执行推理"""
        if not self.is_deployed:
            raise RuntimeError("Model not deployed")
        
        start_time = time.time()
        
        # 执行降维和重构
        reduced = self.model.reduce(input_data)
        reconstructed = self.model.reconstruct(reduced)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return {
            'reduced_state': reduced,
            'reconstructed_state': reconstructed,
            'inference_time': inference_time,
            'compression_ratio': len(input_data) / len(reduced) if len(reduced) > 0 else 0
        }
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if len(self.inference_times) == 0:
            return {'status': 'no_data'}
        
        times = list(self.inference_times)
        
        return {
            'mean_inference_time': np.mean(times),
            'max_inference_time': np.max(times),
            'min_inference_time': np.min(times),
            'throughput': 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
            'total_inferences': len(times)
        }


class RealtimeSimulator:
    """实时模拟器主类"""
    
    def __init__(
        self,
        state_dim: int,
        rom_method: ROMMethod = ROMMethod.POD,
        n_modes: int = 10,
        simulation_dt: float = 0.01
    ):
        self.state_dim = state_dim
        self.simulation_dt = simulation_dt
        
        # 降阶模型
        self.rom = ReducedOrderModel(rom_method, n_modes)
        
        # 边缘部署
        self.edge = EdgeDeployment(self.rom)
        
        # 当前状态
        self.current_full_state: Optional[np.ndarray] = None
        self.current_reduced_state: Optional[np.ndarray] = None
        
        # 模拟历史
        self.simulation_history: deque = deque(maxlen=10000)
        
        # 运行状态
        self.is_running = False
        self.simulation_time = 0.0
    
    def initialize(self, initial_state: np.ndarray) -> None:
        """初始化模拟"""
        self.current_full_state = initial_state.copy()
        self.current_reduced_state = self.rom.reduce(initial_state)
        self.simulation_time = 0.0
        self.simulation_history.clear()
    
    def train_rom(self, training_data: np.ndarray, **kwargs) -> Dict:
        """训练降阶模型"""
        return self.rom.train(training_data, **kwargs)
    
    def step(self, control_input: Optional[np.ndarray] = None) -> Dict:
        """执行单步模拟"""
        if self.current_reduced_state is None:
            raise RuntimeError("Simulator not initialized")
        
        # 简化的动力学模拟
        # 在实际应用中，这里应该使用物理模型或学习到的动力学
        if control_input is not None:
            # 降维控制输入
            control_reduced = self.rom.reduce(control_input) if len(control_input) == self.state_dim else control_input
            
            # 更新降阶状态
            self.current_reduced_state += control_reduced * self.simulation_dt
        
        # 重构完整状态
        self.current_full_state = self.rom.reconstruct(self.current_reduced_state)
        
        # 更新模拟时间
        self.simulation_time += self.simulation_dt
        
        # 记录历史
        self.simulation_history.append({
            'time': self.simulation_time,
            'full_state': self.current_full_state.copy(),
            'reduced_state': self.current_reduced_state.copy()
        })
        
        return {
            'time': self.simulation_time,
            'full_state': self.current_full_state,
            'reduced_state': self.current_reduced_state,
            'compression_ratio': self.state_dim / len(self.current_reduced_state)
        }
    
    def run_simulation(
        self,
        duration: float,
        control_func: Optional[Callable] = None
    ) -> List[Dict]:
        """运行模拟"""
        n_steps = int(duration / self.simulation_dt)
        results = []
        
        for i in range(n_steps):
            control = None
            if control_func:
                control = control_func(self.simulation_time, self.current_full_state)
            
            result = self.step(control)
            results.append(result)
        
        return results
    
    def deploy_to_edge(self, config: Optional[Dict] = None) -> Dict:
        """部署到边缘设备"""
        self.edge.optimize_for_edge()
        return self.edge.deploy(config)
    
    def get_state(self) -> Dict:
        """获取当前状态"""
        return {
            'simulation_time': self.simulation_time,
            'state_dim': self.state_dim,
            'reduced_dim': len(self.current_reduced_state) if self.current_reduced_state is not None else 0,
            'compression_ratio': self.state_dim / len(self.current_reduced_state) if self.current_reduced_state is not None and len(self.current_reduced_state) > 0 else 0,
            'rom_method': self.rom.method.value,
            'history_length': len(self.simulation_history),
            'edge_deployed': self.edge.is_deployed
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Real-time Simulation Module Test")
    print("=" * 60)
    
    # 生成模拟数据
    print("\n1. Generating synthetic data")
    np.random.seed(42)
    n_samples = 500
    state_dim = 100
    
    # 创建具有低维结构的合成数据
    true_modes = 5
    mode_coefficients = np.random.randn(n_samples, true_modes)
    spatial_modes = np.random.randn(state_dim, true_modes)
    snapshots = mode_coefficients @ spatial_modes.T
    snapshots += np.random.randn(n_samples, state_dim) * 0.1
    
    print(f"  Generated {n_samples} snapshots of dimension {state_dim}")
    
    # 测试POD
    print("\n2. Testing POD Reduction")
    pod_rom = ReducedOrderModel(ROMMethod.POD, n_modes=10)
    train_result = pod_rom.train(snapshots)
    print(f"  Training complete")
    print(f"  Energy ratio: {train_result['energy_ratio']:.4f}")
    
    # 测试降维和重构
    test_state = snapshots[0]
    reduced = pod_rom.reduce(test_state)
    reconstructed = pod_rom.reconstruct(reduced)
    error = np.mean((test_state - reconstructed) ** 2)
    print(f"  Reconstruction error: {error:.6f}")
    print(f"  Compression ratio: {pod_rom.get_compression_ratio(state_dim):.2f}x")
    
    # 测试Autoencoder
    print("\n3. Testing Autoencoder Reduction")
    ae_rom = ReducedOrderModel(ROMMethod.AUTOENCODER, n_modes=8)
    ae_result = ae_rom.train(snapshots, epochs=50, batch_size=32)
    print(f"  Training complete")
    print(f"  Final loss: {ae_result['final_loss']:.6f}")
    
    reduced_ae = ae_rom.reduce(test_state)
    reconstructed_ae = ae_rom.reconstruct(reduced_ae)
    error_ae = np.mean((test_state - reconstructed_ae) ** 2)
    print(f"  Reconstruction error: {error_ae:.6f}")
    
    # 测试DMD
    print("\n4. Testing Dynamic Mode Decomposition")
    dmd_rom = ReducedOrderModel(ROMMethod.DMD, n_modes=6)
    dmd_result = dmd_rom.train(snapshots, dt=0.1)
    print(f"  Training complete")
    print(f"  Frequencies: {np.round(dmd_result['frequencies'][:3], 4)}")
    
    # 预测动态
    predictions = dmd_rom.predict_dynamics(n_steps=20)
    print(f"  Predicted {len(predictions)} future states")
    
    # 测试实时模拟器
    print("\n5. Testing Real-time Simulator")
    simulator = RealtimeSimulator(
        state_dim=state_dim,
        rom_method=ROMMethod.POD,
        n_modes=10,
        simulation_dt=0.01
    )
    
    # 训练
    simulator.train_rom(snapshots)
    
    # 初始化
    simulator.initialize(snapshots[0])
    
    # 运行模拟
    def control_func(t, state):
        return np.random.randn(state_dim) * 0.01
    
    results = simulator.run_simulation(duration=0.5, control_func=control_func)
    print(f"  Ran simulation for {len(results)} steps")
    print(f"  Final compression ratio: {results[-1]['compression_ratio']:.2f}x")
    
    # 测试边缘部署
    print("\n6. Testing Edge Deployment")
    deploy_result = simulator.deploy_to_edge({'target_device': 'cpu'})
    print(f"  Deployment status: {deploy_result['status']}")
    
    # 边缘推理测试
    edge_result = simulator.edge.inference(snapshots[10])
    print(f"  Inference time: {edge_result['inference_time']*1000:.3f} ms")
    
    # 性能统计
    stats = simulator.edge.get_performance_stats()
    print(f"  Mean inference time: {stats.get('mean_inference_time', 0)*1000:.3f} ms")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
