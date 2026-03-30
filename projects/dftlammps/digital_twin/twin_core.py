"""
数字孪生核心引擎 (Digital Twin Core Engine)

物理模型与数据驱动的融合框架，实现材料系统的数字孪生表示。
"""

from __future__ import annotations

import asyncio
import copy
import json
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, Callable, Coroutine, Dict, Generic, List, Optional, Protocol,
    Set, Tuple, TypeVar, Union, runtime_checkable, Iterator
)
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


T = TypeVar('T')
ModelType = TypeVar('ModelType', bound='PhysicalModel')


class TwinState(Enum):
    """数字孪生状态枚举"""
    INITIALIZING = auto()
    SYNCING = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()
    TERMINATED = auto()


class ModelType(Enum):
    """物理模型类型"""
    MOLECULAR_DYNAMICS = "md"
    DENSITY_FUNCTIONAL = "dft"
    FINITE_ELEMENT = "fem"
    MACHINE_LEARNING = "ml"
    HYBRID = "hybrid"


@dataclass
class TwinConfiguration:
    """数字孪生配置"""
    name: str = "default_twin"
    description: str = ""
    sync_interval_ms: float = 100.0
    prediction_horizon: int = 100
    physics_weight: float = 0.5  # 物理模型权重 (0-1)
    data_weight: float = 0.5     # 数据驱动模型权重 (0-1)
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000
    enable_uncertainty: bool = True
    enable_adaptive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TwinConfiguration:
        return cls(**data)


@dataclass
class StateVector:
    """状态向量 - 表示系统当前状态"""
    timestamp: float
    data: NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.float64)
    
    def copy(self) -> StateVector:
        """深拷贝状态向量"""
        return StateVector(
            timestamp=self.timestamp,
            data=self.data.copy(),
            metadata=copy.deepcopy(self.metadata)
        )
    
    def distance_to(self, other: StateVector) -> float:
        """计算与另一个状态向量的欧氏距离"""
        return float(np.linalg.norm(self.data - other.data))
    
    def interpolate(self, other: StateVector, alpha: float) -> StateVector:
        """线性插值两个状态向量"""
        new_data = self.data * (1 - alpha) + other.data * alpha
        new_time = self.timestamp * (1 - alpha) + other.timestamp * alpha
        return StateVector(
            timestamp=new_time,
            data=new_data,
            metadata=copy.deepcopy(self.metadata)
        )


@dataclass
class Observation:
    """观测数据 - 来自实验或传感器"""
    timestamp: float
    sensor_id: str
    values: NDArray[np.float64]
    uncertainty: Optional[NDArray[np.float64]] = None
    quality_score: float = 1.0  # 数据质量评分 (0-1)
    
    def __post_init__(self):
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values, dtype=np.float64)


@dataclass
class Prediction:
    """预测结果"""
    horizon: int  # 预测步数
    timestamps: List[float]
    states: List[StateVector]
    confidence_intervals: Optional[List[Tuple[StateVector, StateVector]]] = None
    model_contributions: Dict[str, float] = field(default_factory=dict)


@runtime_checkable
class PhysicalModel(Protocol):
    """物理模型协议"""
    
    def initialize(self, initial_state: StateVector) -> None:
        """初始化模型"""
        ...
    
    def step(self, dt: float) -> StateVector:
        """执行单步模拟"""
        ...
    
    def predict(self, steps: int, dt: float) -> List[StateVector]:
        """多步预测"""
        ...
    
    def get_state(self) -> StateVector:
        """获取当前状态"""
        ...
    
    def set_state(self, state: StateVector) -> None:
        """设置当前状态"""
        ...


@runtime_checkable
class DataDrivenModel(Protocol):
    """数据驱动模型协议"""
    
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """训练模型"""
        ...
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """预测"""
        ...
    
    def update(self, observation: Observation) -> None:
        """在线更新"""
        ...


class PhysicsBasedModel:
    """
    基于物理的模型基类
    支持分子动力学、密度泛函等物理模拟
    """
    
    def __init__(self, model_type: ModelType, params: Dict[str, Any]):
        self.model_type = model_type
        self.params = params
        self._state: Optional[StateVector] = None
        self._history: deque[StateVector] = deque(maxlen=1000)
        self._initialized = False
        
    def initialize(self, initial_state: StateVector) -> None:
        """初始化物理模型"""
        self._state = initial_state.copy()
        self._history.clear()
        self._history.append(initial_state)
        self._initialized = True
        
    def step(self, dt: float) -> StateVector:
        """执行单步物理模拟"""
        if not self._initialized or self._state is None:
            raise RuntimeError("Model not initialized")
        
        # 基于模型类型执行不同的物理计算
        if self.model_type == ModelType.MOLECULAR_DYNAMICS:
            new_state = self._md_step(dt)
        elif self.model_type == ModelType.DENSITY_FUNCTIONAL:
            new_state = self._dft_step(dt)
        elif self.model_type == ModelType.FINITE_ELEMENT:
            new_state = self._fem_step(dt)
        else:
            new_state = self._generic_step(dt)
        
        self._state = new_state
        self._history.append(new_state)
        return new_state
    
    def _md_step(self, dt: float) -> StateVector:
        """分子动力学步进 (简化版Velocity-Verlet)"""
        state = self._state
        if state is None:
            raise RuntimeError("State is None")
            
        # 简化的MD: 假设state.data包含 [positions, velocities, forces]
        n_atoms = len(state.data) // 3 // 3
        pos = state.data[:n_atoms*3].reshape(-1, 3)
        vel = state.data[n_atoms*3:2*n_atoms*3].reshape(-1, 3)
        force = state.data[2*n_atoms*3:].reshape(-1, 3)
        
        # Velocity-Verlet算法
        mass = self.params.get('mass', 1.0)
        vel += 0.5 * force / mass * dt
        pos += vel * dt
        
        # 简化的力计算 (Lennard-Jones势能)
        new_force = self._compute_lj_forces(pos)
        vel += 0.5 * new_force / mass * dt
        
        new_data = np.concatenate([pos.flatten(), vel.flatten(), new_force.flatten()])
        
        return StateVector(
            timestamp=state.timestamp + dt,
            data=new_data,
            metadata={'step_type': 'md'}
        )
    
    def _compute_lj_forces(self, positions: NDArray[np.float64]) -> NDArray[np.float64]:
        """计算Lennard-Jones力"""
        epsilon = self.params.get('epsilon', 1.0)
        sigma = self.params.get('sigma', 1.0)
        n_atoms = len(positions)
        forces = np.zeros_like(positions)
        
        cutoff = 2.5 * sigma
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                
                if r < cutoff and r > 0.01:
                    # Lennard-Jones力
                    sr6 = (sigma / r) ** 6
                    force_mag = 24 * epsilon * (2 * sr6**2 - sr6) / r
                    force_vec = force_mag * r_vec / r
                    
                    forces[i] -= force_vec
                    forces[j] += force_vec
        
        return forces
    
    def _dft_step(self, dt: float) -> StateVector:
        """密度泛函理论步进 (简化表示)"""
        state = self._state
        if state is None:
            raise RuntimeError("State is None")
        # 简化实现 - 实际DFT需要复杂的电子结构计算
        return StateVector(
            timestamp=state.timestamp + dt,
            data=state.data * 0.99,  # 简化的衰减模型
            metadata={'step_type': 'dft'}
        )
    
    def _fem_step(self, dt: float) -> StateVector:
        """有限元步进 (简化表示)"""
        state = self._state
        if state is None:
            raise RuntimeError("State is None")
        # 简化的热传导方程
        alpha = self.params.get('thermal_diffusivity', 0.1)
        laplacian = self._compute_laplacian(state.data)
        new_data = state.data + alpha * laplacian * dt
        
        return StateVector(
            timestamp=state.timestamp + dt,
            data=new_data,
            metadata={'step_type': 'fem'}
        )
    
    def _compute_laplacian(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """计算拉普拉斯算子 (简化版)"""
        result = np.zeros_like(data)
        if len(data) > 2:
            result[1:-1] = data[:-2] - 2 * data[1:-1] + data[2:]
        return result
    
    def _generic_step(self, dt: float) -> StateVector:
        """通用步进"""
        state = self._state
        if state is None:
            raise RuntimeError("State is None")
        return StateVector(
            timestamp=state.timestamp + dt,
            data=state.data,
            metadata={'step_type': 'generic'}
        )
    
    def predict(self, steps: int, dt: float) -> List[StateVector]:
        """多步预测"""
        predictions = []
        for _ in range(steps):
            predictions.append(self.step(dt))
        return predictions
    
    def get_state(self) -> StateVector:
        """获取当前状态"""
        if self._state is None:
            raise RuntimeError("Model not initialized")
        return self._state.copy()
    
    def set_state(self, state: StateVector) -> None:
        """设置当前状态"""
        self._state = state.copy()
        self._history.append(state)


class NeuralSurrogateModel:
    """
    神经网络代理模型
    用于快速近似物理模拟
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 256, 128]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self._build_network()
        self._is_trained = False
        self._training_data: List[Tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        
    def _build_network(self) -> None:
        """构建简单的神经网络权重"""
        self.weights: List[NDArray[np.float64]] = []
        self.biases: List[NDArray[np.float64]] = []
        
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(dims) - 1):
            # Xavier初始化
            limit = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
            w = np.random.uniform(-limit, limit, (dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.weights.append(w)
            self.biases.append(b)
    
    def _relu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """前向传播"""
        current = X
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            current = current @ w + b
            if i < len(self.weights) - 1:  # 最后一层不使用激活函数
                current = self._relu(current)
        
        return current
    
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], 
            epochs: int = 100, learning_rate: float = 0.001) -> None:
        """训练神经网络"""
        n_samples = len(X)
        
        for epoch in range(epochs):
            # 简化的SGD训练
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            
            for idx in indices:
                x_i = X[idx].reshape(1, -1)
                y_i = y[idx].reshape(1, -1)
                
                # 前向传播
                pred = self._forward(x_i)
                loss = np.mean((pred - y_i) ** 2)
                total_loss += loss
                
                # 简化的梯度下降 (数值近似)
                self._numerical_gradient_update(x_i, y_i, learning_rate)
            
            if epoch % 20 == 0:
                avg_loss = total_loss / n_samples
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        self._is_trained = True
    
    def _numerical_gradient_update(self, x: NDArray[np.float64], y: NDArray[np.float64], 
                                    lr: float, epsilon: float = 1e-5) -> None:
        """数值梯度更新 (简化版)"""
        for i in range(len(self.weights)):
            grad_w = np.zeros_like(self.weights[i])
            
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    # 正向扰动
                    self.weights[i][j, k] += epsilon
                    loss_plus = np.mean((self._forward(x) - y) ** 2)
                    
                    # 负向扰动
                    self.weights[i][j, k] -= 2 * epsilon
                    loss_minus = np.mean((self._forward(x) - y) ** 2)
                    
                    # 恢复
                    self.weights[i][j, k] += epsilon
                    
                    grad_w[j, k] = (loss_plus - loss_minus) / (2 * epsilon)
            
            self.weights[i] -= lr * grad_w
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """预测"""
        if not self._is_trained:
            print("Warning: Model not trained, using random predictions")
        return self._forward(X)
    
    def update(self, observation: Observation) -> None:
        """在线更新 (增量学习)"""
        # 将观测添加到训练数据
        if len(self._training_data) > 10000:
            self._training_data.pop(0)
        self._training_data.append((observation.values, observation.values))
        
        # 定期重训练
        if len(self._training_data) % 100 == 0:
            X = np.array([x for x, _ in self._training_data])
            y = np.array([y for _, y in self._training_data])
            self.fit(X, y, epochs=10, learning_rate=0.0001)


class DigitalTwinCore:
    """
    数字孪生核心引擎
    
    融合物理模型和数据驱动模型，实现材料系统的数字孪生表示。
    支持实时同步、预测和不确定性量化。
    """
    
    def __init__(self, config: Optional[TwinConfiguration] = None):
        self.config = config or TwinConfiguration()
        self.twin_id = str(uuid.uuid4())
        self.state = TwinState.INITIALIZING
        
        # 模型组件
        self.physics_model: Optional[PhysicsBasedModel] = None
        self.data_model: Optional[NeuralSurrogateModel] = None
        
        # 状态管理
        self._current_state: Optional[StateVector] = None
        self._state_history: deque[StateVector] = deque(maxlen=10000)
        self._observation_buffer: deque[Observation] = deque(maxlen=1000)
        
        # 事件回调
        self._callbacks: Dict[str, List[Callable[..., Any]]] = {
            'state_change': [],
            'observation': [],
            'prediction': [],
            'error': [],
        }
        
        # 统计信息
        self._stats = {
            'total_steps': 0,
            'total_observations': 0,
            'sync_count': 0,
            'prediction_errors': deque(maxlen=1000),
            'start_time': time.time(),
        }
        
    def register_physics_model(self, model: PhysicsBasedModel) -> None:
        """注册物理模型"""
        self.physics_model = model
        self._notify('state_change', {'event': 'physics_model_registered'})
    
    def register_data_model(self, model: NeuralSurrogateModel) -> None:
        """注册数据驱动模型"""
        self.data_model = model
        self._notify('state_change', {'event': 'data_model_registered'})
    
    def initialize(self, initial_state: StateVector) -> None:
        """初始化数字孪生"""
        self._current_state = initial_state.copy()
        self._state_history.append(initial_state)
        
        if self.physics_model:
            self.physics_model.initialize(initial_state)
        
        if self.data_model:
            # 准备初始训练数据
            pass
        
        self.state = TwinState.RUNNING
        self._notify('state_change', {'event': 'initialized', 'state': initial_state})
    
    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """注册事件回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _notify(self, event: str, data: Dict[str, Any]) -> None:
        """触发事件通知"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def step(self, dt: float) -> StateVector:
        """
        执行单步演化
        
        融合物理模型和数据驱动模型的预测
        """
        if self.state != TwinState.RUNNING:
            raise RuntimeError(f"Cannot step in state {self.state}")
        
        if self._current_state is None:
            raise RuntimeError("Twin not initialized")
        
        # 物理模型预测
        physics_pred = None
        if self.physics_model:
            physics_pred = self.physics_model.step(dt)
        
        # 数据驱动模型预测
        data_pred = None
        if self.data_model:
            X = self._current_state.data.reshape(1, -1)
            pred = self.data_model.predict(X)
            data_pred = StateVector(
                timestamp=self._current_state.timestamp + dt,
                data=pred.flatten(),
                metadata={'source': 'data_model'}
            )
        
        # 模型融合
        new_state = self._fuse_predictions(
            physics_pred, 
            data_pred,
            self.config.physics_weight,
            self.config.data_weight
        )
        
        self._current_state = new_state
        self._state_history.append(new_state)
        self._stats['total_steps'] += 1
        
        self._notify('state_change', {'event': 'step', 'state': new_state})
        
        return new_state
    
    def _fuse_predictions(self, physics: Optional[StateVector],
                          data: Optional[StateVector],
                          w_physics: float, w_data: float) -> StateVector:
        """
        融合物理模型和数据模型的预测
        
        使用加权平均，支持自适应权重调整
        """
        if physics is None and data is None:
            raise ValueError("At least one model must be provided")
        
        if physics is None:
            return data.copy() if data else None
        if data is None:
            return physics.copy()
        
        # 确保维度匹配
        if len(physics.data) != len(data.data):
            # 截断或填充
            min_len = min(len(physics.data), len(data.data))
            physics_data = physics.data[:min_len]
            data_data = data.data[:min_len]
        else:
            physics_data = physics.data
            data_data = data.data
        
        # 归一化权重
        total_weight = w_physics + w_data
        if total_weight == 0:
            total_weight = 1
        
        norm_physics = w_physics / total_weight
        norm_data = w_data / total_weight
        
        # 加权融合
        fused_data = norm_physics * physics_data + norm_data * data_data
        
        return StateVector(
            timestamp=physics.timestamp,
            data=fused_data,
            metadata={
                'fused': True,
                'physics_weight': norm_physics,
                'data_weight': norm_data,
                'physics_timestamp': physics.timestamp,
                'data_timestamp': data.timestamp,
            }
        )
    
    def observe(self, observation: Observation) -> None:
        """
        接收观测数据
        
        用于与实验数据同步
        """
        self._observation_buffer.append(observation)
        self._stats['total_observations'] += 1
        
        self._notify('observation', {'observation': observation})
        
        # 触发同步
        if self.config.enable_adaptive:
            self._adaptive_sync(observation)
    
    def _adaptive_sync(self, observation: Observation) -> None:
        """自适应同步 - 根据观测调整模型"""
        if self._current_state is None:
            return
        
        # 计算观测与预测的偏差
        obs_values = observation.values
        pred_values = self._current_state.data[:len(obs_values)]
        
        error = np.mean((obs_values - pred_values) ** 2)
        self._stats['prediction_errors'].append(error)
        
        # 自适应调整权重
        if self.config.enable_adaptive and len(self._stats['prediction_errors']) > 10:
            recent_errors = list(self._stats['prediction_errors'])[-10:]
            error_trend = np.mean(recent_errors[-5:]) - np.mean(recent_errors[:5])
            
            if error_trend > 0:  # 误差增加，增加数据模型权重
                self.config.data_weight = min(0.9, self.config.data_weight + 0.05)
                self.config.physics_weight = 1.0 - self.config.data_weight
            else:  # 误差减小，保持或微调
                self.config.physics_weight = min(0.9, self.config.physics_weight + 0.01)
                self.config.data_weight = 1.0 - self.config.physics_weight
        
        # 更新数据驱动模型
        if self.data_model:
            self.data_model.update(observation)
    
    def predict(self, steps: int, dt: float) -> Prediction:
        """
        多步预测
        
        使用融合模型进行未来状态预测
        """
        if self._current_state is None:
            raise RuntimeError("Twin not initialized")
        
        # 保存当前状态
        saved_state = self._current_state.copy()
        saved_physics_state = None
        if self.physics_model:
            saved_physics_state = self.physics_model.get_state()
        
        # 执行预测
        predictions: List[StateVector] = []
        timestamps: List[float] = []
        
        for i in range(steps):
            pred = self.step(dt)
            predictions.append(pred.copy())
            timestamps.append(pred.timestamp)
        
        # 恢复状态
        self._current_state = saved_state
        if self.physics_model and saved_physics_state:
            self.physics_model.set_state(saved_physics_state)
        
        # 计算置信区间 (简化版)
        confidence_intervals = None
        if self.config.enable_uncertainty:
            confidence_intervals = self._compute_confidence_intervals(predictions)
        
        return Prediction(
            horizon=steps,
            timestamps=timestamps,
            states=predictions,
            confidence_intervals=confidence_intervals,
            model_contributions={
                'physics': self.config.physics_weight,
                'data': self.config.data_weight
            }
        )
    
    def _compute_confidence_intervals(self, predictions: List[StateVector]) -> List[Tuple[StateVector, StateVector]]:
        """计算预测置信区间"""
        intervals = []
        
        # 基于历史误差估计不确定性
        if len(self._stats['prediction_errors']) > 0:
            error_std = np.std(list(self._stats['prediction_errors']))
        else:
            error_std = 0.1
        
        for pred in predictions:
            lower = StateVector(
                timestamp=pred.timestamp,
                data=pred.data - 2 * error_std,
                metadata={'bound': 'lower'}
            )
            upper = StateVector(
                timestamp=pred.timestamp,
                data=pred.data + 2 * error_std,
                metadata={'bound': 'upper'}
            )
            intervals.append((lower, upper))
        
        return intervals
    
    def get_current_state(self) -> Optional[StateVector]:
        """获取当前状态"""
        return self._current_state.copy() if self._current_state else None
    
    def get_state_history(self, n: Optional[int] = None) -> List[StateVector]:
        """获取状态历史"""
        history = list(self._state_history)
        if n is not None:
            history = history[-n:]
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats['state'] = self.state.name
        stats['twin_id'] = self.twin_id
        stats['uptime'] = time.time() - stats['start_time']
        
        if len(self._stats['prediction_errors']) > 0:
            errors = list(self._stats['prediction_errors'])
            stats['mean_error'] = np.mean(errors)
            stats['std_error'] = np.std(errors)
            stats['max_error'] = np.max(errors)
        
        return stats
    
    def save(self, filepath: str) -> None:
        """保存数字孪生状态"""
        data = {
            'config': self.config.to_dict(),
            'twin_id': self.twin_id,
            'state': self.state.name,
            'current_state': {
                'timestamp': self._current_state.timestamp if self._current_state else None,
                'data': self._current_state.data.tolist() if self._current_state else None,
                'metadata': self._current_state.metadata if self._current_state else None,
            },
            'statistics': self._stats,
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Digital twin saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> DigitalTwinCore:
        """加载数字孪生状态"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        config = TwinConfiguration.from_dict(data['config'])
        twin = cls(config)
        twin.twin_id = data['twin_id']
        twin.state = TwinState[data['state']]
        
        if data['current_state']['timestamp'] is not None:
            twin._current_state = StateVector(
                timestamp=data['current_state']['timestamp'],
                data=np.array(data['current_state']['data']),
                metadata=data['current_state']['metadata']
            )
        
        twin._stats = data['statistics']
        
        print(f"Digital twin loaded from {filepath}")
        return twin
    
    def pause(self) -> None:
        """暂停数字孪生"""
        self.state = TwinState.PAUSED
        self._notify('state_change', {'event': 'paused'})
    
    def resume(self) -> None:
        """恢复数字孪生"""
        self.state = TwinState.RUNNING
        self._notify('state_change', {'event': 'resumed'})
    
    def terminate(self) -> None:
        """终止数字孪生"""
        self.state = TwinState.TERMINATED
        self._notify('state_change', {'event': 'terminated'})


class TwinCluster:
    """
    数字孪生集群
    
    管理多个相关的数字孪生实例
    """
    
    def __init__(self):
        self.twins: Dict[str, DigitalTwinCore] = {}
        self.relationships: Dict[str, List[str]] = {}
    
    def add_twin(self, name: str, twin: DigitalTwinCore) -> None:
        """添加数字孪生"""
        self.twins[name] = twin
        self.relationships[name] = []
    
    def connect(self, twin1: str, twin2: str) -> None:
        """建立孪生间连接"""
        if twin1 in self.relationships and twin2 in self.twins:
            self.relationships[twin1].append(twin2)
    
    def propagate(self, source: str, observation: Observation) -> None:
        """在集群中传播观测"""
        if source not in self.twins:
            return
        
        visited = set()
        queue = [source]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            self.twins[current].observe(observation)
            
            for neighbor in self.relationships.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)


def demo():
    """演示数字孪生核心功能"""
    print("=" * 60)
    print("数字孪生核心引擎演示")
    print("=" * 60)
    
    # 创建配置
    config = TwinConfiguration(
        name="material_twin_demo",
        description="材料系统数字孪生演示",
        physics_weight=0.6,
        data_weight=0.4,
        enable_adaptive=True
    )
    
    # 创建数字孪生
    twin = DigitalTwinCore(config)
    
    # 创建物理模型 (分子动力学)
    physics_model = PhysicsBasedModel(
        model_type=ModelType.MOLECULAR_DYNAMICS,
        params={'mass': 1.0, 'epsilon': 1.0, 'sigma': 1.0}
    )
    
    # 创建数据驱动模型 (神经网络代理)
    data_model = NeuralSurrogateModel(
        input_dim=27,  # 3 atoms * 3 (pos) + 3 (vel) + 3 (force)
        output_dim=27
    )
    
    # 注册模型
    twin.register_physics_model(physics_model)
    twin.register_data_model(data_model)
    
    # 初始化状态 (3个原子，每个有位置、速度、力)
    np.random.seed(42)
    initial_data = np.random.randn(27) * 0.1
    initial_state = StateVector(
        timestamp=0.0,
        data=initial_data,
        metadata={'init': True}
    )
    
    print(f"\n1. 初始化数字孪生")
    print(f"   配置: {config.name}")
    print(f"   物理权重: {config.physics_weight}, 数据权重: {config.data_weight}")
    
    twin.initialize(initial_state)
    
    # 模拟运行
    print(f"\n2. 执行100步模拟")
    dt = 0.001
    for i in range(100):
        state = twin.step(dt)
        if i % 20 == 0:
            energy = np.sum(state.data ** 2)  # 简化的能量计算
            print(f"   Step {i}: t={state.timestamp:.4f}, E={energy:.4f}")
    
    # 添加观测
    print(f"\n3. 添加实验观测")
    for i in range(5):
        obs = Observation(
            timestamp=0.1 + i * 0.02,
            sensor_id=f"sensor_{i}",
            values=np.random.randn(27) * 0.05,
            quality_score=0.9
        )
        twin.observe(obs)
    print(f"   添加了5个观测")
    
    # 预测
    print(f"\n4. 执行50步预测")
    prediction = twin.predict(steps=50, dt=dt)
    print(f"   预测时域: {prediction.horizon} 步")
    print(f"   模型贡献 - 物理: {prediction.model_contributions['physics']:.2f}, 数据: {prediction.model_contributions['data']:.2f}")
    
    # 统计信息
    print(f"\n5. 统计信息")
    stats = twin.get_statistics()
    print(f"   总步数: {stats['total_steps']}")
    print(f"   总观测数: {stats['total_observations']}")
    print(f"   运行时间: {stats['uptime']:.2f}s")
    
    # 保存和加载
    print(f"\n6. 保存和加载测试")
    twin.save("/tmp/demo_twin.pkl")
    twin2 = DigitalTwinCore.load("/tmp/demo_twin.pkl")
    print(f"   加载后的状态: {twin2.state.name}")
    
    print(f"\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return twin


if __name__ == "__main__":
    demo()
