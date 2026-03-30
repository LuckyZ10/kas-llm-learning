"""
材料世界模型 - Material World Model
=====================================

环境动态学习、状态转移预测、多步模拟推演的综合框架。

This module implements a world model for material systems that can:
- Learn environment dynamics from simulation data
- Predict state transitions
- Perform multi-step simulation rollouts
- Support imagination-based planning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import pickle
from pathlib import Path
from collections import deque
import warnings
from abc import ABC, abstractmethod
import copy


class StateType(Enum):
    """状态类型枚举"""
    ATOMIC = auto()          # 原子位置/速度
    ELECTRONIC = auto()      # 电子态密度/能带
    THERMODYNAMIC = auto()   # 温度/压力/能量
    STRUCTURAL = auto()      # 晶格参数/对称性
    COMPOSITIONAL = auto()   # 化学成分
    DEFECT = auto()          # 缺陷状态
    INTERFACE = auto()       # 界面状态


class ActionType(Enum):
    """动作类型枚举"""
    TEMPERATURE_CHANGE = auto()   # 温度变化
    PRESSURE_CHANGE = auto()      # 压力变化
    COMPOSITION_CHANGE = auto()   # 成分变化
    DEFECT_INSERTION = auto()     # 缺陷插入
    FIELD_APPLICATION = auto()    # 外场施加
    MECHANICAL_STRESS = auto()    # 机械应力
    CHEMICAL_REACTION = auto()    # 化学反应


@dataclass
class MaterialState:
    """
    材料状态表示
    
    包含材料在不同尺度上的状态信息
    """
    # 基础标识
    state_id: str
    timestamp: float = 0.0
    
    # 原子尺度
    positions: Optional[np.ndarray] = None          # (N, 3) 原子位置
    velocities: Optional[np.ndarray] = None         # (N, 3) 原子速度  
    forces: Optional[np.ndarray] = None             # (N, 3) 原子力
    atomic_numbers: Optional[np.ndarray] = None     # (N,) 原子序数
    cell: Optional[np.ndarray] = None               # (3, 3) 晶格向量
    
    # 电子结构
    charge_density: Optional[np.ndarray] = None     # 电荷密度
    band_energies: Optional[np.ndarray] = None      # 能带能量
    fermi_energy: Optional[float] = None            # 费米能级
    
    # 热力学量
    temperature: float = 300.0                      # 温度 (K)
    pressure: float = 1.0                           # 压力 (bar)
    total_energy: float = 0.0                       # 总能量
    potential_energy: float = 0.0                   # 势能
    kinetic_energy: float = 0.0                     # 动能
    entropy: Optional[float] = None                 # 熵
    free_energy: Optional[float] = None             # 自由能
    
    # 结构特征
    lattice_params: Optional[Tuple[float, float, float]] = None  # 晶格常数
    angles: Optional[Tuple[float, float, float]] = None          # 晶格角度
    volume: Optional[float] = None                  # 体积
    density: Optional[float] = None                 # 密度
    coordination_numbers: Optional[np.ndarray] = None  # 配位数
    
    # 成分信息
    composition: Optional[Dict[str, float]] = None  # 成分字典
    stoichiometry: Optional[str] = None             # 化学计量比
    
    # 缺陷信息
    defect_positions: Optional[List[Tuple[int, np.ndarray]]] = None
    defect_types: Optional[List[str]] = None
    defect_concentration: float = 0.0
    
    # 统计特征 (用于学习)
    features: Optional[np.ndarray] = None           # 特征向量
    
    def to_vector(self) -> np.ndarray:
        """将状态转换为向量表示"""
        components = []
        
        if self.features is not None:
            return self.features
            
        # 热力学量
        components.extend([
            self.temperature / 1000.0,  # 归一化
            self.pressure / 100.0,
            self.total_energy,
            self.potential_energy,
            self.kinetic_energy
        ])
        
        # 结构特征
        if self.volume is not None:
            components.append(self.volume)
        if self.density is not None:
            components.append(self.density)
        
        # 原子特征统计
        if self.positions is not None:
            components.extend([
                np.mean(np.linalg.norm(self.positions, axis=1)),
                np.std(self.positions),
                len(self.positions)
            ])
        
        return np.array(components, dtype=np.float32)
    
    def compute_displacement(self, other: 'MaterialState') -> float:
        """计算与另一个状态的RMSD位移"""
        if self.positions is None or other.positions is None:
            return 0.0
        diff = self.positions - other.positions
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    def compute_energy_difference(self, other: 'MaterialState') -> float:
        """计算能量差"""
        return self.total_energy - other.total_energy
    
    def clone(self) -> 'MaterialState':
        """深拷贝状态"""
        return copy.deepcopy(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'state_id': self.state_id,
            'timestamp': self.timestamp,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'total_energy': self.total_energy,
            'volume': self.volume,
            'density': self.density,
            'composition': self.composition,
            'stoichiometry': self.stoichiometry,
            'defect_concentration': self.defect_concentration
        }


@dataclass
class MaterialAction:
    """
    材料系统动作
    
    表示对材料系统施加的外部干预
    """
    action_id: str
    action_type: ActionType
    magnitude: float = 0.0
    target_atoms: Optional[List[int]] = None
    duration: float = 0.0  # 持续时间 (fs)
    
    # 具体参数
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """转换为向量表示"""
        type_encoding = np.zeros(len(ActionType))
        type_encoding[self.action_type.value - 1] = 1.0
        
        param_vector = np.array([
            self.magnitude,
            self.duration / 1000.0,
            len(self.target_atoms) if self.target_atoms else 0
        ])
        
        return np.concatenate([type_encoding, param_vector])


@dataclass
class Transition:
    """
    状态转移样本
    
    用于训练世界模型
    """
    state: MaterialState
    action: MaterialAction
    next_state: MaterialState
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_training_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """转换为训练样本 (s, a, s', r, done)"""
        return (
            self.state.to_vector(),
            self.action.to_vector(),
            self.next_state.to_vector(),
            self.reward,
            self.done
        )


class StateEncoder(nn.Module):
    """
    状态编码器
    
    将材料状态编码为低维潜在表示
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        latent_dim: int = 32,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # 构建编码器
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
        
    def _get_activation(self, name: str):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'swish': nn.SiLU()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """编码状态"""
        return self.encoder(state)


class StateDecoder(nn.Module):
    """
    状态解码器
    
    从潜在表示解码材料状态
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int] = [64, 128, 256],
        state_dim: int = 10,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, state_dim))
        self.decoder = nn.Sequential(*layers)
    
    def _get_activation(self, name: str):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """解码潜在表示"""
        return self.decoder(latent)


class DynamicsModel(nn.Module):
    """
    动力学模型
    
    预测状态转移: s_{t+1} = f(s_t, a_t)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        use_probabilistic: bool = True,
        activation: str = 'swish'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_probabilistic = use_probabilistic
        
        # 输入层
        input_dim = state_dim + action_dim
        
        # 核心网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self._get_activation(activation)
            ])
            prev_dim = hidden_dim
        
        self.core = nn.Sequential(*layers)
        
        # 输出头
        if use_probabilistic:
            # 概率模型: 输出均值和对数方差
            self.mean_head = nn.Linear(prev_dim, state_dim)
            self.logvar_head = nn.Linear(prev_dim, state_dim)
        else:
            # 确定性模型
            self.state_head = nn.Linear(prev_dim, state_dim)
        
        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 终止状态预测
        self.done_head = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _get_activation(self, name: str):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),
            'gelu': nn.GELU()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Returns:
            dict with keys: 'next_state', 'reward', 'done', optionally 'logvar'
        """
        x = torch.cat([state, action], dim=-1)
        features = self.core(x)
        
        if self.use_probabilistic:
            mean = self.mean_head(features)
            logvar = self.logvar_head(features)
            logvar = torch.clamp(logvar, -10, 2)  # 限制方差范围
            
            # 重参数化采样
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                next_state = mean + eps * std
            else:
                next_state = mean
        else:
            next_state = self.state_head(features)
            mean = next_state
            logvar = None
        
        reward = self.reward_head(features).squeeze(-1)
        done = self.done_head(features).squeeze(-1)
        
        result = {
            'next_state': next_state,
            'mean': mean,
            'reward': reward,
            'done': done
        }
        
        if logvar is not None:
            result['logvar'] = logvar
        
        return result
    
    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测下一个状态
        
        Returns:
            next_state, reward, done_probability
        """
        with torch.no_grad():
            result = self.forward(state, action)
            
            if deterministic or not self.use_probabilistic:
                next_state = result['mean']
            else:
                std = torch.exp(0.5 * result['logvar'])
                eps = torch.randn_like(std)
                next_state = result['mean'] + eps * std
            
            return next_state, result['reward'], result['done']


class RecurrentDynamicsModel(nn.Module):
    """
    循环动力学模型
    
    使用RNN/LSTM/GRU建模时序依赖关系
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        rnn_type: str = 'gru',
        num_layers: int = 2
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        
        # 输入嵌入
        self.input_embed = nn.Linear(state_dim + action_dim, hidden_dim)
        
        # RNN核心
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers,
                batch_first=True
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                hidden_dim,
                hidden_dim,
                num_layers,
                batch_first=True
            )
        
        # 输出头
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        states: torch.Tensor,  # (batch, seq, state_dim)
        actions: torch.Tensor,  # (batch, seq, action_dim)
        hidden: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播序列"""
        batch_size, seq_len, _ = states.shape
        
        # 嵌入输入
        x = torch.cat([states, actions], dim=-1)
        x = self.input_embed(x)
        
        # RNN前向
        rnn_out, hidden = self.rnn(x, hidden)
        
        # 预测输出
        next_states = self.state_head(rnn_out)
        rewards = self.reward_head(rnn_out).squeeze(-1)
        dones = self.done_head(rnn_out).squeeze(-1)
        
        return {
            'next_states': next_states,
            'rewards': rewards,
            'dones': dones,
            'hidden': hidden
        }
    
    def predict_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步预测"""
        # 添加序列维度
        state = state.unsqueeze(1)
        action = action.unsqueeze(1)
        
        result = self.forward(state, action, hidden)
        
        return (
            result['next_states'].squeeze(1),
            result['rewards'].squeeze(1),
            result['dones'].squeeze(1),
            result['hidden']
        )


class EnsembleDynamicsModel(nn.Module):
    """
    集成动力学模型
    
    使用多个模型集成提高预测鲁棒性和不确定性估计
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_models: int = 5,
        hidden_dims: List[int] = [256, 256],
        use_probabilistic: bool = True
    ):
        super().__init__()
        
        self.num_models = num_models
        
        # 创建模型集成
        self.models = nn.ModuleList([
            DynamicsModel(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                use_probabilistic=use_probabilistic
            )
            for _ in range(num_models)
        ])
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """所有模型的前向传播"""
        return [model(state, action) for model in self.models]
    
    def predict_with_uncertainty(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        带不确定性的预测
        
        Returns predictions with epistemic uncertainty
        """
        predictions = self.forward(state, action)
        
        # 收集所有预测
        next_states = torch.stack([p['next_state'] for p in predictions])
        rewards = torch.stack([p['reward'] for p in predictions])
        dones = torch.stack([p['done'] for p in predictions])
        
        # 计算统计量
        next_state_mean = next_states.mean(dim=0)
        next_state_std = next_states.std(dim=0)
        reward_mean = rewards.mean(dim=0)
        reward_std = rewards.std(dim=0)
        done_mean = dones.mean(dim=0)
        
        return {
            'next_state_mean': next_state_mean,
            'next_state_std': next_state_std,
            'reward_mean': reward_mean,
            'reward_std': reward_std,
            'done_mean': done_mean,
            'next_states': next_states,
            'rewards': rewards,
            'dones': dones
        }


class TransitionDataset(Dataset):
    """转移样本数据集"""
    
    def __init__(self, transitions: List[Transition]):
        self.transitions = transitions
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        s, a, s_next, r, done = self.transitions[idx].to_training_sample()
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(s_next),
            torch.FloatTensor([r]),
            torch.FloatTensor([done])
        )


@dataclass
class WorldModelConfig:
    """世界模型配置"""
    
    # 模型架构
    state_dim: int = 20
    action_dim: int = 10
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    
    # 训练参数
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    validation_split: float = 0.2
    
    # 模型类型
    model_type: str = 'ensemble'  # 'simple', 'recurrent', 'ensemble'
    use_probabilistic: bool = True
    ensemble_size: int = 5
    
    # 损失权重
    state_loss_weight: float = 1.0
    reward_loss_weight: float = 0.5
    done_loss_weight: float = 0.1
    kl_loss_weight: float = 0.01
    
    # 设备
    device: str = 'auto'
    
    # 保存/加载
    model_path: Optional[str] = None
    save_interval: int = 10


class MaterialWorldModel:
    """
    材料世界模型主类
    
    整合状态编码器、动力学模型、训练与推理功能
    """
    
    def __init__(self, config: Optional[WorldModelConfig] = None):
        self.config = config or WorldModelConfig()
        self.device = self._get_device()
        
        # 组件
        self.encoder: Optional[StateEncoder] = None
        self.decoder: Optional[StateDecoder] = None
        self.dynamics: Optional[nn.Module] = None
        
        # 训练状态
        self.is_trained = False
        self.training_history: List[Dict[str, float]] = []
        
        # 经验回放缓冲
        self.replay_buffer: deque = deque(maxlen=100000)
        
        self._build_model()
    
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        if self.config.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.config.device)
    
    def _build_model(self):
        """构建模型组件"""
        cfg = self.config
        
        # 状态编码器
        self.encoder = StateEncoder(
            state_dim=cfg.state_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.hidden_dims
        ).to(self.device)
        
        # 状态解码器
        self.decoder = StateDecoder(
            latent_dim=cfg.latent_dim,
            state_dim=cfg.state_dim,
            hidden_dims=list(reversed(cfg.hidden_dims))
        ).to(self.device)
        
        # 动力学模型
        if cfg.model_type == 'ensemble':
            self.dynamics = EnsembleDynamicsModel(
                state_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                num_models=cfg.ensemble_size,
                hidden_dims=cfg.hidden_dims,
                use_probabilistic=cfg.use_probabilistic
            ).to(self.device)
        elif cfg.model_type == 'recurrent':
            self.dynamics = RecurrentDynamicsModel(
                state_dim=cfg.latent_dim,
                action_dim=cfg.action_dim
            ).to(self.device)
        else:
            self.dynamics = DynamicsModel(
                state_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                hidden_dims=cfg.hidden_dims,
                use_probabilistic=cfg.use_probabilistic
            ).to(self.device)
        
        # 优化器
        params = (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.dynamics.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
    
    def add_transition(self, transition: Transition):
        """添加转移样本到回放缓冲"""
        self.replay_buffer.append(transition)
    
    def add_transitions(self, transitions: List[Transition]):
        """批量添加转移样本"""
        for t in transitions:
            self.replay_buffer.append(t)
    
    def encode_state(self, state: MaterialState) -> torch.Tensor:
        """编码材料状态为潜在向量"""
        state_vec = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.encoder(state_vec)
        return latent
    
    def train(
        self,
        transitions: Optional[List[Transition]] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        训练世界模型
        
        Args:
            transitions: 训练数据，None则使用回放缓冲
            verbose: 是否打印训练信息
            
        Returns:
            训练历史
        """
        if transitions is not None:
            self.add_transitions(transitions)
        
        if len(self.replay_buffer) < self.config.batch_size:
            warnings.warn(f"Not enough samples ({len(self.replay_buffer)} < {self.config.batch_size})")
            return {'loss': []}
        
        # 准备数据
        dataset = TransitionDataset(list(self.replay_buffer))
        
        # 划分训练/验证集
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.config.batch_size
        )
        
        # 训练循环
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # 训练
            self.encoder.train()
            self.decoder.train()
            self.dynamics.train()
            
            train_losses = []
            for batch in train_loader:
                s, a, s_next, r, done = [x.to(self.device) for x in batch]
                
                # 编码状态
                z = self.encoder(s)
                z_next_true = self.encoder(s_next)
                
                # 重建
                s_recon = self.decoder(z)
                s_next_recon = self.decoder(z_next_true)
                
                # 动力学预测
                pred = self.dynamics(z, a)
                z_next_pred = pred['next_state']
                
                # 计算损失
                loss = self._compute_loss(
                    s, s_recon,
                    s_next_true, z_next_pred,
                    r, pred['reward'],
                    done, pred['done'],
                    pred
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.decoder.parameters()) +
                    list(self.dynamics.parameters()),
                    1.0
                )
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # 验证
            val_loss = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                      f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
        
        self.is_trained = True
        self.training_history = history
        
        return history
    
    def _compute_loss(
        self,
        s: torch.Tensor,
        s_recon: torch.Tensor,
        z_next_true: torch.Tensor,
        z_next_pred: torch.Tensor,
        r_true: torch.Tensor,
        r_pred: torch.Tensor,
        done_true: torch.Tensor,
        done_pred: torch.Tensor,
        pred_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算综合损失"""
        cfg = self.config
        
        # 重建损失
        recon_loss = F.mse_loss(s_recon, s)
        
        # 状态转移损失
        state_loss = F.mse_loss(z_next_pred, z_next_true)
        
        # 奖励损失
        reward_loss = F.mse_loss(r_pred, r_true)
        
        # 终止状态损失
        done_loss = F.binary_cross_entropy(done_pred, done_true)
        
        # 概率模型KL损失
        kl_loss = torch.tensor(0.0, device=self.device)
        if 'logvar' in pred_dict and cfg.use_probabilistic:
            kl_loss = -0.5 * torch.sum(
                1 + pred_dict['logvar'] - pred_dict['mean'].pow(2) - pred_dict['logvar'].exp()
            )
            kl_loss = kl_loss / s.size(0)
        
        # 总损失
        total_loss = (
            recon_loss +
            cfg.state_loss_weight * state_loss +
            cfg.reward_loss_weight * reward_loss +
            cfg.done_loss_weight * done_loss +
            cfg.kl_loss_weight * kl_loss
        )
        
        return total_loss
    
    def _validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.encoder.eval()
        self.decoder.eval()
        self.dynamics.eval()
        
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                s, a, s_next, r, done = [x.to(self.device) for x in batch]
                
                z = self.encoder(s)
                z_next_true = self.encoder(s_next)
                
                s_recon = self.decoder(z)
                pred = self.dynamics(z, a)
                
                loss = self._compute_loss(
                    s, s_recon,
                    z_next_true, pred['next_state'],
                    r, pred['reward'],
                    done, pred['done'],
                    pred
                )
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def predict(
        self,
        state: MaterialState,
        action: MaterialAction,
        return_uncertainty: bool = False
    ) -> Tuple[MaterialState, float, float]:
        """
        预测下一个状态
        
        Args:
            state: 当前状态
            action: 执行动作
            return_uncertainty: 是否返回不确定性
            
        Returns:
            (next_state, reward, done_prob) or with uncertainty dict
        """
        if not self.is_trained:
            warnings.warn("Model not trained yet!")
        
        self.encoder.eval()
        self.dynamics.eval()
        
        with torch.no_grad():
            # 编码
            s_vec = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.device)
            a_vec = torch.FloatTensor(action.to_vector()).unsqueeze(0).to(self.device)
            
            z = self.encoder(s_vec)
            
            # 预测
            if isinstance(self.dynamics, EnsembleDynamicsModel) and return_uncertainty:
                result = self.dynamics.predict_with_uncertainty(z, a_vec)
                z_next = result['next_state_mean']
                uncertainty = result['next_state_std'].cpu().numpy()
            else:
                result = self.dynamics(z, a_vec)
                z_next = result['next_state']
                uncertainty = None
            
            # 解码
            s_next_recon = self.decoder(z_next)
            
            # 创建新状态
            next_state = self._vector_to_state(s_next_recon.cpu().numpy()[0], state)
            reward = result['reward_mean' if return_uncertainty else 'reward'].item()
            done = result['done_mean' if return_uncertainty else 'done'].item()
        
        if return_uncertainty and uncertainty is not None:
            return next_state, reward, done, {'state_uncertainty': uncertainty}
        
        return next_state, reward, done
    
    def rollout(
        self,
        initial_state: MaterialState,
        action_sequence: List[MaterialAction],
        return_trajectory: bool = True
    ) -> Union[List[MaterialState], Dict[str, Any]]:
        """
        多步模拟推演
        
        Args:
            initial_state: 初始状态
            action_sequence: 动作序列
            return_trajectory: 是否返回完整轨迹
            
        Returns:
            状态列表或包含详细信息的字典
        """
        states = [initial_state]
        rewards = []
        dones = []
        uncertainties = []
        
        current_state = initial_state
        
        for action in action_sequence:
            result = self.predict(current_state, action, return_uncertainty=True)
            next_state, reward, done, info = result
            
            states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            uncertainties.append(info['state_uncertainty'])
            
            current_state = next_state
            
            if done > 0.5:
                break
        
        if return_trajectory:
            return {
                'states': states,
                'rewards': rewards,
                'dones': dones,
                'uncertainties': uncertainties,
                'total_reward': sum(rewards),
                'length': len(states) - 1
            }
        
        return states
    
    def _vector_to_state(
        self,
        vector: np.ndarray,
        template: MaterialState
    ) -> MaterialState:
        """将向量转换为状态对象"""
        # 简化实现: 基于模板创建新状态，更新可学习的特征
        new_state = template.clone()
        
        # 更新热力学量 (假设向量前几维对应这些)
        if len(vector) >= 5:
            new_state.temperature = vector[0] * 1000.0
            new_state.pressure = vector[1] * 100.0
            new_state.total_energy = vector[2]
            new_state.potential_energy = vector[3]
            new_state.kinetic_energy = vector[4]
        
        new_state.features = vector
        new_state.state_id = f"predicted_{id(new_state)}"
        
        return new_state
    
    def save_checkpoint(self, path: str):
        """保存模型检查点"""
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'dynamics': self.dynamics.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained,
            'history': self.training_history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.dynamics.load_state_dict(checkpoint['dynamics'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.config = checkpoint['config']
        self.is_trained = checkpoint['is_trained']
        self.training_history = checkpoint.get('history', [])
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(
            p.numel() for p in self.encoder.parameters()
        ) + sum(
            p.numel() for p in self.decoder.parameters()
        ) + sum(
            p.numel() for p in self.dynamics.parameters()
        )
        
        return {
            'model_type': self.config.model_type,
            'state_dim': self.config.state_dim,
            'action_dim': self.config.action_dim,
            'latent_dim': self.config.latent_dim,
            'total_parameters': total_params,
            'is_trained': self.is_trained,
            'replay_buffer_size': len(self.replay_buffer),
            'device': str(self.device)
        }


class MultiFidelityWorldModel:
    """
    多保真度世界模型
    
    整合不同精度级别的模型 (DFT -> ML势 -> 经验力场)
    """
    
    def __init__(self):
        self.high_fidelity_model: Optional[MaterialWorldModel] = None  # DFT级
        self.medium_fidelity_model: Optional[MaterialWorldModel] = None  # ML势级
        self.low_fidelity_model: Optional[MaterialWorldModel] = None  # 力场级
        
        self.fidelity_thresholds = {
            'high': 0.1,    # 需要高精度的情况
            'medium': 0.5   # 中等精度足够
        }
    
    def set_models(
        self,
        high: MaterialWorldModel,
        medium: MaterialWorldModel,
        low: MaterialWorldModel
    ):
        """设置不同保真度的模型"""
        self.high_fidelity_model = high
        self.medium_fidelity_model = medium
        self.low_fidelity_model = low
    
    def predict(
        self,
        state: MaterialState,
        action: MaterialAction,
        required_accuracy: float = 0.5
    ) -> Tuple[MaterialState, float, float, str]:
        """
        自适应保真度预测
        
        Args:
            required_accuracy: 所需精度
            
        Returns:
            (next_state, reward, done, fidelity_level)
        """
        if required_accuracy <= self.fidelity_thresholds['high'] and self.high_fidelity_model:
            result = self.high_fidelity_model.predict(state, action)
            return (*result, 'high')
        elif required_accuracy <= self.fidelity_thresholds['medium'] and self.medium_fidelity_model:
            result = self.medium_fidelity_model.predict(state, action)
            return (*result, 'medium')
        else:
            result = self.low_fidelity_model.predict(state, action)
            return (*result, 'low')


# 工具函数
def create_synthetic_transitions(
    num_samples: int = 1000,
    state_dim: int = 20,
    action_dim: int = 10
) -> List[Transition]:
    """
    生成合成转移样本用于测试
    """
    transitions = []
    
    for i in range(num_samples):
        # 随机状态
        state = MaterialState(
            state_id=f"state_{i}",
            timestamp=i * 0.1,
            temperature=np.random.uniform(100, 1000),
            pressure=np.random.uniform(0.1, 100),
            total_energy=np.random.randn(),
            potential_energy=np.random.randn(),
            kinetic_energy=np.random.randn(),
            features=np.random.randn(state_dim)
        )
        
        # 随机动作
        action = MaterialAction(
            action_id=f"action_{i}",
            action_type=np.random.choice(list(ActionType)),
            magnitude=np.random.randn(),
            duration=np.random.uniform(1, 100)
        )
        
        # 模拟下一个状态 (简单线性模型)
        delta = 0.1 * state.to_vector() + 0.05 * action.to_vector()[:state_dim]
        next_features = state.to_vector() + delta + 0.01 * np.random.randn(state_dim)
        
        next_state = MaterialState(
            state_id=f"state_{i+1}",
            timestamp=state.timestamp + 0.1,
            temperature=state.temperature * (1 + 0.01 * action.magnitude),
            pressure=state.pressure * (1 + 0.005 * action.magnitude),
            total_energy=state.total_energy + np.random.randn() * 0.1,
            potential_energy=state.potential_energy + np.random.randn() * 0.05,
            kinetic_energy=state.kinetic_energy + np.random.randn() * 0.05,
            features=next_features
        )
        
        # 奖励和终止
        reward = -abs(next_state.total_energy - (-10.0))  # 期望能量接近-10
        done = next_state.temperature > 2000 or next_state.pressure > 500
        
        transition = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done
        )
        
        transitions.append(transition)
    
    return transitions


if __name__ == "__main__":
    # 测试代码
    print("Testing Material World Model...")
    
    # 生成合成数据
    transitions = create_synthetic_transitions(num_samples=500)
    print(f"Generated {len(transitions)} synthetic transitions")
    
    # 创建并训练模型
    config = WorldModelConfig(
        state_dim=20,
        action_dim=10,
        latent_dim=16,
        num_epochs=20,
        batch_size=32
    )
    
    model = MaterialWorldModel(config)
    print(f"Model created with {model.get_model_info()['total_parameters']} parameters")
    
    # 训练
    history = model.train(transitions, verbose=True)
    print(f"Training completed. Final loss: {history['train_loss'][-1]:.4f}")
    
    # 测试预测
    test_state = transitions[0].state
    test_action = transitions[0].action
    
    result = model.predict(test_state, test_action, return_uncertainty=True)
    next_state, reward, done, info = result
    
    print(f"\nPrediction test:")
    print(f"  Current T: {test_state.temperature:.2f} K")
    print(f"  Predicted T: {next_state.temperature:.2f} K")
    print(f"  Reward: {reward:.4f}")
    print(f"  Done prob: {done:.4f}")
    print(f"  Uncertainty: {np.mean(info['state_uncertainty']):.4f}")
    
    # 测试多步推演
    actions = [transitions[i].action for i in range(10)]
    trajectory = model.rollout(test_state, actions)
    
    print(f"\nRollout test:")
    print(f"  Trajectory length: {trajectory['length']}")
    print(f"  Total reward: {trajectory['total_reward']:.4f}")
    print(f"  Temperature trajectory: {[s.temperature for s in trajectory['states']]}")
    
    print("\nAll tests passed!")
