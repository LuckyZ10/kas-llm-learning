#!/usr/bin/env python3
"""
DQN及其变体实现

包含:
- DQN (Deep Q-Network)
- Dueling DQN
- Rainbow DQN (组合多种改进)

参考: Mnih et al. "Human-level control through deep reinforcement learning", 2015
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, namedtuple
import random

import logging

logger = logging.getLogger(__name__)


@dataclass
class DQNConfig:
    """DQN配置"""
    # 网络结构
    state_dim: int = 128
    action_dim: int = 10
    hidden_dims: List[int] = None
    
    # 训练参数
    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005  # 软更新系数
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # 训练控制
    buffer_size: int = 100000
    batch_size: int = 64
    target_update: int = 100  # 目标网络更新频率
    warmup_steps: int = 1000
    
    # 优先经验回放 (Rainbow)
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 0.001
    
    # Dueling DQN
    use_dueling: bool = True
    
    # 多步学习 (Rainbow)
    n_step: int = 3
    
    # 噪声网络 (Noisy Nets)
    use_noisy: bool = False
    
    # 设备
    device: str = 'auto'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        """添加经验 (最大优先级)"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        experience = Experience(state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """采样批次 (优先采样)"""
        if self.size == 0:
            return None
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 获取样本
        samples = [self.buffer[idx] for idx in indices]
        
        states = torch.FloatTensor([e.state for e in samples])
        actions = torch.LongTensor([e.action for e in samples])
        rewards = torch.FloatTensor([e.reward for e in samples])
        next_states = torch.FloatTensor([e.next_state for e in samples])
        dones = torch.FloatTensor([e.done for e in samples])
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        self.priorities[indices] = priorities + 1e-6  # 避免0优先级
    
    def __len__(self) -> int:
        return self.size


class DuelingQNetwork(nn.Module):
    """Dueling Q网络"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ):
        super().__init__()
        
        # 共享特征层
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        
        self.feature_layer = nn.Sequential(*layers)
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling: Q = V + (A - mean(A))
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q


class QNetwork(nn.Module):
    """标准Q网络"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNAgent:
    """
    DQN代理 (支持Dueling和优先经验回放)
    """
    
    def __init__(self, config: Optional[DQNConfig] = None):
        self.config = config or DQNConfig()
        
        # 创建网络
        if self.config.use_dueling:
            self.q_network = DuelingQNetwork(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dims
            ).to(self.config.device)
            self.target_network = DuelingQNetwork(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dims
            ).to(self.config.device)
        else:
            self.q_network = QNetwork(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dims
            ).to(self.config.device)
            self.target_network = QNetwork(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dims
            ).to(self.config.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # 回放缓冲区
        if self.config.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.config.buffer_size,
                self.config.per_alpha,
                self.config.per_beta,
                self.config.per_beta_increment
            )
        else:
            self.replay_buffer = deque(maxlen=self.config.buffer_size)
        
        # epsilon
        self.epsilon = self.config.epsilon_start
        
        # 训练统计
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> int:
        """选择动作 (epsilon-贪心)"""
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.config.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=-1).cpu().item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """存储转移"""
        if self.config.use_per:
            self.replay_buffer.push(state, action, reward, next_state, done)
        else:
            self.replay_buffer.append(Experience(state, action, reward, next_state, done))
        
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """更新Q网络"""
        min_samples = self.config.batch_size if not self.config.use_per else self.config.warmup_steps
        
        if len(self.replay_buffer) < min_samples:
            return {}
        
        if self.config.use_per:
            # 优先采样
            states, actions, rewards, next_states, dones, weights, indices = \
                self.replay_buffer.sample(self.config.batch_size)
            states = states.to(self.config.device)
            actions = actions.to(self.config.device)
            rewards = rewards.to(self.config.device)
            next_states = next_states.to(self.config.device)
            dones = dones.to(self.config.device)
            weights = weights.to(self.config.device)
        else:
            # 均匀采样
            samples = random.sample(self.replay_buffer, self.config.batch_size)
            states = torch.FloatTensor([e.state for e in samples]).to(self.config.device)
            actions = torch.LongTensor([e.action for e in samples]).to(self.config.device)
            rewards = torch.FloatTensor([e.reward for e in samples]).to(self.config.device)
            next_states = torch.FloatTensor([e.next_state for e in samples]).to(self.config.device)
            dones = torch.FloatTensor([e.done for e in samples]).to(self.config.device)
            weights = torch.ones(self.config.batch_size).to(self.config.device)
        
        # 当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值 (Double DQN)
        with torch.no_grad():
            # 选择动作 (使用online network)
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # 评估动作 (使用target network)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.config.gamma * next_q
        
        # TD误差
        td_errors = torch.abs(current_q - target_q)
        
        # 损失 (带重要性采样权重)
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新优先级
        if self.config.use_per:
            self.replay_buffer.update_priorities(indices, td_errors.cpu().detach().numpy())
        
        # 软更新目标网络
        if self.total_steps % self.config.target_update == 0:
            self._soft_update()
        
        # 衰减epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        return {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'td_error': td_errors.mean().item(),
            'epsilon': self.epsilon,
        }
    
    def _soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )
    
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, Any]:
        """训练一个episode"""
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # 选择动作
            action = self.select_action(state, deterministic=False)
            
            # 执行动作
            result = env.step(action)
            next_state = result.state
            reward = result.reward
            done = result.done
            
            # 存储转移
            self.store_transition(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            
            # 更新
            update_stats = self.update()
            
            if done:
                break
            
            state = next_state
        
        self.episode_rewards.append(episode_reward)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'mean_reward': np.mean(self.episode_rewards),
            **update_stats
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        logger.info(f"Model loaded from {path}")
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取所有动作的Q值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0]


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN代理 (使用Dueling架构)"""
    
    def __init__(self, config: Optional[DQNConfig] = None):
        if config is None:
            config = DQNConfig()
        config.use_dueling = True
        super().__init__(config)


class RainbowDQNAgent(DQNAgent):
    """Rainbow DQN代理 (组合多种改进)"""
    
    def __init__(self, config: Optional[DQNConfig] = None):
        if config is None:
            config = DQNConfig()
        config.use_dueling = True
        config.use_per = True
        config.n_step = 3
        super().__init__(config)
        
        # 多步回放缓冲区
        self.n_step_buffer = deque(maxlen=config.n_step)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """存储转移 (多步)"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.config.n_step:
            return
        
        # 计算n步回报
        n_step_reward = 0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            n_step_reward += (self.config.gamma ** i) * r
        
        first_state, first_action, _, _, _ = self.n_step_buffer[0]
        _, _, _, last_next_state, last_done = self.n_step_buffer[-1]
        
        super().store_transition(
            first_state, first_action, n_step_reward, last_next_state, last_done
        )
