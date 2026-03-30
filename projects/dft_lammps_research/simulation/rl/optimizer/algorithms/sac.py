#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) 算法实现

参考: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", 2018

SAC是一种off-policy算法，通过最大化期望回报和策略熵来学习。
对于材料优化特别适用，因为它能更好地探索连续动作空间。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, namedtuple
import random

import logging

logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    """SAC配置"""
    # 网络结构
    state_dim: int = 128
    action_dim: int = 10
    hidden_dims: List[int] = None
    
    # 训练参数
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # 目标网络软更新系数
    alpha: float = 0.2  # 温度参数 (自动调整时为初始值)
    auto_alpha: bool = True  # 自动调整温度
    target_entropy: Optional[float] = None  # 目标熵
    
    # 训练控制
    buffer_size: int = 1000000
    batch_size: int = 256
    warmup_steps: int = 1000
    update_interval: int = 1
    
    # 其他
    device: str = 'auto'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.target_entropy is None:
            # 目标熵 heuristic: -dim(A)
            self.target_entropy = -self.action_dim


# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """采样批次"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.FloatTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """SAC Actor网络 - 输出高斯分布参数"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 特征提取层
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 输出均值和对数标准差
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作 (重参数化技巧)
        
        Returns:
            (动作, 对数概率)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 重参数化
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        
        # 应用tanh压缩到[-1, 1]
        action = torch.tanh(x_t)
        
        # 计算对数概率 (考虑tanh变换的雅可比行列式)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """获取动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if deterministic:
                mean, _ = self.forward(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.sample(state_tensor)
            
            return action.cpu().numpy()[0]


class CriticNetwork(nn.Module):
    """SAC Critic网络 (Q函数)"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ):
        super().__init__()
        
        # Q(s, a)
        layers = []
        prev_dim = state_dim + action_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SACAgent:
    """
    SAC代理
    
    Soft Actor-Critic算法，适用于连续动作空间的材料优化。
    特点:
    1. Off-policy学习，样本效率高
    2. 最大熵框架，鼓励探索
    3. 自动温度调整
    """
    
    def __init__(self, config: Optional[SACConfig] = None):
        self.config = config or SACConfig()
        
        # 创建网络
        self.actor = ActorNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.config.device)
        
        # 双Q网络
        self.critic1 = CriticNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.config.device)
        self.critic2 = CriticNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.config.device)
        
        # 目标网络
        self.target_critic1 = CriticNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.config.device)
        self.target_critic2 = CriticNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.config.device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.learning_rate)
        
        # 自动温度调整
        if self.config.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.config.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.learning_rate)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(self.config.alpha, device=self.config.device)
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        
        # 训练统计
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """选择动作"""
        return self.actor.get_action(state, deterministic)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        if len(self.replay_buffer) < self.config.warmup_steps:
            return {}
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states = states.to(self.config.device)
        actions = actions.to(self.config.device)
        rewards = rewards.to(self.config.device).unsqueeze(-1)
        next_states = next_states.to(self.config.device)
        dones = dones.to(self.config.device).unsqueeze(-1)
        
        # ========== 更新Critic ==========
        with torch.no_grad():
            # 采样下一动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 目标Q值
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.config.gamma * (target_q - self.alpha * next_log_probs)
        
        # 当前Q值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic损失
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # 更新Critic1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # 更新Critic2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # ========== 更新Actor ==========
        # 重新采样动作
        new_actions, log_probs = self.actor.sample(states)
        
        # 计算Q值
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)
        
        # Actor损失 (最大化Q - alpha * log_prob)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ========== 更新温度 ==========
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.config.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_value = self.alpha.item()
        else:
            alpha_value = self.alpha.item()
        
        # ========== 软更新目标网络 ==========
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': alpha_value,
            'q_value': q.mean().item(),
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
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
            if self.total_steps < self.config.warmup_steps:
                # 随机探索
                action = np.random.uniform(-1, 1, self.config.action_dim)
            else:
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
            if self.total_steps % self.config.update_interval == 0:
                update_stats = self.update()
            else:
                update_stats = {}
            
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
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        logger.info(f"Model loaded from {path}")
    
    def get_q_value(self, state: np.ndarray, action: np.ndarray) -> float:
        """获取Q值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.config.device)
            q1 = self.critic1(state_tensor, action_tensor)
            q2 = self.critic2(state_tensor, action_tensor)
            q = torch.min(q1, q2)
            return q.cpu().item()
