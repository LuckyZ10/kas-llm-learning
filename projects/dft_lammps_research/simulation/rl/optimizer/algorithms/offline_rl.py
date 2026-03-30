#!/usr/bin/env python3
"""
离线强化学习算法

包含:
- CQL (Conservative Q-Learning)
- Decision Transformer

适用于利用已有材料数据库进行学习的场景。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque

import logging

logger = logging.getLogger(__name__)


@dataclass
class CQLConfig:
    """CQL配置"""
    state_dim: int = 128
    action_dim: int = 10
    hidden_dims: List[int] = None
    
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    
    # CQL特定参数
    cql_alpha: float = 1.0  # CQL正则化强度
    cql_temperature: float = 1.0
    target_action_gap: float = 10.0
    
    # 训练控制
    batch_size: int = 256
    n_epochs: int = 100
    
    device: str = 'auto'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class DTConfig:
    """Decision Transformer配置"""
    state_dim: int = 128
    action_dim: int = 10
    hidden_dim: int = 256
    n_layers: int = 3
    n_heads: int = 4
    
    context_length: int = 20  # 上下文长度
    max_episode_length: int = 1000
    
    learning_rate: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 100
    
    device: str = 'auto'
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CQLAgent:
    """
    CQL (Conservative Q-Learning) 代理
    
    参考: Kumar et al. "Conservative Q-Learning for Offline Reinforcement Learning", NeurIPS 2020
    
    CQL通过添加保守性正则化来防止离线RL中的价值高估问题，
    特别适合从有限的材料数据库中学习。
    """
    
    def __init__(self, config: Optional[CQLConfig] = None):
        self.config = config or CQLConfig()
        
        # 创建网络
        from .sac import ActorNetwork, CriticNetwork
        
        self.actor = ActorNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.config.device)
        
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
        
        # 数据集
        self.dataset = []
        
        # 训练统计
        self.epoch = 0
    
    def load_dataset(self, trajectories: List[Dict]):
        """
        加载离线数据集
        
        Args:
            trajectories: 轨迹列表，每个轨迹包含states, actions, rewards
        """
        self.dataset = []
        
        for traj in trajectories:
            states = traj['states']
            actions = traj['actions']
            rewards = traj['rewards']
            next_states = traj.get('next_states', states[1:] + [states[-1]])
            dones = traj.get('dones', [False] * len(rewards))
            
            for i in range(len(rewards)):
                self.dataset.append({
                    'state': states[i],
                    'action': actions[i],
                    'reward': rewards[i],
                    'next_state': next_states[i],
                    'done': dones[i] if i < len(dones) else True
                })
        
        logger.info(f"Loaded dataset with {len(self.dataset)} transitions")
    
    def _sample_batch(self, batch_size: int) -> Tuple:
        """采样批次"""
        indices = np.random.choice(len(self.dataset), batch_size, replace=False)
        batch = [self.dataset[i] for i in indices]
        
        states = torch.FloatTensor([d['state'] for d in batch]).to(self.config.device)
        actions = torch.FloatTensor([d['action'] for d in batch]).to(self.config.device)
        rewards = torch.FloatTensor([d['reward'] for d in batch]).to(self.config.device).unsqueeze(-1)
        next_states = torch.FloatTensor([d['next_state'] for d in batch]).to(self.config.device)
        dones = torch.FloatTensor([d['done'] for d in batch]).to(self.config.device).unsqueeze(-1)
        
        return states, actions, rewards, next_states, dones
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        if len(self.dataset) < self.config.batch_size:
            return {}
        
        n_batches = len(self.dataset) // self.config.batch_size
        
        total_critic1_loss = 0
        total_critic2_loss = 0
        total_actor_loss = 0
        total_cql_loss = 0
        
        for _ in range(n_batches):
            states, actions, rewards, next_states, dones = self._sample_batch(self.config.batch_size)
            
            # ========== Critic更新 (带CQL正则化) ==========
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.sample(next_states)
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards + (1 - dones) * self.config.gamma * target_q
            
            # 当前Q值
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            
            # 标准贝尔曼损失
            bellman_loss1 = F.mse_loss(current_q1, target_q)
            bellman_loss2 = F.mse_loss(current_q2, target_q)
            
            # CQL正则化: 鼓励Q函数对数据集动作的Q值高于其他动作
            # 采样随机动作
            random_actions = torch.FloatTensor(
                np.random.uniform(-1, 1, (self.config.batch_size * 10, self.config.action_dim))
            ).to(self.config.device)
            
            # 重复状态以匹配随机动作数量
            states_repeated = states.repeat(10, 1)
            
            # 计算随机动作的Q值
            random_q1 = self.critic1(states_repeated, random_actions)
            random_q2 = self.critic2(states_repeated, random_actions)
            
            # CQL损失: 数据集动作的Q值应该高于随机动作
            cql_loss1 = torch.logsumexp(random_q1.view(10, -1) / self.config.cql_temperature, dim=0).mean() * self.config.cql_temperature
            cql_loss1 = cql_loss1 - current_q1.mean()
            
            cql_loss2 = torch.logsumexp(random_q2.view(10, -1) / self.config.cql_temperature, dim=0).mean() * self.config.cql_temperature
            cql_loss2 = cql_loss2 - current_q2.mean()
            
            # 总Critic损失
            critic1_loss = bellman_loss1 + self.config.cql_alpha * cql_loss1
            critic2_loss = bellman_loss2 + self.config.cql_alpha * cql_loss2
            
            # 更新Critic
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()
            
            # ========== Actor更新 ==========
            new_actions, log_probs = self.actor.sample(states)
            q1 = self.critic1(states, new_actions)
            q2 = self.critic2(states, new_actions)
            q = torch.min(q1, q2)
            
            actor_loss = (self.config.cql_alpha * log_probs - q).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ========== 软更新目标网络 ==========
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
            
            # 累计损失
            total_critic1_loss += critic1_loss.item()
            total_critic2_loss += critic2_loss.item()
            total_actor_loss += actor_loss.item()
            total_cql_loss += (cql_loss1.item() + cql_loss2.item()) / 2
        
        self.epoch += 1
        
        return {
            'critic1_loss': total_critic1_loss / n_batches,
            'critic2_loss': total_critic2_loss / n_batches,
            'actor_loss': total_actor_loss / n_batches,
            'cql_loss': total_cql_loss / n_batches,
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        return self.actor.get_action(state, deterministic)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        logger.info(f"Model loaded from {path}")


class DecisionTransformer(nn.Module):
    """
    Decision Transformer模型
    
    使用Transformer架构建模序列决策问题，
    将RL视为条件序列建模问题。
    
    Reference: Chen et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling", NeurIPS 2021
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        context_length: int = 20,
        max_episode_length: int = 1000
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        
        # 嵌入层
        self.reward_embedding = nn.Linear(1, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        
        # 时间嵌入
        self.timestep_embedding = nn.Embedding(max_episode_length, hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出层
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # 因果掩码
        self.register_buffer('mask', self._generate_causal_mask(context_length * 3))
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """生成因果掩码"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            returns: (batch, seq_len, 1) 目标回报
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            timesteps: (batch, seq_len)
            
        Returns:
            预测的动作 (batch, seq_len, action_dim)
        """
        batch_size, seq_len = states.shape[:2]
        
        # 嵌入
        reward_emb = self.reward_embedding(returns)
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        time_emb = self.timestep_embedding(timesteps)
        
        # 交错序列: r1, s1, a1, r2, s2, a2, ...
        sequence = torch.zeros(
            batch_size, seq_len * 3, self.hidden_dim,
            device=states.device
        )
        
        sequence[:, 0::3] = reward_emb + time_emb
        sequence[:, 1::3] = state_emb + time_emb
        sequence[:, 2::3] = action_emb + time_emb
        
        # Transformer
        mask = self.mask[:seq_len * 3, :seq_len * 3]
        hidden = self.transformer(sequence, mask=mask, is_causal=True)
        
        # 预测动作 (在每个状态后预测动作)
        action_hidden = hidden[:, 1::3]
        predicted_actions = self.action_predictor(action_hidden)
        
        return predicted_actions


class DecisionTransformerAgent:
    """
    Decision Transformer代理
    
    通过序列建模进行离线强化学习，
    特别适合处理材料优化中的长程依赖问题。
    """
    
    def __init__(self, config: Optional[DTConfig] = None):
        self.config = config or DTConfig()
        
        # 创建模型
        self.model = DecisionTransformer(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
            self.config.n_layers,
            self.config.n_heads,
            self.config.context_length,
            self.config.max_episode_length
        ).to(self.config.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # 数据集
        self.trajectories = []
        
        # 训练统计
        self.epoch = 0
        
        # 在线推理缓存
        self.context_returns = deque(maxlen=self.config.context_length)
        self.context_states = deque(maxlen=self.config.context_length)
        self.context_actions = deque(maxlen=self.config.context_length)
        self.context_timesteps = deque(maxlen=self.config.context_length)
    
    def load_dataset(self, trajectories: List[Dict]):
        """加载离线轨迹数据集"""
        self.trajectories = trajectories
        logger.info(f"Loaded {len(trajectories)} trajectories")
    
    def _sample_batch(self, batch_size: int) -> Tuple:
        """采样批次"""
        # 随机采样轨迹
        traj_indices = np.random.choice(len(self.trajectories), batch_size)
        
        batch_returns = []
        batch_states = []
        batch_actions = []
        batch_timesteps = []
        
        for idx in traj_indices:
            traj = self.trajectories[idx]
            
            # 计算累积回报
            rewards = np.array(traj['rewards'])
            returns = np.cumsum(rewards[::-1])[::-1]
            
            # 随机采样子序列
            traj_len = len(returns)
            if traj_len <= self.config.context_length:
                start_idx = 0
                end_idx = traj_len
            else:
                start_idx = np.random.randint(0, traj_len - self.config.context_length)
                end_idx = start_idx + self.config.context_length
            
            # 填充
            seq_returns = np.zeros(self.config.context_length)
            seq_states = np.zeros((self.config.context_length, self.config.state_dim))
            seq_actions = np.zeros((self.config.context_length, self.config.action_dim))
            seq_timesteps = np.zeros(self.config.context_length, dtype=int)
            
            actual_len = end_idx - start_idx
            seq_returns[:actual_len] = returns[start_idx:end_idx]
            seq_states[:actual_len] = np.array(traj['states'][start_idx:end_idx])
            seq_actions[:actual_len] = np.array(traj['actions'][start_idx:end_idx])
            seq_timesteps[:actual_len] = np.arange(start_idx, end_idx)
            
            batch_returns.append(seq_returns)
            batch_states.append(seq_states)
            batch_actions.append(seq_actions)
            batch_timesteps.append(seq_timesteps)
        
        returns = torch.FloatTensor(batch_returns).unsqueeze(-1).to(self.config.device)
        states = torch.FloatTensor(batch_states).to(self.config.device)
        actions = torch.FloatTensor(batch_actions).to(self.config.device)
        timesteps = torch.LongTensor(batch_timesteps).to(self.config.device)
        
        return returns, states, actions, timesteps
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        if len(self.trajectories) == 0:
            return {}
        
        n_batches = sum(len(t['states']) for t in self.trajectories) // self.config.batch_size
        n_batches = max(1, n_batches // 10)  # 每个epoch采样一部分
        
        total_loss = 0
        
        for _ in range(n_batches):
            returns, states, actions, timesteps = self._sample_batch(self.config.batch_size)
            
            # 预测动作
            predicted_actions = self.model(returns, states, actions, timesteps)
            
            # 动作预测损失 (MSE)
            loss = F.mse_loss(predicted_actions, actions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.epoch += 1
        
        return {
            'loss': total_loss / n_batches,
        }
    
    def reset_context(self, target_return: float):
        """重置上下文 (开始新episode)"""
        self.context_returns.clear()
        self.context_states.clear()
        self.context_actions.clear()
        self.context_timesteps.clear()
        
        # 初始填充
        self.context_returns.append(target_return)
        self.context_states.append(np.zeros(self.config.state_dim))
        self.context_actions.append(np.zeros(self.config.action_dim))
        self.context_timesteps.append(0)
    
    def select_action(
        self,
        state: np.ndarray,
        timestep: int,
        target_return: float
    ) -> np.ndarray:
        """
        选择动作 (在线推理)
        
        Args:
            state: 当前状态
            timestep: 当前时间步
            target_return: 目标累积回报
        """
        # 更新上下文
        self.context_returns.append(target_return)
        self.context_states.append(state)
        self.context_actions.append(np.zeros(self.config.action_dim))  # 占位
        self.context_timesteps.append(timestep)
        
        # 准备输入
        returns = torch.FloatTensor([list(self.context_returns)]).unsqueeze(-1).to(self.config.device)
        states = torch.FloatTensor([list(self.context_states)]).to(self.config.device)
        actions = torch.FloatTensor([list(self.context_actions)]).to(self.config.device)
        timesteps = torch.LongTensor([list(self.context_timesteps)]).to(self.config.device)
        
        # 预测
        with torch.no_grad():
            predicted_actions = self.model(returns, states, actions, timesteps)
            action = predicted_actions[0, -1].cpu().numpy()
        
        # 更新上下文中的动作
        self.context_actions[-1] = action
        
        return action
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Model loaded from {path}")
