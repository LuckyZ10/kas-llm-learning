#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) 算法实现

参考: Schulman et al. "Proximal Policy Optimization Algorithms", 2017

PPO是一种稳定的策略梯度方法，通过裁剪目标函数来防止策略更新过大。
适用于连续和离散动作空间的材料优化问题。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """PPO配置"""
    # 网络结构
    state_dim: int = 128
    action_dim: int = 10
    hidden_dims: List[int] = None
    
    # 训练参数
    learning_rate: float = 3e-4
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    epsilon: float = 0.2  # PPO裁剪参数
    value_coef: float = 0.5  # 价值函数系数
    entropy_coef: float = 0.01  # 熵正则化系数
    max_grad_norm: float = 0.5  # 梯度裁剪
    
    # 训练控制
    n_epochs: int = 10  # 每次更新的epoch数
    batch_size: int = 64
    buffer_size: int = 2048
    n_steps: int = 2048  # 收集步数
    
    # 其他
    device: str = 'auto'
    use_gae: bool = True
    normalize_advantage: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ActorNetwork(nn.Module):
    """策略网络 (Actor)"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        action_type: str = 'continuous'
    ):
        super().__init__()
        self.action_type = action_type
        
        # 构建网络
        layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
            ])
            prev_dim = dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        if action_type == 'continuous':
            # 连续动作: 输出均值和对数标准差
            self.mean_head = nn.Linear(prev_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # 离散动作: 输出logits
            self.logits_head = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Union[Normal, Categorical]:
        features = self.feature_extractor(state)
        
        if self.action_type == 'continuous':
            mean = self.mean_head(features)
            std = torch.exp(self.log_std)
            return Normal(mean, std)
        else:
            logits = self.logits_head(features)
            return Categorical(logits=logits)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor]:
        """获取动作"""
        dist = self.forward(state)
        
        if deterministic:
            if self.action_type == 'continuous':
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1) if self.action_type == 'continuous' else dist.log_prob(action)
        
        return action.detach().cpu().numpy(), log_prob
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作的对数概率和熵"""
        dist = self.forward(state)
        
        if self.action_type == 'continuous':
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """价值网络 (Critic)"""
    
    def __init__(
        self,
        state_dim: int,
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
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class RolloutBuffer:
    """回放缓冲区 - 存储轨迹数据"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: torch.Tensor,
        done: bool,
        next_state: np.ndarray
    ):
        """添加样本"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_states.append(next_state)
    
    def get(self) -> Dict[str, Any]:
        """获取所有数据"""
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': torch.stack(self.log_probs) if self.log_probs else torch.tensor([]),
            'dones': np.array(self.dones),
            'next_states': np.array(self.next_states),
        }
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.next_states.clear()
    
    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """
    PPO代理
    
    用于材料优化的策略梯度方法，支持连续和离散动作空间。
    """
    
    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        
        # 创建网络
        self.actor = ActorNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
            action_type='continuous'
        ).to(self.config.device)
        
        self.critic = CriticNetwork(
            self.config.state_dim,
            self.config.hidden_dims
        ).to(self.config.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate
        )
        
        # 回放缓冲区
        self.buffer = RolloutBuffer()
        
        # 训练统计
        self.episode_rewards = deque(maxlen=100)
        self.training_step = 0
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        选择动作
        
        Returns:
            (动作, 额外信息)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor).cpu().item()
        
        info = {
            'log_prob': log_prob.cpu(),
            'value': value
        }
        
        return action, info
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: torch.Tensor,
        done: bool,
        next_state: np.ndarray
    ):
        """存储转移"""
        self.buffer.add(state, action, reward, value, log_prob, done, next_state)
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE (Generalized Advantage Estimation)
        
        Reference: Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2016
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[-1]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """
        更新策略
        
        Returns:
            训练统计
        """
        if len(self.buffer) == 0:
            return {}
        
        # 获取数据
        data = self.buffer.get()
        states = torch.FloatTensor(data['states']).to(self.config.device)
        actions = torch.FloatTensor(data['actions']).to(self.config.device)
        old_log_probs = data['log_probs'].to(self.config.device)
        
        # 计算下一状态的价值
        with torch.no_grad():
            next_states = torch.FloatTensor(data['next_states']).to(self.config.device)
            next_values = self.critic(next_states).cpu().numpy()
        
        # 计算优势和回报
        advantages, returns = self.compute_gae(
            data['rewards'],
            data['values'],
            data['dones'],
            next_values
        )
        
        advantages = torch.FloatTensor(advantages).to(self.config.device)
        returns = torch.FloatTensor(returns).to(self.config.device)
        
        # 标准化优势
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 训练统计
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        n_updates = 0
        
        # 多epoch训练
        for _ in range(self.config.n_epochs):
            # 生成随机批次
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估当前策略
                log_probs, entropy = self.actor.evaluate_actions(batch_states, batch_actions)
                values = self.critic(batch_states)
                
                # 计算PPO损失
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # 熵奖励 (鼓励探索)
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = actor_loss + self.config.value_coef * critic_loss + self.config.entropy_coef * entropy_loss
                
                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                
                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # 清空缓冲区
        self.buffer.clear()
        
        self.training_step += 1
        
        # 返回统计
        if n_updates > 0:
            return {
                'actor_loss': total_actor_loss / n_updates,
                'critic_loss': total_critic_loss / n_updates,
                'entropy': total_entropy / n_updates,
                'advantage_mean': advantages.mean().item(),
                'return_mean': returns.mean().item(),
            }
        return {}
    
    def train_episode(self, env, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        训练一个episode
        
        Args:
            env: 环境
            max_steps: 最大步数
            
        Returns:
            训练结果
        """
        max_steps = max_steps or self.config.n_steps
        
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # 选择动作
            action, info = self.select_action(state, deterministic=False)
            
            # 执行动作
            result = env.step(action)
            next_state = result.state
            reward = result.reward
            done = result.done
            
            # 存储转移
            self.store_transition(
                state,
                action,
                reward,
                info['value'],
                info['log_prob'],
                done,
                next_state
            )
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        # 更新策略
        update_stats = self.update()
        
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
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        logger.info(f"Model loaded from {path}")
    
    def get_value(self, state: np.ndarray) -> float:
        """获取状态价值"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
            value = self.critic(state_tensor).cpu().item()
        return value
