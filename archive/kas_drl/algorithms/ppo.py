"""
KAS DRL - Proximal Policy Optimization (PPO)
PPO算法实现用于Agent策略优化
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO配置"""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    hidden_dim: int = 256


class ActorNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 均值输出
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        # 对数标准差
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu: [batch, action_dim]
            std: [batch, action_dim]
        """
        features = self.shared(state)
        mu = self.mu(features)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作"""
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PPOAgent:
    """PPO Agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
        device: str = "cpu"
    ):
        self.config = config or PPOConfig()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 网络
        self.actor = ActorNetwork(state_dim, action_dim, self.config.hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, self.config.hidden_dim).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr)
        
        # 内存
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(state_tensor)
                action = mu
                log_prob = torch.zeros(1, 1)
            else:
                action, log_prob = self.actor.sample(state_tensor)
            
            value = self.critic(state_tensor)
        
        # 存储
        self.states.append(state)
        self.actions.append(action.cpu().numpy()[0])
        self.log_probs.append(log_prob.cpu().numpy()[0])
        self.values.append(value.cpu().numpy()[0])
        
        return action.cpu().numpy()[0]
    
    def store_transition(self, reward: float, next_state: np.ndarray, done: bool):
        """存储转移"""
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE优势"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """更新策略"""
        if len(self.states) < self.config.batch_size:
            return {}
        
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).unsqueeze(1).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        # 计算下一状态价值
        with torch.no_grad():
            next_values = self.critic(next_states)
        
        # 计算GAE
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.ppo_epochs):
            # 计算新log prob和熵
            new_log_probs, entropy = self.actor.evaluate(states, actions)
            new_values = self.critic(states)
            
            # 比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            critic_loss = F.mse_loss(new_values, returns)
            
            # 熵奖励
            entropy_loss = -entropy.mean()
            
            # 总损失
            loss = actor_loss + self.config.value_coef * critic_loss + self.config.entropy_coef * entropy_loss
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
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
        
        # 清空内存
        self.clear_memory()
        
        self.training_step += 1
        
        return {
            'actor_loss': total_actor_loss / self.config.ppo_epochs,
            'critic_loss': total_critic_loss / self.config.ppo_epochs,
            'entropy': total_entropy / self.config.ppo_epochs
        }
    
    def clear_memory(self):
        """清空内存"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint['training_step']


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(
        self,
        agent: PPOAgent,
        env,
        num_episodes: int = 1000,
        max_steps: int = 500,
        update_interval: int = 2048,
        eval_interval: int = 100
    ):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.update_interval = update_interval
        self.eval_interval = eval_interval
        
        self.episode_rewards = []
        self.episode_lengths = []
    
    def train(self) -> Dict[str, List]:
        """训练"""
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': []
        }
        
        total_steps = 0
        
        for episode in range(self.num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                # 选择动作
                action = self.agent.select_action(state)
                
                # 执行
                next_state, reward, done, info = self.env.step(action)
                
                # 存储
                self.agent.store_transition(reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                total_steps += 1
                
                # 更新
                if total_steps % self.update_interval == 0:
                    losses = self.agent.update()
                    if losses:
                        history['actor_losses'].append(losses['actor_loss'])
                        history['critic_losses'].append(losses['critic_loss'])
                
                if done:
                    break
            
            history['episode_rewards'].append(episode_reward)
            history['episode_lengths'].append(step + 1)
            
            # 评估
            if episode % self.eval_interval == 0:
                eval_reward = self.evaluate()
                print(f"Episode {episode}, Eval Reward: {eval_reward:.2f}")
        
        return history
    
    def evaluate(self, num_episodes: int = 5) -> float:
        """评估"""
        total_reward = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.max_steps):
                action = self.agent.select_action(state, deterministic=True)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
