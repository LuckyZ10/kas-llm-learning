"""
KAS DRL - Soft Actor-Critic (SAC)
SAC算法实现用于Agent策略优化
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class SACConfig:
    """SAC配置"""
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2  # 温度参数
    automatic_entropy_tuning: bool = True
    target_entropy: Optional[float] = None
    buffer_size: int = 100000
    batch_size: int = 256
    hidden_dim: int = 256
    updates_per_step: int = 1


class SACActor(nn.Module):
    """随机策略网络"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: [batch, action_dim]
            log_std: [batch, action_dim]
        """
        features = self.shared(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 重参数化技巧
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # 计算log prob
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """获取动作"""
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean)
        
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.sample()
        return torch.tanh(x_t)


class SACCritic(nn.Module):
    """Q值网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SACAgent:
    """SAC Agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[SACConfig] = None,
        device: str = "cpu"
    ):
        self.config = config or SACConfig()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor
        self.actor = SACActor(state_dim, action_dim, self.config.hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        
        # Critic (双Q网络)
        self.critic1 = SACCritic(state_dim, action_dim, self.config.hidden_dim).to(device)
        self.critic2 = SACCritic(state_dim, action_dim, self.config.hidden_dim).to(device)
        
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.lr)
        
        # 温度参数
        if self.config.automatic_entropy_tuning:
            if self.config.target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = self.config.target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = self.config.alpha
        
        # 回放缓冲区
        from .ddpg import ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size,
            state_dim,
            action_dim
        )
        
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy()[0]
    
    def update(self) -> Optional[Dict[str, float]]:
        """更新策略"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # 采样
        from .ddpg import ReplayBuffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.detach()
            else:
                alpha = self.alpha
            
            next_q = next_q - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * next_q
        
        # 更新Critic1
        current_q1 = self.critic1(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # 更新Critic2
        current_q2 = self.critic2(states, actions)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha
        
        actor_loss = (alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新温度参数
        if self.config.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_value = self.alpha.item()
        else:
            alpha_value = self.alpha
        
        # 软更新目标网络
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        self.training_step += 1
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': alpha_value,
            'q_value': current_q1.mean().item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'training_step': self.training_step,
            'alpha': self.alpha if not isinstance(self.alpha, torch.Tensor) else self.alpha.item()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        self.training_step = checkpoint['training_step']
        
        if self.config.automatic_entropy_tuning:
            self.alpha = checkpoint['alpha']
            self.log_alpha = torch.tensor([np.log(self.alpha)], requires_grad=True, device=self.device)
