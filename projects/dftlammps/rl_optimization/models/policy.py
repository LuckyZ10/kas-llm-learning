"""
Policy Networks for RL
======================

实现各种策略网络:
- 随机策略 (Stochastic Policy)
- 确定性策略 (Deterministic Policy)
- 离散策略 (Categorical Policy)
- 高斯策略 (Gaussian Policy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class PolicyConfig:
    """策略网络配置"""
    state_dim: int = 128
    action_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "relu"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BasePolicy(nn.Module):
    """策略网络基类"""
    
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # 构建骨干网络
        layers = []
        in_dim = config.state_dim
        
        for i in range(config.num_layers):
            out_dim = config.hidden_dim if i < config.num_layers - 1 else config.action_dim
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < config.num_layers - 1:
                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "silu":
                    layers.append(nn.SiLU())
                elif config.activation == "gelu":
                    layers.append(nn.GELU())
                
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.Dropout(config.dropout))
            
            in_dim = out_dim
        
        self.backbone = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.backbone(state)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作 (numpy接口)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action = self.select_action_tensor(state_tensor, deterministic)
            return action.cpu().numpy()
    
    def select_action_tensor(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """选择动作 (tensor接口)"""
        raise NotImplementedError


class StochasticPolicy(BasePolicy):
    """
    随机策略网络
    
    输出动作分布的参数，从中采样动作
    """
    
    def __init__(
        self,
        config: PolicyConfig,
        action_space_type: str = "continuous"
    ):
        super().__init__(config)
        self.action_space_type = action_space_type
        
        if action_space_type == "continuous":
            # 连续动作空间: 输出均值和对数标准差
            self.mean_layer = nn.Linear(config.action_dim, config.action_dim)
            self.log_std_layer = nn.Linear(config.action_dim, config.action_dim)
        else:
            # 离散动作空间: 输出logits
            self.logits_layer = nn.Linear(config.action_dim, config.action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            连续: (mean, log_std)
            离散: (logits, None)
        """
        features = self.backbone(state)
        
        if self.action_space_type == "continuous":
            mean = self.mean_layer(features)
            log_std = self.log_std_layer(features)
            log_std = torch.clamp(log_std, -20, 2)  # 限制范围
            return mean, log_std
        else:
            logits = self.logits_layer(features)
            return logits, None
    
    def select_action_tensor(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """选择动作"""
        if self.action_space_type == "continuous":
            mean, log_std = self.forward(state)
            
            if deterministic:
                return torch.tanh(mean)
            
            std = torch.exp(log_std)
            noise = torch.randn_like(mean)
            action = mean + std * noise
            return torch.tanh(action)
        else:
            logits, _ = self.forward(state)
            
            if deterministic:
                return torch.argmax(logits, dim=-1)
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            return dist.sample()
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作的对数概率和分布熵
        
        Returns:
            log_prob: 对数概率
            entropy: 熵
            value: 状态值
        """
        if self.action_space_type == "continuous":
            mean, log_std = self.forward(state)
            std = torch.exp(log_std)
            
            # 计算对数概率
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            
            return log_prob, entropy, torch.zeros_like(log_prob)
        else:
            logits, _ = self.forward(state)
            
            dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
            log_prob = dist.log_prob(action).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
            
            return log_prob, entropy, torch.zeros_like(log_prob)


class DeterministicPolicy(BasePolicy):
    """
    确定性策略网络 (DDPG/TD3风格)
    
    直接输出确定的动作
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.action_dim = config.action_dim
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        action = self.backbone(state)
        return torch.tanh(action)  # 输出范围 [-1, 1]
    
    def select_action_tensor(
        self,
        state: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        """选择动作"""
        action = self.forward(state)
        
        if not deterministic:
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -1, 1)
        
        return action


class CategoricalPolicy(BasePolicy):
    """
    离散分类策略
    
    用于离散动作空间
    """
    
    def __init__(
        self,
        config: PolicyConfig,
        num_actions: int = None
    ):
        if num_actions is not None:
            config.action_dim = num_actions
        super().__init__(config)
        self.num_actions = config.action_dim
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播 - 返回logits"""
        logits = self.backbone(state)
        return logits
    
    def select_action_tensor(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """选择动作"""
        logits = self.forward(state)
        
        if deterministic:
            return torch.argmax(logits, dim=-1)
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """获取动作的对数概率"""
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, action.unsqueeze(-1))
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """获取策略熵"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        return entropy


class GaussianPolicy(nn.Module):
    """
    高斯策略网络
    
    用于连续动作空间的高斯分布
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        log_std_min: float = -20,
        log_std_max: float = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = torch.device(device)
        
        # 构建网络
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else action_dim
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
            
            in_dim = out_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 对数标准差
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        mean = self.backbone(state)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            mean, log_std = self.forward(state_tensor)
            
            if deterministic:
                action = mean
            else:
                std = torch.exp(log_std)
                noise = torch.randn_like(mean)
                action = mean + std * noise
            
            return action.cpu().numpy(), mean.cpu().numpy()
    
    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作
        
        Returns:
            log_prob: 对数概率
            entropy: 熵
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 高斯分布的对数概率
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic策略
    
    同时包含策略(Actor)和价值(Critic)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        discrete: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = torch.device(device)
        
        # 共享特征提取器
        shared_layers = []
        in_dim = state_dim
        
        for i in range(num_layers - 1):
            shared_layers.append(nn.Linear(in_dim, hidden_dim))
            shared_layers.append(nn.LayerNorm(hidden_dim))
            shared_layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Actor (策略)
        if discrete:
            self.actor = nn.Linear(hidden_dim, action_dim)
        else:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (价值)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.shared(state)
        
        # 动作
        if self.discrete:
            logits = self.actor(features)
            action_probs = F.softmax(logits, dim=-1)
        else:
            mean = self.actor_mean(features)
            log_std = torch.clamp(self.actor_log_std, -20, 2)
            std = torch.exp(log_std)
            action_probs = (mean, std)
        
        # 价值
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作和价值"""
        features = self.shared(state)
        
        if self.discrete:
            logits = self.actor(features)
            dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            mean = self.actor_mean(features)
            log_std = torch.clamp(self.actor_log_std, -20, 2)
            std = torch.exp(log_std)
            
            dist = torch.distributions.Normal(mean, std)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        
        value = self.critic(features).squeeze(-1)
        
        return action, log_prob, entropy, value
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取状态价值"""
        features = self.shared(state)
        return self.critic(features)


class PolicyNetwork:
    """
    策略网络工厂类
    """
    
    @staticmethod
    def create(
        policy_type: str,
        state_dim: int,
        action_dim: int,
        **kwargs
    ) -> nn.Module:
        """
        创建策略网络
        
        Args:
            policy_type: 策略类型 ("stochastic", "deterministic", "categorical", "gaussian", "actor_critic")
            state_dim: 状态维度
            action_dim: 动作维度
            **kwargs: 其他参数
        """
        config = PolicyConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            **{k: v for k, v in kwargs.items() if k in PolicyConfig.__dataclass_fields__}
        )
        
        if policy_type == "stochastic":
            return StochasticPolicy(
                config,
                action_space_type=kwargs.get("action_space_type", "continuous")
            )
        elif policy_type == "deterministic":
            return DeterministicPolicy(config)
        elif policy_type == "categorical":
            return CategoricalPolicy(config, kwargs.get("num_actions", action_dim))
        elif policy_type == "gaussian":
            return GaussianPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=kwargs.get("hidden_dim", 256),
                num_layers=kwargs.get("num_layers", 3),
                device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
        elif policy_type == "actor_critic":
            return ActorCriticPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=kwargs.get("hidden_dim", 256),
                num_layers=kwargs.get("num_layers", 3),
                discrete=kwargs.get("discrete", False),
                device=kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")


def demo():
    """演示策略网络"""
    print("=" * 60)
    print("Policy Networks Demo")
    print("=" * 60)
    
    state_dim = 10
    action_dim = 4
    batch_size = 8
    
    # 创建随机状态
    state = torch.randn(batch_size, state_dim)
    
    # 1. 随机策略 (连续)
    print("\n1. Stochastic Policy (Continuous)")
    stochastic_policy = PolicyNetwork.create(
        "stochastic",
        state_dim=state_dim,
        action_dim=action_dim,
        action_space_type="continuous"
    )
    
    action = stochastic_policy.select_action_tensor(state, deterministic=False)
    print(f"   Action shape: {action.shape}")
    print(f"   Action range: [{action.min():.2f}, {action.max():.2f}]")
    
    # 2. 确定性策略
    print("\n2. Deterministic Policy")
    det_policy = PolicyNetwork.create(
        "deterministic",
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    action = det_policy.select_action_tensor(state)
    print(f"   Action shape: {action.shape}")
    
    # 3. 分类策略
    print("\n3. Categorical Policy")
    cat_policy = PolicyNetwork.create(
        "categorical",
        state_dim=state_dim,
        action_dim=action_dim,
        num_actions=action_dim
    )
    
    action = cat_policy.select_action_tensor(state)
    print(f"   Action shape: {action.shape}")
    print(f"   Sample actions: {action[:5].tolist()}")
    
    # 4. 高斯策略
    print("\n4. Gaussian Policy")
    gauss_policy = PolicyNetwork.create(
        "gaussian",
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    action_np, mean_np = gauss_policy.select_action(state.numpy())
    print(f"   Action shape: {action_np.shape}")
    
    # 5. Actor-Critic
    print("\n5. Actor-Critic Policy")
    ac_policy = PolicyNetwork.create(
        "actor_critic",
        state_dim=state_dim,
        action_dim=action_dim,
        discrete=True
    )
    
    action, log_prob, entropy, value = ac_policy.get_action_and_value(state)
    print(f"   Action shape: {action.shape}")
    print(f"   Log prob shape: {log_prob.shape}")
    print(f"   Value shape: {value.shape}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
