"""
Offline RL Algorithms
=====================

实现离线强化学习算法:
- CQL (Conservative Q-Learning)
- IQL (Implicit Q-Learning)
- Decision Transformer
- Trajectory Transformer

用于材料设计中的离线策略学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class OfflineRLConfig:
    """离线RL配置"""
    state_dim: int = 128
    action_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2  # SAC temperature or CQL weight
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CQL-specific
    cql_alpha: float = 1.0
    cql_temp: float = 1.0
    
    # IQL-specific
    iql_tau: float = 0.7
    iql_beta: float = 3.0
    
    # DT-specific
    dt_max_len: int = 20
    dt_n_heads: int = 8
    dt_n_layers: int = 6


class QNetwork(nn.Module):
    """Q网络 (双Q网络)"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Q1
        layers1 = []
        in_dim = state_dim + action_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers1.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers1.append(nn.LayerNorm(out_dim))
                layers1.append(nn.ReLU())
            in_dim = out_dim
        
        self.q1 = nn.Sequential(*layers1)
        
        # Q2
        layers2 = []
        in_dim = state_dim + action_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers2.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers2.append(nn.LayerNorm(out_dim))
                layers2.append(nn.ReLU())
            in_dim = out_dim
        
        self.q2 = nn.Sequential(*layers2)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """返回最小Q值"""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
            in_dim = out_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(state)


class CQL(nn.Module):
    """
    Conservative Q-Learning (CQL)
    
    Kumar et al. "Conservative Q-Learning for Offline Reinforcement Learning" (2020)
    
    通过添加保守性项来防止离线RL中的值函数过估计
    """
    
    def __init__(self, config: OfflineRLConfig):
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Q网络
        self.q_network = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
            config.num_layers
        ).to(self.device)
        
        # 目标Q网络
        self.q_target = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
            config.num_layers
        ).to(self.device)
        
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # 策略网络
        from .policy import GaussianPolicy
        self.policy = GaussianPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            device=config.device
        ).to(self.device)
        
        # 优化器
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # 自动调整alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        self.target_entropy = -config.action_dim
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action, _ = self.policy.select_action(state_tensor, deterministic)
            return action
    
    def cql_loss(
        self,
        q_values: torch.Tensor,
        dataset_q_values: torch.Tensor
    ) -> torch.Tensor:
        """
        CQL保守性损失
        
        鼓励Q网络对数据集中的动作给出高值，对采样的动作给出低值
        """
        # 数据集中的动作的Q值
        dataset_q = dataset_q_values.mean()
        
        # 采样动作的Q值 (使用logsumexp)
        sampled_q = torch.logsumexp(q_values / self.config.cql_temp, dim=0).mean()
        sampled_q = sampled_q * self.config.cql_temp
        
        # CQL损失
        cql_loss = sampled_q - dataset_q
        
        return cql_loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        训练步骤
        
        Args:
            batch: 包含 state, action, reward, next_state, done
        """
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        
        # ===== 更新Q网络 =====
        with torch.no_grad():
            # 使用策略采样下一动作
            next_action, next_log_prob = self.policy.select_action(next_state)
            
            # 目标Q值
            target_q1, target_q2 = self.q_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.config.gamma * (
                target_q - self.log_alpha.exp() * next_log_prob.sum(dim=-1, keepdim=True)
            )
        
        # 当前Q值
        current_q1, current_q2 = self.q_network(state, action)
        
        # Bellman损失
        bellman_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # CQL保守性损失
        # 随机采样动作
        num_random = 10
        random_actions = torch.FloatTensor(
            state.shape[0], num_random, self.config.action_dim
        ).uniform_(-1, 1).to(self.device)
        
        # 扩展状态
        state_expanded = state.unsqueeze(1).repeat(1, num_random, 1)
        state_flat = state_expanded.view(-1, self.config.state_dim)
        actions_flat = random_actions.view(-1, self.config.action_dim)
        
        # 随机动作的Q值
        random_q1, random_q2 = self.q_network(state_flat, actions_flat)
        random_q1 = random_q1.view(state.shape[0], num_random)
        random_q2 = random_q2.view(state.shape[0], num_random)
        
        # 当前策略动作的Q值
        current_action, _ = self.policy.select_action(state)
        current_policy_q1, current_policy_q2 = self.q_network(state, current_action)
        
        # CQL损失
        cql_loss_value = self.cql_loss(random_q1, current_q1) + self.cql_loss(random_q2, current_q2)
        
        # 总Q损失
        q_loss = bellman_loss + self.config.cql_alpha * cql_loss_value
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # ===== 更新策略 =====
        new_action, log_prob = self.policy.select_action(state)
        q1_new, q2_new = self.q_network(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.log_alpha.exp().detach() * log_prob.sum(dim=-1, keepdim=True) - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # ===== 更新Alpha =====
        alpha_loss = -(self.log_alpha * (log_prob.sum(dim=-1) + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ===== 软更新目标网络 =====
        for param, target_param in zip(
            self.q_network.parameters(),
            self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'cql_loss': cql_loss_value.item(),
            'alpha': self.log_alpha.exp().item(),
            'mean_q': current_q1.mean().item(),
        }


class IQL(nn.Module):
    """
    Implicit Q-Learning (IQL)
    
    Kostrikov et al. "Offline Reinforcement Learning with Implicit Q-Learning" (2021)
    
    不需要显式的策略提取，使用期望回归避免值函数过估计
    """
    
    def __init__(self, config: OfflineRLConfig):
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # Q网络
        self.q_network = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
            config.num_layers
        ).to(self.device)
        
        # 价值网络 V(s)
        self.v_network = ValueNetwork(
            config.state_dim,
            config.hidden_dim,
            config.num_layers
        ).to(self.device)
        
        # 策略网络
        from .policy import GaussianPolicy
        self.policy = GaussianPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            device=config.device
        ).to(self.device)
        
        # 优化器
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        self.v_optimizer = torch.optim.Adam(
            self.v_network.parameters(),
            lr=config.learning_rate
        )
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
    
    def expectile_loss(
        self,
        diff: torch.Tensor,
        expectile: float
    ) -> torch.Tensor:
        """
        Expectile损失
        
        不对称L2损失，用于学习分位数
        """
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """训练步骤"""
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        
        # ===== 更新V网络 =====
        with torch.no_grad():
            q1, q2 = self.q_network(state, action)
            q = torch.min(q1, q2)
        
        v = self.v_network(state)
        v_loss = self.expectile_loss(q - v, self.config.iql_tau).mean()
        
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        # ===== 更新Q网络 =====
        with torch.no_grad():
            next_v = self.v_network(next_state)
            target_q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.config.gamma * next_v
        
        q1, q2 = self.q_network(state, action)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # ===== 更新策略 =====
        with torch.no_grad():
            q1, q2 = self.q_network(state, action)
            q = torch.min(q1, q2)
            v = self.v_network(state)
            advantage = q - v
            exp_advantage = torch.exp(advantage * self.config.iql_beta)
            exp_advantage = torch.clamp(exp_advantage, max=100.0)
        
        action_pred, _ = self.policy.select_action(state)
        log_prob, _ = self.policy.evaluate(state, action)
        
        policy_loss = -(log_prob * exp_advantage).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'q_loss': q_loss.item(),
            'v_loss': v_loss.item(),
            'policy_loss': policy_loss.item(),
            'mean_advantage': advantage.mean().item(),
        }


class DecisionTransformer(nn.Module):
    """
    Decision Transformer
    
    Chen et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)
    
    将RL视为序列建模问题，使用Transformer预测动作
    """
    
    def __init__(self, config: OfflineRLConfig):
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        
        # 嵌入层
        self.rtg_embed = nn.Linear(1, config.hidden_dim)
        self.state_embed = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_embed = nn.Linear(config.action_dim, config.hidden_dim)
        
        # 时间嵌入
        self.timestep_embed = nn.Embedding(config.dt_max_len, config.hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.dt_n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.dt_n_layers
        )
        
        # 预测头
        self.action_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
            nn.Tanh()
        )
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate
        )
    
    def forward(
        self,
        rtgs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            rtgs: [batch, seq_len, 1] - 回报到go
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len, action_dim]
            timesteps: [batch, seq_len]
            attention_mask: [batch, seq_len]
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # 嵌入
        rtg_embeddings = self.rtg_embed(rtgs)
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        time_embeddings = self.timestep_embed(timesteps)
        
        # 添加时间嵌入
        rtg_embeddings = rtg_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        
        # 交错堆叠: (R, s, a, R, s, a, ...)
        stacked_inputs = torch.stack(
            [rtg_embeddings, state_embeddings, action_embeddings],
            dim=2
        ).view(batch_size, 3 * seq_len, self.config.hidden_dim)
        
        # 因果掩码
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=self.device)
        
        # 扩展掩码
        stacked_attention_mask = torch.stack(
            [attention_mask, attention_mask, attention_mask],
            dim=2
        ).view(batch_size, 3 * seq_len)
        
        # Transformer
        hidden = self.transformer(
            stacked_inputs,
            src_key_padding_mask=~stacked_attention_mask.bool()
        )
        
        # 提取状态对应的隐藏状态
        hidden = hidden[:, 1::3, :]  # 取s的位置
        
        # 预测动作
        action_preds = self.action_predictor(hidden)
        
        return action_preds
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """训练步骤"""
        rtgs = batch['rtgs'].to(self.device)
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        timesteps = batch['timesteps'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # 预测动作
        action_preds = self.forward(rtgs, states, actions, timesteps, attention_mask)
        
        # 动作预测损失
        act_dim = action_preds.shape[2]
        action_preds = action_preds.view(-1, act_dim)
        actions_target = actions.view(-1, act_dim)
        
        if attention_mask is not None:
            mask = attention_mask.view(-1, 1).expand(-1, act_dim)
            loss = F.mse_loss(action_preds * mask, actions_target * mask)
        else:
            loss = F.mse_loss(action_preds, actions_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
        }
    
    def get_action(
        self,
        rtg: float,
        state: np.ndarray,
        actions: List[np.ndarray],
        timesteps: List[int]
    ) -> np.ndarray:
        """获取动作"""
        with torch.no_grad():
            # 准备输入
            rtg_tensor = torch.FloatTensor([[rtg]]).to(self.device)
            state_tensor = torch.FloatTensor([state]).unsqueeze(0).to(self.device)
            
            # 填充历史动作
            action_history = actions + [np.zeros(self.config.action_dim)]
            action_tensor = torch.FloatTensor([action_history]).to(self.device)
            
            timestep_tensor = torch.LongTensor([timesteps + [timesteps[-1] + 1]]).to(self.device)
            
            # 预测
            action_pred = self.forward(
                rtg_tensor.unsqueeze(-1),
                state_tensor,
                action_tensor,
                timestep_tensor
            )
            
            return action_pred[0, -1].cpu().numpy()


class TrajectoryTransformer(DecisionTransformer):
    """
    Trajectory Transformer
    
    扩展Decision Transformer，同时预测状态、动作和奖励
    """
    
    def __init__(self, config: OfflineRLConfig):
        super().__init__(config)
        
        # 额外的预测头
        self.state_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.state_dim)
        )
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
    
    def forward(
        self,
        rtgs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # 嵌入
        rtg_embeddings = self.rtg_embed(rtgs)
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        time_embeddings = self.timestep_embed(timesteps)
        
        rtg_embeddings = rtg_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        
        # 交错堆叠
        stacked_inputs = torch.stack(
            [rtg_embeddings, state_embeddings, action_embeddings],
            dim=2
        ).view(batch_size, 3 * seq_len, self.config.hidden_dim)
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=self.device)
        
        stacked_attention_mask = torch.stack(
            [attention_mask, attention_mask, attention_mask],
            dim=2
        ).view(batch_size, 3 * seq_len)
        
        # Transformer
        hidden = self.transformer(
            stacked_inputs,
            src_key_padding_mask=~stacked_attention_mask.bool()
        )
        
        # 重塑为 [batch, seq_len, 3, hidden_dim]
        hidden = hidden.view(batch_size, seq_len, 3, self.config.hidden_dim)
        
        # 预测
        state_preds = self.state_predictor(hidden[:, :, 0, :])
        action_preds = self.action_predictor(hidden[:, :, 1, :])
        reward_preds = self.reward_predictor(hidden[:, :, 2, :])
        
        return state_preds, action_preds, reward_preds
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """训练步骤"""
        rtgs = batch['rtgs'].to(self.device)
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        timesteps = batch['timesteps'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # 预测
        state_preds, action_preds, reward_preds = self.forward(
            rtgs, states, actions, timesteps, attention_mask
        )
        
        # 损失
        act_dim = action_preds.shape[2]
        
        if attention_mask is not None:
            mask = attention_mask.view(-1, 1)
            
            action_loss = F.mse_loss(
                action_preds.view(-1, act_dim) * mask,
                actions.view(-1, act_dim) * mask
            )
            
            state_loss = F.mse_loss(
                state_preds.view(-1, self.config.state_dim) * mask,
                states.view(-1, self.config.state_dim) * mask
            )
            
            reward_loss = F.mse_loss(
                reward_preds.view(-1, 1) * mask,
                rewards.view(-1, 1) * mask
            )
        else:
            action_loss = F.mse_loss(action_preds, actions)
            state_loss = F.mse_loss(state_preds, states)
            reward_loss = F.mse_loss(reward_preds, rewards)
        
        loss = action_loss + 0.5 * state_loss + 0.5 * reward_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'action_loss': action_loss.item(),
            'state_loss': state_loss.item(),
            'reward_loss': reward_loss.item(),
        }


class OfflineRLAgent:
    """
    离线RL代理包装器
    """
    
    def __init__(
        self,
        algorithm: str = "cql",
        **kwargs
    ):
        config = OfflineRLConfig(**kwargs)
        
        if algorithm.lower() == "cql":
            self.agent = CQL(config)
        elif algorithm.lower() == "iql":
            self.agent = IQL(config)
        elif algorithm.lower() == "decision_transformer":
            self.agent = DecisionTransformer(config)
        elif algorithm.lower() == "trajectory_transformer":
            self.agent = TrajectoryTransformer(config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.algorithm = algorithm
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """训练步骤"""
        return self.agent.train_step(batch)
    
    def select_action(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """选择动作"""
        if hasattr(self.agent, 'select_action'):
            return self.agent.select_action(state, **kwargs)
        elif hasattr(self.agent, 'get_action'):
            return self.agent.get_action(state, **kwargs)
        else:
            raise NotImplementedError


def demo():
    """演示离线RL"""
    print("=" * 60)
    print("Offline RL Demo")
    print("=" * 60)
    
    config = OfflineRLConfig(
        state_dim=10,
        action_dim=4,
        hidden_dim=64,
        num_layers=2
    )
    
    # 1. CQL
    print("\n1. CQL (Conservative Q-Learning)")
    cql = CQL(config)
    
    # 模拟批次数据
    batch_size = 32
    batch = {
        'state': torch.randn(batch_size, config.state_dim),
        'action': torch.randn(batch_size, config.action_dim),
        'reward': torch.randn(batch_size),
        'next_state': torch.randn(batch_size, config.state_dim),
        'done': torch.zeros(batch_size),
    }
    
    stats = cql.train_step(batch)
    print(f"   Training stats: {stats}")
    
    # 2. IQL
    print("\n2. IQL (Implicit Q-Learning)")
    iql = IQL(config)
    stats = iql.train_step(batch)
    print(f"   Training stats: {stats}")
    
    # 3. Decision Transformer
    print("\n3. Decision Transformer")
    dt = DecisionTransformer(config)
    
    seq_len = 10
    batch_dt = {
        'rtgs': torch.randn(batch_size, seq_len, 1),
        'states': torch.randn(batch_size, seq_len, config.state_dim),
        'actions': torch.randn(batch_size, seq_len, config.action_dim),
        'timesteps': torch.randint(0, 100, (batch_size, seq_len)),
    }
    
    stats = dt.train_step(batch_dt)
    print(f"   Training stats: {stats}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
