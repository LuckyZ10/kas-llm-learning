"""
GFlowNet Implementation for Molecular and Crystal Generation
===========================================================

基于流网络的生成模型，用于:
- 分子生成
- 晶体结构生成
- 材料设计

参考:
- Bengio et al. "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation" (2021)
- Malkin et al. "Trajectory Balance: Improved Credit Assignment in GFlowNets" (2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math


@dataclass
class GFlowNetState:
    """GFlowNet状态表示"""
    state_id: int
    features: torch.Tensor
    parent_actions: List[int]
    is_terminal: bool = False
    reward: float = 0.0
    
    def __hash__(self):
        return hash(self.state_id)
    
    def __eq__(self, other):
        if isinstance(other, GFlowNetState):
            return self.state_id == other.state_id
        return False


@dataclass
class GFlowNetConfig:
    """GFlowNet配置"""
    state_dim: int = 128
    action_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-4
    reward_temperature: float = 1.0
    entropy_coef: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FlowNetwork(nn.Module):
    """流网络基础类"""
    
    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        self.config = config
        
        # 状态编码器
        layers = []
        in_dim = config.state_dim
        for i in range(config.num_layers):
            out_dim = config.hidden_dim if i < config.num_layers - 1 else config.action_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < config.num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))
            in_dim = out_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 前向流 (Forward Flow)
        self.forward_flow = nn.Linear(config.action_dim, 1)
        
        # 后向流 (Backward Flow) - 用于 detailed balance
        self.backward_flow = nn.Linear(config.action_dim, 1)
    
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """编码状态"""
        return self.backbone(state)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回前向流和后向流
        
        Returns:
            forward_flow: [batch_size, 1]
            backward_flow: [batch_size, 1]
        """
        encoded = self.encode(state)
        return self.forward_flow(encoded), self.backward_flow(encoded)


class PolicyNetwork(nn.Module):
    """策略网络 - 用于采样动作"""
    
    def __init__(self, config: GFlowNetConfig):
        super().__init__()
        self.config = config
        
        self.policy_head = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim)
        )
    
    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        返回动作logits
        
        Args:
            state: [batch_size, state_dim]
            mask: [batch_size, action_dim] - 有效动作掩码
            
        Returns:
            logits: [batch_size, action_dim]
        """
        logits = self.policy_head(state)
        
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float('-inf'))
        
        return logits
    
    def sample_action(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[int, float]:
        """采样动作"""
        with torch.no_grad():
            logits = self.forward(state, mask)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()


class GFlowNet(nn.Module):
    """
    GFlowNet主类
    
    通过训练流网络来学习从源到汇的流分布，
    使得采样概率与奖励成正比。
    """
    
    def __init__(self, config: Optional[GFlowNetConfig] = None):
        super().__init__()
        self.config = config or GFlowNetConfig()
        
        self.flow_network = FlowNetwork(self.config)
        self.policy_network = PolicyNetwork(self.config)
        
        self.device = torch.device(self.config.device)
        self.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate
        )
        
        # 统计信息
        self.training_stats = {
            'loss': [],
            'mean_reward': [],
            'diversity': [],
        }
    
    def sample_trajectory(
        self,
        env,
        max_steps: int = 100,
        epsilon: float = 0.1
    ) -> List[Tuple[Any, int, float]]:
        """
        从环境中采样一条轨迹
        
        Args:
            env: 环境实例
            max_steps: 最大步数
            epsilon: 探索率
            
        Returns:
            trajectory: [(state, action, log_prob), ...]
        """
        trajectory = []
        state = env.reset()
        
        for step in range(max_steps):
            # 获取有效动作掩码
            valid_actions = env.get_valid_actions()
            mask = torch.zeros(self.config.action_dim, device=self.device)
            mask[valid_actions] = 1
            
            # epsilon-贪心探索
            if np.random.random() < epsilon:
                action = np.random.choice(valid_actions)
                log_prob = math.log(1.0 / len(valid_actions))
            else:
                state_tensor = torch.FloatTensor(state).to(self.device)
                action, log_prob = self.policy_network.sample_action(state_tensor, mask)
            
            trajectory.append((state, action, log_prob))
            
            state, reward, done, info = env.step(action)
            
            if done:
                break
        
        return trajectory
    
    def trajectory_balance_loss(
        self,
        trajectories: List[List[Tuple[Any, int, float]]],
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        轨迹平衡损失 (Trajectory Balance Loss)
        
        这是GFlowNet中最有效的训练目标之一。
        
        Args:
            trajectories: 轨迹列表
            rewards: 每条轨迹的最终奖励
            
        Returns:
            loss: 标量损失
        """
        total_loss = 0.0
        
        for traj, reward in zip(trajectories, rewards):
            # 计算前向流
            forward_flows = []
            backward_flows = []
            log_probs = []
            
            for state, action, log_prob in traj:
                state_tensor = torch.FloatTensor(state).to(self.device)
                f_flow, b_flow = self.flow_network(state_tensor)
                
                forward_flows.append(f_flow.squeeze())
                backward_flows.append(b_flow.squeeze())
                log_probs.append(log_prob)
            
            # 轨迹平衡: Z * prod(P_F) = R * prod(P_B)
            # 取log: logZ + sum(log P_F) = log R + sum(log P_B)
            log_Z = forward_flows[0]  # 初始状态的流
            log_P_F = torch.stack([torch.tensor(lp) for lp in log_probs]).sum()
            log_P_B = torch.stack([f for f in backward_flows[1:]]).sum() if len(backward_flows) > 1 else torch.tensor(0.0)
            log_R = torch.log(reward + 1e-10)
            
            # 平衡损失
            loss = (log_Z + log_P_F - log_R - log_P_B) ** 2
            total_loss += loss
        
        return total_loss / len(trajectories)
    
    def flow_matching_loss(
        self,
        states: List[GFlowNetState],
        next_states: List[GFlowNetState],
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        流匹配损失 (Flow Matching Loss)
        
        匹配流入和流出每个状态的流。
        
        Args:
            states: 当前状态
            next_states: 下一状态
            rewards: 奖励
            
        Returns:
            loss: 标量损失
        """
        losses = []
        
        for state, next_state, reward in zip(states, next_states, rewards):
            state_tensor = torch.FloatTensor(state.features).to(self.device)
            next_tensor = torch.FloatTensor(next_state.features).to(self.device)
            
            f_flow, _ = self.flow_network(state_tensor)
            next_f_flow, _ = self.flow_network(next_tensor)
            
            # 流入 = 流出 (对于非终止状态)
            if not next_state.is_terminal:
                flow_conservation = (f_flow - next_f_flow) ** 2
            else:
                # 终止状态: 流出 = 奖励
                flow_conservation = (f_flow - reward) ** 2
            
            losses.append(flow_conservation)
        
        return torch.stack(losses).mean()
    
    def train_step(
        self,
        trajectories: List[List[Tuple[Any, int, float]]],
        rewards: torch.Tensor,
        loss_type: str = "trajectory_balance"
    ) -> Dict[str, float]:
        """
        训练步骤
        
        Args:
            trajectories: 轨迹列表
            rewards: 奖励张量
            loss_type: 损失类型 ("trajectory_balance" 或 "flow_matching")
            
        Returns:
            stats: 训练统计
        """
        self.train()
        self.optimizer.zero_grad()
        
        if loss_type == "trajectory_balance":
            loss = self.trajectory_balance_loss(trajectories, rewards)
        elif loss_type == "flow_matching":
            # 从轨迹提取状态转换
            states = []
            next_states = []
            for traj in trajectories:
                for i in range(len(traj) - 1):
                    states.append(traj[i][0])
                    next_states.append(traj[i+1][0])
            loss = self.flow_matching_loss(states, next_states, rewards)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        stats = {
            'loss': loss.item(),
            'mean_reward': rewards.mean().item(),
        }
        
        # 更新统计
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def generate_samples(
        self,
        env,
        num_samples: int = 100,
        max_steps: int = 100
    ) -> List[Dict[str, Any]]:
        """
        生成样本
        
        Args:
            env: 环境实例
            num_samples: 样本数量
            max_steps: 最大步数
            
        Returns:
            samples: 生成的样本列表
        """
        self.eval()
        samples = []
        
        for _ in range(num_samples):
            trajectory = self.sample_trajectory(env, max_steps, epsilon=0.0)
            
            # 获取最终状态和奖励
            final_state = trajectory[-1][0] if trajectory else None
            
            # 从环境获取最终样本
            sample = env.get_sample()
            
            samples.append({
                'trajectory': trajectory,
                'final_state': final_state,
                'sample': sample,
            })
        
        return samples


class TrajectoryBalance:
    """
    轨迹平衡训练器
    
    实现 Malkin et al. 的 Trajectory Balance 算法
    """
    
    def __init__(
        self,
        gflownet: GFlowNet,
        learning_rate: float = 1e-4,
        scheduler_gamma: float = 0.95
    ):
        self.gflownet = gflownet
        self.optimizer = torch.optim.Adam(
            gflownet.parameters(),
            lr=learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=scheduler_gamma
        )
    
    def compute_tb_loss(
        self,
        trajectories: List[List[Tuple]],
        rewards: torch.Tensor,
        log_Z: torch.Tensor
    ) -> torch.Tensor:
        """计算轨迹平衡损失"""
        losses = []
        
        for traj, reward in zip(trajectories, rewards):
            # 前向概率
            log_p_forward = sum([log_prob for _, _, log_prob in traj])
            
            # 后向概率 (假设均匀分布)
            log_p_backward = -math.log(len(traj)) * (len(traj) - 1)
            
            # TB损失
            log_reward = torch.log(reward + 1e-10)
            loss = (log_Z + log_p_forward - log_reward - log_p_backward) ** 2
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def train_step(
        self,
        trajectories: List[List[Tuple]],
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """训练步骤"""
        self.optimizer.zero_grad()
        
        # 学习 Z (配分函数)
        log_Z = torch.zeros(1, requires_grad=True, device=self.gflownet.device)
        
        loss = self.compute_tb_loss(trajectories, rewards, log_Z)
        
        loss.backward()
        self.optimizer.step()
        
        return {
            'tb_loss': loss.item(),
            'log_Z': log_Z.item(),
        }
    
    def step_scheduler(self):
        """更新学习率"""
        self.scheduler.step()


class FlowMatchingGFlowNet(GFlowNet):
    """
    流匹配GFlowNet
    
    使用连续时间流匹配进行训练
    """
    
    def __init__(self, config: Optional[GFlowNetConfig] = None):
        super().__init__(config)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )
    
    def forward_with_time(
        self,
        state: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """带时间的前向传播"""
        t_emb = self.time_embed(time.unsqueeze(-1))
        encoded = self.flow_network.encode(state)
        return encoded + t_emb
    
    def compute_flow_matching_loss(
        self,
        states: torch.Tensor,
        times: torch.Tensor,
        target_flows: torch.Tensor
    ) -> torch.Tensor:
        """计算流匹配损失"""
        predicted_flows = self.forward_with_time(states, times)
        loss = F.mse_loss(predicted_flows, target_flows)
        return loss


class MoleculeGFlowNet(GFlowNet):
    """
    分子生成GFlowNet
    
    专门用于分子图的生成
    """
    
    def __init__(
        self,
        atom_types: List[str],
        bond_types: List[str],
        max_atoms: int = 50,
        config: Optional[GFlowNetConfig] = None
    ):
        # 更新配置
        config = config or GFlowNetConfig()
        config.action_dim = len(atom_types) + len(bond_types) + 2  # 添加原子、添加键、终止
        
        super().__init__(config)
        
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_atoms = max_atoms
        
        # 图神经网络编码器
        from dftlammps.gnn_advanced.models.advanced_gnn import GraphTransformer
        self.graph_encoder = GraphTransformer(
            node_dim=len(atom_types),
            edge_dim=len(bond_types),
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        )
    
    def encode_molecule(self, atom_features, bond_features, edge_index):
        """编码分子图"""
        return self.graph_encoder(atom_features, bond_features, edge_index)


class CrystalGFlowNet(GFlowNet):
    """
    晶体结构生成GFlowNet
    
    用于生成晶体结构和组成
    """
    
    def __init__(
        self,
        element_set: List[str],
        space_groups: List[int],
        max_atoms: int = 100,
        config: Optional[GFlowNetConfig] = None
    ):
        config = config or GFlowNetConfig()
        # 动作: 选择元素、选择空间群、设置晶格参数、设置位置
        config.action_dim = len(element_set) + len(space_groups) + 12
        
        super().__init__(config)
        
        self.element_set = element_set
        self.space_groups = space_groups
        self.max_atoms = max_atoms
        
        # 晶体专用编码器
        self.crystal_encoder = nn.Sequential(
            nn.Linear(config.state_dim + 6 + 3, config.hidden_dim),  # 组成 + 晶格 + 位置
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
        )
    
    def encode_crystal(
        self,
        composition: torch.Tensor,
        lattice: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """编码晶体结构"""
        features = torch.cat([composition, lattice, positions], dim=-1)
        return self.crystal_encoder(features)


def demo():
    """演示GFlowNet"""
    print("=" * 60)
    print("GFlowNet Demo")
    print("=" * 60)
    
    # 创建配置
    config = GFlowNetConfig(
        state_dim=32,
        action_dim=16,
        hidden_dim=64,
        num_layers=2
    )
    
    # 创建GFlowNet
    gfn = GFlowNet(config)
    print(f"GFlowNet created with {sum(p.numel() for p in gfn.parameters())} parameters")
    
    # 创建简单的网格环境
    class SimpleGridEnv:
        """简单网格环境用于测试"""
        
        def __init__(self, size=5):
            self.size = size
            self.state = None
            self.position = None
        
        def reset(self):
            self.position = [0, 0]
            self.state = self._get_state()
            return self.state
        
        def _get_state(self):
            state = np.zeros(self.size * self.size)
            state[self.position[0] * self.size + self.position[1]] = 1
            return state
        
        def get_valid_actions(self):
            actions = []
            if self.position[0] > 0:
                actions.append(0)  # 上
            if self.position[0] < self.size - 1:
                actions.append(1)  # 下
            if self.position[1] > 0:
                actions.append(2)  # 左
            if self.position[1] < self.size - 1:
                actions.append(3)  # 右
            actions.append(4)  # 终止
            return actions
        
        def step(self, action):
            if action == 0:
                self.position[0] -= 1
            elif action == 1:
                self.position[0] += 1
            elif action == 2:
                self.position[1] -= 1
            elif action == 3:
                self.position[1] += 1
            
            self.state = self._get_state()
            
            # 奖励基于到目标的距离
            target = [self.size - 1, self.size - 1]
            dist = abs(self.position[0] - target[0]) + abs(self.position[1] - target[1])
            reward = np.exp(-dist / self.size)
            
            done = (action == 4) or (self.position == target)
            
            return self.state, reward, done, {}
        
        def get_sample(self):
            return {'position': self.position.copy()}
    
    env = SimpleGridEnv(size=5)
    
    # 采样轨迹
    print("\nSampling trajectories...")
    trajectories = []
    rewards = []
    
    for i in range(10):
        traj = gfn.sample_trajectory(env, max_steps=20, epsilon=0.2)
        trajectories.append(traj)
        
        # 计算奖励
        final_state = env.reset()
        for state, action, _ in traj:
            final_state, reward, done, _ = env.step(action)
            if done:
                break
        rewards.append(reward)
    
    rewards_tensor = torch.FloatTensor(rewards).to(config.device)
    
    print(f"Sampled {len(trajectories)} trajectories")
    print(f"Mean reward: {rewards_tensor.mean().item():.4f}")
    
    # 训练步骤
    print("\nTraining...")
    for epoch in range(5):
        stats = gfn.train_step(trajectories, rewards_tensor)
        print(f"Epoch {epoch + 1}: loss={stats['loss']:.4f}, mean_reward={stats['mean_reward']:.4f}")
    
    # 生成样本
    print("\nGenerating samples...")
    samples = gfn.generate_samples(env, num_samples=5)
    print(f"Generated {len(samples)} samples")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
