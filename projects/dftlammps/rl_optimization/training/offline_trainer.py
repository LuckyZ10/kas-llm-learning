"""
Offline RL Training
===================

离线强化学习训练器
- 从静态数据集中学习
- CQL/IQL训练
- Decision Transformer训练
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class OfflineRLTrainingConfig:
    """离线RL训练配置"""
    num_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    eval_frequency: int = 10
    save_frequency: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据配置
    max_trajectory_length: int = 100
    normalize_rewards: bool = True
    normalize_states: bool = True
    
    # 算法特定配置
    use_cql: bool = False
    cql_alpha: float = 1.0
    use_iql: bool = False
    iql_tau: float = 0.7


class OfflineDataset:
    """离线数据集"""
    
    def __init__(
        self,
        trajectories: List[List[Tuple]],
        normalize_states: bool = True,
        normalize_rewards: bool = True
    ):
        self.trajectories = trajectories
        self.normalize_states = normalize_states
        self.normalize_rewards = normalize_rewards
        
        # 计算统计信息
        self._compute_statistics()
        
        # 构建经验池
        self.experiences = []
        self._build_experience_pool()
    
    def _compute_statistics(self):
        """计算归一化统计信息"""
        all_states = []
        all_rewards = []
        
        for traj in self.trajectories:
            for exp in traj:
                state = exp[0]
                reward = exp[2] if len(exp) > 2 else 0
                
                if isinstance(state, np.ndarray):
                    all_states.append(state)
                all_rewards.append(reward)
        
        if all_states and self.normalize_states:
            all_states = np.array(all_states)
            self.state_mean = np.mean(all_states, axis=0)
            self.state_std = np.std(all_states, axis=0) + 1e-6
        else:
            self.state_mean = 0
            self.state_std = 1
        
        if all_rewards and self.normalize_rewards:
            self.reward_mean = np.mean(all_rewards)
            self.reward_std = np.std(all_rewards) + 1e-6
        else:
            self.reward_mean = 0
            self.reward_std = 1
    
    def _build_experience_pool(self):
        """构建经验池"""
        for traj in self.trajectories:
            for i, exp in enumerate(traj):
                state = exp[0]
                action = exp[1]
                reward = exp[2] if len(exp) > 2 else 0
                
                # 获取下一状态
                if i < len(traj) - 1:
                    next_state = traj[i + 1][0]
                    done = False
                else:
                    next_state = state
                    done = True
                
                # 归一化
                if isinstance(state, np.ndarray) and self.normalize_states:
                    state = (state - self.state_mean) / self.state_std
                    next_state = (next_state - self.state_mean) / self.state_std
                
                if self.normalize_rewards:
                    reward = (reward - self.reward_mean) / self.reward_std
                
                self.experiences.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                })
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次"""
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        batch = [self.experiences[i] for i in indices]
        
        # 转换为张量
        states = torch.FloatTensor([b['state'] for b in batch])
        actions = torch.FloatTensor([b['action'] for b in batch])
        rewards = torch.FloatTensor([b['reward'] for b in batch])
        next_states = torch.FloatTensor([b['next_state'] for b in batch])
        dones = torch.FloatTensor([b['done'] for b in batch])
        
        return {
            'state': states,
            'action': actions,
            'reward': rewards,
            'next_state': next_states,
            'done': dones,
        }
    
    def sample_trajectories(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样完整轨迹 (用于Decision Transformer)"""
        indices = np.random.choice(len(self.trajectories), batch_size, replace=True)
        
        batch_trajs = [self.trajectories[i] for i in indices]
        
        # 找到最大长度
        max_len = max(len(traj) for traj in batch_trajs)
        max_len = min(max_len, 100)  # 限制最大长度
        
        # 准备序列
        states = []
        actions = []
        rewards = []
        rtgs = []
        timesteps = []
        masks = []
        
        for traj in batch_trajs:
            traj_states = []
            traj_actions = []
            traj_rewards = []
            traj_rtgs = []
            
            # 计算回报到go
            returns = []
            R = 0
            for exp in reversed(traj):
                reward = exp[2] if len(exp) > 2 else 0
                R += reward
                returns.insert(0, R)
            
            # 填充或截断
            for i in range(max_len):
                if i < len(traj):
                    exp = traj[i]
                    traj_states.append(exp[0])
                    traj_actions.append(exp[1])
                    traj_rewards.append(exp[2] if len(exp) > 2 else 0)
                    traj_rtgs.append(returns[i])
                else:
                    # 填充
                    traj_states.append(np.zeros_like(traj[0][0]))
                    traj_actions.append(np.zeros_like(traj[0][1]))
                    traj_rewards.append(0)
                    traj_rtgs.append(0)
            
            states.append(traj_states)
            actions.append(traj_actions)
            rewards.append(traj_rewards)
            rtgs.append(traj_rtgs)
            timesteps.append(list(range(max_len)))
            masks.append([1 if i < len(traj) else 0 for i in range(max_len)])
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'rtgs': torch.FloatTensor(rtgs).unsqueeze(-1),
            'timesteps': torch.LongTensor(timesteps),
            'attention_mask': torch.FloatTensor(masks),
        }
    
    def __len__(self):
        return len(self.experiences)


class OfflineRLTrainer:
    """离线RL训练器"""
    
    def __init__(
        self,
        agent: nn.Module,
        dataset: OfflineDataset,
        config: Optional[OfflineRLTrainingConfig] = None
    ):
        self.agent = agent
        self.dataset = dataset
        self.config = config or OfflineRLTrainingConfig()
        
        self.device = torch.device(self.config.device)
        self.agent.to(self.device)
        
        # 训练统计
        self.training_history = {
            'losses': [],
            'eval_rewards': [],
        }
    
    def train(self) -> Dict[str, List]:
        """训练离线RL代理"""
        print(f"Starting offline RL training for {self.config.num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # 计算每个epoch的步数
            steps_per_epoch = len(self.dataset) // self.config.batch_size
            
            epoch_losses = []
            
            for step in range(steps_per_epoch):
                # 采样批次
                batch = self.dataset.sample(self.config.batch_size)
                
                # 训练步骤
                stats = self.agent.train_step(batch)
                epoch_losses.append(stats.get('loss', stats.get('q_loss', 0)))
            
            # 记录
            mean_loss = np.mean(epoch_losses)
            self.training_history['losses'].append(mean_loss)
            
            # 评估
            if epoch % self.config.eval_frequency == 0:
                print(f"Epoch {epoch}/{self.config.num_epochs} | Loss: {mean_loss:.4f}")
        
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f}s")
        
        return self.training_history
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'agent': self.agent.state_dict(),
            'training_history': self.training_history,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent'])
        self.training_history = checkpoint['training_history']
        print(f"Model loaded from {path}")


class DecisionTransformerTrainer:
    """Decision Transformer训练器"""
    
    def __init__(
        self,
        dt_agent: nn.Module,
        dataset: OfflineDataset,
        config: Optional[OfflineRLTrainingConfig] = None
    ):
        self.dt_agent = dt_agent
        self.dataset = dataset
        self.config = config or OfflineRLTrainingConfig()
        
        self.device = torch.device(self.config.device)
        self.dt_agent.to(self.device)
        
        self.training_history = {'losses': []}
    
    def train(self) -> Dict[str, List]:
        """训练Decision Transformer"""
        print(f"Starting Decision Transformer training")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            steps_per_epoch = len(self.dataset.trajectories) // self.config.batch_size
            
            epoch_losses = []
            
            for step in range(steps_per_epoch):
                # 采样轨迹批次
                batch = self.dataset.sample_trajectories(self.config.batch_size)
                
                # 移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 训练步骤
                stats = self.dt_agent.train_step(batch)
                epoch_losses.append(stats['loss'])
            
            mean_loss = np.mean(epoch_losses)
            self.training_history['losses'].append(mean_loss)
            
            if epoch % self.config.eval_frequency == 0:
                print(f"Epoch {epoch}/{self.config.num_epochs} | Loss: {mean_loss:.4f}")
        
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f}s")
        
        return self.training_history


def demo():
    """演示离线RL训练"""
    print("=" * 60)
    print("Offline RL Trainer Demo")
    print("=" * 60)
    
    # 创建模拟数据集
    num_trajectories = 50
    traj_length = 20
    state_dim = 10
    action_dim = 4
    
    trajectories = []
    for _ in range(num_trajectories):
        traj = []
        for t in range(traj_length):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            traj.append((state, action, reward))
        trajectories.append(traj)
    
    print(f"Created dataset with {num_trajectories} trajectories")
    
    # 创建数据集
    dataset = OfflineDataset(
        trajectories,
        normalize_states=True,
        normalize_rewards=True
    )
    
    print(f"Dataset size: {len(dataset)} experiences")
    
    # 采样测试
    batch = dataset.sample(32)
    print(f"Batch shapes: state={batch['state'].shape}, action={batch['action'].shape}")
    
    # 轨迹采样测试
    traj_batch = dataset.sample_trajectories(4)
    print(f"Trajectory batch shapes: states={traj_batch['states'].shape}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
