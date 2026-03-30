"""
GFlowNet Training
=================

GFlowNet训练器
- 轨迹平衡训练
- 流匹配训练
- 采样和评估
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class GFlowNetTrainingConfig:
    """GFlowNet训练配置"""
    num_iterations: int = 10000
    batch_size: int = 32
    num_samples_per_iter: int = 16
    replay_buffer_size: int = 10000
    learning_rate: float = 1e-4
    gamma_scheduler: float = 0.95
    epsilon_start: float = 0.5
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    reward_temperature_start: float = 1.0
    reward_temperature_end: float = 0.1
    eval_frequency: int = 100
    save_frequency: int = 1000
    max_steps_per_trajectory: int = 50
    loss_type: str = "trajectory_balance"  # "trajectory_balance", "flow_matching"
    use_replay_buffer: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, trajectory: List, reward: float):
        """添加经验"""
        self.buffer.append((trajectory, reward))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """采样批次"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class GFlowNetTrainer:
    """GFlowNet训练器"""
    
    def __init__(
        self,
        gflownet: nn.Module,
        env,
        config: Optional[GFlowNetTrainingConfig] = None
    ):
        self.gflownet = gflownet
        self.env = env
        self.config = config or GFlowNetTrainingConfig()
        
        self.device = torch.device(self.config.device)
        self.gflownet.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.gflownet.parameters(),
            lr=self.config.learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.config.gamma_scheduler
        )
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        
        # 训练状态
        self.iteration = 0
        self.epsilon = self.config.epsilon_start
        self.reward_temperature = self.config.reward_temperature_start
        
        # 统计
        self.training_history = {
            'losses': [],
            'mean_rewards': [],
            'top_k_rewards': [],
            'diversity_scores': [],
        }
        
        # 生成样本缓存
        self.generated_samples = []
    
    def train(self) -> Dict[str, List]:
        """训练GFlowNet"""
        print(f"Starting GFlowNet training for {self.config.num_iterations} iterations")
        
        start_time = time.time()
        
        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            
            # 采样轨迹
            trajectories, rewards = self._sample_trajectories()
            
            # 添加到回放缓冲区
            if self.config.use_replay_buffer:
                for traj, reward in zip(trajectories, rewards):
                    self.replay_buffer.push(traj, reward)
            
            # 准备训练批次
            if self.config.use_replay_buffer and len(self.replay_buffer) >= self.config.batch_size:
                batch = self.replay_buffer.sample(self.config.batch_size)
                train_trajs = [t for t, _ in batch]
                train_rewards = torch.FloatTensor([r for _, r in batch]).to(self.device)
            else:
                train_trajs = trajectories
                train_rewards = torch.FloatTensor(rewards).to(self.device)
            
            # 训练步骤
            stats = self._train_step(train_trajs, train_rewards)
            
            # 记录统计
            self.training_history['losses'].append(stats['loss'])
            self.training_history['mean_rewards'].append(np.mean(rewards))
            
            # 衰减epsilon
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
            
            # 衰减温度
            progress = iteration / self.config.num_iterations
            self.reward_temperature = (
                self.config.reward_temperature_start * (1 - progress) +
                self.config.reward_temperature_end * progress
            )
            
            # 评估
            if iteration % self.config.eval_frequency == 0:
                eval_stats = self.evaluate()
                self.training_history['top_k_rewards'].append(eval_stats['top_k_mean'])
                self.training_history['diversity_scores'].append(eval_stats['diversity'])
                
                print(f"Iter {iteration}/{self.config.num_iterations} | "
                      f"Loss: {stats['loss']:.4f} | "
                      f"Mean Reward: {np.mean(rewards):.4f} | "
                      f"Top-10: {eval_stats['top_k_mean']:.4f} | "
                      f"Diversity: {eval_stats['diversity']:.4f} | "
                      f"Epsilon: {self.epsilon:.3f}")
            
            # 学习率调度
            if iteration % 1000 == 0:
                self.scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f}s")
        
        return self.training_history
    
    def _sample_trajectories(self) -> Tuple[List[List], List[float]]:
        """采样轨迹"""
        trajectories = []
        rewards = []
        
        for _ in range(self.config.num_samples_per_iter):
            traj = self.gflownet.sample_trajectory(
                self.env,
                max_steps=self.config.max_steps_per_trajectory,
                epsilon=self.epsilon
            )
            
            # 计算奖励
            if traj:
                # 从环境获取最终奖励
                final_state = traj[-1][0]
                reward = self._compute_reward(final_state)
                
                # 应用温度
                reward = reward / self.reward_temperature
                
                trajectories.append(traj)
                rewards.append(reward)
        
        return trajectories, rewards
    
    def _compute_reward(self, state: Any) -> float:
        """计算状态奖励"""
        # 使用环境计算奖励
        try:
            reward = self.env.compute_reward()
            return reward
        except:
            return 0.0
    
    def _train_step(
        self,
        trajectories: List[List],
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """训练步骤"""
        self.gflownet.train()
        self.optimizer.zero_grad()
        
        # 使用GFlowNet的训练方法
        stats = self.gflownet.train_step(
            trajectories,
            rewards,
            loss_type=self.config.loss_type
        )
        
        return stats
    
    def evaluate(self, num_samples: int = 100) -> Dict[str, float]:
        """评估GFlowNet"""
        self.gflownet.eval()
        
        samples = []
        rewards = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                traj = self.gflownet.sample_trajectory(
                    self.env,
                    max_steps=self.config.max_steps_per_trajectory,
                    epsilon=0.0  # 无探索
                )
                
                if traj:
                    reward = self._compute_reward(traj[-1][0])
                    samples.append(traj[-1][0])
                    rewards.append(reward)
        
        if not rewards:
            return {'top_k_mean': 0.0, 'diversity': 0.0}
        
        # Top-K平均奖励
        top_k = min(10, len(rewards))
        top_k_mean = np.mean(sorted(rewards, reverse=True)[:top_k])
        
        # 多样性
        diversity = self._compute_diversity(samples)
        
        return {
            'top_k_mean': top_k_mean,
            'diversity': diversity,
            'mean_reward': np.mean(rewards),
            'max_reward': max(rewards),
        }
    
    def _compute_diversity(self, samples: List[Any]) -> float:
        """计算样本多样性"""
        if len(samples) < 2:
            return 0.0
        
        # 计算平均成对距离
        distances = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                dist = self._state_distance(samples[i], samples[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _state_distance(self, state1: Any, state2: Any) -> float:
        """计算状态距离"""
        # 简化实现
        if isinstance(state1, np.ndarray) and isinstance(state2, np.ndarray):
            return np.linalg.norm(state1 - state2)
        
        # 字符串表示的差异
        return 0.0 if str(state1) == str(state2) else 1.0
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'gflownet': self.gflownet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iteration': self.iteration,
            'config': self.config,
            'training_history': self.training_history,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.gflownet.load_state_dict(checkpoint['gflownet'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iteration = checkpoint['iteration']
        self.training_history = checkpoint['training_history']
        print(f"Model loaded from {path}")
    
    def generate_samples(
        self,
        num_samples: int = 100,
        epsilon: float = 0.0
    ) -> List[Dict[str, Any]]:
        """生成样本"""
        self.gflownet.eval()
        
        samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                traj = self.gflownet.sample_trajectory(
                    self.env,
                    max_steps=self.config.max_steps_per_trajectory,
                    epsilon=epsilon
                )
                
                if traj:
                    sample = {
                        'trajectory': traj,
                        'final_state': traj[-1][0],
                        'reward': self._compute_reward(traj[-1][0]),
                    }
                    samples.append(sample)
        
        return samples


def demo():
    """演示GFlowNet训练"""
    print("=" * 60)
    print("GFlowNet Trainer Demo")
    print("=" * 60)
    
    # 导入必要的模块
    from ..models.gflownet import GFlowNet, GFlowNetConfig
    from ..environments.molecule_env import MolecularGraphEnv, MoleculeEnvConfig
    
    # 创建环境
    env_config = MoleculeEnvConfig(
        max_atoms=10,
        atom_types=['C', 'N', 'O', 'H']
    )
    env = MolecularGraphEnv(env_config)
    
    # 创建GFlowNet
    gfn_config = GFlowNetConfig(
        state_dim=env.state_dim if hasattr(env, 'state_dim') else 200,
        action_dim=env.action_dim,
        hidden_dim=128,
        num_layers=2,
    )
    gfn = GFlowNet(gfn_config)
    
    # 创建训练配置
    train_config = GFlowNetTrainingConfig(
        num_iterations=100,
        batch_size=8,
        num_samples_per_iter=8,
        eval_frequency=20,
    )
    
    # 创建训练器
    trainer = GFlowNetTrainer(gfn, env, train_config)
    
    # 训练
    print("\nTraining...")
    history = trainer.train()
    
    # 生成样本
    print("\nGenerating samples...")
    samples = trainer.generate_samples(num_samples=10)
    
    print(f"Generated {len(samples)} samples")
    if samples:
        rewards = [s['reward'] for s in samples]
        print(f"Mean reward: {np.mean(rewards):.3f}")
        print(f"Max reward: {max(rewards):.3f}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
