"""
KAS Training - Trainer
端到端训练框架
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import time
import json
import os
from pathlib import Path


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    eval_interval: int = 50
    save_interval: int = 100
    
    # 学习率调度
    lr_schedule: str = "constant"  # constant, linear, exponential
    lr_decay_rate: float = 0.995
    lr_decay_steps: int = 1000
    
    # 日志
    log_dir: str = "./logs"
    experiment_name: str = "kas_drl"
    
    # 早停
    use_early_stopping: bool = True
    patience: int = 50
    min_delta: float = 0.01
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 混合训练
    use_mixed_precision: bool = False


class Trainer:
    """通用训练器"""
    
    def __init__(
        self,
        agent,
        env,
        config: Optional[TrainingConfig] = None
    ):
        self.agent = agent
        self.env = env
        self.config = config or TrainingConfig()
        
        # 创建日志目录
        self.log_path = Path(self.config.log_dir) / self.config.experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # 历史记录
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'losses': [],
            'timestamps': []
        }
        
        self.best_reward = -np.inf
        self.patience_counter = 0
        self.start_time = time.time()
    
    def train(self, callback: Optional[Callable] = None) -> Dict[str, List]:
        """
        训练Agent
        
        Args:
            callback: 每轮回调函数，接收(episode, metrics)
        
        Returns:
            history: 训练历史
        """
        print(f"Starting training on {self.config.device}")
        print(f"Log directory: {self.log_path}")
        
        for episode in range(self.config.num_episodes):
            episode_metrics = self._train_episode()
            
            # 记录
            self.history['episode_rewards'].append(episode_metrics['reward'])
            self.history['episode_lengths'].append(episode_metrics['length'])
            self.history['timestamps'].append(time.time() - self.start_time)
            
            # 评估
            if episode % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.history['eval_rewards'].append(eval_metrics['mean_reward'])
                
                # 早停检查
                if self._check_early_stopping(eval_metrics['mean_reward']):
                    print(f"Early stopping at episode {episode}")
                    break
                
                # 保存最佳模型
                if eval_metrics['mean_reward'] > self.best_reward:
                    self.best_reward = eval_metrics['mean_reward']
                    self.save_checkpoint('best_model.pt')
                
                print(f"Episode {episode}/{self.config.num_episodes} | "
                      f"Reward: {episode_metrics['reward']:.2f} | "
                      f"Eval: {eval_metrics['mean_reward']:.2f} | "
                      f"Best: {self.best_reward:.2f}")
            
            # 定期保存
            if episode > 0 and episode % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_{episode}.pt')
            
            # 回调
            if callback:
                callback(episode, episode_metrics)
        
        # 保存最终模型
        self.save_checkpoint('final_model.pt')
        self.save_history()
        
        return self.history
    
    def _train_episode(self) -> Dict[str, float]:
        """训练一个回合"""
        state = self.env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(self.config.max_steps_per_episode):
            # 选择动作
            action = self.agent.select_action(state)
            
            # 执行
            next_state, reward, done, info = self.env.step(action)
            
            # 存储转移（如果有这个方法）
            if hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(reward, next_state, done)
            
            # 更新
            if hasattr(self.agent, 'update'):
                loss = self.agent.update()
                if loss:
                    episode_losses.append(loss.get('actor_loss', 0) if isinstance(loss, dict) else loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        return {
            'reward': episode_reward,
            'length': step + 1,
            'loss': np.mean(episode_losses) if episode_losses else 0
        }
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """评估Agent"""
        rewards = []
        lengths = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.config.max_steps_per_episode):
                # 确定性策略
                if hasattr(self.agent, 'select_action'):
                    action = self.agent.select_action(state, deterministic=True)
                else:
                    action = self.agent.select_action(state)
                
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
            lengths.append(step + 1)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
    
    def _check_early_stopping(self, eval_reward: float) -> bool:
        """检查早停条件"""
        if not self.config.use_early_stopping:
            return False
        
        if eval_reward < self.best_reward - self.config.min_delta:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
        
        return self.patience_counter >= self.config.patience
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        if hasattr(self.agent, 'save'):
            self.agent.save(self.log_path / filename)
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.log_path / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        if hasattr(self.agent, 'load'):
            self.agent.load(self.log_path / filename)


class DistributedTrainer(Trainer):
    """分布式训练器（模拟）"""
    
    def __init__(
        self,
        agents: List,
        env_factory: Callable,
        config: Optional[TrainingConfig] = None,
        num_workers: int = 4
    ):
        super().__init__(agents[0], env_factory(), config)
        self.agents = agents
        self.env_factory = env_factory
        self.num_workers = num_workers
    
    def train_parallel(self) -> Dict[str, List]:
        """并行训练（使用多进程模拟）"""
        # 注意：实际实现需要使用torch.multiprocessing
        # 这里简化处理
        return self.train()


class MetaTrainer:
    """元学习训练器"""
    
    def __init__(
        self,
        meta_learner,
        task_sampler,
        config: Optional[TrainingConfig] = None
    ):
        self.meta_learner = meta_learner
        self.task_sampler = task_sampler
        self.config = config or TrainingConfig()
        
        self.log_path = Path(self.config.log_dir) / self.config.experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
    
    def train(self, num_iterations: int = 1000) -> Dict[str, List]:
        """元训练"""
        history = {
            'meta_losses': [],
            'task_losses': [],
            'timestamps': []
        }
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            # 采样任务
            tasks = self.task_sampler.sample_batch(4)
            
            # 元更新
            if hasattr(self.meta_learner, 'outer_step'):
                metrics = self.meta_learner.outer_step(tasks)
                history['meta_losses'].append(metrics['meta_loss'])
                history['task_losses'].append(metrics['mean_task_loss'])
            elif hasattr(self.meta_learner, 'meta_train_step'):
                metrics = self.meta_learner.meta_train_step(self.task_sampler)
                history['task_losses'].append(metrics['mean_task_loss'])
            
            history['timestamps'].append(time.time() - start_time)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{num_iterations} | "
                      f"Task Loss: {history['task_losses'][-1]:.4f}")
        
        # 保存
        history_path = self.log_path / 'meta_training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history


class OnlineLearningManager:
    """在线学习管理器"""
    
    def __init__(
        self,
        agent,
        update_frequency: int = 100,
        min_samples: int = 10
    ):
        self.agent = agent
        self.update_frequency = update_frequency
        self.min_samples = min_samples
        
        self.experience_buffer = []
        self.update_counter = 0
    
    def add_experience(self, state, action, reward, next_state, done):
        """添加经验"""
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        self.update_counter += 1
        
        # 触发更新
        if self.update_counter >= self.update_frequency:
            self._online_update()
            self.update_counter = 0
    
    def _online_update(self):
        """在线更新"""
        if len(self.experience_buffer) < self.min_samples:
            return
        
        # 采样最近的样本
        recent_samples = self.experience_buffer[-self.min_samples:]
        
        # 小批量更新
        for exp in recent_samples:
            if hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(
                    exp['reward'],
                    exp['next_state'],
                    exp['done']
                )
        
        # 执行更新
        if hasattr(self.agent, 'update'):
            for _ in range(5):  # 小批量训练
                self.agent.update()
        
        # 清理旧样本
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.experience_buffer:
            return {}
        
        rewards = [exp['reward'] for exp in self.experience_buffer]
        
        return {
            'total_experiences': len(self.experience_buffer),
            'mean_reward': np.mean(rewards),
            'recent_mean_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        }
