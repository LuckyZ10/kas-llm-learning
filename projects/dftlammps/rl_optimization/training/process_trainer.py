"""
Process Optimization Training
=============================

工艺参数优化训练器
- RL-based optimization
- Bayesian optimization comparison
- Multi-fidelity optimization
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time


@dataclass
class ProcessTrainingConfig:
    """工艺优化训练配置"""
    num_episodes: int = 100
    max_steps_per_episode: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    eval_frequency: int = 10
    
    # 探索
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.98
    
    # 多保真度
    use_multifidelity: bool = False
    fidelity_levels: List[float] = None
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ProcessOptimizationTrainer:
    """工艺参数优化训练器"""
    
    def __init__(
        self,
        agent: torch.nn.Module,
        env,
        config: Optional[ProcessTrainingConfig] = None
    ):
        self.agent = agent
        self.env = env
        self.config = config or ProcessTrainingConfig()
        
        self.device = torch.device(self.config.device)
        self.agent.to(self.device)
        
        # 经验缓冲区
        self.replay_buffer = []
        self.buffer_size = 10000
        
        # 训练统计
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'best_rewards': [],
        }
        
        self.best_reward = float('-inf')
        self.best_params = None
        
        self.epsilon = self.config.epsilon_start
    
    def train(self) -> Dict[str, List]:
        """训练工艺优化代理"""
        print(f"Starting process optimization training")
        print(f"Episodes: {self.config.num_episodes}")
        
        start_time = time.time()
        
        for episode in range(self.config.num_episodes):
            # 运行一个episode
            episode_stats = self._run_episode()
            
            # 更新统计
            self.training_history['episode_rewards'].append(episode_stats['total_reward'])
            self.training_history['episode_lengths'].append(episode_stats['length'])
            
            # 更新最佳
            if episode_stats['best_reward'] > self.best_reward:
                self.best_reward = episode_stats['best_reward']
                self.best_params = episode_stats['best_params']
            
            self.training_history['best_rewards'].append(self.best_reward)
            
            # 训练
            if len(self.replay_buffer) >= self.config.batch_size:
                self._train_step()
            
            # 衰减epsilon
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
            
            # 日志
            if episode % self.config.eval_frequency == 0:
                print(f"Episode {episode}/{self.config.num_episodes} | "
                      f"Reward: {episode_stats['total_reward']:.3f} | "
                      f"Best: {self.best_reward:.3f} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f}s")
        print(f"Best reward: {self.best_reward:.3f}")
        print(f"Best params: {self.best_params}")
        
        return self.training_history
    
    def _run_episode(self) -> Dict[str, Any]:
        """运行一个episode"""
        state = self.env.reset()
        
        total_reward = 0
        episode_experiences = []
        best_reward_in_episode = float('-inf')
        best_params_in_episode = None
        
        for step in range(self.config.max_steps_per_episode):
            # 选择动作
            action = self._select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            episode_experiences.append({
                'state': state.copy() if isinstance(state, np.ndarray) else state,
                'action': action.copy() if isinstance(action, np.ndarray) else action,
                'reward': reward,
                'next_state': next_state.copy() if isinstance(next_state, np.ndarray) else next_state,
                'done': done,
            })
            
            total_reward += reward
            state = next_state
            
            # 记录最佳
            if reward > best_reward_in_episode:
                best_reward_in_episode = reward
                if 'params' in info:
                    best_params_in_episode = info['params']
            
            if done:
                break
        
        # 添加到回放缓冲区
        self.replay_buffer.extend(episode_experiences)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]
        
        return {
            'total_reward': total_reward,
            'length': step + 1,
            'best_reward': best_reward_in_episode,
            'best_params': best_params_in_episode,
        }
    
    def _select_action(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        # Epsilon-贪心
        if np.random.random() < self.epsilon:
            # 随机动作
            return np.random.randn(self.env.action_dim) * 0.5
        
        # 使用策略
        with torch.no_grad():
            if hasattr(self.agent, 'select_action'):
                return self.agent.select_action(state, deterministic=False)
            else:
                # 默认行为
                return np.random.randn(self.env.action_dim) * 0.3
    
    def _train_step(self):
        """训练步骤"""
        # 采样批次
        batch = self._sample_batch()
        
        # 训练
        if hasattr(self.agent, 'train_step'):
            self.agent.train_step(batch)
    
    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        """采样批次"""
        indices = np.random.choice(
            len(self.replay_buffer),
            min(self.config.batch_size, len(self.replay_buffer)),
            replace=False
        )
        
        batch = [self.replay_buffer[i] for i in indices]
        
        return {
            'state': torch.FloatTensor([b['state'] for b in batch]).to(self.device),
            'action': torch.FloatTensor([b['action'] for b in batch]).to(self.device),
            'reward': torch.FloatTensor([b['reward'] for b in batch]).to(self.device),
            'next_state': torch.FloatTensor([b['next_state'] for b in batch]).to(self.device),
            'done': torch.FloatTensor([b['done'] for b in batch]).to(self.device),
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估代理"""
        rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.max_steps_per_episode):
                # 确定性策略
                if hasattr(self.agent, 'select_action'):
                    action = self.agent.select_action(state, deterministic=True)
                else:
                    action = np.zeros(self.env.action_dim)
                
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': max(rewards),
        }


class MultiFidelityOptimizer:
    """
    多保真度优化器
    
    在不同保真度级别上进行优化，降低计算成本
    """
    
    def __init__(
        self,
        low_fidelity_fn: Callable,
        high_fidelity_fn: Callable,
        cost_ratio: float = 0.1
    ):
        self.low_fidelity_fn = low_fidelity_fn
        self.high_fidelity_fn = high_fidelity_fn
        self.cost_ratio = cost_ratio
        
        # 保真度模型 (从低保真度预测高保真度)
        self.fidelity_model = None
    
    def optimize(
        self,
        param_bounds: List[Tuple[float, float]],
        budget: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        多保真度优化
        
        Args:
            param_bounds: 参数范围
            budget: 总预算
            
        Returns:
            best_params, best_value
        """
        # 阶段1: 低保真度预筛选
        print("Phase 1: Low-fidelity screening")
        
        low_fidelity_results = []
        n_low = int(budget * 0.6)  # 60%预算用于低保真度
        
        for i in range(n_low):
            # 随机采样
            params = np.array([
                np.random.uniform(low, high) for low, high in param_bounds
            ])
            
            value = self.low_fidelity_fn(params)
            low_fidelity_results.append((params, value))
        
        # 排序
        low_fidelity_results.sort(key=lambda x: x[1], reverse=True)
        
        # 阶段2: 高保真度验证
        print("Phase 2: High-fidelity validation")
        
        n_high = budget - n_low
        candidates = low_fidelity_results[:n_high * 2]  # 选择前N个候选
        
        high_fidelity_results = []
        for params, low_value in candidates[:n_high]:
            high_value = self.high_fidelity_fn(params)
            high_fidelity_results.append((params, high_value))
        
        # 返回最佳
        best = max(high_fidelity_results, key=lambda x: x[1])
        return best


class AdaptiveOptimizer:
    """
    自适应优化器
    
    根据优化进度动态调整策略
    """
    
    def __init__(
        self,
        base_optimizer: str = "bayesian",
        adaptive_switch: bool = True
    ):
        self.base_optimizer = base_optimizer
        self.adaptive_switch = adaptive_switch
        
        self.optimizers = {}
        self.current_optimizer = base_optimizer
        self.performance_history = []
    
    def optimize(
        self,
        objective_fn: Callable,
        param_bounds: List[Tuple[float, float]],
        budget: int = 100
    ) -> Tuple[np.ndarray, float]:
        """自适应优化"""
        
        if self.base_optimizer == "bayesian":
            from ..environments.process_env import BayesianOptimizer
            
            optimizer = BayesianOptimizer(param_bounds)
            
            for i in range(budget):
                params = optimizer.suggest()
                value = objective_fn(params)
                optimizer.observe(params, value)
                
                # 记录性能
                _, best_so_far = optimizer.get_best()
                self.performance_history.append(best_so_far)
            
            return optimizer.get_best()
        
        elif self.base_optimizer == "random":
            # 随机搜索
            best_params = None
            best_value = float('-inf')
            
            for _ in range(budget):
                params = np.array([
                    np.random.uniform(low, high) for low, high in param_bounds
                ])
                value = objective_fn(params)
                
                if value > best_value:
                    best_value = value
                    best_params = params
            
            return best_params, best_value
        
        else:
            raise ValueError(f"Unknown optimizer: {self.base_optimizer}")


def compare_optimizers(
    objective_fn: Callable,
    param_bounds: List[Tuple[float, float]],
    budget: int = 100,
    num_runs: int = 5
) -> Dict[str, List[float]]:
    """
    比较不同优化器的性能
    
    Returns:
        每个优化器的性能历史
    """
    results = {
        'bayesian': [],
        'random': [],
    }
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        # Bayesian Optimization
        from ..environments.process_env import BayesianOptimizer
        bo = BayesianOptimizer(param_bounds)
        bo_history = []
        
        for _ in range(budget):
            params = bo.suggest()
            value = objective_fn(params)
            bo.observe(params, value)
            _, best = bo.get_best()
            bo_history.append(best)
        
        results['bayesian'].append(bo_history)
        
        # Random Search
        best_value = float('-inf')
        random_history = []
        
        for _ in range(budget):
            params = np.array([
                np.random.uniform(low, high) for low, high in param_bounds
            ])
            value = objective_fn(params)
            best_value = max(best_value, value)
            random_history.append(best_value)
        
        results['random'].append(random_history)
    
    return results


def demo():
    """演示工艺优化训练"""
    print("=" * 60)
    print("Process Optimization Training Demo")
    print("=" * 60)
    
    # 创建模拟环境
    from ..environments.process_env import ParameterEnv, ProcessEnvConfig
    
    # 定义目标函数 (Sphere函数)
    def sphere(x):
        return -np.sum(x**2)
    
    config = ProcessEnvConfig(max_steps=30)
    env = ParameterEnv(
        objective_func=sphere,
        num_params=3,
        bounds=[(-5, 5)] * 3,
        config=config
    )
    
    print(f"Environment created")
    print(f"State dim: {env.state_dim}")
    print(f"Action dim: {env.action_dim}")
    
    # 比较优化器
    print("\nComparing optimizers...")
    results = compare_optimizers(
        sphere,
        [(-5, 5)] * 3,
        budget=50,
        num_runs=3
    )
    
    # 计算平均最终性能
    bo_final = [run[-1] for run in results['bayesian']]
    random_final = [run[-1] for run in results['random']]
    
    print(f"Bayesian Optimization: {np.mean(bo_final):.3f} ± {np.std(bo_final):.3f}")
    print(f"Random Search: {np.mean(random_final):.3f} ± {np.std(random_final):.3f}")
    
    # 多保真度优化
    print("\nMulti-fidelity optimization...")
    
    def high_fidelity(x):
        return sphere(x) + np.random.randn() * 0.01
    
    def low_fidelity(x):
        # 低保真度: 添加更多噪声
        return sphere(x) + np.random.randn() * 0.1
    
    mf_optimizer = MultiFidelityOptimizer(
        low_fidelity_fn=low_fidelity,
        high_fidelity_fn=high_fidelity,
        cost_ratio=0.1
    )
    
    best_params, best_value = mf_optimizer.optimize(
        [(-5, 5)] * 3,
        budget=50
    )
    
    print(f"Best value: {best_value:.3f}")
    print(f"Best params: {best_params}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
