"""
Process Optimization Environments
==================================

实现工艺参数优化环境:
- 合成工艺环境
- 参数优化环境
- 贝叶斯优化 vs RL 比较
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ProcessEnvConfig:
    """工艺环境配置"""
    # 参数范围
    param_bounds: Dict[str, Tuple[float, float]] = None
    # 离散参数选项
    discrete_params: Dict[str, List[Any]] = None
    # 最大步数
    max_steps: int = 50
    # 噪声水平
    noise_level: float = 0.05
    # 是否批量优化
    batch_mode: bool = False
    batch_size: int = 5
    
    def __post_init__(self):
        if self.param_bounds is None:
            self.param_bounds = {
                'temperature': (300.0, 1500.0),
                'pressure': (0.1, 10.0),
                'time': (1.0, 48.0),
                'concentration': (0.01, 1.0),
            }
        if self.discrete_params is None:
            self.discrete_params = {
                'method': ['sol-gel', 'hydrothermal', 'solid-state', 'chemical_vapor'],
                'atmosphere': ['air', 'nitrogen', 'argon', 'vacuum'],
            }


class ProcessOptimizationEnv(ABC):
    """工艺优化环境基类"""
    
    def __init__(self, config: Optional[ProcessEnvConfig] = None):
        self.config = config or ProcessEnvConfig()
        self.current_params = {}
        self.step_count = 0
        self.history = []
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """重置环境"""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        pass
    
    @abstractmethod
    def evaluate(self, params: Dict[str, Any]) -> float:
        """评估参数"""
        pass


class SynthesisEnv(ProcessOptimizationEnv):
    """
    材料合成工艺环境
    
    优化合成参数以获得目标材料性质
    """
    
    def __init__(
        self,
        target_property: str = 'bandgap',
        target_value: float = 1.5,
        config: Optional[ProcessEnvConfig] = None
    ):
        super().__init__(config)
        
        self.target_property = target_property
        self.target_value = target_value
        
        # 连续参数数量
        self.num_continuous = len(self.config.param_bounds)
        self.num_discrete = len(self.config.discrete_params)
        
        # 状态: 当前参数 + 历史性能
        self.state_dim = self.num_continuous + self.num_discrete + 10
        
        # 动作: 参数调整
        self.action_dim = self.num_continuous + self.num_discrete
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 初始化参数为中间值
        self.current_params = {}
        
        for param, (low, high) in self.config.param_bounds.items():
            self.current_params[param] = (low + high) / 2
        
        for param, options in self.config.discrete_params.items():
            self.current_params[param] = options[0]
        
        self.step_count = 0
        self.history = []
        
        return self._get_state_vector()
    
    def _get_state_vector(self) -> np.ndarray:
        """获取状态向量"""
        # 连续参数 (归一化)
        continuous = []
        for param, (low, high) in self.config.param_bounds.items():
            value = self.current_params.get(param, (low + high) / 2)
            normalized = (value - low) / (high - low)
            continuous.append(normalized)
        
        # 离散参数 (one-hot)
        discrete = []
        for param, options in self.config.discrete_params.items():
            value = self.current_params.get(param, options[0])
            one_hot = [1.0 if value == opt else 0.0 for opt in options]
            discrete.extend(one_hot)
        
        # 历史统计
        stats = np.zeros(10)
        if self.history:
            rewards = [h['reward'] for h in self.history]
            stats[0] = np.mean(rewards)
            stats[1] = np.std(rewards) if len(rewards) > 1 else 0
            stats[2] = np.max(rewards)
            stats[3] = len(self.history) / self.config.max_steps
        
        # 组合
        state = np.array(continuous + discrete + stats.tolist())
        
        # 填充
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        
        return state[:self.state_dim].astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        self.step_count += 1
        
        # 解析动作
        action_idx = 0
        
        # 更新连续参数
        for param, (low, high) in self.config.param_bounds.items():
            delta = action[action_idx] * 0.1  # 调整步长
            self.current_params[param] += delta * (high - low)
            self.current_params[param] = np.clip(self.current_params[param], low, high)
            action_idx += 1
        
        # 更新离散参数
        for param, options in self.config.discrete_params.items():
            # 选择概率最高的选项
            logits = action[action_idx:action_idx + len(options)]
            selected_idx = np.argmax(logits)
            self.current_params[param] = options[min(selected_idx, len(options) - 1)]
            action_idx += len(options)
        
        # 评估当前参数
        reward = self.evaluate(self.current_params)
        
        # 记录历史
        self.history.append({
            'params': self.current_params.copy(),
            'reward': reward,
            'step': self.step_count,
        })
        
        # 检查是否结束
        done = self.step_count >= self.config.max_steps
        
        info = {
            'params': self.current_params.copy(),
            'target_property': self.target_property,
            'target_value': self.target_value,
        }
        
        return self._get_state_vector(), reward, done, info
    
    def evaluate(self, params: Dict[str, Any]) -> float:
        """
        评估合成参数
        
        这是一个模拟函数，实际应用中应调用实验或模拟
        """
        # 提取参数
        temp = params.get('temperature', 800)
        pressure = params.get('pressure', 1.0)
        time = params.get('time', 12.0)
        conc = params.get('concentration', 0.1)
        
        # 模拟材料性质 (简化模型)
        # 温度影响
        temp_factor = np.exp(-((temp - 800) / 400) ** 2)
        
        # 压力影响
        pressure_factor = np.log(pressure + 1) / np.log(11)
        
        # 时间影响
        time_factor = 1 - np.exp(-time / 10)
        
        # 浓度影响
        conc_factor = conc * (1 - conc) * 4  # 抛物线，最优在中间
        
        # 合成性质
        property_value = temp_factor * pressure_factor * time_factor * conc_factor
        
        # 添加噪声
        noise = np.random.randn() * self.config.noise_level
        property_value = np.clip(property_value + noise, 0, 1)
        
        # 与目标的接近程度
        reward = 1.0 - abs(property_value - self.target_value)
        
        return reward
    
    def get_best_params(self) -> Dict[str, Any]:
        """获取历史最佳参数"""
        if not self.history:
            return self.current_params
        
        best = max(self.history, key=lambda x: x['reward'])
        return best['params']


class ParameterEnv(ProcessOptimizationEnv):
    """
    通用参数优化环境
    
    用于通用的黑箱函数优化
    """
    
    def __init__(
        self,
        objective_func: Optional[Callable[[np.ndarray], float]] = None,
        num_params: int = 5,
        bounds: Optional[List[Tuple[float, float]]] = None,
        config: Optional[ProcessEnvConfig] = None
    ):
        super().__init__(config)
        
        self.num_params = num_params
        
        if bounds is None:
            self.bounds = [(-5.0, 5.0) for _ in range(num_params)]
        else:
            self.bounds = bounds[:num_params]
        
        # 目标函数
        if objective_func is None:
            self.objective_func = self._default_objective
        else:
            self.objective_func = objective_func
        
        self.state_dim = num_params * 2 + 5
        self.action_dim = num_params
    
    def _default_objective(self, x: np.ndarray) -> float:
        """默认目标函数: Rastrigin函数"""
        A = 10
        return -(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 随机初始化
        self.current_params = np.array([
            np.random.uniform(low, high) for low, high in self.bounds
        ])
        self.step_count = 0
        self.history = []
        return self._get_state_vector()
    
    def _get_state_vector(self) -> np.ndarray:
        """获取状态向量"""
        # 归一化参数
        normalized = []
        for i, (low, high) in enumerate(self.bounds):
            norm = (self.current_params[i] - low) / (high - low)
            normalized.append(norm)
        
        # 参数梯度估计 (有限差分)
        gradients = self._estimate_gradients()
        
        # 历史统计
        stats = np.zeros(5)
        if self.history:
            values = [h['value'] for h in self.history]
            stats[0] = np.mean(values)
            stats[1] = np.std(values) if len(values) > 1 else 0
            stats[2] = np.max(values)
            stats[3] = len(self.history) / self.config.max_steps
            stats[4] = self.step_count / self.config.max_steps
        
        state = np.concatenate([normalized, gradients, stats])
        return state.astype(np.float32)
    
    def _estimate_gradients(self) -> np.ndarray:
        """估计梯度"""
        gradients = np.zeros(self.num_params)
        
        if len(self.history) < 2:
            return gradients
        
        # 使用最近的历史估计梯度
        recent = self.history[-5:]
        if len(recent) >= 2:
            for i in range(self.num_params):
                diffs = []
                for j in range(1, len(recent)):
                    param_diff = recent[j]['params'][i] - recent[j-1]['params'][i]
                    value_diff = recent[j]['value'] - recent[j-1]['value']
                    if abs(param_diff) > 1e-6:
                        diffs.append(value_diff / param_diff)
                if diffs:
                    gradients[i] = np.mean(diffs)
        
        # 归一化
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > 0:
            gradients = gradients / grad_norm
        
        return gradients
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        self.step_count += 1
        
        # 动作是参数调整
        for i in range(self.num_params):
            delta = action[i] * (self.bounds[i][1] - self.bounds[i][0]) * 0.1
            self.current_params[i] += delta
            self.current_params[i] = np.clip(
                self.current_params[i],
                self.bounds[i][0],
                self.bounds[i][1]
            )
        
        # 评估
        value = self.objective_func(self.current_params)
        
        # 添加噪声
        noise = np.random.randn() * self.config.noise_level
        reward = value + noise
        
        # 记录
        self.history.append({
            'params': self.current_params.copy(),
            'value': value,
            'reward': reward,
            'step': self.step_count,
        })
        
        done = self.step_count >= self.config.max_steps
        
        info = {
            'params': self.current_params.copy(),
            'true_value': value,
        }
        
        return self._get_state_vector(), reward, done, info
    
    def evaluate(self, params: Dict[str, Any]) -> float:
        """评估参数"""
        if isinstance(params, dict):
            params_array = np.array(list(params.values()))
        else:
            params_array = np.array(params)
        
        return self.objective_func(params_array)


class BayesianOptimizer:
    """
    贝叶斯优化器
    
    用于与RL方法对比的基线
    """
    
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        acquisition: str = 'ei',  # 'ei', 'ucb', 'pi'
        xi: float = 0.01,
        kappa: float = 2.0
    ):
        self.bounds = bounds
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa
        
        self.X = []
        self.y = []
        
        # 高斯过程
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            
            self.kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=5,
                normalize_y=True
            )
            self.has_sklearn = True
        except ImportError:
            self.has_sklearn = False
    
    def suggest(self) -> np.ndarray:
        """建议下一个采样点"""
        if len(self.X) < 2 or not self.has_sklearn:
            # 随机采样
            return np.array([
                np.random.uniform(low, high) for low, high in self.bounds
            ])
        
        # 拟合GP
        X_array = np.array(self.X)
        y_array = np.array(self.y)
        self.gp.fit(X_array, y_array)
        
        # 优化采集函数
        best_x = None
        best_acq = float('-inf')
        
        # 随机采样多个候选点
        n_candidates = 1000
        candidates = np.array([
            [np.random.uniform(low, high) for low, high in self.bounds]
            for _ in range(n_candidates)
        ])
        
        # 计算采集函数值
        acq_values = self._acquisition_function(candidates)
        
        best_idx = np.argmax(acq_values)
        best_x = candidates[best_idx]
        
        return best_x
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """计算采集函数值"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if self.acquisition == 'ei':
            # 期望改进
            if len(self.y) > 0:
                y_max = np.max(self.y)
                imp = mu - y_max - self.xi
                Z = imp / (sigma + 1e-9)
                ei = imp * (0.5 * (1 + np.sign(Z)))  # 简化的EI
                return ei
            else:
                return mu
        
        elif self.acquisition == 'ucb':
            # 上置信界
            return mu + self.kappa * sigma
        
        elif self.acquisition == 'pi':
            # 改进概率
            if len(self.y) > 0:
                y_max = np.max(self.y)
                imp = mu - y_max - self.xi
                Z = imp / (sigma + 1e-9)
                return 0.5 * (1 + np.sign(Z))  # 简化的PI
            else:
                return np.ones(len(X))
        
        return mu
    
    def observe(self, x: np.ndarray, y: float):
        """观察结果"""
        self.X.append(x)
        self.y.append(y)
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """获取最佳观测"""
        if not self.y:
            return None, float('-inf')
        
        best_idx = np.argmax(self.y)
        return self.X[best_idx], self.y[best_idx]


def demo():
    """演示工艺优化环境"""
    print("=" * 60)
    print("Process Optimization Environments Demo")
    print("=" * 60)
    
    config = ProcessEnvConfig(
        max_steps=20,
        noise_level=0.02
    )
    
    # 1. 合成环境
    print("\n1. Synthesis Environment")
    env = SynthesisEnv(
        target_property='bandgap',
        target_value=0.8,
        config=config
    )
    
    state = env.reset()
    print(f"   State shape: {state.shape}")
    print(f"   Action dim: {env.action_dim}")
    
    # 随机优化
    rewards = []
    for i in range(config.max_steps):
        action = np.random.randn(env.action_dim) * 0.5
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        
        if done:
            print(f"   Finished after {i+1} steps")
            print(f"   Best reward: {max(rewards):.3f}")
            print(f"   Best params: {env.get_best_params()}")
            break
    
    # 2. 参数环境
    print("\n2. Parameter Environment")
    
    # 定义测试函数: Ackley
    def ackley(x):
        a, b, c = 20, 0.2, 2 * np.pi
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        n = len(x)
        return -(-a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e)
    
    param_env = ParameterEnv(
        objective_func=ackley,
        num_params=3,
        bounds=[(-5, 5)] * 3,
        config=config
    )
    
    state = param_env.reset()
    print(f"   State shape: {state.shape}")
    
    for i in range(config.max_steps):
        action = np.random.randn(param_env.action_dim) * 0.3
        state, reward, done, info = param_env.step(action)
        
        if done:
            best = max(param_env.history, key=lambda x: x['reward'])
            print(f"   Best value: {best['value']:.3f}")
            print(f"   Best params: {best['params']}")
            break
    
    # 3. 贝叶斯优化对比
    print("\n3. Bayesian Optimization (for comparison)")
    bo = BayesianOptimizer(bounds=[(-5, 5)] * 3)
    
    for i in range(20):
        x = bo.suggest()
        y = ackley(x)
        bo.observe(x, y)
    
    best_x, best_y = bo.get_best()
    print(f"   Best value: {best_y:.3f}")
    print(f"   Best params: {best_x}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
