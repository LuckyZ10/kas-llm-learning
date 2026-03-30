"""
Reward Function Design Tools
============================

奖励函数设计工具集:
- 基础奖励函数
- 多目标奖励组合
- 奖励学习 (偏好学习、逆强化学习)
- 奖励整形
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class RewardConfig:
    """奖励函数配置"""
    normalize: bool = True
    clip_range: Tuple[float, float] = (-10, 10)
    temperature: float = 1.0
    use_shaping: bool = False
    gamma: float = 0.99


class RewardFunction(ABC):
    """奖励函数基类"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
    
    @abstractmethod
    def compute(self, state: Any, action: Any = None, info: Dict = None) -> float:
        """计算奖励"""
        pass
    
    def __call__(self, state: Any, action: Any = None, info: Dict = None) -> float:
        """调用接口"""
        reward = self.compute(state, action, info)
        
        # 归一化
        if self.config.normalize:
            reward = reward / (abs(reward) + 1e-6)
        
        # 裁剪
        if self.config.clip_range:
            reward = np.clip(reward, *self.config.clip_range)
        
        return reward


class PropertyReward(RewardFunction):
    """
    材料性质奖励
    
    基于目标性质的奖励
    """
    
    def __init__(
        self,
        property_name: str,
        target_value: float,
        property_func: Optional[Callable[[Any], float]] = None,
        tolerance: float = 0.1,
        config: Optional[RewardConfig] = None
    ):
        super().__init__(config)
        
        self.property_name = property_name
        self.target_value = target_value
        self.property_func = property_func or self._default_property_func
        self.tolerance = tolerance
    
    def _default_property_func(self, state: Any) -> float:
        """默认性质函数"""
        if isinstance(state, dict) and self.property_name in state:
            return state[self.property_name]
        return 0.0
    
    def compute(self, state: Any, action: Any = None, info: Dict = None) -> float:
        """计算性质奖励"""
        value = self.property_func(state)
        
        # 高斯型奖励
        diff = abs(value - self.target_value)
        reward = np.exp(-diff / self.tolerance)
        
        return reward


class ValidityReward(RewardFunction):
    """
    化学有效性奖励
    
    奖励化学上有效的分子/结构
    """
    
    def __init__(
        self,
        validity_func: Optional[Callable[[Any], Tuple[bool, float]]] = None,
        config: Optional[RewardConfig] = None
    ):
        super().__init__(config)
        self.validity_func = validity_func or self._default_validity_check
    
    def _default_validity_check(self, state: Any) -> Tuple[bool, float]:
        """默认有效性检查"""
        # 简化检查
        if isinstance(state, dict):
            # 检查是否有原子
            if 'atoms' in state and len(state['atoms']) > 0:
                return True, 1.0
            if 'num_atoms' in state and state['num_atoms'] > 0:
                return True, 1.0
        
        return False, 0.0
    
    def compute(self, state: Any, action: Any = None, info: Dict = None) -> float:
        """计算有效性奖励"""
        is_valid, score = self.validity_func(state)
        return score if is_valid else -1.0


class DiversityReward(RewardFunction):
    """
    多样性奖励
    
    鼓励生成多样化的样本
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        config: Optional[RewardConfig] = None
    ):
        super().__init__(config)
        
        self.similarity_threshold = similarity_threshold
        self.generated_samples = []
        self.max_history = 1000
    
    def compute(self, state: Any, action: Any = None, info: Dict = None) -> float:
        """计算多样性奖励"""
        if not self.generated_samples:
            return 1.0
        
        # 计算与历史样本的相似度
        min_similarity = 1.0
        
        for sample in self.generated_samples:
            sim = self._compute_similarity(state, sample)
            min_similarity = min(min_similarity, sim)
        
        # 多样性奖励: 与最近样本的距离
        diversity = 1.0 - min_similarity
        
        return diversity
    
    def _compute_similarity(self, state1: Any, state2: Any) -> float:
        """计算两个状态的相似度"""
        # 简化实现: 使用字符串表示的相似度
        str1 = str(state1)
        str2 = str(state2)
        
        # Jaccard相似度
        set1 = set(str1)
        set2 = set(str2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def add_sample(self, sample: Any):
        """添加样本到历史"""
        self.generated_samples.append(sample)
        
        # 限制历史大小
        if len(self.generated_samples) > self.max_history:
            self.generated_samples.pop(0)


class NoveltyReward(RewardFunction):
    """
    新颖性奖励
    
    奖励与已知数据库不同的样本
    """
    
    def __init__(
        self,
        reference_database: Optional[List[Any]] = None,
        k: int = 5,
        config: Optional[RewardConfig] = None
    ):
        super().__init__(config)
        
        self.reference_database = reference_database or []
        self.k = k
    
    def compute(self, state: Any, action: Any = None, info: Dict = None) -> float:
        """计算新颖性奖励"""
        if not self.reference_database:
            return 1.0
        
        # 计算到k近邻的平均距离
        distances = []
        
        for ref in self.reference_database:
            dist = self._compute_distance(state, ref)
            distances.append(dist)
        
        # k近邻平均距离
        distances.sort()
        k_dist = np.mean(distances[:self.k])
        
        # 归一化
        novelty = min(k_dist / 10.0, 1.0)
        
        return novelty
    
    def _compute_distance(self, state1: Any, state2: Any) -> float:
        """计算距离"""
        # 简化实现
        if isinstance(state1, dict) and isinstance(state2, dict):
            # 计算字典差异
            all_keys = set(state1.keys()) | set(state2.keys())
            diff = 0
            for key in all_keys:
                val1 = state1.get(key, 0)
                val2 = state2.get(key, 0)
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff += abs(val1 - val2)
                else:
                    diff += 1 if val1 != val2 else 0
            return diff / len(all_keys) if all_keys else 0.0
        
        return 0.0 if state1 == state2 else 1.0


class MultiObjectiveReward(RewardFunction):
    """
    多目标奖励
    
    组合多个奖励函数
    """
    
    def __init__(
        self,
        reward_functions: List[Tuple[RewardFunction, float]],
        method: str = 'weighted_sum',  # 'weighted_sum', 'product', 'pareto'
        config: Optional[RewardConfig] = None
    ):
        super().__init__(config)
        
        self.reward_functions = reward_functions
        self.method = method
        
        # Pareto前沿
        self.pareto_front = []
    
    def compute(self, state: Any, action: Any = None, info: Dict = None) -> float:
        """计算多目标奖励"""
        rewards = []
        
        for reward_func, weight in self.reward_functions:
            r = reward_func.compute(state, action, info)
            rewards.append((r, weight))
        
        if self.method == 'weighted_sum':
            total = sum(r * w for r, w in rewards)
            return total
        
        elif self.method == 'product':
            product = 1.0
            for r, w in rewards:
                product *= (r ** w)
            return product
        
        elif self.method == 'pareto':
            # 计算到Pareto前沿的距离
            objective_vector = [r for r, _ in rewards]
            
            # 更新Pareto前沿
            self._update_pareto_front(objective_vector)
            
            # 计算Pareto支配度
            dominance = self._compute_pareto_dominance(objective_vector)
            
            return dominance
        
        return sum(r * w for r, w in rewards)
    
    def _update_pareto_front(self, objective_vector: List[float]):
        """更新Pareto前沿"""
        # 检查是否被支配
        dominated = False
        to_remove = []
        
        for i, point in enumerate(self.pareto_front):
            if self._dominates(point, objective_vector):
                dominated = True
                break
            if self._dominates(objective_vector, point):
                to_remove.append(i)
        
        if not dominated:
            # 移除被支配的点
            for i in reversed(to_remove):
                self.pareto_front.pop(i)
            
            # 添加新点
            self.pareto_front.append(objective_vector)
    
    def _dominates(self, p1: List[float], p2: List[float]) -> bool:
        """检查p1是否支配p2"""
        better_in_one = False
        
        for v1, v2 in zip(p1, p2):
            if v1 < v2:
                return False
            if v1 > v2:
                better_in_one = True
        
        return better_in_one
    
    def _compute_pareto_dominance(self, objective_vector: List[float]) -> float:
        """计算Pareto支配度"""
        if not self.pareto_front:
            return 1.0
        
        # 计算到最近Pareto点的距离
        min_dist = float('inf')
        
        for point in self.pareto_front:
            dist = np.linalg.norm(np.array(objective_vector) - np.array(point))
            min_dist = min(min_dist, dist)
        
        return np.exp(-min_dist)


class CompositeReward(RewardFunction):
    """
    组合奖励
    
    通过函数组合多个奖励
    """
    
    def __init__(
        self,
        reward_functions: Dict[str, RewardFunction],
        composition_func: Optional[Callable[[Dict[str, float]], float]] = None,
        config: Optional[RewardConfig] = None
    ):
        super().__init__(config)
        
        self.reward_functions = reward_functions
        self.composition_func = composition_func or self._default_composition
    
    def _default_composition(self, rewards: Dict[str, float]) -> float:
        """默认组合函数: 加权平均"""
        return np.mean(list(rewards.values()))
    
    def compute(self, state: Any, action: Any = None, info: Dict = None) -> float:
        """计算组合奖励"""
        rewards = {}
        
        for name, reward_func in self.reward_functions.items():
            rewards[name] = reward_func.compute(state, action, info)
        
        return self.composition_func(rewards)


class RewardModel(nn.Module):
    """
    神经网络奖励模型
    
    学习从状态到奖励的映射
    """
    
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
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)
    
    def predict_reward(self, state: np.ndarray) -> float:
        """预测奖励"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            reward = self.forward(state_tensor)
            return reward.item()


class PreferenceLearning:
    """
    偏好学习
    
    从人类偏好中学习奖励函数
    
    参考: Christiano et al. "Deep Reinforcement Learning from Human Preferences" (2017)
    """
    
    def __init__(
        self,
        state_dim: int,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        
        # 奖励模型
        self.reward_model = RewardModel(state_dim).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.reward_model.parameters(),
            lr=learning_rate
        )
        
        # 偏好数据
        self.preferences = []
    
    def add_preference(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
        preference: int  # 0: prefer 1, 1: prefer 2, 0.5: equal
    ):
        """添加偏好比较"""
        self.preferences.append({
            'state1': state1,
            'state2': state2,
            'preference': preference,
        })
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """训练步骤"""
        if len(self.preferences) < batch_size:
            return {'loss': 0.0}
        
        # 采样批次
        import random
        batch = random.sample(self.preferences, batch_size)
        
        # 准备数据
        states1 = torch.FloatTensor([p['state1'] for p in batch]).to(self.device)
        states2 = torch.FloatTensor([p['state2'] for p in batch]).to(self.device)
        prefs = torch.FloatTensor([p['preference'] for p in batch]).to(self.device)
        
        # 预测奖励
        rewards1 = self.reward_model(states1)
        rewards2 = self.reward_model(states2)
        
        # Bradley-Terry模型
        prob_prefer_1 = torch.sigmoid(rewards1 - rewards2)
        
        # 损失
        # 如果preference=0, 希望prob_prefer_1接近1
        # 如果preference=1, 希望prob_prefer_1接近0
        # 如果preference=0.5, 希望prob_prefer_1接近0.5
        target_prob = 1 - prefs  # 转换为偏好1的概率
        loss = F.binary_cross_entropy(prob_prefer_1.squeeze(), target_prob)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'accuracy': ((prob_prefer_1.squeeze() > 0.5).float() == target_prob).float().mean().item()
        }
    
    def predict_preference(
        self,
        state1: np.ndarray,
        state2: np.ndarray
    ) -> Tuple[float, int]:
        """预测偏好"""
        with torch.no_grad():
            s1 = torch.FloatTensor(state1).unsqueeze(0).to(self.device)
            s2 = torch.FloatTensor(state2).unsqueeze(0).to(self.device)
            
            r1 = self.reward_model(s1)
            r2 = self.reward_model(s2)
            
            prob = torch.sigmoid(r1 - r2).item()
            preference = 0 if prob > 0.5 else 1
            
            return prob, preference


class InverseRL:
    """
    逆强化学习
    
    从专家演示中学习奖励函数
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        
        # 奖励模型
        self.reward_model = RewardModel(state_dim).to(self.device)
        
        # 策略模型 (用于MaxEnt IRL)
        from ..models.policy import CategoricalPolicy
        from ..models.policy import PolicyConfig
        
        config = PolicyConfig(state_dim=state_dim, action_dim=action_dim)
        self.policy = CategoricalPolicy(config).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.reward_model.parameters(),
            lr=learning_rate
        )
        
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )
        
        # 专家演示
        self.expert_demonstrations = []
    
    def add_demonstration(self, trajectory: List[Tuple[np.ndarray, np.ndarray]]):
        """添加专家演示"""
        self.expert_demonstrations.append(trajectory)
    
    def maxent_irl_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        MaxEnt IRL训练步骤
        
        最大化专家轨迹的似然
        """
        if not self.expert_demonstrations:
            return {'loss': 0.0}
        
        # 采样专家轨迹
        import random
        trajectories = random.choices(self.expert_demonstrations, k=batch_size)
        
        # 计算专家轨迹的奖励
        expert_rewards = []
        for traj in trajectories:
            traj_reward = 0
            for state, _ in traj:
                state_tensor = torch.FloatTensor(state).to(self.device)
                r = self.reward_model(state_tensor)
                traj_reward += r
            expert_rewards.append(traj_reward)
        
        expert_rewards = torch.stack(expert_rewards)
        
        # 计算策略轨迹的奖励 (简化: 使用当前策略采样)
        policy_rewards = []
        for _ in range(batch_size):
            # 简化: 随机采样一个状态
            state = np.random.randn(self.policy.config.state_dim)
            state_tensor = torch.FloatTensor(state).to(self.device)
            r = self.reward_model(state_tensor)
            policy_rewards.append(r)
        
        policy_rewards = torch.stack(policy_rewards)
        
        # 损失: 专家奖励应该高于策略奖励
        loss = -torch.mean(expert_rewards) + torch.mean(policy_rewards)
        
        # 添加正则化
        loss += 0.01 * sum(p.pow(2).mean() for p in self.reward_model.parameters())
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'expert_reward': expert_rewards.mean().item(),
            'policy_reward': policy_rewards.mean().item(),
        }


class RewardShaping:
    """
    奖励整形
    
    使用势函数改进学习
    
    参考: Ng et al. "Policy invariance under reward transformations" (1999)
    """
    
    def __init__(self, potential_func: Optional[Callable[[Any], float]] = None, gamma: float = 0.99):
        self.potential_func = potential_func or self._default_potential
        self.gamma = gamma
        self.prev_potential = None
    
    def _default_potential(self, state: Any) -> float:
        """默认势函数"""
        # 使用状态范数作为势
        if isinstance(state, np.ndarray):
            return -np.linalg.norm(state)
        return 0.0
    
    def shape_reward(
        self,
        state: Any,
        reward: float,
        next_state: Any,
        done: bool
    ) -> float:
        """
        整形奖励
        
        F(s, a, s') = R(s, a, s') + γΦ(s') - Φ(s)
        """
        current_potential = self.potential_func(state)
        next_potential = self.potential_func(next_state)
        
        shaped_reward = reward + self.gamma * next_potential - current_potential
        
        self.prev_potential = next_potential
        
        return shaped_reward
    
    def reset(self):
        """重置势函数状态"""
        self.prev_potential = None


class RewardDesigner:
    """
    奖励设计器
    
    提供交互式奖励函数构建工具
    """
    
    def __init__(self):
        self.components = {}
        self.weights = {}
        self.constraints = []
    
    def add_component(
        self,
        name: str,
        reward_func: RewardFunction,
        weight: float = 1.0
    ):
        """添加奖励组件"""
        self.components[name] = reward_func
        self.weights[name] = weight
    
    def set_weight(self, name: str, weight: float):
        """设置组件权重"""
        if name in self.weights:
            self.weights[name] = weight
    
    def add_constraint(
        self,
        constraint_func: Callable[[Any], bool],
        penalty: float = -1.0
    ):
        """添加约束"""
        self.constraints.append((constraint_func, penalty))
    
    def build(self) -> RewardFunction:
        """构建最终奖励函数"""
        # 创建加权组合
        weighted_funcs = [
            (self.components[name], self.weights[name])
            for name in self.components.keys()
        ]
        
        base_reward = MultiObjectiveReward(
            weighted_funcs,
            method='weighted_sum'
        )
        
        return base_reward
    
    def evaluate(self, state: Any) -> Dict[str, float]:
        """评估各组件"""
        results = {}
        
        for name, reward_func in self.components.items():
            results[name] = reward_func.compute(state)
        
        # 计算加权和
        total = sum(results[name] * self.weights[name] for name in results)
        results['total'] = total
        
        return results


def demo():
    """演示奖励函数"""
    print("=" * 60)
    print("Reward Function Design Demo")
    print("=" * 60)
    
    # 1. 属性奖励
    print("\n1. Property Reward")
    property_reward = PropertyReward(
        property_name='bandgap',
        target_value=1.5,
        tolerance=0.2
    )
    
    test_state = {'bandgap': 1.3}
    reward = property_reward(test_state)
    print(f"   State: {test_state}")
    print(f"   Reward: {reward:.3f}")
    
    # 2. 有效性奖励
    print("\n2. Validity Reward")
    validity_reward = ValidityReward()
    
    valid_state = {'num_atoms': 10}
    invalid_state = {'num_atoms': 0}
    
    print(f"   Valid state reward: {validity_reward(valid_state):.3f}")
    print(f"   Invalid state reward: {validity_reward(invalid_state):.3f}")
    
    # 3. 多样性奖励
    print("\n3. Diversity Reward")
    diversity_reward = DiversityReward()
    
    samples = [
        {'atoms': ['C', 'C', 'H']},
        {'atoms': ['C', 'N', 'O']},
        {'atoms': ['O', 'O', 'H']},
    ]
    
    for sample in samples:
        reward = diversity_reward(sample)
        print(f"   Sample {sample['atoms']} diversity: {reward:.3f}")
        diversity_reward.add_sample(sample)
    
    # 4. 多目标奖励
    print("\n4. Multi-Objective Reward")
    multi_reward = MultiObjectiveReward([
        (property_reward, 0.6),
        (validity_reward, 0.4),
    ])
    
    test_state = {'bandgap': 1.4, 'num_atoms': 8}
    reward = multi_reward(test_state)
    print(f"   Combined reward: {reward:.3f}")
    
    # 5. 奖励设计器
    print("\n5. Reward Designer")
    designer = RewardDesigner()
    
    designer.add_component('property', property_reward, weight=0.7)
    designer.add_component('validity', validity_reward, weight=0.3)
    
    results = designer.evaluate(test_state)
    print(f"   Component rewards: {results}")
    
    # 6. 偏好学习
    print("\n6. Preference Learning")
    pref_learning = PreferenceLearning(state_dim=10)
    
    # 添加一些模拟偏好
    for i in range(100):
        s1 = np.random.randn(10)
        s2 = np.random.randn(10)
        
        # 模拟偏好 (基于范数)
        pref = 0 if np.linalg.norm(s1) < np.linalg.norm(s2) else 1
        
        pref_learning.add_preference(s1, s2, pref)
    
    # 训练
    for epoch in range(5):
        stats = pref_learning.train_step(batch_size=32)
        if epoch == 0:
            print(f"   Initial loss: {stats['loss']:.3f}")
    
    print(f"   Final loss: {stats['loss']:.3f}")
    
    # 测试预测
    s1 = np.array([0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0])
    s2 = np.array([1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0, 0])
    prob, pref = pref_learning.predict_preference(s1, s2)
    print(f"   Predicted preference (s1 should be preferred): {pref} (prob={prob:.3f})")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
