"""
KAS DRL - Reward Function
奖励函数设计：用户满意度、能力保留率、收敛速度
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class RewardComponent(Enum):
    """奖励组件类型"""
    USER_SATISFACTION = "user_satisfaction"
    CAPABILITY_RETENTION = "capability_retention"
    CONVERGENCE_SPEED = "convergence_speed"
    RESPONSE_QUALITY = "response_quality"
    EFFICIENCY = "efficiency"
    EXPLORATION = "exploration"


@dataclass
class RewardConfig:
    """奖励配置"""
    # 权重
    user_satisfaction_weight: float = 0.35
    capability_retention_weight: float = 0.20
    convergence_speed_weight: float = 0.20
    response_quality_weight: float = 0.15
    efficiency_weight: float = 0.10
    
    # 折扣因子
    gamma: float = 0.99
    
    # 奖励裁剪
    clip_min: float = -10.0
    clip_max: float = 10.0
    
    # 稀疏奖励阈值
    sparse_threshold: float = 0.7


@dataclass
class InteractionOutcome:
    """交互结果"""
    # 用户满意度
    user_rating: float  # 1-5
    user_feedback_text: str = ""
    accepted: bool = False
    
    # 性能指标
    response_time: float  # 秒
    token_usage: int
    iteration_count: int
    
    # 质量指标
    accuracy: float  # 0-1
    completeness: float  # 0-1
    relevance: float  # 0-1
    
    # 能力变化
    capability_changes: Dict[str, float] = None
    
    def __post_init__(self):
        if self.capability_changes is None:
            self.capability_changes = {}


class RewardFunction:
    """奖励函数"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.reward_history = []
        self.component_history = {comp: [] for comp in RewardComponent}
    
    def compute(
        self,
        outcome: InteractionOutcome,
        previous_state: Optional[Dict[str, Any]] = None,
        current_state: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        计算总奖励
        
        Args:
            outcome: 交互结果
            previous_state: 之前的状态
            current_state: 当前状态
        
        Returns:
            总奖励值
        """
        components = {}
        
        # 1. 用户满意度奖励
        components[RewardComponent.USER_SATISFACTION] = self._compute_user_satisfaction(outcome)
        
        # 2. 能力保留率奖励
        components[RewardComponent.CAPABILITY_RETENTION] = self._compute_capability_retention(outcome)
        
        # 3. 收敛速度奖励
        components[RewardComponent.CONVERGENCE_SPEED] = self._compute_convergence_speed(outcome)
        
        # 4. 响应质量奖励
        components[RewardComponent.RESPONSE_QUALITY] = self._compute_response_quality(outcome)
        
        # 5. 效率奖励
        components[RewardComponent.EFFICIENCY] = self._compute_efficiency(outcome)
        
        # 计算加权总奖励
        total_reward = (
            self.config.user_satisfaction_weight * components[RewardComponent.USER_SATISFACTION] +
            self.config.capability_retention_weight * components[RewardComponent.CAPABILITY_RETENTION] +
            self.config.convergence_speed_weight * components[RewardComponent.CONVERGENCE_SPEED] +
            self.config.response_quality_weight * components[RewardComponent.RESPONSE_QUALITY] +
            self.config.efficiency_weight * components[RewardComponent.EFFICIENCY]
        )
        
        # 裁剪
        total_reward = np.clip(total_reward, self.config.clip_min, self.config.clip_max)
        
        # 记录历史
        self.reward_history.append(total_reward)
        for comp, value in components.items():
            self.component_history[comp].append(value)
        
        return total_reward
    
    def _compute_user_satisfaction(self, outcome: InteractionOutcome) -> float:
        """计算用户满意度奖励"""
        # 基础评分奖励 (归一化到-1到1)
        rating_reward = (outcome.user_rating - 3) / 2.0
        
        # 接受奖励
        acceptance_reward = 1.0 if outcome.accepted else -0.5
        
        # 综合
        satisfaction = 0.6 * rating_reward + 0.4 * acceptance_reward
        
        # 稀疏奖励：如果用户非常满意，给予额外奖励
        if outcome.user_rating >= 4.5 and outcome.accepted:
            satisfaction += 0.5
        
        return satisfaction
    
    def _compute_capability_retention(self, outcome: InteractionOutcome) -> float:
        """计算能力保留率奖励"""
        if not outcome.capability_changes:
            return 0.0
        
        # 计算能力变化
        positive_changes = sum(v for v in outcome.capability_changes.values() if v > 0)
        negative_changes = sum(abs(v) for v in outcome.capability_changes.values() if v < 0)
        
        # 鼓励正向变化，惩罚负向变化
        retention_score = positive_changes - 2.0 * negative_changes
        
        # 归一化
        return np.tanh(retention_score)
    
    def _compute_convergence_speed(self, outcome: InteractionOutcome) -> float:
        """计算收敛速度奖励"""
        # 迭代次数越少越好
        if outcome.iteration_count == 1:
            return 1.0  # 一次成功
        elif outcome.iteration_count <= 3:
            return 0.5
        elif outcome.iteration_count <= 5:
            return 0.0
        else:
            return -0.5 * (outcome.iteration_count - 5) / 5.0
    
    def _compute_response_quality(self, outcome: InteractionOutcome) -> float:
        """计算响应质量奖励"""
        quality = (
            outcome.accuracy * 0.4 +
            outcome.completeness * 0.3 +
            outcome.relevance * 0.3
        )
        
        # 归一化到-1到1
        return 2.0 * quality - 1.0
    
    def _compute_efficiency(self, outcome: InteractionOutcome) -> float:
        """计算效率奖励"""
        # 响应时间奖励 (假设60秒为正常)
        time_score = max(0, 1.0 - outcome.response_time / 60.0)
        
        # Token使用效率 (假设4000为上限)
        token_efficiency = 1.0 - min(outcome.token_usage / 4000.0, 1.0)
        
        return 0.5 * time_score + 0.5 * token_efficiency
    
    def get_statistics(self) -> Dict[str, float]:
        """获取奖励统计"""
        if not self.reward_history:
            return {}
        
        stats = {
            'mean_reward': np.mean(self.reward_history),
            'std_reward': np.std(self.reward_history),
            'min_reward': np.min(self.reward_history),
            'max_reward': np.max(self.reward_history),
            'recent_mean': np.mean(self.reward_history[-100:]) if len(self.reward_history) >= 100 else np.mean(self.reward_history)
        }
        
        # 各组件统计
        for comp, values in self.component_history.items():
            if values:
                stats[f'{comp.value}_mean'] = np.mean(values)
        
        return stats
    
    def reset(self):
        """重置历史"""
        self.reward_history = []
        self.component_history = {comp: [] for comp in RewardComponent}


class CurriculumReward:
    """课程学习奖励 - 逐步增加难度"""
    
    def __init__(self, base_reward_fn: RewardFunction, num_stages: int = 5):
        self.base_reward_fn = base_reward_fn
        self.num_stages = num_stages
        self.current_stage = 0
        self.stage_thresholds = np.linspace(0.3, 0.9, num_stages)
    
    def set_stage(self, stage: int):
        """设置当前阶段"""
        self.current_stage = min(stage, self.num_stages - 1)
    
    def compute(self, outcome: InteractionOutcome, **kwargs) -> float:
        """计算课程奖励"""
        base_reward = self.base_reward_fn.compute(outcome, **kwargs)
        
        # 根据阶段调整奖励
        threshold = self.stage_thresholds[self.current_stage]
        
        # 如果达到当前阶段要求，给予额外奖励
        if outcome.user_rating / 5.0 >= threshold:
            base_reward += 0.5 * (self.current_stage + 1) / self.num_stages
        
        return base_reward
    
    def check_stage_progression(self, recent_rewards: List[float]) -> bool:
        """检查是否应该进入下一阶段"""
        if len(recent_rewards) < 10:
            return False
        
        mean_recent = np.mean(recent_rewards[-10:])
        threshold = self.stage_thresholds[self.current_stage]
        
        return mean_recent > threshold


class PreferenceBasedReward:
    """基于偏好的奖励学习"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        self.reward_model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)
        self.preferences = []  # (state1, state2, preference) preference=1 if state1 > state2
    
    def predict(self, state: torch.Tensor) -> float:
        """预测奖励"""
        with torch.no_grad():
            return self.reward_model(state).item()
    
    def add_preference(self, state1: torch.Tensor, state2: torch.Tensor, preference: int):
        """添加偏好对比"""
        self.preferences.append((state1, state2, preference))
    
    def train_step(self) -> float:
        """训练奖励模型"""
        if len(self.preferences) < 10:
            return 0.0
        
        # 随机采样偏好对
        batch = np.random.choice(len(self.preferences), min(32, len(self.preferences)), replace=False)
        
        loss = 0.0
        for idx in batch:
            s1, s2, pref = self.preferences[idx]
            
            r1 = self.reward_model(s1)
            r2 = self.reward_model(s2)
            
            # Bradley-Terry模型
            if pref == 1:  # s1 > s2
                prob = torch.sigmoid(r1 - r2)
            else:  # s2 > s1
                prob = torch.sigmoid(r2 - r1)
            
            loss -= torch.log(prob + 1e-8)
        
        loss = loss / len(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class MultiObjectiveReward:
    """多目标奖励函数"""
    
    def __init__(self, objectives: List[str], weights: Optional[List[float]] = None):
        self.objectives = objectives
        self.weights = weights or [1.0 / len(objectives)] * len(objectives)
        
        self.reward_functions = {
            'user_satisfaction': self._user_satisfaction,
            'convergence': self._convergence,
            'efficiency': self._efficiency,
            'quality': self._quality,
            'diversity': self._diversity
        }
        
        self.pareto_front = []
    
    def compute(self, outcome: InteractionOutcome) -> np.ndarray:
        """计算多目标奖励向量"""
        rewards = []
        for obj in self.objectives:
            if obj in self.reward_functions:
                rewards.append(self.reward_functions[obj](outcome))
            else:
                rewards.append(0.0)
        
        return np.array(rewards)
    
    def _user_satisfaction(self, outcome: InteractionOutcome) -> float:
        return (outcome.user_rating - 3) / 2.0
    
    def _convergence(self, outcome: InteractionOutcome) -> float:
        return 1.0 - min(outcome.iteration_count / 10.0, 1.0)
    
    def _efficiency(self, outcome: InteractionOutcome) -> float:
        return max(0, 1.0 - outcome.response_time / 60.0)
    
    def _quality(self, outcome: InteractionOutcome) -> float:
        return (outcome.accuracy + outcome.completeness + outcome.relevance) / 3.0
    
    def _diversity(self, outcome: InteractionOutcome) -> float:
        # 基于历史计算的多样性奖励
        return 0.0  # 需要状态追踪
    
    def scalarize(self, reward_vector: np.ndarray, method: str = 'weighted_sum') -> float:
        """标量化多目标奖励"""
        if method == 'weighted_sum':
            return np.dot(reward_vector, self.weights)
        elif method == 'chebyshev':
            # Chebyshev距离
            ideal = np.ones(len(self.objectives))
            return -np.max(self.weights * np.abs(reward_vector - ideal))
        else:
            return np.dot(reward_vector, self.weights)
