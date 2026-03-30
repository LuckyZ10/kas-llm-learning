"""
KAS DRL - State Space Definition
状态空间设计：Agent能力向量、任务特征、用户反馈
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AgentCapability(Enum):
    """Agent能力类型"""
    CODE_REVIEW = "code_review"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class TaskFeatures:
    """任务特征"""
    task_type: str
    complexity: float  # 0-1
    domain: str
    language: str
    file_count: int
    line_count: int
    dependencies: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def to_vector(self) -> np.ndarray:
        """转换为向量表示"""
        # One-hot编码task_type (假设有10种任务类型)
        task_type_vec = np.zeros(10)
        task_type_hash = hash(self.task_type) % 10
        task_type_vec[task_type_hash] = 1.0
        
        # 其他特征
        features = [
            self.complexity,
            min(self.file_count / 100.0, 1.0),  # 归一化
            min(self.line_count / 10000.0, 1.0),
            len(self.dependencies) / 20.0,
            len(self.constraints) / 10.0,
        ]
        
        return np.concatenate([task_type_vec, np.array(features)])


@dataclass
class UserFeedback:
    """用户反馈"""
    rating: float  # 1-5
    response_time: float  # 秒
    iterations: int  # 迭代次数
    accepted: bool
    modifications: int  # 用户修改次数
    
    def to_vector(self) -> np.ndarray:
        """转换为向量"""
        return np.array([
            self.rating / 5.0,
            min(self.response_time / 60.0, 1.0),  # 归一化到1分钟
            min(self.iterations / 10.0, 1.0),
            1.0 if self.accepted else 0.0,
            min(self.modifications / 20.0, 1.0),
        ])


@dataclass
class AgentState:
    """Agent状态"""
    capabilities: Dict[AgentCapability, float]  # 能力水平 0-1
    confidence: float
    success_rate: float
    experience_count: int
    
    def to_vector(self) -> np.ndarray:
        """转换为向量"""
        cap_vector = np.array([
            self.capabilities.get(cap, 0.0) 
            for cap in AgentCapability
        ])
        
        meta_vector = np.array([
            self.confidence,
            self.success_rate,
            min(self.experience_count / 1000.0, 1.0),
        ])
        
        return np.concatenate([cap_vector, meta_vector])


class StateEncoder(nn.Module):
    """状态编码器 - 将多模态状态编码为统一向量"""
    
    def __init__(
        self,
        agent_state_dim: int = 11,  # 8 capabilities + 3 meta
        task_dim: int = 15,         # 10 one-hot + 5 features
        feedback_dim: int = 5,
        hidden_dim: int = 256,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(agent_state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(
        self,
        agent_state: torch.Tensor,
        task_features: torch.Tensor,
        user_feedback: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            agent_state: [batch, agent_state_dim]
            task_features: [batch, task_dim]
            user_feedback: [batch, feedback_dim] or None
        Returns:
            state_vector: [batch, output_dim]
        """
        agent_vec = self.agent_encoder(agent_state)
        task_vec = self.task_encoder(task_features)
        
        if user_feedback is None:
            user_feedback = torch.zeros(agent_state.shape[0], 5, device=agent_state.device)
        
        feedback_vec = self.feedback_encoder(user_feedback)
        
        # 拼接所有特征
        combined = torch.cat([agent_vec, task_vec, feedback_vec], dim=-1)
        
        return self.fusion(combined)


class StateSpace:
    """状态空间管理器"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.encoder = StateEncoder().to(device)
        self.current_state = None
        self.state_history = []
        
    def encode(
        self,
        agent_state: AgentState,
        task_features: TaskFeatures,
        user_feedback: Optional[UserFeedback] = None
    ) -> torch.Tensor:
        """编码当前状态"""
        agent_vec = torch.FloatTensor(agent_state.to_vector()).unsqueeze(0).to(self.device)
        task_vec = torch.FloatTensor(task_features.to_vector()).unsqueeze(0).to(self.device)
        
        if user_feedback:
            feedback_vec = torch.FloatTensor(user_feedback.to_vector()).unsqueeze(0).to(self.device)
        else:
            feedback_vec = None
        
        with torch.no_grad():
            state = self.encoder(agent_vec, task_vec, feedback_vec)
        
        self.current_state = state
        self.state_history.append(state.cpu().numpy())
        
        # 保持历史记录长度
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        return state
    
    def get_state_dim(self) -> int:
        """获取状态维度"""
        return 128
    
    def reset(self):
        """重置状态"""
        self.current_state = None
        self.state_history = []
    
    def get_temporal_context(self, window_size: int = 10) -> torch.Tensor:
        """获取时序上下文"""
        if len(self.state_history) < window_size:
            # 填充
            padding = [self.state_history[0]] * (window_size - len(self.state_history)) if self.state_history else [np.zeros(128)]
            context = padding + self.state_history
        else:
            context = self.state_history[-window_size:]
        
        return torch.FloatTensor(np.array(context)).to(self.device)


class TelemetryStateTracker:
    """遥测数据状态追踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            'response_times': [],
            'success_rates': [],
            'user_ratings': [],
            'token_usage': [],
            'error_rates': []
        }
    
    def update(self, metrics: Dict[str, float]):
        """更新遥测指标"""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                if len(self.metrics[key]) > self.window_size:
                    self.metrics[key] = self.metrics[key][-self.window_size:]
    
    def get_summary(self) -> Dict[str, float]:
        """获取汇总统计"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_trend'] = self._calculate_trend(values)
        return summary
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势"""
        if len(values) < 2:
            return 0.0
        # 简单线性趋势
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def to_state_vector(self) -> np.ndarray:
        """转换为状态向量"""
        summary = self.get_summary()
        return np.array([
            summary.get('response_times_mean', 0),
            summary.get('response_times_std', 0),
            summary.get('response_times_trend', 0),
            summary.get('success_rates_mean', 0),
            summary.get('success_rates_std', 0),
            summary.get('success_rates_trend', 0),
            summary.get('user_ratings_mean', 0),
            summary.get('user_ratings_std', 0),
            summary.get('token_usage_mean', 0),
            summary.get('error_rates_mean', 0),
        ])
