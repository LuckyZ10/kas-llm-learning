"""
KAS DRL - Action Space Definition
动作空间设计：Prompt调整、模板选择、参数优化
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class PromptAdjustment(Enum):
    """Prompt调整类型"""
    EXPAND = "expand"           # 扩展细节
    CONDENSE = "condense"       # 精简
    RESTRUCTURE = "restructure" # 重组结构
    ADD_EXAMPLE = "add_example" # 添加示例
    REMOVE_EXAMPLE = "remove_example"  # 移除示例
    CHANGE_TONE = "change_tone" # 改变语气
    ADD_CONSTRAINT = "add_constraint"  # 添加约束
    REMOVE_CONSTRAINT = "remove_constraint"  # 移除约束


class TemplateType(Enum):
    """模板类型"""
    CONCISE = "concise"         # 简洁
    DETAILED = "detailed"       # 详细
    STEP_BY_STEP = "step_by_step"  # 步骤式
    FEW_SHOT = "few_shot"       # 少样本
    CHAIN_OF_THOUGHT = "chain_of_thought"  # 思维链
    STRUCTURED = "structured"   # 结构化


@dataclass
class PromptAction:
    """Prompt调整动作"""
    adjustment: PromptAdjustment
    strength: float  # 0-1, 调整强度
    target_section: Optional[str] = None  # 目标段落
    
    def to_vector(self) -> np.ndarray:
        """转换为向量"""
        # One-hot编码adjustment
        adj_vec = np.zeros(len(PromptAdjustment))
        adj_vec[list(PromptAdjustment).index(self.adjustment)] = 1.0
        
        return np.concatenate([
            adj_vec,
            np.array([self.strength])
        ])
    
    @staticmethod
    def from_vector(vec: np.ndarray) -> 'PromptAction':
        """从向量解析"""
        adj_idx = np.argmax(vec[:len(PromptAdjustment)])
        adjustment = list(PromptAdjustment)[adj_idx]
        strength = np.clip(vec[len(PromptAdjustment)], 0, 1)
        
        return PromptAction(
            adjustment=adjustment,
            strength=float(strength)
        )


@dataclass
class TemplateAction:
    """模板选择动作"""
    template: TemplateType
    confidence: float  # 0-1
    
    def to_vector(self) -> np.ndarray:
        """转换为向量"""
        # One-hot编码template
        temp_vec = np.zeros(len(TemplateType))
        temp_vec[list(TemplateType).index(self.template)] = 1.0
        
        return np.concatenate([
            temp_vec,
            np.array([self.confidence])
        ])
    
    @staticmethod
    def from_vector(vec: np.ndarray) -> 'TemplateAction':
        """从向量解析"""
        temp_idx = np.argmax(vec[:len(TemplateType)])
        template = list(TemplateType)[temp_idx]
        confidence = np.clip(vec[len(TemplateType)], 0, 1)
        
        return TemplateAction(
            template=template,
            confidence=float(confidence)
        )


@dataclass
class ParameterAction:
    """参数优化动作"""
    temperature: float      # 0-2
    max_tokens: int         # 1-4096
    top_p: float           # 0-1
    frequency_penalty: float # -2 to 2
    presence_penalty: float  # -2 to 2
    
    def to_vector(self) -> np.ndarray:
        """转换为向量 (归一化到0-1)"""
        return np.array([
            self.temperature / 2.0,
            min(self.max_tokens / 4096.0, 1.0),
            self.top_p,
            (self.frequency_penalty + 2) / 4.0,
            (self.presence_penalty + 2) / 4.0,
        ])
    
    @staticmethod
    def from_vector(vec: np.ndarray) -> 'ParameterAction':
        """从向量解析"""
        return ParameterAction(
            temperature=float(np.clip(vec[0] * 2.0, 0, 2)),
            max_tokens=int(np.clip(vec[1] * 4096, 1, 4096)),
            top_p=float(np.clip(vec[2], 0, 1)),
            frequency_penalty=float(np.clip(vec[3] * 4.0 - 2.0, -2, 2)),
            presence_penalty=float(np.clip(vec[4] * 4.0 - 2.0, -2, 2))
        )
    
    def to_llm_params(self) -> Dict[str, Any]:
        """转换为LLM调用参数"""
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty
        }


class ActionSpace:
    """动作空间"""
    
    def __init__(self):
        self.prompt_dim = len(PromptAdjustment) + 1  # adjustment + strength
        self.template_dim = len(TemplateType) + 1    # template + confidence
        self.param_dim = 5  # temperature, max_tokens, top_p, freq_pen, pres_pen
        
        self.total_dim = self.prompt_dim + self.template_dim + self.param_dim
    
    def get_dimensions(self) -> Dict[str, int]:
        """获取各维度大小"""
        return {
            'prompt': self.prompt_dim,
            'template': self.template_dim,
            'param': self.param_dim,
            'total': self.total_dim
        }
    
    def decode(self, action_vector: np.ndarray) -> Tuple[PromptAction, TemplateAction, ParameterAction]:
        """解码动作向量"""
        prompt_vec = action_vector[:self.prompt_dim]
        template_vec = action_vector[self.prompt_dim:self.prompt_dim + self.template_dim]
        param_vec = action_vector[self.prompt_dim + self.template_dim:]
        
        return (
            PromptAction.from_vector(prompt_vec),
            TemplateAction.from_vector(template_vec),
            ParameterAction.from_vector(param_vec)
        )
    
    def encode(
        self,
        prompt_action: PromptAction,
        template_action: TemplateAction,
        param_action: ParameterAction
    ) -> np.ndarray:
        """编码动作"""
        return np.concatenate([
            prompt_action.to_vector(),
            template_action.to_vector(),
            param_action.to_vector()
        ])
    
    def sample_random(self) -> np.ndarray:
        """随机采样动作"""
        return np.random.rand(self.total_dim)
    
    def sample_valid(self) -> Tuple[PromptAction, TemplateAction, ParameterAction]:
        """采样有效动作"""
        # 随机选择调整类型
        prompt_adj = np.random.choice(list(PromptAdjustment))
        prompt_action = PromptAction(
            adjustment=prompt_adj,
            strength=np.random.rand()
        )
        
        # 随机选择模板
        template = np.random.choice(list(TemplateType))
        template_action = TemplateAction(
            template=template,
            confidence=np.random.rand()
        )
        
        # 随机参数
        param_action = ParameterAction(
            temperature=np.random.rand() * 2.0,
            max_tokens=int(np.random.rand() * 4000) + 96,
            top_p=np.random.rand(),
            frequency_penalty=np.random.rand() * 4.0 - 2.0,
            presence_penalty=np.random.rand() * 4.0 - 2.0
        )
        
        return prompt_action, template_action, param_action


class ActionNetwork(nn.Module):
    """动作生成网络"""
    
    def __init__(
        self,
        state_dim: int = 128,
        hidden_dim: int = 256,
        action_dim: int = 17  # 9 + 7 + 5
    ):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Prompt调整头
        self.prompt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 9),  # 8 adjustments + strength
            nn.Sigmoid()
        )
        
        # 模板选择头
        self.template_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 7),  # 6 templates + confidence
            nn.Sigmoid()
        )
        
        # 参数优化头
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),  # 5 parameters
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [batch, state_dim]
        Returns:
            prompt_action, template_action, param_action
        """
        shared_features = self.shared(state)
        
        prompt = self.prompt_head(shared_features)
        template = self.template_head(shared_features)
        params = self.param_head(shared_features)
        
        return prompt, template, params
    
    def get_action(self, state: torch.Tensor) -> np.ndarray:
        """获取动作向量"""
        with torch.no_grad():
            prompt, template, params = self.forward(state)
        
        return torch.cat([prompt, template, params], dim=-1).cpu().numpy()


class HierarchicalActionSpace:
    """分层动作空间 - 高层策略选择任务类型，低层策略执行具体动作"""
    
    def __init__(self):
        self.task_types = [
            'code_generation',
            'code_review',
            'refactoring',
            'documentation',
            'debugging',
            'architecture'
        ]
        
        self.base_action_space = ActionSpace()
    
    def get_high_level_dim(self) -> int:
        """高层动作维度"""
        return len(self.task_types)
    
    def get_low_level_dim(self) -> int:
        """低层动作维度"""
        return self.base_action_space.total_dim
    
    def decode_high_level(self, action: int) -> str:
        """解码高层动作"""
        return self.task_types[action]


class AdaptiveActionMask:
    """自适应动作掩码 - 根据当前状态屏蔽无效动作"""
    
    def __init__(self):
        self.action_space = ActionSpace()
    
    def get_mask(
        self,
        state: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """
        生成动作掩码
        Returns: [action_dim] 0=无效, 1=有效
        """
        mask = torch.ones(self.action_space.total_dim)
        
        # 根据context屏蔽动作
        if context.get('no_examples', False):
            # 屏蔽添加/移除示例
            idx = list(PromptAdjustment).index(PromptAdjustment.ADD_EXAMPLE)
            mask[idx] = 0
            idx = list(PromptAdjustment).index(PromptAdjustment.REMOVE_EXAMPLE)
            mask[idx] = 0
        
        if context.get('simple_task', False):
            # 简单任务禁用复杂模板
            idx = list(TemplateType).index(TemplateType.CHAIN_OF_THOUGHT)
            mask[self.action_space.prompt_dim + idx] = 0
        
        return mask
    
    def apply_mask(
        self,
        action: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """应用掩码"""
        return action * mask
