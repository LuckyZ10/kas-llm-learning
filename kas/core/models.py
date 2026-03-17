"""
KAS - Kimi Agent Studio
简单优先的Agent孵化平台
"""

__version__ = "0.1.0"
__author__ = "KAS Team"

# 保留核心数据结构定义（简化版）
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class CapabilityType(Enum):
    """能力类型"""
    CODE_REVIEW = "code_review"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"


@dataclass
class Capability:
    """能力定义"""
    name: str
    type: CapabilityType
    description: str
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)


@dataclass
class Agent:
    """Agent定义 - 简化版"""
    name: str
    version: str = "0.1.0"
    description: str = ""
    capabilities: List[Capability] = field(default_factory=list)
    system_prompt: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    created_from: str = ""  # 来源项目路径
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'capabilities': [
                {
                    'name': c.name,
                    'type': c.type.value,
                    'description': c.description,
                    'confidence': c.confidence
                }
                for c in self.capabilities
            ],
            'system_prompt': self.system_prompt,
            'model_config': self.model_config,
            'created_from': self.created_from
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Agent':
        """从字典创建"""
        capabilities = [
            Capability(
                name=c['name'],
                type=CapabilityType(c['type']),
                description=c['description'],
                confidence=c.get('confidence', 0.8)
            )
            for c in data.get('capabilities', [])
        ]
        
        return cls(
            name=data['name'],
            version=data.get('version', '0.1.0'),
            description=data.get('description', ''),
            capabilities=capabilities,
            system_prompt=data.get('system_prompt', ''),
            model_config=data.get('model_config', {}),
            created_from=data.get('created_from', '')
        )


# 简单规则引擎配置
DEFAULT_MODEL_CONFIG = {
    'temperature': 0.7,
    'max_tokens': 2000,
    'top_p': 0.9,
}

TASK_TYPE_PARAMS = {
    'simple': {'temperature': 0.3, 'max_tokens': 500},
    'standard': {'temperature': 0.7, 'max_tokens': 2000},
    'complex': {'temperature': 0.7, 'max_tokens': 4000},
    'creative': {'temperature': 0.9, 'max_tokens': 2000},
}


def get_params_for_task(task_type: str = 'standard') -> Dict[str, Any]:
    """根据任务类型获取参数 - 简单规则"""
    return TASK_TYPE_PARAMS.get(task_type, TASK_TYPE_PARAMS['standard'])


__all__ = [
    'Agent',
    'Capability',
    'CapabilityType',
    'get_params_for_task',
    'DEFAULT_MODEL_CONFIG',
]
