"""
推理引擎模块
============

本模块实现了多种推理机制，支持材料科学研究的智能分析：

1. 逻辑推理
   - 演绎推理：从一般到特殊
   - 归纳推理：从特殊到一般
   - 溯因推理：最佳解释推理
   - 类比推理：跨领域知识迁移

2. 概率推理
   - 贝叶斯推理：证据更新信念
   - 因果推理：干预与效果分析
   - 蒙特卡洛方法：采样估计

3. 多跳推理
   - 知识图谱路径搜索
   - 路径排序算法
   - 神经符号推理

使用示例:
---------

```python
from dftlammps.reasoning import ReasoningEngine, Fact, Rule

# 创建推理引擎
engine = ReasoningEngine()

# 添加事实
facts = [
    Fact(id="f1", statement="High-entropy alloys have multiple principal elements"),
    Fact(id="f2", statement="Multiple elements create high configurational entropy")
]

# 混合推理
result = engine.hybrid_reasoning(
    facts=facts,
    query="Is HEA stable?",
    evidence={"experiment": "passed_test"}
)
```
"""

from . import (
    # Core classes
    ReasoningEngine,
    LogicalReasoner,
    ProbabilisticReasoner,
    MultiHopReasoner,
    
    # Data classes
    Fact,
    Rule,
    Belief,
    ReasoningChain,
    KnowledgeNode,
    KnowledgeEdge,
    
    # Enums
    InferenceType,
    LogicOperator
)

__all__ = [
    "ReasoningEngine",
    "LogicalReasoner",
    "ProbabilisticReasoner",
    "MultiHopReasoner",
    "Fact",
    "Rule",
    "Belief",
    "ReasoningChain",
    "KnowledgeNode",
    "KnowledgeEdge",
    "InferenceType",
    "LogicOperator"
]
