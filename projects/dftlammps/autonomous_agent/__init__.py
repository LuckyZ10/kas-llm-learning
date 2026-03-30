"""
自主智能体系统模块
==================

本模块实现了自主材料科学家的核心功能，包括：

1. agent_core.py - 智能体核心
   - 目标理解与分解
   - 自主规划与决策
   - 工具调用与执行
   - 自我反思与修正

2. experiment_planner.py - 实验规划器
   - 假设生成
   - 实验设计优化
   - 资源分配
   - 风险评估

3. literature_agent.py - 文献智能体
   - 自动文献检索
   - 知识提取与整合
   - 研究空白识别

使用示例:
---------

```python
from dftlammps.autonomous_agent import AutonomousAgent, ExperimentPlanner

# 创建自主智能体
agent = AutonomousAgent("MaterialScientist")

# 运行完整流程
result = await agent.run(
    goal_description="发现用于水分解的高效催化剂",
    criteria={
        "requires_experiment": True,
        "target_overpotential": 0.3
    }
)
```
"""

from .agent_core import (
    AutonomousAgent,
    AgentState,
    Task,
    TaskStatus,
    Goal,
    Reflection,
    ToolCall,
    BaseTool,
    ToolRegistry,
    Memory,
    GoalDecomposer,
    Planner,
    Reflector,
    DFTTool,
    LAMMPTool,
    StructureAnalysisTool
)

from .experiment_planner import (
    ExperimentPlanner,
    HypothesisGenerator,
    DesignOptimizer,
    ResourceScheduler,
    RiskAssessor,
    Hypothesis,
    ExperimentDesign,
    ExperimentalVariable,
    ResourceAllocation,
    Risk,
    ExperimentType,
    HypothesisStatus,
    ResourceType
)

from .literature_agent import (
    LiteratureAgent,
    Paper,
    KnowledgeUnit,
    ResearchGap,
    ResearchTrend,
    LiteratureSource,
    KnowledgeType,
    BaseLiteratureSource,
    KnowledgeExtractor,
    KnowledgeIntegrator,
    GapAnalyzer
)

__all__ = [
    # Agent Core
    "AutonomousAgent",
    "AgentState",
    "Task",
    "TaskStatus",
    "Goal",
    "Reflection",
    "ToolCall",
    "BaseTool",
    "ToolRegistry",
    "Memory",
    "GoalDecomposer",
    "Planner",
    "Reflector",
    "DFTTool",
    "LAMMPTool",
    "StructureAnalysisTool",
    
    # Experiment Planner
    "ExperimentPlanner",
    "HypothesisGenerator",
    "DesignOptimizer",
    "ResourceScheduler",
    "RiskAssessor",
    "Hypothesis",
    "ExperimentDesign",
    "ExperimentalVariable",
    "ResourceAllocation",
    "Risk",
    "ExperimentType",
    "HypothesisStatus",
    "ResourceType",
    
    # Literature Agent
    "LiteratureAgent",
    "Paper",
    "KnowledgeUnit",
    "ResearchGap",
    "ResearchTrend",
    "LiteratureSource",
    "KnowledgeType",
    "BaseLiteratureSource",
    "KnowledgeExtractor",
    "KnowledgeIntegrator",
    "GapAnalyzer"
]
