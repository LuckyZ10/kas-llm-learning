# 自主材料科学家系统

## 概述

这是一个基于AI的自主材料科学家系统，实现了从目标理解、文献调研、假设生成、实验规划到自主执行的完整闭环材料发现流程。

## 功能特性

### 1. 自主智能体 (autonomous_agent/)

#### Agent Core (agent_core.py)
- **目标理解与分解**: 自动分析研究目标并分解为可执行任务
- **自主规划与决策**: 基于约束和目标创建最优执行计划
- **工具调用与执行**: 集成DFT、MD等计算工具
- **自我反思与修正**: 执行后分析并改进策略

#### Experiment Planner (experiment_planner.py)
- **假设生成**: 基于知识库和模式识别生成科学假设
- **实验设计优化**: 使用贝叶斯优化、LHS等方法
- **资源分配**: 智能调度计算资源
- **风险评估**: 识别潜在风险并制定缓解策略

#### Literature Agent (literature_agent.py)
- **自动文献检索**: 支持多源文献搜索
- **知识提取与整合**: 从文献中提取结构化知识
- **研究空白识别**: 自动识别研究领域空白

### 2. 推理引擎 (reasoning/)

- **逻辑推理**: 演绎、归纳、溯因、类比推理
- **概率推理**: 贝叶斯更新、因果推断
- **多跳推理**: 知识图谱上的多步推理

### 3. 应用案例

- **自主发现新催化剂**: 完整的水分解催化剂发现流程
- **自主优化合成路线**: 自动化合成参数优化
- **自主撰写研究报告**: 自动分析结果并生成报告

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd dftlammps

# 安装依赖
pip install -r requirements.txt
```

### 基础使用

```python
import asyncio
from dftlammps.autonomous_agent import AutonomousAgent

async def main():
    # 创建自主智能体
    agent = AutonomousAgent("MyScientist")
    
    # 运行完整流程
    result = await agent.run(
        goal_description="发现用于水分解的高效催化剂",
        criteria={
            "requires_experiment": True,
            "target_overpotential": 0.3
        }
    )
    
    print(f"成功率: {result['success_rate']:.2%}")
    print(f"目标达成: {result['goal_achieved']}")

asyncio.run(main())
```

### 实验规划

```python
from dftlammps.autonomous_agent import ExperimentPlanner

async def plan_experiment():
    planner = ExperimentPlanner()
    
    plan = await planner.create_experiment_plan(
        research_question="发现水分解催化剂",
        available_resources={
            ResourceType.CPU: 200,
            ResourceType.GPU: 20
        },
        constraints={
            "max_budget": 10000,
            "num_hypotheses": 5
        }
    )
    
    print(f"预计成本: ${plan['total_estimated_cost']:.2f}")
    print(f"风险分数: {plan['overall_risk_score']:.2f}")
```

### 文献调研

```python
from dftlammps.autonomous_agent import LiteratureAgent

async def research():
    agent = LiteratureAgent()
    
    # 搜索文献
    papers = await agent.search_literature(
        query="catalyst water splitting",
        max_results=20
    )
    
    # 提取知识
    knowledge = await agent.extract_knowledge(papers)
    
    # 识别研究空白
    gaps = await agent.identify_research_gaps()
```

### 推理引擎

```python
from dftlammps.reasoning import ReasoningEngine, Fact

# 创建推理引擎
engine = ReasoningEngine()

# 添加事实
facts = [
    Fact(id="f1", statement="HEAs have high configurational entropy"),
    Fact(id="f2", statement="High entropy stabilizes solid solutions")
]

# 混合推理
result = engine.hybrid_reasoning(
    facts=facts,
    query="Are HEAs stable?",
    evidence={"temperature": 1000}
)
```

## 示例运行

### 运行催化剂发现示例

```bash
cd dftlammps/examples/autonomous_examples
python catalyst_discovery.py
```

## 系统架构

```
dftlammps/
├── autonomous_agent/          # 自主智能体模块
│   ├── agent_core.py          # 智能体核心
│   ├── experiment_planner.py  # 实验规划器
│   ├── literature_agent.py    # 文献智能体
│   └── __init__.py
├── reasoning/                 # 推理引擎
│   └── __init__.py
└── examples/                  # 应用示例
    └── autonomous_examples/
        └── catalyst_discovery.py
```

## 核心概念

### 智能体状态

- `IDLE`: 空闲状态
- `PLANNING`: 规划中
- `EXECUTING`: 执行中
- `REFLECTING`: 反思中
- `LEARNING`: 学习中

### 任务状态

- `PENDING`: 待处理
- `IN_PROGRESS`: 进行中
- `COMPLETED`: 已完成
- `FAILED`: 失败
- `WAITING_FOR_DEPENDENCY`: 等待依赖

### 推理类型

- `DEDUCTIVE`: 演绎推理
- `INDUCTIVE`: 归纳推理
- `ABDUCTIVE`: 溯因推理
- `ANALOGICAL`: 类比推理
- `CAUSAL`: 因果推理
- `PROBABILISTIC`: 概率推理
- `MULTI_HOP`: 多跳推理

## 配置选项

### 智能体配置

```python
agent = AutonomousAgent(
    name="CustomAgent",
    # 自定义配置
)
```

### 规划器配置

```python
planner = ExperimentPlanner(
    # 配置优化算法、约束等
)
```

## 扩展开发

### 添加自定义工具

```python
from dftlammps.autonomous_agent import BaseTool

class CustomTool(BaseTool):
    def __init__(self):
        super().__init__("custom_tool", "描述")
    
    async def execute(self, **kwargs):
        # 实现工具逻辑
        return result

# 注册工具
agent.register_tool(CustomTool())
```

### 添加自定义推理规则

```python
from dftlammps.reasoning import Rule, Fact

# 创建规则
rule = Rule(
    id="custom_rule",
    premises=[Fact(...), Fact(...)],
    conclusion=Fact(...),
    confidence=0.9
)

engine.logical_reasoner.add_rule(rule)
```

## API参考

详细的API文档请参考各模块的docstring。

### 主要类和方法

#### AutonomousAgent
- `understand_goal()`: 理解目标
- `decompose_goal()`: 分解目标
- `plan()`: 规划执行
- `execute_task()`: 执行任务
- `reflect()`: 自我反思
- `run()`: 完整执行流程

#### ExperimentPlanner
- `create_experiment_plan()`: 创建实验计划
- `get_plan_summary()`: 获取计划摘要

#### LiteratureAgent
- `search_literature()`: 搜索文献
- `extract_knowledge()`: 提取知识
- `identify_research_gaps()`: 识别研究空白

#### ReasoningEngine
- `hybrid_reasoning()`: 混合推理
- `scientific_reasoning()`: 科学推理

## 贡献指南

欢迎贡献代码和想法！请参考以下步骤：

1. Fork 仓库
2. 创建特性分支
3. 提交更改
4. 创建 Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或联系开发团队。

## 致谢

感谢所有贡献者和支持本项目的研究机构。
