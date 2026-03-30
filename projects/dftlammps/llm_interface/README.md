# 材料科学LLM集成与智能助手模块文档

## 概述

本模块提供大语言模型(LLM)与材料科学计算软件的集成，支持文献知识提取、代码生成、实验设计辅助等功能。

## 模块结构

```
dftlammps/llm_interface/
├── __init__.py              # 模块初始化
├── materials_gpt.py         # 材料科学GPT
├── code_generator.py        # 智能代码生成
├── chat_assistant.py        # 交互式助手
└── application_examples.py  # 应用案例

dftlammps/knowledge_graph/
├── __init__.py              # 模块初始化
├── kg_core.py              # 知识图谱核心
└── kg_init.py              # 本体初始化
```

## 核心功能

### 1. MaterialsGPT - 材料科学GPT

提供文献知识提取、实验设计、参数推荐和报告生成。

```python
from dftlammps.llm_interface import MaterialsGPT, OpenAIProvider

# 创建GPT实例
gpt = MaterialsGPT(provider=OpenAIProvider(api_key="your_key"))

# 文献知识提取
knowledge = gpt.extract_knowledge_from_literature(text, source="Paper 2024")

# 实验设计
design = gpt.design_experiment(
    objective="Synthesize perovskite solar cell",
    materials=["CsI", "PbI2"],
    constraints={"temperature": "< 200°C"}
)

# 参数推荐
params = gpt.recommend_computation_parameters(
    calc_type="band structure",
    material="Si",
    software="VASP"
)

# 结果解释
interpretation = gpt.interpret_results(results_data, context="Band calculation")

# 报告生成
report = gpt.generate_report("experimental", data, requirements="Include methodology")
```

### 2. CodeGenerator - 智能代码生成

从自然语言生成DFT/MD输入文件，自动修复错误，构建工作流。

```python
from dftlammps.llm_interface import CodeGenerator, CodeLanguage, CalculationType

# 创建代码生成器
generator = CodeGenerator()

# 生成输入文件
code = generator.generate_from_description(
    description="VASP SCF calculation for Si with high accuracy",
    language=CodeLanguage.VASP,
    calc_type=CalculationType.SCF,
    context={"material": "Si", "encut": 520}
)

# 保存生成的文件
generator.save_generated_files(code, output_dir="./input_files")

# 诊断和修复错误
fixes = generator.diagnose_error(
    error_log="Error EDDDAV: Call to ZHEGV failed",
    language=CodeLanguage.VASP,
    current_input=incar_content
)

# 构建工作流
workflow = generator.build_workflow(
    workflow_type="band_structure",
    stages=[
        {"name": "scf", "language": "vasp", "calc_type": "scf"},
        {"name": "bands", "language": "vasp", "calc_type": "bands"}
    ],
    dependencies={1: [0]}
)
```

### 3. ChatAssistant - 交互式助手

提供问答系统、实时计算指导和故障诊断。

```python
from dftlammps.llm_interface import ChatAssistant, ExpertiseLevel

# 创建助手
assistant = ChatAssistant()

# 聊天
response = assistant.chat(
    "What ENCUT should I use for Fe?",
    session_id="user_123",
    context={"user_level": ExpertiseLevel.INTERMEDIATE}
)
print(response.answer)

# 诊断问题
diagnosis = assistant.diagnose(
    error_log="ZHEGV failed",
    software="vasp",
    input_files={"INCAR": incar_content}
)

# 获取实时指导
guidance = assistant.get_guidance(
    current_step="scf_initialization",
    calculation_type="vasp"
)
```

### 4. KnowledgeGraph - 知识图谱

材料实体抽取、关系建模和知识推理。

```python
from dftlammps.knowledge_graph import (
    KnowledgeGraph, 
    create_default_knowledge_graph,
    LiteratureMiningPipeline
)

# 创建知识图谱
kg = create_default_knowledge_graph()

# 从文献提取知识
stats = kg.ingest_text(
    text="CsPbI3 has a band gap of 1.73 eV",
    source="Nature 2024"
)

# 查询知识
results = kg.query("perovskite")

# 推理
inferences = kg.reason("similar materials", entity_id="mat_001")

# 查找相似材料
similar = kg.find_similar_materials("CsPbI3", similarity_threshold=0.5)

# 文献挖掘流水线
pipeline = LiteratureMiningPipeline(kg)
stats = pipeline.process_literature([
    {"text": "...", "source": "Paper 1"},
    {"text": "...", "source": "Paper 2"}
])
```

## 应用案例

### 案例1: 文献挖掘发现新材料

```python
from dftlammps.llm_interface import LiteratureMiningExample

example = LiteratureMiningExample()
example.demonstrate_literature_mining()
```

### 案例2: 自然语言设计计算流程

```python
from dftlammps.llm_interface import NLWorkflowDesignExample

example = NLWorkflowDesignExample()
example.demonstrate_nl_workflow()
```

### 案例3: 智能实验日记生成

```python
from dftlammps.llm_interface import SmartLabNotebookExample

example = SmartLabNotebookExample()
example.demonstrate_smart_notebook()
```

## 快速开始

```python
# 运行所有示例
from dftlammps.llm_interface import IntegrationExample

examples = IntegrationExample()
examples.run_all_examples()
```

## API参考

### MaterialsGPT

| 方法 | 描述 |
|------|------|
| `extract_knowledge_from_literature()` | 从文献提取知识 |
| `design_experiment()` | 设计实验方案 |
| `recommend_computation_parameters()` | 推荐计算参数 |
| `interpret_results()` | 解释计算结果 |
| `generate_report()` | 生成科学报告 |
| `discover_new_materials()` | 发现潜在新材料 |

### CodeGenerator

| 方法 | 描述 |
|------|------|
| `generate_from_description()` | 从描述生成代码 |
| `diagnose_error()` | 诊断并修复错误 |
| `build_workflow()` | 构建计算工作流 |
| `save_generated_files()` | 保存生成的文件 |

### ChatAssistant

| 方法 | 描述 |
|------|------|
| `chat()` | 处理用户消息 |
| `diagnose()` | 诊断计算问题 |
| `get_guidance()` | 获取实时指导 |

### KnowledgeGraph

| 方法 | 描述 |
|------|------|
| `ingest_text()` | 从文本提取知识 |
| `query()` | 查询知识 |
| `reason()` | 知识推理 |
| `find_similar_materials()` | 查找相似材料 |
| `export_to_json()` | 导出到JSON |
| `export_to_neo4j_cypher()` | 导出到Neo4j |

## 依赖要求

```
# 核心依赖
python >= 3.8

# LLM支持 (可选)
openai >= 1.0.0

# 数据处理
numpy
pandas

# 可选依赖
pymatgen  # 材料分析
ase       # 原子模拟
```

## 配置

### 环境变量

```bash
# OpenAI API密钥
export LLM_API_KEY="your-openai-api-key"

# 本地LLM端点 (可选)
export LOCAL_LLM_ENDPOINT="http://localhost:8000"
```

### 配置文件

```python
# config.py
LLM_CONFIG = {
    "default_provider": "openai",
    "openai": {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "local": {
        "endpoint": "http://localhost:8000",
        "model": "llama-2-70b"
    }
}
```

## 最佳实践

1. **代码生成**: 始终验证生成的输入文件，测试小系统后再扩展
2. **参数推荐**: 根据目标性质和计算资源调整推荐参数
3. **知识图谱**: 定期备份和更新知识库
4. **错误诊断**: 结合人工判断，AI建议仅供参考

## 常见问题

### Q: 如何在没有API密钥的情况下使用？
A: 模块提供模拟模式，使用 `LocalLLMProvider` 或在无API密钥时自动启用mock生成。

### Q: 支持哪些DFT软件？
A: 目前支持VASP、Quantum ESPRESSO、CP2K、LAMMPS等主流软件。

### Q: 知识图谱数据存储在哪里？
A: 默认存储在内存中，可以使用 `export_to_json()` 持久化到文件。

## 开发计划

- [ ] 支持更多LLM提供商 (Claude, Gemini等)
- [ ] 集成图形化界面
- [ ] 扩展更多计算方法支持
- [ ] 增强知识图谱推理能力
- [ ] 添加多语言支持

## 贡献

欢迎贡献代码和反馈问题！

## 许可证

MIT License

## 联系我们

- 项目主页: https://github.com/dft-lammps/llm-interface
- 问题反馈: https://github.com/dft-lammps/llm-interface/issues
- 文档: https://dft-lammps.readthedocs.io
