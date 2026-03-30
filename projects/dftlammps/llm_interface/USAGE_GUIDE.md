# DFT-LAMMPS LLM集成模块使用指南

## 简介

本文档介绍如何使用DFT-LAMMPS的LLM集成模块进行材料科学研究。

## 安装

```bash
# 克隆仓库
git clone https://github.com/dft-lammps/dft-lammps.git
cd dft-lammps

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export OPENAI_API_KEY="your-api-key"
```

## 快速入门

### 1. 材料知识提取

```python
from dftlammps.llm_interface import quick_extract

text = """
We synthesized graphene oxide using modified Hummers method.
The material shows high specific surface area of 800 m²/g
and excellent electrochemical performance for supercapacitors.
"""

knowledge = quick_extract(text, source="J. Mater. Chem. 2024")
print(knowledge)
```

### 2. 代码生成

```python
from dftlammps.llm_interface import quick_generate

# 生成VASP输入
code = quick_generate(
    description="Band structure calculation for silicon with HSE06 functional",
    language="vasp",
    calc_type="bands"
)

print(code.main_input)
```

### 3. 智能问答

```python
from dftlammps.llm_interface import quick_chat

answer = quick_chat("What is the best k-point density for metals?")
print(answer)
```

### 4. 错误诊断

```python
from dftlammps.llm_interface import quick_diagnose

report = quick_diagnose(
    error_log="Error EDDDAV: Call to ZHEGV failed",
    software="vasp"
)
print(report.recommended_solutions)
```

## 高级用法

### 自定义LLM提供商

```python
from dftlammps.llm_interface import MaterialsGPT, LocalLLMProvider

# 使用本地LLM
provider = LocalLLMProvider(
    endpoint="http://localhost:8000",
    model_path="/path/to/model"
)

gpt = MaterialsGPT(provider=provider)
```

### 批量处理文献

```python
from dftlammps.llm_interface import LiteratureMiner

miner = LiteratureMiner(gpt)

literature = [
    {"text": "...", "source": "Paper 1"},
    {"text": "...", "source": "Paper 2"},
]

results = miner.mine_from_texts(literature)
```

### 构建知识图谱

```python
from dftlammps.knowledge_graph import create_default_knowledge_graph

kg = create_default_knowledge_graph()

# 添加文献知识
kg.ingest_text("Fe3O4 is a ferrimagnetic material with inverse spinel structure.")

# 查询
results = kg.query("magnetic")

# 导出
kg.export_to_json("knowledge_graph.json")
kg.export_to_neo4j_cypher("import.cypher")
```

## 示例脚本

### 示例1: 完整的计算工作流

```python
#!/usr/bin/env python3
"""
完整计算工作流示例
"""

from dftlammps.llm_interface import (
    MaterialsGPT, 
    CodeGenerator,
    ChatAssistant
)

# 1. 获取参数建议
gpt = MaterialsGPT()
params = gpt.recommend_computation_parameters(
    calc_type="geometry optimization",
    material="TiO2 anatase",
    software="VASP"
)

# 2. 生成输入文件
generator = CodeGenerator()
code = generator.generate_from_description(
    description="TiO2 anatase geometry optimization with VASP",
    language="vasp",
    calc_type="relax"
)

# 3. 检查输入
assistant = ChatAssistant()
review = assistant.chat("Review this VASP input for errors", 
                        context={"code": code.main_input})

# 4. 保存文件
generator.save_generated_files(code, "./TiO2_calculation")

print("工作流准备完成!")
```

### 示例2: 文献综述生成

```python
#!/usr/bin/env python3
"""
自动生成文献综述
"""

from dftlammps.llm_interface import MaterialsGPT
from dftlammps.knowledge_graph import LiteratureMiningPipeline

gpt = MaterialsGPT()
kg = create_default_knowledge_graph()

# 处理文献
pipeline = LiteratureMiningPipeline(kg)
stats = pipeline.process_literature(papers)

# 生成综述
research_trends = pipeline.identify_research_trends()

report = gpt.generate_report(
    report_type="literature_review",
    data={
        "papers_analyzed": stats["documents_processed"],
        "trends": research_trends,
        "key_materials": kg.find_entities_by_type(EntityType.MATERIAL)[:10]
    },
    requirements="Focus on emerging materials and methods"
)

with open("literature_review.txt", "w") as f:
    f.write(report)
```

## 故障排除

### 问题: API调用失败

**解决方案:**
- 检查API密钥是否正确设置
- 验证网络连接
- 查看API配额是否用完

### 问题: 生成的代码有错误

**解决方案:**
- 使用CodeGenerator的diagnose_error功能
- 查看生成的说明和警告
- 参考软件官方文档

### 问题: 知识图谱查询无结果

**解决方案:**
- 确认已添加数据到知识图谱
- 检查查询关键词
- 使用更宽泛的查询条件

## 最佳实践

1. **验证生成内容**: AI生成的内容应始终经过专家验证
2. **迭代优化**: 从简单系统开始，逐步增加复杂度
3. **记录日志**: 保留所有生成和修改记录
4. **定期更新**: 保持知识库和模型参数的最新状态

## 参考资源

- [VASP Wiki](https://www.vasp.at/wiki/index.php)
- [LAMMPS Documentation](https://docs.lammps.org/)
- [Quantum ESPRESSO](https://www.quantum-espresso.org/)
- [Materials Project](https://materialsproject.org/)

## 更新日志

### v1.0.0 (2025)
- 初始版本发布
- 支持MaterialsGPT、CodeGenerator、ChatAssistant
- 知识图谱功能
- 应用案例演示
