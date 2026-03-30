# 神经符号AI与混合推理模块

## 概述

本模块实现神经符号AI（Neuro-Symbolic AI）与混合推理系统，结合神经网络的模式识别能力与符号系统的可解释推理能力，为材料科学提供可解释的AI解决方案。

## 模块结构

```
dftlammps/
├── neuro_symbolic/          # 神经符号AI模块
│   ├── neural_prolog.py     # 神经Prolog与知识图谱推理
│   ├── symbolic_nn.py       # 符号神经网络与双向翻译
│   ├── program_synthesis.py # 程序合成与代码生成
│   ├── applications.py      # 应用案例
│   └── __init__.py          # 模块导出
│
└── knowledge_reasoning/     # 知识推理模块
    ├── ontology_reasoning.py # 本体推理与描述逻辑
    ├── rule_engine.py       # 产生式规则引擎
    ├── case_based_reasoning.py # 案例推理
    └── __init__.py          # 模块导出
```

## 核心功能

### 1. 神经Prolog (neural_prolog.py)

**可微分逻辑编程**
- `NeuralUnification`: 神经合一模块，使用神经网络学习项之间的相似度
- `DifferentiableLogicLayer`: 可微分逻辑层，实现模糊逻辑运算
- `NeuralLogicProgram`: 神经逻辑程序，端到端可微的推理

**神经定理证明**
- `NeuralTheoremProver`: 神经定理证明器
  - 前向链式推理：从已知事实推导新结论
  - 后向链式推理：从目标反向寻找证明
  - 支持可微分的证明搜索

**知识图谱推理**
- `KnowledgeGraphReasoner`: 知识图谱推理
  - 多跳推理与路径查找
  - 神经引导的路径排序
  - 从知识图谱自动学习规则

### 2. 符号神经网络 (symbolic_nn.py)

**双向翻译**
- `NeuralToSymbolicTranslator`: 神经到符号翻译器
  - 使用Gumbel-Softmax实现可微分离散化
  - 从神经网络输出提取符号规则
  
- `SymbolicToNeuralTranslator`: 符号到神经翻译器
  - 符号图神经网络编码
  - 知识图谱嵌入

**概念学习**
- `ConceptLearner`: 概念学习器
  - 自动学习概念层次结构
  - 概念原型学习
  - 关系发现

**层次表示学习**
- `HierarchicalRepresentationLearning`: 层次表示学习
  - 从低级特征到高级抽象
  - 跨层次插值与类比推理

**双向神经符号系统**
- `BidirectionalNeuralSymbolic`: 整合神经到符号和符号到神经的转换
  - 循环一致性约束
  - 符号推理网络
  - 可解释的预测说明

### 3. 程序合成 (program_synthesis.py)

**AST组件**
- `ASTNode`: 抽象语法树节点
- `ASTToCodeConverter`: AST到Python代码转换

**神经程序合成**
- `NeuralProgramSynthesizer`: 基于Transformer的程序合成
  - 从输入-输出示例学习程序
  - 束搜索解码
  - 支持多种程序结构

**代码补全**
- `CodeCompletionModel`: 代码补全模型
  - 基于Transformer的代码生成
  - Top-p (nucleus) 采样

**程序验证**
- `ProgramVerifier`: 程序验证器
  - 自动测试用例生成
  - 安全执行环境
  - 错误报告

**搜索基础合成**
- `SearchBasedSynthesizer`: 基于搜索的程序合成
  - 枚举搜索
  - 遗传算法搜索

**材料DSL**
- `MaterialDSL`: 材料科学领域特定语言
  - 属性过滤
  - 排序与相似度计算
  - 属性预测

### 4. 本体推理 (ontology_reasoning.py)

**概念系统**
- `Concept`: 本体概念
- `Role`: 本体角色/关系
- `Individual`: 本体实例
- `ConceptHierarchy`: 概念层次结构管理

**描述逻辑推理**
- `DescriptionLogicReasoner`: 描述逻辑推理器
  - 概念包含检查（Subsumption）
  - 概念等价与不相交检查
  - 可满足性检查
  - 实例分类
  - 一致性检查
  - 属性推断

**神经本体推理**
- `NeuralOntologyReasoner`: 神经本体推理器
  - 学习的概念嵌入
  - 神经包含概率
  - 模糊分类

### 5. 规则引擎 (rule_engine.py)

**不确定性推理**
- `CertaintyFactor`: 确定性因子
  - 证据组合
  - 不确定性传递

**规则系统**
- `Fact`: 事实/断言
- `Condition`: 规则条件
- `Rule`: 产生式规则

**推理引擎**
- `RuleEngine`: 规则引擎
  - 前向链式推理
  - 后向链式推理
  - 多种冲突消解策略
  - 推理解释

**神经规则学习**
- `NeuralRuleLearner`: 神经规则学习器
  - 从数据中发现规则
  - 确定性预测

### 6. 案例推理 (case_based_reasoning.py)

**案例表示**
- `MaterialCase`: 材料案例
- `CaseStatus`: 案例状态

**相似度计算**
- `SimilarityMetric`: 相似度度量
  - 欧氏距离
  - 余弦相似度
  - 加权相似度

**案例检索**
- `CaseRetriever`: 案例检索器
  - 支持多种索引结构
  - 最近邻搜索

**神经案例编码**
- `NeuralCaseEncoder`: 神经案例编码器
  - 学习案例嵌入
  - 基于重建的训练

**案例适配**
- `CaseAdapter`: 案例适配器
  - 基于差异的适配
  - 参数插值

**CBR系统**
- `CaseBasedReasoner`: 案例推理系统
  - 完整的CBR循环（Retrieve-Reuse-Revise-Retain）
  - 案例库管理
  - 自动学习

## 应用案例 (applications.py)

### 1. 材料知识自动形式化

`MaterialKnowledgeFormalizer`: 将非结构化材料知识自动转换为形式化表示

```python
# 从文本自动构建本体
formalizer = MaterialKnowledgeFormalizer()
hierarchy = formalizer.build_ontology_from_text(text_descriptions)

# 形式化材料属性
formalized = formalizer.formalize_properties(material, hierarchy)

# 从数据发现规则
rules = formalizer.discover_rules_from_data(materials)
```

### 2. 符号-神经混合预测

`HybridPredictor`: 结合神经网络与符号推理的混合预测系统

```python
predictor = HybridPredictor(input_dim=10, num_properties=5)
result = predictor.forward(neural_input, symbolic_input)

# 获取可解释的输出
explanation = predictor.explain_prediction(neural_input, symbolic_input)
```

### 3. 可解释材料设计

`ExplainableMaterialDesigner`: 提供详细推理过程的材料设计系统

```python
designer = ExplainableMaterialDesigner()
result = designer.design_material(target_properties, constraints)

# 包含设计方案、解释、置信度和替代方案
print(result['design'])
print(result['explanation'])
print(result['confidence'])
```

## 使用示例

### 神经Prolog推理

```python
from dftlammps.neuro_symbolic import (
    NeuralTheoremProver, create_material_knowledge_base
)

# 创建知识库
facts, rules = create_material_knowledge_base()

# 创建证明器
prover = NeuralTheoremProver(embedding_dim=128)

# 执行推理
result = prover.backward_chain_reasoning(facts, rules, query, symbol_dict)
```

### 概念学习

```python
from dftlammps.neuro_symbolic import ConceptLearner

# 创建概念学习器
learner = ConceptLearner(input_dim=128, num_initial_concepts=10)

# 学习概念
concept_probs, abstraction, hierarchy = learner(examples)

# 发现新概念
new_concept = learner.learn_new_concept(examples, "NewMaterial")
```

### 规则推理

```python
from dftlammps.knowledge_reasoning import RuleEngine, create_material_rules

# 创建规则引擎
engine = RuleEngine()

# 添加规则
for rule in create_material_rules():
    engine.add_rule(rule)

# 添加事实
engine.add_fact(Fact("hasBandGap", ["silicon"], CertaintyFactor(1.0)))

# 执行推理
engine.forward_chain()

# 查询结果
results = engine.query(Fact("isSemiconductor", ["?x"]))
```

### 案例推理

```python
from dftlammps.knowledge_reasoning import CaseBasedReasoner

# 创建CBR系统
cbr = CaseBasedReasoner()

# 添加案例
for case in create_sample_material_cases():
    cbr.add_case(case)

# 解决问题
solution, similar_cases, explanation = cbr.solve(problem, k=3)

# 保留新案例
cbr.retain(problem, solution, outcome, success_score=0.85)
```

## 技术特点

1. **可微分推理**: 所有核心推理操作都是端到端可微的，支持梯度优化
2. **双向翻译**: 神经网络与符号表示之间的无缝转换
3. **可解释性**: 每个推理步骤都可追踪和解释
4. **不确定性处理**: 支持确定性因子和概率推理
5. **层次化表示**: 从低级特征到高级概念的层次学习
6. **知识整合**: 支持规则、本体和案例多种知识表示

## 代码统计

- 神经符号模块: 3,936 行
- 知识推理模块: 2,217 行
- **总计: 6,153 行**

## 依赖

- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- scikit-learn >= 0.24.0

## 参考文献

1. Garcez, A. S., et al. "Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning." (2019)
2. Kautz, H. "The Third AI Summer: AAAI Robert S. Engelmore Memorial Lecture." (2022)
3. Besold, T. R., et al. "Neural-symbolic learning and reasoning: A survey and interpretation." (2017)
4. De Raedt, L., et al. "From statistical relational to neuro-symbolic artificial intelligence." (2020)

## 作者

DFT-LAMMPS Team

## 许可证

MIT License
