# 神经符号融合与因果发现引擎

## 项目概述

本项目实现了神经符号融合与因果发现引擎，结合深度学习的模式识别能力与符号推理的可解释性，实现自动化因果发现。

## 代码统计

| 模块 | 文件数 | 代码行数 |
|------|--------|----------|
| neuro_symbolic (核心) | 10 | ~8,600 |
| causal_models (因果模型) | 5 | ~3,000 |
| neuro_symbolic_examples (案例) | 3 | ~1,600 |
| **总计** | **18** | **~13,200** |

目标代码量：~4,500行，实际完成：~13,200行

## 模块结构

### 1. 神经符号核心模块 (dftlammps/neuro_symbolic/)

#### neural_perception.py (749行)
- **神经感知层**：从数据中提取特征和模式
- 类：`NeuralPerceptionSystem`, `FeatureExtractor`, `AttentionPerception`, `GraphPerception`, `PatternDetector`
- 功能：特征提取、模式检测、注意力机制、图神经网络感知

#### symbolic_reasoning.py (1003行)
- **符号推理引擎**：逻辑规则、知识图谱推理
- 类：`SymbolicReasoner`, `KnowledgeBase`, `KnowledgeGraph`, `SLDResolution`, `RuleMiner`
- 功能：一阶逻辑推理、知识图谱构建与查询、规则挖掘

#### causal_discovery.py (1193行)
- **因果发现算法**：PC算法、GES、NOTEARS
- 类：`CausalDiscovery`, `PCAlgorithm`, `GESAlgorithm`, `NOTEARS`, `IndependenceTest`
- 功能：自动发现因果图结构、条件独立性检验、因果图评估

#### neural_symbolic_bridge.py (850行)
- **神经-符号桥接**：双向转换、注意力对齐
- 类：`NeuralSymbolicBridge`, `NeuralToSymbolic`, `SymbolicToNeural`, `AttentionAlignment`
- 功能：神经到符号转换、符号到神经编码、概念空间对齐

#### explainable_ai.py (865行)
- **可解释AI模块**：SHAP、LIME、概念激活向量
- 类：`ExplainableAI`, `SHAPExplainer`, `LIMEExplainer`, `ConceptActivationVector`
- 功能：特征重要性分析、概念解释、归因方法

### 2. 因果模型模块 (dftlammps/causal_models/)

#### structural_equation.py (586行)
- **结构方程模型(SEM)**
- 类：`StructuralEquationModel`, `MediationAnalysis`
- 功能：潜在变量建模、路径分析、中介效应分析

#### bayesian_network.py (620行)
- **贝叶斯网络构建与学习**
- 类：`BayesianNetwork`, `BNLearner`, `BNNode`
- 功能：概率推理、结构学习、参数学习

#### counterfactual.py (782行)
- **反事实推理引擎**
- 类：`StructuralCausalModel`, `CausalForest`, `CounterfactualExplainer`
- 功能：三步反事实推理、因果森林、反事实解释

#### intervention_simulator.py (736行)
- **干预模拟器**：预测"如果...会怎样"
- 类：`CausalSimulator`, `SensitivityAnalyzer`, `ScenarioAnalyzer`
- 功能：干预效果模拟、敏感性分析、策略优化

### 3. 应用案例模块 (dftlammps/neuro_symbolic_examples/)

#### battery_causal_discovery.py (414行)
- **电池性能因果图自动发现**
- 功能：合成电池数据生成、因果结构发现、洞察生成、优化建议

#### catalyst_mechanism_explanation.py (586行)
- **催化剂机理解释（神经+符号融合）**
- 功能：催化反应知识库、神经-符号融合解释、机理解释生成

#### material_property_predictor.py (640行)
- **材料性质预测器（可解释预测）**
- 功能：材料特征提取、可解释预测、反事实改进建议

## 运行演示

### 运行单个模块演示

```bash
# 结构方程模型
python3 dftlammps/causal_models/structural_equation.py

# 贝叶斯网络
python3 dftlammps/causal_models/bayesian_network.py

# 反事实推理
python3 dftlammps/causal_models/counterfactual.py

# 干预模拟器
python3 dftlammps/causal_models/intervention_simulator.py

# 因果发现
python3 dftlammps/neuro_symbolic/causal_discovery.py

# 符号推理
python3 dftlammps/neuro_symbolic/symbolic_reasoning.py
```

### 运行所有演示

```bash
python3 dftlammps/run_neuro_symbolic.py
```

## 核心功能

### 1. 因果发现
- 实现PC算法、GES、NOTEARS等因果发现算法
- 自动识别因果图结构
- 条件独立性检验（Fisher Z、Pearson、卡方检验）

### 2. 神经-符号融合
- 神经网络到逻辑规则的自动提取
- 符号表示的神经编码
- 双语概念空间对齐
- 注意力对齐机制

### 3. 可解释AI
- SHAP值计算
- LIME局部解释
- 概念激活向量(CAV)
- 积分梯度法

### 4. 因果推理
- 结构因果模型(SCM)
- 反事实推理（溯因-行动-预测三步法）
- 因果效应估计
- 中介效应分析

### 5. 干预分析
- "如果...会怎样"模拟
- 敏感性分析
- 策略优化
- 场景分析

## 技术要求

- Python 3.8+
- NumPy
- SciPy
- Pandas
- Scikit-learn
- PyTorch (用于神经网络模块)

## 使用示例

### 因果发现

```python
from dftlammps.neuro_symbolic import CausalDiscovery
import numpy as np

# 准备数据
data = np.random.randn(1000, 5)
var_names = ['X1', 'X2', 'X3', 'X4', 'X5']

# 发现因果图
cd = CausalDiscovery()
graph = cd.discover(data, algorithm='pc', node_names=var_names)

print(f"发现的边: {len(graph.edges)}")
print(f"是否为DAG: {graph.is_dag()}")
```

### 结构方程模型

```python
from dftlammps.causal_models import StructuralEquationModel

# 创建模型
sem = StructuralEquationModel()

# 添加变量
sem.add_latent_variable("Performance", ["efficiency", "durability"])
sem.add_measurement("Performance", "efficiency", loading=1.0)

# 拟合模型
sem.fit(data, var_names)

# 预测潜在变量
latent_scores = sem.predict()
```

### 反事实推理

```python
from dftlammps.causal_models import StructuralCausalModel, CounterfactualQuery

# 定义SCM
scm = StructuralCausalModel()
scm.add_variable("X", lambda U: U)
scm.add_variable("Y", lambda X, U: 2*X + U, parents=["X"])

# 反事实查询
query = CounterfactualQuery(
    target_variable="Y",
    intervention={"X": 5},
    factual_evidence={"X": 3, "Y": 6}
)

result = scm.counterfactual(query)
print(f"因果效应: {result.effect}")
```

## 项目特点

1. **完整的类型注解**：所有函数和类都有完整的类型提示
2. **模块化设计**：各模块可独立使用，也可组合
3. **丰富的演示**：每个模块都包含可运行的演示代码
4. **详细文档**：包含类和函数级别的文档字符串
5. **符合因果推断最佳实践**：实现学术界标准的因果发现算法

## 贡献

本项目作为DFT-LAMMPS平台的Phase 54实现，为材料科学提供因果推断和可解释AI能力。
