# 因果AI与可解释材料发现模块

本模块为DFT-MD材料设计平台提供因果推断和可解释AI能力，帮助理解材料性质的因果机制，而非仅仅发现相关性。

## 模块结构

```
dftlammps/causal_ai/
├── __init__.py              # 模块初始化
├── causal_discovery.py      # 因果发现算法 (~950行)
├── explainable_ml.py        # 可解释机器学习 (~800行)
└── mechanistic_model.py     # 机理模型 (~900行)

dftlammps/uncertainty/
├── __init__.py              # 模块初始化
├── bayesian_nn.py           # 贝叶斯神经网络 (~600行)
├── ensemble_methods.py      # 集合方法 (~650行)
└── conformal_prediction.py  # 共形预测 (~600行)

dftlammps/examples/causal/
└── causal_materials_demo.py # 应用案例 (~500行)
```

总代码量: ~4500行

## 核心功能

### 1. 因果发现 (Causal Discovery)

#### 结构学习算法
- **PC算法**: 基于条件独立性检验的因果结构学习
- **GES算法**: 基于评分函数的贪婪等价搜索
- **NOTEARS**: 基于可微分优化的连续优化方法

#### 干预效应估计
- 平均处理效应 (ATE) 估计
- 条件平均处理效应 (CATE) 估计
- 工具变量 (IV) 估计

#### 反事实推理
- 基于结构因果模型 (SCM) 的反事实推理
- 三步流程: 推断、行动、预测

```python
from dftlammps.causal_ai import CausalDiscoveryPipeline, CausalAlgorithm

# 创建因果发现管道
pipeline = CausalDiscoveryPipeline(
    algorithm=CausalAlgorithm.PC,
    independence_test=IndependenceTest.PEARSON,
    alpha=0.05
)

# 学习因果结构
graph = pipeline.fit(data)

# 估计干预效应
pipeline.estimate_intervention_effect(data)
ate_results = pipeline.intervention_estimator.estimate_ate('treatment', 'outcome')
```

### 2. 可解释机器学习 (Explainable ML)

#### 解释方法
- **SHAP**: 基于博弈论的特征重要性
- **LIME**: 局部可解释模型无关解释
- **注意力可视化**: 神经网络注意力权重分析
- **概念激活向量 (CAV)**: 概念层面的解释
- **积分梯度**: 深度网络的属性归因

#### 特征重要性
- 置换重要性
- 全局和局部解释
- 解释稳定性分析

```python
from dftlammps.causal_ai import ExplainableMLPipeline

# 创建可解释管道
explainer = ExplainableMLPipeline(model, feature_names=features)
explainer.fit(X_train, y_train)

# 获取解释
explanations = explainer.explain(X_test, instance_idx=0)

# 全局重要性
global_exp = explainer.global_explanation(X_test, y_test)
```

### 3. 机理模型 (Mechanistic Models)

#### 物理约束神经网络 (PINN)
- 将物理定律嵌入神经网络训练
- 支持PDE约束、边界条件、守恒定律
- 自动微分计算导数

#### 符号回归
- 遗传编程发现符号方程
- 自动复杂度控制和剪枝
- 方程可读性和可解释性

#### 方程发现 (SINDy-like)
- 稀疏回归发现动力学方程
- 构建候选函数库
- 动力系统方程组发现

```python
from dftlammps.causal_ai import MechanisticModelPipeline, PhysicalConstraint

# 定义物理约束
def physics_constraint(u, u_x, u_xx, u_t, x):
    return u_xx + omega**2 * u  # 谐振子方程

constraint = PhysicalConstraint(
    name="harmonic_oscillator",
    constraint_type="pde",
    equation=physics_constraint
)

# 拟合PINN
pipeline = MechanisticModelPipeline()
pipeline.fit_pinn(
    X_data, y_data,
    physics_constraints=[constraint],
    epochs=1000
)

# 符号回归
equation = pipeline.fit_symbolic(X, y, variable_names=['x', 'y'])
```

### 4. 不确定性量化 (Uncertainty Quantification)

#### 贝叶斯神经网络
- 变分推断贝叶斯神经网络
- 蒙特卡洛Dropout
- 分解认知不确定性和偶然不确定性

#### 集合方法
- 深度集合 (Deep Ensembles)
- 快照集成 (Snapshot Ensembles)
- 加权集合
- Bootstrap聚合

#### 共形预测
- 标准共形预测
- 自适应共形预测
- 共形化分位数回归
- 时间序列共形预测

```python
from dftlammps.uncertainty import DeepEnsemble, StandardConformalPredictor

# 深度集合
ensemble = DeepEnsemble(model_builder, n_members=5)
ensemble.fit(X_train, y_train)
pred = ensemble.predict(X_test)  # 包含均值和方差

# 共形预测
cp = StandardConformalPredictor(base_model)
cp.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal, alpha=0.1)
pred_cp = cp.predict(X_test)  # 保证90%覆盖率
```

## 应用案例

### 锂离子电池材料设计

```python
from dftlammps.examples.causal import run_full_demo

# 运行完整演示
results = run_full_demo()
```

该演示展示：
1. **因果关系发现**: 识别材料合成条件、微观结构与性能的因果关系
2. **性质归因分析**: 解释为什么某个材料具有高容量
3. **可靠预测**: 为新材料提供带置信区间的预测
4. **物理方程发现**: 发现容量与组成成分之间的数学关系

## API参考

### 因果发现

| 类/函数 | 描述 |
|---------|------|
| `CausalGraph` | 因果图数据结构 |
| `PCAlgorithm` | PC算法实现 |
| `GESAlgorithm` | GES算法实现 |
| `NOTEARSAlgorithm` | NOTEARS算法实现 |
| `InterventionEffectEstimator` | 干预效应估计 |
| `CounterfactualInference` | 反事实推理 |
| `CausalDiscoveryPipeline` | 完整管道 |

### 可解释ML

| 类/函数 | 描述 |
|---------|------|
| `SHAPExplainer` | SHAP解释器 |
| `LIMEExplainer` | LIME解释器 |
| `AttentionVisualizer` | 注意力可视化 |
| `ConceptActivationVectors` | 概念激活向量 |
| `IntegratedGradients` | 积分梯度 |
| `ExplainableMLPipeline` | 可解释ML管道 |

### 机理模型

| 类/函数 | 描述 |
|---------|------|
| `PhysicsInformedNN` | 物理约束神经网络 |
| `SymbolicRegression` | 符号回归 |
| `EquationDiscovery` | 方程发现 |
| `MechanisticModelPipeline` | 机理模型管道 |

### 不确定性量化

| 类/函数 | 描述 |
|---------|------|
| `BayesianNeuralNetwork` | 贝叶斯神经网络 |
| `DeepEnsemble` | 深度集合 |
| `SnapshotEnsemble` | 快照集成 |
| `StandardConformalPredictor` | 标准共形预测 |
| `AdaptiveConformalPredictor` | 自适应共形预测 |

## 依赖项

必需:
- numpy
- pandas
- scipy
- scikit-learn

可选（增强功能）:
- PyTorch (PINN和贝叶斯NN)
- SHAP (SHAP解释)
- matplotlib (可视化)
- networkx (因果图)

## 使用示例

### 基本用法

```python
import numpy as np
import pandas as pd
from dftlammps.causal_ai import CausalDiscoveryPipeline, CausalAlgorithm

# 生成数据
data = pd.DataFrame({
    'X': np.random.normal(0, 1, 1000),
    'Z': np.random.normal(0, 1, 1000),
    'Y': np.random.normal(0, 1, 1000)
})

# 因果发现
pipeline = CausalDiscoveryPipeline(
    algorithm=CausalAlgorithm.PC
)
graph = pipeline.fit(data)

# 可视化因果图
graph.visualize()
```

### 不确定性量化

```python
from dftlammps.uncertainty import DeepEnsemble, StandardConformalPredictor

# 深度集合
ensemble = DeepEnsemble(model_builder, n_members=5)
ensemble.fit(X_train, y_train)
pred = ensemble.predict(X_test)

print(f"预测: {pred.mean}")
print(f"不确定性: {np.sqrt(pred.variance)}")

# 共形预测
cp = StandardConformalPredictor(model)
cp.fit(X_train, y_train).calibrate(X_cal, y_cal, alpha=0.1)
pred_cp = cp.predict(X_test)

print(f"预测区间: [{pred_cp.lower_bound}, {pred_cp.upper_bound}]")
```

## 参考文献

1. Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
2. Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference.
3. Molnar, C. (2022). Interpretable Machine Learning.
4. Karniadakis, G. E. et al. (2021). Physics-informed machine learning. Nature Reviews Physics.
5. Angelopoulos, A. N., & Bates, S. (2021). Conformal Prediction: A Gentle Introduction.

## 作者

Causal AI Team - DFT-MD平台

## 版本

1.0.0 (2025)
