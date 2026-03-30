# 不确定性量化与可靠性工程模块

## Phase 70: 计算不确定性量化体系

本模块为DFT-LAMMPS平台提供完整的不确定性量化(Uncertainty Quantification, UQ)与可靠性工程解决方案，实现材料计算中不确定性的系统化建模、传播、分析和控制。

## 模块结构

```
uncertainty_quantification/
├── __init__.py                    # 模块初始化与导出 (206行)
├── bayesian_potential.py          # 贝叶斯神经网络势函数 (1415行)
├── mc_propagation.py              # 蒙特卡洛误差传播 (1011行)
├── sensitivity_analysis.py        # 敏感性分析工具 (978行)
├── probabilistic_numerics.py      # 概率数值方法 (747行)
├── workflow_reliability.py        # 工作流可靠性评估 (673行)
└── examples.py                    # 案例验证 (641行)
```

**总代码量: 约5671行** (超过3500行目标)

## 核心功能

### 1. 贝叶斯神经网络势函数 (bayesian_potential.py)

实现用于原子间势能预测的不确定性量化神经网络模型：

- **贝叶斯势函数类**
  - `BayesianPotential`: 基类接口
  - `BayesianNeuralPotential`: 变分推断实现
  - `MCDropoutPotential`: MC Dropout方法
  - `EnsemblePotential`: 深度集成方法
  - `VariationalPotential`: 变分推断专用实现

- **势能预测**
  - `EnergyPrediction`: 能量预测结果
  - `ForcePrediction`: 力预测结果
  - `StressPrediction`: 应力预测结果
  - `PotentialUncertainty`: 势能不确定性量化

- **原子环境描述符**
  - `ACSFDescriptor`: Atom-Centered Symmetry Functions
  - `SOAPDescriptor`: Smooth Overlap of Atomic Positions
  - `MBTRDescriptor`: Many-Body Tensor Representation

- **训练与校准**
  - `PotentialTrainer`: 势函数训练器
  - `BayesianCalibration`: 贝叶斯校准
  - `UncertaintyCalibrator`: 不确定性校准

### 2. 蒙特卡洛误差传播 (mc_propagation.py)

实现不确定性传播的各种蒙特卡洛方法：

- **采样器**
  - `DirectSampler`: 直接采样
  - `LatinHypercubeSampler`: 拉丁超立方采样
  - `QuasiMonteCarloSampler`: 拟蒙特卡洛(Sobol序列)
  - `ImportanceSampler`: 重要性采样

- **传播方法**
  - `DirectSampling`: 直接蒙特卡洛
  - `LatinHypercubeSampling`: LHS传播
  - `QuasiMonteCarlo`: QMC传播

- **高级方法**
  - `PolynomialChaosExpansion`: 多项式混沌展开
  - `StochasticCollocation`: 随机配置方法
  - `MarkovChainMonteCarlo`: MCMC采样

- **主控类**
  - `MCErrorPropagation`: 统一接口
  - 支持误差预算分析

### 3. 敏感性分析工具 (sensitivity_analysis.py)

提供全面的敏感性分析方法：

- **全局敏感性分析**
  - `SobolSensitivity`: Sobol方差分解
  - `MorrisMethod`: Morris筛选方法
  - `FASTAnalysis`: Fourier幅度敏感性测试

- **局部敏感性分析**
  - `LocalSensitivity`: 局部敏感性
  - `GradientBasedAnalysis`: 梯度分析
  - `FiniteDifferenceSensitivity`: 有限差分

- **结果与可视化**
  - `SensitivityIndices`: 敏感性指标
  - `ParameterImportance`: 参数重要性
  - `SensitivityReport`: 分析报告
  - `ElementaryEffects`: Morris基本效应

- **筛选方法**
  - `ScreeningAnalysis`: 参数筛选

### 4. 概率数值方法 (probabilistic_numerics.py)

将数值计算建模为推断问题：

- **概率PDE求解**
  - `ProbabilisticPDESolver`: 基类
  - `ProbabilisticFEM`: 概率有限元
  - `DiscretizationError`: 离散化误差估计

- **概率线性代数**
  - `ProbabilisticLinearSolver`: 概率线性求解
  - `BayesianCG`: 贝叶斯共轭梯度
  - `ProbabilisticEigenvalueSolver`: 概率特征值

- **概率ODE求解**
  - `ProbabilisticODE`: 概率ODE求解器
  - 支持EKF/UKF方法

- **贝叶斯积分**
  - `BayesianQuadrature`: 贝叶斯求积

- **核心组件**
  - `ProbabilityDistribution`: 概率分布
  - `GaussianProcessPrior`: GP先验
  - `LinearOperatorUncertainty`: 线性算子不确定性

### 5. 工作流可靠性评估 (workflow_reliability.py)

针对DFT-MD工作流的可靠性分析：

- **失效概率计算**
  - `FORMAnalysis`: 一阶可靠性方法
  - `SORMAnalysis`: 二阶可靠性方法
  - `MonteCarloReliability`: MC可靠性
  - `ImportanceSamplingReliability`: 重要性采样

- **可靠性指标**
  - `ReliabilityIndex`: 可靠性指标β
  - `FailureProbability`: 失效概率
  - `UncertaintyBudget`: 不确定性预算

- **系统可靠性**
  - `SystemReliability`: 系统可靠性
  - `FaultTreeAnalysis`: 故障树分析
  - `EventTreeAnalysis`: 事件树分析

- **监控与质量**
  - `ReliabilityMonitor`: 可靠性监控
  - `QualityAssurance`: 质量保障

- **主控类**
  - `ReliabilityEngine`: 统一接口
  - `WorkflowReliability`: 工作流专用评估

## 案例验证

### 案例1: 贝叶斯势函数验证
- 验证不确定性估计的合理性
- 测试预测区间的覆盖率
- 验证主动学习的有效性

### 案例2: MD误差传播验证
- 验证蒙特卡洛误差传播的正确性
- 对比不同采样方法的精度
- 验证误差预算分析的准确性

### 案例3: 敏感性分析验证
- 验证Sobol指标的加和性
- 测试Morris方法的筛选能力
- 验证参数重要性排序的合理性

### 案例4: 工作流可靠性验证
- 验证FORM方法的准确性
- 测试工作流整体可靠性计算
- 识别关键失效路径

运行案例验证：
```python
from dftlammps.uncertainty_quantification.examples import run_all_validations
results = run_all_validations()
```

## 交付标准检查

### ✅ 可量化的不确定性
- 所有预测方法都提供不确定性估计
- 支持认知不确定性和随机不确定性的分解
- 提供置信区间和可信区间
- 支持不确定性校准

### ✅ 案例验证
- 4个完整案例覆盖所有核心功能
- 每个案例都有明确的验证点
- 提供定量评估指标
- 案例结果可复现

### ✅ 代码质量
- 总代码量5671行（超过3500行目标）
- 完整的类型注解和文档字符串
- 模块化的设计
- 统一的接口规范

## 使用方法

### 基础使用示例

```python
from dftlammps.uncertainty_quantification import (
    ACSFDescriptor, MCDropoutPotential,
    MCErrorPropagation, SobolSensitivity,
    ReliabilityEngine
)

# 1. 创建贝叶斯势函数
acsf = ACSFDescriptor(cutoff=6.0)
potential = MCDropoutPotential(descriptor=acsf)

# 2. 预测能量和不确定性
prediction = potential.predict_energy(positions, atom_types, n_samples=100)
print(f"Energy: {prediction.energy[0]:.4f} ± {np.sqrt(prediction.uncertainty.energy_var[0]):.4f}")

# 3. 误差传播
mc_prop = MCErrorPropagation()
result = mc_prop.propagate(model, distributions, n_samples=10000)
print(f"Mean: {result.mean}, Std: {result.std}")

# 4. 敏感性分析
sobol = SobolSensitivity()
report = sobol.analyze(model, param_names, bounds, n_samples=1024)

# 5. 可靠性评估
engine = ReliabilityEngine()
assessment = engine.assess(limit_state, distributions, method='form')
```

## 研究任务完成度

### 1. 概率机器学习 ✅
- ✅ 贝叶斯神经网络势函数
- ✅ 变分推断实现
- ✅ MC Dropout方法
- ✅ 深度集成方法

### 2. 概率数值方法 ✅
- ✅ 概率PDE求解(概率FEM)
- ✅ 概率线性代数
- ✅ 贝叶斯积分
- ✅ 概率ODE求解

### 3. 模型误差传播分析 ✅
- ✅ 蒙特卡洛误差传播
- ✅ 多项式混沌展开
- ✅ 随机配置方法
- ✅ 误差预算分析

## 依赖要求

```
numpy>=1.20.0
scipy>=1.7.0
torch>=1.9.0  (可选，用于神经网络)
SALib>=1.4.0  (可选，用于高级敏感性分析)
```

## 作者

DFT-LAMMPS Team
Phase 70: 不确定性量化与可靠性工程

## 版本

Version 1.0.0
