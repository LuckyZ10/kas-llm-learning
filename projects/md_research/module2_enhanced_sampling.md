# 分子动力学前沿研究 - 模块2：增强采样新方法
**研究时间**: 2025年
**研究方向**: OPES、Metadynamics、TAMD

---

## 2.1 OPES (On-the-fly Probability Enhanced Sampling)

### 核心创新 (Invernizzi & Parrinello, 2020)
OPES是一种全新的增强采样框架，从根本上重新思考了metadynamics：
- **从偏置势到概率分布**: 不再简单累积高斯偏置，而是实时估计目标分布与真实分布的比值
- **准静态偏置**: 偏置势在短暂初始瞬态后变为准静态，通过简单重加权即可计算自由能
- **对次优CV更鲁棒**: 特别设计用于处理非最优集体变量

### 数学原理
```
V(x) = -(1/β) log(p_target(x) / P(x))
```
其中通过重加权实时更新概率估计，每次迭代更新一次样本。

### OPES变体

#### OPES_METAD
- 采样metadynamics-like目标分布（如well-tempered分布）
- 偏置形式: V_n(Q) = (1-γ^(-1))(1/β)log(P̃_n(Q)/Z_n + ε)
- 关键参数 **ΔE (能垒参数)**: 用于设置偏置因子γ，选择约等于需克服的能量势垒

#### OPES_EXPANDED  
- 采样扩展系综目标分布（replica-exchange-like）
- 支持多种扩展系综：
  - Multicanonical (多正则系综)
  - Multibaric (多压强系综)  
  - Multithermal-baric (温压联合系综)
  - Multi-umbrella (多伞形采样)

#### OPES_METAD_EXPLORE (探索模式)
- 专注于快速探索自由能面
- 适用于复杂高能垒系统

---

## 2.2 OneOPES: 混合增强采样 (2023-2024)

### 核心概念 (Rizzi et al., 2023; Gervasio group)
OneOPES结合了**OPES**与**副本交换分子动力学(REMD)**的优势：
- 使用多个副本，具有不同的偏置势强度
- 基础副本（第一个）包含单个OPES偏置
- 高级副本包含多个OPES偏置势，强度递增
- 最高副本可快速探索整个自由能景观

### 技术细节
- 副本间交换频率: 每1000-10000步
- 目标交换接受率: ≥20%
- 可同时对多个集体变量进行偏置
- 结合OPES MultiThermal偏置实现多温度探索

### 2024年重要应用
- **蛋白质-配体结合自由能计算** (Karrenbrock et al., 2024)
  - 高精度预测结合自由能
  - 在小分子受体上实现自动化
- **聚合物吸附研究** (Glisman et al., 2024)
  - 8个副本系统
  - 结合距离、回转半径、配位数等多个CV

---

## 2.3 Well-Tempered Metadynamics (持续发展)

### 基本原理
- 通过添加时间依赖的外部偏置势来增强CV采样
- 偏置势由排斥性高斯核组成，周期性沉积在当前CV位置
- 偏置势抵消自由能面，导致CV空间更均匀的采样

### Well-Tempered变体核心公式
```
V_n(s) = Σ G(s,s_k) exp[-βV_{k-1}(s_k)/(γ-1)]
```
收敛时:
```
V(s) = -(1-1/γ)F(s) + c
```

### 2024-2025年新发展

#### 与机器学习势结合
- **ReaxFF-Metadynamics**: 在反应力场MD中使用WT-MetaD研究稀有事件
- 应用于CO2矿化等复杂反应过程

#### 多副本Metadynamics变体
- **Parallel-bias metadynamics**: 并行偏置多个CV
- **Bias-exchange metadynamics**: 副本间交换偏置
- **Replica exchange with collective variable tempering**: 减少副本数量

---

## 2.4 Temperature-Accelerated MD (TAMD)

### 基本原理
TAMD与aMD采取不同策略：
- **aMD**: 修改能量景观（降低势垒）
- **TAMD**: 提高CV温度，加速自由能面探索

### 数学形式 (Overdamped dynamics)
```
γẋ = -∇V(x) + μ(X-q(x))∇q(x) + √(2γβ^(-1))ẇ
Ẋ = -μ(X-q(x)) + √(2β̃^(-1))Ẇ
```
其中 β̃ = 1/(k_B T̃) 且 T̃ > T

### 优势
- CV以更高温度探索自由能景观
- 不影响原始系统约束动力学的采样精度
- 自由能梯度计算保持准确

### TAMD三变体 (Vanden-Eijnden, NYU)
1. **自由能探索**: 大规模蛋白质构象采样
2. **Single sweep结合**: 非参数化映射自由能景观
3. **参数化估计**: 假设自由能函数形式，在线优化参数

### Temperature-Accelerated Dynamics (TAD)
Voter等人开发的加速动力学方法：
- 在高温下运行模拟加速跃迁
- 过滤出在原始温度下不会发生的虚假跃迁
- 从高温 extrapolate 到低温的态间跃迁速率

**加速比**:
```
SU_TAD = exp[E_BA(1/(k_B T_low) - 1/(k_B T_high))]
```

### 最新发展 (LANL, 2024)
- **Speculatively Parallel TAD (SpecTAD)**: 推测性并行化
- **Spatially Parallel TAD (ParTAD)**: 空间并行处理大系统
- **Extended TAD (XTAD)**: 扩展TAD用于大规模系统

---

## 2.5 机器学习与增强采样结合 (2024-2025热点)

### 数据驱动集体变量发现

#### 深度学习方法
- **Bonati et al. (2020-2021)**: 
  - J. Phys. Chem. Lett. 11, 2998-3004 (2020)
  - Proc. Natl Acad. Sci. USA 118, e2113533118 (2021)
  - 深度学习慢模式用于稀有事件采样

#### TLC框架 (ICML 2025)
- **Time-Lagged Generation学习CV**
- 从时间滞后条件生成模型学习CV
- 捕获慢动态行为而非静态玻尔兹曼分布
- 在SMD和OPES中验证效果优于现有MLCV方法

### 强化学习增强采样
- 使用UCB (Upper Confidence Bound)算法平衡探索与利用
- 动态调整采样策略

---

## 2.6 软件实现

### PLUMED 2.8+
- **OPES模块**: 完整实现OPES_METAD和OPES_EXPANDED
- **配置**: --enable-modules=opes
- 包含多种扩展CV (ECVs)

### 其他软件
- **Colvars**: ABF、Metadynamics多种变体
- **SSAGES**: 多种增强采样方法
- **PMFlib**: ABF、约束动力学、Metadynamics

---

## 2.7 方法选择指南

| 方法 | 适用场景 | 优势 | 局限 |
|------|---------|------|------|
| **OPES** | 通用增强采样 | 快速收敛、参数少、对次优CV鲁棒 | 相对较新 |
| **OneOPES** | 复杂生物分子、结合自由能 | 多CV同时加速、高精度 | 计算成本高 |
| **WT-MetaD** | 成熟协议、已知好CV | 大量文献、多种后处理 | 收敛慢于OPES |
| **TAMD** | 自由能计算、CV探索 | 不改变势能面、无偏 | 需要 careful 参数选择 |
| **TAD** | 材料中稀有事件 | 可恢复正确动力学 | 需要过滤虚假跃迁 |

---

## 2.8 未来趋势

1. **MLCV自动化**: 深度学习自动发现最优CV
2. **混合方法**: 多种增强采样技术组合
3. **不确定性量化**: 评估增强采样结果的可靠性
4. **大规模并行**: GPU加速与副本并行结合
5. **与ML势结合**: 在机器学习势上使用增强采样
