# 主动学习在机器学习势训练中的应用：技术报告

## 目录
1. [概述](#1-概述)
2. [DP-GEN工作流详解](#2-dp-gen工作流详解)
3. [不确定性量化方法](#3-不确定性量化方法)
4. [探索策略](#4-探索策略)
5. [模型压缩](#5-模型压缩)
6. [收敛判断标准](#6-收敛判断标准)
7. [最佳实践](#7-最佳实践)
8. [参考文献](#8-参考文献)

---

## 1. 概述

### 1.1 主动学习的动机

在机器学习势（ML Potential）训练中，数据的质量和覆盖度直接决定了模型的精度和泛化能力。传统的静态数据采样方法存在以下问题：

- **数据冗余**：大量计算资源用于相似构型的重复计算
- **覆盖不足**：关键过渡态和稀有事件难以被充分采样
- **效率低下**：无法根据模型当前状态动态调整采样策略

主动学习（Active Learning）通过迭代式的"探索-标注-重训练"循环，实现了训练数据的自适应生成，显著提高了数据效率和模型精度。

### 1.2 DP-GEN框架

Deep Potential GENerator (DP-GEN) 是一个实现并发学习（Concurrent Learning）的开源框架，由深势科技（DeepModeling）开发。其核心思想是：

> **"让模型告诉我们哪些构型需要DFT计算"**

DP-GEN工作流包含三个核心阶段：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Exploration │ ──▶ │   Labeling  │ ──▶ │   Training  │
│    (探索)    │     │   (DFT标注) │     │   (重训练)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
       ▲                                       │
       └───────────────────────────────────────┘
                    (循环迭代)
```

---

## 2. DP-GEN工作流详解

### 2.1 探索阶段 (Exploration)

探索阶段使用当前的ML势进行分子动力学(MD)模拟，生成大量构型。关键步骤：

#### 2.1.1 温度/压力扰动
```python
# 温度范围设置 (K)
temperatures = np.linspace(300, 2000, 8)  # 从室温到高温

# 压力范围设置 (GPa)  
pressures = np.linspace(-5, 50, 6)  # 从负压力到高压
```

**物理意义**：
- 高温采样：探索热激发下的构型空间，捕捉非谐效应
- 高压采样：探索致密相变和键合变化
- 负压力：探索拉伸和断裂行为

#### 2.1.2 系综选择

| 系综 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| NVT | 固定体积系统 | 稳定，易于收敛 | 无法捕捉热膨胀 |
| NPT | 实际材料模拟 | 符合实验条件 | 计算开销较大 |
| NVE | 微正则采样 | 能量守恒 | 温度漂移 |

#### 2.1.3 模型偏差计算

使用**模型集成（Model Ensemble）**量化不确定性：

```python
# 训练 N 个独立模型
n_models = 4
models = [train_model(seed=i) for i in range(n_models)]

# 对每个构型计算预测分布
forces = [model.predict(coords) for model in models]

# 力偏差: 标准差最大值
ε_F,max = max_i √[⟨||F_i||²⟩ - ⟨||F_i||⟩²]

# 能量偏差: 相对标准差  
ε_E = √[⟨E²⟩ - ⟨E⟩²] / N_atoms
```

### 2.2 标注阶段 (Labeling)

标注阶段对筛选出的候选构型进行第一性原理计算。

#### 2.2.1 候选构型选择标准

候选构型满足：**θ_lo ≤ ε_F,max < θ_hi**

其中：
- **θ_lo (下限)**：约 0.05 eV/Å
- **θ_hi (上限)**：约 0.15-0.35 eV/Å

选择逻辑：
- **ε < θ_lo**: 模型已能准确预测，无需DFT计算
- **θ_lo ≤ ε < θ_hi**: 模型不确定性适中，加入训练集
- **ε ≥ θ_hi**: 不确定性过高，可能是物理不稳定构型

#### 2.2.2 DFT计算参数建议

```fortran
! VASP INCAR 示例
ENCUT = 500          ! 截断能，建议 400-600 eV
ISMEAR = 0           ! Gaussian smearing
SIGMA = 0.05         ! 展宽
EDIFF = 1E-6         ! 电子收敛
NSW = 0              ! 单点计算，不优化
IBRION = -1          ! 不移动离子
LCHARG = .FALSE.     ! 不保存CHGCAR
LWAVE = .FALSE.      ! 不保存WAVECAR
```

### 2.3 训练阶段 (Training)

#### 2.3.1 损失函数设计

```python
loss = p_e * |E_DFT - E_ML|² + p_f * |F_DFT - F_ML|² + p_v * |V_DFT - V_ML|²
```

**自适应权重策略**：

| 训练阶段 | p_e | p_f | p_v | 策略说明 |
|---------|-----|-----|-----|---------|
| 初期 | 0.02 | 1000 | 0.01 | 优先学习力 |
| 中期 | 0.1 | 100 | 0.1 | 平衡学习 |
| 后期 | 1.0 | 1.0 | 1.0 | 精细调整能量 |

#### 2.3.2 学习率调度

```python
learning_rate = start_lr * (stop_lr/start_lr)^(step/decay_steps)

# 典型参数
start_lr = 0.001      # 初始学习率
stop_lr = 3.51e-8     # 最终学习率 (接近单精度极限)
decay_steps = 5000    # 衰减步数
```

---

## 3. 不确定性量化方法

### 3.1 模型偏差类型

#### 3.1.1 力偏差 (Force Deviation)

$$
\epsilon_{F,\max} = \max_i \sqrt{\langle ||\mathbf{F}_i||^2 \rangle - \langle ||\mathbf{F}_i|| \rangle^2}
$$

这是DP-GEN中最常用的不确定性指标，因为：
- 力对局部环境敏感
- 与MD稳定性直接相关
- 计算成本低

#### 3.1.2 能量偏差 (Energy Deviation)

$$
\epsilon_E = \frac{\sqrt{\langle E^2 \rangle - \langle E \rangle^2}}{N_{\text{atoms}}}
$$

适用场景：
- 热力学性质计算
- 相变研究
- 全局结构优化

#### 3.1.3 维里偏差 (Virial Deviation)

$$
\epsilon_V = \frac{\sqrt{\langle ||\mathbf{V}||^2 \rangle - \langle ||\mathbf{V}|| \rangle^2}}{N_{\text{atoms}}}
$$

适用场景：
- 高压研究
- 弹性常数计算
- 应力-应变分析

### 3.2 阈值设定指南

#### 3.2.1 金属体系
```python
# 金属建议阈值 (eV/Å)
f_trust_lo = 0.05
f_trust_hi = 0.15
```

#### 3.2.2 分子/软物质
```python
# 分子体系建议阈值 (eV/Å)
f_trust_lo = 0.03
f_trust_hi = 0.10
```

#### 3.2.3 高温/高压极端条件
```python
# 极端条件建议阈值 (eV/Å)
f_trust_lo = 0.08
f_trust_hi = 0.25
```

### 3.3 自适应阈值调整

```python
def adjust_thresholds(candidate_ratio, target_ratio=0.1):
    """
    根据候选结构比例动态调整阈值
    
    目标: 维持 5-15% 的候选结构比例
    """
    if candidate_ratio > target_ratio * 1.5:
        # 候选过多，提高阈值
        f_trust_lo *= 1.1
        f_trust_hi *= 1.1
    elif candidate_ratio < target_ratio * 0.5:
        # 候选过少，降低阈值
        f_trust_lo *= 0.9
        f_trust_hi = max(f_trust_hi * 0.95, f_trust_lo + 0.05)
```

---

## 4. 探索策略

### 4.1 温度扰动策略

#### 4.1.1 线性温度调度
```python
temperatures = np.linspace(T_min, T_max, n_points)
```

#### 4.1.2 指数温度调度
```python
temperatures = T_min * (T_max/T_min) ** (np.arange(n_points) / (n_points-1))
```

#### 4.1.3 自适应温度调度
基于当前迭代的不确定性分布调整温度范围：
```python
if avg_uncertainty < threshold_low:
    T_max *= 1.2  # 扩大高温探索
elif avg_uncertainty > threshold_high:
    T_max *= 0.9  # 降低温度，精细化采样
```

### 4.2 压力扰动策略

```python
# 压力扫描范围 (GPa)
pressures = [-5, 0, 10, 30, 50]

# 特殊相变压力点 (示例: Si的相变)
phase_transition_pressures = [12, 90]  # GPa
```

### 4.3 结构变形策略

#### 4.3.1 应变模式
```python
deformation_modes = {
    'uniaxial': [[1+ε, 0, 0], [0, 1, 0], [0, 0, 1]],    # 单轴拉伸/压缩
    'biaxial': [[1+ε, 0, 0], [0, 1+ε, 0], [0, 0, 1]],   # 双轴应变
    'shear': [[1, ε, 0], [0, 1, 0], [0, 0, 1]],         # 剪切变形
    'volumetric': [[f, 0, 0], [0, f, 0], [0, 0, f]]     # 等体积应变, f=(1+ε)^(1/3)
}
```

#### 4.3.2 缺陷结构生成
```python
def generate_defect_structures(bulk, defect_type='vacancy'):
    """
    生成缺陷结构
    
    defect_type:
        - 'vacancy': 空位缺陷
        - 'interstitial': 间隙原子
        - 'substitution': 替位掺杂
        - 'dislocation': 位错
    """
    # 实现细节...
```

### 4.4 表面结构采样

```python
from ase.build import surface

# 生成不同晶面的表面
miller_indices = [(1,0,0), (1,1,0), (1,1,1), (2,1,0)]
for hkl in miller_indices:
    slab = surface(bulk_structure, hkl, n_layers=6, vacuum=15.0)
```

### 4.5 AIMD采样策略

对于复杂系统，可直接使用短时间的AIMD轨迹作为初始探索数据：

```python
# AIMD参数
aimd_params = {
    'time_step': 1.0,      # fs
    'temperature': 900,    # K (高温加速探索)
    'n_steps': 5000,       # 5 ps
    'ensemble': 'NVT'
}
```

**注意**：AIMD计算成本高，建议仅在必要时使用，或用于生成初始训练集。

---

## 5. 模型压缩

### 5.1 压缩原理

DeePMD-kit的模型压缩技术通过以下方式加速推理：

1. **表格化推理（Tabulated Inference）**：将嵌入网络预计算为查找表
2. **算子融合（Operator Merging）**：减少GPU kernel启动开销
3. **精确邻居索引（Precise Neighbor Indexing）**：优化邻居列表构建

```
原始模型推理: 神经网络前向传播 (计算密集型)
        ↓
压缩模型推理: 查表 + 简单运算 (内存密集型)
```

### 5.2 压缩命令

```bash
# 冻结模型
dp freeze -o graph.pb

# 压缩模型
dp compress -i graph.pb -o graph-compress.pb \
    -s 0.01           # 表格步长 \
    -e 5.0            # 外推范围 \
    -f 100            # 溢出检查频率
```

### 5.3 性能提升

根据系统不同，压缩模型可实现：

| 指标 | 原始模型 | 压缩模型 | 加速比 |
|------|---------|---------|--------|
| 推理速度 | 1x | 10-20x | 10-20倍 |
| 内存占用 | 1x | 0.05x | 减少95% |
| 精度损失 | - | < 0.1% | 可忽略 |

### 5.4 压缩版DeepPot-SE

对于`se_e2_a`和`se_e3`类型的描述符，压缩效果最佳：

```json
{
  "model": {
    "descriptor": {
      "type": "se_e2_a",
      "neuron": [25, 50, 100],
      "axis_neuron": 16
    }
  }
}
```

---

## 6. 收敛判断标准

### 6.1 误差阈值标准

```python
CONVERGENCE_CRITERIA = {
    'max_force_error': 0.05,      # eV/Å
    'max_energy_error': 0.001,    # eV/atom  
    'max_virial_error': 0.01,     # eV
    'candidate_ratio_threshold': 0.05  # 5%
}
```

### 6.2 收敛判断逻辑

```python
def check_convergence(history, current_stats):
    """
    收敛判断函数
    
    需同时满足以下条件:
    1. 力误差 < threshold (连续3轮)
    2. 能量误差 < threshold (连续3轮)
    3. 候选结构比例 < 5%
    4. 达到最小迭代次数
    """
    # 检查误差
    force_ok = all(r['force_rmse'] < 0.05 for r in history[-3:])
    energy_ok = all(r['energy_rmse'] < 0.001 for r in history[-3:])
    
    # 检查候选比例
    candidate_ratio = current_stats['candidate'] / current_stats['total']
    ratio_ok = candidate_ratio < 0.05
    
    # 检查迭代次数
    min_iter_ok = len(history) >= 5
    
    return force_ok and energy_ok and ratio_ok and min_iter_ok
```

### 6.3 收敛曲线分析

典型的主动学习收敛曲线：

```
误差
  │
  │    ╲
  │     ╲_______
  │              ╲________
  │                       ╲_______
  │                                ╲______
  │                                       ╲____
  │                                             ╲
  └────────────────────────────────────────────────▶ 迭代
    初期          中期              后期
    (快速下降)     (缓慢优化)        (收敛平稳)
```

### 6.4 收敛诊断

如果收敛缓慢，检查：

1. **阈值是否合适**：
   - 阈值过高 → 候选太少，数据不足
   - 阈值过低 → 候选太多，计算资源浪费

2. **探索策略是否充分**：
   - 温度范围是否足够宽
   - 是否覆盖了所有感兴趣的相空间区域

3. **模型容量是否足够**：
   - 增加网络宽度/深度
   - 考虑使用DPA-1/DPA-2等先进架构

---

## 7. 最佳实践

### 7.1 初始训练集构建

#### 7.1.1 最小数据集规模
```
简单体系 (1-2元素): 50-100 结构
中等复杂度 (3-4元素): 200-500 结构
复杂体系 (5+ 元素): 500-1000+ 结构
```

#### 7.1.2 初始数据多样性
```python
initial_structures = [
    # 晶体结构
    bulk_crystal,
    
    # 表面结构
    (1,0,0), (1,1,0), (1,1,1) surfaces,
    
    # 缺陷结构
    vacancy_structure,
    interstitial_structure,
    
    # 变形结构
    strained_structures (ε = -0.1 to +0.1),
    
    # AIMD快照
    aimd_snapshots (T = 300, 600, 900 K)
]
```

### 7.2 迭代策略

#### 7.2.1 迭代频率
```
推荐设置:
- 每轮最多标注: 50-100 结构
- 最大迭代次数: 10-20 轮
- 最小迭代次数: 5 轮 (避免过早收敛)
```

#### 7.2.2 数据平衡
```python
# 确保各类结构比例合理
structure_weights = {
    'bulk': 0.4,
    'surface': 0.2,
    'defect': 0.2,
    'high_temp': 0.2
}
```

### 7.3 模型集成

```python
# 训练4个独立模型，使用不同随机种子
n_models = 4
seeds = [1, 1001, 2001, 3001]

for i, seed in enumerate(seeds):
    config.seed = seed
    train_model(config, output_dir=f"model_{i}")
```

### 7.4 质量控制检查

```python
def quality_checks(model, test_data):
    """质量控制检查清单"""
    
    # 1. 能量-力一致性检查
    e_forces = -np.gradient(energies)
    force_consistency = np.allclose(e_forces, forces, atol=0.1)
    
    # 2. 对称性检查
    # 晶体结构应有预期的对称性
    
    # 3. 渐进行为检查  
    # 远距离应有正确的渐进行为
    
    # 4. 相稳定性检查
    # 各相的相对稳定性应与DFT一致
    
    return all_checks_passed
```

### 7.5 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 候选结构过多 | 阈值过低 | 提高 f_trust_lo |
| 候选结构过少 | 阈值过高/模型过拟合 | 降低 f_trust_lo，检查模型 |
| MD不稳定 | 力预测不准确 | 增加训练数据，降低初始温度 |
| 收敛缓慢 | 探索不充分 | 扩大温度/压力范围 |
| 特定区域误差大 | 数据覆盖不足 | 针对性采样该区域 |

### 7.6 计算资源规划

```
典型资源需求（以100原子体系为例）：

每轮DFT计算:
- 每结构: ~100 CPU核心小时
- 50结构/轮: ~5000 CPU核心小时

模型训练:
- 每模型: ~4 GPU小时
- 4模型集成: ~16 GPU小时

完整工作流 (10轮):
- DFT: ~50000 CPU核心小时
- 训练: ~160 GPU小时
- MD探索: ~1000 CPU核心小时
```

---

## 8. 参考文献

### 8.1 核心文献

1. **DP-GEN原始论文**
   - Zhang, Y., et al. "DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models." *Computer Physics Communications* 253 (2020): 107206.

2. **DeePMD-kit v2**
   - Wang, H., et al. "DeePMD-kit v2: A software package for deep potential models." *The Journal of Chemical Physics* 156.12 (2022): 124801.

3. **Deep Potential方法**
   - Zhang, L., et al. "Active learning of uniformly accurate interatomic potentials for materials simulation." *Physical Review Materials* 3.2 (2019): 023804.

### 8.2 应用案例

4. **FePt合金**
   - "Development of a deep potential model for FePt alloys: DFT-level accuracy in high-temperature mechanical simulations"

5. **Zr金属**
   - "Development of a Machine Learning Interatomic Potential for Zirconium and Its Verification in Molecular Dynamics"

6. **Mo-Re合金**
   - "A high accuracy machine-learning potential model for Mo-Re binary alloy"

### 8.3 相关资源

- DP-GEN GitHub: https://github.com/deepmodeling/dpgen
- DeePMD-kit文档: https://docs.deepmodeling.com/projects/deepmd
- DeepModeling教程: https://tutorials.deepmodeling.com

---

## 附录：快速参考表

### A1. 阈值推荐表

| 体系类型 | f_trust_lo | f_trust_hi | 适用场景 |
|---------|------------|------------|---------|
| 轻元素 | 0.03 | 0.10 | H, C, N, O 小分子 |
| 金属 | 0.05 | 0.15 | Cu, Al, Fe 等 |
| 高温金属 | 0.08 | 0.25 | T > 1000K |
| 离子化合物 | 0.05 | 0.15 | NaCl, LiF 等 |
| 复杂氧化物 | 0.10 | 0.30 | 多元氧化物 |

### A2. 训练参数模板

```json
{
  "model": {
    "descriptor": {
      "type": "se_e2_a",
      "rcut": 6.0,
      "rcut_smth": 0.5,
      "sel": [46, 46],
      "neuron": [25, 50, 100],
      "axis_neuron": 16
    },
    "fitting_net": {
      "neuron": [240, 240, 240]
    }
  },
  "learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 3.51e-8,
    "decay_steps": 5000
  },
  "loss": {
    "start_pref_e": 0.02,
    "limit_pref_e": 1,
    "start_pref_f": 1000,
    "limit_pref_f": 1,
    "start_pref_v": 0.01,
    "limit_pref_v": 1
  },
  "training": {
    "numb_steps": 1000000,
    "batch_size": "auto"
  }
}
```

---

*报告生成日期: 2025-03-09*  
*版本: 1.0*
