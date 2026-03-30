# 机器学习力场前沿研究综述报告
## 模块3：长程相互作用模型（Long-Range Interactions）

### 3.1 长程相互作用的重要性

在分子模拟中，长程相互作用（特别是静电相互作用）对许多物理化学性质至关重要：
- 离子体系的晶格能
- 极性分子的偶极矩
- 介电常数
- 相变行为
- 界面/表面性质

**挑战**:
- 传统ML势通常只描述短程相互作用（截断半径3-6 Å）
- 长程库仑相互作用衰减缓慢（~1/r）
- 直接学习长程相互作用需要超大截断半径，计算成本极高

---

### 3.2 DeePMD长程修正（DPLR）

**方法**: Deep Potential Long Range (DPLR)
**发表**: DeePMD-kit官方文档，持续更新

#### 3.2.1 理论基础

DPLR模型将总能量分解为短程和长程贡献：
```
E = E_DP + E_ele

其中:
- E_DP: 标准短程深度势能，拟合 (E_DFT - E_ele)
- E_ele: 静电相互作用能
```

**静电能计算**（傅里叶空间）：
```
E_ele = (1/2πV) Σ_{m≠0, |m|≤L} [exp(-π²m²/β²) / m²] S²(m)

其中:
- β: 高斯展宽参数
- S(m): 结构因子
- V: 系统体积
```

**结构因子**:
```
S(m) = Σ_i q_i exp(-2πim·r_i) + Σ_n q_n exp(-2πim·W_n)

其中:
- r_i: 离子坐标
- q_i: 离子电荷
- W_n: Wannier中心坐标（来自深度Wannier模型）
```

#### 3.2.2 两步训练流程

**第一步：训练深度Wannier模型（DW）**
```json
{
  "fitting_net": {
    "type": "dipole",
    "dipole_type": [0],
    "neuron": [128, 128, 128]
  },
  "loss": {
    "type": "tensor",
    "pref": 0.0,
    "pref_atomic": 1.0
  }
}
```
- 预测Wannier中心（WC）相对于原子的位置
- 使用原子偶极矩数据训练（来自VASP+Wannier90）

**第二步：训练DPLR能量模型**
```json
{
  "modifier": {
    "type": "dipole_charge",
    "model_name": "dw.pb",
    "model_charge_map": [-8],
    "sys_charge_map": [6, 1],
    "ewald_h": 1.00,
    "ewald_beta": 0.40
  }
}
```

#### 3.2.3 LAMMPS集成

**DPLR MD模拟配置**:
```
# 原子类型定义
group real_atom type 1 2      # O和H
group virtual_atom type 3      # Wannier中心

# 势函数设置
pair_style deepmd ener.pb
pair_coeff * *

# 虚拟键（用于映射）
bond_style zero
bond_coeff *
special_bonds lj/coul 1 1 1 angle no

# 长程静电（PPPM-DPLR）
kspace_style pppm/dplr 1e-5
kspace_modify gewald 0.40 diff ik mesh 64 64 64

# DPLR fix
fix 0 all dplr model ener.pb type_associate 1 3 bond_type 1
fix_modify 0 virial yes
```

#### 3.2.4 DPLR误差分析

**高斯近似引入的误差**:
- 主要误差来源：偶极-四极相互作用
- 衰减：~r⁻⁴（比原始库仑相互作用的r⁻¹快得多）

**适用体系**:
- 水/冰体系
- 离子液体
- 电解质溶液
- 极性晶体

---

### 3.3 电偶极矩预测

#### 3.3.1 原子偶极模型

**深度偶极模型**:
```
μ_atom = NN(原子环境)
μ_total = Σ_i μ_atom,i
```

**应用**:
- 红外光谱预测
- 介电响应计算
- 极化率预测

#### 3.3.2 MGNN偶极矩预测 (2025)

**MGNN架构**:
- 输出模块支持向量和张量输出
- 可直接预测偶极矩和极化率

**乙醇真空红外光谱预测结果**:
- 与从头算计算高度一致
- 可用于拉曼光谱预测

---

### 3.4 其他长程修正方法

#### 3.4.1 多范围修正（Range-Corrected）MLP

**思想**:
- 将相互作用分为短程、中程、长程
- 不同范围使用不同模型或解析表达式

**实现** (2025年文献):
```
E = E_short(MLP) + E_medium(MLP/correction) + E_long(analytical)
```

**应用场景**:
- QM/MM计算
- 大尺度粗粒化模拟

#### 3.4.2 可学习的长程核

**方法**:
- 使用神经网络学习有效的长程相互作用核
- 结合物理约束（如高斯展宽）

**优势**:
- 比显式Ewald求和更快
- 可包含极化效应

---

### 3.5 长程ML势的比较

| 方法 | 精度 | 计算成本 | 实现复杂度 | 适用体系 |
|------|------|---------|-----------|---------|
| DPLR | 高 | 中等 | 中等 | 水、离子体系 |
| 多范围修正 | 中等-高 | 低 | 低 | QM/MM |
| 可学习核 | 中等 | 低 | 高 | 通用 |
| 大截断MLP | 高 | 很高 | 低 | 小体系 |

---

### 3.6 实践建议

#### 3.6.1 体系选择指南

**使用DPLR**:
- 水/冰体系
- 离子液体
- 电解质
- 需要精确静电的极性体系

**使用多范围修正**:
- QM/MM模拟
- 大尺度生物分子
- 计算资源受限场景

**使用大截断MLP**:
- 小体系（<1000原子）
- 静电效应不主导
- 验证性计算

#### 3.6.2 训练数据准备

**DPLR训练数据**:
1. DFT计算（VASP/CP2K等）
2. Wannier90计算Wannier中心
3. 原子偶极矩数据：`atomic_dipole = wannier_position - atom_position`

**数据格式**:
```
data/
  ├── set.000/
  │   ├── coord.npy
  │   ├── energy.npy
  │   ├── force.npy
  │   └── atomic_dipole.npy  # DPLR必需
  └── type.raw
```

---

### 3.7 2025年最新进展

#### 3.7.1 极化MLP

**最新研究**:
- 显式包含极化响应的MLP
- 可变形电荷模型
- 环境依赖的偶极矩

#### 3.7.2 高效Ewald实现

**优化方向**:
- GPU加速的PPPM算法
- 混合精度计算
- 自适应网格

---

### 3.8 模块总结

**关键要点**:
1. **DPLR**是目前最成熟的长程ML势方案
2. **两步训练**（DW模型+DPLR模型）是标准流程
3. **Wannier中心**是连接短程MLP和长程静电的桥梁
4. **偶极矩预测**能力对光谱计算至关重要

**实践建议**:
- 极性体系必须考虑长程修正
- DPLR与LAMMPS集成完善，适合生产环境
- 大体系考虑多范围修正或粗粒化

---

*报告生成时间: 2026-03-08*
*下一模块: 多元素体系训练*
