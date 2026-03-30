# DFT方法学前沿研究 - 模块3：量子嵌入方法

## 研究时间：2026-03-08
## 模块状态：✅ 完成

---

## 一、DMET（密度矩阵嵌入理论）最新进展

### 1.1 理论框架

DMET是一种基于波函数的量子嵌入方法，核心思想：
- 将系统分为**杂质（Impurity）**和**环境（Environment）**
- 通过**Schmidt分解**获得杂质和bath轨道
- 环境部分用低精度方法处理，杂质部分用高精度方法

**关键公式**：
```
|Ψ⟩ = Σ_α^N λ_α |A_α⟩ |B_α⟩
```
其中|A_α⟩是杂质态，|B_α⟩是bath态。

### 1.2 DMET变体

| 变体 | 特点 | 应用场景 |
|------|------|----------|
| **Full DMET** | 完全自洽，优化关联势和化学势 | 强关联体系基态 |
| **One-shot DMET** | 仅优化化学势 | 快速预测 |
| **Single Impurity DMET** | 单杂质+环境 | 局域缺陷 |

### 1.3 最新软件实现

**InQuanto 2.1.1（2025）**：
- 支持多种DMET变体
- 量子计算集成
- 活性空间求解器接口

**libDMET + PennyLane教程**：
```python
# DMET自洽循环四步：
1. 构造杂质哈密顿量
2. 求解杂质问题（FCI/量子算法）
3. 计算全体系能量
4. 更新杂质-环境相互作用
```

### 1.4 非正交局域轨道DMET（2024）

**北京大学姜鸿课题组进展**：
- 突破传统正交局域轨道限制
- 非正交分解Slater行列式
- 应用于3d/4f单离子磁体
- 零场分裂参数准确预测

---

## 二、DFT+DMFT最新进展

### 2.1 核心概念对比

| 特性 | DMET | DMFT |
|------|------|------|
| 嵌入对象 | 密度矩阵 | Green函数 |
| 频率依赖 | 静态 | 动态 |
| 适合问题 | 基态能量 | 光谱、激发态 |

### 2.2 中国物理B最新成果（2024.10）

**任新国课题组LCNAO-DFT+DMFT**：
- 线性组合数值原子轨道框架
- 电荷自洽DFT+DMFT形式化
- 连接三种CT-QMC杂质求解器
- 验证：3d、4f、5f强关联电子体系

**技术特点**：
- 全势全电子DFT代码接口
- 可扩展架构
- 桥接不同LCNAO DFT包

### 2.3 平移对称性恢复（2024）

**全轨道DMFT新进展**：
- 重叠原子中心杂质片段
- 对称性适应的嵌入问题设计
- 二维氮化硼和石墨烯测试
- GW + DMFT组合计算

### 2.4 量子计算+DMFT（2024.04）

**IBM Quantum实现**：
- 14量子比特硬件实验
- Lehmann表示量子杂质求解器
- Ca₂CuO₂Cl₂实际材料计算
- 误差缓解+零噪声外推

---

## 三、子系统DFT（Subsystem DFT）

### 3.1 冻结密度嵌入（FDE）

**WIREs 2024综述更新**：
- 非可加动能泛函（NAKE）新近似
- 势重建策略改进
- 3-FDE方法处理共价键合体系
- 投影基嵌入（PbE）

### 3.2 投影基嵌入（PbE）

**特点**：
- 用投影算符替代NAKP项
- KS方程中的正交性保证
- WF-in-DFT嵌入的重要参考

### 3.3 活性空间嵌入通用框架（2024.12）

**Nature Computational Science**：
- 轨道空间分离的混合量子-经典计算框架
- 分子和周期性体系统一处理
- 范围分离DFT嵌入
- Wannier函数定位方案

---

## 四、Wannier函数最新进展

### 4.1 Wannier函数软件生态系统（2024-2025）

**Reviews of Modern Physics封面文章**（2024）：
- Wannier引擎：计算Wannier函数
- 接口代码：连接DFT/GW与Wannier
- Wannier启用代码：加速精确模拟

**在线注册表**：
https://wannier-developers.github.io/wannier-ecosystem-registry

### 4.2 Wannier90 v3.x功能

- 自旋or投影
- 并行执行（MPI）
- van der Waals相互作用计算
- Landauer-Buttiker输运
- Boltzmann输运

### 4.3 PDWF算法扩展（2025）

**投影性判别Wannier函数**：
- 磁性体系支持
- 自旋-轨道耦合处理
- 氢化投影器自动添加
- AiiDA工作流自动化

**性能指标**：
- 成功率 >98%（η₂ ≤ 20 meV）
- 金属：至Fermi能级+2 eV
- 半导体：至CBM+2 eV

### 4.4 选择局域化Wannier函数

**固定中心约束**：
- 修改泛函最小化
- 保持高局域度
- 并发选择局域化

---

## 五、嵌入方法对比与应用

### 5.1 方法选择指南

| 应用场景 | 推荐方法 | 原因 |
|----------|----------|------|
| 分子基态能量 | DMET | 精确、可扩展 |
| 固体光谱 | DFT+DMFT | 动态关联 |
| 溶剂化效应 | Subsystem DFT | 高效、物理清晰 |
| 缺陷态 | Wannier+DMFT | 局域轨道 |
| 量子计算 | Quantum Embedding | 减少量子比特 |

### 5.2 应用案例

**1. 氢化Liₙ团簇能量预测**
- DMET与CCSD结果对比
- 活性空间4e6o和6e7o

**2. Ce基重费米子超导体**
- DFT+DMFT + 有效模型
- 压力诱导QCP和SC-KP转变

**3. LiNiO₂阴极**
- DMFT解释室温绝缘行为
- Mott-电荷转移带隙

---

## 六、关键文献

1. Wouters et al., JCTC 2016: "A Practical Guide to DMET"
2. Qu et al., Chin. Phys. B 2024: "Charge self-consistent DFT+DMFT with LCNAO"
3. Marrazzo et al., Rev. Mod. Phys. 2024: "Wannier-function software ecosystem"
4. Qiao et al., npj Comput. Mater. 2025: "Robust Wannierization including magnetization"
5. Jacob et al., WIREs 2024: "Subsystem DFT update"
6. Rossmannek et al., npj 2024: "Active space embedding framework"

---

**模块3完成时间**：2026-03-08 17:35 GMT+8
**下一模块**：强关联体系计算（DFT+U、动态平均场、杂化泛函）
