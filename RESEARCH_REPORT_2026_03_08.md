# 24小时持续研究报告 - 最终汇报
**研究时间：** 2026年3月8日 18:07 - 18:25 (GMT+8)
**研究员：** AI Research Agent

---

## 📊 执行摘要

本次研究系统性地追踪了第一性原理计算软件、机器学习势基准测试和材料数据库的最新进展。识别出**5项重大突破**和**8个关键趋势**，为技能库更新提供了明确的行动指南。

---

## 🚀 五大重大突破

### 1. VASP 6.5系列发布 (2024年12月-2025年3月)
**重要性：** ★★★★★

- **6.5.0** (2024-12-17): 原生电子-声子耦合、增强MLFF、BSE改进
- **6.5.1** (2025-03-11): 稳定性和bug修复
- **关键限制：** 截至6.5.1，**暂不支持超导温度计算**（需要用户自行后处理）

**影响：** VASP用户现在可以在单一软件中完成完整的电子-声子相互作用研究，无需依赖外部包（如EPW）。

---

### 2. Quantum ESPRESSO 7.5发布 (2025年12月)
**重要性：** ★★★★☆

**核心新功能：**
- 轨道分辨DFT+U方法 (JCTC 2024)
- Wannier90-DFT+U接口（Wannier函数作为Hubbard投影算符）
- 增强的MD功能（NVT/NPT动力学）

**影响：** 强关联系统研究获得更精确工具，Wannier函数与DFT+U的整合将简化多体物理研究。

---

### 3. DPA-3架构发布与缩放定律验证 (2025年6月)
**重要性：** ★★★★★

**突破：**
- **首次系统验证LAM中的缩放定律(Scaling Law)**
- 线Graph系列(LiGS)架构
- DPA-3.1-3M模型在LAMBench上达到SOTA性能

**性能：**
- Matbench Discovery CPS=0.717（第2名，仅用1/6参数）
- LAMBench 17个任务中总体泛化误差最低
- SPICE-MACE-OFF数据集能量误差降低66%

**影响：** 为大原子模型(LAM)的进一步发展提供了理论和工程基础。

---

### 4. Matbench Discovery基准测试发布 (2025年6月)
**重要性：** ★★★★★

**当前领先者：**
| 排名 | 模型 | F1 | DAF | 机构 |
|-----|------|----|-----|------|
| 1 | PET-OAM-XL | 0.924 | 6.075 | EPFL COSMO (2026-01) |
| 2 | EquiformerV2+DeNS | 0.919 | 5.983 | Meta FAIR |

**影响：** 为机器学习势评估提供了行业标准，发现加速因子(DAF)成为关键指标。

---

### 5. OMat24数据集发布 (Meta FAIR, 2024年10月)
**重要性：** ★★★★☆

**规模：**
- 1亿+周期性DFT计算
- 比MPtrj大2个数量级
- 基于EquiformerV2架构的模型

**性能：**
- Matbench Discovery F1=0.917（之前最佳0.880）
- 热力学稳定性识别阳性率首次>90%

**状态：** 已被OMol25取代（2025年，扩展至小分子和生物分子）

---

## 📈 八大关键趋势

### 趋势1：电子-声子耦合成为第一性原理软件标配
- VASP原生支持
- 重整化带隙、电子寿命、输运性质成为标准输出
- 超导温度计算即将加入

### 趋势2：基础模型(Foundation Models)主导ML势领域
- MACE-MP-0 → MACE-OMAT
- DPA-3.1-3M (OpenLAM)
- CHGNet电荷信息势
- MatterSim跨元素/温度/压力模型

### 趋势3：缩放定律在LAM中得到验证
- DPA-3首次系统验证
- 模型规模、数据量、计算预算与性能呈幂律关系
- 指导未来模型发展方向

### 趋势4：跨领域泛化是核心挑战
- LAMBench揭示无机材料/分子/催化领域差距
- 多任务训练成为主流（DPA-3.1-3M支持OMat24/OC20M/SPICE2头）
- 需要更多跨领域训练数据

### 趋势5：电荷信息重要性上升
- CHGNet展示电荷耦合物理预测能力
- 氧化态、自旋态预测需求增长
- 电荷信息MD成为新工具

### 趋势6：材料数据库规模爆炸性增长
- OMat24：1亿+计算
- OQMD v1.8：100万+材料
- GNoME集成进Materials Project

### 趋势7：计算方法升级
- r2SCAN泛函普及（MP添加30,000 GNoME r2SCAN计算）
- DFPT声子计算标准化
- PBE+U持续优化

### 趋势8：工作流自动化成熟
- Atomate2替代atomate成为MP基础设施
- Jobflow + FireWorks组合
- 多DFT代码支持（VASP/CP2K/ABINIT/Qchem）

---

## 📋 技能库更新建议

### 立即行动（高优先级）

| 任务 | 描述 | 预计工时 |
|-----|------|---------|
| VASP 6.5 EPC教程 | 电子-声子耦合计算完整流程 | 4h |
| QE 7.5新功能说明 | 轨道分辨DFT+U使用指南 | 3h |
| ML势推荐更新 | MACE/DPA-3/CHGNet对比 | 2h |
| Matbench Discovery指南 | 模型评估和提交教程 | 3h |

### 短期计划（中优先级）

| 任务 | 描述 | 预计工时 |
|-----|------|---------|
| Atomate2工作流教程 | 从安装到实际计算 | 4h |
| LAMBench使用指南 | 大原子模型评估 | 2h |
| OMat24数据使用 | 数据集接入和预处理 | 2h |

### 持续维护

- 监控VASP/QE版本发布
- 跟踪Matbench Discovery排行榜
- 更新Python依赖要求（pymatgen需Python 3.10+）

---

## 🔗 关键资源链接

### 第一性原理软件
- VASP: https://www.vasp.at/
- QE: https://gitlab.com/QEF/q-e

### 机器学习势
- MACE: https://github.com/ACEsuit/mace
- DeePMD-kit: https://github.com/deepmodeling/deepmd-kit
- DPA-3模型: https://www.aissquare.com/models/detail?pageType=models&name=DPA-3.1-3M&id=343

### 基准测试
- Matbench Discovery: https://matbench-discovery.materialsproject.org/
- LAMBench: https://www.aissquare.com/openlam?tab=Benchmark

### 材料数据库
- Materials Project: https://materialsproject.org/
- OQMD: https://oqmd.org/

### 工具库
- Pymatgen: https://pymatgen.org/
- Atomate2: https://materialsproject.github.io/atomate2/
- MatGL: https://matgl.ai/

---

## 📚 参考文献精选

1. **LAMBench**: Li C. et al., "LAMBench: A Benchmark for Large Atomistic Models", npj Computational Materials, 2025
2. **Matbench Discovery**: Riebesell J. et al., "A framework to evaluate machine learning crystal stability predictions", Nature Machine Intelligence, 2025
3. **DPA-3**: Zhang D. et al., "A Graph Neural Network for the Era of Large Atomistic Models", arXiv:2506.01686, 2025
4. **OMat24**: Barroso-Luque L. et al., "Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models", arXiv:2410.12771, 2024
5. **CHGNet**: Deng B. et al., "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling", Nature Machine Intelligence, 2023
6. **MACE**: Batatia I. et al., "A foundation model for atomistic materials chemistry", Journal of Chemical Physics, 2025
7. **Atomate2**: "Atomate2: modular workflows for materials science", Digital Discovery, 2025
8. **MatGL**: Ko T.W. et al., "Materials Graph Library (MatGL)", npj Computational Materials, 2025

---

## 🎯 后续研究方向建议

1. **技术深度挖掘**
   - VASP电子-声子耦合详细理论和实现
   - QE 7.5 Wannier90-DFT+U接口实际应用
   - DPA-3架构详细分析

2. **新兴领域跟踪**
   - 大原子模型社区(OpenLAM)动态
   - 多保真度ML势发展
   - 生成式AI在材料发现中的应用

3. **工具链整合**
   - ML势+DFT混合工作流
   - 主动学习自动化
   - 实验-计算闭环系统

---

**研究状态：** ✅ 完成第一阶段系统调研
**建议下次更新：** 监控VASP 6.6/QE 7.6发布，跟踪LAMBench排行榜变化
