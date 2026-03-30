# 24小时持续研究报告 - 2026年3月8日

## 研究状态：进行中

---

## 一、第一性原理计算软件最新版本

### 1.1 VASP 6.5+ 新功能 (2024年12月-2025年3月)

#### VASP 6.5.0 (2024年12月17日发布)
**核心新功能：**
- **电子-声子耦合 (Electron-phonon coupling)** - VASP原生支持，无需外部包
  - 支持随机采样方法和多体微扰理论两种方法
  - 可计算重整化带隙
  - 支持电子寿命和输运性质计算（电导率、载流子迁移率、热电系数、ZT值）
  
- **机器学习力场 (MLFF) 增强**
  - On-the-fly主动学习训练
  - 支持快速预测模式 (ML_MODE = refit)，提速20-100倍
  - 贝叶斯误差估计和RMSE分析
  
- **Bethe-Salpeter方程 (BSE) 改进**
- **py4vasp数据分析和可视化工具**

#### VASP 6.5.1 (2025年3月11日发布)
- 大量bug修复和性能改进
- MLFF稳定性增强

**参考资料：**
- https://www.vasp.at/info/post/
- VASP官方Wiki - Best practices for MLFF

---

### 1.2 Quantum ESPRESSO 7.5 新功能 (2025年12月发布)

#### 核心新功能：
1. **轨道分辨DFT+U方法 (Orbital Resolved DFT+U)**
   - 作者：E. Macke, I. Timrov
   - 参考文献：JCTC 2024, 20(11), 4824-4843

2. **Wannier90与DFT+U接口**
   - 可使用Wannier函数作为Hubbard投影算符
   - 作者：I. Timrov, A. Carta, C. Ederer等
   - arXiv:2411.03937

3. **pp.x扩展**
   - 支持可视化DFT+U的Hubbard投影算符

4. **分子动力学增强**
   - NVT动力学：使用Nose-Hoover恒温器，支持Verlet和velocity-Verlet
   - NPT动力学：支持Nose-Hoover恒温器 + Parrinello-Rahman/Wentzcovich气压恒温器 + Beeman MD引擎

5. **Bug修复**
   - 完全相对论赝势读取问题
   - Berry Phase计算警告
   - 多处理器对称化故障
   - 非线性PAW情况崩溃修复

**兼容性：**
- THERMO_PW 2.1.1 已适配QE 7.5

**参考资料：**
- https://gitlab.com/QEF/q-e/-/releases/qe-7.5
- 中文介绍：http://mp.weixin.qq.com/s?__biz=MzU1MDkyODA0MA==&mid=2247486179&idx=3&sn=2e2324acf3a64a74e6b81224e915bb5f

---

## 二、机器学习势最新基准测试

### 2.1 LAMBench - 大型原子模型综合基准 (2025年)

**发表：** Nature npj Computational Materials, 2025
**代码：** https://github.com/deepmodeling/lambench
**排行榜：** https://www.aissquare.com/openlam?tab=Benchmark

#### 评估的三个核心能力：
1. **泛化性 (Generalizability)** - 跨领域原子系统预测准确性
2. **适应性 (Adaptability)** - 微调用于结构-性质关系任务
3. **适用性 (Applicability)** - 真实模拟中的稳定性和效率

#### 测试领域：
- **无机材料 (Inorganic Materials)**：MDR Phonon benchmark（声子性质）、弹性benchmark
- **分子 (Molecules)**：TorsionNet500（扭转能量）、Wiggle150（高应变构型）
- **催化 (Catalysis)**：OC20NEB-OOD（反应能垒预测）

#### 测试的10个LAMs (截至2025年8月1日)：
| 模型 | 类型 | 表现亮点 |
|------|------|---------|
| DPA-3.1-3M (MPtrj) | 多任务 | 力场泛化性最佳 |
| SevenNet-MF-ompa (MPA) | 多任务 | 无机材料领域最佳 |
| GRACE-2L-OAM | 单任务 | 无机材料表现优异 |
| Orb-v3 | 单任务 | 分子领域表现良好 |
| MACE-MP-0 | 基础模型 | 广泛适用 |
| MatterSim-v1-5M | 单任务 | - |

#### 关键发现：
- 当前LAM与理想通用PES仍有显著差距
- 跨领域训练数据对泛化性至关重要
- 多保真度推理支持满足不同XC泛函需求
- 保持模型保守性和可微性对MD模拟稳定性重要

---

### 2.2 Matbench Discovery 基准

**核心指标：**
- 评估模型预测材料稳定性的能力
- 与LAMBench无机材料领域测试结果高度一致

---

### 2.3 经典基准数据集更新

#### QM9
- 134k小分子（C,H,O,N,F，≤9重原子）
- 分子性质预测（能量、HOMO/LUMO、偶极矩）
- 链接：https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

#### MD17
- 8种小有机分子MD轨迹（苯、乙醇等）
- ~3-4M构型
- 能量和力预测
- 链接：http://quantum-machine.org/datasets/#md-datasets

#### MD22 (新增)
- 大分子/生物分子片段（42-370原子）
- 0.2M构型
- 链接：https://www.openqdc.io/datasets/md22

---

### 2.4 MACE基础模型最新进展

#### MACE-MP-0 (2024年发布)
**训练数据：**
- Materials Project MPtrj数据集
- ~150,000种晶体材料
- ~1.5M结构（PBE+U级别）

**覆盖范围：**
- 89种元素
- 适用于30多种原子模拟类型（冰结构、MOF、催化剂、非晶结构、液固界面等）

#### 最新迭代版本 (2025年)
| 模型名称 | 训练数据集 | 特点 |
|---------|-----------|------|
| matpbs-pbe-omat-ft | MATPES-PBS | 无+U修正 |
| mpa-0-medium | MPtrj + sAlex | 高压稳定性改进 |
| mp-0b3-medium | MPtrj | 声子性质改进 |
| omat-0-medium | OMAT | 优秀声子性质 |

**关键发现：**
- 从MPtrj到OMAT的改进主要来自数据集扩展
- 包含Boltzmann分布采样的rattled结构和MD轨迹
- 对非谐固体（如立方钙钛矿）的PES描述更准确

---

### 2.5 ML哈密顿量模型基准

| 模型 | 类型 | 特点 |
|------|------|------|
| DeepH | 消息传递GNN | 从晶体结构直接预测哈密顿量 |
| DeepH-E3 | E(3)-等变Transformer | 严格旋转平移等变性 |
| HamGNN | E(3)-等变卷积GNN | 紧束缚哈密顿量预测 |
| DeepH-hybrid | 杂化泛函预测器 | 直接预测杂化泛函哈密顿量 |
| xDeepH | 自旋轨道GNN | 磁性超结构预测 |
| WANet | SO(2)卷积+MoE | 大规模分子哈密顿量预测 |

---

## 三、材料数据库最新发布

### 3.1 Materials Project 2025更新

#### 当前版本：v2025.09.25 (2025年9月25日上线)

#### 2025年重要更新：

**v2025.09.25 (2025年9月25日)**
- 插入电极集合修正
- 修复过滤错误，添加~1,200插入电极文档
- 确保所有数据来自相同热力学凸包(GGA_GGA+U)

**v2025.06.11 (2025年6月11日)**
- 迁移~1,500种材料的DFPT声子数据（与atomate2新声子工作流一致）
- 新schema支持高效存储声子能带和DOS

**v2025.04.18 (2025年4月18日)**
- 添加30,000种GNoME来源材料（r2SCAN计算）
- Princeton大学Kingsbury实验室贡献1,133计算（PBEsol和r2SCAN混合）

**v2025.02.28 (2025年2月28日)**
- 弹性集合更新：废弃290个不合理弹性模量文档
- 弹性模量范围检查：-100 GPa 至 800 GPa

**v2025.02.12 (2025年2月12日)**
- 添加1,073种镱(Yb)材料（Yb_3赝势+r2SCAN重新弛豫）
- ~30种新型杂化无机/有机甲酸盐钙钛矿

#### 历史重要更新：
- **MaterialsProject2020Compatibility**：新能量校正方案
- 支持不确定性量化
- 新增Br, I, Se, Si, Sb, Te的校正

**参考资料：**
- https://docs.materialsproject.org/changes/database-versions
- https://next-gen.materialsproject.org/

---

### 3.2 OQMD (Open Quantum Materials Database) 更新

#### 当前版本：v1.8 (2026年2月发布)
- 数据库大小：21.1 GB
- 包含超过100万种材料
- 包含有机和假设化合物

#### 版本历史：
| 版本 | 发布时间 | 大小 | 特点 |
|-----|---------|------|------|
| v1.8 | 2026年2月 | 21.1 GB | 大量新结构 |
| v1.7 | 2025年5月 | 19.2 GB | 增量更新 |
| v1.6 | 2023年11月 | 16.5 GB | - |
| v1.5 | 2021年10月 | 15.0 GB | - |
| v1.4 | 2020年10月 | 12.0 GB | qmpy API重大变更 |

#### API兼容性：
- qmpy API v1.4兼容所有v1.4+数据库
- 支持OPTIMADE API标准(v1.0.0)
- Base URL: https://oqmd.org/optimade

#### 应用领域：
- 热电材料
- 电池材料
- 高强度合金
- 大规模凸包稳定性分析

**aiOQ数据库：**
- FactSage中可用
- 包含475,887种化合物

**参考资料：**
- https://oqmd.org/download/

---

### 3.3 其他重要数据库更新

#### CoRE MOF 2025
- 简化Web界面
- 支持拖放CIF文件计算几何描述符
- 预测水和热稳定性

#### QMOF数据库
- 已整合进Materials Project
- 提供DFT衍生性质（优化结构、带隙、能带结构）

---

## 四、研究发现汇总

### 关键趋势：

1. **第一性原理软件发展方向：**
   - 电子-声子耦合成为标准功能
   - MLFF原生集成成为主流
   - 更强的Wannier函数支持

2. **机器学习势发展方向：**
   - 基础模型（Foundation Models）崛起
   - 跨领域泛化成为核心挑战
   - 多保真度支持需求增长

3. **材料数据库发展方向：**
   - 更大规模数据集（GNoME等）
   - 更高级的计算方法（r2SCAN）
   - 更好的API标准化（OPTIMADE）

### 新发现的重要进展：

#### 6. OMat24数据集 (Meta FAIR, 2024年10月)
**重大进展：** Meta发布了Open Materials 2024 (OMat24)

**数据集规模：**
- 超过1亿次周期性DFT计算
- 比MPtrj数据集大约2个数量级
- 使用PBE泛函（可选Hubbard U修正）

**数据采样技术：**
- Rattled Boltzmann采样
- 从头算分子动力学(AIMD)
- Rattled弛豫
- 大量非零力和应力的结构

**模型性能：**
- 基于EquiformerV2架构
- 最大模型约1.5亿参数
- Matbench Discovery上F1分数0.917（之前最佳0.880）
- 热力学稳定性识别阳性率首次超过90%

**状态：** 已被OMol25取代（2025年发布）

**链接：**
- 论文：https://arxiv.org/abs/2410.12771
- 数据集：https://huggingface.co/datasets/fairchem/OMAT24
- 模型：https://huggingface.co/fairchem/OMAT24

---

#### 7. DPA-3模型架构发布 (2025年6月)
**重大突破：** Deep Potential团队发布DPA-3

**论文：** https://arxiv.org/abs/2506.01686

**核心创新：**
- **线 graph 系列 (Line Graph Series - LiGS)** 架构
- 同时迭代构建原子、键、角、二面体的高阶图结构
- 严格保持物理不变性（平移、旋转、置换对称性、能量守恒）

**关键验证：**
- **首次系统验证LAM中的缩放定律(Scaling Law)**
- 泛化误差与参数规模、训练步数、计算预算呈幂律关系
- R² = 0.981的高度一致性

**性能表现：**

| 基准测试 | 表现 |
|---------|------|
| SPICE-MACE-OFF | 能量预测误差降低66%（比MACE-OFF23(L)参数更少） |
| TorsionNet-500 | 500个分子扭转势垒预测全部达到化学精度 |
| Matbench Discovery | DPA3-L24模型CPS=0.717，排名第二（仅次于eSEN-30M-MP） |
| LAMBench | DPA-3.1-3M达到SOTA性能，17个力场预测任务中总体泛化误差最低 |

**模型版本：**
- DPA-3.1-3M：327万参数
- 训练数据：OpenLAM dataset v1 - 163M结构
- 代码：https://github.com/deepmodeling/deepmd-kit/releases/tag/v3.1.0
- 快速开始：https://bohrium.dp.tech/notebooks/57111746135

---

#### 8. Matbench Discovery最新排名 (2026年1月)
**链接：** https://matbench-discovery.materialsproject.org/

**当前领先模型：**

| 排名 | 模型 | F1分数 | DAF | 机构 |
|-----|------|--------|-----|------|
| 1 | PET-OAM-XL | 0.924 | 6.075 | EPFL COSMO实验室 |
| 2 | EquiformerV2+DeNS | 0.919 | 5.983 | Meta FAIR |
| 3 | EquFlash | 0.915 | - | - |
| - | DPA3-L24 | - | - | Deep Potential |

**关键指标：**
- **DAF (Discovery Acceleration Factor)**：相比随机选择发现稳定材料的倍数
- 最大可能DAF ≈ 6.54
- PET-OAM-XL达到6.075，接近最优性能

**PET-OAM-XL详情：**
- 基于PET-MAD架构扩展
- 专为材料发现任务优化
- 论文：https://arxiv.org/abs/2601.16195
- EPFL COSMO实验室，2026年1月登顶

---

#### 9. VASP 6.5电子-声子耦合现状
**重要发现：**
- 截至VASP 6.5.1，**暂不支持超导温度计算**
- 用户需要通过ELPH_DRIVER自行后处理电子-声子矩阵元
- 开发团队正在努力添加此功能

**已有功能：**
- 电子-声子矩阵元计算
- 带隙重整化（ZPR）
- 电子寿命
- 输运性质（电导率、迁移率、热电系数）

---

#### 10. MatGL (Materials Graph Library) 更新 (2025年11月)
**论文：** npj Computational Materials 2025
**链接：** https://matgl.ai/

**v2.0.0重大更新 (2025年11月13日)：**
- 新增QET架构
- PyG (PyTorch Geometric) 后端成为默认
- 预训练分子势和PyG框架

**支持的架构：**
| 架构 | 描述 |
|-----|------|
| M3GNet | 通用图深度学习原子间势 |
| MEGNet | 分子和晶体通用ML框架 |
| CHGNet | 电荷信息原子建模预训练势 |
| TensorNet | 笛卡尔张量表示 |
| SO3Net | 等变神经网络 |
| QET (新增) | 最新架构 |

**预训练模型：**
- M3GNet通用势
- CHGNet电荷信息势
- 多保真度MEGNet

---

#### 11. CHGNet电荷信息势
**论文：** Nature Machine Intelligence 2023
**特点：**
- 预训练通用神经网络势
- 融入电荷信息
- 4层原子卷积层
- 捕获最长20Å的长程相互作用
- 适用于固态材料（如LixMnO2的电荷信息MD）

**性能：**
- 比从头算MD快100倍
- 能预测电荷耦合物理（如Mn离子电荷 disproportionation）

---

#### 12. Atomate2工作流框架 (2025年)
**论文：** Digital Discovery 2025
**链接：** https://materialsproject.github.io/atomate2/

**核心功能：**
- 100+计算材料科学工作流
- Materials Project数据库基础设施
- 多DFT代码支持（VASP、CP2K、ABINIT、Qchem）
- 与AMSET、phonopy、Lobsterpy集成

**工作流引擎：**
- Jobflow：工作流定义语言
- FireWorks：大规模工作流管理
- jobflow-remote：远程执行

**2025年培训资源：**
- CECAM研讨会：https://www.cecam.org/workshop-details/automated-ab-initio-workflows-with-jobflow-and-atomate2-1276
- 视频教程：
  - Jobflow和Jobflow-remote
  - Atomate2基础
  - Atomate2高级工作流（Part 1 & 2）

**教程：** https://github.com/materialsproject/atomate2/tree/main/tutorials

---

#### 13. Pymatgen最新状态 (2024-2025)
**链接：** https://pymatgen.org/

**版本要求：**
- Python 3.10+
- 遵循Scientific Python软件栈支持计划

**主要功能：**
1. 灵活的Element、Site、Molecule、Structure类
2. 广泛的I/O支持（VASP、ABINIT、CIF、Gaussian、XYZ等）
3. 相图、Pourbaix图生成
4. 电子结构分析（DOS、能带结构）
5. Materials Project REST API集成

**学习资源：**
- 官方文档：https://pymatgen.org/
- Matgenb教程：https://matgenb.materialsvirtuallab.org/
- Dr. Anubhav Jain YouTube教程

**扩展包：**
- pymatgen-analysis-diffusion：扩散分析（NEB、MD轨迹、RDF、van Hove关联函数）
- pymatgen-analysis-defects：缺陷分析

---

### 持续跟踪项目

| 项目 | 链接 | 状态 |
|-----|------|------|
| VASP官方 | https://www.vasp.at/ | 6.5.1已发布 |
| QE GitLab | https://gitlab.com/QEF/q-e | 7.5已发布 |
| Materials Project | https://materialsproject.org/ | v2025.09.25 |
| OQMD | https://oqmd.org/ | v1.8 |
| Matbench Discovery | https://matbench-discovery.materialsproject.org/ | 活跃更新 |
| LAMBench | https://www.aissquare.com/openlam | 活跃 |
| MACE | https://github.com/ACEsuit/mace | 持续更新 |
| DeePMD-kit | https://github.com/deepmodeling/deepmd-kit | v3.1.0 |
| OpenLAM | https://www.aissquare.com/openlam | 大原子模型社区 |
| Atomate2 | https://materialsproject.github.io/atomate2/ | 活跃开发 |
| MatGL | https://matgl.ai/ | v2.0.0 |
| Pymatgen | https://pymatgen.org/ | Python 3.10+ |

---

## 五、研究发现汇总与趋势分析

### 5.1 第一性原理软件发展趋势

1. **电子-声子耦合成为标配**
   - VASP 6.5原生支持
   - QE持续增强
   - 超导温度计算即将加入

2. **机器学习势原生集成**
   - VASP MLFF On-the-fly训练
   - 速度提升20-100倍

3. **更强的关联电子处理**
   - QE 7.5轨道分辨DFT+U
   - Wannier函数深度整合

### 5.2 机器学习势发展趋势

1. **基础模型(Foundation Models)主导**
   - MACE-MP-0、MACE-OMAT
   - DPA-3.1-3M
   - OMat24训练模型

2. **缩放定律(Scaling Law)验证**
   - DPA-3首次系统验证LAM缩放定律
   - 数据规模和模型大小同样重要

3. **跨领域泛化是核心挑战**
   - LAMBench揭示无机/分子/催化领域差距
   - 多任务训练成为主流

4. **电荷信息重要性上升**
   - CHGNet展示电荷耦合物理预测能力
   - 氧化态、自旋态预测需求增长

### 5.3 材料数据库发展趋势

1. **规模爆炸性增长**
   - OMat24：1亿+计算
   - OQMD v1.8：100万+材料
   - GNoME集成进MP

2. **计算方法升级**
   - r2SCAN泛函普及
   - PBE+U持续优化
   - 声子DFPT标准化

3. **API标准化**
   - OPTIMADE生态系统扩展
   - ASE统一接口

### 5.4 工作流自动化趋势

1. **Atomate2成为主流**
   - 替代原始atomate
   - Jobflow + FireWorks组合
   - 多DFT代码支持

2. **ML势工作流整合**
   - 训练-测试-部署流水线
   - 主动学习自动化

---

## 六、建议的技能库更新

### 高优先级更新：
1. [ ] 添加VASP 6.5电子-声子耦合计算示例
2. [ ] 添加QE 7.5新功能说明
3. [ ] 更新ML势推荐列表（MACE、DPA-3、CHGNet）
4. [ ] 添加Matbench Discovery评估指南
5. [ ] 更新Materials Project API使用文档

### 中优先级更新：
1. [ ] 添加Atomate2工作流教程
2. [ ] 更新LAMBench基准测试说明
3. [ ] 添加OMat24数据集使用指南
4. [ ] 更新MatGL使用示例

### 持续维护：
1. [ ] 监控各软件新版本发布
2. [ ] 跟踪Matbench Discovery排行榜
3. [ ] 更新Python依赖版本要求
4. [ ] 维护最佳实践文档

---

*研究持续进行中...*

---

## 研究时间戳
- 研究开始：2026-03-08 18:07 GMT+8
- 最后更新：2026-03-08 18:15 GMT+8
- 下次更新目标：继续深入各细分领域
