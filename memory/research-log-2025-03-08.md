# 持续研究日志 - 2025-03-08

## 第一轮搜索完成时间：17:14 GMT+8

---

## 🔬 第一性原理计算新进展

### 1. Floquet理论+TDDFT超快动力学 (2025 Roadmap)
- **来源**: IOP Science, 2025-11
- **突破**: 动态能带工程，激光驱动瞬态能带结构调控
- **方法**: Floquet理论 + TDDFT/多体微扰论 + TR-ARPES验证
- **进展**: 
  - 拓扑绝缘体和绝缘体中Floquet态的实验验证
  - 耦合TDDFT-Maxwell描述超强光-物质耦合
  - 量子计算开始应用于Floquet分析

### 2. DFT计算摩擦学突破 (MDPI 2025-10)
- **突破**: 应变工程实现石墨烯本征超润滑
- **关键发现**: 35%双轴压缩应变→势垒完全平坦(∆Emax≈0)
- **机制**: 应变平滑化层间电荷密度波动
- **应用**: 超低摩擦材料设计

### 3. 非平衡格林函数(NEGF)进展
- 计算复杂度持续改进
- 为驱动系统中电子关联提供稳健的第一性原理框架

---

## 🧬 分子动力学前沿方法

### 1. TAIP框架 - 在线测试时自适应ML势
- **论文**: Nature Communications, 2025-02-22
- **团队**: 复旦大学/上海AI Lab
- **问题**: MLIP训练/测试分布漂移导致MD崩溃
- **解决**: 双层级自监督学习（全局结构+原子局部环境）
- **效果**: 无需额外数据桥接域间隙

### 2. 动态性质精细化ML势
- **论文**: Nature Communications, 2025-01-18
- **团队**: 韩斌/于强
- **突破**: 利用光谱/输运系数优化ML势
- **方法**: 伴随法+梯度截断解决内存/梯度爆炸
- **意义**: 从振动光谱数据提取微观相互作用（逆问题）

### 3. ML势稳定性提升
- **方法**: Potential Averaging (PA) 势能平均
- **应用**: Allegro模型 + LAMMPS
- **效果**: 大尺度MD模拟稳定性显著提升

### 4. GPU加速大规模MD
- **平台**: LAMMPS ML-IAP-Kokkos (NVIDIA/LANL/Sandia)
- **技术**: PyTorch MLIP + Cython桥接 + 端到端GPU加速
- **支持**: cuEquivariance等变模型

---

## 🤖 材料发现AI最新突破

### 生成模型新架构

| 模型 | 类型 | 特点 | 年份 |
|------|------|------|------|
| MatterGPT | Transformer | 多性质逆向设计（形成能+带隙） | 2024 |
| AtomGPT | NLP+结构 | 超导材料原子结构生成 | 2024 |
| ChargeDIFF | 扩散模型 | 首个纳入3D电荷密度 | 2025 |
| AlloyGAN | GAN+LLM | 金属玻璃热力学预测（误差<8%） | 2025 |
| CDVAE | VAE | NJIT多价离子电池材料发现 | 2025 |

### GNN前沿进展

- **EOSnet**: 带隙MAE=0.163 eV, 金属分类97.7%
- **KA-GNN**: Kolmogorov-Arnold网络，提升可解释性
- **CTGNN**: 晶体Transformer，钙钛矿预测领先
- **Hybrid-LLM-GNN**: 25%精度提升

### 自主实验室(SDL)

- **AlabOS**: 可重构工作流管理框架
- **NanoChef**: 银纳米颗粒优化，100次实验达最优
- **Rainbow**: 多机器人金属卤化物钙钛矿合成
- **PLD自动化**: 贝叶斯优化+原位拉曼，仅采样0.25%参数空间

### 实际发现案例

1. **MIT SCIGEN + DiffCSP** (2025-09)
   - 生成1000万Archimedean晶格材料候选
   - 100万通过稳定性筛选
   - 成功合成TiPdBi和TiPbSb两种新化合物

2. **NJIT多价离子电池材料** (2025-07)
   - CDVAE + LLM双AI方法
   - 发现5种全新多孔过渡金属氧化物结构
   - 适用于镁/钙/铝/锌离子电池

---

## 🔬 4. 第一性原理+量子计算交叉进展（第二轮新增）

### 量子-经典混合神经网络(QANN)
- **论文**: arXiv:2512.13115 (2025-12)
- **突破**: 量子启发式ANN用于拓扑材料发现
- **方法**: 将成分描述符映射到量子概率幅，引入元素间配对关联
- **成果**: 高通量筛选发现5种全新拓扑化合物

### 2D量子材料自旋量子比特设计
- **论文**: arXiv:2503.31xxxx (2025-03)
- **目标**: 单层MoS₂中可调控自旋缺陷量子比特
- **发现**: 6种热力学稳定反位缺陷(MX, M=Mg,Ca,Sr,Ba,Zn,Cd)
- **应用**: 量子信息科学，室温量子比特操控

### 非磁性替代永磁材料发现
- **论文**: arXiv:2506.22627 (2025-06)
- **方法**: CGCNN + 第一性原理 + GPU超算
- **发现**: Fe-Co-Zr三元空间中9种热力学稳定新化合物 + 81种低能量亚稳相
- **优化**: Fe₅Co₁₈Zr₆→Fe₅Co₁₆Zr₆Mn₄, 各向异性K₁=1.1 MJ/m³

---

## 🧬 5. ML势最新进展（第二轮新增）

### 开源数据集与模型（2025）

| 名称 | 类型 | 特点 | 链接 |
|------|------|------|------|
| **OMat24** | 数据集 | 无机材料数据集，用于预训练 | arXiv:2410.12771 |
| **OMol25** | 数据集 | 有机分子数据集，氧化态教学 | arXiv:2505.08762 |
| **MACE-OFF** | 模型 | 短程可迁移ML力场（有机分子） | JACS 2025 |
| **MatterSim** | 模型 | 跨元素、温度、压力的深度学习 | arXiv:2405.04967 |
| **Orb-v3** | 模型 | 大规模原子模拟 | arXiv:2504.06231 |
| **UMA** | 模型族 | 通用原子模型家族 | arXiv:2506.23971 |
| **eSEN-OMol25** | 模型 | 电荷自旋守恒的有机分子势 | - |
| **Egret-1** | 模型 | 生物有机模拟预训练势 | arXiv:2504.20955 |

### 长程相互作用突破
- **论文**: Nature Communications 2025-11
- **方法**: 等变GNN + 显式极化长程物理
- **框架**: 可微分Ewald求和 + 电荷平衡
- **意义**: 离子系统、异质界面模拟精度提升

### 关键基准测试
- **MLIP Arena**: ICLR 2025，公平透明评估平台
- **CHIPS-FF**: 通用ML力场材料性质评估
- **Matbench Discovery**: 晶体稳定性预测框架

---

## 🤖 6. 生成模型与扩散模型最新进展（第二轮新增）

### 晶体结构生成模型（2025 arXiv）

| 模型 | 架构 | 创新点 |
|------|------|--------|
| **SymmCD** | 扩散模型 | 对称性保持晶体生成 | arXiv:2502.03638 |
| **WyckoffDiff** | 扩散模型 | Wyckoff位置对称性生成 | arXiv:2502.06485 |
| **Equivariant Hypergraph Diffusion** | 超图扩散 | 晶体结构预测等变超图 | arXiv:2501.18850 |
| **CrystalGRW** | 测地随机游走 | 目标性质晶体生成 | arXiv:2501.08998 |
| **CHGGen** | GNN+扩散 | host-guided修复生成 | 2025 |
| **FlowMM** | 黎曼流匹配 | 材料生成流匹配 | arXiv:2406.04713 |
| **CrysLLMGen** | LLM+扩散 | NeurIPS 2025, LLM+扩散结合 | arXiv:2025 |
| **PCCD** | 点云扩散 | 点云表示晶体扩散 | iScience 2025 |
| **CrysTens-3D** | 3D体素扩散 | 原子特征编码+3D体素 | MRS 2025 |

### 语言模型生成晶体
- **方法**: 直接生成XYZ/CIF/PDB文件
- **代表**: CrysText, LLM+晶体生成
- **突破**: 自然语言描述→晶体结构

---

## 🌐 7. GNN最新架构（第二轮新增）

### 等变模型进展
- **EquiformerV2**: 扩展至高阶表示的等变Transformer
- **GemNet-OC**: 大规模多样化分子模拟GNN
- **SO3KRATES**: SO(3)等变架构

### 注意力机制改进
- **CTGNN**: 晶体Transformer GNN (arXiv:2405.11502)
- **ACGNet**: 可解释注意力晶体图网络
- **DR-Label**: 标签解构重建提升催化GNN

### 多模态融合
- **MolPROP**: 多模态语言+图融合分子性质预测
- **Hybrid Transformer Graph**: 四体相互作用混合框架

---

## 📝 技能库维护建议（更新）

### 建议新增技能模块
1. `skills/floquet-tddft/` - 超快动力学计算
2. `skills/mlip-stability/` - ML势稳定性提升
3. `skills/generative-materials/` - 生成式材料设计
4. `skills/autonomous-lab/` - 自主实验平台接口
5. `skills/equivariant-gnn/` - 等变图神经网络
6. `skills/diffusion-crystal/` - 扩散模型晶体生成
7. `skills/long-range-mlp/` - 长程ML势
8. `skills/quantum-ann/` - 量子-经典混合AI

### 建议更新现有技能
- 更新CHGNet技能至最新版本
- 更新MACE技能至MACE-OFF
- 添加Orb/Mettalsim/UMA支持
- 更新DeepMD-kit至v3版本

---

## 🔧 8. 开源代码库与工具平台（第三轮新增）

### 主流ML势GitHub仓库汇总

| 名称 | 仓库 | 最新版本 | 特点 |
|------|------|----------|------|
| **DeepMD-kit** | github.com/deepmodeling/deepmd-kit | v3 (2025) | 多后端框架，支持DPA3 |
| **MACE** | github.com/ACEsuit/mace | 2025 | 等变消息传递，MACE-MP-0 |
| **MatterSim** | github.com/microsoft/mattersim | 2024 | 跨温度/压力深度学习 |
| **CHGNet** | github.com/CederGroupHub/chgnet | 2025 | 电荷感知，稳定性预测 |
| **NequIP** | github.com/mir-group/nequip | - | E(3)等变GNN |
| **Allegro** | github.com/mir-group/allegro | - | 严格局部等变网络 |
| **ORB** | github.com/orbital-materials/orb | v3 (2025) | 快速可扩展 |
| **GAP** | github.com/libAtoms/GAP | - | 高斯过程势 |
| **SNAP/FitSNAP** | github.com/FitSNAP/FitSNAP | - | 谱近邻分析 |
| **MTP** | github.com/ashapeev/mlip-2 | - | 矩张量势 |
| **NEP** | github.com/brucefan1983/GPUMD | - | GPU分子动力学 |
| **ACE** | github.com/ACEsuit/ACE1pack | - | 原子簇展开 |

### 晶体结构生成开源工具

| 工具 | 类型 | 链接 |
|------|------|------|
| **MAGUS 2.0** | 晶体结构预测 | gitlab.com/bigd4/magus |
| **PyXtal** | 晶体生成+对称性分析 | github.com/qzhu2017/PyXtal |
| **CDVAE** | 扩散VAE晶体生成 | github.com/txie-93/cdvae |
| **MatterGen** | 微软生成模型 | github.com/microsoft/mattergen |
| **FlowMM** | 黎曼流匹配 | github.com/bkmi/flowmm |

### MD模拟平台ML势支持更新

- **LAMMPS**: SNAP、DeePMD、PINN、OpenMSCG、FitSNAP
- **GROMACS**: TorchMD-Net、MACE等变GNN
- **OpenMM 8.0**: 原生ML势支持、Deep Potential插件
- **CP2K**: 混合QM/ML框架、Active Learning
- **AMBER**: DPMD、MACE离子液体
- **NAMD**: Colvars+RL自适应采样、DeepDriveMD
- **JAX-MD**: 端到端可微MD (NequIP/MACE原生支持)
- **ASE 3.27** (2025-12): 统一接口，支持所有主流ML势

---

## 📊 9. 不确定性量化(UQ)最新方法（第三轮新增）

### 证据深度学习(Evidential Deep Learning)
- **论文**: Nature Communications 2025-12-20
- **方法**: eIP - 证据深度学习ML势
- **突破**: 单次前向传播估计不确定性，区分偶然/认知不确定性
- **应用**: 主动学习、UDD不确定性驱动动力学

### UQ方法对比

| 方法 | 计算成本 | 准确性 | 特点 |
|------|----------|--------|------|
| **集成方法** | 高(4x) | 高 | 多模型训练，可靠 |
| **MC Dropout** | 中 | 中 | 多次推理 |
| **GMM** | 中 | 中 | EM迭代 |
| **MVE** | 低 | 低 | 均值方差估计 |
| **eIP (证据)** | 低 | 高 | 单次前向传播 |
| **LTAU** | 低 | 中 | 训练轨迹分析 |
| **LLPR** | 低 | 中 | 最后层刚性 |

---

## 🔬 10. 声子/热导率ML势基准测试（第三轮新增）

### 大规模基准研究 (arXiv:2509.03401)
- **测试**: 6种uMLPs在2,429种晶体材料上
- **模型**: EquiformerV2、MatterSim、MACE、CHGNet
- **发现**: 
  - EquiformerV2在预测原子力和三阶IFC表现最强
  - 微调的EquiformerV2在二阶IFC、LTC预测上最优
  - MatterSim力精度较低但IFC预测中等（误差抵消）

### 磁性材料ML势
- **论文**: Physical Review E 2025-11
- **发现**: 
  - 非磁性训练数据足以预测顺磁熔体动态性质
  - 自旋极化训练对铁磁相静态性质至关重要

---

## 📝 技能库维护建议（第三轮更新）

### 紧急更新项
1. **新增技能**: `skills/matgl/` - Materials Graph Library
2. **更新技能**: ASE → 3.27.0版本支持
3. **新增技能**: `skills/evidential-uq/` - 证据深度学习UQ
4. **新增技能**: `skills/magus/` - MAGUS晶体结构预测

---

## 🔄 持续搜索状态更新

- [x] 第一轮：基础搜索完成
- [x] 第二轮：arXiv最新论文搜索完成
- [x] 第三轮：代码/GitHub/工具平台搜索完成
- [ ] 第四轮：方法学详细技术深挖
- [ ] 第五轮：跨领域交叉进展

---

*研究状态：**24小时持续运行中** | 最后更新：17:26 GMT+8*
