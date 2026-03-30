# 材料数据库与自动化工具研究报告
## 模块4: 性质预测模型 (CGCNN / MegNet / OrbNet)

**研究时间**: 2026-03-08 16:50+

---

## 1. 图神经网络用于材料性质预测概述

图神经网络(GNNs)已成为材料性质预测的变革性技术，通过将晶体结构表示为图(原子为节点、键为边)，直接学习结构-性质关系。

### 1.1 主要模型时间线
| 年份 | 模型 | 机构 | 关键创新 |
|------|------|------|---------|
| 2018 | **CGCNN** | MIT | 首个晶体图卷积神经网络 |
| 2019 | **MEGNet** | UCSD/MVL | 全局状态属性，多任务学习 |
| 2020 | **OrbNet** | Caltech | 原子轨道特征，量子化学精度 |
| 2021 | **ALIGNN** | NIST | 原子线消息传递，键角信息 |
| 2023 | **MatGNet** | 多机构 | Mat2vec嵌入，多头注意力 |

---

## 2. CGCNN (Crystal Graph Convolutional Neural Network)

### 2.1 核心架构
CGCNN是首个将晶体结构直接表示为图进行深度学习的框架。

```
晶体结构(CIF)
     ↓
晶体图构建(原子→节点, 键→边)
     ↓
图卷积层(R卷积层堆叠)
     ↓
池化层(Set2Set)
     ↓
全连接层
     ↓
性质预测
```

### 2.2 图构建策略
- **节点特征**: 原子序数one-hot编码
- **边特征**: 扩展高斯核的键距
- **多图边**: 允许多重边表示不同键类型

### 2.3 多任务扩展 (MT-CGCNN)
```python
# 同时预测多个相关性质
shared_representation = CGCNN(structure)
formation_energy = FC_Ef(shared_representation)
band_gap = FC_Eg(shared_representation)
fermi_energy = FC_Ef(shared_representation)

# 加权损失函数
total_loss = w1*L_Ef + w2*L_Eg + w3*L_Ef
```

### 2.4 性能指标
| 性质 | MAE (CGCNN) | MAE (MT-CGCNN) |
|------|-------------|----------------|
| 形成能 | 0.039 eV/atom | 0.036 eV/atom |
| 带隙 | 0.388 eV | 0.350 eV |

---

## 3. MEGNet (MatErials Graph Network)

### 3.1 架构创新
MEGNet引入了**全局状态属性**，允许预测温度/压力依赖的性质。

```
┌─────────────────────────────────────────┐
│           MEGNet 架构                   │
├─────────────────────────────────────────┤
│  节点(原子) ←──→ 边(键) ←──→ 全局状态   │
│      ↑              ↑            ↑      │
│   原子序数       键距信息    温度/压力/熵 │
└─────────────────────────────────────────┘
```

### 3.2 关键特性
- **Set2Set池化**: 处理可变大小图
- **全局状态输入**: 支持状态依赖性质预测
- **多体交互**: 通过消息传递捕获复杂交互

### 3.3 性能对比
| 数据集 | 性质 | MAE |
|--------|------|-----|
| QM9 (分子) | 多种性质 | SOTA (2019) |
| Materials Project | 形成能 | 0.028 eV/atom |
| Materials Project | 带隙 | 0.33 eV |

### 3.4 元素嵌入可视化
MEGNet训练后可提取元素嵌入向量，揭示化学周期性规律：
```python
from megnet.models import MEGNetModel

model = MEGNetModel.from_file('mefnet_model.hdf5')
element_embeddings = model.get_element_embeddings()
# 可用于迁移学习
```

---

## 4. OrbNet

### 4.1 核心创新
OrbNet是第一个使用**原子轨道(AO)特征**的深度学习模型，直接从薛定谔方程学习。

### 4.2 轨道特征化
```
传统GNN: 原子 → 节点, 键 → 边
OrbNet: 电子轨道 → 节点, 轨道交互 → 边
```

### 4.3 对称适应原子轨道(SAAO)特征
| 特征矩阵 | 维度 | 含义 |
|---------|------|------|
| Fock矩阵(F) | N×N | 单电子哈密顿量 |
| 密度矩阵(P) | N×N | 电子占据 |
| 重叠矩阵(S) | N×N | 轨道重叠 |
| 哈密顿矩阵(H) | N×N | 总哈密顿量 |

### 4.4 加速效果
- **计算速度**: 比DFT快**1000倍**
- **可处理分子**: 比训练数据大**10倍**的分子
- **实时交互**: 支持分子实时操作和可视化

### 4.5 OrbNet-Spin扩展
处理开壳层系统(自由基、激发态):
```
输入: α-自旋和β-自旋的分离Fock/密度矩阵
输出: 单重态/三重态能量预测

性能: MAE < 1 kcal/mol (化学精度)
```

---

## 5. 模型对比与选择

### 5.1 性能基准 (Materials Project)
| 模型 | 形成能MAE | 带隙MAE | 特点 |
|------|-----------|---------|------|
| CGCNN | 0.039 | 0.388 | 经典基线 |
| MEGNet | 0.028 | 0.33 | 全局状态 |
| SchNet | 0.035 | - | 连续滤波 |
| ALIGNN | 0.022 | 0.218 | 键角信息 |
| GeoCGNN | 0.024 | - | 几何增强 |
| MF-CGNN | 0.022 | 0.215 | 最小特征工程 |

### 5.2 适用场景
| 应用场景 | 推荐模型 | 原因 |
|---------|---------|------|
| 快速筛选 | MEGNet | 预训练模型可用 |
| 带隙预测 | ALIGNN | 最低MAE |
| 分子系统 | OrbNet | 量子化学精度 |
| 状态依赖性质 | MEGNet | 全局状态输入 |
| 可解释性需求 | CGCNN | 局部化学环境贡献 |

---

## 6. 集成建议

### 6.1 材料筛选工作流
```
候选结构生成
      ↓
CGCNN/MEGNet快速预测形成能
      ↓
ALIGNN精细预测带隙/弹性
      ↓
DFT验证关键候选
      ↓
实验合成
```

### 6.2 多保真度策略
```python
# 三级筛选
level1 = megnet.predict(structures)  # 快速初筛
level2 = alignn.predict(candidates)   # 精细预测
level3 = dft.calculate(finalists)     # 精确验证
```

---

**模块4研究完成**
