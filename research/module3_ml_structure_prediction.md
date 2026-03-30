# 材料数据库与自动化工具研究报告
## 模块3: 机器学习结构预测 (M3GNet / CHGNet / GNoME)

**研究时间**: 2026-03-08 16:40+

---

## 1. 通用机器学习原子间势 (Universal MLIPs) 概述

通用机器学习原子间势是材料科学的重大突破，能够在元素周期表范围内进行快速、准确的结构预测。

### 1.1 主要模型对比
| 模型 | 机构 | 架构 | 独特特性 | 年份 |
|------|------|------|---------|------|
| **M3GNet** | UC Berkeley/MVL | GNN + 三体相互作用 | 含应力预测 | 2022 |
| **CHGNet** | UC Berkeley/Ceder组 | GNN + 磁矩 | 电荷信息建模 | 2023 |
| **GNoME** | Google DeepMind | GNN | 主动学习大规模发现 | 2023 |
| **MACE** | 多机构 | 等变消息传递 | 高精度力预测 | 2023 |
| **SevenNet** | 韩国 | NequIP架构 | 高效并行 | 2024 |

---

## 2. M3GNet (Materials 3-body Graph Network)

### 2.1 架构创新
- **三体相互作用**: 显式包含键角信息，超越传统二体势
- **位置包含图**: 输入包含原子坐标和3×3晶格矩阵
- **自动微分**: 通过反向传播计算力(能量梯度)和应力

### 2.2 核心组件
```
位置包含图 → 图特征化 → 多体计算模块 → 图卷积 → 读出层
                ↓
         球贝塞尔函数 + 球谐函数展开
```

### 2.3 训练数据
- **数据集**: MPF.2021.2.8 (Materials Project Trajectory)
- **规模**: 150万+ 无机结构
- **元素覆盖**: 89种元素

### 2.4 使用示例
```python
from m3gnet.models import M3GNet, Potential
from m3gnet.trainers import PotentialTrainer
import tensorflow as tf

# 加载预训练模型
m3gnet = M3GNet.load("M3GNet-MP-2021.2.8-PES")
potential = Potential(model=m3gnet)

# 结构弛豫
from m3gnet.models import Relaxer
relaxer = Relaxer(potential=potential)
relaxed_structure = relaxer.relax(structure)
```

### 2.5 性能指标
| 性质 | MAE |
|------|-----|
| 能量 | ~30 meV/atom |
| 力 | ~77 meV/Å |
| 应力 | ~0.35 GPa |

---

## 3. CHGNet (Crystal Hamiltonian Graph Neural Network)

### 3.1 核心创新
CHGNet是首个显式包含**磁矩(magnetic moments)**的通用势函数，能够学习电子轨道占据信息。

### 3.2 电荷信息建模
```
晶体结构(未知电荷) → CHGNet → 能量 + 力 + 应力 + 磁矩
                                    ↓
                              电子轨道占据信息
```

### 3.3 架构特点
- **原子图 + 键图**: 同时表示成对键和键角信息
- **4层原子卷积**: 捕获最长20Å的长程相互作用
- **400,438可训练参数**

### 3.4 性能对比
| 模型版本 | 能量MAE | 力MAE | 应力MAE | 磁矩MAE |
|---------|---------|-------|---------|---------|
| CHGNet v0.2.0 | 33 meV/atom | 79 meV/Å | 0.351 GPa | - |
| CHGNet v0.3.0 | 30 meV/atom | 77 meV/Å | 0.348 GPa | 0.032 μB |

### 3.5 应用案例
1. **Li-Mn-O体系电荷信息MD**: 模拟α-LMO到ε-LMO相变
2. **LiFePO₄有限温度相图**: 预测温度-组分相稳定性
3. **石榴石导体Li扩散**: 离子电导率与激活能预测

### 3.6 使用示例
```python
from chgnet.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics

# 加载预训练模型
chgnet = CHGNet.load()

# 分子动力学模拟
md = MolecularDynamics(
    atoms=structure,
    model=chgnet,
    ensemble="nvt",
    temperature=1000,
    timestep=2.0,  # fs
)
md.run(steps=100000)
```

---

## 4. GNoME (Graph Networks for Materials Exploration)

### 4.1 项目规模 (Google DeepMind)
- **发现材料**: 220万新晶体结构
- **稳定材料**: 38万种预测稳定
- **等效人工年**: ~800年研究工作量
- **实验验证**: 736种已合成

### 4.2 核心架构
```
┌─────────────────────────────────────────────────────────────┐
│                    GNoME 主动学习循环                         │
├─────────────────────────────────────────────────────────────┤
│  1. 结构生成 → 2. GNN预测稳定性 → 3. DFT验证 → 4. 数据回传    │
│      ↑                                              ↓       │
│      └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 双管道生成策略
| 管道 | 方法 | 特点 |
|------|------|------|
| **结构管道** | 基于已知晶体结构生成候选 | 结构相似性 |
| **组分管道** | 基于化学式的随机方法 | 探索性更强 |

### 4.4 性能提升
```
预测准确率: 50% → 80%+
能量误差: ~11 meV/atom
```

### 4.5 重要发现
- **52,000种**类石墨烯层状化合物 (之前仅~1,000种)
- **528种**锂离子导体 (25倍于已知)
- **15种**锂过渡金属氧化物 (超导/电池应用)

### 4.6 A-Lab自主合成验证
- **机构**: Lawrence Berkeley National Lab
- **成功率**: 41/58 = 71%
- **运行时间**: 17天连续运行
- **方法**: AI驱动机器人合成系统

---

## 5. 模型对比与选择指南

### 5.1 适用场景
| 应用场景 | 推荐模型 | 原因 |
|---------|---------|------|
| 快速结构弛豫 | M3GNet | 速度快, 稳定性好 |
| 含磁体系模拟 | CHGNet | 唯一含磁矩势函数 |
| 电池材料研究 | CHGNet | 电荷信息对离子体系关键 |
| 大规模材料发现 | GNoME | 主动学习, 高通量生成 |
| 高精度力预测 | MACE | 等变架构, 力精度高 |

### 5.2 Matbench Discovery基准
CHGNet在该基准测试中取得了SOTA性能，在分布外材料稳定性预测中表现优异。

### 5.3 微调策略
所有模型都支持迁移学习:
```python
# CHGNet微调示例
from chgnet.model import CHGNet
from chgnet.trainer import Trainer

model = CHGNet.load()
trainer = Trainer(model=model, ...)
trainer.train(train_loader, val_loader)
```

---

## 6. 集成建议

### 6.1 材料发现工作流
```
GNoME生成候选结构
        ↓
CHGNet/M3GNet快速弛豫筛选
        ↓
DFT验证稳定结构
        ↓
性质预测模型(模块4)
        ↓
实验合成(A-Lab)
```

### 6.2 与DFT的协同
- **ML势**: 快速探索配置空间
- **DFT**: 验证和精确计算
- **主动学习**: 迭代改进模型

---

**模块3研究完成**
