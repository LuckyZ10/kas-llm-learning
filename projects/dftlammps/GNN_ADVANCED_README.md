# GNN材料表示学习与预训练模型模块

## 模块概述

本模块实现了材料发现中的高级图神经网络（GNN）技术，包括预训练模型、等变网络、层次化表示学习以及基础大模型架构。

## 目录结构

```
dftlammps/
├── gnn_advanced/
│   ├── __init__.py              # 模块导出
│   ├── pretrained_models.py     # 预训练GNN模型 (882行)
│   ├── equivariant_gnn.py       # E(3)/SE(3)等变网络 (752行)
│   └── hierarchical_gnn.py      # 层次化GNN (761行)
│
└── foundation_models/
    ├── __init__.py              # 模块导出
    ├── material_foundation.py   # 材料基础大模型 (949行)
    └── applications.py          # 应用案例 (888行)
```

**总计: 4,325行代码**

---

## 1. 高级GNN模块 (gnn_advanced/)

### 1.1 预训练模型 (pretrained_models.py)

实现了大规模材料图的预训练框架。

**核心功能:**
- **大规模材料图预训练**: 支持100+种元素的原子嵌入
- **自监督学习任务**:
  - 掩码原子预测 (Masked Atom Prediction)
  - 对比学习 (Contrastive Learning)
  - 能量/力联合预测
- **迁移学习适配**: PropertyPredictionHead支持多任务微调
- **少样本学习**: FewShotLearner实现Prototypical Network和MAML

**主要类:**
```python
- PretrainConfig          # 预训练配置
- AtomEmbedding          # 原子嵌入层
- DistanceExpansion      # 距离展开层
- PretrainedGNN          # 预训练GNN主模型
- ContrastiveLearner     # 对比学习模块
- PretrainingTrainer     # 预训练管理器
- FewShotLearner         # 少样本学习器
```

### 1.2 等变GNN (equivariant_gnn.py)

实现了E(3)和SE(3)等变网络，保证物理对称性。

**核心功能:**
- **E(3)等变性**: 旋转、平移、反射不变性
- **SE(3)等变性**: 特殊欧几里得群等变
- **张量场网络**: TensorFieldNetwork实现
- **球谐函数**: 方向信息编码
- **SE(3) Transformer**: 基于注意力的等变消息传递

**主要类:**
```python
- EquivariantConfig       # 等变网络配置
- SphericalHarmonics     # 球谐函数层
- TensorProduct          # 等变张量积
- EquivariantConv        # 等变卷积层
- SE3TransformerLayer    # SE(3) Transformer
- EquivariantGNN         # 等变GNN主模型
- TensorFieldNetwork     # 张量场网络
```

**对称性保证:**
```python
# 测试等变性
rotation = random_rotation()
errors = model.get_equivariance_error(data, rotation)
print(f"能量不变性误差: {errors['energy_invariance_error']}")
print(f"力等变性误差: {errors['force_equivariance_error']}")
```

### 1.3 层次GNN (hierarchical_gnn.py)

实现了原子-化学键-晶胞多尺度表示学习。

**核心功能:**
- **原子级别**: AtomLevel捕获局部化学环境
- **化学键级别**: BondLevel建模键网络和拓扑
- **晶胞级别**: CellLevel学习全局晶体表示
- **跨层注意力**: CrossLevelAttention融合多尺度特征

**层次架构:**
```
原子 (Atom Level)
    ↓
化学键 (Bond Level) ← 基于距离和共价半径推断
    ↓
晶胞 (Cell Level) ← Transformer处理
    ↓
跨层融合 (Cross-Level Attention)
    ↓
多任务输出
```

**主要类:**
```python
- HierarchicalConfig     # 层次网络配置
- AtomLevel             # 原子级别层
- AtomConv              # 原子卷积
- BondLevel             # 化学键级别层
- BondConv              # 化学键卷积
- CellLevel             # 晶胞级别层
- HierarchicalGNN       # 层次GNN主模型
- CrossLevelAttention   # 跨层注意力
```

---

## 2. 基础大模型模块 (foundation_models/)

### 2.1 材料基础模型 (material_foundation.py)

实现了材料科学领域的大规模预训练Transformer架构。

**核心功能:**
- **大规模Transformer架构**: 支持多层多头注意力
- **多任务预训练**: 同时预测能量、力、带隙、模量等
- **位置编码**: 空间位置编码（基于频率）
- **图注意力**: GraphMultiHeadAttention处理图结构
- **对比学习**: 自监督表示学习

**主要类:**
```python
- FoundationModelConfig      # 基础模型配置
- SinusoidalPositionalEncoding   # 正弦位置编码
- SpatialPositionalEncoding      # 空间位置编码
- GraphMultiHeadAttention        # 图多头注意力
- TransformerEncoderLayer        # Transformer编码层
- MaterialFoundationModel        # 材料基础大模型
- MultiTaskTrainer              # 多任务训练器
- PrototypicalNetwork           # 原型网络
- ZeroShotPredictor             # 零样本预测器
- PretrainingTasks              # 预训练任务集合
```

**预训练任务:**
1. 掩码原子预测 (Masked Atom Prediction)
2. 对比学习 (Contrastive Learning)
3. 去噪任务 (Denoising)
4. 多属性联合预测

### 2.2 应用案例 (applications.py)

实现了GNN在材料发现中的实际应用场景。

**核心应用:**

#### 零样本材料性质预测 (ZeroShotMaterialDiscovery)
- 基于嵌入相似度的性质推断
- 无需训练的快速预测
- 多属性同时预测
- 不确定性量化

```python
predictor = ZeroShotMaterialDiscovery(model)
predictor.build_property_database(materials, properties)
pred, conf, neighbors = predictor.predict_zero_shot(
    new_material, 'bandgap', k=5
)
```

#### 跨域迁移学习 (CrossDomainTransfer)
- 源域到目标域的模型迁移
- 域差异分析 (MMD, 分布距离)
- 多种适应策略:
  - Fine-tune: 全模型微调
  - Feature Extraction: 特征提取
  - Adapter: 轻量级适配器

```python
transfer = CrossDomainTransfer(source_model)
gap_metrics = transfer.analyze_domain_gap(source_data, target_data)
adapted_model = transfer.transfer_with_adaptation(
    target_data, target_labels, adaptation_strategy='adapter'
)
```

#### 大规模筛选加速 (LargeScaleScreening)
- 并行批量筛选
- 主动学习策略
- 不确定性引导采样
- 自动化报告生成

```python
screener = LargeScaleScreening(model)
results = screener.parallel_screening(
    candidates, target_property='energy',
    target_range=(-5, -2)
)
labeled = screener.active_learning_screening(
    candidates, oracle_function
)
```

---

## 3. 使用示例

### 3.1 快速开始

```python
from dftlammps.gnn_advanced import PretrainedGNN, PretrainConfig
from torch_geometric.data import Data
import torch

# 配置
config = PretrainConfig(hidden_dim=256, num_layers=6)

# 创建模型
model = PretrainedGNN(config)

# 准备数据
data = Data(
    atomic_numbers=torch.randint(1, 100, (20,)),
    pos=torch.randn(20, 3) * 10,
    batch=torch.zeros(20, dtype=torch.long)
)

# 前向传播
output = model(data)
print(f"能量: {output['energy'].item():.4f} eV")
print(f"力形状: {output['forces'].shape}")
```

### 3.2 等变网络

```python
from dftlammps.gnn_advanced import EquivariantGNN, EquivariantConfig

config = EquivariantConfig(
    hidden_dim=128,
    num_layers=4,
    num_scalars=64,
    num_vectors=16,
    lmax=2
)

model = EquivariantGNN(config)

# 测试等变性
rotation = random_rotation()
errors = model.get_equivariance_error(data, rotation)
```

### 3.3 层次GNN

```python
from dftlammps.gnn_advanced import HierarchicalGNN, HierarchicalConfig

config = HierarchicalConfig(
    atom_hidden_dim=128,
    bond_hidden_dim=64,
    cell_hidden_dim=256,
    atom_num_layers=4,
    bond_num_layers=2
)

model = HierarchicalGNN(config)
output = model(data, return_hierarchy=True)

# 访问层次特征
hierarchy = output['hierarchy']
print(hierarchy['atom_features'].shape)
print(hierarchy['bond_features'].shape)
print(hierarchy['cell_features'].shape)
```

### 3.4 基础大模型

```python
from dftlammps.foundation_models import (
    MaterialFoundationModel,
    FoundationModelConfig
)

config = FoundationModelConfig(
    hidden_dim=512,
    num_layers=12,
    num_heads=16
)

model = MaterialFoundationModel(config)
output = model(data, return_features=True)

# 多任务预测
print(f"能量: {output['energy'].item()}")
print(f"带隙: {output['bandgap'].item()}")
print(f"体积模量: {output['bulk_modulus'].item()}")
```

### 3.5 零样本预测

```python
from dftlammps.foundation_models import ZeroShotMaterialDiscovery

predictor = ZeroShotMaterialDiscovery(model)
predictor.build_property_database(known_materials, properties)

# 预测新材料
pred, conf, neighbors = predictor.predict_zero_shot(
    new_material, 'bandgap', k=5
)
print(f"预测带隙: {pred:.3f} eV (置信度: {conf:.3f})")
```

---

## 4. 技术特点

### 4.1 物理一致性
- **等变性保证**: 所有模型满足E(3)对称性
- **能量守恒**: 力通过能量梯度计算
- **化学合理性**: 基于共价半径和电负性

### 4.2 计算效率
- **层次化设计**: 多尺度处理减少计算量
- **稀疏注意力**: 只关注局部邻域
- **批量处理**: 支持大规模并行筛选

### 4.3 可扩展性
- **模块化架构**: 易于扩展新的网络层
- **配置驱动**: 通过配置文件控制模型行为
- **多任务支持**: 灵活的输出头设计

---

## 5. 依赖要求

```
torch >= 1.10.0
torch-geometric >= 2.0.0
torch-scatter >= 2.0.0
numpy >= 1.20.0
```

---

## 6. 参考文献

1. Schütt et al. "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions"
2. Klicpera et al. "Directional Message Passing for Molecular Graphs"
3. Batzner et al. "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials"
4. Thomas et al. "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds"
5. Qiao et al. "Uni-Mol: A Universal 3D Molecular Representation Learning Framework"

---

## 7. 作者

图神经网络专家模块
创建日期: 2026-03-09
代码总行数: 4,325行
