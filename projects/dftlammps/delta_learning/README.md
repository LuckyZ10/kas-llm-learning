# Delta Learning (Δ-Learning) Module

## 概述

Δ-学习模块实现从低精度DFT到高精度方法的机器学习修正。

核心公式:
```
E_high = E_low + ΔE_ML
F_high = F_low + ΔF_ML
σ_high = σ_low + Δσ_ML
```

其中ΔE_ML, ΔF_ML, Δσ_ML由神经网络从数据中学习。

## 主要特性

- **多目标学习**: 同时预测能量、力、应力修正
- **SOAP描述符**: 平滑原子位置重叠描述符
- **ACE描述符**: 原子簇展开描述符 (高阶多体)
- **转移学习**: 从小体系学习，应用到大体系

## 快速开始

### 基本用法

```python
from dftlammps.delta_learning import DeltaLearningInterface, DeltaLearningConfig

# 创建配置
config = DeltaLearningConfig(
    descriptor_type='soap',
    energy_weight=1.0,
    force_weight=50.0,
)

# 创建模型
delta_model = DeltaLearningInterface(config)

# 准备训练数据
structures = [
    {
        'positions': np.array([[0, 0, 0], [0, 0, 0.74]]),  # H2
        'atom_types': np.array([0, 0]),
        'energy_low': -31.5,      # PBE能量 (eV)
        'energy_high': -31.8,     # CCSD(T)能量 (eV)
        'forces_low': np.array([[0, 0, 0.1], [0, 0, -0.1]]),
        'forces_high': np.array([[0, 0, 0.12], [0, 0, -0.12]]),
    },
    # ... 更多结构
]

# 训练
delta_model.fit(structures, validation_split=0.1)

# 预测
new_structure = {
    'positions': np.array([[0, 0, 0], [0, 0, 1.0]]),
    'atom_types': np.array([0, 0]),
}
correction = delta_model.predict(new_structure)
# correction = {'energy_delta': ..., 'forces_delta': ...}

# 应用修正
low_accuracy_energy = -31.0  # PBE计算
high_accuracy_energy = delta_model.correct_energy(
    new_structure, 
    low_accuracy_energy
)
```

### 转移学习

```python
from dftlammps.delta_learning import transfer_learning_delta_model

# 从预训练模型创建新模型
new_model = transfer_learning_delta_model(
    pretrained_path='water_delta_model.pt',
    new_config=config,
    freeze_layers=[0, 1]  # 冻结前两层
)

# 在新体系上训练
new_structures = [...]  # 新体系的结构
new_model.fit(new_structures)
```

### 创建流程

```python
from dftlammps.delta_learning import create_delta_learning_pipeline

# 一键创建完整流程
delta_model = create_delta_learning_pipeline(
    low_level_dft='pbe',
    high_level_method='ccsd_t',
    descriptor_type='soap'
)
```

## 描述符类型

### SOAP (推荐)

```python
config = DeltaLearningConfig(
    descriptor_type='soap',
    descriptor_dim=100,
)
```

SOAP特点:
- 旋转、平移、置换不变
- 平滑截断
- 广泛适用于分子和材料

### ACE

```python
config = DeltaLearningConfig(
    descriptor_type='ace',
    descriptor_dim=150,
)
```

ACE特点:
- 高阶多体相互作用
- 更好的外推能力
- 适合复杂体系

## 配置选项

```python
DeltaLearningConfig(
    # 模型架构
    descriptor_type='soap',           # 描述符类型
    descriptor_dim=100,               # 描述符维度
    hidden_dims=[256, 256, 128, 64], # 隐藏层
    activation='silu',                # 激活函数
    use_batch_norm=True,              # 批归一化
    dropout_rate=0.1,                 # Dropout率
    
    # 训练目标
    predict_energy=True,              # 预测能量
    predict_forces=True,              # 预测力
    predict_stress=False,             # 预测应力
    
    # 损失权重
    energy_weight=1.0,                # 能量损失权重
    force_weight=50.0,                # 力损失权重
    stress_weight=1.0,                # 应力损失权重
    smoothness_weight=0.01,           # 平滑性权重
    
    # 训练参数
    learning_rate=1e-3,               # 学习率
    weight_decay=1e-5,                # 权重衰减
    batch_size=32,                    # 批量大小
    max_epochs=1000,                  # 最大epoch数
    early_stopping_patience=100,      # 早停耐心值
    
    # 转移学习
    use_transfer_learning=False,      # 启用转移学习
    pretrained_model_path=None,       # 预训练模型路径
    freeze_layers=[],                 # 冻结的层
    fine_tune_lr_ratio=0.1,           # 微调学习率比例
)
```

## 训练数据格式

```python
structure = {
    # 必需
    'positions': np.array([...]),      # 原子位置 [n_atoms, 3]
    'atom_types': np.array([...]),     # 原子类型 [n_atoms]
    
    # 周期性体系
    'cell': np.array([...]),           # 晶胞矩阵 [3, 3]
    
    # 能量 (eV)
    'energy_low': float,               # 低精度能量
    'energy_high': float,              # 高精度能量
    
    # 力 (eV/Å)
    'forces_low': np.array([...]),     # 低精度力 [n_atoms, 3]
    'forces_high': np.array([...]),    # 高精度力 [n_atoms, 3]
    
    # 应力 (GPa或eV/Å³)
    'stress_low': np.array([...]),     # 低精度应力 [6]或[3,3]
    'stress_high': np.array([...]),    # 高精度应力
}
```

## 高级用法

### 自定义描述符

```python
from dftlammps.delta_learning import DeltaLearningModel

# 创建自定义描述符计算器
class MyDescriptor(nn.Module):
    def forward(self, positions, atom_types, cell=None):
        # 自定义描述符计算
        return descriptors

# 使用自定义描述符
config = DeltaLearningConfig(descriptor_dim=50)
model = DeltaLearningModel(config, descriptor_dim=50)
```

### 批量预测

```python
# 多个结构批量预测
structures = [struct1, struct2, struct3, ...]
predictions = []

for struct in structures:
    pred = delta_model.predict(struct)
    predictions.append(pred)
```

### 保存和加载

```python
# 保存模型
delta_model.save('my_delta_model.pt')

# 加载模型
delta_model = DeltaLearningInterface()
delta_model.load('my_delta_model.pt')
```

## 性能提示

1. **数据量**: 通常需要100-1000个结构用于训练
2. **力权重**: 力通常比能量重要50倍
3. **描述符**: SOAP适合大多数情况，ACE适合高阶相互作用
4. **转移学习**: 从相似体系预训练可以显著减少所需数据

## 故障排除

### 训练不收敛

- 降低学习率
- 增加训练数据
- 检查数据归一化
- 减少网络复杂度

### 过拟合

- 增加Dropout
- 增加权重衰减
- 使用早停
- 增加训练数据

### 预测不合理

- 检查训练数据范围是否覆盖预测范围
- 验证描述符计算正确
- 检查能量/力单位一致

## 参考文献

1. Ramakrishnan et al., "Big Data Meets Quantum Chemistry Approximations: The Δ-Machine Learning Approach", J. Chem. Theory Comput. 2015
2. Bartók et al., "On representing chemical environments", Phys. Rev. B 2013
3. Drautz, "Atomic cluster expansion for accurate and transferable interatomic potentials", Phys. Rev. B 2019
