# DFT-LAMMPS RL Optimization Module

强化学习在材料发现和工艺优化中的应用

## 概述

本模块实现了强化学习在材料科学中的应用，包括：

- **GFlowNet分子生成器**: 基于流网络的分子和晶体生成
- **离线强化学习**: CQL、IQL、Decision Transformer用于材料设计
- **工艺参数优化**: RL与贝叶斯优化的比较与应用
- **奖励函数设计工具**: 多目标奖励、偏好学习、逆强化学习
- **高通量筛选集成**: 将RL生成器集成到筛选工作流

## 安装

```bash
# 基础依赖
pip install torch numpy scipy

# 可选依赖 (用于分子操作)
pip install rdkit-pypi

# 可选依赖 (用于贝叶斯优化)
pip install scikit-learn
```

## 快速开始

### 1. GFlowNet分子生成

```python
from dftlammps.rl_optimization import (
    MoleculeGFlowNet, GFlowNetConfig,
    MolecularGraphEnv, MoleculeEnvConfig,
    GFlowNetTrainer, GFlowNetTrainingConfig
)

# 创建环境
env_config = MoleculeEnvConfig(
    max_atoms=20,
    atom_types=['C', 'N', 'O', 'H']
)
env = MolecularGraphEnv(env_config)

# 创建GFlowNet
gfn_config = GFlowNetConfig(
    state_dim=200,
    action_dim=env.action_dim,
    hidden_dim=256
)
gfn = MoleculeGFlowNet(
    atom_types=['C', 'N', 'O', 'H'],
    bond_types=['SINGLE', 'DOUBLE'],
    config=gfn_config
)

# 训练
trainer = GFlowNetTrainer(gfn, env)
history = trainer.train()

# 生成样本
samples = trainer.generate_samples(num_samples=100)
```

### 2. 离线强化学习

```python
from dftlammps.rl_optimization import CQL, IQL, OfflineRLConfig
from dftlammps.rl_optimization.training import OfflineRLTrainer, OfflineDataset

# 准备离线数据集
trajectories = [...]  # 从现有数据加载
dataset = OfflineDataset(trajectories)

# 创建CQL代理
config = OfflineRLConfig(
    state_dim=128,
    action_dim=64,
    cql_alpha=1.0
)
agent = CQL(config)

# 训练
trainer = OfflineRLTrainer(agent, dataset)
history = trainer.train()
```

### 3. 工艺参数优化

```python
from dftlammps.rl_optimization import SynthesisEnv, ProcessEnvConfig

# 创建环境
config = ProcessEnvConfig(
    param_bounds={
        'temperature': (300, 1200),
        'pressure': (0.1, 10),
    }
)
env = SynthesisEnv(
    target_property='bandgap',
    target_value=1.5,
    config=config
)

# 训练优化代理
from dftlammps.rl_optimization.training import ProcessOptimizationTrainer
trainer = ProcessOptimizationTrainer(agent, env)
history = trainer.train()
```

## 模块结构

```
rl_optimization/
├── models/
│   ├── gflownet.py          # GFlowNet实现
│   ├── policy.py            # 策略网络
│   └── offline_rl.py        # 离线RL算法
├── environments/
│   ├── molecule_env.py      # 分子生成环境
│   ├── material_env.py      # 材料设计环境
│   └── process_env.py       # 工艺优化环境
├── rewards/
│   └── reward_design.py     # 奖励函数设计
├── training/
│   ├── gflownet_trainer.py  # GFlowNet训练
│   ├── offline_trainer.py   # 离线RL训练
│   └── process_trainer.py   # 工艺优化训练
├── integration/
│   ├── screening_rl.py      # 筛选集成
│   └── multi_objective.py   # 多目标优化
└── examples/
    ├── example_gflownet_molecule.py
    ├── example_offline_rl.py
    ├── example_process_optimization.py
    └── example_reward_design.py
```

## 示例

### 运行示例

```bash
# GFlowNet分子生成
python -m dftlammps.rl_optimization.examples.example_gflownet_molecule

# 离线强化学习
python -m dftlammps.rl_optimization.examples.example_offline_rl

# 工艺参数优化
python -m dftlammps.rl_optimization.examples.example_process_optimization

# 奖励函数设计
python -m dftlammps.rl_optimization.examples.example_reward_design
```

## 算法说明

### GFlowNet

GFlowNet通过训练流网络来学习生成对象，使得生成概率与奖励成正比。

**关键特性:**
- 轨迹平衡 (Trajectory Balance) 损失
- 流匹配 (Flow Matching) 损失
- 支持分子图和晶体结构生成

### 离线强化学习

#### CQL (Conservative Q-Learning)

防止离线RL中的值函数过估计，通过添加保守性项来约束Q值。

#### IQL (Implicit Q-Learning)

使用期望回归避免值函数过估计，不需要显式的策略提取。

#### Decision Transformer

将RL视为序列建模问题，使用Transformer预测动作。

### 工艺参数优化

#### 贝叶斯优化 vs RL

- **贝叶斯优化**: 适用于昂贵的黑箱函数，样本效率更高
- **RL**: 适用于序列决策问题，可以学习长期策略

## 奖励函数设计

### 内置奖励组件

- `PropertyReward`: 基于目标性质的奖励
- `ValidityReward`: 化学有效性奖励
- `DiversityReward`: 样本多样性奖励
- `NoveltyReward`: 新颖性奖励

### 多目标优化

```python
from dftlammps.rl_optimization import MultiObjectiveReward

multi_reward = MultiObjectiveReward([
    (property_reward, 0.5),
    (validity_reward, 0.3),
    (diversity_reward, 0.2),
], method='weighted_sum')
```

### 偏好学习

```python
from dftlammps.rl_optimization import PreferenceLearning

pref_learning = PreferenceLearning(state_dim=128)

# 添加偏好比较
pref_learning.add_preference(state1, state2, preference=0)

# 训练
pref_learning.train_step()
```

## 性能优化

### GPU加速

```python
config = GFlowNetConfig(device='cuda')
gfn = GFlowNet(config)
```

### 多保真度优化

```python
from dftlammps.rl_optimization.training import MultiFidelityOptimizer

optimizer = MultiFidelityOptimizer(
    low_fidelity_fn=fast_approximation,
    high_fidelity_fn=expensive_simulation,
    cost_ratio=0.1
)
```

## 集成到高通量筛选

```python
from dftlammps.rl_optimization import ScreeningRLIntegration

integration = ScreeningRLIntegration(
    generator=gflownet,
    scorer=predictive_model,
    config=ScreeningRLConfig(batch_size=100)
)

results = integration.generate_and_screen(env)
```

## 引用

如果使用了本模块，请引用以下论文:

- GFlowNet: Bengio et al. "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation" (2021)
- CQL: Kumar et al. "Conservative Q-Learning for Offline Reinforcement Learning" (2020)
- IQL: Kostrikov et al. "Offline Reinforcement Learning with Implicit Q-Learning" (2021)
- Decision Transformer: Chen et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)

## 许可证

MIT License
