# DFT-LAMMPS 前沿方法模块

## 概述

本模块集成了2024-2025年材料计算和AI领域的最新前沿方法,包括:

- **生成式AI材料设计**: CDVAE、DiffCSP扩散模型,流匹配模型
- **图神经网络新材料**: MACE等变GNN, ALIGNN, DimeNet++
- **物理信息神经网络**: PINNs求解PDE,神经算子(FNO/DeepONet)
- **自动化实验**: 自动驾驶实验室,机器人合成规划,闭环发现系统
- **文献追踪**: arXiv监控,论文分析,趋势检测,自动集成

## 模块结构

```
dftlammps/
├── frontier/                 # 前沿方法模块
│   ├── diffusion_materials.py     # 扩散模型生成晶体
│   ├── flow_matching.py           # 流匹配模型
│   ├── llm_materials_design.py    # LLM材料设计
│   ├── mace_integration.py        # MACE等变GNN
│   ├── alignn_wrapper.py          # ALIGNN原子线图
│   ├── dimenet_plus_plus.py       # DimeNet++方向消息传递
│   ├── pinns_for_pde.py           # PINNs求解PDE
│   ├── neural_operators.py        # 神经算子
│   ├── physics_informed_ml.py     # 物理约束ML势
│   ├── self_driving_lab.py        # 自动驾驶实验室
│   ├── robotic_synthesis.py       # 机器人合成规划
│   └── closed_loop_discovery.py   # 闭环发现系统
│
├── literature_scanner/       # 文献追踪模块
│   ├── arxiv_monitor.py           # arXiv监控
│   ├── paper_analyzer.py          # 论文分析
│   ├── trend_detector.py          # 趋势检测
│   └── auto_importer.py           # 自动集成
│
└── frontier_examples/        # 前沿案例
    ├── diffusion_battery_cathode.py  # 扩散模型设计电池正极
    ├── mace_active_learning.py       # MACE主动学习
    ├── self_driving_perovskite.py    # 自动驾驶钙钛矿发现
    └── pinn_phase_field.py           # PINN求解相场方程
```

## 快速开始

### 1. 扩散模型生成晶体

```python
from dftlammps.frontier import CrystalGenerator

# 创建生成器
generator = CrystalGenerator(model_type='cdvae', device='cuda')

# 生成结构
structures = generator.generate(
    num_structures=10,
    num_atoms_range=(5, 50)
)

# 优化特定属性
optimized = generator.optimize_for_property(
    'band_gap', target_value=1.5
)
```

### 2. MACE主动学习

```python
from dftlammps.frontier import MACE, MACEActiveLearner

# 创建模型
model = MACE(num_elements=20, hidden_channels=128)

# 主动学习
learner = MACEActiveLearner(model)
selected = learner.select_samples(candidate_pool, n_select=5)
```

### 3. PINNs求解相场方程

```python
from dftlammps.frontier import PhaseFieldPINN, SirenNetwork

# 创建网络
network = SirenNetwork(in_features=3, hidden_features=128)

# 创建PINN
pinn = PhaseFieldPINN(
    network=network,
    equation_type="cahn_hilliard"
)

# 训练
train_pinn(pinn, domain_bounds=[(0, 1), (0, 1)])
```

### 4. 自动驾驶实验室

```python
from dftlammps.frontier import SelfDrivingLab

# 创建实验室
lab = SelfDrivingLab()

# 设置目标
lab.set_target(
    target_formula="LiFePO4",
    target_properties={'band_gap': (2.0, 0.3)}
)

# 运行发现
results = lab.run_discovery(max_experiments=50)
```

## 技术细节

### 扩散模型

**CDVAE** (Crystal Diffusion Variational Autoencoder)
- 使用VAE学习晶体结构的潜在表示
- 在潜在空间中进行条件扩散生成
- 支持属性引导生成

**DiffCSP** (Diffusion for Crystal Structure Prediction)
- 组分条件下的结构生成
- 处理周期性边界条件
- 分数坐标+晶格参数联合生成

**流匹配模型**
- 比扩散模型更快的生成速度
- 更直的生成轨迹
- 黎曼流形上的流匹配

### 图神经网络

**MACE**
- 高阶等变消息传递
- 10-100x样本效率
- 精确的力预测

**ALIGNN**
- 线图捕获角度信息
-  Materials Project SOTA性能
- 力场版本支持

**DimeNet++**
- 方向消息传递
- 球谐函数特征
- 3-body相互作用建模

### 物理信息神经网络

**PINNs**
- 物理方程作为损失约束
- 自动微分计算导数
- 相场、扩散、弹性方程

**神经算子**
- FNO: 傅里叶空间卷积
- DeepONet: 学习非线性算子
- 分辨率不变性

### 自动化实验

**闭环发现**
- 计算筛选→合成→表征→反馈
- 贝叶斯优化
- 自适应学习

## 参考文献

### 扩散模型
- Xie et al. (2022) "Crystal Diffusion Variational Autoencoder"
- Jiao et al. (2023) "Crystal Structure Prediction by Jointly Modeling Spatial and Periodic Invariances"
- Lipman et al. (2023) "Flow Matching for Generative Modeling"

### 图神经网络
- Batatia et al. (2022) "MACE: Higher Order Equivariant Message Passing Neural Networks"
- Choudhary et al. (2021) "Atomistic Line Graph Neural Network"
- Gasteiger et al. (2021) "Fast and Uncertainty-Aware Directional Message Passing"

### 物理信息神经网络
- Raissi et al. (2019) "Physics-informed neural networks"
- Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations"

### 自动化实验
- Szymanski et al. (2021) "Toward autonomous design and synthesis of novel inorganic materials"
- MacLeod et al. (2020) "Self-driving laboratory for accelerated discovery"

## 许可

MIT License
