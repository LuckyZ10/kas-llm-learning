# DFTLammps 自适应多尺度耦合模块

## 概述

本模块实现了自动多尺度耦合与智能分辨率选择功能，能够在分子动力学模拟中自动判断何时切换DFT和ML势函数分辨率。

## 模块结构

```
dftlammps/adaptive_multiscale/
├── __init__.py              # 模块导出
├── resolution_adapter.py    # 分辨率适配器 (~650行)
├── error_estimators.py      # 误差估计器 (~700行)
├── coupling_controller.py   # 耦合控制器 (~680行)
├── application_examples.py  # 应用案例 (~520行)

dftlammps/smart_sampling/
├── __init__.py              # 智能采样模块 (~900行)
```

## 核心功能

### 1. 分辨率适配器 (resolution_adapter.py)

#### ResolutionLevel
- `DFT_HIGH`: 高精度DFT (HSE06等)
- `DFT_STANDARD`: 标准DFT (PBE等)
- `ML_DENSE`: 稠密神经网络势
- `ML_STANDARD`: 标准ML势
- `ML_FAST`: 快速ML势
- `MM_CLASSICAL`: 经典力场

#### AdaptiveResolutionManager
主要管理器，提供：
- 自动DFT↔ML势切换
- 误差驱动的分辨率调整
- 计算成本-精度权衡

```python
from dftlammps.adaptive_multiscale import create_default_manager

manager = create_default_manager(target_accuracy=0.001, budget_ms=1000.0)

result = manager.step(
    positions=positions,
    forces_ml=forces_ml,
    uncertainty=uncertainty_estimate,
    system_context={'geometry_changed': True}
)
```

### 2. 误差估计器 (error_estimators.py)

#### EnsembleErrorEstimator
基于深度集成模型的不确定性估计：
- 集成成员分歧分析
- 数据噪声(aleatoric)与模型不确定性(epistemic)分离
- 距离训练集修正

#### GradientSensitivityEstimator
基于输入扰动的敏感度分析：
- 有限差分法评估预测稳定性
- 位置扰动下的能量/力变化

#### AdaptiveSamplingTrigger
智能采样触发器：
- 基于不确定性阈值的触发
- 探索vs利用的平衡
- DFT调用率控制

```python
from dftlammps.adaptive_multiscale import EnsembleErrorEstimator, AdaptiveSamplingTrigger

estimator = EnsembleErrorEstimator(ensemble_models)
trigger = AdaptiveSamplingTrigger(estimator)

should_trigger, info = trigger.should_trigger_dft(positions, atomic_numbers)
```

### 3. 耦合控制器 (coupling_controller.py)

#### AdaptiveBoundaryOptimizer
自适应QM/MM边界优化：
- 动态QM区域调整
- 缓冲区层定义
- 连接原子(link atoms)处理

#### InformationCoordinator
信息传递协调：
- 力混合 (Force mixing)
- 能量插值
- 平滑过渡

#### LoadBalancer
负载均衡：
- 处理器分区优化
- 并行效率分析
- 动态重平衡建议

```python
from dftlammps.adaptive_multiscale import CouplingController

controller = CouplingController()
qm_region = controller.initialize(positions, atomic_numbers)

result = controller.step(
    positions, atomic_numbers,
    forces_qm, forces_mm, forces_ml,
    error_indicators
)
mixed_forces = result['mixed_forces']
```

### 4. 智能采样 (smart_sampling)

#### AdaptiveImportanceSampler
自适应重要性采样：
- 不确定性加权
- 能量加权 (类Boltzmann)
- 新颖性加权
- 系统/残差/多项式采样方法

#### RareEventDetector
稀有事件检测：
- 键断裂/形成检测
- 过渡态识别
- 相变检测
- 扩散跳跃追踪

#### ParallelEfficiencyOptimizer
并行效率优化：
- 任务分配优化
- 动态批大小调整
- 负载均衡分析

```python
from dftlammps.smart_sampling import SmartSamplingManager

sampler = SmartSamplingManager()

# 采样配置
indices = sampler.sample_configurations(
    positions_batch, energies, uncertainties, n_samples=10
)

# 检测稀有事件
events = sampler.process_trajectory_frame(
    timestep, positions, atomic_numbers, energy, forces
)
```

## 应用案例

### 1. 自适应裂纹扩展模拟

```python
from dftlammps.adaptive_multiscale.application_examples import (
    AdaptiveCrackSimulation, CrackSimulationConfig
)

config = CrackSimulationConfig(
    system_size=(100, 100, 10),
    initial_crack_length=20.0
)

simulation = AdaptiveCrackSimulation(config)
simulation.initialize(positions, atomic_numbers)

for step in range(100):
    result = simulation.step(positions, atomic_numbers, stresses, ...)
    # QM区域自动跟随裂纹尖端
    # 高应力区域使用DFT，体相使用ML
```

### 2. 智能催化反应路径

```python
from dftlammps.adaptive_multiscale.application_examples import (
    ReactionPathExplorer, CatalyticReactionConfig
)

config = CatalyticReactionConfig(
    catalyst_type="Pt",
    surface_plane="111",
    adsorbate="CO"
)

explorer = ReactionPathExplorer(config)

for step in range(100):
    result = explorer.step(positions, atomic_numbers, energy, forces)
    # 自动识别过渡态和中间体
    # 关键点位使用高精度DFT
```

### 3. 自动缺陷演化跟踪

```python
from dftlammps.adaptive_multiscale.application_examples import (
    DefectTracker, DefectTrackingConfig
)

config = DefectTrackingConfig(
    material_type="Si",
    temperature_range=(300, 800)
)

tracker = DefectTracker(config)

for step in range(100):
    result = tracker.step(positions, atomic_numbers, reference_positions, ...)
    # 自动检测空位、间隙原子、位错
    # 缺陷附近使用高分辨率
```

## 配置参数

### AdaptiveResolutionManager 配置

```python
config = {
    'target_accuracy': 0.001,      # 目标精度 (eV/atom)
    'budget_ms': 1000.0,           # 每步计算预算 (ms)
    'hysteresis': 0.15,            # 切换滞后阈值
    'min_steps': 5                 # 最小停留步数
}
```

### EnsembleErrorEstimator 配置

```python
estimator = EnsembleErrorEstimator(
    ensemble_models=models,
    temperature=300.0,
    energy_weight=0.3,
    force_weight=0.7
)
```

### AdaptiveBoundaryOptimizer 配置

```python
optimizer = AdaptiveBoundaryOptimizer(
    min_qm_radius=4.0,            # 最小QM半径 (Å)
    max_qm_radius=12.0,           # 最大QM半径 (Å)
    buffer_thickness=2.0,         # 缓冲区厚度 (Å)
    error_threshold=0.1           # 误差阈值
)
```

## 性能指标

### 计算效率指标

- `wall_time`: 实际耗时
- `cpu_time`: CPU时间
- `memory_usage_mb`: 内存使用
- `energy_evaluations`: 能量评估次数
- `parallel_efficiency`: 并行效率

### 精度指标

- `energy_rmse`: 能量均方根误差
- `force_rmse`: 力均方根误差
- `confidence_score`: 置信度评分
- `uncertainty_quantile`: 不确定性分位数

### 边界质量指标

- `boundary_energy`: 边界能量
- `force_continuity`: 力连续性
- `charge_neutrality_error`: 电荷中性误差

## 算法细节

### 分辨率选择算法

1. **Pareto最优分析**: 识别成本-精度Pareto前沿
2. **加权评分**: 自适应权重平衡精度与成本
3. **滞后机制**: 防止分辨率快速振荡

### 不确定性量化

1. **集成方差**: 深度集成成员间的分歧
2. **梯度敏感度**: 输入扰动下的输出变化
3. **距离修正**: 到训练集的距离加权

### QM/MM边界优化

1. **种子扩展**: 从高误差原子开始区域生长
2. **缓冲区层**: 过渡区域的平滑处理
3. **连接原子**: 切断键的适当处理

## 测试

每个模块包含自测试代码：

```bash
# 测试分辨率适配器
python dftlammps/adaptive_multiscale/resolution_adapter.py

# 测试误差估计器
python dftlammps/adaptive_multiscale/error_estimators.py

# 测试耦合控制器
python dftlammps/adaptive_multiscale/coupling_controller.py

# 测试智能采样
python dftlammps/smart_sampling/__init__.py

# 运行应用案例
python dftlammps/adaptive_multiscale/application_examples.py
```

## 依赖项

- numpy >= 1.20.0
- scipy >= 1.7.0
- Python >= 3.8

## 版本历史

- **1.0.0** (2026-03-09)
  - 初始版本
  - 实现核心自适应多尺度功能
  - 包含三个完整应用案例

## 作者

DFTLammps Team

## 许可证

MIT License
