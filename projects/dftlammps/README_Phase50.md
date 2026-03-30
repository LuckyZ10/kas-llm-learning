# DFT-LAMMPS 数字孪生与实时预测系统 (Phase 50)

## 系统概述

本系统构建了一套完整的材料系统数字孪生平台，实现了物理模型与数据驱动模型的融合、实时同步、预测性维护、不确定性量化和可视化监控。

## 目录结构

```
dftlammps/
├── digital_twin/           # 核心数字孪生模块
│   ├── twin_core.py        # 数字孪生核心引擎 (~950行)
│   ├── real_time_sync.py   # 实时同步机制 (~850行)
│   ├── predictive_model.py # 预测性维护与寿命预测 (~850行)
│   └── uncertainty_quantification.py  # 不确定性量化 (~870行)
│
├── twin_visualization/     # 可视化模块
│   ├── dashboard.py        # 实时监控仪表板 (~690行)
│   ├── renderer_3d.py      # 三维结构演化渲染 (~720行)
│   └── anomaly_alert.py    # 异常检测与预警 (~750行)
│
└── twin_examples/          # 应用案例
    ├── battery_twin.py     # 电池数字孪生 (~530行)
    └── catalyst_twin.py    # 催化剂数字孪生 (~690行)
```

## 总代码量: ~6,900行 Python

## 核心功能

### 1. 数字孪生核心引擎 (twin_core.py)

- **DigitalTwinCore**: 核心引擎类，融合物理模型和数据驱动模型
- **PhysicsBasedModel**: 物理模型基类，支持MD/DFT/FEM等多种物理模拟
- **NeuralSurrogateModel**: 神经网络代理模型，快速近似物理模拟
- **TwinCluster**: 多孪生集群管理

主要特性：
- 加权模型融合 (物理权重 + 数据权重)
- 自适应权重调整
- 状态历史管理
- 事件回调机制
- 保存/加载功能

### 2. 实时同步机制 (real_time_sync.py)

- **RealTimeSynchronizer**: 实时同步器
- **DataTransformer**: 数据格式转换
- **QualityFilter**: 数据质量过滤
- **BidirectionalMapper**: 实验-模拟双向映射

主要特性：
- 支持EXP→SIM、SIM→EXP、双向同步
- 实时、批量、事件驱动、定时同步模式
- 自适应速率调整
- 拥塞控制
- 数据质量评估

### 3. 预测性维护 (predictive_model.py)

- **PredictiveMaintenanceEngine**: 预测维护引擎
- **DegradationModel**: 退化模型基类
  - ExponentialDegradationModel: 指数退化模型
  - PowerLawDegradationModel: 幂律退化模型
  - ParisLawModel: 疲劳裂纹扩展模型
- **ParticleFilterRUL**: 粒子滤波RUL预测

主要特性：
- 健康指数计算
- RUL (剩余使用寿命) 预测
- 故障预测
- 维护建议生成
- 多方法融合预测

### 4. 不确定性量化 (uncertainty_quantification.py)

- **UQEngine**: 不确定性量化引擎
- **MonteCarloUQ**: Monte Carlo方法
- **BootstrapUQ**: Bootstrap重采样
- **EnsembleUQ**: 集成模型UQ
- **BayesianUQ**: 贝叶斯推断
- **SensitivityAnalyzer**: 敏感性分析
- **ConfidenceEstimator**: 置信度估计

主要特性：
- 多种UQ方法
- Sobol敏感性分析
- Morris筛选
- 置信度校准
- 不确定性传播追踪

### 5. 可视化模块

#### 实时监控仪表板 (dashboard.py)
- **Dashboard**: 实时仪表板
- **MatplotlibRenderer**: 静态图表渲染
- **PlotlyRenderer**: 交互式图表
- **WebDashboard**: Web界面

#### 三维渲染 (renderer_3d.py)
- **Structure3D**: 3D结构表示
- **MatplotlibRenderer3D**: 3D渲染器
- **VolumeRenderer**: 体积渲染
- **EvolutionAnimator**: 演化动画

#### 异常检测 (anomaly_alert.py)
- **AnomalyDetectionSystem**: 异常检测系统
- **StatisticalDetector**: 统计检测器
- **IsolationForestDetector**: 隔离森林
- **AutoencoderDetector**: 自编码器
- **AlertManager**: 预警管理

### 6. 应用案例

#### 电池数字孪生 (battery_twin.py)
- **BatteryDigitalTwin**: 电池数字孪生系统
- **BatteryPhysicsModel**: 电池物理模型 (等效电路 + 老化模型)
- 容量衰减预测
- SOH/SOC估算
- 充放电循环模拟

#### 催化剂数字孪生 (catalyst_twin.py)
- **CatalystDigitalTwin**: 催化剂数字孪生系统
- **CatalystPhysicsModel**: 催化剂物理模型
- 活性位点演化追踪
- 失活机理预测 (烧结、积碳、中毒)
- 再生策略推荐

## 快速开始

### 运行完整演示

```python
import sys
sys.path.insert(0, 'dftlammps')

# 运行单个模块演示
from digital_twin.twin_core import demo as core_demo
core_demo()

from digital_twin.predictive_model import demo as pred_demo
pred_demo()

from twin_examples.battery_twin import demo as battery_demo
battery_demo()

from twin_examples.catalyst_twin import demo as catalyst_demo
catalyst_demo()
```

### 创建自定义数字孪生

```python
from dftlammps import (
    DigitalTwinCore, TwinConfiguration, StateVector,
    PhysicsBasedModel, ModelType
)

# 创建配置
config = TwinConfiguration(
    name="my_twin",
    physics_weight=0.6,
    data_weight=0.4
)

# 创建数字孪生
twin = DigitalTwinCore(config)

# 注册物理模型
physics_model = PhysicsBasedModel(
    model_type=ModelType.MOLECULAR_DYNAMICS,
    params={'mass': 1.0}
)
twin.register_physics_model(physics_model)

# 初始化
initial_state = StateVector(
    timestamp=0.0,
    data=np.array([...])
)
twin.initialize(initial_state)

# 运行模拟
for i in range(100):
    state = twin.step(dt=0.001)
```

## 依赖要求

- Python >= 3.8
- NumPy >= 1.20
- Matplotlib >= 3.3 (可视化)
- Flask >= 2.0 (Web界面，可选)
- Plotly >= 5.0 (交互式图表，可选)
- Pillow >= 8.0 (图像处理，可选)

## 架构特点

1. **模块化设计**: 核心、同步、预测、UQ、可视化分离
2. **类型注解**: 完整的类型提示支持
3. **可扩展性**: 易于添加新的物理模型和数据驱动模型
4. **容错性**: 异常处理和降级机制
5. **实时性**: 支持实时数据流处理

## 许可证

MIT License

## 作者

DFT-LAMMPS Development Team
