# 材料数字孪生与实时预测模块 (Material Digital Twin & Real-time Prediction)

## 📊 模块统计 (Module Statistics)

| 模块 | 文件数 | 代码行数 | 说明 |
|------|--------|----------|------|
| `digital_twin/` | 4 | 3,269 | 数字孪生核心模块 |
| `realtime_sim/` | 2 | 1,049 | 实时模拟与降阶模型 |
| `examples/digital_twin_cases/` | 5 | 3,016 | 应用案例与文档 |
| **总计** | **11** | **~7,334** | 完整数字孪生系统 |

## 🏗️ 架构概览 (Architecture Overview)

```
┌─────────────────────────────────────────────────────────────────┐
│                    数字孪生核心 (Digital Twin Core)              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ 物理-数据融合 │  │ 状态同步器   │  │   预测性维护         │  │
│  │ Hybrid Model │  │ Synchronizer │  │ Predictive Maint.    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ 传感器融合    │      │ 预测模型      │      │ 实时模拟      │
│ Sensor Fusion│      │ Predictive   │      │ Realtime Sim │
│              │      │ Models       │      │              │
│ • Kalman     │      │              │      │ • POD        │
│ • Particle   │      │ • RUL预测    │      │ • DMD        │
│ • Anomaly    │      │ • 退化模型   │      │ • Autoencoder│
└──────────────┘      │ • 故障预警   │      │ • 边缘部署   │
                      └──────────────┘      └──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      应用案例 (Applications)                     │
├─────────────────────────────────────────────────────────────────┤
│  🔋 Battery Health    🏗️ Structural Lifetime    ⚗️ Catalyst    │
│     Digital Twin          Prediction           Deactivation     │
│                                                              │
│  • SOC/SOH监测          • 疲劳裂纹预测        • 活性监测      │
│  • 容量退化预测          • 寿命评估            • 失活机理识别  │
│  • 充电优化              • 检测计划优化        • 再生策略      │
│  • 安全预警              • 风险评估            • 操作优化      │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 核心功能 (Core Features)

### 1. 物理-数据融合 (Physics-Data Fusion)
- **PhysicsInformedLayer**: 物理约束神经网络层
- **HybridDigitalTwin**: 融合物理模型和数据驱动模型
- **不确定性量化**: 贝叶斯神经网络风格的不确定性估计

### 2. 传感器融合 (Sensor Fusion)
- **多传感器集成**: 支持温度、应变、化学、光学等传感器
- **滤波算法**: 卡尔曼滤波、粒子滤波、自适应噪声滤波
- **异常检测**: 统计方法和自编码器异常检测

### 3. 预测模型 (Predictive Models)
- **RUL预测**: LSTM和注意力机制剩余寿命预测
- **退化模型**: 线性、指数、幂律、Weibull等
- **故障预警**: 多级预警系统和风险评估

### 4. 实时模拟 (Real-time Simulation)
- **降阶模型**: POD、DMD、自编码器、DeepONet
- **在线学习**: 增量更新和自适应学习
- **边缘部署**: 模型量化和TorchScript转换

## 📦 模块详情 (Module Details)

### digital_twin/twin_core.py (993行)
- `DigitalTwinSystem`: 主数字孪生系统类
- `HybridDigitalTwin`: 混合神经网络模型
- `StateSynchronizer`: 状态同步器
- `PredictiveMaintenance`: 预测性维护

### digital_twin/sensor_fusion.py (1080行)
- `MultiSensorFusion`: 多传感器融合
- `SensorNetwork`: 传感器网络管理
- `AnomalyDetector`: 异常检测器
- `KalmanFilter/ParticleFilter`: 滤波器

### digital_twin/predictive_model.py (1078行)
- `PredictiveMaintenanceSuite`: 预测维护套件
- `RULPredictor`: 剩余寿命预测器
- `DegradationCurve`: 退化曲线模型
- `FailureWarningSystem`: 故障预警系统

### realtime_sim/rom_simulator.py (993行)
- `RealtimeSimulator`: 实时模拟器
- `ReducedOrderModel`: 降阶模型
- `ProperOrthogonalDecomposition`: POD
- `DynamicModeDecomposition`: DMD
- `OnlineLearner`: 在线学习
- `EdgeDeployment`: 边缘部署

## 🔬 应用案例 (Application Cases)

### 电池健康数字孪生 (652行)
```python
battery_twin = BatteryDigitalTwin(battery_id="EV_001")
battery_twin.initialize()
result = battery_twin.update_from_sensors(voltage=3.7, current=10.0)
prediction = battery_twin.predict_life()
```

### 结构材料寿命预测 (902行)
```python
structural_twin = StructuralDigitalTwin(component_id="Bridge_A1")
structural_twin.update_from_measurements(strains=[100, 105, ...])
life_pred = structural_twin.predict_lifetime()
schedule = structural_twin.optimize_inspection_schedule()
```

### 催化剂失活监测 (1032行)
```python
catalyst_twin = CatalystDigitalTwin(reactor_id="R101")
catalyst_twin.update(temperature=773, conversion=85.0)
mechanisms = catalyst_twin.identify_mechanisms()
regen_plan = catalyst_twin.plan_regeneration()
```

## 📖 文档 (Documentation)

详细文档请参见 `examples/digital_twin_cases/README.md`

## 🧪 验证 (Verification)

运行验证脚本：
```bash
python verify_digital_twin.py
```

## 📝 依赖 (Dependencies)

- numpy
- scipy
- torch
- scikit-learn

## 📄 许可证 (License)

MIT License
