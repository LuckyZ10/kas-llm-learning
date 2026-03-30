"""
Battery Health Digital Twin
电池健康数字孪生应用案例

实现电池系统的数字孪生，包括：
- 实时电池状态监测
- 容量退化预测
- 安全预警
- 寿命优化建议
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import time
import warnings

# 导入数字孪生模块
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from dftlammps.digital_twin import (
    DigitalTwinSystem,
    MaterialState,
    PhysicsParameters,
    SensorNetwork,
    SensorReading,
    SensorType,
    PredictiveMaintenanceSuite,
    DegradationModel,
    FailureMode
)
from dftlammps.digital_twin.sensor_fusion import (
    KalmanFilter,
    CalibrationParams
)


@dataclass
class BatteryState:
    """电池状态数据类"""
    timestamp: float
    
    # 电气参数
    voltage: float = 3.7          # V
    current: float = 0.0          # A
    power: float = 0.0            # W
    
    # 热参数
    temperature: float = 25.0     # °C
    temperature_gradient: float = 0.0  # °C/min
    
    # 化学参数
    soc: float = 100.0            # State of Charge (%)
    soh: float = 100.0            # State of Health (%)
    internal_resistance: float = 0.1  # Ohm
    
    # 老化参数
    cycle_count: int = 0
    cumulative_throughput: float = 0.0  # Ah
    
    # 退化指标
    capacity_fade: float = 0.0    # %
    resistance_growth: float = 0.0  # %
    
    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.voltage,
            self.current,
            self.power,
            self.temperature,
            self.soc,
            self.soh,
            self.internal_resistance,
            self.cycle_count / 1000,  # 归一化
            self.cumulative_throughput / 10000,
            self.capacity_fade,
            self.resistance_growth
        ])


class BatteryDigitalTwin:
    """电池数字孪生系统"""
    
    def __init__(
        self,
        battery_id: str,
        nominal_capacity: float = 2.5,  # Ah
        nominal_voltage: float = 3.7,   # V
        chemistry: str = "Li-ion"
    ):
        self.battery_id = battery_id
        self.nominal_capacity = nominal_capacity
        self.nominal_voltage = nominal_voltage
        self.chemistry = chemistry
        
        # 核心数字孪生系统
        self.twin = DigitalTwinSystem(
            material_id=battery_id,
            state_dim=32,
            physics_dim=10,
            observation_dim=16
        )
        
        # 预测维护套件
        self.predictor = PredictiveMaintenanceSuite(
            input_dim=11,
            sequence_length=30
        )
        
        # 传感器网络
        self.sensor_network = SensorNetwork()
        self._setup_sensors()
        
        # 状态历史
        self.state_history: deque = deque(maxlen=10000)
        self.battery_states: deque = deque(maxlen=10000)
        
        # 当前状态
        self.current_state: Optional[BatteryState] = None
        
        # 校准参数
        self.voltage_calibration = 1.0
        self.current_calibration = 1.0
        
        # 统计
        self.charge_cycles = 0
        self.discharge_cycles = 0
        self.total_energy_throughput = 0.0  # kWh
    
    def _setup_sensors(self) -> None:
        """设置传感器网络"""
        # 电压传感器
        self.sensor_network.add_sensor(
            "voltage_main",
            SensorType.ELECTROCHEMICAL,
            measurement_dim=1,
            filter_type="kalman",
            calibration=CalibrationParams(
                offset=np.array([0.0]),
                scale=np.array([self.voltage_calibration])
            )
        )
        
        # 电流传感器
        self.sensor_network.add_sensor(
            "current_main",
            SensorType.ELECTROCHEMICAL,
            measurement_dim=1,
            filter_type="kalman",
            calibration=CalibrationParams(
                offset=np.array([0.0]),
                scale=np.array([self.current_calibration])
            )
        )
        
        # 温度传感器（多点）
        for i in range(3):
            self.sensor_network.add_sensor(
                f"temperature_{i}",
                SensorType.TEMPERATURE,
                measurement_dim=1,
                filter_type="adaptive",
                metadata={'location': f'cell_{i}'}
            )
        
        # 压力传感器（安全监测）
        self.sensor_network.add_sensor(
            "pressure_internal",
            SensorType.PRESSURE,
            measurement_dim=1,
            filter_type="kalman"
        )
    
    def initialize(self, initial_state: Optional[BatteryState] = None) -> None:
        """初始化电池孪生"""
        if initial_state is None:
            initial_state = BatteryState(
                timestamp=time.time(),
                voltage=self.nominal_voltage,
                soc=100.0,
                soh=100.0
            )
        
        self.current_state = initial_state
        
        # 初始化数字孪生
        material_state = self._battery_to_material_state(initial_state)
        self.twin.initialize(material_state)
        
        # 初始化预测器
        self.predictor.update_state(
            features=initial_state.to_vector(),
            health_indicator=initial_state.soh / 100.0,
            timestamp=initial_state.timestamp
        )
        
        print(f"Battery Digital Twin initialized for {self.battery_id}")
    
    def _battery_to_material_state(self, battery: BatteryState) -> MaterialState:
        """转换电池状态到材料状态"""
        state = MaterialState()
        state.material_id = self.battery_id
        state.timestamp = battery.timestamp
        
        # 将电池参数映射到材料参数
        state.conductivity = 1.0 / (battery.internal_resistance + 1e-8)
        state.degradation_index = battery.capacity_fade / 100.0
        
        # 物理参数
        state.physics = PhysicsParameters(
            temperature=battery.temperature + 273.15,
            pressure=101325.0
        )
        
        return state
    
    def update_from_sensors(
        self,
        voltage: Optional[float] = None,
        current: Optional[float] = None,
        temperatures: Optional[List[float]] = None,
        pressure: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """从传感器数据更新"""
        if timestamp is None:
            timestamp = time.time()
        
        # 处理传感器数据
        if voltage is not None:
            reading = SensorReading(
                sensor_id="voltage_main",
                sensor_type=SensorType.ELECTROCHEMICAL,
                timestamp=timestamp,
                value=np.array([voltage]),
                unit="V"
            )
            self.sensor_network.ingest_data(reading)
        
        if current is not None:
            reading = SensorReading(
                sensor_id="current_main",
                sensor_type=SensorType.ELECTROCHEMICAL,
                timestamp=timestamp,
                value=np.array([current]),
                unit="A"
            )
            self.sensor_network.ingest_data(reading)
        
        if temperatures:
            for i, temp in enumerate(temperatures[:3]):
                reading = SensorReading(
                    sensor_id=f"temperature_{i}",
                    sensor_type=SensorType.TEMPERATURE,
                    timestamp=timestamp,
                    value=np.array([temp]),
                    unit="Celsius"
                )
                self.sensor_network.ingest_data(reading)
        
        # 获取融合结果
        fused = self.sensor_network.get_fused_state()
        
        # 更新电池状态
        new_state = self._fused_to_battery_state(fused, timestamp)
        self.current_state = new_state
        self.battery_states.append(new_state)
        
        # 更新数字孪生
        material_state = self._battery_to_material_state(new_state)
        twin_result = self.twin.update_from_observation(
            observation=new_state.to_vector()[:12],
            physics=material_state.physics
        )
        
        # 更新预测器
        predictor_result = self.predictor.update_state(
            features=new_state.to_vector(),
            health_indicator=new_state.soh / 100.0,
            timestamp=timestamp
        )
        
        # 更新统计
        self._update_statistics(new_state)
        
        return {
            'battery_state': new_state,
            'fused_sensor_data': fused,
            'twin_update': twin_result,
            'predictor_update': predictor_result,
            'warnings': predictor_result.get('warnings', [])
        }
    
    def _fused_to_battery_state(
        self,
        fused: Dict,
        timestamp: float
    ) -> BatteryState:
        """从融合数据转换到电池状态"""
        # 简化的转换逻辑
        state = BatteryState(timestamp=timestamp)
        
        if fused.get('status') == 'fused':
            fused_val = fused.get('fused_value', np.zeros(5))
            if len(fused_val) > 0:
                state.voltage = fused_val[0]
            if len(fused_val) > 1:
                state.temperature = fused_val[1]
        
        # 从历史继承其他参数
        if self.current_state:
            state.soc = self._estimate_soc(state.voltage)
            state.soh = self.current_state.soh
            state.internal_resistance = self.current_state.internal_resistance
            state.cycle_count = self.current_state.cycle_count
        
        return state
    
    def _estimate_soc(self, voltage: float) -> float:
        """从电压估计SOC（简化模型）"""
        # 简化的开路电压-SOC关系
        v_min = 2.5  # 完全放电
        v_max = 4.2  # 完全充电
        soc = (voltage - v_min) / (v_max - v_min) * 100
        return max(0, min(100, soc))
    
    def _update_statistics(self, state: BatteryState) -> None:
        """更新统计数据"""
        # 能量吞吐量
        power = state.voltage * state.current
        self.total_energy_throughput += abs(power) / 1000 / 3600  # kWh
        
        # 循环计数（简化）
        if self.current_state:
            if state.current > 0.1 and self.current_state.current <= 0.1:
                self.charge_cycles += 1
            elif state.current < -0.1 and self.current_state.current >= -0.1:
                self.discharge_cycles += 1
    
    def predict_life(
        self,
        prediction_horizon: int = 1000,
        usage_profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """预测电池寿命"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        # 使用预测器预测RUL
        rul_result = self.predictor.rul_predictor.predict_rul_hybrid(
            current_health=self.current_state.soh / 100.0,
            current_time=time.time()
        )
        
        # 风险评估
        risk = self.predictor.warning_system.get_risk_assessment(rul_result)
        
        # 预测未来健康度
        future_states = self.predictor.predict_future_states(
            num_steps=min(prediction_horizon, 100),
            step_size=1.0
        )
        
        # 根据使用场景调整
        cycle_life_factor = 1.0
        if usage_profile == 'aggressive':
            cycle_life_factor = 0.7
        elif usage_profile == 'gentle':
            cycle_life_factor = 1.3
        
        adjusted_rul = rul_result.rul_mean * cycle_life_factor
        
        return {
            'current_soh': self.current_state.soh,
            'predicted_rul_cycles': adjusted_rul,
            'predicted_rul_years': adjusted_rul / 365,
            'risk_level': risk['risk_level'],
            'risk_score': risk['risk_score'],
            'recommended_action': risk['recommended_action'],
            'future_trajectory': future_states,
            'confidence': rul_result.confidence
        }
    
    def optimize_charging(
        self,
        target_soc: float = 100.0,
        time_constraint: Optional[float] = None,
        temperature_constraint: float = 45.0
    ) -> Dict[str, Any]:
        """优化充电策略"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        current_soc = self.current_state.soc
        
        if current_soc >= target_soc:
            return {'status': 'already_charged', 'current_soc': current_soc}
        
        # 基于当前健康度优化充电策略
        soh_factor = self.current_state.soh / 100.0
        
        # 老化电池需要更温和的充电
        max_current = 1.0 * soh_factor  # C-rate
        
        if time_constraint:
            # 需要快速充电
            required_rate = (target_soc - current_soc) / 100 * self.nominal_capacity / (time_constraint / 3600)
            max_current = min(max_current, required_rate / self.nominal_capacity)
        
        # 温度限制
        temp_margin = temperature_constraint - self.current_state.temperature
        if temp_margin < 10:
            max_current *= 0.5
        
        # 分阶段充电策略
        soc_gap = target_soc - current_soc
        
        strategy = []
        
        # CC阶段
        cc_current = min(max_current, 1.0)
        cc_soc_limit = min(80, target_soc)
        if current_soc < cc_soc_limit:
            strategy.append({
                'phase': 'CC',
                'current': cc_current,
                'target_soc': cc_soc_limit,
                'estimated_time': (cc_soc_limit - current_soc) / 100 * self.nominal_capacity / cc_current * 3600
            })
        
        # CV阶段
        if target_soc > 80:
            strategy.append({
                'phase': 'CV',
                'voltage': 4.2,
                'target_soc': target_soc,
                'estimated_time': 1800  # 简化估计
            })
        
        return {
            'current_soc': current_soc,
            'target_soc': target_soc,
            'strategy': strategy,
            'estimated_total_time': sum(s.get('estimated_time', 0) for s in strategy),
            'health_preservation_tips': [
                '避免长时间处于高SOC状态',
                '充电时保持温度在适宜范围',
                '定期进行完整充放电循环以校准SOC'
            ]
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        # 计算退化率
        if len(self.battery_states) > 10:
            recent_soh = [s.soh for s in list(self.battery_states)[-10:]]
            degradation_rate = (recent_soh[0] - recent_soh[-1]) / len(recent_soh)
        else:
            degradation_rate = 0
        
        return {
            'battery_id': self.battery_id,
            'current_state': {
                'soc': self.current_state.soc,
                'soh': self.current_state.soh,
                'voltage': self.current_state.voltage,
                'temperature': self.current_state.temperature,
                'internal_resistance': self.current_state.internal_resistance
            },
            'degradation': {
                'capacity_fade': 100 - self.current_state.soh,
                'resistance_growth': self.current_state.resistance_growth,
                'degradation_rate_per_cycle': degradation_rate
            },
            'usage_statistics': {
                'charge_cycles': self.charge_cycles,
                'discharge_cycles': self.discharge_cycles,
                'total_energy_throughput_kwh': self.total_energy_throughput
            },
            'sensor_status': self.sensor_network.get_network_status(),
            'twin_status': self.twin.get_system_status()
        }
    
    def detect_anomalies(self) -> List[Dict]:
        """检测异常"""
        if self.current_state is None:
            return []
        
        anomalies = []
        
        # 电压异常
        if self.current_state.voltage > 4.3 or self.current_state.voltage < 2.0:
            anomalies.append({
                'type': 'voltage_anomaly',
                'severity': 'critical',
                'value': self.current_state.voltage,
                'message': f'电压异常: {self.current_state.voltage:.2f}V'
            })
        
        # 温度异常
        if self.current_state.temperature > 60:
            anomalies.append({
                'type': 'thermal_anomaly',
                'severity': 'critical',
                'value': self.current_state.temperature,
                'message': f'温度过高: {self.current_state.temperature:.1f}°C'
            })
        elif self.current_state.temperature > 45:
            anomalies.append({
                'type': 'thermal_warning',
                'severity': 'warning',
                'value': self.current_state.temperature,
                'message': f'温度偏高: {self.current_state.temperature:.1f}°C'
            })
        
        # 内阻异常（增长过快）
        if self.current_state.resistance_growth > 50:
            anomalies.append({
                'type': 'resistance_anomaly',
                'severity': 'warning',
                'value': self.current_state.resistance_growth,
                'message': f'内阻增长异常: {self.current_state.resistance_growth:.1f}%'
            })
        
        return anomalies


def simulate_battery_degradation(
    initial_capacity: float = 2.5,
    cycles: int = 1000,
    c_rate: float = 1.0,
    temperature: float = 25.0
) -> List[BatteryState]:
    """模拟电池退化过程"""
    states = []
    
    # 退化模型参数
    cycle_degradation = 0.02  # % per cycle
    temperature_factor = 1.0 + 0.02 * max(0, temperature - 25)
    c_rate_factor = 1.0 + 0.1 * max(0, c_rate - 1)
    
    soh = 100.0
    
    for cycle in range(cycles):
        # 更新SOH
        degradation = cycle_degradation * temperature_factor * c_rate_factor
        soh -= degradation
        soh = max(0, soh)
        
        # 创建状态
        state = BatteryState(
            timestamp=time.time() + cycle * 3600,
            voltage=3.7 + np.random.randn() * 0.1,
            current=0.0,
            temperature=temperature + np.random.randn() * 2,
            soc=np.random.uniform(20, 80),
            soh=soh,
            internal_resistance=0.1 * (100 / soh),
            cycle_count=cycle,
            capacity_fade=100 - soh
        )
        
        states.append(state)
    
    return states


if __name__ == "__main__":
    print("=" * 70)
    print("Battery Health Digital Twin - Application Demo")
    print("=" * 70)
    
    # 创建电池数字孪生
    print("\n1. Creating Battery Digital Twin")
    battery_twin = BatteryDigitalTwin(
        battery_id="EV_Battery_001",
        nominal_capacity=50.0,  # 50 Ah电动车电池
        nominal_voltage=3.7,
        chemistry="NMC"
    )
    
    # 初始化
    battery_twin.initialize()
    print(f"  Initialized battery: {battery_twin.battery_id}")
    print(f"  Nominal capacity: {battery_twin.nominal_capacity} Ah")
    
    # 模拟运行
    print("\n2. Simulating Battery Operation")
    
    # 生成模拟数据
    simulated_states = simulate_battery_degradation(
        initial_capacity=50.0,
        cycles=100,
        c_rate=1.5,
        temperature=35.0
    )
    
    for i, state in enumerate(simulated_states[:20]):
        result = battery_twin.update_from_sensors(
            voltage=state.voltage,
            current=state.current,
            temperatures=[state.temperature],
            timestamp=state.timestamp
        )
        
        if i % 5 == 0:
            print(f"  Cycle {i}: SOH = {state.soh:.1f}%, Temp = {state.temperature:.1f}°C")
    
    # 健康报告
    print("\n3. Health Report")
    report = battery_twin.get_health_report()
    print(f"  Current SOH: {report['current_state']['soh']:.1f}%")
    print(f"  Capacity fade: {report['degradation']['capacity_fade']:.1f}%")
    print(f"  Total cycles: {report['usage_statistics']['charge_cycles']}")
    
    # 寿命预测
    print("\n4. Life Prediction")
    prediction = battery_twin.predict_life(usage_profile='normal')
    print(f"  Predicted RUL: {prediction['predicted_rul_cycles']:.0f} cycles")
    print(f"  Risk level: {prediction['risk_level']}")
    print(f"  Recommendation: {prediction['recommended_action']}")
    
    # 充电优化
    print("\n5. Charging Optimization")
    charge_plan = battery_twin.optimize_charging(
        target_soc=90.0,
        time_constraint=3600
    )
    print(f"  Current SOC: {charge_plan['current_soc']:.1f}%")
    print(f"  Strategy phases: {len(charge_plan['strategy'])}")
    print(f"  Estimated time: {charge_plan['estimated_total_time']/60:.1f} min")
    
    # 异常检测
    print("\n6. Anomaly Detection")
    anomalies = battery_twin.detect_anomalies()
    if anomalies:
        for anomaly in anomalies:
            print(f"  {anomaly['severity'].upper()}: {anomaly['message']}")
    else:
        print("  No anomalies detected")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
