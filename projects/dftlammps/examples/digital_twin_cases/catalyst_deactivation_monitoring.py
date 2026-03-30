"""
Catalyst Deactivation Monitoring
催化剂失活监测应用案例

实现催化剂系统的数字孪生，包括：
- 催化剂活性实时监测
- 失活机理识别
- 再生策略优化
- 反应器性能预测
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time
import warnings

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
    FailureMode,
    create_catalyst_twin
)
from dftlammps.realtime_sim import (
    ReducedOrderModel,
    ROMMethod,
    RealtimeSimulator
)


class CatalystType(Enum):
    """催化剂类型"""
    ZEOLITE = "zeolite"
    METAL_OXIDE = "metal_oxide"
    PRECIOUS_METAL = "precious_metal"
    ENZYME = "enzyme"
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"


class DeactivationMechanism(Enum):
    """失活机理"""
    SINTERING = "sintering"                    # 烧结
    COKING = "coking"                          # 积碳
    POISONING = "poisoning"                    # 中毒
    THERMAL_DEGRADATION = "thermal_degradation"  # 热降解
    ATTRITION = "attrition"                    # 磨损
    PHASE_CHANGE = "phase_change"              # 相变
    LEACHING = "leaching"                      # 浸出


@dataclass
class ReactionCondition:
    """反应条件"""
    temperature: float = 300.0      # K
    pressure: float = 1.0           # bar
    feed_flow_rate: float = 1.0     # mL/min
    feed_composition: Dict[str, float] = field(default_factory=dict)
    space_velocity: float = 1.0     # h^-1


@dataclass
class CatalystProperties:
    """催化剂物性"""
    surface_area: float = 100.0     # m²/g
    pore_volume: float = 0.5        # cm³/g
    particle_size: float = 50.0     # μm
    metal_dispersion: float = 0.5   # fraction
    acidity: float = 100.0          # μmol/g
    crystallinity: float = 0.9      # fraction


@dataclass
class CatalystState:
    """催化剂状态数据类"""
    timestamp: float
    
    # 标识
    reactor_id: str = ""
    catalyst_type: CatalystType = CatalystType.ZEOLITE
    
    # 反应条件
    conditions: ReactionCondition = field(default_factory=ReactionCondition)
    
    # 催化剂物性
    properties: CatalystProperties = field(default_factory=CatalystProperties)
    
    # 性能指标
    conversion: float = 0.0         # %
    selectivity: float = 0.0        # %
    yield_rate: float = 0.0         # mol/h
    activity: float = 1.0           # 相对活性
    
    # 失活指标
    deactivation_rate: float = 0.0  # %/h
    time_on_stream: float = 0.0     # h
    
    # 积碳量
    coke_content: float = 0.0       # wt%
    coke_precursors: List[str] = field(default_factory=list)
    
    # 中毒情况
    poison_coverage: float = 0.0    # %
    poison_species: List[str] = field(default_factory=list)
    
    # 结构变化
    sintering_degree: float = 0.0   # %
    attrition_loss: float = 0.0     # %
    
    # 热稳定性
    thermal_stability_index: float = 1.0
    
    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.conditions.temperature / 1000,
            self.conditions.pressure / 10,
            self.properties.surface_area / 500,
            self.properties.pore_volume,
            self.properties.metal_dispersion,
            self.conversion / 100,
            self.selectivity / 100,
            self.activity,
            self.deactivation_rate,
            self.time_on_stream / 1000,
            self.coke_content / 10,
            self.poison_coverage / 100,
            self.sintering_degree / 100,
            self.thermal_stability_index
        ])
    
    def calculate_ton(self) -> float:
        """计算周转数 (Turnover Number)"""
        if self.properties.metal_dispersion > 0:
            return self.yield_rate * self.time_on_stream / self.properties.metal_dispersion
        return 0.0
    
    def estimate_regeneration_potential(self) -> float:
        """估计再生潜力"""
        # 基于积碳量的简单估计
        if self.coke_content > 0:
            # 积碳可完全烧除
            return min(1.0, 1.0 - self.sintering_degree / 100)
        return 1.0


class DeactivationKinetics:
    """失活动力学模型"""
    
    def __init__(self):
        # 失活速率常数
        self.k_deactivation = 0.001  # h^-1
        
        # 各失活机理的权重
        self.mechanism_weights = {
            DeactivationMechanism.SINTERING: 0.2,
            DeactivationMechanism.COKING: 0.5,
            DeactivationMechanism.POISONING: 0.2,
            DeactivationMechanism.THERMAL_DEGRADATION: 0.1
        }
        
        # 阿累尼乌斯参数
        self.Ea_sintering = 200e3     # J/mol
        self.Ea_coking = 100e3        # J/mol
        
        self.R = 8.314                # J/(mol·K)
    
    def calculate_sintering_rate(
        self,
        temperature: float,
        current_dispersion: float,
        time: float
    ) -> float:
        """
        计算烧结速率
        
        基于Huttig模型: dD/dt = -k * D^n
        """
        # 阿累尼乌斯方程
        k = self.k_deactivation * np.exp(-self.Ea_sintering / (self.R * temperature))
        
        # 烧结速率
        n = 3  # 烧结级数
        rate = k * (current_dispersion ** n)
        
        return rate
    
    def calculate_coking_rate(
        self,
        temperature: float,
        conversion: float,
        feed_composition: Dict[str, float]
    ) -> float:
        """
        计算积碳速率
        
        基于Voorhies方程: C_coke = A * t^n
        """
        # 温度效应
        k_coke = 0.01 * np.exp(-self.Ea_coking / (self.R * temperature))
        
        # 转化率效应
        conversion_factor = conversion ** 1.5
        
        # 原料组成效应（不饱和度）
        unsaturated_content = feed_composition.get('olefins', 0) + feed_composition.get('aromatics', 0)
        composition_factor = 1 + 2 * unsaturated_content
        
        rate = k_coke * conversion_factor * composition_factor
        
        return rate
    
    def calculate_poisoning_rate(
        self,
        poison_concentration: float,
        adsorption_constant: float = 1.0
    ) -> float:
        """
        计算中毒速率
        
        基于Langmuir吸附等温线
        """
        # 覆盖率变化
        theta = adsorption_constant * poison_concentration / (1 + adsorption_constant * poison_concentration)
        
        return theta
    
    def predict_activity(
        self,
        initial_activity: float,
        time_on_stream: float,
        deactivation_order: float = 2.0
    ) -> float:
        """
        预测催化剂活性
        
        通用失活动力学: a = (1 + k_d * t)^(1-n) / (n-1)
        """
        if deactivation_order == 1:
            activity = initial_activity * np.exp(-self.k_deactivation * time_on_stream)
        else:
            activity = initial_activity * (
                1 + (deactivation_order - 1) * self.k_deactivation * time_on_stream
            ) ** (1 / (1 - deactivation_order))
        
        return max(0, min(1, activity))


class MechanismIdentifier:
    """失活机理识别器"""
    
    def __init__(self):
        # 特征阈值
        self.thresholds = {
            'coke_content_high': 5.0,      # wt%
            'temperature_high': 700,        # K
            'surface_area_loss_high': 50,   # %
            'poison_level_high': 10         # %
        }
        
        # 机理特征模式
        self.patterns = {
            DeactivationMechanism.SINTERING: {
                'surface_area_drop': True,
                'crystallinity_change': True,
                'coke_content_low': True
            },
            DeactivationMechanism.COKING: {
                'coke_content_high': True,
                'pore_blocking': True,
                'reversible': True
            },
            DeactivationMechanism.POISONING: {
                'poison_detected': True,
                'selectivity_change': True,
                'irreversible': True
            }
        }
    
    def identify_mechanisms(self, state: CatalystState) -> List[Dict]:
        """
        识别可能的失活机理
        
        Returns:
            识别出的机理列表
        """
        identified = []
        
        # 检查积碳
        if state.coke_content > self.thresholds['coke_content_high']:
            identified.append({
                'mechanism': DeactivationMechanism.COKING,
                'confidence': min(1.0, state.coke_content / 10),
                'evidence': f'High coke content: {state.coke_content:.2f} wt%'
            })
        
        # 检查烧结
        if state.sintering_degree > 20:
            identified.append({
                'mechanism': DeactivationMechanism.SINTERING,
                'confidence': state.sintering_degree / 100,
                'evidence': f'Sintering degree: {state.sintering_degree:.1f}%'
            })
        
        # 检查中毒
        if state.poison_coverage > 5:
            identified.append({
                'mechanism': DeactivationMechanism.POISONING,
                'confidence': state.poison_coverage / 100,
                'evidence': f'Poison coverage: {state.poison_coverage:.1f}%'
            })
        
        # 检查热降解
        if state.conditions.temperature > self.thresholds['temperature_high']:
            identified.append({
                'mechanism': DeactivationMechanism.THERMAL_DEGRADATION,
                'confidence': (state.conditions.temperature - 700) / 100,
                'evidence': f'High temperature: {state.conditions.temperature:.0f} K'
            })
        
        # 按置信度排序
        identified.sort(key=lambda x: x['confidence'], reverse=True)
        
        return identified
    
    def recommend_analysis(self, state: CatalystState) -> List[str]:
        """推荐表征分析方法"""
        methods = []
        
        if state.coke_content > 2:
            methods.extend(['TGA', 'TEM', 'Raman'])
        
        if state.sintering_degree > 10:
            methods.extend(['BET', 'XRD', 'TEM'])
        
        if state.poison_coverage > 1:
            methods.extend(['XPS', 'ICP-MS', 'SEM-EDS'])
        
        methods.extend(['XRD', 'NH3-TPD'])
        
        return list(set(methods))


class RegenerationOptimizer:
    """再生策略优化器"""
    
    def __init__(self):
        # 再生参数范围
        self.temperature_range = (400, 600)  # K
        self.oxygen_range = (1, 21)          # %
        self.time_range = (1, 24)            # h
    
    def optimize_coke_burning(
        self,
        coke_content: float,
        catalyst_type: CatalystType,
        time_constraint: Optional[float] = None
    ) -> Dict:
        """
        优化积碳烧除再生
        
        Args:
            coke_content: 积碳含量 wt%
            catalyst_type: 催化剂类型
            time_constraint: 时间限制 (h)
        
        Returns:
            优化的再生条件
        """
        # 基于积碳量确定基本参数
        if coke_content < 2:
            base_temp = 450
            base_time = 2
        elif coke_content < 5:
            base_temp = 500
            base_time = 4
        else:
            base_temp = 550
            base_time = 8
        
        # 催化剂类型调整
        if catalyst_type == CatalystType.ZEOLITE:
            max_temp = 600  # 避免破坏沸石结构
        elif catalyst_type == CatalystType.PRECIOUS_METAL:
            max_temp = 550  # 避免金属烧结
        else:
            max_temp = 600
        
        temp = min(base_temp, max_temp)
        
        # 时间约束
        if time_constraint and base_time > time_constraint:
            # 提高温度缩短时间
            time_factor = base_time / time_constraint
            temp = min(temp * (1 + 0.1 * np.log(time_factor)), max_temp)
            regen_time = time_constraint
        else:
            regen_time = base_time
        
        return {
            'regeneration_type': 'coke_burning',
            'temperature_K': temp,
            'temperature_C': temp - 273.15,
            'oxygen_content': 5,  # %
            'ramp_rate': 2,       # K/min
            'hold_time': regen_time,
            'cooling_rate': 5,    # K/min
            'estimated_recovery': min(0.95, 1 - 0.05 * coke_content / 10)
        }
    
    def optimize_poison_removal(
        self,
        poison_species: List[str],
        poison_coverage: float
    ) -> Dict:
        """优化中毒催化剂再生"""
        
        strategies = []
        
        for poison in poison_species:
            if poison in ['sulfur', 'sulfide']:
                strategies.append({
                    'poison': poison,
                    'method': 'oxidative_regeneration',
                    'temperature': 600,
                    'atmosphere': 'air',
                    'duration': 4
                })
            elif poison in ['nitrogen', 'ammonia']:
                strategies.append({
                    'poison': poison,
                    'method': 'thermal_desorption',
                    'temperature': 500,
                    'atmosphere': 'inert',
                    'duration': 2
                })
            elif poison in ['metals', 'arsenic']:
                strategies.append({
                    'poison': poison,
                    'method': 'acid_washing',
                    'solution': 'dilute_HCl',
                    'temperature': 353,
                    'duration': 1
                })
        
        return {
            'regeneration_type': 'poison_removal',
            'strategies': strategies,
            'estimated_recovery': max(0.3, 1 - poison_coverage / 100)
        }
    
    def calculate_regeneration_cost(
        self,
        regeneration_params: Dict,
        catalyst_value: float
    ) -> Dict:
        """计算再生成本效益"""
        
        # 能源成本估算
        temp = regeneration_params.get('temperature_K', 500)
        hold_time = regeneration_params.get('hold_time', 4)
        
        # 简化的能源成本计算
        energy_cost = (temp - 300) * hold_time * 0.1  # 相对单位
        
        # 活性恢复价值
        recovery = regeneration_params.get('estimated_recovery', 0.8)
        recovered_value = catalyst_value * recovery
        
        # 成本效益比
        benefit_cost = recovered_value / (energy_cost + 1)
        
        return {
            'energy_cost': energy_cost,
            'recovered_value': recovered_value,
            'benefit_cost_ratio': benefit_cost,
            'recommended': benefit_cost > 2.0
        }


class CatalystDigitalTwin:
    """催化剂数字孪生系统"""
    
    def __init__(
        self,
        reactor_id: str,
        catalyst_type: CatalystType = CatalystType.ZEOLITE,
        initial_mass: float = 1.0  # kg
    ):
        self.reactor_id = reactor_id
        self.catalyst_type = catalyst_type
        self.initial_mass = initial_mass
        
        # 核心数字孪生
        self.twin = create_catalyst_twin(reactor_id)
        
        # 失活动力学
        self.kinetics = DeactivationKinetics()
        
        # 机理识别
        self.mechanism_identifier = MechanismIdentifier()
        
        # 再生优化
        self.regeneration_optimizer = RegenerationOptimizer()
        
        # 预测套件
        self.predictor = PredictiveMaintenanceSuite(
            input_dim=14,
            sequence_length=50
        )
        
        # 传感器网络
        self.sensor_network = SensorNetwork()
        self._setup_sensors()
        
        # 状态
        self.current_state: Optional[CatalystState] = None
        self.state_history: deque = deque(maxlen=10000)
        
        # 反应器模型（降阶）
        self.reactor_rom: Optional[ReducedOrderModel] = None
    
    def _setup_sensors(self) -> None:
        """设置传感器"""
        # 温度传感器（多点）
        for i in range(5):
            self.sensor_network.add_sensor(
                f"reactor_temp_{i}",
                SensorType.TEMPERATURE,
                measurement_dim=1,
                filter_type="kalman"
            )
        
        # 压力传感器
        self.sensor_network.add_sensor(
            "reactor_pressure",
            SensorType.PRESSURE,
            measurement_dim=1,
            filter_type="kalman"
        )
        
        # 在线分析
        self.sensor_network.add_sensor(
            "conversion_analyzer",
            SensorType.OPTICAL,
            measurement_dim=1,
            filter_type="adaptive"
        )
        
        # 热重分析（在线TGA）
        self.sensor_network.add_sensor(
            "coke_tga",
            SensorType.THERMAL_CAMERA,
            measurement_dim=1,
            filter_type="kalman"
        )
    
    def initialize(self, initial_state: Optional[CatalystState] = None) -> None:
        """初始化"""
        if initial_state is None:
            initial_state = CatalystState(
                timestamp=time.time(),
                reactor_id=self.reactor_id,
                catalyst_type=self.catalyst_type,
                activity=1.0
            )
        
        self.current_state = initial_state
        
        # 初始化数字孪生
        material_state = self._catalyst_to_material_state(initial_state)
        self.twin.initialize(material_state)
        
        # 初始化预测器
        self.predictor.update_state(
            features=initial_state.to_vector(),
            health_indicator=initial_state.activity,
            timestamp=initial_state.timestamp
        )
        
        print(f"Catalyst Digital Twin initialized for {self.reactor_id}")
    
    def _catalyst_to_material_state(self, catalyst: CatalystState) -> MaterialState:
        """转换催化剂状态到材料状态"""
        state = MaterialState()
        state.material_id = self.reactor_id
        state.timestamp = catalyst.timestamp
        
        # 映射参数
        state.degradation_index = 1 - catalyst.activity
        state.porosity = catalyst.properties.pore_volume
        state.conductivity = catalyst.properties.metal_dispersion
        
        # 物理参数
        state.physics = PhysicsParameters(
            temperature=catalyst.conditions.temperature,
            pressure=catalyst.conditions.pressure * 1e5  # bar to Pa
        )
        
        return state
    
    def update(
        self,
        temperature: Optional[float] = None,
        pressure: Optional[float] = None,
        conversion: Optional[float] = None,
        coke_content: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """更新催化剂状态"""
        if timestamp is None:
            timestamp = time.time()
        
        # 处理传感器数据
        if temperature:
            for i in range(5):
                reading = SensorReading(
                    sensor_id=f"reactor_temp_{i}",
                    sensor_type=SensorType.TEMPERATURE,
                    timestamp=timestamp,
                    value=np.array([temperature + np.random.randn() * 2]),
                    unit="Kelvin"
                )
                self.sensor_network.ingest_data(reading)
        
        # 更新状态
        if self.current_state:
            new_state = CatalystState(
                timestamp=timestamp,
                reactor_id=self.reactor_id,
                catalyst_type=self.catalyst_type,
                conditions=self.current_state.conditions,
                properties=self.current_state.properties
            )
            
            if temperature:
                new_state.conditions.temperature = temperature
            if pressure:
                new_state.conditions.pressure = pressure
            if conversion:
                new_state.conversion = conversion
            if coke_content:
                new_state.coke_content = coke_content
            
            # 计算活性
            if self.current_state.time_on_stream > 0:
                new_state.activity = self.kinetics.predict_activity(
                    1.0,
                    new_state.time_on_stream
                )
            
            self.current_state = new_state
            self.state_history.append(new_state)
            
            # 更新数字孪生和预测器
            self._update_models(new_state, timestamp)
            
            # 识别失活机理
            mechanisms = self.mechanism_identifier.identify_mechanisms(new_state)
            
            return {
                'state': new_state,
                'identified_mechanisms': mechanisms,
                'recommended_analysis': self.mechanism_identifier.recommend_analysis(new_state)
            }
        
        return {'error': 'Not initialized'}
    
    def _update_models(self, state: CatalystState, timestamp: float) -> None:
        """更新模型"""
        # 更新数字孪生
        material_state = self._catalyst_to_material_state(state)
        self.twin.update_from_observation(
            observation=state.to_vector()[:12],
            physics=material_state.physics
        )
        
        # 更新预测器
        self.predictor.update_state(
            features=state.to_vector(),
            health_indicator=state.activity,
            timestamp=timestamp
        )
    
    def predict_performance(
        self,
        time_horizon: float = 1000,  # h
        operating_conditions: Optional[ReactionCondition] = None
    ) -> Dict[str, Any]:
        """预测未来性能"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        # 使用降阶模型或经验模型预测
        future_times = np.linspace(
            self.current_state.time_on_stream,
            self.current_state.time_on_stream + time_horizon,
            100
        )
        
        activities = []
        conversions = []
        
        for t in future_times:
            # 活性预测
            activity = self.kinetics.predict_activity(1.0, t)
            activities.append(activity)
            
            # 转化率预测（简化模型）
            base_conversion = self.current_state.conversion
            conversion = base_conversion * activity
            conversions.append(conversion)
        
        # RUL预测
        rul = self.predictor.rul_predictor.predict_rul_hybrid(
            current_health=self.current_state.activity,
            current_time=self.current_state.time_on_stream
        )
        
        # 确定更换时机
        replacement_threshold = 0.3  # 30%活性
        time_to_replacement = None
        for i, act in enumerate(activities):
            if act < replacement_threshold:
                time_to_replacement = future_times[i] - self.current_state.time_on_stream
                break
        
        return {
            'current_activity': self.current_state.activity,
            'predicted_activities': list(zip(future_times.tolist(), activities)),
            'predicted_conversions': list(zip(future_times.tolist(), conversions)),
            'predicted_rul_hours': rul.rul_mean,
            'time_to_replacement_hours': time_to_replacement,
            'recommended_regeneration_time': time_to_replacement * 0.7 if time_to_replacement else None
        }
    
    def optimize_operation(
        self,
        target_conversion: float = 80.0,
        cost_weight: float = 0.5
    ) -> Dict[str, Any]:
        """优化操作条件"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        # 当前活性
        activity = self.current_state.activity
        
        # 温度优化
        # 简化的阿累尼乌斯关系
        Ea = 80e3  # J/mol
        R = 8.314
        T_ref = 573  # K
        
        # 要达到目标转化率所需温度
        current_conversion = self.current_state.conversion
        
        if current_conversion < target_conversion:
            # 需要提高温度
            temp_increase = (Ea / R) * np.log(target_conversion / max(current_conversion, 1))
            optimized_temp = min(T_ref + temp_increase, 700)  # 上限700K
        else:
            optimized_temp = self.current_state.conditions.temperature
        
        # 空速优化
        optimized_whsv = 2.0 / activity  # 降低空速补偿活性下降
        
        # 计算优化后的预期寿命
        # 更高温度加速失活
        temp_penalty = np.exp((optimized_temp - 573) / 50)
        expected_life = 1000 / (activity * temp_penalty)
        
        return {
            'current_conditions': {
                'temperature': self.current_state.conditions.temperature,
                'conversion': current_conversion
            },
            'optimized_conditions': {
                'temperature_K': optimized_temp,
                'temperature_C': optimized_temp - 273.15,
                'whsv': optimized_whsv
            },
            'expected_performance': {
                'conversion': target_conversion,
                'expected_life_hours': expected_life
            },
            'trade_offs': [
                'Higher temperature increases conversion but accelerates deactivation',
                'Lower space velocity improves conversion but reduces throughput'
            ]
        }
    
    def plan_regeneration(
        self,
        urgency: str = 'normal',
        cost_constraint: Optional[float] = None
    ) -> Dict[str, Any]:
        """制定再生计划"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        # 识别失活机理
        mechanisms = self.mechanism_identifier.identify_mechanisms(self.current_state)
        
        # 根据主要机理选择再生策略
        strategies = []
        
        for mech in mechanisms[:2]:  # 考虑前两种机理
            mechanism = mech['mechanism']
            
            if mechanism == DeactivationMechanism.COKING:
                strategy = self.regeneration_optimizer.optimize_coke_burning(
                    self.current_state.coke_content,
                    self.catalyst_type
                )
                strategies.append(strategy)
            
            elif mechanism == DeactivationMechanism.POISONING:
                strategy = self.regeneration_optimizer.optimize_poison_removal(
                    self.current_state.poison_species,
                    self.current_state.poison_coverage
                )
                strategies.append(strategy)
        
        # 评估成本效益
        cost_benefits = []
        for strategy in strategies:
            if 'temperature_K' in strategy:
                cb = self.regeneration_optimizer.calculate_regeneration_cost(
                    strategy,
                    catalyst_value=1000  # 假设价值
                )
                cost_benefits.append(cb)
        
        return {
            'identified_mechanisms': [m['mechanism'].value for m in mechanisms],
            'regeneration_strategies': strategies,
            'cost_benefit_analysis': cost_benefits,
            'recommended_strategy': strategies[0] if strategies else None,
            'expected_recovery': strategies[0]['estimated_recovery'] if strategies else 0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        return {
            'reactor_id': self.reactor_id,
            'catalyst_type': self.catalyst_type.value,
            'current_state': {
                'activity': self.current_state.activity,
                'conversion': self.current_state.conversion,
                'selectivity': self.current_state.selectivity,
                'time_on_stream': self.current_state.time_on_stream,
                'coke_content': self.current_state.coke_content
            },
            'properties': {
                'surface_area': self.current_state.properties.surface_area,
                'metal_dispersion': self.current_state.properties.metal_dispersion
            },
            'deactivation': {
                'rate': self.current_state.deactivation_rate,
                'mechanisms': [
                    m['mechanism'].value
                    for m in self.mechanism_identifier.identify_mechanisms(self.current_state)
                ]
            },
            'history_length': len(self.state_history)
        }


def simulate_catalyst_deactivation(
    initial_activity: float = 1.0,
    hours: int = 1000,
    temperature: float = 600,
    coke_formation_rate: float = 0.01
) -> List[CatalystState]:
    """模拟催化剂失活过程"""
    states = []
    
    kinetics = DeactivationKinetics()
    
    for h in range(0, hours, 10):
        # 计算活性
        activity = kinetics.predict_activity(initial_activity, h)
        
        # 积碳
        coke = coke_formation_rate * h * (1 + 0.1 * (temperature - 573) / 100)
        
        # 烧结程度
        sintering = 5 * np.log(1 + h / 100) if h > 0 else 0
        
        state = CatalystState(
            timestamp=time.time() + h * 3600,
            reactor_id="R101",
            catalyst_type=CatalystType.ZEOLITE,
            conditions=ReactionCondition(
                temperature=temperature,
                pressure=2.0
            ),
            activity=activity,
            conversion=85 * activity,
            selectivity=90 - coke * 0.5,
            deactivation_rate=-0.001 * activity,
            time_on_stream=h,
            coke_content=coke,
            sintering_degree=min(50, sintering)
        )
        
        states.append(state)
    
    return states


if __name__ == "__main__":
    print("=" * 70)
    print("Catalyst Deactivation Monitoring - Application Demo")
    print("=" * 70)
    
    # 创建催化剂数字孪生
    print("\n1. Creating Catalyst Digital Twin")
    catalyst_twin = CatalystDigitalTwin(
        reactor_id="FCC_Reactor_R101",
        catalyst_type=CatalystType.ZEOLITE,
        initial_mass=50.0
    )
    
    # 初始化
    initial_properties = CatalystProperties(
        surface_area=350.0,
        pore_volume=0.45,
        metal_dispersion=0.6,
        acidity=150.0
    )
    
    initial_state = CatalystState(
        timestamp=time.time(),
        reactor_id="FCC_Reactor_R101",
        catalyst_type=CatalystType.ZEOLITE,
        properties=initial_properties,
        conditions=ReactionCondition(
            temperature=773,
            pressure=2.5,
            feed_flow_rate=100
        ),
        activity=1.0,
        conversion=85.0,
        selectivity=92.0
    )
    
    catalyst_twin.initialize(initial_state)
    print(f"  Initialized reactor: {catalyst_twin.reactor_id}")
    print(f"  Initial activity: {initial_state.activity:.2f}")
    
    # 模拟失活过程
    print("\n2. Simulating Catalyst Deactivation")
    simulated_states = simulate_catalyst_deactivation(
        initial_activity=1.0,
        hours=500,
        temperature=773,
        coke_formation_rate=0.02
    )
    
    for i, state in enumerate(simulated_states[:20]):
        result = catalyst_twin.update(
            temperature=state.conditions.temperature,
            conversion=state.conversion,
            coke_content=state.coke_content,
            timestamp=state.timestamp
        )
        
        if i % 5 == 0:
            print(f"  Hour {state.time_on_stream}: Activity = {state.activity:.3f}, "
                  f"Coke = {state.coke_content:.2f} wt%")
            
            mechanisms = result.get('identified_mechanisms', [])
            if mechanisms:
                print(f"    Identified mechanisms: {[m['mechanism'].value for m in mechanisms[:2]]}")
    
    # 性能预测
    print("\n3. Performance Prediction")
    prediction = catalyst_twin.predict_performance(time_horizon=500)
    print(f"  Current activity: {prediction['current_activity']:.3f}")
    print(f"  Predicted RUL: {prediction['predicted_rul_hours']:.0f} hours")
    print(f"  Time to replacement: {prediction['time_to_replacement_hours']:.0f} hours")
    if prediction['recommended_regeneration_time']:
        print(f"  Recommended regeneration: {prediction['recommended_regeneration_time']:.0f} hours")
    
    # 操作优化
    print("\n4. Operation Optimization")
    optimization = catalyst_twin.optimize_operation(target_conversion=80)
    print(f"  Current conversion: {optimization['current_conditions']['conversion']:.1f}%")
    print(f"  Optimized temperature: {optimization['optimized_conditions']['temperature_C']:.0f}°C")
    print(f"  Expected conversion: {optimization['expected_performance']['conversion']:.1f}%")
    print(f"  Expected life: {optimization['expected_performance']['expected_life_hours']:.0f} hours")
    
    # 再生计划
    print("\n5. Regeneration Planning")
    regen_plan = catalyst_twin.plan_regeneration()
    print(f"  Identified mechanisms: {regen_plan['identified_mechanisms']}")
    if regen_plan['regeneration_strategies']:
        strategy = regen_plan['regeneration_strategies'][0]
        print(f"  Strategy: {strategy['regeneration_type']}")
        if 'temperature_C' in strategy:
            print(f"  Temperature: {strategy['temperature_C']:.0f}°C")
            print(f"  Hold time: {strategy['hold_time']:.1f} hours")
        print(f"  Expected recovery: {regen_plan['expected_recovery']*100:.0f}%")
    
    # 状态报告
    print("\n6. Status Report")
    status = catalyst_twin.get_status()
    print(f"  Activity: {status['current_state']['activity']:.3f}")
    print(f"  Conversion: {status['current_state']['conversion']:.1f}%")
    print(f"  Time on stream: {status['current_state']['time_on_stream']:.0f} hours")
    print(f"  Deactivation mechanisms: {status['deactivation']['mechanisms']}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
