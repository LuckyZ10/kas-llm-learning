"""
Structural Material Lifetime Prediction
结构材料寿命预测应用案例

实现结构材料的数字孪生，包括：
- 疲劳裂纹监测与预测
- 应力-应变分析
- 断裂韧性评估
- 维护计划优化
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
    create_structural_twin
)
from dftlammps.digital_twin.sensor_fusion import (
    KalmanFilter,
    CalibrationParams
)


class LoadingType(Enum):
    """加载类型"""
    STATIC = "static"
    CYCLIC = "cyclic"
    RANDOM = "random"
    IMPACT = "impact"
    THERMAL = "thermal"


@dataclass
class CrackInfo:
    """裂纹信息"""
    position: np.ndarray           # 位置 [x, y, z]
    length: float = 0.0            # 裂纹长度 (mm)
    depth: float = 0.0             # 裂纹深度 (mm)
    orientation: float = 0.0       # 方向角 (度)
    growth_rate: float = 0.0       # 增长率 (mm/cycle)
    stress_intensity_factor: float = 0.0  # 应力强度因子 K


@dataclass
class StructuralState:
    """结构状态数据类"""
    timestamp: float
    
    # 几何参数
    component_id: str = ""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # 应力参数 (Voigt notation)
    stress: np.ndarray = field(default_factory=lambda: np.zeros(6))
    strain: np.ndarray = field(default_factory=lambda: np.zeros(6))
    stress_amplitude: float = 0.0
    mean_stress: float = 0.0
    
    # 材料参数
    elastic_modulus: float = 200e9  # Pa
    yield_strength: float = 250e6   # Pa
    ultimate_strength: float = 400e6  # Pa
    fracture_toughness: float = 50e6  # Pa√m
    
    # 疲劳参数
    fatigue_cycles: int = 0
    fatigue_damage: float = 0.0
    remaining_fatigue_life: float = float('inf')
    
    # 裂纹信息
    cracks: List[CrackInfo] = field(default_factory=list)
    
    # 环境参数
    temperature: float = 25.0       # °C
    humidity: float = 50.0          # %
    corrosion_exposure: float = 0.0  # 腐蚀暴露时间
    
    # 振动参数
    natural_frequencies: List[float] = field(default_factory=list)
    vibration_amplitude: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            *self.stress[:3],  # 正应力分量
            *self.strain[:3],  # 正应变分量
            self.stress_amplitude / 1e6,  # MPa
            self.mean_stress / 1e6,
            self.elastic_modulus / 1e9,   # GPa
            self.yield_strength / 1e6,    # MPa
            self.fracture_toughness / 1e6,
            self.fatigue_cycles / 1e6,    # 归一化
            self.fatigue_damage,
            self.temperature,
            self.humidity / 100,
            len(self.cracks) / 10  # 裂纹数量（归一化）
        ])
    
    def get_von_mises_stress(self) -> float:
        """计算von Mises等效应力"""
        s11, s22, s33 = self.stress[0], self.stress[1], self.stress[2]
        s12, s13, s23 = self.stress[3], self.stress[4], self.stress[5]
        
        return np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 +
                              6*(s12**2 + s13**2 + s23**2)))
    
    def get_principal_stresses(self) -> np.ndarray:
        """计算主应力"""
        # 构建应力张量
        sigma = np.array([
            [self.stress[0], self.stress[3], self.stress[4]],
            [self.stress[3], self.stress[1], self.stress[5]],
            [self.stress[4], self.stress[5], self.stress[2]]
        ])
        
        eigenvalues = np.linalg.eigvalsh(sigma)
        return np.sort(eigenvalues)[::-1]  # 降序排列


class FatigueLifePredictor:
    """疲劳寿命预测器"""
    
    def __init__(self):
        # S-N曲线参数
        self.fatigue_strength_coefficient = 1000e6  # σf'
        self.fatigue_strength_exponent = -0.1      # b
        self.fatigue_ductility_coefficient = 1.0    # εf'
        self.fatigue_ductility_exponent = -0.6      # c
        
        # 平均应力修正 (Goodman准则)
        self.use_goodman_correction = True
        
        # Miner累积损伤
        self.miner_exponent = 1.0
    
    def predict_sn_curve(
        self,
        stress_amplitude: float,
        material_ultimate_strength: float,
        cycles_to_failure: Optional[int] = None
    ) -> Dict:
        """
        基于S-N曲线预测寿命
        
        Args:
            stress_amplitude: 应力幅值 (Pa)
            material_ultimate_strength: 材料极限强度 (Pa)
            cycles_to_failure: 用于校准的失效循环数
        
        Returns:
            预测结果
        """
        if cycles_to_failure:
            # 校准模型
            self.fatigue_strength_coefficient = stress_amplitude * (cycles_to_failure ** (-self.fatigue_strength_exponent))
        
        # Basquin方程: σa = σf' * (2N)^b
        # 求解 N
        Nf = 0.5 * (stress_amplitude / self.fatigue_strength_coefficient) ** (1 / self.fatigue_strength_exponent)
        
        # 疲劳极限
        fatigue_limit = 0.5 * material_ultimate_strength
        
        return {
            'cycles_to_failure': int(Nf),
            'fatigue_limit': fatigue_limit,
            'infinite_life': stress_amplitude < fatigue_limit,
            'stress_amplitude': stress_amplitude
        }
    
    def apply_goodman_correction(
        self,
        stress_amplitude: float,
        mean_stress: float,
        ultimate_strength: float
    ) -> float:
        """Goodman平均应力修正"""
        if not self.use_goodman_correction or ultimate_strength == 0:
            return stress_amplitude
        
        # Goodman方程: σa / σN + σm / σu = 1
        # 修正后的应力幅
        corrected_amplitude = stress_amplitude / (1 - mean_stress / ultimate_strength)
        
        return corrected_amplitude
    
    def calculate_damage(
        self,
        stress_history: List[Tuple[float, float]],  # (应力幅, 循环数)
        material_ultimate_strength: float
    ) -> float:
        """
        使用Miner准则计算累积损伤
        
        Args:
            stress_history: 应力历史 [(应力幅, 循环数), ...]
            material_ultimate_strength: 材料极限强度
        
        Returns:
            累积损伤值
        """
        total_damage = 0.0
        
        for stress_amp, n_cycles in stress_history:
            # 预测该应力幅下的寿命
            life_pred = self.predict_sn_curve(stress_amp, material_ultimate_strength)
            Nf = life_pred['cycles_to_failure']
            
            if Nf > 0:
                damage = n_cycles / Nf
                total_damage += damage
        
        return total_damage
    
    def rainflow_count(
        self,
        stress_series: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        雨流计数法提取应力循环
        
        Args:
            stress_series: 应力时间序列
        
        Returns:
            应力循环列表 [(应力幅, 平均应力), ...]
        """
        # 简化的雨流计数实现
        cycles = []
        
        # 找到峰谷值
        peaks_valleys = []
        for i in range(1, len(stress_series) - 1):
            if (stress_series[i] > stress_series[i-1] and 
                stress_series[i] > stress_series[i+1]):
                peaks_valleys.append(('peak', stress_series[i], i))
            elif (stress_series[i] < stress_series[i-1] and 
                  stress_series[i] < stress_series[i+1]):
                peaks_valleys.append(('valley', stress_series[i], i))
        
        # 配对形成循环
        for i in range(0, len(peaks_valleys) - 1, 2):
            if i + 1 < len(peaks_valleys):
                peak_val = peaks_valleys[i][1]
                valley_val = peaks_valleys[i+1][1]
                
                stress_amp = abs(peak_val - valley_val) / 2
                mean_stress = (peak_val + valley_val) / 2
                
                cycles.append((stress_amp, mean_stress))
        
        return cycles


class CrackGrowthPredictor:
    """裂纹增长预测器 (Paris定律)"""
    
    def __init__(self):
        # Paris定律参数
        self.C = 1e-12  # 材料常数
        self.m = 3.0    # 指数
        
        # 应力强度因子阈值
        self.K_threshold = 5e6  # Pa√m
        
        # 临界应力强度因子
        self.K_critical = 50e6  # Pa√m
    
    def calculate_stress_intensity(
        self,
        stress: float,
        crack_length: float,
        geometry_factor: float = 1.12
    ) -> float:
        """
        计算应力强度因子 K
        
        Args:
            stress: 远场应力 (Pa)
            crack_length: 裂纹长度 (m)
            geometry_factor: 几何修正因子
        
        Returns:
            应力强度因子 (Pa√m)
        """
        return geometry_factor * stress * np.sqrt(np.pi * crack_length)
    
    def predict_growth_rate(self, delta_K: float) -> float:
        """
        预测裂纹增长速率 (Paris定律)
        
        da/dN = C * (ΔK)^m
        
        Returns:
            裂纹增长速率 (m/cycle)
        """
        if delta_K < self.K_threshold:
            return 0.0
        
        return self.C * (delta_K ** self.m)
    
    def predict_life(
        self,
        initial_crack_length: float,
        critical_crack_length: float,
        stress_range: float,
        geometry_factor: float = 1.12
    ) -> Dict:
        """
        预测裂纹扩展到临界尺寸的寿命
        
        Args:
            initial_crack_length: 初始裂纹长度 (m)
            critical_crack_length: 临界裂纹长度 (m)
            stress_range: 应力范围 (Pa)
            geometry_factor: 几何修正因子
        
        Returns:
            寿命预测结果
        """
        a = initial_crack_length
        N = 0
        
        # 数值积分计算寿命
        da = 0.001 * initial_crack_length  # 步长
        
        while a < critical_crack_length:
            # 当前应力强度因子
            K = self.calculate_stress_intensity(stress_range, a, geometry_factor)
            
            if K >= self.K_critical:
                break
            
            # 增长速率
            dadN = self.predict_growth_rate(K)
            
            if dadN <= 0:
                break
            
            # 更新
            dN = da / dadN
            N += dN
            a += da
        
        return {
            'cycles_to_failure': int(N),
            'critical_length_reached': a >= critical_crack_length,
            'final_crack_length': a
        }


class StructuralDigitalTwin:
    """结构材料数字孪生系统"""
    
    def __init__(
        self,
        component_id: str,
        material_type: str = "steel",
        geometry: Optional[Dict] = None
    ):
        self.component_id = component_id
        self.material_type = material_type
        self.geometry = geometry or {}
        
        # 核心数字孪生
        self.twin = create_structural_twin(component_id)
        
        # 预测器
        self.fatigue_predictor = FatigueLifePredictor()
        self.crack_predictor = CrackGrowthPredictor()
        self.predictor = PredictiveMaintenanceSuite(
            input_dim=13,
            sequence_length=50
        )
        
        # 传感器网络
        self.sensor_network = SensorNetwork()
        self._setup_sensors()
        
        # 状态历史
        self.state_history: deque = deque(maxlen=10000)
        self.current_state: Optional[StructuralState] = None
        
        # 加载历史
        self.loading_history: deque = deque(maxlen=10000)
        
        # 裂纹数据库
        self.detected_cracks: List[CrackInfo] = []
    
    def _setup_sensors(self) -> None:
        """设置传感器网络"""
        # 应变片阵列
        for i in range(6):
            self.sensor_network.add_sensor(
                f"strain_gauge_{i}",
                SensorType.STRAIN_GAUGE,
                measurement_dim=1,
                filter_type="kalman"
            )
        
        # 加速度计
        for i in range(3):
            self.sensor_network.add_sensor(
                f"accelerometer_{i}",
                SensorType.ACCELEROMETER,
                measurement_dim=3,
                filter_type="kalman"
            )
        
        # 位移传感器
        self.sensor_network.add_sensor(
            "displacement_01",
            SensorType.DISPLACEMENT,
            measurement_dim=1,
            filter_type="kalman"
        )
        
        # 温度传感器
        self.sensor_network.add_sensor(
            "temperature_01",
            SensorType.TEMPERATURE,
            measurement_dim=1,
            filter_type="adaptive"
        )
        
        # 声发射传感器（裂纹检测）
        self.sensor_network.add_sensor(
            "acoustic_emission",
            SensorType.ACOUSTIC,
            measurement_dim=1,
            filter_type="adaptive"
        )
    
    def initialize(self, initial_state: Optional[StructuralState] = None) -> None:
        """初始化结构孪生"""
        if initial_state is None:
            initial_state = StructuralState(
                timestamp=time.time(),
                component_id=self.component_id
            )
        
        self.current_state = initial_state
        
        # 初始化数字孪生
        material_state = self._structural_to_material_state(initial_state)
        self.twin.initialize(material_state)
        
        # 初始化预测器
        self.predictor.update_state(
            features=initial_state.to_vector(),
            health_indicator=1.0 - initial_state.fatigue_damage,
            timestamp=initial_state.timestamp
        )
        
        print(f"Structural Digital Twin initialized for {self.component_id}")
    
    def _structural_to_material_state(
        self,
        structural: StructuralState
    ) -> MaterialState:
        """转换结构状态到材料状态"""
        state = MaterialState()
        state.material_id = self.component_id
        state.timestamp = structural.timestamp
        
        # 映射参数
        state.elastic_modulus = structural.elastic_modulus
        state.yield_strength = structural.yield_strength
        state.fracture_toughness = structural.fracture_toughness
        state.fatigue_cycles = structural.fatigue_cycles
        state.degradation_index = structural.fatigue_damage
        
        # 物理参数
        state.physics = PhysicsParameters(
            temperature=structural.temperature + 273.15,
            stress=structural.stress,
            strain=structural.strain
        )
        
        return state
    
    def update_from_measurements(
        self,
        strains: Optional[List[float]] = None,
        accelerations: Optional[List[np.ndarray]] = None,
        displacement: Optional[float] = None,
        temperature: Optional[float] = None,
        acoustic_emission: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """从测量数据更新"""
        if timestamp is None:
            timestamp = time.time()
        
        # 处理应变数据
        if strains:
            for i, strain in enumerate(strains[:6]):
                reading = SensorReading(
                    sensor_id=f"strain_gauge_{i}",
                    sensor_type=SensorType.STRAIN_GAUGE,
                    timestamp=timestamp,
                    value=np.array([strain]),
                    unit="microstrain"
                )
                self.sensor_network.ingest_data(reading)
        
        # 处理温度
        if temperature is not None:
            reading = SensorReading(
                sensor_id="temperature_01",
                sensor_type=SensorType.TEMPERATURE,
                timestamp=timestamp,
                value=np.array([temperature]),
                unit="Celsius"
            )
            self.sensor_network.ingest_data(reading)
        
        # 处理声发射（裂纹检测）
        if acoustic_emission is not None:
            reading = SensorReading(
                sensor_id="acoustic_emission",
                sensor_type=SensorType.ACOUSTIC,
                timestamp=timestamp,
                value=np.array([acoustic_emission]),
                unit="dB"
            )
            self.sensor_network.ingest_data(reading)
            
            # 检测裂纹萌生
            if acoustic_emission > 60:  # 阈值
                self._detect_crack(timestamp)
        
        # 获取融合数据
        fused = self.sensor_network.get_fused_state()
        
        # 更新结构状态
        new_state = self._update_structural_state(fused, timestamp)
        self.current_state = new_state
        self.state_history.append(new_state)
        
        # 更新数字孪生
        material_state = self._structural_to_material_state(new_state)
        twin_result = self.twin.update_from_observation(
            observation=new_state.to_vector()[:12],
            physics=material_state.physics
        )
        
        # 更新预测器
        predictor_result = self.predictor.update_state(
            features=new_state.to_vector(),
            health_indicator=1.0 - new_state.fatigue_damage,
            timestamp=timestamp
        )
        
        # 计算疲劳损伤
        self._update_fatigue_damage(new_state)
        
        return {
            'structural_state': new_state,
            'fused_data': fused,
            'twin_update': twin_result,
            'predictor_result': predictor_result,
            'cracks_detected': len(self.detected_cracks)
        }
    
    def _update_structural_state(
        self,
        fused: Dict,
        timestamp: float
    ) -> StructuralState:
        """从融合数据更新结构状态"""
        state = StructuralState(timestamp=timestamp)
        state.component_id = self.component_id
        
        if self.current_state:
            # 继承历史状态
            state.fatigue_cycles = self.current_state.fatigue_cycles
            state.fatigue_damage = self.current_state.fatigue_damage
            state.elastic_modulus = self.current_state.elastic_modulus
            state.yield_strength = self.current_state.yield_strength
            state.cracks = self.current_state.cracks.copy()
        
        # 从传感器数据更新
        if fused.get('status') == 'fused':
            fused_val = fused.get('fused_value', np.zeros(5))
            # 简化的映射
            state.temperature = fused_val[0] if len(fused_val) > 0 else 25.0
        
        return state
    
    def _detect_crack(self, timestamp: float) -> None:
        """检测裂纹"""
        crack = CrackInfo(
            position=np.random.randn(3) * 0.1,  # 模拟位置
            length=0.1,
            depth=0.05,
            growth_rate=0.0
        )
        self.detected_cracks.append(crack)
        
        if self.current_state:
            self.current_state.cracks.append(crack)
    
    def _update_fatigue_damage(self, state: StructuralState) -> None:
        """更新疲劳损伤"""
        # 简化的疲劳损伤累积
        if state.stress_amplitude > 0:
            # 获取S-N曲线预测
            life_pred = self.fatigue_predictor.predict_sn_curve(
                state.stress_amplitude,
                state.ultimate_strength
            )
            
            # 计算增量损伤
            if life_pred['cycles_to_failure'] > 0:
                delta_damage = 1.0 / life_pred['cycles_to_failure']
                state.fatigue_damage += delta_damage
                state.fatigue_damage = min(1.0, state.fatigue_damage)
                
                # 更新剩余寿命
                if state.fatigue_damage < 1.0:
                    state.remaining_fatigue_life = (
                        (1.0 - state.fatigue_damage) / delta_damage
                    )
    
    def predict_lifetime(
        self,
        loading_scenario: Optional[LoadingType] = None,
        years: int = 20
    ) -> Dict[str, Any]:
        """预测结构寿命"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        # 疲劳寿命预测
        fatigue_pred = self.fatigue_predictor.predict_sn_curve(
            self.current_state.stress_amplitude,
            self.current_state.ultimate_strength
        )
        
        # RUL预测
        rul_result = self.predictor.rul_predictor.predict_rul_hybrid(
            current_health=1.0 - self.current_state.fatigue_damage,
            current_time=time.time()
        )
        
        # 裂纹增长预测
        crack_predictions = []
        for crack in self.detected_cracks:
            pred = self.crack_predictor.predict_life(
                initial_crack_length=crack.length / 1000,  # mm to m
                critical_crack_length=0.05,  # 假设临界尺寸
                stress_range=self.current_state.stress_amplitude
            )
            crack_predictions.append({
                'initial_length': crack.length,
                'predicted_cycles': pred['cycles_to_failure']
            })
        
        # 风险评估
        risk = self.predictor.warning_system.get_risk_assessment(rul_result)
        
        return {
            'current_damage': self.current_state.fatigue_damage,
            'cycles_to_failure_fatigue': fatigue_pred['cycles_to_failure'],
            'predicted_rul_cycles': rul_result.rul_mean,
            'crack_predictions': crack_predictions,
            'risk_level': risk['risk_level'],
            'risk_score': risk['risk_score'],
            'recommended_action': risk['recommended_action'],
            'dominant_failure_mode': self._determine_failure_mode()
        }
    
    def _determine_failure_mode(self) -> str:
        """确定主导失效模式"""
        if not self.current_state:
            return "unknown"
        
        # 基于当前状态判断
        if self.current_state.cracks:
            return "fracture"
        elif self.current_state.fatigue_damage > 0.5:
            return "fatigue"
        elif self.current_state.stress_amplitude > self.current_state.yield_strength * 0.8:
            return "plastic_deformation"
        else:
            return "fatigue"
    
    def optimize_inspection_schedule(
        self,
        risk_tolerance: float = 0.1,
        max_interval_years: float = 5.0
    ) -> Dict[str, Any]:
        """优化检测计划"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        # 基于风险等级确定检测间隔
        life_pred = self.predict_lifetime()
        risk_level = life_pred.get('risk_level', 'low')
        
        inspection_intervals = {
            'critical': 0.25,   # 3个月
            'high': 0.5,        # 6个月
            'medium': 1.0,      # 1年
            'low': 2.0,         # 2年
            'info': max_interval_years
        }
        
        interval = inspection_intervals.get(risk_level, max_interval_years)
        
        # 生成检测计划
        schedule = []
        current_time = 0
        while current_time < max_interval_years:
            schedule.append({
                'year': current_time,
                'inspection_type': self._determine_inspection_type(current_time),
                'focus_areas': self._determine_focus_areas()
            })
            current_time += interval
        
        return {
            'risk_level': risk_level,
            'recommended_interval_years': interval,
            'inspection_schedule': schedule,
            'ndt_methods': self._recommend_ndt_methods()
        }
    
    def _determine_inspection_type(self, year: float) -> str:
        """确定检测类型"""
        if year == 0:
            return "baseline"
        elif year % 2 == 0:
            return "comprehensive"
        else:
            return "routine"
    
    def _determine_focus_areas(self) -> List[str]:
        """确定重点关注区域"""
        areas = []
        
        if self.current_state:
            # 高应力区域
            if self.current_state.stress_amplitude > 200e6:
                areas.append("high_stress_zones")
            
            # 裂纹位置
            if self.detected_cracks:
                areas.append("known_crack_locations")
            
            # 连接部位
            areas.append("joints_and_connections")
        
        return areas
    
    def _recommend_ndt_methods(self) -> List[str]:
        """推荐无损检测方法"""
        methods = ['visual_inspection']
        
        if self.detected_cracks:
            methods.extend(['ultrasonic_testing', 'eddy_current'])
        
        if self.current_state and self.current_state.fatigue_damage > 0.3:
            methods.append('acoustic_emission')
        
        return methods
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        if self.current_state is None:
            return {'error': 'Not initialized'}
        
        return {
            'component_id': self.component_id,
            'current_state': {
                'von_mises_stress': self.current_state.get_von_mises_stress() / 1e6,
                'principal_stresses': self.current_state.get_principal_stresses() / 1e6,
                'fatigue_damage': self.current_state.fatigue_damage,
                'fatigue_cycles': self.current_state.fatigue_cycles,
                'num_cracks': len(self.current_state.cracks)
            },
            'detected_cracks': [
                {
                    'length': c.length,
                    'depth': c.depth,
                    'growth_rate': c.growth_rate
                }
                for c in self.detected_cracks
            ],
            'twin_status': self.twin.get_system_status(),
            'sensor_status': self.sensor_network.get_network_status()
        }


def simulate_fatigue_loading(
    cycles: int = 1000,
    stress_amplitude: float = 150e6,
    mean_stress: float = 50e6,
    sample_rate: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """模拟疲劳加载"""
    t = np.linspace(0, cycles, cycles * sample_rate)
    
    # 正弦载荷
    stress = mean_stress + stress_amplitude * np.sin(2 * np.pi * t)
    
    # 添加噪声
    stress += np.random.randn(len(t)) * 5e6
    
    return t, stress


if __name__ == "__main__":
    print("=" * 70)
    print("Structural Material Lifetime Prediction - Application Demo")
    print("=" * 70)
    
    # 创建结构数字孪生
    print("\n1. Creating Structural Digital Twin")
    structural_twin = StructuralDigitalTwin(
        component_id="Bridge_Girder_A1",
        material_type="Structural Steel Q345",
        geometry={'length': 30.0, 'width': 0.5, 'height': 1.2}
    )
    
    # 初始化
    initial_state = StructuralState(
        timestamp=time.time(),
        component_id="Bridge_Girder_A1",
        elastic_modulus=206e9,
        yield_strength=345e6,
        ultimate_strength=470e6,
        fracture_toughness=80e6
    )
    structural_twin.initialize(initial_state)
    print(f"  Initialized component: {structural_twin.component_id}")
    
    # 模拟疲劳加载
    print("\n2. Simulating Fatigue Loading")
    t, stress = simulate_fatigue_loading(cycles=100, stress_amplitude=180e6)
    
    # 更新数字孪生
    for i in range(0, len(stress), 100):
        strain = stress[i] / initial_state.elastic_modulus * 1e6  # 微应变
        
        result = structural_twin.update_from_measurements(
            strains=[strain] * 6,
            temperature=25.0 + np.random.randn() * 2,
            acoustic_emission=np.random.uniform(30, 70),
            timestamp=time.time() + i / 100
        )
        
        if i % 5000 == 0:
            current = structural_twin.current_state
            print(f"  Cycle {i//100}: Damage = {current.fatigue_damage:.4f}, Cycles = {current.fatigue_cycles}")
    
    # 寿命预测
    print("\n3. Lifetime Prediction")
    life_pred = structural_twin.predict_lifetime(loading_scenario=LoadingType.CYCLIC)
    print(f"  Current damage: {life_pred['current_damage']:.4f}")
    print(f"  Cycles to failure (fatigue): {life_pred['cycles_to_failure_fatigue']}")
    print(f"  Predicted RUL: {life_pred['predicted_rul_cycles']:.0f} cycles")
    print(f"  Risk level: {life_pred['risk_level']}")
    print(f"  Dominant failure mode: {life_pred['dominant_failure_mode']}")
    
    # 检测计划优化
    print("\n4. Inspection Schedule Optimization")
    schedule = structural_twin.optimize_inspection_schedule()
    print(f"  Risk level: {schedule['risk_level']}")
    print(f"  Recommended interval: {schedule['recommended_interval_years']:.1f} years")
    print(f"  Number of inspections (5 years): {len(schedule['inspection_schedule'])}")
    print(f"  NDT methods: {', '.join(schedule['ndt_methods'])}")
    
    # 状态报告
    print("\n5. Status Report")
    report = structural_twin.get_status_report()
    print(f"  Von Mises stress: {report['current_state']['von_mises_stress']:.1f} MPa")
    print(f"  Fatigue damage: {report['current_state']['fatigue_damage']:.4f}")
    print(f"  Detected cracks: {report['current_state']['num_cracks']}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
