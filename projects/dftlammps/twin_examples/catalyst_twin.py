"""
催化剂数字孪生 (Catalyst Digital Twin)

实现催化剂的数字孪生系统，包括活性位点演化和失活预测。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


try:
    from ..digital_twin.twin_core import (
        DigitalTwinCore, TwinConfiguration, StateVector, Observation,
        PhysicsBasedModel, ModelType, NeuralSurrogateModel
    )
    from ..digital_twin.real_time_sync import (
        RealTimeSynchronizer, SyncConfiguration, SyncDirection, SyncMode
    )
    from ..digital_twin.predictive_model import (
        PredictiveMaintenanceEngine, DegradationModel,
        ExponentialDegradationModel, PowerLawDegradationModel,
        HealthIndicator, RULPrediction, FailureMode
    )
    from ..digital_twin.uncertainty_quantification import (
        UQEngine, UQMethod, UncertaintyEstimate
    )
    from ..twin_visualization.dashboard import Dashboard, DashboardConfig
    from ..twin_visualization.anomaly_alert import AnomalyDetectionSystem
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from digital_twin.twin_core import (
        DigitalTwinCore, TwinConfiguration, StateVector, Observation,
        PhysicsBasedModel, ModelType, NeuralSurrogateModel
    )
    from digital_twin.real_time_sync import (
        RealTimeSynchronizer, SyncConfiguration, SyncDirection, SyncMode
    )
    from digital_twin.predictive_model import (
        PredictiveMaintenanceEngine, DegradationModel,
        ExponentialDegradationModel, PowerLawDegradationModel,
        HealthIndicator, RULPrediction, FailureMode
    )
    from digital_twin.uncertainty_quantification import (
        UQEngine, UQMethod, UncertaintyEstimate
    )
    from twin_visualization.dashboard import Dashboard, DashboardConfig
    from twin_visualization.anomaly_alert import AnomalyDetectionSystem


class CatalystType(Enum):
    """催化剂类型"""
    HETEROGENEOUS = "heterogeneous"  # 多相催化
    HOMOGENEOUS = "homogeneous"      # 均相催化
    ENZYME = "enzyme"                # 酶催化
    PHOTOCATALYST = "photocatalyst"  # 光催化
    ELECTROCATALYST = "electrocatalyst"  # 电催化


class DeactivationMechanism(Enum):
    """失活机理"""
    SINTERING = "sintering"          # 烧结
    COKING = "coking"                # 积碳
    POISONING = "poisoning"          # 中毒
    ACTIVE_PHASE_CHANGE = "phase_change"  # 活性相变化
    LEACHING = "leaching"            # 活性组分流失
    THERMAL_DEGRADATION = "thermal"  # 热降解


@dataclass
class ActiveSite:
    """活性位点"""
    site_id: int
    site_type: str           # 位点类型 (台阶、扭结、晶面等)
    coordination_number: int  # 配位数
    position: NDArray[np.float64]
    binding_energy: float = 0.0   # 吸附能 (eV)
    activity: float = 1.0         # 相对活性 (0-1)
    coverage: float = 0.0         # 表面覆盖度
    
    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float64)


@dataclass
class ReactionCondition:
    """反应条件"""
    temperature: float = 300.0       # K
    pressure: float = 1.0            # bar
    flow_rate: float = 100.0         # mL/min
    reactant_concentration: float = 0.1  # mol/L
    
    # 对于电催化
    potential: float = 0.0           # V vs RHE
    current_density: float = 0.0     # mA/cm²


@dataclass
class CatalystSpecification:
    """催化剂规格"""
    # 基本信息
    name: str = "Catalyst-1"
    catalyst_type: CatalystType = CatalystType.HETEROGENEOUS
    active_metal: str = "Pt"
    support: str = "Al2O3"
    
    # 物理性质
    surface_area: float = 100.0      # m²/g
    pore_volume: float = 0.5         # cm³/g
    particle_size: float = 5.0       # nm
    metal_loading: float = 1.0       # wt%
    
    # 初始活性位点
    initial_active_sites: int = 1000
    initial_dispersion: float = 0.5  # 金属分散度


@dataclass
class CatalystState:
    """催化剂状态"""
    timestamp: float
    
    # 反应条件
    temperature: float = 300.0       # K
    pressure: float = 1.0            # bar
    
    # 性能指标
    conversion: float = 0.0          # 转化率
    selectivity: float = 0.0         # 选择性
    yield_rate: float = 0.0          # 产率
    
    # 结构参数
    active_site_count: int = 0
    average_particle_size: float = 5.0  # nm
    dispersion: float = 0.5
    
    # 失活状态
    time_on_stream: float = 0.0      # h
    deactivation_rate: float = 0.0   # %/h
    coke_content: float = 0.0        # wt%
    poison_content: float = 0.0      # wt%
    
    @property
    def activity(self) -> float:
        """计算相对活性"""
        return self.active_site_count / 1000  # 假设初始1000个位点


class CatalystPhysicsModel(PhysicsBasedModel):
    """
    催化剂物理模型
    
    模拟催化剂的结构演化和失活过程
    """
    
    def __init__(self, spec: CatalystSpecification):
        super().__init__(ModelType.HYBRID, {
            'Ea_sintering': 150000,   # 烧结活化能 (J/mol)
            'k_coking': 0.001,        # 积碳速率常数
            'k_poison': 0.0001,       # 中毒速率常数
            'particle_growth_rate': 0.01,  # nm/h
        })
        self.spec = spec
        
        # 活性位点集合
        self.active_sites: Dict[int, ActiveSite] = {}
        self._init_active_sites()
        
        # 反应网络 (简化)
        self.reaction_rate_constant = 1.0  # 1/s
        
    def _init_active_sites(self) -> None:
        """初始化活性位点"""
        np.random.seed(42)
        
        for i in range(self.spec.initial_active_sites):
            # 随机位置 (纳米颗粒表面)
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = self.spec.particle_size / 2
            
            pos = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            
            # 位点类型 (基于配位数)
            cn = np.random.choice([6, 8, 9], p=[0.1, 0.3, 0.6])
            site_type = {6: 'kink', 8: 'step', 9: 'terrace'}.get(cn, 'terrace')
            
            # 活性与配位数相关 (低配位 = 高活性)
            activity = 1.0 - (cn - 6) / 6 * 0.5
            
            site = ActiveSite(
                site_id=i,
                site_type=site_type,
                coordination_number=cn,
                position=pos,
                binding_energy=-1.0 - np.random.rand() * 1.0,  # -1 to -2 eV
                activity=activity
            )
            
            self.active_sites[i] = site
    
    def step(self, dt: float) -> StateVector:
        """执行催化剂演化步进"""
        if self._state is None:
            raise RuntimeError("Model not initialized")
        
        state_data = self._state.data
        temperature = state_data[0]  # K
        pressure = state_data[1]     # bar
        time_on_stream = state_data[2]  # h
        coke_content = state_data[3]    # wt%
        
        # 烧结效应 (Ostwald熟化)
        T_kelvin = temperature
        R = 8.314
        sintering_rate = np.exp(-self.params['Ea_sintering'] / (R * T_kelvin))
        
        # 粒子尺寸增长
        new_particle_size = self.spec.particle_size + \
                           self.params['particle_growth_rate'] * sintering_rate * dt / 3600
        
        # 积碳增长
        coking_rate = self.params['k_coking'] * pressure * np.exp(-50000/(R*T_kelvin))
        new_coke = coke_content + coking_rate * dt / 3600
        
        # 活性位点失活
        active_count = len(self.active_sites)
        deactivation = 0.0
        
        # 烧结导致的位点减少
        sites_lost_sintering = int(active_count * sintering_rate * dt / 3600 * 0.1)
        
        # 积碳覆盖导致的位点阻塞
        sites_blocked_coke = int(active_count * (new_coke / 10))  # 假设10%积碳完全覆盖
        
        # 更新位点
        for _ in range(sites_lost_sintering):
            if self.active_sites:
                # 优先失去高配位位点 (较不稳定)
                site_to_remove = max(self.active_sites.values(), 
                                    key=lambda s: s.coordination_number)
                del self.active_sites[site_to_remove.site_id]
        
        # 计算剩余活性
        remaining_sites = len(self.active_sites)
        activity_factor = remaining_sites / self.spec.initial_active_sites
        activity_factor *= (1 - sites_blocked_coke / max(remaining_sites, 1))
        
        # 计算转化率 (简化)
        conversion = self._calculate_conversion(temperature, pressure, activity_factor)
        
        new_data = np.array([
            temperature,
            pressure,
            time_on_stream + dt/3600,
            new_coke,
            conversion,
            activity_factor,
            remaining_sites
        ])
        
        return StateVector(
            timestamp=self._state.timestamp + dt,
            data=new_data,
            metadata={
                'step_type': 'catalyst',
                'active_sites': remaining_sites,
                'particle_size': new_particle_size
            }
        )
    
    def _calculate_conversion(self, T: float, P: float, 
                             activity: float) -> float:
        """计算转化率 (Arrhenius方程)"""
        R = 8.314
        Ea = 80000  # J/mol
        
        rate_constant = self.reaction_rate_constant * np.exp(-Ea / (R * T))
        conversion = 1 - np.exp(-rate_constant * P * activity)
        
        return float(np.clip(conversion, 0, 1))
    
    def get_active_site_distribution(self) -> Dict[str, int]:
        """获取活性位点类型分布"""
        distribution = {}
        for site in self.active_sites.values():
            site_type = site.site_type
            distribution[site_type] = distribution.get(site_type, 0) + 1
        return distribution
    
    def get_site_activity_map(self) -> NDArray[np.float64]:
        """获取位点活性分布图"""
        activities = [s.activity for s in self.active_sites.values()]
        return np.array(activities)


class CatalystDigitalTwin:
    """
    催化剂数字孪生系统
    
    整合催化剂物理模型、活性位点演化跟踪和失活预测
    """
    
    def __init__(self, spec: Optional[CatalystSpecification] = None,
                 name: str = "catalyst_twin"):
        self.spec = spec or CatalystSpecification()
        self.name = name
        
        # 核心数字孪生
        self.config = TwinConfiguration(
            name=name,
            physics_weight=0.7,
            data_weight=0.3,
            enable_adaptive=True
        )
        self.core = DigitalTwinCore(self.config)
        
        # 物理模型
        self.physics_model = CatalystPhysicsModel(self.spec)
        self.core.register_physics_model(self.physics_model)
        
        # 数据驱动模型 (活性预测)
        self.data_model = NeuralSurrogateModel(
            input_dim=7,   # [T, P, t, coke, conversion, activity, sites]
            output_dim=7
        )
        self.core.register_data_model(self.data_model)
        
        # 同步器
        sync_config = SyncConfiguration(
            direction=SyncDirection.BIDIRECTIONAL,
            mode=SyncMode.EVENT_DRIVEN
        )
        self.synchronizer = RealTimeSynchronizer(self.core, sync_config)
        
        # 预测引擎
        self.pm_engine = PredictiveMaintenanceEngine({
            'failure_threshold': 0.3,  # 活性降至30%视为失效
            'window_size': 50
        })
        
        # 注册失活模型
        self.pm_engine.register_degradation_model(
            'power_law', PowerLawDegradationModel(failure_threshold=0.3)
        )
        
        # 异常检测
        self.anomaly_system = AnomalyDetectionSystem()
        
        # 历史
        self._history: List[CatalystState] = []
        self._site_evolution: List[Dict[str, Any]] = []
        
    def initialize(self, condition: Optional[ReactionCondition] = None) -> None:
        """初始化催化剂数字孪生"""
        if condition is None:
            condition = ReactionCondition()
        
        state_vector = StateVector(
            timestamp=0.0,
            data=np.array([
                condition.temperature,
                condition.pressure,
                0.0,   # time on stream
                0.0,   # coke content
                0.0,   # conversion
                1.0,   # activity factor
                self.spec.initial_active_sites
            ]),
            metadata={'catalyst_init': True}
        )
        
        self.core.initialize(state_vector)
        
        # 记录初始状态
        self._record_state()
        
        print(f"Catalyst Digital Twin '{self.name}' initialized")
        print(f"  Type: {self.spec.catalyst_type.value}")
        print(f"  Active Metal: {self.spec.active_metal}/{self.spec.support}")
        print(f"  Initial Active Sites: {self.spec.initial_active_sites}")
    
    def simulate_operation(self, duration_hours: float,
                          condition: Optional[ReactionCondition] = None,
                          dt: float = 60.0) -> List[CatalystState]:
        """
        模拟催化剂运行
        
        Args:
            duration_hours: 运行时间 (小时)
            condition: 反应条件
            dt: 时间步长 (秒)
        """
        if condition:
            # 更新条件
            state = self.core.get_current_state()
            if state:
                state.data[0] = condition.temperature
                state.data[1] = condition.pressure
        
        n_steps = int(duration_hours * 3600 / dt)
        
        for i in range(n_steps):
            # 步进
            self.core.step(dt)
            
            # 记录状态
            if i % 10 == 0:
                self._record_state()
            
            # 更新健康评估
            state = self.core.get_current_state()
            if state:
                features = np.array([state.data[4], state.data[5], state.data[6]])
                self.pm_engine.update(features)
        
        return self._history
    
    def _record_state(self) -> None:
        """记录当前状态"""
        state = self.core.get_current_state()
        if state:
            cat_state = self._state_to_catalyst_state(state)
            self._history.append(cat_state)
            
            # 记录位点分布
            site_dist = self.physics_model.get_active_site_distribution()
            self._site_evolution.append({
                'timestamp': state.timestamp,
                'distribution': site_dist,
                'total_sites': len(self.physics_model.active_sites)
            })
    
    def _state_to_catalyst_state(self, state: StateVector) -> CatalystState:
        """转换StateVector到CatalystState"""
        return CatalystState(
            timestamp=state.timestamp,
            temperature=state.data[0],
            pressure=state.data[1],
            time_on_stream=state.data[2],
            coke_content=state.data[3],
            conversion=state.data[4],
            active_site_count=int(state.data[6])
        )
    
    def predict_activity(self, future_hours: float) -> Tuple[float, float]:
        """预测未来活性"""
        prediction = self.core.predict(
            steps=int(future_hours * 3600 / 60),
            dt=60.0
        )
        
        if prediction.states:
            final_state = prediction.states[-1]
            activity = final_state.data[5]
            return activity, self.config.data_weight
        
        return 0.0, 0.0
    
    def predict_deactivation(self, mechanism: DeactivationMechanism) -> Dict[str, Any]:
        """预测特定失活机理的影响"""
        if mechanism == DeactivationMechanism.SINTERING:
            # 烧结导致粒子长大，分散度下降
            current_sites = len(self.physics_model.active_sites)
            # 假设烧结导致粒子尺寸翻倍，表面积减半
            predicted_sites = int(current_sites * 0.5)
            
        elif mechanism == DeactivationMechanism.COKING:
            # 积碳增加
            current_coke = self.core.get_current_state().data[3] if self.core.get_current_state() else 0
            predicted_coke = min(current_coke * 2, 20)  # 最多20%
            predicted_sites = int(self.spec.initial_active_sites * (1 - predicted_coke/20))
            
        else:
            predicted_sites = len(self.physics_model.active_sites)
        
        return {
            'mechanism': mechanism.value,
            'predicted_active_sites': predicted_sites,
            'activity_retention': predicted_sites / self.spec.initial_active_sites
        }
    
    def get_site_evolution_analysis(self) -> Dict[str, Any]:
        """获取活性位点演化分析"""
        if not self._site_evolution:
            return {}
        
        # 分析位点类型变化
        initial = self._site_evolution[0]
        current = self._site_evolution[-1]
        
        analysis = {
            'initial_distribution': initial['distribution'],
            'current_distribution': current['distribution'],
            'total_loss': initial['total_sites'] - current['total_sites'],
            'loss_percentage': (initial['total_sites'] - current['total_sites']) / initial['total_sites'] * 100
        }
        
        # 计算每种位点的损失
        site_type_loss = {}
        for site_type in set(list(initial['distribution'].keys()) + list(current['distribution'].keys())):
            initial_count = initial['distribution'].get(site_type, 0)
            current_count = current['distribution'].get(site_type, 0)
            loss = initial_count - current_count
            site_type_loss[site_type] = {
                'initial': initial_count,
                'current': current_count,
                'loss': loss,
                'loss_pct': loss / initial_count * 100 if initial_count > 0 else 0
            }
        
        analysis['site_type_analysis'] = site_type_loss
        
        return analysis
    
    def recommend_regeneration(self) -> Optional[Dict[str, Any]]:
        """推荐再生策略"""
        current = self._history[-1] if self._history else None
        
        if current is None or current.coke_content < 2.0:
            return None
        
        # 根据积碳量推荐再生条件
        if current.coke_content > 10.0:
            method = "oxidative_burnoff"
            temperature = 773  # 500°C
            duration = 4  # hours
        elif current.coke_content > 5.0:
            method = "controlled_oxidation"
            temperature = 673  # 400°C
            duration = 2
        else:
            method = "mild_regeneration"
            temperature = 573  # 300°C
            duration = 1
        
        return {
            'recommended': True,
            'method': method,
            'temperature_k': temperature,
            'duration_hours': duration,
            'expected_recovery': 0.85 if method == "oxidative_burnoff" else 0.95,
            'coke_to_remove': current.coke_content
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """生成催化剂状态报告"""
        current = self._history[-1] if self._history else None
        
        site_analysis = self.get_site_evolution_analysis()
        regeneration = self.recommend_regeneration()
        
        return {
            'catalyst_info': {
                'name': self.name,
                'type': self.spec.catalyst_type.value,
                'composition': f"{self.spec.active_metal}/{self.spec.support}"
            },
            'current_state': asdict(current) if current else {},
            'site_evolution': site_analysis,
            'performance_trend': 'degrading' if current and current.conversion < 0.5 else 'stable',
            'regeneration_recommendation': regeneration,
            'rul_prediction': self.pm_engine.predict_rul().rul_cycles if current else 0
        }


def demo():
    """演示催化剂数字孪生"""
    print("=" * 70)
    print("⚗️ 催化剂数字孪生系统演示")
    print("=" * 70)
    
    # 创建催化剂规格
    spec = CatalystSpecification(
        name="Pt-Al2O3-Reformer",
        catalyst_type=CatalystType.HETEROGENEOUS,
        active_metal="Pt",
        support="Al2O3",
        surface_area=150.0,
        metal_loading=0.5,
        initial_active_sites=500
    )
    
    # 创建数字孪生
    print("\n1. 创建催化剂数字孪生")
    catalyst_twin = CatalystDigitalTwin(spec, name="Reformer_Catalyst_01")
    
    # 初始化
    condition = ReactionCondition(
        temperature=773.0,    # 500°C
        pressure=20.0,        # 20 bar
        flow_rate=200.0
    )
    catalyst_twin.initialize(condition)
    
    print(f"   初始活性位点: {spec.initial_active_sites}")
    print(f"   初始位点分布: {catalyst_twin.physics_model.get_active_site_distribution()}")
    
    # 模拟运行
    print("\n2. 模拟催化反应运行")
    print(f"   条件: {condition.temperature}K, {condition.pressure}bar")
    
    # 分段模拟不同时间
    time_segments = [24, 48, 72, 96]  # 小时
    
    for i, hours in enumerate(time_segments):
        print(f"\n   阶段 {i+1}: 运行 {hours} 小时")
        
        # 逐渐增加温度模拟工业过程
        condition.temperature += 5  # 升温补偿活性下降
        
        states = catalyst_twin.simulate_operation(
            duration_hours=hours,
            condition=condition
        )
        
        current = states[-1]
        print(f"   转化率: {current.conversion:.1%}")
        print(f"   剩余活性位点: {current.active_site_count}")
        print(f"   积碳含量: {current.coke_content:.2f} wt%")
        print(f"   平均粒径: {current.average_particle_size:.2f} nm")
    
    # 位点演化分析
    print("\n3. 活性位点演化分析")
    analysis = catalyst_twin.get_site_evolution_analysis()
    
    print(f"   总位点损失: {analysis['total_loss']} ({analysis['loss_percentage']:.1f}%)")
    print("   各类型位点变化:")
    
    for site_type, data in analysis['site_type_analysis'].items():
        print(f"      {site_type:10s}: {data['initial']:4d} → {data['current']:4d} "
              f"(损失 {data['loss_pct']:.1f}%)")
    
    # 失活预测
    print("\n4. 失活机理预测")
    
    for mechanism in [DeactivationMechanism.SINTERING, 
                      DeactivationMechanism.COKING]:
        pred = catalyst_twin.predict_deactivation(mechanism)
        print(f"   {mechanism.value:20s}: "
              f"预测活性保持率 = {pred['activity_retention']:.1%}")
    
    # 再生建议
    print("\n5. 催化剂再生建议")
    regen = catalyst_twin.recommend_regeneration()
    
    if regen:
        print(f"   建议再生: ✓")
        print(f"   方法: {regen['method']}")
        print(f"   条件: {regen['temperature_k']}K, {regen['duration_hours']}小时")
        print(f"   预期恢复: {regen['expected_recovery']:.1%}")
    else:
        print("   建议再生: ✗ (积碳量不足)")
    
    # 活性预测
    print("\n6. 未来活性预测")
    future_activity, confidence = catalyst_twin.predict_activity(future_hours=168)  # 1周
    print(f"   预测1周后活性: {future_activity:.1%} (置信度: {confidence:.1%})")
    
    # 生成报告
    print("\n7. 生成催化剂状态报告")
    report = catalyst_twin.generate_report()
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  催化剂数字孪生报告                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  催化剂名称: {report['catalyst_info']['name']:18s}              ║
    ║  类型:      {report['catalyst_info']['type']:18s}              ║
    ║  组成:      {report['catalyst_info']['composition']:18s}              ║
    ║  当前转化率: {report['current_state'].get('conversion', 0)*100:16.1f}%            ║
    ║  当前活性:  {report['current_state'].get('activity', 0)*100:17.1f}%            ║
    ║  趋势:      {report['performance_trend']:18s}              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)
    
    return catalyst_twin


if __name__ == "__main__":
    demo()
