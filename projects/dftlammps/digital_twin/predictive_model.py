"""
预测性维护与寿命预测 (Predictive Maintenance & Life Prediction)

实现材料系统的预测性维护、剩余使用寿命(RUL)预测和故障预警。
"""

from __future__ import annotations

import json
import pickle
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Protocol, Set,
    Tuple, TypeVar, Union, runtime_checkable, Iterator
)
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial import polynomial as P

warnings.filterwarnings('ignore')


try:
    from .twin_core import StateVector, Observation, Prediction
except ImportError:
    from twin_core import StateVector, Observation, Prediction


T = TypeVar('T')


class DegradationMode(Enum):
    """退化模式"""
    FATIGUE = "fatigue"                 # 疲劳
    CORROSION = "corrosion"             # 腐蚀
    WEAR = "wear"                       # 磨损
    THERMAL = "thermal"                 # 热退化
    ELECTROCHEMICAL = "electrochemical"  # 电化学退化
    CREEP = "creep"                     # 蠕变
    UNKNOWN = "unknown"


class FailureMode(Enum):
    """故障模式"""
    FRACTURE = "fracture"               # 断裂
    YIELDING = "yielding"               # 屈服
    BUCKLING = "buckling"               # 屈曲
    DELAMINATION = "delamination"       # 分层
    SHORT_CIRCUIT = "short_circuit"     # 短路
    CAPACITY_LOSS = "capacity_loss"     # 容量损失
    CATALYST_DEACTIVATION = "catalyst_deactivation"  # 催化剂失活


class HealthLevel(Enum):
    """健康等级"""
    EXCELLENT = (0, 0.9, "green")
    GOOD = (0.7, 0.9, "lightgreen")
    FAIR = (0.5, 0.7, "yellow")
    DEGRADED = (0.3, 0.5, "orange")
    CRITICAL = (0.1, 0.3, "red")
    FAILED = (0.0, 0.1, "black")
    
    def __init__(self, lower: float, upper: float, color: str):
        self.lower = lower
        self.upper = upper
        self.color = color
    
    @classmethod
    def from_health_index(cls, hi: float) -> HealthLevel:
        """从健康指数获取等级"""
        for level in cls:
            if level.lower <= hi < level.upper:
                return level
        return cls.FAILED if hi < 0.1 else cls.EXCELLENT


@dataclass
class HealthIndicator:
    """健康指标"""
    timestamp: float
    value: float  # 0-1，1表示完全健康
    confidence: float = 1.0
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    
    @property
    def level(self) -> HealthLevel:
        return HealthLevel.from_health_index(self.value)


@dataclass
class RULPrediction:
    """剩余使用寿命预测结果"""
    timestamp: float
    rul_cycles: float           # 剩余循环次数
    rul_time_hours: float       # 剩余时间(小时)
    confidence_interval: Tuple[float, float]  # 置信区间
    prediction_method: str
    degradation_trend: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        if not isinstance(self.degradation_trend, np.ndarray):
            self.degradation_trend = np.array(self.degradation_trend, dtype=np.float64)


@dataclass
class MaintenanceRecommendation:
    """维护建议"""
    timestamp: float
    urgency: str  # 'immediate', 'soon', 'scheduled', 'none'
    action_type: str
    description: str
    estimated_cost: float
    estimated_downtime_hours: float
    recommended_window: Tuple[float, float]  # 建议维护时间窗口


@dataclass
class FaultPrediction:
    """故障预测"""
    timestamp: float
    failure_mode: FailureMode
    probability: float
    expected_time: float
    severity: str  # 'critical', 'high', 'medium', 'low'
    contributing_features: Dict[str, float] = field(default_factory=dict)


class FeatureExtractor:
    """
    特征提取器
    
    从历史数据中提取退化相关的特征
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._buffer: deque[NDArray[np.float64]] = deque(maxlen=window_size)
        
    def add_sample(self, sample: NDArray[np.float64]) -> None:
        """添加样本"""
        self._buffer.append(sample)
    
    def extract_features(self) -> Dict[str, float]:
        """提取时域和频域特征"""
        if len(self._buffer) < 10:
            return {}
        
        data = np.array(list(self._buffer))
        
        features = {}
        
        # 时域特征
        for i in range(data.shape[1] if len(data.shape) > 1 else 1):
            if len(data.shape) > 1:
                channel = data[:, i]
            else:
                channel = data
            
            prefix = f"ch{i}_" if len(data.shape) > 1 else ""
            
            features[f"{prefix}mean"] = float(np.mean(channel))
            features[f"{prefix}std"] = float(np.std(channel))
            features[f"{prefix}rms"] = float(np.sqrt(np.mean(channel**2)))
            features[f"{prefix}peak"] = float(np.max(np.abs(channel)))
            features[f"{prefix}crest_factor"] = features[f"{prefix}peak"] / (features[f"{prefix}rms"] + 1e-10)
            features[f"{prefix}skewness"] = float(np.mean((channel - np.mean(channel))**3) / (np.std(channel)**3 + 1e-10))
            features[f"{prefix}kurtosis"] = float(np.mean((channel - np.mean(channel))**4) / (np.std(channel)**4 + 1e-10))
            
            # 趋势特征
            if len(channel) > 1:
                x = np.arange(len(channel))
                slope, _ = np.polyfit(x, channel, 1)
                features[f"{prefix}trend_slope"] = float(slope)
        
        # 频域特征 (FFT)
        if len(data) >= 32:
            for i in range(data.shape[1] if len(data.shape) > 1 else 1):
                if len(data.shape) > 1:
                    channel = data[:, i]
                else:
                    channel = data
                
                prefix = f"ch{i}_" if len(data.shape) > 1 else ""
                
                fft = np.fft.fft(channel)
                psd = np.abs(fft)**2
                freqs = np.fft.fftfreq(len(channel))
                
                # 频带能量
                features[f"{prefix}spectral_centroid"] = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))
                features[f"{prefix}spectral_bandwidth"] = float(np.sqrt(np.sum((freqs - features[f"{prefix}spectral_centroid"])**2 * psd) / (np.sum(psd) + 1e-10)))
        
        return features


class DegradationModel(ABC):
    """
    退化模型基类
    
    用于建模材料或系统的退化过程
    """
    
    def __init__(self, failure_threshold: float = 0.2):
        self.failure_threshold = failure_threshold
        self._is_fitted = False
        self._params: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, cycles: NDArray[np.float64], 
            health_indices: NDArray[np.float64]) -> None:
        """拟合退化模型"""
        pass
    
    @abstractmethod
    def predict_health(self, cycles: NDArray[np.float64]) -> NDArray[np.float64]:
        """预测健康指数"""
        pass
    
    def predict_rul(self, current_cycle: float, 
                    current_health: float) -> float:
        """预测剩余寿命 (需子类实现或使用数值方法)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        # 数值搜索RUL
        for future_cycle in range(int(current_cycle), int(current_cycle) + 10000):
            health = self.predict_health(np.array([future_cycle]))[0]
            if health <= self.failure_threshold:
                return future_cycle - current_cycle
        
        return float('inf')


class ExponentialDegradationModel(DegradationModel):
    """
    指数退化模型
    
    H(t) = exp(-λ * t^β)
    
    其中H是健康指数，t是时间/循环次数
    """
    
    def __init__(self, failure_threshold: float = 0.2):
        super().__init__(failure_threshold)
        self.lambda_param = 0.001
        self.beta = 1.0
    
    def fit(self, cycles: NDArray[np.float64], 
            health_indices: NDArray[np.float64]) -> None:
        """拟合指数模型参数"""
        # 对数线性化: ln(-ln(H)) = ln(λ) + β * ln(t)
        valid_idx = (health_indices > 0) & (health_indices < 1) & (cycles > 0)
        
        if np.sum(valid_idx) < 2:
            raise ValueError("Insufficient valid data points")
        
        y = np.log(-np.log(health_indices[valid_idx]))
        x = np.log(cycles[valid_idx])
        
        # 线性回归
        A = np.vstack([x, np.ones(len(x))]).T
        beta, ln_lambda = np.linalg.lstsq(A, y, rcond=None)[0]
        
        self.beta = beta
        self.lambda_param = np.exp(ln_lambda)
        self._is_fitted = True
        
        self._params = {'lambda': self.lambda_param, 'beta': self.beta}
    
    def predict_health(self, cycles: NDArray[np.float64]) -> NDArray[np.float64]:
        """预测健康指数"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        return np.exp(-self.lambda_param * cycles**self.beta)


class PowerLawDegradationModel(DegradationModel):
    """
    幂律退化模型
    
    H(t) = 1 - a * t^b
    
    适用于疲劳、磨损等退化过程
    """
    
    def __init__(self, failure_threshold: float = 0.2):
        super().__init__(failure_threshold)
        self.a = 0.001
        self.b = 1.0
    
    def fit(self, cycles: NDArray[np.float64], 
            health_indices: NDArray[np.float64]) -> None:
        """拟合幂律模型"""
        # 变形: ln(1-H) = ln(a) + b * ln(t)
        valid_idx = (cycles > 0) & (health_indices < 1)
        
        if np.sum(valid_idx) < 2:
            raise ValueError("Insufficient valid data points")
        
        y = np.log(1 - health_indices[valid_idx])
        x = np.log(cycles[valid_idx])
        
        A = np.vstack([x, np.ones(len(x))]).T
        b, ln_a = np.linalg.lstsq(A, y, rcond=None)[0]
        
        self.b = b
        self.a = np.exp(ln_a)
        self._is_fitted = True
        
        self._params = {'a': self.a, 'b': self.b}
    
    def predict_health(self, cycles: NDArray[np.float64]) -> NDArray[np.float64]:
        """预测健康指数"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        health = 1 - self.a * cycles**self.b
        return np.clip(health, 0, 1)


class ParisLawModel(DegradationModel):
    """
    Paris定律模型 (疲劳裂纹扩展)
    
    da/dN = C * (ΔK)^m
    
    用于金属疲劳寿命预测
    """
    
    def __init__(self, failure_threshold: float = 0.2):
        super().__init__(failure_threshold)
        self.C = 1e-12
        self.m = 3.0
        self.initial_crack_size = 0.001
        self.critical_crack_size = 0.01
    
    def fit(self, cycles: NDArray[np.float64], 
            crack_sizes: NDArray[np.float64]) -> None:
        """拟合Paris定律参数"""
        # 计算裂纹扩展速率
        da_dN = np.diff(crack_sizes) / np.diff(cycles)
        a_mid = (crack_sizes[:-1] + crack_sizes[1:]) / 2
        
        # 假设ΔK ∝ sqrt(a)
        delta_K = np.sqrt(a_mid)
        
        # 对数线性回归
        log_da_dN = np.log(da_dN[da_dN > 0])
        log_delta_K = np.log(delta_K[da_dN > 0])
        
        A = np.vstack([log_delta_K, np.ones(len(log_delta_K))]).T
        m, log_C = np.linalg.lstsq(A, log_da_dN, rcond=None)[0]
        
        self.m = m
        self.C = np.exp(log_C)
        self._is_fitted = True
    
    def predict_health(self, cycles: NDArray[np.float64]) -> NDArray[np.float64]:
        """预测健康指数 (基于裂纹尺寸归一化)"""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        # 数值积分求解裂纹扩展
        crack_sizes = []
        a = self.initial_crack_size
        
        for _ in cycles:
            delta_K = np.sqrt(a)
            da_dN = self.C * delta_K**self.m
            a += da_dN
            crack_sizes.append(a)
            
            if a >= self.critical_crack_size:
                break
        
        # 转换为健康指数
        crack_sizes = np.array(crack_sizes)
        health = 1 - (crack_sizes - self.initial_crack_size) / \
                     (self.critical_crack_size - self.initial_crack_size)
        
        return np.clip(health, 0, 1)


class ParticleFilterRUL:
    """
    粒子滤波RUL预测
    
    使用粒子滤波进行非线性、非高斯退化过程的状态估计
    """
    
    def __init__(self, n_particles: int = 1000):
        self.n_particles = n_particles
        self.particles: Optional[NDArray[np.float64]] = None
        self.weights: Optional[NDArray[np.float64]] = None
        self._degradation_model: Optional[DegradationModel] = None
        
    def initialize(self, initial_health: float, initial_std: float = 0.05) -> None:
        """初始化粒子"""
        self.particles = np.random.normal(initial_health, initial_std, self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def set_degradation_model(self, model: DegradationModel) -> None:
        """设置退化模型"""
        self._degradation_model = model
    
    def predict(self, dt: float = 1.0) -> None:
        """预测步骤"""
        if self.particles is None:
            raise RuntimeError("Particles not initialized")
        
        # 应用退化模型
        if self._degradation_model:
            # 简化的退化传播
            degradation_rate = 0.001
            noise = np.random.normal(0, 0.01, self.n_particles)
            self.particles -= degradation_rate * dt + noise
            self.particles = np.clip(self.particles, 0, 1)
        
    def update(self, observation: float, measurement_noise: float = 0.05) -> None:
        """更新步骤"""
        if self.particles is None or self.weights is None:
            raise RuntimeError("Particles not initialized")
        
        # 计算似然
        likelihood = np.exp(-0.5 * ((self.particles - observation) / measurement_noise)**2)
        likelihood += 1e-300  # 避免除零
        
        # 更新权重
        self.weights *= likelihood
        self.weights /= np.sum(self.weights)
        
        # 重采样
        self._resample()
    
    def _resample(self) -> None:
        """系统重采样"""
        if self.weights is None or self.particles is None:
            return
        
        # 有效粒子数
        n_eff = 1.0 / np.sum(self.weights**2)
        
        if n_eff < self.n_particles / 2:
            # 执行重采样
            indices = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def estimate_rul(self, failure_threshold: float = 0.2) -> Tuple[float, float]:
        """估计RUL"""
        if self.particles is None:
            return 0.0, 0.0
        
        # 对每个粒子预测到失效
        rul_samples = []
        
        for particle in self.particles:
            rul = 0
            health = particle
            
            while health > failure_threshold and rul < 10000:
                health -= 0.001  # 简化的退化
                rul += 1
            
            rul_samples.append(rul)
        
        rul_mean = np.mean(rul_samples)
        rul_std = np.std(rul_samples)
        
        return rul_mean, rul_std


class PredictiveMaintenanceEngine:
    """
    预测性维护引擎
    
    整合健康评估、RUL预测和维护决策
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 组件
        self.feature_extractor = FeatureExtractor(
            window_size=self.config.get('window_size', 100)
        )
        self.degradation_models: Dict[str, DegradationModel] = {}
        self.particle_filter = ParticleFilterRUL(
            n_particles=self.config.get('n_particles', 1000)
        )
        
        # 状态
        self._health_history: deque[HealthIndicator] = deque(maxlen=10000)
        self._rul_history: deque[RULPrediction] = deque(maxlen=1000)
        self._fault_predictions: deque[FaultPrediction] = deque(maxlen=1000)
        
        # 参数
        self.failure_threshold = self.config.get('failure_threshold', 0.2)
        self.warning_threshold = self.config.get('warning_threshold', 0.5)
        
    def register_degradation_model(self, name: str, 
                                    model: DegradationModel) -> None:
        """注册退化模型"""
        self.degradation_models[name] = model
    
    def update(self, measurement: NDArray[np.float64], 
               cycle: Optional[float] = None) -> HealthIndicator:
        """
        更新健康状态
        
        接收新的测量数据，更新健康评估
        """
        # 特征提取
        self.feature_extractor.add_sample(measurement)
        features = self.feature_extractor.extract_features()
        
        # 计算健康指数
        health_value = self._calculate_health_index(features)
        
        # 确定贡献因子
        contributing = {}
        if 'trend_slope' in features:
            if features['trend_slope'] < 0:
                contributing['negative_trend'] = abs(features['trend_slope'])
        if 'rms' in features:
            if features['rms'] > 1.0:
                contributing['high_vibration'] = features['rms']
        
        indicator = HealthIndicator(
            timestamp=time.time(),
            value=health_value,
            confidence=0.9,
            contributing_factors=contributing
        )
        
        self._health_history.append(indicator)
        
        return indicator
    
    def _calculate_health_index(self, features: Dict[str, float]) -> float:
        """计算健康指数"""
        # 基于特征的综合健康评估
        health = 1.0
        
        # 趋势惩罚
        if 'trend_slope' in features:
            if features['trend_slope'] < -0.01:
                health -= abs(features['trend_slope']) * 10
        
        # RMS惩罚
        if 'rms' in features:
            if features['rms'] > 0.5:
                health -= (features['rms'] - 0.5) * 0.2
        
        # 峰度惩罚 (异常检测)
        if 'kurtosis' in features:
            if features['kurtosis'] > 3.0:
                health -= (features['kurtosis'] - 3.0) * 0.05
        
        return float(np.clip(health, 0, 1))
    
    def predict_rul(self, method: str = "particle_filter") -> RULPrediction:
        """
        预测剩余使用寿命
        """
        timestamp = time.time()
        
        if len(self._health_history) < 10:
            return RULPrediction(
                timestamp=timestamp,
                rul_cycles=0,
                rul_time_hours=0,
                confidence_interval=(0, 0),
                prediction_method="insufficient_data"
            )
        
        # 获取历史健康指数
        health_values = np.array([h.value for h in self._health_history])
        cycles = np.arange(len(health_values))
        
        if method == "particle_filter":
            rul, uncertainty = self._predict_rul_particle_filter()
        elif method == "exponential":
            rul, uncertainty = self._predict_rul_model(ExponentialDegradationModel())
        elif method == "power_law":
            rul, uncertainty = self._predict_rul_model(PowerLawDegradationModel())
        else:
            rul, uncertainty = self._predict_rul_linear()
        
        prediction = RULPrediction(
            timestamp=timestamp,
            rul_cycles=rul,
            rul_time_hours=rul / 24.0,  # 假设每天一个循环
            confidence_interval=(max(0, rul - uncertainty), rul + uncertainty),
            prediction_method=method,
            degradation_trend=health_values[-50:]
        )
        
        self._rul_history.append(prediction)
        
        return prediction
    
    def _predict_rul_particle_filter(self) -> Tuple[float, float]:
        """使用粒子滤波预测RUL"""
        current_health = self._health_history[-1].value
        
        # 初始化或更新粒子滤波
        if self.particle_filter.particles is None:
            self.particle_filter.initialize(current_health)
        
        self.particle_filter.predict()
        self.particle_filter.update(current_health)
        
        rul_mean, rul_std = self.particle_filter.estimate_rul(self.failure_threshold)
        
        return rul_mean, 2 * rul_std
    
    def _predict_rul_model(self, model: DegradationModel) -> Tuple[float, float]:
        """使用退化模型预测RUL"""
        health_values = np.array([h.value for h in self._health_history])
        cycles = np.arange(len(health_values))
        
        try:
            model.fit(cycles, health_values)
            current_cycle = len(health_values)
            current_health = health_values[-1]
            
            rul = model.predict_rul(current_cycle, current_health)
            
            # 估计不确定性
            uncertainty = rul * 0.2  # 20% 不确定性
            
            return rul, uncertainty
        except Exception as e:
            print(f"Model prediction failed: {e}")
            return 0, 0
    
    def _predict_rul_linear(self) -> Tuple[float, float]:
        """使用线性外推预测RUL"""
        health_values = np.array([h.value for h in self._health_history])
        cycles = np.arange(len(health_values))
        
        # 拟合线性趋势
        if len(health_values) >= 2:
            slope, intercept = np.polyfit(cycles[-20:], health_values[-20:], 1)
            
            if slope >= 0:
                return float('inf'), float('inf')
            
            # 外推到失效阈值
            rul = (self.failure_threshold - intercept) / slope - len(health_values)
            
            # 估计不确定性
            residuals = health_values[-20:] - (slope * cycles[-20:] + intercept)
            std_residual = np.std(residuals)
            uncertainty = abs(3 * std_residual / slope)
            
            return max(0, rul), uncertainty
        
        return 0, 0
    
    def predict_faults(self) -> List[FaultPrediction]:
        """预测潜在故障"""
        predictions = []
        
        if len(self._health_history) < 5:
            return predictions
        
        current_health = self._health_history[-1]
        
        # 基于健康水平预测故障
        if current_health.level in [HealthLevel.DEGRADED, HealthLevel.CRITICAL]:
            rul = self.predict_rul()
            
            fault = FaultPrediction(
                timestamp=time.time(),
                failure_mode=FailureMode.CAPACITY_LOSS,
                probability=1 - current_health.value,
                expected_time=rul.rul_time_hours,
                severity='high' if current_health.level == HealthLevel.CRITICAL else 'medium',
                contributing_features=current_health.contributing_factors
            )
            
            predictions.append(fault)
            self._fault_predictions.append(fault)
        
        return predictions
    
    def recommend_maintenance(self) -> Optional[MaintenanceRecommendation]:
        """
        生成维护建议
        """
        if len(self._health_history) == 0:
            return None
        
        current_health = self._health_history[-1]
        rul = self.predict_rul() if len(self._rul_history) > 0 else None
        
        # 确定紧急程度
        if current_health.level == HealthLevel.CRITICAL:
            urgency = 'immediate'
            action = '紧急停机检修'
            cost = 50000
            downtime = 48
        elif current_health.level == HealthLevel.DEGRADED:
            urgency = 'soon'
            action = '计划维护'
            cost = 15000
            downtime = 24
        elif current_health.level == HealthLevel.FAIR:
            urgency = 'scheduled'
            action = '预防性检查'
            cost = 5000
            downtime = 8
        else:
            return None
        
        # 计算建议时间窗口
        if rul:
            recommended_start = time.time() + rul.rul_time_hours * 3600 * 0.3
            recommended_end = time.time() + rul.rul_time_hours * 3600 * 0.8
        else:
            recommended_start = time.time() + 7 * 24 * 3600
            recommended_end = time.time() + 30 * 24 * 3600
        
        return MaintenanceRecommendation(
            timestamp=time.time(),
            urgency=urgency,
            action_type=action,
            description=f"基于健康等级 {current_health.level.name} 的维护建议",
            estimated_cost=cost,
            estimated_downtime_hours=downtime,
            recommended_window=(recommended_start, recommended_end)
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取完整健康状态"""
        current = self._health_history[-1] if self._health_history else None
        rul = self._rul_history[-1] if self._rul_history else None
        
        return {
            'current_health': {
                'value': current.value if current else None,
                'level': current.level.name if current else None,
                'confidence': current.confidence if current else None,
            },
            'rul': {
                'cycles': rul.rul_cycles if rul else None,
                'hours': rul.rul_time_hours if rul else None,
                'confidence_interval': rul.confidence_interval if rul else None,
            },
            'trend': 'degrading' if current and current.value < 0.8 else 'stable',
            'recommendation': self.recommend_maintenance()
        }


def demo():
    """演示预测性维护功能"""
    print("=" * 60)
    print("预测性维护与寿命预测演示")
    print("=" * 60)
    
    # 创建引擎
    engine = PredictiveMaintenanceEngine({
        'window_size': 50,
        'failure_threshold': 0.2,
        'warning_threshold': 0.5
    })
    
    # 模拟退化数据
    print("\n1. 模拟系统退化过程")
    np.random.seed(42)
    
    n_cycles = 500
    true_health = np.ones(n_cycles)
    
    # 生成退化曲线 (电池容量衰减)
    for i in range(1, n_cycles):
        # 指数衰减 + 噪声
        true_health[i] = true_health[i-1] * 0.998 + np.random.normal(0, 0.005)
    
    true_health = np.clip(true_health, 0, 1)
    
    # 注入测量数据
    print(f"   模拟 {n_cycles} 个循环...")
    for i in range(n_cycles):
        measurement = np.array([true_health[i], 
                               true_health[i] * 0.9 + np.random.normal(0, 0.02),
                               true_health[i] * 1.1 + np.random.normal(0, 0.02)])
        engine.update(measurement, cycle=i)
        
        if i % 100 == 0:
            health = engine._health_history[-1]
            print(f"   Cycle {i}: Health = {health.value:.3f} ({health.level.name})")
    
    # RUL预测
    print(f"\n2. 剩余使用寿命预测")
    
    for method in ['linear', 'exponential', 'particle_filter']:
        try:
            rul = engine.predict_rul(method=method)
            print(f"   {method:20s}: RUL = {rul.rul_cycles:.1f} cycles "
                  f"(置信区间: [{rul.confidence_interval[0]:.1f}, {rul.confidence_interval[1]:.1f}])")
        except Exception as e:
            print(f"   {method:20s}: 预测失败 ({e})")
    
    # 故障预测
    print(f"\n3. 故障预测")
    faults = engine.predict_faults()
    if faults:
        for fault in faults:
            print(f"   故障模式: {fault.failure_mode.value}")
            print(f"   概率: {fault.probability:.2%}")
            print(f"   预计时间: {fault.expected_time:.1f} hours")
            print(f"   严重度: {fault.severity}")
    else:
        print("   暂无故障预测")
    
    # 维护建议
    print(f"\n4. 维护建议")
    recommendation = engine.recommend_maintenance()
    if recommendation:
        print(f"   紧急程度: {recommendation.urgency}")
        print(f"   行动类型: {recommendation.action_type}")
        print(f"   估计成本: ${recommendation.estimated_cost:,.0f}")
        print(f"   估计停机时间: {recommendation.estimated_downtime_hours} hours")
    else:
        print("   无需维护")
    
    # 完整状态报告
    print(f"\n5. 健康状态总结")
    status = engine.get_health_status()
    print(f"   当前健康: {status['current_health']['value']:.3f}")
    print(f"   健康等级: {status['current_health']['level']}")
    print(f"   趋势: {status['trend']}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return engine


if __name__ == "__main__":
    demo()
