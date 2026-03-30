"""
Sensor Fusion Module
传感器融合模块

提供多源数据集成、噪声过滤与校准、异常检测功能
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Union, Any
from collections import deque
from enum import Enum
import threading
import time
from abc import ABC, abstractmethod
import warnings


class SensorType(Enum):
    """传感器类型枚举"""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    STRAIN_GAUGE = "strain_gauge"
    ACCELEROMETER = "accelerometer"
    DISPLACEMENT = "displacement"
    ELECTROCHEMICAL = "electrochemical"
    OPTICAL = "optical"
    ACOUSTIC = "acoustic"
    THERMAL_CAMERA = "thermal_camera"
    XRAY = "xray"
    CUSTOM = "custom"


class DataQuality(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"      # > 0.95
    GOOD = "good"                # 0.85 - 0.95
    FAIR = "fair"                # 0.70 - 0.85
    POOR = "poor"                # 0.50 - 0.70
    UNRELIABLE = "unreliable"    # < 0.50


@dataclass
class SensorReading:
    """传感器读数数据类"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    value: np.ndarray
    unit: str = ""
    quality_score: float = 1.0
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.value, (int, float)):
            self.value = np.array([self.value])


@dataclass
class CalibrationParams:
    """校准参数数据类"""
    offset: np.ndarray = field(default_factory=lambda: np.zeros(1))
    scale: np.ndarray = field(default_factory=lambda: np.ones(1))
    linear_coeffs: Optional[np.ndarray] = None
    nonlinear_coeffs: Optional[np.ndarray] = None
    temperature_compensation: Optional[Callable] = None
    
    def apply(self, raw_value: np.ndarray, temperature: float = 25.0) -> np.ndarray:
        """应用校准"""
        calibrated = (raw_value - self.offset) * self.scale
        
        if self.linear_coeffs is not None:
            calibrated = np.polyval(self.linear_coeffs, calibrated)
        
        if self.temperature_compensation:
            calibrated = self.temperature_compensation(calibrated, temperature)
        
        return calibrated


class KalmanFilter:
    """卡尔曼滤波器"""
    
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # 状态
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)
        
        # 状态转移矩阵（假设恒速模型）
        self.F = np.eye(state_dim)
        if state_dim >= 2 * measurement_dim:
            for i in range(measurement_dim):
                self.F[i, i + measurement_dim] = 1.0
        
        # 观测矩阵
        self.H = np.zeros((measurement_dim, state_dim))
        self.H[:measurement_dim, :measurement_dim] = np.eye(measurement_dim)
        
        # 过程噪声
        self.Q = process_noise * np.eye(state_dim)
        
        # 观测噪声
        self.R = measurement_noise * np.eye(measurement_dim)
    
    def predict(self, dt: float = 1.0) -> np.ndarray:
        """预测步骤"""
        # 更新状态转移矩阵
        if self.state_dim >= 2 * self.measurement_dim:
            for i in range(self.measurement_dim):
                self.F[i, i + self.measurement_dim] = dt
        
        # 预测状态
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        
        return self.state[:self.measurement_dim]
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """更新步骤"""
        # 计算卡尔曼增益
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        
        # 更新协方差
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ self.R @ K.T
        
        return self.state[:self.measurement_dim]
    
    def filter(self, measurement: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """完整滤波步骤"""
        self.predict(dt)
        return self.update(measurement)
    
    def get_confidence(self) -> float:
        """获取置信度"""
        trace = np.trace(self.covariance[:self.measurement_dim, :self.measurement_dim])
        return 1.0 / (1.0 + trace)


class ParticleFilter:
    """粒子滤波器（用于非线性/非高斯系统）"""
    
    def __init__(
        self,
        state_dim: int,
        num_particles: int = 1000,
        resample_threshold: float = 0.5
    ):
        self.state_dim = state_dim
        self.num_particles = num_particles
        self.resample_threshold = resample_threshold
        
        # 初始化粒子
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
    
    def predict(
        self,
        transition_func: Callable,
        noise_std: float = 0.1
    ) -> None:
        """预测步骤"""
        for i in range(self.num_particles):
            self.particles[i] = transition_func(self.particles[i])
            self.particles[i] += np.random.randn(self.state_dim) * noise_std
    
    def update(
        self,
        measurement: np.ndarray,
        likelihood_func: Callable
    ) -> None:
        """更新步骤"""
        for i in range(self.num_particles):
            self.weights[i] *= likelihood_func(measurement, self.particles[i])
        
        # 归一化权重
        self.weights += 1e-300  # 避免除零
        self.weights /= np.sum(self.weights)
        
        # 重采样
        self._resample()
    
    def _resample(self) -> None:
        """系统重采样"""
        effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        
        if effective_sample_size < self.resample_threshold * self.num_particles:
            # 执行重采样
            indices = np.random.choice(
                self.num_particles,
                size=self.num_particles,
                p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate(self) -> Tuple[np.ndarray, float]:
        """获取估计状态和方差"""
        mean = np.average(self.particles, weights=self.weights, axis=0)
        variance = np.average(
            (self.particles - mean) ** 2,
            weights=self.weights,
            axis=0
        )
        return mean, np.sqrt(variance)


class AdaptiveNoiseFilter:
    """自适应噪声过滤器"""
    
    def __init__(
        self,
        window_size: int = 50,
        adaptation_rate: float = 0.1
    ):
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        self.history: deque = deque(maxlen=window_size)
        self.noise_estimate = 1.0
        self.trend = 0.0
    
    def filter(self, value: float) -> Tuple[float, float]:
        """
        自适应滤波
        
        Returns:
            (filtered_value, confidence)
        """
        self.history.append(value)
        
        if len(self.history) < 3:
            return value, 0.5
        
        data = np.array(self.history)
        
        # 估计噪声水平
        local_noise = np.std(data[-10:]) if len(data) >= 10 else np.std(data)
        self.noise_estimate = (
            (1 - self.adaptation_rate) * self.noise_estimate +
            self.adaptation_rate * local_noise
        )
        
        # 计算加权移动平均
        weights = np.exp(-0.1 * np.arange(len(data)))
        weights /= weights.sum()
        filtered = np.average(data, weights=weights[::-1])
        
        # 计算置信度
        confidence = 1.0 / (1.0 + self.noise_estimate)
        
        return filtered, confidence
    
    def detect_outlier(self, value: float, threshold: float = 3.0) -> bool:
        """检测异常值"""
        if len(self.history) < 5:
            return False
        
        data = np.array(self.history)
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        
        return abs(value - mean) > threshold * std


class MultiSensorFusion:
    """多传感器融合核心类"""
    
    def __init__(self):
        self.sensors: Dict[str, Dict] = {}
        self.filters: Dict[str, Any] = {}
        self.calibrations: Dict[str, CalibrationParams] = {}
        
        # 融合权重（动态调整）
        self.fusion_weights: Dict[str, float] = {}
        
        # 时间同步窗口
        self.sync_window = 0.1  # 100ms
        
        # 历史数据
        self.history: deque = deque(maxlen=10000)
        
        # 数据质量统计
        self.quality_stats: Dict[str, Dict] = {}
        
        # 线程锁
        self._lock = threading.RLock()
    
    def register_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        measurement_dim: int,
        filter_type: str = "kalman",
        calibration: Optional[CalibrationParams] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """注册传感器"""
        with self._lock:
            self.sensors[sensor_id] = {
                'type': sensor_type,
                'dim': measurement_dim,
                'last_reading': None,
                'last_timestamp': None,
                'metadata': metadata or {}
            }
            
            # 初始化滤波器
            if filter_type == "kalman":
                self.filters[sensor_id] = KalmanFilter(
                    state_dim=measurement_dim * 2,
                    measurement_dim=measurement_dim
                )
            elif filter_type == "particle":
                self.filters[sensor_id] = ParticleFilter(
                    state_dim=measurement_dim
                )
            elif filter_type == "adaptive":
                self.filters[sensor_id] = AdaptiveNoiseFilter()
            else:
                self.filters[sensor_id] = None
            
            # 设置校准
            self.calibrations[sensor_id] = calibration or CalibrationParams()
            
            # 初始化权重
            self.fusion_weights[sensor_id] = 1.0
            
            # 初始化质量统计
            self.quality_stats[sensor_id] = {
                'total_readings': 0,
                'valid_readings': 0,
                'outliers': 0,
                'quality_sum': 0.0
            }
    
    def process_reading(self, reading: SensorReading) -> Optional[SensorReading]:
        """处理传感器读数"""
        with self._lock:
            if reading.sensor_id not in self.sensors:
                warnings.warn(f"Unknown sensor: {reading.sensor_id}")
                return None
            
            sensor_info = self.sensors[reading.sensor_id]
            
            # 更新统计
            stats = self.quality_stats[reading.sensor_id]
            stats['total_readings'] += 1
            
            # 校准
            calibrated_value = self.calibrations[reading.sensor_id].apply(
                reading.value,
                reading.metadata.get('temperature', 25.0)
            )
            
            # 滤波
            filtered_value = calibrated_value
            confidence = reading.confidence
            
            filter_obj = self.filters.get(reading.sensor_id)
            if isinstance(filter_obj, KalmanFilter):
                filtered_value = filter_obj.filter(calibrated_value)
                confidence = filter_obj.get_confidence()
            elif isinstance(filter_obj, ParticleFilter):
                # 简化的粒子滤波更新
                def likelihood(m, p):
                    return np.exp(-0.5 * np.sum((m - p) ** 2))
                filter_obj.update(calibrated_value, likelihood)
                filtered_value, _ = filter_obj.estimate()
            elif isinstance(filter_obj, AdaptiveNoiseFilter):
                if len(calibrated_value) == 1:
                    filtered_value, confidence = filter_obj.filter(calibrated_value[0])
                    filtered_value = np.array([filtered_value])
            
            # 创建处理后读数
            processed = SensorReading(
                sensor_id=reading.sensor_id,
                sensor_type=reading.sensor_type,
                timestamp=reading.timestamp,
                value=filtered_value,
                unit=reading.unit,
                quality_score=reading.quality_score * confidence,
                confidence=confidence,
                metadata={
                    **reading.metadata,
                    'raw_value': reading.value,
                    'calibrated': True,
                    'filtered': True
                }
            )
            
            # 更新传感器状态
            sensor_info['last_reading'] = processed
            sensor_info['last_timestamp'] = reading.timestamp
            
            # 更新统计
            if processed.quality_score > 0.5:
                stats['valid_readings'] += 1
            stats['quality_sum'] += processed.quality_score
            
            # 存储历史
            self.history.append({
                'timestamp': reading.timestamp,
                'reading': processed
            })
            
            return processed
    
    def fuse_sensors(
        self,
        sensor_ids: Optional[List[str]] = None,
        time_window: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        融合多个传感器数据
        
        Returns:
            融合结果字典
        """
        with self._lock:
            if sensor_ids is None:
                sensor_ids = list(self.sensors.keys())
            
            window = time_window or self.sync_window
            current_time = time.time()
            
            # 收集同步窗口内的数据
            readings: Dict[str, SensorReading] = {}
            
            for sid in sensor_ids:
                sensor = self.sensors.get(sid)
                if sensor and sensor['last_reading']:
                    if current_time - sensor['last_timestamp'] <= window:
                        readings[sid] = sensor['last_reading']
            
            if not readings:
                return {'status': 'no_data', 'fused_value': None}
            
            # 动态调整权重
            self._update_fusion_weights(readings)
            
            # 执行融合
            fused_value, uncertainty = self._weighted_fusion(readings)
            
            # 评估数据质量
            quality = self._assess_quality(readings)
            
            return {
                'status': 'fused',
                'fused_value': fused_value,
                'uncertainty': uncertainty,
                'quality': quality,
                'num_sensors': len(readings),
                'sensor_weights': {
                    sid: self.fusion_weights.get(sid, 0)
                    for sid in readings.keys()
                },
                'timestamp': current_time
            }
    
    def _update_fusion_weights(
        self,
        readings: Dict[str, SensorReading]
    ) -> None:
        """动态更新融合权重"""
        total_quality = sum(
            r.quality_score * r.confidence
            for r in readings.values()
        )
        
        if total_quality > 0:
            for sid, reading in readings.items():
                quality_weight = (reading.quality_score * reading.confidence) / total_quality
                
                # 时间衰减
                age = time.time() - reading.timestamp
                time_weight = np.exp(-age / self.sync_window)
                
                # 综合权重
                self.fusion_weights[sid] = quality_weight * time_weight
    
    def _weighted_fusion(
        self,
        readings: Dict[str, SensorReading]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """加权融合"""
        # 对齐维度
        max_dim = max(r.value.shape[0] for r in readings.values())
        
        values = []
        weights = []
        
        for sid, reading in readings.items():
            # 填充或截断
            val = reading.value
            if val.shape[0] < max_dim:
                val = np.pad(val, (0, max_dim - val.shape[0]))
            elif val.shape[0] > max_dim:
                val = val[:max_dim]
            
            values.append(val)
            weights.append(self.fusion_weights.get(sid, 1.0))
        
        values = np.array(values)
        weights = np.array(weights)
        weights /= weights.sum()  # 归一化
        
        # 加权平均
        fused = np.average(values, axis=0, weights=weights)
        
        # 加权标准差作为不确定性
        uncertainty = np.sqrt(
            np.average((values - fused) ** 2, axis=0, weights=weights)
        )
        
        return fused, uncertainty
    
    def _assess_quality(self, readings: Dict[str, SensorReading]) -> DataQuality:
        """评估数据质量"""
        avg_quality = np.mean([r.quality_score for r in readings.values()])
        
        if avg_quality > 0.95:
            return DataQuality.EXCELLENT
        elif avg_quality > 0.85:
            return DataQuality.GOOD
        elif avg_quality > 0.70:
            return DataQuality.FAIR
        elif avg_quality > 0.50:
            return DataQuality.POOR
        else:
            return DataQuality.UNRELIABLE
    
    def calibrate_sensor(
        self,
        sensor_id: str,
        reference_values: np.ndarray,
        measured_values: np.ndarray,
        order: int = 1
    ) -> CalibrationParams:
        """
        校准传感器
        
        Args:
            sensor_id: 传感器ID
            reference_values: 参考真值
            measured_values: 测量值
            order: 拟合阶数
        
        Returns:
            校准参数
        """
        # 线性拟合
        if order == 1:
            coeffs = np.polyfit(measured_values, reference_values, 1)
            calib = CalibrationParams(
                offset=0,
                scale=1.0,
                linear_coeffs=coeffs
            )
        else:
            # 多项式拟合
            coeffs = np.polyfit(measured_values, reference_values, order)
            calib = CalibrationParams(
                nonlinear_coeffs=coeffs
            )
        
        self.calibrations[sensor_id] = calib
        return calib
    
    def get_sensor_stats(self, sensor_id: Optional[str] = None) -> Dict:
        """获取传感器统计信息"""
        with self._lock:
            if sensor_id:
                stats = self.quality_stats.get(sensor_id, {})
                if stats.get('total_readings', 0) > 0:
                    return {
                        'sensor_id': sensor_id,
                        'total_readings': stats['total_readings'],
                        'valid_ratio': stats['valid_readings'] / stats['total_readings'],
                        'average_quality': stats['quality_sum'] / stats['total_readings'],
                        'current_weight': self.fusion_weights.get(sensor_id, 0)
                    }
                return stats
            else:
                return {
                    sid: self.get_sensor_stats(sid)
                    for sid in self.sensors.keys()
                }


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(
        self,
        window_size: int = 100,
        sensitivity: float = 3.0,
        use_ml: bool = True
    ):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.use_ml = use_ml
        
        # 历史数据窗口
        self.data_window: deque = deque(maxlen=window_size)
        
        # 统计模型
        self.mean = None
        self.std = None
        self.covariance = None
        
        # ML模型（自编码器）
        if use_ml:
            self.autoencoder = None
            self.threshold = None
        
        # 异常历史
        self.anomaly_history: deque = deque(maxlen=1000)
        
        # 检测模式
        self.detection_modes = ['statistical', 'isolation_forest', 'autoencoder']
        self.active_mode = 'statistical'
    
    def fit(self, data: np.ndarray) -> None:
        """拟合检测模型"""
        self.data_window.clear()
        
        for sample in data:
            self.data_window.append(sample)
        
        data_array = np.array(self.data_window)
        
        # 统计模型
        self.mean = np.mean(data_array, axis=0)
        self.std = np.std(data_array, axis=0) + 1e-8
        
        if data_array.shape[1] > 1:
            self.covariance = np.cov(data_array.T)
        
        # 训练自编码器
        if self.use_ml and len(data_array) > 50:
            self._train_autoencoder(data_array)
    
    def _train_autoencoder(self, data: np.ndarray) -> None:
        """训练自编码器"""
        input_dim = data.shape[1]
        hidden_dim = max(input_dim // 2, 4)
        
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        self.autoencoder = Autoencoder(input_dim, hidden_dim)
        
        # 简单训练
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        self.autoencoder.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.autoencoder(data_tensor)
            loss = criterion(output, data_tensor)
            loss.backward()
            optimizer.step()
        
        # 设置阈值
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder(data_tensor)
            errors = torch.mean((reconstructed - data_tensor) ** 2, dim=1)
            self.threshold = torch.quantile(errors, 0.95).item()
    
    def detect(self, sample: np.ndarray) -> Dict[str, Any]:
        """
        检测异常
        
        Returns:
            检测结果字典
        """
        # 添加到窗口
        self.data_window.append(sample)
        
        result = {
            'is_anomaly': False,
            'score': 0.0,
            'method': self.active_mode,
            'confidence': 0.0
        }
        
        if len(self.data_window) < 10:
            return result
        
        if self.active_mode == 'statistical':
            result = self._statistical_detect(sample)
        elif self.active_mode == 'autoencoder' and self.autoencoder:
            result = self._autoencoder_detect(sample)
        
        # 记录异常
        if result['is_anomaly']:
            self.anomaly_history.append({
                'timestamp': time.time(),
                'sample': sample,
                'score': result['score']
            })
        
        return result
    
    def _statistical_detect(self, sample: np.ndarray) -> Dict[str, Any]:
        """统计方法检测"""
        if self.mean is None:
            return {'is_anomaly': False, 'score': 0.0, 'method': 'statistical'}
        
        # Z-score
        z_score = np.abs((sample - self.mean) / self.std)
        max_z = np.max(z_score)
        
        # Mahalanobis距离（多维）
        if self.covariance is not None and sample.shape[0] > 1:
            try:
                inv_cov = np.linalg.inv(self.covariance)
                diff = sample - self.mean
                mahal_dist = np.sqrt(diff @ inv_cov @ diff)
            except:
                mahal_dist = 0
        else:
            mahal_dist = 0
        
        score = max(max_z, mahal_dist)
        is_anomaly = score > self.sensitivity
        
        return {
            'is_anomaly': is_anomaly,
            'score': float(score),
            'z_score': float(max_z),
            'mahalanobis': float(mahal_dist),
            'method': 'statistical',
            'confidence': min(1.0, score / self.sensitivity)
        }
    
    def _autoencoder_detect(self, sample: np.ndarray) -> Dict[str, Any]:
        """自编码器检测"""
        if self.autoencoder is None or self.threshold is None:
            return {'is_anomaly': False, 'score': 0.0, 'method': 'autoencoder'}
        
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            reconstructed = self.autoencoder(sample_tensor)
            error = torch.mean((reconstructed - sample_tensor) ** 2).item()
        
        score = error / (self.threshold + 1e-8)
        is_anomaly = error > self.threshold
        
        return {
            'is_anomaly': is_anomaly,
            'score': float(score),
            'reconstruction_error': float(error),
            'threshold': self.threshold,
            'method': 'autoencoder',
            'confidence': min(1.0, score)
        }
    
    def get_anomaly_summary(self, time_window: Optional[float] = None) -> Dict:
        """获取异常摘要"""
        if not self.anomaly_history:
            return {'total_anomalies': 0}
        
        current_time = time.time()
        
        if time_window:
            recent_anomalies = [
                a for a in self.anomaly_history
                if current_time - a['timestamp'] <= time_window
            ]
        else:
            recent_anomalies = list(self.anomaly_history)
        
        if not recent_anomalies:
            return {'total_anomalies': 0, 'time_window': time_window}
        
        scores = [a['score'] for a in recent_anomalies]
        
        return {
            'total_anomalies': len(recent_anomalies),
            'time_window': time_window,
            'mean_score': np.mean(scores),
            'max_score': np.max(scores),
            'anomaly_rate': len(recent_anomalies) / max(len(self.data_window), 1),
            'last_anomaly_time': recent_anomalies[-1]['timestamp']
        }


class SensorNetwork:
    """传感器网络管理器"""
    
    def __init__(self):
        self.fusion = MultiSensorFusion()
        self.anomaly_detector = AnomalyDetector()
        
        # 注册的传感器
        self.registered_sensors: Dict[str, Dict] = {}
        
        # 数据回调
        self.data_callbacks: List[Callable] = []
        self.anomaly_callbacks: List[Callable] = []
        
        # 运行状态
        self.is_running = False
        self._lock = threading.RLock()
    
    def add_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        measurement_dim: int = 1,
        filter_type: str = "kalman",
        calibration: Optional[CalibrationParams] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """添加传感器到网络"""
        self.fusion.register_sensor(
            sensor_id, sensor_type, measurement_dim,
            filter_type, calibration, metadata
        )
        
        self.registered_sensors[sensor_id] = {
            'type': sensor_type,
            'added_at': time.time(),
            'status': 'active'
        }
    
    def ingest_data(self, reading: SensorReading) -> Optional[Dict]:
        """
        摄入传感器数据
        
        Returns:
            处理结果
        """
        with self._lock:
            # 处理读数
            processed = self.fusion.process_reading(reading)
            
            if processed is None:
                return None
            
            # 异常检测
            anomaly_result = self.anomaly_detector.detect(processed.value)
            
            # 回调
            for callback in self.data_callbacks:
                try:
                    callback(processed)
                except Exception as e:
                    warnings.warn(f"Data callback error: {e}")
            
            if anomaly_result['is_anomaly']:
                for callback in self.anomaly_callbacks:
                    try:
                        callback(processed, anomaly_result)
                    except Exception as e:
                        warnings.warn(f"Anomaly callback error: {e}")
            
            return {
                'processed': processed,
                'anomaly': anomaly_result,
                'fused': None
            }
    
    def get_fused_state(self, sensor_ids: Optional[List[str]] = None) -> Dict:
        """获取融合状态"""
        return self.fusion.fuse_sensors(sensor_ids)
    
    def register_data_callback(self, callback: Callable) -> None:
        """注册数据回调"""
        self.data_callbacks.append(callback)
    
    def register_anomaly_callback(self, callback: Callable) -> None:
        """注册异常回调"""
        self.anomaly_callbacks.append(callback)
    
    def get_network_status(self) -> Dict:
        """获取网络状态"""
        return {
            'num_sensors': len(self.registered_sensors),
            'sensors': self.registered_sensors,
            'fusion_stats': self.fusion.get_sensor_stats(),
            'anomaly_summary': self.anomaly_detector.get_anomaly_summary()
        }


# 便捷函数
def create_temperature_sensor(sensor_id: str, location: str = "") -> SensorReading:
    """创建温度传感器读数模板"""
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=SensorType.TEMPERATURE,
        timestamp=time.time(),
        value=np.array([25.0]),
        unit="Celsius",
        metadata={'location': location}
    )


def create_strain_sensor(sensor_id: str, direction: str = "axial") -> SensorReading:
    """创建应变传感器读数模板"""
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=SensorType.STRAIN_GAUGE,
        timestamp=time.time(),
        value=np.array([0.0]),
        unit="microstrain",
        metadata={'direction': direction}
    )


def create_battery_sensor(sensor_id: str, metric: str = "voltage") -> SensorReading:
    """创建电池传感器读数模板"""
    sensor_type = SensorType.ELECTROCHEMICAL
    value = np.array([3.7])
    unit = "V"
    
    if metric == "current":
        value = np.array([0.0])
        unit = "A"
    elif metric == "temperature":
        sensor_type = SensorType.TEMPERATURE
        value = np.array([25.0])
        unit = "C"
    
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        timestamp=time.time(),
        value=value,
        unit=unit,
        metadata={'metric': metric}
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Sensor Fusion Module Test")
    print("=" * 60)
    
    # 测试多传感器融合
    print("\n1. Testing Multi-Sensor Fusion")
    fusion = MultiSensorFusion()
    
    # 注册传感器
    fusion.register_sensor(
        "temp_01", SensorType.TEMPERATURE, 1, "kalman"
    )
    fusion.register_sensor(
        "strain_01", SensorType.STRAIN_GAUGE, 1, "adaptive"
    )
    fusion.register_sensor(
        "pressure_01", SensorType.PRESSURE, 1, "kalman"
    )
    
    print(f"  Registered {len(fusion.sensors)} sensors")
    
    # 模拟数据
    print("\n2. Simulating Sensor Data")
    np.random.seed(42)
    for i in range(20):
        # 温度传感器
        temp_reading = SensorReading(
            sensor_id="temp_01",
            sensor_type=SensorType.TEMPERATURE,
            timestamp=time.time() + i * 0.1,
            value=np.array([25.0 + np.random.randn() * 0.5]),
            unit="Celsius",
            quality_score=0.95
        )
        fusion.process_reading(temp_reading)
        
        # 应变传感器
        strain_reading = SensorReading(
            sensor_id="strain_01",
            sensor_type=SensorType.STRAIN_GAUGE,
            timestamp=time.time() + i * 0.1,
            value=np.array([100.0 + np.random.randn() * 10]),
            unit="microstrain",
            quality_score=0.90
        )
        fusion.process_reading(strain_reading)
    
    print(f"  Processed 40 readings")
    
    # 融合结果
    print("\n3. Fusion Result")
    result = fusion.fuse_sensors()
    print(f"  Status: {result['status']}")
    print(f"  Fused value shape: {result['fused_value'].shape if result['fused_value'] is not None else None}")
    print(f"  Data quality: {result['quality']}")
    print(f"  Num sensors: {result['num_sensors']}")
    
    # 异常检测测试
    print("\n4. Testing Anomaly Detection")
    detector = AnomalyDetector(window_size=50)
    
    # 生成正常数据
    normal_data = np.random.randn(100, 2) * 0.5 + np.array([1.0, 2.0])
    detector.fit(normal_data)
    print(f"  Fitted detector with {len(normal_data)} samples")
    
    # 测试正常样本
    normal_sample = np.array([1.1, 2.1])
    result = detector.detect(normal_sample)
    print(f"  Normal sample: anomaly={result['is_anomaly']}, score={result['score']:.4f}")
    
    # 测试异常样本
    anomaly_sample = np.array([5.0, 6.0])
    result = detector.detect(anomaly_sample)
    print(f"  Anomaly sample: anomaly={result['is_anomaly']}, score={result['score']:.4f}")
    
    # 传感器网络测试
    print("\n5. Testing Sensor Network")
    network = SensorNetwork()
    
    network.add_sensor("battery_voltage", SensorType.ELECTROCHEMICAL, 1)
    network.add_sensor("battery_current", SensorType.ELECTROCHEMICAL, 1)
    network.add_sensor("battery_temp", SensorType.TEMPERATURE, 1)
    
    # 模拟电池数据
    for i in range(10):
        voltage = SensorReading(
            sensor_id="battery_voltage",
            sensor_type=SensorType.ELECTROCHEMICAL,
            timestamp=time.time(),
            value=np.array([3.7 + np.random.randn() * 0.05]),
            unit="V"
        )
        network.ingest_data(voltage)
    
    status = network.get_network_status()
    print(f"  Network sensors: {status['num_sensors']}")
    print(f"  Anomaly summary: {status['anomaly_summary']}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
