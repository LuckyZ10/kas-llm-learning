"""
异常检测与预警 (Anomaly Detection & Alert)

实现材料系统的实时异常检测和预警系统。
"""

from __future__ import annotations

import json
import time
import threading
import asyncio
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Protocol, Set,
    Tuple, TypeVar, Union, Iterator
)
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


try:
    from ..digital_twin.twin_core import StateVector, Observation, TwinState
    from ..digital_twin.predictive_model import HealthIndicator, FailureMode
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from digital_twin.twin_core import StateVector, Observation, TwinState
    from digital_twin.predictive_model import HealthIndicator, FailureMode


class AlertLevel(Enum):
    """预警等级"""
    INFO = ("info", "ℹ️", 0)
    LOW = ("low", "🔍", 1)
    MEDIUM = ("medium", "⚠️", 2)
    HIGH = ("high", "🚨", 3)
    CRITICAL = ("critical", "🔥", 4)
    
    def __init__(self, code: str, icon: str, priority: int):
        self.code = code
        self.icon = icon
        self.priority = priority


class AnomalyType(Enum):
    """异常类型"""
    TEMPERATURE_SPIKE = "temperature_spike"
    PRESSURE_DROP = "pressure_drop"
    VIBRATION_ABNORMAL = "vibration_abnormal"
    STRUCTURAL_DEFORMATION = "structural_deformation"
    CAPACITY_DEGRADATION = "capacity_degradation"
    UNEXPECTED_BEHAVIOR = "unexpected_behavior"
    SENSOR_FAULT = "sensor_fault"
    COMMUNICATION_ERROR = "communication_error"


@dataclass
class Alert:
    """预警信息"""
    alert_id: str
    timestamp: float
    level: AlertLevel
    anomaly_type: AnomalyType
    title: str
    description: str
    source: str
    related_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp,
            'level': self.level.code,
            'anomaly_type': self.anomaly_type.value,
            'title': self.title,
            'description': self.description,
            'source': self.source,
            'related_metrics': self.related_metrics,
            'recommendations': self.recommendations,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }


@dataclass
class AnomalyScore:
    """异常评分"""
    timestamp: float
    overall_score: float  # 0-1
    component_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    
    @property
    def is_anomaly(self) -> bool:
        return self.overall_score > 0.7


class AnomalyDetector(ABC):
    """异常检测器基类"""
    
    def __init__(self, name: str, threshold: float = 0.7):
        self.name = name
        self.threshold = threshold
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, normal_data: NDArray[np.float64]) -> None:
        """使用正常数据训练"""
        pass
    
    @abstractmethod
    def detect(self, data: NDArray[np.float64]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        检测异常
        
        Returns:
            (是否异常, 异常分数, 详细信息)
        """
        pass


class StatisticalDetector(AnomalyDetector):
    """
    基于统计的异常检测器
    
    使用均值、标准差和分布特性检测异常
    """
    
    def __init__(self, name: str = "statistical", threshold: float = 3.0):
        super().__init__(name, threshold)
        self.mean: Optional[NDArray[np.float64]] = None
        self.std: Optional[NDArray[np.float64]] = None
        self.cov: Optional[NDArray[np.float64]] = None
        
    def fit(self, normal_data: NDArray[np.float64]) -> None:
        """拟合统计模型"""
        self.mean = np.mean(normal_data, axis=0)
        self.std = np.std(normal_data, axis=0) + 1e-10
        
        if normal_data.shape[1] < 10:
            self.cov = np.cov(normal_data.T)
        
        self._is_fitted = True
    
    def detect(self, data: NDArray[np.float64]) -> Tuple[bool, float, Dict[str, Any]]:
        """检测统计异常"""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted")
        
        # Z-score检测
        z_scores = np.abs((data - self.mean) / self.std)
        max_z = np.max(z_scores)
        
        # Mahalanobis距离 (如果协方差可计算)
        mahal_distance = 0.0
        if self.cov is not None:
            try:
                cov_inv = np.linalg.inv(self.cov)
                diff = data - self.mean
                mahal_distance = float(np.sqrt(diff @ cov_inv @ diff))
            except np.linalg.LinAlgError:
                pass
        
        # 综合异常分数
        anomaly_score = min(1.0, max_z / self.threshold)
        
        is_anomaly = max_z > self.threshold or mahal_distance > self.threshold
        
        details = {
            'max_z_score': float(max_z),
            'mahalanobis_distance': mahal_distance,
            'mean_deviation': float(np.mean(z_scores))
        }
        
        return is_anomaly, anomaly_score, details


class IsolationForestDetector(AnomalyDetector):
    """
    隔离森林异常检测器
    
    基于随机划分的异常检测算法
    """
    
    def __init__(self, name: str = "isolation_forest", 
                 n_trees: int = 100, 
                 sample_size: int = 256,
                 threshold: float = 0.6):
        super().__init__(name, threshold)
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees: List[Dict] = []
        self._height_limit = int(np.ceil(np.log2(sample_size)))
    
    def fit(self, normal_data: NDArray[np.float64]) -> None:
        """训练隔离森林"""
        n_samples = len(normal_data)
        
        for _ in range(self.n_trees):
            # 随机采样
            indices = np.random.choice(n_samples, min(self.sample_size, n_samples), replace=False)
            sample = normal_data[indices]
            
            # 构建树
            tree = self._build_tree(sample, 0)
            self.trees.append(tree)
        
        self._is_fitted = True
    
    def _build_tree(self, data: NDArray[np.float64], current_height: int) -> Dict:
        """递归构建隔离树"""
        n_samples = len(data)
        
        if current_height >= self._height_limit or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}
        
        # 随机选择特征和分割点
        feature = np.random.randint(0, data.shape[1])
        min_val, max_val = np.min(data[:, feature]), np.max(data[:, feature])
        split_val = np.random.uniform(min_val, max_val)
        
        # 分割数据
        left_mask = data[:, feature] < split_val
        right_mask = ~left_mask
        
        left_tree = self._build_tree(data[left_mask], current_height + 1)
        right_tree = self._build_tree(data[right_mask], current_height + 1)
        
        return {
            'type': 'internal',
            'feature': feature,
            'split': split_val,
            'left': left_tree,
            'right': right_tree
        }
    
    def _path_length(self, x: NDArray[np.float64], tree: Dict, current_height: int) -> float:
        """计算路径长度"""
        if tree['type'] == 'leaf':
            return current_height + self._c(tree['size'])
        
        if x[tree['feature']] < tree['split']:
            return self._path_length(x, tree['left'], current_height + 1)
        else:
            return self._path_length(x, tree['right'], current_height + 1)
    
    def _c(self, n: int) -> float:
        """平均路径长度修正"""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    
    def detect(self, data: NDArray[np.float64]) -> Tuple[bool, float, Dict[str, Any]]:
        """检测异常"""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted")
        
        # 计算平均路径长度
        path_lengths = []
        for tree in self.trees:
            path_lengths.append(self._path_length(data, tree, 0))
        
        avg_path_length = np.mean(path_lengths)
        
        # 异常分数
        anomaly_score = 2 ** (-avg_path_length / self._c(self.sample_size))
        
        is_anomaly = anomaly_score > self.threshold
        
        details = {
            'avg_path_length': float(avg_path_length),
            'n_trees': len(self.trees)
        }
        
        return is_anomaly, anomaly_score, details


class AutoencoderDetector(AnomalyDetector):
    """
    自编码器异常检测器
    
    使用神经网络学习正常数据的压缩表示
    """
    
    def __init__(self, name: str = "autoencoder",
                 encoding_dim: int = 8,
                 threshold: float = 0.1):
        super().__init__(name, threshold)
        self.encoding_dim = encoding_dim
        self.weights: Dict[str, NDArray[np.float64]] = {}
        self.reconstruction_threshold = threshold
    
    def fit(self, normal_data: NDArray[np.float64], 
            epochs: int = 100, 
            learning_rate: float = 0.001) -> None:
        """训练自编码器"""
        input_dim = normal_data.shape[1]
        
        # 初始化权重 (简化版)
        self.weights['W1'] = np.random.randn(input_dim, self.encoding_dim) * 0.01
        self.weights['b1'] = np.zeros(self.encoding_dim)
        self.weights['W2'] = np.random.randn(self.encoding_dim, input_dim) * 0.01
        self.weights['b2'] = np.zeros(input_dim)
        
        # 简化的训练 (使用PCA近似)
        # 实际应用中应该使用反向传播
        from numpy.linalg import svd
        U, s, Vt = svd(normal_data - np.mean(normal_data, axis=0))
        self.weights['W1'] = Vt[:self.encoding_dim].T
        self.weights['W2'] = Vt[:self.encoding_dim]
        
        # 计算重构误差阈值
        reconstructed = self._reconstruct(normal_data)
        errors = np.mean((normal_data - reconstructed)**2, axis=1)
        self.reconstruction_threshold = np.percentile(errors, 95) * 1.5
        
        self._is_fitted = True
    
    def _encode(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """编码"""
        return np.tanh(x @ self.weights['W1'] + self.weights['b1'])
    
    def _decode(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """解码"""
        return z @ self.weights['W2'] + self.weights['b2']
    
    def _reconstruct(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """重构"""
        z = self._encode(x)
        return self._decode(z)
    
    def detect(self, data: NDArray[np.float64]) -> Tuple[bool, float, Dict[str, Any]]:
        """检测异常"""
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted")
        
        reconstructed = self._reconstruct(data)
        reconstruction_error = float(np.mean((data - reconstructed)**2))
        
        anomaly_score = min(1.0, reconstruction_error / self.reconstruction_threshold)
        is_anomaly = reconstruction_error > self.reconstruction_threshold
        
        details = {
            'reconstruction_error': reconstruction_error,
            'threshold': self.reconstruction_threshold
        }
        
        return is_anomaly, anomaly_score, details


class AlertManager:
    """
    预警管理器
    
    管理预警的生成、分发和生命周期
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alerts: deque[Alert] = deque(maxlen=1000)
        self.active_alerts: Dict[str, Alert] = {}
        self._callbacks: List[Callable[[Alert], None]] = []
        self._rules: List[Dict[str, Any]] = []
        
        # 默认规则
        self._add_default_rules()
    
    def _add_default_rules(self) -> None:
        """添加默认预警规则"""
        self._rules.append({
            'name': 'critical_health',
            'condition': lambda data: data.get('health', 1.0) < 0.3,
            'level': AlertLevel.CRITICAL,
            'type': AnomalyType.CAPACITY_DEGRADATION,
            'title': 'Critical Health Level',
            'description': 'System health has dropped below critical threshold'
        })
        
        self._rules.append({
            'name': 'high_temperature',
            'condition': lambda data: data.get('temperature', 0) > 350,
            'level': AlertLevel.HIGH,
            'type': AnomalyType.TEMPERATURE_SPIKE,
            'title': 'High Temperature Warning',
            'description': 'Temperature exceeds safe operating limits'
        })
        
        self._rules.append({
            'name': 'rapid_degradation',
            'condition': lambda data: data.get('degradation_rate', 0) > 0.1,
            'level': AlertLevel.MEDIUM,
            'type': AnomalyType.CAPACITY_DEGRADATION,
            'title': 'Rapid Degradation Detected',
            'description': 'System is degrading faster than expected'
        })
    
    def add_rule(self, rule: Dict[str, Any]) -> None:
        """添加预警规则"""
        self._rules.append(rule)
    
    def on_alert(self, callback: Callable[[Alert], None]) -> None:
        """注册预警回调"""
        self._callbacks.append(callback)
    
    def evaluate(self, data: Dict[str, Any]) -> List[Alert]:
        """评估数据并生成预警"""
        triggered = []
        
        for rule in self._rules:
            try:
                if rule['condition'](data):
                    alert = self._create_alert(rule, data)
                    
                    # 检查是否重复
                    if alert.alert_id not in self.active_alerts:
                        self.active_alerts[alert.alert_id] = alert
                        self.alerts.append(alert)
                        triggered.append(alert)
                        
                        # 触发回调
                        for callback in self._callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                print(f"Alert callback error: {e}")
            except Exception as e:
                print(f"Rule evaluation error: {e}")
        
        return triggered
    
    def _create_alert(self, rule: Dict[str, Any], data: Dict[str, Any]) -> Alert:
        """创建预警"""
        alert_id = f"{rule['name']}_{int(time.time())}"
        
        recommendations = self._generate_recommendations(rule, data)
        
        return Alert(
            alert_id=alert_id,
            timestamp=time.time(),
            level=rule['level'],
            anomaly_type=rule['type'],
            title=rule['title'],
            description=rule['description'],
            source='alert_manager',
            related_metrics={k: v for k, v in data.items() if isinstance(v, (int, float))},
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, rule: Dict[str, Any], 
                                  data: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if rule['name'] == 'critical_health':
            recommendations = [
                '立即检查系统状态',
                '准备维护计划',
                '考虑降载运行',
                '联系技术支持'
            ]
        elif rule['name'] == 'high_temperature':
            recommendations = [
                '检查冷却系统',
                '降低负载',
                '增加通风',
                '监控温度趋势'
            ]
        elif rule['name'] == 'rapid_degradation':
            recommendations = [
                '分析退化原因',
                '调整操作参数',
                '增加监测频率',
                '准备备用方案'
            ]
        
        return recommendations
    
    def acknowledge(self, alert_id: str) -> bool:
        """确认预警"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            return True
        return False
    
    def resolve(self, alert_id: str) -> bool:
        """解决预警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            del self.active_alerts[alert_id]
            return True
        return False
    
    def get_active_alerts(self, min_level: Optional[AlertLevel] = None) -> List[Alert]:
        """获取活动预警"""
        alerts = list(self.active_alerts.values())
        
        if min_level:
            alerts = [a for a in alerts if a.level.priority >= min_level.priority]
        
        return sorted(alerts, key=lambda a: a.level.priority, reverse=True)
    
    def get_alert_history(self, n: int = 100) -> List[Alert]:
        """获取预警历史"""
        return list(self.alerts)[-n:]


class AnomalyDetectionSystem:
    """
    异常检测系统
    
    整合多种检测器和预警管理
    """
    
    def __init__(self):
        self.detectors: Dict[str, AnomalyDetector] = {}
        self.alert_manager = AlertManager()
        self._data_buffer: deque[NDArray[np.float64]] = deque(maxlen=1000)
        self._scores_buffer: deque[AnomalyScore] = deque(maxlen=1000)
        
        # 统计
        self._total_detections = 0
        self._true_positives = 0
    
    def add_detector(self, detector: AnomalyDetector) -> None:
        """添加检测器"""
        self.detectors[detector.name] = detector
    
    def train(self, normal_data: NDArray[np.float64]) -> None:
        """训练所有检测器"""
        for detector in self.detectors.values():
            print(f"Training {detector.name}...")
            detector.fit(normal_data)
    
    def detect(self, data: NDArray[np.float64]) -> AnomalyScore:
        """执行异常检测"""
        self._data_buffer.append(data)
        
        # 集成多个检测器的结果
        scores = {}
        is_anomaly_votes = []
        
        for name, detector in self.detectors.items():
            try:
                is_anom, score, details = detector.detect(data)
                scores[name] = score
                is_anomaly_votes.append(is_anom)
            except Exception as e:
                print(f"Detector {name} failed: {e}")
                scores[name] = 0.0
        
        # 综合评分 (平均)
        overall_score = np.mean(list(scores.values())) if scores else 0.0
        
        # 多数投票
        is_anomaly = sum(is_anomaly_votes) > len(is_anomaly_votes) / 2
        
        score = AnomalyScore(
            timestamp=time.time(),
            overall_score=overall_score,
            component_scores=scores,
            confidence=1.0 - np.std(list(scores.values())) if scores else 0.0
        )
        
        self._scores_buffer.append(score)
        self._total_detections += 1
        
        # 触发预警
        if is_anomaly:
            self._true_positives += 1
            self.alert_manager.evaluate({
                'anomaly_score': overall_score,
                'detector_scores': scores,
                'timestamp': time.time()
            })
        
        return score
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_detections': self._total_detections,
            'anomalies_detected': self._true_positives,
            'detection_rate': self._true_positives / max(1, self._total_detections),
            'active_alerts': len(self.alert_manager.active_alerts),
            'detectors': list(self.detectors.keys())
        }


class RealTimeMonitor:
    """
    实时监控器
    
    持续监控数据流并触发检测
    """
    
    def __init__(self, detection_system: AnomalyDetectionSystem,
                 check_interval: float = 1.0):
        self.detection_system = detection_system
        self.check_interval = check_interval
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._data_queue: deque[NDArray[np.float64]] = deque(maxlen=100)
        self._on_anomaly_callbacks: List[Callable[[AnomalyScore], None]] = []
    
    def on_anomaly(self, callback: Callable[[AnomalyScore], None]) -> None:
        """注册异常回调"""
        self._on_anomaly_callbacks.append(callback)
    
    def feed_data(self, data: NDArray[np.float64]) -> None:
        """输入数据"""
        self._data_queue.append(data)
    
    def start(self) -> None:
        """启动监控"""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        print("Real-time monitor started")
    
    def stop(self) -> None:
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        print("Real-time monitor stopped")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            if self._data_queue:
                data = self._data_queue.popleft()
                score = self.detection_system.detect(data)
                
                if score.is_anomaly:
                    for callback in self._on_anomaly_callbacks:
                        try:
                            callback(score)
                        except Exception as e:
                            print(f"Anomaly callback error: {e}")
            
            time.sleep(self.check_interval)


def demo():
    """演示异常检测功能"""
    print("=" * 60)
    print("异常检测与预警演示")
    print("=" * 60)
    
    # 创建检测系统
    system = AnomalyDetectionSystem()
    
    # 添加检测器
    system.add_detector(StatisticalDetector(threshold=3.0))
    system.add_detector(IsolationForestDetector(n_trees=50, threshold=0.6))
    system.add_detector(AutoencoderDetector(encoding_dim=4, threshold=0.1))
    
    print("\n1. 训练检测器")
    
    # 生成正常训练数据
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # 正常数据 (多元高斯)
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    normal_data = np.random.multivariate_normal(mean, cov, n_samples)
    
    system.train(normal_data)
    print(f"   使用 {n_samples} 个样本训练完成")
    
    # 测试数据
    print("\n2. 检测测试")
    
    # 正常测试数据
    print("\n   a) 正常数据检测")
    normal_test = np.random.multivariate_normal(mean, cov, 10)
    for i, data in enumerate(normal_test):
        score = system.detect(data)
        status = "⚠️ 异常" if score.is_anomaly else "✓ 正常"
        print(f"      样本 {i+1}: 分数={score.overall_score:.3f} {status}")
    
    # 异常测试数据 (偏移)
    print("\n   b) 异常数据检测 (均值偏移)")
    anomaly_data = np.random.multivariate_normal(mean + 5, cov, 10)
    for i, data in enumerate(anomaly_data):
        score = system.detect(data)
        status = "⚠️ 异常" if score.is_anomaly else "✓ 正常"
        print(f"      样本 {i+1}: 分数={score.overall_score:.3f} {status}")
    
    # 预警管理
    print("\n3. 预警管理")
    
    # 模拟触发预警
    alert_data = {
        'health': 0.25,
        'temperature': 380,
        'degradation_rate': 0.15
    }
    
    triggered = system.alert_manager.evaluate(alert_data)
    print(f"   触发预警数: {len(triggered)}")
    
    for alert in triggered:
        print(f"\n   [{alert.level.icon}] {alert.title}")
        print(f"      类型: {alert.anomaly_type.value}")
        print(f"      描述: {alert.description}")
        print(f"      建议:")
        for rec in alert.recommendations[:3]:
            print(f"         - {rec}")
    
    # 查看活动预警
    active = system.alert_manager.get_active_alerts()
    print(f"\n   活动预警总数: {len(active)}")
    
    # 统计
    print("\n4. 检测统计")
    stats = system.get_statistics()
    print(f"   总检测次数: {stats['total_detections']}")
    print(f"   异常检测数: {stats['anomalies_detected']}")
    print(f"   检测率: {stats['detection_rate']:.2%}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return system


if __name__ == "__main__":
    demo()
