"""
实时同步机制 (Real-Time Synchronization)

实现实验数据与模拟数据的双向映射和实时同步。
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Protocol, Set,
    Tuple, TypeVar, Union, runtime_checkable, Iterator, AsyncIterator
)
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    from .twin_core import (
        StateVector, Observation, DigitalTwinCore, TwinConfiguration
    )
except ImportError:
    from twin_core import (
        StateVector, Observation, DigitalTwinCore, TwinConfiguration
    )


T = TypeVar('T')


class SyncDirection(Enum):
    """同步方向"""
    EXP_TO_SIM = "exp_to_sim"      # 实验 -> 模拟
    SIM_TO_EXP = "sim_to_exp"      # 模拟 -> 实验
    BIDIRECTIONAL = "bidirectional"  # 双向同步


class SyncMode(Enum):
    """同步模式"""
    REALTIME = "realtime"          # 实时同步
    BATCH = "batch"                # 批量同步
    EVENT_DRIVEN = "event_driven"  # 事件驱动
    SCHEDULED = "scheduled"        # 定时同步


class DataQuality(Enum):
    """数据质量等级"""
    EXCELLENT = 1.0
    GOOD = 0.8
    FAIR = 0.6
    POOR = 0.4
    UNUSABLE = 0.0


@dataclass
class SyncConfiguration:
    """同步配置"""
    # 基本设置
    direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    mode: SyncMode = SyncMode.REALTIME
    
    # 时序设置
    sample_rate_hz: float = 100.0           # 采样率
    sync_interval_ms: float = 10.0          # 同步间隔
    max_latency_ms: float = 100.0           # 最大允许延迟
    
    # 数据质量设置
    min_quality_threshold: float = 0.5      # 最小质量阈值
    outlier_detection: bool = True          # 异常值检测
    filtering_enabled: bool = True          # 滤波启用
    
    # 缓冲设置
    buffer_size: int = 1000                 # 缓冲区大小
    max_backlog: int = 10000                # 最大积压
    
    # 容错设置
    retry_attempts: int = 3                 # 重试次数
    timeout_ms: float = 5000.0              # 超时时间
    
    # 自适应设置
    adaptive_rate: bool = True              # 自适应速率
    congestion_control: bool = True         # 拥塞控制


@dataclass
class DataMapping:
    """数据映射规则"""
    source_field: str
    target_field: str
    transform: Optional[Callable[[Any], Any]] = None
    scale: float = 1.0
    offset: float = 0.0
    
    def apply(self, value: Any) -> Any:
        """应用映射变换"""
        if self.transform:
            value = self.transform(value)
        return value * self.scale + self.offset


@dataclass
class SyncRecord:
    """同步记录"""
    timestamp: float
    direction: SyncDirection
    data_id: str
    source_data: Any
    target_data: Any
    latency_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SyncMetrics:
    """同步指标"""
    total_syncs: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    data_dropped: int = 0
    buffer_utilization: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_syncs == 0:
            return 0.0
        return self.successful_syncs / self.total_syncs
    
    @property
    def average_latency_ms(self) -> float:
        if self.total_syncs == 0:
            return 0.0
        return self.total_latency_ms / self.total_syncs


class DataTransformer:
    """
    数据转换器
    
    处理实验数据与模拟数据之间的格式转换和单位映射
    """
    
    def __init__(self):
        self.mappings: Dict[str, DataMapping] = {}
        self._calibration_params: Dict[str, Any] = {}
    
    def add_mapping(self, name: str, mapping: DataMapping) -> None:
        """添加映射规则"""
        self.mappings[name] = mapping
    
    def transform(self, data: Dict[str, Any], direction: SyncDirection) -> Dict[str, Any]:
        """执行数据转换"""
        result = {}
        
        for key, value in data.items():
            if key in self.mappings:
                mapping = self.mappings[key]
                
                if direction == SyncDirection.EXP_TO_SIM:
                    result[mapping.target_field] = mapping.apply(value)
                else:
                    result[mapping.source_field] = self._inverse_apply(mapping, value)
            else:
                result[key] = value
        
        return result
    
    def _inverse_apply(self, mapping: DataMapping, value: Any) -> Any:
        """逆向应用映射"""
        result = (value - mapping.offset) / mapping.scale
        return result
    
    def calibrate(self, exp_data: List[Any], sim_data: List[Any]) -> None:
        """
        校准映射参数
        
        使用线性回归校准scale和offset
        """
        exp_array = np.array(exp_data)
        sim_array = np.array(sim_data)
        
        # 简单线性回归
        A = np.vstack([exp_array, np.ones(len(exp_array))]).T
        scale, offset = np.linalg.lstsq(A, sim_array, rcond=None)[0]
        
        self._calibration_params['scale'] = float(scale)
        self._calibration_params['offset'] = float(offset)


class QualityFilter:
    """
    数据质量过滤器
    
    检测和处理异常值、缺失值和噪声
    """
    
    def __init__(self, config: SyncConfiguration):
        self.config = config
        self._history: deque[NDArray[np.float64]] = deque(maxlen=100)
        self._outlier_threshold = 3.0  # 标准差倍数
    
    def filter(self, data: NDArray[np.float64]) -> Tuple[NDArray[np.float64], DataQuality]:
        """过滤数据并评估质量"""
        # 检查NaN和Inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return np.zeros_like(data), DataQuality.UNUSABLE
        
        # 异常值检测
        if self.config.outlier_detection and len(self._history) > 10:
            data = self._remove_outliers(data)
        
        # 平滑滤波
        if self.config.filtering_enabled:
            data = self._apply_filter(data)
        
        # 质量评估
        quality = self._assess_quality(data)
        
        self._history.append(data)
        
        return data, quality
    
    def _remove_outliers(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """移除异常值"""
        history_array = np.array(list(self._history))
        mean = np.mean(history_array, axis=0)
        std = np.std(history_array, axis=0)
        
        # 标记异常值
        outliers = np.abs(data - mean) > self._outlier_threshold * std
        
        # 用历史均值替换异常值
        result = data.copy()
        result[outliers] = mean[outliers]
        
        return result
    
    def _apply_filter(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """应用低通滤波"""
        if len(self._history) < 3:
            return data
        
        # 简单的移动平均
        history_array = np.array(list(self._history)[-3:])
        weights = np.array([0.2, 0.3, 0.5])
        smoothed = np.average(np.vstack([history_array, data]), 
                             axis=0, 
                             weights=np.concatenate([[0]*3, [1]]) * weights[-1])
        
        return 0.7 * data + 0.3 * smoothed
    
    def _assess_quality(self, data: NDArray[np.float64]) -> DataQuality:
        """评估数据质量"""
        if len(self._history) < 5:
            return DataQuality.GOOD
        
        # 计算信号稳定性
        history_array = np.array(list(self._history)[-10:])
        variance = np.var(history_array, axis=0)
        mean_variance = np.mean(variance)
        
        # 根据方差评估质量
        if mean_variance < 0.01:
            return DataQuality.EXCELLENT
        elif mean_variance < 0.1:
            return DataQuality.GOOD
        elif mean_variance < 1.0:
            return DataQuality.FAIR
        elif mean_variance < 10.0:
            return DataQuality.POOR
        else:
            return DataQuality.UNUSABLE


class ExperimentDataSource:
    """
    实验数据源
    
    抽象接口，支持多种实验数据采集方式
    """
    
    def __init__(self, source_id: str, config: SyncConfiguration):
        self.source_id = source_id
        self.config = config
        self._connected = False
        self._buffer: queue.Queue[Observation] = queue.Queue(maxsize=config.buffer_size)
        self._callbacks: List[Callable[[Observation], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def connect(self) -> bool:
        """连接数据源"""
        self._connected = True
        return True
    
    def disconnect(self) -> None:
        """断开数据源"""
        self._connected = False
        self._running = False
    
    def on_data(self, callback: Callable[[Observation], None]) -> None:
        """注册数据回调"""
        self._callbacks.append(callback)
    
    def start_acquisition(self) -> None:
        """开始数据采集"""
        self._running = True
        self._thread = threading.Thread(target=self._acquisition_loop)
        self._thread.daemon = True
        self._thread.start()
    
    def stop_acquisition(self) -> None:
        """停止数据采集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _acquisition_loop(self) -> None:
        """采集循环 (在子线程中运行)"""
        while self._running:
            try:
                data = self._read_data()
                if data is not None:
                    observation = self._convert_to_observation(data)
                    
                    # 放入缓冲区
                    try:
                        self._buffer.put_nowait(observation)
                    except queue.Full:
                        pass
                    
                    # 触发回调
                    for callback in self._callbacks:
                        try:
                            callback(observation)
                        except Exception as e:
                            print(f"Callback error: {e}")
                
                time.sleep(1.0 / self.config.sample_rate_hz)
            except Exception as e:
                print(f"Acquisition error: {e}")
    
    def _read_data(self) -> Optional[Dict[str, Any]]:
        """读取原始数据 (子类实现)"""
        return None
    
    def _convert_to_observation(self, data: Dict[str, Any]) -> Observation:
        """转换为观测对象 (子类实现)"""
        return Observation(
            timestamp=time.time(),
            sensor_id=self.source_id,
            values=np.array([])
        )
    
    def get_observation(self, timeout: float = 1.0) -> Optional[Observation]:
        """获取观测数据"""
        try:
            return self._buffer.get(timeout=timeout)
        except queue.Empty:
            return None


class SimulatedExperimentSource(ExperimentDataSource):
    """
    模拟实验数据源
    
    用于测试和演示
    """
    
    def __init__(self, source_id: str, config: SyncConfiguration, 
                 signal_type: str = "sine", noise_level: float = 0.1):
        super().__init__(source_id, config)
        self.signal_type = signal_type
        self.noise_level = noise_level
        self._counter = 0
        
    def _read_data(self) -> Optional[Dict[str, Any]]:
        """生成模拟数据"""
        self._counter += 1
        t = self._counter / self.config.sample_rate_hz
        
        if self.signal_type == "sine":
            value = np.sin(2 * np.pi * 1.0 * t) + np.random.randn() * self.noise_level
        elif self.signal_type == "random_walk":
            value = np.random.randn() * self.noise_level + self._counter * 0.001
        else:
            value = np.random.randn() * self.noise_level
        
        return {
            'timestamp': time.time(),
            'value': value,
            'temperature': 300 + 10 * np.sin(t),
            'pressure': 1.0 + 0.1 * np.cos(t),
        }
    
    def _convert_to_observation(self, data: Dict[str, Any]) -> Observation:
        """转换为观测"""
        values = np.array([
            data['value'],
            data['temperature'],
            data['pressure']
        ])
        
        return Observation(
            timestamp=data['timestamp'],
            sensor_id=self.source_id,
            values=values,
            quality_score=0.9
        )


class RealTimeSynchronizer:
    """
    实时同步器
    
    管理实验数据与数字孪生之间的实时双向同步
    """
    
    def __init__(self, twin: DigitalTwinCore, config: SyncConfiguration):
        self.twin = twin
        self.config = config
        self.sources: Dict[str, ExperimentDataSource] = {}
        self.transformer = DataTransformer()
        self.quality_filter = QualityFilter(config)
        
        # 同步状态
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 统计信息
        self.metrics = SyncMetrics()
        self._sync_history: deque[SyncRecord] = deque(maxlen=1000)
        
        # 缓冲区
        self._exp_buffer: deque[Observation] = deque(maxlen=config.buffer_size)
        self._sim_buffer: deque[StateVector] = deque(maxlen=config.buffer_size)
        
    def add_source(self, source: ExperimentDataSource) -> None:
        """添加数据源"""
        self.sources[source.source_id] = source
        source.on_data(self._on_experiment_data)
    
    def _on_experiment_data(self, observation: Observation) -> None:
        """处理实验数据到达"""
        with self._lock:
            # 质量过滤
            filtered_values, quality = self.quality_filter.filter(observation.values)
            
            if quality.value < self.config.min_quality_threshold:
                self.metrics.data_dropped += 1
                return
            
            # 创建过滤后的观测
            filtered_obs = Observation(
                timestamp=observation.timestamp,
                sensor_id=observation.sensor_id,
                values=filtered_values,
                quality_score=quality.value
            )
            
            self._exp_buffer.append(filtered_obs)
            
            # 触发同步
            if self.config.mode == SyncMode.EVENT_DRIVEN:
                self._sync_exp_to_sim(filtered_obs)
    
    def start(self) -> None:
        """启动同步"""
        self._running = True
        
        # 启动所有数据源
        for source in self.sources.values():
            source.connect()
            source.start_acquisition()
        
        # 启动同步线程
        if self.config.mode == SyncMode.REALTIME:
            self._sync_thread = threading.Thread(target=self._sync_loop)
            self._sync_thread.daemon = True
            self._sync_thread.start()
        
        print(f"Real-time synchronizer started (mode: {self.config.mode.value})")
    
    def stop(self) -> None:
        """停止同步"""
        self._running = False
        
        # 停止所有数据源
        for source in self.sources.values():
            source.stop_acquisition()
            source.disconnect()
        
        # 等待同步线程
        if self._sync_thread:
            self._sync_thread.join(timeout=2.0)
        
        print("Real-time synchronizer stopped")
    
    def _sync_loop(self) -> None:
        """同步循环"""
        while self._running:
            start_time = time.time()
            
            try:
                # 实验 -> 模拟
                if self.config.direction in [SyncDirection.EXP_TO_SIM, SyncDirection.BIDIRECTIONAL]:
                    self._process_exp_buffer()
                
                # 模拟 -> 实验 (获取控制输出)
                if self.config.direction in [SyncDirection.SIM_TO_EXP, SyncDirection.BIDIRECTIONAL]:
                    self._process_sim_buffer()
                
                # 自适应速率调整
                if self.config.adaptive_rate:
                    self._adjust_sync_rate()
                
            except Exception as e:
                print(f"Sync error: {e}")
            
            # 计算实际耗时并调整睡眠
            elapsed = (time.time() - start_time) * 1000
            sleep_time = max(0, (self.config.sync_interval_ms - elapsed) / 1000)
            time.sleep(sleep_time)
    
    def _process_exp_buffer(self) -> None:
        """处理实验数据缓冲区"""
        with self._lock:
            while self._exp_buffer:
                observation = self._exp_buffer.popleft()
                self._sync_exp_to_sim(observation)
    
    def _process_sim_buffer(self) -> None:
        """处理模拟数据缓冲区"""
        # 获取当前状态并发送给实验系统
        state = self.twin.get_current_state()
        if state:
            self._sim_buffer.append(state)
            # 这里可以实现向实验系统的反馈控制
    
    def _sync_exp_to_sim(self, observation: Observation) -> None:
        """同步实验数据到模拟"""
        start_time = time.time()
        data_id = str(uuid.uuid4())
        
        try:
            # 发送给数字孪生
            self.twin.observe(observation)
            
            # 记录同步
            latency = (time.time() - start_time) * 1000
            record = SyncRecord(
                timestamp=start_time,
                direction=SyncDirection.EXP_TO_SIM,
                data_id=data_id,
                source_data=observation,
                target_data=None,
                latency_ms=latency,
                success=True
            )
            
            self._update_metrics(record)
            self._sync_history.append(record)
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            record = SyncRecord(
                timestamp=start_time,
                direction=SyncDirection.EXP_TO_SIM,
                data_id=data_id,
                source_data=observation,
                target_data=None,
                latency_ms=latency,
                success=False,
                error_message=str(e)
            )
            self._update_metrics(record)
    
    def _update_metrics(self, record: SyncRecord) -> None:
        """更新统计指标"""
        self.metrics.total_syncs += 1
        
        if record.success:
            self.metrics.successful_syncs += 1
        else:
            self.metrics.failed_syncs += 1
        
        self.metrics.total_latency_ms += record.latency_ms
        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, record.latency_ms)
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, record.latency_ms)
        
        # 更新缓冲区利用率
        buffer_usage = len(self._exp_buffer) / self.config.buffer_size
        self.metrics.buffer_utilization = buffer_usage
    
    def _adjust_sync_rate(self) -> None:
        """自适应调整同步速率"""
        if not self.config.congestion_control:
            return
        
        # 根据缓冲区利用率调整
        if self.metrics.buffer_utilization > 0.8:
            # 拥塞，增加同步频率
            self.config.sync_interval_ms = max(1.0, self.config.sync_interval_ms * 0.9)
        elif self.metrics.buffer_utilization < 0.2:
            # 空闲，降低同步频率
            self.config.sync_interval_ms = min(100.0, self.config.sync_interval_ms * 1.1)
    
    def manual_sync(self, observation: Observation) -> bool:
        """手动同步单条数据"""
        try:
            self._sync_exp_to_sim(observation)
            return True
        except Exception as e:
            print(f"Manual sync failed: {e}")
            return False
    
    def get_metrics(self) -> SyncMetrics:
        """获取同步指标"""
        return self.metrics
    
    def get_sync_history(self, n: int = 100) -> List[SyncRecord]:
        """获取同步历史"""
        return list(self._sync_history)[-n:]


class BidirectionalMapper:
    """
    双向映射器
    
    建立实验域和模拟域之间的双向映射关系
    """
    
    def __init__(self):
        self.exp_to_sim_map: Dict[str, str] = {}
        self.sim_to_exp_map: Dict[str, str] = {}
        self._transforms: Dict[str, Tuple[Callable, Callable]] = {}
    
    def register_mapping(self, exp_field: str, sim_field: str,
                        forward_transform: Callable[[Any], Any] = lambda x: x,
                        backward_transform: Callable[[Any], Any] = lambda x: x) -> None:
        """注册字段映射"""
        self.exp_to_sim_map[exp_field] = sim_field
        self.sim_to_exp_map[sim_field] = exp_field
        self._transforms[f"{exp_field}->{sim_field}"] = (forward_transform, backward_transform)
    
    def exp_to_sim(self, exp_data: Dict[str, Any]) -> Dict[str, Any]:
        """实验数据转模拟数据"""
        result = {}
        for exp_field, value in exp_data.items():
            if exp_field in self.exp_to_sim_map:
                sim_field = self.exp_to_sim_map[exp_field]
                transform_key = f"{exp_field}->{sim_field}"
                if transform_key in self._transforms:
                    value = self._transforms[transform_key][0](value)
                result[sim_field] = value
            else:
                result[exp_field] = value
        return result
    
    def sim_to_exp(self, sim_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟数据转实验数据"""
        result = {}
        for sim_field, value in sim_data.items():
            if sim_field in self.sim_to_exp_map:
                exp_field = self.sim_to_exp_map[sim_field]
                transform_key = f"{exp_field}->{sim_field}"
                if transform_key in self._transforms:
                    value = self._transforms[transform_key][1](value)
                result[exp_field] = value
            else:
                result[sim_field] = value
        return result


class AsyncSynchronizer:
    """
    异步同步器
    
    支持asyncio的高性能异步同步
    """
    
    def __init__(self, twin: DigitalTwinCore, config: SyncConfiguration):
        self.twin = twin
        self.config = config
        self._running = False
        self._queue: asyncio.Queue[Observation] = asyncio.Queue(maxsize=config.buffer_size)
    
    async def start(self) -> None:
        """启动异步同步"""
        self._running = True
        await asyncio.gather(
            self._producer(),
            self._consumer()
        )
    
    async def stop(self) -> None:
        """停止异步同步"""
        self._running = False
    
    async def _producer(self) -> None:
        """生产者协程"""
        while self._running:
            # 模拟数据生成
            observation = Observation(
                timestamp=time.time(),
                sensor_id="async_sensor",
                values=np.random.randn(3)
            )
            
            try:
                await asyncio.wait_for(
                    self._queue.put(observation),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(self.config.sync_interval_ms / 1000)
    
    async def _consumer(self) -> None:
        """消费者协程"""
        while self._running:
            try:
                observation = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.1
                )
                
                # 同步到数字孪生
                self.twin.observe(observation)
                
            except asyncio.TimeoutError:
                pass


def demo():
    """演示实时同步功能"""
    print("=" * 60)
    print("实时同步机制演示")
    print("=" * 60)
    
    # 导入twin_core
    try:
        from .twin_core import DigitalTwinCore, TwinConfiguration, StateVector
    except ImportError:
        from twin_core import DigitalTwinCore, TwinConfiguration, StateVector
    
    # 创建数字孪生
    twin_config = TwinConfiguration()
    twin = DigitalTwinCore(twin_config)
    
    # 初始化
    initial_state = StateVector(
        timestamp=0.0,
        data=np.zeros(3),
        metadata={'init': True}
    )
    twin.initialize(initial_state)
    
    # 创建同步配置
    sync_config = SyncConfiguration(
        direction=SyncDirection.BIDIRECTIONAL,
        mode=SyncMode.REALTIME,
        sample_rate_hz=50.0,
        sync_interval_ms=20.0,
        adaptive_rate=True,
        congestion_control=True
    )
    
    # 创建同步器
    synchronizer = RealTimeSynchronizer(twin, sync_config)
    
    # 添加模拟数据源
    source1 = SimulatedExperimentSource("sensor_1", sync_config, signal_type="sine")
    source2 = SimulatedExperimentSource("sensor_2", sync_config, signal_type="random_walk")
    
    synchronizer.add_source(source1)
    synchronizer.add_source(source2)
    
    print("\n1. 启动实时同步")
    print(f"   模式: {sync_config.mode.value}")
    print(f"   采样率: {sync_config.sample_rate_hz} Hz")
    print(f"   同步间隔: {sync_config.sync_interval_ms} ms")
    
    synchronizer.start()
    
    # 运行一段时间
    print("\n2. 运行3秒采集数据...")
    time.sleep(3)
    
    # 停止
    print("\n3. 停止同步")
    synchronizer.stop()
    
    # 查看统计
    print("\n4. 同步统计")
    metrics = synchronizer.get_metrics()
    print(f"   总同步次数: {metrics.total_syncs}")
    print(f"   成功次数: {metrics.successful_syncs}")
    print(f"   成功率: {metrics.success_rate:.2%}")
    print(f"   平均延迟: {metrics.average_latency_ms:.2f} ms")
    print(f"   最小延迟: {metrics.min_latency_ms:.2f} ms")
    print(f"   最大延迟: {metrics.max_latency_ms:.2f} ms")
    print(f"   丢弃数据: {metrics.data_dropped}")
    
    # 演示双向映射
    print("\n5. 双向映射演示")
    mapper = BidirectionalMapper()
    
    # 注册映射
    mapper.register_mapping(
        "temperature_celsius", "temp_kelvin",
        forward_transform=lambda c: c + 273.15,
        backward_transform=lambda k: k - 273.15
    )
    mapper.register_mapping(
        "pressure_bar", "pressure_pa",
        forward_transform=lambda bar: bar * 1e5,
        backward_transform=lambda pa: pa / 1e5
    )
    
    exp_data = {
        "temperature_celsius": 25.0,
        "pressure_bar": 1.5,
        "other_field": "value"
    }
    
    sim_data = mapper.exp_to_sim(exp_data)
    print(f"   实验数据: {exp_data}")
    print(f"   模拟数据: {sim_data}")
    
    back_to_exp = mapper.sim_to_exp(sim_data)
    print(f"   还原数据: {back_to_exp}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return synchronizer


if __name__ == "__main__":
    demo()
