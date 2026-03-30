"""
指标收集模块
Prometheus格式指标、系统监控、API统计
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps
import logging
import psutil

logger = logging.getLogger(__name__)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """增加计数器"""
        with self._lock:
            key = self._format_key(name, labels)
            self._counters[key] += value
    
    def decrement(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """减少计数器"""
        with self._lock:
            key = self._format_key(name, labels)
            self._counters[key] -= value
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """设置 gauges"""
        with self._lock:
            key = self._format_key(name, labels)
            self._gauges[key] = value
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录直方图值"""
        with self._lock:
            key = self._format_key(name, labels)
            self._histograms[key].append(value)
            # 限制存储数量
            if len(self._histograms[key]) > 10000:
                self._histograms[key] = self._histograms[key][-5000:]
    
    def timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """记录计时"""
        with self._lock:
            key = self._format_key(name, labels)
            self._timers[key].append(duration)
    
    def time(self, name: str, labels: Optional[Dict[str, str]] = None):
        """计时上下文管理器"""
        return TimerContext(self, name, labels)
    
    def _format_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """格式化key"""
        if not labels:
            return name
        label_str = ",".join([f'{k}="{v}"' for k, v in sorted(labels.items())])
        return f'{name}{{{label_str}}}'
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """获取计数器值"""
        key = self._format_key(name, labels)
        return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """获取gauge值"""
        key = self._format_key(name, labels)
        return self._gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """获取直方图统计"""
        import statistics
        
        key = self._format_key(name, labels)
        values = self._histograms.get(key, [])
        
        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "sum": sum(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        }
    
    def get_timer_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """获取计时统计"""
        key = self._format_key(name, labels)
        times = list(self._timers.get(key, []))
        
        if not times:
            return {"count": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}
        
        import statistics
        return {
            "count": len(times),
            "avg_ms": statistics.mean(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "p95_ms": sorted(times)[int(len(times) * 0.95)] * 1000,
        }
    
    def to_prometheus_format(self) -> str:
        """导出为Prometheus格式"""
        lines = []
        
        # 计数器
        for key, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in sorted(self._gauges.items()):
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")
        
        # 直方图
        for key, values in sorted(self._histograms.items()):
            if values:
                base_name = key.split('{')[0]
                lines.append(f"# TYPE {base_name} histogram")
                lines.append(f'{key}_count {len(values)}')
                lines.append(f'{key}_sum {sum(values)}')
        
        return "\n".join(lines)
    
    def get_all_metrics(self) -> Dict[str, any]:
        """获取所有指标"""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {k: self.get_histogram_stats(k.split('{')[0]) for k in self._histograms.keys()},
            "timers": {k: self.get_timer_stats(k.split('{')[0]) for k in self._timers.keys()},
        }


class TimerContext:
    """计时上下文"""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]]):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.timer(self.name, duration, self.labels)


# 全局指标收集器
metrics = MetricsCollector()


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics = metrics_collector or metrics
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self, interval: int = 60):
        """启动监控"""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self._thread.daemon = True
        self._thread.start()
        logger.info(f"System monitor started with {interval}s interval")
    
    def stop(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("System monitor stopped")
    
    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self._running:
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(interval)
    
    def _collect_metrics(self):
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.gauge("system_cpu_percent", cpu_percent)
        
        # 内存使用
        memory = psutil.virtual_memory()
        self.metrics.gauge("system_memory_percent", memory.percent)
        self.metrics.gauge("system_memory_used_gb", memory.used / (1024**3))
        self.metrics.gauge("system_memory_available_gb", memory.available / (1024**3))
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        self.metrics.gauge("system_disk_percent", disk.percent)
        self.metrics.gauge("system_disk_free_gb", disk.free / (1024**3))
        
        # 网络IO
        net_io = psutil.net_io_counters()
        self.metrics.gauge("system_network_bytes_sent", net_io.bytes_sent)
        self.metrics.gauge("system_network_bytes_recv", net_io.bytes_recv)
        
        # 进程数
        self.metrics.gauge("system_process_count", len(psutil.pids()))
        
        # 负载平均
        try:
            load_avg = psutil.getloadavg()
            self.metrics.gauge("system_load_avg_1m", load_avg[0])
            self.metrics.gauge("system_load_avg_5m", load_avg[1])
            self.metrics.gauge("system_load_avg_15m", load_avg[2])
        except AttributeError:
            pass  # Windows不支持


class APIMetrics:
    """API指标收集"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics = metrics_collector or metrics
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None
    ):
        """记录API请求"""
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status": str(status_code),
        }
        
        # 总请求数
        self.metrics.increment("api_requests_total", 1, labels)
        
        # 响应时间
        self.metrics.histogram("api_request_duration_seconds", duration, labels)
        self.metrics.timer("api_request_duration", duration, labels)
        
        # 状态码分类
        status_class = f"{status_code // 100}xx"
        self.metrics.increment(f"api_requests_{status_class}", 1, {"endpoint": endpoint})
        
        # 用户请求统计
        if user_id:
            self.metrics.increment("api_requests_by_user", 1, {"user_id": user_id})
    
    def record_task_submission(self, task_type: str, user_id: str):
        """记录任务提交"""
        self.metrics.increment("tasks_submitted_total", 1, {"type": task_type})
        self.metrics.increment("tasks_submitted_by_user", 1, {"user_id": user_id, "type": task_type})
    
    def record_task_completion(self, task_type: str, status: str, duration: float):
        """记录任务完成"""
        self.metrics.increment("tasks_completed_total", 1, {"type": task_type, "status": status})
        self.metrics.histogram("task_duration_seconds", duration, {"type": task_type})
    
    def record_auth_attempt(self, success: bool, auth_type: str = "password"):
        """记录认证尝试"""
        status = "success" if success else "failure"
        self.metrics.increment("auth_attempts_total", 1, {"status": status, "type": auth_type})


def track_time(name: str, labels: Optional[Dict[str, str]] = None):
    """计时装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                metrics.timer(name, duration, labels)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                metrics.timer(name, duration, labels)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# 创建全局实例
api_metrics = APIMetrics()
system_monitor = SystemMonitor()


import asyncio
