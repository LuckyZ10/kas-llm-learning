"""
KAS Core - Resource Quota Limiter

Resource monitoring and quota management for safe execution.
Supports cross-platform (Windows/Linux/macOS) resource tracking.
"""

import os
import sys
import time
import signal
import threading
import functools
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict
from enum import Enum
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ResourceStatus(Enum):
    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


@dataclass
class ResourceQuota:
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 512
    max_execution_time: int = 60
    max_file_size_mb: int = 10
    max_output_size_mb: int = 5

    def validate(self) -> List[str]:
        errors = []
        if not 0 <= self.max_cpu_percent <= 100:
            errors.append("max_cpu_percent must be between 0 and 100")
        if self.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive")
        if self.max_execution_time <= 0:
            errors.append("max_execution_time must be positive")
        if self.max_file_size_mb <= 0:
            errors.append("max_file_size_mb must be positive")
        if self.max_output_size_mb <= 0:
            errors.append("max_output_size_mb must be positive")
        return errors


@dataclass
class QuotaStatus:
    within_limits: bool = True
    cpu_status: str = "ok"
    memory_status: str = "ok"
    time_status: str = "ok"
    file_status: str = "ok"
    output_status: str = "ok"
    violations: List[str] = field(default_factory=list)
    current_cpu: float = 0.0
    current_memory_mb: float = 0.0
    current_time: float = 0.0

    def add_violation(self, message: str):
        self.violations.append(message)
        self.within_limits = False


class ResourceMonitor:
    def __init__(self, quota: Optional[ResourceQuota] = None):
        self.quota = quota or ResourceQuota()
        self._start_time: Optional[float] = None
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._samples: List[Dict[str, float]] = []
        self._max_samples = 100

        if not PSUTIL_AVAILABLE:
            raise ImportError(
                "psutil is required for resource monitoring. "
                "Install it with: pip install psutil>=5.9.0"
            )

        self._process = psutil.Process()

    def get_cpu_usage(self) -> float:
        try:
            return self._process.cpu_percent(interval=0.1)
        except psutil.Error:
            return 0.0

    def get_memory_usage(self) -> float:
        try:
            mem_info = self._process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except psutil.Error:
            return 0.0

    def get_execution_time(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_file_size_mb(self, file_path: str) -> float:
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except OSError:
            return 0.0

    def check_quota(self, quota: Optional[ResourceQuota] = None) -> QuotaStatus:
        check_quota = quota or self.quota
        status = QuotaStatus()

        cpu_usage = self.get_cpu_usage()
        memory_mb = self.get_memory_usage()
        exec_time = self.get_execution_time()

        status.current_cpu = cpu_usage
        status.current_memory_mb = memory_mb
        status.current_time = exec_time

        warning_threshold = 0.8

        if cpu_usage > check_quota.max_cpu_percent:
            status.cpu_status = "exceeded"
            status.add_violation(
                f"CPU usage {cpu_usage:.1f}% exceeds limit {check_quota.max_cpu_percent}%"
            )
        elif cpu_usage > check_quota.max_cpu_percent * warning_threshold:
            status.cpu_status = "warning"

        if memory_mb > check_quota.max_memory_mb:
            status.memory_status = "exceeded"
            status.add_violation(
                f"Memory usage {memory_mb:.1f}MB exceeds limit {check_quota.max_memory_mb}MB"
            )
        elif memory_mb > check_quota.max_memory_mb * warning_threshold:
            status.memory_status = "warning"

        if exec_time > check_quota.max_execution_time:
            status.time_status = "exceeded"
            status.add_violation(
                f"Execution time {exec_time:.1f}s exceeds limit {check_quota.max_execution_time}s"
            )
        elif exec_time > check_quota.max_execution_time * warning_threshold:
            status.time_status = "warning"

        return status

    def start_monitoring(self, interval: float = 0.5):
        with self._lock:
            if self._monitoring:
                return

            self._start_time = time.time()
            self._monitoring = True
            self._stop_event.clear()
            self._samples.clear()

            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self._monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        with self._lock:
            if not self._monitoring:
                return self._get_summary()

            self._monitoring = False
            self._stop_event.set()

            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2.0)

            return self._get_summary()

    def _monitor_loop(self, interval: float):
        while not self._stop_event.is_set():
            try:
                sample = {
                    "timestamp": time.time(),
                    "cpu": self.get_cpu_usage(),
                    "memory_mb": self.get_memory_usage(),
                    "elapsed": self.get_execution_time()
                }

                with self._lock:
                    self._samples.append(sample)
                    if len(self._samples) > self._max_samples:
                        self._samples = self._samples[-self._max_samples:]

                self._stop_event.wait(interval)
            except Exception:
                break

    def _get_summary(self) -> Dict[str, Any]:
        if not self._samples:
            return {
                "duration": self.get_execution_time(),
                "samples": 0
            }

        cpus = [s["cpu"] for s in self._samples]
        memories = [s["memory_mb"] for s in self._samples]

        return {
            "duration": self.get_execution_time(),
            "samples": len(self._samples),
            "cpu": {
                "avg": sum(cpus) / len(cpus) if cpus else 0,
                "max": max(cpus) if cpus else 0,
                "min": min(cpus) if cpus else 0
            },
            "memory_mb": {
                "avg": sum(memories) / len(memories) if memories else 0,
                "max": max(memories) if memories else 0,
                "min": min(memories) if memories else 0
            }
        }

    def get_current_stats(self) -> Dict[str, float]:
        return {
            "cpu_percent": self.get_cpu_usage(),
            "memory_mb": self.get_memory_usage(),
            "elapsed_seconds": self.get_execution_time()
        }


class ResourceLimiter:
    def __init__(self, quota: Optional[ResourceQuota] = None):
        self.quota = quota or ResourceQuota()
        self.monitor = ResourceMonitor(self.quota) if PSUTIL_AVAILABLE else None
        self._timeout_triggered = False

    def execute(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        self._timeout_triggered = False

        if self.monitor:
            self.monitor.start_monitoring()

        try:
            if sys.platform == "win32":
                result = self._execute_with_thread_timeout(func, *args, **kwargs)
            else:
                result = self._execute_with_signal_timeout(func, *args, **kwargs)

            return result
        finally:
            if self.monitor:
                self.monitor.stop_monitoring()

    def _execute_with_thread_timeout(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        result_container = {"result": None, "error": None, "completed": False}

        def target():
            try:
                result_container["result"] = func(*args, **kwargs)
                result_container["completed"] = True
            except Exception as e:
                result_container["error"] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.quota.max_execution_time)

        if thread.is_alive():
            self._timeout_triggered = True
            raise TimeoutError(
                f"Function execution exceeded time limit of {self.quota.max_execution_time}s"
            )

        if result_container["error"]:
            raise result_container["error"]

        return result_container["result"]

    def _execute_with_signal_timeout(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        def timeout_handler(signum, frame):
            self._timeout_triggered = True
            raise TimeoutError(
                f"Function execution exceeded time limit of {self.quota.max_execution_time}s"
            )

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.quota.max_execution_time)

        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        return result

    def check_file_size(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return True

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return size_mb <= self.quota.max_file_size_mb

    def check_output_size(self, output: str) -> bool:
        size_mb = len(output.encode('utf-8')) / (1024 * 1024)
        return size_mb <= self.quota.max_output_size_mb

    def get_status(self) -> Optional[QuotaStatus]:
        if self.monitor:
            return self.monitor.check_quota()
        return None

    def was_timeout_triggered(self) -> bool:
        return self._timeout_triggered


def with_quota(
    max_time: int = 60,
    max_memory_mb: int = 512,
    max_cpu_percent: float = 80.0
):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            quota = ResourceQuota(
                max_cpu_percent=max_cpu_percent,
                max_memory_mb=max_memory_mb,
                max_execution_time=max_time
            )
            limiter = ResourceLimiter(quota)
            return limiter.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def create_default_quota() -> ResourceQuota:
    return ResourceQuota(
        max_cpu_percent=80.0,
        max_memory_mb=512,
        max_execution_time=60,
        max_file_size_mb=10,
        max_output_size_mb=5
    )


def create_strict_quota() -> ResourceQuota:
    return ResourceQuota(
        max_cpu_percent=50.0,
        max_memory_mb=256,
        max_execution_time=30,
        max_file_size_mb=5,
        max_output_size_mb=2
    )


def create_relaxed_quota() -> ResourceQuota:
    return ResourceQuota(
        max_cpu_percent=95.0,
        max_memory_mb=2048,
        max_execution_time=300,
        max_file_size_mb=100,
        max_output_size_mb=50
    )
