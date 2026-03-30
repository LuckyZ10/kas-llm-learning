#!/usr/bin/env python3
"""
fault_tolerance.py
==================
容错与重试机制模块

功能：
- 智能重试策略
- 断路器模式
- 故障分类与处理
- 指数退避
- 死信队列
"""

import os
import sys
import json
import time
import logging
import traceback
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable, Type, Union
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import random

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """故障类型"""
    TRANSIENT = "transient"      # 瞬时故障，可重试
    PERMANENT = "permanent"      # 永久故障，不应重试
    RESOURCE = "resource"        # 资源不足
    AUTHENTICATION = "auth"      # 认证失败
    NETWORK = "network"          # 网络故障
    TIMEOUT = "timeout"          # 超时
    SCHEDULER = "scheduler"      # 调度器错误
    UNKNOWN = "unknown"          # 未知


class FailureSeverity(Enum):
    """故障严重程度"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FailureRecord:
    """故障记录"""
    job_id: Optional[str]
    failure_type: FailureType
    severity: FailureSeverity
    error_message: str
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "failure_type": self.failure_type.value,
            "severity": self.severity.name,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


@dataclass
class RetryPolicy:
    """重试策略"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    backoff_strategy: str = "exponential"  # exponential, linear, fixed
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )
    non_retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (SystemExit, KeyboardInterrupt)
    )
    on_retry_callback: Optional[Callable] = None
    on_failure_callback: Optional[Callable] = None
    
    def calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        if self.backoff_strategy == "exponential":
            delay = self.initial_delay * (self.backoff_factor ** attempt)
        elif self.backoff_strategy == "linear":
            delay = self.initial_delay * (attempt + 1)
        else:  # fixed
            delay = self.initial_delay
        
        # 添加抖动，避免惊群效应
        jitter = random.uniform(0, delay * 0.1)
        delay += jitter
        
        return min(delay, self.max_delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """判断是否应重试"""
        if attempt >= self.max_attempts - 1:
            return False
        
        # 检查非可重试异常
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # 检查可重试异常
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return False


class CircuitBreakerState(Enum):
    """断路器状态"""
    CLOSED = "closed"       # 正常，允许请求
    OPEN = "open"           # 故障，拒绝请求
    HALF_OPEN = "half_open" # 测试恢复


class CircuitBreaker:
    """断路器 - 防止级联故障"""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()
        
        # 统计
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
    
    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            return self._state
    
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            
            if self._state == CircuitBreakerState.OPEN:
                # 检查是否过了恢复期
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitBreakerState.HALF_OPEN
                        self._half_open_calls = 0
                        self._success_count = 0
                        logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                        return True
                return False
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                return self._half_open_calls < self.half_open_max_calls
            
            return False
    
    def record_success(self):
        """记录成功"""
        with self._lock:
            self._total_successes += 1
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitBreakerState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED (recovered)")
            else:
                self._failure_count = 0
    
    def record_failure(self):
        """记录失败"""
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker {self.name} OPEN (recovery failed)")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker {self.name} OPEN (threshold reached)")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行带断路器保护的调用"""
        if not self.can_execute():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls += 1
        
        self._total_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None
            }


class CircuitBreakerOpenError(Exception):
    """断路器打开错误"""
    pass


class FailureClassifier:
    """故障分类器"""
    
    # 错误模式映射
    ERROR_PATTERNS: Dict[FailureType, List[str]] = {
        FailureType.TRANSIENT: [
            "temporary",
            "try again",
            "rate limit",
            "throttled",
            "service unavailable",
            "503"
        ],
        FailureType.PERMANENT: [
            "not found",
            "invalid",
            "bad request",
            "forbidden",
            "unauthorized",
            "400",
            "403",
            "404"
        ],
        FailureType.RESOURCE: [
            "no space",
            "disk full",
            "quota exceeded",
            "insufficient",
            "out of memory",
            "oom"
        ],
        FailureType.AUTHENTICATION: [
            "authentication",
            "unauthorized",
            "permission denied",
            "access denied",
            "401",
            "403"
        ],
        FailureType.NETWORK: [
            "connection",
            "timeout",
            "refused",
            "reset",
            "unreachable",
            "dns",
            "network"
        ],
        FailureType.TIMEOUT: [
            "timeout",
            "timed out",
            "deadline exceeded"
        ],
        FailureType.SCHEDULER: [
            "sbatch",
            "qsub",
            "bsub",
            "queue",
            "partition",
            "reservation"
        ]
    }
    
    @classmethod
    def classify(cls, error_message: str, exception: Exception = None) -> Tuple[FailureType, FailureSeverity]:
        """
        分类故障
        
        Args:
            error_message: 错误消息
            exception: 异常对象
        
        Returns:
            (FailureType, FailureSeverity)
        """
        error_lower = error_message.lower()
        
        # 根据模式匹配
        for failure_type, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_lower:
                    severity = cls._determine_severity(failure_type, error_message)
                    return failure_type, severity
        
        # 根据异常类型判断
        if exception:
            exc_type = type(exception).__name__.lower()
            if "timeout" in exc_type:
                return FailureType.TIMEOUT, FailureSeverity.MEDIUM
            elif "connection" in exc_type:
                return FailureType.NETWORK, FailureSeverity.HIGH
            elif "auth" in exc_type or "permission" in exc_type:
                return FailureType.AUTHENTICATION, FailureSeverity.HIGH
        
        return FailureType.UNKNOWN, FailureSeverity.MEDIUM
    
    @classmethod
    def _determine_severity(cls, failure_type: FailureType, error_message: str) -> FailureSeverity:
        """确定严重程度"""
        # 基于故障类型的默认严重度
        severity_map = {
            FailureType.TRANSIENT: FailureSeverity.LOW,
            FailureType.TIMEOUT: FailureSeverity.MEDIUM,
            FailureType.NETWORK: FailureSeverity.MEDIUM,
            FailureType.RESOURCE: FailureSeverity.HIGH,
            FailureType.AUTHENTICATION: FailureSeverity.HIGH,
            FailureType.SCHEDULER: FailureSeverity.MEDIUM,
            FailureType.PERMANENT: FailureSeverity.CRITICAL,
            FailureType.UNKNOWN: FailureSeverity.MEDIUM
        }
        
        severity = severity_map.get(failure_type, FailureSeverity.MEDIUM)
        
        # 基于消息内容调整
        if "critical" in error_message.lower() or "fatal" in error_message.lower():
            severity = FailureSeverity(severity.value + 1) if severity.value < 4 else severity
        
        return severity


class JobRetryHandler:
    """作业重试处理器"""
    
    def __init__(
        self,
        default_policy: RetryPolicy = None,
        max_history: int = 1000
    ):
        self.default_policy = default_policy or RetryPolicy()
        self._failure_history: deque = deque(maxlen=max_history)
        self._retry_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        
        # 故障类型特定的策略
        self._policies: Dict[FailureType, RetryPolicy] = {
            FailureType.TRANSIENT: RetryPolicy(
                max_attempts=5,
                initial_delay=1.0,
                backoff_factor=2.0
            ),
            FailureType.TIMEOUT: RetryPolicy(
                max_attempts=3,
                initial_delay=5.0,
                backoff_factor=1.5
            ),
            FailureType.NETWORK: RetryPolicy(
                max_attempts=5,
                initial_delay=2.0,
                backoff_factor=2.0
            ),
            FailureType.RESOURCE: RetryPolicy(
                max_attempts=3,
                initial_delay=30.0,
                backoff_factor=1.5
            ),
            FailureType.SCHEDULER: RetryPolicy(
                max_attempts=3,
                initial_delay=10.0,
                backoff_factor=2.0
            )
        }
    
    def execute_with_retry(
        self,
        func: Callable,
        job_id: Optional[str] = None,
        policy: RetryPolicy = None,
        *args,
        **kwargs
    ) -> Any:
        """
        带重试的执行
        
        Args:
            func: 要执行的函数
            job_id: 作业ID
            policy: 重试策略
            *args, **kwargs: 函数参数
        
        Returns:
            函数返回值
        
        Raises:
            最后一次异常
        """
        policy = policy or self.default_policy
        last_exception = None
        
        for attempt in range(policy.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                # 成功，重置重试计数
                if job_id:
                    with self._lock:
                        self._retry_counts[job_id] = 0
                
                return result
                
            except Exception as e:
                last_exception = e
                error_message = str(e)
                
                # 分类故障
                failure_type, severity = FailureClassifier.classify(error_message, e)
                
                # 记录故障
                record = FailureRecord(
                    job_id=job_id,
                    failure_type=failure_type,
                    severity=severity,
                    error_message=error_message,
                    traceback=traceback.format_exc(),
                    context={"attempt": attempt}
                )
                
                with self._lock:
                    self._failure_history.append(record)
                    if job_id:
                        self._retry_counts[job_id] += 1
                
                # 获取特定类型的策略
                type_policy = self._policies.get(failure_type, policy)
                
                # 检查是否应重试
                if not type_policy.should_retry(e, attempt):
                    logger.error(f"Non-retryable failure for job {job_id}: {error_message}")
                    if policy.on_failure_callback:
                        policy.on_failure_callback(record)
                    raise
                
                # 计算延迟
                delay = type_policy.calculate_delay(attempt)
                
                logger.warning(
                    f"Job {job_id} failed (attempt {attempt + 1}/{type_policy.max_attempts}): "
                    f"{error_message}. Retrying in {delay:.1f}s..."
                )
                
                # 重试回调
                if type_policy.on_retry_callback:
                    type_policy.on_retry_callback(record, attempt, delay)
                
                time.sleep(delay)
        
        # 所有重试都失败了
        logger.error(f"Job {job_id} failed after {policy.max_attempts} attempts")
        raise last_exception
    
    def get_failure_stats(self) -> dict:
        """获取故障统计"""
        with self._lock:
            stats = defaultdict(lambda: {"count": 0, "last": None})
            
            for record in self._failure_history:
                ft = record.failure_type.value
                stats[ft]["count"] += 1
                stats[ft]["last"] = record.timestamp.isoformat()
            
            return dict(stats)
    
    def get_job_retry_count(self, job_id: str) -> int:
        """获取作业重试次数"""
        with self._lock:
            return self._retry_counts.get(job_id, 0)
    
    def reset_job_retry_count(self, job_id: str):
        """重置作业重试计数"""
        with self._lock:
            if job_id in self._retry_counts:
                del self._retry_counts[job_id]


class FaultToleranceManager:
    """容错管理器"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handler = JobRetryHandler()
        self._dead_letter_queue: deque = deque(maxlen=100)
        self._lock = threading.Lock()
    
    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """获取或创建断路器"""
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout
                )
            return self.circuit_breakers[name]
    
    def execute(
        self,
        func: Callable,
        circuit_breaker_name: Optional[str] = None,
        retry_policy: RetryPolicy = None,
        job_id: Optional[str] = None,
        fallback: Callable = None,
        *args,
        **kwargs
    ) -> Any:
        """
        带容错的执行
        
        Args:
            func: 要执行的函数
            circuit_breaker_name: 断路器名称
            retry_policy: 重试策略
            job_id: 作业ID
            fallback: 失败时的回退函数
            *args, **kwargs: 函数参数
        
        Returns:
            函数返回值或回退值
        """
        # 断路器保护
        if circuit_breaker_name:
            cb = self.get_circuit_breaker(circuit_breaker_name)
            if not cb.can_execute():
                if fallback:
                    logger.warning(f"Circuit breaker {circuit_breaker_name} open, using fallback")
                    return fallback(*args, **kwargs)
                raise CircuitBreakerOpenError(f"Circuit breaker {circuit_breaker_name} is OPEN")
        
        try:
            # 带重试的执行
            result = self.retry_handler.execute_with_retry(
                func, job_id, retry_policy, *args, **kwargs
            )
            
            # 记录成功
            if circuit_breaker_name:
                cb.record_success()
            
            return result
            
        except Exception as e:
            # 记录失败
            if circuit_breaker_name:
                cb.record_failure()
            
            # 添加到死信队列
            self._add_to_dead_letter(job_id, func, args, kwargs, e)
            
            # 使用回退
            if fallback:
                logger.warning(f"Execution failed, using fallback: {e}")
                return fallback(*args, **kwargs)
            
            raise
    
    def _add_to_dead_letter(
        self,
        job_id: Optional[str],
        func: Callable,
        args: tuple,
        kwargs: dict,
        error: Exception
    ):
        """添加到死信队列"""
        entry = {
            "job_id": job_id,
            "function": func.__name__,
            "args": args,
            "kwargs": kwargs,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        with self._lock:
            self._dead_letter_queue.append(entry)
    
    def get_dead_letter_queue(self) -> List[dict]:
        """获取死信队列内容"""
        with self._lock:
            return list(self._dead_letter_queue)
    
    def clear_dead_letter_queue(self):
        """清空死信队列"""
        with self._lock:
            self._dead_letter_queue.clear()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "circuit_breakers": {
                name: cb.get_stats()
                for name, cb in self.circuit_breakers.items()
            },
            "failure_stats": self.retry_handler.get_failure_stats(),
            "dead_letter_count": len(self._dead_letter_queue)
        }


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None,
    on_failure: Callable = None
):
    """重试装饰器"""
    policy = RetryPolicy(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        retryable_exceptions=retryable_exceptions,
        on_retry_callback=on_retry,
        on_failure_callback=on_failure
    )
    
    def decorator(func: Callable) -> Callable:
        handler = JobRetryHandler(policy)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return handler.execute_with_retry(func, None, policy, *args, **kwargs)
        
        return wrapper
    return decorator


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """断路器装饰器"""
    def decorator(func: Callable) -> Callable:
        cb = CircuitBreaker(name, failure_threshold, recovery_timeout)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        return wrapper
    return decorator
