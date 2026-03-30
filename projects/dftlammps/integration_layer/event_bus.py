"""
DFT-LAMMPS 事件总线系统
=======================
模块间异步通信

基于发布-订阅模式的模块间通信机制

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from queue import Queue, Empty
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union, Coroutine
)
from functools import wraps


logger = logging.getLogger("event_bus")


class EventPriority(Enum):
    """事件优先级"""
    CRITICAL = 0      # 关键
    HIGH = 1          # 高
    NORMAL = 2        # 正常
    LOW = 3           # 低
    BACKGROUND = 4    # 后台


class EventType(Enum):
    """事件类型"""
    # 计算事件
    CALCULATION_STARTED = "calculation.started"
    CALCULATION_COMPLETED = "calculation.completed"
    CALCULATION_FAILED = "calculation.failed"
    
    # 数据事件
    DATA_CREATED = "data.created"
    DATA_UPDATED = "data.updated"
    DATA_DELETED = "data.deleted"
    
    # 模块事件
    MODULE_REGISTERED = "module.registered"
    MODULE_INITIALIZED = "module.initialized"
    MODULE_ERROR = "module.error"
    MODULE_SHUTDOWN = "module.shutdown"
    
    # 工作流事件
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_STEP_STARTED = "workflow.step.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    
    # 系统事件
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_INFO = "system.info"
    
    # 自定义事件
    CUSTOM = "custom"


@dataclass
class Event:
    """事件定义"""
    event_type: EventType               # 事件类型
    source: str                         # 事件源（模块名）
    data: Dict[str, Any] = field(default_factory=dict)  # 事件数据
    
    # 元数据
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None  # 关联ID（用于追踪）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        return cls(
            event_type=EventType(data["event_type"]),
            source=data["source"],
            data=data.get("data", {}),
            event_id=data.get("event_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            priority=EventPriority(data.get("priority", 2)),
            correlation_id=data.get("correlation_id")
        )


# 处理器类型
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventSubscription:
    """事件订阅"""
    
    def __init__(
        self,
        subscription_id: str,
        event_type: Optional[EventType],
        handler: Union[EventHandler, AsyncEventHandler],
        filter_func: Optional[Callable[[Event], bool]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        is_async: bool = False
    ):
        self.subscription_id = subscription_id
        self.event_type = event_type
        self.handler = handler
        self.filter_func = filter_func
        self.priority = priority
        self.is_async = is_async
        self.active = True
    
    def matches(self, event: Event) -> bool:
        """检查事件是否匹配订阅"""
        if not self.active:
            return False
        
        if self.event_type is not None and event.event_type != self.event_type:
            return False
        
        if self.filter_func and not self.filter_func(event):
            return False
        
        return True
    
    def cancel(self) -> None:
        """取消订阅"""
        self.active = False


class EventBus:
    """
    事件总线
    
    提供模块间的发布-订阅通信机制
    
    Example:
        bus = EventBus.get_instance()
        
        # 订阅事件
        def on_calculation(event):
            print(f"Calculation completed: {event.data}")
        
        subscription = bus.subscribe(
            EventType.CALCULATION_COMPLETED, 
            on_calculation
        )
        
        # 发布事件
        bus.publish(Event(
            event_type=EventType.CALCULATION_COMPLETED,
            source="vasp_module",
            data={"energy": -100.5}
        ))
        
        # 取消订阅
        subscription.cancel()
    """
    
    _instance: Optional[EventBus] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> EventBus:
        """获取事件总线实例"""
        return cls()
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self._subscriptions: Dict[Optional[EventType], List[EventSubscription]] = defaultdict(list)
            self._event_queue: Queue = Queue()
            self._running = False
            self._worker_thread: Optional[threading.Thread] = None
            self._async_loop: Optional[asyncio.AbstractEventLoop] = None
            self._history: List[Event] = []
            self._history_limit = 1000
            self._initialized = True
            self._stats = {
                "published": 0,
                "delivered": 0,
                "errors": 0
            }
    
    def subscribe(
        self,
        event_type: Optional[EventType],
        handler: Union[EventHandler, AsyncEventHandler],
        filter_func: Optional[Callable[[Event], bool]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        is_async: bool = False
    ) -> EventSubscription:
        """
        订阅事件
        
        Args:
            event_type: 事件类型（None表示订阅所有事件）
            handler: 事件处理器
            filter_func: 可选的过滤器函数
            priority: 处理器优先级
            is_async: 是否为异步处理器
        """
        subscription_id = str(uuid.uuid4())
        
        subscription = EventSubscription(
            subscription_id=subscription_id,
            event_type=event_type,
            handler=handler,
            filter_func=filter_func,
            priority=priority,
            is_async=is_async
        )
        
        self._subscriptions[event_type].append(subscription)
        
        # 按优先级排序
        self._subscriptions[event_type].sort(key=lambda s: s.priority.value)
        
        logger.debug(f"Subscribed to {event_type}: {subscription_id}")
        return subscription
    
    def unsubscribe(self, subscription: EventSubscription) -> bool:
        """取消订阅"""
        subscription.cancel()
        
        if subscription.event_type in self._subscriptions:
            self._subscriptions[subscription.event_type] = [
                s for s in self._subscriptions[subscription.event_type]
                if s.subscription_id != subscription.subscription_id
            ]
            return True
        
        return False
    
    def publish(self, event: Event, asynchronous: bool = True) -> None:
        """
        发布事件
        
        Args:
            event: 要发布的事件
            asynchronous: 是否异步处理
        """
        if asynchronous:
            self._event_queue.put(event)
        else:
            self._dispatch_event(event)
        
        # 记录历史
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history.pop(0)
        
        self._stats["published"] += 1
        logger.debug(f"Published event: {event.event_type.value}")
    
    def publish_sync(self, event: Event) -> None:
        """同步发布事件"""
        self.publish(event, asynchronous=False)
    
    def start(self) -> None:
        """启动事件处理"""
        if self._running:
            return
        
        self._running = True
        
        # 启动工作线程
        self._worker_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info("Event bus started")
    
    def stop(self) -> None:
        """停止事件处理"""
        self._running = False
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        
        logger.info("Event bus stopped")
    
    def clear_history(self) -> None:
        """清除历史"""
        self._history.clear()
    
    def get_history(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """获取历史事件"""
        events = self._history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if source:
            events = [e for e in events if e.source == source]
        
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self._stats.copy()
    
    def create_correlation_id(self) -> str:
        """创建关联ID"""
        return str(uuid.uuid4())
    
    def _event_loop(self) -> None:
        """事件处理循环"""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.1)
                self._dispatch_event(event)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in event loop: {e}")
                self._stats["errors"] += 1
    
    def _dispatch_event(self, event: Event) -> None:
        """分发事件"""
        # 收集匹配的订阅
        matching_subscriptions = []
        
        # 特定类型订阅
        if event.event_type in self._subscriptions:
            matching_subscriptions.extend(
                s for s in self._subscriptions[event.event_type] if s.matches(event)
            )
        
        # 通配符订阅
        if None in self._subscriptions:
            matching_subscriptions.extend(
                s for s in self._subscriptions[None] if s.matches(event)
            )
        
        # 按优先级排序
        matching_subscriptions.sort(key=lambda s: s.priority.value)
        
        # 分发
        for subscription in matching_subscriptions:
            try:
                if subscription.is_async:
                    # 异步处理
                    asyncio.create_task(subscription.handler(event))
                else:
                    # 同步处理
                    subscription.handler(event)
                
                self._stats["delivered"] += 1
                
            except Exception as e:
                logger.error(f"Error handling event {event.event_id}: {e}")
                self._stats["errors"] += 1


# 装饰器：事件处理器
def event_handler(
    event_type: EventType,
    priority: EventPriority = EventPriority.NORMAL,
    filter_func: Optional[Callable[[Event], bool]] = None
):
    """
    装饰器：将函数标记为事件处理器
    
    Example:
        @event_handler(EventType.CALCULATION_COMPLETED)
        def on_calculation(event):
            print(f"Calculation: {event.data}")
    """
    def decorator(func: Callable) -> Callable:
        bus = EventBus.get_instance()
        subscription = bus.subscribe(
            event_type=event_type,
            handler=func,
            filter_func=filter_func,
            priority=priority
        )
        
        func._event_subscription = subscription
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# 便捷函数
def publish_event(
    event_type: EventType,
    source: str,
    data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    priority: EventPriority = EventPriority.NORMAL
) -> None:
    """便捷函数：发布事件"""
    event = Event(
        event_type=event_type,
        source=source,
        data=data,
        priority=priority,
        correlation_id=correlation_id
    )
    EventBus.get_instance().publish(event)


def subscribe_to(
    event_type: EventType,
    handler: Callable[[Event], None],
    **kwargs
) -> EventSubscription:
    """便捷函数：订阅事件"""
    return EventBus.get_instance().subscribe(event_type, handler, **kwargs)


# 预定义的事件工厂
def calculation_started_event(
    calculation_id: str,
    calculation_type: str,
    source: str = "unknown"
) -> Event:
    """创建计算开始事件"""
    return Event(
        event_type=EventType.CALCULATION_STARTED,
        source=source,
        data={
            "calculation_id": calculation_id,
            "calculation_type": calculation_type
        }
    )


def calculation_completed_event(
    calculation_id: str,
    result: Dict[str, Any],
    source: str = "unknown"
) -> Event:
    """创建计算完成事件"""
    return Event(
        event_type=EventType.CALCULATION_COMPLETED,
        source=source,
        data={
            "calculation_id": calculation_id,
            "result": result
        }
    )


def workflow_step_event(
    workflow_id: str,
    step_id: str,
    step_name: str,
    status: str,
    source: str = "unknown"
) -> Event:
    """创建工作流步骤事件"""
    return Event(
        event_type=EventType.WORKFLOW_STEP_STARTED if status == "started" 
                   else EventType.WORKFLOW_STEP_COMPLETED,
        source=source,
        data={
            "workflow_id": workflow_id,
            "step_id": step_id,
            "step_name": step_name,
            "status": status
        }
    )


class EventLogger:
    """
    事件日志记录器
    
    自动记录特定类型的事件
    """
    
    def __init__(self, bus: Optional[EventBus] = None):
        self.bus = bus or EventBus.get_instance()
        self._subscriptions: List[EventSubscription] = []
        self._logs: List[Dict[str, Any]] = []
    
    def start_logging(self, event_types: Optional[List[EventType]] = None) -> None:
        """开始记录"""
        types_to_log = event_types or list(EventType)
        
        for event_type in types_to_log:
            sub = self.bus.subscribe(
                event_type=event_type,
                handler=self._log_event
            )
            self._subscriptions.append(sub)
    
    def stop_logging(self) -> None:
        """停止记录"""
        for sub in self._subscriptions:
            self.bus.unsubscribe(sub)
        self._subscriptions.clear()
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """获取日志"""
        return self._logs.copy()
    
    def clear_logs(self) -> None:
        """清除日志"""
        self._logs.clear()
    
    def _log_event(self, event: Event) -> None:
        """记录事件"""
        self._logs.append({
            "timestamp": event.timestamp,
            "event_type": event.event_type.value,
            "source": event.source,
            "data_summary": {k: str(v)[:50] for k, v in event.data.items()}
        })


class EventReplayer:
    """
    事件重放器
    
    重放历史事件用于调试或恢复
    """
    
    def __init__(self, bus: Optional[EventBus] = None):
        self.bus = bus or EventBus.get_instance()
    
    def replay(
        self,
        events: List[Event],
        speed: float = 1.0,
        callback: Optional[Callable[[Event], None]] = None
    ) -> None:
        """
        重放事件
        
        Args:
            events: 要重放的事件列表
            speed: 重放速度（1.0为原速，2.0为2倍速）
            callback: 每个事件的处理回调
        """
        if not events:
            return
        
        # 按时间排序
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # 重放
        last_time = sorted_events[0].timestamp
        
        for event in sorted_events:
            # 计算延迟
            delay = (event.timestamp - last_time) / speed
            if delay > 0:
                time.sleep(delay)
            
            # 发布事件
            self.bus.publish(event)
            
            if callback:
                callback(event)
            
            last_time = event.timestamp


# 导出主要类
__all__ = [
    'EventBus',
    'Event',
    'EventType',
    'EventPriority',
    'EventSubscription',
    'EventHandler',
    'event_handler',
    'publish_event',
    'subscribe_to',
    'EventLogger',
    'EventReplayer'
]