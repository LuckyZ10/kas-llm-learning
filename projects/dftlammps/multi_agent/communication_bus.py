"""
Communication Bus - 通信总线
实现Agent间的消息传递机制
"""
from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import uuid

from .agent_core import Message, MessageType


class ChannelType(Enum):
    """通道类型"""
    DIRECT = "direct"           # 点对点
    BROADCAST = "broadcast"     # 广播
    MULTICAST = "multicast"     # 组播
    PUBSUB = "pubsub"           # 发布订阅


@dataclass
class Channel:
    """通信通道"""
    channel_id: str
    channel_type: ChannelType
    members: Set[str] = None
    topic: Optional[str] = None
    
    def __post_init__(self):
        if self.members is None:
            self.members = set()


class MessageFilter:
    """消息过滤器"""
    
    def __init__(
        self,
        sender_ids: Optional[Set[str]] = None,
        message_types: Optional[Set[MessageType]] = None,
        content_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    ):
        self.sender_ids = sender_ids
        self.message_types = message_types
        self.content_predicate = content_predicate
    
    def matches(self, message: Message) -> bool:
        """检查消息是否匹配过滤器"""
        if self.sender_ids and message.sender_id not in self.sender_ids:
            return False
        if self.message_types and message.message_type not in self.message_types:
            return False
        if self.content_predicate and not self.content_predicate(message.content):
            return False
        return True


class CommunicationBus:
    """
    通信总线 - 管理中心化的Agent通信
    支持同步和异步模式
    """
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}  # agent_id -> agent
        self.channels: Dict[str, Channel] = {}
        self.message_history: List[Message] = []
        self.max_history = 10000
        
        # 消息队列
        self._queues: Dict[str, asyncio.Queue[Message]] = {}
        
        # 订阅管理
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> agent_ids
        self._agent_topics: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> topics
        
        # 消息拦截器
        self._interceptors: List[Callable[[Message], Optional[Message]]] = []
        
        # 统计
        self.stats = {
            "messages_routed": 0,
            "messages_dropped": 0,
            "bytes_transferred": 0
        }
        
        # 运行状态
        self._running = False
        self._router_task: Optional[asyncio.Task] = None
        
        # 回调注册
        self._callbacks: Dict[str, List[Callable[[Message], None]]] = defaultdict(list)
    
    def register_agent(self, agent_id: str, agent: Any) -> None:
        """注册Agent到总线"""
        self.agents[agent_id] = agent
        self._queues[agent_id] = asyncio.Queue()
    
    def unregister_agent(self, agent_id: str) -> None:
        """从总线注销Agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self._queues:
            del self._queues[agent_id]
        
        # 清理订阅
        for topic in list(self._agent_topics[agent_id]):
            self.unsubscribe(agent_id, topic)
    
    def create_channel(
        self,
        channel_type: ChannelType,
        channel_id: Optional[str] = None,
        topic: Optional[str] = None,
        members: Optional[List[str]] = None
    ) -> Channel:
        """创建通信通道"""
        channel_id = channel_id or str(uuid.uuid4())
        
        channel = Channel(
            channel_id=channel_id,
            channel_type=channel_type,
            members=set(members or []),
            topic=topic
        )
        
        self.channels[channel_id] = channel
        return channel
    
    async def send(self, message: Message) -> bool:
        """
        发送消息
        返回是否成功发送
        """
        # 应用拦截器
        for interceptor in self._interceptors:
            message = interceptor(message)
            if message is None:
                self.stats["messages_dropped"] += 1
                return False
        
        # 存储历史
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # 路由消息
        if message.receiver_id:
            # 点对点消息
            await self._route_direct(message)
        else:
            # 广播消息
            await self._route_broadcast(message)
        
        self.stats["messages_routed"] += 1
        self.stats["bytes_transferred"] += len(json.dumps(message.to_dict()))
        
        # 触发回调
        await self._trigger_callbacks(message)
        
        return True
    
    async def _route_direct(self, message: Message) -> None:
        """路由直接消息"""
        receiver_id = message.receiver_id
        
        if receiver_id in self._queues:
            await self._queues[receiver_id].put(message)
        
        # 也放入Agent的inbox（如果Agent有inbox属性）
        if receiver_id in self.agents:
            agent = self.agents[receiver_id]
            if hasattr(agent, 'inbox'):
                await agent.inbox.put(message)
    
    async def _route_broadcast(self, message: Message) -> None:
        """路由广播消息"""
        # 获取消息相关的topic
        topics = message.metadata.get("topics", [])
        
        target_agents = set()
        
        # 根据topic路由
        for topic in topics:
            target_agents.update(self._subscriptions.get(topic, set()))
        
        # 如果没有指定topic，广播给所有Agent
        if not topics:
            target_agents = set(self.agents.keys())
        
        # 发送给目标Agent
        for agent_id in target_agents:
            if agent_id != message.sender_id:  # 不发给发送者
                await self._route_direct(
                    Message(
                        sender_id=message.sender_id,
                        receiver_id=agent_id,
                        message_type=message.message_type,
                        content=message.content,
                        metadata={**message.metadata, "broadcast": True}
                    )
                )
    
    def subscribe(self, agent_id: str, topic: str) -> None:
        """订阅主题"""
        self._subscriptions[topic].add(agent_id)
        self._agent_topics[agent_id].add(topic)
    
    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """取消订阅"""
        self._subscriptions[topic].discard(agent_id)
        self._agent_topics[agent_id].discard(topic)
    
    async def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        接收消息（同步风格接口）
        用于同步模式的Agent
        """
        if agent_id not in self._queues:
            return None
        
        try:
            return await asyncio.wait_for(
                self._queues[agent_id].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    def add_interceptor(
        self,
        interceptor: Callable[[Message], Optional[Message]]
    ) -> None:
        """添加消息拦截器"""
        self._interceptors.append(interceptor)
    
    def register_callback(
        self,
        message_type: str,
        callback: Callable[[Message], None]
    ) -> None:
        """注册消息回调"""
        self._callbacks[message_type].append(callback)
    
    async def _trigger_callbacks(self, message: Message) -> None:
        """触发回调"""
        callbacks = self._callbacks.get(message.message_type.name, [])
        callbacks.extend(self._callbacks.get("*", []))
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def get_message_history(
        self,
        sender_id: Optional[str] = None,
        receiver_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[Message]:
        """获取消息历史"""
        filtered = self.message_history
        
        if sender_id:
            filtered = [m for m in filtered if m.sender_id == sender_id]
        if receiver_id:
            filtered = [m for m in filtered if m.receiver_id == receiver_id]
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
        
        return filtered[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "registered_agents": len(self.agents),
            "active_channels": len(self.channels),
            "message_history_size": len(self.message_history),
            "subscriptions": sum(len(s) for s in self._subscriptions.values())
        }


class AsyncCommunicationBus(CommunicationBus):
    """
    异步通信总线 - 支持更复杂的异步模式
    """
    
    def __init__(self):
        super().__init__()
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._request_timeout = 30.0
    
    async def request_response(
        self,
        sender_id: str,
        receiver_id: str,
        content: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[Message]:
        """
        请求-响应模式
        发送请求并等待响应
        """
        request_id = str(uuid.uuid4())
        
        # 创建future等待响应
        future = asyncio.Future()
        self._pending_requests[request_id] = future
        
        # 发送请求
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.COMMUNICATION,
            content=content,
            metadata={"request_id": request_id, "is_request": True}
        )
        
        await self.send(message)
        
        try:
            # 等待响应
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_requests.pop(request_id, None)
    
    async def send_response(
        self,
        request_message: Message,
        content: Dict[str, Any]
    ) -> None:
        """发送响应"""
        request_id = request_message.metadata.get("request_id")
        
        response = Message(
            sender_id=request_message.receiver_id,
            receiver_id=request_message.sender_id,
            message_type=MessageType.RESULT,
            content=content,
            metadata={"request_id": request_id, "is_response": True}
        )
        
        await self.send(response)
        
        # 如果有pending future，设置结果
        if request_id in self._pending_requests:
            self._pending_requests[request_id].set_result(response)
    
    async def publish(
        self,
        sender_id: str,
        topic: str,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.COMMUNICATION
    ) -> None:
        """发布消息到主题"""
        message = Message(
            sender_id=sender_id,
            receiver_id=None,  # 广播
            message_type=message_type,
            content=content,
            metadata={"topics": [topic]}
        )
        
        await self.send(message)


class SyncCommunicationBus:
    """
    同步通信总线 - 用于同步Agent
    使用线程安全的队列
    """
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self._queues: Dict[str, List[Message]] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.message_history: List[Message] = []
    
    def register_agent(self, agent_id: str, agent: Any) -> None:
        """注册Agent"""
        self.agents[agent_id] = agent
    
    def send_sync(self, message: Message) -> bool:
        """同步发送消息"""
        self.message_history.append(message)
        
        if message.receiver_id:
            # 点对点
            if message.receiver_id in self._queues:
                self._queues[message.receiver_id].append(message)
                return True
        else:
            # 广播给所有
            for agent_id in self.agents:
                if agent_id != message.sender_id:
                    self._queues[agent_id].append(message)
            return True
        
        return False
    
    def receive_sync(self, agent_id: str, block: bool = False, timeout: Optional[float] = None) -> Optional[Message]:
        """同步接收消息"""
        import time
        
        start_time = time.time()
        
        while True:
            if agent_id in self._queues and self._queues[agent_id]:
                return self._queues[agent_id].pop(0)
            
            if not block:
                return None
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.01)


class MessageRouter:
    """
    智能消息路由器
    基于内容和Agent能力路由消息
    """
    
    def __init__(self, bus: CommunicationBus):
        self.bus = bus
        self._routing_rules: List[Callable[[Message], Optional[str]]] = []
        self._agent_capabilities: Dict[str, Set[str]] = {}
    
    def register_capability(self, agent_id: str, capabilities: List[str]) -> None:
        """注册Agent能力"""
        self._agent_capabilities[agent_id] = set(capabilities)
    
    def add_routing_rule(self, rule: Callable[[Message], Optional[str]]) -> None:
        """添加路由规则"""
        self._routing_rules.append(rule)
    
    async def route_by_capability(self, message: Message, required_capability: str) -> List[str]:
        """基于能力路由消息"""
        capable_agents = [
            agent_id for agent_id, caps in self._agent_capabilities.items()
            if required_capability in caps
        ]
        
        # 发送给所有有能力的Agent
        for agent_id in capable_agents:
            routed_message = Message(
                sender_id=message.sender_id,
                receiver_id=agent_id,
                message_type=message.message_type,
                content=message.content,
                metadata={**message.metadata, "routed_by_capability": required_capability}
            )
            await self.bus.send(routed_message)
        
        return capable_agents
    
    async def route_by_content_type(self, message: Message) -> None:
        """基于内容类型路由"""
        content_type = message.content.get("type")
        
        if content_type:
            # 查找能处理该内容类型的Agent
            for agent_id, caps in self._agent_capabilities.items():
                if f"handle_{content_type}" in caps:
                    await self.bus.send(Message(
                        sender_id=message.sender_id,
                        receiver_id=agent_id,
                        message_type=message.message_type,
                        content=message.content,
                        metadata=message.metadata
                    ))


class CommunicationBusManager:
    """
    通信总线管理器
    管理多个通信总线实例
    """
    
    _instance: Optional[CommunicationBusManager] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._buses: Dict[str, CommunicationBus] = {}
            cls._instance._default_bus: Optional[CommunicationBus] = None
        return cls._instance
    
    def create_bus(self, name: str, async_mode: bool = True) -> CommunicationBus:
        """创建新的通信总线"""
        if async_mode:
            bus = AsyncCommunicationBus()
        else:
            bus = CommunicationBus()
        
        self._buses[name] = bus
        
        if self._default_bus is None:
            self._default_bus = bus
        
        return bus
    
    def get_bus(self, name: Optional[str] = None) -> Optional[CommunicationBus]:
        """获取通信总线"""
        if name:
            return self._buses.get(name)
        return self._default_bus
    
    def set_default_bus(self, name: str) -> bool:
        """设置默认总线"""
        if name in self._buses:
            self._default_bus = self._buses[name]
            return True
        return False
    
    def list_buses(self) -> List[str]:
        """列出所有总线"""
        return list(self._buses.keys())


# 便捷函数
def get_communication_bus(name: Optional[str] = None) -> CommunicationBus:
    """获取通信总线实例"""
    manager = CommunicationBusManager()
    bus = manager.get_bus(name)
    if bus is None:
        bus = manager.create_bus(name or "default")
    return bus
