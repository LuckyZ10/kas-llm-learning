"""
KAS Agent Communication Protocol - 通信管理器模块

提供统一的通信管理，支持多传输层、自动重连、消息确认等高级功能。
"""
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

from kas.core.protocol.message import (
    Message, MessageType, MessagePriority, MessageBuilder
)
from kas.core.protocol.transport import Transport, LocalTransport
from kas.core.protocol.router import MessageRouter, RoutingStrategy


class ConnectionState(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class PendingMessage:
    """
    待确认消息
    
    Attributes:
        message: 原始消息
        sent_at: 发送时间
        retry_count: 重试次数
        ack_event: 确认事件
        timeout: 超时时间（秒）
    """
    message: Message
    sent_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    ack_event: asyncio.Event = field(default_factory=asyncio.Event)
    timeout: float = 30.0


@dataclass
class TransportConfig:
    """
    传输层配置
    
    Attributes:
        name: 传输层名称
        transport_class: 传输类
        config: 传输层配置参数
        priority: 优先级（数值越小优先级越高）
        auto_reconnect: 是否自动重连
        reconnect_interval: 重连间隔（秒）
        max_retries: 最大重试次数
    """
    name: str
    transport_class: type
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    auto_reconnect: bool = True
    reconnect_interval: float = 5.0
    max_retries: int = 3


class CommunicationManager:
    """
    通信管理器
    
    统一管理多个传输层，提供：
    1. 传输层管理 - 支持同时使用多种传输方式
    2. 自动重连 - 连接断开时自动重连
    3. 消息确认 - ACK机制确保消息送达
    4. 负载均衡 - 智能选择最佳传输层
    5. 消息队列 - 离线消息缓存
    
    Example:
        # 创建管理器
        manager = CommunicationManager("agent1")
        
        # 配置传输层
        manager.add_transport(TransportConfig(
            name="local",
            transport_class=LocalTransport,
            config={},
            priority=0
        ))
        
        # 启动
        await manager.start()
        
        # 发送消息
        await manager.send(message)
        
        # 停止
        await manager.stop()
    """
    
    def __init__(self, agent_id: str, enable_ack: bool = True):
        """
        Args:
            agent_id: 本Agent标识
            enable_ack: 是否启用消息确认机制
        """
        self.agent_id = agent_id
        self.enable_ack = enable_ack
        
        # 传输层管理
        self._transports: Dict[str, Transport] = {}
        self._transport_configs: Dict[str, TransportConfig] = {}
        self._transport_states: Dict[str, ConnectionState] = {}
        self._transport_tasks: Dict[str, asyncio.Task] = {}
        
        # 消息路由器
        self._router = MessageRouter()
        self._router.register_agent(agent_id)
        
        # 消息处理器
        self._message_handlers: Dict[MessageType, List[Callable[[Message], None]]] = {}
        self._default_handler: Optional[Callable[[Message], None]] = None
        
        # 消息确认
        self._pending_messages: Dict[str, PendingMessage] = {}
        self._ack_timeout = 30.0
        
        # 消息队列（离线缓存）
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._offline_mode = False
        
        # 心跳
        self._heartbeat_interval = 30.0
        self._last_heartbeat: Dict[str, datetime] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # 重连
        self._reconnect_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # 统计
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_acked": 0,
            "messages_failed": 0,
            "reconnects": 0,
            "started_at": None
        }
        
        # 注册默认消息处理器
        self._router.subscribe(
            agent_id, "*",
            self._on_message_received
        )
    
    # ==================== 传输层管理 ====================
    
    def add_transport(self, config: TransportConfig) -> None:
        """
        添加传输层
        
        Args:
            config: 传输层配置
        """
        self._transport_configs[config.name] = config
        self._transport_states[config.name] = ConnectionState.DISCONNECTED
        logger.info(f"Transport config added: {config.name} ({config.transport_class.__name__})")
    
    def remove_transport(self, name: str) -> None:
        """移除传输层"""
        if name in self._transports:
            asyncio.create_task(self._disconnect_transport(name))
        del self._transport_configs[name]
        del self._transport_states[name]
    
    async def _connect_transport(self, name: str) -> bool:
        """连接指定传输层"""
        config = self._transport_configs.get(name)
        if not config:
            return False
        
        self._transport_states[name] = ConnectionState.CONNECTING
        
        try:
            transport = config.transport_class(self.agent_id, **config.config)
            success = await transport.connect()
            
            if success:
                self._transports[name] = transport
                self._transport_states[name] = ConnectionState.CONNECTED
                transport.on_message(self._create_transport_handler(name))
                
                # 启动监听任务
                self._transport_tasks[name] = asyncio.create_task(
                    self._transport_listen_loop(name)
                )
                
                logger.info(f"Transport connected: {name}")
                return True
            else:
                self._transport_states[name] = ConnectionState.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect transport {name}: {e}")
            self._transport_states[name] = ConnectionState.ERROR
            return False
    
    async def _disconnect_transport(self, name: str) -> None:
        """断开指定传输层"""
        if name in self._transport_tasks:
            self._transport_tasks[name].cancel()
            try:
                await self._transport_tasks[name]
            except asyncio.CancelledError:
                pass
            del self._transport_tasks[name]
        
        if name in self._transports:
            await self._transports[name].disconnect()
            del self._transports[name]
        
        self._transport_states[name] = ConnectionState.DISCONNECTED
        logger.info(f"Transport disconnected: {name}")
    
    def _create_transport_handler(self, transport_name: str) -> Callable[[Message], None]:
        """创建传输层消息处理器"""
        async def handler(message: Message):
            await self._on_transport_message(transport_name, message)
        return handler
    
    async def _transport_listen_loop(self, name: str):
        """传输层监听循环"""
        transport = self._transports.get(name)
        if not transport:
            return
        
        while not self._stop_event.is_set():
            try:
                message = await transport.receive()
                if message:
                    await self._on_transport_message(name, message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Transport {name} listen error: {e}")
                await asyncio.sleep(1)
    
    # ==================== 生命周期管理 ====================
    
    async def start(self) -> bool:
        """启动通信管理器"""
        logger.info(f"Starting CommunicationManager for {self.agent_id}")
        
        self._stats["started_at"] = datetime.now().isoformat()
        
        # 连接所有传输层
        for name in self._transport_configs:
            await self._connect_transport(name)
        
        # 启动重连任务
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())
        
        # 启动心跳任务
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # 启动消息队列处理器
        asyncio.create_task(self._message_queue_processor())
        
        logger.info("CommunicationManager started")
        return True
    
    async def stop(self) -> None:
        """停止通信管理器"""
        logger.info("Stopping CommunicationManager")
        
        self._stop_event.set()
        
        # 取消任务
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        for task in self._transport_tasks.values():
            task.cancel()
        
        # 断开所有传输层
        for name in list(self._transports.keys()):
            await self._disconnect_transport(name)
        
        # 清理待确认消息
        for pending in self._pending_messages.values():
            pending.ack_event.set()  # 释放等待者
        
        logger.info("CommunicationManager stopped")
    
    # ==================== 消息发送 ====================
    
    async def send(self, message: Message, wait_ack: bool = False,
                   timeout: float = 30.0) -> bool:
        """
        发送消息
        
        Args:
            message: 要发送的消息
            wait_ack: 是否等待确认
            timeout: 超时时间（秒）
        
        Returns:
            是否发送成功
        """
        # 设置发送者
        if not message.sender:
            message.sender = self.agent_id
        
        # 选择最佳传输层
        transport_name = self._select_transport()
        if not transport_name:
            # 没有可用传输层，放入离线队列
            if self._offline_mode:
                try:
                    self._message_queue.put_nowait((message, wait_ack, timeout))
                    return True
                except asyncio.QueueFull:
                    logger.error("Message queue full")
                    return False
            else:
                logger.error("No available transport")
                return False
        
        transport = self._transports[transport_name]
        
        # 如果需要确认，添加到待确认列表
        if wait_ack and self.enable_ack:
            pending = PendingMessage(
                message=message,
                timeout=timeout
            )
            self._pending_messages[message.id] = pending
        
        # 发送
        try:
            success = await transport.send(message)
            if success:
                self._stats["messages_sent"] += 1
                
                if wait_ack and self.enable_ack:
                    # 等待确认
                    try:
                        await asyncio.wait_for(
                            pending.ack_event.wait(),
                            timeout=timeout
                        )
                        self._stats["messages_acked"] += 1
                        return True
                    except asyncio.TimeoutError:
                        logger.warning(f"ACK timeout for message {message.id}")
                        self._stats["messages_failed"] += 1
                        # 重试
                        return await self._retry_message(pending)
                return True
            else:
                self._stats["messages_failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Send error: {e}")
            self._stats["messages_failed"] += 1
            return False
    
    async def send_request(self, receiver: str, payload: Dict[str, Any],
                          timeout: float = 30.0) -> Optional[Message]:
        """
        发送请求并等待响应
        
        Args:
            receiver: 接收者
            payload: 请求负载
            timeout: 超时时间
        
        Returns:
            响应消息，或None（超时）
        """
        request = MessageBuilder().request() \
            .from_agent(self.agent_id) \
            .to_agent(receiver) \
            .with_payload(payload) \
            .build()
        
        # 创建响应等待事件
        response_event = asyncio.Event()
        response_msg: Optional[Message] = None
        
        async def response_handler(msg: Message):
            nonlocal response_msg
            if (msg.type == MessageType.RESPONSE and
                msg.correlation_id == request.id):
                response_msg = msg
                response_event.set()
        
        # 临时订阅响应
        self._router.subscribe(
            self.agent_id,
            f"response_to_{request.id}",
            response_handler
        )
        
        try:
            # 发送请求
            success = await self.send(request)
            if not success:
                return None
            
            # 等待响应
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
            return response_msg
            
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout to {receiver}")
            return None
        finally:
            # 清理订阅
            self._router.unsubscribe_all(self.agent_id)
    
    async def broadcast(self, payload: Dict[str, Any],
                        msg_type: MessageType = MessageType.EVENT) -> int:
        """
        广播消息
        
        Args:
            payload: 消息负载
            msg_type: 消息类型
        
        Returns:
            成功发送的传输层数量
        """
        message = Message(
            type=msg_type,
            sender=self.agent_id,
            receiver=None,  # None表示广播
            payload=payload
        )
        
        count = 0
        for name, transport in self._transports.items():
            try:
                if await transport.send(message):
                    count += 1
            except Exception as e:
                logger.error(f"Broadcast error on {name}: {e}")
        
        return count
    
    async def _retry_message(self, pending: PendingMessage) -> bool:
        """重试发送消息"""
        config = self._transport_configs.get(self._select_transport())
        if not config:
            return False
        
        if pending.retry_count >= config.max_retries:
            logger.error(f"Max retries exceeded for message {pending.message.id}")
            return False
        
        pending.retry_count += 1
        logger.info(f"Retrying message {pending.message.id} (attempt {pending.retry_count})")
        
        # 重新发送
        return await self.send(pending.message, wait_ack=True, timeout=pending.timeout)
    
    def _select_transport(self) -> Optional[str]:
        """选择最佳传输层"""
        available = [
            (name, self._transport_configs[name].priority)
            for name, state in self._transport_states.items()
            if state == ConnectionState.CONNECTED and name in self._transports
        ]
        
        if not available:
            return None
        
        # 按优先级排序
        available.sort(key=lambda x: x[1])
        return available[0][0]
    
    # ==================== 消息接收处理 ====================
    
    async def _on_transport_message(self, transport_name: str, message: Message) -> None:
        """处理从传输层接收的消息"""
        self._stats["messages_received"] += 1
        
        # 更新心跳时间
        self._last_heartbeat[transport_name] = datetime.now()
        
        # 处理确认消息
        if message.type == MessageType.ACK and self.enable_ack:
            await self._handle_ack(message)
            return
        
        # 处理心跳消息
        if message.type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(transport_name, message)
            return
        
        # 如果需要确认，发送ACK
        if self.enable_ack and message.headers.get("require_ack") == "true":
            await self._send_ack(message)
        
        # 路由消息
        await self._router.route(message)
    
    async def _on_message_received(self, message: Message) -> None:
        """路由器回调 - 消息到达"""
        # 调用注册的消息处理器
        handlers = self._message_handlers.get(message.type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
        
        # 调用默认处理器
        if self._default_handler:
            try:
                if asyncio.iscoroutinefunction(self._default_handler):
                    await self._default_handler(message)
                else:
                    self._default_handler(message)
            except Exception as e:
                logger.error(f"Default handler error: {e}")
    
    async def _handle_ack(self, ack_message: Message) -> None:
        """处理确认消息"""
        acked_msg_id = ack_message.payload.get("acknowledged_msg_id")
        if acked_msg_id and acked_msg_id in self._pending_messages:
            pending = self._pending_messages.pop(acked_msg_id)
            pending.ack_event.set()
            logger.debug(f"Message {acked_msg_id} acknowledged")
    
    async def _send_ack(self, original_message: Message) -> None:
        """发送确认消息"""
        ack = original_message.create_ack()
        await self.send(ack)
    
    # ==================== 消息处理器注册 ====================
    
    def on_message(self, msg_type: MessageType,
                   handler: Callable[[Message], None]) -> None:
        """
        注册消息处理器
        
        Args:
            msg_type: 消息类型
            handler: 处理函数
        """
        if msg_type not in self._message_handlers:
            self._message_handlers[msg_type] = []
        self._message_handlers[msg_type].append(handler)
    
    def on_any_message(self, handler: Callable[[Message], None]) -> None:
        """注册默认消息处理器（处理所有类型）"""
        self._default_handler = handler
    
    def register_handler(self, msg_type: MessageType,
                        handler: Callable[[Message], None]) -> None:
        """兼容旧接口的消息处理器注册"""
        self.on_message(msg_type, handler)
    
    # ==================== 心跳机制 ====================
    
    async def _heartbeat_loop(self) -> None:
        """心跳发送循环"""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._heartbeat_interval
                )
            except asyncio.TimeoutError:
                # 发送心跳
                for name, transport in self._transports.items():
                    if self._transport_states.get(name) == ConnectionState.CONNECTED:
                        try:
                            heartbeat = MessageBuilder().heartbeat() \
                                .from_agent(self.agent_id) \
                                .with_payload({
                                    "transport": name,
                                    "stats": transport.get_stats()
                                }) \
                                .build()
                            await transport.send(heartbeat)
                        except Exception as e:
                            logger.warning(f"Failed to send heartbeat on {name}: {e}")
    
    async def _handle_heartbeat(self, transport_name: str, message: Message) -> None:
        """处理心跳消息"""
        logger.debug(f"Heartbeat received from {message.sender} via {transport_name}")
        # 可以在这里实现健康检查逻辑
    
    # ==================== 自动重连 ====================
    
    async def _reconnect_loop(self) -> None:
        """自动重连循环"""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=5.0  # 每5秒检查一次
                )
            except asyncio.TimeoutError:
                for name, config in self._transport_configs.items():
                    if not config.auto_reconnect:
                        continue
                    
                    state = self._transport_states.get(name)
                    if state in [ConnectionState.DISCONNECTED, ConnectionState.ERROR]:
                        self._stats["reconnects"] += 1
                        logger.info(f"Attempting to reconnect transport: {name}")
                        await self._connect_transport(name)
    
    # ==================== 消息队列处理 ====================
    
    async def _message_queue_processor(self) -> None:
        """消息队列处理器（离线模式）"""
        while not self._stop_event.is_set():
            try:
                message, wait_ack, timeout = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                # 尝试发送
                if await self.send(message, wait_ack, timeout):
                    self._message_queue.task_done()
                else:
                    # 发送失败，放回队列
                    try:
                        self._message_queue.put_nowait((message, wait_ack, timeout))
                    except asyncio.QueueFull:
                        logger.error("Message queue full, dropping message")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message queue processor error: {e}")
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        transport_stats = {
            name: {
                "state": state.value,
                "stats": self._transports[name].get_stats() if name in self._transports else None
            }
            for name, state in self._transport_states.items()
        }
        
        return {
            **self._stats,
            "transports": transport_stats,
            "pending_messages": len(self._pending_messages),
            "queue_size": self._message_queue.qsize()
        }
    
    def get_transport_states(self) -> Dict[str, ConnectionState]:
        """获取传输层状态"""
        return self._transport_states.copy()
