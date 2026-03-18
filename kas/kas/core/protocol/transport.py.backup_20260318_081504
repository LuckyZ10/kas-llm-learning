"""
KAS Agent Communication Protocol - 传输层模块

提供多种传输方式的实现：本地内存、TCP、WebSocket
所有传输都支持异步操作。
"""
import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Set, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 网络相关导入
import socket
from asyncio import StreamReader, StreamWriter

# WebSocket支持（可选）
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.client import WebSocketClientProtocol
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.debug("websockets not available, WebSocketTransport disabled")

from kas.core.protocol.message import Message, MessageType, SerializationFormat


class Transport(ABC):
    """
    传输层抽象基类
    
    所有传输方式（本地内存、TCP、WebSocket）的基类。
    定义统一的传输接口。
    
    Attributes:
        transport_id: 传输层唯一标识
        agent_id: 本Agent标识
        message_handler: 消息处理回调函数
        connected: 是否已连接
        stats: 传输统计信息
    """
    
    def __init__(self, agent_id: str):
        self.transport_id = str(uuid.uuid4())
        self.agent_id = agent_id
        self.message_handler: Optional[Callable[[Message], None]] = None
        self.connected = False
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "errors": 0,
            "connected_since": None
        }
        self._closing = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """建立连接"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    async def send(self, message: Message) -> bool:
        """
        发送消息
        
        Args:
            message: 要发送的消息
        
        Returns:
            是否发送成功
        """
        pass
    
    @abstractmethod
    async def receive(self) -> Optional[Message]:
        """
        接收消息（阻塞）
        
        Returns:
            接收到的消息，或None
        """
        pass
    
    def on_message(self, handler: Callable[[Message], None]) -> None:
        """
        设置消息处理器
        
        Args:
            handler: 消息处理回调函数
        """
        self.message_handler = handler
    
    async def start_listening(self) -> None:
        """开始监听消息（后台任务）"""
        while not self._closing and self.connected:
            try:
                message = await self.receive()
                if message and self.message_handler:
                    self._stats["messages_received"] += 1
                    asyncio.create_task(self._handle_message(message))
            except Exception as e:
                if not self._closing:
                    logger.error(f"Error in transport {self.transport_id}: {e}")
                    self._stats["errors"] += 1
    
    async def _handle_message(self, message: Message) -> None:
        """内部消息处理"""
        try:
            if self.message_handler:
                await self.message_handler(message)
        except Exception as e:
            logger.error(f"Message handler error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取传输统计信息"""
        return self._stats.copy()
    
    def _update_sent_stats(self, bytes_count: int) -> None:
        """更新发送统计"""
        self._stats["messages_sent"] += 1
        self._stats["bytes_sent"] += bytes_count
    
    def _update_received_stats(self, bytes_count: int) -> None:
        """更新接收统计"""
        self._stats["messages_received"] += 1
        self._stats["bytes_received"] += bytes_count


class LocalTransport(Transport):
    """
    本地内存传输
    
    用于同一进程内Agent间的通信，使用asyncio Queue实现。
    性能最高，无序列化开销。
    
    Example:
        transport = LocalTransport("agent1")
        await transport.connect()
        await transport.send(message)
    """
    
    # 类级别的消息总线，用于进程内通信
    _message_bus: Dict[str, asyncio.Queue] = {}
    _lock = asyncio.Lock()
    
    def __init__(self, agent_id: str, max_queue_size: int = 1000):
        super().__init__(agent_id)
        self.max_queue_size = max_queue_size
        self._queue: Optional[asyncio.Queue] = None
        self._listening = False
    
    async def connect(self) -> bool:
        """建立连接 - 创建消息队列"""
        async with self._lock:
            if self.agent_id not in self._message_bus:
                self._message_bus[self.agent_id] = asyncio.Queue(
                    maxsize=self.max_queue_size
                )
            self._queue = self._message_bus[self.agent_id]
        
        self.connected = True
        self._stats["connected_since"] = datetime.now().isoformat()
        logger.debug(f"LocalTransport connected for agent {self.agent_id}")
        return True
    
    async def disconnect(self) -> None:
        """断开连接 - 移除消息队列"""
        self._closing = True
        self.connected = False
        async with self._lock:
            if self.agent_id in self._message_bus:
                del self._message_bus[self.agent_id]
        logger.debug(f"LocalTransport disconnected for agent {self.agent_id}")
    
    async def send(self, message: Message) -> bool:
        """
        发送消息到目标Agent的队列
        
        Args:
            message: 要发送的消息
        
        Returns:
            是否发送成功
        """
        if not self.connected:
            logger.error("Transport not connected")
            return False
        
        target_id = message.receiver
        if not target_id:
            # 广播模式
            async with self._lock:
                for agent_id, queue in self._message_bus.items():
                    if agent_id != self.agent_id:
                        try:
                            queue.put_nowait(message)
                        except asyncio.QueueFull:
                            logger.warning(f"Queue full for agent {agent_id}")
            self._update_sent_stats(1)
            return True
        
        # 点对点发送
        async with self._lock:
            if target_id not in self._message_bus:
                logger.warning(f"Target agent {target_id} not found in local bus")
                return False
            target_queue = self._message_bus[target_id]
        
        try:
            target_queue.put_nowait(message)
            self._update_sent_stats(1)
            return True
        except asyncio.QueueFull:
            logger.error(f"Queue full for agent {target_id}")
            return False
    
    async def receive(self) -> Optional[Message]:
        """从队列接收消息"""
        if not self._queue:
            return None
        
        try:
            message = await self._queue.get()
            self._update_received_stats(1)
            return message
        except asyncio.CancelledError:
            return None
    
    @classmethod
    async def register_agent(cls, agent_id: str, max_queue_size: int = 1000) -> None:
        """注册Agent到本地总线"""
        async with cls._lock:
            if agent_id not in cls._message_bus:
                cls._message_bus[agent_id] = asyncio.Queue(maxsize=max_queue_size)
    
    @classmethod
    async def unregister_agent(cls, agent_id: str) -> None:
        """从本地总线注销Agent"""
        async with cls._lock:
            if agent_id in cls._message_bus:
                del cls._message_bus[agent_id]
    
    @classmethod
    def get_registered_agents(cls) -> List[str]:
        """获取已注册的Agent列表"""
        return list(cls._message_bus.keys())


class TCPTransport(Transport):
    """
    TCP传输
    
    用于跨进程/跨机通信，基于asyncio的TCP实现。
    支持服务端和客户端模式。
    
    Example:
        # 服务端
        transport = TCPTransport("agent1", host="0.0.0.0", port=8888)
        await transport.start_server()
        
        # 客户端
        transport = TCPTransport("agent2", server_host="localhost", server_port=8888)
        await transport.connect()
    """
    
    def __init__(self, agent_id: str, 
                 host: str = "0.0.0.0", port: int = 0,
                 server_host: Optional[str] = None, server_port: Optional[int] = None,
                 serialization_format: SerializationFormat = SerializationFormat.JSON):
        super().__init__(agent_id)
        self.host = host
        self.port = port
        self.server_host = server_host or host
        self.server_port = server_port or port
        self.serialization_format = serialization_format
        
        self._server: Optional[asyncio.Server] = None
        self._reader: Optional[StreamReader] = None
        self._writer: Optional[StreamWriter] = None
        self._clients: Dict[str, StreamWriter] = {}  # 服务端模式下的客户端连接
        self._receive_buffer = asyncio.Queue(maxsize=1000)
    
    async def start_server(self) -> bool:
        """启动TCP服务端"""
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                self.host,
                self.port
            )
            self.port = self._server.sockets[0].getsockname()[1]
            self.connected = True
            self._stats["connected_since"] = datetime.now().isoformat()
            logger.info(f"TCP server started on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start TCP server: {e}")
            return False
    
    async def _handle_client(self, reader: StreamReader, writer: StreamWriter):
        """处理客户端连接"""
        client_addr = writer.get_extra_info('peername')
        client_id = f"{client_addr[0]}:{client_addr[1]}"
        self._clients[client_id] = writer
        logger.debug(f"Client connected: {client_id}")
        
        try:
            while not self._closing:
                # 读取消息长度（4字节）
                length_data = await reader.read(4)
                if len(length_data) < 4:
                    break
                
                msg_length = int.from_bytes(length_data, 'big')
                
                # 读取消息数据
                data = await reader.read(msg_length)
                if len(data) < msg_length:
                    break
                
                # 反序列化
                message = Message.deserialize(data)
                await self._receive_buffer.put(message)
                self._update_received_stats(len(data))
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            del self._clients[client_id]
            writer.close()
            await writer.wait_closed()
            logger.debug(f"Client disconnected: {client_id}")
    
    async def connect(self) -> bool:
        """连接到TCP服务端"""
        try:
            self._reader, self._writer = await asyncio.open_connection(
                self.server_host,
                self.server_port
            )
            self.connected = True
            self._stats["connected_since"] = datetime.now().isoformat()
            
            # 启动接收任务
            asyncio.create_task(self._receive_loop())
            
            logger.info(f"Connected to TCP server at {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TCP server: {e}")
            return False
    
    async def _receive_loop(self):
        """接收循环"""
        try:
            while not self._closing and self.connected:
                # 读取消息长度
                length_data = await self._reader.read(4)
                if len(length_data) < 4:
                    break
                
                msg_length = int.from_bytes(length_data, 'big')
                
                # 读取消息数据
                data = await self._reader.read(msg_length)
                if len(data) < msg_length:
                    break
                
                # 反序列化并放入缓冲区
                message = Message.deserialize(data)
                await self._receive_buffer.put(message)
                self._update_received_stats(len(data))
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._closing:
                logger.error(f"Receive loop error: {e}")
                self._stats["errors"] += 1
        finally:
            self.connected = False
    
    async def disconnect(self) -> None:
        """断开连接"""
        self._closing = True
        self.connected = False
        
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        # 关闭所有客户端连接
        for writer in self._clients.values():
            writer.close()
        self._clients.clear()
        
        logger.info("TCP transport disconnected")
    
    async def send(self, message: Message) -> bool:
        """发送消息"""
        if not self.connected:
            logger.error("TCP transport not connected")
            return False
        
        try:
            data = message.serialize(format=self.serialization_format)
            length_prefix = len(data).to_bytes(4, 'big')
            
            if self._writer:  # 客户端模式
                self._writer.write(length_prefix + data)
                await self._writer.drain()
            elif self._clients:  # 服务端模式 - 广播
                for writer in self._clients.values():
                    writer.write(length_prefix + data)
                await asyncio.gather(*[w.drain() for w in self._clients.values()])
            
            self._update_sent_stats(len(data) + 4)
            return True
        except Exception as e:
            logger.error(f"Failed to send TCP message: {e}")
            self._stats["errors"] += 1
            return False
    
    async def receive(self) -> Optional[Message]:
        """接收消息"""
        try:
            message = await asyncio.wait_for(
                self._receive_buffer.get(),
                timeout=1.0
            )
            return message
        except asyncio.TimeoutError:
            return None


class WebSocketTransport(Transport):
    """
    WebSocket传输
    
    用于实时双向通信，支持浏览器客户端。
    需要 websockets 库支持。
    
    Example:
        # 服务端
        transport = WebSocketTransport("agent1", host="0.0.0.0", port=8765)
        await transport.start_server()
        
        # 客户端
        transport = WebSocketTransport("agent2", server_url="ws://localhost:8765")
        await transport.connect()
    """
    
    def __init__(self, agent_id: str,
                 host: str = "0.0.0.0", port: int = 0,
                 server_url: Optional[str] = None,
                 serialization_format: SerializationFormat = SerializationFormat.JSON):
        if not WEBSOCKET_AVAILABLE:
            raise RuntimeError("websockets library not installed")
        
        super().__init__(agent_id)
        self.host = host
        self.port = port
        self.server_url = server_url
        self.serialization_format = serialization_format
        
        self._server = None
        self._websocket = None
        self._clients = set()
        self._receive_buffer = asyncio.Queue(maxsize=1000)
    
    async def start_server(self) -> bool:
        """启动WebSocket服务端"""
        try:
            self._server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port
            )
            self.port = self._server.sockets[0].getsockname()[1]
            self.connected = True
            self._stats["connected_since"] = datetime.now().isoformat()
            logger.info(f"WebSocket server started on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    async def _handle_client(self, websocket, path: str):
        """处理WebSocket客户端连接"""
        self._clients.add(websocket)
        client_addr = websocket.remote_address
        logger.debug(f"WebSocket client connected: {client_addr}")
        
        try:
            async for message in websocket:
                try:
                    data = message.encode('utf-8') if isinstance(message, str) else message
                    msg_obj = Message.deserialize(data)
                    await self._receive_buffer.put(msg_obj)
                    self._update_received_stats(len(data))
                except Exception as e:
                    logger.error(f"Failed to process WebSocket message: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.debug(f"WebSocket client disconnected: {client_addr}")
    
    async def connect(self) -> bool:
        """连接到WebSocket服务端"""
        if not self.server_url:
            raise ValueError("server_url required for client mode")
        
        try:
            self._websocket = await websockets.connect(self.server_url)
            self.connected = True
            self._stats["connected_since"] = datetime.now().isoformat()
            
            # 启动接收任务
            asyncio.create_task(self._receive_loop())
            
            logger.info(f"Connected to WebSocket server at {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket server: {e}")
            return False
    
    async def _receive_loop(self):
        """接收循环"""
        try:
            async for message in self._websocket:
                try:
                    data = message.encode('utf-8') if isinstance(message, str) else message
                    msg_obj = Message.deserialize(data)
                    await self._receive_buffer.put(msg_obj)
                    self._update_received_stats(len(data))
                except Exception as e:
                    logger.error(f"Failed to process received message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            if not self._closing:
                logger.error(f"WebSocket receive error: {e}")
                self._stats["errors"] += 1
        finally:
            self.connected = False
    
    async def disconnect(self) -> None:
        """断开连接"""
        self._closing = True
        self.connected = False
        
        if self._websocket:
            await self._websocket.close()
        
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        # 关闭所有客户端连接
        for client in list(self._clients):
            await client.close()
        self._clients.clear()
        
        logger.info("WebSocket transport disconnected")
    
    async def send(self, message: Message) -> bool:
        """发送消息"""
        if not self.connected:
            logger.error("WebSocket transport not connected")
            return False
        
        try:
            data = message.serialize(format=self.serialization_format)
            
            if self._websocket:  # 客户端模式
                await self._websocket.send(data)
            elif self._clients:  # 服务端模式
                if self._clients:
                    await asyncio.gather(*[client.send(data) for client in self._clients])
            
            self._update_sent_stats(len(data))
            return True
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            self._stats["errors"] += 1
            return False
    
    async def receive(self) -> Optional[Message]:
        """接收消息"""
        try:
            msg = await asyncio.wait_for(
                self._receive_buffer.get(),
                timeout=1.0
            )
            return msg
        except asyncio.TimeoutError:
            return None
