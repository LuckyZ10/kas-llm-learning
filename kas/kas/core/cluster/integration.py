"""
KAS Distributed Agent Cluster - Phase 5.1集成模块

将分布式集群与Phase 5.1的通信协议集成，实现：
1. 使用TCPTransport/WebSocketTransport进行节点间通信
2. 与Crew系统集成，支持分布式Crew
3. 统一的消息处理和路由
"""
import asyncio
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from kas.core.protocol import (
    Message, MessageType, MessagePriority, MessageBuilder,
    CommunicationManager, TransportConfig, ConnectionState,
    TCPTransport, WebSocketTransport, LocalTransport
)
from kas.core.protocol.router import RoutingStrategy


class ClusterMessageHandler:
    """
    集群消息处理器
    
    处理集群内部的各种消息类型，桥接ClusterNode和CommunicationManager。
    """
    
    # 消息类型定义
    MSG_TYPE_JOIN_REQUEST = "cluster.join_request"
    MSG_TYPE_JOIN_RESPONSE = "cluster.join_response"
    MSG_TYPE_HEARTBEAT = "cluster.heartbeat"
    MSG_TYPE_HEARTBEAT_RESPONSE = "cluster.heartbeat_response"
    MSG_TYPE_VOTE_REQUEST = "cluster.vote_request"
    MSG_TYPE_VOTE_RESPONSE = "cluster.vote_response"
    MSG_TYPE_LEAVE_NOTIFY = "cluster.leave_notify"
    MSG_TYPE_STATE_REQUEST = "cluster.state_request"
    MSG_TYPE_STATE_RESPONSE = "cluster.state_response"
    MSG_TYPE_APPEND_ENTRIES = "cluster.append_entries"
    MSG_TYPE_APPEND_ENTRIES_RESPONSE = "cluster.append_entries_response"
    MSG_TYPE_TASK_ASSIGN = "cluster.task_assign"
    MSG_TYPE_TASK_RESULT = "cluster.task_result"
    MSG_TYPE_TASK_MIGRATE = "cluster.task_migrate"
    MSG_TYPE_BROADCAST = "cluster.broadcast"
    
    def __init__(self, cluster_node, cluster_manager, state_store=None):
        """
        初始化消息处理器
        
        Args:
            cluster_node: ClusterNode实例
            cluster_manager: ClusterManager实例
            state_store: DistributedStateStore实例（可选）
        """
        from kas.core.cluster.node import ClusterNode
        from kas.core.cluster.manager import ClusterManager
        from kas.core.cluster.state import DistributedStateStore
        
        self.node: ClusterNode = cluster_node
        self.manager: ClusterManager = cluster_manager
        self.state_store: Optional[DistributedStateStore] = state_store
        
        # 消息处理映射
        self._handlers: Dict[str, Callable] = {
            self.MSG_TYPE_JOIN_REQUEST: self._handle_join_request,
            self.MSG_TYPE_JOIN_RESPONSE: self._handle_join_response,
            self.MSG_TYPE_HEARTBEAT: self._handle_heartbeat,
            self.MSG_TYPE_HEARTBEAT_RESPONSE: self._handle_heartbeat_response,
            self.MSG_TYPE_VOTE_REQUEST: self._handle_vote_request,
            self.MSG_TYPE_VOTE_RESPONSE: self._handle_vote_response,
            self.MSG_TYPE_LEAVE_NOTIFY: self._handle_leave_notify,
            self.MSG_TYPE_STATE_REQUEST: self._handle_state_request,
            self.MSG_TYPE_APPEND_ENTRIES: self._handle_append_entries,
            self.MSG_TYPE_TASK_ASSIGN: self._handle_task_assign,
            self.MSG_TYPE_TASK_RESULT: self._handle_task_result,
            self.MSG_TYPE_TASK_MIGRATE: self._handle_task_migrate,
        }
        
        # 待处理的响应（用于request-response模式）
        self._pending_responses: Dict[str, asyncio.Future] = {}
        
        logger.info("ClusterMessageHandler initialized")
    
    async def handle_message(self, message: Message) -> None:
        """
        处理接收到的消息
        
        Args:
            message: 消息对象
        """
        # 获取消息类型
        msg_type = message.payload.get("cluster_msg_type")
        
        if not msg_type:
            logger.warning(f"Received message without cluster_msg_type: {message}")
            return
        
        # 查找处理器
        handler = self._handlers.get(msg_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error handling message {msg_type}: {e}")
        else:
            logger.warning(f"No handler for message type: {msg_type}")
    
    async def _handle_join_request(self, message: Message) -> None:
        """处理加入请求"""
        node_info_data = message.payload.get("node_info", {})
        
        from kas.core.cluster.node import NodeInfo
        node_info = NodeInfo.from_dict(node_info_data)
        
        # 添加到成员列表
        self.node.add_member(node_info)
        
        # 发送响应
        response_payload = {
            "cluster_msg_type": self.MSG_TYPE_JOIN_RESPONSE,
            "success": True,
            "term": self.node.get_term(),
            "leader_id": self.node.get_leader_id(),
            "members": [
                self.node.info.to_dict(),
                *[m.to_dict() for m in self.node.get_members().values()]
            ]
        }
        
        await self._send_response(message, response_payload)
        logger.info(f"Processed join request from {node_info.node_id}")
    
    async def _handle_join_response(self, message: Message) -> None:
        """处理加入响应"""
        # 设置响应结果
        correlation_id = message.correlation_id
        if correlation_id in self._pending_responses:
            self._pending_responses[correlation_id].set_result(message.payload)
    
    async def _handle_heartbeat(self, message: Message) -> None:
        """处理心跳"""
        heartbeat_data = message.payload.get("heartbeat", {})
        result = await self.node.handle_heartbeat(heartbeat_data)
        
        # 发送响应
        response_payload = {
            "cluster_msg_type": self.MSG_TYPE_HEARTBEAT_RESPONSE,
            **result
        }
        await self._send_response(message, response_payload)
    
    async def _handle_heartbeat_response(self, message: Message) -> None:
        """处理心跳响应"""
        correlation_id = message.correlation_id
        if correlation_id in self._pending_responses:
            self._pending_responses[correlation_id].set_result(message.payload)
    
    async def _handle_vote_request(self, message: Message) -> None:
        """处理投票请求"""
        request_data = message.payload.get("request", {})
        result = await self.node.handle_vote_request(request_data)
        
        response_payload = {
            "cluster_msg_type": self.MSG_TYPE_VOTE_RESPONSE,
            **result
        }
        await self._send_response(message, response_payload)
    
    async def _handle_vote_response(self, message: Message) -> None:
        """处理投票响应"""
        correlation_id = message.correlation_id
        if correlation_id in self._pending_responses:
            self._pending_responses[correlation_id].set_result(message.payload)
    
    async def _handle_leave_notify(self, message: Message) -> None:
        """处理离开通知"""
        node_id = message.payload.get("node_id")
        if node_id:
            self.node.remove_member(node_id)
            logger.info(f"Processed leave notification from {node_id}")
    
    async def _handle_state_request(self, message: Message) -> None:
        """处理状态请求"""
        if not self.state_store:
            return
        
        result = await self.state_store.handle_state_request(message.payload)
        
        response_payload = {
            "cluster_msg_type": self.MSG_TYPE_STATE_RESPONSE,
            **result
        }
        await self._send_response(message, response_payload)
    
    async def _handle_append_entries(self, message: Message) -> None:
        """处理AppendEntries（Raft）"""
        if not self.state_store:
            return
        
        request = message.payload.get("request", {})
        result = await self.state_store.handle_append_entries(request)
        
        response_payload = {
            "cluster_msg_type": self.MSG_TYPE_APPEND_ENTRIES_RESPONSE,
            **result
        }
        await self._send_response(message, response_payload)
    
    async def _handle_task_assign(self, message: Message) -> None:
        """处理任务分配"""
        task_data = message.payload.get("task_data", {})
        task_id = task_data.get("task_id")
        
        logger.info(f"Received task assignment: {task_id}")
        
        # 触发任务执行（由外部任务处理器执行）
        # 这里只是确认接收
        response_payload = {
            "cluster_msg_type": self.MSG_TYPE_TASK_RESULT,
            "task_id": task_id,
            "accepted": True,
            "node_id": self.node.node_id
        }
        await self._send_response(message, response_payload)
    
    async def _handle_task_result(self, message: Message) -> None:
        """处理任务结果"""
        correlation_id = message.correlation_id
        if correlation_id in self._pending_responses:
            self._pending_responses[correlation_id].set_result(message.payload)
    
    async def _handle_task_migrate(self, message: Message) -> None:
        """处理任务迁移"""
        task_id = message.payload.get("task_id")
        from_node = message.payload.get("from_node")
        to_node = message.payload.get("to_node")
        
        if to_node == self.node.node_id:
            logger.info(f"Task {task_id} migrated to this node from {from_node}")
            # 这里应该接管任务执行
    
    async def _send_response(self, request_message: Message,
                             payload: Dict[str, Any]) -> None:
        """发送响应消息"""
        # 实际发送由ClusterIntegration处理
        # 这里只是准备响应
        pass
    
    def create_request_future(self, correlation_id: str) -> asyncio.Future:
        """创建请求等待Future"""
        future = asyncio.Future()
        self._pending_responses[correlation_id] = future
        
        # 设置超时清理
        async def cleanup():
            await asyncio.sleep(10.0)
            if correlation_id in self._pending_responses:
                if not self._pending_responses[correlation_id].done():
                    self._pending_responses[correlation_id].cancel()
                del self._pending_responses[correlation_id]
        
        asyncio.create_task(cleanup())
        return future


class ClusterIntegration:
    """
    集群集成器
    
    将分布式集群与Phase 5.1的通信协议集成：
    1. 使用CommunicationManager进行消息收发
    2. 配置TCPTransport/WebSocketTransport连接远程节点
    3. 提供统一的集群消息发送接口
    4. 与Crew系统集成
    
    Example:
        # 创建集成器
        integration = ClusterIntegration(node, manager, state_store)
        await integration.start()
        
        # 配置传输层
        integration.configure_tcp_transport(port=8001)
        
        # 连接到其他节点
        await integration.connect_to_node("node2", "192.168.1.2", 8001)
        
        # 发送集群消息
        await integration.send_cluster_message(
            target="node2",
            msg_type=ClusterMessageHandler.MSG_TYPE_HEARTBEAT,
            payload={...}
        )
    """
    
    def __init__(self, cluster_node, cluster_manager,
                 state_store=None, enable_local_transport: bool = True):
        """
        初始化集群集成器
        
        Args:
            cluster_node: ClusterNode实例
            cluster_manager: ClusterManager实例
            state_store: DistributedStateStore实例（可选）
            enable_local_transport: 是否启用本地传输（用于单机测试）
        """
        from kas.core.cluster.node import ClusterNode
        from kas.core.cluster.manager import ClusterManager
        from kas.core.cluster.state import DistributedStateStore
        
        self.node: ClusterNode = cluster_node
        self.manager: ClusterManager = cluster_manager
        self.state_store: Optional[DistributedStateStore] = state_store
        
        # 创建通信管理器
        self.comm_manager = CommunicationManager(cluster_node.node_id)
        
        # 创建消息处理器
        self.msg_handler = ClusterMessageHandler(cluster_node, cluster_manager, state_store)
        
        # 已连接节点（node_id -> transport名称）
        self._node_connections: Dict[str, str] = {}
        
        # 远程节点地址（node_id -> (host, port)）
        self._node_addresses: Dict[str, tuple] = {}
        
        # 是否启用本地传输（单机测试模式）
        self._enable_local_transport = enable_local_transport
        
        # 设置消息回调
        self._setup_callbacks()
        
        logger.info("ClusterIntegration initialized")
    
    def _setup_callbacks(self) -> None:
        """设置回调函数"""
        # 设置ClusterManager的消息发送回调
        self.manager.set_send_message_callback(self._send_message_callback)
        
        # 设置ClusterNode的消息处理回调
        self.node.set_message_handler(self._node_message_callback)
        
        # 注册消息处理器到CommunicationManager
        self.comm_manager.on_any_message(self._on_comm_message)
    
    # ==================== 生命周期管理 ====================
    
    async def start(self) -> bool:
        """启动集群集成"""
        logger.info("Starting ClusterIntegration")
        
        # 配置并启动本地传输（用于进程内通信）
        if self._enable_local_transport:
            self.configure_local_transport()
        
        # 启动通信管理器
        await self.comm_manager.start()
        
        return True
    
    async def stop(self) -> None:
        """停止集群集成"""
        logger.info("Stopping ClusterIntegration")
        
        await self.comm_manager.stop()
    
    # ==================== 传输层配置 ====================
    
    def configure_local_transport(self, priority: int = 0) -> None:
        """配置本地传输（用于单机测试）"""
        config = TransportConfig(
            name="local",
            transport_class=LocalTransport,
            config={},
            priority=priority,
            auto_reconnect=True
        )
        self.comm_manager.add_transport(config)
        logger.info("Local transport configured")
    
    def configure_tcp_transport(self, port: int,
                                 host: str = "0.0.0.0",
                                 priority: int = 1) -> None:
        """
        配置TCP传输
        
        Args:
            port: 监听端口
            host: 监听地址
            priority: 优先级
        """
        config = TransportConfig(
            name="tcp_server",
            transport_class=TCPTransport,
            config={
                "host": host,
                "port": port
            },
            priority=priority,
            auto_reconnect=True
        )
        self.comm_manager.add_transport(config)
        logger.info(f"TCP transport configured on {host}:{port}")
    
    def configure_tcp_client(self, server_host: str, server_port: int,
                             priority: int = 1) -> None:
        """
        配置TCP客户端传输
        
        Args:
            server_host: 服务器地址
            server_port: 服务器端口
            priority: 优先级
        """
        config = TransportConfig(
            name=f"tcp_client_{server_host}_{server_port}",
            transport_class=TCPTransport,
            config={
                "host": "0.0.0.0",
                "port": 0,
                "server_host": server_host,
                "server_port": server_port
            },
            priority=priority,
            auto_reconnect=True
        )
        self.comm_manager.add_transport(config)
        logger.info(f"TCP client configured to connect to {server_host}:{server_port}")
    
    def configure_websocket_transport(self, port: int,
                                       host: str = "0.0.0.0",
                                       priority: int = 1) -> None:
        """
        配置WebSocket传输
        
        Args:
            port: 监听端口
            host: 监听地址
            priority: 优先级
        """
        config = TransportConfig(
            name="websocket_server",
            transport_class=WebSocketTransport,
            config={
                "host": host,
                "port": port
            },
            priority=priority,
            auto_reconnect=True
        )
        self.comm_manager.add_transport(config)
        logger.info(f"WebSocket transport configured on {host}:{port}")
    
    # ==================== 节点连接管理 ====================
    
    async def connect_to_node(self, node_id: str, host: str, port: int) -> bool:
        """
        连接到远程节点
        
        Args:
            node_id: 节点ID
            host: 主机地址
            port: 端口
        
        Returns:
            是否成功连接
        """
        # 记录节点地址
        self._node_addresses[node_id] = (host, port)
        
        # 配置TCP客户端
        transport_name = f"tcp_to_{node_id}"
        config = TransportConfig(
            name=transport_name,
            transport_class=TCPTransport,
            config={
                "host": "0.0.0.0",
                "port": 0,
                "server_host": host,
                "server_port": port
            },
            priority=2,
            auto_reconnect=True
        )
        
        self.comm_manager.add_transport(config)
        
        # 重新初始化连接
        await self.comm_manager._connect_transport(transport_name)
        
        self._node_connections[node_id] = transport_name
        logger.info(f"Connected to node {node_id} at {host}:{port}")
        
        return True
    
    async def disconnect_from_node(self, node_id: str) -> None:
        """断开与节点的连接"""
        if node_id in self._node_connections:
            transport_name = self._node_connections[node_id]
            await self.comm_manager._disconnect_transport(transport_name)
            del self._node_connections[node_id]
            logger.info(f"Disconnected from node {node_id}")
    
    # ==================== 消息发送 ====================
    
    async def send_cluster_message(self, target: str, msg_type: str,
                                    payload: Dict[str, Any],
                                    wait_response: bool = False,
                                    timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        发送集群消息
        
        Args:
            target: 目标节点ID
            msg_type: 消息类型
            payload: 消息负载
            wait_response: 是否等待响应
            timeout: 超时时间
        
        Returns:
            如果wait_response为True，返回响应数据；否则返回None
        """
        # 构建完整payload
        full_payload = {
            "cluster_msg_type": msg_type,
            **payload
        }
        
        # 创建消息
        msg_builder = MessageBuilder()
        msg_builder.from_agent(self.node.node_id)
        msg_builder.to_agent(target)
        
        if msg_type.endswith("_response"):
            msg_builder.response()
        else:
            msg_builder.request()
        
        message = (msg_builder
                   .with_payload(full_payload)
                   .with_priority(MessagePriority.HIGH)
                   .build())
        
        if wait_response:
            # 创建等待Future
            future = self.msg_handler.create_request_future(message.id)
            
            # 发送消息
            await self.comm_manager.send(message)
            
            # 等待响应
            try:
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for response to {msg_type}")
                return None
            except asyncio.CancelledError:
                return None
        else:
            # 直接发送，不等待响应
            await self.comm_manager.send(message)
            return None
    
    async def broadcast_cluster_message(self, msg_type: str,
                                         payload: Dict[str, Any]) -> int:
        """
        广播集群消息到所有节点
        
        Args:
            msg_type: 消息类型
            payload: 消息负载
        
        Returns:
            成功发送的节点数
        """
        full_payload = {
            "cluster_msg_type": msg_type,
            **payload
        }
        
        message = (MessageBuilder()
                   .broadcast()
                   .from_agent(self.node.node_id)
                   .with_payload(full_payload)
                   .with_priority(MessagePriority.NORMAL)
                   .build())
        
        return await self.comm_manager.broadcast(full_payload)
    
    # ==================== 回调函数 ====================
    
    async def _send_message_callback(self, msg_type: str,
                                     data: Dict[str, Any]) -> Any:
        """
        ClusterManager的消息发送回调
        
        这个回调函数被ClusterManager调用，用于发送消息到其他节点。
        """
        target = data.get("target")
        
        if not target:
            # 广播模式
            if msg_type == "task_assign":
                return await self.broadcast_cluster_message(
                    ClusterMessageHandler.MSG_TYPE_TASK_ASSIGN,
                    {"task_data": data.get("task_data")}
                )
            return None
        
        # 点对点发送
        handler_map = {
            "join_request": ClusterMessageHandler.MSG_TYPE_JOIN_REQUEST,
            "heartbeat": ClusterMessageHandler.MSG_TYPE_HEARTBEAT,
            "vote_request": ClusterMessageHandler.MSG_TYPE_VOTE_REQUEST,
            "leave_notify": ClusterMessageHandler.MSG_TYPE_LEAVE_NOTIFY,
            "task_assign": ClusterMessageHandler.MSG_TYPE_TASK_ASSIGN,
            "task_migrate": ClusterMessageHandler.MSG_TYPE_TASK_MIGRATE,
            "state_request": ClusterMessageHandler.MSG_TYPE_STATE_REQUEST,
            "append_entries": ClusterMessageHandler.MSG_TYPE_APPEND_ENTRIES,
        }
        
        cluster_msg_type = handler_map.get(msg_type)
        if not cluster_msg_type:
            logger.warning(f"Unknown message type: {msg_type}")
            return None
        
        # 构建payload
        payload_key = msg_type.replace("_request", "").replace("_notify", "")
        payload = {payload_key: data}
        
        # 发送并等待响应
        response = await self.send_cluster_message(
            target=target,
            msg_type=cluster_msg_type,
            payload=payload,
            wait_response=True,
            timeout=5.0
        )
        
        return response
    
    async def _node_message_callback(self, msg_type: str,
                                     data: Dict[str, Any]) -> Any:
        """
        ClusterNode的消息处理回调
        
        这个回调函数被ClusterNode调用，用于发送协议级别的消息。
        实际处理委托给_send_message_callback。
        """
        return await self._send_message_callback(msg_type, data)
    
    async def _on_comm_message(self, message: Message) -> None:
        """
        CommunicationManager的消息接收回调
        
        处理从Phase 5.1通信层接收到的消息。
        """
        # 检查是否是集群消息
        payload = message.payload
        if not isinstance(payload, dict):
            return
        
        if "cluster_msg_type" not in payload:
            # 不是集群消息，忽略
            return
        
        # 处理集群消息
        await self.msg_handler.handle_message(message)
        
        # 处理响应类型消息
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self.msg_handler._pending_responses:
            future = self.msg_handler._pending_responses.pop(correlation_id)
            if not future.done():
                future.set_result(payload)
    
    # ==================== Crew系统集成 ====================
    
    async def register_crew(self, crew_id: str, capabilities: List[str]) -> bool:
        """
        注册Crew到集群
        
        将当前节点的Crew能力注册到集群中，使得其他节点可以分发任务给这个Crew。
        
        Args:
            crew_id: Crew标识
            capabilities: Crew能力列表
        
        Returns:
            是否成功注册
        """
        # 更新节点能力
        self.node.info.capabilities.extend(capabilities)
        self.node.info.capabilities = list(set(self.node.info.capabilities))  # 去重
        
        # 广播Crew注册信息
        await self.broadcast_cluster_message(
            ClusterMessageHandler.MSG_TYPE_BROADCAST,
            {
                "event": "crew_registered",
                "crew_id": crew_id,
                "node_id": self.node.node_id,
                "capabilities": capabilities
            }
        )
        
        logger.info(f"Crew {crew_id} registered with capabilities: {capabilities}")
        return True
    
    async def find_crew_for_task(self, required_capabilities: List[str]) -> Optional[str]:
        """
        查找能够执行任务的Crew
        
        Args:
            required_capabilities: 需要的Crew能力
        
        Returns:
            节点ID，或None
        """
        # 使用ClusterManager的服务发现功能
        available_nodes = self.manager.discover_services()
        
        for node_info in available_nodes:
            node_caps = set(node_info.get("capabilities", []))
            if set(required_capabilities).issubset(node_caps):
                return node_info["node_id"]
        
        return None
    
    async def distribute_crew_task(self, crew_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分发Crew任务
        
        Args:
            crew_id: 目标Crew ID
            task_data: 任务数据
        
        Returns:
            任务执行结果
        """
        # 查找Crew所在的节点
        target_node = None
        for node_id, info in self.node.get_members().items():
            if crew_id in info.capabilities:
                target_node = node_id
                break
        
        if not target_node:
            return {"success": False, "error": f"Crew {crew_id} not found"}
        
        # 发送任务
        response = await self.send_cluster_message(
            target=target_node,
            msg_type=ClusterMessageHandler.MSG_TYPE_TASK_ASSIGN,
            payload={
                "task_data": {
                    "type": "crew_task",
                    "crew_id": crew_id,
                    **task_data
                }
            },
            wait_response=True,
            timeout=60.0
        )
        
        return response or {"success": False, "error": "No response"}
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "node_id": self.node.node_id,
            "connections": len(self._node_connections),
            "pending_responses": len(self.msg_handler._pending_responses),
            "transport_states": {
                name: state.value
                for name, state in self.comm_manager.get_transport_states().items()
            }
        }
