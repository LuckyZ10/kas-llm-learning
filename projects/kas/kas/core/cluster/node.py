"""
KAS Distributed Agent Cluster - 集群节点管理模块

提供集群节点的生命周期管理，包括节点注册、发现、心跳和Leader选举。
"""
import asyncio
import time
import uuid
import hashlib
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """节点状态"""
    JOINING = "joining"       # 正在加入集群
    ACTIVE = "active"         # 活跃状态
    SUSPECT = "suspect"       # 疑似故障
    INACTIVE = "inactive"     # 不活跃
    LEAVING = "leaving"       # 正在离开
    LEFT = "left"             # 已离开


class NodeRole(Enum):
    """节点角色"""
    FOLLOWER = "follower"     # 跟随者
    CANDIDATE = "candidate"   # 候选人
    LEADER = "leader"         # 领导者


@dataclass
class NodeInfo:
    """
    节点信息
    
    Attributes:
        node_id: 节点唯一标识
        host: 主机地址
        port: 端口
        metadata: 节点元数据
        state: 节点状态
        role: 节点角色
        joined_at: 加入时间
        last_heartbeat: 最后心跳时间
        version: 节点版本
        capabilities: 节点能力列表
        load: 当前负载（0-100）
    """
    node_id: str
    host: str
    port: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: NodeState = NodeState.JOINING
    role: NodeRole = NodeRole.FOLLOWER
    joined_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)
    load: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "metadata": self.metadata,
            "state": self.state.value,
            "role": self.role.value,
            "joined_at": self.joined_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "version": self.version,
            "capabilities": self.capabilities,
            "load": self.load
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        """从字典创建"""
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            metadata=data.get("metadata", {}),
            state=NodeState(data.get("state", "joining")),
            role=NodeRole(data.get("role", "follower")),
            joined_at=datetime.fromisoformat(data["joined_at"]) if data.get("joined_at") else datetime.now(),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]) if data.get("last_heartbeat") else datetime.now(),
            version=data.get("version", "1.0.0"),
            capabilities=data.get("capabilities", []),
            load=data.get("load", 0.0)
        )
    
    def is_healthy(self, timeout_seconds: float = 30.0) -> bool:
        """检查节点是否健康"""
        if self.state not in [NodeState.ACTIVE, NodeState.JOINING]:
            return False
        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        return elapsed < timeout_seconds
    
    def address(self) -> str:
        """获取节点地址"""
        return f"{self.host}:{self.port}"


@dataclass
class ClusterConfig:
    """
    集群配置
    
    Attributes:
        cluster_name: 集群名称
        heartbeat_interval: 心跳间隔（秒）
        heartbeat_timeout: 心跳超时（秒）
        election_timeout_min: 选举超时最小值（毫秒）
        election_timeout_max: 选举超时最大值（毫秒）
        node_timeout: 节点超时时间（秒）
        retry_interval: 重试间隔（秒）
        max_retries: 最大重试次数
    """
    cluster_name: str = "kas-cluster"
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 15.0
    election_timeout_min: int = 150
    election_timeout_max: int = 300
    node_timeout: float = 30.0
    retry_interval: float = 3.0
    max_retries: int = 5


class ClusterNode:
    """
    集群节点
    
    表示集群中的一个节点，负责：
    1. 节点生命周期管理（加入/离开）
    2. 心跳维护
    3. Leader选举（Raft算法简化版）
    4. 成员发现
    5. 状态同步
    
    Example:
        config = ClusterConfig(cluster_name="my-cluster")
        node = ClusterNode("node1", "localhost", 8001, config)
        await node.start()
        await node.join_cluster(["localhost:8000"])  # 加入现有集群
    """
    
    def __init__(self, node_id: str, host: str, port: int,
                 config: Optional[ClusterConfig] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        初始化集群节点
        
        Args:
            node_id: 节点唯一标识
            host: 主机地址
            port: 服务端口
            config: 集群配置
            metadata: 节点元数据
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.config = config or ClusterConfig()
        self.metadata = metadata or {}
        
        # 节点信息
        self.info = NodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            metadata=self.metadata,
            state=NodeState.JOINING,
            role=NodeRole.FOLLOWER
        )
        
        # 集群成员（node_id -> NodeInfo）
        self._members: Dict[str, NodeInfo] = {}
        
        # 当前Leader
        self._leader_id: Optional[str] = None
        
        # 任期（Raft）
        self._term: int = 0
        
        # 投票状态
        self._voted_for: Optional[str] = None
        
        # 选举定时器
        self._election_timer: Optional[asyncio.Task] = None
        self._election_timeout: float = 0.0
        
        # 心跳任务
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # 健康检查任务
        self._health_check_task: Optional[asyncio.Task] = None
        
        # 事件回调
        self._callbacks: Dict[str, List[Callable]] = {
            "on_leader_change": [],
            "on_member_join": [],
            "on_member_leave": [],
            "on_state_change": [],
            "on_heartbeat": []
        }
        
        # 运行状态
        self._running = False
        self._stop_event = asyncio.Event()
        
        # 消息处理器（由ClusterManager设置）
        self._message_handler: Optional[Callable] = None
        
        logger.info(f"ClusterNode initialized: {node_id} at {host}:{port}")
    
    # ==================== 生命周期管理 ====================
    
    async def start(self) -> bool:
        """启动节点"""
        if self._running:
            return True
        
        logger.info(f"Starting node {self.node_id}")
        self._running = True
        self.info.state = NodeState.JOINING
        
        # 启动健康检查
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # 启动选举定时器（如果是第一个节点，将成为Leader）
        self._reset_election_timer()
        
        return True
    
    async def stop(self) -> None:
        """停止节点"""
        if not self._running:
            return
        
        logger.info(f"Stopping node {self.node_id}")
        self._running = False
        self._stop_event.set()
        self.info.state = NodeState.LEAVING
        
        # 通知其他节点离开
        await self._notify_leave()
        
        # 取消任务
        if self._election_timer:
            self._election_timer.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        
        self.info.state = NodeState.LEFT
        logger.info(f"Node {self.node_id} stopped")
    
    async def join_cluster(self, seed_nodes: List[str]) -> bool:
        """
        加入现有集群
        
        Args:
            seed_nodes: 种子节点地址列表 ["host:port", ...]
        
        Returns:
            是否成功加入
        """
        if not seed_nodes:
            # 没有种子节点，作为第一个节点启动（将成为Leader）
            logger.info(f"No seed nodes, starting as first node")
            await self._become_leader()
            return True
        
        for seed in seed_nodes:
            try:
                logger.info(f"Attempting to join via seed: {seed}")
                if await self._request_join(seed):
                    return True
            except Exception as e:
                logger.warning(f"Failed to join via {seed}: {e}")
        
        logger.error("Failed to join cluster via any seed node")
        return False
    
    async def leave_cluster(self) -> None:
        """离开集群"""
        await self.stop()
    
    # ==================== Leader选举（Raft简化版） ====================
    
    def is_leader(self) -> bool:
        """检查是否为Leader"""
        return self.info.role == NodeRole.LEADER
    
    def get_leader_id(self) -> Optional[str]:
        """获取当前Leader的ID"""
        return self._leader_id
    
    def get_term(self) -> int:
        """获取当前任期"""
        return self._term
    
    async def _become_leader(self) -> None:
        """成为Leader"""
        logger.info(f"Node {self.node_id} becoming LEADER for term {self._term}")
        
        self.info.role = NodeRole.LEADER
        self._leader_id = self.node_id
        self.info.state = NodeState.ACTIVE
        
        # 取消选举定时器
        if self._election_timer:
            self._election_timer.cancel()
        
        # 启动心跳发送
        self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
        
        # 触发回调
        await self._trigger_callback("on_leader_change", self.node_id)
    
    async def _become_follower(self, term: int, leader_id: Optional[str] = None) -> None:
        """成为Follower"""
        if term > self._term:
            self._term = term
            self._voted_for = None
        
        was_leader = self.is_leader()
        self.info.role = NodeRole.FOLLOWER
        self._leader_id = leader_id
        self.info.state = NodeState.ACTIVE
        
        # 停止心跳发送（如果是Leader降级）
        if was_leader and self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        # 重置选举定时器
        self._reset_election_timer()
        
        logger.info(f"Node {self.node_id} became FOLLOWER for term {term}")
        
        if leader_id:
            await self._trigger_callback("on_leader_change", leader_id)
    
    async def _become_candidate(self) -> None:
        """成为候选人并开始选举"""
        self._term += 1
        self.info.role = NodeRole.CANDIDATE
        self._voted_for = self.node_id
        
        logger.info(f"Node {self.node_id} became CANDIDATE for term {self._term}")
        
        # 请求投票
        votes = 1  # 自己的票
        votes_needed = (len(self._members) + 1) // 2 + 1
        
        vote_requests = []
        for member_id, member_info in self._members.items():
            vote_requests.append(self._request_vote(member_info))
        
        if vote_requests:
            results = await asyncio.gather(*vote_requests, return_exceptions=True)
            for result in results:
                if result is True:
                    votes += 1
        
        # 检查是否获得多数票
        if votes >= votes_needed:
            await self._become_leader()
        else:
            # 选举失败，重新开始
            self._reset_election_timer()
    
    def _reset_election_timer(self) -> None:
        """重置选举定时器"""
        if self._election_timer:
            self._election_timer.cancel()
        
        # 随机超时，避免同时发起选举
        import random
        timeout_ms = random.randint(
            self.config.election_timeout_min,
            self.config.election_timeout_max
        )
        self._election_timeout = timeout_ms / 1000.0
        
        self._election_timer = asyncio.create_task(self._election_timeout_handler())
    
    async def _election_timeout_handler(self) -> None:
        """选举超时处理"""
        try:
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=self._election_timeout
            )
        except asyncio.TimeoutError:
            if self._running and not self.is_leader():
                logger.info(f"Election timeout, starting new election")
                await self._become_candidate()
    
    # ==================== 心跳机制 ====================
    
    async def _send_heartbeats(self) -> None:
        """Leader发送心跳"""
        while self._running and self.is_leader():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.heartbeat_interval
                )
            except asyncio.TimeoutError:
                # 发送心跳给所有成员
                heartbeat_tasks = []
                for member_id, member_info in self._members.items():
                    if member_info.state == NodeState.ACTIVE:
                        heartbeat_tasks.append(self._send_heartbeat(member_info))
                
                if heartbeat_tasks:
                    await asyncio.gather(*heartbeat_tasks, return_exceptions=True)
    
    async def _send_heartbeat(self, member: NodeInfo) -> bool:
        """发送心跳到指定节点"""
        try:
            # 通过消息处理器发送（由ClusterManager实现）
            if self._message_handler:
                await self._message_handler("heartbeat", {
                    "term": self._term,
                    "leader_id": self.node_id,
                    "leader_commit": 0,  # 简化版，不实现完整Raft
                    "target": member.node_id
                })
            return True
        except Exception as e:
            logger.warning(f"Failed to send heartbeat to {member.node_id}: {e}")
            return False
    
    async def handle_heartbeat(self, heartbeat: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理收到的心跳
        
        Returns:
            响应信息
        """
        term = heartbeat.get("term", 0)
        leader_id = heartbeat.get("leader_id")
        
        # 更新最后心跳时间
        self.info.last_heartbeat = datetime.now()
        
        if term > self._term:
            # 发现更高任期，转为Follower
            await self._become_follower(term, leader_id)
        elif term == self._term:
            if self.is_leader() and leader_id != self.node_id:
                # 出现两个Leader，转为Follower
                await self._become_follower(term, leader_id)
            elif leader_id != self._leader_id:
                self._leader_id = leader_id
                await self._trigger_callback("on_leader_change", leader_id)
        
        # 重置选举定时器
        self._reset_election_timer()
        
        await self._trigger_callback("on_heartbeat", heartbeat)
        
        return {
            "success": True,
            "term": self._term,
            "node_id": self.node_id,
            "match_index": 0  # 简化版
        }
    
    # ==================== 成员管理 ====================
    
    def get_members(self) -> Dict[str, NodeInfo]:
        """获取所有集群成员"""
        return self._members.copy()
    
    def get_active_members(self) -> Dict[str, NodeInfo]:
        """获取活跃成员"""
        return {
            node_id: info
            for node_id, info in self._members.items()
            if info.is_healthy(self.config.node_timeout)
        }
    
    def get_member_count(self) -> int:
        """获取成员数量（包括自己）"""
        return len(self._members) + 1
    
    def add_member(self, node_info: NodeInfo) -> None:
        """添加成员"""
        if node_info.node_id != self.node_id:
            is_new = node_info.node_id not in self._members
            self._members[node_info.node_id] = node_info
            
            if is_new:
                logger.info(f"Member added: {node_info.node_id}")
                asyncio.create_task(self._trigger_callback("on_member_join", node_info))
    
    def remove_member(self, node_id: str) -> None:
        """移除成员"""
        if node_id in self._members:
            node_info = self._members.pop(node_id)
            logger.info(f"Member removed: {node_id}")
            asyncio.create_task(self._trigger_callback("on_member_leave", node_info))
            
            # 如果是Leader，需要重新选举
            if node_id == self._leader_id:
                self._leader_id = None
                self._reset_election_timer()
    
    def update_member_heartbeat(self, node_id: str) -> None:
        """更新成员心跳"""
        if node_id in self._members:
            self._members[node_id].last_heartbeat = datetime.now()
    
    async def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.heartbeat_interval
                )
            except asyncio.TimeoutError:
                now = datetime.now()
                timeout = timedelta(seconds=self.config.node_timeout)
                
                # 检查成员健康状态
                for node_id, info in list(self._members.items()):
                    if now - info.last_heartbeat > timeout:
                        logger.warning(f"Node {node_id} suspected failed")
                        info.state = NodeState.SUSPECT
                        
                        # 如果是Leader失联，触发选举
                        if node_id == self._leader_id:
                            self._leader_id = None
                            if not self.is_leader():
                                self._reset_election_timer()
    
    # ==================== 网络请求处理 ====================
    
    async def _request_join(self, seed_address: str) -> bool:
        """请求加入集群"""
        # 通过消息处理器发送加入请求
        if self._message_handler:
            try:
                response = await self._message_handler("join_request", {
                    "node_info": self.info.to_dict(),
                    "target_address": seed_address
                })
                
                if response and response.get("success"):
                    # 更新成员列表
                    members = response.get("members", [])
                    for member_data in members:
                        if member_data["node_id"] != self.node_id:
                            self.add_member(NodeInfo.from_dict(member_data))
                    
                    # 更新Leader信息
                    leader_id = response.get("leader_id")
                    if leader_id:
                        await self._become_follower(response.get("term", 0), leader_id)
                    
                    self.info.state = NodeState.ACTIVE
                    logger.info(f"Successfully joined cluster via {seed_address}")
                    return True
            except Exception as e:
                logger.warning(f"Join request failed: {e}")
        
        return False
    
    async def _request_vote(self, member: NodeInfo) -> bool:
        """请求投票"""
        if self._message_handler:
            try:
                response = await self._message_handler("vote_request", {
                    "term": self._term,
                    "candidate_id": self.node_id,
                    "last_log_index": 0,
                    "last_log_term": 0,
                    "target": member.node_id
                })
                
                if response and response.get("vote_granted"):
                    return True
            except Exception as e:
                logger.warning(f"Vote request to {member.node_id} failed: {e}")
        
        return False
    
    async def handle_vote_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理投票请求"""
        term = request.get("term", 0)
        candidate_id = request.get("candidate_id")
        
        vote_granted = False
        
        if term > self._term:
            # 更新任期并转为Follower
            self._term = term
            self._voted_for = None
            await self._become_follower(term)
        
        if term == self._term:
            if self._voted_for is None or self._voted_for == candidate_id:
                # 检查候选人的日志是否至少和自己一样新（简化版，总是同意）
                self._voted_for = candidate_id
                vote_granted = True
                self._reset_election_timer()
        
        return {
            "term": self._term,
            "vote_granted": vote_granted
        }
    
    async def _notify_leave(self) -> None:
        """通知其他节点离开"""
        if self._message_handler:
            for member_id in self._members:
                try:
                    await self._message_handler("leave_notify", {
                        "node_id": self.node_id,
                        "target": member_id
                    })
                except Exception as e:
                    logger.warning(f"Failed to notify {member_id}: {e}")
    
    # ==================== 事件回调 ====================
    
    def on(self, event: str, callback: Callable) -> None:
        """注册事件回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def off(self, event: str, callback: Callable) -> None:
        """取消事件回调"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    async def _trigger_callback(self, event: str, data: Any) -> None:
        """触发事件回调"""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def set_message_handler(self, handler: Callable) -> None:
        """设置消息处理器（由ClusterManager调用）"""
        self._message_handler = handler
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取节点统计信息"""
        return {
            "node_id": self.node_id,
            "state": self.info.state.value,
            "role": self.info.role.value,
            "term": self._term,
            "leader_id": self._leader_id,
            "members": len(self._members),
            "active_members": len(self.get_active_members()),
            "is_leader": self.is_leader()
        }
