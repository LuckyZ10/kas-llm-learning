"""
KAS Distributed Agent Cluster - 集群管理器模块

提供集群级别的管理功能，包括节点管理、负载均衡、故障检测和恢复。
"""
import asyncio
import hashlib
import random
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from kas.core.cluster.node import ClusterNode, NodeInfo, NodeState, NodeRole


@dataclass
class LoadBalancerConfig:
    """
    负载均衡配置
    
    Attributes:
        strategy: 负载均衡策略 (round_robin, random, least_load, consistent_hash)
        virtual_nodes: 一致性哈希虚拟节点数
        health_check_interval: 健康检查间隔（秒）
    """
    strategy: str = "consistent_hash"  # 默认使用一致性哈希
    virtual_nodes: int = 150
    health_check_interval: float = 10.0


class ConsistentHashRing:
    """
    一致性哈希环
    
    用于负载均衡和任务分片，确保：
    1. 相同key总是映射到相同节点
    2. 节点加入/离开时最小化数据迁移
    """
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}  # hash -> node_id
        self.nodes: Dict[str, Set[int]] = {}  # node_id -> set of hash values
        self.sorted_keys: List[int] = []
    
    def add_node(self, node_id: str) -> None:
        """添加节点到哈希环"""
        if node_id in self.nodes:
            return
        
        hash_values = set()
        for i in range(self.virtual_nodes):
            # 为每个虚拟节点计算hash
            key = f"{node_id}:{i}"
            hash_val = self._hash(key)
            self.ring[hash_val] = node_id
            hash_values.add(hash_val)
        
        self.nodes[node_id] = hash_values
        self.sorted_keys = sorted(self.ring.keys())
        logger.debug(f"Added node {node_id} to hash ring")
    
    def remove_node(self, node_id: str) -> None:
        """从哈希环移除节点"""
        if node_id not in self.nodes:
            return
        
        for hash_val in self.nodes[node_id]:
            del self.ring[hash_val]
        
        del self.nodes[node_id]
        self.sorted_keys = sorted(self.ring.keys())
        logger.debug(f"Removed node {node_id} from hash ring")
    
    def get_node(self, key: str) -> Optional[str]:
        """获取key对应的节点"""
        if not self.ring:
            return None
        
        hash_val = self._hash(key)
        
        # 找到第一个大于等于hash_val的节点
        for key_hash in self.sorted_keys:
            if key_hash >= hash_val:
                return self.ring[key_hash]
        
        # 如果没有找到，返回第一个节点（环形）
        return self.ring[self.sorted_keys[0]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """获取key对应的多个节点（用于副本）"""
        if not self.ring or count <= 0:
            return []
        
        hash_val = self._hash(key)
        nodes = []
        seen = set()
        
        # 从hash_val位置开始遍历
        start_idx = 0
        for i, key_hash in enumerate(self.sorted_keys):
            if key_hash >= hash_val:
                start_idx = i
                break
        
        # 收集不重复的节点
        for i in range(len(self.sorted_keys)):
            idx = (start_idx + i) % len(self.sorted_keys)
            node_id = self.ring[self.sorted_keys[idx]]
            if node_id not in seen:
                seen.add(node_id)
                nodes.append(node_id)
                if len(nodes) >= count:
                    break
        
        return nodes
    
    def _hash(self, key: str) -> int:
        """计算key的hash值"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def get_all_nodes(self) -> List[str]:
        """获取所有节点"""
        return list(self.nodes.keys())


class ClusterManager:
    """
    集群管理器
    
    管理整个Agent集群，负责：
    1. 节点加入/离开管理
    2. 负载均衡（多种策略）
    3. 故障检测和自动恢复
    4. 服务发现
    5. 任务分发
    
    Example:
        manager = ClusterManager(node)
        await manager.start()
        
        # 分发任务
        target_node = manager.select_node_for_task("task-123", {"cpu": 50})
        await manager.distribute_task(task_data, target_node)
    """
    
    def __init__(self, node: ClusterNode, 
                 lb_config: Optional[LoadBalancerConfig] = None):
        """
        初始化集群管理器
        
        Args:
            node: 本节点实例
            lb_config: 负载均衡配置
        """
        self.node = node
        self.lb_config = lb_config or LoadBalancerConfig()
        
        # 一致性哈希环
        self._hash_ring = ConsistentHashRing(self.lb_config.virtual_nodes)
        
        # 轮询计数器
        self._round_robin_counter = 0
        
        # 任务分配记录
        self._task_assignments: Dict[str, str] = {}  # task_id -> node_id
        
        # 故障恢复任务
        self._recovery_tasks: Dict[str, asyncio.Task] = {}
        
        # 节点健康状态
        self._node_health: Dict[str, Dict[str, Any]] = {}
        
        # 故障检测任务
        self._failure_detection_task: Optional[asyncio.Task] = None
        
        # 运行状态
        self._running = False
        self._stop_event = asyncio.Event()
        
        # 消息发送回调
        self._send_message_callback: Optional[Callable] = None
        
        # 注册节点事件回调
        self._setup_node_callbacks()
        
        logger.info(f"ClusterManager initialized for node {node.node_id}")
    
    def _setup_node_callbacks(self) -> None:
        """设置节点事件回调"""
        self.node.on("on_member_join", self._on_member_join)
        self.node.on("on_member_leave", self._on_member_leave)
        self.node.on("on_leader_change", self._on_leader_change)
    
    # ==================== 生命周期管理 ====================
    
    async def start(self) -> bool:
        """启动集群管理器"""
        if self._running:
            return True
        
        logger.info("Starting ClusterManager")
        self._running = True
        
        # 设置节点消息处理器
        self.node.set_message_handler(self._handle_node_message)
        
        # 启动故障检测
        self._failure_detection_task = asyncio.create_task(
            self._failure_detection_loop()
        )
        
        # 将本节点添加到哈希环
        self._hash_ring.add_node(self.node.node_id)
        
        # 添加已知成员到哈希环
        for member_id in self.node.get_members():
            self._hash_ring.add_node(member_id)
        
        return True
    
    async def stop(self) -> None:
        """停止集群管理器"""
        if not self._running:
            return
        
        logger.info("Stopping ClusterManager")
        self._running = False
        self._stop_event.set()
        
        if self._failure_detection_task:
            self._failure_detection_task.cancel()
        
        # 取消所有恢复任务
        for task in self._recovery_tasks.values():
            task.cancel()
        
        logger.info("ClusterManager stopped")
    
    # ==================== 负载均衡 ====================
    
    def select_node_for_task(self, task_id: str, 
                             requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        为任务选择节点
        
        Args:
            task_id: 任务标识
            requirements: 任务要求（如CPU、内存等）
        
        Returns:
            选中的节点ID，或None
        """
        strategy = self.lb_config.strategy
        available_nodes = self._get_available_nodes(requirements)
        
        if not available_nodes:
            logger.warning("No available nodes for task")
            return None
        
        if strategy == "consistent_hash":
            # 使用一致性哈希
            node_id = self._hash_ring.get_node(task_id)
            if node_id in available_nodes:
                return node_id
            # 如果hash到的节点不可用，使用备选
            return random.choice(list(available_nodes))
        
        elif strategy == "round_robin":
            # 轮询
            nodes = list(available_nodes)
            idx = self._round_robin_counter % len(nodes)
            self._round_robin_counter += 1
            return nodes[idx]
        
        elif strategy == "random":
            # 随机
            return random.choice(list(available_nodes))
        
        elif strategy == "least_load":
            # 最小负载
            return self._select_least_loaded(available_nodes)
        
        else:
            # 默认使用一致性哈希
            return self._hash_ring.get_node(task_id)
    
    def select_nodes_for_sharding(self, task_id: str, shard_count: int) -> List[str]:
        """
        为任务分片选择多个节点
        
        Args:
            task_id: 任务标识
            shard_count: 分片数量
        
        Returns:
            节点ID列表
        """
        available_nodes = self._get_available_nodes()
        
        if not available_nodes:
            return []
        
        if len(available_nodes) <= shard_count:
            return list(available_nodes)
        
        # 使用一致性哈希选择多个节点
        nodes = self._hash_ring.get_nodes(task_id, shard_count)
        
        # 过滤掉不可用的节点
        valid_nodes = [n for n in nodes if n in available_nodes]
        
        # 如果不够，随机补充
        while len(valid_nodes) < shard_count and len(valid_nodes) < len(available_nodes):
            remaining = available_nodes - set(valid_nodes)
            valid_nodes.append(random.choice(list(remaining)))
        
        return valid_nodes[:shard_count]
    
    def _get_available_nodes(self, requirements: Optional[Dict[str, Any]] = None) -> Set[str]:
        """获取可用节点"""
        available = set()
        
        # 添加本节点（如果是活跃的）
        if self.node.info.state == NodeState.ACTIVE:
            available.add(self.node.node_id)
        
        # 添加活跃成员
        for node_id, info in self.node.get_active_members().items():
            if info.state == NodeState.ACTIVE:
                # 检查是否满足要求
                if self._meets_requirements(info, requirements):
                    available.add(node_id)
        
        return available
    
    def _meets_requirements(self, node_info: NodeInfo, 
                            requirements: Optional[Dict[str, Any]]) -> bool:
        """检查节点是否满足任务要求"""
        if not requirements:
            return True
        
        # 检查CPU负载
        if "max_load" in requirements:
            if node_info.load > requirements["max_load"]:
                return False
        
        # 检查能力要求
        if "capabilities" in requirements:
            required_caps = set(requirements["capabilities"])
            node_caps = set(node_info.capabilities)
            if not required_caps.issubset(node_caps):
                return False
        
        return True
    
    def _select_least_loaded(self, available_nodes: Set[str]) -> Optional[str]:
        """选择负载最小的节点"""
        best_node = None
        best_load = float('inf')
        
        for node_id in available_nodes:
            if node_id == self.node.node_id:
                load = self.node.info.load
            else:
                member = self.node.get_members().get(node_id)
                load = member.load if member else float('inf')
            
            if load < best_load:
                best_load = load
                best_node = node_id
        
        return best_node
    
    # ==================== 任务分发 ====================
    
    async def distribute_task(self, task_data: Dict[str, Any],
                              target_node: Optional[str] = None) -> Dict[str, Any]:
        """
        分发任务到指定节点
        
        Args:
            task_data: 任务数据
            target_node: 目标节点（None则自动选择）
        
        Returns:
            分发结果
        """
        task_id = task_data.get("task_id", str(random.uuid4()))
        
        if target_node is None:
            target_node = self.select_node_for_task(task_id, task_data.get("requirements"))
        
        if not target_node:
            return {
                "success": False,
                "error": "No available node for task"
            }
        
        # 记录任务分配
        self._task_assignments[task_id] = target_node
        
        # 发送任务
        try:
            if self._send_message_callback:
                result = await self._send_message_callback("task_assign", {
                    "task_id": task_id,
                    "task_data": task_data,
                    "target": target_node,
                    "source": self.node.node_id
                })
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "assigned_node": target_node,
                    "result": result
                }
            else:
                return {
                    "success": False,
                    "error": "No message handler configured"
                }
        
        except Exception as e:
            logger.error(f"Failed to distribute task {task_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def broadcast_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        广播任务到所有节点
        
        Args:
            task_data: 任务数据
        
        Returns:
            各节点的响应
        """
        results = {}
        tasks = []
        
        # 收集所有可用节点
        all_nodes = self._get_available_nodes()
        
        for node_id in all_nodes:
            tasks.append(self._send_task_to_node(task_data, node_id))
        
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for node_id, response in zip(all_nodes, responses):
                if isinstance(response, Exception):
                    results[node_id] = {"success": False, "error": str(response)}
                else:
                    results[node_id] = response
        
        return {
            "success": True,
            "results": results,
            "total_nodes": len(all_nodes),
            "success_count": sum(1 for r in results.values() if r.get("success"))
        }
    
    async def _send_task_to_node(self, task_data: Dict[str, Any],
                                  node_id: str) -> Dict[str, Any]:
        """发送任务到指定节点"""
        if self._send_message_callback:
            return await self._send_message_callback("task_assign", {
                "task_data": task_data,
                "target": node_id,
                "source": self.node.node_id
            })
        return {"success": False, "error": "No message handler"}
    
    # ==================== 故障检测与恢复 ====================
    
    async def _failure_detection_loop(self) -> None:
        """故障检测循环"""
        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.lb_config.health_check_interval
                )
            except asyncio.TimeoutError:
                await self._check_node_health()
    
    async def _check_node_health(self) -> None:
        """检查节点健康状态"""
        now = datetime.now()
        
        for node_id, info in list(self.node.get_members().items()):
            # 获取或创建健康记录
            if node_id not in self._node_health:
                self._node_health[node_id] = {
                    "failures": 0,
                    "last_check": now,
                    "status": "healthy"
                }
            
            health = self._node_health[node_id]
            
            # 检查是否健康
            if info.is_healthy(self.node.config.node_timeout):
                if health["status"] != "healthy":
                    logger.info(f"Node {node_id} is now healthy")
                    health["status"] = "healthy"
                    health["failures"] = 0
            else:
                health["failures"] += 1
                health["status"] = "suspected"
                
                logger.warning(
                    f"Node {node_id} suspected failed "
                    f"({health['failures']} failures)"
                )
                
                # 连续多次失败，标记为失败并触发恢复
                if health["failures"] >= 3:
                    await self._handle_node_failure(node_id)
    
    async def _handle_node_failure(self, node_id: str) -> None:
        """处理节点故障"""
        logger.error(f"Node {node_id} marked as failed")
        
        # 更新健康状态
        if node_id in self._node_health:
            self._node_health[node_id]["status"] = "failed"
        
        # 从哈希环移除
        self._hash_ring.remove_node(node_id)
        
        # 迁移任务
        await self._migrate_tasks_from_failed_node(node_id)
        
        # 从成员列表移除
        self.node.remove_member(node_id)
    
    async def _migrate_tasks_from_failed_node(self, failed_node: str) -> None:
        """从故障节点迁移任务"""
        # 找出分配给故障节点的任务
        tasks_to_migrate = [
            task_id for task_id, node_id in self._task_assignments.items()
            if node_id == failed_node
        ]
        
        if not tasks_to_migrate:
            return
        
        logger.info(f"Migrating {len(tasks_to_migrate)} tasks from {failed_node}")
        
        for task_id in tasks_to_migrate:
            # 重新选择节点
            new_node = self.select_node_for_task(task_id)
            
            if new_node:
                logger.info(f"Migrating task {task_id} to {new_node}")
                self._task_assignments[task_id] = new_node
                
                # 通知新节点接管任务
                if self._send_message_callback:
                    try:
                        await self._send_message_callback("task_migrate", {
                            "task_id": task_id,
                            "from_node": failed_node,
                            "to_node": new_node
                        })
                    except Exception as e:
                        logger.error(f"Failed to migrate task {task_id}: {e}")
            else:
                logger.error(f"No available node for task {task_id}")
    
    # ==================== 事件处理 ====================
    
    def _on_member_join(self, node_info: NodeInfo) -> None:
        """处理成员加入"""
        logger.info(f"Member joined: {node_info.node_id}")
        self._hash_ring.add_node(node_info.node_id)
        
        # 如果是Leader，发送集群状态给新成员
        if self.node.is_leader():
            asyncio.create_task(self._send_cluster_state(node_info.node_id))
    
    def _on_member_leave(self, node_info: NodeInfo) -> None:
        """处理成员离开"""
        logger.info(f"Member left: {node_info.node_id}")
        self._hash_ring.remove_node(node_info.node_id)
        
        if node_info.node_id in self._node_health:
            del self._node_health[node_info.node_id]
    
    def _on_leader_change(self, leader_id: str) -> None:
        """处理Leader变更"""
        logger.info(f"Leader changed to: {leader_id}")
        
        # 如果是新Leader，重新平衡任务
        if leader_id == self.node.node_id:
            asyncio.create_task(self._rebalance_tasks())
    
    async def _send_cluster_state(self, node_id: str) -> None:
        """发送集群状态给新成员"""
        state = {
            "members": [m.to_dict() for m in self.node.get_members().values()],
            "task_assignments": self._task_assignments,
            "timestamp": datetime.now().isoformat()
        }
        
        if self._send_message_callback:
            try:
                await self._send_message_callback("cluster_state", {
                    "state": state,
                    "target": node_id
                })
            except Exception as e:
                logger.error(f"Failed to send cluster state: {e}")
    
    async def _rebalance_tasks(self) -> None:
        """重新平衡任务分配"""
        logger.info("Rebalancing tasks")
        
        # 获取所有可用节点
        available = self._get_available_nodes()
        
        # 重新分配任务
        for task_id, node_id in list(self._task_assignments.items()):
            if node_id not in available:
                # 重新选择节点
                new_node = self.select_node_for_task(task_id)
                if new_node:
                    self._task_assignments[task_id] = new_node
                    logger.info(f"Reassigned task {task_id} to {new_node}")
    
    # ==================== 消息处理 ====================
    
    async def _handle_node_message(self, msg_type: str, data: Dict[str, Any]) -> Any:
        """处理节点消息"""
        if msg_type == "join_request":
            return await self._handle_join_request(data)
        elif msg_type == "heartbeat":
            return await self.node.handle_heartbeat(data)
        elif msg_type == "vote_request":
            return await self.node.handle_vote_request(data)
        elif msg_type == "leave_notify":
            return await self._handle_leave_notify(data)
        elif msg_type == "health_check":
            return await self._handle_health_check()
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            return None
    
    async def _handle_join_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理加入请求"""
        node_info_data = data.get("node_info", {})
        node_info = NodeInfo.from_dict(node_info_data)
        
        # 添加成员
        self.node.add_member(node_info)
        self._hash_ring.add_node(node_info.node_id)
        
        logger.info(f"Accepted join request from {node_info.node_id}")
        
        return {
            "success": True,
            "term": self.node.get_term(),
            "leader_id": self.node.get_leader_id(),
            "members": [
                self.node.info.to_dict(),
                *[m.to_dict() for m in self.node.get_members().values()]
            ]
        }
    
    async def _handle_leave_notify(self, data: Dict[str, Any]) -> None:
        """处理离开通知"""
        node_id = data.get("node_id")
        if node_id:
            self.node.remove_member(node_id)
            self._hash_ring.remove_node(node_id)
    
    async def _handle_health_check(self) -> Dict[str, Any]:
        """处理健康检查"""
        return {
            "node_id": self.node.node_id,
            "state": self.node.info.state.value,
            "role": self.node.info.role.value,
            "load": self.node.info.load,
            "timestamp": datetime.now().isoformat()
        }
    
    def set_send_message_callback(self, callback: Callable) -> None:
        """设置消息发送回调"""
        self._send_message_callback = callback
    
    # ==================== 服务发现 ====================
    
    def discover_services(self, service_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        发现服务
        
        Args:
            service_type: 服务类型过滤器
        
        Returns:
            服务列表
        """
        services = []
        
        # 添加本节点
        if self.node.info.state == NodeState.ACTIVE:
            if service_type is None or service_type in self.node.info.capabilities:
                services.append({
                    "node_id": self.node.node_id,
                    "address": self.node.info.address(),
                    "capabilities": self.node.info.capabilities,
                    "load": self.node.info.load
                })
        
        # 添加其他成员
        for node_id, info in self.node.get_active_members().items():
            if service_type is None or service_type in info.capabilities:
                services.append({
                    "node_id": node_id,
                    "address": info.address(),
                    "capabilities": info.capabilities,
                    "load": info.load
                })
        
        return services
    
    def get_node_for_service(self, service_type: str) -> Optional[str]:
        """
        获取提供指定服务的节点
        
        Args:
            service_type: 服务类型
        
        Returns:
            节点ID或None
        """
        available = []
        
        # 检查本节点
        if service_type in self.node.info.capabilities:
            available.append((self.node.node_id, self.node.info.load))
        
        # 检查其他成员
        for node_id, info in self.node.get_active_members().items():
            if service_type in info.capabilities:
                available.append((node_id, info.load))
        
        if not available:
            return None
        
        # 选择负载最小的
        available.sort(key=lambda x: x[1])
        return available[0][0]
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取集群管理器统计信息"""
        return {
            "node_id": self.node.node_id,
            "is_leader": self.node.is_leader(),
            "members": len(self.node.get_members()),
            "active_members": len(self.node.get_active_members()),
            "hash_ring_nodes": len(self._hash_ring.get_all_nodes()),
            "task_assignments": len(self._task_assignments),
            "lb_strategy": self.lb_config.strategy
        }
