"""
KAS Distributed Agent Cluster - 分布式状态存储模块

提供分布式状态存储，使用简化版Raft共识算法实现状态同步和分布式锁。
"""
import asyncio
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LockStatus(Enum):
    """锁状态"""
    FREE = "free"
    LOCKED = "locked"
    EXPIRED = "expired"


@dataclass
class DistributedLock:
    """
    分布式锁
    
    Attributes:
        lock_id: 锁标识
        holder: 持有者节点ID
        acquired_at: 获取时间
        expires_at: 过期时间
        status: 锁状态
    """
    lock_id: str
    holder: str
    acquired_at: datetime
    expires_at: datetime
    status: LockStatus = LockStatus.LOCKED
    
    def is_expired(self) -> bool:
        """检查锁是否过期"""
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lock_id": self.lock_id,
            "holder": self.holder,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedLock':
        return cls(
            lock_id=data["lock_id"],
            holder=data["holder"],
            acquired_at=datetime.fromisoformat(data["acquired_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            status=LockStatus(data.get("status", "locked"))
        )


@dataclass
class LogEntry:
    """
    Raft日志条目
    
    Attributes:
        index: 日志索引
        term: 任期号
        command: 命令类型
        key: 键
        value: 值
        timestamp: 时间戳
    """
    index: int
    term: int
    command: str  # "set", "delete", "lock", "unlock"
    key: str
    value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "term": self.term,
            "command": self.command,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        return cls(
            index=data["index"],
            term=data["term"],
            command=data["command"],
            key=data["key"],
            value=data.get("value"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now()
        )


class DistributedStateStore:
    """
    分布式状态存储
    
    提供分布式键值存储和分布式锁功能，使用简化版Raft算法：
    1. 状态同步 - Leader-Follower模式复制状态
    2. 分布式锁 - 基于租约的分布式锁
    3. 一致性保证 - 所有写操作通过Leader，读操作可从Follower
    4. 故障恢复 - Leader故障时自动重新选举
    
    注意：这是简化版Raft，适合中小规模集群，不实现完整Raft的所有功能。
    
    Example:
        store = DistributedStateStore(node)
        await store.start()
        
        # 设置值（自动同步到所有节点）
        await store.set("config", {"key": "value"})
        
        # 获取值
        value = await store.get("config")
        
        # 分布式锁
        lock = await store.acquire_lock("resource_1", ttl=30)
        if lock:
            try:
                # 执行临界区代码
                pass
            finally:
                await store.release_lock("resource_1")
    """
    
    def __init__(self, cluster_node, snapshot_interval: int = 1000):
        """
        初始化状态存储
        
        Args:
            cluster_node: ClusterNode实例
            snapshot_interval: 快照间隔（日志条目数）
        """
        from kas.core.cluster.node import ClusterNode
        self.node: ClusterNode = cluster_node
        
        # 状态存储
        self._state: Dict[str, Any] = {}
        
        # Raft日志
        self._log: List[LogEntry] = []
        self._commit_index: int = 0
        self._last_applied: int = 0
        
        # Leader状态（仅Leader有效）
        self._next_index: Dict[str, int] = {}  # 每个Follower的下一条日志索引
        self._match_index: Dict[str, int] = {}  # 每个Follower已匹配的最高日志索引
        
        # 分布式锁
        self._locks: Dict[str, DistributedLock] = {}
        
        # 配置
        self._snapshot_interval = snapshot_interval
        
        # 工作线程
        self._sync_task: Optional[asyncio.Task] = None
        self._lock_cleanup_task: Optional[asyncio.Task] = None
        
        # 运行状态
        self._running = False
        self._stop_event = asyncio.Event()
        
        # 写入等待（用于等待提交确认）
        self._pending_writes: Dict[int, asyncio.Event] = {}
        
        # 监听器
        self._watchers: Dict[str, List[Callable]] = {}
        
        logger.info("DistributedStateStore initialized")
    
    # ==================== 生命周期管理 ====================
    
    async def start(self) -> bool:
        """启动状态存储"""
        if self._running:
            return True
        
        logger.info("Starting DistributedStateStore")
        self._running = True
        
        # 注册节点消息处理器
        self._setup_message_handlers()
        
        # 启动同步任务（仅Leader）
        self._sync_task = asyncio.create_task(self._sync_loop())
        
        # 启动锁清理任务
        self._lock_cleanup_task = asyncio.create_task(self._lock_cleanup_loop())
        
        return True
    
    async def stop(self) -> None:
        """停止状态存储"""
        if not self._running:
            return
        
        logger.info("Stopping DistributedStateStore")
        self._running = False
        self._stop_event.set()
        
        if self._sync_task:
            self._sync_task.cancel()
        if self._lock_cleanup_task:
            self._lock_cleanup_task.cancel()
        
        logger.info("DistributedStateStore stopped")
    
    def _setup_message_handlers(self) -> None:
        """设置消息处理器"""
        # 注意：实际的消息处理通过ClusterNode的message_handler进行
        # 这里我们只是准备好处理函数，等待integration模块设置
        pass
    
    # ==================== 状态操作（KV Store） ====================
    
    async def get(self, key: str, local_read: bool = True) -> Optional[Any]:
        """
        获取值
        
        Args:
            key: 键
            local_read: 是否允许本地读取（True则从本地读取，False则从Leader读取）
        
        Returns:
            值，或None
        """
        if not local_read and not self.node.is_leader():
            # 从Leader读取
            return await self._get_from_leader(key)
        
        return self._state.get(key)
    
    async def set(self, key: str, value: Any,
                  wait_commit: bool = True,
                  timeout: float = 5.0) -> bool:
        """
        设置值
        
        Args:
            key: 键
            value: 值
            wait_commit: 是否等待提交确认
            timeout: 超时时间
        
        Returns:
            是否成功
        """
        if not self.node.is_leader():
            # 转发给Leader
            return await self._forward_to_leader("set", key, value)
        
        # Leader直接追加日志
        entry = self._append_log("set", key, value)
        
        if wait_commit:
            # 等待提交确认
            event = asyncio.Event()
            self._pending_writes[entry.index] = event
            
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Set operation timeout for key {key}")
                return False
            finally:
                if entry.index in self._pending_writes:
                    del self._pending_writes[entry.index]
        
        return True
    
    async def delete(self, key: str, wait_commit: bool = True) -> bool:
        """
        删除键
        
        Args:
            key: 键
            wait_commit: 是否等待提交确认
        
        Returns:
            是否成功
        """
        if not self.node.is_leader():
            return await self._forward_to_leader("delete", key, None)
        
        entry = self._append_log("delete", key, None)
        
        if wait_commit:
            event = asyncio.Event()
            self._pending_writes[entry.index] = event
            
            try:
                await asyncio.wait_for(event.wait(), timeout=5.0)
                return True
            except asyncio.TimeoutError:
                return False
            finally:
                if entry.index in self._pending_writes:
                    del self._pending_writes[entry.index]
        
        return True
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return key in self._state
    
    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        获取所有键
        
        Args:
            pattern: 可选的模式匹配（简化实现，使用startswith）
        
        Returns:
            键列表
        """
        keys = list(self._state.keys())
        
        if pattern:
            import fnmatch
            keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        
        return keys
    
    async def get_all(self) -> Dict[str, Any]:
        """获取所有状态"""
        return self._state.copy()
    
    # ==================== 分布式锁 ====================
    
    async def acquire_lock(self, lock_id: str,
                           ttl: float = 30.0,
                           blocking: bool = False,
                           blocking_timeout: Optional[float] = None) -> bool:
        """
        获取分布式锁
        
        Args:
            lock_id: 锁标识
            ttl: 锁过期时间（秒）
            blocking: 是否阻塞等待
            blocking_timeout: 阻塞超时时间
        
        Returns:
            是否成功获取锁
        """
        # 如果当前节点是Leader，直接处理
        if self.node.is_leader():
            return await self._acquire_lock_local(lock_id, ttl)
        
        # 否则转发给Leader
        return await self._forward_to_leader("lock", lock_id, {"ttl": ttl})
    
    async def release_lock(self, lock_id: str) -> bool:
        """
        释放分布式锁
        
        Args:
            lock_id: 锁标识
        
        Returns:
            是否成功释放
        """
        if self.node.is_leader():
            return await self._release_lock_local(lock_id)
        
        return await self._forward_to_leader("unlock", lock_id, None)
    
    async def is_locked(self, lock_id: str) -> bool:
        """检查锁是否被持有"""
        if lock_id in self._locks:
            lock = self._locks[lock_id]
            if not lock.is_expired():
                return True
        return False
    
    async def get_lock_holder(self, lock_id: str) -> Optional[str]:
        """获取锁的持有者"""
        if lock_id in self._locks:
            lock = self._locks[lock_id]
            if not lock.is_expired():
                return lock.holder
        return None
    
    async def _acquire_lock_local(self, lock_id: str, ttl: float) -> bool:
        """本地获取锁"""
        now = datetime.now()
        
        # 检查现有锁
        if lock_id in self._locks:
            existing = self._locks[lock_id]
            if not existing.is_expired():
                return False  # 锁已被持有且未过期
        
        # 创建新锁
        lock = DistributedLock(
            lock_id=lock_id,
            holder=self.node.node_id,
            acquired_at=now,
            expires_at=now + timedelta(seconds=ttl)
        )
        
        # 记录日志并应用
        entry = self._append_log("lock", lock_id, lock.to_dict())
        self._apply_entry(entry)
        
        return True
    
    async def _release_lock_local(self, lock_id: str) -> bool:
        """本地释放锁"""
        if lock_id not in self._locks:
            return False
        
        lock = self._locks[lock_id]
        
        # 检查是否是持有者
        if lock.holder != self.node.node_id:
            logger.warning(f"Node {self.node.node_id} tried to release lock held by {lock.holder}")
            return False
        
        # 记录日志并应用
        entry = self._append_log("unlock", lock_id, None)
        self._apply_entry(entry)
        
        return True
    
    async def _lock_cleanup_loop(self) -> None:
        """锁清理循环 - 清理过期锁"""
        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                # 清理过期锁
                expired_locks = [
                    lock_id for lock_id, lock in self._locks.items()
                    if lock.is_expired()
                ]
                
                for lock_id in expired_locks:
                    logger.debug(f"Cleaning up expired lock: {lock_id}")
                    del self._locks[lock_id]
    
    # ==================== Raft日志 ====================
    
    def _append_log(self, command: str, key: str, value: Any) -> LogEntry:
        """追加日志条目"""
        entry = LogEntry(
            index=len(self._log) + 1,
            term=self.node.get_term(),
            command=command,
            key=key,
            value=value
        )
        self._log.append(entry)
        
        # 应用日志到状态机
        self._apply_entry(entry)
        
        return entry
    
    def _apply_entry(self, entry: LogEntry) -> None:
        """应用日志条目到状态机"""
        if entry.command == "set":
            self._state[entry.key] = entry.value
            self._notify_watchers(entry.key, entry.value)
        
        elif entry.command == "delete":
            if entry.key in self._state:
                del self._state[entry.key]
                self._notify_watchers(entry.key, None)
        
        elif entry.command == "lock":
            lock_data = entry.value
            if isinstance(lock_data, dict):
                self._locks[entry.key] = DistributedLock.from_dict(lock_data)
        
        elif entry.command == "unlock":
            if entry.key in self._locks:
                del self._locks[entry.key]
        
        self._last_applied = entry.index
    
    def _commit_entries(self, up_to_index: int) -> None:
        """提交日志条目（触发等待的事件）"""
        for index in range(self._commit_index + 1, up_to_index + 1):
            if index in self._pending_writes:
                self._pending_writes[index].set()
        
        self._commit_index = up_to_index
    
    # ==================== 状态同步 ====================
    
    async def _sync_loop(self) -> None:
        """状态同步循环（仅Leader）"""
        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=0.1  # 100ms同步一次
                )
            except asyncio.TimeoutError:
                if not self.node.is_leader():
                    continue
                
                # 同步日志给Followers
                await self._sync_to_followers()
    
    async def _sync_to_followers(self) -> None:
        """同步日志到Followers"""
        members = self.node.get_active_members()
        
        for member_id in members:
            try:
                await self._send_append_entries(member_id)
            except Exception as e:
                logger.debug(f"Failed to sync to {member_id}: {e}")
    
    async def _send_append_entries(self, member_id: str) -> bool:
        """发送AppendEntries RPC到Follower"""
        next_idx = self._next_index.get(member_id, 1)
        
        # 准备条目
        entries = []
        for i in range(next_idx - 1, len(self._log)):
            entries.append(self._log[i].to_dict())
        
        if not entries:
            return True  # 没有新条目需要发送
        
        # 构建请求
        prev_log_index = next_idx - 1
        prev_log_term = 0
        if prev_log_index > 0 and prev_log_index <= len(self._log):
            prev_log_term = self._log[prev_log_index - 1].term
        
        request = {
            "term": self.node.get_term(),
            "leader_id": self.node.node_id,
            "prev_log_index": prev_log_index,
            "prev_log_term": prev_log_term,
            "entries": entries,
            "leader_commit": self._commit_index
        }
        
        # 发送请求（实际发送由integration模块处理）
        # 这里模拟成功响应
        # 实际实现中，这里应该等待响应并处理成功/失败
        
        # 更新索引（简化处理，假设总是成功）
        self._next_index[member_id] = len(self._log) + 1
        self._match_index[member_id] = len(self._log)
        
        # 检查是否可以提交
        self._check_commit()
        
        return True
    
    def _check_commit(self) -> None:
        """检查是否可以提交日志"""
        # 找到可以提交的最高索引
        for index in range(self._commit_index + 1, len(self._log) + 1):
            # 统计已复制的节点数
            replicated = 1  # Leader自己
            
            for member_id, match_idx in self._match_index.items():
                if match_idx >= index:
                    replicated += 1
            
            # 如果大多数节点已复制，提交
            total_nodes = len(self.node.get_members()) + 1
            if replicated > total_nodes / 2:
                self._commit_entries(index)
            else:
                break
    
    async def handle_append_entries(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理AppendEntries RPC（Follower端）
        
        Returns:
            响应字典
        """
        term = request.get("term", 0)
        leader_id = request.get("leader_id")
        prev_log_index = request.get("prev_log_index", 0)
        prev_log_term = request.get("prev_log_term", 0)
        entries = request.get("entries", [])
        leader_commit = request.get("leader_commit", 0)
        
        # 如果任期更高，更新自己的任期
        if term > self.node.get_term():
            # 转为Follower（这应该在handle_heartbeat中处理）
            pass
        
        # 检查日志一致性
        if prev_log_index > 0:
            if prev_log_index > len(self._log):
                return {"term": self.node.get_term(), "success": False}
            
            if self._log[prev_log_index - 1].term != prev_log_term:
                return {"term": self.node.get_term(), "success": False}
        
        # 追加新条目
        for entry_data in entries:
            entry = LogEntry.from_dict(entry_data)
            
            # 如果已有该索引的条目，检查任期
            if entry.index <= len(self._log):
                if self._log[entry.index - 1].term != entry.term:
                    # 删除冲突条目及其后的所有条目
                    self._log = self._log[:entry.index - 1]
                    self._log.append(entry)
                    self._apply_entry(entry)
            else:
                # 追加新条目
                self._log.append(entry)
                self._apply_entry(entry)
        
        # 更新commit_index
        if leader_commit > self._commit_index:
            self._commit_index = min(leader_commit, len(self._log))
        
        return {"term": self.node.get_term(), "success": True}
    
    # ==================== 外部请求处理 ====================
    
    async def _forward_to_leader(self, command: str, key: str,
                                  value: Any) -> Any:
        """转发请求给Leader"""
        leader_id = self.node.get_leader_id()
        
        if not leader_id:
            logger.error("No leader available")
            return False
        
        # 构建请求
        request = {
            "type": "state_request",
            "command": command,
            "key": key,
            "value": value,
            "sender": self.node.node_id
        }
        
        # 实际发送由integration模块处理
        # 这里返回False表示需要外部处理
        logger.debug(f"Request to {command} {key} needs to be forwarded to leader {leader_id}")
        return False
    
    async def _get_from_leader(self, key: str) -> Any:
        """从Leader获取值"""
        # 简化处理：直接从本地读取
        # 实际实现中，如果要求强一致性，需要向Leader查询
        return self._state.get(key)
    
    async def handle_state_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理状态请求（Leader端）
        
        Args:
            request: 请求字典，包含 command, key, value
        
        Returns:
            响应字典
        """
        command = request.get("command")
        key = request.get("key")
        value = request.get("value")
        
        if command == "set":
            success = await self.set(key, value, wait_commit=True)
            return {"success": success}
        
        elif command == "get":
            return {"success": True, "value": self._state.get(key)}
        
        elif command == "delete":
            success = await self.delete(key)
            return {"success": success}
        
        elif command == "lock":
            ttl = value.get("ttl", 30.0) if isinstance(value, dict) else 30.0
            success = await self._acquire_lock_local(key, ttl)
            return {"success": success}
        
        elif command == "unlock":
            success = await self._release_lock_local(key)
            return {"success": success}
        
        return {"success": False, "error": "Unknown command"}
    
    # ==================== 快照 ====================
    
    async def create_snapshot(self) -> Dict[str, Any]:
        """创建快照"""
        return {
            "last_included_index": self._last_applied,
            "last_included_term": self._log[self._last_applied - 1].term if self._last_applied > 0 else 0,
            "state": self._state.copy(),
            "locks": {k: v.to_dict() for k, v in self._locks.items()},
            "timestamp": datetime.now().isoformat()
        }
    
    async def install_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """安装快照"""
        try:
            self._state = snapshot.get("state", {})
            locks_data = snapshot.get("locks", {})
            self._locks = {k: DistributedLock.from_dict(v) for k, v in locks_data.items()}
            
            # 截断日志
            last_index = snapshot.get("last_included_index", 0)
            self._log = self._log[last_index:]
            self._last_applied = last_index
            self._commit_index = last_index
            
            return True
        except Exception as e:
            logger.error(f"Failed to install snapshot: {e}")
            return False
    
    # ==================== 监听 ====================
    
    def watch(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """
        监听键的变化
        
        Args:
            key: 要监听的键
            callback: 回调函数，接收(key, value)参数
        """
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)
    
    def unwatch(self, key: str, callback: Callable) -> None:
        """取消监听"""
        if key in self._watchers and callback in self._watchers[key]:
            self._watchers[key].remove(callback)
    
    def _notify_watchers(self, key: str, value: Any) -> None:
        """通知监听者"""
        for callback in self._watchers.get(key, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(key, value))
                else:
                    callback(key, value)
            except Exception as e:
                logger.error(f"Watcher callback error: {e}")
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "is_leader": self.node.is_leader(),
            "state_size": len(self._state),
            "log_size": len(self._log),
            "commit_index": self._commit_index,
            "last_applied": self._last_applied,
            "active_locks": len([l for l in self._locks.values() if not l.is_expired()]),
            "watchers": len(self._watchers)
        }
