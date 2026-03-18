"""
KAS Agent Communication Protocol - 消息路由模块

提供消息路由功能，支持发布/订阅、点对点和广播路由模式。
"""
import asyncio
import fnmatch
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from kas.core.protocol.message import Message, MessageType


class RoutingStrategy(Enum):
    """路由策略枚举"""
    DIRECT = "direct"       # 点对点直接路由
    PUB_SUB = "pub_sub"     # 发布/订阅模式
    BROADCAST = "broadcast" # 广播模式
    ROUND_ROBIN = "round_robin"  # 轮询模式
    HASH = "hash"           # 哈希路由


@dataclass
class Subscription:
    """
    订阅信息
    
    Attributes:
        id: 订阅唯一标识
        subscriber_id: 订阅者ID
        pattern: 订阅模式（支持通配符 * 和 ?）
        handler: 消息处理回调
        filter_fn: 可选的消息过滤函数
        created_at: 创建时间
    """
    id: str
    subscriber_id: str
    pattern: str
    handler: Callable[[Message], None]
    filter_fn: Optional[Callable[[Message], bool]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Route:
    """
    路由规则
    
    Attributes:
        source: 源Agent模式（None表示任意）
        target: 目标Agent模式（None表示任意）
        msg_type: 消息类型过滤器
        strategy: 路由策略
        priority: 路由优先级
        transform_fn: 消息转换函数
    """
    source: Optional[str] = None
    target: Optional[str] = None
    msg_type: Optional[MessageType] = None
    strategy: RoutingStrategy = RoutingStrategy.DIRECT
    priority: int = 0
    transform_fn: Optional[Callable[[Message], Message]] = None


class MessageRouter:
    """
    消息路由器
    
    负责消息的路由分发，支持多种路由策略：
    1. 点对点路由 - 直接发送到指定Agent
    2. 发布/订阅 - 基于模式的订阅机制
    3. 广播路由 - 发送到所有Agent
    4. 轮询路由 - 轮询选择目标
    5. 哈希路由 - 基于哈希值选择目标
    
    Example:
        router = MessageRouter()
        
        # 订阅消息
        router.subscribe("agent1", "events.*", handler)
        
        # 发布消息
        await router.publish(message)
        
        # 点对点发送
        await router.route_direct(message)
    """
    
    def __init__(self):
        # 订阅表：pattern -> [Subscription]
        self._subscriptions: Dict[str, List[Subscription]] = {}
        
        # Agent订阅索引：agent_id -> [subscription_id]
        self._agent_subscriptions: Dict[str, Set[str]] = {}
        
        # 路由规则表
        self._routes: List[Route] = []
        
        # 轮询计数器
        self._round_robin_counters: Dict[str, int] = {}
        
        # 已注册Agent
        self._registered_agents: Set[str] = set()
        
        # 消息中间件链
        self._middleware: List[Callable[[Message], Optional[Message]]] = []
        
        # 统计信息
        self._stats = {
            "messages_routed": 0,
            "messages_dropped": 0,
            "subscriptions_created": 0,
            "subscriptions_removed": 0
        }
        
        self._lock = asyncio.Lock()
    
    # ==================== 订阅管理 ====================
    
    def subscribe(self, subscriber_id: str, pattern: str,
                  handler: Callable[[Message], None],
                  filter_fn: Optional[Callable[[Message], bool]] = None) -> str:
        """
        订阅消息
        
        Args:
            subscriber_id: 订阅者ID
            pattern: 订阅模式（如 "events.*", "agent.*.status"）
            handler: 消息处理函数
            filter_fn: 可选的消息过滤函数
        
        Returns:
            订阅ID
        """
        import uuid
        subscription_id = str(uuid.uuid4())
        
        subscription = Subscription(
            id=subscription_id,
            subscriber_id=subscriber_id,
            pattern=pattern,
            handler=handler,
            filter_fn=filter_fn
        )
        
        if pattern not in self._subscriptions:
            self._subscriptions[pattern] = []
        self._subscriptions[pattern].append(subscription)
        
        if subscriber_id not in self._agent_subscriptions:
            self._agent_subscriptions[subscriber_id] = set()
        self._agent_subscriptions[subscriber_id].add(subscription_id)
        
        self._stats["subscriptions_created"] += 1
        logger.debug(f"Subscription created: {subscriber_id} -> {pattern}")
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        取消订阅
        
        Args:
            subscription_id: 订阅ID
        
        Returns:
            是否成功取消
        """
        for pattern, subs in self._subscriptions.items():
            for sub in subs:
                if sub.id == subscription_id:
                    subs.remove(sub)
                    
                    # 更新Agent订阅索引
                    if sub.subscriber_id in self._agent_subscriptions:
                        self._agent_subscriptions[sub.subscriber_id].discard(subscription_id)
                    
                    self._stats["subscriptions_removed"] += 1
                    logger.debug(f"Subscription removed: {subscription_id}")
                    return True
        return False
    
    def unsubscribe_all(self, agent_id: str) -> int:
        """
        取消Agent的所有订阅
        
        Args:
            agent_id: Agent ID
        
        Returns:
            取消的订阅数量
        """
        count = 0
        if agent_id in self._agent_subscriptions:
            for sub_id in list(self._agent_subscriptions[agent_id]):
                if self.unsubscribe(sub_id):
                    count += 1
            del self._agent_subscriptions[agent_id]
        return count
    
    def get_subscriptions(self, agent_id: Optional[str] = None) -> List[Subscription]:
        """
        获取订阅列表
        
        Args:
            agent_id: 可选的Agent过滤器
        
        Returns:
            订阅列表
        """
        if agent_id:
            result = []
            for pattern, subs in self._subscriptions.items():
                for sub in subs:
                    if sub.subscriber_id == agent_id:
                        result.append(sub)
            return result
        else:
            return [sub for subs in self._subscriptions.values() for sub in subs]
    
    # ==================== 路由功能 ====================
    
    async def route(self, message: Message, strategy: Optional[RoutingStrategy] = None) -> List[str]:
        """
        路由消息
        
        Args:
            message: 要路由的消息
            strategy: 路由策略（None表示自动选择）
        
        Returns:
            接收者ID列表
        """
        # 应用中间件
        for middleware in self._middleware:
            try:
                message = middleware(message) or message
            except Exception as e:
                logger.error(f"Middleware error: {e}")
        
        # 确定路由策略
        if strategy is None:
            strategy = self._determine_strategy(message)
        
        # 执行路由
        if strategy == RoutingStrategy.DIRECT:
            return await self._route_direct(message)
        elif strategy == RoutingStrategy.PUB_SUB:
            return await self._route_pub_sub(message)
        elif strategy == RoutingStrategy.BROADCAST:
            return await self._route_broadcast(message)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return await self._route_round_robin(message)
        elif strategy == RoutingStrategy.HASH:
            return await self._route_hash(message)
        else:
            logger.warning(f"Unknown routing strategy: {strategy}")
            return []
    
    async def route_direct(self, message: Message) -> bool:
        """
        点对点路由
        
        Args:
            message: 消息（receiver必须指定）
        
        Returns:
            是否路由成功
        """
        recipients = await self._route_direct(message)
        return len(recipients) > 0
    
    async def publish(self, message: Message, topic: Optional[str] = None) -> List[str]:
        """
        发布消息（Pub/Sub模式）
        
        Args:
            message: 消息
            topic: 主题（覆盖消息的topic信息）
        
        Returns:
            接收者ID列表
        """
        if topic:
            message.headers["topic"] = topic
        return await self._route_pub_sub(message)
    
    async def broadcast(self, message: Message) -> List[str]:
        """
        广播消息
        
        Args:
            message: 消息
        
        Returns:
            接收者ID列表
        """
        return await self._route_broadcast(message)
    
    # ==================== 内部路由实现 ====================
    
    def _determine_strategy(self, message: Message) -> RoutingStrategy:
        """根据消息内容确定路由策略"""
        if message.receiver is None or message.type == MessageType.BROADCAST:
            return RoutingStrategy.BROADCAST
        elif "topic" in message.headers:
            return RoutingStrategy.PUB_SUB
        else:
            return RoutingStrategy.DIRECT
    
    async def _route_direct(self, message: Message) -> List[str]:
        """点对点路由实现"""
        if not message.receiver:
            logger.warning("Direct routing requires receiver")
            return []
        
        target = message.receiver
        delivered = []
        
        # 查找匹配的订阅
        for pattern, subs in self._subscriptions.items():
            if self._match_pattern(target, pattern):
                for sub in subs:
                    if sub.filter_fn and not sub.filter_fn(message):
                        continue
                    
                    try:
                        if asyncio.iscoroutinefunction(sub.handler):
                            await sub.handler(message)
                        else:
                            sub.handler(message)
                        delivered.append(sub.subscriber_id)
                    except Exception as e:
                        logger.error(f"Handler error for {sub.subscriber_id}: {e}")
        
        self._stats["messages_routed"] += len(delivered)
        return delivered
    
    async def _route_pub_sub(self, message: Message) -> List[str]:
        """发布/订阅路由实现"""
        topic = message.headers.get("topic", message.payload.get("topic", "*"))
        delivered = []
        
        for pattern, subs in self._subscriptions.items():
            if self._match_pattern(topic, pattern):
                for sub in subs:
                    if sub.filter_fn and not sub.filter_fn(message):
                        continue
                    
                    try:
                        if asyncio.iscoroutinefunction(sub.handler):
                            await sub.handler(message)
                        else:
                            sub.handler(message)
                        delivered.append(sub.subscriber_id)
                    except Exception as e:
                        logger.error(f"Handler error for {sub.subscriber_id}: {e}")
        
        self._stats["messages_routed"] += len(delivered)
        return delivered
    
    async def _route_broadcast(self, message: Message) -> List[str]:
        """广播路由实现"""
        delivered = []
        
        # 发送给所有已注册Agent
        for agent_id in self._registered_agents:
            if agent_id == message.sender:
                continue  # 不发送给自己
            
            # 查找该Agent的订阅
            if agent_id in self._agent_subscriptions:
                for sub_id in self._agent_subscriptions[agent_id]:
                    for subs in self._subscriptions.values():
                        for sub in subs:
                            if sub.id == sub_id:
                                try:
                                    if asyncio.iscoroutinefunction(sub.handler):
                                        await sub.handler(message)
                                    else:
                                        sub.handler(message)
                                    delivered.append(agent_id)
                                except Exception as e:
                                    logger.error(f"Handler error for {agent_id}: {e}")
        
        self._stats["messages_routed"] += len(delivered)
        return delivered
    
    async def _route_round_robin(self, message: Message) -> List[str]:
        """轮询路由实现"""
        # 获取可用的目标
        targets = list(self._registered_agents)
        if message.sender in targets:
            targets.remove(message.sender)
        
        if not targets:
            return []
        
        # 轮询选择
        counter_key = message.headers.get("group", "default")
        counter = self._round_robin_counters.get(counter_key, 0)
        target = targets[counter % len(targets)]
        self._round_robin_counters[counter_key] = counter + 1
        
        # 设置接收者并路由
        message.receiver = target
        return await self._route_direct(message)
    
    async def _route_hash(self, message: Message) -> List[str]:
        """哈希路由实现"""
        targets = list(self._registered_agents)
        if message.sender in targets:
            targets.remove(message.sender)
        
        if not targets:
            return []
        
        # 基于消息ID哈希选择目标
        hash_key = message.headers.get("hash_key", message.id)
        target_index = hash(hash_key) % len(targets)
        target = targets[target_index]
        
        message.receiver = target
        return await self._route_direct(message)
    
    def _match_pattern(self, value: str, pattern: str) -> bool:
        """
        匹配模式
        
        支持通配符：
        - * 匹配任意字符序列
        - ? 匹配单个字符
        """
        return fnmatch.fnmatch(value, pattern)
    
    # ==================== Agent管理 ====================
    
    def register_agent(self, agent_id: str) -> None:
        """注册Agent到路由器"""
        self._registered_agents.add(agent_id)
        logger.debug(f"Agent registered: {agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """从路由器注销Agent"""
        self._registered_agents.discard(agent_id)
        self.unsubscribe_all(agent_id)
        logger.debug(f"Agent unregistered: {agent_id}")
    
    def get_registered_agents(self) -> List[str]:
        """获取已注册的Agent列表"""
        return list(self._registered_agents)
    
    # ==================== 路由规则 ====================
    
    def add_route(self, route: Route) -> None:
        """添加路由规则"""
        self._routes.append(route)
        self._routes.sort(key=lambda r: r.priority, reverse=True)
        logger.debug(f"Route added: {route}")
    
    def remove_route(self, source: Optional[str] = None,
                     target: Optional[str] = None) -> int:
        """移除路由规则"""
        count = 0
        self._routes = [
            r for r in self._routes
            if not ((source is None or r.source == source) and
                    (target is None or r.target == target))
        ]
        return count
    
    def clear_routes(self) -> None:
        """清除所有路由规则"""
        self._routes.clear()
    
    # ==================== 中间件 ====================
    
    def use(self, middleware: Callable[[Message], Optional[Message]]) -> None:
        """
        添加中间件
        
        中间件可以对消息进行处理、转换或过滤。
        返回None表示丢弃消息。
        """
        self._middleware.append(middleware)
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        return {
            **self._stats,
            "registered_agents": len(self._registered_agents),
            "active_subscriptions": sum(len(subs) for subs in self._subscriptions.values()),
            "routing_rules": len(self._routes)
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "messages_routed": 0,
            "messages_dropped": 0,
            "subscriptions_created": 0,
            "subscriptions_removed": 0
        }
