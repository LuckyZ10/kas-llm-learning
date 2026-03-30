"""
KAS Distributed Agent Cluster - 分布式Agent集群

Phase 5.2 实现：分布式Agent集群支持，让Agent可以跨机器协作。

主要功能：
1. 集群节点管理 - ClusterNode提供节点注册、发现、心跳和Leader选举
2. 集群管理器 - ClusterManager提供负载均衡、故障检测和恢复
3. 分布式任务调度 - DistributedScheduler提供任务分片和结果聚合
4. 分布式状态存储 - DistributedStateStore提供简化版Raft共识
5. Phase 5.1集成 - ClusterIntegration与通信协议集成

Example:
    # 快速入门
    import asyncio
    from kas.core.cluster import ClusterNode, ClusterManager, ClusterIntegration
    
    async def main():
        # 创建节点
        node = ClusterNode("node1", "localhost", 8001)
        await node.start()
        
        # 创建管理器
        manager = ClusterManager(node)
        await manager.start()
        
        # 创建集成器
        integration = ClusterIntegration(node, manager)
        integration.configure_tcp_transport(8001)
        await integration.start()
        
        # 加入集群
        await node.join_cluster(["localhost:8000"])
        
        # 使用集群...
        
        await integration.stop()
        await manager.stop()
        await node.stop()
"""

# 节点管理
from kas.core.cluster.node import (
    ClusterNode,
    NodeInfo,
    NodeState,
    NodeRole,
    ClusterConfig,
)

# 集群管理
from kas.core.cluster.manager import (
    ClusterManager,
    LoadBalancerConfig,
    ConsistentHashRing,
)

# 任务调度
from kas.core.cluster.scheduler import (
    DistributedScheduler,
    Task,
    TaskShard,
    TaskStatus,
    TaskType,
    TaskResult,
    TaskAggregator,
)

# 状态存储
from kas.core.cluster.state import (
    DistributedStateStore,
    DistributedLock,
    LockStatus,
    LogEntry,
)

# Phase 5.1集成
from kas.core.cluster.integration import (
    ClusterIntegration,
    ClusterMessageHandler,
)

__all__ = [
    # 节点管理
    "ClusterNode",
    "NodeInfo",
    "NodeState",
    "NodeRole",
    "ClusterConfig",
    
    # 集群管理
    "ClusterManager",
    "LoadBalancerConfig",
    "ConsistentHashRing",
    
    # 任务调度
    "DistributedScheduler",
    "Task",
    "TaskShard",
    "TaskStatus",
    "TaskType",
    "TaskResult",
    "TaskAggregator",
    
    # 状态存储
    "DistributedStateStore",
    "DistributedLock",
    "LockStatus",
    "LogEntry",
    
    # Phase 5.1集成
    "ClusterIntegration",
    "ClusterMessageHandler",
]

__version__ = "5.2.0"
