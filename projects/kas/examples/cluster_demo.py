"""
KAS Distributed Agent Cluster - 使用示例

演示如何使用分布式集群功能。
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

from kas.core.cluster import (
    ClusterNode, ClusterManager, ClusterIntegration,
    DistributedScheduler, Task, TaskType, TaskStatus
)
from kas.core.cluster.integration import ClusterMessageHandler


async def demo_single_node_cluster():
    """示例1: 单节点集群（快速测试）"""
    print("\n=== 示例1: 单节点集群 ===\n")
    
    # 创建节点
    node1 = ClusterNode("node1", "127.0.0.1", 8001)
    await node1.start()
    
    # 创建管理器
    manager1 = ClusterManager(node1)
    await manager1.start()
    
    # 创建集成器
    integration1 = ClusterIntegration(node1, manager1, enable_local_transport=True)
    integration1.configure_local_transport()
    await integration1.start()
    
    print(f"节点1启动: {node1.node_id}")
    print(f"角色: {node1.get_role().value}")
    print(f"状态: {node1.get_state().value}")
    
    # 等待心跳周期
    await asyncio.sleep(2)
    
    # 停止
    await integration1.stop()
    await manager1.stop()
    await node1.stop()
    
    print("单节点集群测试完成")


async def demo_two_node_cluster():
    """示例2: 双节点集群（Leader-Follower）"""
    print("\n=== 示例2: 双节点集群 ===\n")
    
    # 创建第一个节点（将成为Leader）
    node1 = ClusterNode("node1", "127.0.0.1", 8001)
    await node1.start()
    manager1 = ClusterManager(node1)
    await manager1.start()
    integration1 = ClusterIntegration(node1, manager1, enable_local_transport=True)
    integration1.configure_local_transport()
    await integration1.start()
    
    # 创建第二个节点
    node2 = ClusterNode("node2", "127.0.0.1", 8002)
    await node2.start()
    manager2 = ClusterManager(node2)
    await manager2.start()
    integration2 = ClusterIntegration(node2, manager2, enable_local_transport=True)
    integration2.configure_local_transport()
    await integration2.start()
    
    print(f"节点1: {node1.node_id} - {node1.get_role().value}")
    print(f"节点2: {node2.node_id} - {node2.get_role().value}")
    
    # 节点2加入集群
    print("\n节点2尝试加入集群...")
    success = await node2.join_cluster(["127.0.0.1:8001"])
    print(f"加入结果: {'成功' if success else '失败'}")
    
    # 等待一段时间
    await asyncio.sleep(3)
    
    # 查看集群成员
    print(f"\n节点1的成员列表:")
    for member_id, member_info in node1.get_members().items():
        print(f"  - {member_id} ({member_info.role.value})")
    
    print(f"\n节点2的成员列表:")
    for member_id, member_info in node2.get_members().items():
        print(f"  - {member_id} ({member_info.role.value})")
    
    # 停止
    await integration2.stop()
    await manager2.stop()
    await node2.stop()
    
    await integration1.stop()
    await manager1.stop()
    await node1.stop()
    
    print("\n双节点集群测试完成")


async def demo_distributed_scheduler():
    """示例3: 分布式任务调度"""
    print("\n=== 示例3: 分布式任务调度 ===\n")
    
    # 创建节点
    node = ClusterNode("scheduler_node", "127.0.0.1", 8003)
    await node.start()
    
    manager = ClusterManager(node)
    await manager.start()
    
    # 创建调度器
    scheduler = DistributedScheduler(node, manager)
    await scheduler.start()
    
    print(f"调度器启动: {node.node_id}")
    
    # 提交简单任务
    print("\n提交任务...")
    
    task = Task(
        task_id="task_001",
        task_type=TaskType.SINGLE,
        payload={"action": "compute", "data": [1, 2, 3, 4, 5]},
        max_retries=2
    )
    
    result = await scheduler.submit_task(task, wait_result=True, timeout=5.0)
    print(f"任务结果: {result}")
    
    # 提交分片任务
    print("\n提交分片任务...")
    
    shard_task = Task(
        task_id="shard_task_001",
        task_type=TaskType.SHARDED,
        payload={"data": list(range(100))},
        shard_count=4
    )
    
    result = await scheduler.submit_task(shard_task, wait_result=True, timeout=10.0)
    print(f"分片任务结果: {result}")
    
    # 停止
    await scheduler.stop()
    await manager.stop()
    await node.stop()
    
    print("\n分布式调度器测试完成")


async def demo_distributed_state():
    """示例4: 分布式状态存储"""
    print("\n=== 示例4: 分布式状态存储 ===\n")
    
    from kas.core.cluster.state import DistributedStateStore
    
    # 创建节点
    node = ClusterNode("state_node", "127.0.0.1", 8004)
    await node.start()
    
    # 创建状态存储
    store = DistributedStateStore(node)
    await store.start()
    
    print(f"状态存储启动: {node.node_id}")
    print(f"是否是Leader: {node.is_leader()}")
    
    # 设置状态
    print("\n设置状态...")
    success = await store.set("config", {"name": "test_cluster", "version": "1.0"})
    print(f"设置状态: {'成功' if success else '失败'}")
    
    success = await store.set("counter", 0)
    print(f"设置计数器: {'成功' if success else '失败'}")
    
    # 获取状态
    print("\n获取状态...")
    config = await store.get("config")
    print(f"config = {config}")
    
    counter = await store.get("counter")
    print(f"counter = {counter}")
    
    # 分布式锁
    print("\n测试分布式锁...")
    lock_acquired = await store.acquire_lock("resource_1", ttl=10.0)
    print(f"获取锁: {'成功' if lock_acquired else '失败'}")
    
    if lock_acquired:
        print("执行临界区代码...")
        await asyncio.sleep(1)
        
        lock_released = await store.release_lock("resource_1")
        print(f"释放锁: {'成功' if lock_released else '失败'}")
    
    # 查看统计
    print("\n状态存储统计:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 停止
    await store.stop()
    await node.stop()
    
    print("\n分布式状态存储测试完成")


async def demo_leader_election():
    """示例5: Leader选举"""
    print("\n=== 示例5: Leader选举 ===\n")
    
    # 创建3个节点
    nodes = []
    managers = []
    integrations = []
    
    for i in range(3):
        port = 8010 + i
        node = ClusterNode(f"node_{i+1}", "127.0.0.1", port)
        await node.start()
        
        manager = ClusterManager(node)
        await manager.start()
        
        integration = ClusterIntegration(node, manager, enable_local_transport=True)
        integration.configure_local_transport()
        await integration.start()
        
        nodes.append(node)
        managers.append(manager)
        integrations.append(integration)
        
        print(f"节点{i+1}启动: {node.node_id} - {node.get_role().value}")
    
    # 节点2和3加入节点1的集群
    print("\n节点加入集群...")
    await nodes[1].join_cluster(["127.0.0.1:8010"])
    await nodes[2].join_cluster(["127.0.0.1:8010"])
    
    # 等待选举
    print("等待Leader选举...")
    await asyncio.sleep(3)
    
    # 查看Leader
    print("\nLeader状态:")
    for i, node in enumerate(nodes):
        role = node.get_role().value
        is_leader = node.is_leader()
        print(f"  节点{i+1} ({node.node_id}): {role} {'(Leader)' if is_leader else ''}")
    
    # 模拟Leader故障
    leader_idx = None
    for i, node in enumerate(nodes):
        if node.is_leader():
            leader_idx = i
            break
    
    if leader_idx is not None:
        print(f"\n模拟Leader (节点{leader_idx+1}) 故障...")
        await integrations[leader_idx].stop()
        await managers[leader_idx].stop()
        await nodes[leader_idx].stop()
        
        # 等待新Leader选举
        print("等待新Leader选举...")
        await asyncio.sleep(5)
        
        # 查看新的Leader
        print("\n新Leader状态:")
        for i, node in enumerate(nodes):
            if i == leader_idx:
                print(f"  节点{i+1}: 已停止")
                continue
            role = node.get_role().value
            is_leader = node.is_leader()
            print(f"  节点{i+1} ({node.node_id}): {role} {'(Leader)' if is_leader else ''}")
    
    # 停止剩余节点
    for i in range(len(nodes)):
        if i == leader_idx:
            continue
        await integrations[i].stop()
        await managers[i].stop()
        await nodes[i].stop()
    
    print("\nLeader选举测试完成")


async def main():
    """运行所有示例"""
    print("=" * 60)
    print("KAS Distributed Agent Cluster - 演示")
    print("=" * 60)
    
    try:
        await demo_single_node_cluster()
        await demo_two_node_cluster()
        await demo_distributed_scheduler()
        await demo_distributed_state()
        await demo_leader_election()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
