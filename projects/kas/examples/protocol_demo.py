"""
KAS Agent Communication Protocol - 使用示例

演示如何使用通信协议进行Agent间通信。
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

from kas.core.protocol import (
    CommunicationManager,
    TransportConfig,
    LocalTransport,
    TCPTransport,
    MessageBuilder,
    MessageType,
    MessageRouter,
    RoutingStrategy,
)
from kas.core.protocol.integration import (
    CrewCommunicationAdapter,
    AgentCommunicator,
)


async def demo_local_transport():
    """示例1: 本地传输层通信"""
    print("\n=== 示例1: 本地传输层通信 ===\n")
    
    # 创建两个Agent的通信管理器
    agent1 = CommunicationManager("agent1")
    agent2 = CommunicationManager("agent2")
    
    # 配置本地传输
    agent1.add_transport(TransportConfig(
        name="local",
        transport_class=LocalTransport,
        config={},
        priority=0
    ))
    
    agent2.add_transport(TransportConfig(
        name="local",
        transport_class=LocalTransport,
        config={},
        priority=0
    ))
    
    # 启动
    await agent1.start()
    await agent2.start()
    
    # 设置消息处理器
    received_messages = []
    
    def message_handler(msg):
        print(f"Agent2 收到消息: {msg.payload}")
        received_messages.append(msg)
    
    agent2.on_any_message(message_handler)
    
    # Agent1 发送消息给 Agent2
    msg = MessageBuilder().request() \
        .from_agent("agent1") \
        .to_agent("agent2") \
        .with_payload({"action": "process_data", "data": [1, 2, 3]}) \
        .build()
    
    success = await agent1.send(msg)
    print(f"Agent1 发送消息: {'成功' if success else '失败'}")
    
    # 等待消息处理
    await asyncio.sleep(0.5)
    
    # 停止
    await agent1.stop()
    await agent2.stop()
    
    print(f"\n总共收到 {len(received_messages)} 条消息")


async def demo_pub_sub():
    """示例2: 发布/订阅模式"""
    print("\n=== 示例2: 发布/订阅模式 ===\n")
    
    # 创建路由器
    router = MessageRouter()
    
    # 注册Agent
    router.register_agent("agent1")
    router.register_agent("agent2")
    router.register_agent("agent3")
    
    # Agent订阅不同主题
    received = {"agent2": [], "agent3": []}
    
    def handler_factory(agent_id):
        def handler(msg):
            print(f"{agent_id} 收到: {msg.payload}")
            received[agent_id].append(msg)
        return handler
    
    router.subscribe("agent2", "events.*", handler_factory("agent2"))
    router.subscribe("agent3", "events.critical.*", handler_factory("agent3"))
    
    # 发布消息
    msg1 = MessageBuilder().event() \
        .from_agent("agent1") \
        .with_payload({"type": "info", "content": "系统启动"}) \
        .build()
    msg1.headers["topic"] = "events.system"
    
    msg2 = MessageBuilder().event() \
        .from_agent("agent1") \
        .with_payload({"type": "error", "content": "磁盘空间不足"}) \
        .build()
    msg2.headers["topic"] = "events.critical.disk"
    
    print("发布消息到 events.system...")
    await router.publish(msg1)
    
    print("发布消息到 events.critical.disk...")
    await router.publish(msg2)
    
    await asyncio.sleep(0.5)
    
    print(f"\nAgent2 收到 {len(received['agent2'])} 条消息")
    print(f"Agent3 收到 {len(received['agent3'])} 条消息")


async def demo_crew_communication():
    """示例3: Crew成员间通信"""
    print("\n=== 示例3: Crew成员间通信 ===\n")
    
    # 创建Crew通信适配器
    crew = CrewCommunicationAdapter("ContractReviewCrew")
    await crew.start()
    
    # 创建Agent通信器
    alice = crew.create_agent_communicator("Alice")
    bob = crew.create_agent_communicator("Bob")
    carol = crew.create_agent_communicator("Carol")
    
    # 设置消息处理器
    bob_messages = []
    carol_messages = []
    
    def bob_handler(payload):
        print(f"Bob 收到: {payload}")
        bob_messages.append(payload)
    
    def carol_handler(payload):
        print(f"Carol 收到: {payload}")
        carol_messages.append(payload)
    
    bob.on_message(bob_handler)
    carol.on_message(carol_handler)
    
    # Alice 分配任务
    print("Alice 分配任务...")
    await alice.send_message({
        "type": "task_assignment",
        "task": "OCR识别",
        "contract_id": "CT-2024-001"
    }, to_agent="Bob")
    
    await alice.send_message({
        "type": "task_assignment",
        "task": "条款分析",
        "contract_id": "CT-2024-001"
    }, to_agent="Carol")
    
    # Bob 完成任务后报告
    await asyncio.sleep(0.5)
    print("\nBob 完成任务...")
    await bob.send_message({
        "type": "task_complete",
        "task": "OCR识别",
        "result": "识别完成: 10页内容"
    })
    
    await asyncio.sleep(0.5)
    
    # 广播通知
    print("\nAlice 广播通知...")
    await crew.broadcast_to_crew({
        "type": "notification",
        "content": "合同审查完成"
    })
    
    await asyncio.sleep(0.5)
    
    # 停止
    await crew.stop()
    
    print(f"\nBob 总共收到 {len(bob_messages)} 条消息")
    print(f"Carol 总共收到 {len(carol_messages)} 条消息")


async def demo_request_response():
    """示例4: 请求-响应模式"""
    print("\n=== 示例4: 请求-响应模式 ===\n")
    
    # 创建服务端
    server = CommunicationManager("server")
    server.add_transport(TransportConfig(
        name="local",
        transport_class=LocalTransport,
        config={},
        priority=0
    ))
    
    # 创建客户端
    client = CommunicationManager("client")
    client.add_transport(TransportConfig(
        name="local",
        transport_class=LocalTransport,
        config={},
        priority=0
    ))
    
    # 启动
    await server.start()
    await client.start()
    
    # 服务端处理请求
    async def handle_request(msg):
        print(f"Server 收到请求: {msg.payload}")
        
        # 处理请求
        data = msg.payload.get("data", [])
        result = sum(data)
        
        # 发送响应
        response = MessageBuilder().response() \
            .from_agent("server") \
            .to_agent("client") \
            .correlates_to(msg.id) \
            .with_payload({"result": result}) \
            .build()
        
        await server.send(response)
        print(f"Server 发送响应: {result}")
    
    server.on_message(MessageType.REQUEST, handle_request)
    
    # 客户端发送请求
    print("Client 发送请求...")
    response = await client.send_request(
        receiver="server",
        payload={"data": [10, 20, 30, 40]},
        timeout=5.0
    )
    
    if response:
        print(f"Client 收到响应: {response.payload}")
    else:
        print("Client 请求超时")
    
    # 停止
    await server.stop()
    await client.stop()


async def main():
    """运行所有示例"""
    print("=" * 60)
    print("KAS Agent Communication Protocol - 演示")
    print("=" * 60)
    
    try:
        await demo_local_transport()
        await demo_pub_sub()
        await demo_crew_communication()
        await demo_request_response()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
