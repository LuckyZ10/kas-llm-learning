"""
KAS Agent Communication Protocol - 与现有系统集成

提供与Crew Workflow、Memory System的集成。
"""
import asyncio
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

from kas.core.protocol import (
    CommunicationManager,
    TransportConfig,
    LocalTransport,
    Message,
    MessageType,
    MessageBuilder,
)
from kas.core.protocol.manager import ConnectionState


class CrewCommunicationAdapter:
    """
    Crew通信适配器
    
    将通信协议集成到Crew系统中，使Agent可以通过标准化协议通信。
    """
    
    def __init__(self, crew_name: str, supervisor=None):
        self.crew_name = crew_name
        self.supervisor = supervisor
        self.manager = CommunicationManager(f"crew:{crew_name}")
        self._agent_communicators: Dict[str, "AgentCommunicator"] = {}
        
        # 添加本地传输层
        self.manager.add_transport(TransportConfig(
            name="local",
            transport_class=LocalTransport,
            config={},
            priority=0
        ))
    
    async def start(self) -> bool:
        """启动Crew通信"""
        success = await self.manager.start()
        if success:
            logger.info(f"Crew communication started: {self.crew_name}")
        return success
    
    async def stop(self):
        """停止Crew通信"""
        await self.manager.stop()
        logger.info(f"Crew communication stopped: {self.crew_name}")
    
    def create_agent_communicator(self, agent_name: str) -> "AgentCommunicator":
        """为Agent创建通信器"""
        communicator = AgentCommunicator(
            agent_name=agent_name,
            crew_name=self.crew_name,
            manager=self.manager
        )
        self._agent_communicators[agent_name] = communicator
        return communicator
    
    async def broadcast_to_crew(self, message: Dict[str, Any], exclude: Optional[str] = None):
        """广播消息给所有Crew成员"""
        for agent_name, comm in self._agent_communicators.items():
            if agent_name != exclude:
                await comm.send_message(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        return self.manager.get_stats()


class AgentCommunicator:
    """
    Agent通信器
    
    为单个Agent提供简化的通信接口。
    """
    
    def __init__(self, agent_name: str, crew_name: str, manager: CommunicationManager):
        self.agent_name = agent_name
        self.crew_name = crew_name
        self.manager = manager
        self.agent_id = f"{crew_name}:{agent_name}"
        self._message_handlers = []
        
        # 注册消息处理器
        self.manager.on_any_message(self._handle_message)
    
    async def send_message(self, payload: Dict[str, Any], to_agent: Optional[str] = None) -> bool:
        """
        发送消息
        
        Args:
            payload: 消息内容
            to_agent: 目标Agent名称（None表示广播）
        """
        msg = MessageBuilder().event() \
            .from_agent(self.agent_id) \
            .with_payload(payload) \
            .build()
        
        if to_agent:
            msg.receiver = f"{self.crew_name}:{to_agent}"
        
        return await self.manager.send(msg)
    
    async def send_request(self, payload: Dict[str, Any], to_agent: str, timeout: float = 30.0) -> Optional[Dict]:
        """
        发送请求并等待响应
        
        Args:
            payload: 请求内容
            to_agent: 目标Agent
            timeout: 超时时间
        
        Returns:
            响应内容
        """
        receiver = f"{self.crew_name}:{to_agent}"
        response = await self.manager.send_request(receiver, payload, timeout)
        
        if response:
            return response.payload
        return None
    
    def on_message(self, handler):
        """注册消息处理器"""
        self._message_handlers.append(handler)
    
    async def _handle_message(self, message: Message):
        """内部消息处理"""
        # 检查是否是发给本Agent的
        if message.receiver and message.receiver != self.agent_id:
            return
        
        # 调用注册的处理器
        for handler in self._message_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message.payload)
                else:
                    handler(message.payload)
            except Exception as e:
                logger.error(f"Message handler error: {e}")


class WorkflowCommunicationBridge:
    """
    工作流通信桥
    
    将通信协议集成到工作流引擎中，使工作流任务可以通过消息传递数据。
    """
    
    def __init__(self, workflow_engine):
        self.workflow_engine = workflow_engine
        self.manager: Optional[CommunicationManager] = None
        self._task_channels: Dict[str, Any] = {}
    
    async def initialize(self):
        """初始化通信桥"""
        # 创建通信管理器
        crew_name = getattr(self.workflow_engine, 'crew_name', 'workflow')
        self.manager = CommunicationManager(f"workflow:{crew_name}")
        
        # 添加本地传输
        self.manager.add_transport(TransportConfig(
            name="local",
            transport_class=LocalTransport,
            config={},
            priority=0
        ))
        
        await self.manager.start()
        logger.info("Workflow communication bridge initialized")
    
    async def dispatch_task_with_communication(self, task_id: str, agent_name: str,
                                               task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分发任务并建立通信通道
        
        Args:
            task_id: 任务ID
            agent_name: Agent名称
            task_data: 任务数据
        
        Returns:
            任务结果
        """
        # 创建任务通信通道
        channel = TaskCommunicationChannel(task_id, self.manager)
        self._task_channels[task_id] = channel
        
        try:
            # 发送任务消息
            msg = MessageBuilder().request() \
                .from_agent("workflow") \
                .to_agent(agent_name) \
                .with_payload({
                    "task_id": task_id,
                    "task": task_data
                }) \
                .build()
            
            # 等待结果
            response = await self.manager.send_request(
                receiver=agent_name,
                payload=msg.payload,
                timeout=task_data.get("timeout", 300)
            )
            
            if response:
                return response.payload
            else:
                return {"status": "timeout"}
                
        finally:
            del self._task_channels[task_id]
    
    async def cleanup(self):
        """清理资源"""
        if self.manager:
            await self.manager.stop()


class TaskCommunicationChannel:
    """任务通信通道 - 用于工作流任务间通信"""
    
    def __init__(self, task_id: str, manager: CommunicationManager):
        self.task_id = task_id
        self.manager = manager
        self._messages = []
    
    async def send_progress(self, progress: float, message: str = ""):
        """发送进度更新"""
        await self.manager.broadcast({
            "type": "task_progress",
            "task_id": self.task_id,
            "progress": progress,
            "message": message
        })
    
    async def send_result(self, result: Any):
        """发送结果"""
        await self.manager.broadcast({
            "type": "task_result",
            "task_id": self.task_id,
            "result": result
        })
