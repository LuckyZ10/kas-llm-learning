"""
Agent Core - 智能体基类
实现感知-推理-行动循环（Perceive-Reason-Act Loop）
"""
from __future__ import annotations
import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar
from collections import deque
import json


class MessageType(Enum):
    """消息类型枚举"""
    PERCEPTION = auto()      # 感知输入
    REASONING = auto()       # 推理过程
    ACTION = auto()          # 行动指令
    COMMUNICATION = auto()   # 通信消息
    RESULT = auto()          # 结果返回
    ERROR = auto()           # 错误信息
    SYSTEM = auto()          # 系统消息


class AgentStatus(Enum):
    """智能体状态枚举"""
    IDLE = auto()
    PERCEIVING = auto()
    REASONING = auto()
    ACTING = auto()
    COMMUNICATING = auto()
    ERROR = auto()


@dataclass
class Message:
    """标准化消息协议"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: Optional[str] = None  # None表示广播
    message_type: MessageType = MessageType.SYSTEM
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            receiver_id=data.get("receiver_id"),
            message_type=MessageType[data["message_type"]],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class Observation:
    """感知观测数据"""
    source: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence
        }


@dataclass
class Action:
    """行动定义"""
    action_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 优先级，数字越大优先级越高
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "params": self.params,
            "priority": self.priority
        }


T = TypeVar('T')


class MemoryBuffer(Generic[T]):
    """有限容量的记忆缓冲区"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: deque[T] = deque(maxlen=capacity)
        self.index: Dict[str, T] = {}  # 用于快速查找
    
    def add(self, item: T, key: Optional[str] = None) -> None:
        """添加记忆项"""
        self.buffer.append(item)
        if key:
            self.index[key] = item
    
    def get_recent(self, n: int = 10) -> List[T]:
        """获取最近的n条记忆"""
        return list(self.buffer)[-n:]
    
    def get_all(self) -> List[T]:
        """获取所有记忆"""
        return list(self.buffer)
    
    def search(self, predicate: Callable[[T], bool]) -> List[T]:
        """根据条件搜索记忆"""
        return [item for item in self.buffer if predicate(item)]
    
    def get_by_key(self, key: str) -> Optional[T]:
        """通过key获取记忆"""
        return self.index.get(key)
    
    def clear(self) -> None:
        """清空记忆"""
        self.buffer.clear()
        self.index.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)


class WorkingMemory:
    """工作记忆 - 短期上下文"""
    
    def __init__(self, capacity: int = 50):
        self.observations: MemoryBuffer[Observation] = MemoryBuffer(capacity)
        self.messages: MemoryBuffer[Message] = MemoryBuffer(capacity)
        self.current_context: Dict[str, Any] = {}
        self.focus: Optional[str] = None  # 当前关注点
    
    def add_observation(self, observation: Observation) -> None:
        self.observations.add(observation)
    
    def add_message(self, message: Message) -> None:
        self.messages.add(message)
    
    def set_context(self, key: str, value: Any) -> None:
        self.current_context[key] = value
    
    def get_context(self, key: str) -> Optional[Any]:
        return self.current_context.get(key)
    
    def get_relevant_observations(self, n: int = 5) -> List[Observation]:
        """获取相关的最近观测"""
        return self.observations.get_recent(n)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """获取工作记忆摘要"""
        return {
            "recent_observations": [o.to_dict() for o in self.observations.get_recent(5)],
            "recent_messages": [m.to_dict() for m in self.messages.get_recent(5)],
            "current_context": self.current_context,
            "focus": self.focus
        }


class LongTermMemory:
    """长期记忆 - 持久化知识"""
    
    def __init__(self):
        self.facts: Dict[str, Any] = {}
        self.experiences: MemoryBuffer[Dict[str, Any]] = MemoryBuffer(10000)
        self.learned_patterns: Dict[str, Any] = {}
        self.relationships: Dict[str, Set[str]] = {}  # 实体关系图
    
    def store_fact(self, key: str, value: Any) -> None:
        """存储事实知识"""
        self.facts[key] = value
    
    def retrieve_fact(self, key: str) -> Optional[Any]:
        """检索事实"""
        return self.facts.get(key)
    
    def add_experience(self, experience: Dict[str, Any], key: Optional[str] = None) -> None:
        """添加经验"""
        experience["timestamp"] = datetime.now().isoformat()
        self.experiences.add(experience, key)
    
    def find_similar_experiences(self, context: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """查找相似经验（简单实现）"""
        def similarity_score(exp: Dict[str, Any]) -> float:
            score = 0.0
            for k, v in context.items():
                if k in exp and exp[k] == v:
                    score += 1.0
            return score
        
        experiences = self.experiences.get_all()
        scored = [(exp, similarity_score(exp)) for exp in experiences]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, score in scored[:limit] if score > 0]
    
    def learn_pattern(self, pattern_name: str, pattern_data: Any) -> None:
        """学习模式"""
        self.learned_patterns[pattern_name] = pattern_data
    
    def add_relationship(self, entity1: str, entity2: str, relation_type: str = "related") -> None:
        """添加实体关系"""
        if entity1 not in self.relationships:
            self.relationships[entity1] = set()
        self.relationships[entity1].add(f"{relation_type}:{entity2}")
    
    def get_related(self, entity: str) -> Set[str]:
        """获取相关实体"""
        relations = self.relationships.get(entity, set())
        return {r.split(":", 1)[1] for r in relations}


class AgentCapability:
    """Agent能力定义"""
    
    def __init__(self, name: str, description: str, handler: Callable[..., Any]):
        self.name = name
        self.description = description
        self.handler = handler
    
    async def execute(self, **kwargs) -> Any:
        """执行能力"""
        if asyncio.iscoroutinefunction(self.handler):
            return await self.handler(**kwargs)
        return self.handler(**kwargs)


class BaseAgent(ABC):
    """
    智能体基类 - 实现感知-推理-行动循环
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: str = "Agent",
        description: str = "",
        memory_capacity: int = 1000
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.status = AgentStatus.IDLE
        
        # 记忆系统
        self.working_memory = WorkingMemory(capacity=50)
        self.long_term_memory = LongTermMemory()
        
        # 消息系统
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.outbox: asyncio.Queue[Message] = asyncio.Queue()
        self.message_handlers: Dict[MessageType, List[Callable[[Message], Any]]] = {
            mt: [] for mt in MessageType
        }
        
        # 能力注册
        self.capabilities: Dict[str, AgentCapability] = {}
        
        # 配置
        self.config = {
            "max_reasoning_iterations": 10,
            "perception_interval": 1.0,
            "action_timeout": 30.0,
            "enable_async": True
        }
        
        # 统计
        self.stats = {
            "perceptions": 0,
            "reasoning_steps": 0,
            "actions_executed": 0,
            "messages_received": 0,
            "messages_sent": 0,
            "errors": 0
        }
        
        # 运行控制
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
    
    def register_capability(self, capability: AgentCapability) -> None:
        """注册能力"""
        self.capabilities[capability.name] = capability
    
    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Any]
    ) -> None:
        """注册消息处理器"""
        self.message_handlers[message_type].append(handler)
    
    # ===== 感知阶段 =====
    
    @abstractmethod
    async def perceive(self) -> List[Observation]:
        """
        感知环境 - 从环境获取信息
        子类必须实现
        """
        pass
    
    async def _perception_loop(self) -> None:
        """感知循环"""
        while self._running:
            try:
                self.status = AgentStatus.PERCEIVING
                observations = await self.perceive()
                
                for obs in observations:
                    self.working_memory.add_observation(obs)
                    self.stats["perceptions"] += 1
                
                # 处理感知结果（触发推理）
                if observations:
                    await self._on_new_observations(observations)
                
                await asyncio.sleep(self.config["perception_interval"])
            except Exception as e:
                self.stats["errors"] += 1
                await self._handle_error("perception", e)
    
    async def _on_new_observations(self, observations: List[Observation]) -> None:
        """新观测到达时的回调"""
        pass
    
    # ===== 推理阶段 =====
    
    @abstractmethod
    async def reason(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        推理决策 - 基于感知信息做出决策
        子类必须实现
        """
        pass
    
    async def _reasoning_process(self, trigger: Optional[Any] = None) -> Dict[str, Any]:
        """推理过程"""
        self.status = AgentStatus.REASONING
        
        # 构建上下文
        context = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "working_memory": self.working_memory.get_context_summary(),
            "long_term_facts": self.long_term_memory.facts,
            "trigger": trigger,
            "capabilities": list(self.capabilities.keys())
        }
        
        try:
            result = await self.reason(context)
            self.stats["reasoning_steps"] += 1
            
            # 存储推理结果
            self.working_memory.set_context("last_reasoning", result)
            
            return result
        except Exception as e:
            self.stats["errors"] += 1
            await self._handle_error("reasoning", e)
            return {"error": str(e)}
    
    # ===== 行动阶段 =====
    
    @abstractmethod
    async def act(self, decision: Dict[str, Any]) -> List[Action]:
        """
        执行行动 - 将决策转化为具体行动
        子类必须实现
        """
        pass
    
    async def _action_execution(self, actions: List[Action]) -> List[Any]:
        """执行行动序列"""
        self.status = AgentStatus.ACTING
        results = []
        
        # 按优先级排序
        actions.sort(key=lambda a: a.priority, reverse=True)
        
        for action in actions:
            try:
                if action.action_type in self.capabilities:
                    capability = self.capabilities[action.action_type]
                    result = await capability.execute(**action.params)
                    results.append({
                        "action": action.to_dict(),
                        "result": result,
                        "success": True
                    })
                    self.stats["actions_executed"] += 1
                else:
                    results.append({
                        "action": action.to_dict(),
                        "error": f"Unknown capability: {action.action_type}",
                        "success": False
                    })
            except Exception as e:
                self.stats["errors"] += 1
                results.append({
                    "action": action.to_dict(),
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    # ===== 通信阶段 =====
    
    async def send_message(
        self,
        content: Dict[str, Any],
        receiver_id: Optional[str] = None,
        message_type: MessageType = MessageType.COMMUNICATION,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """发送消息"""
        message = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
        
        await self.outbox.put(message)
        self.stats["messages_sent"] += 1
        return message
    
    async def broadcast(
        self,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.COMMUNICATION
    ) -> Message:
        """广播消息"""
        return await self.send_message(content, None, message_type)
    
    async def _message_processing_loop(self) -> None:
        """消息处理循环"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.inbox.get(),
                    timeout=1.0
                )
                
                self.status = AgentStatus.COMMUNICATING
                self.working_memory.add_message(message)
                self.stats["messages_received"] += 1
                
                # 调用处理器
                handlers = self.message_handlers.get(message.message_type, [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        self.stats["errors"] += 1
                
                # 处理完消息后可能触发推理
                await self._on_message_received(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.stats["errors"] += 1
                await self._handle_error("message_processing", e)
    
    async def _on_message_received(self, message: Message) -> None:
        """消息接收回调"""
        pass
    
    # ===== 主循环 =====
    
    async def run_cycle(self) -> None:
        """运行一个完整的感知-推理-行动循环"""
        try:
            # 感知
            observations = await self.perceive()
            for obs in observations:
                self.working_memory.add_observation(obs)
            
            # 推理
            context = {
                "observations": [o.to_dict() for o in observations],
                "agent_state": self.status.name
            }
            decision = await self._reasoning_process(context)
            
            # 行动
            actions = await self.act(decision)
            if actions:
                await self._action_execution(actions)
                
        except Exception as e:
            self.stats["errors"] += 1
            await self._handle_error("cycle", e)
    
    async def run(self) -> None:
        """启动Agent主循环"""
        self._running = True
        
        # 启动后台任务
        if self.config["enable_async"]:
            self._tasks.add(asyncio.create_task(self._perception_loop()))
            self._tasks.add(asyncio.create_task(self._message_processing_loop()))
        
        while self._running:
            await self.run_cycle()
            await asyncio.sleep(0.1)
    
    async def stop(self) -> None:
        """停止Agent"""
        self._running = False
        
        # 取消所有任务
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    async def _handle_error(self, phase: str, error: Exception) -> None:
        """错误处理"""
        self.status = AgentStatus.ERROR
        error_msg = f"Error in {phase}: {str(error)}"
        
        # 发送错误消息
        await self.send_message(
            {"phase": phase, "error": str(error)},
            message_type=MessageType.ERROR
        )
    
    # ===== 工具方法 =====
    
    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.name,
            "description": self.description,
            "stats": self.stats.copy(),
            "capabilities": list(self.capabilities.keys()),
            "working_memory_size": len(self.working_memory.observations),
            "long_term_facts": len(self.long_term_memory.facts)
        }
    
    def export_memory(self) -> Dict[str, Any]:
        """导出记忆"""
        return {
            "working_memory": self.working_memory.get_context_summary(),
            "long_term_facts": self.long_term_memory.facts,
            "experiences_count": len(self.long_term_memory.experiences),
            "learned_patterns": self.long_term_memory.learned_patterns
        }
    
    def import_memory(self, data: Dict[str, Any]) -> None:
        """导入记忆"""
        if "long_term_facts" in data:
            self.long_term_memory.facts.update(data["long_term_facts"])
        if "learned_patterns" in data:
            self.long_term_memory.learned_patterns.update(data["learned_patterns"])


class ReactiveAgent(BaseAgent):
    """
    反应式智能体 - 简单的感知-行动映射
    适用于快速响应场景
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rules: List[Callable[[List[Observation]], Optional[Action]]] = []
    
    def add_rule(self, rule: Callable[[List[Observation]], Optional[Action]]) -> None:
        """添加反应规则"""
        self.rules.append(rule)
    
    async def perceive(self) -> List[Observation]:
        """感知 - 默认实现，子类可覆盖"""
        return []
    
    async def reason(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """推理 - 简化版，基于规则匹配"""
        observations = context.get("observations", [])
        
        for rule in self.rules:
            action = rule(observations)
            if action:
                return {"action": action}
        
        return {"action": None}
    
    async def act(self, decision: Dict[str, Any]) -> List[Action]:
        """行动"""
        action = decision.get("action")
        if action:
            return [action]
        return []


class DeliberativeAgent(BaseAgent):
    """
    慎思式智能体 - 完整的感知-推理-行动循环
    支持复杂的规划和决策
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.goals: List[Dict[str, Any]] = []
        self.plans: List[Dict[str, Any]] = []
        self.current_plan: Optional[Dict[str, Any]] = None
    
    def add_goal(self, goal: Dict[str, Any]) -> None:
        """添加目标"""
        self.goals.append(goal)
    
    async def plan(self, goal: Dict[str, Any], context: Dict[str, Any]) -> List[Action]:
        """
        规划 - 生成达成目标的行动序列
        子类应实现具体的规划算法
        """
        return []
    
    async def perceive(self) -> List[Observation]:
        """感知 - 默认实现"""
        return []
    
    async def reason(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """推理 - 包含目标选择和规划"""
        # 选择目标
        if self.goals:
            goal = self._select_goal(context)
            
            # 规划
            plan = await self.plan(goal, context)
            
            return {
                "goal": goal,
                "plan": plan
            }
        
        return {}
    
    def _select_goal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """目标选择策略 - 默认选择第一个"""
        return self.goals[0] if self.goals else {}
    
    async def act(self, decision: Dict[str, Any]) -> List[Action]:
        """行动 - 执行计划"""
        plan = decision.get("plan", [])
        return plan
