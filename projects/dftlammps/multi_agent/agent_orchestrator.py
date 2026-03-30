"""
Agent Orchestrator - 智能体编排器
协调多个Agent协作
"""
from __future__ import annotations
import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .agent_core import BaseAgent, Message, MessageType, AgentStatus
from .communication_bus import CommunicationBus, AsyncCommunicationBus, get_communication_bus
from .consensus_mechanism import ConsensusManager, MajorityConsensus


class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class CollaborationMode(Enum):
    """协作模式"""
    HIERARCHICAL = auto()      # 层级协作
    PEER_TO_PEER = auto()      # 点对点协作
    MARKET_BASED = auto()      # 基于市场的协作
    CONSENSUS_BASED = auto()   # 基于共识的协作


@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    description: str
    required_capabilities: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "required_capabilities": self.required_capabilities,
            "assigned_to": self.assigned_to,
            "status": self.status.name,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "metadata": self.metadata
        }


@dataclass
class Workflow:
    """工作流定义"""
    id: str
    name: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    task_order: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def add_task(self, task: Task, after: Optional[List[str]] = None) -> None:
        """添加任务"""
        self.tasks[task.id] = task
        
        if after:
            task.dependencies = after
            # 在依赖任务后插入
            max_idx = -1
            for dep_id in after:
                if dep_id in self.task_order:
                    max_idx = max(max_idx, self.task_order.index(dep_id))
            self.task_order.insert(max_idx + 1, task.id)
        else:
            self.task_order.append(task.id)
    
    def get_ready_tasks(self) -> List[Task]:
        """获取可执行的任务（依赖已完成）"""
        ready = []
        for task_id in self.task_order:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                # 检查依赖
                deps_completed = all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_completed:
                    ready.append(task)
        return ready


class AgentRegistry:
    """Agent注册中心"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.agent_groups: Dict[str, Set[str]] = defaultdict(set)
        self.agent_status: Dict[str, AgentStatus] = {}
    
    def register(self, agent: BaseAgent) -> None:
        """注册Agent"""
        self.agents[agent.agent_id] = agent
        self.agent_status[agent.agent_id] = agent.status
        
        # 提取能力
        for cap_name in agent.capabilities.keys():
            self.agent_capabilities[agent.agent_id].add(cap_name)
    
    def unregister(self, agent_id: str) -> None:
        """注销Agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_capabilities[agent_id]
            del self.agent_status[agent_id]
            
            # 从所有组中移除
            for group in self.agent_groups.values():
                group.discard(agent_id)
    
    def add_to_group(self, agent_id: str, group: str) -> None:
        """将Agent添加到组"""
        self.agent_groups[group].add(agent_id)
    
    def remove_from_group(self, agent_id: str, group: str) -> None:
        """从组中移除Agent"""
        self.agent_groups[group].discard(agent_id)
    
    def find_agents_by_capability(
        self,
        capability: str,
        available_only: bool = True
    ) -> List[str]:
        """根据能力查找Agent"""
        matching = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if capability in capabilities:
                if not available_only or self.agent_status.get(agent_id) == AgentStatus.IDLE:
                    matching.append(agent_id)
        return matching
    
    def find_agents_by_group(self, group: str) -> List[str]:
        """根据组查找Agent"""
        return list(self.agent_groups[group])
    
    def update_status(self, agent_id: str, status: AgentStatus) -> None:
        """更新Agent状态"""
        self.agent_status[agent_id] = status
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取Agent信息"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            "agent_id": agent_id,
            "name": agent.name,
            "status": self.agent_status.get(agent_id, AgentStatus.IDLE).name,
            "capabilities": list(self.agent_capabilities[agent_id]),
            "groups": [
                group for group, members in self.agent_groups.items()
                if agent_id in members
            ]
        }


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.pending_tasks: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.scheduling_strategy: str = "capability_match"
    
    def submit_task(self, task: Task) -> None:
        """提交任务"""
        self.pending_tasks.append(task)
        # 按优先级排序
        self.pending_tasks.sort(key=lambda t: t.priority, reverse=True)
    
    def schedule(self) -> Optional[Tuple[Task, str]]:
        """
        调度任务
        返回 (任务, AgentID) 或 None
        """
        if not self.pending_tasks:
            return None
        
        for task in self.pending_tasks:
            # 找到合适的Agent
            agent_id = self._find_best_agent(task)
            if agent_id:
                task.assigned_to = agent_id
                task.status = TaskStatus.ASSIGNED
                
                self.pending_tasks.remove(task)
                self.active_tasks[task.id] = task
                
                return task, agent_id
        
        return None
    
    def _find_best_agent(self, task: Task) -> Optional[str]:
        """找到最适合执行任务的Agent"""
        candidates = []
        
        for capability in task.required_capabilities:
            agents = self.registry.find_agents_by_capability(capability)
            candidates.extend(agents)
        
        if not candidates:
            return None
        
        # 去重并选择负载最低的
        candidate_set = set(candidates)
        
        # 计算每个候选的当前任务数
        task_counts = defaultdict(int)
        for active_task in self.active_tasks.values():
            if active_task.assigned_to:
                task_counts[active_task.assigned_to] += 1
        
        # 选择任务最少的
        best_agent = min(candidate_set, key=lambda a: task_counts[a])
        return best_agent
    
    def complete_task(self, task_id: str, result: Any) -> None:
        """完成任务"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            self.completed_tasks[task_id] = task
    
    def fail_task(self, task_id: str, error: str) -> None:
        """标记任务失败"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task.status = TaskStatus.FAILED
            task.metadata["error"] = error
            self.completed_tasks[task_id] = task


class AgentOrchestrator:
    """
    Agent编排器
    协调多个Agent的协作
    """
    
    def __init__(
        self,
        communication_bus: Optional[CommunicationBus] = None,
        collaboration_mode: CollaborationMode = CollaborationMode.HIERARCHICAL
    ):
        self.communication_bus = communication_bus or get_communication_bus()
        self.collaboration_mode = collaboration_mode
        
        # 核心组件
        self.registry = AgentRegistry()
        self.scheduler = TaskScheduler(self.registry)
        self.consensus_manager = ConsensusManager()
        
        # 工作流管理
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Set[str] = set()
        
        # 协作管理
        self.collaboration_sessions: Dict[str, Dict[str, Any]] = {}
        
        # 监控和回调
        self.task_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.on_workflow_complete: Optional[Callable[[str, Dict], None]] = None
        
        # 运行状态
        self._running = False
        self._orchestrator_task: Optional[asyncio.Task] = None
        
        # 设置通信
        self._setup_communication()
    
    def _setup_communication(self) -> None:
        """设置通信"""
        # 订阅系统消息
        if hasattr(self.communication_bus, 'register_callback'):
            self.communication_bus.register_callback(
                MessageType.SYSTEM.name,
                self._handle_system_message
            )
            self.communication_bus.register_callback(
                MessageType.RESULT.name,
                self._handle_result_message
            )
    
    def register_agent(self, agent: BaseAgent, groups: Optional[List[str]] = None) -> None:
        """注册Agent"""
        self.registry.register(agent)
        self.communication_bus.register_agent(agent.agent_id, agent)
        
        if groups:
            for group in groups:
                self.registry.add_to_group(agent.agent_id, group)
        
        # 向Agent注册消息处理器
        agent.register_message_handler(
            MessageType.ACTION,
            self._create_task_handler(agent)
        )
    
    def _create_task_handler(self, agent: BaseAgent) -> Callable:
        """创建任务处理器"""
        async def handler(message: Message) -> None:
            task_data = message.content.get("task")
            if task_data:
                # 更新任务状态
                task_id = task_data.get("id")
                if task_id in self.scheduler.active_tasks:
                    self.scheduler.active_tasks[task_id].status = TaskStatus.IN_PROGRESS
                    self.scheduler.active_tasks[task_id].started_at = datetime.now()
        
        return handler
    
    def create_workflow(self, name: str, workflow_id: Optional[str] = None) -> Workflow:
        """创建工作流"""
        workflow = Workflow(
            id=workflow_id or str(uuid.uuid4()),
            name=name
        )
        self.workflows[workflow.id] = workflow
        return workflow
    
    def add_task_to_workflow(
        self,
        workflow_id: str,
        task: Task,
        after: Optional[List[str]] = None
    ) -> bool:
        """添加任务到工作流"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.add_task(task, after)
        
        # 同时提交到调度器
        self.scheduler.submit_task(task)
        
        return True
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """执行工作流"""
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id]
        self.active_workflows.add(workflow_id)
        workflow.status = TaskStatus.IN_PROGRESS
        
        results = {}
        
        try:
            while True:
                # 获取可执行的任务
                ready_tasks = workflow.get_ready_tasks()
                
                if not ready_tasks:
                    # 检查是否全部完成
                    all_done = all(
                        t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                        for t in workflow.tasks.values()
                    )
                    if all_done:
                        break
                    
                    await asyncio.sleep(0.1)
                    continue
                
                # 调度并执行任务
                for task in ready_tasks:
                    scheduled = self.scheduler.schedule()
                    if scheduled and scheduled[0].id == task.id:
                        task, agent_id = scheduled
                        
                        # 发送任务给Agent
                        await self._assign_task(task, agent_id)
                
                await asyncio.sleep(0.1)
            
            # 收集结果
            workflow.status = TaskStatus.COMPLETED
            workflow.completed_at = datetime.now()
            
            results = {
                task_id: task.result
                for task_id, task in workflow.tasks.items()
            }
            
            if self.on_workflow_complete:
                if asyncio.iscoroutinefunction(self.on_workflow_complete):
                    await self.on_workflow_complete(workflow_id, results)
                else:
                    self.on_workflow_complete(workflow_id, results)
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "results": results
            }
            
        except Exception as e:
            workflow.status = TaskStatus.FAILED
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e)
            }
        finally:
            self.active_workflows.discard(workflow_id)
    
    async def _assign_task(self, task: Task, agent_id: str) -> None:
        """分配任务给Agent"""
        message = Message(
            sender_id="orchestrator",
            receiver_id=agent_id,
            message_type=MessageType.ACTION,
            content={"task": task.to_dict()},
            metadata={"workflow_id": task.metadata.get("workflow_id")}
        )
        
        await self.communication_bus.send(message)
    
    def _handle_system_message(self, message: Message) -> None:
        """处理系统消息"""
        pass
    
    def _handle_result_message(self, message: Message) -> None:
        """处理结果消息"""
        content = message.content
        task_id = content.get("task_id")
        result = content.get("result")
        success = content.get("success", True)
        
        if task_id:
            if success:
                self.scheduler.complete_task(task_id, result)
            else:
                self.scheduler.fail_task(task_id, content.get("error", "Unknown error"))
    
    async def start_collaboration(
        self,
        collaboration_id: str,
        participants: List[str],
        goal: Dict[str, Any]
    ) -> None:
        """启动协作会话"""
        self.collaboration_sessions[collaboration_id] = {
            "participants": participants,
            "goal": goal,
            "started_at": datetime.now(),
            "status": "active",
            "messages": []
        }
        
        # 通知参与者
        for participant in participants:
            await self.communication_bus.send(Message(
                sender_id="orchestrator",
                receiver_id=participant,
                message_type=MessageType.SYSTEM,
                content={
                    "event": "collaboration_started",
                    "collaboration_id": collaboration_id,
                    "goal": goal
                }
            ))
    
    async def coordinate_consensus(
        self,
        proposal_id: str,
        content: Dict[str, Any],
        description: str,
        participants: Optional[List[str]] = None
    ) -> str:
        """
        协调共识决策
        返回提案ID
        """
        if participants is None:
            participants = list(self.registry.agents.keys())
        
        # 注册参与者
        for p in participants:
            self.consensus_manager.register_participant(p)
        
        # 创建提案
        proposal = self.consensus_manager.create_proposal(
            proposal_id=proposal_id,
            proposer_id="orchestrator",
            content=content,
            description=description,
            required_approvals=len(participants) // 2 + 1
        )
        
        # 通知参与者投票
        for participant in participants:
            await self.communication_bus.send(Message(
                sender_id="orchestrator",
                receiver_id=participant,
                message_type=MessageType.SYSTEM,
                content={
                    "event": "vote_request",
                    "proposal_id": proposal_id,
                    "description": description,
                    "content": content
                }
            ))
        
        return proposal_id
    
    async def start(self) -> None:
        """启动编排器"""
        self._running = True
        await self.consensus_manager.start()
        self._orchestrator_task = asyncio.create_task(self._main_loop())
    
    async def stop(self) -> None:
        """停止编排器"""
        self._running = False
        
        if self._orchestrator_task:
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except asyncio.CancelledError:
                pass
        
        await self.consensus_manager.stop()
    
    async def _main_loop(self) -> None:
        """主循环"""
        while self._running:
            # 持续调度任务
            self.scheduler.schedule()
            
            # 更新Agent状态
            for agent_id, agent in self.registry.agents.items():
                self.registry.update_status(agent_id, agent.status)
            
            await asyncio.sleep(0.5)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "agents": {
                agent_id: self.registry.get_agent_info(agent_id)
                for agent_id in self.registry.agents.keys()
            },
            "tasks": {
                "pending": len(self.scheduler.pending_tasks),
                "active": len(self.scheduler.active_tasks),
                "completed": len(self.scheduler.completed_tasks)
            },
            "workflows": {
                wid: {
                    "name": w.name,
                    "status": w.status.name,
                    "tasks": len(w.tasks)
                }
                for wid, w in self.workflows.items()
            },
            "active_collaborations": list(self.collaboration_sessions.keys())
        }


class MultiAgentCollaboration:
    """
    多Agent协作辅助类
    简化协作操作
    """
    
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
    
    async def collaborative_problem_solving(
        self,
        problem: Dict[str, Any],
        agent_roles: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        协作问题解决
        
        Args:
            problem: 问题定义
            agent_roles: {角色: [Agent ID列表]}
        """
        collaboration_id = f"collab_{uuid.uuid4().hex[:8]}"
        
        # 收集所有参与者
        all_participants = []
        for agents in agent_roles.values():
            all_participants.extend(agents)
        all_participants = list(set(all_participants))
        
        # 启动协作
        await self.orchestrator.start_collaboration(
            collaboration_id,
            all_participants,
            problem
        )
        
        # 创建工作流
        workflow = self.orchestrator.create_workflow(
            f"Collaborative_Solving_{collaboration_id}"
        )
        
        # 为每个角色创建任务
        step = 0
        for role, agents in agent_roles.items():
            task = Task(
                id=f"task_{collaboration_id}_{step}",
                name=f"{role}_contribution",
                description=f"Contribution from {role}",
                required_capabilities=[role],
                metadata={
                    "role": role,
                    "collaboration_id": collaboration_id,
                    "workflow_id": workflow.id
                }
            )
            self.orchestrator.add_task_to_workflow(workflow.id, task)
            step += 1
        
        # 执行工作流
        result = await self.orchestrator.execute_workflow(workflow.id)
        
        return {
            "collaboration_id": collaboration_id,
            "result": result
        }
    
    async def peer_review_process(
        self,
        work_id: str,
        authors: List[str],
        reviewers: List[str]
    ) -> Dict[str, Any]:
        """
        同行评审流程
        """
        workflow = self.orchestrator.create_workflow(f"Review_{work_id}")
        
        # 作者提交
        submit_task = Task(
            id=f"submit_{work_id}",
            name="submit_work",
            description="Authors submit work for review",
            required_capabilities=["author"]
        )
        self.orchestrator.add_task_to_workflow(workflow.id, submit_task)
        
        # 评审
        review_tasks = []
        for i, reviewer in enumerate(reviewers):
            review_task = Task(
                id=f"review_{work_id}_{i}",
                name=f"review_by_{reviewer}",
                description=f"Review by {reviewer}",
                required_capabilities=["review"],
                dependencies=[submit_task.id]
            )
            self.orchestrator.add_task_to_workflow(
                workflow.id,
                review_task,
                after=[submit_task.id]
            )
            review_tasks.append(review_task)
        
        # 汇总
        consolidate_task = Task(
            id=f"consolidate_{work_id}",
            name="consolidate_reviews",
            description="Consolidate all reviews",
            required_capabilities=["consolidate"],
            dependencies=[t.id for t in review_tasks]
        )
        self.orchestrator.add_task_to_workflow(
            workflow.id,
            consolidate_task,
            after=[t.id for t in review_tasks]
        )
        
        return await self.orchestrator.execute_workflow(workflow.id)


class MarketBasedOrchestrator(AgentOrchestrator):
    """
    基于市场的编排器
    使用拍卖机制分配任务
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auctions: Dict[str, Dict[str, Any]] = {}
    
    async def auction_task(self, task: Task, timeout: float = 5.0) -> Optional[str]:
        """
        拍卖任务
        返回中标Agent ID
        """
        auction_id = f"auction_{task.id}"
        
        # 找到所有有能力的Agent
        candidates = []
        for capability in task.required_capabilities:
            candidates.extend(self.registry.find_agents_by_capability(capability, False))
        
        candidates = list(set(candidates))
        
        # 发送拍卖邀请
        bids = {}
        
        for candidate in candidates:
            # 实际应用中，这里应该发送消息并等待回复
            # 简化：假设Agent会根据自己的负载出价
            agent = self.registry.agents[candidate]
            load = sum(
                1 for t in self.scheduler.active_tasks.values()
                if t.assigned_to == candidate
            )
            # 负载越低，出价越高（更愿意接受）
            bid = 100 / (load + 1)
            bids[candidate] = bid
        
        if not bids:
            return None
        
        # 选择最高出价
        winner = max(bids.items(), key=lambda x: x[1])
        return winner[0]
    
    async def _assign_task(self, task: Task, agent_id: str) -> None:
        """重写任务分配，使用拍卖"""
        winner = await self.auction_task(task)
        if winner:
            task.assigned_to = winner
            await super()._assign_task(task, winner)
