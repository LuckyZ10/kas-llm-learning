"""
Coordinator Agent - 协调员Agent
负责任务分配、进度管理、冲突解决
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

from ..multi_agent.agent_core import (
    DeliberativeAgent, Message, MessageType,
    Observation, Action, AgentCapability
)


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


class ConflictType(Enum):
    """冲突类型"""
    RESOURCE = "resource"           # 资源竞争
    PRIORITY = "priority"           # 优先级冲突
    METHODOLOGY = "methodology"     # 方法论分歧
    CREDIT = "credit"               # 贡献归属
    DEADLINE = "deadline"           # 时间冲突


@dataclass
class Project:
    """项目"""
    id: str
    name: str
    description: str
    objectives: List[str] = field(default_factory=list)
    
    # 时间线
    start_date: datetime = field(default_factory=datetime.now)
    target_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    
    # 任务
    tasks: Dict[str, 'TaskInfo'] = field(default_factory=dict)
    task_order: List[str] = field(default_factory=list)
    
    # 参与者
    participants: Set[str] = field(default_factory=set)
    lead_agent: Optional[str] = None
    
    # 状态
    status: str = "planning"  # planning, active, paused, completed, cancelled
    progress: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "objectives": self.objectives,
            "status": self.status,
            "progress": self.progress,
            "participants": list(self.participants),
            "lead_agent": self.lead_agent,
            "task_count": len(self.tasks),
            "start_date": self.start_date.isoformat(),
            "target_completion": self.target_completion.isoformat() if self.target_completion else None
        }


@dataclass
class TaskInfo:
    """任务信息"""
    id: str
    name: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # 分配
    assigned_to: Optional[str] = None
    reviewers: List[str] = field(default_factory=list)
    
    # 时间
    estimated_duration: Optional[timedelta] = None
    deadline: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 依赖
    dependencies: List[str] = field(default_factory=list)
    
    # 状态
    status: str = "pending"  # pending, assigned, in_progress, review, completed, blocked
    
    # 结果
    deliverables: List[str] = field(default_factory=list)
    output: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.name,
            "assigned_to": self.assigned_to,
            "status": self.status,
            "dependencies": self.dependencies,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class Conflict:
    """冲突"""
    id: str
    conflict_type: ConflictType
    description: str
    
    # 涉及方
    parties: List[str] = field(default_factory=list)
    
    # 相关资源/任务
    related_items: List[str] = field(default_factory=list)
    
    # 状态
    status: str = "identified"  # identified, mediating, resolved, escalated
    
    # 解决方案
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "conflict_type": self.conflict_type.value,
            "description": self.description,
            "parties": self.parties,
            "status": self.status,
            "resolution": self.resolution,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class Schedule:
    """调度计划"""
    id: str
    project_id: str
    
    # 任务安排
    task_schedule: Dict[str, Tuple[datetime, datetime]] = field(default_factory=dict)
    
    # 资源分配
    resource_allocation: Dict[str, List[str]] = field(default_factory=dict)
    
    # 里程碑
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class CoordinatorAgent(DeliberativeAgent):
    """
    协调员Agent
    负责协调多Agent系统的运行
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "CoordinatorAgent")
        kwargs.setdefault("description", "Coordinates tasks, manages progress, resolves conflicts")
        super().__init__(**kwargs)
        
        # 项目库
        self.projects: Dict[str, Project] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.conflicts: Dict[str, Conflict] = {}
        
        # 调度
        self.schedules: Dict[str, Schedule] = {}
        
        # Agent负载
        self.agent_loads: Dict[str, int] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}
        
        # 注册能力
        self._register_capabilities()
        
        # 注册消息处理器
        self.register_message_handler(
            MessageType.COMMUNICATION,
            self._handle_coordination_message
        )
        
        self.register_message_handler(
            MessageType.SYSTEM,
            self._handle_system_message
        )
    
    def _register_capabilities(self) -> None:
        """注册专业能力"""
        self.register_capability(AgentCapability(
            name="create_project",
            description="Create new research project",
            handler=self._create_project_handler
        ))
        
        self.register_capability(AgentCapability(
            name="assign_task",
            description="Assign task to appropriate agent",
            handler=self._assign_task_handler
        ))
        
        self.register_capability(AgentCapability(
            name="schedule_tasks",
            description="Create task schedule",
            handler=self._schedule_tasks_handler
        ))
        
        self.register_capability(AgentCapability(
            name="monitor_progress",
            description="Monitor project progress",
            handler=self._monitor_progress_handler
        ))
        
        self.register_capability(AgentCapability(
            name="resolve_conflict",
            description="Resolve conflicts between agents",
            handler=self._resolve_conflict_handler
        ))
        
        self.register_capability(AgentCapability(
            name="reallocate_resources",
            description="Reallocate resources based on priorities",
            handler=self._reallocate_resources_handler
        ))
    
    async def perceive(self) -> List[Observation]:
        """感知"""
        observations = []
        
        # 检查消息
        while not self.inbox.empty():
            try:
                message = self.inbox.get_nowait()
                observations.append(Observation(
                    source="message",
                    data=message.content,
                    confidence=0.95
                ))
            except asyncio.QueueEmpty:
                break
        
        # 检查即将到期的任务
        now = datetime.now()
        for task in self.tasks.values():
            if task.status == "in_progress" and task.deadline:
                time_remaining = task.deadline - now
                if time_remaining < timedelta(hours=24) and time_remaining > timedelta(0):
                    observations.append(Observation(
                        source=f"task_{task.id}",
                        data={
                            "type": "deadline_approaching",
                            "task_id": task.id,
                            "hours_remaining": time_remaining.total_seconds() / 3600
                        },
                        confidence=0.9
                    ))
        
        # 检查冲突
        for conflict in self.conflicts.values():
            if conflict.status == "identified":
                observations.append(Observation(
                    source=f"conflict_{conflict.id}",
                    data={
                        "type": "new_conflict",
                        "conflict_id": conflict.id,
                        "conflict_type": conflict.conflict_type.value
                    },
                    confidence=0.9
                ))
        
        return observations
    
    async def reason(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """推理"""
        decision = {"actions": [], "reasoning_log": []}
        
        trigger = context.get("trigger")
        
        if trigger:
            trigger_type = trigger.get("type")
            
            if trigger_type == "deadline_approaching":
                task_id = trigger.get("task_id")
                task = self.tasks.get(task_id)
                
                if task:
                    # 检查是否需要重新分配或延长
                    hours_remaining = trigger.get("hours_remaining", 0)
                    
                    if hours_remaining < 4:
                        # 紧急：可能需要重新分配
                        decision["actions"].append({
                            "type": "escalate_task",
                            "task_id": task_id,
                            "reason": "deadline_approaching"
                        })
                    else:
                        # 发送提醒
                        decision["actions"].append({
                            "type": "send_reminder",
                            "task_id": task_id
                        })
            
            elif trigger_type == "new_conflict":
                conflict_id = trigger.get("conflict_id")
                decision["actions"].append({
                    "type": "mediate_conflict",
                    "conflict_id": conflict_id
                })
            
            elif trigger_type == "task_completed":
                task_id = trigger.get("task_id")
                # 检查依赖此任务的其他任务
                dependent_tasks = [
                    t for t in self.tasks.values()
                    if task_id in t.dependencies and t.status == "blocked"
                ]
                
                for dep_task in dependent_tasks:
                    # 检查所有依赖是否都完成
                    if all(
                        self.tasks[dep_id].status == "completed"
                        for dep_id in dep_task.dependencies
                    ):
                        decision["actions"].append({
                            "type": "unblock_task",
                            "task_id": dep_task.id
                        })
        
        # 定期检查整体进度
        for project in self.projects.values():
            if project.status == "active":
                old_progress = project.progress
                new_progress = self._calculate_project_progress(project)
                
                if new_progress != old_progress:
                    project.progress = new_progress
                    
                    # 检查是否完成
                    if new_progress >= 100:
                        decision["actions"].append({
                            "type": "complete_project",
                            "project_id": project.id
                        })
        
        return decision
    
    async def act(self, decision: Dict[str, Any]) -> List[Action]:
        """行动"""
        actions = []
        
        for action_data in decision.get("actions", []):
            action_type = action_data.get("type")
            
            if action_type == "assign_task":
                actions.append(Action(
                    action_type="execute_assignment",
                    params={
                        "task_id": action_data.get("task_id"),
                        "agent_id": action_data.get("agent_id")
                    },
                    priority=3
                ))
            
            elif action_type == "mediate_conflict":
                actions.append(Action(
                    action_type="mediate",
                    params={"conflict_id": action_data.get("conflict_id")},
                    priority=2
                ))
            
            elif action_type == "send_reminder":
                actions.append(Action(
                    action_type="remind",
                    params={"task_id": action_data.get("task_id")},
                    priority=1
                ))
            
            elif action_type == "complete_project":
                actions.append(Action(
                    action_type="finalize_project",
                    params={"project_id": action_data.get("project_id")},
                    priority=2
                ))
        
        return actions
    
    # ===== 核心能力实现 =====
    
    async def create_project(
        self,
        name: str,
        description: str,
        objectives: List[str],
        participants: List[str],
        lead_agent: Optional[str] = None
    ) -> Project:
        """
        创建新项目
        """
        project_id = f"proj_{uuid.uuid4().hex[:8]}"
        
        project = Project(
            id=project_id,
            name=name,
            description=description,
            objectives=objectives,
            participants=set(participants),
            lead_agent=lead_agent,
            status="planning"
        )
        
        self.projects[project_id] = project
        
        # 更新Agent负载
        for participant in participants:
            self.agent_loads[participant] = self.agent_loads.get(participant, 0)
        
        return project
    
    async def add_task(
        self,
        project_id: str,
        name: str,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        required_capabilities: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        deadline: Optional[datetime] = None
    ) -> Optional[TaskInfo]:
        """
        添加任务到项目
        """
        if project_id not in self.projects:
            return None
        
        project = self.projects[project_id]
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = TaskInfo(
            id=task_id,
            name=name,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            deadline=deadline
        )
        
        # 如果有依赖，初始状态设为blocked
        if task.dependencies:
            task.status = "blocked"
        
        self.tasks[task_id] = task
        project.tasks[task_id] = task
        project.task_order.append(task_id)
        
        # 自动分配
        await self._auto_assign_task(task, required_capabilities)
        
        return task
    
    async def _auto_assign_task(
        self,
        task: TaskInfo,
        required_capabilities: Optional[List[str]] = None
    ) -> bool:
        """
        自动分配任务给合适的Agent
        """
        best_agent = None
        best_score = -1
        
        for agent_id, caps in self.agent_capabilities.items():
            score = 0
            
            # 能力匹配
            if required_capabilities:
                matches = sum(1 for cap in required_capabilities if cap in caps)
                score += matches * 10
            
            # 负载均衡（偏好负载低的）
            load = self.agent_loads.get(agent_id, 0)
            score -= load * 2
            
            # 当前状态
            if agent_id in [t.assigned_to for t in self.tasks.values() if t.status == "in_progress"]:
                score -= 5
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        if best_agent:
            task.assigned_to = best_agent
            task.status = "assigned"
            self.agent_loads[best_agent] = self.agent_loads.get(best_agent, 0) + 1
            return True
        
        return False
    
    async def assign_task(
        self,
        task_id: str,
        agent_id: str
    ) -> bool:
        """
        手动分配任务
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # 如果已分配给其他Agent，减少其负载
        if task.assigned_to:
            self.agent_loads[task.assigned_to] = max(
                0, self.agent_loads.get(task.assigned_to, 0) - 1
            )
        
        task.assigned_to = agent_id
        task.status = "assigned"
        self.agent_loads[agent_id] = self.agent_loads.get(agent_id, 0) + 1
        
        # 通知Agent
        await self.send_message(
            {
                "event": "task_assigned",
                "task": task.to_dict()
            },
            receiver_id=agent_id,
            message_type=MessageType.COMMUNICATION
        )
        
        return True
    
    async def create_schedule(self, project_id: str) -> Optional[Schedule]:
        """
        为项目创建调度计划
        """
        if project_id not in self.projects:
            return None
        
        project = self.projects[project_id]
        
        schedule_id = f"schedule_{uuid.uuid4().hex[:8]}"
        
        task_schedule = {}
        resource_allocation = {}
        
        current_time = datetime.now()
        
        for task_id in project.task_order:
            task = project.tasks[task_id]
            
            # 计算开始时间（考虑依赖）
            if task.dependencies:
                latest_end = current_time
                for dep_id in task.dependencies:
                    if dep_id in task_schedule:
                        dep_end = task_schedule[dep_id][1]
                        latest_end = max(latest_end, dep_end)
                start_time = latest_end
            else:
                start_time = current_time
            
            # 估算持续时间
            duration = task.estimated_duration or timedelta(hours=8)
            end_time = start_time + duration
            
            task_schedule[task_id] = (start_time, end_time)
            
            # 资源分配
            if task.assigned_to:
                if task.assigned_to not in resource_allocation:
                    resource_allocation[task.assigned_to] = []
                resource_allocation[task.assigned_to].append(task_id)
        
        schedule = Schedule(
            id=schedule_id,
            project_id=project_id,
            task_schedule=task_schedule,
            resource_allocation=resource_allocation
        )
        
        self.schedules[schedule_id] = schedule
        
        return schedule
    
    async def identify_conflict(
        self,
        conflict_type: ConflictType,
        description: str,
        parties: List[str],
        related_items: Optional[List[str]] = None
    ) -> Conflict:
        """
        识别并记录冲突
        """
        conflict_id = f"conflict_{uuid.uuid4().hex[:8]}"
        
        conflict = Conflict(
            id=conflict_id,
            conflict_type=conflict_type,
            description=description,
            parties=parties,
            related_items=related_items or [],
            status="identified"
        )
        
        self.conflicts[conflict_id] = conflict
        
        return conflict
    
    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution_strategy: str = "negotiation"
    ) -> Dict[str, Any]:
        """
        解决冲突
        """
        if conflict_id not in self.conflicts:
            return {"error": "Conflict not found"}
        
        conflict = self.conflicts[conflict_id]
        conflict.status = "mediating"
        
        resolution_result = {
            "conflict_id": conflict_id,
            "strategy": resolution_strategy,
            "success": False
        }
        
        if resolution_strategy == "negotiation":
            # 协商解决
            # 简化：找到各方都能接受的妥协方案
            resolution = await self._negotiate_resolution(conflict)
            
            if resolution:
                conflict.resolution = resolution
                conflict.status = "resolved"
                conflict.resolved_at = datetime.now()
                conflict.resolved_by = self.agent_id
                
                resolution_result["success"] = True
                resolution_result["resolution"] = resolution
        
        elif resolution_strategy == "arbitration":
            # 仲裁
            # 由协调者决定
            resolution = await self._arbitrate_resolution(conflict)
            
            conflict.resolution = resolution
            conflict.status = "resolved"
            conflict.resolved_at = datetime.now()
            conflict.resolved_by = self.agent_id
            
            resolution_result["success"] = True
            resolution_result["resolution"] = resolution
        
        elif resolution_strategy == "voting":
            # 投票
            resolution = await self._vote_resolution(conflict)
            
            if resolution:
                conflict.resolution = resolution
                conflict.status = "resolved"
                conflict.resolved_at = datetime.now()
                
                resolution_result["success"] = True
                resolution_result["resolution"] = resolution
        
        return resolution_result
    
    async def _negotiate_resolution(self, conflict: Conflict) -> Optional[str]:
        """协商解决"""
        # 简化实现
        if conflict.conflict_type == ConflictType.RESOURCE:
            return "Resource sharing arrangement agreed"
        elif conflict.conflict_type == ConflictType.PRIORITY:
            return "Priorities reordered by consensus"
        elif conflict.conflict_type == ConflictType.METHODOLOGY:
            return "Hybrid approach adopted"
        return "Agreement reached through discussion"
    
    async def _arbitrate_resolution(self, conflict: Conflict) -> str:
        """仲裁解决"""
        # 协调者直接决定
        if conflict.conflict_type == ConflictType.RESOURCE:
            return "Resource allocated based on project priority"
        elif conflict.conflict_type == ConflictType.DEADLINE:
            return "Deadline extended with clear milestones"
        return "Coordinator decision: proceed with proposal from " + conflict.parties[0]
    
    async def _vote_resolution(self, conflict: Conflict) -> Optional[str]:
        """投票解决"""
        # 简化：随机选择
        import random
        winner = random.choice(conflict.parties)
        return f"Majority vote favors {winner}'s proposal"
    
    async def update_task_status(
        self,
        task_id: str,
        new_status: str,
        output: Optional[Any] = None
    ) -> bool:
        """
        更新任务状态
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        old_status = task.status
        task.status = new_status
        
        if new_status == "in_progress" and old_status != "in_progress":
            task.started_at = datetime.now()
        
        if new_status == "completed":
            task.completed_at = datetime.now()
            task.output = output
            
            # 减少Agent负载
            if task.assigned_to:
                self.agent_loads[task.assigned_to] = max(
                    0, self.agent_loads.get(task.assigned_to, 0) - 1
                )
            
            # 通知依赖任务
            await self._notify_dependent_tasks(task_id)
        
        return True
    
    async def _notify_dependent_tasks(self, completed_task_id: str) -> None:
        """通知依赖任务"""
        for task in self.tasks.values():
            if completed_task_id in task.dependencies and task.status == "blocked":
                # 检查所有依赖是否都完成
                if all(
                    self.tasks[dep_id].status == "completed"
                    for dep_id in task.dependencies
                ):
                    task.status = "pending"
                    
                    # 通知任务可以开始
                    if task.assigned_to:
                        await self.send_message(
                            {
                                "event": "dependencies_met",
                                "task_id": task.id
                            },
                            receiver_id=task.assigned_to,
                            message_type=MessageType.COMMUNICATION
                        )
    
    def _calculate_project_progress(self, project: Project) -> float:
        """计算项目进度"""
        if not project.tasks:
            return 0.0
        
        completed = sum(1 for t in project.tasks.values() if t.status == "completed")
        total = len(project.tasks)
        
        return (completed / total) * 100
    
    def get_project_dashboard(self, project_id: str) -> Optional[Dict[str, Any]]:
        """获取项目仪表板"""
        if project_id not in self.projects:
            return None
        
        project = self.projects[project_id]
        
        # 任务状态统计
        status_counts = {}
        for task in project.tasks.values():
            status = task.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # 即将到期的任务
        now = datetime.now()
        upcoming_deadlines = [
            task.to_dict() for task in project.tasks.values()
            if task.deadline and task.deadline > now and
            (task.deadline - now).days <= 3 and
            task.status != "completed"
        ]
        
        # 按Agent统计
        agent_tasks = {}
        for task in project.tasks.values():
            agent = task.assigned_to or "unassigned"
            if agent not in agent_tasks:
                agent_tasks[agent] = {"total": 0, "completed": 0}
            agent_tasks[agent]["total"] += 1
            if task.status == "completed":
                agent_tasks[agent]["completed"] += 1
        
        return {
            "project": project.to_dict(),
            "task_status_distribution": status_counts,
            "upcoming_deadlines": upcoming_deadlines,
            "agent_workload": agent_tasks,
            "active_conflicts": [
                c.to_dict() for c in self.conflicts.values()
                if c.status == "identified" or c.status == "mediating"
            ]
        }
    
    # ===== 消息处理器 =====
    
    async def _handle_coordination_message(self, message: Message) -> None:
        """处理协调消息"""
        content = message.content
        event = content.get("event")
        
        if event == "task_status_update":
            task_id = content.get("task_id")
            new_status = content.get("status")
            output = content.get("output")
            
            await self.update_task_status(task_id, new_status, output)
        
        elif event == "conflict_report":
            await self.identify_conflict(
                ConflictType(content.get("conflict_type", "resource")),
                content.get("description", ""),
                content.get("parties", []),
                content.get("related_items")
            )
        
        elif event == "request_task":
            agent_id = message.sender_id
            # Agent请求任务
            # 查找可分配的任务
            for task in self.tasks.values():
                if task.status == "pending" and not task.assigned_to:
                    await self.assign_task(task.id, agent_id)
                    break
    
    async def _handle_system_message(self, message: Message) -> None:
        """处理系统消息"""
        pass
    
    # ===== 能力处理器 =====
    
    async def _create_project_handler(self, **kwargs) -> Dict[str, Any]:
        project = await self.create_project(
            kwargs.get("name", ""),
            kwargs.get("description", ""),
            kwargs.get("objectives", []),
            kwargs.get("participants", []),
            kwargs.get("lead_agent")
        )
        return {"project": project.to_dict()}
    
    async def _assign_task_handler(self, **kwargs) -> Dict[str, Any]:
        success = await self.assign_task(
            kwargs.get("task_id"),
            kwargs.get("agent_id")
        )
        return {"success": success}
    
    async def _schedule_tasks_handler(self, **kwargs) -> Dict[str, Any]:
        schedule = await self.create_schedule(kwargs.get("project_id"))
        return {"schedule": schedule.__dict__ if schedule else None}
    
    async def _monitor_progress_handler(self, **kwargs) -> Dict[str, Any]:
        dashboard = self.get_project_dashboard(kwargs.get("project_id"))
        return {"dashboard": dashboard}
    
    async def _resolve_conflict_handler(self, **kwargs) -> Dict[str, Any]:
        result = await self.resolve_conflict(
            kwargs.get("conflict_id"),
            kwargs.get("strategy", "negotiation")
        )
        return result
    
    async def _reallocate_resources_handler(self, **kwargs) -> Dict[str, Any]:
        # 重新分配资源
        project_id = kwargs.get("project_id")
        # 简化实现
        return {"status": "reallocation_completed"}
    
    # ===== 公共API =====
    
    def get_projects(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取项目列表"""
        projects = self.projects.values()
        if status:
            projects = [p for p in projects if p.status == status]
        return [p.to_dict() for p in projects]
    
    def get_tasks(
        self,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取任务列表"""
        tasks = self.tasks.values()
        
        if project_id:
            tasks = [t for t in tasks if t.id in self.projects.get(project_id, Project("", "", "")).tasks]
        
        if agent_id:
            tasks = [t for t in tasks if t.assigned_to == agent_id]
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return [t.to_dict() for t in tasks]
    
    def register_agent_capabilities(self, agent_id: str, capabilities: List[str]) -> None:
        """注册Agent能力"""
        self.agent_capabilities[agent_id] = set(capabilities)
        if agent_id not in self.agent_loads:
            self.agent_loads[agent_id] = 0
