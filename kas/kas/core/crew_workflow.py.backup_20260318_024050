"""
KAS Crew 工作流引擎
实现基于 YAML 的工作流定义和执行
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from kas.core.sandbox.supervisor import SandboxSupervisor, CrewConfig
from kas.core.sandbox.message_bus import MessageBus, MessageType

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    """工作流任务"""
    id: str
    name: str
    agent: str
    task: str
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # 执行条件表达式
    timeout: int = 30  # 超时秒数
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "agent": self.agent,
            "task": self.task,
            "depends_on": self.depends_on,
            "condition": self.condition,
            "timeout": self.timeout,
            "status": self.status.value,
            "result": self.result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


@dataclass
class Workflow:
    """工作流定义"""
    name: str
    description: str
    tasks: List[WorkflowTask]
    variables: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Workflow':
        """从字典加载工作流"""
        tasks = []
        for task_data in data.get('tasks', []):
            tasks.append(WorkflowTask(
                id=task_data.get('id', f"task_{len(tasks)}"),
                name=task_data.get('name', ''),
                agent=task_data.get('agent', ''),
                task=task_data.get('task', ''),
                depends_on=task_data.get('depends_on', []),
                condition=task_data.get('condition'),
                timeout=task_data.get('timeout', 30)
            ))
        
        return cls(
            name=data.get('name', 'unnamed'),
            description=data.get('description', ''),
            tasks=tasks,
            variables=data.get('variables', {})
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Workflow':
        """从 YAML 文件加载工作流"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "tasks": [t.to_dict() for t in self.tasks],
            "variables": self.variables
        }


class WorkflowEngine:
    """
    工作流引擎
    
    执行基于依赖关系的任务调度:
    1. 解析任务依赖图
    2. 按拓扑排序执行
    3. 支持条件判断
    4. 超时和错误处理
    5. 结果传递
    """
    
    def __init__(self, supervisor: SandboxSupervisor):
        """
        Args:
            supervisor: 沙盒监督器
        """
        self.supervisor = supervisor
        self.workflows: Dict[str, Workflow] = {}
        self._executions: Dict[str, Dict] = {}  # 执行记录
    
    def load_workflow(self, yaml_path: Path) -> Workflow:
        """加载工作流"""
        workflow = Workflow.from_yaml(yaml_path)
        self.workflows[workflow.name] = workflow
        return workflow
    
    def save_workflow(self, workflow: Workflow, yaml_path: Path):
        """保存工作流到 YAML"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(workflow.to_dict(), f, allow_unicode=True, sort_keys=False)
    
    def execute(self, crew_name: str, workflow: Workflow, 
                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行工作流
        
        Args:
            crew_name: Crew 名称
            workflow: 工作流定义
            context: 执行上下文（变量、输入等）
        
        Returns:
            执行结果
        """
        execution_id = f"{workflow.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        context = context or {}
        
        logger.info(f"Starting workflow execution: {execution_id}")
        
        # 初始化执行记录
        self._executions[execution_id] = {
            "workflow_name": workflow.name,
            "crew_name": crew_name,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "tasks": {t.id: t for t in workflow.tasks},
            "context": context.copy()
        }
        
        try:
            # 拓扑排序获取执行顺序
            execution_order = self._topological_sort(workflow.tasks)
            
            for task in execution_order:
                # 检查依赖是否完成
                if not self._check_dependencies(task, execution_id):
                    logger.warning(f"Task {task.id} dependencies not met, skipping")
                    task.status = TaskStatus.SKIPPED
                    continue
                
                # 检查条件
                if task.condition and not self._evaluate_condition(task.condition, execution_id):
                    logger.info(f"Task {task.id} condition not met, skipping")
                    task.status = TaskStatus.SKIPPED
                    continue
                
                # 执行任务
                self._execute_task(crew_name, task, execution_id)
                
                # 检查是否失败
                if task.status == TaskStatus.FAILED:
                    logger.error(f"Task {task.id} failed, stopping workflow")
                    break
            
            # 更新执行状态
            self._executions[execution_id]["completed_at"] = datetime.now().isoformat()
            self._executions[execution_id]["status"] = "completed"
            
            # 收集结果
            results = {}
            for task in workflow.tasks:
                results[task.id] = {
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error
                }
            
            return {
                "execution_id": execution_id,
                "workflow_name": workflow.name,
                "status": "completed",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self._executions[execution_id]["status"] = "failed"
            self._executions[execution_id]["error"] = str(e)
            
            return {
                "execution_id": execution_id,
                "workflow_name": workflow.name,
                "status": "failed",
                "error": str(e)
            }
    
    def _topological_sort(self, tasks: List[WorkflowTask]) -> List[WorkflowTask]:
        """拓扑排序任务"""
        task_map = {t.id: t for t in tasks}
        visited = set()
        result = []
        
        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep_id in task.depends_on:
                    visit(dep_id)
                result.append(task)
        
        for task in tasks:
            visit(task.id)
        
        return result
    
    def _check_dependencies(self, task: WorkflowTask, execution_id: str) -> bool:
        """检查任务依赖是否完成"""
        execution = self._executions[execution_id]
        
        for dep_id in task.depends_on:
            dep_task = execution["tasks"].get(dep_id)
            if not dep_task:
                return False
            if dep_task.status not in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]:
                return False
        
        return True
    
    def _evaluate_condition(self, condition: str, execution_id: str) -> bool:
        """评估条件表达式"""
        execution = self._executions[execution_id]
        context = execution["context"]
        
        try:
            # 简单的条件求值
            # 支持: context.get('has_image'), results.get('task_1')
            namespace = {
                "context": context,
                "results": {t.id: t.result for t in execution["tasks"].values()}
            }
            return eval(condition, namespace)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False
    
    def _execute_task(self, crew_name: str, task: WorkflowTask, execution_id: str):
        """执行单个任务"""
        logger.info(f"Executing task: {task.id} on agent {task.agent}")
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        
        try:
            # 构建任务上下文
            execution = self._executions[execution_id]
            task_context = {
                "execution_id": execution_id,
                "workflow_name": execution["workflow_name"],
                "task_id": task.id,
                "variables": execution["context"]
            }
            
            # 添加依赖任务的结果
            for dep_id in task.depends_on:
                dep_task = execution["tasks"].get(dep_id)
                if dep_task and dep_task.result:
                    task_context[f"{dep_id}_result"] = dep_task.result
            
            # 分发任务并等待结果
            result = self.supervisor.dispatch_task(
                crew_name=crew_name,
                agent_name=task.agent,
                task=task.task,
                context=task_context,
                wait_result=True,
                timeout=task.timeout
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            
            logger.info(f"Task {task.id} completed")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()
            
            logger.error(f"Task {task.id} failed: {e}")
    
    def get_execution(self, execution_id: str) -> Optional[Dict]:
        """获取执行记录"""
        execution = self._executions.get(execution_id)
        if execution:
            return {
                "execution_id": execution_id,
                "workflow_name": execution["workflow_name"],
                "crew_name": execution["crew_name"],
                "started_at": execution["started_at"],
                "completed_at": execution.get("completed_at"),
                "status": execution["status"],
                "error": execution.get("error"),
                "tasks": {tid: t.to_dict() for tid, t in execution["tasks"].items()}
            }
        return None
    
    def list_executions(self, workflow_name: Optional[str] = None) -> List[Dict]:
        """列出执行记录"""
        results = []
        for exec_id, execution in self._executions.items():
            if workflow_name is None or execution["workflow_name"] == workflow_name:
                results.append({
                    "execution_id": exec_id,
                    "workflow_name": execution["workflow_name"],
                    "crew_name": execution["crew_name"],
                    "started_at": execution["started_at"],
                    "status": execution["status"]
                })
        return results
