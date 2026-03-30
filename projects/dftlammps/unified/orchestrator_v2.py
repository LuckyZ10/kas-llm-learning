"""
DFT+LAMMPS Orchestrator V2 - Super Orchestrator
===============================================
跨模块工作流编排器 - 智能任务调度与全局优化

功能：
1. 跨模块工作流定义与执行
2. 智能任务调度算法
3. 依赖管理与并行执行
4. 全局优化策略
5. 动态资源分配
"""

import asyncio
import heapq
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import time
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .common import (
    DFTLAMMPSError, WorkflowError, ParallelError, TimeoutError,
    get_logger, generate_id, log_execution, retry, log_context
)
from .config_system import GlobalConfig
from .unified_api import UnifiedAPIRouter, APIRequest, HTTPMethod

logger = get_logger("orchestrator_v2")


# =============================================================================
# 工作流类型定义
# =============================================================================

class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRYING = auto()


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = auto()      # 顺序执行
    PARALLEL = auto()        # 并行执行
    DISTRIBUTED = auto()     # 分布式执行
    HYBRID = auto()          # 混合模式


@dataclass
class ResourceRequirements:
    """资源需求"""
    cpu_cores: int = 1
    memory_gb: float = 4.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    disk_gb: float = 10.0
    time_limit_seconds: float = 3600.0
    
    def fits_in(self, available: 'ResourceRequirements') -> bool:
        """检查资源是否满足"""
        return (
            self.cpu_cores <= available.cpu_cores and
            self.memory_gb <= available.memory_gb and
            self.gpu_count <= available.gpu_count and
            self.gpu_memory_gb <= available.gpu_memory_gb and
            self.disk_gb <= available.disk_gb
        )


@dataclass
class TaskResult:
    """任务结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: Any = None, **kwargs) -> 'TaskResult':
        """创建成功结果"""
        return cls(success=True, data=data, **kwargs)
    
    @classmethod
    def fail(cls, error: str, **kwargs) -> 'TaskResult':
        """创建失败结果"""
        return cls(success=False, error=error, **kwargs)


@dataclass
class Task:
    """任务定义"""
    id: str = field(default_factory=lambda: generate_id("task"))
    name: str = ""
    description: str = ""
    module: str = ""           # 所属模块
    operation: str = ""        # 操作类型
    params: Dict[str, Any] = field(default_factory=dict)
    
    # 执行控制
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    
    # 资源需求
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    
    # 依赖关系
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID
    dependents: List[str] = field(default_factory=list)    # 依赖此任务的任务ID
    
    # 执行控制
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: float = 3600.0
    
    # 结果
    result: Optional[TaskResult] = None
    
    # 时间戳
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.name,
            "priority": self.priority.name,
            "module": self.module,
            "operation": self.operation,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": self.get_execution_time(),
            "result": asdict(self.result) if self.result else None
        }
    
    def get_execution_time(self) -> float:
        """获取执行时间"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return 0.0
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """检查是否可以执行（依赖是否满足）"""
        return all(dep in completed_tasks for dep in self.dependencies)


@dataclass
class Workflow:
    """工作流定义"""
    id: str = field(default_factory=lambda: generate_id("wf"))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    tasks: Dict[str, Task] = field(default_factory=dict)
    
    # 全局设置
    max_parallel_tasks: int = 4
    global_timeout: float = 86400.0  # 24小时
    checkpoint_interval: int = 100
    
    # 状态
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # 结果聚合
    aggregate_results: bool = True
    result_aggregator: Optional[Callable[[List[TaskResult]], Any]] = None
    
    def add_task(self, task: Task) -> Task:
        """添加任务"""
        self.tasks[task.id] = task
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def get_ready_tasks(self) -> List[Task]:
        """获取准备就绪的任务"""
        completed = {
            t.id for t in self.tasks.values() 
            if t.status == TaskStatus.COMPLETED
        }
        
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and task.can_execute(completed):
                ready.append(task)
        
        # 按优先级排序
        ready.sort(key=lambda t: (t.priority.value, t.created_at))
        return ready
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """获取依赖图"""
        graph = defaultdict(list)
        for task in self.tasks.values():
            for dep in task.dependencies:
                graph[dep].append(task.id)
        return dict(graph)
    
    def validate(self) -> List[str]:
        """验证工作流"""
        errors = []
        
        # 检查循环依赖
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = self.tasks.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        if has_cycle(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    errors.append(f"Circular dependency detected involving task {task_id}")
        
        # 检查不存在的依赖
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    errors.append(f"Task {task.id} depends on non-existent task {dep}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.name,
            "task_count": len(self.tasks),
            "created_at": self.created_at,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()}
        }


# =============================================================================
# 任务执行器
# =============================================================================

class TaskExecutor(ABC):
    """任务执行器基类"""
    
    @abstractmethod
    async def execute(self, task: Task, context: 'ExecutionContext') -> TaskResult:
        """执行任务"""
        pass
    
    @abstractmethod
    def can_execute(self, task: Task) -> bool:
        """检查是否可以执行此任务"""
        pass


class ModuleTaskExecutor(TaskExecutor):
    """模块任务执行器 - 调用统一API"""
    
    def __init__(self, router: UnifiedAPIRouter):
        self.router = router
        self._handlers: Dict[str, Callable] = {}
    
    def register_handler(self, module: str, operation: str, handler: Callable) -> None:
        """注册处理函数"""
        key = f"{module}:{operation}"
        self._handlers[key] = handler
    
    def can_execute(self, task: Task) -> bool:
        """检查是否可以执行"""
        key = f"{task.module}:{task.operation}"
        return key in self._handlers or self._get_api_route(task) is not None
    
    def _get_api_route(self, task: Task) -> Optional[str]:
        """获取API路由"""
        # 尝试标准路径格式
        path = f"/modules/{task.module}/{task.operation}"
        route = self.router.registry.get_route("POST", path)
        if route:
            return path
        return None
    
    async def execute(self, task: Task, context: 'ExecutionContext') -> TaskResult:
        """执行任务"""
        start_time = time.time()
        
        try:
            # 优先使用注册的处理器
            key = f"{task.module}:{task.operation}"
            handler = self._handlers.get(key)
            
            if handler:
                result = await handler(task.params, context)
            else:
                # 通过API调用
                api_path = self._get_api_route(task)
                if not api_path:
                    return TaskResult.fail(f"No handler for {key}")
                
                response = await self.router.route(APIRequest(
                    path=api_path,
                    method=HTTPMethod.POST,
                    body={
                        "params": task.params,
                        "task_id": task.id,
                        "workflow_id": context.workflow_id
                    }
                ))
                
                if response.status.value == "error":
                    return TaskResult.fail(
                        response.error.get("message", "Unknown error")
                    )
                result = response.data
            
            execution_time = time.time() - start_time
            return TaskResult.ok(
                data=result,
                execution_time=execution_time,
                resource_usage={"cpu_time": execution_time}
            )
            
        except Exception as e:
            logger.exception(f"Task execution failed: {task.id}")
            return TaskResult.fail(str(e))


class PythonFunctionExecutor(TaskExecutor):
    """Python函数执行器"""
    
    def __init__(self):
        self._functions: Dict[str, Callable] = {}
    
    def register(self, name: str, func: Callable) -> None:
        """注册函数"""
        self._functions[name] = func
    
    def can_execute(self, task: Task) -> bool:
        """检查是否可以执行"""
        return task.operation in self._functions
    
    async def execute(self, task: Task, context: 'ExecutionContext') -> TaskResult:
        """执行任务"""
        start_time = time.time()
        func = self._functions.get(task.operation)
        
        if not func:
            return TaskResult.fail(f"Function not found: {task.operation}")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**task.params)
            else:
                # 在线程池中运行同步函数
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(**task.params))
            
            execution_time = time.time() - start_time
            return TaskResult.ok(
                data=result,
                execution_time=execution_time
            )
        except Exception as e:
            return TaskResult.fail(str(e))


class ShellCommandExecutor(TaskExecutor):
    """Shell命令执行器"""
    
    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = work_dir
    
    def can_execute(self, task: Task) -> bool:
        """检查是否可以执行"""
        return task.module == "shell" or task.operation == "exec"
    
    async def execute(self, task: Task, context: 'ExecutionContext') -> TaskResult:
        """执行shell命令"""
        import subprocess
        
        command = task.params.get("command", "")
        if not command:
            return TaskResult.fail("No command specified")
        
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=task.params.get("cwd") or self.work_dir
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=task.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            
            if process.returncode == 0:
                return TaskResult.ok(
                    data={
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode(),
                        "returncode": process.returncode
                    },
                    execution_time=execution_time
                )
            else:
                return TaskResult.fail(
                    error=f"Command failed with code {process.returncode}: {stderr.decode()}",
                    execution_time=execution_time
                )
                
        except asyncio.TimeoutError:
            return TaskResult.fail(
                error=f"Command timed out after {task.timeout_seconds}s",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return TaskResult.fail(str(e))


# =============================================================================
# 执行上下文
# =============================================================================

@dataclass
class ExecutionContext:
    """执行上下文"""
    workflow_id: str
    task_id: Optional[str] = None
    parent_context: Optional['ExecutionContext'] = None
    
    # 共享数据存储
    shared_data: Dict[str, Any] = field(default_factory=dict)
    
    # 结果缓存
    result_cache: Dict[str, TaskResult] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取共享数据"""
        return self.shared_data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置共享数据"""
        self.shared_data[key] = value
    
    def cache_result(self, task_id: str, result: TaskResult) -> None:
        """缓存结果"""
        self.result_cache[task_id] = result
    
    def get_cached_result(self, task_id: str) -> Optional[TaskResult]:
        """获取缓存的结果"""
        return self.result_cache.get(task_id)


# =============================================================================
# 智能调度器
# =============================================================================

class SmartScheduler:
    """
    智能任务调度器
    
    功能：
    - 优先级调度
    - 资源感知调度
    - 负载均衡
    - 动态调整
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._task_queue: List[Tuple[int, float, Task]] = []  # (priority, timestamp, task)
        self._running_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        
        # 资源跟踪
        self._available_resources = ResourceRequirements(
            cpu_cores=mp.cpu_count(),
            memory_gb=32.0,  # 假设
            gpu_count=0
        )
        self._used_resources = ResourceRequirements()
        
        # 统计
        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_wait_time": 0.0,
            "total_exec_time": 0.0
        }
    
    async def submit(self, task: Task) -> None:
        """提交任务"""
        async with self._lock:
            task.status = TaskStatus.QUEUED
            heapq.heappush(
                self._task_queue,
                (task.priority.value, time.time(), task)
            )
            self._stats["tasks_submitted"] += 1
            logger.debug(f"Task {task.id} submitted with priority {task.priority.name}")
    
    async def get_next_task(self) -> Optional[Task]:
        """获取下一个可执行的任务"""
        async with self._lock:
            # 检查资源限制
            if len(self._running_tasks) >= self.max_workers:
                return None
            
            # 查找资源匹配的任务
            temp_queue = []
            selected_task = None
            
            while self._task_queue and not selected_task:
                priority, timestamp, task = heapq.heappop(self._task_queue)
                
                # 检查资源需求
                if self._has_sufficient_resources(task.resources):
                    selected_task = task
                else:
                    temp_queue.append((priority, timestamp, task))
            
            # 放回未选择的任务
            for item in temp_queue:
                heapq.heappush(self._task_queue, item)
            
            if selected_task:
                self._running_tasks[selected_task.id] = selected_task
                self._allocate_resources(selected_task.resources)
                selected_task.status = TaskStatus.RUNNING
                selected_task.started_at = time.time()
            
            return selected_task
    
    async def complete_task(self, task: Task, result: TaskResult) -> None:
        """完成任务"""
        async with self._lock:
            task.result = result
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            task.completed_at = time.time()
            
            if task.id in self._running_tasks:
                del self._running_tasks[task.id]
            
            self._completed_tasks[task.id] = task
            self._release_resources(task.resources)
            
            # 更新统计
            if result.success:
                self._stats["tasks_completed"] += 1
            else:
                self._stats["tasks_failed"] += 1
            
            wait_time = task.started_at - task.created_at if task.started_at else 0
            self._stats["total_wait_time"] += wait_time
            self._stats["total_exec_time"] += result.execution_time
    
    def _has_sufficient_resources(self, required: ResourceRequirements) -> bool:
        """检查是否有足够资源"""
        available = ResourceRequirements(
            cpu_cores=self._available_resources.cpu_cores - self._used_resources.cpu_cores,
            memory_gb=self._available_resources.memory_gb - self._used_resources.memory_gb,
            gpu_count=self._available_resources.gpu_count - self._used_resources.gpu_count,
            gpu_memory_gb=self._available_resources.gpu_memory_gb - self._used_resources.gpu_memory_gb,
        )
        return required.fits_in(available)
    
    def _allocate_resources(self, resources: ResourceRequirements) -> None:
        """分配资源"""
        self._used_resources.cpu_cores += resources.cpu_cores
        self._used_resources.memory_gb += resources.memory_gb
        self._used_resources.gpu_count += resources.gpu_count
        self._used_resources.gpu_memory_gb += resources.gpu_memory_gb
    
    def _release_resources(self, resources: ResourceRequirements) -> None:
        """释放资源"""
        self._used_resources.cpu_cores -= resources.cpu_cores
        self._used_resources.memory_gb -= resources.memory_gb
        self._used_resources.gpu_count -= resources.gpu_count
        self._used_resources.gpu_memory_gb -= resources.gpu_memory_gb
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._stats["tasks_completed"] + self._stats["tasks_failed"]
        avg_wait = self._stats["total_wait_time"] / total if total > 0 else 0
        avg_exec = self._stats["total_exec_time"] / total if total > 0 else 0
        
        return {
            **self._stats,
            "avg_wait_time": avg_wait,
            "avg_exec_time": avg_exec,
            "queue_length": len(self._task_queue),
            "running_count": len(self._running_tasks),
            "completed_count": len(self._completed_tasks)
        }


# =============================================================================
# 全局优化器
# =============================================================================

class GlobalOptimizer:
    """
    全局优化器
    
    功能：
    - 任务重排序优化
    - 批处理优化
    - 缓存策略
    - 资源预分配
    """
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._task_patterns: Dict[str, List[Dict]] = defaultdict(list)
    
    def optimize_workflow(self, workflow: Workflow) -> Workflow:
        """优化工作流"""
        # 任务合并优化
        workflow = self._merge_similar_tasks(workflow)
        
        # 重排序优化
        workflow = self._reorder_for_parallelism(workflow)
        
        # 资源预分配优化
        workflow = self._optimize_resources(workflow)
        
        return workflow
    
    def _merge_similar_tasks(self, workflow: Workflow) -> Workflow:
        """合并相似任务"""
        # 找出可以批处理的相似任务
        task_groups = defaultdict(list)
        
        for task in workflow.tasks.values():
            if not task.dependencies:  # 只合并无依赖的任务
                key = f"{task.module}:{task.operation}"
                task_groups[key].append(task)
        
        # 合并逻辑（简化版）
        for key, tasks in task_groups.items():
            if len(tasks) > 1:
                logger.debug(f"Found {len(tasks)} similar tasks for {key}")
        
        return workflow
    
    def _reorder_for_parallelism(self, workflow: Workflow) -> Workflow:
        """重排序以提高并行度"""
        # 计算每个任务的关键路径长度
        critical_path = {}
        
        def get_path_length(task_id: str, memo: Dict = {}) -> float:
            if task_id in memo:
                return memo[task_id]
            
            task = workflow.get_task(task_id)
            if not task or not task.dependencies:
                memo[task_id] = 0
                return 0
            
            max_dep_length = max(
                get_path_length(dep, memo) for dep in task.dependencies
            )
            memo[task_id] = max_dep_length + 1
            return memo[task_id]
        
        for task_id in workflow.tasks:
            critical_path[task_id] = get_path_length(task_id)
        
        # 根据关键路径长度调整优先级
        for task_id, path_length in critical_path.items():
            task = workflow.get_task(task_id)
            if task and path_length > 2:
                # 关键路径上的任务提高优先级
                if task.priority.value < TaskPriority.HIGH.value:
                    task.priority = TaskPriority.HIGH
        
        return workflow
    
    def _optimize_resources(self, workflow: Workflow) -> Workflow:
        """优化资源分配"""
        # 分析任务历史执行数据，调整资源分配
        for task in workflow.tasks.values():
            pattern_key = f"{task.module}:{task.operation}"
            history = self._task_patterns.get(pattern_key, [])
            
            if history:
                # 根据历史平均执行时间调整超时
                avg_time = sum(h.get("exec_time", 0) for h in history) / len(history)
                task.timeout_seconds = max(task.timeout_seconds, avg_time * 2)
        
        return workflow
    
    def record_task_execution(self, task: Task, result: TaskResult) -> None:
        """记录任务执行数据"""
        pattern_key = f"{task.module}:{task.operation}"
        self._task_patterns[pattern_key].append({
            "exec_time": result.execution_time,
            "success": result.success,
            "timestamp": time.time()
        })
        
        # 限制历史记录大小
        if len(self._task_patterns[pattern_key]) > 100:
            self._task_patterns[pattern_key] = self._task_patterns[pattern_key][-100:]


# =============================================================================
# 超级编排器 V2
# =============================================================================

class OrchestratorV2:
    """
    超级编排器 V2
    
    核心功能：
    - 跨模块工作流编排
    - 智能任务调度
    - 全局优化
    - 动态资源管理
    """
    
    def __init__(self, config: Optional[GlobalConfig] = None):
        self.config = config
        self.scheduler = SmartScheduler()
        self.optimizer = GlobalOptimizer()
        self.executors: List[TaskExecutor] = []
        
        # 工作流管理
        self._workflows: Dict[str, Workflow] = {}
        self._contexts: Dict[str, ExecutionContext] = {}
        
        # 事件回调
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 运行状态
        self._running = False
        self._main_loop_task: Optional[asyncio.Task] = None
    
    def add_executor(self, executor: TaskExecutor) -> None:
        """添加执行器"""
        self.executors.append(executor)
    
    def on(self, event: str, handler: Callable) -> None:
        """注册事件处理器"""
        self._event_handlers[event].append(handler)
    
    async def _emit(self, event: str, data: Any) -> None:
        """触发事件"""
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    async def start(self) -> None:
        """启动编排器"""
        if self._running:
            return
        
        self._running = True
        self._main_loop_task = asyncio.create_task(self._main_loop())
        logger.info("Orchestrator V2 started")
    
    async def stop(self) -> None:
        """停止编排器"""
        self._running = False
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Orchestrator V2 stopped")
    
    async def _main_loop(self) -> None:
        """主调度循环"""
        while self._running:
            try:
                # 获取下一个任务
                task = await self.scheduler.get_next_task()
                
                if task:
                    # 异步执行任务
                    asyncio.create_task(self._execute_task(task))
                else:
                    # 没有可执行的任务，等待
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.exception("Error in main loop")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Task) -> None:
        """执行任务"""
        # 找到合适的执行器
        executor = None
        for exec in self.executors:
            if exec.can_execute(task):
                executor = exec
                break
        
        if not executor:
            result = TaskResult.fail(f"No executor available for task {task.id}")
        else:
            # 获取执行上下文
            context = self._contexts.get(task.metadata.get("workflow_id", ""))
            
            # 执行任务
            try:
                result = await asyncio.wait_for(
                    executor.execute(task, context),
                    timeout=task.timeout_seconds
                )
            except asyncio.TimeoutError:
                result = TaskResult.fail(f"Task timeout after {task.timeout_seconds}s")
            except Exception as e:
                result = TaskResult.fail(str(e))
        
        # 处理重试
        if not result.success and task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            logger.warning(f"Retrying task {task.id} (attempt {task.retry_count})")
            await self.scheduler.submit(task)
            return
        
        # 完成任务
        await self.scheduler.complete_task(task, result)
        
        # 优化器记录
        self.optimizer.record_task_execution(task, result)
        
        # 触发事件
        await self._emit("task_completed", {
            "task_id": task.id,
            "success": result.success,
            "workflow_id": task.metadata.get("workflow_id")
        })
        
        # 检查工作流状态
        workflow_id = task.metadata.get("workflow_id")
        if workflow_id:
            await self._check_workflow_completion(workflow_id)
    
    def create_workflow(self, name: str, description: str = "") -> Workflow:
        """创建工作流"""
        workflow = Workflow(name=name, description=description)
        self._workflows[workflow.id] = workflow
        self._contexts[workflow.id] = ExecutionContext(workflow_id=workflow.id)
        return workflow
    
    def add_task_to_workflow(self, workflow_id: str, task: Task) -> Task:
        """添加任务到工作流"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise WorkflowError(f"Workflow not found: {workflow_id}")
        
        task.metadata["workflow_id"] = workflow_id
        workflow.add_task(task)
        return task
    
    async def execute_workflow(self, workflow_id: str, 
                              wait: bool = True) -> Dict[str, Any]:
        """执行工作流"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise WorkflowError(f"Workflow not found: {workflow_id}")
        
        # 验证工作流
        errors = workflow.validate()
        if errors:
            raise WorkflowError(f"Workflow validation failed: {errors}")
        
        # 优化工作流
        workflow = self.optimizer.optimize_workflow(workflow)
        
        # 提交就绪的任务
        workflow.status = TaskStatus.RUNNING
        workflow.started_at = time.time()
        
        ready_tasks = workflow.get_ready_tasks()
        for task in ready_tasks:
            await self.scheduler.submit(task)
        
        await self._emit("workflow_started", {"workflow_id": workflow_id})
        
        if wait:
            # 等待完成
            while workflow.status == TaskStatus.RUNNING:
                await asyncio.sleep(0.5)
                
                # 检查是否完成
                all_done = all(
                    t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                    for t in workflow.tasks.values()
                )
                
                if all_done:
                    workflow.status = TaskStatus.COMPLETED
                    workflow.completed_at = time.time()
                    break
            
            return self._aggregate_workflow_results(workflow)
        
        return {"workflow_id": workflow_id, "status": "started"}
    
    async def _check_workflow_completion(self, workflow_id: str) -> None:
        """检查工作流完成状态"""
        workflow = self._workflows.get(workflow_id)
        if not workflow or workflow.status != TaskStatus.RUNNING:
            return
        
        # 提交新就绪的任务
        completed = {
            t.id for t in workflow.tasks.values()
            if t.status == TaskStatus.COMPLETED
        }
        
        for task in workflow.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                task.can_execute(completed)):
                await self.scheduler.submit(task)
        
        # 检查是否全部完成
        all_done = all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            for t in workflow.tasks.values()
        )
        
        if all_done:
            workflow.status = TaskStatus.COMPLETED
            workflow.completed_at = time.time()
            await self._emit("workflow_completed", {"workflow_id": workflow_id})
    
    def _aggregate_workflow_results(self, workflow: Workflow) -> Dict[str, Any]:
        """聚合工作流结果"""
        results = {}
        failed_tasks = []
        
        for task in workflow.tasks.values():
            if task.result:
                results[task.name or task.id] = {
                    "success": task.result.success,
                    "data": task.result.data,
                    "error": task.result.error,
                    "execution_time": task.result.execution_time
                }
                if not task.result.success:
                    failed_tasks.append(task.name or task.id)
        
        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.name,
            "execution_time": workflow.completed_at - workflow.started_at if workflow.completed_at else 0,
            "results": results,
            "failed_tasks": failed_tasks,
            "all_success": len(failed_tasks) == 0
        }
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.name,
            "task_count": len(workflow.tasks),
            "tasks": {t.id: t.to_dict() for t in workflow.tasks.values()}
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "scheduler": self.scheduler.get_stats(),
            "workflows": {
                "total": len(self._workflows),
                "running": sum(1 for w in self._workflows.values() if w.status == TaskStatus.RUNNING)
            }
        }


# =============================================================================
# 便捷函数
# =============================================================================

# 全局编排器实例
_global_orchestrator: Optional[OrchestratorV2] = None


def get_orchestrator(config: Optional[GlobalConfig] = None) -> OrchestratorV2:
    """获取全局编排器"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = OrchestratorV2(config)
    return _global_orchestrator


async def run_workflow(tasks: List[Task], 
                      name: str = "workflow",
                      config: Optional[GlobalConfig] = None) -> Dict[str, Any]:
    """
    便捷函数：运行工作流
    
    Args:
        tasks: 任务列表
        name: 工作流名称
        config: 配置
        
    Returns:
        工作流执行结果
    """
    orch = get_orchestrator(config)
    
    # 启动编排器
    await orch.start()
    
    try:
        # 创建工作流
        workflow = orch.create_workflow(name)
        
        # 添加任务
        for task in tasks:
            orch.add_task_to_workflow(workflow.id, task)
        
        # 执行并等待
        result = await orch.execute_workflow(workflow.id, wait=True)
        return result
        
    finally:
        await orch.stop()


# =============================================================================
# 工作流构建器
# =============================================================================

class WorkflowBuilder:
    """工作流构建器 - 链式API"""
    
    def __init__(self, name: str = "", orchestrator: Optional[OrchestratorV2] = None):
        self.orchestrator = orchestrator or get_orchestrator()
        self.workflow = self.orchestrator.create_workflow(name)
        self._last_task: Optional[Task] = None
    
    def add_task(self, 
                name: str,
                module: str,
                operation: str,
                params: Dict[str, Any] = None,
                depends_on: Optional[Union[str, List[str]]] = None,
                priority: TaskPriority = TaskPriority.NORMAL,
                resources: Optional[ResourceRequirements] = None) -> 'WorkflowBuilder':
        """添加任务"""
        task = Task(
            name=name,
            module=module,
            operation=operation,
            params=params or {},
            priority=priority,
            resources=resources or ResourceRequirements()
        )
        
        # 设置依赖
        if depends_on:
            if isinstance(depends_on, str):
                task.dependencies = [depends_on]
            else:
                task.dependencies = depends_on
        elif self._last_task:
            # 默认依赖上一个任务
            task.dependencies = [self._last_task.id]
        
        self.orchestrator.add_task_to_workflow(self.workflow.id, task)
        self._last_task = task
        
        return self
    
    def parallel(self, *tasks: Task) -> 'WorkflowBuilder':
        """添加并行任务组"""
        prev_task = self._last_task
        
        for task in tasks:
            if prev_task and not task.dependencies:
                task.dependencies = [prev_task.id]
            self.orchestrator.add_task_to_workflow(self.workflow.id, task)
        
        self._last_task = None  # 并行后无明确上一个任务
        return self
    
    async def execute(self, wait: bool = True) -> Dict[str, Any]:
        """执行工作流"""
        return await self.orchestrator.execute_workflow(self.workflow.id, wait)
    
    def get_workflow(self) -> Workflow:
        """获取工作流对象"""
        return self.workflow
