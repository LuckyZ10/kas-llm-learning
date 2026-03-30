"""
DFT-LAMMPS 资源调度器
=====================
计算资源跨模块分配

智能分配和管理计算资源，优化模块执行效率。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import PriorityQueue, Queue, Empty
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union, Callable


logger = logging.getLogger("resource_scheduler")


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    LICENSE = "license"  # 软件许可证


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceRequirement:
    """资源需求"""
    resource_type: ResourceType
    amount: float                     # 数量
    unit: str                         # 单位
    exclusive: bool = False           # 是否独占
    duration_estimate: float = 3600.0  # 预计持续时间（秒）
    
    def __hash__(self) -> int:
        return hash((self.resource_type, self.amount, self.unit))


@dataclass
class ResourceAllocation:
    """资源分配"""
    resource_type: ResourceType
    amount: float
    unit: str
    allocated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class ComputeTask:
    """计算任务"""
    task_id: str
    task_type: str                    # 任务类型（如"vasp", "lammps"）
    module_name: str                  # 执行模块
    priority: TaskPriority = TaskPriority.NORMAL
    
    # 资源需求
    requirements: List[ResourceRequirement] = field(default_factory=list)
    
    # 执行信息
    command: Optional[str] = None
    working_dir: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    
    # 回调
    on_start: Optional[Callable[[str], None]] = None
    on_complete: Optional[Callable[[str, Any], None]] = None
    on_error: Optional[Callable[[str, Exception], None]] = None
    
    # 状态
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    
    def __lt__(self, other: ComputeTask) -> bool:
        """用于优先级队列比较"""
        return self.priority.value < other.priority.value


class Resource:
    """资源基类"""
    
    def __init__(
        self,
        resource_id: str,
        resource_type: ResourceType,
        total_capacity: float,
        unit: str
    ):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.total_capacity = total_capacity
        self.unit = unit
        
        self._allocated: Dict[str, ResourceAllocation] = {}
        self._lock = threading.Lock()
    
    @property
    def available(self) -> float:
        """可用资源"""
        with self._lock:
            self._cleanup_expired()
            return self.total_capacity - sum(
                alloc.amount for alloc in self._allocated.values()
            )
    
    @property
    def utilization(self) -> float:
        """资源利用率"""
        if self.total_capacity == 0:
            return 0.0
        return (self.total_capacity - self.available) / self.total_capacity
    
    def allocate(
        self,
        task_id: str,
        amount: float,
        duration: Optional[float] = None
    ) -> Optional[ResourceAllocation]:
        """分配资源"""
        with self._lock:
            self._cleanup_expired()
            
            if amount > self.available:
                return None
            
            allocation = ResourceAllocation(
                resource_type=self.resource_type,
                amount=amount,
                unit=self.unit,
                expires_at=time.time() + duration if duration else None
            )
            
            self._allocated[task_id] = allocation
            logger.debug(f"Allocated {amount} {self.unit} of {self.resource_type.value} to {task_id}")
            return allocation
    
    def release(self, task_id: str) -> bool:
        """释放资源"""
        with self._lock:
            if task_id in self._allocated:
                del self._allocated[task_id]
                logger.debug(f"Released {self.resource_type.value} from {task_id}")
                return True
            return False
    
    def _cleanup_expired(self) -> None:
        """清理过期分配"""
        expired = [
            task_id for task_id, alloc in self._allocated.items()
            if alloc.is_expired()
        ]
        for task_id in expired:
            del self._allocated[task_id]
            logger.debug(f"Cleaned up expired allocation for {task_id}")


class CPUResource(Resource):
    """CPU资源"""
    
    def __init__(self, resource_id: str, cores: int):
        super().__init__(resource_id, ResourceType.CPU, float(cores), "cores")


class GPUResource(Resource):
    """GPU资源"""
    
    def __init__(
        self,
        resource_id: str,
        device_id: int,
        memory_gb: float,
        compute_capability: str = ""
    ):
        super().__init__(resource_id, ResourceType.GPU, memory_gb, "GB")
        self.device_id = device_id
        self.compute_capability = compute_capability
        self._memory_per_task: Dict[str, float] = {}
    
    def allocate(
        self,
        task_id: str,
        amount: float,
        duration: Optional[float] = None
    ) -> Optional[ResourceAllocation]:
        """分配GPU显存"""
        allocation = super().allocate(task_id, amount, duration)
        if allocation:
            self._memory_per_task[task_id] = amount
        return allocation
    
    def release(self, task_id: str) -> bool:
        """释放GPU"""
        if task_id in self._memory_per_task:
            del self._memory_per_task[task_id]
        return super().release(task_id)


class MemoryResource(Resource):
    """内存资源"""
    
    def __init__(self, resource_id: str, total_gb: float):
        super().__init__(resource_id, ResourceType.MEMORY, total_gb, "GB")


class LicenseResource(Resource):
    """许可证资源"""
    
    def __init__(self, resource_id: str, software: str, total_licenses: int):
        super().__init__(resource_id, ResourceType.LICENSE, float(total_licenses), "licenses")
        self.software = software


class ResourcePool:
    """
    资源池
    
    管理所有可用资源
    """
    
    def __init__(self):
        self._resources: Dict[ResourceType, List[Resource]] = {
            rtype: [] for rtype in ResourceType
        }
        self._lock = threading.Lock()
    
    def add_resource(self, resource: Resource) -> None:
        """添加资源"""
        with self._lock:
            self._resources[resource.resource_type].append(resource)
        logger.info(f"Added resource: {resource.resource_id} ({resource.resource_type.value})")
    
    def remove_resource(self, resource_id: str) -> bool:
        """移除资源"""
        with self._lock:
            for rtype, resources in self._resources.items():
                for i, res in enumerate(resources):
                    if res.resource_id == resource_id:
                        resources.pop(i)
                        logger.info(f"Removed resource: {resource_id}")
                        return True
        return False
    
    def get_resources(
        self,
        resource_type: Optional[ResourceType] = None
    ) -> List[Resource]:
        """获取资源"""
        with self._lock:
            if resource_type:
                return self._resources[resource_type].copy()
            else:
                return [
                    res for resources in self._resources.values()
                    for res in resources
                ]
    
    def check_availability(
        self,
        requirement: ResourceRequirement
    ) -> bool:
        """检查资源可用性"""
        with self._lock:
            resources = self._resources[requirement.resource_type]
            
            if requirement.exclusive:
                # 独占资源：需要一个完全空闲的资源
                for res in resources:
                    if res.available >= res.total_capacity * 0.99:
                        return True
                return False
            else:
                # 共享资源：总可用量足够即可
                total_available = sum(res.available for res in resources)
                return total_available >= requirement.amount
    
    def allocate_resources(
        self,
        task_id: str,
        requirements: List[ResourceRequirement]
    ) -> Dict[ResourceType, ResourceAllocation]:
        """
        为任务分配资源
        
        使用最佳适应算法
        """
        allocations = {}
        
        with self._lock:
            for req in requirements:
                resources = self._resources[req.resource_type]
                
                # 按可用量排序（最佳适应）
                sorted_resources = sorted(
                    resources,
                    key=lambda r: r.available
                )
                
                allocated = False
                for res in sorted_resources:
                    allocation = res.allocate(
                        task_id,
                        req.amount,
                        req.duration_estimate
                    )
                    if allocation:
                        allocations[req.resource_type] = allocation
                        allocated = True
                        break
                
                if not allocated:
                    # 分配失败，回滚
                    self._rollback_allocations(task_id, allocations)
                    raise ResourceError(
                        f"Cannot allocate {req.amount} {req.unit} of {req.resource_type.value}"
                    )
        
        return allocations
    
    def release_resources(self, task_id: str) -> None:
        """释放任务占用的所有资源"""
        with self._lock:
            for resources in self._resources.values():
                for res in resources:
                    res.release(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取资源统计"""
        stats = {}
        
        for rtype, resources in self._resources.items():
            total = sum(res.total_capacity for res in resources)
            available = sum(res.available for res in resources)
            
            stats[rtype.value] = {
                "total": total,
                "available": available,
                "utilized": total - available,
                "utilization_rate": (total - available) / total if total > 0 else 0,
                "count": len(resources)
            }
        
        return stats
    
    def _rollback_allocations(
        self,
        task_id: str,
        allocations: Dict[ResourceType, ResourceAllocation]
    ) -> None:
        """回滚分配"""
        for rtype in allocations:
            for res in self._resources[rtype]:
                res.release(task_id)


class ResourceScheduler:
    """
    资源调度器
    
    智能调度计算任务，优化资源利用
    
    Example:
        scheduler = ResourceScheduler()
        
        # 添加资源
        scheduler.add_cpu("cpu_1", 16)
        scheduler.add_gpu("gpu_1", 0, 8.0)
        
        # 提交任务
        task = ComputeTask(
            task_id="task_1",
            task_type="vasp",
            module_name="vasp_module",
            requirements=[
                ResourceRequirement(ResourceType.CPU, 4, "cores"),
                ResourceRequirement(ResourceType.MEMORY, 32, "GB")
            ]
        )
        scheduler.submit_task(task)
        
        # 启动调度
        scheduler.start()
    """
    
    def __init__(self):
        self.resource_pool = ResourcePool()
        self._task_queue: PriorityQueue[Tuple[int, float, ComputeTask]] = PriorityQueue()
        self._running_tasks: Dict[str, ComputeTask] = {}
        self._completed_tasks: Dict[str, ComputeTask] = {}
        
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 策略配置
        self._max_concurrent_tasks = 10
        self._task_timeout = 3600 * 24  # 24小时
        self._preemption_enabled = False
    
    def add_cpu(self, resource_id: str, cores: int) -> None:
        """添加CPU资源"""
        self.resource_pool.add_resource(CPUResource(resource_id, cores))
    
    def add_gpu(
        self,
        resource_id: str,
        device_id: int,
        memory_gb: float,
        compute_capability: str = ""
    ) -> None:
        """添加GPU资源"""
        self.resource_pool.add_resource(
            GPUResource(resource_id, device_id, memory_gb, compute_capability)
        )
    
    def add_memory(self, resource_id: str, total_gb: float) -> None:
        """添加内存资源"""
        self.resource_pool.add_resource(MemoryResource(resource_id, total_gb))
    
    def add_license(
        self,
        resource_id: str,
        software: str,
        total_licenses: int
    ) -> None:
        """添加许可证资源"""
        self.resource_pool.add_resource(
            LicenseResource(resource_id, software, total_licenses)
        )
    
    def submit_task(self, task: ComputeTask) -> None:
        """提交任务"""
        with self._lock:
            # 检查任务ID是否重复
            if task.task_id in self._running_tasks or task.task_id in self._completed_tasks:
                raise ValueError(f"Task ID already exists: {task.task_id}")
        
        # 加入队列（使用优先级和时间戳作为排序键）
        entry = (task.priority.value, time.time(), task)
        self._task_queue.put(entry)
        task.status = TaskStatus.QUEUED
        
        logger.info(f"Submitted task: {task.task_id} (priority: {task.priority.name})")
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            # 检查运行中的任务
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                self.resource_pool.release_resources(task_id)
                del self._running_tasks[task_id]
                self._completed_tasks[task_id] = task
                logger.info(f"Cancelled running task: {task_id}")
                return True
            
            # 检查队列中的任务（需要重建队列）
            new_queue = PriorityQueue()
            found = False
            
            while not self._task_queue.empty():
                priority, timestamp, task = self._task_queue.get()
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    self._completed_tasks[task_id] = task
                    found = True
                    logger.info(f"Cancelled queued task: {task_id}")
                else:
                    new_queue.put((priority, timestamp, task))
            
            self._task_queue = new_queue
            return found
    
    def start(self) -> None:
        """启动调度器"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        logger.info("Resource scheduler started")
    
    def stop(self) -> None:
        """停止调度器"""
        self._running = False
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=30.0)
        
        # 释放所有资源
        with self._lock:
            for task in self._running_tasks.values():
                self.resource_pool.release_resources(task.task_id)
        
        logger.info("Resource scheduler stopped")
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self._lock:
            if task_id in self._running_tasks:
                return self._running_tasks[task_id].status
            if task_id in self._completed_tasks:
                return self._completed_tasks[task_id].status
        
        # 检查队列
        for _, _, task in list(self._task_queue.queue):
            if task.task_id == task_id:
                return task.status
        
        return None
    
    def get_queue_length(self) -> int:
        """获取队列长度"""
        return self._task_queue.qsize()
    
    def get_running_count(self) -> int:
        """获取运行中任务数"""
        with self._lock:
            return len(self._running_tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计"""
        return {
            "resource_stats": self.resource_pool.get_stats(),
            "queue_length": self.get_queue_length(),
            "running_tasks": self.get_running_count(),
            "completed_tasks": len(self._completed_tasks),
            "max_concurrent": self._max_concurrent_tasks
        }
    
    def _scheduler_loop(self) -> None:
        """调度循环"""
        while self._running:
            try:
                # 检查资源状态
                self._cleanup_finished_tasks()
                
                # 尝试调度任务
                if len(self._running_tasks) < self._max_concurrent_tasks:
                    self._try_schedule_task()
                
                time.sleep(0.1)  # 100ms轮询
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def _try_schedule_task(self) -> bool:
        """尝试调度一个任务"""
        try:
            # 非阻塞获取
            _, _, task = self._task_queue.get(timeout=0.1)
        except Empty:
            return False
        
        # 检查资源是否满足
        can_allocate = all(
            self.resource_pool.check_availability(req)
            for req in task.requirements
        )
        
        if not can_allocate:
            # 资源不足，放回队列
            self._task_queue.put((task.priority.value, time.time(), task))
            return False
        
        # 分配资源
        try:
            allocations = self.resource_pool.allocate_resources(
                task.task_id,
                task.requirements
            )
            
            # 启动任务
            self._start_task(task)
            return True
            
        except ResourceError as e:
            logger.error(f"Failed to allocate resources for {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self._completed_tasks[task.task_id] = task
            return False
    
    def _start_task(self, task: ComputeTask) -> None:
        """启动任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        with self._lock:
            self._running_tasks[task.task_id] = task
        
        # 在工作线程中执行任务
        worker_thread = threading.Thread(
            target=self._execute_task,
            args=(task,),
            daemon=True
        )
        worker_thread.start()
        
        if task.on_start:
            task.on_start(task.task_id)
        
        logger.info(f"Started task: {task.task_id}")
    
    def _execute_task(self, task: ComputeTask) -> None:
        """执行任务"""
        try:
            # 这里调用实际的任务执行逻辑
            # 实际实现中会根据task.task_type调用相应的模块
            result = self._run_module_task(task)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            if task.on_complete:
                task.on_complete(task.task_id, result)
            
            logger.info(f"Completed task: {task.task_id}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            if task.on_error:
                task.on_error(task.task_id, e)
            
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # 释放资源
            self.resource_pool.release_resources(task.task_id)
    
    def _run_module_task(self, task: ComputeTask) -> Any:
        """运行模块任务（实际执行逻辑）"""
        # 这是一个占位符，实际实现会调用相应的模块
        logger.debug(f"Executing task {task.task_id} with module {task.module_name}")
        
        # 模拟执行时间
        time.sleep(1)
        
        return {"status": "success", "task_id": task.task_id}
    
    def _cleanup_finished_tasks(self) -> None:
        """清理已完成任务"""
        with self._lock:
            finished = [
                task_id for task_id, task in self._running_tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            ]
            
            for task_id in finished:
                task = self._running_tasks.pop(task_id)
                self._completed_tasks[task_id] = task


class ResourceError(Exception):
    """资源错误"""
    pass


class AdaptiveScheduler(ResourceScheduler):
    """
    自适应调度器
    
    根据历史执行数据动态调整调度策略
    """
    
    def __init__(self):
        super().__init__()
        self._task_history: List[Dict[str, Any]] = []
        self._module_stats: Dict[str, Dict[str, Any]] = {}
    
    def submit_task(self, task: ComputeTask) -> None:
        """提交任务（自动优化资源需求）"""
        # 根据历史数据优化资源需求
        optimized_requirements = self._optimize_requirements(task)
        task.requirements = optimized_requirements
        
        super().submit_task(task)
    
    def record_execution(
        self,
        task: ComputeTask,
        actual_resources: Dict[str, float],
        execution_time: float
    ) -> None:
        """记录执行数据"""
        record = {
            "task_type": task.task_type,
            "module": task.module_name,
            "requested": {
                req.resource_type.value: req.amount
                for req in task.requirements
            },
            "actual": actual_resources,
            "execution_time": execution_time,
            "timestamp": time.time()
        }
        
        self._task_history.append(record)
        
        # 更新模块统计
        if task.module_name not in self._module_stats:
            self._module_stats[task.module_name] = {
                "count": 0,
                "avg_time": 0,
                "avg_cpu": 0,
                "avg_memory": 0
            }
        
        stats = self._module_stats[task.module_name]
        stats["count"] += 1
        
        # 移动平均
        n = stats["count"]
        stats["avg_time"] = (stats["avg_time"] * (n - 1) + execution_time) / n
        stats["avg_cpu"] = (stats["avg_cpu"] * (n - 1) + actual_resources.get("cpu", 0)) / n
        stats["avg_memory"] = (stats["avg_memory"] * (n - 1) + actual_resources.get("memory", 0)) / n
    
    def get_recommendations(self) -> Dict[str, Any]:
        """获取资源推荐"""
        recommendations = {}
        
        for module, stats in self._module_stats.items():
            recommendations[module] = {
                "recommended_cpu": stats["avg_cpu"] * 1.2,  # 20% buffer
                "recommended_memory": stats["avg_memory"] * 1.2,
                "expected_duration": stats["avg_time"]
            }
        
        return recommendations
    
    def _optimize_requirements(
        self,
        task: ComputeTask
    ) -> List[ResourceRequirement]:
        """优化资源需求"""
        if task.module_name not in self._module_stats:
            return task.requirements
        
        stats = self._module_stats[task.module_name]
        optimized = []
        
        for req in task.requirements:
            if req.resource_type == ResourceType.CPU:
                # 使用历史平均值+缓冲
                optimized_amount = min(
                    req.amount,
                    stats["avg_cpu"] * 1.2
                )
                optimized.append(ResourceRequirement(
                    resource_type=req.resource_type,
                    amount=optimized_amount,
                    unit=req.unit,
                    exclusive=req.exclusive,
                    duration_estimate=stats["avg_time"]
                ))
            else:
                optimized.append(req)
        
        return optimized


# 便捷函数
def get_scheduler() -> ResourceScheduler:
    """获取全局调度器实例"""
    if not hasattr(get_scheduler, '_instance'):
        get_scheduler._instance = AdaptiveScheduler()
    return get_scheduler._instance


def submit_module_task(
    task_type: str,
    module_name: str,
    requirements: List[ResourceRequirement],
    **kwargs
) -> str:
    """便捷函数：提交模块任务"""
    import uuid
    
    task = ComputeTask(
        task_id=str(uuid.uuid4()),
        task_type=task_type,
        module_name=module_name,
        requirements=requirements,
        **kwargs
    )
    
    get_scheduler().submit_task(task)
    return task.task_id