"""
Workflow Orchestrator Module
============================

Intelligent workflow orchestration with dynamic task scheduling,
dependency resolution, automatic retry/fallback, and adaptive resource allocation.

Author: DFT+LAMMPS Automation Team
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from functools import wraps
import heapq
import threading
import copy

import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    CANCELLED = auto()
    TIMEOUT = auto()
    SKIPPED = auto()


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    LICENSE = "license"


class RetryStrategy(Enum):
    """Retry strategies for failed tasks."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    ADAPTIVE = "adaptive"


class DependencyType(Enum):
    """Types of task dependencies."""
    REQUIRED = "required"           # Must complete successfully
    OPTIONAL = "optional"           # Try to complete but not required
    SOFT = "soft"                   # Can proceed in parallel
    CONDITIONAL = "conditional"     # Based on condition


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResourceRequirements:
    """Resource requirements for a task."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    disk_gb: float = 10.0
    time_limit: Optional[timedelta] = None
    custom_resources: Dict[str, Any] = field(default_factory=dict)
    
    def __add__(self, other: ResourceRequirements) -> ResourceRequirements:
        """Combine two resource requirements."""
        return ResourceRequirements(
            cpu_cores=self.cpu_cores + other.cpu_cores,
            memory_gb=self.memory_gb + other.memory_gb,
            gpu_count=self.gpu_count + other.gpu_count,
            disk_gb=self.disk_gb + other.disk_gb,
            time_limit=max(self.time_limit, other.time_limit) if self.time_limit and other.time_limit else (self.time_limit or other.time_limit),
            custom_resources={**self.custom_resources, **other.custom_resources}
        )
    
    def fits_in(self, available: ResourceRequirements) -> bool:
        """Check if requirements fit in available resources."""
        return (
            self.cpu_cores <= available.cpu_cores and
            self.memory_gb <= available.memory_gb and
            self.gpu_count <= available.gpu_count and
            self.disk_gb <= available.disk_gb
        )


@dataclass
class TaskMetrics:
    """Metrics collected during task execution."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    cpu_time: float = 0.0
    memory_peak: float = 0.0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    retry_count: int = 0
    failure_reason: Optional[str] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate task duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        dur = self.duration
        return dur.total_seconds() if dur else 0.0


@dataclass
class RetryConfig:
    """Configuration for task retry behavior."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 300.0
    retry_on: Tuple[type, ...] = (Exception,)
    fallback_action: Optional[Callable] = None
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry."""
        if self.strategy == RetryStrategy.FIXED_DELAY:
            return self.base_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
            return min(delay, self.max_delay)
        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
            return min(delay, self.max_delay)
        elif self.strategy == RetryStrategy.ADAPTIVE:
            # Adaptive based on historical data
            return self._adaptive_delay(attempt)
        return self.base_delay
    
    def _adaptive_delay(self, attempt: int) -> float:
        """Calculate adaptive delay based on historical success patterns."""
        # Simplified implementation - would use historical data in practice
        return min(self.base_delay * (1.5 ** attempt), self.max_delay)


@dataclass
class TaskDependency:
    """Represents a dependency between tasks."""
    task_id: str
    dependency_type: DependencyType = DependencyType.REQUIRED
    condition: Optional[Callable[[Any], bool]] = None
    
    def is_satisfied(self, task_result: Any) -> bool:
        """Check if dependency condition is satisfied."""
        if self.dependency_type == DependencyType.OPTIONAL:
            return True
        if self.condition:
            return self.condition(task_result)
        return task_result is not None


T = TypeVar('T')

@dataclass
class Task(Generic[T]):
    """
    Represents a unit of work in the workflow.
    
    Attributes:
        id: Unique task identifier
        name: Human-readable task name
        func: Callable to execute
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        dependencies: List of task dependencies
        resources: Resource requirements
        priority: Task priority
        retry_config: Retry configuration
        timeout: Maximum execution time
        metadata: Additional task metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "unnamed_task"
    func: Optional[Callable[..., T]] = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[TaskDependency] = field(default_factory=list)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    priority: TaskPriority = TaskPriority.NORMAL
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies satisfied)."""
        return self.status == TaskStatus.PENDING
    
    @property
    def is_finished(self) -> bool:
        """Check if task has finished execution."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, 
                              TaskStatus.CANCELLED, TaskStatus.SKIPPED)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status.name,
            'priority': self.priority.name,
            'resources': {
                'cpu_cores': self.resources.cpu_cores,
                'memory_gb': self.resources.memory_gb,
                'gpu_count': self.resources.gpu_count,
            },
            'metrics': {
                'start_time': self.metrics.start_time.isoformat() if self.metrics.start_time else None,
                'end_time': self.metrics.end_time.isoformat() if self.metrics.end_time else None,
                'duration_seconds': self.metrics.duration_seconds,
                'retry_count': self.metrics.retry_count,
            },
            'dependencies': [d.task_id for d in self.dependencies],
        }


@dataclass
class WorkflowConfig:
    """Configuration for workflow orchestration."""
    # Parallelism settings
    max_concurrent_tasks: int = 10
    max_workers: int = 4
    
    # Resource management
    total_resources: ResourceRequirements = field(default_factory=lambda: ResourceRequirements(
        cpu_cores=16, memory_gb=64.0, gpu_count=4, disk_gb=1000.0
    ))
    
    # Scheduling
    scheduling_policy: str = "priority"  # priority, fifo, fair-share
    preemption_enabled: bool = False
    
    # Retry and fault tolerance
    default_retry_config: RetryConfig = field(default_factory=RetryConfig)
    checkpoint_interval: int = 300  # seconds
    
    # Monitoring
    enable_metrics: bool = True
    metrics_collection_interval: float = 1.0
    
    # Adaptive settings
    enable_adaptive_scaling: bool = True
    scaling_threshold: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'max_workers': self.max_workers,
            'scheduling_policy': self.scheduling_policy,
            'preemption_enabled': self.preemption_enabled,
            'checkpoint_interval': self.checkpoint_interval,
            'enable_metrics': self.enable_metrics,
        }


# =============================================================================
# Dependency Graph
# =============================================================================

class DependencyGraph:
    """
    Manages task dependencies and determines execution order.
    
    Uses topological sorting with cycle detection.
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
    
    def add_task(self, task: Task) -> None:
        """Add a task to the graph."""
        with self._lock:
            self.tasks[task.id] = task
            for dep in task.dependencies:
                self._dependencies[task.id].add(dep.task_id)
                self._dependents[dep.task_id].add(task.id)
    
    def remove_task(self, task_id: str) -> None:
        """Remove a task from the graph."""
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
            
            # Remove from dependencies
            for deps in self._dependencies.values():
                deps.discard(task_id)
            if task_id in self._dependencies:
                del self._dependencies[task_id]
            
            # Remove from dependents
            for deps in self._dependents.values():
                deps.discard(task_id)
            if task_id in self._dependents:
                del self._dependents[task_id]
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (all dependencies satisfied)."""
        with self._lock:
            ready = []
            for task_id, task in self.tasks.items():
                if task.status != TaskStatus.PENDING:
                    continue
                
                # Check all dependencies
                all_satisfied = True
                for dep in task.dependencies:
                    if dep.task_id not in self.tasks:
                        all_satisfied = False
                        break
                    dep_task = self.tasks[dep.task_id]
                    if not dep.is_satisfied(dep_task.result):
                        if dep.dependency_type == DependencyType.REQUIRED:
                            all_satisfied = False
                            break
                
                if all_satisfied:
                    ready.append(task)
            
            return ready
    
    def get_execution_order(self) -> List[List[Task]]:
        """
        Get tasks organized by execution level (parallel groups).
        
        Returns a list where each inner list contains tasks that can
        execute in parallel.
        """
        with self._lock:
            # Kahn's algorithm for topological sort
            in_degree = {tid: len(deps) for tid, deps in self._dependencies.items()}
            queue = [tid for tid, deg in in_degree.items() if deg == 0]
            levels = []
            
            while queue:
                current_level = queue.copy()
                levels.append([self.tasks[tid] for tid in current_level])
                queue = []
                
                for tid in current_level:
                    for dependent in self._dependents[tid]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)
            
            return levels
    
    def detect_cycles(self) -> Optional[List[str]]:
        """Detect cycles in the dependency graph."""
        with self._lock:
            WHITE, GRAY, BLACK = 0, 1, 2
            color = {tid: WHITE for tid in self.tasks}
            path = []
            
            def dfs(node: str) -> Optional[List[str]]:
                color[node] = GRAY
                path.append(node)
                
                for dependent in self._dependents.get(node, []):
                    if color[dependent] == GRAY:
                        # Cycle detected
                        cycle_start = path.index(dependent)
                        return path[cycle_start:] + [dependent]
                    if color[dependent] == WHITE:
                        result = dfs(dependent)
                        if result:
                            return result
                
                path.pop()
                color[node] = BLACK
                return None
            
            for tid in self.tasks:
                if color[tid] == WHITE:
                    cycle = dfs(tid)
                    if cycle:
                        return cycle
            
            return None
    
    def get_critical_path(self) -> List[str]:
        """
        Calculate the critical path (longest path) through the workflow.
        
        Returns list of task IDs on the critical path.
        """
        with self._lock:
            # Calculate earliest start/finish times
            es = {tid: 0 for tid in self.tasks}  # earliest start
            ef = {}  # earliest finish
            
            for level in self.get_execution_order():
                for task in level:
                    tid = task.id
                    # Duration estimate (could use historical data)
                    duration = task.metadata.get('estimated_duration', 1.0)
                    ef[tid] = es[tid] + duration
                    
                    for dependent in self._dependents.get(tid, []):
                        es[dependent] = max(es[dependent], ef[tid])
            
            # Find the path with maximum total duration
            max_end_time = max(ef.values()) if ef else 0
            critical_tasks = [tid for tid, finish in ef.items() if finish == max_end_time]
            
            # Backtrack to find full path
            path = []
            if critical_tasks:
                current = critical_tasks[0]
                path.append(current)
                
                while self._dependencies.get(current):
                    # Find predecessor with latest finish time
                    pred = max(self._dependencies[current], 
                              key=lambda x: ef.get(x, 0))
                    path.append(pred)
                    current = pred
            
            return list(reversed(path))


# =============================================================================
# Resource Manager
# =============================================================================

class ResourceManager:
    """
    Manages computational resources and allocation.
    
    Provides adaptive resource allocation based on workload and
    historical performance data.
    """
    
    def __init__(self, total_resources: ResourceRequirements):
        self.total_resources = total_resources
        self.available_resources = copy.deepcopy(total_resources)
        self.allocated_resources: Dict[str, ResourceRequirements] = {}
        self._lock = threading.RLock()
        
        # Historical data for adaptive allocation
        self._usage_history: List[Dict[str, Any]] = []
        self._efficiency_scores: Dict[str, float] = {}
    
    def allocate(self, task_id: str, requirements: ResourceRequirements) -> bool:
        """Allocate resources for a task."""
        with self._lock:
            if not requirements.fits_in(self.available_resources):
                logger.warning(f"Insufficient resources for task {task_id}")
                return False
            
            self.allocated_resources[task_id] = requirements
            self.available_resources = ResourceRequirements(
                cpu_cores=self.available_resources.cpu_cores - requirements.cpu_cores,
                memory_gb=self.available_resources.memory_gb - requirements.memory_gb,
                gpu_count=self.available_resources.gpu_count - requirements.gpu_count,
                disk_gb=self.available_resources.disk_gb - requirements.disk_gb,
                time_limit=self.available_resources.time_limit,
            )
            
            logger.debug(f"Allocated resources for {task_id}: {requirements}")
            return True
    
    def release(self, task_id: str) -> None:
        """Release resources allocated to a task."""
        with self._lock:
            if task_id not in self.allocated_resources:
                return
            
            resources = self.allocated_resources.pop(task_id)
            self.available_resources = ResourceRequirements(
                cpu_cores=self.available_resources.cpu_cores + resources.cpu_cores,
                memory_gb=self.available_resources.memory_gb + resources.memory_gb,
                gpu_count=self.available_resources.gpu_count + resources.gpu_count,
                disk_gb=self.available_resources.disk_gb + resources.disk_gb,
                time_limit=self.available_resources.time_limit,
            )
            
            logger.debug(f"Released resources for {task_id}")
    
    def get_available(self) -> ResourceRequirements:
        """Get currently available resources."""
        with self._lock:
            return copy.deepcopy(self.available_resources)
    
    def record_usage(self, task_id: str, actual_usage: ResourceRequirements) -> None:
        """Record actual resource usage for adaptive learning."""
        with self._lock:
            self._usage_history.append({
                'task_id': task_id,
                'timestamp': datetime.now(),
                'usage': actual_usage,
            })
            
            # Calculate efficiency score
            if task_id in self.allocated_resources:
                allocated = self.allocated_resources[task_id]
                efficiency = min(
                    actual_usage.cpu_cores / max(allocated.cpu_cores, 1),
                    actual_usage.memory_gb / max(allocated.memory_gb, 1),
                )
                self._efficiency_scores[task_id] = efficiency
    
    def suggest_resources(self, task_type: str, 
                         historical_data: List[Dict[str, Any]]) -> ResourceRequirements:
        """
        Suggest resource allocation based on historical data.
        
        Uses simple statistical analysis to predict optimal resources.
        """
        if not historical_data:
            return ResourceRequirements()
        
        # Extract usage patterns
        cpu_usage = [d['usage'].cpu_cores for d in historical_data]
        memory_usage = [d['usage'].memory_gb for d in historical_data]
        
        # Use percentiles for conservative estimation
        suggested_cpu = int(np.percentile(cpu_usage, 90) * 1.2)
        suggested_memory = float(np.percentile(memory_usage, 90) * 1.2)
        
        return ResourceRequirements(
            cpu_cores=max(1, suggested_cpu),
            memory_gb=max(1.0, suggested_memory),
        )
    
    def scale_resources(self, factor: float) -> None:
        """Scale total resources by a factor."""
        with self._lock:
            self.total_resources = ResourceRequirements(
                cpu_cores=int(self.total_resources.cpu_cores * factor),
                memory_gb=self.total_resources.memory_gb * factor,
                gpu_count=int(self.total_resources.gpu_count * factor),
                disk_gb=self.total_resources.disk_gb * factor,
                time_limit=self.total_resources.time_limit,
            )
            
            # Recalculate available
            used_cpu = sum(r.cpu_cores for r in self.allocated_resources.values())
            used_mem = sum(r.memory_gb for r in self.allocated_resources.values())
            used_gpu = sum(r.gpu_count for r in self.allocated_resources.values())
            used_disk = sum(r.disk_gb for r in self.allocated_resources.values())
            
            self.available_resources = ResourceRequirements(
                cpu_cores=self.total_resources.cpu_cores - used_cpu,
                memory_gb=self.total_resources.memory_gb - used_mem,
                gpu_count=self.total_resources.gpu_count - used_gpu,
                disk_gb=self.total_resources.disk_gb - used_disk,
                time_limit=self.total_resources.time_limit,
            )


# =============================================================================
# Task Scheduler
# =============================================================================

class TaskScheduler:
    """
    Schedules tasks based on priority, resources, and dependencies.
    
    Supports multiple scheduling policies:
    - Priority: Higher priority tasks first
    - FIFO: First in, first out
    - Fair-share: Balance resources across users/projects
    """
    
    def __init__(self, config: WorkflowConfig, resource_manager: ResourceManager):
        self.config = config
        self.resource_manager = resource_manager
        self._queue: List[Tuple[int, float, str, Task]] = []  # (priority, timestamp, id, task)
        self._queue_lock = threading.RLock()
        self._counter = 0
    
    def enqueue(self, task: Task) -> None:
        """Add a task to the scheduling queue."""
        with self._queue_lock:
            priority_value = task.priority.value
            timestamp = time.time()
            self._counter += 1
            
            heapq.heappush(
                self._queue,
                (priority_value, timestamp, self._counter, task)
            )
            
            logger.debug(f"Enqueued task {task.id} with priority {task.priority.name}")
    
    def dequeue(self) -> Optional[Task]:
        """Get the next task that can be scheduled."""
        with self._queue_lock:
            available = self.resource_manager.get_available()
            
            # Find first task that fits in available resources
            temp_queue = []
            selected_task = None
            
            while self._queue and not selected_task:
                prio, ts, cnt, task = heapq.heappop(self._queue)
                
                if task.resources.fits_in(available):
                    selected_task = task
                else:
                    temp_queue.append((prio, ts, cnt, task))
            
            # Put back tasks that couldn't be scheduled
            for item in temp_queue:
                heapq.heappush(self._queue, item)
            
            if selected_task:
                logger.debug(f"Dequeued task {selected_task.id}")
            
            return selected_task
    
    def peek(self) -> Optional[Task]:
        """Look at the next task without removing it."""
        with self._queue_lock:
            if self._queue:
                return self._queue[0][3]
            return None
    
    def remove(self, task_id: str) -> bool:
        """Remove a specific task from the queue."""
        with self._queue_lock:
            for i, (prio, ts, cnt, task) in enumerate(self._queue):
                if task.id == task_id:
                    self._queue.pop(i)
                    heapq.heapify(self._queue)
                    return True
            return False
    
    def get_queue_length(self) -> int:
        """Get number of tasks in queue."""
        with self._queue_lock:
            return len(self._queue)
    
    def clear(self) -> None:
        """Clear the scheduling queue."""
        with self._queue_lock:
            self._queue.clear()


# =============================================================================
# Retry Handler
# =============================================================================

class RetryHandler:
    """
    Handles task retry logic with various backoff strategies.
    """
    
    def __init__(self):
        self._retry_counts: Dict[str, int] = {}
        self._failure_history: List[Dict[str, Any]] = []
    
    def should_retry(self, task: Task, exception: Exception) -> bool:
        """Determine if a task should be retried."""
        retry_count = self._retry_counts.get(task.id, 0)
        
        if retry_count >= task.retry_config.max_retries:
            return False
        
        if not isinstance(exception, task.retry_config.retry_on):
            return False
        
        return True
    
    def get_retry_delay(self, task: Task) -> float:
        """Get the delay before next retry attempt."""
        retry_count = self._retry_counts.get(task.id, 0)
        return task.retry_config.calculate_delay(retry_count)
    
    def record_failure(self, task: Task, exception: Exception) -> None:
        """Record a task failure."""
        self._retry_counts[task.id] = self._retry_counts.get(task.id, 0) + 1
        
        self._failure_history.append({
            'task_id': task.id,
            'timestamp': datetime.now(),
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'retry_count': self._retry_counts[task.id],
        })
        
        task.metrics.retry_count = self._retry_counts[task.id]
        task.metrics.failure_reason = str(exception)
    
    def execute_fallback(self, task: Task) -> Any:
        """Execute fallback action if configured."""
        if task.retry_config.fallback_action:
            logger.info(f"Executing fallback for task {task.id}")
            return task.retry_config.fallback_action(task)
        return None
    
    def get_failure_patterns(self) -> Dict[str, Any]:
        """Analyze failure patterns for system improvement."""
        if not self._failure_history:
            return {}
        
        patterns = defaultdict(int)
        for failure in self._failure_history:
            patterns[failure['exception_type']] += 1
        
        return {
            'total_failures': len(self._failure_history),
            'unique_exception_types': len(patterns),
            'most_common': max(patterns.items(), key=lambda x: x[1]) if patterns else None,
            'patterns': dict(patterns),
        }


# =============================================================================
# Workflow Orchestrator
# =============================================================================

class WorkflowOrchestrator:
    """
    Main orchestrator for workflow execution.
    
    Coordinates task scheduling, resource allocation, dependency management,
    retry logic, and monitoring.
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        
        # Core components
        self.dependency_graph = DependencyGraph()
        self.resource_manager = ResourceManager(self.config.total_resources)
        self.task_scheduler = TaskScheduler(self.config, self.resource_manager)
        self.retry_handler = RetryHandler()
        
        # Execution state
        self._tasks: Dict[str, Task] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks: Dict[str, Task] = {}
        self._failed_tasks: Dict[str, Task] = {}
        
        # Threading and async
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._stop_event = threading.Event()
        
        # Metrics
        self._workflow_start_time: Optional[datetime] = None
        self._workflow_end_time: Optional[datetime] = None
        self._metrics_collector: Optional[threading.Thread] = None
    
    def add_task(self, task: Task) -> str:
        """Add a task to the workflow."""
        with self._lock:
            self._tasks[task.id] = task
            self.dependency_graph.add_task(task)
            
            logger.info(f"Added task {task.id} ({task.name}) to workflow")
            return task.id
    
    def add_tasks(self, tasks: List[Task]) -> List[str]:
        """Add multiple tasks to the workflow."""
        return [self.add_task(t) for t in tasks]
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the workflow."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            
            task = self._tasks.pop(task_id)
            self.dependency_graph.remove_task(task_id)
            self.task_scheduler.remove(task_id)
            
            logger.info(f"Removed task {task_id} ({task.name}) from workflow")
            return True
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def validate_workflow(self) -> Tuple[bool, Optional[str]]:
        """
        Validate the workflow for cycles and other issues.
        
        Returns (is_valid, error_message)
        """
        # Check for cycles
        cycle = self.dependency_graph.detect_cycles()
        if cycle:
            return False, f"Dependency cycle detected: {' -> '.join(cycle)}"
        
        # Check for undefined dependencies
        for task in self._tasks.values():
            for dep in task.dependencies:
                if dep.task_id not in self._tasks:
                    return False, f"Task {task.id} has undefined dependency: {dep.task_id}"
        
        return True, None
    
    async def run(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Args:
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with execution results and statistics
        """
        # Validate workflow
        is_valid, error = self.validate_workflow()
        if not is_valid:
            raise ValueError(f"Invalid workflow: {error}")
        
        self._workflow_start_time = datetime.now()
        logger.info("Starting workflow execution")
        
        # Start metrics collection
        if self.config.enable_metrics:
            self._start_metrics_collection()
        
        try:
            # Execute with timeout if specified
            if timeout:
                result = await asyncio.wait_for(
                    self._execute_workflow(),
                    timeout=timeout
                )
            else:
                result = await self._execute_workflow()
            
            self._workflow_end_time = datetime.now()
            return result
            
        except asyncio.TimeoutError:
            logger.error("Workflow execution timed out")
            self._cancel_all_tasks()
            raise
        finally:
            self._stop_metrics_collection()
    
    async def _execute_workflow(self) -> Dict[str, Any]:
        """Internal workflow execution loop."""
        # Queue all ready tasks
        ready_tasks = self.dependency_graph.get_ready_tasks()
        for task in ready_tasks:
            self.task_scheduler.enqueue(task)
        
        # Main execution loop
        while True:
            with self._lock:
                # Check if all tasks are complete
                all_finished = all(
                    t.is_finished for t in self._tasks.values()
                )
                
                if all_finished and not self._running_tasks:
                    break
            
            # Schedule new tasks
            await self._schedule_tasks()
            
            # Process completed tasks
            await self._process_completed_tasks()
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.01)
        
        return self._get_execution_summary()
    
    async def _schedule_tasks(self) -> None:
        """Schedule tasks that are ready to run."""
        with self._lock:
            while len(self._running_tasks) < self.config.max_concurrent_tasks:
                task = self.task_scheduler.dequeue()
                if not task:
                    break
                
                # Allocate resources
                if not self.resource_manager.allocate(task.id, task.resources):
                    # Put back in queue if allocation fails
                    self.task_scheduler.enqueue(task)
                    break
                
                # Create async task
                task.status = TaskStatus.SCHEDULED
                asyncio_task = asyncio.create_task(
                    self._execute_task(task),
                    name=f"task_{task.id}"
                )
                self._running_tasks[task.id] = asyncio_task
                
                logger.info(f"Scheduled task {task.id} ({task.name})")
    
    async def _execute_task(self, task: Task) -> None:
        """Execute a single task with retry logic."""
        task.status = TaskStatus.RUNNING
        task.metrics.start_time = datetime.now()
        
        logger.info(f"Starting task {task.id} ({task.name})")
        
        try:
            # Execute with timeout if specified
            if task.timeout:
                result = await asyncio.wait_for(
                    self._run_task_function(task),
                    timeout=task.timeout
                )
            else:
                result = await self._run_task_function(task)
            
            # Task completed successfully
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.metrics.end_time = datetime.now()
            
            self._completed_tasks[task.id] = task
            
            logger.info(f"Task {task.id} completed successfully "
                       f"({task.metrics.duration_seconds:.2f}s)")
            
            # Queue dependent tasks
            await self._queue_dependent_tasks(task)
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.metrics.end_time = datetime.now()
            task.metrics.failure_reason = "Execution timeout"
            
            logger.error(f"Task {task.id} timed out")
            await self._handle_task_failure(task, TimeoutError("Task execution timed out"))
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.metrics.end_time = datetime.now()
            task.metrics.failure_reason = str(e)
            
            logger.error(f"Task {task.id} failed: {e}")
            await self._handle_task_failure(task, e)
        
        finally:
            # Release resources
            self.resource_manager.release(task.id)
            
            with self._lock:
                if task.id in self._running_tasks:
                    del self._running_tasks[task.id]
    
    async def _run_task_function(self, task: Task) -> Any:
        """Run the actual task function."""
        if task.func is None:
            return None
        
        # Run in thread pool for CPU-bound tasks
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: task.func(*task.args, **task.kwargs)
        )
    
    async def _handle_task_failure(self, task: Task, exception: Exception) -> None:
        """Handle task failure with retry logic."""
        self.retry_handler.record_failure(task, exception)
        
        if self.retry_handler.should_retry(task, exception):
            # Schedule retry
            delay = self.retry_handler.get_retry_delay(task)
            task.status = TaskStatus.RETRYING
            
            logger.info(f"Retrying task {task.id} in {delay:.2f}s "
                       f"(attempt {task.metrics.retry_count + 1}/"
                       f"{task.retry_config.max_retries + 1})")
            
            await asyncio.sleep(delay)
            task.status = TaskStatus.PENDING
            self.task_scheduler.enqueue(task)
        else:
            # Max retries exceeded
            self._failed_tasks[task.id] = task
            
            # Try fallback
            try:
                fallback_result = self.retry_handler.execute_fallback(task)
                if fallback_result is not None:
                    task.result = fallback_result
                    task.status = TaskStatus.COMPLETED
                    self._completed_tasks[task.id] = task
                    del self._failed_tasks[task.id]
                    
                    logger.info(f"Task {task.id} completed via fallback")
                    await self._queue_dependent_tasks(task)
            except Exception as fallback_error:
                logger.error(f"Fallback for task {task.id} also failed: {fallback_error}")
    
    async def _queue_dependent_tasks(self, completed_task: Task) -> None:
        """Queue tasks that depend on the completed task."""
        ready_tasks = self.dependency_graph.get_ready_tasks()
        for task in ready_tasks:
            if task.status == TaskStatus.PENDING:
                self.task_scheduler.enqueue(task)
    
    async def _process_completed_tasks(self) -> None:
        """Process any completed async tasks."""
        # This is handled by the individual task coroutines
        await asyncio.sleep(0)
    
    def _cancel_all_tasks(self) -> None:
        """Cancel all running tasks."""
        with self._lock:
            for task_id, async_task in list(self._running_tasks.items()):
                async_task.cancel()
                if task_id in self._tasks:
                    self._tasks[task_id].status = TaskStatus.CANCELLED
    
    def _start_metrics_collection(self) -> None:
        """Start background metrics collection."""
        def collect_metrics():
            while not self._stop_event.is_set():
                # Collect system metrics
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'running_tasks': len(self._running_tasks),
                    'completed_tasks': len(self._completed_tasks),
                    'failed_tasks': len(self._failed_tasks),
                    'queue_length': self.task_scheduler.get_queue_length(),
                }
                logger.debug(f"Metrics: {metrics}")
                
                self._stop_event.wait(self.config.metrics_collection_interval)
        
        self._metrics_collector = threading.Thread(target=collect_metrics)
        self._metrics_collector.start()
    
    def _stop_metrics_collection(self) -> None:
        """Stop background metrics collection."""
        if self._metrics_collector:
            self._stop_event.set()
            self._metrics_collector.join(timeout=5.0)
    
    def _get_execution_summary(self) -> Dict[str, Any]:
        """Generate execution summary."""
        total_tasks = len(self._tasks)
        completed = len(self._completed_tasks)
        failed = len(self._failed_tasks)
        
        duration = None
        if self._workflow_start_time and self._workflow_end_time:
            duration = (self._workflow_end_time - self._workflow_start_time).total_seconds()
        
        # Calculate resource efficiency
        total_cpu_time = sum(
            t.metrics.duration_seconds * t.resources.cpu_cores
            for t in self._completed_tasks.values()
        )
        
        return {
            'success': failed == 0,
            'total_tasks': total_tasks,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_tasks if total_tasks > 0 else 0,
            'duration_seconds': duration,
            'total_cpu_time': total_cpu_time,
            'tasks': {tid: t.to_dict() for tid, t in self._tasks.items()},
            'failure_patterns': self.retry_handler.get_failure_patterns(),
            'critical_path': self.dependency_graph.get_critical_path(),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            'total_tasks': len(self._tasks),
            'running': len(self._running_tasks),
            'completed': len(self._completed_tasks),
            'failed': len(self._failed_tasks),
            'queue_length': self.task_scheduler.get_queue_length(),
            'available_resources': {
                'cpu_cores': self.resource_manager.available_resources.cpu_cores,
                'memory_gb': self.resource_manager.available_resources.memory_gb,
                'gpu_count': self.resource_manager.available_resources.gpu_count,
            },
        }
    
    def pause(self) -> None:
        """Pause workflow execution."""
        logger.info("Pausing workflow execution")
        # Implementation would track pause state and prevent new scheduling
    
    def resume(self) -> None:
        """Resume workflow execution."""
        logger.info("Resuming workflow execution")
        # Implementation would clear pause state
    
    def shutdown(self) -> None:
        """Shutdown the orchestrator and clean up resources."""
        logger.info("Shutting down workflow orchestrator")
        self._cancel_all_tasks()
        self._executor.shutdown(wait=True)
        self._stop_metrics_collection()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_simple_workflow(
    tasks: List[Tuple[str, Callable, Tuple, Dict]],
    dependencies: Optional[Dict[str, List[str]]] = None
) -> WorkflowOrchestrator:
    """
    Create a simple workflow from task definitions.
    
    Args:
        tasks: List of (name, func, args, kwargs) tuples
        dependencies: Dictionary mapping task names to dependency names
        
    Returns:
        Configured WorkflowOrchestrator
    """
    orchestrator = WorkflowOrchestrator()
    task_map = {}
    
    # Create tasks
    for name, func, args, kwargs in tasks:
        task = Task(name=name, func=func, args=args, kwargs=kwargs)
        task_map[name] = task.id
        orchestrator.add_task(task)
    
    # Set up dependencies
    if dependencies:
        for task_name, dep_names in dependencies.items():
            if task_name in task_map:
                task = orchestrator.get_task(task_map[task_name])
                for dep_name in dep_names:
                    if dep_name in task_map:
                        task.dependencies.append(
                            TaskDependency(task_id=task_map[dep_name])
                        )
    
    return orchestrator


def run_workflow_sync(
    orchestrator: WorkflowOrchestrator,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Run a workflow synchronously."""
    return asyncio.run(orchestrator.run(timeout))


# =============================================================================
# Decorators
# =============================================================================

def workflow_task(
    priority: TaskPriority = TaskPriority.NORMAL,
    resources: Optional[ResourceRequirements] = None,
    retry_config: Optional[RetryConfig] = None,
    timeout: Optional[float] = None
):
    """Decorator to mark a function as a workflow task."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Attach task configuration
        wrapper._task_config = {
            'priority': priority,
            'resources': resources or ResourceRequirements(),
            'retry_config': retry_config or RetryConfig(),
            'timeout': timeout,
            'func': func,
        }
        
        return wrapper
    return decorator


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example workflow demonstration
    logging.basicConfig(level=logging.INFO)
    
    def task_a():
        time.sleep(0.5)
        return "Result A"
    
    def task_b(x):
        time.sleep(0.3)
        return f"Result B with {x}"
    
    def task_c(x, y):
        time.sleep(0.4)
        return f"Result C with {x} and {y}"
    
    # Create workflow
    orchestrator = WorkflowOrchestrator(
        config=WorkflowConfig(max_concurrent_tasks=3)
    )
    
    # Add tasks
    task1 = Task(name="Task A", func=task_a, priority=TaskPriority.HIGH)
    task2 = Task(name="Task B", func=task_b, args=("input",), 
                 dependencies=[TaskDependency(task1.id)])
    task3 = Task(name="Task C", func=task_c, args=("x", "y"),
                 dependencies=[TaskDependency(task1.id)])
    
    orchestrator.add_task(task1)
    orchestrator.add_task(task2)
    orchestrator.add_task(task3)
    
    # Run workflow
    result = asyncio.run(orchestrator.run())
    print(f"Workflow result: {result}")
