"""
KAS Distributed Agent Cluster - 分布式任务调度器模块

提供分布式任务的调度、分片、状态跟踪和结果聚合功能。
"""
import asyncio
import uuid
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"           # 等待执行
    SCHEDULING = "scheduling"     # 正在调度
    RUNNING = "running"           # 运行中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 失败
    CANCELLED = "cancelled"       # 已取消
    TIMEOUT = "timeout"           # 超时


class TaskType(Enum):
    """任务类型"""
    SINGLE = "single"             # 单节点任务
    SHARDED = "sharded"           # 分片任务（多节点并行）
    PIPELINE = "pipeline"         # 流水线任务（多阶段）
    MAP_REDUCE = "map_reduce"     # MapReduce任务
    BROADCAST = "broadcast"       # 广播任务（所有节点）


@dataclass
class TaskShard:
    """
    任务分片
    
    Attributes:
        shard_id: 分片ID
        task_id: 所属任务ID
        node_id: 分配的节点
        data: 分片数据
        status: 分片状态
        result: 分片结果
        started_at: 开始时间
        completed_at: 完成时间
        error: 错误信息
    """
    shard_id: str
    task_id: str
    node_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "shard_id": self.shard_id,
            "task_id": self.task_id,
            "node_id": self.node_id,
            "data": self.data,
            "status": self.status.value,
            "result": self.result,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error
        }


@dataclass
class Task:
    """
    任务定义
    
    Attributes:
        task_id: 任务唯一标识
        task_type: 任务类型
        name: 任务名称
        payload: 任务数据
        status: 任务状态
        shards: 任务分片
        created_at: 创建时间
        started_at: 开始时间
        completed_at: 完成时间
        timeout: 超时时间（秒）
        priority: 优先级（1-10，越小越高）
        dependencies: 依赖任务ID列表
        result_aggregator: 结果聚合函数名
        retry_count: 重试次数
        max_retries: 最大重试次数
    """
    task_id: str
    task_type: TaskType
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    shards: Dict[str, TaskShard] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: float = 300.0
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    result_aggregator: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "name": self.name,
            "payload": self.payload,
            "status": self.status.value,
            "shards": {k: v.to_dict() for k, v in self.shards.items()},
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "timeout": self.timeout,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "result_aggregator": self.result_aggregator,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


@dataclass
class TaskResult:
    """
    任务结果
    
    Attributes:
        task_id: 任务ID
        status: 最终状态
        result: 结果数据
        shards_results: 各分片结果
        started_at: 开始时间
        completed_at: 完成时间
        execution_time_ms: 执行时间（毫秒）
        error: 错误信息
    """
    task_id: str
    status: TaskStatus
    result: Any = None
    shards_results: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "shards_results": self.shards_results,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error
        }


class TaskAggregator:
    """任务结果聚合器"""
    
    @staticmethod
    def concat(shards_results: Dict[str, Any]) -> Any:
        """连接所有分片结果"""
        return list(shards_results.values())
    
    @staticmethod
    def sum(shards_results: Dict[str, Any]) -> Any:
        """求和聚合"""
        results = shards_results.values()
        if all(isinstance(r, (int, float)) for r in results):
            return sum(results)
        return list(results)
    
    @staticmethod
    def average(shards_results: Dict[str, Any]) -> Any:
        """平均值聚合"""
        results = list(shards_results.values())
        if not results:
            return 0
        if all(isinstance(r, (int, float)) for r in results):
            return sum(results) / len(results)
        return list(results)
    
    @staticmethod
    def count(shards_results: Dict[str, Any]) -> int:
        """计数聚合"""
        return len(shards_results)
    
    @staticmethod
    def first(shards_results: Dict[str, Any]) -> Any:
        """取第一个结果"""
        return next(iter(shards_results.values())) if shards_results else None
    
    @staticmethod
    def merge_dicts(shards_results: Dict[str, Any]) -> Dict:
        """合并字典结果"""
        merged = {}
        for result in shards_results.values():
            if isinstance(result, dict):
                merged.update(result)
        return merged
    
    @staticmethod
    def get_aggregator(name: str) -> Callable:
        """获取聚合器函数"""
        aggregators = {
            "concat": TaskAggregator.concat,
            "sum": TaskAggregator.sum,
            "average": TaskAggregator.average,
            "count": TaskAggregator.count,
            "first": TaskAggregator.first,
            "merge_dicts": TaskAggregator.merge_dicts,
        }
        return aggregators.get(name, TaskAggregator.concat)


class DistributedScheduler:
    """
    分布式任务调度器
    
    负责任务的分布式调度：
    1. 任务分片 - 将大任务拆分到多个节点
    2. 任务状态跟踪 - 实时监控任务执行状态
    3. 结果聚合 - 收集和合并分片结果
    4. 故障恢复 - 失败任务重试和迁移
    5. 依赖管理 - 处理任务依赖关系
    
    Example:
        scheduler = DistributedScheduler(cluster_manager)
        await scheduler.start()
        
        # 提交单节点任务
        task = await scheduler.submit_task({
            "name": "process_data",
            "type": "single",
            "payload": {"data": "..."}
        })
        
        # 提交分片任务
        sharded_task = await scheduler.submit_sharded_task({
            "name": "batch_process",
            "shards": 4,
            "shard_data": [...]
        })
        
        # 等待结果
        result = await scheduler.wait_for_task(task.task_id)
    """
    
    def __init__(self, cluster_manager):
        """
        初始化调度器
        
        Args:
            cluster_manager: 集群管理器实例
        """
        from kas.core.cluster.manager import ClusterManager
        self.cluster_manager: ClusterManager = cluster_manager
        
        # 任务存储
        self._tasks: Dict[str, Task] = {}
        self._task_results: Dict[str, TaskResult] = {}
        
        # 任务队列（按优先级排序）
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # 任务事件（用于等待任务完成）
        self._task_events: Dict[str, asyncio.Event] = {}
        
        # 执行中的任务
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # 任务完成回调
        self._task_callbacks: Dict[str, List[Callable]] = {}
        
        # 工作线程
        self._scheduler_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # 运行状态
        self._running = False
        self._stop_event = asyncio.Event()
        
        # 统计信息
        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0
        }
        
        logger.info("DistributedScheduler initialized")
    
    # ==================== 生命周期管理 ====================
    
    async def start(self) -> bool:
        """启动调度器"""
        if self._running:
            return True
        
        logger.info("Starting DistributedScheduler")
        self._running = True
        
        # 启动调度循环
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # 启动监控循环
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        # 启动清理循环
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        return True
    
    async def stop(self) -> None:
        """停止调度器"""
        if not self._running:
            return
        
        logger.info("Stopping DistributedScheduler")
        self._running = False
        self._stop_event.set()
        
        # 取消所有运行中的任务
        for task in self._running_tasks.values():
            task.cancel()
        
        # 取消工作线程
        if self._scheduler_task:
            self._scheduler_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # 设置所有等待事件
        for event in self._task_events.values():
            event.set()
        
        logger.info("DistributedScheduler stopped")
    
    # ==================== 任务提交 ====================
    
    async def submit_task(self, task_def: Dict[str, Any]) -> Task:
        """
        提交任务
        
        Args:
            task_def: 任务定义
                - name: 任务名称
                - type: 任务类型 (single, sharded, pipeline, map_reduce, broadcast)
                - payload: 任务数据
                - timeout: 超时时间
                - priority: 优先级
                - dependencies: 依赖任务ID列表
        
        Returns:
            Task对象
        """
        task_id = task_def.get("task_id") or str(uuid.uuid4())
        task_type = TaskType(task_def.get("type", "single"))
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            name=task_def.get("name", "unnamed_task"),
            payload=task_def.get("payload", {}),
            timeout=task_def.get("timeout", 300.0),
            priority=task_def.get("priority", 5),
            dependencies=task_def.get("dependencies", []),
            result_aggregator=task_def.get("aggregator", "concat")
        )
        
        # 如果是分片任务，创建分片
        if task_type == TaskType.SHARDED:
            shards_data = task_def.get("shards_data", [])
            num_shards = task_def.get("num_shards", len(shards_data))
            
            if not shards_data and num_shards > 0:
                # 如果没有提供分片数据，将payload均匀分割
                shards_data = self._split_payload(task.payload, num_shards)
            
            for i, shard_data in enumerate(shards_data):
                shard_id = f"{task_id}_shard_{i}"
                task.shards[shard_id] = TaskShard(
                    shard_id=shard_id,
                    task_id=task_id,
                    data=shard_data
                )
        
        # 存储任务
        self._tasks[task_id] = task
        self._task_events[task_id] = asyncio.Event()
        
        # 加入队列（优先级越小越优先）
        await self._task_queue.put((task.priority, task_id))
        
        self._stats["tasks_submitted"] += 1
        logger.info(f"Task submitted: {task_id} ({task.name})")
        
        return task
    
    async def submit_sharded_task(self, name: str,
                                   shards_data: List[Dict[str, Any]],
                                   payload: Optional[Dict] = None,
                                   aggregator: str = "concat",
                                   timeout: float = 300.0,
                                   priority: int = 5) -> Task:
        """
        提交分片任务
        
        Args:
            name: 任务名称
            shards_data: 各分片数据列表
            payload: 共享的任务数据
            aggregator: 结果聚合方式
            timeout: 超时时间
            priority: 优先级
        
        Returns:
            Task对象
        """
        return await self.submit_task({
            "name": name,
            "type": "sharded",
            "payload": payload or {},
            "shards_data": shards_data,
            "num_shards": len(shards_data),
            "aggregator": aggregator,
            "timeout": timeout,
            "priority": priority
        })
    
    async def submit_map_reduce_task(self, name: str,
                                      map_data: List[Any],
                                      reducer: str = "concat",
                                      timeout: float = 300.0) -> Task:
        """
        提交MapReduce任务
        
        Args:
            name: 任务名称
            map_data: 待处理数据列表
            reducer: 归约函数
            timeout: 超时时间
        
        Returns:
            Task对象
        """
        # 每个数据项作为一个分片
        shards_data = [{"item": item, "index": i}
                       for i, item in enumerate(map_data)]
        
        return await self.submit_task({
            "name": name,
            "type": "map_reduce",
            "shards_data": shards_data,
            "payload": {"mapper": "default"},
            "aggregator": reducer,
            "timeout": timeout
        })
    
    def _split_payload(self, payload: Dict[str, Any],
                       num_shards: int) -> List[Dict[str, Any]]:
        """将payload分割为多个分片"""
        shards = []
        
        # 如果payload包含items列表，按items分割
        if "items" in payload and isinstance(payload["items"], list):
            items = payload["items"]
            items_per_shard = max(1, len(items) // num_shards)
            
            for i in range(num_shards):
                start = i * items_per_shard
                end = start + items_per_shard if i < num_shards - 1 else len(items)
                shard_payload = {**payload, "items": items[start:end]}
                shards.append(shard_payload)
        else:
            # 否则，复制payload到每个分片
            for i in range(num_shards):
                shards.append({**payload, "shard_index": i, "total_shards": num_shards})
        
        return shards
    
    # ==================== 任务调度 ====================
    
    async def _scheduler_loop(self) -> None:
        """调度器主循环"""
        while self._running:
            try:
                # 等待任务
                priority, task_id = await self._task_queue.get()
                
                if not self._running:
                    break
                
                task = self._tasks.get(task_id)
                if not task:
                    continue
                
                # 检查依赖
                if task.dependencies:
                    deps_satisfied = await self._check_dependencies(task)
                    if not deps_satisfied:
                        # 依赖未满足，放回队列
                        await self._task_queue.put((priority + 1, task_id))
                        continue
                
                # 调度任务
                self._running_tasks[task_id] = asyncio.create_task(
                    self._execute_task(task)
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
    
    async def _execute_task(self, task: Task) -> None:
        """执行任务"""
        task.status = TaskStatus.SCHEDULING
        task.started_at = datetime.now()
        
        try:
            if task.task_type == TaskType.SINGLE:
                await self._execute_single_task(task)
            elif task.task_type == TaskType.SHARDED:
                await self._execute_sharded_task(task)
            elif task.task_type == TaskType.MAP_REDUCE:
                await self._execute_map_reduce_task(task)
            elif task.task_type == TaskType.BROADCAST:
                await self._execute_broadcast_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self._stats["tasks_cancelled"] += 1
        except Exception as e:
            logger.error(f"Task {task.task_id} execution error: {e}")
            task.status = TaskStatus.FAILED
            
            # 重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                await self._task_queue.put((task.priority, task.task_id))
            else:
                self._stats["tasks_failed"] += 1
                await self._complete_task(task, error=str(e))
        
        finally:
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]
    
    async def _execute_single_task(self, task: Task) -> None:
        """执行单节点任务"""
        task.status = TaskStatus.RUNNING
        
        # 选择节点
        node_id = self.cluster_manager.select_node_for_task(
            task.task_id,
            task.payload.get("requirements")
        )
        
        if not node_id:
            raise RuntimeError("No available node for task")
        
        # 分发任务
        result = await self.cluster_manager.distribute_task({
            "task_id": task.task_id,
            "type": "single",
            "payload": task.payload
        }, node_id)
        
        if result.get("success"):
            await self._complete_task(task, result=result.get("result"))
        else:
            raise RuntimeError(result.get("error", "Task execution failed"))
    
    async def _execute_sharded_task(self, task: Task) -> None:
        """执行分片任务"""
        task.status = TaskStatus.RUNNING
        
        # 为每个分片选择节点
        shard_count = len(task.shards)
        nodes = self.cluster_manager.select_nodes_for_sharding(task.task_id, shard_count)
        
        if not nodes:
            raise RuntimeError("No available nodes for sharded task")
        
        # 循环分配节点
        shard_tasks = []
        for i, (shard_id, shard) in enumerate(task.shards.items()):
            node_id = nodes[i % len(nodes)]
            shard.node_id = node_id
            shard.status = TaskStatus.RUNNING
            shard.started_at = datetime.now()
            
            shard_tasks.append(self._execute_shard(shard, node_id))
        
        # 等待所有分片完成
        await asyncio.gather(*shard_tasks, return_exceptions=True)
        
        # 检查分片状态
        failed_shards = [s for s in task.shards.values() if s.status == TaskStatus.FAILED]
        
        if failed_shards:
            # 有分片失败，尝试重试
            for shard in failed_shards:
                if shard.retry_count < task.max_retries:
                    shard.retry_count = shard.retry_count + 1 if hasattr(shard, 'retry_count') else 1
                    new_node = self.cluster_manager.select_node_for_task(shard.shard_id)
                    if new_node:
                        shard.node_id = new_node
                        shard.status = TaskStatus.PENDING
                        await self._execute_shard(shard, new_node)
            
            # 再次检查
            failed_shards = [s for s in task.shards.values() if s.status == TaskStatus.FAILED]
            if failed_shards:
                raise RuntimeError(f"{len(failed_shards)} shards failed")
        
        # 聚合结果
        shards_results = {sid: shard.result for sid, shard in task.shards.items()}
        aggregator = TaskAggregator.get_aggregator(task.result_aggregator or "concat")
        final_result = aggregator(shards_results)
        
        await self._complete_task(task, result=final_result, shards_results=shards_results)
    
    async def _execute_shard(self, shard: TaskShard, node_id: str) -> None:
        """执行单个分片"""
        try:
            result = await self.cluster_manager.distribute_task({
                "task_id": shard.shard_id,
                "type": "shard",
                "payload": shard.data,
                "parent_task": shard.task_id
            }, node_id)
            
            if result.get("success"):
                shard.status = TaskStatus.COMPLETED
                shard.result = result.get("result")
            else:
                shard.status = TaskStatus.FAILED
                shard.error = result.get("error")
                
        except Exception as e:
            shard.status = TaskStatus.FAILED
            shard.error = str(e)
        finally:
            shard.completed_at = datetime.now()
    
    async def _execute_map_reduce_task(self, task: Task) -> None:
        """执行MapReduce任务（与sharded类似，但强调map/reduce语义）"""
        # Map阶段
        await self._execute_sharded_task(task)
        # Reduce阶段已通过aggregator完成
    
    async def _execute_broadcast_task(self, task: Task) -> None:
        """执行广播任务"""
        task.status = TaskStatus.RUNNING
        
        result = await self.cluster_manager.broadcast_task({
            "task_id": task.task_id,
            "payload": task.payload
        })
        
        if result.get("success"):
            await self._complete_task(task, result=result.get("results"))
        else:
            raise RuntimeError("Broadcast task failed")
    
    async def _complete_task(self, task: Task, result: Any = None,
                             shards_results: Optional[Dict] = None,
                             error: Optional[str] = None) -> None:
        """完成任务"""
        task.completed_at = datetime.now()
        
        if error:
            task.status = TaskStatus.FAILED
        else:
            task.status = TaskStatus.COMPLETED
            self._stats["tasks_completed"] += 1
        
        # 创建结果
        execution_time = 0.0
        if task.started_at and task.completed_at:
            execution_time = (task.completed_at - task.started_at).total_seconds() * 1000
        
        task_result = TaskResult(
            task_id=task.task_id,
            status=task.status,
            result=result,
            shards_results=shards_results or {},
            started_at=task.started_at,
            completed_at=task.completed_at,
            execution_time_ms=execution_time,
            error=error
        )
        
        self._task_results[task.task_id] = task_result
        
        # 触发事件
        if task.task_id in self._task_events:
            self._task_events[task.task_id].set()
        
        # 调用回调
        await self._trigger_callbacks(task.task_id, task_result)
        
        logger.info(f"Task {task.task_id} completed with status {task.status.value}")
    
    async def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖是否满足"""
        for dep_id in task.dependencies:
            if dep_id not in self._task_results:
                return False
            
            dep_result = self._task_results[dep_id]
            if dep_result.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    # ==================== 任务监控 ====================
    
    async def _monitor_loop(self) -> None:
        """监控循环 - 检查任务超时"""
        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                now = datetime.now()
                
                for task in self._tasks.values():
                    if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED,
                                          TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
                        # 检查是否超时
                        if task.started_at:
                            elapsed = (now - task.started_at).total_seconds()
                            if elapsed > task.timeout:
                                logger.warning(f"Task {task.task_id} timed out")
                                task.status = TaskStatus.TIMEOUT
                                await self._complete_task(task, error="Task timeout")
    
    async def _cleanup_loop(self) -> None:
        """清理循环 - 清理旧任务"""
        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=300.0  # 每5分钟清理一次
                )
            except asyncio.TimeoutError:
                # 清理已完成超过1小时的任务
                cutoff = datetime.now() - timedelta(hours=1)
                
                for task_id in list(self._tasks.keys()):
                    task = self._tasks[task_id]
                    if (task.completed_at and task.completed_at < cutoff):
                        del self._tasks[task_id]
                        if task_id in self._task_results:
                            del self._task_results[task_id]
                        if task_id in self._task_events:
                            del self._task_events[task_id]
                        if task_id in self._task_callbacks:
                            del self._task_callbacks[task_id]
    
    # ==================== 任务查询与控制 ====================
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务信息"""
        return self._tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        return self._task_results.get(task_id)
    
    async def wait_for_task(self, task_id: str,
                            timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 等待超时时间
        
        Returns:
            任务结果，或None（超时）
        """
        # 检查是否已完成
        if task_id in self._task_results:
            return self._task_results[task_id]
        
        # 等待完成事件
        event = self._task_events.get(task_id)
        if not event:
            return None
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._task_results.get(task_id)
        except asyncio.TimeoutError:
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否成功取消
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return False
        
        # 取消运行中的任务
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
        
        task.status = TaskStatus.CANCELLED
        await self._complete_task(task, error="Task cancelled")
        
        return True
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """
        列出任务
        
        Args:
            status: 状态过滤器
        
        Returns:
            任务列表
        """
        tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return tasks
    
    def on_task_complete(self, task_id: str, callback: Callable) -> None:
        """注册任务完成回调"""
        if task_id not in self._task_callbacks:
            self._task_callbacks[task_id] = []
        self._task_callbacks[task_id].append(callback)
    
    async def _trigger_callbacks(self, task_id: str, result: TaskResult) -> None:
        """触发任务完成回调"""
        callbacks = self._task_callbacks.get(task_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Task callback error: {e}")
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        return {
            **self._stats,
            "tasks_pending": len([t for t in self._tasks.values()
                                  if t.status == TaskStatus.PENDING]),
            "tasks_running": len([t for t in self._tasks.values()
                                 if t.status == TaskStatus.RUNNING]),
            "tasks_in_memory": len(self._tasks),
            "results_in_memory": len(self._task_results),
            "queue_size": self._task_queue.qsize()
        }
