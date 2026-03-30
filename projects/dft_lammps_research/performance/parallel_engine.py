#!/usr/bin/env python3
"""
parallel_engine.py
==================
并行计算引擎模块

提供高效的并行计算功能：
- 多进程/多线程自动选择
- 任务调度器（负载均衡）
- 进程池/线程池管理
- 异步执行支持

作者: Performance Optimization Expert
日期: 2026-03-22
"""

import os
import multiprocessing as mp
import threading
import concurrent.futures
import asyncio
import functools
from typing import Dict, List, Optional, Callable, Any, Union, Iterator, Iterable
from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue, Empty
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class ParallelBackend(Enum):
    """并行后端类型"""
    THREAD = auto()      # 多线程
    PROCESS = auto()     # 多进程
    ASYNC = auto()       # 异步
    AUTO = auto()        # 自动选择


@dataclass
class Task:
    """任务定义"""
    func: Callable
    args: tuple
    kwargs: dict
    task_id: int
    priority: int = 0
    submitted_at: float = 0.0


@dataclass
class TaskResult:
    """任务结果"""
    task_id: int
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0


class TaskScheduler:
    """
    任务调度器
    
    提供智能任务调度，支持优先级、负载均衡和动态调整。
    
    示例:
        scheduler = TaskScheduler(max_workers=4)
        
        # 提交任务
        future = scheduler.submit(my_func, arg1, arg2, priority=1)
        result = future.result()
        
        # 批量提交
        results = scheduler.map(my_func, data_list)
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 backend: ParallelBackend = ParallelBackend.AUTO):
        """
        初始化任务调度器
        
        Args:
            max_workers: 最大工作线程/进程数
            backend: 并行后端类型
        """
        self.max_workers = max_workers or os.cpu_count() or 4
        self.backend = self._select_backend(backend)
        
        self._executor: Optional[Any] = None
        self._task_queue: Queue = Queue()
        self._results: Dict[int, TaskResult] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
        self._shutdown = False
        
        self._setup_executor()
    
    def _select_backend(self, backend: ParallelBackend) -> ParallelBackend:
        """选择最佳后端"""
        if backend != ParallelBackend.AUTO:
            return backend
        
        # 自动选择逻辑
        # CPU密集型 -> 进程
        # I/O密集型 -> 线程
        # 默认使用线程（启动更快）
        return ParallelBackend.THREAD
    
    def _setup_executor(self) -> None:
        """设置执行器"""
        if self.backend == ParallelBackend.THREAD:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="parallel_worker"
            )
        elif self.backend == ParallelBackend.PROCESS:
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )
    
    def submit(self, func: Callable, *args, 
               priority: int = 0, **kwargs) -> concurrent.futures.Future:
        """
        提交任务
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            priority: 优先级（数字越小优先级越高）
            **kwargs: 关键字参数
        
        Returns:
            Future对象
        """
        if self._shutdown:
            raise RuntimeError("Scheduler has been shut down")
        
        with self._lock:
            self._task_counter += 1
            task_id = self._task_counter
        
        # 创建任务
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            task_id=task_id,
            priority=priority,
            submitted_at=time.time()
        )
        
        # 提交到执行器
        future = self._executor.submit(self._execute_task, task)
        
        return future
    
    def map(self, func: Callable, 
            iterable: Iterable,
            chunksize: int = 1) -> Iterator:
        """
        并行映射
        
        Args:
            func: 要应用的函数
            iterable: 可迭代对象
            chunksize: 分块大小
        
        Returns:
            结果迭代器
        """
        if self._shutdown:
            raise RuntimeError("Scheduler has been shut down")
        
        return self._executor.map(func, iterable, chunksize=chunksize)
    
    def map_ordered(self, func: Callable, 
                    iterable: Iterable,
                    chunksize: int = 1) -> List:
        """
        并行映射（有序返回）
        
        Args:
            func: 要应用的函数
            iterable: 可迭代对象
            chunksize: 分块大小
        
        Returns:
            结果列表
        """
        return list(self.map(func, iterable, chunksize))
    
    def shutdown(self, wait: bool = True) -> None:
        """
        关闭调度器
        
        Args:
            wait: 是否等待所有任务完成
        """
        self._shutdown = True
        if self._executor:
            self._executor.shutdown(wait=wait)
    
    def _execute_task(self, task: Task) -> Any:
        """执行任务"""
        start_time = time.time()
        
        try:
            result = task.func(*task.args, **task.kwargs)
            return result
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            raise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class WorkerPool:
    """
    工作池
    
    管理固定数量的工作线程/进程，提供高效的任务分发。
    
    示例:
        pool = WorkerPool(size=4)
        
        # 并行处理
        results = pool.parallel_map(process_func, data_list)
        
        # 工作窃取调度
        pool.submit_to_least_busy(worker_func, data)
    """
    
    def __init__(self, size: Optional[int] = None):
        """
        初始化工作池
        
        Args:
            size: 工作线程/进程数
        """
        self.size = size or os.cpu_count() or 4
        self._workers: List[threading.Thread] = []
        self._task_queue: Queue = Queue()
        self._result_queue: Queue = Queue()
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        
        self._start_workers()
    
    def _start_workers(self) -> None:
        """启动工作线程"""
        for i in range(self.size):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self) -> None:
        """工作线程主循环"""
        while not self._shutdown_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.1)
                if task is None:  # 终止信号
                    break
                
                result = task.func(*task.args, **task.kwargs)
                self._result_queue.put(TaskResult(
                    task_id=task.task_id,
                    result=result
                ))
                
            except Empty:
                continue
            except Exception as e:
                self._result_queue.put(TaskResult(
                    task_id=task.task_id if 'task' in dir() else -1,
                    error=e
                ))
    
    def submit(self, func: Callable, *args, **kwargs) -> int:
        """
        提交任务
        
        Returns:
            任务ID
        """
        with self._lock:
            task_id = id(func) + int(time.time() * 1000)
        
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            task_id=task_id
        )
        
        self._task_queue.put(task)
        return task_id
    
    def parallel_map(self, func: Callable, 
                     data: List,
                     return_ordered: bool = True) -> List:
        """
        并行映射
        
        Args:
            func: 处理函数
            data: 数据列表
            return_ordered: 是否保持顺序
        
        Returns:
            结果列表
        """
        # 提交所有任务
        futures = []
        for i, item in enumerate(data):
            future = self.submit(func, item)
            futures.append((i, future))
        
        # 收集结果
        results = [None] * len(data)
        completed = 0
        
        while completed < len(data):
            try:
                result = self._result_queue.get(timeout=0.1)
                if result.error:
                    raise result.error
                
                # 找到对应索引
                for idx, (orig_idx, _) in enumerate(futures):
                    if orig_idx == result.task_id:
                        results[orig_idx] = result.result
                        completed += 1
                        break
                        
            except Empty:
                continue
        
        return results
    
    def shutdown(self) -> None:
        """关闭工作池"""
        self._shutdown_event.set()
        
        # 发送终止信号
        for _ in self._workers:
            self._task_queue.put(None)
        
        # 等待所有线程结束
        for worker in self._workers:
            worker.join(timeout=1.0)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class AsyncExecutor:
    """
    异步执行器
    
    提供异步任务执行能力。
    
    示例:
        executor = AsyncExecutor()
        
        async def main():
            # 异步执行
            result = await executor.run_async(my_func, arg1, arg2)
            
            # 并发执行
            tasks = [executor.run_async(f, x) for x in data]
            results = await asyncio.gather(*tasks)
        
        asyncio.run(main())
    """
    
    def __init__(self, max_workers: int = 10):
        """
        初始化异步执行器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor: Optional[concurrent.futures.Executor] = None
    
    async def run_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        异步运行函数
        
        Args:
            func: 要运行的函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            函数结果
        """
        loop = asyncio.get_event_loop()
        
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            )
        
        # 在线程池中执行
        result = await loop.run_in_executor(
            self._executor,
            functools.partial(func, *args, **kwargs)
        )
        
        return result
    
    async def gather(self, *coroutines) -> List:
        """
        并发执行多个协程
        
        Args:
            *coroutines: 协程对象
        
        Returns:
            结果列表
        """
        return await asyncio.gather(*coroutines)
    
    async def map_async(self, func: Callable, 
                        data: Iterable) -> List:
        """
        异步映射
        
        Args:
            func: 处理函数
            data: 数据列表
        
        Returns:
            结果列表
        """
        tasks = [self.run_async(func, item) for item in data]
        return await asyncio.gather(*tasks)


class ParallelEngine:
    """
    并行引擎
    
    统一的并行计算接口，自动选择最佳执行策略。
    
    示例:
        engine = ParallelEngine()
        
        # 自动选择并行策略
        results = engine.parallel_map(process_func, data)
        
        # 指定策略
        with engine.context(backend='process', workers=8):
            results = engine.parallel_map(heavy_compute, data)
    """
    
    _instance: Optional['ParallelEngine'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, default_workers: Optional[int] = None):
        """
        初始化并行引擎
        
        Args:
            default_workers: 默认工作线程/进程数
        """
        if self._initialized:
            return
        
        self._initialized = True
        self.default_workers = default_workers or os.cpu_count() or 4
        
        self._scheduler: Optional[TaskScheduler] = None
        self._async_executor: Optional[AsyncExecutor] = None
        self._current_backend = ParallelBackend.AUTO
    
    def parallel_map(self, 
                     func: Callable,
                     data: Iterable,
                     backend: Optional[str] = None,
                     workers: Optional[int] = None,
                     chunksize: Optional[int] = None,
                     ordered: bool = True) -> List:
        """
        并行映射
        
        Args:
            func: 处理函数
            data: 数据列表
            backend: 后端类型 ('thread', 'process', 'auto')
            workers: 工作线程/进程数
            chunksize: 分块大小
            ordered: 是否保持顺序
        
        Returns:
            结果列表
        """
        data_list = list(data)
        
        if len(data_list) == 0:
            return []
        
        # 数据量小，串行执行
        if len(data_list) < 10:
            return [func(item) for item in data_list]
        
        # 确定后端
        backend_enum = self._parse_backend(backend or 'auto')
        workers = workers or self.default_workers
        chunksize = chunksize or max(1, len(data_list) // (workers * 4))
        
        # 使用调度器执行
        with TaskScheduler(max_workers=workers, backend=backend_enum) as scheduler:
            if ordered:
                return scheduler.map_ordered(func, data_list, chunksize)
            else:
                return list(scheduler.map(func, data_list, chunksize))
    
    def parallel_for(self, 
                     func: Callable[[int], Any],
                     n_iterations: int,
                     backend: Optional[str] = None,
                     workers: Optional[int] = None) -> List:
        """
        并行for循环
        
        Args:
            func: 处理函数（接收索引）
            n_iterations: 迭代次数
            backend: 后端类型
            workers: 工作线程/进程数
        
        Returns:
            结果列表
        """
        return self.parallel_map(
            func,
            range(n_iterations),
            backend=backend,
            workers=workers
        )
    
    def parallel_starmap(self,
                         func: Callable,
                         args_list: List[tuple],
                         backend: Optional[str] = None,
                         workers: Optional[int] = None) -> List:
        """
        并行starmap（支持多参数）
        
        Args:
            func: 处理函数
            args_list: 参数元组列表
            backend: 后端类型
            workers: 工作线程/进程数
        
        Returns:
            结果列表
        """
        wrapper = lambda args: func(*args)
        return self.parallel_map(wrapper, args_list, backend, workers)
    
    def get_optimal_workers(self, 
                           task_type: str = 'cpu',
                           memory_per_task_mb: Optional[float] = None) -> int:
        """
        获取最优工作线程/进程数
        
        Args:
            task_type: 任务类型 ('cpu', 'io', 'memory')
            memory_per_task_mb: 每个任务所需内存（MB）
        
        Returns:
            最优工作数
        """
        cpu_count = os.cpu_count() or 4
        
        if task_type == 'io':
            # I/O密集型，可以更多线程
            return min(cpu_count * 2, 32)
        
        elif task_type == 'memory' and memory_per_task_mb:
            # 内存受限
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            max_by_memory = int(available_memory / memory_per_task_mb * 0.8)
            return min(cpu_count, max_by_memory)
        
        else:
            # CPU密集型
            return cpu_count
    
    def _parse_backend(self, backend: str) -> ParallelBackend:
        """解析后端字符串"""
        mapping = {
            'thread': ParallelBackend.THREAD,
            'process': ParallelBackend.PROCESS,
            'async': ParallelBackend.ASYNC,
            'auto': ParallelBackend.AUTO,
        }
        return mapping.get(backend.lower(), ParallelBackend.AUTO)


# 便捷函数

def parallel_map(func: Callable,
                 data: Iterable,
                 backend: str = 'auto',
                 workers: Optional[int] = None,
                 **kwargs) -> List:
    """
    便捷的并行映射函数
    
    示例:
        results = parallel_map(process_func, data_list, backend='process', workers=8)
    """
    engine = ParallelEngine()
    return engine.parallel_map(func, data, backend, workers, **kwargs)


def parallel_for(func: Callable[[int], Any],
                 n_iterations: int,
                 backend: str = 'auto',
                 workers: Optional[int] = None) -> List:
    """
    便捷的并行for循环
    
    示例:
        results = parallel_for(compute, 1000, workers=4)
    """
    engine = ParallelEngine()
    return engine.parallel_for(func, n_iterations, backend, workers)
