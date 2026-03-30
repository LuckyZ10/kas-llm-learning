#!/usr/bin/env python3
"""
profiler.py
===========
性能分析器模块

提供全面的性能分析功能：
- 函数级性能分析（装饰器方式）
- 代码块性能分析（上下文管理器）
- 瓶颈自动识别
- 火焰图数据生成
- 统计报告生成

作者: Performance Optimization Expert
日期: 2026-03-22
"""

import time
import functools
import inspect
import sys
import threading
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
import json
import statistics
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FunctionStats:
    """函数统计信息"""
    name: str
    module: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    @property
    def avg_time(self) -> float:
        """平均执行时间"""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    @property
    def std_time(self) -> float:
        """执行时间标准差"""
        if len(self.times) < 2:
            return 0.0
        return statistics.stdev(self.times)
    
    @property
    def p50_time(self) -> float:
        """中位数执行时间"""
        if not self.times:
            return 0.0
        return np.percentile(list(self.times), 50)
    
    @property
    def p95_time(self) -> float:
        """95分位执行时间"""
        if not self.times:
            return 0.0
        return np.percentile(list(self.times), 95)
    
    @property
    def p99_time(self) -> float:
        """99分位执行时间"""
        if not self.times:
            return 0.0
        return np.percentile(list(self.times), 99)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "module": self.module,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "std_time": self.std_time,
            "p50_time": self.p50_time,
            "p95_time": self.p95_time,
            "p99_time": self.p99_time,
        }


@dataclass
class CallFrame:
    """调用帧信息"""
    function_name: str
    module_name: str
    filename: str
    line_number: int
    start_time: float
    end_time: Optional[float] = None
    children: List['CallFrame'] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """执行持续时间"""
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


class Profiler:
    """
    性能分析器
    
    提供全面的性能分析功能，包括函数级和代码块级分析。
    
    示例:
        profiler = Profiler()
        
        # 方式1: 装饰器
        @profiler.profile
        def my_function():
            pass
        
        # 方式2: 上下文管理器
        with profiler.measure("block_name"):
            # 代码块
            pass
        
        # 生成报告
        profiler.generate_report()
    """
    
    _instance: Optional['Profiler'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, enabled: bool = True, max_stats: int = 10000):
        """
        初始化分析器
        
        Args:
            enabled: 是否启用分析
            max_stats: 最大统计记录数
        """
        if self._initialized:
            return
        
        self._initialized = True
        self.enabled = enabled
        self.max_stats = max_stats
        
        # 统计数据
        self._stats: Dict[str, FunctionStats] = {}
        self._call_stack: List[CallFrame] = []
        self._call_tree: List[CallFrame] = []
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 配置
        self._track_memory = False
        self._track_gc = False
        
        logger.debug("Profiler initialized")
    
    def enable(self) -> None:
        """启用分析"""
        self.enabled = True
        logger.info("Profiler enabled")
    
    def disable(self) -> None:
        """禁用分析"""
        self.enabled = False
        logger.info("Profiler disabled")
    
    def reset(self) -> None:
        """重置所有统计数据"""
        with self._lock:
            self._stats.clear()
            self._call_stack.clear()
            self._call_tree.clear()
        logger.debug("Profiler stats reset")
    
    def profile(self, func: Optional[Callable] = None, *, 
                name: Optional[str] = None,
                track_memory: bool = False) -> Callable:
        """
        性能分析装饰器
        
        Args:
            func: 要分析的函数
            name: 自定义名称（可选）
            track_memory: 是否追踪内存使用
        
        Returns:
            装饰后的函数
        """
        def decorator(f: Callable) -> Callable:
            func_name = name or f.__name__
            module_name = f.__module__
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return f(*args, **kwargs)
                
                # 获取函数信息
                try:
                    filename = inspect.getfile(f)
                    line_no = inspect.getsourcelines(f)[1]
                except (OSError, TypeError):
                    filename = "unknown"
                    line_no = 0
                
                # 创建调用帧
                frame = CallFrame(
                    function_name=func_name,
                    module_name=module_name,
                    filename=filename,
                    line_number=line_no,
                    start_time=time.perf_counter()
                )
                
                # 记录调用栈
                with self._lock:
                    if self._call_stack:
                        self._call_stack[-1].children.append(frame)
                    else:
                        self._call_tree.append(frame)
                    self._call_stack.append(frame)
                
                # 记录内存（可选）
                mem_before = 0
                if track_memory:
                    mem_before = self._get_memory_usage()
                
                try:
                    start = time.perf_counter()
                    result = f(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    
                    # 更新统计
                    with self._lock:
                        key = f"{module_name}.{func_name}"
                        if key not in self._stats:
                            self._stats[key] = FunctionStats(
                                name=func_name,
                                module=module_name
                            )
                        
                        stat = self._stats[key]
                        stat.call_count += 1
                        stat.total_time += elapsed
                        stat.min_time = min(stat.min_time, elapsed)
                        stat.max_time = max(stat.max_time, elapsed)
                        stat.times.append(elapsed)
                        
                        # 更新调用帧
                        frame.end_time = time.perf_counter()
                        self._call_stack.pop()
                    
                    return result
                    
                except Exception as e:
                    with self._lock:
                        if self._call_stack and self._call_stack[-1] == frame:
                            frame.end_time = time.perf_counter()
                            self._call_stack.pop()
                    raise
            
            return wrapper
        
        if func is not None:
            return decorator(func)
        return decorator
    
    @contextmanager
    def measure(self, name: str, track_memory: bool = False):
        """
        代码块性能测量上下文管理器
        
        Args:
            name: 代码块名称
            track_memory: 是否追踪内存
        
        示例:
            with profiler.measure("data_loading"):
                data = load_large_dataset()
        """
        if not self.enabled:
            yield
            return
        
        start = time.perf_counter()
        mem_before = self._get_memory_usage() if track_memory else 0
        
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            mem_after = self._get_memory_usage() if track_memory else 0
            
            with self._lock:
                key = f"block:{name}"
                if key not in self._stats:
                    self._stats[key] = FunctionStats(
                        name=name,
                        module="__block__"
                    )
                
                stat = self._stats[key]
                stat.call_count += 1
                stat.total_time += elapsed
                stat.min_time = min(stat.min_time, elapsed)
                stat.max_time = max(stat.max_time, elapsed)
                stat.times.append(elapsed)
    
    def get_stats(self, name: Optional[str] = None) -> Union[FunctionStats, Dict[str, FunctionStats]]:
        """
        获取统计信息
        
        Args:
            name: 函数名称（None则返回所有）
        
        Returns:
            统计信息
        """
        with self._lock:
            if name is not None:
                return self._stats.get(name)
            return dict(self._stats)
    
    def get_slowest_functions(self, n: int = 10) -> List[FunctionStats]:
        """
        获取最慢的函数
        
        Args:
            n: 返回数量
        
        Returns:
            按总时间排序的函数列表
        """
        with self._lock:
            sorted_stats = sorted(
                self._stats.values(),
                key=lambda x: x.total_time,
                reverse=True
            )
        return sorted_stats[:n]
    
    def get_hotspots(self, n: int = 10) -> List[FunctionStats]:
        """
        获取热点函数（调用次数最多）
        
        Args:
            n: 返回数量
        
        Returns:
            按调用次数排序的函数列表
        """
        with self._lock:
            sorted_stats = sorted(
                self._stats.values(),
                key=lambda x: x.call_count,
                reverse=True
            )
        return sorted_stats[:n]
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        生成性能报告
        
        Args:
            output_path: 输出文件路径（None则返回字符串）
        
        Returns:
            报告内容
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE PROFILE REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # 概览
        with self._lock:
            total_calls = sum(s.call_count for s in self._stats.values())
            total_time = sum(s.total_time for s in self._stats.values())
            
        lines.append(f"Total Functions Profiled: {len(self._stats)}")
        lines.append(f"Total Calls: {total_calls}")
        lines.append(f"Total Time: {total_time:.4f}s")
        lines.append("")
        
        # 最慢函数
        lines.append("-" * 80)
        lines.append("TOP 10 SLOWEST FUNCTIONS (by total time)")
        lines.append("-" * 80)
        lines.append(f"{'Function':<40} {'Calls':<10} {'Total(s)':<12} {'Avg(ms)':<10} {'Max(ms)':<10}")
        lines.append("-" * 80)
        
        for stat in self.get_slowest_functions(10):
            func_name = f"{stat.module}.{stat.name}"
            if len(func_name) > 38:
                func_name = "..." + func_name[-35:]
            lines.append(
                f"{func_name:<40} {stat.call_count:<10} "
                f"{stat.total_time:<12.4f} {stat.avg_time*1000:<10.2f} "
                f"{stat.max_time*1000:<10.2f}"
            )
        
        lines.append("")
        
        # 热点函数
        lines.append("-" * 80)
        lines.append("TOP 10 HOTSPOTS (by call count)")
        lines.append("-" * 80)
        lines.append(f"{'Function':<40} {'Calls':<10} {'Avg(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10}")
        lines.append("-" * 80)
        
        for stat in self.get_hotspots(10):
            func_name = f"{stat.module}.{stat.name}"
            if len(func_name) > 38:
                func_name = "..." + func_name[-35:]
            lines.append(
                f"{func_name:<40} {stat.call_count:<10} "
                f"{stat.avg_time*1000:<10.2f} {stat.p95_time*1000:<10.2f} "
                f"{stat.p99_time*1000:<10.2f}"
            )
        
        lines.append("")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def export_json(self, output_path: Path) -> None:
        """
        导出统计数据为JSON
        
        Args:
            output_path: 输出路径
        """
        with self._lock:
            data = {
                "timestamp": time.time(),
                "stats": {k: v.to_dict() for k, v in self._stats.items()},
                "call_tree": self._serialize_call_tree()
            }
        
        output_path = Path(output_path)
        output_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Stats exported to {output_path}")
    
    def _serialize_call_tree(self) -> List[Dict]:
        """序列化调用树"""
        def serialize_frame(frame: CallFrame) -> Dict:
            return {
                "function": frame.function_name,
                "module": frame.module_name,
                "filename": frame.filename,
                "line": frame.line_number,
                "duration": frame.duration,
                "children": [serialize_frame(c) for c in frame.children]
            }
        
        return [serialize_frame(f) for f in self._call_tree]
    
    def _get_memory_usage(self) -> int:
        """获取当前内存使用量（字节）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0


class BottleneckAnalyzer:
    """
    瓶颈分析器
    
    自动识别性能瓶颈并提供优化建议。
    """
    
    def __init__(self, profiler: Optional[Profiler] = None):
        """
        初始化瓶颈分析器
        
        Args:
            profiler: 性能分析器实例
        """
        self.profiler = profiler or Profiler()
    
    def analyze(self) -> Dict[str, Any]:
        """
        分析性能瓶颈
        
        Returns:
            瓶颈分析报告
        """
        stats = self.profiler.get_stats()
        
        if not stats:
            return {"status": "no_data", "bottlenecks": []}
        
        bottlenecks = []
        
        # 1. 高耗时瓶颈
        total_time = sum(s.total_time for s in stats.values())
        for stat in stats.values():
            percentage = (stat.total_time / total_time) * 100 if total_time > 0 else 0
            if percentage > 20:
                bottlenecks.append({
                    "type": "high_time_consumption",
                    "function": f"{stat.module}.{stat.name}",
                    "severity": "high" if percentage > 50 else "medium",
                    "percentage": percentage,
                    "total_time": stat.total_time,
                    "suggestion": self._get_time_suggestion(stat)
                })
        
        # 2. 高频调用瓶颈
        for stat in stats.values():
            if stat.call_count > 10000:
                bottlenecks.append({
                    "type": "high_call_frequency",
                    "function": f"{stat.module}.{stat.name}",
                    "severity": "high" if stat.call_count > 100000 else "medium",
                    "call_count": stat.call_count,
                    "suggestion": self._get_frequency_suggestion(stat)
                })
        
        # 3. 高方差瓶颈（不稳定）
        for stat in stats.values():
            if stat.call_count >= 10 and stat.std_time > stat.avg_time * 0.5:
                bottlenecks.append({
                    "type": "high_variance",
                    "function": f"{stat.module}.{stat.name}",
                    "severity": "medium",
                    "cv": stat.std_time / stat.avg_time if stat.avg_time > 0 else 0,
                    "suggestion": "Consider investigating variable input sizes or external dependencies"
                })
        
        # 排序
        severity_order = {"high": 0, "medium": 1, "low": 2}
        bottlenecks.sort(key=lambda x: severity_order.get(x["severity"], 3))
        
        return {
            "status": "analyzed",
            "total_bottlenecks": len(bottlenecks),
            "bottlenecks": bottlenecks
        }
    
    def _get_time_suggestion(self, stat: FunctionStats) -> str:
        """获取时间优化建议"""
        suggestions = []
        
        if stat.avg_time > 0.1:  # > 100ms
            suggestions.append("Consider Numba JIT compilation")
        
        if "loop" in stat.name.lower() or "iterate" in stat.name.lower():
            suggestions.append("Check for vectorization opportunities with NumPy")
        
        if stat.call_count > 1000:
            suggestions.append("Consider caching/memoization")
        
        return "; ".join(suggestions) if suggestions else "Review algorithm complexity"
    
    def _get_frequency_suggestion(self, stat: FunctionStats) -> str:
        """获取频率优化建议"""
        if stat.avg_time < 0.001:  # < 1ms
            return "Function is lightweight, consider inlining or batching calls"
        else:
            return "High call frequency with non-trivial cost, consider caching or algorithm optimization"


# 便捷函数

def profile_function(func: Optional[Callable] = None, **kwargs) -> Callable:
    """
    便捷的函数分析装饰器
    
    使用全局Profiler实例。
    
    示例:
        @profile_function
        def my_func():
            pass
        
        @profile_function(name="custom_name")
        def another_func():
            pass
    """
    profiler = Profiler()
    return profiler.profile(func, **kwargs)


@contextmanager
def PerformanceContext(name: str, **kwargs):
    """
    便捷的代码块分析上下文管理器
    
    使用全局Profiler实例。
    
    示例:
        with PerformanceContext("data_loading"):
            data = load_data()
    """
    profiler = Profiler()
    with profiler.measure(name, **kwargs):
        yield
