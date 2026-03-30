#!/usr/bin/env python3
"""
Serverless Computing Module
无服务器计算模块 - 提供函数即服务、自动伸缩和按需计费

Features:
- Function as a Service (FaaS)
- Auto-scaling based on demand
- Pay-per-use billing
- Cold start optimization
- Multi-runtime support
- Event-driven architecture

Author: DFTLammps Platform
Version: 2.0.0
"""

import asyncio
import json
import time
import hashlib
import base64
import pickle
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Union, Set, Tuple
from enum import Enum, auto
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from datetime import datetime, timedelta
import random
import functools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FunctionRuntime(Enum):
    """函数运行时环境"""
    PYTHON = "python"
    NODEJS = "nodejs"
    GO = "go"
    RUST = "rust"
    JVM = "jvm"
    CUSTOM = "custom"


class FunctionState(Enum):
    """函数状态"""
    COLD = "cold"           # 未初始化
    INITIALIZING = "initializing"  # 初始化中
    WARM = "warm"           # 已预热，可立即执行
    RUNNING = "running"     # 执行中
    ERROR = "error"         # 错误状态
    TERMINATING = "terminating"  # 终止中


class ScalingPolicy(Enum):
    """扩缩容策略"""
    REQUEST_BASED = "request_based"      # 基于请求数
    CPU_BASED = "cpu_based"              # 基于CPU使用率
    CUSTOM_METRIC = "custom_metric"       # 基于自定义指标
    SCHEDULED = "scheduled"              # 基于时间表
    PREDICTIVE = "predictive"            # 预测性伸缩


class BillingGranularity(Enum):
    """计费粒度"""
    PER_MILLISECOND = 0.001
    PER_100MS = 0.1
    PER_SECOND = 1.0
    PER_MINUTE = 60.0


@dataclass
class FunctionResource:
    """函数资源配置"""
    memory_mb: int = 512
    cpu_cores: float = 1.0
    timeout_seconds: int = 300
    ephemeral_storage_mb: int = 512
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FunctionSpec:
    """函数规格定义"""
    function_id: str
    name: str
    runtime: FunctionRuntime
    handler: str
    code: Union[str, bytes]  # 代码或代码包路径
    requirements: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    resources: FunctionResource = field(default_factory=FunctionResource)
    layers: List[str] = field(default_factory=list)
    vpc_config: Optional[Dict] = None
    
    # 元数据
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.function_id:
            self.function_id = hashlib.md5(
                f"{self.name}-{time.time()}".encode()
            ).hexdigest()[:16]


@dataclass
class FunctionInstance:
    """函数实例 (执行环境)"""
    instance_id: str
    function_id: str
    state: FunctionState = FunctionState.COLD
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    current_task: Optional[str] = None
    
    # 运行时上下文
    runtime_context: Any = None
    
    async def initialize(self, spec: FunctionSpec):
        """初始化函数实例"""
        self.state = FunctionState.INITIALIZING
        logger.info(f"Initializing function instance {self.instance_id}")
        
        # 模拟初始化延迟
        await asyncio.sleep(0.5)
        
        self.state = FunctionState.WARM
        self.last_used = datetime.utcnow()
        logger.info(f"Function instance {self.instance_id} is warm")
    
    async def execute(self, event: Dict, context: Dict) -> Dict:
        """执行函数"""
        self.state = FunctionState.RUNNING
        self.current_task = context.get("request_id")
        
        start_time = time.time()
        
        try:
            # 实际实现会调用运行时执行代码
            await asyncio.sleep(random.uniform(0.01, 0.1))  # 模拟执行
            
            result = {
                "statusCode": 200,
                "body": json.dumps({"message": "Success", "input": event}),
                "headers": {"Content-Type": "application/json"}
            }
            
        except Exception as e:
            result = {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)}),
                "headers": {"Content-Type": "application/json"}
            }
            self.state = FunctionState.ERROR
        
        execution_time = (time.time() - start_time) * 1000
        self.execution_count += 1
        self.total_execution_time_ms += execution_time
        self.last_used = datetime.utcnow()
        self.state = FunctionState.WARM
        self.current_task = None
        
        return result
    
    async def terminate(self):
        """终止实例"""
        self.state = FunctionState.TERMINATING
        logger.info(f"Terminating function instance {self.instance_id}")
        await asyncio.sleep(0.1)


@dataclass
class ExecutionRequest:
    """执行请求"""
    request_id: str
    function_id: str
    event: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    
    # QoS参数
    priority: int = 5  # 1-10
    max_latency_ms: float = 1000.0
    cold_start_acceptable: bool = True
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = hashlib.md5(
                f"{self.function_id}-{time.time()}".encode()
            ).hexdigest()[:16]


@dataclass
class ExecutionResult:
    """执行结果"""
    request_id: str
    function_id: str
    instance_id: str
    status_code: int
    response: Any
    
    # 性能指标
    cold_start: bool = False
    initialization_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    billing_duration_ms: float = 0.0
    
    # 计费
    billed_memory_mb: int = 0
    cost_usd: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScalingConfig:
    """自动伸缩配置"""
    min_instances: int = 0
    max_instances: int = 100
    target_concurrency: float = 10.0  # 每个实例的目标并发数
    scale_up_threshold: float = 0.7   # CPU/内存使用率超过此值时扩容
    scale_down_threshold: float = 0.3  # CPU/内存使用率低于此值时缩容
    
    # 冷却时间
    scale_up_cooldown_seconds: int = 30
    scale_down_cooldown_seconds: int = 120
    
    # 预测性伸缩
    enable_predictive: bool = False
    prediction_window_minutes: int = 10
    
    # 定时伸缩
    scheduled_scaling: List[Dict] = field(default_factory=list)


@dataclass
class BillingRecord:
    """计费记录"""
    record_id: str
    function_id: str
    request_id: str
    instance_id: str
    
    # 计费项
    duration_ms: float
    memory_mb: int
    cpu_cores: float
    
    # 费用
    compute_cost: float
    request_cost: float
    total_cost: float
    
    # 元数据
    start_time: datetime
    end_time: datetime
    region: str = "us-east-1"


class BillingCalculator:
    """计费计算器"""
    
    # 定价 (USD)
    PRICING = {
        "compute_per_gb_second": 0.0000166667,  # $ per GB-second
        "request_per_million": 0.20,            # $ per million requests
        "provisioned_concurrency": 0.000004646,  # $ per GB-second
    }
    
    def __init__(self, granularity: BillingGranularity = BillingGranularity.PER_100MS):
        self.granularity = granularity
        self.records: List[BillingRecord] = []
    
    def calculate_cost(
        self,
        duration_ms: float,
        memory_mb: int,
        cpu_cores: float = 1.0,
        is_provisioned: bool = False
    ) -> Tuple[float, float]:
        """
        计算执行成本
        
        Returns: (compute_cost, request_cost)
        """
        # 按粒度向上取整
        duration_units = self._round_to_granularity(duration_ms / 1000)
        memory_gb = memory_mb / 1024
        
        # 计算费用
        price_per_unit = self.PRICING["provisioned_concurrency"] if is_provisioned else self.PRICING["compute_per_gb_second"]
        compute_cost = duration_units * memory_gb * price_per_unit
        
        # 请求费用 (按百万计算)
        request_cost = self.PRICING["request_per_million"] / 1000000
        
        return round(compute_cost, 10), round(request_cost, 10)
    
    def _round_to_granularity(self, duration_seconds: float) -> float:
        """按粒度向上取整"""
        granularity = self.granularity.value
        return max(granularity, ((duration_seconds // granularity) + 1) * granularity)
    
    def create_record(
        self,
        function_id: str,
        request_id: str,
        instance_id: str,
        duration_ms: float,
        memory_mb: int,
        cpu_cores: float,
        start_time: datetime
    ) -> BillingRecord:
        """创建计费记录"""
        compute_cost, request_cost = self.calculate_cost(duration_ms, memory_mb, cpu_cores)
        
        record = BillingRecord(
            record_id=hashlib.md5(f"{request_id}-{time.time()}".encode()).hexdigest()[:16],
            function_id=function_id,
            request_id=request_id,
            instance_id=instance_id,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            cpu_cores=cpu_cores,
            compute_cost=compute_cost,
            request_cost=request_cost,
            total_cost=round(compute_cost + request_cost, 10),
            start_time=start_time,
            end_time=datetime.utcnow()
        )
        
        self.records.append(record)
        return record
    
    def get_usage_report(
        self,
        function_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """获取使用报告"""
        # 过滤记录
        filtered = self.records
        
        if function_id:
            filtered = [r for r in filtered if r.function_id == function_id]
        
        if start_time:
            filtered = [r for r in filtered if r.start_time >= start_time]
        
        if end_time:
            filtered = [r for r in filtered if r.end_time <= end_time]
        
        # 聚合统计
        total_requests = len(filtered)
        total_duration_ms = sum(r.duration_ms for r in filtered)
        total_compute_cost = sum(r.compute_cost for r in filtered)
        total_request_cost = sum(r.request_cost for r in filtered)
        
        # 按函数分组
        by_function = defaultdict(lambda: {"requests": 0, "duration_ms": 0, "cost": 0})
        for r in filtered:
            by_function[r.function_id]["requests"] += 1
            by_function[r.function_id]["duration_ms"] += r.duration_ms
            by_function[r.function_id]["cost"] += r.total_cost
        
        return {
            "total_requests": total_requests,
            "total_duration_ms": round(total_duration_ms, 2),
            "total_duration_gb_seconds": round(total_duration_ms * sum(r.memory_mb for r in filtered) / 1024 / 1000, 4),
            "total_compute_cost": round(total_compute_cost, 6),
            "total_request_cost": round(total_request_cost, 6),
            "total_cost": round(total_compute_cost + total_request_cost, 6),
            "average_duration_ms": round(total_duration_ms / max(total_requests, 1), 2),
            "by_function": dict(by_function)
        }


class AutoScaler:
    """自动伸缩器"""
    
    def __init__(self):
        self.configs: Dict[str, ScalingConfig] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_scale_up: Dict[str, datetime] = {}
        self.last_scale_down: Dict[str, datetime] = {}
        
        # 预测模型 (简化版)
        self.prediction_windows: Dict[str, List[float]] = defaultdict(list)
    
    def configure(self, function_id: str, config: ScalingConfig):
        """配置自动伸缩"""
        self.configs[function_id] = config
        logger.info(f"Configured auto-scaling for {function_id}")
    
    def record_metric(self, function_id: str, metric_name: str, value: float):
        """记录指标"""
        self.metrics_history[f"{function_id}:{metric_name}"].append({
            "timestamp": datetime.utcnow(),
            "value": value
        })
    
    async def evaluate_scaling(
        self,
        function_id: str,
        current_instances: int,
        pending_requests: int,
        avg_cpu_usage: float,
        avg_memory_usage: float
    ) -> Tuple[int, str]:
        """
        评估是否需要扩缩容
        
        Returns: (目标实例数, 原因)
        """
        config = self.configs.get(function_id)
        if not config:
            return current_instances, "no_config"
        
        now = datetime.utcnow()
        
        # 检查冷却时间
        if function_id in self.last_scale_up:
            elapsed = (now - self.last_scale_up[function_id]).total_seconds()
            if elapsed < config.scale_up_cooldown_seconds:
                return current_instances, "cooldown"
        
        if function_id in self.last_scale_down:
            elapsed = (now - self.last_scale_down[function_id]).total_seconds()
            if elapsed < config.scale_down_cooldown_seconds:
                return current_instances, "cooldown"
        
        # 计算当前并发
        current_concurrency = pending_requests / max(current_instances, 1)
        
        # 计算所需实例数
        needed_instances = max(
            config.min_instances,
            min(
                config.max_instances,
                int(pending_requests / config.target_concurrency) + 1
            )
        )
        
        # 基于资源使用率调整
        if avg_cpu_usage > config.scale_up_threshold or avg_memory_usage > config.scale_up_threshold:
            if current_instances < config.max_instances:
                self.last_scale_up[function_id] = now
                return min(current_instances + 1, config.max_instances), "high_resource_usage"
        
        if avg_cpu_usage < config.scale_down_threshold and avg_memory_usage < config.scale_down_threshold:
            if current_instances > config.min_instances and pending_requests == 0:
                self.last_scale_down[function_id] = now
                return max(current_instances - 1, config.min_instances), "low_resource_usage"
        
        # 基于并发调整
        if current_concurrency > config.target_concurrency * 1.5:
            if current_instances < config.max_instances:
                self.last_scale_up[function_id] = now
                return min(current_instances + 2, config.max_instances), "high_concurrency"
        
        if current_concurrency < config.target_concurrency * 0.5 and pending_requests == 0:
            if current_instances > config.min_instances:
                self.last_scale_down[function_id] = now
                return max(current_instances - 1, config.min_instances), "low_concurrency"
        
        return current_instances, "stable"
    
    def predict_load(self, function_id: str, minutes_ahead: int = 10) -> float:
        """预测未来负载 (简化版移动平均)"""
        history = self.metrics_history.get(f"{function_id}:requests_per_second", deque())
        
        if len(history) < 10:
            return 0.0
        
        # 计算移动平均
        recent_values = [h["value"] for h in list(history)[-30:]]
        avg = sum(recent_values) / len(recent_values)
        
        # 简单趋势预测
        if len(recent_values) >= 10:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
            predicted = avg + trend * minutes_ahead * 6  # 6 samples per minute
            return max(0, predicted)
        
        return avg


class FunctionRegistry:
    """函数注册表"""
    
    def __init__(self):
        self.functions: Dict[str, FunctionSpec] = {}
        self.instances: Dict[str, Dict[str, FunctionInstance]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def register(self, spec: FunctionSpec) -> bool:
        """注册函数"""
        async with self._lock:
            self.functions[spec.function_id] = spec
            logger.info(f"Registered function: {spec.name} ({spec.function_id})")
            return True
    
    async def unregister(self, function_id: str) -> bool:
        """注销函数"""
        async with self._lock:
            if function_id in self.functions:
                # 终止所有实例
                for instance_id, instance in self.instances.get(function_id, {}).items():
                    await instance.terminate()
                
                del self.functions[function_id]
                if function_id in self.instances:
                    del self.instances[function_id]
                
                logger.info(f"Unregistered function: {function_id}")
                return True
            return False
    
    def get_function(self, function_id: str) -> Optional[FunctionSpec]:
        """获取函数规格"""
        return self.functions.get(function_id)
    
    async def get_or_create_instance(
        self,
        function_id: str,
        warm_start: bool = True
    ) -> Tuple[FunctionInstance, bool]:
        """
        获取或创建函数实例
        
        Returns: (实例, 是否冷启动)
        """
        async with self._lock:
            spec = self.functions.get(function_id)
            if not spec:
                raise ValueError(f"Function {function_id} not found")
            
            instances = self.instances.get(function_id, {})
            
            # 查找可用的warm实例
            if warm_start:
                for instance_id, instance in instances.items():
                    if instance.state == FunctionState.WARM:
                        return instance, False
            
            # 创建新实例
            instance_id = f"{function_id}-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            instance = FunctionInstance(instance_id=instance_id, function_id=function_id)
            
            # 初始化
            await instance.initialize(spec)
            
            self.instances[function_id][instance_id] = instance
            
            return instance, True
    
    async def release_instance(self, function_id: str, instance_id: str):
        """释放实例 (返回warm池)"""
        async with self._lock:
            instance = self.instances.get(function_id, {}).get(instance_id)
            if instance:
                instance.state = FunctionState.WARM
                instance.current_task = None
    
    async def terminate_idle_instances(
        self,
        function_id: Optional[str] = None,
        idle_timeout_seconds: int = 600
    ) -> int:
        """终止空闲实例"""
        terminated = 0
        now = datetime.utcnow()
        
        async with self._lock:
            functions_to_check = [function_id] if function_id else list(self.instances.keys())
            
            for fid in functions_to_check:
                instances = self.instances.get(fid, {})
                to_terminate = []
                
                for instance_id, instance in instances.items():
                    if instance.state == FunctionState.WARM:
                        idle_time = (now - instance.last_used).total_seconds()
                        if idle_time > idle_timeout_seconds:
                            to_terminate.append(instance_id)
                
                for instance_id in to_terminate:
                    instance = instances[instance_id]
                    await instance.terminate()
                    del instances[instance_id]
                    terminated += 1
        
        return terminated
    
    def get_instance_stats(self, function_id: str) -> Dict:
        """获取实例统计信息"""
        instances = self.instances.get(function_id, {})
        
        state_counts = defaultdict(int)
        total_executions = 0
        
        for instance in instances.values():
            state_counts[instance.state.value] += 1
            total_executions += instance.execution_count
        
        return {
            "total_instances": len(instances),
            "state_distribution": dict(state_counts),
            "total_executions": total_executions
        }


class ServerlessOrchestrator:
    """无服务器编排器"""
    
    def __init__(self):
        self.registry = FunctionRegistry()
        self.auto_scaler = AutoScaler()
        self.billing = BillingCalculator()
        
        # 执行队列
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.pending_requests: Dict[str, ExecutionRequest] = {}
        
        # 运行时统计
        self.execution_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "error": 0,
            "cold_start_count": 0,
            "total_latency_ms": 0
        })
        
        # 后台任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """启动编排器"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Serverless orchestrator started")
    
    async def stop(self):
        """停止编排器"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._scaling_task:
            self._scaling_task.cancel()
        
        logger.info("Serverless orchestrator stopped")
    
    async def _cleanup_loop(self):
        """清理循环 - 终止空闲实例"""
        while self._running:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                terminated = await self.registry.terminate_idle_instances(
                    idle_timeout_seconds=300  # 5分钟空闲超时
                )
                if terminated > 0:
                    logger.info(f"Terminated {terminated} idle instances")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _scaling_loop(self):
        """伸缩循环"""
        while self._running:
            try:
                await asyncio.sleep(10)  # 每10秒评估一次
                
                for function_id in self.registry.functions.keys():
                    stats = self.registry.get_instance_stats(function_id)
                    current_instances = stats["total_instances"]
                    
                    # 获取待处理请求数
                    pending = len([
                        r for r in self.pending_requests.values()
                        if r.function_id == function_id
                    ])
                    
                    # 评估伸缩
                    target, reason = await self.auto_scaler.evaluate_scaling(
                        function_id,
                        current_instances,
                        pending,
                        0.5,  # 模拟CPU使用率
                        0.5   # 模拟内存使用率
                    )
                    
                    if target != current_instances:
                        logger.info(
                            f"Scaling {function_id}: {current_instances} -> {target} ({reason})"
                        )
                        
                        if target > current_instances:
                            # 预热实例
                            for _ in range(target - current_instances):
                                await self.registry.get_or_create_instance(
                                    function_id, warm_start=False
                                )
                        # 缩容由 cleanup_loop 处理
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling error: {e}")
    
    async def deploy_function(self, spec: FunctionSpec) -> str:
        """
        部署函数
        
        Returns: 函数ID
        """
        await self.registry.register(spec)
        
        # 默认伸缩配置
        self.auto_scaler.configure(spec.function_id, ScalingConfig())
        
        return spec.function_id
    
    async def invoke(
        self,
        function_id: str,
        event: Dict,
        headers: Optional[Dict] = None,
        priority: int = 5
    ) -> ExecutionResult:
        """
        调用函数
        
        完整的执行流程：
        1. 获取/创建函数实例
        2. 执行函数
        3. 记录计费
        4. 释放实例
        """
        start_time = datetime.utcnow()
        
        # 创建执行请求
        request = ExecutionRequest(
            request_id="",
            function_id=function_id,
            event=event,
            headers=headers or {},
            priority=priority
        )
        
        self.pending_requests[request.request_id] = request
        
        try:
            # 获取函数规格
            spec = self.registry.get_function(function_id)
            if not spec:
                raise ValueError(f"Function {function_id} not found")
            
            # 获取或创建实例
            instance, is_cold_start = await self.registry.get_or_create_instance(
                function_id,
                warm_start=not request.cold_start_acceptable
            )
            
            init_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # 执行
            exec_start = time.time()
            response = await instance.execute(event, {"request_id": request.request_id})
            exec_time_ms = (time.time() - exec_start) * 1000
            
            # 计费
            billing_record = self.billing.create_record(
                function_id=function_id,
                request_id=request.request_id,
                instance_id=instance.instance_id,
                duration_ms=exec_time_ms,
                memory_mb=spec.resources.memory_mb,
                cpu_cores=spec.resources.cpu_cores,
                start_time=start_time
            )
            
            # 释放实例
            await self.registry.release_instance(function_id, instance.instance_id)
            
            # 更新统计
            self.execution_stats[function_id]["total"] += 1
            if response.get("statusCode", 500) < 400:
                self.execution_stats[function_id]["success"] += 1
            else:
                self.execution_stats[function_id]["error"] += 1
            
            if is_cold_start:
                self.execution_stats[function_id]["cold_start_count"] += 1
            
            self.execution_stats[function_id]["total_latency_ms"] += exec_time_ms
            
            # 记录指标用于伸缩
            self.auto_scaler.record_metric(
                function_id, "execution_time_ms", exec_time_ms
            )
            
            return ExecutionResult(
                request_id=request.request_id,
                function_id=function_id,
                instance_id=instance.instance_id,
                status_code=response.get("statusCode", 200),
                response=response,
                cold_start=is_cold_start,
                initialization_time_ms=init_time_ms if is_cold_start else 0,
                execution_time_ms=exec_time_ms,
                billing_duration_ms=billing_record.duration_ms,
                billed_memory_mb=spec.resources.memory_mb,
                cost_usd=billing_record.total_cost
            )
            
        finally:
            del self.pending_requests[request.request_id]
    
    async def invoke_async(
        self,
        function_id: str,
        event: Dict,
        headers: Optional[Dict] = None
    ) -> str:
        """异步调用函数"""
        # 简化实现：直接同步调用
        # 实际实现会返回 invocation ID，结果通过回调/S3等异步返回
        result = await self.invoke(function_id, event, headers)
        return result.request_id
    
    def get_function_metrics(self, function_id: str) -> Dict:
        """获取函数指标"""
        stats = self.execution_stats[function_id]
        total = stats["total"]
        
        instance_stats = self.registry.get_instance_stats(function_id)
        billing_report = self.billing.get_usage_report(function_id)
        
        return {
            "function_id": function_id,
            "executions": {
                "total": total,
                "success": stats["success"],
                "error": stats["error"],
                "success_rate": round(stats["success"] / max(total, 1), 4)
            },
            "cold_start": {
                "count": stats["cold_start_count"],
                "rate": round(stats["cold_start_count"] / max(total, 1), 4)
            },
            "latency": {
                "avg_ms": round(stats["total_latency_ms"] / max(total, 1), 2)
            },
            "instances": instance_stats,
            "billing": billing_report
        }
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """获取所有函数指标"""
        return {
            fid: self.get_function_metrics(fid)
            for fid in self.registry.functions.keys()
        }
    
    async def provision_concurrency(
        self,
        function_id: str,
        concurrent_executions: int
    ) -> bool:
        """预置并发 - 预热指定数量的实例"""
        logger.info(f"Provisioning {concurrent_executions} concurrent instances for {function_id}")
        
        for i in range(concurrent_executions):
            await self.registry.get_or_create_instance(function_id, warm_start=False)
        
        return True
    
    async def delete_function(self, function_id: str) -> bool:
        """删除函数"""
        return await self.registry.unregister(function_id)


# 便捷装饰器
def faas_function(
    runtime: FunctionRuntime = FunctionRuntime.PYTHON,
    memory_mb: int = 512,
    timeout_seconds: int = 300,
    **kwargs
):
    """函数装饰器 - 将Python函数转换为FaaS函数"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **func_kwargs):
            return await func(*args, **func_kwargs)
        
        # 附加元数据
        wrapper._faas_config = {
            "runtime": runtime,
            "memory_mb": memory_mb,
            "timeout_seconds": timeout_seconds,
            "handler": func.__name__,
            **kwargs
        }
        wrapper._faas_func = func
        
        return wrapper
    return decorator


# 示例使用
async def demo():
    """无服务器计算演示"""
    
    # 创建编排器
    orchestrator = ServerlessOrchestrator()
    await orchestrator.start()
    
    try:
        # 定义函数规格
        functions = [
            FunctionSpec(
                function_id="",
                name="molecule-predict",
                runtime=FunctionRuntime.PYTHON,
                handler="predict.handler",
                code="import json; def handler(event, context): return {'prediction': 0.95}",
                resources=FunctionResource(memory_mb=1024, cpu_cores=1.0, timeout_seconds=30),
                environment={"MODEL_PATH": "/models/v1"},
                description="Predict molecular properties"
            ),
            FunctionSpec(
                function_id="",
                name="structure-optimize",
                runtime=FunctionRuntime.PYTHON,
                handler="optimize.handler",
                code="import json; def handler(event, context): return {'optimized': True}",
                resources=FunctionResource(memory_mb=2048, cpu_cores=2.0, timeout_seconds=60),
                description="Optimize molecular structure"
            ),
            FunctionSpec(
                function_id="",
                name="data-preprocess",
                runtime=FunctionRuntime.PYTHON,
                handler="preprocess.handler",
                code="import json; def handler(event, context): return {'processed': len(event.get('data', []))}",
                resources=FunctionResource(memory_mb=512, cpu_cores=0.5, timeout_seconds=10),
                description="Preprocess simulation data"
            ),
        ]
        
        deployed_functions = []
        
        # 部署函数
        print("=== 部署函数 ===")
        for spec in functions:
            fid = await orchestrator.deploy_function(spec)
            deployed_functions.append(fid)
            print(f"  Deployed: {spec.name} ({fid[:8]}...)")
            print(f"    Memory: {spec.resources.memory_mb}MB, Timeout: {spec.resources.timeout_seconds}s")
        
        # 预置并发
        print("\n=== 预置并发 ===")
        await orchestrator.provision_concurrency(deployed_functions[0], 3)
        print(f"  Provisioned 3 warm instances for {deployed_functions[0][:8]}...")
        
        # 调用函数
        print("\n=== 函数调用测试 ===")
        
        # 第一次调用 (warm start)
        result1 = await orchestrator.invoke(
            deployed_functions[0],
            {"smiles": "CCO", "properties": ["energy", "dipole"]},
            priority=8
        )
        print(f"  Invoke 1: cold_start={result1.cold_start}, "
              f"latency={result1.execution_time_ms:.2f}ms, "
              f"cost=${result1.cost_usd:.8f}")
        
        # 第二次调用 (应该 warm)
        result2 = await orchestrator.invoke(
            deployed_functions[0],
            {"smiles": "CCCO", "properties": ["energy"]},
            priority=5
        )
        print(f"  Invoke 2: cold_start={result2.cold_start}, "
              f"latency={result2.execution_time_ms:.2f}ms, "
              f"cost=${result2.cost_usd:.8f}")
        
        # 批量调用测试
        print("\n=== 批量调用测试 ===")
        tasks = [
            orchestrator.invoke(
                deployed_functions[1],
                {"structure": f"molecule_{i}", "iterations": 100}
            )
            for i in range(10)
        ]
        
        batch_results = await asyncio.gather(*tasks)
        
        total_cost = sum(r.cost_usd for r in batch_results)
        cold_starts = sum(1 for r in batch_results if r.cold_start)
        avg_latency = sum(r.execution_time_ms for r in batch_results) / len(batch_results)
        
        print(f"  10 invocations: avg_latency={avg_latency:.2f}ms, "
              f"cold_starts={cold_starts}, total_cost=${total_cost:.8f}")
        
        # 等待计费记录累积
        await asyncio.sleep(1)
        
        # 查看指标
        print("\n=== 函数指标 ===")
        for fid in deployed_functions[:2]:
            metrics = orchestrator.get_function_metrics(fid)
            print(f"  {fid[:8]}...:")
            print(f"    Executions: {metrics['executions']['total']} "
                  f"(success: {metrics['executions']['success']})")
            print(f"    Cold start rate: {metrics['cold_start']['rate']:.1%}")
            print(f"    Avg latency: {metrics['latency']['avg_ms']:.2f}ms")
            print(f"    Total cost: ${metrics['billing']['total_cost']:.8f}")
        
        # 配置自动伸缩
        print("\n=== 自动伸缩配置 ===")
        scaling_config = ScalingConfig(
            min_instances=1,
            max_instances=50,
            target_concurrency=10,
            scale_up_threshold=0.7,
            scale_down_threshold=0.3
        )
        orchestrator.auto_scaler.configure(deployed_functions[0], scaling_config)
        print(f"  Configured auto-scaling for {deployed_functions[0][:8]}...")
        print(f"    Min: {scaling_config.min_instances}, Max: {scaling_config.max_instances}")
        print(f"    Target concurrency: {scaling_config.target_concurrency}")
        
        # 计费报告
        print("\n=== 计费报告 ===")
        all_billing = orchestrator.billing.get_usage_report()
        print(f"  Total requests: {all_billing['total_requests']}")
        print(f"  Total duration: {all_billing['total_duration_gb_seconds']:.4f} GB-seconds")
        print(f"  Total cost: ${all_billing['total_cost']:.8f}")
        
        # 按函数查看
        print("\n  By function:")
        for fid, data in all_billing['by_function'].items():
            print(f"    {fid[:8]}...: {data['requests']} requests, ${data['cost']:.8f}")
        
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(demo())
