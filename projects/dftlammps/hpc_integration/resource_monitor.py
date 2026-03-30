#!/usr/bin/env python3
"""
resource_monitor.py
===================
计算资源监控模块

监控功能：
- 集群整体状态
- 队列统计信息
- 节点健康状态
- GPU利用率
- 作业资源使用
- 实时告警
"""

import os
import re
import json
import time
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
import subprocess

from .cluster_connector import ClusterConnector

logger = logging.getLogger(__name__)


@dataclass
class NodeStatus:
    """节点状态"""
    name: str
    state: str  # idle, allocated, mixed, down, drained
    total_cores: int = 0
    allocated_cores: int = 0
    total_gpus: int = 0
    allocated_gpus: int = 0
    memory_total_gb: float = 0.0
    memory_free_gb: float = 0.0
    features: List[str] = field(default_factory=list)
    partitions: List[str] = field(default_factory=list)
    last_update: Optional[datetime] = None
    
    @property
    def available_cores(self) -> int:
        return self.total_cores - self.allocated_cores
    
    @property
    def available_gpus(self) -> int:
        return self.total_gpus - self.allocated_gpus
    
    @property
    def is_available(self) -> bool:
        return self.state in ["idle", "mixed"]


@dataclass
class QueueStats:
    """队列统计"""
    name: str
    available: bool
    priority: int = 0
    time_limit: Optional[str] = None
    max_nodes: int = 0
    max_cores: int = 0
    running_jobs: int = 0
    pending_jobs: int = 0
    suspended_jobs: int = 0
    total_nodes: int = 0
    idle_nodes: int = 0
    avg_wait_time: Optional[timedelta] = None
    
    @property
    def total_jobs(self) -> int:
        return self.running_jobs + self.pending_jobs + self.suspended_jobs
    
    @property
    def utilization(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return (self.total_nodes - self.idle_nodes) / self.total_nodes


@dataclass
class GPUStats:
    """GPU统计"""
    node: str
    gpu_id: int
    type: str
    utilization_pct: float = 0.0
    memory_used_mb: int = 0
    memory_total_mb: int = 0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    processes: List[Dict] = field(default_factory=list)
    
    @property
    def memory_utilization_pct(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


@dataclass
class JobResourceUsage:
    """作业资源使用"""
    job_id: str
    job_name: str
    user: str
    partition: str
    state: str
    nodes: List[str] = field(default_factory=list)
    cores: int = 0
    gpus: int = 0
    memory_used_gb: float = 0.0
    cpu_time_hours: float = 0.0
    wall_time_hours: float = 0.0
    efficiency_pct: float = 0.0


@dataclass
class ClusterMetrics:
    """集群指标"""
    timestamp: datetime
    total_nodes: int = 0
    idle_nodes: int = 0
    allocated_nodes: int = 0
    down_nodes: int = 0
    total_cores: int = 0
    allocated_cores: int = 0
    total_gpus: int = 0
    allocated_gpus: int = 0
    running_jobs: int = 0
    pending_jobs: int = 0
    avg_cluster_utilization: float = 0.0
    queues: List[QueueStats] = field(default_factory=list)
    nodes: List[NodeStatus] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_nodes": self.total_nodes,
            "idle_nodes": self.idle_nodes,
            "allocated_nodes": self.allocated_nodes,
            "down_nodes": self.down_nodes,
            "total_cores": self.total_cores,
            "allocated_cores": self.allocated_cores,
            "total_gpus": self.total_gpus,
            "allocated_gpus": self.allocated_gpus,
            "running_jobs": self.running_jobs,
            "pending_jobs": self.pending_jobs,
            "avg_cluster_utilization": self.avg_cluster_utilization,
            "queue_summary": [
                {"name": q.name, "running": q.running_jobs, "pending": q.pending_jobs}
                for q in self.queues
            ]
        }


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(
        self,
        connector: ClusterConnector,
        scheduler_type: str = "slurm",
        poll_interval: int = 60,
        history_size: int = 1000
    ):
        self.connector = connector
        self.scheduler_type = scheduler_type.lower()
        self.poll_interval = poll_interval
        self.history_size = history_size
        
        # 数据存储
        self._metrics_history: deque = deque(maxlen=history_size)
        self._nodes: Dict[str, NodeStatus] = {}
        self._queues: Dict[str, QueueStats] = {}
        self._gpu_stats: Dict[str, List[GPUStats]] = {}
        self._job_usage: Dict[str, JobResourceUsage] = {}
        
        # 告警回调
        self._alert_handlers: List[Callable] = []
        self._alert_conditions: List[dict] = []
        
        # 后台线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
    
    def start_monitoring(self):
        """启动后台监控"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """停止后台监控"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            try:
                self.update_metrics()
                self._check_alerts()
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
            
            # 等待下次轮询
            self._stop_event.wait(self.poll_interval)
    
    def update_metrics(self) -> ClusterMetrics:
        """更新并返回集群指标"""
        metrics = ClusterMetrics(timestamp=datetime.now())
        
        try:
            if self.scheduler_type == "slurm":
                self._update_slurm_metrics(metrics)
            elif self.scheduler_type == "pbs":
                self._update_pbs_metrics(metrics)
            elif self.scheduler_type == "lsf":
                self._update_lsf_metrics(metrics)
            
            # 存储历史
            with self._lock:
                self._metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
            return metrics
    
    def _update_slurm_metrics(self, metrics: ClusterMetrics):
        """更新Slurm指标"""
        # 节点信息
        code, stdout, _ = self.connector.execute(
            "sinfo -N -o '%N|%T|%c|%C|%G|%m|%e|%f|%P' --noheader"
        )
        
        if code == 0:
            nodes = []
            for line in stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 9:
                    # 解析核心分配信息 %C = allocated/idle/other/total
                    core_info = parts[3].split("/")
                    
                    node = NodeStatus(
                        name=parts[0],
                        state=parts[1].lower(),
                        total_cores=int(core_info[3]) if len(core_info) > 3 else int(parts[2]),
                        allocated_cores=int(core_info[0]) if len(core_info) > 0 else 0,
                        total_gpus=len(parts[4].split(",")) if parts[4] else 0,
                        memory_total_gb=float(parts[5]) / 1024 if parts[5] else 0,
                        memory_free_gb=float(parts[6]) / 1024 if parts[6] else 0,
                        features=parts[7].split(",") if parts[7] else [],
                        partitions=parts[8].split(",") if parts[8] else [],
                        last_update=datetime.now()
                    )
                    nodes.append(node)
            
            with self._lock:
                for node in nodes:
                    self._nodes[node.name] = node
            
            metrics.nodes = nodes
            metrics.total_nodes = len(nodes)
            metrics.idle_nodes = sum(1 for n in nodes if n.state == "idle")
            metrics.allocated_nodes = sum(1 for n in nodes if n.state == "allocated")
            metrics.down_nodes = sum(1 for n in nodes if n.state in ["down", "drained"])
            metrics.total_cores = sum(n.total_cores for n in nodes)
            metrics.allocated_cores = sum(n.allocated_cores for n in nodes)
        
        # 队列信息
        code, stdout, _ = self.connector.execute(
            "sinfo -o '%P|%a|%l|%D|%t|%N' --noheader"
        )
        
        if code == 0:
            queues = []
            for line in stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 6:
                    queue = QueueStats(
                        name=parts[0].rstrip("*"),
                        available=parts[1] == "up",
                        time_limit=parts[2],
                        total_nodes=int(parts[3]) if parts[3].isdigit() else 0,
                        running_jobs=0,  # 将在下面更新
                        pending_jobs=0
                    )
                    queues.append(queue)
            
            # 获取作业统计
            code, stdout, _ = self.connector.execute(
                "squeue -o '%T|%P' --noheader"
            )
            
            if code == 0:
                queue_jobs = defaultdict(lambda: {"running": 0, "pending": 0})
                for line in stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) >= 2:
                        state, partition = parts[0], parts[1]
                        if state == "RUNNING":
                            queue_jobs[partition]["running"] += 1
                        elif state == "PENDING":
                            queue_jobs[partition]["pending"] += 1
                
                for queue in queues:
                    queue.running_jobs = queue_jobs[queue.name]["running"]
                    queue.pending_jobs = queue_jobs[queue.name]["pending"]
            
            with self._lock:
                for queue in queues:
                    self._queues[queue.name] = queue
            
            metrics.queues = queues
            metrics.running_jobs = sum(q.running_jobs for q in queues)
            metrics.pending_jobs = sum(q.pending_jobs for q in queues)
            
            # 计算集群利用率
            if metrics.total_cores > 0:
                metrics.avg_cluster_utilization = metrics.allocated_cores / metrics.total_cores
    
    def _update_pbs_metrics(self, metrics: ClusterMetrics):
        """更新PBS指标"""
        # PBS qstat -q
        code, stdout, _ = self.connector.execute("qstat -q")
        
        if code == 0:
            queues = []
            lines = stdout.strip().split("\n")
            # 解析表格
            in_table = False
            for line in lines:
                if "Queue" in line and "Memory" in line:
                    in_table = True
                    continue
                if in_table and line.strip() and not line.startswith("-"):
                    parts = line.split()
                    if len(parts) >= 5:
                        queue = QueueStats(
                            name=parts[0],
                            available=True,
                            max_nodes=int(parts[4]) if parts[4].isdigit() else 0,
                            running_jobs=int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 0,
                            pending_jobs=int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else 0
                        )
                        queues.append(queue)
            
            metrics.queues = queues
            metrics.running_jobs = sum(q.running_jobs for q in queues)
            metrics.pending_jobs = sum(q.pending_jobs for q in queues)
    
    def _update_lsf_metrics(self, metrics: ClusterMetrics):
        """更新LSF指标"""
        # bqueues
        code, stdout, _ = self.connector.execute(
            "bqueues -o 'queue_name priority status max njobs pend run susp' -noheader"
        )
        
        if code == 0:
            queues = []
            for line in stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    queue = QueueStats(
                        name=parts[0],
                        available=parts[2] == "Open",
                        priority=int(parts[1]) if parts[1].isdigit() else 0,
                        running_jobs=int(parts[5]) if parts[5].isdigit() else 0,
                        pending_jobs=int(parts[4]) if parts[4].isdigit() else 0,
                        suspended_jobs=int(parts[6]) if parts[6].isdigit() else 0
                    )
                    queues.append(queue)
            
            metrics.queues = queues
            metrics.running_jobs = sum(q.running_jobs for q in queues)
            metrics.pending_jobs = sum(q.pending_jobs for q in queues)
        
        # bhosts for node info
        code, stdout, _ = self.connector.execute(
            "bhosts -o 'host_name status njobs max mem' -noheader"
        )
        
        if code == 0:
            nodes = []
            for line in stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    node = NodeStatus(
                        name=parts[0],
                        state=parts[1].lower(),
                        total_cores=int(parts[3]) if parts[3].isdigit() else 0,
                        allocated_cores=int(parts[2]) if parts[2].isdigit() else 0
                    )
                    nodes.append(node)
            
            metrics.nodes = nodes
            metrics.total_nodes = len(nodes)
    
    def get_gpu_stats(self, node: Optional[str] = None) -> List[GPUStats]:
        """
        获取GPU统计
        
        Args:
            node: 指定节点，None则查询所有节点
        
        Returns:
            GPUStats列表
        """
        stats = []
        
        if node:
            nodes_to_query = [node]
        else:
            # 获取有GPU的节点
            with self._lock:
                nodes_to_query = [
                    n.name for n in self._nodes.values()
                    if n.total_gpus > 0
                ]
        
        for node_name in nodes_to_query:
            # 使用nvidia-smi
            nvidia_smi_cmd = (
                "nvidia-smi --query-gpu=index,name,utilization.gpu,"
                "memory.used,memory.total,temperature.gpu,power.draw "
                "--format=csv,noheader,nounits"
            )
            
            code, stdout, _ = self.connector.execute(
                f"ssh {node_name} '{nvidia_smi_cmd}' 2>/dev/null || echo 'ERROR'"
            )
            
            if code == 0 and "ERROR" not in stdout:
                node_gpus = []
                for line in stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 7:
                        gpu = GPUStats(
                            node=node_name,
                            gpu_id=int(parts[0]),
                            type=parts[1],
                            utilization_pct=float(parts[2]) if parts[2] else 0.0,
                            memory_used_mb=int(parts[3]) if parts[3] else 0,
                            memory_total_mb=int(parts[4]) if parts[4] else 0,
                            temperature_c=float(parts[5]) if parts[5] else 0.0,
                            power_draw_w=float(parts[6]) if parts[6] else 0.0
                        )
                        node_gpus.append(gpu)
                
                with self._lock:
                    self._gpu_stats[node_name] = node_gpus
                
                stats.extend(node_gpus)
        
        return stats
    
    def get_job_resource_usage(self, job_id: Optional[str] = None) -> List[JobResourceUsage]:
        """获取作业资源使用"""
        usages = []
        
        if self.scheduler_type == "slurm":
            # sacct for completed jobs, sstat for running
            if job_id:
                # 特定作业
                code, stdout, _ = self.connector.execute(
                    f"sacct -j {job_id} -o 'JobID,JobName,User,Partition,State,NNodes,NCPUS,MaxRSS,TotalCPU,Elapsed' -P --noheader"
                )
            else:
                # 所有运行中作业
                code, stdout, _ = self.connector.execute(
                    "sstat -o 'JobID,JobName,User,Partition,MaxRSS,AveCPU' -P --noheader"
                )
            
            if code == 0:
                for line in stdout.strip().split("\n"):
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) >= 5:
                        usage = JobResourceUsage(
                            job_id=parts[0],
                            job_name=parts[1],
                            user=parts[2],
                            partition=parts[3],
                            state=parts[4],
                            memory_used_gb=self._parse_memory(parts[7]) if len(parts) > 7 else 0.0
                        )
                        usages.append(usage)
        
        return usages
    
    def _parse_memory(self, mem_str: str) -> float:
        """解析内存字符串为GB"""
        if not mem_str:
            return 0.0
        
        mem_str = mem_str.strip().upper()
        
        try:
            if mem_str.endswith("K"):
                return float(mem_str[:-1]) / (1024 * 1024)
            elif mem_str.endswith("M"):
                return float(mem_str[:-1]) / 1024
            elif mem_str.endswith("G"):
                return float(mem_str[:-1])
            elif mem_str.endswith("T"):
                return float(mem_str[:-1]) * 1024
            else:
                # Assume KB
                return float(mem_str) / (1024 * 1024)
        except ValueError:
            return 0.0
    
    def get_current_metrics(self) -> Optional[ClusterMetrics]:
        """获取当前指标"""
        with self._lock:
            if self._metrics_history:
                return self._metrics_history[-1]
            return None
    
    def get_metrics_history(
        self,
        since: Optional[datetime] = None,
        duration: Optional[timedelta] = None
    ) -> List[ClusterMetrics]:
        """
        获取历史指标
        
        Args:
            since: 起始时间
            duration: 持续时间
        
        Returns:
            ClusterMetrics列表
        """
        with self._lock:
            metrics_list = list(self._metrics_history)
        
        if since:
            metrics_list = [m for m in metrics_list if m.timestamp >= since]
        
        if duration:
            cutoff = datetime.now() - duration
            metrics_list = [m for m in metrics_list if m.timestamp >= cutoff]
        
        return metrics_list
    
    def add_alert_condition(
        self,
        condition_type: str,
        threshold: float,
        comparator: str = ">",
        callback: Optional[Callable] = None
    ):
        """
        添加告警条件
        
        Args:
            condition_type: 条件类型 (utilization, pending_jobs, queue_wait_time)
            threshold: 阈值
            comparator: 比较器 (>, <, >=, <=, ==)
            callback: 告警回调函数
        """
        condition = {
            "type": condition_type,
            "threshold": threshold,
            "comparator": comparator,
            "callback": callback,
            "last_triggered": None
        }
        self._alert_conditions.append(condition)
    
    def register_alert_handler(self, handler: Callable):
        """注册告警处理器"""
        self._alert_handlers.append(handler)
    
    def _check_alerts(self):
        """检查告警条件"""
        metrics = self.get_current_metrics()
        if not metrics:
            return
        
        for condition in self._alert_conditions:
            triggered = False
            value = None
            
            if condition["type"] == "utilization":
                value = metrics.avg_cluster_utilization
            elif condition["type"] == "pending_jobs":
                value = metrics.pending_jobs
            elif condition["type"] == "running_jobs":
                value = metrics.running_jobs
            
            if value is not None:
                comparator = condition["comparator"]
                threshold = condition["threshold"]
                
                if comparator == ">":
                    triggered = value > threshold
                elif comparator == "<":
                    triggered = value < threshold
                elif comparator == ">=":
                    triggered = value >= threshold
                elif comparator == "<=":
                    triggered = value <= threshold
                elif comparator == "==":
                    triggered = value == threshold
                
                if triggered:
                    # 防抖动：至少间隔5分钟
                    last = condition.get("last_triggered")
                    if last is None or (datetime.now() - last) > timedelta(minutes=5):
                        condition["last_triggered"] = datetime.now()
                        self._trigger_alert(condition, value)
    
    def _trigger_alert(self, condition: dict, value: float):
        """触发告警"""
        alert_msg = (
            f"ALERT: {condition['type']} {condition['comparator']} "
            f"{condition['threshold']} (current: {value:.2f})"
        )
        logger.warning(alert_msg)
        
        # 调用回调
        if condition["callback"]:
            try:
                condition["callback"](condition, value)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # 调用全局处理器
        for handler in self._alert_handlers:
            try:
                handler(condition, value)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def estimate_wait_time(
        self,
        nodes: int = 1,
        cores: int = 1,
        gpus: int = 0,
        walltime_hours: float = 1.0,
        queue: Optional[str] = None
    ) -> Optional[timedelta]:
        """
        估算等待时间
        
        Args:
            nodes: 节点数
            cores: 核心数
            gpus: GPU数
            walltime_hours: 运行时间
            queue: 队列名
        
        Returns:
            预估等待时间
        """
        with self._lock:
            if queue and queue in self._queues:
                q = self._queues[queue]
                # 简单估算：每个pending作业增加5分钟
                base_wait = q.pending_jobs * 5
                
                # 考虑队列利用率
                if q.utilization > 0.9:
                    base_wait *= 2
                
                return timedelta(minutes=min(base_wait, 1440))  # 最多24小时
            
            # 默认估算
            total_pending = sum(q.pending_jobs for q in self._queues.values())
            return timedelta(minutes=min(total_pending * 3, 1440))
    
    def get_best_queue(
        self,
        nodes: int = 1,
        cores: int = 1,
        gpus: int = 0,
        walltime_hours: float = 1.0
    ) -> Optional[str]:
        """
        获取最佳队列
        
        Args:
            nodes: 节点数
            cores: 核心数
            gpus: GPU数
            walltime_hours: 运行时间
        
        Returns:
            最佳队列名
        """
        with self._lock:
            candidates = []
            
            for name, queue in self._queues.items():
                if not queue.available:
                    continue
                
                # 检查资源限制
                if queue.max_nodes > 0 and nodes > queue.max_nodes:
                    continue
                
                # 计算评分 (越低越好)
                score = queue.pending_jobs + queue.running_jobs * 0.5
                
                candidates.append((name, score, queue.avg_wait_time or timedelta(hours=1)))
            
            if candidates:
                # 按评分排序
                candidates.sort(key=lambda x: (x[1], x[2]))
                return candidates[0][0]
            
            return None
    
    def export_metrics(self, path: str):
        """导出指标到文件"""
        with self._lock:
            data = {
                "export_time": datetime.now().isoformat(),
                "metrics_history": [m.to_dict() for m in self._metrics_history],
                "nodes": [asdict(n) for n in self._nodes.values()],
                "queues": [asdict(q) for q in self._queues.values()]
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {path}")


@contextmanager
def monitor_context(
    connector: ClusterConnector,
    scheduler_type: str = "slurm",
    poll_interval: int = 60
):
    """监控上下文管理器"""
    monitor = ResourceMonitor(connector, scheduler_type, poll_interval)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()
