"""
Monitoring and metrics collection for HPC jobs and clusters.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core.base import BaseJobScheduler
from ..core.job import JobStatus, JobInfo
from ..core.cluster import QueueInfo, NodeInfo

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Monitoring alert."""
    id: str
    level: AlertLevel
    message: str
    source: str  # job_id, cluster_name, etc.
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'level': self.level.value,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'acknowledged': self.acknowledged,
        }


@dataclass
class JobMetrics:
    """Metrics for a job."""
    job_id: str
    
    # Time metrics
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    queue_wait_seconds: Optional[float] = None
    runtime_seconds: Optional[float] = None
    
    # Resource metrics
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_limit_mb: Optional[float] = None
    
    # I/O metrics
    read_bytes: Optional[int] = None
    write_bytes: Optional[int] = None
    read_ops: Optional[int] = None
    write_ops: Optional[int] = None
    
    # Network metrics (if applicable)
    network_rx_bytes: Optional[int] = None
    network_tx_bytes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'submit_time': self.submit_time.isoformat() if self.submit_time else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'queue_wait_seconds': self.queue_wait_seconds,
            'runtime_seconds': self.runtime_seconds,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_limit_mb': self.memory_limit_mb,
            'read_bytes': self.read_bytes,
            'write_bytes': self.write_bytes,
            'read_ops': self.read_ops,
            'write_ops': self.write_ops,
            'network_rx_bytes': self.network_rx_bytes,
            'network_tx_bytes': self.network_tx_bytes,
        }


@dataclass
class ClusterMetrics:
    """Metrics for a cluster."""
    timestamp: datetime
    
    # Node metrics
    total_nodes: int = 0
    available_nodes: int = 0
    allocated_nodes: int = 0
    offline_nodes: int = 0
    
    # Core metrics
    total_cores: int = 0
    available_cores: int = 0
    allocated_cores: int = 0
    
    # Job metrics
    pending_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    
    # Queue metrics
    queue_depths: Dict[str, int] = field(default_factory=dict)
    
    # Utilization
    cpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_nodes': self.total_nodes,
            'available_nodes': self.available_nodes,
            'allocated_nodes': self.allocated_nodes,
            'offline_nodes': self.offline_nodes,
            'total_cores': self.total_cores,
            'available_cores': self.available_cores,
            'allocated_cores': self.allocated_cores,
            'pending_jobs': self.pending_jobs,
            'running_jobs': self.running_jobs,
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'queue_depths': self.queue_depths,
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
        }


class JobMonitor:
    """Monitor for HPC jobs."""
    
    def __init__(self, scheduler: BaseJobScheduler):
        self.scheduler = scheduler
        self._monitored_jobs: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[Alert] = []
        self._alert_handlers: List[Callable[[Alert], None]] = []
        self._metrics_history: Dict[str, List[JobMetrics]] = {}
        self._lock = asyncio.Lock()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler callback."""
        self._alert_handlers.append(handler)
    
    async def start_monitoring(
        self,
        job_id: str,
        poll_interval: int = 30,
        error_patterns: List[str] = None
    ) -> None:
        """
        Start monitoring a job.
        
        Args:
            job_id: Job ID to monitor
            poll_interval: Seconds between checks
            error_patterns: List of error patterns to watch for in output
        """
        async with self._lock:
            if job_id in self._monitored_jobs:
                logger.warning(f"Already monitoring job {job_id}")
                return
            
            self._monitored_jobs[job_id] = {
                'poll_interval': poll_interval,
                'error_patterns': error_patterns or [],
                'start_time': datetime.now(),
                'previous_status': None,
            }
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop(job_id))
    
    async def _monitoring_loop(self, job_id: str) -> None:
        """Main monitoring loop for a job."""
        while True:
            async with self._lock:
                if job_id not in self._monitored_jobs:
                    break
                
                config = self._monitored_jobs[job_id]
            
            try:
                # Get current status
                status = await self.scheduler.get_job_status(job_id)
                info = await self.scheduler.get_job_info(job_id)
                
                # Check for status changes
                previous_status = config.get('previous_status')
                if previous_status and status != previous_status:
                    self._create_alert(
                        AlertLevel.INFO,
                        f"Job {job_id} status changed from {previous_status.value} to {status.value}",
                        job_id
                    )
                
                config['previous_status'] = status
                
                # Collect metrics
                metrics = await self._collect_job_metrics(job_id, info)
                
                if job_id not in self._metrics_history:
                    self._metrics_history[job_id] = []
                self._metrics_history[job_id].append(metrics)
                
                # Check for errors
                if error_patterns := config.get('error_patterns'):
                    await self._check_for_errors(job_id, error_patterns)
                
                # Check for resource issues
                self._check_resource_usage(job_id, metrics)
                
                # Check if job is done
                if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    await self._handle_job_completion(job_id, status)
                    async with self._lock:
                        del self._monitored_jobs[job_id]
                    break
                
            except Exception as e:
                logger.error(f"Error monitoring job {job_id}: {e}")
                self._create_alert(
                    AlertLevel.ERROR,
                    f"Monitoring error for job {job_id}: {e}",
                    job_id
                )
            
            await asyncio.sleep(config['poll_interval'])
    
    async def _collect_job_metrics(self, job_id: str, info: JobInfo) -> JobMetrics:
        """Collect metrics for a job."""
        metrics = JobMetrics(job_id=job_id)
        
        # Set timing info
        metrics.submit_time = info.submit_time
        metrics.start_time = info.start_time
        
        if info.start_time and info.submit_time:
            metrics.queue_wait_seconds = (
                info.start_time - info.submit_time
            ).total_seconds()
        
        if info.start_time:
            metrics.runtime_seconds = (
                datetime.now() - info.start_time
            ).total_seconds()
        
        # Try to get resource usage from scheduler
        try:
            result = await self.scheduler.get_job_result(job_id)
            # Extract resource usage if available
        except Exception:
            pass
        
        return metrics
    
    async def _check_for_errors(self, job_id: str, error_patterns: List[str]) -> None:
        """Check job output for error patterns."""
        try:
            info = await self.scheduler.get_job_info(job_id)
            
            # Try to read stderr if available
            if info.additional_info.get('stderr_path'):
                # This would require connector access to read files
                pass
        except Exception:
            pass
    
    def _check_resource_usage(self, job_id: str, metrics: JobMetrics) -> None:
        """Check for resource usage issues."""
        # Check for high memory usage
        if metrics.memory_percent and metrics.memory_percent > 95:
            self._create_alert(
                AlertLevel.WARNING,
                f"Job {job_id} is using {metrics.memory_percent:.1f}% of allocated memory",
                job_id,
                {'memory_percent': metrics.memory_percent}
            )
    
    async def _handle_job_completion(self, job_id: str, status: JobStatus) -> None:
        """Handle job completion."""
        if status == JobStatus.FAILED:
            self._create_alert(
                AlertLevel.ERROR,
                f"Job {job_id} has failed",
                job_id
            )
        elif status == JobStatus.COMPLETED:
            self._create_alert(
                AlertLevel.INFO,
                f"Job {job_id} completed successfully",
                job_id
            )
    
    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        source: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Create and dispatch an alert."""
        alert = Alert(
            id=f"{source}_{datetime.now().timestamp()}",
            level=level,
            message=message,
            source=source,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        self._alerts.append(alert)
        
        # Dispatch to handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_alerts(
        self,
        level: AlertLevel = None,
        source: str = None,
        unacknowledged_only: bool = False
    ) -> List[Alert]:
        """Get alerts matching criteria."""
        alerts = self._alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if source:
            alerts = [a for a in alerts if a.source == source]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_metrics_history(self, job_id: str) -> List[JobMetrics]:
        """Get metrics history for a job."""
        return self._metrics_history.get(job_id, [])
    
    def stop_monitoring(self, job_id: str) -> None:
        """Stop monitoring a job."""
        if job_id in self._monitored_jobs:
            del self._monitored_jobs[job_id]


class ClusterMonitor:
    """Monitor for HPC clusters."""
    
    def __init__(self, scheduler: BaseJobScheduler, cluster_name: str):
        self.scheduler = scheduler
        self.cluster_name = cluster_name
        self._monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._metrics_history: List[ClusterMetrics] = []
        self._alert_handlers: List[Callable[[Alert], None]] = []
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler callback."""
        self._alert_handlers.append(handler)
    
    async def start_monitoring(self, interval: int = 60) -> None:
        """Start cluster monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
    
    async def stop_monitoring(self) -> None:
        """Stop cluster monitoring."""
        self._monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
    
    async def _monitoring_loop(self, interval: int) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = await self._collect_cluster_metrics()
                self._metrics_history.append(metrics)
                
                # Keep only last 24 hours of data
                cutoff = datetime.now() - timedelta(hours=24)
                self._metrics_history = [
                    m for m in self._metrics_history
                    if m.timestamp > cutoff
                ]
                
                # Check for issues
                self._check_cluster_health(metrics)
                
            except Exception as e:
                logger.error(f"Error collecting cluster metrics: {e}")
            
            await asyncio.sleep(interval)
    
    async def _collect_cluster_metrics(self) -> ClusterMetrics:
        """Collect cluster-wide metrics."""
        metrics = ClusterMetrics(timestamp=datetime.now())
        
        try:
            # Get queue info
            queues = await self.scheduler.get_queues()
            
            for queue in queues:
                metrics.queue_depths[queue.name] = queue.jobs_queued
                metrics.pending_jobs += queue.jobs_queued
                metrics.running_jobs += queue.jobs_running
            
            # Get node info
            nodes = await self.scheduler.get_nodes()
            
            metrics.total_nodes = len(nodes)
            metrics.allocated_nodes = sum(
                1 for n in nodes if n.state == 'allocated'
            )
            metrics.available_nodes = sum(
                1 for n in nodes if n.state == 'idle'
            )
            metrics.offline_nodes = sum(
                1 for n in nodes if n.state in ['down', 'offline', 'drain']
            )
            
            metrics.total_cores = sum(n.total_cores for n in nodes)
            metrics.available_cores = sum(n.free_cores for n in nodes)
            metrics.allocated_cores = metrics.total_cores - metrics.available_cores
            
            # Calculate utilization
            if metrics.total_cores > 0:
                metrics.cpu_utilization = (
                    metrics.allocated_cores / metrics.total_cores * 100
                )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _check_cluster_health(self, metrics: ClusterMetrics) -> None:
        """Check cluster health and generate alerts."""
        # Check for offline nodes
        if metrics.offline_nodes > metrics.total_nodes * 0.1:  # >10% offline
            self._create_alert(
                AlertLevel.WARNING,
                f"{metrics.offline_nodes} nodes are offline in cluster {self.cluster_name}",
                self.cluster_name
            )
        
        # Check for high queue depth
        for queue, depth in metrics.queue_depths.items():
            if depth > 100:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"Queue {queue} has {depth} pending jobs",
                    self.cluster_name,
                    {'queue': queue, 'depth': depth}
                )
    
    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        source: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Create and dispatch an alert."""
        alert = Alert(
            id=f"{source}_{datetime.now().timestamp()}",
            level=level,
            message=message,
            source=source,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_metrics_history(
        self,
        since: datetime = None
    ) -> List[ClusterMetrics]:
        """Get metrics history."""
        if since:
            return [m for m in self._metrics_history if m.timestamp >= since]
        return self._metrics_history
    
    def get_current_metrics(self) -> Optional[ClusterMetrics]:
        """Get most recent metrics."""
        if self._metrics_history:
            return self._metrics_history[-1]
        return None
