"""
High-level API for HPC Connector.

This module provides a simplified interface for common HPC operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from datetime import datetime

from .core.cluster import ClusterConfig, ClusterManager
from .core.job import JobConfig, JobResult, JobStatus, JobPriority
from .core.exceptions import HPCConnectorError
from .connectors import ConnectorFactory
from .schedulers import SchedulerFactory
from .data import DataPipeline, TransferProgress
from .monitoring import JobMonitor, ClusterMonitor, AlertLevel
from .recovery import FaultRecovery, RecoveryConfig, RecoveryStrategy

logger = logging.getLogger(__name__)


@dataclass
class HPCClientConfig:
    """Configuration for HPC client."""
    cluster_name: str
    enable_monitoring: bool = True
    enable_recovery: bool = True
    recovery_config: RecoveryConfig = None


class HPCClient:
    """
    High-level client for HPC operations.
    
    Example:
        ```python
        from hpc_connector import HPCClient, ClusterConfig
        
        config = ClusterConfig.from_dict({...})
        client = HPCClient(config)
        
        async with client:
            job_id = await client.submit_job({
                'name': 'my_job',
                'command': 'python train.py',
                'work_dir': '/home/user/work',
                'resources': {
                    'nodes': 2,
                    'cores_per_node': 4,
                    'walltime': '2:00:00'
                }
            })
            
            result = await client.wait_for_job(job_id)
            print(f"Job completed with exit code {result.exit_code}")
        ```
    """
    
    def __init__(self, cluster_config: ClusterConfig, client_config: HPCClientConfig = None):
        self.cluster_config = cluster_config
        self.client_config = client_config or HPCClientConfig(cluster_name=cluster_config.name)
        
        self._connector = None
        self._scheduler = None
        self._data_pipeline = None
        self._job_monitor = None
        self._cluster_monitor = None
        self._fault_recovery = None
        self._initialized = False
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self) -> None:
        """Initialize the client and connect to cluster."""
        if self._initialized:
            return
        
        # Create connector
        self._connector = ConnectorFactory.create(self.cluster_config)
        await self._connector.connect()
        
        # Create scheduler
        self._scheduler = SchedulerFactory.create(
            self.cluster_config.cluster_type,
            self._connector
        )
        
        # Create data pipeline
        self._data_pipeline = DataPipeline(
            self._connector,
            self.cluster_config.staging_dir
        )
        
        # Create monitors if enabled
        if self.client_config.enable_monitoring:
            self._job_monitor = JobMonitor(self._scheduler)
            self._cluster_monitor = ClusterMonitor(
                self._scheduler,
                self.cluster_config.name
            )
            await self._cluster_monitor.start_monitoring()
        
        # Create fault recovery if enabled
        if self.client_config.enable_recovery:
            recovery_config = self.client_config.recovery_config or RecoveryConfig()
            self._fault_recovery = FaultRecovery(
                self._scheduler,
                recovery_config
            )
            await self._fault_recovery.load_state()
        
        self._initialized = True
        logger.info(f"HPC client initialized for cluster {self.cluster_config.name}")
    
    async def close(self) -> None:
        """Close the client and disconnect from cluster."""
        if not self._initialized:
            return
        
        if self._cluster_monitor:
            await self._cluster_monitor.stop_monitoring()
        
        if self._connector:
            await self._connector.disconnect()
        
        self._initialized = False
        logger.info("HPC client closed")
    
    async def submit_job(
        self,
        job_spec: Union[JobConfig, Dict[str, Any]],
        monitor: bool = True,
        enable_recovery: bool = True
    ) -> str:
        """
        Submit a job to the cluster.
        
        Args:
            job_spec: Job configuration (JobConfig or dict)
            monitor: Whether to start monitoring the job
            enable_recovery: Whether to enable fault recovery for this job
            
        Returns:
            Job ID
        """
        self._ensure_initialized()
        
        if isinstance(job_spec, dict):
            job_config = JobConfig.from_dict(job_spec)
        else:
            job_config = job_spec
        
        # Submit job
        job_id = await self._scheduler.submit_job(job_config)
        
        # Register for recovery if enabled
        if enable_recovery and self._fault_recovery:
            await self._fault_recovery.register_job(job_id, job_config)
            await self._fault_recovery.start_monitoring(job_id)
        
        # Start monitoring if enabled
        if monitor and self._job_monitor:
            await self._job_monitor.start_monitoring(job_id)
        
        return job_id
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        self._ensure_initialized()
        return await self._scheduler.cancel_job(job_id)
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status."""
        self._ensure_initialized()
        return await self._scheduler.get_job_status(job_id)
    
    async def get_job_info(self, job_id: str) -> Dict[str, Any]:
        """Get detailed job information."""
        self._ensure_initialized()
        info = await self._scheduler.get_job_info(job_id)
        return info.to_dict()
    
    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = None,
        on_status_change: Callable[[JobStatus, JobStatus], None] = None
    ) -> JobResult:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None for no limit)
            on_status_change: Callback(old_status, new_status)
            
        Returns:
            Job result
        """
        self._ensure_initialized()
        
        start_time = datetime.now()
        
        while True:
            status = await self.get_job_status(job_id)
            
            if status in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.TIMEOUT
            ]:
                return await self._scheduler.get_job_result(job_id)
            
            # Check timeout
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    raise TimeoutError(f"Timeout waiting for job {job_id}")
            
            await asyncio.sleep(poll_interval)
    
    async def wait_for_jobs(
        self,
        job_ids: List[str],
        poll_interval: int = 30
    ) -> Dict[str, JobResult]:
        """
        Wait for multiple jobs to complete.
        
        Args:
            job_ids: List of job IDs
            poll_interval: Seconds between checks
            
        Returns:
            Dictionary mapping job_id to JobResult
        """
        self._ensure_initialized()
        return await self._scheduler.wait_for_jobs(job_ids, poll_interval)
    
    async def list_jobs(
        self,
        user: str = None,
        status: JobStatus = None,
        queue: str = None
    ) -> List[Dict[str, Any]]:
        """List jobs."""
        self._ensure_initialized()
        jobs = await self._scheduler.list_jobs(user, status, queue)
        return [job.to_dict() for job in jobs]
    
    async def get_queues(self) -> List[Dict[str, Any]]:
        """Get available queues/partitions."""
        self._ensure_initialized()
        queues = await self._scheduler.get_queues()
        return [queue.to_dict() for queue in queues]
    
    async def get_nodes(self, queue: str = None) -> List[Dict[str, Any]]:
        """Get compute nodes."""
        self._ensure_initialized()
        nodes = await self._scheduler.get_nodes(queue)
        return [node.to_dict() for node in nodes]
    
    # Data transfer methods
    
    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        progress_callback: Callable[[TransferProgress], None] = None
    ) -> None:
        """Upload a file to the cluster."""
        self._ensure_initialized()
        await self._data_pipeline._upload_with_progress(
            local_path, remote_path, progress_callback
        )
    
    async def download_file(
        self,
        remote_path: str,
        local_path: str,
        progress_callback: Callable[[TransferProgress], None] = None
    ) -> None:
        """Download a file from the cluster."""
        self._ensure_initialized()
        await self._data_pipeline._download_with_progress(
            remote_path, local_path, progress_callback
        )
    
    async def sync_to_remote(
        self,
        local_dir: str,
        remote_dir: str,
        exclude_patterns: List[str] = None
    ) -> None:
        """Sync local directory to remote."""
        self._ensure_initialized()
        await self._data_pipeline.sync_to_remote(
            local_dir, remote_dir, exclude_patterns
        )
    
    async def sync_from_remote(
        self,
        remote_dir: str,
        local_dir: str,
        exclude_patterns: List[str] = None
    ) -> None:
        """Sync remote directory to local."""
        self._ensure_initialized()
        await self._data_pipeline.sync_from_remote(
            remote_dir, local_dir, exclude_patterns
        )
    
    async def stage_job_data(
        self,
        job_id: str,
        input_files: List[str],
        output_patterns: List[str],
        local_work_dir: str,
        remote_work_dir: str
    ) -> Dict[str, str]:
        """Stage data for a job."""
        self._ensure_initialized()
        return await self._data_pipeline.stage_job_data(
            job_id, input_files, output_patterns,
            local_work_dir, remote_work_dir
        )
    
    async def retrieve_job_output(
        self,
        job_id: str,
        output_patterns: List[str],
        local_work_dir: str,
        remote_work_dir: str
    ) -> Any:
        """Retrieve job output."""
        self._ensure_initialized()
        return await self._data_pipeline.retrieve_job_output(
            job_id, output_patterns, local_work_dir, remote_work_dir
        )
    
    # Monitoring methods
    
    def add_alert_handler(self, handler: Callable[[Any], None]) -> None:
        """Add an alert handler."""
        if self._job_monitor:
            self._job_monitor.add_alert_handler(handler)
        if self._cluster_monitor:
            self._cluster_monitor.add_alert_handler(handler)
    
    def get_job_alerts(
        self,
        level: AlertLevel = None,
        unacknowledged_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get job alerts."""
        if not self._job_monitor:
            return []
        
        alerts = self._job_monitor.get_alerts(level, None, unacknowledged_only)
        return [alert.to_dict() for alert in alerts]
    
    # Recovery methods
    
    async def save_checkpoint(
        self,
        job_id: str,
        checkpoint_path: str,
        step: int = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Save a checkpoint for a job."""
        if self._fault_recovery:
            await self._fault_recovery.save_checkpoint(
                job_id, checkpoint_path, step, metadata
            )
    
    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self._initialized:
            raise HPCConnectorError("Client not initialized. Use async context manager or call initialize()")


class HPCPool:
    """
    Pool of HPC clients for managing multiple clusters.
    """
    
    def __init__(self):
        self._clients: Dict[str, HPCClient] = {}
        self._cluster_manager = ClusterManager()
    
    def add_cluster(self, config: ClusterConfig) -> None:
        """Add a cluster to the pool."""
        self._cluster_manager.register(config)
    
    def remove_cluster(self, name: str) -> None:
        """Remove a cluster from the pool."""
        self._cluster_manager.unregister(name)
        if name in self._clients:
            del self._clients[name]
    
    async def get_client(self, name: str) -> HPCClient:
        """Get or create a client for a cluster."""
        if name not in self._clients:
            config = self._cluster_manager.get(name)
            if not config:
                raise ValueError(f"Cluster not found: {name}")
            
            self._clients[name] = HPCClient(config)
        
        return self._clients[name]
    
    async def submit_to_best_cluster(
        self,
        job_spec: Union[JobConfig, Dict[str, Any]],
        cluster_preference: List[str] = None
    ) -> tuple:
        """
        Submit job to the best available cluster.
        
        Returns:
            Tuple of (job_id, cluster_name)
        """
        clusters = cluster_preference or self._cluster_manager.list_clusters()
        
        for cluster_name in clusters:
            try:
                client = await self.get_client(cluster_name)
                await client.initialize()
                
                # Check queue availability
                queues = await client.get_queues()
                available = any(q['free_nodes'] > 0 for q in queues)
                
                if available:
                    job_id = await client.submit_job(job_spec)
                    return job_id, cluster_name
                
            except Exception as e:
                logger.warning(f"Could not use cluster {cluster_name}: {e}")
                continue
        
        # If no cluster available, submit to first preference anyway
        if clusters:
            client = await self.get_client(clusters[0])
            await client.initialize()
            job_id = await client.submit_job(job_spec)
            return job_id, clusters[0]
        
        raise HPCConnectorError("No clusters available")
    
    async def close_all(self) -> None:
        """Close all clients."""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
