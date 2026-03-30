"""
Abstract base classes for HPC connectors and job schedulers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
import asyncio
import logging

from .exceptions import (
    ConnectionError,
    JobSubmissionError,
    JobMonitorError,
    AuthenticationError,
)
from .job import JobConfig, JobResult, JobInfo, JobStatus
from .cluster import ClusterConfig, QueueInfo, NodeInfo


logger = logging.getLogger(__name__)


class BaseHPCConnector(ABC):
    """Abstract base class for HPC cluster connectors."""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self._connected = False
        self._connection = None
        self._lock = asyncio.Lock()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to cluster."""
        return self._connected
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the cluster.
        
        Raises:
            AuthenticationError: If authentication fails
            ConnectionError: If connection cannot be established
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the cluster."""
        pass
    
    @abstractmethod
    async def execute(self, command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a command on the cluster.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with 'stdout', 'stderr', and 'exit_code'
        """
        pass
    
    @abstractmethod
    async def upload_file(
        self, 
        local_path: str, 
        remote_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Upload a file to the cluster.
        
        Args:
            local_path: Local file path
            remote_path: Remote file path
            progress_callback: Optional callback(bytes_transferred, total_bytes)
        """
        pass
    
    @abstractmethod
    async def download_file(
        self, 
        remote_path: str, 
        local_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Download a file from the cluster.
        
        Args:
            remote_path: Remote file path
            local_path: Local file path
            progress_callback: Optional callback(bytes_transferred, total_bytes)
        """
        pass
    
    @abstractmethod
    async def upload_directory(
        self,
        local_path: str,
        remote_path: str,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Upload a directory to the cluster.
        
        Args:
            local_path: Local directory path
            remote_path: Remote directory path
            exclude_patterns: Patterns to exclude (e.g., ['*.pyc', '__pycache__'])
            progress_callback: Optional callback(bytes_transferred, total_bytes)
        """
        pass
    
    @abstractmethod
    async def download_directory(
        self,
        remote_path: str,
        local_path: str,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """
        Download a directory from the cluster.
        
        Args:
            remote_path: Remote directory path
            local_path: Local directory path
            exclude_patterns: Patterns to exclude
            progress_callback: Optional callback(bytes_transferred, total_bytes)
        """
        pass
    
    @abstractmethod
    async def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the cluster."""
        pass
    
    @abstractmethod
    async def list_directory(self, remote_path: str) -> List[Dict[str, Any]]:
        """
        List directory contents.
        
        Returns:
            List of file info dictionaries with 'name', 'size', 'mode', 'mtime', etc.
        """
        pass
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


class BaseJobScheduler(ABC):
    """Abstract base class for job schedulers."""
    
    def __init__(self, connector: BaseHPCConnector):
        self.connector = connector
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    @abstractmethod
    async def submit_job(self, job_config: JobConfig) -> str:
        """
        Submit a job to the scheduler.
        
        Args:
            job_config: Job configuration
            
        Returns:
            Job ID assigned by the scheduler
            
        Raises:
            JobSubmissionError: If submission fails
        """
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a submitted job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if cancellation was successful
        """
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get the current status of a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Current job status
        """
        pass
    
    @abstractmethod
    async def get_job_info(self, job_id: str) -> JobInfo:
        """
        Get detailed information about a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job information
        """
        pass
    
    @abstractmethod
    async def get_job_result(self, job_id: str) -> JobResult:
        """
        Get the result of a completed job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job result
        """
        pass
    
    @abstractmethod
    async def list_jobs(
        self,
        user: Optional[str] = None,
        status: Optional[JobStatus] = None,
        queue: Optional[str] = None
    ) -> List[JobInfo]:
        """
        List jobs matching criteria.
        
        Args:
            user: Filter by user
            status: Filter by status
            queue: Filter by queue
            
        Returns:
            List of job information
        """
        pass
    
    @abstractmethod
    async def get_queues(self) -> List[QueueInfo]:
        """
        Get information about available queues/partitions.
        
        Returns:
            List of queue information
        """
        pass
    
    @abstractmethod
    async def get_nodes(self, queue: Optional[str] = None) -> List[NodeInfo]:
        """
        Get information about compute nodes.
        
        Args:
            queue: Filter by queue/partition
            
        Returns:
            List of node information
        """
        pass
    
    @abstractmethod
    async def estimate_start_time(self, job_id: str) -> Optional[str]:
        """
        Estimate when a job will start.
        
        Args:
            job_id: Job ID
            
        Returns:
            Estimated start time string or None
        """
        pass
    
    async def monitor_job(
        self,
        job_id: str,
        poll_interval: int = 30,
        on_status_change: Optional[Callable[[JobStatus, JobStatus], None]] = None,
        on_complete: Optional[Callable[[JobResult], None]] = None
    ) -> JobResult:
        """
        Monitor a job until completion.
        
        Args:
            job_id: Job ID to monitor
            poll_interval: Seconds between status checks
            on_status_change: Callback(old_status, new_status)
            on_complete: Callback(result)
            
        Returns:
            Job result
        """
        previous_status = None
        
        while True:
            current_status = await self.get_job_status(job_id)
            
            if previous_status is not None and current_status != previous_status:
                if on_status_change:
                    on_status_change(previous_status, current_status)
            
            if current_status in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.TIMEOUT
            ]:
                result = await self.get_job_result(job_id)
                if on_complete:
                    on_complete(result)
                return result
            
            previous_status = current_status
            await asyncio.sleep(poll_interval)
    
    async def start_monitoring(
        self,
        job_id: str,
        poll_interval: int = 30,
        on_status_change: Optional[Callable[[JobStatus, JobStatus], None]] = None,
        on_complete: Optional[Callable[[JobResult], None]] = None
    ) -> None:
        """
        Start background monitoring of a job.
        
        Args:
            job_id: Job ID to monitor
            poll_interval: Seconds between status checks
            on_status_change: Callback(old_status, new_status)
            on_complete: Callback(result)
        """
        if job_id in self._monitoring_tasks:
            logger.warning(f"Already monitoring job {job_id}")
            return
        
        task = asyncio.create_task(
            self._monitoring_loop(job_id, poll_interval, on_status_change, on_complete)
        )
        self._monitoring_tasks[job_id] = task
    
    async def _monitoring_loop(
        self,
        job_id: str,
        poll_interval: int,
        on_status_change: Optional[Callable],
        on_complete: Optional[Callable]
    ) -> None:
        """Internal monitoring loop."""
        try:
            await self.monitor_job(job_id, poll_interval, on_status_change, on_complete)
        except Exception as e:
            logger.error(f"Error monitoring job {job_id}: {e}")
        finally:
            if job_id in self._monitoring_tasks:
                del self._monitoring_tasks[job_id]
    
    def stop_monitoring(self, job_id: str) -> None:
        """Stop background monitoring of a job."""
        if job_id in self._monitoring_tasks:
            self._monitoring_tasks[job_id].cancel()
            del self._monitoring_tasks[job_id]
    
    async def wait_for_jobs(
        self,
        job_ids: List[str],
        poll_interval: int = 30
    ) -> Dict[str, JobResult]:
        """
        Wait for multiple jobs to complete.
        
        Args:
            job_ids: List of job IDs
            poll_interval: Seconds between status checks
            
        Returns:
            Dictionary mapping job_id to JobResult
        """
        results = {}
        pending = set(job_ids)
        
        while pending:
            for job_id in list(pending):
                status = await self.get_job_status(job_id)
                if status in [
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                    JobStatus.TIMEOUT
                ]:
                    results[job_id] = await self.get_job_result(job_id)
                    pending.remove(job_id)
            
            if pending:
                await asyncio.sleep(poll_interval)
        
        return results
