"""
Fault recovery and checkpoint management for HPC jobs.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import json

from ..core.base import BaseJobScheduler
from ..core.job import JobConfig, JobStatus, JobResult
from ..core.exceptions import RecoveryError

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Job recovery strategies."""
    NONE = "none"  # No automatic recovery
    RESTART = "restart"  # Restart from beginning
    CHECKPOINT = "checkpoint"  # Resume from checkpoint
    RESUBMIT = "resubmit"  # Submit as new job


@dataclass
class CheckpointInfo:
    """Checkpoint information."""
    job_id: str
    timestamp: datetime
    checkpoint_path: str
    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'timestamp': self.timestamp.isoformat(),
            'checkpoint_path': self.checkpoint_path,
            'step': self.step,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CheckpointInfo":
        return cls(
            job_id=data['job_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            checkpoint_path=data['checkpoint_path'],
            step=data.get('step', 0),
            metadata=data.get('metadata', {}),
        )


@dataclass
class RecoveryConfig:
    """Configuration for job recovery."""
    strategy: RecoveryStrategy = RecoveryStrategy.RESTART
    max_retries: int = 3
    retry_delay: int = 60  # seconds between retries
    checkpoint_enabled: bool = False
    checkpoint_interval: int = 3600  # seconds
    checkpoint_keep_count: int = 3
    on_recovery: Optional[Callable[[str, str], None]] = None  # job_id, attempt
    on_recovery_failed: Optional[Callable[[str, str], None]] = None  # job_id, reason


@dataclass
class JobRecord:
    """Record of a job for recovery purposes."""
    job_id: str
    config: JobConfig
    submit_time: datetime
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[CheckpointInfo] = field(default_factory=list)
    current_status: JobStatus = JobStatus.PENDING
    recovery_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'config': self.config.to_dict(),
            'submit_time': self.submit_time.isoformat(),
            'attempts': self.attempts,
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'current_status': self.current_status.value,
            'recovery_count': self.recovery_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "JobRecord":
        return cls(
            job_id=data['job_id'],
            config=JobConfig.from_dict(data['config']),
            submit_time=datetime.fromisoformat(data['submit_time']),
            attempts=data.get('attempts', []),
            checkpoints=[CheckpointInfo.from_dict(c) for c in data.get('checkpoints', [])],
            current_status=JobStatus(data.get('current_status', 'pending')),
            recovery_count=data.get('recovery_count', 0),
        )


class FaultRecovery:
    """Fault recovery manager for HPC jobs."""
    
    def __init__(
        self,
        scheduler: BaseJobScheduler,
        config: RecoveryConfig = None,
        state_file: str = None
    ):
        self.scheduler = scheduler
        self.config = config or RecoveryConfig()
        self.state_file = state_file or ".recovery_state.json"
        self._jobs: Dict[str, JobRecord] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def register_job(
        self,
        job_id: str,
        job_config: JobConfig
    ) -> JobRecord:
        """Register a job for fault recovery."""
        async with self._lock:
            record = JobRecord(
                job_id=job_id,
                config=job_config,
                submit_time=datetime.now()
            )
            self._jobs[job_id] = record
            await self._save_state()
            return record
    
    async def start_monitoring(self, job_id: str) -> None:
        """Start monitoring a job for failure."""
        if job_id in self._monitoring_tasks:
            return
        
        task = asyncio.create_task(self._monitor_job(job_id))
        self._monitoring_tasks[job_id] = task
    
    async def _monitor_job(self, job_id: str) -> None:
        """Monitor a job and handle failures."""
        try:
            while True:
                status = await self.scheduler.get_job_status(job_id)
                
                async with self._lock:
                    if job_id not in self._jobs:
                        break
                    
                    record = self._jobs[job_id]
                    record.current_status = status
                
                # Check for failure conditions
                if status in [JobStatus.FAILED, JobStatus.TIMEOUT]:
                    await self._handle_failure(job_id, status)
                    break
                
                if status in [JobStatus.COMPLETED, JobStatus.CANCELLED]:
                    break
                
                await asyncio.sleep(30)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring job {job_id}: {e}")
        finally:
            if job_id in self._monitoring_tasks:
                del self._monitoring_tasks[job_id]
    
    async def _handle_failure(self, job_id: str, failure_status: JobStatus) -> Optional[str]:
        """Handle job failure and attempt recovery."""
        async with self._lock:
            if job_id not in self._jobs:
                return None
            
            record = self._jobs[job_id]
            
            if record.recovery_count >= self.config.max_retries:
                logger.error(f"Max retries exceeded for job {job_id}")
                if self.config.on_recovery_failed:
                    self.config.on_recovery_failed(job_id, "max_retries_exceeded")
                return None
            
            record.attempts.append({
                'job_id': job_id,
                'status': failure_status.value,
                'timestamp': datetime.now().isoformat(),
            })
            record.recovery_count += 1
        
        # Attempt recovery
        try:
            new_job_id = await self._attempt_recovery(record)
            
            if new_job_id:
                logger.info(
                    f"Job {job_id} recovered as {new_job_id} "
                    f"(attempt {record.recovery_count})"
                )
                
                if self.config.on_recovery:
                    self.config.on_recovery(job_id, new_job_id)
                
                # Update record with new job ID
                async with self._lock:
                    record.job_id = new_job_id
                    await self._save_state()
                
                # Start monitoring the new job
                await self.start_monitoring(new_job_id)
                
                return new_job_id
            else:
                logger.error(f"Recovery failed for job {job_id}")
                if self.config.on_recovery_failed:
                    self.config.on_recovery_failed(job_id, "recovery_failed")
                return None
        
        except Exception as e:
            logger.error(f"Error during recovery of job {job_id}: {e}")
            if self.config.on_recovery_failed:
                self.config.on_recovery_failed(job_id, str(e))
            return None
    
    async def _attempt_recovery(self, record: JobRecord) -> Optional[str]:
        """Attempt to recover a failed job."""
        if self.config.strategy == RecoveryStrategy.NONE:
            return None
        
        # Wait before retry
        await asyncio.sleep(self.config.retry_delay)
        
        # Create recovery job config
        recovery_config = self._create_recovery_config(record)
        
        try:
            new_job_id = await self.scheduler.submit_job(recovery_config)
            return new_job_id
        except Exception as e:
            logger.error(f"Failed to submit recovery job: {e}")
            return None
    
    def _create_recovery_config(self, record: JobRecord) -> JobConfig:
        """Create job configuration for recovery."""
        config = record.config
        
        # Modify config based on recovery strategy
        if self.config.strategy == RecoveryStrategy.CHECKPOINT:
            # Use latest checkpoint if available
            if record.checkpoints:
                latest = record.checkpoints[-1]
                config.checkpoint_enabled = True
                # Add checkpoint resume to command
                config.command = f"{config.command} --resume-from {latest.checkpoint_path}"
        
        elif self.config.strategy == RecoveryStrategy.RESUBMIT:
            # Modify job name to indicate recovery
            config.name = f"{config.name}_retry{record.recovery_count}"
        
        # Increase resources if needed
        if record.recovery_count > 1:
            # Try with more resources on subsequent retries
            config.resources.walltime = self._increase_walltime(
                config.resources.walltime
            )
        
        return config
    
    def _increase_walltime(self, walltime: str) -> str:
        """Increase walltime for retry."""
        # Parse walltime string (HH:MM:SS or MM:SS or SS)
        parts = walltime.split(':')
        
        if len(parts) == 3:
            hours = int(parts[0])
            return f"{hours + 1}:{parts[1]}:{parts[2]}"
        elif len(parts) == 2:
            minutes = int(parts[0])
            if minutes < 60:
                return f"{minutes + 10}:{parts[1]}"
            else:
                return f"{minutes // 60 + 1}:{minutes % 60:02d}:{parts[1]}"
        else:
            return walltime
    
    async def save_checkpoint(
        self,
        job_id: str,
        checkpoint_path: str,
        step: int = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Record a checkpoint for a job."""
        async with self._lock:
            if job_id not in self._jobs:
                return
            
            record = self._jobs[job_id]
            
            checkpoint = CheckpointInfo(
                job_id=job_id,
                timestamp=datetime.now(),
                checkpoint_path=checkpoint_path,
                step=step or len(record.checkpoints),
                metadata=metadata or {}
            )
            
            record.checkpoints.append(checkpoint)
            
            # Keep only recent checkpoints
            if len(record.checkpoints) > self.config.checkpoint_keep_count:
                record.checkpoints = record.checkpoints[-self.config.checkpoint_keep_count:]
            
            await self._save_state()
    
    async def get_latest_checkpoint(self, job_id: str) -> Optional[CheckpointInfo]:
        """Get the latest checkpoint for a job."""
        async with self._lock:
            if job_id not in self._jobs:
                return None
            
            checkpoints = self._jobs[job_id].checkpoints
            return checkpoints[-1] if checkpoints else None
    
    async def stop_monitoring(self, job_id: str) -> None:
        """Stop monitoring a job."""
        if job_id in self._monitoring_tasks:
            self._monitoring_tasks[job_id].cancel()
            del self._monitoring_tasks[job_id]
    
    async def _save_state(self) -> None:
        """Save recovery state to file."""
        try:
            state = {
                'jobs': {jid: record.to_dict() for jid, record in self._jobs.items()},
                'timestamp': datetime.now().isoformat(),
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save recovery state: {e}")
    
    async def load_state(self) -> None:
        """Load recovery state from file."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            for job_id, data in state.get('jobs', {}).items():
                self._jobs[job_id] = JobRecord.from_dict(data)
            
            logger.info(f"Loaded recovery state for {len(self._jobs)} jobs")
        
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Failed to load recovery state: {e}")
    
    def get_job_record(self, job_id: str) -> Optional[JobRecord]:
        """Get job record by ID."""
        return self._jobs.get(job_id)
    
    def list_jobs(self) -> List[str]:
        """List all registered job IDs."""
        return list(self._jobs.keys())
    
    async def cleanup_completed(self, max_age: timedelta = None) -> None:
        """Remove old completed job records."""
        max_age = max_age or timedelta(days=7)
        cutoff = datetime.now() - max_age
        
        async with self._lock:
            to_remove = []
            for job_id, record in self._jobs.items():
                if (record.current_status in [JobStatus.COMPLETED, JobStatus.CANCELLED]
                    and record.submit_time < cutoff):
                    to_remove.append(job_id)
            
            for job_id in to_remove:
                del self._jobs[job_id]
            
            if to_remove:
                await self._save_state()
                logger.info(f"Cleaned up {len(to_remove)} old job records")


class CheckpointManager:
    """Manager for job checkpointing."""
    
    def __init__(self, scheduler: BaseJobScheduler):
        self.scheduler = scheduler
    
    async def list_checkpoints(self, job_id: str, checkpoint_dir: str) -> List[Dict[str, Any]]:
        """List available checkpoints for a job."""
        result = await self.scheduler.connector.execute(
            f"find {checkpoint_dir} -name '*checkpoint*' -type f 2>/dev/null | sort -r"
        )
        
        if result['exit_code'] != 0:
            return []
        
        checkpoints = []
        for line in result['stdout'].strip().split('\n'):
            if line:
                # Get file info
                stat_result = await self.scheduler.connector.execute(
                    f"stat -c '%s|%Y' {line}"
                )
                if stat_result['exit_code'] == 0:
                    parts = stat_result['stdout'].strip().split('|')
                    if len(parts) == 2:
                        checkpoints.append({
                            'path': line,
                            'size': int(parts[0]),
                            'mtime': datetime.fromtimestamp(int(parts[1])).isoformat(),
                        })
        
        return checkpoints
    
    async def restore_checkpoint(
        self,
        job_id: str,
        checkpoint_path: str,
        restore_dir: str
    ) -> bool:
        """Restore a checkpoint to a directory."""
        try:
            # Create restore directory
            await self.scheduler.connector.execute(f"mkdir -p {restore_dir}")
            
            # Copy checkpoint files
            result = await self.scheduler.connector.execute(
                f"cp -r {checkpoint_path}/* {restore_dir}/"
            )
            
            return result['exit_code'] == 0
        
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    async def cleanup_old_checkpoints(
        self,
        checkpoint_dir: str,
        keep_count: int = 3
    ) -> None:
        """Clean up old checkpoint files."""
        checkpoints = await self.list_checkpoints("", checkpoint_dir)
        
        if len(checkpoints) > keep_count:
            to_delete = checkpoints[keep_count:]
            for checkpoint in to_delete:
                await self.scheduler.connector.execute(
                    f"rm -rf {checkpoint['path']}"
                )
            
            logger.info(f"Cleaned up {len(to_delete)} old checkpoints")
