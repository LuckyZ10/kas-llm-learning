"""
Job configuration and status models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union
import json


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass
class ResourceRequest:
    """Resource request specification."""
    nodes: int = 1
    cores_per_node: int = 1
    memory_per_node: str = "4GB"
    gpus_per_node: int = 0
    walltime: str = "1:00:00"  # HH:MM:SS
    queue: Optional[str] = None
    partition: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "cores_per_node": self.cores_per_node,
            "memory_per_node": self.memory_per_node,
            "gpus_per_node": self.gpus_per_node,
            "walltime": self.walltime,
            "queue": self.queue,
            "partition": self.partition,
            "constraints": self.constraints,
        }


@dataclass
class JobConfig:
    """Job configuration."""
    name: str
    command: str
    work_dir: str
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    priority: JobPriority = JobPriority.NORMAL
    
    # Environment
    modules: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    
    # I/O
    stdin: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Job IDs to depend on
    
    # Checkpointing
    checkpoint_enabled: bool = False
    checkpoint_interval: int = 3600  # seconds
    checkpoint_dir: Optional[str] = None
    
    # Notifications
    notify_on_start: bool = False
    notify_on_complete: bool = False
    notify_email: Optional[str] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "work_dir": self.work_dir,
            "resources": self.resources.to_dict(),
            "priority": self.priority.value,
            "modules": self.modules,
            "environment": self.environment,
            "stdin": self.stdin,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "dependencies": self.dependencies,
            "checkpoint_enabled": self.checkpoint_enabled,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "notify_on_start": self.notify_on_start,
            "notify_on_complete": self.notify_on_complete,
            "notify_email": self.notify_email,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobConfig":
        resources = ResourceRequest(**data.get("resources", {}))
        priority = JobPriority(data.get("priority", JobPriority.NORMAL.value))
        
        return cls(
            name=data["name"],
            command=data["command"],
            work_dir=data["work_dir"],
            resources=resources,
            priority=priority,
            modules=data.get("modules", []),
            environment=data.get("environment", {}),
            stdin=data.get("stdin"),
            stdout=data.get("stdout"),
            stderr=data.get("stderr"),
            dependencies=data.get("dependencies", []),
            checkpoint_enabled=data.get("checkpoint_enabled", False),
            checkpoint_interval=data.get("checkpoint_interval", 3600),
            checkpoint_dir=data.get("checkpoint_dir"),
            notify_on_start=data.get("notify_on_start", False),
            notify_on_complete=data.get("notify_on_complete", False),
            notify_email=data.get("notify_email"),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class JobResult:
    """Job execution result."""
    job_id: str
    status: JobStatus
    
    # Timing
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Resource usage
    nodes_used: int = 0
    cores_used: int = 0
    memory_used: Optional[str] = None
    walltime_used: Optional[str] = None
    
    # Exit status
    exit_code: Optional[int] = None
    
    # Output
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    output_files: Dict[str, str] = field(default_factory=dict)
    
    # Error info
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "submit_time": self.submit_time.isoformat() if self.submit_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "nodes_used": self.nodes_used,
            "cores_used": self.cores_used,
            "memory_used": self.memory_used,
            "walltime_used": self.walltime_used,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "output_files": self.output_files,
            "error_message": self.error_message,
        }


@dataclass
class JobInfo:
    """Real-time job information."""
    job_id: str
    name: str
    status: JobStatus
    
    # Owner
    user: Optional[str] = None
    group: Optional[str] = None
    account: Optional[str] = None
    
    # Resources allocated
    nodes: List[str] = field(default_factory=list)
    cores: int = 0
    gpus: int = 0
    memory: Optional[str] = None
    
    # Queue info
    queue: Optional[str] = None
    priority: Optional[int] = None
    
    # Timing
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    estimated_start: Optional[datetime] = None
    time_limit: Optional[str] = None
    time_used: Optional[str] = None
    
    # Progress
    percent_complete: Optional[float] = None
    
    # Additional info
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status.value,
            "user": self.user,
            "group": self.group,
            "account": self.account,
            "nodes": self.nodes,
            "cores": self.cores,
            "gpus": self.gpus,
            "memory": self.memory,
            "queue": self.queue,
            "priority": self.priority,
            "submit_time": self.submit_time.isoformat() if self.submit_time else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "estimated_start": self.estimated_start.isoformat() if self.estimated_start else None,
            "time_limit": self.time_limit,
            "time_used": self.time_used,
            "percent_complete": self.percent_complete,
            "additional_info": self.additional_info,
        }
