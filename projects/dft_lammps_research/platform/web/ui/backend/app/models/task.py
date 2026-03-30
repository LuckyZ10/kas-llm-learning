"""
Task Model - Individual computational tasks
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
from sqlalchemy import String, Text, DateTime, ForeignKey, Integer, Float, JSON, Boolean, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.db.database import Base

if TYPE_CHECKING:
    from app.models.workflow import Workflow


class TaskType(str, enum.Enum):
    DFT_SINGLE_POINT = "dft_single_point"
    DFT_RELAXATION = "dft_relaxation"
    DFT_MD = "dft_md"
    LAMMPS_MD = "lammps_md"
    LAMMPS_MINIMIZATION = "lammps_minimization"
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"
    STRUCTURE_GENERATION = "structure_generation"
    ANALYSIS = "analysis"
    DATA_EXPORT = "data_export"


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(int, enum.Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class Task(Base):
    __tablename__ = "tasks"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Task type and status
    task_type: Mapped[TaskType] = mapped_column(Enum(TaskType))
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus), default=TaskStatus.PENDING)
    priority: Mapped[TaskPriority] = mapped_column(Integer, default=TaskPriority.NORMAL)
    
    # Execution details
    command: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Shell command or script
    script_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Script content
    working_directory: Mapped[str] = mapped_column(String(512), nullable=False)
    
    # Configuration
    input_files: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    output_files: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    environment_variables: Mapped[Dict[str, str]] = mapped_column(JSON, default=dict)
    resource_requirements: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)  # CPU, memory, GPU
    
    # Results
    result_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Resource usage
    cpu_time_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    memory_peak_mb: Mapped[float] = mapped_column(Float, default=0.0)
    gpu_memory_mb: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Retry configuration
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Node reference (for workflow visualization)
    node_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Foreign keys
    workflow_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflows.id"), nullable=False)
    parent_task_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("tasks.id"), nullable=True)
    
    # Relationships
    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="tasks")
    child_tasks: Mapped[list["Task"]] = relationship("Task", backref="parent", remote_side=[id])
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "priority": self.priority,
            "working_directory": self.working_directory,
            "input_files": self.input_files,
            "output_files": self.output_files,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "exit_code": self.exit_code,
            "cpu_time_seconds": self.cpu_time_seconds,
            "memory_peak_mb": self.memory_peak_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "node_id": self.node_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "queued_at": self.queued_at.isoformat() if self.queued_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "workflow_id": self.workflow_id,
            "parent_task_id": self.parent_task_id,
        }
    
    def get_duration_seconds(self) -> Optional[float]:
        """Calculate task duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now(self.started_at.tzinfo) - self.started_at).total_seconds()
        return None
