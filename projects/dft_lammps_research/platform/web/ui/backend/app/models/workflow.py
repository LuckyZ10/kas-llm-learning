"""
Workflow Model - Workflow definitions and instances
"""
import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING, Dict, Any
from sqlalchemy import String, Text, DateTime, ForeignKey, Integer, JSON, Boolean, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.db.database import Base

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.task import Task


class WorkflowType(str, enum.Enum):
    ACTIVE_LEARNING = "active_learning"
    HIGH_THROUGHPUT = "high_throughput"
    ML_TRAINING = "ml_training"
    MD_SIMULATION = "md_simulation"
    DFT_CALCULATION = "dft_calculation"
    CUSTOM = "custom"


class WorkflowStatus(str, enum.Enum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Workflow(Base):
    __tablename__ = "workflows"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Workflow type and status
    workflow_type: Mapped[WorkflowType] = mapped_column(Enum(WorkflowType))
    status: Mapped[WorkflowStatus] = mapped_column(Enum(WorkflowStatus), default=WorkflowStatus.DRAFT)
    
    # Workflow definition (node graph)
    definition: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)  # Node graph definition
    
    # Execution state
    current_node_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    execution_context: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)  # Runtime variables
    
    # Statistics
    total_tasks: Mapped[int] = mapped_column(Integer, default=0)
    completed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    failed_tasks: Mapped[int] = mapped_column(Integer, default=0)
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Resource usage
    cpu_time_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    memory_gb: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Foreign keys
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    parent_workflow_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("workflows.id"), nullable=True)
    
    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="workflows")
    tasks: Mapped[List["Task"]] = relationship("Task", back_populates="workflow", cascade="all, delete-orphan")
    sub_workflows: Mapped[List["Workflow"]] = relationship("Workflow", backref="parent", remote_side=[id])
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "workflow_type": self.workflow_type.value,
            "status": self.status.value,
            "definition": self.definition,
            "current_node_id": self.current_node_id,
            "execution_context": self.execution_context,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "progress_percent": self.progress_percent,
            "cpu_time_seconds": self.cpu_time_seconds,
            "memory_gb": self.memory_gb,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "project_id": self.project_id,
            "parent_workflow_id": self.parent_workflow_id,
        }
