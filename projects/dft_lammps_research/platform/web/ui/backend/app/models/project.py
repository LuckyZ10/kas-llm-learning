"""
Project Model - Research project management
"""
import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, Text, DateTime, ForeignKey, Integer, Float, JSON, Boolean, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.db.database import Base

if TYPE_CHECKING:
    from app.models.workflow import Workflow
    from app.models.user import User


class ProjectStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class Project(Base):
    __tablename__ = "projects"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Project metadata
    status: Mapped[ProjectStatus] = mapped_column(Enum(ProjectStatus), default=ProjectStatus.DRAFT)
    project_type: Mapped[str] = mapped_column(String(50), default="battery_screening")  # battery_screening, alloy_design, etc.
    
    # Research parameters
    target_properties: Mapped[dict] = mapped_column(JSON, default=dict)  # {"ionic_conductivity": {"min": 0.1, "max": 10}}
    material_system: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # Li-Mn-O, Si-anode, etc.
    
    # Paths
    work_directory: Mapped[str] = mapped_column(String(512), nullable=False)
    
    # Statistics
    total_structures: Mapped[int] = mapped_column(Integer, default=0)
    completed_calculations: Mapped[int] = mapped_column(Integer, default=0)
    failed_calculations: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Foreign keys
    owner_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="projects")
    workflows: Mapped[List["Workflow"]] = relationship("Workflow", back_populates="project", cascade="all, delete-orphan")
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "project_type": self.project_type,
            "target_properties": self.target_properties,
            "material_system": self.material_system,
            "work_directory": self.work_directory,
            "total_structures": self.total_structures,
            "completed_calculations": self.completed_calculations,
            "failed_calculations": self.failed_calculations,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "owner_id": self.owner_id,
        }
