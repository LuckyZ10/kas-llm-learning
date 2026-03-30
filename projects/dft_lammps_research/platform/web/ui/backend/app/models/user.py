"""
User Model - User authentication and authorization
"""
import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import String, Text, DateTime, Boolean, JSON, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.db.database import Base

if TYPE_CHECKING:
    from app.models.project import Project


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class User(Base):
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Security
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.RESEARCHER)
    
    # Profile
    avatar_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # API access
    api_key: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True, index=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    projects: Mapped[List["Project"]] = relationship("Project", back_populates="owner")
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        data = {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "role": self.role.value,
            "avatar_url": self.avatar_url,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }
        if include_sensitive:
            data["api_key"] = self.api_key
        return data
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        if self.is_superuser:
            return True
        
        role_permissions = {
            UserRole.ADMIN: ["*"],
            UserRole.RESEARCHER: ["project:create", "project:read", "project:update", 
                                  "workflow:create", "workflow:read", "workflow:update",
                                  "task:create", "task:read", "task:update"],
            UserRole.VIEWER: ["project:read", "workflow:read", "task:read"],
        }
        
        permissions = role_permissions.get(self.role, [])
        return "*" in permissions or permission in permissions
