"""
Pydantic Schemas for Projects
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    project_type: str = "battery_screening"
    target_properties: Dict[str, Any] = Field(default_factory=dict)
    material_system: Optional[str] = None
    work_directory: str


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[str] = None
    target_properties: Optional[Dict[str, Any]] = None


class ProjectResponse(ProjectBase):
    id: str
    status: str
    total_structures: int
    completed_calculations: int
    failed_calculations: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    owner_id: str
    
    class Config:
        from_attributes = True


class ProjectList(BaseModel):
    items: List[ProjectResponse]
    total: int
    page: int
    page_size: int
