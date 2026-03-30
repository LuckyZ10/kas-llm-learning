"""
Pydantic Schemas for Tasks
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class TaskBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    task_type: str


class TaskCreate(TaskBase):
    workflow_id: str
    working_directory: str
    command: Optional[str] = None
    script_content: Optional[str] = None
    input_files: Dict[str, Any] = Field(default_factory=dict)
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 5
    node_id: Optional[str] = None


class TaskUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    exit_code: Optional[int] = None
    cpu_time_seconds: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


class TaskResponse(TaskBase):
    id: str
    status: str
    priority: int
    workflow_id: str
    working_directory: str
    input_files: Dict[str, Any]
    output_files: Dict[str, Any]
    result_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    exit_code: Optional[int]
    cpu_time_seconds: float
    memory_peak_mb: float
    gpu_memory_mb: float
    retry_count: int
    max_retries: int
    node_id: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    queued_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    parent_task_id: Optional[str]
    
    class Config:
        from_attributes = True


class TaskList(BaseModel):
    items: list[TaskResponse]
    total: int
    page: int
    page_size: int
