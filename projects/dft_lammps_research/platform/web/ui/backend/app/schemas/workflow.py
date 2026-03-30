"""
Pydantic Schemas for Workflows
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class Position(BaseModel):
    x: float
    y: float


class WorkflowNodeData(BaseModel):
    label: str
    description: Optional[str] = None
    node_type: str  # dft_calculation, md_simulation, ml_training, etc.
    config: Dict[str, Any] = Field(default_factory=dict)


class WorkflowNode(BaseModel):
    id: str
    type: str = "default"
    position: Position
    data: WorkflowNodeData


class WorkflowEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Optional[str] = "default"
    animated: bool = False
    label: Optional[str] = None


class WorkflowDefinition(BaseModel):
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]


class WorkflowBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    workflow_type: str


class WorkflowCreate(WorkflowBase):
    project_id: str
    definition: Optional[WorkflowDefinition] = None


class WorkflowUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[str] = None
    definition: Optional[WorkflowDefinition] = None
    execution_context: Optional[Dict[str, Any]] = None


class WorkflowExecutionRequest(BaseModel):
    initial_context: Dict[str, Any] = Field(default_factory=dict)
    start_node_id: Optional[str] = None


class WorkflowResponse(WorkflowBase):
    id: str
    status: str
    project_id: str
    definition: Dict[str, Any]
    execution_context: Dict[str, Any]
    current_node_id: Optional[str]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    progress_percent: float
    cpu_time_seconds: float
    memory_gb: float
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class WorkflowList(BaseModel):
    items: List[WorkflowResponse]
    total: int
    page: int
    page_size: int
