"""
Workflow API Endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.workflow import Workflow, WorkflowStatus, WorkflowType
from app.models.user import User
from app.schemas.workflow import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse, WorkflowList,
    WorkflowNode, WorkflowExecutionRequest
)
from app.services.auth import get_current_user
from app.services.workflow_engine import WorkflowEngine

router = APIRouter()


@router.get("", response_model=WorkflowList)
async def list_workflows(
    project_id: Optional[str] = None,
    workflow_type: Optional[WorkflowType] = None,
    status: Optional[WorkflowStatus] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List workflows with optional filtering"""
    query = select(Workflow)
    
    if project_id:
        query = query.where(Workflow.project_id == project_id)
    
    if workflow_type:
        query = query.where(Workflow.workflow_type == workflow_type)
    
    if status:
        query = query.where(Workflow.status == status)
    
    # Get total count
    count_result = await db.execute(query)
    total = len(count_result.scalars().all())
    
    # Get paginated results
    query = query.order_by(desc(Workflow.created_at)).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    workflows = result.scalars().all()
    
    return WorkflowList(
        items=[WorkflowResponse(**w.to_dict()) for w in workflows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    workflow_data: WorkflowCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new workflow"""
    workflow = Workflow(
        name=workflow_data.name,
        description=workflow_data.description,
        workflow_type=workflow_data.workflow_type,
        project_id=workflow_data.project_id,
        definition=workflow_data.definition.model_dump() if workflow_data.definition else {"nodes": [], "edges": []},
    )
    
    db.add(workflow)
    await db.flush()
    await db.refresh(workflow)
    
    return WorkflowResponse(**workflow.to_dict())


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get workflow by ID"""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return WorkflowResponse(**workflow.to_dict())


@router.patch("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    workflow_data: WorkflowUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update workflow"""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    update_data = workflow_data.model_dump(exclude_unset=True)
    if "definition" in update_data and update_data["definition"]:
        update_data["definition"] = update_data["definition"].model_dump()
    
    for field, value in update_data.items():
        setattr(workflow, field, value)
    
    await db.flush()
    await db.refresh(workflow)
    
    return WorkflowResponse(**workflow.to_dict())


@router.post("/{workflow_id}/execute", response_model=WorkflowResponse)
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecutionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start workflow execution"""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow.status not in [WorkflowStatus.DRAFT, WorkflowStatus.PAUSED, WorkflowStatus.FAILED]:
        raise HTTPException(status_code=400, detail=f"Cannot execute workflow in {workflow.status} state")
    
    # Start workflow execution
    engine = WorkflowEngine(db)
    await engine.start_workflow(workflow, request.initial_context)
    
    return WorkflowResponse(**workflow.to_dict())


@router.post("/{workflow_id}/pause", response_model=WorkflowResponse)
async def pause_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Pause workflow execution"""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow.status != WorkflowStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Can only pause running workflows")
    
    workflow.status = WorkflowStatus.PAUSED
    await db.flush()
    await db.refresh(workflow)
    
    return WorkflowResponse(**workflow.to_dict())


@router.post("/{workflow_id}/cancel", response_model=WorkflowResponse)
async def cancel_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Cancel workflow execution"""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if workflow.status not in [WorkflowStatus.RUNNING, WorkflowStatus.QUEUED, WorkflowStatus.PAUSED]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel workflow in {workflow.status} state")
    
    workflow.status = WorkflowStatus.CANCELLED
    await db.flush()
    await db.refresh(workflow)
    
    return WorkflowResponse(**workflow.to_dict())


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete workflow"""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    await db.delete(workflow)
    await db.flush()
    
    return None
