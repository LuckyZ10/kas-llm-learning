"""
Task API Endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.task import Task, TaskStatus
from app.models.user import User
from app.schemas.task import TaskCreate, TaskUpdate, TaskResponse, TaskList
from app.services.auth import get_current_user

router = APIRouter()


@router.get("", response_model=TaskList)
async def list_tasks(
    workflow_id: Optional[str] = None,
    status: Optional[TaskStatus] = None,
    task_type: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List tasks with optional filtering"""
    query = select(Task)
    
    if workflow_id:
        query = query.where(Task.workflow_id == workflow_id)
    
    if status:
        query = query.where(Task.status == status)
    
    if task_type:
        query = query.where(Task.task_type == task_type)
    
    # Get total count
    count_result = await db.execute(query)
    total = len(count_result.scalars().all())
    
    # Get paginated results
    query = query.order_by(desc(Task.created_at)).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    return TaskList(
        items=[TaskResponse(**t.to_dict()) for t in tasks],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get task by ID"""
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskResponse(**task.to_dict())


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task_data: TaskCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new task"""
    task = Task(
        name=task_data.name,
        description=task_data.description,
        task_type=task_data.task_type,
        workflow_id=task_data.workflow_id,
        working_directory=task_data.working_directory,
        command=task_data.command,
        script_content=task_data.script_content,
        input_files=task_data.input_files,
        resource_requirements=task_data.resource_requirements,
        priority=task_data.priority,
        node_id=task_data.node_id,
    )
    
    db.add(task)
    await db.flush()
    await db.refresh(task)
    
    return TaskResponse(**task.to_dict())


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: str,
    task_data: TaskUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update task"""
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    update_data = task_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(task, field, value)
    
    await db.flush()
    await db.refresh(task)
    
    return TaskResponse(**task.to_dict())


@router.get("/{task_id}/logs")
async def get_task_logs(
    task_id: str,
    lines: int = Query(100, ge=1, le=10000),
    current_user: User = Depends(get_current_user),
):
    """Get task logs"""
    # This would read from log files
    # For now, return placeholder
    return {"task_id": task_id, "logs": [], "lines": lines}
