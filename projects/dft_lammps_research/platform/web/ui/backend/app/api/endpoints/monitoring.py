"""
Monitoring API Endpoints - Real-time system and task monitoring
"""
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.task import Task
from app.models.workflow import Workflow
from app.models.user import User
from app.services.auth import get_current_user
from app.services.monitoring import MonitoringService

router = APIRouter()


@router.get("/stats")
async def get_system_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get overall system statistics"""
    service = MonitoringService(db)
    return await service.get_system_stats()


@router.get("/workflows/active")
async def get_active_workflows(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get currently active workflows"""
    service = MonitoringService(db)
    return await service.get_active_workflows(limit)


@router.get("/tasks/recent")
async def get_recent_tasks(
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get recent tasks"""
    service = MonitoringService(db)
    return await service.get_recent_tasks(status, limit)


@router.get("/resources")
async def get_resource_usage(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get resource usage statistics"""
    service = MonitoringService(db)
    return await service.get_resource_usage()


@router.get("/training")
async def get_training_metrics(
    model_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get ML training metrics"""
    service = MonitoringService(db)
    return await service.get_training_metrics(model_id)


@router.get("/md/{trajectory_id}")
async def get_md_metrics(
    trajectory_id: str,
    metric: str = Query("temperature", enum=["temperature", "energy", "pressure", "volume"]),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get MD simulation metrics"""
    service = MonitoringService(db)
    return await service.get_md_metrics(trajectory_id, metric)


@router.get("/al/progress")
async def get_active_learning_progress(
    project_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get active learning progress"""
    service = MonitoringService(db)
    return await service.get_al_progress(project_id)
