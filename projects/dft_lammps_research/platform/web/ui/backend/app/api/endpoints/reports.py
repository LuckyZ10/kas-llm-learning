"""
Report Generation API Endpoints
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.user import User
from app.services.auth import get_current_user
from app.services.report_generator import ReportGenerator

router = APIRouter()


@router.post("/project/{project_id}")
async def generate_project_report(
    project_id: str,
    format: str = Query("pdf", enum=["pdf", "html", "markdown"]),
    include_structures: bool = True,
    include_charts: bool = True,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate project report"""
    generator = ReportGenerator(db)
    
    report_path = await generator.generate_project_report(
        project_id=project_id,
        format=format,
        include_structures=include_structures,
        include_charts=include_charts,
    )
    
    if not report_path:
        raise HTTPException(status_code=404, detail="Project not found or report generation failed")
    
    return FileResponse(
        path=report_path,
        filename=f"project_report_{project_id}.{format}",
        media_type="application/pdf" if format == "pdf" else "text/html",
    )


@router.post("/workflow/{workflow_id}")
async def generate_workflow_report(
    workflow_id: str,
    format: str = Query("pdf", enum=["pdf", "html"]),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate workflow execution report"""
    generator = ReportGenerator(db)
    
    report_path = await generator.generate_workflow_report(
        workflow_id=workflow_id,
        format=format,
    )
    
    if not report_path:
        raise HTTPException(status_code=404, detail="Workflow not found or report generation failed")
    
    return FileResponse(
        path=report_path,
        filename=f"workflow_report_{workflow_id}.{format}",
        media_type="application/pdf" if format == "pdf" else "text/html",
    )


@router.post("/screening")
async def generate_screening_report(
    project_id: str,
    top_n: int = Query(50, ge=1, le=500),
    format: str = Query("pdf", enum=["pdf", "html", "csv"]),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate screening results report"""
    generator = ReportGenerator(db)
    
    report_path = await generator.generate_screening_report(
        project_id=project_id,
        top_n=top_n,
        format=format,
    )
    
    if not report_path:
        raise HTTPException(status_code=404, detail="No screening results found")
    
    return FileResponse(
        path=report_path,
        filename=f"screening_report_{project_id}.{format}",
        media_type="application/pdf" if format == "pdf" else "text/csv" if format == "csv" else "text/html",
    )
