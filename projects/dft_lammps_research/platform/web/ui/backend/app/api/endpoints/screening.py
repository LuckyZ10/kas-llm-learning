"""
Screening Results API Endpoints
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.screening_result import ScreeningResult
from app.models.user import User
from app.schemas.screening import (
    ScreeningResultResponse, ScreeningResultList,
    ScreeningFilter, ScreeningComparison
)
from app.services.auth import get_current_user

router = APIRouter()


@router.get("", response_model=ScreeningResultList)
async def list_screening_results(
    project_id: Optional[str] = None,
    formula: Optional[str] = None,
    min_ionic_conductivity: Optional[float] = None,
    max_formation_energy: Optional[float] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = "overall_score",
    sort_order: str = "desc",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List screening results with filtering"""
    query = select(ScreeningResult)
    
    if project_id:
        query = query.where(ScreeningResult.project_id == project_id)
    
    if formula:
        query = query.where(ScreeningResult.formula.ilike(f"%{formula}%"))
    
    if min_ionic_conductivity:
        query = query.where(ScreeningResult.ionic_conductivity >= min_ionic_conductivity)
    
    if max_formation_energy:
        query = query.where(ScreeningResult.formation_energy <= max_formation_energy)
    
    # Get total count
    count_result = await db.execute(query)
    total = len(count_result.scalars().all())
    
    # Sorting
    if sort_order == "desc":
        query = query.order_by(desc(getattr(ScreeningResult, sort_by, ScreeningResult.overall_score)))
    else:
        query = query.order_by(getattr(ScreeningResult, sort_by, ScreeningResult.overall_score))
    
    # Pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    results = result.scalars().all()
    
    return ScreeningResultList(
        items=[ScreeningResultResponse(**r.to_dict()) for r in results],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{result_id}", response_model=ScreeningResultResponse)
async def get_screening_result(
    result_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get screening result by ID"""
    result = await db.execute(select(ScreeningResult).where(ScreeningResult.id == result_id))
    screening = result.scalar_one_or_none()
    
    if not screening:
        raise HTTPException(status_code=404, detail="Screening result not found")
    
    return ScreeningResultResponse(**screening.to_dict())


@router.post("/filter", response_model=ScreeningResultList)
async def filter_screening_results(
    filters: ScreeningFilter,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Advanced filtering of screening results"""
    query = select(ScreeningResult)
    
    conditions = []
    
    if filters.project_id:
        conditions.append(ScreeningResult.project_id == filters.project_id)
    
    if filters.formula_contains:
        conditions.append(ScreeningResult.formula.ilike(f"%{filters.formula_contains}%"))
    
    if filters.property_ranges:
        for prop, range_def in filters.property_ranges.items():
            column = getattr(ScreeningResult, prop, None)
            if column:
                if "min" in range_def:
                    conditions.append(column >= range_def["min"])
                if "max" in range_def:
                    conditions.append(column <= range_def["max"])
    
    if conditions:
        query = query.where(and_(*conditions))
    
    count_result = await db.execute(query)
    total = len(count_result.scalars().all())
    
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    results = result.scalars().all()
    
    return ScreeningResultList(
        items=[ScreeningResultResponse(**r.to_dict()) for r in results],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/compare")
async def compare_structures(
    comparison: ScreeningComparison,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compare multiple structures"""
    result = await db.execute(
        select(ScreeningResult).where(ScreeningResult.id.in_(comparison.structure_ids))
    )
    structures = result.scalars().all()
    
    if len(structures) != len(comparison.structure_ids):
        raise HTTPException(status_code=404, detail="Some structures not found")
    
    return {
        "structures": [s.to_dict() for s in structures],
        "properties": comparison.properties,
    }
