"""
API Gateway Router

Main API endpoints for DFT+LAMMPS platform
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks, Request
from pydantic import BaseModel, Field

from api_platform.gateway.main import gateway, limiter
from api_platform.auth.permissions import PermissionChecker, Permission

checker = PermissionChecker()
router = APIRouter()


# ============== Request/Response Schemas ==============

class ProjectCreate(BaseModel):
    """Create a new project"""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, max_length=2000, description="Project description")
    project_type: str = Field("battery_screening", description="Type of project")
    target_properties: Dict[str, Any] = Field(default_factory=dict, description="Target material properties")
    material_system: Optional[str] = Field(None, description="Material system (e.g., Li-S, Li-P-S)")
    tags: List[str] = Field(default_factory=list, description="Project tags")


class ProjectUpdate(BaseModel):
    """Update project"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[str] = None
    target_properties: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class ProjectResponse(BaseModel):
    """Project response"""
    id: str
    name: str
    description: Optional[str]
    project_type: str
    status: str
    target_properties: Dict[str, Any]
    material_system: Optional[str]
    tags: List[str]
    total_structures: int
    completed_calculations: int
    failed_calculations: int
    created_at: datetime
    updated_at: Optional[datetime]
    owner_id: str
    
    class Config:
        from_attributes = True


class ProjectList(BaseModel):
    """List of projects"""
    items: List[ProjectResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class CalculationSubmit(BaseModel):
    """Submit a calculation"""
    structure: Dict[str, Any] = Field(..., description="Atomic structure data")
    calculation_type: str = Field(..., description="Type of calculation (dft, lammps, ml)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Calculation parameters")
    priority: int = Field(5, ge=1, le=10, description="Priority (1-10)")
    

class CalculationResponse(BaseModel):
    """Calculation response"""
    id: str
    project_id: str
    calculation_type: str
    status: str
    structure: Dict[str, Any]
    parameters: Dict[str, Any]
    priority: int
    results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class BatchSubmit(BaseModel):
    """Submit batch calculations"""
    project_id: str
    calculations: List[CalculationSubmit] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    """Batch submission response"""
    batch_id: str
    project_id: str
    total: int
    submitted: int
    failed: int
    calculation_ids: List[str]
    estimated_completion: Optional[datetime]


class StructureUpload(BaseModel):
    """Upload structure data"""
    name: str
    format: str = Field(..., description="File format (poscar, cif, xyz, json)")
    data: str = Field(..., description="Structure data (base64 encoded or raw)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StructureResponse(BaseModel):
    """Structure response"""
    id: str
    name: str
    format: str
    cell: List[List[float]]
    species: List[str]
    positions: List[List[float]]
    metadata: Dict[str, Any]
    created_at: datetime


class FilterParams(BaseModel):
    """Common filter parameters"""
    status: Optional[str] = None
    project_type: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    search: Optional[str] = None


# ============== Projects Endpoints ==============

@router.get(
    "/projects",
    response_model=ProjectList,
    summary="List projects",
    description="Get a paginated list of projects with optional filtering"
)
@limiter.limit("60/minute")
async def list_projects(
    request: Request,
    status: Optional[str] = None,
    project_type: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = Query("created_at", enum=["name", "created_at", "updated_at", "status"]),
    sort_order: str = Query("desc", enum=["asc", "desc"]),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    List all projects for the authenticated user.
    
    - **status**: Filter by project status
    - **project_type**: Filter by project type
    - **search**: Search in name and description
    - **sort_by**: Sort field
    - **sort_order**: Sort direction
    """
    # In production: query database
    # For now, return mock data
    return ProjectList(
        items=[],
        total=0,
        page=page,
        page_size=page_size,
        has_next=False,
        has_prev=False,
    )


@router.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create project",
    description="Create a new research project"
)
@limiter.limit("30/minute")
async def create_project(
    request: Request,
    project_data: ProjectCreate,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Create a new project"""
    # In production: save to database
    project_id = f"proj_{datetime.utcnow().timestamp()}"
    
    return ProjectResponse(
        id=project_id,
        name=project_data.name,
        description=project_data.description,
        project_type=project_data.project_type,
        status="created",
        target_properties=project_data.target_properties,
        material_system=project_data.material_system,
        tags=project_data.tags,
        total_structures=0,
        completed_calculations=0,
        failed_calculations=0,
        created_at=datetime.utcnow(),
        updated_at=None,
        owner_id=auth["client_id"],
    )


@router.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    summary="Get project",
    description="Get project details by ID"
)
@limiter.limit("60/minute")
async def get_project(
    request: Request,
    project_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Get project by ID"""
    # In production: query database
    raise HTTPException(status_code=404, detail="Project not found")


@router.patch(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    summary="Update project",
    description="Update project details"
)
@limiter.limit("30/minute")
async def update_project(
    request: Request,
    project_id: str,
    project_data: ProjectUpdate,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Update project"""
    # In production: update database
    raise HTTPException(status_code=404, detail="Project not found")


@router.delete(
    "/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete project",
    description="Delete a project and all associated data"
)
@limiter.limit("10/minute")
async def delete_project(
    request: Request,
    project_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Delete project"""
    # Check permission
    if not checker.has_permission(auth.get("permissions", []), Permission.DELETE_RESOURCES):
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # In production: delete from database
    return None


# ============== Calculations Endpoints ==============

@router.get(
    "/projects/{project_id}/calculations",
    response_model=List[CalculationResponse],
    summary="List calculations",
    description="List all calculations for a project"
)
@limiter.limit("60/minute")
async def list_calculations(
    request: Request,
    project_id: str,
    status: Optional[str] = None,
    calculation_type: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    auth: dict = Depends(gateway.authenticate_request)
):
    """List calculations for a project"""
    return []


@router.post(
    "/projects/{project_id}/calculations",
    response_model=CalculationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit calculation",
    description="Submit a new calculation"
)
@limiter.limit("30/minute")
async def submit_calculation(
    request: Request,
    project_id: str,
    calc_data: CalculationSubmit,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    Submit a calculation for processing.
    
    Returns immediately with calculation ID. Use the status endpoint to check progress.
    """
    calc_id = f"calc_{datetime.utcnow().timestamp()}"
    
    # In production: queue the calculation
    # background_tasks.add_task(process_calculation, calc_id, calc_data)
    
    return CalculationResponse(
        id=calc_id,
        project_id=project_id,
        calculation_type=calc_data.calculation_type,
        status="queued",
        structure=calc_data.structure,
        parameters=calc_data.parameters,
        priority=calc_data.priority,
        results=None,
        error_message=None,
        created_at=datetime.utcnow(),
        started_at=None,
        completed_at=None,
    )


@router.get(
    "/calculations/{calculation_id}",
    response_model=CalculationResponse,
    summary="Get calculation",
    description="Get calculation status and results"
)
@limiter.limit("60/minute")
async def get_calculation(
    request: Request,
    calculation_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Get calculation by ID"""
    raise HTTPException(status_code=404, detail="Calculation not found")


@router.post(
    "/projects/{project_id}/calculations/batch",
    response_model=BatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch",
    description="Submit multiple calculations in a batch"
)
@limiter.limit("10/minute")
async def submit_batch(
    request: Request,
    project_id: str,
    batch_data: BatchSubmit,
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    Submit a batch of calculations.
    
    Maximum 100 calculations per batch.
    """
    # Check batch permission
    if not checker.has_permission(auth.get("permissions", []), Permission.BATCH_OPERATIONS):
        raise HTTPException(status_code=403, detail="Batch operations require upgrade")
    
    batch_id = f"batch_{datetime.utcnow().timestamp()}"
    calc_ids = [f"calc_{i}_{datetime.utcnow().timestamp()}" for i in range(len(batch_data.calculations))]
    
    return BatchResponse(
        batch_id=batch_id,
        project_id=project_id,
        total=len(batch_data.calculations),
        submitted=len(batch_data.calculations),
        failed=0,
        calculation_ids=calc_ids,
        estimated_completion=None,
    )


# ============== Structures Endpoints ==============

@router.post(
    "/projects/{project_id}/structures",
    response_model=StructureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload structure",
    description="Upload a crystal structure"
)
@limiter.limit("30/minute")
async def upload_structure(
    request: Request,
    project_id: str,
    structure: StructureUpload,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Upload structure to project"""
    structure_id = f"struct_{datetime.utcnow().timestamp()}"
    
    return StructureResponse(
        id=structure_id,
        name=structure.name,
        format=structure.format,
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        species=["Li", "S"],
        positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        metadata=structure.metadata,
        created_at=datetime.utcnow(),
    )


@router.get(
    "/structures/{structure_id}",
    response_model=StructureResponse,
    summary="Get structure",
    description="Get structure by ID"
)
@limiter.limit("60/minute")
async def get_structure(
    request: Request,
    structure_id: str,
    format: Optional[str] = Query(None, description="Export format"),
    auth: dict = Depends(gateway.authenticate_request)
):
    """Get structure by ID"""
    raise HTTPException(status_code=404, detail="Structure not found")


# ============== Results Endpoints ==============

@router.get(
    "/calculations/{calculation_id}/results",
    summary="Get results",
    description="Get calculation results"
)
@limiter.limit("60/minute")
async def get_results(
    request: Request,
    calculation_id: str,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Get calculation results"""
    return {"calculation_id": calculation_id, "results": {}}


@router.get(
    "/projects/{project_id}/results/export",
    summary="Export results",
    description="Export all project results"
)
@limiter.limit("10/minute")
async def export_results(
    request: Request,
    project_id: str,
    format: str = Query("json", enum=["json", "csv", "xlsx", "hdf5"]),
    auth: dict = Depends(gateway.authenticate_request)
):
    """
    Export project results in various formats.
    
    - **json**: JSON format (default)
    - **csv**: CSV format
    - **xlsx**: Excel format
    - **hdf5**: HDF5 format for large datasets
    """
    # Check export permission
    if not checker.has_permission(auth.get("permissions", []), Permission.EXPORT_DATA):
        raise HTTPException(status_code=403, detail="Export requires upgrade")
    
    return {"project_id": project_id, "format": format, "download_url": ""}


# ============== Usage Endpoints ==============

@router.get(
    "/usage",
    summary="Get usage stats",
    description="Get API usage statistics"
)
@limiter.limit("30/minute")
async def get_usage(
    request: Request,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    auth: dict = Depends(gateway.authenticate_request)
):
    """Get usage statistics for the authenticated user"""
    # Check permission
    if not checker.has_permission(auth.get("permissions", []), Permission.VIEW_USAGE):
        raise HTTPException(status_code=403, detail="Usage viewing requires upgrade")
    
    return {
        "client_id": auth["client_id"],
        "tier": auth["tier"],
        "period": {
            "start": start_date,
            "end": end_date,
        },
        "requests": {
            "total": 0,
            "successful": 0,
            "failed": 0,
        },
        "calculations": {
            "total": 0,
            "completed": 0,
            "failed": 0,
        },
        "storage": {
            "used_gb": 0,
            "limit_gb": checker.get_tier_limits(auth["tier"])["storage_gb"],
        },
    }
