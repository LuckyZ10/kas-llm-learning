#!/usr/bin/env python3
"""
DFT-LAMMPS Web API Server
=========================
FastAPI-based RESTful API for the DFT-LAMMPS platform.

Features:
- Task submission and management
- Status tracking
- Results retrieval
- File upload/download
- Authentication and authorization

Author: DFT-LAMMPS Web Team
Version: 1.0.0
"""

import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import logging

# FastAPI
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Pydantic
from pydantic import BaseModel, Field, validator

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dftlammps.core.workflow import IntegratedMaterialsWorkflow, IntegratedWorkflowConfig
from dftlammps.core.workflow import MaterialsProjectConfig, DFTStageConfig
from dftlammps.core.workflow import MLPotentialConfig, MDStageConfig, AnalysisConfig
from dftlammps.core.dft_bridge import DFTToLAMMPSBridge
from dftlammps.core.md_simulation import MDSimulationRunner, MDConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results"))
STATIC_DIR = Path(os.environ.get("STATIC_DIR", "./static"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Pydantic Models
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime

class TaskStatus(str):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowType(str):
    FULL = "full_workflow"           # Structure → DFT → ML → MD → Analysis
    DFT_ONLY = "dft_only"            # DFT calculation only
    ML_TRAINING = "ml_training"      # ML potential training
    MD_SIMULATION = "md_simulation"  # MD simulation only
    SCREENING = "screening"          # High-throughput screening
    FITTING = "force_field_fitting"  # Force field parameter fitting

class DFTConfig(BaseModel):
    code: str = Field(default="vasp", description="DFT code: vasp, espresso, abacus")
    functional: str = Field(default="PBE", description="Exchange-correlation functional")
    encut: float = Field(default=520, ge=200, le=1000, description="Energy cutoff in eV")
    kpoints_density: float = Field(default=0.25, gt=0, le=1.0, description="K-point density")
    ediff: float = Field(default=1e-6, gt=0, description="Electronic convergence")
    ncores: int = Field(default=4, ge=1, le=128, description="Number of CPU cores")
    max_steps: int = Field(default=200, ge=10, le=1000, description="Maximum ionic steps")

class MLConfig(BaseModel):
    framework: str = Field(default="deepmd", description="ML framework: deepmd, nep, mace")
    preset: str = Field(default="fast", description="Training preset: fast, accurate, light")
    num_models: int = Field(default=4, ge=1, le=10, description="Number of ensemble models")
    max_iterations: int = Field(default=10, ge=1, le=50, description="Active learning iterations")

class MDConfigInput(BaseModel):
    ensemble: str = Field(default="nvt", description="Ensemble: nve, nvt, npt")
    temperatures: List[float] = Field(default=[300, 500, 700], description="Temperatures in K")
    timestep: float = Field(default=1.0, gt=0, le=5.0, description="Timestep in fs")
    nsteps_equil: int = Field(default=50000, ge=1000, description="Equilibration steps")
    nsteps_prod: int = Field(default=100000, ge=10000, description="Production steps")
    nprocs: int = Field(default=4, ge=1, le=64, description="Number of processors")

class AnalysisConfigInput(BaseModel):
    compute_diffusion: bool = Field(default=True)
    compute_conductivity: bool = Field(default=True)
    compute_activation_energy: bool = Field(default=True)
    compute_vibration: bool = Field(default=False)

class SubmitTaskRequest(BaseModel):
    workflow_type: str = Field(..., description="Type of workflow to run")
    name: str = Field(..., min_length=1, max_length=100, description="Task name")
    description: Optional[str] = Field(default=None, max_length=500)
    
    # Input structure
    material_id: Optional[str] = Field(default=None, description="Materials Project ID (mp-XXXX)")
    formula: Optional[str] = Field(default=None, description="Chemical formula")
    structure_file_id: Optional[str] = Field(default=None, description="Uploaded structure file ID")
    
    # Workflow stages configuration
    dft_config: Optional[DFTConfig] = Field(default_factory=DFTConfig)
    ml_config: Optional[MLConfig] = Field(default_factory=MLConfig)
    md_config: Optional[MDConfigInput] = Field(default_factory=MDConfigInput)
    analysis_config: Optional[AnalysisConfigInput] = Field(default_factory=AnalysisConfigInput)
    
    # Stage control
    skip_dft: bool = Field(default=False)
    skip_ml: bool = Field(default=False)
    skip_md: bool = Field(default=False)
    skip_analysis: bool = Field(default=False)
    
    # Resources
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1-10)")
    max_runtime_hours: float = Field(default=24.0, gt=0, le=168, description="Maximum runtime in hours")

class TaskInfo(BaseModel):
    task_id: str
    name: str
    workflow_type: str
    status: str
    priority: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = Field(default=0.0, ge=0, le=100)
    message: Optional[str] = None
    error_message: Optional[str] = None
    runtime_seconds: Optional[float] = None

class TaskDetail(TaskInfo):
    description: Optional[str] = None
    config: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    logs: Optional[List[str]] = None
    files: Optional[List[str]] = None

class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]
    total: int
    page: int
    page_size: int

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    content_type: str
    uploaded_at: datetime

class CalculationResult(BaseModel):
    task_id: str
    workflow_type: str
    status: str
    results: Dict[str, Any]
    download_url: Optional[str] = None

# =============================================================================
# In-Memory Task Store (Replace with database in production)
# =============================================================================

class TaskStore:
    """Simple in-memory task storage. Replace with database for production."""
    
    def __init__(self):
        self._tasks: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def create_task(self, task_id: str, data: Dict) -> Dict:
        async with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "status": TaskStatus.PENDING,
                "progress": 0.0,
                "created_at": datetime.utcnow(),
                "started_at": None,
                "completed_at": None,
                "message": "Task created",
                "error_message": None,
                "runtime_seconds": None,
                "results": None,
                "logs": [],
                **data
            }
            return self._tasks[task_id]
    
    async def get_task(self, task_id: str) -> Optional[Dict]:
        async with self._lock:
            return self._tasks.get(task_id)
    
    async def update_task(self, task_id: str, updates: Dict) -> Optional[Dict]:
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(updates)
                return self._tasks[task_id]
            return None
    
    async def list_tasks(self, status: Optional[str] = None, 
                        limit: int = 20, offset: int = 0) -> tuple:
        async with self._lock:
            tasks = list(self._tasks.values())
            if status:
                tasks = [t for t in tasks if t["status"] == status]
            # Sort by created_at descending
            tasks.sort(key=lambda x: x["created_at"], reverse=True)
            total = len(tasks)
            tasks = tasks[offset:offset + limit]
            return tasks, total
    
    async def delete_task(self, task_id: str) -> bool:
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False

task_store = TaskStore()

# =============================================================================
# Authentication (Simplified - use proper auth in production)
# =============================================================================

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token. In production, use proper JWT validation."""
    token = credentials.credentials
    # Simplified: accept any non-empty token
    # In production, validate JWT, check expiration, user permissions, etc.
    if not token or token == "undefined":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return {"user_id": "user_123", "role": "researcher"}

# =============================================================================
# Background Task Worker
# =============================================================================

async def run_workflow_task(task_id: str, config: SubmitTaskRequest):
    """Background task to run the workflow."""
    
    try:
        # Update status to running
        await task_store.update_task(task_id, {
            "status": TaskStatus.RUNNING,
            "started_at": datetime.utcnow(),
            "message": "Initializing workflow"
        })
        
        logger.info(f"Starting workflow task {task_id}: {config.name}")
        
        # Create workflow configuration
        workflow_config = IntegratedWorkflowConfig(
            workflow_name=config.name,
            working_dir=str(RESULTS_DIR / task_id),
            dft_config=DFTStageConfig(
                code=config.dft_config.code,
                functional=config.dft_config.functional,
                encut=config.dft_config.encut,
                kpoints_density=config.dft_config.kpoints_density,
                ediff=config.dft_config.ediff,
                ncores=config.dft_config.ncores,
                max_steps=config.dft_config.max_steps
            ),
            ml_config=MLPotentialConfig(
                framework=config.ml_config.framework,
                preset=config.ml_config.preset,
                num_models=config.ml_config.num_models,
                max_iterations=config.ml_config.max_iterations
            ),
            md_config=MDStageConfig(
                ensemble=config.md_config.ensemble,
                temperatures=config.md_config.temperatures,
                timestep=config.md_config.timestep,
                nsteps_equil=config.md_config.nsteps_equil,
                nsteps_prod=config.md_config.nsteps_prod,
                nprocs=config.md_config.nprocs
            ),
            analysis_config=AnalysisConfig(
                compute_diffusion=config.analysis_config.compute_diffusion,
                compute_conductivity=config.analysis_config.compute_conductivity,
                compute_activation_energy=config.analysis_config.compute_activation_energy,
                compute_vibration=config.analysis_config.compute_vibration
            )
        )
        
        # Configure stages
        workflow_config.stages["dft_calculation"].enabled = not config.skip_dft
        workflow_config.stages["ml_training"].enabled = not config.skip_ml
        workflow_config.stages["md_simulation"].enabled = not config.skip_md
        workflow_config.stages["analysis"].enabled = not config.skip_analysis
        
        # Create and run workflow
        workflow = IntegratedMaterialsWorkflow(workflow_config)
        
        # Progress updates
        await task_store.update_task(task_id, {
            "progress": 10.0,
            "message": "Fetching structure"
        })
        
        # Run workflow
        results = await asyncio.to_thread(
            workflow.run,
            material_id=config.material_id,
            formula=config.formula
        )
        
        # Convert results to JSON-serializable format
        serializable_results = convert_to_serializable(results)
        
        # Update task as completed
        await task_store.update_task(task_id, {
            "status": TaskStatus.COMPLETED,
            "completed_at": datetime.utcnow(),
            "progress": 100.0,
            "message": "Workflow completed successfully",
            "runtime_seconds": (datetime.utcnow() - 
                              (await task_store.get_task(task_id))["started_at"]).total_seconds(),
            "results": serializable_results
        })
        
        logger.info(f"Workflow task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Workflow task {task_id} failed: {str(e)}")
        await task_store.update_task(task_id, {
            "status": TaskStatus.FAILED,
            "completed_at": datetime.utcnow(),
            "error_message": str(e),
            "message": "Workflow failed"
        })

def convert_to_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'to_dict'):
        return convert_to_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj

# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting DFT-LAMMPS Web API Server")
    yield
    logger.info("Shutting down DFT-LAMMPS Web API Server")

app = FastAPI(
    title="DFT-LAMMPS Web API",
    description="""
    RESTful API for DFT-LAMMPS materials simulation platform.
    
    ## Features
    
    * **Workflow Management**: Submit and manage complex simulation workflows
    * **Structure Input**: Support for Materials Project IDs, formulas, and file uploads
    * **DFT Calculations**: VASP, Quantum ESPRESSO, and ABACUS support
    * **ML Potentials**: DeepMD, NEP, and MACE training and inference
    * **MD Simulations**: LAMMPS-based molecular dynamics with various ensembles
    * **Analysis**: Diffusion coefficients, conductivity, activation energy
    
    ## Authentication
    
    All endpoints require Bearer token authentication.
    """,
    version=API_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint for health check."""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        timestamp=datetime.utcnow()
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return await root()

# -----------------------------------------------------------------------------
# Task Management Endpoints
# -----------------------------------------------------------------------------

@app.post(f"{API_PREFIX}/tasks", response_model=TaskInfo, status_code=status.HTTP_202_ACCEPTED, tags=["Tasks"])
async def submit_task(
    request: SubmitTaskRequest,
    background_tasks: BackgroundTasks,
    user: Dict = Depends(verify_token)
):
    """
    Submit a new simulation task.
    
    The task will be queued and executed asynchronously.
    Use the returned task_id to track progress.
    """
    task_id = str(uuid.uuid4())
    
    # Create task record
    await task_store.create_task(task_id, {
        "name": request.name,
        "description": request.description,
        "workflow_type": request.workflow_type,
        "priority": request.priority,
        "user_id": user.get("user_id"),
        "config": request.dict()
    })
    
    # Start background task
    background_tasks.add_task(run_workflow_task, task_id, request)
    
    logger.info(f"Task {task_id} submitted by user {user.get('user_id')}")
    
    task = await task_store.get_task(task_id)
    return TaskInfo(**task)

@app.get(f"{API_PREFIX}/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    user: Dict = Depends(verify_token)
):
    """List all tasks with optional filtering."""
    offset = (page - 1) * page_size
    tasks, total = await task_store.list_tasks(status=status, limit=page_size, offset=offset)
    
    return TaskListResponse(
        tasks=[TaskInfo(**t) for t in tasks],
        total=total,
        page=page,
        page_size=page_size
    )

@app.get(f"{API_PREFIX}/tasks/{{task_id}}", response_model=TaskDetail, tags=["Tasks"])
async def get_task(
    task_id: str,
    user: Dict = Depends(verify_token)
):
    """Get detailed information about a specific task."""
    task = await task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    return TaskDetail(**task)

@app.delete(f"{API_PREFIX}/tasks/{{task_id}}", status_code=status.HTTP_204_NO_CONTENT, tags=["Tasks"])
async def delete_task(
    task_id: str,
    user: Dict = Depends(verify_token)
):
    """Delete a task and its associated results."""
    task = await task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    # Delete result files
    result_dir = RESULTS_DIR / task_id
    if result_dir.exists():
        import shutil
        shutil.rmtree(result_dir)
    
    await task_store.delete_task(task_id)
    logger.info(f"Task {task_id} deleted by user {user.get('user_id')}")

@app.post(f"{API_PREFIX}/tasks/{{task_id}}/cancel", response_model=TaskInfo, tags=["Tasks"])
async def cancel_task(
    task_id: str,
    user: Dict = Depends(verify_token)
):
    """Cancel a running or pending task."""
    task = await task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    if task["status"] not in [TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel task with status {task['status']}"
        )
    
    await task_store.update_task(task_id, {
        "status": TaskStatus.CANCELLED,
        "completed_at": datetime.utcnow(),
        "message": "Task cancelled by user"
    })
    
    logger.info(f"Task {task_id} cancelled by user {user.get('user_id')}")
    
    updated_task = await task_store.get_task(task_id)
    return TaskInfo(**updated_task)

# -----------------------------------------------------------------------------
# File Upload/Download Endpoints
# -----------------------------------------------------------------------------

@app.post(f"{API_PREFIX}/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    file: UploadFile = File(...),
    user: Dict = Depends(verify_token)
):
    """Upload a structure file (POSCAR, CIF, XYZ, etc.)."""
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    filename = f"{file_id}{file_ext}"
    filepath = UPLOAD_DIR / filename
    
    # Save file
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    
    file_size = len(content)
    
    logger.info(f"File uploaded: {file.filename} ({file_size} bytes) as {file_id}")
    
    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=file_size,
        content_type=file.content_type,
        uploaded_at=datetime.utcnow()
    )

@app.get(f"{API_PREFIX}/tasks/{{task_id}}/download/{{filename}}", tags=["Files"])
async def download_file(
    task_id: str,
    filename: str,
    user: Dict = Depends(verify_token)
):
    """Download a result file from a task."""
    task = await task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    file_path = RESULTS_DIR / task_id / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {filename} not found"
        )
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.get(f"{API_PREFIX}/tasks/{{task_id}}/results", response_model=CalculationResult, tags=["Results"])
async def get_results(
    task_id: str,
    user: Dict = Depends(verify_token)
):
    """Get the results of a completed task."""
    task = await task_store.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task is not completed. Current status: {task['status']}"
        )
    
    download_url = f"/api/v1/tasks/{task_id}/download/workflow_report.json"
    
    return CalculationResult(
        task_id=task_id,
        workflow_type=task["workflow_type"],
        status=task["status"],
        results=task.get("results", {}),
        download_url=download_url
    )

# -----------------------------------------------------------------------------
# Structure Query Endpoints
# -----------------------------------------------------------------------------

@app.get(f"{API_PREFIX}/structures/search", tags=["Structures"])
async def search_structures(
    query: str = Query(..., description="Search query (formula or material_id)"),
    limit: int = Query(10, ge=1, le=50),
    user: Dict = Depends(verify_token)
):
    """
    Search for structures in Materials Project.
    
    This is a proxy endpoint that queries the Materials Project API.
    """
    try:
        from mp_api.client import MPRester
        
        mpr = MPRester()
        
        # Try to determine if query is a material_id or formula
        if query.startswith("mp-"):
            docs = mpr.summary.search(material_ids=[query], fields=[
                "material_id", "formula_pretty", "structure",
                "energy_per_atom", "band_gap", "symmetry"
            ])
        else:
            docs = mpr.summary.search(formula=query, fields=[
                "material_id", "formula_pretty", "structure",
                "energy_per_atom", "band_gap", "symmetry"
            ], num_chunks=1, chunk_size=limit)
        
        results = []
        for doc in docs[:limit]:
            results.append({
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "energy_per_atom": doc.energy_per_atom,
                "band_gap": doc.band_gap,
                "symmetry": doc.symmetry.symbol if doc.symmetry else None
            })
        
        return {"results": results, "total": len(results)}
        
    except Exception as e:
        logger.error(f"Materials Project search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Materials Project API unavailable"
        )

# -----------------------------------------------------------------------------
# Workflow Presets Endpoints
# -----------------------------------------------------------------------------

@app.get(f"{API_PREFIX}/presets", tags=["Presets"])
async def get_presets(user: Dict = Depends(verify_token)):
    """Get available workflow presets and default configurations."""
    return {
        "workflow_types": [
            {"id": "full_workflow", "name": "Full Workflow", "description": "Complete pipeline: DFT → ML → MD → Analysis"},
            {"id": "dft_only", "name": "DFT Only", "description": "DFT calculation and structure optimization"},
            {"id": "ml_training", "name": "ML Training", "description": "Train machine learning potential"},
            {"id": "md_simulation", "name": "MD Simulation", "description": "Molecular dynamics simulation"},
            {"id": "screening", "name": "High-Throughput Screening", "description": "Screen multiple materials"},
            {"id": "fitting", "name": "Force Field Fitting", "description": "Fit classical force field parameters"}
        ],
        "dft_codes": ["vasp", "espresso", "abacus"],
        "ml_frameworks": ["deepmd", "nep", "mace"],
        "ensembles": ["nve", "nvt", "npt"],
        "functionals": ["PBE", "PBEsol", "SCAN", "HSE06"],
        "default_configs": {
            "fast": {
                "dft": {"encut": 400, "kpoints_density": 0.3, "max_steps": 100},
                "ml": {"preset": "fast", "num_models": 2},
                "md": {"nsteps_equil": 10000, "nsteps_prod": 50000}
            },
            "balanced": {
                "dft": {"encut": 520, "kpoints_density": 0.25, "max_steps": 200},
                "ml": {"preset": "fast", "num_models": 4},
                "md": {"nsteps_equil": 50000, "nsteps_prod": 100000}
            },
            "accurate": {
                "dft": {"encut": 700, "kpoints_density": 0.2, "max_steps": 300},
                "ml": {"preset": "accurate", "num_models": 4},
                "md": {"nsteps_equil": 100000, "nsteps_prod": 500000}
            }
        }
    }

# =============================================================================
# Static Files (for frontend)
# =============================================================================

# Mount static files (frontend build)
if (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)
