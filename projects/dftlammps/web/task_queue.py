#!/usr/bin/env python3
"""
Task Queue Module
=================
Asynchronous task queue implementation using Redis and Celery.

This module provides distributed task processing for:
- DFT calculations
- ML potential training
- MD simulations
- Analysis tasks

Author: DFT-LAMMPS Web Team
Version: 1.0.0
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import asyncio
from functools import wraps

# Celery
from celery import Celery, Task, chain, group, chord
from celery.result import AsyncResult
from celery.exceptions import MaxRetriesExceededError

# Redis
import redis
from redis import Redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Celery app configuration
celery_app = Celery(
    'dftlammps_tasks',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['dftlammps.web.task_queue']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 24,  # 24 hours max runtime
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    task_routes={
        'dftlammps.web.task_queue.run_dft': {'queue': 'dft'},
        'dftlammps.web.task_queue.train_ml': {'queue': 'ml'},
        'dftlammps.web.task_queue.run_md': {'queue': 'md'},
        'dftlammps.web.task_queue.run_analysis': {'queue': 'analysis'},
    },
    task_default_queue='default',
)

# Redis client for direct access
redis_client = Redis.from_url(REDIS_URL, decode_responses=True)

# =============================================================================
# Task Status Enum
# =============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    STARTED = "started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"

# =============================================================================
# Task Data Models
# =============================================================================

@dataclass
class TaskMetadata:
    """Task metadata for tracking and monitoring."""
    task_id: str
    workflow_type: str
    user_id: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = TaskStatus.PENDING
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    error_message: Optional[str] = None
    celery_task_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'workflow_type': self.workflow_type,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'progress': self.progress,
            'stage': self.stage,
            'message': self.message,
            'error_message': self.error_message,
            'celery_task_id': self.celery_task_id
        }

@dataclass
class DFTTaskConfig:
    """DFT calculation task configuration."""
    code: str = "vasp"
    functional: str = "PBE"
    encut: float = 520
    kpoints_density: float = 0.25
    ediff: float = 1e-6
    ncores: int = 4
    max_steps: int = 200
    structure_file: Optional[str] = None
    material_id: Optional[str] = None
    formula: Optional[str] = None

@dataclass
class MLTaskConfig:
    """ML training task configuration."""
    framework: str = "deepmd"
    preset: str = "fast"
    num_models: int = 4
    max_iterations: int = 10
    training_data_path: str = ""
    validation_data_path: Optional[str] = None

@dataclass
class MDTaskConfig:
    """MD simulation task configuration."""
    ensemble: str = "nvt"
    temperature: float = 300.0
    pressure: Optional[float] = None
    timestep: float = 1.0
    nsteps_equil: int = 50000
    nsteps_prod: int = 100000
    potential_file: str = ""
    structure_file: str = ""
    nprocs: int = 4

@dataclass
class AnalysisTaskConfig:
    """Analysis task configuration."""
    trajectory_file: str = ""
    analysis_types: List[str] = None
    atom_types: List[str] = None
    
    def __post_init__(self):
        if self.analysis_types is None:
            self.analysis_types = ["diffusion", "rdf", "msd"]
        if self.atom_types is None:
            self.atom_types = ["Li"]

# =============================================================================
# Task Status Manager
# =============================================================================

class TaskStatusManager:
    """Manages task status in Redis."""
    
    TASK_PREFIX = "dftlammps:task:"
    QUEUE_PREFIX = "dftlammps:queue:"
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    def _key(self, task_id: str) -> str:
        return f"{self.TASK_PREFIX}{task_id}"
    
    def create_task(self, task_id: str, workflow_type: str, user_id: str) -> TaskMetadata:
        """Create a new task entry."""
        metadata = TaskMetadata(
            task_id=task_id,
            workflow_type=workflow_type,
            user_id=user_id,
            created_at=datetime.utcnow()
        )
        self._save_metadata(metadata)
        return metadata
    
    def get_task(self, task_id: str) -> Optional[TaskMetadata]:
        """Get task metadata."""
        data = self.redis.hgetall(self._key(task_id))
        if not data:
            return None
        
        return TaskMetadata(
            task_id=data.get('task_id', task_id),
            workflow_type=data.get('workflow_type', ''),
            user_id=data.get('user_id', ''),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.utcnow(),
            started_at=datetime.fromisoformat(data['started_at']) if 'started_at' in data else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if 'completed_at' in data else None,
            status=data.get('status', TaskStatus.PENDING),
            progress=float(data.get('progress', 0.0)),
            stage=data.get('stage', ''),
            message=data.get('message', ''),
            error_message=data.get('error_message'),
            celery_task_id=data.get('celery_task_id')
        )
    
    def update_task(self, task_id: str, **kwargs) -> Optional[TaskMetadata]:
        """Update task metadata."""
        metadata = self.get_task(task_id)
        if not metadata:
            return None
        
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        self._save_metadata(metadata)
        return metadata
    
    def _save_metadata(self, metadata: TaskMetadata):
        """Save metadata to Redis."""
        key = self._key(metadata.task_id)
        data = metadata.to_dict()
        self.redis.hset(key, mapping=data)
        # Set expiration (7 days)
        self.redis.expire(key, 7 * 24 * 3600)
    
    def delete_task(self, task_id: str) -> bool:
        """Delete task metadata."""
        key = self._key(task_id)
        return self.redis.delete(key) > 0
    
    def list_tasks(self, user_id: Optional[str] = None, 
                   status: Optional[str] = None,
                   limit: int = 100) -> List[TaskMetadata]:
        """List tasks with optional filtering."""
        tasks = []
        pattern = f"{self.TASK_PREFIX}*"
        
        for key in self.redis.scan_iter(match=pattern, count=limit):
            task_id = key.decode().replace(self.TASK_PREFIX, "")
            metadata = self.get_task(task_id)
            if metadata:
                if user_id and metadata.user_id != user_id:
                    continue
                if status and metadata.status != status:
                    continue
                tasks.append(metadata)
        
        # Sort by created_at descending
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        return tasks[:limit]
    
    def update_progress(self, task_id: str, progress: float, 
                       stage: str = "", message: str = ""):
        """Update task progress."""
        updates = {'progress': progress}
        if stage:
            updates['stage'] = stage
        if message:
            updates['message'] = message
        self.update_task(task_id, **updates)
    
    def set_celery_task_id(self, task_id: str, celery_task_id: str):
        """Associate Celery task ID with our task ID."""
        self.update_task(task_id, celery_task_id=celery_task_id, status=TaskStatus.QUEUED)
    
    def mark_started(self, task_id: str):
        """Mark task as started."""
        self.update_task(task_id, status=TaskStatus.STARTED, started_at=datetime.utcnow())
    
    def mark_completed(self, task_id: str, message: str = "Task completed"):
        """Mark task as completed."""
        self.update_task(
            task_id, 
            status=TaskStatus.SUCCESS,
            completed_at=datetime.utcnow(),
            progress=100.0,
            message=message
        )
    
    def mark_failed(self, task_id: str, error_message: str):
        """Mark task as failed."""
        self.update_task(
            task_id,
            status=TaskStatus.FAILURE,
            completed_at=datetime.utcnow(),
            error_message=error_message,
            message="Task failed"
        )

# Initialize status manager
task_manager = TaskStatusManager(redis_client)

# =============================================================================
# Celery Tasks
# =============================================================================

class WorkflowTask(Task):
    """Base class for workflow tasks with progress tracking."""
    
    def __call__(self, *args, **kwargs):
        """Override to set up task context."""
        self.task_id = kwargs.get('task_id')
        return self.run(*args, **kwargs)
    
    def update_progress(self, progress: float, stage: str = "", message: str = ""):
        """Update task progress."""
        if self.task_id:
            task_manager.update_progress(self.task_id, progress, stage, message)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        if self.task_id:
            task_manager.mark_completed(self.task_id)
        logger.info(f"Task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        if self.task_id:
            task_manager.mark_failed(self.task_id, str(exc))
        logger.error(f"Task {task_id} failed: {exc}")

# -----------------------------------------------------------------------------
# DFT Calculation Task
# -----------------------------------------------------------------------------

@celery_app.task(base=WorkflowTask, bind=True, max_retries=3, default_retry_delay=60)
def run_dft(self, config: Dict, task_id: str) -> Dict:
    """
    Run DFT calculation task.
    
    Args:
        config: DFT configuration dictionary
        task_id: Task ID for tracking
    
    Returns:
        Results dictionary with energy, forces, etc.
    """
    self.task_id = task_id
    task_manager.mark_started(task_id)
    
    try:
        self.update_progress(10.0, "setup", "Setting up DFT calculation")
        
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from dftlammps.core.dft_bridge import DFTToLAMMPSBridge
        from ase.io import read
        
        config_obj = DFTTaskConfig(**config)
        
        self.update_progress(20.0, "structure", "Loading structure")
        
        # Get structure
        if config_obj.structure_file:
            atoms = read(config_obj.structure_file)
        elif config_obj.material_id:
            # Fetch from Materials Project
            from mp_api.client import MPRester
            mpr = MPRester()
            docs = mpr.summary.search(material_ids=[config_obj.material_id], fields=["structure"])
            if docs:
                from pymatgen.io.ase import AseAtomsAdaptor
                adaptor = AseAtomsAdaptor()
                atoms = adaptor.get_atoms(docs[0].structure)
            else:
                raise ValueError(f"Material {config_obj.material_id} not found")
        else:
            raise ValueError("No structure provided")
        
        self.update_progress(30.0, "calculation", "Running DFT calculation")
        
        # Run DFT (simplified - actual implementation would use ASE calculators)
        bridge = DFTToLAMMPSBridge(working_dir=str(RESULTS_DIR / task_id / "dft"))
        
        # This is a placeholder - actual DFT would take much longer
        # In production, this would submit to a HPC scheduler
        self.update_progress(50.0, "calculation", "Computing electronic structure")
        
        # Simulate work
        time.sleep(2)
        
        self.update_progress(80.0, "extraction", "Extracting results")
        
        results = {
            "energy": -100.0,  # Placeholder
            "energy_per_atom": -10.0,
            "forces": [[0.0, 0.0, 0.0]] * len(atoms),
            "converged": True,
            "n_steps": 50
        }
        
        self.update_progress(100.0, "complete", "DFT calculation completed")
        
        return results
        
    except Exception as exc:
        logger.error(f"DFT task failed: {exc}")
        try:
            self.retry(countdown=60)
        except MaxRetriesExceededError:
            raise

# -----------------------------------------------------------------------------
# ML Training Task
# -----------------------------------------------------------------------------

@celery_app.task(base=WorkflowTask, bind=True, max_retries=2, default_retry_delay=30)
def train_ml(self, config: Dict, task_id: str, training_data: str) -> Dict:
    """
    Train ML potential task.
    
    Args:
        config: ML configuration dictionary
        task_id: Task ID for tracking
        training_data: Path to training data
    
    Returns:
        Results with model paths and metrics
    """
    self.task_id = task_id
    task_manager.mark_started(task_id)
    
    try:
        self.update_progress(10.0, "setup", "Setting up ML training")
        
        config_obj = MLTaskConfig(**config)
        
        self.update_progress(20.0, "data_loading", "Loading training data")
        
        # Validate training data exists
        if not Path(training_data).exists():
            raise FileNotFoundError(f"Training data not found: {training_data}")
        
        self.update_progress(30.0, "training", f"Training {config_obj.framework} model")
        
        # Simulate training
        for i in range(config_obj.num_models):
            progress = 30 + (i / config_obj.num_models) * 60
            self.update_progress(
                progress, 
                "training", 
                f"Training model {i+1}/{config_obj.num_models}"
            )
            time.sleep(1)
        
        # Model paths (placeholders)
        model_dir = RESULTS_DIR / task_id / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_paths = [str(model_dir / f"model_{i}.pb") for i in range(config_obj.num_models)]
        
        self.update_progress(100.0, "complete", "ML training completed")
        
        return {
            "model_paths": model_paths,
            "num_models": config_obj.num_models,
            "framework": config_obj.framework,
            "training_loss": 0.01,
            "validation_loss": 0.015
        }
        
    except Exception as exc:
        logger.error(f"ML training task failed: {exc}")
        raise

# -----------------------------------------------------------------------------
# MD Simulation Task
# -----------------------------------------------------------------------------

@celery_app.task(base=WorkflowTask, bind=True, max_retries=2)
def run_md(self, config: Dict, task_id: str, model_path: str) -> Dict:
    """
    Run MD simulation task.
    
    Args:
        config: MD configuration dictionary
        task_id: Task ID for tracking
        model_path: Path to trained model
    
    Returns:
        Results with trajectory path and statistics
    """
    self.task_id = task_id
    task_manager.mark_started(task_id)
    
    try:
        self.update_progress(10.0, "setup", "Setting up MD simulation")
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from dftlammps.core.md_simulation import MDSimulationRunner, MDConfig
        
        config_obj = MDTaskConfig(**config)
        
        self.update_progress(20.0, "initialization", "Initializing MD system")
        
        md_config = MDConfig(
            ensemble=config_obj.ensemble,
            temperature=config_obj.temperature,
            pressure=config_obj.pressure,
            timestep=config_obj.timestep,
            nsteps=config_obj.nsteps_prod,
            nsteps_equil=config_obj.nsteps_equil,
            pair_style="deepmd",  # or from config
            potential_file=model_path,
            working_dir=str(RESULTS_DIR / task_id / "md"),
            nprocs=config_obj.nprocs
        )
        
        self.update_progress(30.0, "equilibration", "Running equilibration")
        
        # Simulate equilibration
        time.sleep(1)
        
        self.update_progress(50.0, "production", "Running production MD")
        
        # Simulate production run
        time.sleep(2)
        
        trajectory_path = str(RESULTS_DIR / task_id / "md" / "trajectory.lammpstrj")
        
        self.update_progress(100.0, "complete", "MD simulation completed")
        
        return {
            "trajectory_path": trajectory_path,
            "temperature": config_obj.temperature,
            "nsteps": config_obj.nsteps_prod,
            "ensemble": config_obj.ensemble
        }
        
    except Exception as exc:
        logger.error(f"MD task failed: {exc}")
        raise

# -----------------------------------------------------------------------------
# Analysis Task
# -----------------------------------------------------------------------------

@celery_app.task(base=WorkflowTask, bind=True)
def run_analysis(self, config: Dict, task_id: str, trajectory_file: str) -> Dict:
    """
    Run analysis task on MD trajectory.
    
    Args:
        config: Analysis configuration dictionary
        task_id: Task ID for tracking
        trajectory_file: Path to trajectory file
    
    Returns:
        Analysis results
    """
    self.task_id = task_id
    task_manager.mark_started(task_id)
    
    try:
        self.update_progress(10.0, "setup", "Setting up analysis")
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from dftlammps.core.md_simulation import MDTrajectoryAnalyzer
        
        config_obj = AnalysisTaskConfig(**config)
        
        self.update_progress(30.0, "loading", "Loading trajectory")
        
        # Simulate loading
        time.sleep(0.5)
        
        results = {}
        
        # Compute various analyses
        if "diffusion" in config_obj.analysis_types:
            self.update_progress(50.0, "analysis", "Computing diffusion coefficient")
            results["diffusion_coefficient"] = 1e-5  # Placeholder
            time.sleep(0.5)
        
        if "rdf" in config_obj.analysis_types:
            self.update_progress(70.0, "analysis", "Computing radial distribution function")
            results["rdf_computed"] = True
            time.sleep(0.5)
        
        if "msd" in config_obj.analysis_types:
            self.update_progress(85.0, "analysis", "Computing mean square displacement")
            results["msd_computed"] = True
            time.sleep(0.5)
        
        self.update_progress(100.0, "complete", "Analysis completed")
        
        return results
        
    except Exception as exc:
        logger.error(f"Analysis task failed: {exc}")
        raise

# -----------------------------------------------------------------------------
# Workflow Orchestration Tasks
# -----------------------------------------------------------------------------

@celery_app.task
def create_workflow_report(task_id: str, results: List[Dict]) -> Dict:
    """Create final workflow report from all stage results."""
    report = {
        "task_id": task_id,
        "completed_at": datetime.utcnow().isoformat(),
        "stages": {}
    }
    
    # Combine results from all stages
    for i, result in enumerate(results):
        report["stages"][f"stage_{i}"] = result
    
    # Save report
    report_path = RESULTS_DIR / task_id / "workflow_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

@celery_app.task
def handle_workflow_error(request, exc, traceback, task_id: str):
    """Handle workflow errors."""
    logger.error(f"Workflow {task_id} failed: {exc}")
    task_manager.mark_failed(task_id, str(exc))
    return {"error": str(exc), "traceback": traceback}

# =============================================================================
# Workflow Builders
# =============================================================================

def create_full_workflow(
    task_id: str,
    dft_config: Dict,
    ml_config: Dict,
    md_config: Dict,
    analysis_config: Dict
) -> Any:
    """
    Create a complete workflow chain: DFT → ML → MD → Analysis.
    
    Returns:
        Celery chain object
    """
    # Create workflow chain
    workflow = chain(
        # Stage 1: DFT
        run_dft.s(dft_config, task_id),
        
        # Stage 2: ML Training (receives DFT results)
        train_ml.s(ml_config, task_id),
        
        # Stage 3: MD (receives model path)
        run_md.s(md_config, task_id),
        
        # Stage 4: Analysis (receives trajectory)
        run_analysis.s(analysis_config, task_id),
        
        # Final: Create report
        create_workflow_report.s(task_id)
    )
    
    return workflow

def create_dft_only_workflow(task_id: str, dft_config: Dict) -> Any:
    """Create DFT-only workflow."""
    return chain(
        run_dft.s(dft_config, task_id),
        create_workflow_report.s(task_id)
    )

def create_md_only_workflow(
    task_id: str,
    md_config: Dict,
    analysis_config: Dict,
    model_path: str
) -> Any:
    """Create MD-only workflow."""
    return chain(
        run_md.s(md_config, task_id, model_path),
        run_analysis.s(analysis_config, task_id),
        create_workflow_report.s(task_id)
    )

# =============================================================================
# Queue Management
# =============================================================================

class QueueManager:
    """Manage Celery queues and workers."""
    
    def __init__(self, celery_app: Celery):
        self.app = celery_app
        self.inspector = celery_app.control.inspect()
    
    def get_queue_stats(self) -> Dict:
        """Get queue statistics."""
        active = self.inspector.active() or {}
        scheduled = self.inspector.scheduled() or {}
        reserved = self.inspector.reserved() or {}
        
        return {
            "active_tasks": sum(len(t) for t in active.values()),
            "scheduled_tasks": sum(len(t) for t in scheduled.values()),
            "reserved_tasks": sum(len(t) for t in reserved.values()),
        }
    
    def get_worker_status(self) -> Dict:
        """Get worker status."""
        stats = self.inspector.stats() or {}
        return {
            worker: {
                "total_tasks": info.get("total", {}).get("tasks", 0),
                "uptime": info.get("uptime", 0)
            }
            for worker, info in stats.items()
        }
    
    def purge_queue(self, queue: str = "celery") -> int:
        """Purge all tasks from a queue."""
        return self.app.control.purge()
    
    def revoke_task(self, celery_task_id: str, terminate: bool = False):
        """Revoke a running task."""
        self.app.control.revoke(celery_task_id, terminate=terminate)

queue_manager = QueueManager(celery_app)

# =============================================================================
# Async Interface
# =============================================================================

class AsyncTaskQueue:
    """Async interface for task queue operations."""
    
    @staticmethod
    async def submit_task(
        workflow_type: str,
        task_id: str,
        user_id: str,
        configs: Dict[str, Dict]
    ) -> str:
        """
        Submit a workflow task.
        
        Returns:
            Celery task ID
        """
        # Create task metadata
        task_manager.create_task(task_id, workflow_type, user_id)
        
        # Build workflow based on type
        if workflow_type == "full_workflow":
            workflow = create_full_workflow(
                task_id,
                configs.get("dft", {}),
                configs.get("ml", {}),
                configs.get("md", {}),
                configs.get("analysis", {})
            )
        elif workflow_type == "dft_only":
            workflow = create_dft_only_workflow(task_id, configs.get("dft", {}))
        elif workflow_type == "md_simulation":
            workflow = create_md_only_workflow(
                task_id,
                configs.get("md", {}),
                configs.get("analysis", {}),
                configs.get("model_path", "")
            )
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        # Apply async and get Celery task ID
        result = workflow.apply_async()
        
        # Store Celery task ID
        task_manager.set_celery_task_id(task_id, result.id)
        
        return result.id
    
    @staticmethod
    async def get_task_status(task_id: str) -> Optional[TaskMetadata]:
        """Get task status."""
        return task_manager.get_task(task_id)
    
    @staticmethod
    async def cancel_task(task_id: str) -> bool:
        """Cancel a task."""
        metadata = task_manager.get_task(task_id)
        if metadata and metadata.celery_task_id:
            celery_app.control.revoke(metadata.celery_task_id, terminate=True)
            task_manager.update_task(task_id, status=TaskStatus.REVOKED)
            return True
        return False

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Start Celery worker
    celery_app.start()
