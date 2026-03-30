"""
Monitoring Service - Collect and aggregate monitoring data
"""
from typing import List, Dict, Any, Optional
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.models.task import Task, TaskStatus
from app.models.workflow import Workflow, WorkflowStatus
from app.models.project import Project
from app.core.config import settings

logger = structlog.get_logger()


class MonitoringService:
    """Service for collecting monitoring data"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        # Count workflows by status
        workflow_counts = {}
        for status in WorkflowStatus:
            result = await self.db.execute(
                select(func.count()).select_from(Workflow).where(Workflow.status == status)
            )
            workflow_counts[status.value] = result.scalar()
        
        # Count tasks by status
        task_counts = {}
        for status in TaskStatus:
            result = await self.db.execute(
                select(func.count()).select_from(Task).where(Task.status == status)
            )
            task_counts[status.value] = result.scalar()
        
        # Resource usage
        result = await self.db.execute(
            select(func.sum(Task.cpu_time_seconds), func.sum(Task.memory_peak_mb))
            .where(Task.status == TaskStatus.COMPLETED)
        )
        total_cpu, total_memory = result.fetchone()
        
        return {
            "workflows": workflow_counts,
            "tasks": task_counts,
            "resources": {
                "total_cpu_hours": round((total_cpu or 0) / 3600, 2),
                "total_memory_gb": round((total_memory or 0) / 1024, 2),
            },
            "timestamp": logger.bind().kwargs.get("timestamp"),
        }
    
    async def get_active_workflows(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get currently active workflows"""
        result = await self.db.execute(
            select(Workflow)
            .where(Workflow.status.in_([WorkflowStatus.RUNNING, WorkflowStatus.QUEUED]))
            .order_by(desc(Workflow.started_at))
            .limit(limit)
        )
        workflows = result.scalars().all()
        
        return [{
            "id": w.id,
            "name": w.name,
            "type": w.workflow_type.value,
            "status": w.status.value,
            "progress": w.progress_percent,
            "started_at": w.started_at.isoformat() if w.started_at else None,
        } for w in workflows]
    
    async def get_recent_tasks(self, status: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tasks"""
        query = select(Task)
        
        if status:
            query = query.where(Task.status == status)
        
        query = query.order_by(desc(Task.updated_at)).limit(limit)
        
        result = await self.db.execute(query)
        tasks = result.scalars().all()
        
        return [t.to_dict() for t in tasks]
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        # Get running tasks
        result = await self.db.execute(
            select(Task).where(Task.status == TaskStatus.RUNNING)
        )
        running_tasks = result.scalars().all()
        
        return {
            "running_tasks": len(running_tasks),
            "tasks": [{
                "id": t.id,
                "name": t.name,
                "type": t.task_type.value,
                "cpu_time": t.cpu_time_seconds,
                "memory_mb": t.memory_peak_mb,
            } for t in running_tasks],
        }
    
    async def get_training_metrics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get ML training metrics from lcurve.out files"""
        import pandas as pd
        from pathlib import Path
        
        lcurve_path = Path(settings.MODELS_PATH) / "lcurve.out"
        
        if not lcurve_path.exists():
            # Search subdirectories
            for subdir in Path(settings.MODELS_PATH).rglob("lcurve.out"):
                lcurve_path = subdir
                break
        
        if not lcurve_path.exists():
            return {"error": "No training data found"}
        
        try:
            df = pd.read_csv(lcurve_path, sep=r'\s+', comment='#', header=None)
            if len(df.columns) >= 7:
                df.columns = ['batch', 'lr', 'loss', 'energy_rmse', 'energy_rmse_traj',
                             'force_rmse', 'force_rmse_traj'][:len(df.columns)]
            
            # Return last 1000 points for performance
            df = df.tail(1000)
            
            return {
                "current": {
                    "loss": float(df['loss'].iloc[-1]) if 'loss' in df.columns else None,
                    "force_rmse": float(df['force_rmse'].iloc[-1]) if 'force_rmse' in df.columns else None,
                    "energy_rmse": float(df['energy_rmse'].iloc[-1]) if 'energy_rmse' in df.columns else None,
                    "lr": float(df['lr'].iloc[-1]) if 'lr' in df.columns else None,
                },
                "history": {
                    "batch": df['batch'].tolist() if 'batch' in df.columns else [],
                    "loss": df['loss'].tolist() if 'loss' in df.columns else [],
                    "force_rmse": df['force_rmse'].tolist() if 'force_rmse' in df.columns else [],
                    "energy_rmse": df['energy_rmse'].tolist() if 'energy_rmse' in df.columns else [],
                }
            }
        except Exception as e:
            logger.error(f"Failed to read training metrics", error=str(e))
            return {"error": str(e)}
    
    async def get_md_metrics(self, trajectory_id: str, metric: str) -> Dict[str, Any]:
        """Get MD simulation metrics from LAMMPS logs"""
        import pandas as pd
        from pathlib import Path
        
        # Find log file
        log_files = list(Path(settings.MD_RESULTS_PATH).rglob("log.lammps"))
        
        if not log_files:
            return {"error": "No MD log files found"}
        
        log_path = log_files[0]
        
        try:
            # Parse LAMMPS log
            data = []
            with open(log_path) as f:
                lines = f.readlines()
            
            in_thermo = False
            headers = []
            
            for line in lines:
                if "Step" in line and "Temp" in line:
                    headers = line.strip().split()
                    in_thermo = True
                    continue
                
                if in_thermo:
                    if line.strip() == "" or "Loop" in line:
                        in_thermo = False
                        continue
                    try:
                        values = [float(x) for x in line.strip().split()]
                        if len(values) == len(headers):
                            data.append(dict(zip(headers, values)))
                    except:
                        pass
            
            df = pd.DataFrame(data)
            
            # Map metric to column
            metric_map = {
                "temperature": ["Temp", "temp"],
                "energy": ["TotEng", "TotEnergy", "etotal", "pe"],
                "pressure": ["Press", "press"],
                "volume": ["Volume", "vol"],
            }
            
            column = None
            for col in metric_map.get(metric, [metric]):
                if col in df.columns:
                    column = col
                    break
            
            if not column:
                return {"error": f"Metric {metric} not found in log"}
            
            return {
                "trajectory_id": trajectory_id,
                "metric": metric,
                "steps": df['Step'].tolist() if 'Step' in df.columns else list(range(len(df))),
                "values": df[column].tolist(),
            }
            
        except Exception as e:
            logger.error(f"Failed to read MD metrics", error=str(e))
            return {"error": str(e)}
    
    async def get_al_progress(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get active learning progress"""
        import json
        from pathlib import Path
        
        al_path = Path(settings.AL_WORKFLOW_PATH)
        
        if not al_path.exists():
            return {"error": "No active learning data found"}
        
        iterations = []
        
        for iter_dir in sorted(al_path.glob("iter_*")):
            iter_num = int(iter_dir.name.split("_")[1])
            
            iter_data = {
                "iteration": iter_num,
            }
            
            # Read stats
            stats_file = iter_dir / "exploration_stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                    iter_data.update(stats)
            
            iterations.append(iter_data)
        
        return {
            "iterations": iterations,
            "current_iteration": len(iterations),
        }
