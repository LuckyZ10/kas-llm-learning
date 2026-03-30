"""
Workflow Tool Integrations

Integrations for workflow orchestration tools:
- Apache Airflow
- Prefect
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import structlog

logger = structlog.get_logger()


@dataclass
class WorkflowTask:
    """Represents a workflow task"""
    task_id: str
    task_type: str
    params: Dict[str, Any]
    dependencies: List[str]
    retries: int = 3
    retry_delay_seconds: int = 60


@dataclass
class WorkflowDefinition:
    """Workflow definition"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    schedule: Optional[str] = None  # Cron expression
    tags: List[str] = None


class WorkflowProvider(ABC):
    """Abstract base for workflow providers"""
    
    @abstractmethod
    def create_workflow(self, definition: WorkflowDefinition) -> str:
        """Create a workflow and return workflow ID"""
        pass
    
    @abstractmethod
    def trigger_workflow(self, workflow_id: str, params: Optional[Dict] = None) -> str:
        """Trigger a workflow run"""
        pass
    
    @abstractmethod
    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Get workflow run status"""
        pass


# ==================== Apache Airflow Integration ====================

class AirflowProvider(WorkflowProvider):
    """
    Apache Airflow Integration
    
    Creates DAGs for DFT+LAMMPS workflows
    """
    
    def __init__(self, dags_folder: str = "~/airflow/dags"):
        self.dags_folder = dags_folder
    
    def create_dft_workflow_dag(
        self,
        dag_id: str,
        project_id: str,
        structures: List[Dict[str, Any]],
        calculation_params: Dict[str, Any],
        schedule: Optional[str] = None
    ) -> str:
        """
        Create an Airflow DAG for DFT calculations
        
        Args:
            dag_id: Unique DAG identifier
            project_id: DFT+LAMMPS project ID
            structures: List of structures to calculate
            calculation_params: DFT calculation parameters
            schedule: Cron schedule (optional)
        
        Returns:
            Path to generated DAG file
        """
        dag_code = f'''"""
DFT+LAMMPS Workflow DAG: {dag_id}
Auto-generated at {datetime.utcnow().isoformat()}
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
import json

# DAG Configuration
PROJECT_ID = "{project_id}"
STRUCTURES = {json.dumps(structures, indent=4)}
CALC_PARAMS = {json.dumps(calculation_params, indent=4)}

# Default arguments
default_args = {{
    'owner': 'dft-lammps',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}}

# DAG definition
with DAG(
    '{dag_id}',
    default_args=default_args,
    description='DFT calculations for {len(structures)} structures',
    schedule_interval={repr(schedule) if schedule else 'None'},
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['dft', 'lammps', 'materials'],
) as dag:

    def submit_calculations(**context):
        """Submit all calculations"""
        from dft_lammps import Client
        
        client = Client()
        calc_ids = []
        
        for structure in STRUCTURES:
            calc = client.calculations.submit(
                project_id=PROJECT_ID,
                structure=structure,
                calculation_type='dft',
                parameters=CALC_PARAMS
            )
            calc_ids.append(calc.id)
        
        # Push calculation IDs to XCom
        context['ti'].xcom_push(key='calculation_ids', value=calc_ids)
        return calc_ids

    def wait_for_calculations(**context):
        """Wait for all calculations to complete"""
        from dft_lammps import Client
        import time
        
        client = Client()
        ti = context['ti']
        calc_ids = ti.xcom_pull(task_ids='submit', key='calculation_ids')
        
        completed = []
        while len(completed) < len(calc_ids):
            for calc_id in calc_ids:
                if calc_id in completed:
                    continue
                
                calc = client.calculations.get(calc_id)
                if calc.status in ['completed', 'failed']:
                    completed.append(calc_id)
            
            if len(completed) < len(calc_ids):
                time.sleep(10)
        
        return completed

    def process_results(**context):
        """Process and export results"""
        from dft_lammps import Client
        
        client = Client()
        ti = context['ti']
        calc_ids = ti.xcom_pull(task_ids='wait', key='return_value')
        
        results = []
        for calc_id in calc_ids:
            calc = client.calculations.get(calc_id)
            results.append({{
                'id': calc_id,
                'status': calc.status,
                'results': calc.results
            }})
        
        # Export results
        client.projects.export_results(
            project_id=PROJECT_ID,
            format='json'
        )
        
        return results

    # Tasks
    submit_task = PythonOperator(
        task_id='submit',
        python_callable=submit_calculations,
    )

    wait_task = PythonOperator(
        task_id='wait',
        python_callable=wait_for_calculations,
    )

    process_task = PythonOperator(
        task_id='process',
        python_callable=process_results,
    )

    # Dependencies
    submit_task >> wait_task >> process_task
'''
        
        # Write DAG file
        import os
        dag_path = os.path.join(os.path.expanduser(self.dags_folder), f"{dag_id}.py")
        
        with open(dag_path, 'w') as f:
            f.write(dag_code)
        
        logger.info("airflow_dag_created", dag_id=dag_id, path=dag_path)
        return dag_path
    
    def create_workflow(self, definition: WorkflowDefinition) -> str:
        """Create workflow from definition"""
        # Convert generic workflow to Airflow DAG
        return self.create_dft_workflow_dag(
            dag_id=definition.workflow_id,
            project_id=definition.tasks[0].params.get("project_id", ""),
            structures=definition.tasks[0].params.get("structures", []),
            calculation_params=definition.tasks[0].params.get("params", {}),
            schedule=definition.schedule
        )
    
    def trigger_workflow(self, dag_id: str, params: Optional[Dict] = None) -> str:
        """Trigger DAG run via Airflow API"""
        import requests
        
        # Airflow REST API
        base_url = "http://localhost:8080/api/v1"
        
        response = requests.post(
            f"{base_url}/dags/{dag_id}/dagRuns",
            json={"conf": params or {}},
            auth=("admin", "admin")  # Use proper auth
        )
        
        if response.status_code == 200:
            run_id = response.json().get("dag_run_id")
            logger.info("dag_triggered", dag_id=dag_id, run_id=run_id)
            return run_id
        else:
            raise RuntimeError(f"Failed to trigger DAG: {response.text}")
    
    def get_status(self, dag_id: str, run_id: str) -> Dict[str, Any]:
        """Get DAG run status"""
        import requests
        
        base_url = "http://localhost:8080/api/v1"
        
        response = requests.get(
            f"{base_url}/dags/{dag_id}/dagRuns/{run_id}",
            auth=("admin", "admin")
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "dag_id": dag_id,
                "run_id": run_id,
                "state": data.get("state"),
                "start_date": data.get("start_date"),
                "end_date": data.get("end_date"),
            }
        else:
            return {"error": response.text}


# ==================== Prefect Integration ====================

class PrefectProvider(WorkflowProvider):
    """
    Prefect (v2+) Integration
    
    Creates Prefect flows for DFT+LAMMPS workflows
    """
    
    def __init__(self, prefect_api_url: str = "http://localhost:4200"):
        self.api_url = prefect_api_url
        self.flows: Dict[str, Any] = {}
    
    def create_dft_flow(
        self,
        flow_name: str,
        project_id: str
    ) -> Callable:
        """
        Create a Prefect flow for DFT calculations
        
        Returns:
            Prefect flow function
        """
        try:
            from prefect import flow, task
            from prefect.tasks import task_input_hash
            from dft_lammps import Client
            
            @task(cache_key_fn=task_input_hash, retries=3)
            def submit_calculation_task(structure: Dict, params: Dict) -> str:
                """Submit a single calculation"""
                client = Client()
                calc = client.calculations.submit(
                    project_id=project_id,
                    structure=structure,
                    calculation_type='dft',
                    parameters=params
                )
                return calc.id
            
            @task(retries=5, retry_delay_seconds=60)
            def wait_for_calculation_task(calc_id: str) -> Dict:
                """Wait for calculation to complete"""
                client = Client()
                calc = client.calculations.wait(calc_id, timeout=3600)
                return {
                    "id": calc_id,
                    "status": calc.status,
                    "results": calc.results
                }
            
            @task
            def process_results_task(results: List[Dict]) -> Dict:
                """Process and aggregate results"""
                successful = [r for r in results if r["status"] == "completed"]
                failed = [r for r in results if r["status"] == "failed"]
                
                return {
                    "total": len(results),
                    "successful": len(successful),
                    "failed": len(failed),
                    "results": successful
                }
            
            @flow(name=flow_name, log_prints=True)
            def dft_batch_flow(
                structures: List[Dict],
                calculation_params: Dict[str, Any]
            ):
                """
                DFT batch calculation flow
                
                Args:
                    structures: List of structures to calculate
                    calculation_params: DFT calculation parameters
                """
                logger = get_run_logger()
                logger.info(f"Starting DFT batch flow for {{len(structures)}} structures")
                
                # Submit all calculations
                calc_futures = [
                    submit_calculation_task.submit(structure, calculation_params)
                    for structure in structures
                ]
                
                # Wait for all to complete
                calc_ids = [f.result() for f in calc_futures]
                logger.info(f"Submitted {{len(calc_ids)}} calculations")
                
                # Wait for completion
                results_futures = [
                    wait_for_calculation_task.submit(calc_id)
                    for calc_id in calc_ids
                ]
                results = [f.result() for f in results_futures]
                
                # Process results
                summary = process_results_task(results)
                logger.info(f"Flow complete: {{summary['successful']}}/{{summary['total']}} successful")
                
                return summary
            
            self.flows[flow_name] = dft_batch_flow
            return dft_batch_flow
            
        except ImportError:
            raise ImportError("Prefect not installed. Run: pip install prefect")
    
    def create_workflow(self, definition: WorkflowDefinition) -> str:
        """Create Prefect flow from definition"""
        flow = self.create_dft_flow(
            flow_name=definition.name,
            project_id=definition.tasks[0].params.get("project_id", "")
        )
        return definition.workflow_id
    
    def trigger_workflow(self, flow_name: str, params: Optional[Dict] = None) -> str:
        """Trigger Prefect flow run"""
        if flow_name not in self.flows:
            raise ValueError(f"Flow not found: {flow_name}")
        
        flow = self.flows[flow_name]
        
        # Run the flow
        result = flow(**(params or {}))
        
        return str(result)
    
    def get_status(self, run_id: str) -> Dict[str, Any]:
        """Get Prefect flow run status"""
        try:
            from prefect import get_client
            
            # This would use Prefect client to query run status
            return {
                "run_id": run_id,
                "state": "unknown",  # Would query actual state
            }
        except ImportError:
            return {"error": "Prefect not installed"}
    
    def create_deployment(
        self,
        flow_name: str,
        deployment_name: str,
        schedule: Optional[str] = None
    ):
        """Create Prefect deployment for scheduled runs"""
        try:
            from prefect.deployments import Deployment
            
            if flow_name not in self.flows:
                raise ValueError(f"Flow not found: {flow_name}")
            
            flow = self.flows[flow_name]
            
            deployment = Deployment.build_from_flow(
                flow=flow,
                name=deployment_name,
                schedule=schedule,
                work_queue_name="dft-lammps"
            )
            
            deployment.apply()
            
            logger.info(
                "prefect_deployment_created",
                flow=flow_name,
                deployment=deployment_name
            )
            
        except ImportError:
            raise ImportError("Prefect not installed")


# ==================== Workflow Templates ====================

def create_screening_workflow(
    project_id: str,
    structures_source: str,  # 'file', 'database', 'generated'
    screening_criteria: Dict[str, Any],
    provider: str = "airflow"
) -> WorkflowDefinition:
    """
    Create a materials screening workflow
    
    Template for high-throughput screening campaigns
    """
    tasks = [
        WorkflowTask(
            task_id="load_structures",
            task_type="data_loading",
            params={"source": structures_source},
            dependencies=[]
        ),
        WorkflowTask(
            task_id="preprocess",
            task_type="preprocessing",
            params={"normalize": True, "validate": True},
            dependencies=["load_structures"]
        ),
        WorkflowTask(
            task_id="submit_calculations",
            task_type="calculation",
            params={
                "project_id": project_id,
                "calculation_type": "dft",
                "batch_size": 10
            },
            dependencies=["preprocess"]
        ),
        WorkflowTask(
            task_id="wait_and_collect",
            task_type="collection",
            params={"timeout": 3600 * 24},  # 24 hours
            dependencies=["submit_calculations"]
        ),
        WorkflowTask(
            task_id="screen_results",
            task_type="filtering",
            params=screening_criteria,
            dependencies=["wait_and_collect"]
        ),
        WorkflowTask(
            task_id="export_results",
            task_type="export",
            params={"format": "csv"},
            dependencies=["screen_results"]
        ),
    ]
    
    return WorkflowDefinition(
        workflow_id=f"screening_{project_id}",
        name="Materials Screening Workflow",
        description=f"High-throughput screening for project {project_id}",
        tasks=tasks,
        tags=["screening", "high-throughput"]
    )


def create_ml_training_workflow(
    project_id: str,
    training_set_size: int = 1000,
    provider: str = "prefect"
) -> WorkflowDefinition:
    """
    Create an ML potential training workflow
    
    Template for generating training data and training ML potentials
    """
    tasks = [
        WorkflowTask(
            task_id="generate_structures",
            task_type="generation",
            params={"count": training_set_size, "method": "mcsqs"},
            dependencies=[]
        ),
        WorkflowTask(
            task_id="dft_calculations",
            task_type="calculation",
            params={
                "project_id": project_id,
                "calculation_type": "dft",
                "properties": ["energy", "forces", "stress"]
            },
            dependencies=["generate_structures"]
        ),
        WorkflowTask(
            task_id="validate_data",
            task_type="validation",
            params={"check_convergence": True},
            dependencies=["dft_calculations"]
        ),
        WorkflowTask(
            task_id="train_potential",
            task_type="training",
            params={"model_type": "nep", "epochs": 1000},
            dependencies=["validate_data"]
        ),
        WorkflowTask(
            task_id="validate_potential",
            task_type="validation",
            params={"test_set_ratio": 0.2},
            dependencies=["train_potential"]
        ),
    ]
    
    return WorkflowDefinition(
        workflow_id=f"ml_training_{project_id}",
        name="ML Potential Training Workflow",
        description=f"Training ML potential with {training_set_size} structures",
        tasks=tasks,
        tags=["machine-learning", "training", "nep"]
    )
