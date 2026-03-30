"""
Workflow Engine - Execute workflow definitions
"""
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.models.workflow import Workflow, WorkflowStatus
from app.models.task import Task, TaskStatus
from app.websocket.manager import broadcast_workflow_update, broadcast_task_update

logger = structlog.get_logger()


class WorkflowEngine:
    """Workflow execution engine"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def start_workflow(self, workflow: Workflow, initial_context: Dict[str, Any]):
        """Start workflow execution"""
        logger.info(f"Starting workflow", workflow_id=workflow.id)
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.execution_context = initial_context
        workflow.progress_percent = 0.0
        
        await self.db.flush()
        
        # Broadcast update
        await broadcast_workflow_update(workflow.id, "running", 0.0)
        
        # Start executing from the first node or specified start node
        await self._execute_next_node(workflow)
    
    async def _execute_next_node(self, workflow: Workflow):
        """Execute the next node in the workflow"""
        definition = workflow.definition
        nodes = definition.get("nodes", [])
        edges = definition.get("edges", [])
        
        if not nodes:
            logger.warning(f"No nodes in workflow", workflow_id=workflow.id)
            await self._complete_workflow(workflow)
            return
        
        # Find current or starting node
        current_node_id = workflow.current_node_id
        if not current_node_id:
            # Start from first node
            current_node_id = nodes[0]["id"]
        
        workflow.current_node_id = current_node_id
        
        # Find the node definition
        current_node = None
        for node in nodes:
            if node["id"] == current_node_id:
                current_node = node
                break
        
        if not current_node:
            logger.error(f"Current node not found", workflow_id=workflow.id, node_id=current_node_id)
            await self._fail_workflow(workflow, f"Node {current_node_id} not found")
            return
        
        # Create and queue task for this node
        await self._create_node_task(workflow, current_node)
    
    async def _create_node_task(self, workflow: Workflow, node: Dict[str, Any]):
        """Create a task from a workflow node"""
        node_data = node.get("data", {})
        node_type = node_data.get("node_type", "generic")
        
        task = Task(
            name=node_data.get("label", f"Task for {node['id']}"),
            description=node_data.get("description"),
            task_type=node_type,
            workflow_id=workflow.id,
            working_directory=f"{workflow.project_id}/{workflow.id}",
            node_id=node["id"],
            status=TaskStatus.PENDING,
        )
        
        self.db.add(task)
        await self.db.flush()
        
        # Broadcast task creation
        await broadcast_task_update(task.id, "pending", task.to_dict())
        
        logger.info(f"Created task for node", task_id=task.id, node_id=node["id"])
    
    async def _complete_workflow(self, workflow: Workflow):
        """Mark workflow as completed"""
        workflow.status = WorkflowStatus.COMPLETED
        workflow.progress_percent = 100.0
        await self.db.flush()
        
        await broadcast_workflow_update(workflow.id, "completed", 100.0)
        logger.info(f"Workflow completed", workflow_id=workflow.id)
    
    async def _fail_workflow(self, workflow: Workflow, error_message: str):
        """Mark workflow as failed"""
        workflow.status = WorkflowStatus.FAILED
        await self.db.flush()
        
        await broadcast_workflow_update(workflow.id, "failed", workflow.progress_percent)
        logger.error(f"Workflow failed", workflow_id=workflow.id, error=error_message)
