"""
PBS/Torque scheduler implementation.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..core.base import BaseJobScheduler
from ..core.job import JobConfig, JobResult, JobInfo, JobStatus, ResourceRequest, JobPriority
from ..core.cluster import QueueInfo, NodeInfo
from ..core.exceptions import JobSubmissionError, JobMonitorError
from ..connectors.ssh_connector import SSHConnector

logger = logging.getLogger(__name__)


class PBSScheduler(BaseJobScheduler):
    """PBS/Torque workload manager implementation."""
    
    STATUS_MAP = {
        'C': JobStatus.COMPLETED,
        'E': JobStatus.COMPLETING,
        'H': JobStatus.PENDING,  # Held
        'Q': JobStatus.QUEUED,
        'R': JobStatus.RUNNING,
        'T': JobStatus.PENDING,  # Moving
        'W': JobStatus.PENDING,  # Waiting
        'S': JobStatus.SUSPENDED,
        'F': JobStatus.FAILED,
        'U': JobStatus.UNKNOWN,
    }
    
    def __init__(self, connector: SSHConnector):
        super().__init__(connector)
        self.connector = connector
        self._is_torque = False  # Will detect if Torque vs PBS Pro
    
    async def submit_job(self, job_config: JobConfig) -> str:
        """Submit a job to PBS/Torque."""
        script = self._generate_job_script(job_config)
        
        script_path = f"/tmp/{job_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pbs"
        
        try:
            # Create script on remote
            result = await self.connector.execute(
                f"cat > {script_path} << 'EOFSCRIPT'\n{script}\nEOFSCRIPT"
            )
            
            if result['exit_code'] != 0:
                raise JobSubmissionError(
                    f"Failed to create job script: {result['stderr']}"
                )
            
            # Submit job
            result = await self.connector.execute(
                f"cd {job_config.work_dir} && qsub {script_path}"
            )
            
            if result['exit_code'] != 0:
                raise JobSubmissionError(
                    f"Job submission failed: {result['stderr']}"
                )
            
            # Parse job ID
            job_id = result['stdout'].strip().split('.')[0]  # Remove server name
            
            # Clean up script
            await self.connector.execute(f"rm {script_path}")
            
            logger.info(f"Submitted job {job_id}: {job_config.name}")
            return job_id
            
        except Exception as e:
            if not isinstance(e, JobSubmissionError):
                raise JobSubmissionError(f"Job submission failed: {e}")
            raise
    
    def _generate_job_script(self, job_config: JobConfig) -> str:
        """Generate PBS job script."""
        lines = ["#!/bin/bash"]
        
        res = job_config.resources
        
        # PBS directives
        lines.append(f"#PBS -N {job_config.name}")
        lines.append(f"#PBS -l nodes={res.nodes}:ppn={res.cores_per_node}")
        lines.append(f"#PBS -l walltime={res.walltime}")
        
        if res.memory_per_node:
            lines.append(f"#PBS -l mem={res.memory_per_node}")
        
        if res.gpus_per_node > 0:
            lines.append(f"#PBS -l gpus={res.gpus_per_node}")
        
        if res.queue:
            lines.append(f"#PBS -q {res.queue}")
        elif self.connector.config.default_queue:
            lines.append(f"#PBS -q {self.connector.config.default_queue}")
        
        # Priority
        if job_config.priority == JobPriority.HIGH:
            lines.append("#PBS -p 100")
        elif job_config.priority == JobPriority.URGENT:
            lines.append("#PBS -p 200")
        elif job_config.priority == JobPriority.LOW:
            lines.append("#PBS -p -100")
        
        if job_config.stdout:
            lines.append(f"#PBS -o {job_config.stdout}")
        
        if job_config.stderr:
            lines.append(f"#PBS -e {job_config.stderr}")
        elif job_config.stdout:
            lines.append("#PBS -j oe")  # Join stdout and stderr
        
        if job_config.dependencies:
            dep_str = ":".join(job_config.dependencies)
            lines.append(f"#PBS -W depend=afterok:{dep_str}")
        
        if job_config.notify_email:
            lines.append(f"#PBS -M {job_config.notify_email}")
            mail_events = []
            if job_config.notify_on_start:
                mail_events.append("b")  # Begin
            if job_config.notify_on_complete:
                mail_events.extend(["e", "a"])  # End, Abort
            if mail_events:
                lines.append(f"#PBS -m {''.join(mail_events)}")
        
        # Environment setup
        for module in job_config.modules:
            lines.append(f"module load {module}")
        
        for key, value in job_config.environment.items():
            lines.append(f'export {key}="{value}"')
        
        # Checkpoint
        if job_config.checkpoint_enabled:
            lines.append(f"#PBS -c s")  # Checkpoint on error
        
        # Change to working directory
        lines.append(f"cd {job_config.work_dir}")
        
        # Job command
        lines.append(job_config.command)
        
        return "\n".join(lines)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a PBS job."""
        result = await self.connector.execute(f"qdel {job_id}")
        return result['exit_code'] == 0
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status."""
        result = await self.connector.execute(
            f"qstat -f {job_id} 2>/dev/null | grep job_state"
        )
        
        if result['exit_code'] != 0 or not result['stdout'].strip():
            # Job might be completed, check qstat -x (for Torque) or tracejob
            result = await self.connector.execute(f"qstat -x {job_id} 2>/dev/null")
            if result['exit_code'] != 0:
                return JobStatus.UNKNOWN
        
        match = re.search(r'job_state = (\w)', result['stdout'])
        if match:
            state = match.group(1)
            return self.STATUS_MAP.get(state, JobStatus.UNKNOWN)
        
        return JobStatus.UNKNOWN
    
    async def get_job_info(self, job_id: str) -> JobInfo:
        """Get detailed job information."""
        result = await self.connector.execute(f"qstat -f {job_id}")
        
        if result['exit_code'] != 0:
            raise JobMonitorError(f"Could not get job info for {job_id}")
        
        return self._parse_qstat_output(result['stdout'], job_id)
    
    def _parse_qstat_output(self, output: str, job_id: str) -> JobInfo:
        """Parse qstat -f output."""
        info = JobInfo(job_id=job_id, name="", status=JobStatus.UNKNOWN)
        
        for line in output.split('\n'):
            line = line.strip()
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'Job_Name':
                    info.name = value
                elif key == 'Job_Owner':
                    info.user = value.split('@')[0]
                elif key == 'job_state':
                    info.status = self.STATUS_MAP.get(value, JobStatus.UNKNOWN)
                elif key == 'queue':
                    info.queue = value
                elif key == 'Resource_List.nodes':
                    # Parse nodes=2:ppn=4 format
                    match = re.match(r'(\d+)(?::ppn=(\d+))?', value)
                    if match:
                        info.cores = int(match.group(1)) * (int(match.group(2)) if match.group(2) else 1)
                elif key == 'exec_host':
                    info.nodes = self._parse_exec_host(value)
        
        return info
    
    def _parse_exec_host(self, exec_host: str) -> List[str]:
        """Parse exec_host format (e.g., node1/0+node1/1+node2/0)."""
        nodes = set()
        for part in exec_host.split('+'):
            node = part.split('/')[0]
            nodes.add(node)
        return list(nodes)
    
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get job result."""
        # Try qstat -x for completed jobs (Torque)
        result = await self.connector.execute(f"qstat -x -f {job_id}")
        
        if result['exit_code'] != 0:
            return JobResult(job_id=job_id, status=JobStatus.UNKNOWN)
        
        output = result['stdout']
        job_result = JobResult(job_id=job_id, status=JobStatus.UNKNOWN)
        
        for line in output.split('\n'):
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'job_state':
                    job_result.status = self.STATUS_MAP.get(value, JobStatus.UNKNOWN)
                elif key == 'exit_status':
                    try:
                        job_result.exit_code = int(value)
                    except ValueError:
                        pass
                elif key == 'resources_used.mem':
                    job_result.memory_used = value
                elif key == 'resources_used.walltime':
                    job_result.walltime_used = value
        
        return job_result
    
    async def list_jobs(
        self,
        user: Optional[str] = None,
        status: Optional[JobStatus] = None,
        queue: Optional[str] = None
    ) -> List[JobInfo]:
        """List jobs."""
        cmd = "qstat -u" if user else "qstat"
        if user:
            cmd += f" {user}"
        
        result = await self.connector.execute(cmd)
        
        if result['exit_code'] != 0:
            return []
        
        jobs = []
        lines = result['stdout'].strip().split('\n')
        
        # Skip header lines
        for line in lines[2:] if len(lines) > 2 else lines:
            parts = line.split()
            if len(parts) < 6:
                continue
            
            job_id = parts[0].split('.')[0]
            job_status = self.STATUS_MAP.get(parts[4], JobStatus.UNKNOWN)
            
            if status and job_status != status:
                continue
            
            if queue and parts[2] != queue:
                continue
            
            jobs.append(JobInfo(
                job_id=job_id,
                name=parts[3],
                user=parts[1],
                status=job_status,
                queue=parts[2],
            ))
        
        return jobs
    
    async def get_queues(self) -> List[QueueInfo]:
        """Get queue information."""
        result = await self.connector.execute("qstat -q")
        
        if result['exit_code'] != 0:
            return []
        
        queues = []
        lines = result['stdout'].strip().split('\n')
        
        # Find queue list section
        in_queue_list = False
        for line in lines:
            if 'Queue' in line and 'Memory' in line:
                in_queue_list = True
                continue
            
            if in_queue_list and line.strip() and not line.startswith('-'):
                parts = line.split()
                if len(parts) >= 5:
                    queues.append(QueueInfo(
                        name=parts[0],
                        state='open' if 'enabled' in line.lower() else 'closed',
                        max_nodes=int(parts[2]) if parts[2].isdigit() else 0,
                        jobs_running=int(parts[3]) if parts[3].isdigit() else 0,
                        jobs_queued=int(parts[4]) if parts[4].isdigit() else 0,
                    ))
        
        return queues
    
    async def get_nodes(self, queue: Optional[str] = None) -> List[NodeInfo]:
        """Get node information."""
        result = await self.connector.execute("pbsnodes -a")
        
        if result['exit_code'] != 0:
            return []
        
        nodes = []
        current_node = None
        
        for line in result['stdout'].split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            # New node entry
            if not line.startswith(' '):
                if current_node:
                    nodes.append(current_node)
                current_node = NodeInfo(name=line, state='unknown')
            elif current_node:
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'state':
                        current_node.state = value
                    elif key == 'np':
                        current_node.total_cores = int(value)
                    elif key == 'properties':
                        current_node.features = value.split(',')
                    elif key == 'status':
                        # Parse status for memory info
                        mem_match = re.search(r'physmem=(\d+)(\w+)', value)
                        if mem_match:
                            current_node.total_memory = f"{mem_match.group(1)}{mem_match.group(2)}"
                        load_match = re.search(r'loadave=([\d.]+)', value)
                        if load_match:
                            current_node.load_average = float(load_match.group(1))
        
        if current_node:
            nodes.append(current_node)
        
        return nodes
    
    async def estimate_start_time(self, job_id: str) -> Optional[str]:
        """Estimate job start time."""
        # PBS Pro has showstart command
        result = await self.connector.execute(f"showstart {job_id}")
        
        if result['exit_code'] == 0:
            # Parse showstart output
            for line in result['stdout'].split('\n'):
                if 'Estimated start' in line or 'start time' in line.lower():
                    return line.split(':')[-1].strip()
        
        return None
