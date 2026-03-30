"""
LSF (Load Sharing Facility) scheduler implementation.
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


class LSFScheduler(BaseJobScheduler):
    """IBM LSF workload manager implementation."""
    
    STATUS_MAP = {
        'PEND': JobStatus.PENDING,
        'PSUSP': JobStatus.SUSPENDED,
        'RUN': JobStatus.RUNNING,
        'USUSP': JobStatus.SUSPENDED,
        'SSUSP': JobStatus.SUSPENDED,
        'DONE': JobStatus.COMPLETED,
        'EXIT': JobStatus.FAILED,
        'POST_DONE': JobStatus.COMPLETED,
        'WAIT': JobStatus.PENDING,
        'ZOMBI': JobStatus.UNKNOWN,
    }
    
    def __init__(self, connector: SSHConnector):
        super().__init__(connector)
        self.connector = connector
    
    async def submit_job(self, job_config: JobConfig) -> str:
        """Submit a job to LSF."""
        # Build bsub command
        cmd_parts = ['bsub']
        
        res = job_config.resources
        
        # Resource requirements
        cmd_parts.extend(['-J', job_config.name])
        cmd_parts.extend(['-n', str(res.nodes * res.cores_per_node)])
        cmd_parts.extend(['-W', res.walltime.replace(':', '')])
        
        if res.nodes > 1:
            cmd_parts.extend(['-R', f'span[hosts={res.nodes}]'])
        
        if res.memory_per_node:
            mem_mb = self._parse_memory_to_mb(res.memory_per_node)
            cmd_parts.extend(['-M', str(mem_mb)])
            cmd_parts.extend(['-R', f'rusage[mem={mem_mb}]'])
        
        if res.gpus_per_node > 0:
            cmd_parts.extend(['-gpu', f'num={res.gpus_per_node}'])
        
        if res.queue:
            cmd_parts.extend(['-q', res.queue])
        elif self.connector.config.default_queue:
            cmd_parts.extend(['-q', self.connector.config.default_queue])
        
        # Priority
        if job_config.priority == JobPriority.HIGH:
            cmd_parts.extend(['-sp', '50'])
        elif job_config.priority == JobPriority.URGENT:
            cmd_parts.extend(['-sp', '100'])
        
        if job_config.stdout:
            cmd_parts.extend(['-o', job_config.stdout])
        
        if job_config.stderr:
            cmd_parts.extend(['-e', job_config.stderr])
        
        if job_config.dependencies:
            dep_str = " && ".join([f"done({dep})" for dep in job_config.dependencies])
            cmd_parts.extend(['-w', dep_str])
        
        if job_config.notify_email:
            cmd_parts.extend(['-u', job_config.notify_email])
            if job_config.notify_on_start:
                cmd_parts.append('-B')  # Notify at start
            if job_config.notify_on_complete:
                cmd_parts.append('-N')  # Notify at completion
        
        # Environment
        env_setup = ""
        if job_config.modules:
            env_setup += "; ".join([f"module load {m}" for m in job_config.modules])
        for key, value in job_config.environment.items():
            env_setup += f'; export {key}="{value}"' if env_setup else f'export {key}="{value}"'
        
        # Job command
        job_cmd = job_config.command
        if env_setup:
            job_cmd = f"{env_setup}; cd {job_config.work_dir}; {job_cmd}"
        else:
            job_cmd = f"cd {job_config.work_dir}; {job_cmd}"
        
        cmd_parts.append(f'"{job_cmd}"')
        
        # Submit
        result = await self.connector.execute(" ".join(cmd_parts))
        
        if result['exit_code'] != 0:
            raise JobSubmissionError(f"Job submission failed: {result['stderr']}")
        
        # Parse job ID
        match = re.search(r'Job <(\d+)>', result['stdout'])
        if not match:
            raise JobSubmissionError(
                f"Could not parse job ID from output: {result['stdout']}"
            )
        
        job_id = match.group(1)
        logger.info(f"Submitted job {job_id}: {job_config.name}")
        return job_id
    
    def _parse_memory_to_mb(self, memory_str: str) -> int:
        """Parse memory string to MB."""
        match = re.match(r'(\d+)(\w*)', memory_str.upper())
        if not match:
            return 4096
        
        value = int(match.group(1))
        unit = match.group(2)
        
        multipliers = {
            '': 1, 'B': 1,
            'K': 1/1024, 'KB': 1/1024,
            'M': 1, 'MB': 1,
            'G': 1024, 'GB': 1024,
            'T': 1024*1024, 'TB': 1024*1024,
        }
        
        return int(value * multipliers.get(unit, 1))
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an LSF job."""
        result = await self.connector.execute(f"bkill {job_id}")
        return result['exit_code'] == 0
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status."""
        result = await self.connector.execute(f"bjobs {job_id} -noheader")
        
        if result['exit_code'] != 0 or not result['stdout'].strip():
            return JobStatus.UNKNOWN
        
        parts = result['stdout'].split()
        if len(parts) >= 3:
            state = parts[2]
            return self.STATUS_MAP.get(state, JobStatus.UNKNOWN)
        
        return JobStatus.UNKNOWN
    
    async def get_job_info(self, job_id: str) -> JobInfo:
        """Get detailed job information."""
        result = await self.connector.execute(f"bjobs -l {job_id}")
        
        if result['exit_code'] != 0:
            raise JobMonitorError(f"Could not get job info for {job_id}")
        
        return self._parse_bjobs_output(result['stdout'], job_id)
    
    def _parse_bjobs_output(self, output: str, job_id: str) -> JobInfo:
        """Parse bjobs -l output."""
        info = JobInfo(job_id=job_id, name="", status=JobStatus.UNKNOWN)
        
        for line in output.split('\n'):
            line = line.strip()
            
            if line.startswith('Job <'):
                match = re.search(r'Job <(\d+)>, Job Name <([^\u003e]+)>', line)
                if match:
                    info.name = match.group(2)
            
            if 'Status' in line:
                match = re.search(r'Status <(\w+)>', line)
                if match:
                    info.status = self.STATUS_MAP.get(match.group(1), JobStatus.UNKNOWN)
            
            if 'Queue' in line:
                match = re.search(r'Queue <([^\u003e]+)>', line)
                if match:
                    info.queue = match.group(1)
            
            if 'Exec Host' in line:
                match = re.search(r'Exec Host\s+(\S+)', line)
                if match:
                    info.nodes = match.group(1).split(':')
        
        return info
    
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get job result."""
        result = await self.connector.execute(f"bjobs -l {job_id}")
        
        if result['exit_code'] != 0:
            return JobResult(job_id=job_id, status=JobStatus.UNKNOWN)
        
        job_result = JobResult(job_id=job_id, status=JobStatus.UNKNOWN)
        
        for line in result['stdout'].split('\n'):
            if 'Status' in line:
                match = re.search(r'Status <(\w+)>', line)
                if match:
                    job_result.status = self.STATUS_MAP.get(match.group(1), JobStatus.UNKNOWN)
            
            if 'Exited with exit code' in line:
                match = re.search(r'exit code (\d+)', line)
                if match:
                    job_result.exit_code = int(match.group(1))
            
            if 'MAX MEM' in line or 'max mem' in line.lower():
                match = re.search(r'([\d.]+)\s*(\w+)', line)
                if match:
                    job_result.memory_used = f"{match.group(1)} {match.group(2)}"
            
            if 'CPU time' in line:
                match = re.search(r'CPU time\s*:\s*([\d.]+)', line)
                if match:
                    job_result.walltime_used = match.group(1)
        
        return job_result
    
    async def list_jobs(
        self,
        user: Optional[str] = None,
        status: Optional[JobStatus] = None,
        queue: Optional[str] = None
    ) -> List[JobInfo]:
        """List jobs."""
        cmd = "bjobs -noheader"
        if user:
            cmd += f" -u {user}"
        
        result = await self.connector.execute(cmd)
        
        if result['exit_code'] != 0:
            return []
        
        jobs = []
        for line in result['stdout'].strip().split('\n'):
            parts = line.split()
            if len(parts) < 4:
                continue
            
            job_status = self.STATUS_MAP.get(parts[2], JobStatus.UNKNOWN)
            
            if status and job_status != status:
                continue
            
            if queue and parts[3] != queue:
                continue
            
            jobs.append(JobInfo(
                job_id=parts[0],
                user=parts[1],
                status=job_status,
                queue=parts[3],
            ))
        
        return jobs
    
    async def get_queues(self) -> List[QueueInfo]:
        """Get queue information."""
        result = await self.connector.execute("bqueues -l")
        
        if result['exit_code'] != 0:
            return []
        
        queues = []
        current_queue = None
        
        for line in result['stdout'].split('\n'):
            line = line.strip()
            
            if line.startswith('QUEUE:'):
                if current_queue:
                    queues.append(current_queue)
                name = line.split(':')[1].strip()
                current_queue = QueueInfo(name=name, state='open')
            
            elif current_queue:
                if 'NJOBS' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        current_queue.jobs_running = int(parts[1]) if parts[1].isdigit() else 0
                        current_queue.jobs_queued = int(parts[3]) if parts[3].isdigit() else 0
        
        if current_queue:
            queues.append(current_queue)
        
        return queues
    
    async def get_nodes(self, queue: Optional[str] = None) -> List[NodeInfo]:
        """Get node information."""
        cmd = "bhosts -w"
        result = await self.connector.execute(cmd)
        
        if result['exit_code'] != 0:
            return []
        
        nodes = []
        lines = result['stdout'].strip().split('\n')
        
        # Skip header
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            
            nodes.append(NodeInfo(
                name=parts[0],
                state=parts[1].lower(),
                total_cores=int(parts[3]) if parts[3].isdigit() else 0,
                free_cores=int(parts[4]) if parts[4].isdigit() else 0,
                load_average=float(parts[5]) if parts[5].replace('.', '').isdigit() else None,
            ))
        
        return nodes
    
    async def estimate_start_time(self, job_id: str) -> Optional[str]:
        """Estimate job start time."""
        result = await self.connector.execute(f"bjobs {job_id} -noheader")
        
        if result['exit_code'] == 0:
            # For pending jobs, show estimated start time
            for line in result['stdout'].split('\n'):
                parts = line.split()
                if len(parts) >= 6 and parts[2] == 'PEND':
                    # Get more details with -l
                    detail_result = await self.connector.execute(f"bjobs -l {job_id}")
                    if detail_result['exit_code'] == 0:
                        # Parse for estimated start time
                        pass
        
        return None
