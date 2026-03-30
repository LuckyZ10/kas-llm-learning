"""
SLURM scheduler implementation.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..core.base import BaseJobScheduler
from ..core.job import JobConfig, JobResult, JobInfo, JobStatus, ResourceRequest
from ..core.cluster import QueueInfo, NodeInfo
from ..core.exceptions import JobSubmissionError, JobMonitorError
from ..connectors.ssh_connector import SSHConnector

logger = logging.getLogger(__name__)


class SlurmScheduler(BaseJobScheduler):
    """SLURM workload manager implementation."""
    
    STATUS_MAP = {
        'PENDING': JobStatus.PENDING,
        'RUNNING': JobStatus.RUNNING,
        'SUSPENDED': JobStatus.SUSPENDED,
        'COMPLETING': JobStatus.COMPLETING,
        'COMPLETED': JobStatus.COMPLETED,
        'FAILED': JobStatus.FAILED,
        'CANCELLED': JobStatus.CANCELLED,
        'TIMEOUT': JobStatus.TIMEOUT,
        'NODE_FAIL': JobStatus.FAILED,
        'PREEMPTED': JobStatus.CANCELLED,
        'OUT_OF_MEMORY': JobStatus.FAILED,
    }
    
    def __init__(self, connector: SSHConnector):
        super().__init__(connector)
        self.connector = connector
    
    async def submit_job(self, job_config: JobConfig) -> str:
        """Submit a job to SLURM."""
        script = self._generate_job_script(job_config)
        
        # Write script to temporary file
        script_path = f"/tmp/{job_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sh"
        
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
                f"cd {job_config.work_dir} && sbatch {script_path}"
            )
            
            if result['exit_code'] != 0:
                raise JobSubmissionError(
                    f"Job submission failed: {result['stderr']}"
                )
            
            # Parse job ID from output
            match = re.search(r'Submitted batch job (\d+)', result['stdout'])
            if not match:
                raise JobSubmissionError(
                    f"Could not parse job ID from output: {result['stdout']}"
                )
            
            job_id = match.group(1)
            
            # Clean up script
            await self.connector.execute(f"rm {script_path}")
            
            logger.info(f"Submitted job {job_id}: {job_config.name}")
            return job_id
            
        except Exception as e:
            if not isinstance(e, JobSubmissionError):
                raise JobSubmissionError(f"Job submission failed: {e}")
            raise
    
    def _generate_job_script(self, job_config: JobConfig) -> str:
        """Generate SLURM job script."""
        lines = ["#!/bin/bash"]
        
        # SLURM directives
        res = job_config.resources
        lines.append(f"#SBATCH --job-name={job_config.name}")
        lines.append(f"#SBATCH --nodes={res.nodes}")
        lines.append(f"#SBATCH --ntasks-per-node={res.cores_per_node}")
        lines.append(f"#SBATCH --time={res.walltime}")
        
        if res.memory_per_node:
            lines.append(f"#SBATCH --mem={res.memory_per_node}")
        
        if res.gpus_per_node > 0:
            lines.append(f"#SBATCH --gpus-per-node={res.gpus_per_node}")
        
        if res.partition:
            lines.append(f"#SBATCH --partition={res.partition}")
        elif self.connector.config.default_partition:
            lines.append(f"#SBATCH --partition={self.connector.config.default_partition}")
        
        if res.constraints:
            lines.append(f"#SBATCH --constraint={'&'.join(res.constraints)}")
        
        if job_config.priority == JobPriority.HIGH:
            lines.append("#SBATCH --priority=high")
        elif job_config.priority == JobPriority.URGENT:
            lines.append("#SBATCH --priority=urgent")
        
        if job_config.stdout:
            lines.append(f"#SBATCH --output={job_config.stdout}")
        
        if job_config.stderr:
            lines.append(f"#SBATCH --error={job_config.stderr}")
        
        if job_config.dependencies:
            dep_str = ":".join(job_config.dependencies)
            lines.append(f"#SBATCH --dependency=afterok:{dep_str}")
        
        if job_config.notify_email:
            lines.append(f"#SBATCH --mail-user={job_config.notify_email}")
            mail_types = []
            if job_config.notify_on_start:
                mail_types.append("BEGIN")
            if job_config.notify_on_complete:
                mail_types.append("END")
            if mail_types:
                lines.append(f"#SBATCH --mail-type={','.join(mail_types)}")
        
        # Environment setup
        if self.connector.config.module_system:
            for module in job_config.modules:
                lines.append(f"module load {module}")
        
        for key, value in job_config.environment.items():
            lines.append(f'export {key}="{value}"')
        
        # Checkpoint setup
        if job_config.checkpoint_enabled:
            lines.append(f"#SBATCH --checkpoint={job_config.checkpoint_interval}")
            if job_config.checkpoint_dir:
                lines.append(f"#SBATCH --checkpoint-dir={job_config.checkpoint_dir}")
        
        # Change to working directory
        lines.append(f"cd {job_config.work_dir}")
        
        # Job command
        lines.append(job_config.command)
        
        return "\n".join(lines)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a SLURM job."""
        result = await self.connector.execute(f"scancel {job_id}")
        return result['exit_code'] == 0
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status."""
        result = await self.connector.execute(
            f"scontrol show job {job_id} --oneliner"
        )
        
        if result['exit_code'] != 0:
            # Check if job is in sacct (completed jobs)
            result = await self.connector.execute(
                f'sacct -j {job_id} --format=State --noheader -X'
            )
            if result['exit_code'] == 0 and result['stdout'].strip():
                state = result['stdout'].strip().split()[0]
                return self.STATUS_MAP.get(state, JobStatus.UNKNOWN)
            return JobStatus.UNKNOWN
        
        # Parse JobState from scontrol output
        match = re.search(r'JobState=(\S+)', result['stdout'])
        if match:
            state = match.group(1)
            return self.STATUS_MAP.get(state, JobStatus.UNKNOWN)
        
        return JobStatus.UNKNOWN
    
    async def get_job_info(self, job_id: str) -> JobInfo:
        """Get detailed job information."""
        # Use scontrol for running jobs, sacct for completed
        result = await self.connector.execute(
            f"scontrol show job {job_id} --oneliner"
        )
        
        if result['exit_code'] != 0:
            # Try sacct for completed jobs
            result = await self.connector.execute(
                f"sacct -j {job_id} --format=JobID,JobName,State,User,Partition,"
                f"NCPUS,NNodes,Start,End,Elapsed,ExitCode --noheader -X -P"
            )
            if result['exit_code'] != 0:
                raise JobMonitorError(f"Could not get job info for {job_id}")
            
            return self._parse_sacct_output(result['stdout'], job_id)
        
        return self._parse_scontrol_output(result['stdout'], job_id)
    
    def _parse_scontrol_output(self, output: str, job_id: str) -> JobInfo:
        """Parse scontrol output."""
        info = JobInfo(job_id=job_id, name="", status=JobStatus.UNKNOWN)
        
        # Extract fields
        patterns = {
            'name': r'JobName=(\S+)',
            'user': r'UserId=(\S+)',
            'partition': r'Partition=(\S+)',
            'state': r'JobState=(\S+)',
            'cores': r'NumCPUs=(\d+)',
            'nodes': r'NumNodes=(\d+)',
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                value = match.group(1)
                if field == 'name':
                    info.name = value
                elif field == 'user':
                    info.user = value.split('(')[0]  # Remove group info
                elif field == 'partition':
                    info.queue = value
                elif field == 'state':
                    info.status = self.STATUS_MAP.get(value, JobStatus.UNKNOWN)
                elif field == 'cores':
                    info.cores = int(value)
                elif field == 'nodes':
                    # Parse nodelist
                    nodes_match = re.search(r'ExecHost=(\S+)', output)
                    if nodes_match:
                        info.nodes = self._parse_nodelist(nodes_match.group(1))
        
        return info
    
    def _parse_sacct_output(self, output: str, job_id: str) -> JobInfo:
        """Parse sacct output."""
        parts = output.strip().split('|')
        if len(parts) < 10:
            return JobInfo(job_id=job_id, name="", status=JobStatus.UNKNOWN)
        
        status = self.STATUS_MAP.get(parts[2], JobStatus.UNKNOWN)
        
        info = JobInfo(
            job_id=job_id,
            name=parts[1],
            status=status,
            user=parts[3],
            queue=parts[4],
            cores=int(parts[5]) if parts[5].isdigit() else 0,
        )
        
        return info
    
    def _parse_nodelist(self, nodelist: str) -> List[str]:
        """Parse SLURM nodelist format (e.g., node[01-05],node07)."""
        nodes = []
        parts = nodelist.split(',')
        
        for part in parts:
            match = re.match(r'(\w+)\[(\d+)-(\d+)\]', part)
            if match:
                prefix = match.group(1)
                start = int(match.group(2))
                end = int(match.group(3))
                width = len(match.group(2))
                for i in range(start, end + 1):
                    nodes.append(f"{prefix}{i:0{width}d}")
            else:
                nodes.append(part)
        
        return nodes
    
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get job result."""
        # Get info from sacct
        result = await self.connector.execute(
            f"sacct -j {job_id} --format=JobID,State,ExitCode,MaxRSS,"
            f"Start,End,Elapsed,NNodes,NCPUS --noheader -X -P"
        )
        
        if result['exit_code'] != 0 or not result['stdout'].strip():
            return JobResult(job_id=job_id, status=JobStatus.UNKNOWN)
        
        parts = result['stdout'].strip().split('|')
        
        # Parse exit code (format: exit_code:signal)
        exit_code = None
        if len(parts) > 2 and parts[2]:
            try:
                exit_code = int(parts[2].split(':')[0])
            except ValueError:
                pass
        
        job_result = JobResult(
            job_id=job_id,
            status=self.STATUS_MAP.get(parts[1], JobStatus.UNKNOWN),
            exit_code=exit_code,
            memory_used=parts[3] if len(parts) > 3 else None,
            nodes_used=int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else 0,
            cores_used=int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else 0,
        )
        
        return job_result
    
    async def list_jobs(
        self,
        user: Optional[str] = None,
        status: Optional[JobStatus] = None,
        queue: Optional[str] = None
    ) -> List[JobInfo]:
        """List jobs."""
        cmd = "squeue --format='%i|%j|%u|%T|%P|%C|%D|%N' --noheader"
        
        if user:
            cmd += f" -u {user}"
        if queue:
            cmd += f" -p {queue}"
        
        result = await self.connector.execute(cmd)
        
        if result['exit_code'] != 0:
            return []
        
        jobs = []
        for line in result['stdout'].strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) < 8:
                continue
            
            job_status = self.STATUS_MAP.get(parts[3], JobStatus.UNKNOWN)
            
            if status and job_status != status:
                continue
            
            jobs.append(JobInfo(
                job_id=parts[0],
                name=parts[1],
                user=parts[2],
                status=job_status,
                queue=parts[4],
                cores=int(parts[5]) if parts[5].isdigit() else 0,
                nodes=parts[7].split(',') if parts[7] else [],
            ))
        
        return jobs
    
    async def get_queues(self) -> List[QueueInfo]:
        """Get queue/partition information."""
        result = await self.connector.execute(
            "sinfo --format='%P|%a|%D|%c|%C|%G|%m' --noheader"
        )
        
        if result['exit_code'] != 0:
            return []
        
        queues = []
        for line in result['stdout'].strip().split('\n'):
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 7:
                continue
            
            # Parse CPU info (allocated/idle/other/total)
            cpu_parts = parts[4].split('/')
            
            queues.append(QueueInfo(
                name=parts[0].rstrip('*'),  # Remove default marker
                state=parts[1],
                total_nodes=int(parts[2]),
                total_cores=int(cpu_parts[3]) if len(cpu_parts) > 3 else 0,
                free_cores=int(cpu_parts[1]) if len(cpu_parts) > 1 else 0,
            ))
        
        return queues
    
    async def get_nodes(self, queue: Optional[str] = None) -> List[NodeInfo]:
        """Get node information."""
        cmd = "sinfo -N --format='%N|%T|%c|%C|%m|%e|%G|%f' --noheader"
        if queue:
            cmd += f" -p {queue}"
        
        result = await self.connector.execute(cmd)
        
        if result['exit_code'] != 0:
            return []
        
        nodes = []
        seen = set()
        
        for line in result['stdout'].strip().split('\n'):
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 8:
                continue
            
            name = parts[0]
            if name in seen:
                continue
            seen.add(name)
            
            # Parse CPU info
            cpu_parts = parts[3].split('/')
            
            nodes.append(NodeInfo(
                name=name,
                state=parts[1],
                total_cores=int(parts[2]),
                free_cores=int(cpu_parts[1]) if len(cpu_parts) > 1 else 0,
                total_memory=parts[4],
                free_memory=parts[5],
                features=parts[7].split(',') if parts[7] else [],
            ))
        
        return nodes
    
    async def estimate_start_time(self, job_id: str) -> Optional[str]:
        """Estimate job start time."""
        result = await self.connector.execute(
            f'squeue -j {job_id} --format="%S" --noheader'
        )
        
        if result['exit_code'] == 0:
            start_time = result['stdout'].strip()
            if start_time and start_time != 'N/A':
                return start_time
        
        return None
