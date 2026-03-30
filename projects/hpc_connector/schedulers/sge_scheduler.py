"""
SGE (Sun Grid Engine) / Oracle Grid Engine scheduler implementation.
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


class SGEScheduler(BaseJobScheduler):
    """SGE/Oracle Grid Engine workload manager implementation."""
    
    STATUS_MAP = {
        'qw': JobStatus.PENDING,
        'qwr': JobStatus.PENDING,
        'hqw': JobStatus.PENDING,  # Hold
        'hRwq': JobStatus.PENDING,
        'r': JobStatus.RUNNING,
        't': JobStatus.RUNNING,  # Transferring
        'Rr': JobStatus.RUNNING,  # Restarting
        'Rt': JobStatus.RUNNING,
        's': JobStatus.SUSPENDED,
        'ts': JobStatus.SUSPENDED,
        'S': JobStatus.SUSPENDED,
        'tS': JobStatus.SUSPENDED,
        'T': JobStatus.SUSPENDED,
        'tT': JobStatus.SUSPENDED,
        'Rs': JobStatus.SUSPENDED,
        'Rts': JobStatus.SUSPENDED,
        'RS': JobStatus.SUSPENDED,
        'RtS': JobStatus.SUSPENDED,
        'RT': JobStatus.SUSPENDED,
        'RtT': JobStatus.SUSPENDED,
        'Eqw': JobStatus.PENDING,  # Error
        'dr': JobStatus.COMPLETING,  # Deleting running
        'dt': JobStatus.COMPLETING,
        'dRr': JobStatus.COMPLETING,
        'dRt': JobStatus.COMPLETING,
        'ds': JobStatus.COMPLETING,
        'dS': JobStatus.COMPLETING,
        'dT': JobStatus.COMPLETING,
        'dRs': JobStatus.COMPLETING,
        'dRS': JobStatus.COMPLETING,
        'dRT': JobStatus.COMPLETING,
    }
    
    def __init__(self, connector: SSHConnector):
        super().__init__(connector)
        self.connector = connector
    
    async def submit_job(self, job_config: JobConfig) -> str:
        """Submit a job to SGE."""
        # Build qsub command
        cmd_parts = ['qsub']
        
        res = job_config.resources
        
        # Resource requirements
        cmd_parts.extend(['-N', job_config.name])
        
        # Parallel environment
        pe_name = res.queue or 'mpi'  # Default PE
        total_slots = res.nodes * res.cores_per_node
        cmd_parts.extend(['-pe', pe_name, str(total_slots)])
        
        cmd_parts.extend(['-l', f'h_rt={res.walltime}'])
        
        if res.memory_per_node:
            cmd_parts.extend(['-l', f'mem_free={res.memory_per_node},h_vmem={res.memory_per_node}'])
        
        if res.gpus_per_node > 0:
            cmd_parts.extend(['-l', f'gpu={res.gpus_per_node}'])
        
        if res.queue and res.queue not in ['mpi', 'smp', 'orte']:
            cmd_parts.extend(['-q', res.queue])
        
        # Priority
        if job_config.priority == JobPriority.HIGH:
            cmd_parts.extend(['-p', '100'])
        elif job_config.priority == JobPriority.URGENT:
            cmd_parts.extend(['-p', '500'])
        elif job_config.priority == JobPriority.LOW:
            cmd_parts.extend(['-p', '-100'])
        
        if job_config.stdout:
            cmd_parts.extend(['-o', job_config.stdout])
        
        if job_config.stderr:
            cmd_parts.extend(['-e', job_config.stderr])
        elif job_config.stdout:
            cmd_parts.append('-j y')  # Join stdout/stderr
        
        if job_config.dependencies:
            dep_str = ",".join(job_config.dependencies)
            cmd_parts.extend(['-hold_jid', dep_str])
        
        if job_config.notify_email:
            cmd_parts.extend(['-M', job_config.notify_email])
            mail_events = []
            if job_config.notify_on_start:
                mail_events.append('b')
            if job_config.notify_on_complete:
                mail_events.append('e')
            if mail_events:
                cmd_parts.extend(['-m', ''.join(mail_events)])
        
        # Checkpointing
        if job_config.checkpoint_enabled:
            cmd_parts.extend(['-ckpt', 'restarter'])
        
        # Environment setup script
        env_script = self._generate_env_script(job_config)
        
        # Create temporary script
        script_path = f"/tmp/{job_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sge"
        script_content = f"""#!/bin/bash
{env_script}
cd {job_config.work_dir}
{job_config.command}
"""
        
        # Write and submit
        result = await self.connector.execute(
            f"cat > {script_path} << 'EOFSCRIPT'\n{script_content}\nEOFSCRIPT"
        )
        
        if result['exit_code'] != 0:
            raise JobSubmissionError(f"Failed to create job script: {result['stderr']}")
        
        cmd_parts.append(script_path)
        
        result = await self.connector.execute(" ".join(cmd_parts))
        
        if result['exit_code'] != 0:
            raise JobSubmissionError(f"Job submission failed: {result['stderr']}")
        
        # Parse job ID
        match = re.search(r'Your job (\d+)', result['stdout'])
        if not match:
            raise JobSubmissionError(
                f"Could not parse job ID from output: {result['stdout']}"
            )
        
        job_id = match.group(1)
        
        # Clean up script
        await self.connector.execute(f"rm {script_path}")
        
        logger.info(f"Submitted job {job_id}: {job_config.name}")
        return job_id
    
    def _generate_env_script(self, job_config: JobConfig) -> str:
        """Generate environment setup script."""
        lines = []
        
        for module in job_config.modules:
            lines.append(f"module load {module}")
        
        for key, value in job_config.environment.items():
            lines.append(f'export {key}="{value}"')
        
        return "\n".join(lines)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an SGE job."""
        result = await self.connector.execute(f"qdel {job_id}")
        return result['exit_code'] == 0
    
    async def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status."""
        result = await self.connector.execute(f"qstat -j {job_id} 2>&1")
        
        if "Following jobs do not exist" in result['stdout']:
            # Job completed or doesn't exist, check qacct
            result = await self.connector.execute(f"qacct -j {job_id} 2>&1")
            if result['exit_code'] == 0:
                # Parse exit status from qacct
                match = re.search(r'exit_status\s+(\d+)', result['stdout'])
                if match:
                    exit_code = int(match.group(1))
                    if exit_code == 0:
                        return JobStatus.COMPLETED
                    else:
                        return JobStatus.FAILED
            return JobStatus.UNKNOWN
        
        if result['exit_code'] != 0:
            return JobStatus.UNKNOWN
        
        # Parse job state from qstat -j output
        match = re.search(r'job_state\s+(\w+)', result['stdout'])
        if match:
            state = match.group(1)
            return self.STATUS_MAP.get(state, JobStatus.UNKNOWN)
        
        # Try qstat for simpler output
        result = await self.connector.execute(f"qstat | grep '^{job_id}'")
        if result['exit_code'] == 0:
            parts = result['stdout'].split()
            if len(parts) >= 5:
                state = parts[4]
                return self.STATUS_MAP.get(state, JobStatus.UNKNOWN)
        
        return JobStatus.UNKNOWN
    
    async def get_job_info(self, job_id: str) -> JobInfo:
        """Get detailed job information."""
        result = await self.connector.execute(f"qstat -j {job_id}")
        
        if result['exit_code'] != 0:
            # Try qacct for completed jobs
            result = await self.connector.execute(f"qacct -j {job_id}")
            if result['exit_code'] != 0:
                raise JobMonitorError(f"Could not get job info for {job_id}")
            return self._parse_qacct_output(result['stdout'], job_id)
        
        return self._parse_qstat_j_output(result['stdout'], job_id)
    
    def _parse_qstat_j_output(self, output: str, job_id: str) -> JobInfo:
        """Parse qstat -j output."""
        info = JobInfo(job_id=job_id, name="", status=JobStatus.UNKNOWN)
        
        for line in output.split('\n'):
            line = line.strip()
            
            if line.startswith('job_name:'):
                info.name = line.split(':', 1)[1].strip()
            elif line.startswith('job_state:'):
                state = line.split(':', 1)[1].strip()
                info.status = self.STATUS_MAP.get(state, JobStatus.UNKNOWN)
            elif line.startswith('owner:'):
                info.user = line.split(':', 1)[1].strip()
            elif line.startswith('queue_name:'):
                info.queue = line.split(':', 1)[1].strip()
            elif line.startswith('slots:'):
                try:
                    info.cores = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('hard_request:'):
                if 'h_rt' in line:
                    match = re.search(r'h_rt=(\d+)', line)
                    if match:
                        info.time_limit = match.group(1)
        
        return info
    
    def _parse_qacct_output(self, output: str, job_id: str) -> JobInfo:
        """Parse qacct output."""
        info = JobInfo(job_id=job_id, name="", status=JobStatus.UNKNOWN)
        
        for line in output.split('\n'):
            line = line.strip()
            
            if line.startswith('jobname'):
                info.name = line.split(None, 1)[1] if len(line.split()) > 1 else ""
            elif line.startswith('owner'):
                info.user = line.split(None, 1)[1] if len(line.split()) > 1 else ""
            elif line.startswith('qname'):
                info.queue = line.split(None, 1)[1] if len(line.split()) > 1 else ""
            elif line.startswith('slots'):
                try:
                    info.cores = int(line.split()[1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('exit_status'):
                try:
                    exit_code = int(line.split()[1])
                    info.status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
                except (ValueError, IndexError):
                    pass
        
        return info
    
    async def get_job_result(self, job_id: str) -> JobResult:
        """Get job result."""
        result = await self.connector.execute(f"qacct -j {job_id}")
        
        if result['exit_code'] != 0:
            return JobResult(job_id=job_id, status=JobStatus.UNKNOWN)
        
        job_result = JobResult(job_id=job_id, status=JobStatus.UNKNOWN)
        
        for line in result['stdout'].split('\n'):
            line = line.strip()
            
            if line.startswith('exit_status'):
                try:
                    exit_code = int(line.split()[1])
                    job_result.exit_code = exit_code
                    job_result.status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
                except (ValueError, IndexError):
                    pass
            elif line.startswith('maxvmem'):
                match = re.search(r'([\d.]+)(\w+)', line)
                if match:
                    job_result.memory_used = f"{match.group(1)}{match.group(2)}"
            elif line.startswith('ru_wallclock'):
                try:
                    seconds = int(line.split()[1])
                    hours = seconds // 3600
                    minutes = (seconds % 3600) // 60
                    secs = seconds % 60
                    job_result.walltime_used = f"{hours}:{minutes:02d}:{secs:02d}"
                except (ValueError, IndexError):
                    pass
            elif line.startswith('slots'):
                try:
                    job_result.cores_used = int(line.split()[1])
                except (ValueError, IndexError):
                    pass
        
        return job_result
    
    async def list_jobs(
        self,
        user: Optional[str] = None,
        status: Optional[JobStatus] = None,
        queue: Optional[str] = None
    ) -> List[JobInfo]:
        """List jobs."""
        cmd = "qstat -u '*'" if not user else f"qstat -u {user}"
        
        result = await self.connector.execute(cmd)
        
        if result['exit_code'] != 0:
            return []
        
        jobs = []
        lines = result['stdout'].strip().split('\n')
        
        # Skip header lines (usually 2)
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 5:
                continue
            
            job_status = self.STATUS_MAP.get(parts[4], JobStatus.UNKNOWN)
            
            if status and job_status != status:
                continue
            
            if queue and parts[3] != queue:
                continue
            
            jobs.append(JobInfo(
                job_id=parts[0],
                name=parts[2],
                user=parts[3],
                status=job_status,
                queue=parts[0].split('@')[1] if '@' in parts[0] else parts[3],
            ))
        
        return jobs
    
    async def get_queues(self) -> List[QueueInfo]:
        """Get queue information."""
        result = await self.connector.execute("qstat -g c")
        
        if result['exit_code'] != 0:
            return []
        
        queues = []
        lines = result['stdout'].strip().split('\n')
        
        # Skip header
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 5:
                continue
            
            queues.append(QueueInfo(
                name=parts[0],
                state='open' if 'a' in parts[1] or 'A' in parts[1] else 'closed',
                total_cores=int(parts[2]) if parts[2].isdigit() else 0,
                allocated_cores=int(parts[3]) if parts[3].isdigit() else 0,
                jobs_running=int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0,
            ))
        
        return queues
    
    async def get_nodes(self, queue: Optional[str] = None) -> List[NodeInfo]:
        """Get node information."""
        result = await self.connector.execute("qstat -f")
        
        if result['exit_code'] != 0:
            return []
        
        nodes = []
        lines = result['stdout'].strip().split('\n')
        
        # Parse host states
        for line in lines:
            if '@' in line and not line.startswith('-'):
                parts = line.split()
                if len(parts) >= 6:
                    name = parts[0].split('@')[1] if '@' in parts[0] else parts[0]
                    
                    nodes.append(NodeInfo(
                        name=name,
                        state=parts[2].lower() if len(parts) > 2 else 'unknown',
                        total_cores=int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0,
                        load_average=float(parts[4]) if len(parts) > 4 and parts[4].replace('.', '').isdigit() else None,
                    ))
        
        return nodes
    
    async def estimate_start_time(self, job_id: str) -> Optional[str]:
        """Estimate job start time."""
        result = await self.connector.execute(f"qstat -j {job_id}")
        
        if result['exit_code'] == 0:
            for line in result['stdout'].split('\n'):
                if 'estimated_start_time' in line.lower():
                    match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                    if match:
                        return match.group(1)
        
        return None
