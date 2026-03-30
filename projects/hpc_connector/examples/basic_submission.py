#!/usr/bin/env python3
"""
Example: Basic job submission and monitoring.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpc_connector import HPCClient, ClusterConfig
from hpc_connector.core.job import JobStatus


async def main():
    """Run basic job submission example."""
    # Load cluster configuration
    config = ClusterConfig.from_dict({
        "name": "example-cluster",
        "cluster_type": "slurm",
        "ssh": {
            "host": "localhost",  # Replace with actual cluster
            "port": 22,
            "user": "username",
            "auth_method": "key",
            "key_file": "~/.ssh/id_rsa",
        },
        "work_dir": "/tmp/hpc_example",
        "default_partition": "compute",
    })
    
    async with HPCClient(config) as client:
        print(f"Connected to cluster: {config.name}")
        
        # Get available queues
        print("\nAvailable queues:")
        queues = await client.get_queues()
        for queue in queues[:3]:  # Show first 3
            print(f"  - {queue['name']}: {queue['jobs_running']} running, {queue['jobs_queued']} queued")
        
        # Submit a job
        job_spec = {
            "name": "hello_world",
            "command": "echo 'Hello from HPC!' && sleep 5 && echo 'Job complete'",
            "work_dir": "/tmp/hpc_example",
            "resources": {
                "nodes": 1,
                "cores_per_node": 1,
                "walltime": "00:05:00",
            },
            "stdout": "hello.out",
            "stderr": "hello.err",
        }
        
        print("\nSubmitting job...")
        job_id = await client.submit_job(job_spec)
        print(f"Job submitted: {job_id}")
        
        # Monitor job
        print("\nMonitoring job...")
        def on_status_change(old, new):
            print(f"  Status: {old.value} -> {new.value}")
        
        result = await client.wait_for_job(
            job_id,
            poll_interval=5,
            on_status_change=on_status_change
        )
        
        print(f"\nJob finished with status: {result.status.value}")
        print(f"Exit code: {result.exit_code}")
        
        # List recent jobs
        print("\nRecent jobs:")
        jobs = await client.list_jobs(status=JobStatus.COMPLETED, limit=5)
        for job in jobs:
            print(f"  - {job['job_id']}: {job['name']} ({job['status']})")


if __name__ == "__main__":
    asyncio.run(main())
