#!/usr/bin/env python3
"""
Example: Workflow with dependent jobs.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpc_connector import HPCClient, ClusterConfig


async def main():
    """Run workflow example with dependent jobs."""
    config = ClusterConfig.from_dict({
        "name": "example-cluster",
        "cluster_type": "slurm",
        "ssh": {
            "host": "localhost",
            "port": 22,
            "user": "username",
            "auth_method": "key",
            "key_file": "~/.ssh/id_rsa",
        },
        "work_dir": "/tmp/hpc_workflow",
    })
    
    async with HPCClient(config) as client:
        print("Workflow Example: Data Processing Pipeline\n")
        
        # Step 1: Data preprocessing
        print("1. Submitting preprocessing job...")
        preprocess_job = {
            "name": "preprocess",
            "command": "echo 'Preprocessing data...' && sleep 10",
            "work_dir": "/tmp/hpc_workflow",
            "resources": {
                "nodes": 1,
                "cores_per_node": 4,
                "walltime": "00:15:00",
            },
        }
        
        preprocess_id = await client.submit_job(preprocess_job)
        print(f"   Preprocessing job: {preprocess_id}")
        
        # Step 2: Parallel processing (depends on preprocessing)
        print("\n2. Submitting parallel processing jobs...")
        processing_jobs = []
        for i in range(3):
            job = {
                "name": f"process_chunk_{i}",
                "command": f"echo 'Processing chunk {i}...' && sleep 20",
                "work_dir": "/tmp/hpc_workflow",
                "resources": {
                    "nodes": 1,
                    "cores_per_node": 8,
                    "walltime": "00:30:00",
                },
                "dependencies": [preprocess_id],
            }
            job_id = await client.submit_job(job)
            processing_jobs.append(job_id)
            print(f"   Processing job {i}: {job_id}")
        
        # Step 3: Aggregation (depends on all processing jobs)
        print("\n3. Submitting aggregation job...")
        aggregate_job = {
            "name": "aggregate",
            "command": "echo 'Aggregating results...' && sleep 5",
            "work_dir": "/tmp/hpc_workflow",
            "resources": {
                "nodes": 1,
                "cores_per_node": 2,
                "walltime": "00:10:00",
            },
            "dependencies": processing_jobs,
        }
        
        aggregate_id = await client.submit_job(aggregate_job)
        print(f"   Aggregation job: {aggregate_id}")
        
        # Step 4: Reporting (depends on aggregation)
        print("\n4. Submitting report generation job...")
        report_job = {
            "name": "generate_report",
            "command": "echo 'Generating final report...' && sleep 5",
            "work_dir": "/tmp/hpc_workflow",
            "resources": {
                "nodes": 1,
                "cores_per_node": 1,
                "walltime": "00:05:00",
            },
            "dependencies": [aggregate_id],
        }
        
        report_id = await client.submit_job(report_job)
        print(f"   Report job: {report_id}")
        
        # Wait for workflow completion
        print("\nWaiting for workflow completion...")
        all_jobs = [preprocess_id] + processing_jobs + [aggregate_id, report_id]
        results = await client.wait_for_jobs(all_jobs)
        
        print("\nWorkflow Results:")
        for job_id, result in results.items():
            print(f"  {job_id}: {result.status.value} (exit code: {result.exit_code})")
        
        # Summary
        successful = sum(1 for r in results.values() if r.status.value == "completed")
        print(f"\nSummary: {successful}/{len(results)} jobs completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
