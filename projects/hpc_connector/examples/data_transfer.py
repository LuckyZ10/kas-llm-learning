#!/usr/bin/env python3
"""
Example: Data transfer and synchronization.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpc_connector import HPCClient, ClusterConfig
from hpc_connector.data import TransferProgress


def progress_callback(progress: TransferProgress):
    """Display transfer progress."""
    bar_length = 40
    filled = int(bar_length * progress.percentage / 100)
    bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)
    
    print(f"\r[{bar}] {progress.percentage:.1f}% "
          f"({progress.bytes_transferred}/{progress.total_bytes} bytes) "
          f"{progress.speed_mbps:.2f} MB/s", end='', flush=True)


async def main():
    """Run data transfer example."""
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
        "work_dir": "/tmp/hpc_example",
    })
    
    async with HPCClient(config) as client:
        print("Data Transfer Example\n")
        
        # Example 1: Upload a single file
        print("1. Uploading file...")
        # await client.upload_file(
        #     local_path="/local/path/to/file.txt",
        #     remote_path="/remote/path/to/file.txt",
        #     progress_callback=progress_callback
        # )
        print("   (Skipped - update paths for your environment)")
        
        # Example 2: Download a file
        print("\n2. Downloading file...")
        # await client.download_file(
        #     remote_path="/remote/path/to/results.dat",
        #     local_path="/local/path/to/results.dat",
        #     progress_callback=progress_callback
        # )
        print("   (Skipped - update paths for your environment)")
        
        # Example 3: Sync directory to remote
        print("\n3. Syncing local directory to remote...")
        # await client.sync_to_remote(
        #     local_dir="/local/project",
        #     remote_dir="/remote/project",
        #     exclude_patterns=["*.pyc", "__pycache__", ".git"]
        # )
        print("   (Skipped - update paths for your environment)")
        
        # Example 4: Stage job data
        print("\n4. Staging job data...")
        # staging_info = await client.stage_job_data(
        #     job_id="12345",
        #     input_files=["input.dat", "params.json", "model.pt"],
        #     output_patterns=["output*.dat", "results/*.json", "checkpoint*.pt"],
        #     local_work_dir="/local/job001",
        #     remote_work_dir="/remote/job001"
        # )
        print("   (Skipped - update paths for your environment)")
        
        # Example 5: Retrieve job output
        print("\n5. Retrieving job output...")
        # manifest = await client.retrieve_job_output(
        #     job_id="12345",
        #     output_patterns=["output*.dat", "results/*.json"],
        #     local_work_dir="/local/job001",
        #     remote_work_dir="/remote/job001"
        # )
        print("   (Skipped - update paths for your environment)")


if __name__ == "__main__":
    asyncio.run(main())
