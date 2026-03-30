#!/usr/bin/env python3
"""
Example: Fault recovery with checkpointing.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpc_connector import HPCClient, ClusterConfig
from hpc_connector.recovery import RecoveryConfig, RecoveryStrategy


def on_recovery(old_job_id: str, new_job_id: str):
    """Handle job recovery."""
    print(f"  [Recovery] Job {old_job_id} recovered as {new_job_id}")


def on_recovery_failed(job_id: str, reason: str):
    """Handle recovery failure."""
    print(f"  [Recovery Failed] Job {job_id}: {reason}")


async def main():
    """Run fault recovery example."""
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
        "work_dir": "/tmp/hpc_recovery",
    })
    
    # Configure recovery
    recovery_config = RecoveryConfig(
        strategy=RecoveryStrategy.RESTART,
        max_retries=3,
        retry_delay=30,
        on_recovery=on_recovery,
        on_recovery_failed=on_recovery_failed,
    )
    
    client_config = {
        "cluster_name": config.name,
        "enable_recovery": True,
        "recovery_config": recovery_config,
    }
    
    async with HPCClient(config) as client:
        print("Fault Recovery Example\n")
        
        # Example 1: Simple job with automatic restart on failure
        print("1. Submitting job with automatic restart...")
        job_spec = {
            "name": "reliable_job",
            "command": "echo 'Running job...' && sleep 30",
            "work_dir": "/tmp/hpc_recovery",
            "resources": {
                "nodes": 1,
                "cores_per_node": 4,
                "walltime": "00:10:00",
            },
        }
        
        job_id = await client.submit_job(
            job_spec,
            enable_recovery=True
        )
        print(f"   Job submitted: {job_id}")
        print("   (If this job fails, it will be automatically restarted)")
        
        # Example 2: Checkpoint-enabled job
        print("\n2. Submitting checkpoint-enabled job...")
        checkpoint_job = {
            "name": "checkpoint_job",
            "command": "python long_simulation.py --checkpoint-enabled",
            "work_dir": "/tmp/hpc_recovery/checkpoint",
            "resources": {
                "nodes": 4,
                "cores_per_node": 16,
                "walltime": "24:00:00",
            },
            "checkpoint_enabled": True,
            "checkpoint_interval": 3600,  # Every hour
            "checkpoint_dir": "/tmp/hpc_recovery/checkpoint/checkpoints",
        }
        
        # Manually save checkpoint during job execution
        # (In real usage, this would be done by the job itself)
        await client.save_checkpoint(
            job_id=job_id,
            checkpoint_path="/tmp/hpc_recovery/checkpoint/checkpoints/iter_1000",
            step=1000,
            metadata={"iteration": 1000, "loss": 0.123}
        )
        print("   Checkpoint saved for job")
        
        # Example 3: Monitoring alerts
        print("\n3. Setting up alert handling...")
        
        def alert_handler(alert):
            print(f"   [Alert] {alert.level.value}: {alert.message}")
        
        client.add_alert_handler(alert_handler)
        print("   Alert handler registered")
        
        # Get alerts
        print("\n4. Checking for alerts...")
        alerts = client.get_job_alerts()
        if alerts:
            for alert in alerts[:5]:
                print(f"   - {alert['level']}: {alert['message']}")
        else:
            print("   No alerts")


if __name__ == "__main__":
    asyncio.run(main())
