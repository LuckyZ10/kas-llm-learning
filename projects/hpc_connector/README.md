# HPC Connector

A production-grade HPC cluster connection and job scheduling system for Python.

## Features

- **Multi-Platform Support**: SLURM, PBS/Torque, LSF, SGE, AWS ParallelCluster, Aliyun Batch, Tencent Batch
- **Unified API**: Single interface for all cluster types
- **Async I/O**: High-performance asynchronous operations
- **Data Pipeline**: Automatic file staging, sync, and checkpoint management
- **Monitoring**: Real-time job monitoring and alerting
- **Fault Recovery**: Automatic retry and checkpoint resume
- **Type Hints**: Full type annotation support

## Installation

```bash
pip install hpc-connector
```

Or from source:

```bash
git clone https://github.com/your-org/hpc-connector.git
cd hpc-connector
pip install -e .
```

## Quick Start

```python
import asyncio
from hpc_connector import HPCClient, ClusterConfig

async def main():
    # Configure cluster
    config = ClusterConfig.from_dict({
        "name": "my-cluster",
        "cluster_type": "slurm",
        "ssh": {
            "host": "login.cluster.org",
            "user": "username",
            "auth_method": "key",
            "key_file": "~/.ssh/id_rsa",
        },
        "default_partition": "compute",
    })
    
    # Connect and submit job
    async with HPCClient(config) as client:
        job_id = await client.submit_job({
            "name": "my_job",
            "command": "python train.py",
            "work_dir": "/scratch/username/work",
            "resources": {
                "nodes": 2,
                "cores_per_node": 8,
                "walltime": "4:00:00",
            },
        })
        
        # Wait for completion
        result = await client.wait_for_job(job_id)
        print(f"Job completed: {result.status}")

asyncio.run(main())
```

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| SLURM | ✅ Supported | Most common, full features |
| PBS/Torque | ✅ Supported | Includes PBS Pro |
| LSF | ✅ Supported | IBM Platform LSF |
| SGE | ✅ Supported | Oracle Grid Engine |
| AWS ParallelCluster | 🚧 Planned | Via boto3 |
| Aliyun Batch | 🚧 Planned | Via aliyun SDK |
| Tencent Batch | 🚧 Planned | Via qcloud SDK |

## Architecture

```
hpc_connector/
├── core/               # Base classes and models
│   ├── base.py        # Abstract base classes
│   ├── cluster.py     # Cluster configuration
│   ├── job.py         # Job models
│   └── exceptions.py  # Custom exceptions
├── connectors/        # Connection implementations
│   ├── ssh_connector.py
│   └── __init__.py
├── schedulers/        # Job scheduler implementations
│   ├── slurm_scheduler.py
│   ├── pbs_scheduler.py
│   ├── lsf_scheduler.py
│   └── sge_scheduler.py
├── data/              # Data pipeline
│   └── pipeline.py
├── monitoring/        # Monitoring and alerting
│   └── monitor.py
├── recovery/          # Fault recovery
│   └── fault_recovery.py
├── api.py            # High-level client API
└── examples/         # Usage examples
```

## Core Components

### Cluster Configuration

```python
from hpc_connector.core.cluster import ClusterConfig, ClusterType

config = ClusterConfig.from_dict({
    "name": "slurm-cluster",
    "cluster_type": "slurm",
    "ssh": {
        "host": "login.cluster.org",
        "port": 22,
        "user": "username",
        "auth_method": "key",
        "key_file": "~/.ssh/id_rsa",
    },
    "work_dir": "/scratch/username",
    "default_partition": "compute",
    "max_nodes": 128,
    "max_walltime": "72:00:00",
})
```

### Job Submission

```python
from hpc_connector.core.job import JobConfig, ResourceRequest, JobPriority

job = JobConfig(
    name="simulation",
    command="mpirun -np 64 ./sim.exe",
    work_dir="/scratch/work",
    resources=ResourceRequest(
        nodes=4,
        cores_per_node=16,
        memory_per_node="64GB",
        walltime="24:00:00",
        partition="compute",
    ),
    priority=JobPriority.HIGH,
    modules=["intelmpi/2019", "mkl/2019"],
    environment={"OMP_NUM_THREADS": "16"},
)
```

### Data Pipeline

```python
# Upload with progress tracking
def progress_callback(progress):
    print(f"{progress.percentage:.1f}% complete")

await client.upload_file(
    local_path="/local/data",
    remote_path="/remote/data",
    progress_callback=progress_callback
)

# Sync directories
await client.sync_to_remote(
    local_dir="/local/project",
    remote_dir="/remote/project",
    exclude_patterns=["*.pyc", "__pycache__", ".git"]
)
```

### Monitoring

```python
# Add alert handler
def on_alert(alert):
    if alert.level == AlertLevel.ERROR:
        send_notification(alert.message)

client.add_alert_handler(on_alert)

# Get job metrics
alerts = client.get_job_alerts(
    level=AlertLevel.WARNING,
    unacknowledged_only=True
)
```

### Fault Recovery

```python
from hpc_connector.recovery import RecoveryConfig, RecoveryStrategy

config = RecoveryConfig(
    strategy=RecoveryStrategy.RESTART,
    max_retries=3,
    retry_delay=60,
)

client = HPCClient(cluster_config, recovery_config=config)

# Job will be automatically restarted on failure
job_id = await client.submit_job(job_spec, enable_recovery=True)
```

## Examples

See `examples/` directory for complete examples:

- `basic_submission.py` - Simple job submission
- `data_transfer.py` - File upload/download
- `workflow.py` - Multi-job workflows
- `fault_recovery.py` - Checkpoint and recovery

## Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hpc_connector
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Installation instructions
- Configuration guide
- Security best practices
- Monitoring setup
- Troubleshooting

## API Reference

### HPCClient

Main client class for cluster operations.

```python
class HPCClient:
    async def submit_job(job_spec) -> str
    async def cancel_job(job_id) -> bool
    async def get_job_status(job_id) -> JobStatus
    async def wait_for_job(job_id) -> JobResult
    async def list_jobs(...) -> List[Dict]
    async def get_queues() -> List[Dict]
    async def upload_file(local, remote, callback)
    async def download_file(remote, local, callback)
    async def sync_to_remote(local_dir, remote_dir)
```

### ClusterConfig

Configuration for HPC cluster connection.

```python
class ClusterConfig:
    name: str
    cluster_type: ClusterType
    ssh: SSHConfig
    work_dir: str
    default_partition: Optional[str]
    max_nodes: int
    max_walltime: str
```

### JobConfig

Job specification.

```python
class JobConfig:
    name: str
    command: str
    work_dir: str
    resources: ResourceRequest
    priority: JobPriority
    modules: List[str]
    environment: Dict[str, str]
    dependencies: List[str]
    checkpoint_enabled: bool
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: https://hpc-connector.readthedocs.io
- Issues: https://github.com/your-org/hpc-connector/issues
- Discussions: https://github.com/your-org/hpc-connector/discussions
