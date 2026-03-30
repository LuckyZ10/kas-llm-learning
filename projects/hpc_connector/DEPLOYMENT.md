# HPC Connector Deployment Guide

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Security](#security)
4. [Cluster Setup](#cluster-setup)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

## Installation

### Requirements

- Python 3.8+
- asyncssh library
- SSH access to target clusters

### Install from Source

```bash
# Clone repository
git clone <repository-url>
cd hpc_connector

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hpc_connector/ ./hpc_connector/
COPY config/ ./config/

CMD ["python", "-m", "hpc_connector.server"]
```

## Configuration

### Cluster Configuration File

Create `clusters.yaml`:

```yaml
clusters:
  - name: slurm-production
    cluster_type: slurm
    ssh:
      host: login.slurm-cluster.org
      port: 22
      user: ${SLURM_USER}
      auth_method: key
      key_file: ~/.ssh/slurm_key
      timeout: 30
      keepalive_interval: 60
    work_dir: /scratch/${SLURM_USER}
    module_system: lmod
    environment_setup:
      - source /etc/profile.d/modules.sh
    default_partition: compute
    max_nodes: 128
    max_walltime: "72:00:00"
    data_staging_enabled: true
    staging_dir: /scratch/${SLURM_USER}/staging
    monitoring_interval: 30

  - name: pbs-cluster
    cluster_type: pbs
    ssh:
      host: login.pbs-cluster.org
      user: ${PBS_USER}
      auth_method: key
      key_file: ~/.ssh/pbs_key
    work_dir: /home/${PBS_USER}/work
    default_queue: batch
    max_nodes: 64
    max_walltime: "48:00:00"
```

### Environment Variables

```bash
# Cluster credentials
export SLURM_USER=username
export SLURM_KEY_PASSPHRASE=secret
export PBS_USER=username

# Monitoring
export ALERT_WEBHOOK_URL=https://hooks.slack.com/...
export PROMETHEUS_ENDPOINT=http://prometheus:9090

# Recovery
export CHECKPOINT_BASE_PATH=/shared/checkpoints
```

### Python Configuration

```python
from hpc_connector import HPCClient, ClusterConfig
from hpc_connector.core.cluster import ClusterManager

# Load from file
manager = ClusterManager()
manager.load_from_file("clusters.yaml")

# Or create programmatically
config = ClusterConfig.from_dict({
    "name": "my-cluster",
    "cluster_type": "slurm",
    "ssh": {
        "host": "login.cluster.org",
        "user": "username",
        "auth_method": "key",
        "key_file": "~/.ssh/id_rsa",
    },
})
```

## Security

### SSH Key Management

1. **Generate dedicated keys**:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/hpc_connector_key -N ""
```

2. **Set appropriate permissions**:
```bash
chmod 600 ~/.ssh/hpc_connector_key
chmod 644 ~/.ssh/hpc_connector_key.pub
```

3. **Add to cluster authorized_keys**:
```bash
ssh-copy-id -i ~/.ssh/hpc_connector_key.pub user@login.cluster.org
```

4. **Use SSH agent**:
```bash
eval $(ssh-agent -s)
ssh-add ~/.ssh/hpc_connector_key
```

### Key Passphrases

For encrypted keys:

```python
config = ClusterConfig.from_dict({
    "ssh": {
        "auth_method": "key_with_passphrase",
        "key_file": "~/.ssh/encrypted_key",
        # Passphrase loaded from environment or secret manager
        "key_passphrase": os.environ.get("SSH_PASSPHRASE"),
    }
})
```

### Jump Host / Bastion

For clusters behind bastion hosts:

```yaml
ssh:
  host: compute-node.internal
  user: username
  auth_method: key
  proxy_host: bastion.cluster.org
  proxy_port: 22
  proxy_user: bastion-user
```

### Secrets Management

#### Using HashiCorp Vault

```python
import hvac

client = hvac.Client(url='https://vault.example.com')
client.auth.kubernetes.login(role='hpc-connector')

secret = client.secrets.kv.v2.read_secret_version(
    path='clusters/slurm'
)

config = ClusterConfig.from_dict(secret['data']['data'])
```

#### Using AWS Secrets Manager

```python
import boto3

secrets = boto3.client('secretsmanager')
response = secrets.get_secret_value(SecretId='hpc/slurm-cluster')

import json
config_data = json.loads(response['SecretString'])
config = ClusterConfig.from_dict(config_data)
```

## Cluster Setup

### SLURM Cluster

1. **Ensure SSH access**:
```bash
# Test connection
ssh -i ~/.ssh/key user@login.cluster.org "sinfo"
```

2. **Configure partitions**:
```bash
scontrol show partitions
```

3. **Verify job submission**:
```bash
sbatch --wrap="echo test" --partition=compute
```

### PBS/Torque Cluster

1. **Verify qsub works**:
```bash
ssh user@login.cluster.org "echo 'echo test' | qsub"
```

2. **Check queue configuration**:
```bash
qstat -q
```

### LSF Cluster

1. **Test bsub**:
```bash
ssh user@login.cluster.org "bsub -I echo test"
```

2. **Check queues**:
```bash
bqueues
```

### SGE Cluster

1. **Test qsub**:
```bash
ssh user@login.cluster.org "qsub -b y echo test"
```

2. **Check queues**:
```bash
qstat -g c
```

## Monitoring

### Prometheus Integration

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
jobs_submitted = Counter('hpc_jobs_submitted_total', 'Total jobs submitted', ['cluster'])
job_duration = Histogram('hpc_job_duration_seconds', 'Job duration', ['cluster', 'status'])

# Start metrics server
start_http_server(8000)

# Use in code
jobs_submitted.labels(cluster='slurm').inc()
```

### Grafana Dashboard

Import dashboard JSON from `monitoring/grafana-dashboard.json`.

Key panels:
- Jobs by status
- Queue depth over time
- Resource utilization
- Error rates
- Transfer speeds

### Alerting

Configure alert rules:

```yaml
groups:
  - name: hpc_alerts
    rules:
      - alert: HighQueueDepth
        expr: hpc_queue_depth > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High queue depth on {{ $labels.cluster }}"
      
      - alert: JobFailureRate
        expr: rate(hpc_jobs_failed_total[1h]) > 0.1
        for: 10m
        labels:
          severity: critical
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hpc_connector.log'),
        logging.StreamHandler()
    ]
)
```

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to cluster

```bash
# Test SSH connectivity
ssh -v -i ~/.ssh/key user@login.cluster.org

# Check key permissions
ls -la ~/.ssh/

# Verify key is loaded
ssh-add -l
```

**Solution**:
- Verify key permissions (600 for private key)
- Check firewall rules
- Ensure key is added to authorized_keys on cluster

### Job Submission Failures

**Problem**: Jobs fail to submit

```python
# Enable debug logging
import logging
logging.getLogger('hpc_connector').setLevel(logging.DEBUG)

# Check scheduler output
result = await connector.execute("scontrol show config")
print(result['stdout'])
```

**Common causes**:
- Invalid resource requests
- Queue/partition not available
- User limits exceeded
- Environment setup issues

### Data Transfer Issues

**Problem**: Slow or failed transfers

```bash
# Test bandwidth
iperf3 -c login.cluster.org

# Check disk quotas
ssh user@login.cluster.org "df -h"
```

**Optimizations**:
- Enable compression: `compress: true`
- Use multiple connections
- Transfer in parallel
- Exclude unnecessary files

### Performance Tuning

1. **Increase SSH multiplexing**:
```yaml
ssh:
  keepalive_interval: 30
  compress: true
```

2. **Adjust transfer chunk size**:
```python
# In data pipeline
chunk_size = 262144  # 256KB
```

3. **Limit concurrent transfers**:
```python
self._transfer_semaphore = asyncio.Semaphore(10)
```

### Debug Mode

```python
# Enable detailed debugging
import asyncio
asyncio.get_event_loop().set_debug(True)

# Log all SSH commands
class DebugConnector(SSHConnector):
    async def execute(self, command, **kwargs):
        print(f"[DEBUG] Executing: {command}")
        return await super().execute(command, **kwargs)
```

## Maintenance

### Regular Tasks

1. **Rotate log files**:
```bash
logrotate -f /etc/logrotate.d/hpc-connector
```

2. **Clean up old recovery state**:
```python
await recovery.cleanup_completed(max_age=timedelta(days=30))
```

3. **Update SSH keys**:
```bash
# Generate new key
ssh-keygen -t ed25519 -f ~/.ssh/hpc_new

# Distribute to clusters
for host in cluster1 cluster2 cluster3; do
    ssh-copy-id -i ~/.ssh/hpc_new.pub user@$host
done
```

### Backup and Recovery

Backup configuration:
```bash
tar czf hpc-config-backup.tar.gz ~/.hpc_connector/ clusters.yaml
```

Restore:
```bash
tar xzf hpc-config-backup.tar.gz -C ~/
```

## Support

For issues and questions:
- Check logs: `tail -f hpc_connector.log`
- Enable debug mode
- Review cluster-specific documentation
- Contact cluster administrators
