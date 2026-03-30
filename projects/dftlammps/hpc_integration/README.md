# HPC Integration Module

HPC计算集群集成模块 - 将DFT-MLIP/LAMMPS系统连接到真实计算资源。

## 功能特性

### 1. 集群连接管理 (`cluster_connector`)
- **SSH直连**: 通过Paramiko库实现安全的SSH连接
- **连接池**: 支持多连接池管理，提高并发性能
- **Kubernetes**: 支持云原生HPC部署
- **Slurm REST API**: 通过REST API管理作业
- **本地模式**: 本地执行用于测试和开发

### 2. 增强型作业提交 (`job_submitter`)
- **多调度器支持**: Slurm, PBS/Torque, LSF
- **作业模板**: 预定义的VASP, LAMMPS, NEP, DeepMD模板
- **作业数组**: 批量提交和管理
- **依赖管理**: 支持复杂的工作流依赖
- **容器支持**: 自动集成Docker/Singularity

### 3. 资源监控 (`resource_monitor`)
- **实时监控**: 集群利用率、队列状态
- **GPU监控**: NVIDIA GPU利用率和温度
- **作业监控**: 资源使用和效率分析
- **告警系统**: 可配置的告警条件
- **等待时间估算**: 智能估算作业等待时间

### 4. 数据同步 (`data_sync`)
- **自动同步**: 作业前后自动数据同步
- **增量同步**: 仅传输变更的文件
- **文件监控**: 实时检测文件变更
- **压缩传输**: 支持传输压缩

### 5. 容错与重试 (`fault_tolerance`)
- **断路器模式**: 防止级联故障
- **智能重试**: 指数退避和抖动
- **故障分类**: 自动分类故障类型
- **死信队列**: 记录失败的作业

### 6. 断点续算 (`checkpoint_manager`)
- **自动检查点**: 定期创建计算状态快照
- **状态验证**: 文件完整性校验
- **恢复分析**: 分析恢复能力和节省时间
- **多存储后端**: 本地和远程存储支持

### 7. 容器运行时 (`container_runtime`)
- **Singularity/Apptainer**: HPC首选容器方案
- **Docker**: rootless模式支持
- **自动镜像拉取**: 支持多种镜像源
- **GPU支持**: NVIDIA GPU直通

### 8. 存储后端 (`storage_backend`)
- **S3兼容**: AWS S3和兼容服务
- **MinIO**: 本地对象存储
- **SFTP**: SSH文件传输
- **本地存储**: 开发和测试使用

## 快速开始

### 安装依赖

```bash
pip install paramiko boto3 requests
```

### 基本用法

```python
from hpc_integration import create_hpc_client
from hpc_integration.cluster_connector import ClusterConfig

# 配置集群
config = ClusterConfig(
    name="my_cluster",
    host="hpc.example.com",
    username="user",
    key_file="~/.ssh/id_rsa",
    scheduler_type="slurm"
)

# 创建客户端并连接
client = create_hpc_client(
    cluster_config=config.to_dict(),
    enable_monitoring=True,
    enable_fault_tolerance=True,
    enable_checkpointing=True
)

with client:
    # 提交VASP作业
    from hpc_integration.job_submitter import JobTemplate
    
    template = JobTemplate.for_vasp(
        name="vasp_relax",
        nodes=2,
        cores_per_node=32,
        walltime_hours=24.0
    )
    
    job = client.submit_job(
        template=template,
        working_dir="./vasp_calc"
    )
    
    print(f"Job submitted: {job.job_id}")
    
    # 等待完成
    status = client.wait_for_job(job.job_id)
    print(f"Job completed with status: {status}")
```

### 批量提交

```python
from hpc_integration.job_submitter import JobArrayBuilder

# 创建模板
template = JobTemplate.for_vasp(name="batch_calc")

# 创建作业数组构建器
builder = JobArrayBuilder(template, Path("./batch_work"))

# 添加多个作业
for i in range(100):
    builder.add_job(
        working_dir=f"./calc_{i:04d}",
        job_name=f"calc_{i:04d}"
    )

# 提交数组作业
array_job = client.submit_array(builder)
```

### 检查点与恢复

```python
# 创建检查点
checkpoint = client.create_checkpoint(
    job_id="vasp_001",
    working_dir=Path("./calc"),
    metadata={"iteration": 100}
)

# 检查恢复能力
capability = client.can_resume("vasp_001", Path("./calc"))
if capability["can_resume"]:
    print(f"可以从检查点恢复，预计节省: {capability['estimated_time_saved_seconds']} 秒")
```

### 容错执行

```python
from hpc_integration.fault_tolerance import with_retry

@with_retry(max_attempts=5, backoff_factor=2.0)
def submit_with_retry():
    return client.submit_job(template, working_dir)

# 或使用断路器
from hpc_integration import FaultToleranceManager

ft = FaultToleranceManager()
result = ft.execute(
    risky_function,
    circuit_breaker_name="my_service",
    retry_policy=RetryPolicy(max_attempts=3)
)
```

## 配置示例

### 集群配置 (cluster_config.json)

```json
{
  "name": "slurm_cluster",
  "host": "hpc.university.edu",
  "scheduler_type": "slurm",
  "username": "researcher",
  "key_file": "~/.ssh/id_rsa",
  "remote_work_dir": "/scratch/researcher/dft",
  "default_queue": "normal",
  "max_nodes": 100
}
```

### 存储配置 (storage_config.json)

```json
{
  "type": "s3",
  "endpoint": "https://s3.amazonaws.com",
  "access_key": "YOUR_ACCESS_KEY",
  "secret_key": "YOUR_SECRET_KEY",
  "bucket": "dft-data",
  "region": "us-east-1"
}
```

### MinIO配置

```json
{
  "type": "minio",
  "endpoint": "http://localhost:9000",
  "access_key": "minioadmin",
  "secret_key": "minioadmin",
  "bucket": "dft-calculations"
}
```

## 高级用法

### 主动学习工作流

```python
from hpc_integration.examples.workflow_examples import ActiveLearningWorkflow

workflow = ActiveLearningWorkflow(client)
results = workflow.run_active_learning_loop(
    initial_structures=["struct1", "struct2"],
    work_dir=Path("./active_learning"),
    max_iterations=10
)
```

### 多尺度模拟

```python
from hpc_integration.examples.workflow_examples import MultiscaleWorkflow

workflow = MultiscaleWorkflow(client)
results = workflow.run_multiscale_simulation(
    system_config={"material": "LiCoO2", "size": "10nm"},
    work_dir=Path("./multiscale")
)
```

### 容器执行

```python
result = client.run_in_container(
    image="docker://vasp:v6.4.0",
    command=["mpirun", "vasp_std"],
    working_dir=Path("./calc"),
    bind_mounts={"./data": "/data"},
    gpus=1
)
```

## API参考

### 核心类

- `HPCIntegrationClient`: 主客户端类
- `ClusterConnector`: 集群连接器基类
- `EnhancedJobSubmitter`: 作业提交器
- `ResourceMonitor`: 资源监控器
- `CheckpointManager`: 检查点管理器
- `FaultToleranceManager`: 容错管理器
- `DataSyncManager`: 数据同步管理器

### 枚举类型

- `CalculationType`: VASP, LAMMPS, NEP, DEEPMD, etc.
- `ContainerEngine`: SINGULARITY, DOCKER, PODMAN
- `SyncDirection`: UPLOAD, DOWNLOAD, BIDIRECTIONAL
- `FailureType`: TRANSIENT, PERMANENT, NETWORK, etc.

## 测试

```bash
# 运行所有测试
cd /root/.openclaw/workspace/dftlammps/hpc_integration
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_hpc_integration.py::TestCircuitBreaker -v
```

## 示例

查看 `examples/` 目录:
- `basic_usage.py`: 基础用法示例
- `workflow_examples.py`: 复杂工作流示例
- `batch_submission.py`: 批量提交示例

## 故障排除

### SSH连接问题

```python
# 检查连接
connector = SSHClusterConnector(config)
if not connector.connect():
    print("连接失败，检查:")
    print("1. SSH密钥是否正确")
    print("2. 主机名和端口")
    print("3. 防火墙设置")
```

### 作业提交失败

```python
# 使用dry_run模式调试
template = JobTemplate.for_vasp()
spec = JobSpec(template=template, working_dir="./test")

# 仅生成脚本，不提交
submitted = submitter.submit(spec, dry_run=True)
```

### 检查点恢复失败

```python
# 验证文件完整性
capability = manager.analyze_resume_capability(job_id, work_dir)
if not capability.can_resume:
    print(f"缺少文件: {capability.missing_files}")
```

## 贡献

欢迎提交Issue和Pull Request!

## 许可证

MIT License

## 作者

DFT-MLIP HPC Integration Team

## 更新日志

### v1.0.0 (2026-03-10)
- 初始版本发布
- 支持Slurm/PBS/LSF调度器
- 实现断点续算功能
- 集成S3/MinIO存储
- 添加容器运行时支持
