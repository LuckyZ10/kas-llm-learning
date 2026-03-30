#!/usr/bin/env python3
"""
hpc_integration/__init__.py
===========================
HPC计算集群集成模块 - 将系统连接到真实计算资源

功能模块：
1. cluster_connector - 集群连接管理
2. job_submitter - 增强型作业提交器 (Slurm/PBS/LSF)
3. resource_monitor - 计算资源监控
4. data_sync - 数据自动同步 (S3/MinIO/SFTP)
5. fault_tolerance - 容错与重试机制
6. checkpoint_manager - 断点续算管理
7. container_runtime - 容器运行时 (Docker/Singularity)
8. storage_backend - 对象存储后端

作者: DFT-MLIP HPC Integration Team
日期: 2026-03-10
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DFT-MLIP HPC Integration Team"

# 核心组件导入
from .cluster_connector import (
    ClusterConnector,
    ClusterConfig,
    ConnectionPool,
    SSHClusterConnector,
    KubernetesConnector,
    get_connector
)

from .job_submitter import (
    EnhancedJobSubmitter,
    JobTemplate,
    JobArrayBuilder,
    CalculationType,
    SubmissionStrategy
)

from .resource_monitor import (
    ResourceMonitor,
    ClusterMetrics,
    NodeStatus,
    QueueStats,
    GPUStats,
    monitor_context
)

from .data_sync import (
    DataSyncManager,
    SyncTask,
    SyncDirection,
    FileWatcher,
    AutoSyncPolicy
)

from .fault_tolerance import (
    FaultToleranceManager,
    RetryPolicy,
    CircuitBreaker,
    JobRetryHandler,
    FailureClassifier
)

from .checkpoint_manager import (
    CheckpointManager,
    Checkpoint,
    CalculationState,
    ResumeCapability,
    create_periodic_checkpoint
)

from .container_runtime import (
    ContainerRuntime,
    SingularityRuntime,
    DockerRuntime,
    ContainerImage,
    pull_image
)

from .storage_backend import (
    StorageBackend,
    S3Backend,
    MinIOBackend,
    SFTPBackend,
    StorageConfig,
    get_storage_backend
)

# 便捷函数
def create_hpc_client(
    cluster_config: dict = None,
    storage_config: dict = None,
    enable_monitoring: bool = True,
    enable_fault_tolerance: bool = True,
    enable_checkpointing: bool = True
):
    """
    创建HPC集成客户端
    
    Args:
        cluster_config: 集群配置字典
        storage_config: 存储配置字典
        enable_monitoring: 启用资源监控
        enable_fault_tolerance: 启用容错机制
        enable_checkpointing: 启用断点续算
    
    Returns:
        HPCIntegrationClient实例
    """
    from .client import HPCIntegrationClient
    return HPCIntegrationClient(
        cluster_config=cluster_config,
        storage_config=storage_config,
        enable_monitoring=enable_monitoring,
        enable_fault_tolerance=enable_fault_tolerance,
        enable_checkpointing=enable_checkpointing
    )


def submit_dft_workflow(
    working_dir: str,
    structures: list,
    calc_type: str = "vasp",
    resources: dict = None,
    checkpoint_interval: int = 3600
):
    """
    便捷函数: 提交DFT计算工作流
    
    Args:
        working_dir: 工作目录
        structures: 结构文件列表
        calc_type: 计算类型
        resources: 资源配置
        checkpoint_interval: 检查点间隔(秒)
    
    Returns:
        WorkflowHandle实例
    """
    client = create_hpc_client()
    return client.submit_dft_workflow(
        working_dir=working_dir,
        structures=structures,
        calc_type=calc_type,
        resources=resources,
        checkpoint_interval=checkpoint_interval
    )


__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 集群连接
    "ClusterConnector",
    "ClusterConfig", 
    "ConnectionPool",
    "SSHClusterConnector",
    "KubernetesConnector",
    "get_connector",
    
    # 作业提交
    "EnhancedJobSubmitter",
    "JobTemplate",
    "JobArrayBuilder",
    "CalculationType",
    "SubmissionStrategy",
    
    # 资源监控
    "ResourceMonitor",
    "ClusterMetrics",
    "NodeStatus",
    "QueueStats",
    "GPUStats",
    "monitor_context",
    
    # 数据同步
    "DataSyncManager",
    "SyncTask",
    "SyncDirection",
    "FileWatcher",
    "AutoSyncPolicy",
    
    # 容错
    "FaultToleranceManager",
    "RetryPolicy",
    "CircuitBreaker",
    "JobRetryHandler",
    "FailureClassifier",
    
    # 检查点
    "CheckpointManager",
    "Checkpoint",
    "CalculationState",
    "ResumeCapability",
    "create_periodic_checkpoint",
    
    # 容器
    "ContainerRuntime",
    "SingularityRuntime",
    "DockerRuntime",
    "ContainerImage",
    "pull_image",
    
    # 存储
    "StorageBackend",
    "S3Backend",
    "MinIOBackend",
    "SFTPBackend",
    "StorageConfig",
    "get_storage_backend",
    
    # 便捷函数
    "create_hpc_client",
    "submit_dft_workflow",
]
