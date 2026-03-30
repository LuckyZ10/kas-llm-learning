#!/usr/bin/env python3
"""
client.py
=========
HPC集成客户端 - 统一入口

将所有HPC集成功能整合为统一的客户端接口。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field

# 导入所有模块
from .cluster_connector import (
    ClusterConfig, ClusterConnector, get_connector
)
from .job_submitter import (
    EnhancedJobSubmitter, JobTemplate, JobArrayBuilder,
    JobSpec, CalculationType, SubmittedJob
)
from .resource_monitor import (
    ResourceMonitor, ClusterMetrics, monitor_context
)
from .data_sync import (
    DataSyncManager, SyncTask, SyncDirection,
    AutoSyncPolicy, FileWatcher
)
from .fault_tolerance import (
    FaultToleranceManager, RetryPolicy, CircuitBreaker
)
from .checkpoint_manager import (
    CheckpointManager, Checkpoint, CalculationState,
    ResumeCapability
)
from .container_runtime import (
    ContainerRuntime, ContainerConfig, ContainerImage,
    ContainerEngine, get_runtime, detect_available_runtime
)
from .storage_backend import (
    StorageBackend, StorageConfig, S3Backend, MinIOBackend,
    SFTPBackend, get_storage_backend
)

logger = logging.getLogger(__name__)


@dataclass
class HPCClientConfig:
    """HPC客户端配置"""
    # 集群配置
    cluster_config: Optional[ClusterConfig] = None
    
    # 存储配置
    storage_config: Optional[StorageConfig] = None
    
    # 功能开关
    enable_monitoring: bool = True
    enable_fault_tolerance: bool = True
    enable_checkpointing: bool = True
    enable_auto_sync: bool = True
    
    # 默认设置
    default_scheduler: str = "slurm"
    default_queue: Optional[str] = None
    default_container_engine: str = "singularity"
    
    # 路径
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    staging_dir: Path = field(default_factory=lambda: Path("./staging"))
    
    @classmethod
    def from_file(cls, path: str) -> "HPCClientConfig":
        """从文件加载配置"""
        with open(path) as f:
            data = json.load(f)
        
        if "cluster_config" in data:
            data["cluster_config"] = ClusterConfig(**data["cluster_config"])
        if "storage_config" in data:
            data["storage_config"] = StorageConfig(**data["storage_config"])
        if "checkpoint_dir" in data:
            data["checkpoint_dir"] = Path(data["checkpoint_dir"])
        if "staging_dir" in data:
            data["staging_dir"] = Path(data["staging_dir"])
        
        return cls(**data)
    
    def to_file(self, path: str):
        """保存配置到文件"""
        data = {
            "cluster_config": self.cluster_config.to_dict() if self.cluster_config else None,
            "storage_config": self.storage_config.to_dict() if self.storage_config else None,
            "enable_monitoring": self.enable_monitoring,
            "enable_fault_tolerance": self.enable_fault_tolerance,
            "enable_checkpointing": self.enable_checkpointing,
            "enable_auto_sync": self.enable_auto_sync,
            "default_scheduler": self.default_scheduler,
            "default_queue": self.default_queue,
            "default_container_engine": self.default_container_engine,
            "checkpoint_dir": str(self.checkpoint_dir),
            "staging_dir": str(self.staging_dir)
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class HPCIntegrationClient:
    """
    HPC集成客户端
    
    统一接口，整合所有HPC功能：
    - 集群连接
    - 作业提交
    - 资源监控
    - 数据同步
    - 容错处理
    - 断点续算
    """
    
    def __init__(self, config: HPCClientConfig = None, **kwargs):
        """
        初始化HPC客户端
        
        Args:
            config: HPCClientConfig实例
            **kwargs: 快捷配置参数
        """
        if config is None:
            config = HPCClientConfig(**kwargs)
        
        self.config = config
        
        # 初始化组件
        self._connector: Optional[ClusterConnector] = None
        self._job_submitter: Optional[EnhancedJobSubmitter] = None
        self._resource_monitor: Optional[ResourceMonitor] = None
        self._data_sync: Optional[DataSyncManager] = None
        self._fault_tolerance: Optional[FaultToleranceManager] = None
        self._checkpoint_manager: Optional[CheckpointManager] = None
        self._storage_backend: Optional[StorageBackend] = None
        
        # 连接状态
        self._connected = False
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化组件"""
        # 容错管理器（最先初始化，其他组件可能依赖）
        if self.config.enable_fault_tolerance:
            self._fault_tolerance = FaultToleranceManager()
        
        # 存储后端
        if self.config.storage_config:
            self._storage_backend = get_storage_backend(self.config.storage_config)
        
        # 检查点管理器
        if self.config.enable_checkpointing:
            from .checkpoint_manager import LocalCheckpointStorage
            storage = LocalCheckpointStorage(self.config.checkpoint_dir)
            self._checkpoint_manager = CheckpointManager(storage)
    
    def connect(self) -> bool:
        """
        连接到HPC集群
        
        Returns:
            是否成功
        """
        if not self.config.cluster_config:
            logger.error("No cluster configuration provided")
            return False
        
        try:
            # 创建连接器
            self._connector = get_connector(self.config.cluster_config)
            
            # 建立连接
            if not self._connector.connect():
                logger.error("Failed to connect to cluster")
                return False
            
            # 初始化作业提交器
            self._job_submitter = EnhancedJobSubmitter(
                self._connector,
                self.config.default_scheduler
            )
            
            # 初始化资源监控
            if self.config.enable_monitoring:
                self._resource_monitor = ResourceMonitor(
                    self._connector,
                    self.config.default_scheduler
                )
                self._resource_monitor.start_monitoring()
            
            # 初始化数据同步
            if self.config.enable_auto_sync and self._storage_backend:
                self._data_sync = DataSyncManager(
                    self._storage_backend,
                    AutoSyncPolicy()
                )
                self._data_sync.start_auto_sync()
            
            self._connected = True
            logger.info("HPC client connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._resource_monitor:
            self._resource_monitor.stop_monitoring()
        
        if self._data_sync:
            self._data_sync.stop_auto_sync()
        
        if self._connector:
            self._connector.disconnect()
        
        self._connected = False
        logger.info("HPC client disconnected")
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
    
    # ========== 作业提交接口 ==========
    
    def submit_job(
        self,
        template: JobTemplate,
        working_dir: Path,
        dependencies: List[str] = None,
        dry_run: bool = False
    ) -> SubmittedJob:
        """
        提交单个作业
        
        Args:
            template: 作业模板
            working_dir: 工作目录
            dependencies: 依赖的作业ID列表
            dry_run: 仅生成脚本不提交
        
        Returns:
            SubmittedJob实例
        """
        self._ensure_connected()
        
        spec = JobSpec(
            template=template,
            working_dir=Path(working_dir),
            dependencies=dependencies or []
        )
        
        # 使用容错执行
        if self._fault_tolerance:
            return self._fault_tolerance.execute(
                self._job_submitter.submit,
                circuit_breaker_name="job_submission",
                job_id=None,
                spec=spec,
                dry_run=dry_run
            )
        else:
            return self._job_submitter.submit(spec, dry_run)
    
    def submit_dft_workflow(
        self,
        working_dir: str,
        structures: List[str],
        calc_type: str = "vasp",
        resources: dict = None,
        checkpoint_interval: int = 3600
    ) -> "WorkflowHandle":
        """
        提交DFT计算工作流
        
        Args:
            working_dir: 工作目录
            structures: 结构文件列表
            calc_type: 计算类型
            resources: 资源配置
            checkpoint_interval: 检查点间隔
        
        Returns:
            WorkflowHandle实例
        """
        self._ensure_connected()
        
        work_dir = Path(working_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建作业模板
        if calc_type == "vasp":
            template = JobTemplate.for_vasp(**(resources or {}))
        elif calc_type == "lammps":
            template = JobTemplate.for_lammps(**(resources or {}))
        else:
            template = JobTemplate(
                name=f"{calc_type}_calc",
                calculation_type=CalculationType(calc_type),
                **(resources or {})
            )
        
        # 创建作业数组构建器
        builder = JobArrayBuilder(template, work_dir)
        
        for i, structure in enumerate(structures):
            job_work_dir = work_dir / f"calc_{i:04d}"
            builder.add_job(
                working_dir=job_work_dir,
                job_name=f"{calc_type}_{i:04d}",
                custom_inputs={
                    'pre_commands': [f"cp {structure} ./"]
                }
            )
        
        # 提交作业数组
        submitted = self._job_submitter.submit_array(builder)
        
        return WorkflowHandle(
            client=self,
            job_id=submitted.job_id,
            working_dir=work_dir,
            num_jobs=len(structures)
        )
    
    def get_job_status(self, job_id: str) -> dict:
        """获取作业状态"""
        self._ensure_connected()
        
        # 使用调度器查询
        from .hpc.scheduler import create_scheduler
        scheduler = create_scheduler(self.config.default_scheduler)
        jobs = scheduler.query(job_id)
        
        if jobs:
            job = jobs[0]
            return {
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status.value,
                "elapsed_time": str(job.elapsed_time) if job.elapsed_time else None
            }
        
        return {"job_id": job_id, "status": "unknown"}
    
    def cancel_job(self, job_id: str) -> bool:
        """取消作业"""
        self._ensure_connected()
        
        from .hpc.scheduler import create_scheduler
        scheduler = create_scheduler(self.config.default_scheduler)
        return scheduler.cancel(job_id)
    
    def wait_for_job(self, job_id: str, timeout: float = None) -> str:
        """等待作业完成"""
        self._ensure_connected()
        
        from .hpc.scheduler import create_scheduler
        scheduler = create_scheduler(self.config.default_scheduler)
        status = scheduler.wait_for_job(job_id, timeout=timeout)
        return status.value
    
    # ========== 资源监控接口 ==========
    
    def get_cluster_metrics(self) -> Optional[ClusterMetrics]:
        """获取集群指标"""
        if self._resource_monitor:
            return self._resource_monitor.get_current_metrics()
        return None
    
    def estimate_wait_time(
        self,
        nodes: int = 1,
        cores: int = 1,
        gpus: int = 0,
        walltime_hours: float = 1.0,
        queue: str = None
    ) -> str:
        """估算等待时间"""
        if self._resource_monitor:
            wait_time = self._resource_monitor.estimate_wait_time(
                nodes, cores, gpus, walltime_hours, queue
            )
            if wait_time:
                return str(wait_time)
        return "unknown"
    
    def get_best_queue(
        self,
        nodes: int = 1,
        cores: int = 1,
        gpus: int = 0
    ) -> Optional[str]:
        """获取最佳队列"""
        if self._resource_monitor:
            return self._resource_monitor.get_best_queue(nodes, cores, gpus)
        return None
    
    # ========== 数据同步接口 ==========
    
    def sync_to_remote(
        self,
        local_path: Path,
        remote_path: str,
        wait: bool = False
    ) -> dict:
        """同步到远程"""
        if not self._data_sync:
            return {"success": False, "error": "Data sync not enabled"}
        
        result = self._data_sync.sync_before_job(local_path, remote_path)
        
        return {
            "success": result.success,
            "files_transferred": result.files_transferred,
            "bytes_transferred": result.bytes_transferred,
            "errors": result.errors
        }
    
    def sync_from_remote(
        self,
        remote_path: str,
        local_path: Path,
        wait: bool = False
    ) -> dict:
        """从远程同步"""
        if not self._data_sync:
            return {"success": False, "error": "Data sync not enabled"}
        
        result = self._data_sync.sync_after_job(local_path, remote_path)
        
        return {
            "success": result.success,
            "files_transferred": result.files_transferred,
            "bytes_transferred": result.bytes_transferred,
            "errors": result.errors
        }
    
    # ========== 检查点接口 ==========
    
    def create_checkpoint(
        self,
        job_id: str,
        working_dir: Path,
        metadata: dict = None
    ) -> Optional[Checkpoint]:
        """创建检查点"""
        if self._checkpoint_manager:
            return self._checkpoint_manager.create_checkpoint(
                job_id=job_id,
                working_dir=working_dir,
                metadata=metadata or {}
            )
        return None
    
    def can_resume(self, job_id: str, working_dir: Path) -> dict:
        """检查是否可以恢复"""
        if self._checkpoint_manager:
            capability = self._checkpoint_manager.analyze_resume_capability(
                job_id, working_dir
            )
            return capability.to_dict()
        return {"can_resume": False}
    
    def list_checkpoints(self, job_id: str = None) -> List[dict]:
        """列出检查点"""
        if self._checkpoint_manager:
            checkpoints = self._checkpoint_manager.list_checkpoints(job_id)
            return [ckpt.to_dict() for ckpt in checkpoints]
        return []
    
    # ========== 容器接口 ==========
    
    def run_in_container(
        self,
        image: str,
        command: List[str],
        working_dir: Path = None,
        bind_mounts: dict = None,
        gpus: int = 0
    ) -> dict:
        """
        在容器中运行命令
        
        Args:
            image: 容器镜像
            command: 执行命令
            working_dir: 工作目录
            bind_mounts: 挂载映射
            gpus: GPU数量
        
        Returns:
            执行结果
        """
        container_image = ContainerImage.from_string(image)
        
        config = ContainerConfig(
            image=container_image,
            engine=ContainerEngine(self.config.default_container_engine),
            bind_mounts=bind_mounts or {},
            gpus=gpus if gpus > 0 else None
        )
        
        runtime = get_runtime(config)
        
        result = runtime.run(command, working_dir)
        
        return {
            "success": result.success,
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": result.duration_seconds
        }
    
    # ========== 内部方法 ==========
    
    def _ensure_connected(self):
        """确保已连接"""
        if not self._connected:
            raise ConnectionError("Not connected to HPC cluster. Call connect() first.")
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


class WorkflowHandle:
    """工作流句柄"""
    
    def __init__(
        self,
        client: HPCIntegrationClient,
        job_id: str,
        working_dir: Path,
        num_jobs: int
    ):
        self.client = client
        self.job_id = job_id
        self.working_dir = working_dir
        self.num_jobs = num_jobs
        self._status = "submitted"
    
    @property
    def status(self) -> str:
        """获取状态"""
        info = self.client.get_job_status(self.job_id)
        return info.get("status", "unknown")
    
    def wait(self, timeout: float = None) -> str:
        """等待完成"""
        return self.client.wait_for_job(self.job_id, timeout)
    
    def cancel(self) -> bool:
        """取消工作流"""
        return self.client.cancel_job(self.job_id)
    
    def sync_results(self) -> dict:
        """同步结果"""
        return self.client.sync_from_remote(
            str(self.working_dir),
            self.working_dir
        )
