#!/usr/bin/env python3
"""
tests/test_hpc_integration.py
=============================
HPC集成模块测试
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hpc_integration.cluster_connector import (
    ClusterConfig, SSHClusterConnector, LocalConnector,
    ConnectionPool, get_connector
)
from hpc_integration.job_submitter import (
    JobTemplate, JobArrayBuilder, JobSpec, EnhancedJobSubmitter,
    CalculationType, SubmittedJob
)
from hpc_integration.resource_monitor import (
    ResourceMonitor, ClusterMetrics, NodeStatus, QueueStats
)
from hpc_integration.fault_tolerance import (
    FaultToleranceManager, RetryPolicy, CircuitBreaker,
    CircuitBreakerState, FailureClassifier, FailureType
)
from hpc_integration.checkpoint_manager import (
    CheckpointManager, LocalCheckpointStorage, Checkpoint,
    CalculationState, ResumeCapability
)
from hpc_integration.container_runtime import (
    ContainerConfig, ContainerImage, ContainerEngine,
    SingularityRuntime, DockerRuntime
)
from hpc_integration.storage_backend import (
    StorageConfig, LocalBackend
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """临时目录fixture"""
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture
def cluster_config():
    """集群配置fixture"""
    return ClusterConfig(
        name="test_cluster",
        host="localhost",
        scheduler_type="slurm"
    )


@pytest.fixture
def mock_connector():
    """模拟连接器fixture"""
    connector = Mock()
    connector.execute.return_value = (0, "output", "")
    connector.exists.return_value = True
    connector.mkdir.return_value = True
    connector.upload.return_value = True
    connector.download.return_value = True
    return connector


@pytest.fixture
def vasp_template():
    """VASP模板fixture"""
    return JobTemplate.for_vasp(
        name="vasp_test",
        nodes=1,
        cores_per_node=8,
        walltime_hours=2.0
    )


# =============================================================================
# Cluster Connector Tests
# =============================================================================

class TestClusterConfig:
    """集群配置测试"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = ClusterConfig(
            name="test",
            host="hpc.example.com",
            scheduler_type="slurm"
        )
        
        assert config.name == "test"
        assert config.host == "hpc.example.com"
        assert config.scheduler_type == "slurm"
        assert config.port == 22  # 默认值
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = ClusterConfig(
            name="test",
            host="localhost"
        )
        
        data = config.to_dict()
        
        assert data["name"] == "test"
        assert data["host"] == "localhost"
        assert "scheduler_type" in data
    
    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "name": "test",
            "host": "localhost",
            "scheduler_type": "pbs"
        }
        
        config = ClusterConfig.from_dict(data)
        
        assert config.name == "test"
        assert config.scheduler_type == "pbs"


class TestLocalConnector:
    """本地连接器测试"""
    
    def test_connect(self, cluster_config):
        """测试连接"""
        connector = LocalConnector(cluster_config)
        
        assert connector.connect() is True
        assert connector.is_connected() is True
    
    def test_execute(self, cluster_config):
        """测试命令执行"""
        connector = LocalConnector(cluster_config)
        connector.connect()
        
        code, stdout, stderr = connector.execute("echo 'hello'")
        
        assert code == 0
        assert "hello" in stdout
    
    def test_file_operations(self, cluster_config, temp_dir):
        """测试文件操作"""
        connector = LocalConnector(cluster_config)
        connector.connect()
        
        # 创建目录
        test_dir = temp_dir / "test_dir"
        assert connector.mkdir(str(test_dir)) is True
        assert test_dir.exists()
        
        # 检查存在
        assert connector.exists(str(test_dir)) is True
        assert connector.exists("/nonexistent/path") is False


# =============================================================================
# Job Submitter Tests
# =============================================================================

class TestJobTemplate:
    """作业模板测试"""
    
    def test_vasp_template_creation(self):
        """测试VASP模板创建"""
        template = JobTemplate.for_vasp(
            name="vasp_relax",
            nodes=2,
            cores_per_node=32
        )
        
        assert template.name == "vasp_relax"
        assert template.calculation_type == CalculationType.VASP
        assert template.nodes == 2
        assert template.cores_per_node == 32
        assert "vasp" in template.modules
    
    def test_lammps_template_creation(self):
        """测试LAMMPS模板创建"""
        template = JobTemplate.for_lammps(
            name="md_sim",
            nodes=1,
            cores_per_node=16
        )
        
        assert template.calculation_type == CalculationType.LAMMPS
        assert "lammps" in template.modules
        assert template.executable == "lmp"
    
    def test_ml_template_creation(self):
        """测试ML训练模板"""
        template = JobTemplate.for_ml_training(
            name="deepmd_train",
            num_gpus=2
        )
        
        assert template.calculation_type == CalculationType.DEEPMD
        assert template.gpus == 2
        assert template.gpu_type == "a100"
    
    def test_template_serialization(self):
        """测试模板序列化"""
        template = JobTemplate.for_vasp()
        
        data = template.to_dict()
        restored = JobTemplate.from_dict(data)
        
        assert restored.name == template.name
        assert restored.nodes == template.nodes
        assert restored.calculation_type == template.calculation_type


class TestJobArrayBuilder:
    """作业数组构建器测试"""
    
    def test_add_job(self, vasp_template, temp_dir):
        """测试添加作业"""
        builder = JobArrayBuilder(vasp_template, temp_dir)
        
        idx = builder.add_job(
            working_dir=temp_dir / "job1",
            job_name="test_job_1"
        )
        
        assert idx == 1
        assert len(builder) == 1
    
    def test_add_multiple_jobs(self, vasp_template, temp_dir):
        """测试添加多个作业"""
        builder = JobArrayBuilder(vasp_template, temp_dir)
        
        for i in range(5):
            builder.add_job(
                working_dir=temp_dir / f"job_{i}",
                job_name=f"job_{i}"
            )
        
        assert len(builder) == 5
    
    def test_build_array_job(self, vasp_template, temp_dir):
        """测试构建数组作业"""
        builder = JobArrayBuilder(vasp_template, temp_dir)
        
        for i in range(3):
            builder.add_job(working_dir=temp_dir / f"job_{i}")
        
        array_spec = builder.build_array_job()
        
        assert array_spec.array_range == (1, 3)
        assert array_spec.job_name.endswith("_array")


class TestEnhancedJobSubmitter:
    """增强型作业提交器测试"""
    
    def test_generate_slurm_script(self, mock_connector, vasp_template, temp_dir):
        """测试生成Slurm脚本"""
        submitter = EnhancedJobSubmitter(mock_connector, "slurm")
        
        spec = JobSpec(
            template=vasp_template,
            working_dir=temp_dir
        )
        
        script = submitter._generate_script(spec)
        
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=vasp_test" in script
        assert "#SBATCH --nodes=1" in script
        assert "mpirun" in script
    
    def test_generate_pbs_script(self, mock_connector, vasp_template, temp_dir):
        """测试生成PBS脚本"""
        submitter = EnhancedJobSubmitter(mock_connector, "pbs")
        
        spec = JobSpec(
            template=vasp_template,
            working_dir=temp_dir
        )
        
        script = submitter._generate_script(spec)
        
        assert "#!/bin/bash" in script
        assert "#PBS -N vasp_test" in script
    
    def test_parse_slurm_job_id(self):
        """测试解析Slurm作业ID"""
        submitter = EnhancedJobSubmitter(Mock(), "slurm")
        
        job_id = submitter._parse_job_id("Submitted batch job 12345")
        
        assert job_id == "12345"


# =============================================================================
# Resource Monitor Tests
# =============================================================================

class TestResourceMonitor:
    """资源监控器测试"""
    
    def test_initialization(self, mock_connector):
        """测试初始化"""
        monitor = ResourceMonitor(mock_connector, "slurm")
        
        assert monitor.connector == mock_connector
        assert monitor.scheduler_type == "slurm"
    
    def test_get_current_metrics_empty(self, mock_connector):
        """测试获取空指标"""
        monitor = ResourceMonitor(mock_connector, "slurm")
        
        metrics = monitor.get_current_metrics()
        
        assert metrics is None
    
    def test_estimate_wait_time(self, mock_connector):
        """测试等待时间估算"""
        monitor = ResourceMonitor(mock_connector, "slurm")
        
        # Mock队列数据
        from hpc_integration.resource_monitor import QueueStats
        monitor._queues["normal"] = QueueStats(
            name="normal",
            available=True,
            pending_jobs=10
        )
        
        wait_time = monitor.estimate_wait_time(queue="normal")
        
        assert wait_time is not None


# =============================================================================
# Fault Tolerance Tests
# =============================================================================

class TestCircuitBreaker:
    """断路器测试"""
    
    def test_initial_state(self):
        """测试初始状态"""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute() is True
    
    def test_failure_counting(self):
        """测试故障计数"""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        
        assert cb._failure_count == 2
        assert cb.state == CircuitBreakerState.CLOSED  # 还未达到阈值
    
    def test_open_state(self):
        """测试打开状态"""
        cb = CircuitBreaker("test", failure_threshold=2)
        
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() is False
    
    def test_recovery(self):
        """测试恢复"""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        
        # 触发打开
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # 等待恢复
        import time
        time.sleep(0.2)
        
        assert cb.can_execute() is True  # 应该进入HALF_OPEN
    
    def test_successful_call(self):
        """测试成功调用"""
        cb = CircuitBreaker("test")
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        
        assert result == "success"
        assert cb._total_successes == 1
    
    def test_failed_call(self):
        """测试失败调用"""
        cb = CircuitBreaker("test")
        
        def fail_func():
            raise ValueError("error")
        
        with pytest.raises(ValueError):
            cb.call(fail_func)
        
        assert cb._total_failures == 1


class TestFailureClassifier:
    """故障分类器测试"""
    
    def test_classify_transient_error(self):
        """测试分类瞬时错误"""
        failure_type, severity = FailureClassifier.classify(
            "Service temporarily unavailable, try again"
        )
        
        assert failure_type == FailureType.TRANSIENT
    
    def test_classify_resource_error(self):
        """测试分类资源错误"""
        failure_type, severity = FailureClassifier.classify(
            "No space left on device"
        )
        
        assert failure_type == FailureType.RESOURCE
    
    def test_classify_timeout_error(self):
        """测试分类超时错误"""
        failure_type, severity = FailureClassifier.classify(
            "Connection timeout"
        )
        
        assert failure_type == FailureType.TIMEOUT


class TestRetryPolicy:
    """重试策略测试"""
    
    def test_exponential_delay(self):
        """测试指数退避延迟"""
        policy = RetryPolicy(
            initial_delay=1.0,
            backoff_factor=2.0,
            backoff_strategy="exponential"
        )
        
        d0 = policy.calculate_delay(0)
        d1 = policy.calculate_delay(1)
        d2 = policy.calculate_delay(2)
        
        assert d0 < d1 < d2
        assert d1 >= 1.9  # ~2.0 with jitter
        assert d2 >= 3.8  # ~4.0 with jitter
    
    def test_linear_delay(self):
        """测试线性延迟"""
        policy = RetryPolicy(
            initial_delay=1.0,
            backoff_strategy="linear"
        )
        
        d0 = policy.calculate_delay(0)
        d1 = policy.calculate_delay(1)
        
        assert d1 >= d0 * 1.8  # ~2x with jitter
    
    def test_should_retry_logic(self):
        """测试重试判断逻辑"""
        policy = RetryPolicy(
            max_attempts=3,
            retryable_exceptions=(ValueError,)
        )
        
        assert policy.should_retry(ValueError("test"), 0) is True
        assert policy.should_retry(ValueError("test"), 2) is False  # 最后一次
        assert policy.should_retry(TypeError("test"), 0) is False  # 非可重试异常


# =============================================================================
# Checkpoint Manager Tests
# =============================================================================

class TestCheckpointManager:
    """检查点管理器测试"""
    
    def test_create_checkpoint(self, temp_dir):
        """测试创建检查点"""
        storage = LocalCheckpointStorage(temp_dir)
        manager = CheckpointManager(storage)
        
        # 创建测试文件
        (temp_dir / "test.txt").write_text("test content")
        
        checkpoint = manager.create_checkpoint(
            job_id="test_job",
            working_dir=temp_dir,
            state=CalculationState.RUNNING,
            iteration=10,
            total_steps=100,
            completed_steps=50
        )
        
        assert checkpoint is not None
        assert checkpoint.job_id == "test_job"
        assert checkpoint.progress_pct == 50.0
        assert "test.txt" in checkpoint.file_checksums
    
    def test_list_checkpoints(self, temp_dir):
        """测试列出检查点"""
        storage = LocalCheckpointStorage(temp_dir)
        manager = CheckpointManager(storage)
        
        # 创建多个检查点
        for i in range(3):
            manager.create_checkpoint(
                job_id="job_1",
                working_dir=temp_dir,
                completed_steps=i * 10
            )
        
        checkpoints = manager.list_checkpoints("job_1")
        
        assert len(checkpoints) == 3
    
    def test_resume_capability(self, temp_dir):
        """测试恢复能力分析"""
        storage = LocalCheckpointStorage(temp_dir)
        manager = CheckpointManager(storage)
        
        # 创建检查点
        manager.create_checkpoint(
            job_id="test_job",
            working_dir=temp_dir,
            completed_steps=50
        )
        
        capability = manager.analyze_resume_capability("test_job", temp_dir)
        
        assert isinstance(capability, ResumeCapability)


# =============================================================================
# Container Runtime Tests
# =============================================================================

class TestContainerImage:
    """容器镜像测试"""
    
    def test_from_string_docker(self):
        """测试从Docker格式字符串解析"""
        image = ContainerImage.from_string("ubuntu:20.04")
        
        assert image.name == "ubuntu"
        assert image.tag == "20.04"
        assert image.registry is None
    
    def test_from_string_with_registry(self):
        """测试带仓库的字符串解析"""
        image = ContainerImage.from_string("docker.io/library/ubuntu:latest")
        
        assert image.registry == "docker.io"
        assert image.name == "library/ubuntu"
        assert image.tag == "latest"
    
    def test_full_name(self):
        """测试完整名称"""
        image = ContainerImage(
            name="vasp",
            tag="6.4.0",
            registry="hpc.registry.io"
        )
        
        assert image.full_name == "hpc.registry.io/vasp:6.4.0"


class TestContainerConfig:
    """容器配置测试"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = ContainerConfig(
            image=ContainerImage(name="test", tag="latest"),
            engine=ContainerEngine.SINGULARITY,
            gpus=2,
            bind_mounts={"/host": "/container"}
        )
        
        assert config.engine == ContainerEngine.SINGULARITY
        assert config.gpus == 2
        assert "/host" in config.bind_mounts


# =============================================================================
# Storage Backend Tests
# =============================================================================

class TestLocalStorageBackend:
    """本地存储后端测试"""
    
    def test_connect(self, temp_dir):
        """测试连接"""
        config = StorageConfig(
            type="local",
            local_path=str(temp_dir)
        )
        backend = LocalBackend(config)
        
        assert backend.connect() is True
    
    def test_file_operations(self, temp_dir):
        """测试文件操作"""
        config = StorageConfig(
            type="local",
            local_path=str(temp_dir)
        )
        backend = LocalBackend(config)
        backend.connect()
        
        # 创建测试文件
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        # 测试存在性
        assert backend.exists("test.txt") is True
        assert backend.is_file("test.txt") is True
        assert backend.is_dir("test.txt") is False
    
    def test_upload_download(self, temp_dir):
        """测试上传下载"""
        config = StorageConfig(
            type="local",
            local_path=str(temp_dir / "remote")
        )
        backend = LocalBackend(config)
        backend.connect()
        
        # 上传
        local_file = temp_dir / "local.txt"
        local_file.write_text("content")
        
        result = backend.upload_file(local_file, "uploaded.txt")
        
        assert result.success is True
        assert result.files_transferred == 1
        
        # 下载
        download_path = temp_dir / "downloaded.txt"
        result = backend.download_file("uploaded.txt", download_path)
        
        assert result.success is True
        assert download_path.exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """集成测试"""
    
    def test_full_job_lifecycle(self, mock_connector, vasp_template, temp_dir):
        """测试完整作业生命周期"""
        # 1. 创建提交器
        submitter = EnhancedJobSubmitter(mock_connector, "slurm")
        
        # 2. 创建作业规格
        spec = JobSpec(
            template=vasp_template,
            working_dir=temp_dir
        )
        
        # 3. 生成脚本
        script = submitter._generate_script(spec)
        assert "#!/bin/bash" in script
        
        # 4. Mock提交
        mock_connector.execute.return_value = (0, "Submitted batch job 12345", "")
        
        # 5. 提交作业
        # submitted = submitter.submit(spec)  # 需要完整mock
    
    def test_fault_tolerance_integration(self):
        """测试容错集成"""
        ft_manager = FaultToleranceManager()
        
        # 使用断路器包装函数
        def risky_operation():
            import random
            if random.random() < 0.5:
                raise ValueError("Random failure")
            return "success"
        
        # 多次调用
        for _ in range(5):
            try:
                result = ft_manager.execute(
                    risky_operation,
                    circuit_breaker_name="test_op",
                    retry_policy=RetryPolicy(max_attempts=2)
                )
            except:
                pass
        
        # 检查统计
        stats = ft_manager.get_stats()
        assert "circuit_breakers" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
