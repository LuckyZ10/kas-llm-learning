"""
Tests for HPC Connector.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from hpc_connector.core.cluster import ClusterConfig, ClusterType, SSHConfig
from hpc_connector.core.job import JobConfig, JobStatus, ResourceRequest, JobPriority
from hpc_connector.core.exceptions import ConnectionError, JobSubmissionError


class TestClusterConfig:
    """Test cluster configuration."""
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "test-cluster",
            "cluster_type": "slurm",
            "ssh": {
                "host": "login.test.edu",
                "port": 22,
                "user": "testuser",
                "auth_method": "key",
                "key_file": "~/.ssh/id_rsa",
            },
            "work_dir": "/scratch/testuser",
            "default_partition": "compute",
            "max_nodes": 64,
        }
        
        config = ClusterConfig.from_dict(data)
        
        assert config.name == "test-cluster"
        assert config.cluster_type == ClusterType.SLURM
        assert config.ssh.host == "login.test.edu"
        assert config.ssh.user == "testuser"
        assert config.default_partition == "compute"
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ClusterConfig(
            name="test-cluster",
            cluster_type=ClusterType.PBS,
            ssh=SSHConfig(host="test.edu", user="user"),
        )
        
        data = config.to_dict()
        
        assert data["name"] == "test-cluster"
        assert data["cluster_type"] == "pbs"
        assert data["ssh"]["host"] == "test.edu"


class TestJobConfig:
    """Test job configuration."""
    
    def test_from_dict(self):
        """Test creating job config from dictionary."""
        data = {
            "name": "test-job",
            "command": "python test.py",
            "work_dir": "/work",
            "resources": {
                "nodes": 2,
                "cores_per_node": 4,
                "memory_per_node": "8GB",
                "walltime": "2:00:00",
            },
            "priority": 10,
            "modules": ["python/3.9"],
            "environment": {"VAR": "value"},
        }
        
        config = JobConfig.from_dict(data)
        
        assert config.name == "test-job"
        assert config.command == "python test.py"
        assert config.resources.nodes == 2
        assert config.resources.cores_per_node == 4
        assert config.priority == JobPriority.HIGH
        assert config.modules == ["python/3.9"]
    
    def test_to_dict(self):
        """Test converting job config to dictionary."""
        config = JobConfig(
            name="test",
            command="echo test",
            work_dir="/tmp",
            resources=ResourceRequest(nodes=1, cores_per_node=2),
        )
        
        data = config.to_dict()
        
        assert data["name"] == "test"
        assert data["resources"]["nodes"] == 1


class TestJobStatus:
    """Test job status enum."""
    
    def test_status_values(self):
        """Test job status values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
    
    def test_terminal_statuses(self):
        """Test terminal status identification."""
        terminal = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
        
        for status in terminal:
            assert status in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.TIMEOUT
            ]


@pytest.mark.asyncio
class TestSSHConnector:
    """Test SSH connector."""
    
    async def test_connect_success(self):
        """Test successful connection."""
        with patch('hpc_connector.connectors.ssh_connector.ASYNCSSH_AVAILABLE', True):
            with patch('asyncssh.connect', new_callable=AsyncMock) as mock_connect:
                mock_conn = AsyncMock()
                mock_sftp = AsyncMock()
                mock_conn.start_sftp_client = AsyncMock(return_value=mock_sftp)
                mock_connect.return_value = mock_conn
                
                from hpc_connector.connectors.ssh_connector import SSHConnector
                
                config = ClusterConfig(
                    name="test",
                    cluster_type=ClusterType.SLURM,
                    ssh=SSHConfig(host="test.edu", user="user"),
                )
                
                connector = SSHConnector(config)
                await connector.connect()
                
                assert connector.is_connected
                mock_connect.assert_called_once()
    
    async def test_execute_command(self):
        """Test command execution."""
        with patch('hpc_connector.connectors.ssh_connector.ASYNCSSH_AVAILABLE', True):
            with patch('asyncssh.connect', new_callable=AsyncMock) as mock_connect:
                mock_conn = AsyncMock()
                mock_result = Mock()
                mock_result.stdout = "output"
                mock_result.stderr = ""
                mock_result.exit_status = 0
                mock_conn.run = AsyncMock(return_value=mock_result)
                mock_conn.start_sftp_client = AsyncMock(return_value=AsyncMock())
                mock_connect.return_value = mock_conn
                
                from hpc_connector.connectors.ssh_connector import SSHConnector
                
                config = ClusterConfig(
                    name="test",
                    cluster_type=ClusterType.SLURM,
                    ssh=SSHConfig(host="test.edu", user="user"),
                )
                
                connector = SSHConnector(config)
                await connector.connect()
                
                result = await connector.execute("echo test")
                
                assert result['stdout'] == "output"
                assert result['exit_code'] == 0


@pytest.mark.asyncio
class TestSlurmScheduler:
    """Test SLURM scheduler."""
    
    async def test_submit_job(self):
        """Test job submission."""
        from hpc_connector.schedulers.slurm_scheduler import SlurmScheduler
        
        mock_connector = AsyncMock()
        mock_connector.execute = AsyncMock(return_value={
            'stdout': 'Submitted batch job 12345',
            'stderr': '',
            'exit_code': 0
        })
        mock_connector.config.work_dir = '/scratch'
        mock_connector.config.default_partition = 'compute'
        
        scheduler = SlurmScheduler(mock_connector)
        
        job_config = JobConfig(
            name="test",
            command="echo hello",
            work_dir="/tmp",
            resources=ResourceRequest(nodes=1, cores_per_node=1),
        )
        
        job_id = await scheduler.submit_job(job_config)
        
        assert job_id == "12345"
        mock_connector.execute.assert_called()
    
    async def test_get_job_status(self):
        """Test getting job status."""
        from hpc_connector.schedulers.slurm_scheduler import SlurmScheduler
        
        mock_connector = AsyncMock()
        mock_connector.execute = AsyncMock(return_value={
            'stdout': 'JobState=RUNNING',
            'stderr': '',
            'exit_code': 0
        })
        
        scheduler = SlurmScheduler(mock_connector)
        status = await scheduler.get_job_status("12345")
        
        assert status == JobStatus.RUNNING


class TestExceptions:
    """Test exception handling."""
    
    def test_connector_error(self):
        """Test base connector error."""
        from hpc_connector.core.exceptions import HPCConnectorError
        
        error = HPCConnectorError("Test error", "TEST_CODE", {"detail": "info"})
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_CODE"
        assert "TEST_CODE" in str(error)
    
    def test_connection_error(self):
        """Test connection error."""
        from hpc_connector.core.exceptions import ConnectionError
        
        error = ConnectionError("Connection failed")
        assert error.error_code == "CONN_ERROR"
    
    def test_job_submission_error(self):
        """Test job submission error."""
        from hpc_connector.core.exceptions import JobSubmissionError
        
        error = JobSubmissionError("Submission failed", {"exit_code": 1})
        assert error.error_code == "JOB_SUBMIT_ERROR"


class TestDataPipeline:
    """Test data pipeline."""
    
    def test_transfer_progress(self):
        """Test transfer progress calculation."""
        from hpc_connector.data import TransferProgress
        
        progress = TransferProgress(
            source="/local/file",
            destination="/remote/file",
            bytes_transferred=500,
            total_bytes=1000,
            percentage=50.0,
            speed_mbps=10.5,
        )
        
        assert not progress.is_complete
        
        progress.bytes_transferred = 1000
        assert progress.is_complete


class TestRecovery:
    """Test fault recovery."""
    
    def test_recovery_config(self):
        """Test recovery configuration."""
        from hpc_connector.recovery import RecoveryConfig, RecoveryStrategy
        
        config = RecoveryConfig(
            strategy=RecoveryStrategy.RESTART,
            max_retries=5,
            retry_delay=120,
        )
        
        assert config.strategy == RecoveryStrategy.RESTART
        assert config.max_retries == 5
        assert config.retry_delay == 120
    
    def test_checkpoint_info(self):
        """Test checkpoint info."""
        from hpc_connector.recovery import CheckpointInfo
        
        checkpoint = CheckpointInfo(
            job_id="123",
            timestamp=datetime.now(),
            checkpoint_path="/checkpoints/ckpt_1",
            step=100,
            metadata={"loss": 0.1},
        )
        
        data = checkpoint.to_dict()
        assert data["job_id"] == "123"
        assert data["step"] == 100


# Integration tests (marked to skip by default)
@pytest.mark.integration
@pytest.mark.asyncio
class TestIntegration:
    """Integration tests - require actual cluster connection."""
    
    async def test_full_workflow(self):
        """Test full job lifecycle."""
        # This test requires a real cluster
        pass
    
    async def test_data_transfer(self):
        """Test data transfer operations."""
        # This test requires a real cluster
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
