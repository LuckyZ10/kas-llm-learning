"""Tests for HPC scheduler module."""

import pytest

from dftlammps.hpc import (
    SchedulerType,
    JobStatus,
    ResourceRequest,
    JobSpec,
)


class TestSchedulerType:
    """Test SchedulerType enum."""
    
    def test_enum_values(self):
        """Test enum value assignments."""
        assert SchedulerType.SLURM.value == "slurm"
        assert SchedulerType.PBS.value == "pbs"
        assert SchedulerType.LSF.value == "lsf"
        assert SchedulerType.LOCAL.value == "local"


class TestJobStatus:
    """Test JobStatus enum."""
    
    def test_enum_values(self):
        """Test enum value assignments."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"
        assert JobStatus.TIMEOUT.value == "timeout"


class TestResourceRequest:
    """Test ResourceRequest dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        resources = ResourceRequest()
        assert resources.num_nodes == 1
        assert resources.num_cores_per_node == 1
        assert resources.num_gpus == 0
        assert resources.walltime_hours == 24.0
        assert resources.exclusive is False
    
    def test_slurm_conversion(self):
        """Test conversion to Slurm format."""
        resources = ResourceRequest(
            num_nodes=2,
            num_cores_per_node=32,
            walltime_hours=48.0,
            partition="gpu"
        )
        slurm_str = resources.to_slurm()
        assert "#SBATCH --nodes=2" in slurm_str
        assert "#SBATCH --ntasks-per-node=32" in slurm_str
        assert "#SBATCH --partition=gpu" in slurm_str
    
    def test_pbs_conversion(self):
        """Test conversion to PBS format."""
        resources = ResourceRequest(
            num_nodes=2,
            num_cores_per_node=32,
            walltime_hours=24.0
        )
        pbs_str = resources.to_pbs()
        assert "#PBS" in pbs_str
        assert "nodes=2:ppn=32" in pbs_str


class TestJobSpec:
    """Test JobSpec dataclass."""
    
    def test_default_values(self):
        """Test default job specification."""
        job = JobSpec(name="test_job")
        assert job.name == "test_job"
        assert job.working_dir == "."
        assert job.dependencies == []
    
    def test_full_specification(self):
        """Test complete job specification."""
        resources = ResourceRequest(num_nodes=2, num_cores_per_node=32)
        job = JobSpec(
            name="dft_calculation",
            executable="python",
            arguments=["run.py", "--config", "config.yaml"],
            resources=resources,
            working_dir="./calc",
            environment_vars={"OMP_NUM_THREADS": "4"},
            dependencies=["job1", "job2"]
        )
        assert job.name == "dft_calculation"
        assert job.executable == "python"
        assert len(job.arguments) == 4
        assert job.resources.num_nodes == 2
