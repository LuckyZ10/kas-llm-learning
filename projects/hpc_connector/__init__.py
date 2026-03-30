"""
HPC Connector - A production-grade HPC cluster connection and job scheduling system.

This package provides unified interfaces for connecting to various HPC clusters
and managing computational jobs across different scheduling systems.
"""

__version__ = "1.0.0"
__author__ = "DFT Platform Team"

from .core.base import BaseHPCConnector, BaseJobScheduler
from .core.cluster import ClusterConfig, ClusterManager
from .core.job import JobConfig, JobStatus, JobResult
from .core.exceptions import (
    HPCConnectorError,
    AuthenticationError,
    ConnectionError,
    JobSubmissionError,
    JobMonitorError,
    DataTransferError,
    ResourceError,
)

__all__ = [
    "BaseHPCConnector",
    "BaseJobScheduler",
    "ClusterConfig",
    "ClusterManager",
    "JobConfig",
    "JobStatus",
    "JobResult",
    "HPCConnectorError",
    "AuthenticationError",
    "ConnectionError",
    "JobSubmissionError",
    "JobMonitorError",
    "DataTransferError",
    "ResourceError",
]
