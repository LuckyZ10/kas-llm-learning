"""
DFTLammps Global Compute Module
全球分布式计算模块

This module provides global-scale deployment and edge computing capabilities for the DFTLammps platform.

Modules:
    multi_cloud: Multi-cloud orchestration (AWS/Azure/GCP)
    edge_deployment: Edge computing and low-latency inference
    serverless: Serverless computing with auto-scaling

Author: DFTLammps Platform Team
Version: 2.0.0
"""

from .multi_cloud import (
    MultiCloudOrchestrator,
    CloudProvider,
    CloudRegion,
    InstanceSpec,
    ComputeTask,
    CostOptimizer,
    AWSProvider,
    AzureProvider,
    GCPProvider
)

from .edge_deployment import (
    EdgeDeploymentOrchestrator,
    EdgeNode,
    EdgeNodeManager,
    LatencyAwareScheduler,
    DataLocalityManager,
    EdgeInferenceEngine,
    GeoLocation,
    HardwareCapabilities
)

from .serverless import (
    ServerlessOrchestrator,
    FunctionSpec,
    FunctionRuntime,
    ExecutionRequest,
    BillingCalculator,
    AutoScaler
)

__version__ = "2.0.0"
__all__ = [
    # Multi-cloud
    "MultiCloudOrchestrator",
    "CloudProvider",
    "CloudRegion",
    "InstanceSpec",
    "ComputeTask",
    "CostOptimizer",
    "AWSProvider",
    "AzureProvider",
    "GCPProvider",
    
    # Edge deployment
    "EdgeDeploymentOrchestrator",
    "EdgeNode",
    "EdgeNodeManager",
    "LatencyAwareScheduler",
    "DataLocalityManager",
    "EdgeInferenceEngine",
    "GeoLocation",
    "HardwareCapabilities",
    
    # Serverless
    "ServerlessOrchestrator",
    "FunctionSpec",
    "FunctionRuntime",
    "ExecutionRequest",
    "BillingCalculator",
    "AutoScaler"
]
