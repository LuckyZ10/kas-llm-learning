"""
Connector factory and registry.
"""

from typing import Dict, Type

from ..core.base import BaseHPCConnector
from ..core.cluster import ClusterConfig, ClusterType
from .ssh_connector import SSHConnector


class ConnectorFactory:
    """Factory for creating HPC connectors."""
    
    _registry: Dict[ClusterType, Type[BaseHPCConnector]] = {
        ClusterType.SLURM: SSHConnector,
        ClusterType.PBS: SSHConnector,
        ClusterType.TORQUE: SSHConnector,
        ClusterType.LSF: SSHConnector,
        ClusterType.SGE: SSHConnector,
    }
    
    @classmethod
    def register(
        cls,
        cluster_type: ClusterType,
        connector_class: Type[BaseHPCConnector]
    ) -> None:
        """Register a connector class for a cluster type."""
        cls._registry[cluster_type] = connector_class
    
    @classmethod
    def create(cls, config: ClusterConfig) -> BaseHPCConnector:
        """
        Create a connector for the given cluster configuration.
        
        Args:
            config: Cluster configuration
            
        Returns:
            HPC connector instance
        """
        connector_class = cls._registry.get(config.cluster_type)
        if connector_class is None:
            raise ValueError(f"No connector registered for cluster type: {config.cluster_type}")
        
        return connector_class(config)
    
    @classmethod
    def supported_types(cls) -> list:
        """Get list of supported cluster types."""
        return list(cls._registry.keys())
