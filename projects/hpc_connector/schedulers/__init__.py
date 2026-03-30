"""
Scheduler factory and registry.
"""

from typing import Dict, Type

from ..core.base import BaseJobScheduler
from ..core.cluster import ClusterType
from ..connectors.ssh_connector import SSHConnector


class SchedulerFactory:
    """Factory for creating job schedulers."""
    
    _registry: Dict[ClusterType, Type[BaseJobScheduler]] = {}
    
    @classmethod
    def register(
        cls,
        cluster_type: ClusterType,
        scheduler_class: Type[BaseJobScheduler]
    ) -> None:
        """Register a scheduler class for a cluster type."""
        cls._registry[cluster_type] = scheduler_class
    
    @classmethod
    def create(cls, cluster_type: ClusterType, connector: SSHConnector) -> BaseJobScheduler:
        """
        Create a scheduler for the given cluster type.
        
        Args:
            cluster_type: Type of cluster
            connector: SSH connector instance
            
        Returns:
            Job scheduler instance
        """
        scheduler_class = cls._registry.get(cluster_type)
        if scheduler_class is None:
            raise ValueError(f"No scheduler registered for cluster type: {cluster_type}")
        
        return scheduler_class(connector)
    
    @classmethod
    def supported_types(cls) -> list:
        """Get list of supported cluster types."""
        return list(cls._registry.keys())


# Register default schedulers
def _register_defaults():
    from .slurm_scheduler import SlurmScheduler
    from .pbs_scheduler import PBSScheduler
    from .lsf_scheduler import LSFScheduler
    from .sge_scheduler import SGEScheduler
    
    SchedulerFactory.register(ClusterType.SLURM, SlurmScheduler)
    SchedulerFactory.register(ClusterType.PBS, PBSScheduler)
    SchedulerFactory.register(ClusterType.TORQUE, PBSScheduler)
    SchedulerFactory.register(ClusterType.LSF, LSFScheduler)
    SchedulerFactory.register(ClusterType.SGE, SGEScheduler)


_register_defaults()
