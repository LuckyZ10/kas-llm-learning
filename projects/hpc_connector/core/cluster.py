"""
Cluster configuration and management models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto


class ClusterType(Enum):
    """Supported cluster types."""
    SLURM = "slurm"
    PBS = "pbs"
    TORQUE = "torque"
    LSF = "lsf"
    SGE = "sge"
    AWS_PARALLELCLUSTER = "aws_parallelcluster"
    ALIYUN_BATCH = "aliyun_batch"
    TENCENT_BATCH = "tencent_batch"


class AuthMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    KEY = "key"
    KEY_WITH_PASSPHRASE = "key_with_passphrase"
    KERBEROS = "kerberos"
    TOKEN = "token"
    MFA = "mfa"


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    port: int = 22
    user: str
    
    # Authentication
    auth_method: AuthMethod = AuthMethod.KEY
    password: Optional[str] = None
    key_file: Optional[str] = None
    key_passphrase: Optional[str] = None
    
    # Connection options
    timeout: int = 30
    keepalive_interval: int = 60
    compress: bool = True
    
    # Jump host / proxy
    proxy_host: Optional[str] = None
    proxy_port: int = 22
    proxy_user: Optional[str] = None
    
    # Advanced options
    strict_host_key_checking: bool = True
    known_hosts_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "auth_method": self.auth_method.value,
            "key_file": self.key_file,
            "timeout": self.timeout,
            "keepalive_interval": self.keepalive_interval,
            "compress": self.compress,
            "proxy_host": self.proxy_host,
            "proxy_port": self.proxy_port,
            "proxy_user": self.proxy_user,
            "strict_host_key_checking": self.strict_host_key_checking,
            "known_hosts_file": self.known_hosts_file,
        }


@dataclass
class ClusterConfig:
    """Cluster configuration."""
    name: str
    cluster_type: ClusterType
    ssh: SSHConfig
    
    # Environment
    work_dir: str = "~"
    module_system: Optional[str] = None  # lmod, modules, etc.
    environment_setup: List[str] = field(default_factory=list)
    
    # Resource defaults
    default_queue: Optional[str] = None
    default_partition: Optional[str] = None
    max_nodes: int = 100
    max_walltime: str = "24:00:00"
    
    # Data transfer
    data_staging_enabled: bool = True
    staging_dir: Optional[str] = None
    
    # Monitoring
    monitoring_interval: int = 30  # seconds
    
    # Features and capabilities
    features: List[str] = field(default_factory=list)
    
    # Custom settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cluster_type": self.cluster_type.value,
            "ssh": self.ssh.to_dict(),
            "work_dir": self.work_dir,
            "module_system": self.module_system,
            "environment_setup": self.environment_setup,
            "default_queue": self.default_queue,
            "default_partition": self.default_partition,
            "max_nodes": self.max_nodes,
            "max_walltime": self.max_walltime,
            "data_staging_enabled": self.data_staging_enabled,
            "staging_dir": self.staging_dir,
            "monitoring_interval": self.monitoring_interval,
            "features": self.features,
            "settings": self.settings,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterConfig":
        ssh_data = data.get("ssh", {})
        ssh_config = SSHConfig(
            host=ssh_data["host"],
            port=ssh_data.get("port", 22),
            user=ssh_data["user"],
            auth_method=AuthMethod(ssh_data.get("auth_method", "key")),
            key_file=ssh_data.get("key_file"),
            timeout=ssh_data.get("timeout", 30),
            keepalive_interval=ssh_data.get("keepalive_interval", 60),
            compress=ssh_data.get("compress", True),
            proxy_host=ssh_data.get("proxy_host"),
            proxy_port=ssh_data.get("proxy_port", 22),
            proxy_user=ssh_data.get("proxy_user"),
            strict_host_key_checking=ssh_data.get("strict_host_key_checking", True),
            known_hosts_file=ssh_data.get("known_hosts_file"),
        )
        
        return cls(
            name=data["name"],
            cluster_type=ClusterType(data["cluster_type"]),
            ssh=ssh_config,
            work_dir=data.get("work_dir", "~"),
            module_system=data.get("module_system"),
            environment_setup=data.get("environment_setup", []),
            default_queue=data.get("default_queue"),
            default_partition=data.get("default_partition"),
            max_nodes=data.get("max_nodes", 100),
            max_walltime=data.get("max_walltime", "24:00:00"),
            data_staging_enabled=data.get("data_staging_enabled", True),
            staging_dir=data.get("staging_dir"),
            monitoring_interval=data.get("monitoring_interval", 30),
            features=data.get("features", []),
            settings=data.get("settings", {}),
        )


@dataclass
class QueueInfo:
    """Queue/Partition information."""
    name: str
    state: str  # open, closed, draining, etc.
    
    # Resource limits
    min_nodes: int = 1
    max_nodes: int = 1
    max_walltime: str = "24:00:00"
    max_jobs_per_user: Optional[int] = None
    
    # Current status
    total_nodes: int = 0
    free_nodes: int = 0
    allocated_nodes: int = 0
    jobs_running: int = 0
    jobs_queued: int = 0
    
    # Features
    features: List[str] = field(default_factory=list)
    
    # Pricing (for cloud clusters)
    price_per_hour: Optional[float] = None
    currency: str = "USD"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "max_walltime": self.max_walltime,
            "max_jobs_per_user": self.max_jobs_per_user,
            "total_nodes": self.total_nodes,
            "free_nodes": self.free_nodes,
            "allocated_nodes": self.allocated_nodes,
            "jobs_running": self.jobs_running,
            "jobs_queued": self.jobs_queued,
            "features": self.features,
            "price_per_hour": self.price_per_hour,
            "currency": self.currency,
        }


@dataclass
class NodeInfo:
    """Compute node information."""
    name: str
    state: str  # idle, allocated, down, draining, etc.
    
    # Resources
    total_cores: int = 0
    free_cores: int = 0
    total_memory: str = "0GB"
    free_memory: str = "0GB"
    total_gpus: int = 0
    free_gpus: int = 0
    
    # Features
    features: List[str] = field(default_factory=list)
    
    # Load
    load_average: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "total_cores": self.total_cores,
            "free_cores": self.free_cores,
            "total_memory": self.total_memory,
            "free_memory": self.free_memory,
            "total_gpus": self.total_gpus,
            "free_gpus": self.free_gpus,
            "features": self.features,
            "load_average": self.load_average,
        }


class ClusterManager:
    """Manages cluster configurations."""
    
    def __init__(self):
        self._clusters: Dict[str, ClusterConfig] = {}
    
    def register(self, config: ClusterConfig) -> None:
        """Register a cluster configuration."""
        self._clusters[config.name] = config
    
    def unregister(self, name: str) -> None:
        """Unregister a cluster."""
        if name in self._clusters:
            del self._clusters[name]
    
    def get(self, name: str) -> Optional[ClusterConfig]:
        """Get cluster configuration by name."""
        return self._clusters.get(name)
    
    def list_clusters(self) -> List[str]:
        """List all registered cluster names."""
        return list(self._clusters.keys())
    
    def load_from_file(self, path: str) -> None:
        """Load cluster configurations from file."""
        import json
        import yaml
        
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        if isinstance(data, list):
            for cluster_data in data:
                config = ClusterConfig.from_dict(cluster_data)
                self.register(config)
        elif isinstance(data, dict) and 'clusters' in data:
            for cluster_data in data['clusters']:
                config = ClusterConfig.from_dict(cluster_data)
                self.register(config)
        else:
            config = ClusterConfig.from_dict(data)
            self.register(config)
    
    def save_to_file(self, path: str) -> None:
        """Save cluster configurations to file."""
        import json
        
        data = {
            "clusters": [config.to_dict() for config in self._clusters.values()]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
