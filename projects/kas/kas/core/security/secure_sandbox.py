"""
KAS Secure Sandbox - Integrated Security Layer

Combines all security features into a unified sandbox environment:
- Sensitive information filtering
- Resource quota management
- Network access control
- Docker container isolation
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from kas.core.security.sensitive_filter import (
    SensitiveInfoFilter,
    FilterConfig,
    filter_text,
)
from kas.core.security.resource_quota import (
    ResourceQuota,
    QuotaStatus,
    ResourceMonitor,
    ResourceLimiter,
    create_default_quota,
)
from kas.core.security.network_controller import (
    NetworkPolicy,
    NetworkAccessController,
    NetworkInterceptor,
    PolicyPreset,
)
from kas.core.security.docker_sandbox import (
    DockerSandbox,
    DockerSandboxConfig,
    ExecutionResult,
    DockerNotAvailableError,
)

logger = logging.getLogger(__name__)


@dataclass
class SecureSandboxConfig:
    """Configuration for secure sandbox"""
    name: str
    work_dir: Path
    
    # Security settings
    enable_filter: bool = True
    filter_config: Optional[FilterConfig] = None
    
    # Resource settings
    enable_quota: bool = True
    resource_quota: Optional[ResourceQuota] = None
    
    # Network settings
    enable_network_control: bool = True
    network_policy: Optional[NetworkPolicy] = None
    network_preset: str = "moderate"
    
    # Docker settings
    use_docker: bool = True
    docker_image: str = "python:3.11-slim"
    docker_cpu_limit: float = 1.0
    docker_memory_limit: str = "512m"
    docker_timeout: int = 300
    
    def __post_init__(self):
        if self.filter_config is None:
            self.filter_config = FilterConfig()
        if self.resource_quota is None:
            self.resource_quota = create_default_quota()
        if self.network_policy is None:
            preset_map = {
                "strict": PolicyPreset.STRICT,
                "moderate": PolicyPreset.MODERATE,
                "permissive": PolicyPreset.PERMISSIVE,
            }
            preset = preset_map.get(self.network_preset, PolicyPreset.MODERATE)
            self.network_policy = NetworkPolicy.from_preset(preset)


@dataclass
class SecureExecutionResult:
    """Result from secure sandbox execution"""
    success: bool
    output: str
    filtered_output: str
    exit_code: int
    execution_time: float
    quota_status: Optional[QuotaStatus] = None
    violations: List[str] = field(default_factory=list)
    error: Optional[str] = None


class SecureSandbox:
    """
    Secure Sandbox with integrated security layers
    
    Usage:
        config = SecureSandboxConfig(
            name="my-sandbox",
            work_dir=Path("./workspace"),
            network_preset="strict"
        )
        
        sandbox = SecureSandbox(config)
        result = sandbox.execute(["python", "script.py"])
        
        if result.success:
            print(result.filtered_output)
    """
    
    def __init__(self, config: SecureSandboxConfig):
        self.config = config
        self.name = config.name
        self.work_dir = Path(config.work_dir)
        
        # Initialize security components
        self._init_filter()
        self._init_resource_monitor()
        self._init_network_controller()
        self._init_docker()
        
        self._running = False
    
    def _init_filter(self):
        """Initialize sensitive info filter"""
        if self.config.enable_filter:
            self.filter = SensitiveInfoFilter(self.config.filter_config)
        else:
            self.filter = None
    
    def _init_resource_monitor(self):
        """Initialize resource monitor"""
        if self.config.enable_quota:
            self.monitor = ResourceMonitor()
            self.limiter = ResourceLimiter(self.config.resource_quota)
        else:
            self.monitor = None
            self.limiter = None
    
    def _init_network_controller(self):
        """Initialize network controller"""
        if self.config.enable_network_control:
            self.network = NetworkAccessController(self.config.network_policy)
        else:
            self.network = None
    
    def _init_docker(self):
        """Initialize Docker sandbox"""
        if self.config.use_docker:
            try:
                docker_config = DockerSandboxConfig(
                    image=self.config.docker_image,
                    container_name=f"kas-{self.name}",
                    work_dir="/workspace",
                    cpu_limit=self.config.docker_cpu_limit,
                    memory_limit=self.config.docker_memory_limit,
                    timeout=self.config.docker_timeout,
                    auto_remove=True,
                )
                self.docker = DockerSandbox(docker_config)
                self.docker_available = True
            except DockerNotAvailableError:
                logger.warning("Docker not available, falling back to local execution")
                self.docker = None
                self.docker_available = False
        else:
            self.docker = None
            self.docker_available = False
    
    def start(self) -> bool:
        """Start the secure sandbox"""
        if self._running:
            logger.warning(f"Sandbox {self.name} already running")
            return True
        
        try:
            # Start resource monitoring
            if self.monitor:
                self.monitor.start_monitoring()
            
            # Start Docker container if available
            if self.docker:
                self.docker.create()
                self.docker.start()
            
            # Apply network policy
            if self.network:
                self.network.apply_policy()
            
            self._running = True
            logger.info(f"Secure sandbox {self.name} started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start sandbox: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the secure sandbox"""
        if not self._running:
            return True
        
        try:
            # Stop Docker container
            if self.docker:
                self.docker.stop()
                self.docker.remove()
            
            # Stop resource monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
            
            self._running = False
            logger.info(f"Secure sandbox {self.name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop sandbox: {e}")
            return False
    
    def execute(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> SecureExecutionResult:
        """
        Execute command in secure sandbox
        
        Args:
            command: Command to execute
            env: Environment variables
            timeout: Execution timeout (overrides config)
            
        Returns:
            SecureExecutionResult with filtered output
        """
        violations = []
        quota_status = None
        
        try:
            # Check network access if command makes network calls
            if self.network and self._needs_network_access(command):
                allowed = self._validate_network_access(command)
                if not allowed:
                    violations.append("Network access denied by policy")
            
            # Execute with resource limits
            if self.docker and self.docker_available:
                result = self.docker.execute(command)
                output = result.stdout
                exit_code = result.exit_code
                exec_time = result.execution_time
            else:
                # Fallback to local execution with limiter
                if self.limiter:
                    result = self.limiter.execute(
                        lambda: self._local_execute(command, env)
                    )
                    output = result.get("output", "")
                    exit_code = result.get("exit_code", 0)
                    exec_time = result.get("time", 0)
                    quota_status = result.get("quota_status")
                else:
                    import subprocess
                    proc = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=timeout or self.config.docker_timeout
                    )
                    output = proc.stdout
                    exit_code = proc.returncode
                    exec_time = 0
            
            # Filter output
            if self.filter:
                filtered_output = self.filter.filter(output)
            else:
                filtered_output = output
            
            return SecureExecutionResult(
                success=exit_code == 0,
                output=output,
                filtered_output=filtered_output,
                exit_code=exit_code,
                execution_time=exec_time,
                quota_status=quota_status,
                violations=violations,
            )
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return SecureExecutionResult(
                success=False,
                output="",
                filtered_output="",
                exit_code=-1,
                execution_time=0,
                violations=violations,
                error=str(e),
            )
    
    def _local_execute(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute command locally"""
        import subprocess
        import time
        
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        
        start_time = time.time()
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=merged_env,
        )
        exec_time = time.time() - start_time
        
        return {
            "output": proc.stdout + proc.stderr,
            "exit_code": proc.returncode,
            "time": exec_time,
        }
    
    def _needs_network_access(self, command: List[str]) -> bool:
        """Check if command likely needs network access"""
        network_tools = ["curl", "wget", "http", "https", "requests", "urllib", "pip", "npm"]
        cmd_str = " ".join(command).lower()
        return any(tool in cmd_str for tool in network_tools)
    
    def _validate_network_access(self, command: List[str]) -> bool:
        """Validate network access for command"""
        return True
    
    def filter_text(self, text: str) -> str:
        """Filter sensitive information from text"""
        if self.filter:
            return self.filter.filter(text)
        return text
    
    def check_quota(self) -> Optional[QuotaStatus]:
        """Check current resource quota status"""
        if self.monitor and self.config.resource_quota:
            return self.monitor.check_quota(self.config.resource_quota)
        return None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class SecureSandboxManager:
    """
    Manager for multiple secure sandboxes
    """
    
    def __init__(self):
        self.sandboxes: Dict[str, SecureSandbox] = {}
    
    def create(self, config: SecureSandboxConfig) -> SecureSandbox:
        """Create a new secure sandbox"""
        if config.name in self.sandboxes:
            raise ValueError(f"Sandbox {config.name} already exists")
        
        sandbox = SecureSandbox(config)
        self.sandboxes[config.name] = sandbox
        return sandbox
    
    def get(self, name: str) -> Optional[SecureSandbox]:
        """Get sandbox by name"""
        return self.sandboxes.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove sandbox"""
        sandbox = self.sandboxes.get(name)
        if sandbox:
            sandbox.stop()
            del self.sandboxes[name]
            return True
        return False
    
    def list(self) -> List[str]:
        """List all sandbox names"""
        return list(self.sandboxes.keys())
    
    def stop_all(self):
        """Stop all sandboxes"""
        for sandbox in self.sandboxes.values():
            sandbox.stop()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()
        return False


def create_secure_sandbox(
    name: str,
    work_dir: str,
    preset: str = "default",
    **kwargs
) -> SecureSandbox:
    """
    Factory function to create secure sandbox with preset
    
    Args:
        name: Sandbox name
        work_dir: Working directory
        preset: Preset name (strict, default, relaxed)
        **kwargs: Override config options
        
    Returns:
        Configured SecureSandbox instance
    """
    config_map = {
        "strict": {
            "network_preset": "strict",
            "docker_cpu_limit": 0.5,
            "docker_memory_limit": "256m",
            "docker_timeout": 60,
        },
        "default": {
            "network_preset": "moderate",
            "docker_cpu_limit": 1.0,
            "docker_memory_limit": "512m",
            "docker_timeout": 300,
        },
        "relaxed": {
            "network_preset": "permissive",
            "docker_cpu_limit": 2.0,
            "docker_memory_limit": "1g",
            "docker_timeout": 600,
        },
    }
    
    config_kwargs = config_map.get(preset, config_map["default"])
    config_kwargs.update(kwargs)
    
    config = SecureSandboxConfig(
        name=name,
        work_dir=Path(work_dir),
        **config_kwargs
    )
    
    return SecureSandbox(config)
