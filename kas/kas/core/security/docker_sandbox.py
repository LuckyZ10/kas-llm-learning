"""
Docker Sandbox - Container isolation for secure code execution
"""
import asyncio
import io
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DockerNotAvailableError(Exception):
    """Raised when Docker is not installed or not running"""
    pass


class SandboxCreationError(Exception):
    """Raised when sandbox creation fails"""
    pass


class SandboxExecutionError(Exception):
    """Raised when command execution fails"""
    pass


class SandboxTimeoutError(Exception):
    """Raised when execution times out"""
    pass


@dataclass
class DockerSandboxConfig:
    """Docker sandbox configuration"""
    image: str = "python:3.11-slim"
    container_name: str = ""
    work_dir: str = "/workspace"
    volumes: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    network_mode: str = "bridge"
    cpu_limit: float = 1.0
    memory_limit: str = "512m"
    timeout: int = 60
    auto_remove: bool = True

    def __post_init__(self):
        if not self.container_name:
            import uuid
            self.container_name = f"kas_sandbox_{uuid.uuid4().hex[:8]}"


@dataclass
class ExecutionResult:
    """Result of command execution in sandbox"""
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    timed_out: bool = False


class DockerSandbox:
    """
    Docker sandbox for isolated code execution
    
    Manages Docker container lifecycle and provides
    secure command execution environment
    """
    
    def __init__(self, config: DockerSandboxConfig):
        self.config = config
        self._client = None
        self._container = None
        self._created = False
        self._started = False
    
    @property
    def client(self):
        """Lazy load Docker client"""
        if self._client is None:
            try:
                import docker
                self._client = docker.from_env()
            except ImportError:
                raise DockerNotAvailableError(
                    "Docker package not installed. Run: pip install docker>=6.0.0"
                )
            except Exception as e:
                raise DockerNotAvailableError(f"Docker not available: {e}")
        return self._client
    
    def create(self) -> bool:
        """Create Docker container"""
        if self._created:
            logger.warning(f"Container {self.config.container_name} already created")
            return True
        
        try:
            volumes = {}
            for host_path, container_path in self.config.volumes.items():
                volumes[host_path] = {
                    "bind": container_path,
                    "mode": "rw"
                }
            
            self._container = self.client.containers.create(
                image=self.config.image,
                name=self.config.container_name,
                working_dir=self.config.work_dir,
                environment=self.config.environment,
                volumes=volumes if volumes else None,
                network_mode=self.config.network_mode,
                cpu_quota=int(self.config.cpu_limit * 100000),
                mem_limit=self.config.memory_limit,
                auto_remove=self.config.auto_remove,
                detach=True,
                stdin_open=True,
                tty=True
            )
            
            self._created = True
            logger.info(f"Created container: {self.config.container_name}")
            return True
            
        except Exception as e:
            raise SandboxCreationError(f"Failed to create container: {e}")
    
    def start(self) -> bool:
        """Start the container"""
        if not self._created or self._container is None:
            raise SandboxCreationError("Container not created. Call create() first.")
        
        if self._started:
            logger.warning(f"Container {self.config.container_name} already started")
            return True
        
        try:
            self._container.start()
            self._started = True
            logger.info(f"Started container: {self.config.container_name}")
            return True
        except Exception as e:
            raise SandboxExecutionError(f"Failed to start container: {e}")
    
    def execute(self, command: List[str]) -> ExecutionResult:
        """
        Execute command in container (synchronous)
        
        Args:
            command: Command and arguments to execute
            
        Returns:
            ExecutionResult with output and status
        """
        if not self._started or self._container is None:
            raise SandboxExecutionError("Container not started. Call start() first.")
        
        start_time = time.time()
        
        try:
            exit_code, output = self._container.exec_run(
                cmd=command,
                workdir=self.config.work_dir,
                demux=True
            )
            
            stdout = output[0].decode("utf-8") if output[0] else ""
            stderr = output[1].decode("utf-8") if output[1] else ""
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                timed_out=False
            )
            
        except Exception as e:
            if "timeout" in str(e).lower():
                raise SandboxTimeoutError(
                    f"Command execution timed out after {self.config.timeout}s"
                )
            raise SandboxExecutionError(f"Command execution failed: {e}")
    
    async def execute_async(self, command: List[str]) -> ExecutionResult:
        """
        Execute command in container (asynchronous)
        
        Args:
            command: Command and arguments to execute
            
        Returns:
            ExecutionResult with output and status
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.execute, command),
                timeout=self.config.timeout
            )
            return result
        except asyncio.TimeoutError:
            raise SandboxTimeoutError(
                f"Command execution timed out after {self.config.timeout}s"
            )
    
    def stop(self) -> bool:
        """Stop the container"""
        if self._container is None:
            return True
        
        try:
            self._container.stop(timeout=10)
            self._started = False
            logger.info(f"Stopped container: {self.config.container_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False
    
    def remove(self) -> bool:
        """Remove the container"""
        if self._container is None:
            return True
        
        try:
            self._container.remove(force=True)
            self._created = False
            self._started = False
            self._container = None
            logger.info(f"Removed container: {self.config.container_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove container: {e}")
            return False
    
    def get_logs(self) -> str:
        """Get container logs"""
        if self._container is None:
            return ""
        
        try:
            logs = self._container.logs()
            return logs.decode("utf-8") if logs else ""
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return ""
    
    def get_status(self) -> str:
        """Get container status"""
        if self._container is None:
            return "not_created"
        
        try:
            self._container.reload()
            return self._container.status
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return "unknown"
    
    def copy_to_container(self, host_path: str, container_path: str) -> bool:
        """
        Copy file from host to container
        
        Args:
            host_path: Path on host machine
            container_path: Destination path in container
            
        Returns:
            True if successful
        """
        if self._container is None:
            raise SandboxExecutionError("Container not created")
        
        try:
            host_file = Path(host_path)
            if not host_file.exists():
                raise FileNotFoundError(f"Host file not found: {host_path}")
            
            with open(host_file, "rb") as f:
                data = f.read()
            
            import io
            self._container.put_archive(
                container_path,
                self._create_tar_archive(host_file.name, data)
            )
            
            logger.debug(f"Copied {host_path} to {container_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy file to container: {e}")
            return False
    
    def copy_from_container(self, container_path: str, host_path: str) -> bool:
        """
        Copy file from container to host
        
        Args:
            container_path: Path in container
            host_path: Destination path on host
            
        Returns:
            True if successful
        """
        if self._container is None:
            raise SandboxExecutionError("Container not created")
        
        try:
            import tarfile
            import io
            
            bits, stat = self._container.get_archive(container_path)
            
            tar_stream = io.BytesIO()
            for chunk in bits:
                tar_stream.write(chunk)
            tar_stream.seek(0)
            
            host_file = Path(host_path)
            host_file.parent.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(fileobj=tar_stream) as tar:
                member = tar.getmembers()[0]
                with tar.extractfile(member) as f:
                    with open(host_file, "wb") as out:
                        out.write(f.read())
            
            logger.debug(f"Copied {container_path} to {host_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy file from container: {e}")
            return False
    
    def _create_tar_archive(self, name: str, data: bytes) -> io.BytesIO:
        """Create a tar archive with a single file"""
        import tarfile
        import io
        
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        
        tar_stream.seek(0)
        return tar_stream
    
    def __enter__(self):
        self.create()
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.remove()
        return False


class DockerSandboxManager:
    """
    Manager for multiple Docker sandbox instances
    
    Provides batch creation, destruction, and resource cleanup
    """
    
    def __init__(self):
        self._sandboxes: Dict[str, DockerSandbox] = {}
    
    def create_sandbox(
        self,
        name: str,
        config: Optional[DockerSandboxConfig] = None
    ) -> DockerSandbox:
        """
        Create and register a new sandbox
        
        Args:
            name: Unique sandbox name
            config: Sandbox configuration (uses defaults if None)
            
        Returns:
            Created DockerSandbox instance
        """
        if name in self._sandboxes:
            raise SandboxCreationError(f"Sandbox '{name}' already exists")
        
        if config is None:
            config = DockerSandboxConfig()
        else:
            config.container_name = name
        
        sandbox = DockerSandbox(config)
        self._sandboxes[name] = sandbox
        
        return sandbox
    
    def get_sandbox(self, name: str) -> Optional[DockerSandbox]:
        """Get sandbox by name"""
        return self._sandboxes.get(name)
    
    def remove_sandbox(self, name: str) -> bool:
        """Stop and remove a sandbox"""
        sandbox = self._sandboxes.get(name)
        if sandbox is None:
            return False
        
        sandbox.stop()
        sandbox.remove()
        del self._sandboxes[name]
        return True
    
    def list_sandboxes(self) -> List[str]:
        """List all registered sandbox names"""
        return list(self._sandboxes.keys())
    
    def create_all(self) -> Dict[str, bool]:
        """Create all registered sandboxes"""
        results = {}
        for name, sandbox in self._sandboxes.items():
            try:
                sandbox.create()
                sandbox.start()
                results[name] = True
            except Exception as e:
                logger.error(f"Failed to create sandbox {name}: {e}")
                results[name] = False
        return results
    
    def destroy_all(self) -> Dict[str, bool]:
        """Stop and remove all sandboxes"""
        results = {}
        for name, sandbox in self._sandboxes.items():
            try:
                sandbox.stop()
                sandbox.remove()
                results[name] = True
            except Exception as e:
                logger.error(f"Failed to destroy sandbox {name}: {e}")
                results[name] = False
        
        self._sandboxes.clear()
        return results
    
    def cleanup(self) -> int:
        """
        Clean up orphaned containers
        
        Returns:
            Number of containers removed
        """
        try:
            import docker
            client = docker.from_env()
            
            removed = 0
            for container in client.containers.list(all=True):
                if container.name.startswith("kas_sandbox_"):
                    try:
                        container.remove(force=True)
                        removed += 1
                        logger.info(f"Cleaned up orphaned container: {container.name}")
                    except Exception as e:
                        logger.error(f"Failed to remove {container.name}: {e}")
            
            return removed
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sandboxes"""
        status = {}
        for name, sandbox in self._sandboxes.items():
            status[name] = {
                "status": sandbox.get_status(),
                "created": sandbox._created,
                "started": sandbox._started,
                "config": {
                    "image": sandbox.config.image,
                    "memory_limit": sandbox.config.memory_limit,
                    "cpu_limit": sandbox.config.cpu_limit
                }
            }
        return status
    
    def __len__(self) -> int:
        return len(self._sandboxes)
    
    def __contains__(self, name: str) -> bool:
        return name in self._sandboxes
