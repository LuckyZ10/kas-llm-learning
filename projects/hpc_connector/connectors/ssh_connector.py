"""
SSH-based HPC connector implementation.
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import fnmatch

from ..core.base import BaseHPCConnector
from ..core.cluster import ClusterConfig, AuthMethod
from ..core.exceptions import ConnectionError, AuthenticationError, DataTransferError

logger = logging.getLogger(__name__)

# Try to import asyncssh
try:
    import asyncssh
    ASYNCSSH_AVAILABLE = True
except ImportError:
    ASYNCSSH_AVAILABLE = False
    logger.warning("asyncssh not available, SSH functionality will be limited")


class SSHConnector(BaseHPCConnector):
    """
    SSH-based connector for HPC clusters.
    
    Uses asyncssh for asynchronous SSH operations.
    """
    
    def __init__(self, config: ClusterConfig):
        super().__init__(config)
        self._ssh_conn = None
        self._sftp = None
    
    async def connect(self) -> None:
        """Establish SSH connection to the cluster."""
        if not ASYNCSSH_AVAILABLE:
            raise ImportError("asyncssh is required for SSH connections")
        
        async with self._lock:
            if self._connected:
                return
            
            try:
                ssh_config = self.config.ssh
                
                # Prepare connection options
                conn_options = {
                    'host': ssh_config.host,
                    'port': ssh_config.port,
                    'username': ssh_config.user,
                    'connect_timeout': ssh_config.timeout,
                    'keepalive_interval': ssh_config.keepalive_interval,
                    'compress': ssh_config.compress,
                }
                
                # Configure authentication
                if ssh_config.auth_method == AuthMethod.PASSWORD:
                    conn_options['password'] = ssh_config.password
                elif ssh_config.auth_method in [AuthMethod.KEY, AuthMethod.KEY_WITH_PASSPHRASE]:
                    if ssh_config.key_file:
                        key_path = os.path.expanduser(ssh_config.key_file)
                        if ssh_config.auth_method == AuthMethod.KEY_WITH_PASSPHRASE:
                            conn_options['client_keys'] = [asyncssh.read_private_key(
                                key_path, passphrase=ssh_config.key_passphrase
                            )]
                        else:
                            conn_options['client_keys'] = [key_path]
                
                # Configure proxy/jump host
                if ssh_config.proxy_host:
                    proxy_tunnel = await asyncssh.connect(
                        ssh_config.proxy_host,
                        port=ssh_config.proxy_port,
                        username=ssh_config.proxy_user or ssh_config.user,
                    )
                    conn_options['tunnel'] = proxy_tunnel
                
                # Configure host key checking
                if not ssh_config.strict_host_key_checking:
                    conn_options['known_hosts'] = None
                elif ssh_config.known_hosts_file:
                    conn_options['known_hosts'] = ssh_config.known_hosts_file
                
                # Establish connection
                self._ssh_conn = await asyncssh.connect(**conn_options)
                self._sftp = await self._ssh_conn.start_sftp_client()
                
                self._connected = True
                logger.info(f"Connected to {ssh_config.host}:{ssh_config.port}")
                
            except asyncssh.PermissionDenied as e:
                raise AuthenticationError(
                    f"Authentication failed for {ssh_config.user}@{ssh_config.host}",
                    {"error": str(e)}
                )
            except Exception as e:
                raise ConnectionError(
                    f"Failed to connect to {ssh_config.host}:{ssh_config.port}",
                    {"error": str(e)}
                )
    
    async def disconnect(self) -> None:
        """Close SSH connection."""
        async with self._lock:
            if self._sftp:
                self._sftp.exit()
                self._sftp = None
            
            if self._ssh_conn:
                self._ssh_conn.close()
                await self._ssh_conn.wait_closed()
                self._ssh_conn = None
            
            self._connected = False
            logger.info("Disconnected from cluster")
    
    async def execute(
        self, 
        command: str, 
        timeout: Optional[int] = None,
        work_dir: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute a command on the cluster.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            work_dir: Working directory for command
            environment: Environment variables to set
            
        Returns:
            Dictionary with 'stdout', 'stderr', and 'exit_code'
        """
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        # Build environment setup
        env_setup = []
        if self.config.environment_setup:
            env_setup.extend(self.config.environment_setup)
        if environment:
            for key, value in environment.items():
                env_setup.append(f'export {key}="{value}"')
        
        # Build full command
        full_command = "; ".join(env_setup) if env_setup else ""
        if work_dir:
            full_command += f"; cd {work_dir}" if full_command else f"cd {work_dir}"
        if full_command:
            full_command += f"; {command}"
        else:
            full_command = command
        
        try:
            result = await self._ssh_conn.run(
                full_command,
                timeout=timeout
            )
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.exit_status
            }
        except asyncio.TimeoutError:
            raise TimeoutError(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise ConnectionError(f"Command execution failed: {e}")
    
    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Upload a file to the cluster."""
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        try:
            local_path = os.path.expanduser(local_path)
            local_size = os.path.getsize(local_path)
            
            if progress_callback:
                # Use chunked transfer with progress
                await self._upload_with_progress(
                    local_path, remote_path, local_size, progress_callback
                )
            else:
                await self._sftp.put(local_path, remote_path)
            
            logger.info(f"Uploaded {local_path} to {remote_path}")
        except Exception as e:
            raise DataTransferError(
                f"Failed to upload {local_path} to {remote_path}",
                {"error": str(e)}
            )
    
    async def _upload_with_progress(
        self,
        local_path: str,
        remote_path: str,
        total_size: int,
        progress_callback: Callable[[int, int], None]
    ) -> None:
        """Upload file with progress tracking."""
        bytes_transferred = 0
        chunk_size = 65536  # 64KB chunks
        
        async with self._sftp.open(remote_path, 'wb') as remote_file:
            with open(local_path, 'rb') as local_file:
                while True:
                    chunk = local_file.read(chunk_size)
                    if not chunk:
                        break
                    await remote_file.write(chunk)
                    bytes_transferred += len(chunk)
                    progress_callback(bytes_transferred, total_size)
    
    async def download_file(
        self,
        remote_path: str,
        local_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Download a file from the cluster."""
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        try:
            local_path = os.path.expanduser(local_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Get remote file size
            try:
                stat = await self._sftp.stat(remote_path)
                remote_size = stat.size
            except:
                remote_size = 0
            
            if progress_callback and remote_size > 0:
                await self._download_with_progress(
                    remote_path, local_path, remote_size, progress_callback
                )
            else:
                await self._sftp.get(remote_path, local_path)
            
            logger.info(f"Downloaded {remote_path} to {local_path}")
        except Exception as e:
            raise DataTransferError(
                f"Failed to download {remote_path} to {local_path}",
                {"error": str(e)}
            )
    
    async def _download_with_progress(
        self,
        remote_path: str,
        local_path: str,
        total_size: int,
        progress_callback: Callable[[int, int], None]
    ) -> None:
        """Download file with progress tracking."""
        bytes_transferred = 0
        chunk_size = 65536  # 64KB chunks
        
        async with self._sftp.open(remote_path, 'rb') as remote_file:
            with open(local_path, 'wb') as local_file:
                while True:
                    chunk = await remote_file.read(chunk_size)
                    if not chunk:
                        break
                    local_file.write(chunk)
                    bytes_transferred += len(chunk)
                    progress_callback(bytes_transferred, total_size)
    
    async def upload_directory(
        self,
        local_path: str,
        remote_path: str,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Upload a directory to the cluster."""
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        exclude_patterns = exclude_patterns or []
        local_path = os.path.expanduser(local_path)
        
        # Calculate total size first
        total_size = 0
        files_to_upload = []
        
        for root, dirs, files in os.walk(local_path):
            # Filter out excluded directories
            dirs[:] = [
                d for d in dirs 
                if not any(fnmatch.fnmatch(d, p) for p in exclude_patterns)
            ]
            
            for file in files:
                if any(fnmatch.fnmatch(file, p) for p in exclude_patterns):
                    continue
                
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, local_path)
                remote_file = os.path.join(remote_path, rel_path).replace('\\', '/')
                
                size = os.path.getsize(local_file)
                total_size += size
                files_to_upload.append((local_file, remote_file, size))
        
        # Upload files
        bytes_transferred = 0
        for local_file, remote_file, size in files_to_upload:
            remote_dir = os.path.dirname(remote_file)
            await self._sftp.makedirs(remote_dir, exist_ok=True)
            
            if progress_callback:
                await self._upload_with_progress(
                    local_file, remote_file, size,
                    lambda b, t: progress_callback(bytes_transferred + b, total_size)
                )
                bytes_transferred += size
            else:
                await self._sftp.put(local_file, remote_file)
        
        logger.info(f"Uploaded directory {local_path} to {remote_path}")
    
    async def download_directory(
        self,
        remote_path: str,
        local_path: str,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Download a directory from the cluster."""
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        exclude_patterns = exclude_patterns or []
        local_path = os.path.expanduser(local_path)
        os.makedirs(local_path, exist_ok=True)
        
        # Get file list and sizes
        files_to_download = []
        total_size = 0
        
        async for entry in self._sftp.scandir(remote_path):
            if entry.filename in ['.', '..']:
                continue
            if any(fnmatch.fnmatch(entry.filename, p) for p in exclude_patterns):
                continue
            
            remote_entry = f"{remote_path}/{entry.filename}"
            local_entry = os.path.join(local_path, entry.filename)
            
            if await self._sftp.isdir(remote_entry):
                # Recursively download subdirectory
                sub_files, sub_size = await self._get_directory_files(
                    remote_entry, local_entry, exclude_patterns
                )
                files_to_download.extend(sub_files)
                total_size += sub_size
            else:
                size = entry.attrs.size if entry.attrs else 0
                files_to_download.append((remote_entry, local_entry, size))
                total_size += size
        
        # Download files
        bytes_transferred = 0
        for remote_file, local_file, size in files_to_download:
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            
            if progress_callback:
                await self._download_with_progress(
                    remote_file, local_file, size,
                    lambda b, t: progress_callback(bytes_transferred + b, total_size)
                )
                bytes_transferred += size
            else:
                await self._sftp.get(remote_file, local_file)
        
        logger.info(f"Downloaded directory {remote_path} to {local_path}")
    
    async def _get_directory_files(
        self,
        remote_path: str,
        local_path: str,
        exclude_patterns: List[str]
    ) -> tuple:
        """Get list of files in a remote directory."""
        files = []
        total_size = 0
        
        async for entry in self._sftp.scandir(remote_path):
            if entry.filename in ['.', '..']:
                continue
            if any(fnmatch.fnmatch(entry.filename, p) for p in exclude_patterns):
                continue
            
            remote_entry = f"{remote_path}/{entry.filename}"
            local_entry = os.path.join(local_path, entry.filename)
            
            if await self._sftp.isdir(remote_entry):
                sub_files, sub_size = await self._get_directory_files(
                    remote_entry, local_entry, exclude_patterns
                )
                files.extend(sub_files)
                total_size += sub_size
            else:
                size = entry.attrs.size if entry.attrs else 0
                files.append((remote_entry, local_entry, size))
                total_size += size
        
        return files, total_size
    
    async def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the cluster."""
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        try:
            await self._sftp.stat(remote_path)
            return True
        except asyncssh.SFTPNoSuchFile:
            return False
    
    async def list_directory(self, remote_path: str) -> List[Dict[str, Any]]:
        """List directory contents."""
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        entries = []
        async for entry in self._sftp.scandir(remote_path):
            if entry.filename in ['.', '..']:
                continue
            
            info = {
                'name': entry.filename,
                'path': f"{remote_path}/{entry.filename}",
            }
            
            if entry.attrs:
                info.update({
                    'size': entry.attrs.size,
                    'uid': entry.attrs.uid,
                    'gid': entry.attrs.gid,
                    'mode': oct(entry.attrs.permissions)[-3:] if entry.attrs.permissions else None,
                    'atime': entry.attrs.atime,
                    'mtime': entry.attrs.mtime,
                })
            
            entries.append(info)
        
        return entries
