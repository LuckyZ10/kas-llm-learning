#!/usr/bin/env python3
"""
cluster_connector.py
====================
集群连接管理模块

提供多种集群连接方式：
- SSH直连
- Kubernetes编排
- Slurm REST API
- 本地模拟

支持连接池管理和自动重连。
"""

import os
import re
import sys
import json
import time
import socket
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta
import subprocess
import hashlib

# 可选依赖
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """集群配置"""
    # 基本信息
    name: str
    host: str
    scheduler_type: str = "slurm"  # slurm, pbs, lsf
    
    # SSH配置
    port: int = 22
    username: Optional[str] = None
    key_file: Optional[str] = None
    password: Optional[str] = None
    
    # 路径配置
    remote_work_dir: str = "~/dft_calc"
    local_staging_dir: str = "./staging"
    
    # 计算资源配置
    default_queue: Optional[str] = None
    max_nodes: int = 100
    max_cores_per_node: int = 128
    max_gpus_per_node: int = 8
    
    # 限制
    max_walltime_hours: float = 168.0  # 7天
    max_jobs_per_user: int = 100
    
    # 连接池
    connection_pool_size: int = 5
    connection_timeout: int = 30
    
    # 认证
    api_token: Optional[str] = None
    api_endpoint: Optional[str] = None
    
    # 容器
    container_runtime: Optional[str] = None  # singularity, docker
    default_container_image: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ClusterConfig":
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: str) -> "ClusterConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class ConnectionPool:
    """SSH连接池"""
    
    def __init__(self, config: ClusterConfig, max_size: int = 5):
        self.config = config
        self.max_size = max_size
        self._pool: List[Any] = []
        self._lock = threading.Lock()
        self._created = 0
    
    def get_connection(self, timeout: float = 30.0) -> Any:
        """获取连接"""
        with self._lock:
            # 尝试获取现有连接
            while self._pool:
                conn = self._pool.pop()
                if self._is_alive(conn):
                    return conn
            
            # 创建新连接
            if self._created < self.max_size:
                self._created += 1
                return self._create_connection()
        
        # 等待可用连接
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                    if self._is_alive(conn):
                        return conn
            time.sleep(0.1)
        
        raise TimeoutError("Could not acquire connection from pool")
    
    def return_connection(self, conn: Any):
        """归还连接"""
        with self._lock:
            if self._is_alive(conn) and len(self._pool) < self.max_size:
                self._pool.append(conn)
            else:
                self._close_connection(conn)
                self._created -= 1
    
    def _create_connection(self) -> Any:
        """创建新连接"""
        if not PARAMIKO_AVAILABLE:
            raise ImportError("paramiko required for SSH connections")
        
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        connect_kwargs = {
            "hostname": self.config.host,
            "port": self.config.port,
            "username": self.config.username or os.getenv("USER"),
            "timeout": self.config.connection_timeout,
        }
        
        if self.config.key_file:
            connect_kwargs["key_filename"] = self.config.key_file
        elif self.config.password:
            connect_kwargs["password"] = self.config.password
        
        client.connect(**connect_kwargs)
        return client
    
    def _is_alive(self, conn: Any) -> bool:
        """检查连接是否存活"""
        try:
            transport = conn.get_transport()
            return transport is not None and transport.is_active()
        except Exception:
            return False
    
    def _close_connection(self, conn: Any):
        """关闭连接"""
        try:
            conn.close()
        except Exception:
            pass
    
    def close_all(self):
        """关闭所有连接"""
        with self._lock:
            for conn in self._pool:
                self._close_connection(conn)
            self._pool.clear()
            self._created = 0


class ClusterConnector(ABC):
    """集群连接器基类"""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self._connected = False
        self._connection_time: Optional[datetime] = None
    
    @abstractmethod
    def connect(self) -> bool:
        """建立连接"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """执行命令，返回 (returncode, stdout, stderr)"""
        pass
    
    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> bool:
        """上传文件"""
        pass
    
    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> bool:
        """下载文件"""
        pass
    
    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """检查远程路径是否存在"""
        pass
    
    @abstractmethod
    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """创建远程目录"""
        pass
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
    
    def get_connection_info(self) -> dict:
        """获取连接信息"""
        return {
            "connected": self._connected,
            "connection_time": self._connection_time.isoformat() if self._connection_time else None,
            "cluster_name": self.config.name,
            "host": self.config.host,
            "scheduler": self.config.scheduler_type
        }
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


class SSHClusterConnector(ClusterConnector):
    """SSH集群连接器"""
    
    def __init__(self, config: ClusterConfig, use_pool: bool = True):
        super().__init__(config)
        self.use_pool = use_pool
        self._pool: Optional[ConnectionPool] = None
        self._persistent_client: Optional[Any] = None
    
    def connect(self) -> bool:
        """建立SSH连接"""
        if not PARAMIKO_AVAILABLE:
            logger.error("paramiko not available for SSH connections")
            return False
        
        try:
            if self.use_pool:
                self._pool = ConnectionPool(
                    self.config,
                    max_size=self.config.connection_pool_size
                )
            else:
                # 创建持久连接
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                connect_kwargs = {
                    "hostname": self.config.host,
                    "port": self.config.port,
                    "username": self.config.username or os.getenv("USER"),
                    "timeout": self.config.connection_timeout,
                }
                
                if self.config.key_file:
                    connect_kwargs["key_filename"] = self.config.key_file
                elif self.config.password:
                    connect_kwargs["password"] = self.config.password
                
                client.connect(**connect_kwargs)
                self._persistent_client = client
            
            self._connected = True
            self._connection_time = datetime.now()
            logger.info(f"Connected to cluster {self.config.name} ({self.config.host})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.host}: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """断开SSH连接"""
        if self._pool:
            self._pool.close_all()
            self._pool = None
        
        if self._persistent_client:
            self._persistent_client.close()
            self._persistent_client = None
        
        self._connected = False
        logger.info(f"Disconnected from cluster {self.config.name}")
    
    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """执行远程命令"""
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        if self._pool:
            client = self._pool.get_connection(timeout=timeout or self.config.connection_timeout)
            try:
                stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
                exit_status = stdout.channel.recv_exit_status()
                return exit_status, stdout.read().decode(), stderr.read().decode()
            finally:
                self._pool.return_connection(client)
        else:
            stdin, stdout, stderr = self._persistent_client.exec_command(command, timeout=timeout)
            exit_status = stdout.channel.recv_exit_status()
            return exit_status, stdout.read().decode(), stderr.read().decode()
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        """上传文件"""
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        try:
            if self._pool:
                client = self._pool.get_connection()
                try:
                    sftp = client.open_sftp()
                    sftp.put(local_path, remote_path)
                    return True
                finally:
                    self._pool.return_connection(client)
            else:
                sftp = self._persistent_client.open_sftp()
                sftp.put(local_path, remote_path)
                return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        """下载文件"""
        if not self._connected:
            raise ConnectionError("Not connected to cluster")
        
        try:
            if self._pool:
                client = self._pool.get_connection()
                try:
                    sftp = client.open_sftp()
                    sftp.get(remote_path, local_path)
                    return True
                finally:
                    self._pool.return_connection(client)
            else:
                sftp = self._persistent_client.open_sftp()
                sftp.get(remote_path, local_path)
                return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def exists(self, remote_path: str) -> bool:
        """检查远程路径是否存在"""
        code, _, _ = self.execute(f"test -e {remote_path}")
        return code == 0
    
    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """创建远程目录"""
        flag = "-p" if parents else ""
        code, _, _ = self.execute(f"mkdir {flag} {remote_path}")
        return code == 0
    
    def glob(self, pattern: str) -> List[str]:
        """远程文件glob"""
        code, stdout, _ = self.execute(f"ls -1 {pattern} 2>/dev/null")
        if code == 0:
            return [l.strip() for l in stdout.strip().split("\n") if l.strip()]
        return []


class KubernetesConnector(ClusterConnector):
    """Kubernetes集群连接器（用于云HPC）"""
    
    def __init__(self, config: ClusterConfig, namespace: str = "default"):
        super().__init__(config)
        self.namespace = namespace
        self._api_client = None
    
    def connect(self) -> bool:
        """连接到K8s集群"""
        # 这里需要kubernetes客户端库
        # 简化实现，实际使用时需要完整K8s API集成
        try:
            # 检查kubectl是否可用
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                timeout=10
            )
            self._connected = result.returncode == 0
            if self._connected:
                self._connection_time = datetime.now()
            return self._connected
        except Exception as e:
            logger.error(f"K8s connection failed: {e}")
            return False
    
    def disconnect(self):
        """断开K8s连接"""
        self._connected = False
        self._api_client = None
    
    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """在K8s中执行命令（通过kubectl exec）"""
        # 简化实现
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        """上传到K8s PVC"""
        # 使用kubectl cp
        result = subprocess.run(
            ["kubectl", "cp", local_path, f"{self.namespace}/{remote_path}"],
            capture_output=True
        )
        return result.returncode == 0
    
    def download(self, remote_path: str, local_path: str) -> bool:
        """从K8s下载"""
        result = subprocess.run(
            ["kubectl", "cp", f"{self.namespace}/{remote_path}", local_path],
            capture_output=True
        )
        return result.returncode == 0
    
    def exists(self, remote_path: str) -> bool:
        """检查K8s中路径是否存在"""
        code, _, _ = self.execute(f"kubectl exec -n {self.namespace} -- test -e {remote_path}")
        return code == 0
    
    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """在K8s中创建目录"""
        flag = "-p" if parents else ""
        code, _, _ = self.execute(f"kubectl exec -n {self.namespace} -- mkdir {flag} {remote_path}")
        return code == 0


class SlurmRestConnector(ClusterConnector):
    """Slurm REST API连接器"""
    
    def __init__(self, config: ClusterConfig):
        super().__init__(config)
        if not config.api_endpoint:
            raise ValueError("api_endpoint required for REST connector")
        self.base_url = config.api_endpoint.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "X-SLURM-USER-NAME": config.username or os.getenv("USER", "")
        }
        if config.api_token:
            self.headers["Authorization"] = f"Bearer {config.api_token}"
    
    def connect(self) -> bool:
        """测试REST API连接"""
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available")
            return False
        
        try:
            response = requests.get(
                f"{self.base_url}/slurmdb/v0.0.38/ping",
                headers=self.headers,
                timeout=10
            )
            self._connected = response.status_code == 200
            if self._connected:
                self._connection_time = datetime.now()
            return self._connected
        except Exception as e:
            logger.error(f"REST API connection failed: {e}")
            return False
    
    def disconnect(self):
        """断开REST连接"""
        self._connected = False
    
    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """通过REST执行命令（不支持，返回错误）"""
        return 1, "", "REST connector does not support direct command execution"
    
    def submit_job(self, job_script: str) -> str:
        """通过REST提交作业"""
        import base64
        
        payload = {
            "script": base64.b64encode(job_script.encode()).decode(),
            "job": {
                "name": "rest_job"
            }
        }
        
        response = requests.post(
            f"{self.base_url}/slurm/v0.0.38/job/submit",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            return str(data.get("job_id", "unknown"))
        else:
            raise RuntimeError(f"Job submission failed: {response.text}")
    
    def query_job(self, job_id: str) -> dict:
        """通过REST查询作业"""
        response = requests.get(
            f"{self.base_url}/slurm/v0.0.38/job/{job_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Job query failed: {response.text}")
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        """REST API不支持直接文件上传"""
        logger.warning("REST connector does not support direct file upload")
        return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        """REST API不支持直接文件下载"""
        logger.warning("REST connector does not support direct file download")
        return False
    
    def exists(self, remote_path: str) -> bool:
        """REST API不支持文件存在检查"""
        return False
    
    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """REST API不支持目录创建"""
        return False


class LocalConnector(ClusterConnector):
    """本地连接器（用于测试）"""
    
    def connect(self) -> bool:
        """本地始终可用"""
        self._connected = True
        self._connection_time = datetime.now()
        return True
    
    def disconnect(self):
        """断开本地连接"""
        self._connected = False
    
    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """本地执行命令"""
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    
    def upload(self, local_path: str, remote_path: str) -> bool:
        """本地复制文件"""
        import shutil
        try:
            shutil.copy2(local_path, remote_path)
            return True
        except Exception as e:
            logger.error(f"Local copy failed: {e}")
            return False
    
    def download(self, remote_path: str, local_path: str) -> bool:
        """本地复制文件"""
        return self.upload(remote_path, local_path)
    
    def exists(self, remote_path: str) -> bool:
        """检查本地路径"""
        return Path(remote_path).exists()
    
    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """创建本地目录"""
        try:
            Path(remote_path).mkdir(parents=parents, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Local mkdir failed: {e}")
            return False


def get_connector(config: Union[ClusterConfig, str, dict]) -> ClusterConnector:
    """
    获取合适的连接器
    
    Args:
        config: ClusterConfig实例、配置文件路径或配置字典
    
    Returns:
        ClusterConnector实例
    """
    if isinstance(config, str):
        config = ClusterConfig.from_file(config)
    elif isinstance(config, dict):
        config = ClusterConfig.from_dict(config)
    
    # 根据配置选择连接器类型
    if config.api_endpoint and config.scheduler_type == "slurm":
        return SlurmRestConnector(config)
    elif config.host in ["localhost", "127.0.0.1", None]:
        return LocalConnector(config)
    else:
        return SSHClusterConnector(config)


@contextmanager
def connector_context(config: Union[ClusterConfig, str, dict]):
    """连接器上下文管理器"""
    conn = get_connector(config)
    try:
        conn.connect()
        yield conn
    finally:
        conn.disconnect()
