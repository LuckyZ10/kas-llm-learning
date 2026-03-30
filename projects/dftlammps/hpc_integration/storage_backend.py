#!/usr/bin/env python3
"""
storage_backend.py
==================
存储后端模块

支持的存储后端：
- S3 (AWS S3)
- MinIO (兼容S3的本地存储)
- SFTP (SSH文件传输)
- Local (本地文件系统，用于测试)
"""

import os
import io
import re
import json
import time
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Iterator
from pathlib import Path
from datetime import datetime

from .data_sync import SyncResult

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False


@dataclass
class StorageConfig:
    """存储配置"""
    # 通用配置
    type: str  # s3, minio, sftp, local
    
    # S3/MinIO配置
    endpoint: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: str = "us-east-1"
    bucket: str = "dft-lammps"
    
    # 路径前缀
    prefix: str = ""
    
    # SFTP配置
    host: Optional[str] = None
    port: int = 22
    username: Optional[str] = None
    password: Optional[str] = None
    key_file: Optional[str] = None
    
    # 本地配置
    local_path: Optional[str] = None
    
    # 传输配置
    multipart_threshold: int = 8 * 1024 * 1024  # 8MB
    max_concurrency: int = 10
    use_ssl: bool = True
    verify_ssl: bool = True
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0


class StorageBackend(ABC):
    """存储后端基类"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self._connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """连接存储"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """检查路径是否存在"""
        pass
    
    @abstractmethod
    def is_file(self, remote_path: str) -> bool:
        """检查是否为文件"""
        pass
    
    @abstractmethod
    def is_dir(self, remote_path: str) -> bool:
        """检查是否为目录"""
        pass
    
    @abstractmethod
    def list_files(
        self,
        remote_path: str,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[str]:
        """列出文件"""
        pass
    
    @abstractmethod
    def upload_file(
        self,
        local_path: Path,
        remote_path: str
    ) -> SyncResult:
        """上传单个文件"""
        pass
    
    @abstractmethod
    def download_file(
        self,
        remote_path: str,
        local_path: Path
    ) -> SyncResult:
        """下载单个文件"""
        pass
    
    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        """删除文件"""
        pass
    
    @abstractmethod
    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """创建目录"""
        pass
    
    def upload_directory(
        self,
        local_path: Path,
        remote_path: str,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        delete_extraneous: bool = False
    ) -> SyncResult:
        """
        上传整个目录
        
        Args:
            local_path: 本地路径
            remote_path: 远程路径
            include_patterns: 包含模式
            exclude_patterns: 排除模式
            delete_extraneous: 删除远程多余文件
        
        Returns:
            SyncResult实例
        """
        if not local_path.exists():
            return SyncResult(success=False, errors=[f"Local path does not exist: {local_path}"])
        
        files_transferred = 0
        bytes_transferred = 0
        errors = []
        
        include_patterns = include_patterns or ["*"]
        exclude_patterns = exclude_patterns or []
        
        for file_path in self._walk_local_files(local_path, include_patterns, exclude_patterns):
            rel_path = file_path.relative_to(local_path)
            target_path = f"{remote_path}/{rel_path}"
            
            try:
                result = self.upload_file(file_path, target_path)
                if result.success:
                    files_transferred += result.files_transferred
                    bytes_transferred += result.bytes_transferred
                else:
                    errors.extend(result.errors)
            except Exception as e:
                errors.append(str(e))
        
        return SyncResult(
            success=len(errors) == 0,
            files_transferred=files_transferred,
            bytes_transferred=bytes_transferred,
            errors=errors
        )
    
    def download_directory(
        self,
        remote_path: str,
        local_path: Path,
        include_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        delete_extraneous: bool = False
    ) -> SyncResult:
        """
        下载整个目录
        
        Args:
            remote_path: 远程路径
            local_path: 本地路径
            include_patterns: 包含模式
            exclude_patterns: 排除模式
            delete_extraneous: 删除本地多余文件
        
        Returns:
            SyncResult实例
        """
        files_transferred = 0
        bytes_transferred = 0
        errors = []
        
        include_patterns = include_patterns or ["*"]
        exclude_patterns = exclude_patterns or []
        
        try:
            remote_files = self.list_files(remote_path, recursive=True)
            
            for remote_file in remote_files:
                # 检查过滤
                file_name = os.path.basename(remote_file)
                if not self._matches_patterns(file_name, include_patterns):
                    continue
                if self._matches_patterns(file_name, exclude_patterns):
                    continue
                
                rel_path = remote_file[len(remote_path):].lstrip("/")
                target_path = local_path / rel_path
                
                try:
                    result = self.download_file(remote_file, target_path)
                    if result.success:
                        files_transferred += result.files_transferred
                        bytes_transferred += result.bytes_transferred
                    else:
                        errors.extend(result.errors)
                except Exception as e:
                    errors.append(str(e))
        
        except Exception as e:
            errors.append(str(e))
        
        return SyncResult(
            success=len(errors) == 0,
            files_transferred=files_transferred,
            bytes_transferred=bytes_transferred,
            errors=errors
        )
    
    def _walk_local_files(
        self,
        base_path: Path,
        include_patterns: List[str],
        exclude_patterns: List[str]
    ) -> Iterator[Path]:
        """遍历本地文件"""
        if base_path.is_file():
            if self._matches_patterns(base_path.name, include_patterns):
                if not self._matches_patterns(base_path.name, exclude_patterns):
                    yield base_path
            return
        
        for path in base_path.rglob("*"):
            if path.is_file():
                if self._matches_patterns(path.name, include_patterns):
                    if not self._matches_patterns(path.name, exclude_patterns):
                        yield path
    
    def _matches_patterns(self, name: str, patterns: List[str]) -> bool:
        """检查文件名是否匹配模式"""
        import fnmatch
        return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
    
    def _compute_md5(self, file_path: Path) -> str:
        """计算文件MD5"""
        hash_obj = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()


class S3Backend(StorageBackend):
    """S3存储后端"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._client = None
        self._transfer = None
    
    def connect(self) -> bool:
        """连接S3"""
        if not BOTO3_AVAILABLE:
            logger.error("boto3 not available for S3 backend")
            return False
        
        try:
            boto_config = Config(
                max_pool_connections=self.config.max_concurrency,
                retries={'max_attempts': self.config.max_retries}
            )
            
            session_kwargs = {
                "region_name": self.config.region
            }
            
            if self.config.access_key and self.config.secret_key:
                session_kwargs["aws_access_key_id"] = self.config.access_key
                session_kwargs["aws_secret_access_key"] = self.config.secret_key
            
            session = boto3.Session(**session_kwargs)
            
            client_kwargs = {
                "config": boto_config,
                "verify": self.config.verify_ssl
            }
            
            if self.config.endpoint:
                client_kwargs["endpoint_url"] = self.config.endpoint
                client_kwargs["use_ssl"] = self.config.use_ssl
            
            self._client = session.client('s3', **client_kwargs)
            
            # 测试连接
            self._client.head_bucket(Bucket=self.config.bucket)
            
            self._connected = True
            logger.info(f"Connected to S3 bucket: {self.config.bucket}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to S3: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """断开S3连接"""
        if self._client:
            self._client.close()
            self._client = None
        self._connected = False
    
    def exists(self, remote_path: str) -> bool:
        """检查对象是否存在"""
        if not self._connected:
            return False
        
        key = self._make_key(remote_path)
        
        try:
            self._client.head_object(Bucket=self.config.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def is_file(self, remote_path: str) -> bool:
        """检查是否为文件"""
        return self.exists(remote_path)
    
    def is_dir(self, remote_path: str) -> bool:
        """检查是否为目录（在S3中模拟）"""
        key = self._make_key(remote_path)
        if not key.endswith('/'):
            key += '/'
        
        response = self._client.list_objects_v2(
            Bucket=self.config.bucket,
            Prefix=key,
            MaxKeys=1
        )
        return response.get('KeyCount', 0) > 0
    
    def list_files(
        self,
        remote_path: str,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[str]:
        """列出文件"""
        if not self._connected:
            return []
        
        key = self._make_key(remote_path)
        if not key.endswith('/'):
            key += '/'
        
        files = []
        paginator = self._client.get_paginator('list_objects_v2')
        
        delimiter = '' if recursive else '/'
        
        for page in paginator.paginate(
            Bucket=self.config.bucket,
            Prefix=key,
            Delimiter=delimiter
        ):
            # 文件
            for obj in page.get('Contents', []):
                file_key = obj['Key']
                if file_key != key:  # 排除目录本身
                    rel_path = file_key[len(self.config.prefix):].lstrip('/')
                    files.append(rel_path)
            
            # 子目录中的文件（非递归时通过CommonPrefixes获取）
            if not recursive:
                for prefix in page.get('CommonPrefixes', []):
                    prefix_key = prefix['Prefix']
                    files.extend(self.list_files(
                        prefix_key[len(self.config.prefix):].rstrip('/'),
                        pattern,
                        recursive=False
                    ))
        
        # 过滤
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), pattern)]
        
        return files
    
    def upload_file(self, local_path: Path, remote_path: str) -> SyncResult:
        """上传文件到S3"""
        if not self._connected:
            return SyncResult(success=False, errors=["Not connected"])
        
        key = self._make_key(remote_path)
        
        try:
            extra_args = {}
            
            # 计算MD5用于校验
            if self.config.verify_ssl:
                md5 = self._compute_md5(local_path)
                extra_args['ContentMD5'] = md5
            
            # 上传
            self._client.upload_file(
                str(local_path),
                self.config.bucket,
                key,
                ExtraArgs=extra_args,
                Callback=self._upload_progress_callback if logger.isEnabledFor(logging.DEBUG) else None
            )
            
            file_size = local_path.stat().st_size
            
            return SyncResult(
                success=True,
                files_transferred=1,
                bytes_transferred=file_size
            )
            
        except Exception as e:
            return SyncResult(success=False, errors=[str(e)])
    
    def download_file(self, remote_path: str, local_path: Path) -> SyncResult:
        """从S3下载文件"""
        if not self._connected:
            return SyncResult(success=False, errors=["Not connected"])
        
        key = self._make_key(remote_path)
        
        try:
            # 确保目录存在
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 下载
            self._client.download_file(
                self.config.bucket,
                key,
                str(local_path)
            )
            
            # 获取文件大小
            response = self._client.head_object(Bucket=self.config.bucket, Key=key)
            file_size = response['ContentLength']
            
            return SyncResult(
                success=True,
                files_transferred=1,
                bytes_transferred=file_size
            )
            
        except Exception as e:
            return SyncResult(success=False, errors=[str(e)])
    
    def delete_file(self, remote_path: str) -> bool:
        """删除S3对象"""
        if not self._connected:
            return False
        
        key = self._make_key(remote_path)
        
        try:
            self._client.delete_object(Bucket=self.config.bucket, Key=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {remote_path}: {e}")
            return False
    
    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """在S3中创建目录（创建空对象作为标记）"""
        if not self._connected:
            return False
        
        key = self._make_key(remote_path)
        if not key.endswith('/'):
            key += '/'
        
        try:
            self._client.put_object(Bucket=self.config.bucket, Key=key, Body=b'')
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {remote_path}: {e}")
            return False
    
    def _make_key(self, remote_path: str) -> str:
        """生成S3 key"""
        path = remote_path.lstrip('/')
        if self.config.prefix:
            return f"{self.config.prefix}/{path}"
        return path
    
    def _upload_progress_callback(self, bytes_transferred):
        """上传进度回调"""
        logger.debug(f"Uploaded {bytes_transferred} bytes")


class MinIOBackend(S3Backend):
    """MinIO存储后端（兼容S3）"""
    
    def __init__(self, config: StorageConfig):
        # MinIO默认不使用SSL
        config.use_ssl = config.use_ssl if config.endpoint and config.endpoint.startswith('https') else False
        super().__init__(config)
    
    def connect(self) -> bool:
        """连接MinIO"""
        if not self.config.endpoint:
            logger.error("MinIO endpoint required")
            return False
        
        # MinIO使用与S3相同的API
        return super().connect()


class SFTPBackend(StorageBackend):
    """SFTP存储后端"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._client = None
        self._sftp = None
    
    def connect(self) -> bool:
        """连接SFTP"""
        if not PARAMIKO_AVAILABLE:
            logger.error("paramiko not available for SFTP backend")
            return False
        
        try:
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_kwargs = {
                "hostname": self.config.host,
                "port": self.config.port,
                "username": self.config.username,
                "timeout": 30
            }
            
            if self.config.key_file:
                connect_kwargs["key_filename"] = self.config.key_file
            elif self.config.password:
                connect_kwargs["password"] = self.config.password
            
            self._client.connect(**connect_kwargs)
            self._sftp = self._client.open_sftp()
            
            self._connected = True
            logger.info(f"Connected to SFTP: {self.config.host}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SFTP: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """断开SFTP连接"""
        if self._sftp:
            self._sftp.close()
            self._sftp = None
        if self._client:
            self._client.close()
            self._client = None
        self._connected = False
    
    def exists(self, remote_path: str) -> bool:
        """检查路径是否存在"""
        if not self._connected:
            return False
        
        try:
            self._sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False
    
    def is_file(self, remote_path: str) -> bool:
        """检查是否为文件"""
        if not self._connected:
            return False
        
        try:
            stat = self._sftp.stat(remote_path)
            return not stat.st_mode & 0o40000  # S_IFDIR
        except FileNotFoundError:
            return False
    
    def is_dir(self, remote_path: str) -> bool:
        """检查是否为目录"""
        if not self._connected:
            return False
        
        try:
            stat = self._sftp.stat(remote_path)
            return bool(stat.st_mode & 0o40000)  # S_IFDIR
        except FileNotFoundError:
            return False
    
    def list_files(
        self,
        remote_path: str,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[str]:
        """列出文件"""
        if not self._connected:
            return []
        
        files = []
        
        try:
            entries = self._sftp.listdir_attr(remote_path)
            
            for entry in entries:
                full_path = f"{remote_path}/{entry.filename}"
                
                if entry.st_mode & 0o40000:  # 目录
                    if recursive:
                        files.extend(self.list_files(full_path, pattern, recursive))
                else:
                    import fnmatch
                    if fnmatch.fnmatch(entry.filename, pattern):
                        files.append(full_path)
        
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
        
        return files
    
    def upload_file(self, local_path: Path, remote_path: str) -> SyncResult:
        """上传文件到SFTP"""
        if not self._connected:
            return SyncResult(success=False, errors=["Not connected"])
        
        try:
            # 确保远程目录存在
            remote_dir = os.path.dirname(remote_path)
            self._mkdir_p(remote_dir)
            
            # 上传
            self._sftp.put(str(local_path), remote_path)
            
            file_size = local_path.stat().st_size
            
            return SyncResult(
                success=True,
                files_transferred=1,
                bytes_transferred=file_size
            )
            
        except Exception as e:
            return SyncResult(success=False, errors=[str(e)])
    
    def download_file(self, remote_path: str, local_path: Path) -> SyncResult:
        """从SFTP下载文件"""
        if not self._connected:
            return SyncResult(success=False, errors=["Not connected"])
        
        try:
            # 确保本地目录存在
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 下载
            self._sftp.get(remote_path, str(local_path))
            
            # 获取文件大小
            stat = self._sftp.stat(remote_path)
            file_size = stat.st_size
            
            return SyncResult(
                success=True,
                files_transferred=1,
                bytes_transferred=file_size
            )
            
        except Exception as e:
            return SyncResult(success=False, errors=[str(e)])
    
    def delete_file(self, remote_path: str) -> bool:
        """删除远程文件"""
        if not self._connected:
            return False
        
        try:
            self._sftp.remove(remote_path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {remote_path}: {e}")
            return False
    
    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """创建远程目录"""
        if not self._connected:
            return False
        
        try:
            if parents:
                self._mkdir_p(remote_path)
            else:
                self._sftp.mkdir(remote_path)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {remote_path}: {e}")
            return False
    
    def _mkdir_p(self, remote_path: str):
        """递归创建目录"""
        parts = remote_path.strip('/').split('/')
        current = ""
        
        for part in parts:
            current += f"/{part}"
            try:
                self._sftp.mkdir(current)
            except IOError:
                # 目录可能已存在
                pass


class LocalBackend(StorageBackend):
    """本地存储后端（用于测试）"""
    
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._base_path = Path(config.local_path or "/tmp/storage")
    
    def connect(self) -> bool:
        """连接本地存储（始终成功）"""
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._connected = True
        return True
    
    def disconnect(self):
        """断开本地存储"""
        self._connected = False
    
    def _resolve_path(self, remote_path: str) -> Path:
        """解析远程路径为本地路径"""
        # 移除前导斜杠
        path = remote_path.lstrip('/')
        return self._base_path / path
    
    def exists(self, remote_path: str) -> bool:
        """检查路径是否存在"""
        return self._resolve_path(remote_path).exists()
    
    def is_file(self, remote_path: str) -> bool:
        """检查是否为文件"""
        return self._resolve_path(remote_path).is_file()
    
    def is_dir(self, remote_path: str) -> bool:
        """检查是否为目录"""
        return self._resolve_path(remote_path).is_dir()
    
    def list_files(
        self,
        remote_path: str,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[str]:
        """列出文件"""
        path = self._resolve_path(remote_path)
        
        if not path.exists():
            return []
        
        files = []
        
        if recursive:
            for file_path in path.rglob(pattern):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(self._base_path))
                    files.append(rel_path)
        else:
            for file_path in path.glob(pattern):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(self._base_path))
                    files.append(rel_path)
        
        return files
    
    def upload_file(self, local_path: Path, remote_path: str) -> SyncResult:
        """复制文件到本地存储"""
        import shutil
        
        try:
            target = self._resolve_path(remote_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(local_path), str(target))
            
            file_size = local_path.stat().st_size
            
            return SyncResult(
                success=True,
                files_transferred=1,
                bytes_transferred=file_size
            )
            
        except Exception as e:
            return SyncResult(success=False, errors=[str(e)])
    
    def download_file(self, remote_path: str, local_path: Path) -> SyncResult:
        """从本地存储复制文件"""
        import shutil
        
        try:
            source = self._resolve_path(remote_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(local_path))
            
            file_size = source.stat().st_size
            
            return SyncResult(
                success=True,
                files_transferred=1,
                bytes_transferred=file_size
            )
            
        except Exception as e:
            return SyncResult(success=False, errors=[str(e)])
    
    def delete_file(self, remote_path: str) -> bool:
        """删除本地文件"""
        try:
            path = self._resolve_path(remote_path)
            path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to delete {remote_path}: {e}")
            return False
    
    def mkdir(self, remote_path: str, parents: bool = True) -> bool:
        """创建本地目录"""
        try:
            path = self._resolve_path(remote_path)
            path.mkdir(parents=parents, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {remote_path}: {e}")
            return False


def get_storage_backend(config: StorageConfig) -> StorageBackend:
    """
    获取存储后端实例
    
    Args:
        config: 存储配置
    
    Returns:
        StorageBackend实例
    """
    backends = {
        "s3": S3Backend,
        "minio": MinIOBackend,
        "sftp": SFTPBackend,
        "local": LocalBackend
    }
    
    backend_class = backends.get(config.type.lower())
    if not backend_class:
        raise ValueError(f"Unknown storage type: {config.type}")
    
    return backend_class(config)
