#!/usr/bin/env python3
"""
checkpoint_manager.py
====================
断点续传和容错机制模块

功能：
1. 自动保存中间状态
2. 崩溃后从断点恢复
3. 任务重试和失败转移
4. 分布式检查点管理
5. 数据一致性保证

作者: HPC & Performance Optimization Expert
日期: 2026-03-09
"""

import os
import sys
import json
import pickle
import hashlib
import shutil
import signal
import atexit
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
from contextlib import contextmanager
from abc import ABC, abstractmethod
import traceback
import tempfile

# 尝试导入可选依赖
try:
    import fsspec
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据类定义
# =============================================================================

class CheckpointStatus(Enum):
    """检查点状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"


@dataclass
class CheckpointMetadata:
    """检查点元数据"""
    checkpoint_id: str
    created_at: datetime
    version: str = "1.0"
    
    # 任务信息
    task_name: str = ""
    task_type: str = ""
    iteration: int = 0
    total_iterations: int = 0
    
    # 状态信息
    status: CheckpointStatus = CheckpointStatus.PENDING
    progress_percentage: float = 0.0
    
    # 资源使用
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    elapsed_time_seconds: float = 0.0
    
    # 依赖
    parent_checkpoint_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    # 恢复信息
    resume_count: int = 0
    last_resume_at: Optional[datetime] = None
    
    # 标签和元数据
    tags: Dict[str, str] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        d = asdict(self)
        d['status'] = self.status.value
        d['created_at'] = self.created_at.isoformat()
        d['last_resume_at'] = self.last_resume_at.isoformat() if self.last_resume_at else None
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointMetadata":
        """从字典创建"""
        d = d.copy()
        d['status'] = CheckpointStatus(d['status'])
        d['created_at'] = datetime.fromisoformat(d['created_at'])
        if d['last_resume_at']:
            d['last_resume_at'] = datetime.fromisoformat(d['last_resume_at'])
        return cls(**d)


@dataclass
class CheckpointConfig:
    """检查点配置"""
    # 存储配置
    checkpoint_dir: str = "./checkpoints"
    storage_backend: str = "local"  # local, s3, gcs, hdfs
    storage_options: Dict[str, Any] = field(default_factory=dict)
    
    # 保存策略
    save_interval_seconds: Optional[float] = 600.0  # 10分钟
    save_interval_iterations: int = 100
    keep_last_n: int = 5  # 保留最近N个检查点
    keep_best: bool = True  # 保留最佳检查点
    
    # 压缩和校验
    compression: str = "gzip"  # gzip, bz2, lz4, None
    verify_checksum: bool = True
    
    # 容错
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    auto_resume: bool = True
    
    # 异步保存
    async_save: bool = False
    async_queue_size: int = 2
    
    # 清理
    cleanup_on_complete: bool = False
    cleanup_interval_hours: float = 24.0


@dataclass
class RecoveryInfo:
    """恢复信息"""
    can_resume: bool
    last_checkpoint_id: Optional[str] = None
    last_checkpoint_path: Optional[str] = None
    last_iteration: int = 0
    recovered_state: Optional[Any] = None
    error_message: Optional[str] = None


# =============================================================================
# 存储后端
# =============================================================================

class StorageBackend(ABC):
    """存储后端基类"""
    
    @abstractmethod
    def save(self, path: str, data: bytes) -> bool:
        """保存数据"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> Optional[bytes]:
        """加载数据"""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """检查路径是否存在"""
        pass
    
    @abstractmethod
    def list_dir(self, path: str) -> List[str]:
        """列出目录内容"""
        pass
    
    @abstractmethod
    def delete(self, path: str) -> bool:
        """删除路径"""
        pass
    
    @abstractmethod
    def makedirs(self, path: str) -> bool:
        """创建目录"""
        pass


class LocalStorageBackend(StorageBackend):
    """本地文件系统存储"""
    
    def save(self, path: str, data: bytes) -> bool:
        try:
            # 原子写入
            temp_path = path + ".tmp"
            with open(temp_path, 'wb') as f:
                f.write(data)
            os.rename(temp_path, path)
            return True
        except Exception as e:
            logger.error(f"Failed to save to {path}: {e}")
            return False
    
    def load(self, path: str) -> Optional[bytes]:
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load from {path}: {e}")
            return None
    
    def exists(self, path: str) -> bool:
        return os.path.exists(path)
    
    def list_dir(self, path: str) -> List[str]:
        if not os.path.isdir(path):
            return []
        return os.listdir(path)
    
    def delete(self, path: str) -> bool:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False
    
    def makedirs(self, path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False


class CloudStorageBackend(StorageBackend):
    """云存储后端（使用fsspec）"""
    
    def __init__(self, protocol: str, storage_options: Dict[str, Any]):
        if not FSSPEC_AVAILABLE:
            raise ImportError("fsspec is required for cloud storage")
        
        self.protocol = protocol
        self.storage_options = storage_options
        self.fs = fsspec.filesystem(protocol, **storage_options)
    
    def _make_path(self, path: str) -> str:
        """构造完整路径"""
        if not path.startswith(f"{self.protocol}://"):
            path = f"{self.protocol}://{path}"
        return path
    
    def save(self, path: str, data: bytes) -> bool:
        try:
            path = self._make_path(path)
            with self.fs.open(path, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            logger.error(f"Failed to save to {path}: {e}")
            return False
    
    def load(self, path: str) -> Optional[bytes]:
        try:
            path = self._make_path(path)
            with self.fs.open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load from {path}: {e}")
            return None
    
    def exists(self, path: str) -> bool:
        path = self._make_path(path)
        return self.fs.exists(path)
    
    def list_dir(self, path: str) -> List[str]:
        path = self._make_path(path)
        if not self.fs.isdir(path):
            return []
        return self.fs.listdir(path)
    
    def delete(self, path: str) -> bool:
        try:
            path = self._make_path(path)
            self.fs.delete(path, recursive=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False
    
    def makedirs(self, path: str) -> bool:
        try:
            path = self._make_path(path)
            self.fs.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False


def create_storage_backend(config: CheckpointConfig) -> StorageBackend:
    """创建存储后端"""
    if config.storage_backend == "local":
        return LocalStorageBackend()
    elif config.storage_backend in ["s3", "gcs", "hdfs", "azure"]:
        return CloudStorageBackend(config.storage_backend, config.storage_options)
    else:
        raise ValueError(f"Unknown storage backend: {config.storage_backend}")


# =============================================================================
# 检查点管理器
# =============================================================================

class CheckpointManager:
    """
    检查点管理器
    
    提供全面的检查点管理功能，包括保存、加载、恢复和清理
    """
    
    def __init__(self, config: Optional[CheckpointConfig] = None):
        self.config = config or CheckpointConfig()
        self.backend = create_storage_backend(self.config)
        
        # 初始化检查点目录
        self.backend.makedirs(self.config.checkpoint_dir)
        
        # 状态跟踪
        self._current_checkpoint_id: Optional[str] = None
        self._current_metadata: Optional[CheckpointMetadata] = None
        self._checkpoint_history: List[str] = []
        
        # 异步保存
        self._async_queue: Optional[queue.Queue] = None
        self._async_thread: Optional[threading.Thread] = None
        self._stop_async = threading.Event()
        
        if self.config.async_save:
            self._start_async_worker()
        
        # 注册信号处理器
        self._register_signal_handlers()
        
        # 自动清理旧检查点
        if self.config.cleanup_interval_hours > 0:
            self._cleanup_old_checkpoints()
    
    def _register_signal_handlers(self):
        """注册信号处理器以实现优雅退出"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint before exit...")
            self._emergency_save()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 注册退出回调
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """清理资源"""
        if self.config.async_save:
            self._stop_async.set()
            if self._async_thread:
                self._async_thread.join(timeout=5.0)
    
    def _start_async_worker(self):
        """启动异步保存工作线程"""
        self._async_queue = queue.Queue(maxsize=self.config.async_queue_size)
        
        def async_worker():
            while not self._stop_async.is_set():
                try:
                    task = self._async_queue.get(timeout=1.0)
                    if task is None:
                        break
                    checkpoint_id, state, metadata = task
                    self._do_save(checkpoint_id, state, metadata)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Async save error: {e}")
        
        self._async_thread = threading.Thread(target=async_worker, daemon=True)
        self._async_thread.start()
    
    def _generate_checkpoint_id(self, task_name: str = "") -> str:
        """生成检查点ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = hashlib.md5(f"{task_name}_{time.time()}".encode()).hexdigest()[:8]
        return f"{task_name}_{timestamp}_{unique}" if task_name else f"checkpoint_{timestamp}_{unique}"
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> str:
        """获取检查点路径"""
        return os.path.join(self.config.checkpoint_dir, checkpoint_id)
    
    def save(
        self,
        state: Any,
        task_name: str = "",
        task_type: str = "",
        iteration: int = 0,
        total_iterations: int = 0,
        progress_percentage: float = 0.0,
        tags: Optional[Dict[str, str]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        保存检查点
        
        Args:
            state: 要保存的状态（需可序列化）
            task_name: 任务名称
            task_type: 任务类型
            iteration: 当前迭代
            total_iterations: 总迭代数
            progress_percentage: 进度百分比
            tags: 标签
            custom_data: 自定义数据
            is_best: 是否为最佳检查点
        
        Returns:
            检查点ID
        """
        checkpoint_id = self._generate_checkpoint_id(task_name)
        
        # 创建元数据
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            created_at=datetime.now(),
            task_name=task_name,
            task_type=task_type,
            iteration=iteration,
            total_iterations=total_iterations,
            status=CheckpointStatus.IN_PROGRESS,
            progress_percentage=progress_percentage,
            parent_checkpoint_id=self._current_checkpoint_id,
            tags=tags or {},
            custom_data=custom_data or {}
        )
        
        # 异步或同步保存
        if self.config.async_save:
            self._async_queue.put((checkpoint_id, state, metadata))
        else:
            self._do_save(checkpoint_id, state, metadata)
        
        # 更新当前检查点
        self._current_checkpoint_id = checkpoint_id
        self._current_metadata = metadata
        self._checkpoint_history.append(checkpoint_id)
        
        # 清理旧检查点
        self._rotate_checkpoints()
        
        # 保存最佳检查点
        if is_best:
            self._save_best_checkpoint(checkpoint_id, state, metadata)
        
        logger.info(f"Checkpoint saved: {checkpoint_id} (iteration {iteration})")
        return checkpoint_id
    
    def _do_save(self, checkpoint_id: str, state: Any, metadata: CheckpointMetadata):
        """执行实际保存操作"""
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        self.backend.makedirs(checkpoint_path)
        
        try:
            # 序列化状态
            state_data = self._serialize(state)
            
            # 应用压缩
            if self.config.compression:
                state_data = self._compress(state_data)
            
            # 计算校验和
            if self.config.verify_checksum:
                checksum = hashlib.sha256(state_data).hexdigest()
                metadata.custom_data["checksum"] = checksum
            
            # 保存状态文件
            state_path = os.path.join(checkpoint_path, "state.pkl")
            self.backend.save(state_path, state_data)
            
            # 保存元数据
            metadata.disk_usage_mb = len(state_data) / (1024 * 1024)
            metadata.status = CheckpointStatus.COMPLETED
            
            meta_path = os.path.join(checkpoint_path, "metadata.json")
            meta_data = json.dumps(metadata.to_dict(), indent=2).encode()
            self.backend.save(meta_path, meta_data)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            metadata.status = CheckpointStatus.FAILED
    
    def _serialize(self, state: Any) -> bytes:
        """序列化状态"""
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化状态"""
        return pickle.loads(data)
    
    def _compress(self, data: bytes) -> bytes:
        """压缩数据"""
        if self.config.compression == "gzip":
            import gzip
            return gzip.compress(data)
        elif self.config.compression == "bz2":
            import bz2
            return bz2.compress(data)
        elif self.config.compression == "lz4":
            try:
                import lz4.frame
                return lz4.frame.compress(data)
            except ImportError:
                logger.warning("lz4 not available, skipping compression")
                return data
        return data
    
    def _decompress(self, data: bytes) -> bytes:
        """解压数据"""
        if self.config.compression == "gzip":
            import gzip
            return gzip.decompress(data)
        elif self.config.compression == "bz2":
            import bz2
            return bz2.decompress(data)
        elif self.config.compression == "lz4":
            try:
                import lz4.frame
                return lz4.frame.decompress(data)
            except ImportError:
                return data
        return data
    
    def load(self, checkpoint_id: Optional[str] = None) -> Tuple[Any, CheckpointMetadata]:
        """
        加载检查点
        
        Args:
            checkpoint_id: 检查点ID，None则加载最新的
        
        Returns:
            (状态, 元数据)
        """
        if checkpoint_id is None:
            checkpoint_id = self._find_latest_checkpoint()
            if checkpoint_id is None:
                raise FileNotFoundError("No checkpoint found")
        
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        # 加载元数据
        meta_path = os.path.join(checkpoint_path, "metadata.json")
        meta_data = self.backend.load(meta_path)
        if meta_data is None:
            raise FileNotFoundError(f"Checkpoint metadata not found: {checkpoint_id}")
        
        metadata = CheckpointMetadata.from_dict(json.loads(meta_data))
        
        # 加载状态
        state_path = os.path.join(checkpoint_path, "state.pkl")
        state_data = self.backend.load(state_path)
        if state_data is None:
            raise FileNotFoundError(f"Checkpoint state not found: {checkpoint_id}")
        
        # 验证校验和
        if self.config.verify_checksum and "checksum" in metadata.custom_data:
            expected_checksum = metadata.custom_data["checksum"]
            actual_checksum = hashlib.sha256(state_data).hexdigest()
            if expected_checksum != actual_checksum:
                metadata.status = CheckpointStatus.CORRUPTED
                raise ValueError(f"Checkpoint checksum mismatch: {checkpoint_id}")
        
        # 解压
        if self.config.compression:
            state_data = self._decompress(state_data)
        
        state = self._deserialize(state_data)
        
        # 更新恢复信息
        metadata.resume_count += 1
        metadata.last_resume_at = datetime.now()
        
        self._current_checkpoint_id = checkpoint_id
        self._current_metadata = metadata
        
        logger.info(f"Checkpoint loaded: {checkpoint_id} (iteration {metadata.iteration})")
        return state, metadata
    
    def recover(self, task_name: str = "") -> RecoveryInfo:
        """
        尝试恢复任务状态
        
        Args:
            task_name: 任务名称过滤器
        
        Returns:
            恢复信息
        """
        try:
            # 查找最新的可恢复检查点
            checkpoint_id = self._find_latest_checkpoint(task_name)
            
            if checkpoint_id is None:
                return RecoveryInfo(
                    can_resume=False,
                    error_message="No checkpoint found"
                )
            
            # 加载检查点
            state, metadata = self.load(checkpoint_id)
            
            # 检查状态
            if metadata.status == CheckpointStatus.FAILED:
                return RecoveryInfo(
                    can_resume=False,
                    last_checkpoint_id=checkpoint_id,
                    error_message="Checkpoint was marked as failed"
                )
            
            if metadata.status == CheckpointStatus.CORRUPTED:
                # 尝试回退到父检查点
                if metadata.parent_checkpoint_id:
                    logger.warning(f"Checkpoint corrupted, trying parent: {metadata.parent_checkpoint_id}")
                    return self.recover_from(metadata.parent_checkpoint_id)
                else:
                    return RecoveryInfo(
                        can_resume=False,
                        last_checkpoint_id=checkpoint_id,
                        error_message="Checkpoint corrupted, no parent available"
                    )
            
            return RecoveryInfo(
                can_resume=True,
                last_checkpoint_id=checkpoint_id,
                last_checkpoint_path=self._get_checkpoint_path(checkpoint_id),
                last_iteration=metadata.iteration,
                recovered_state=state
            )
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return RecoveryInfo(
                can_resume=False,
                error_message=str(e)
            )
    
    def recover_from(self, checkpoint_id: str) -> RecoveryInfo:
        """从指定检查点恢复"""
        try:
            state, metadata = self.load(checkpoint_id)
            
            return RecoveryInfo(
                can_resume=True,
                last_checkpoint_id=checkpoint_id,
                last_checkpoint_path=self._get_checkpoint_path(checkpoint_id),
                last_iteration=metadata.iteration,
                recovered_state=state
            )
        except Exception as e:
            return RecoveryInfo(
                can_resume=False,
                last_checkpoint_id=checkpoint_id,
                error_message=str(e)
            )
    
    def _find_latest_checkpoint(self, task_name: str = "") -> Optional[str]:
        """查找最新的检查点"""
        checkpoints = self.list_checkpoints(task_name)
        
        if not checkpoints:
            return None
        
        # 按创建时间排序
        checkpoints.sort(key=lambda x: x["metadata"].created_at, reverse=True)
        
        # 返回第一个非失败的检查点
        for cp in checkpoints:
            if cp["metadata"].status not in [CheckpointStatus.FAILED, CheckpointStatus.CORRUPTED]:
                return cp["checkpoint_id"]
        
        return checkpoints[0]["checkpoint_id"] if checkpoints else None
    
    def list_checkpoints(self, task_name: str = "", status: Optional[CheckpointStatus] = None) -> List[Dict]:
        """
        列出检查点
        
        Args:
            task_name: 任务名称过滤器
            status: 状态过滤器
        
        Returns:
            检查点信息列表
        """
        checkpoints = []
        
        try:
            entries = self.backend.list_dir(self.config.checkpoint_dir)
        except Exception:
            return checkpoints
        
        for entry in entries:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, entry)
            meta_path = os.path.join(checkpoint_path, "metadata.json")
            
            if not self.backend.exists(meta_path):
                continue
            
            try:
                meta_data = self.backend.load(meta_path)
                if meta_data is None:
                    continue
                
                metadata = CheckpointMetadata.from_dict(json.loads(meta_data))
                
                # 过滤
                if task_name and not metadata.task_name.startswith(task_name):
                    continue
                if status and metadata.status != status:
                    continue
                
                checkpoints.append({
                    "checkpoint_id": entry,
                    "path": checkpoint_path,
                    "metadata": metadata
                })
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata {entry}: {e}")
        
        return checkpoints
    
    def _rotate_checkpoints(self):
        """轮转检查点，保留最近的N个"""
        if self.config.keep_last_n <= 0:
            return
        
        checkpoints = self.list_checkpoints()
        checkpoints.sort(key=lambda x: x["metadata"].created_at, reverse=True)
        
        # 保留keep_last_n个
        to_delete = checkpoints[self.config.keep_last_n:]
        
        for cp in to_delete:
            # 不删除最佳检查点
            if "best" in cp["checkpoint_id"]:
                continue
            
            try:
                self.backend.delete(cp["path"])
                logger.debug(f"Deleted old checkpoint: {cp['checkpoint_id']}")
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {cp['checkpoint_id']}: {e}")
    
    def _save_best_checkpoint(self, checkpoint_id: str, state: Any, metadata: CheckpointMetadata):
        """保存最佳检查点"""
        best_id = f"{metadata.task_name}_best" if metadata.task_name else "best"
        best_path = self._get_checkpoint_path(best_id)
        
        try:
            self.backend.makedirs(best_path)
            
            # 复制状态
            source_path = os.path.join(self._get_checkpoint_path(checkpoint_id), "state.pkl")
            dest_path = os.path.join(best_path, "state.pkl")
            
            state_data = self.backend.load(source_path)
            if state_data:
                self.backend.save(dest_path, state_data)
            
            # 保存元数据
            best_metadata = CheckpointMetadata(
                checkpoint_id=best_id,
                created_at=datetime.now(),
                task_name=metadata.task_name,
                task_type=metadata.task_type,
                iteration=metadata.iteration,
                status=CheckpointStatus.COMPLETED,
                progress_percentage=metadata.progress_percentage,
                tags={**metadata.tags, "is_best": "true"}
            )
            
            meta_path = os.path.join(best_path, "metadata.json")
            meta_data = json.dumps(best_metadata.to_dict(), indent=2).encode()
            self.backend.save(meta_path, meta_data)
            
            logger.info(f"Best checkpoint saved: {best_id}")
            
        except Exception as e:
            logger.error(f"Failed to save best checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self):
        """清理过期的检查点"""
        cutoff = datetime.now() - timedelta(hours=self.config.cleanup_interval_hours)
        
        checkpoints = self.list_checkpoints()
        for cp in checkpoints:
            if cp["metadata"].created_at < cutoff:
                try:
                    self.backend.delete(cp["path"])
                    logger.info(f"Cleaned up old checkpoint: {cp['checkpoint_id']}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup checkpoint {cp['checkpoint_id']}: {e}")
    
    def _emergency_save(self):
        """紧急保存当前状态"""
        if self._current_metadata:
            try:
                # 创建紧急检查点
                emergency_id = f"emergency_{int(time.time())}"
                emergency_path = self._get_checkpoint_path(emergency_id)
                self.backend.makedirs(emergency_path)
                
                # 保存元数据
                emergency_metadata = CheckpointMetadata(
                    checkpoint_id=emergency_id,
                    created_at=datetime.now(),
                    task_name=self._current_metadata.task_name,
                    status=CheckpointStatus.FAILED,
                    tags={"emergency": "true", "parent": self._current_checkpoint_id or ""}
                )
                
                meta_path = os.path.join(emergency_path, "metadata.json")
                meta_data = json.dumps(emergency_metadata.to_dict()).encode()
                self.backend.save(meta_path, meta_data)
                
                logger.info(f"Emergency checkpoint created: {emergency_id}")
            except Exception as e:
                logger.error(f"Emergency save failed: {e}")
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除指定检查点"""
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        return self.backend.delete(checkpoint_path)
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict]:
        """获取检查点信息"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_id)
        meta_path = os.path.join(checkpoint_path, "metadata.json")
        
        if not self.backend.exists(meta_path):
            return None
        
        meta_data = self.backend.load(meta_path)
        if meta_data is None:
            return None
        
        metadata = CheckpointMetadata.from_dict(json.loads(meta_data))
        
        return {
            "checkpoint_id": checkpoint_id,
            "path": checkpoint_path,
            "metadata": metadata
        }


# =============================================================================
# 容错任务执行器
# =============================================================================

class FaultTolerantExecutor:
    """
    容错任务执行器
    
    自动重试失败的任务，支持指数退避和断路器模式
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        
        # 断路器状态
        self._failure_count = 0
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_timeout = circuit_breaker_timeout
        self._circuit_open_until: Optional[float] = None
        
        # 统计
        self._success_count = 0
        self._failure_count_total = 0
        self._retry_count = 0
    
    def _is_circuit_open(self) -> bool:
        """检查断路器是否打开"""
        if self._circuit_open_until is None:
            return False
        
        if time.time() > self._circuit_open_until:
            # 重置断路器
            self._circuit_open_until = None
            self._failure_count = 0
            return False
        
        return True
    
    def _open_circuit(self):
        """打开断路器"""
        self._circuit_open_until = time.time() + self._circuit_breaker_timeout
        logger.warning(f"Circuit breaker opened for {self._circuit_breaker_timeout}s")
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行函数，自动重试
        
        Args:
            func: 要执行的函数
            *args, **kwargs: 函数参数
        
        Returns:
            函数返回值
        
        Raises:
            RuntimeError: 所有重试失败后
        """
        if self._is_circuit_open():
            raise RuntimeError("Circuit breaker is open")
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # 成功，重置失败计数
                if attempt > 0:
                    self._failure_count = max(0, self._failure_count - 1)
                
                self._success_count += 1
                return result
                
            except Exception as e:
                last_exception = e
                self._failure_count += 1
                self._failure_count_total += 1
                
                # 检查是否需要打开断路器
                if self._failure_count >= self._circuit_breaker_threshold:
                    self._open_circuit()
                    raise RuntimeError(f"Circuit breaker opened after {self._failure_count} failures") from e
                
                if attempt < self.max_retries:
                    # 计算延迟（指数退避）
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    self._retry_count += 1
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise RuntimeError(f"Task failed after {self.max_retries + 1} attempts") from last_exception
    
    def get_stats(self) -> Dict:
        """获取执行统计"""
        return {
            "success_count": self._success_count,
            "failure_count": self._failure_count_total,
            "retry_count": self._retry_count,
            "failure_rate": self._failure_count_total / max(1, self._success_count + self._failure_count_total),
            "circuit_open": self._is_circuit_open()
        }


# =============================================================================
# 自动检查点装饰器
# =============================================================================

def checkpointed(
    checkpoint_dir: str = "./checkpoints",
    save_interval_iterations: int = 100,
    save_interval_seconds: Optional[float] = None,
    task_name: Optional[str] = None
):
    """
    自动检查点装饰器
    
    自动为迭代函数添加检查点功能
    
    示例：
        @checkpointed(checkpoint_dir="./my_checkpoints", save_interval_iterations=10)
        def train_model(iteration, state):
            # 训练逻辑
            return new_state
    """
    def decorator(func: Callable):
        config = CheckpointConfig(
            checkpoint_dir=checkpoint_dir,
            save_interval_iterations=save_interval_iterations,
            save_interval_seconds=save_interval_seconds
        )
        manager = CheckpointManager(config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 尝试恢复
            recovery = manager.recover(task_name or func.__name__)
            
            start_iteration = 0
            state = None
            
            if recovery.can_resume:
                logger.info(f"Resuming from checkpoint: {recovery.last_checkpoint_id}")
                start_iteration = recovery.last_iteration + 1
                state = recovery.recovered_state
            
            # 执行函数
            iteration = start_iteration
            last_save_time = time.time()
            
            try:
                while True:
                    # 调用函数
                    result = func(iteration, state, *args, **kwargs)
                    
                    # 检查是否应该保存
                    should_save = (
                        iteration % save_interval_iterations == 0
                    )
                    
                    if save_interval_seconds:
                        should_save = should_save or (
                            time.time() - last_save_time >= save_interval_seconds
                        )
                    
                    if should_save:
                        manager.save(
                            state=result,
                            task_name=task_name or func.__name__,
                            iteration=iteration
                        )
                        last_save_time = time.time()
                    
                    state = result
                    iteration += 1
                    
            except Exception as e:
                logger.error(f"Function failed at iteration {iteration}: {e}")
                # 保存失败前的状态
                manager.save(
                    state=state,
                    task_name=task_name or func.__name__,
                    iteration=iteration,
                    tags={"status": "failed", "error": str(e)}
                )
                raise
            
            return state
        
        return wrapper
    return decorator


# =============================================================================
# 上下文管理器
# =============================================================================

@contextmanager
def checkpoint_context(
    checkpoint_manager: CheckpointManager,
    task_name: str,
    task_type: str = "",
    auto_save: bool = True,
    save_interval_iterations: int = 100
):
    """
    检查点上下文管理器
    
    示例：
        with checkpoint_context(manager, "my_task") as ctx:
            for i in range(1000):
                result = do_work(i)
                ctx.save_if_needed(i, result)
    """
    class CheckpointContext:
        def __init__(self, manager, task_name, task_type):
            self.manager = manager
            self.task_name = task_name
            self.task_type = task_type
            self.iteration = 0
            self._last_save_iteration = 0
        
        def save(self, iteration: int, state: Any, **kwargs):
            """手动保存检查点"""
            self.manager.save(
                state=state,
                task_name=self.task_name,
                task_type=self.task_type,
                iteration=iteration,
                **kwargs
            )
            self._last_save_iteration = iteration
        
        def save_if_needed(self, iteration: int, state: Any, **kwargs):
            """根据间隔自动保存"""
            if iteration - self._last_save_iteration >= save_interval_iterations:
                self.save(iteration, state, **kwargs)
        
        def recover(self) -> RecoveryInfo:
            """恢复状态"""
            return self.manager.recover(self.task_name)
    
    ctx = CheckpointContext(checkpoint_manager, task_name, task_type)
    
    # 尝试恢复
    if checkpoint_manager.config.auto_resume:
        recovery = ctx.recover()
        if recovery.can_resume:
            ctx.iteration = recovery.last_iteration
            logger.info(f"Resumed {task_name} from iteration {ctx.iteration}")
    
    try:
        yield ctx
    finally:
        # 最终保存
        if auto_save:
            logger.info(f"Final checkpoint for {task_name}")


# =============================================================================
# 高级API
# =============================================================================

def create_checkpoint_manager(
    checkpoint_dir: str = "./checkpoints",
    **kwargs
) -> CheckpointManager:
    """
    创建检查点管理器的便捷函数
    
    Args:
        checkpoint_dir: 检查点目录
        **kwargs: 其他配置参数
    
    Returns:
        CheckpointManager实例
    """
    config = CheckpointConfig(checkpoint_dir=checkpoint_dir, **kwargs)
    return CheckpointManager(config)


def resume_or_start(
    checkpoint_dir: str,
    task_name: str,
    init_func: Callable[[], Any],
    resume_func: Optional[Callable[[Any, int], Any]] = None
) -> Tuple[Any, int, bool]:
    """
    自动恢复或开始新任务
    
    Args:
        checkpoint_dir: 检查点目录
        task_name: 任务名称
        init_func: 初始化函数，返回初始状态
        resume_func: 恢复后的处理函数
    
    Returns:
        (状态, 起始迭代, 是否恢复)
    """
    manager = create_checkpoint_manager(checkpoint_dir)
    recovery = manager.recover(task_name)
    
    if recovery.can_resume:
        state = recovery.recovered_state
        start_iteration = recovery.last_iteration + 1
        
        if resume_func:
            state = resume_func(state, start_iteration)
        
        logger.info(f"Resumed task '{task_name}' from iteration {start_iteration}")
        return state, start_iteration, True
    else:
        state = init_func()
        logger.info(f"Started new task '{task_name}'")
        return state, 0, False


# =============================================================================
# 示例用法
# =============================================================================

if __name__ == "__main__":
    # 示例1: 基本使用
    config = CheckpointConfig(
        checkpoint_dir="./test_checkpoints",
        save_interval_iterations=10,
        keep_last_n=3
    )
    
    manager = CheckpointManager(config)
    
    # 模拟训练循环
    for iteration in range(100):
        state = {"iteration": iteration, "loss": 1.0 / (iteration + 1)}
        
        # 每10次迭代保存检查点
        if iteration % 10 == 0:
            manager.save(
                state=state,
                task_name="training",
                iteration=iteration,
                progress_percentage=iteration
            )
    
    # 示例2: 恢复
    recovery = manager.recover("training")
    if recovery.can_resume:
        print(f"Can resume from iteration {recovery.last_iteration}")
        print(f"Recovered state: {recovery.recovered_state}")
    
    # 示例3: 容错执行
    executor = FaultTolerantExecutor(max_retries=3)
    
    def risky_operation():
        import random
        if random.random() < 0.7:
            raise RuntimeError("Random failure")
        return "success"
    
    try:
        result = executor.execute(risky_operation)
        print(f"Result: {result}")
    except RuntimeError as e:
        print(f"Failed after retries: {e}")
    
    # 示例4: 使用装饰器
    @checkpointed(checkpoint_dir="./decorator_checkpoints", save_interval_iterations=5)
    def long_running_task(iteration, state, max_iterations=100):
        if state is None:
            state = {"count": 0, "data": []}
        
        state["count"] += 1
        state["data"].append(iteration)
        
        # 模拟工作
        time.sleep(0.1)
        
        if iteration >= max_iterations:
            raise StopIteration("Task complete")
        
        return state
    
    print("checkpoint_manager module loaded successfully")
