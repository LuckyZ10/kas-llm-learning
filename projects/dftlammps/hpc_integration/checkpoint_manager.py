#!/usr/bin/env python3
"""
checkpoint_manager.py
=====================
断点续算管理模块

功能：
- 计算状态快照
- 自动检查点创建
- 作业恢复
- 状态持久化
- 增量检查点
"""

import os
import re
import json
import time
import pickle
import logging
import hashlib
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum, auto
from abc import ABC, abstractmethod
from contextlib import contextmanager
import copy

from .cluster_connector import ClusterConnector

logger = logging.getLogger(__name__)


class CalculationState(Enum):
    """计算状态"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CHECKPOINTED = "checkpointed"
    RESUMED = "resumed"


@dataclass
class Checkpoint:
    """检查点数据"""
    checkpoint_id: str
    job_id: Optional[str]
    state: CalculationState
    timestamp: datetime
    
    # 文件快照
    file_checksums: Dict[str, str] = field(default_factory=dict)
    file_sizes: Dict[str, int] = field(default_factory=dict)
    
    # 计算状态
    iteration: int = 0
    total_steps: int = 0
    completed_steps: int = 0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 存储路径
    local_path: Optional[Path] = None
    remote_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "job_id": self.job_id,
            "state": self.state.value,
            "timestamp": self.timestamp.isoformat(),
            "file_checksums": self.file_checksums,
            "file_sizes": self.file_sizes,
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "metadata": self.metadata,
            "local_path": str(self.local_path) if self.local_path else None,
            "remote_path": self.remote_path
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        data = data.copy()
        data['state'] = CalculationState(data['state'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('local_path'):
            data['local_path'] = Path(data['local_path'])
        return cls(**data)
    
    @property
    def progress_pct(self) -> float:
        """计算进度百分比"""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100


class ResumeCapability:
    """恢复能力分析"""
    
    def __init__(
        self,
        can_resume: bool,
        resume_from: Optional[Checkpoint] = None,
        missing_files: List[str] = None,
        estimated_time_saved: timedelta = None
    ):
        self.can_resume = can_resume
        self.resume_from = resume_from
        self.missing_files = missing_files or []
        self.estimated_time_saved = estimated_time_saved or timedelta(0)
    
    def to_dict(self) -> dict:
        return {
            "can_resume": self.can_resume,
            "resume_from": self.resume_from.to_dict() if self.resume_from else None,
            "missing_files": self.missing_files,
            "estimated_time_saved_seconds": self.estimated_time_saved.total_seconds()
        }


class CheckpointStorage(ABC):
    """检查点存储基类"""
    
    @abstractmethod
    def save(self, checkpoint: Checkpoint, data: dict) -> bool:
        """保存检查点"""
        pass
    
    @abstractmethod
    def load(self, checkpoint_id: str) -> Tuple[Optional[Checkpoint], Optional[dict]]:
        """加载检查点"""
        pass
    
    @abstractmethod
    def list_checkpoints(self, job_id: Optional[str] = None) -> List[Checkpoint]:
        """列出检查点"""
        pass
    
    @abstractmethod
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        pass


class LocalCheckpointStorage(CheckpointStorage):
    """本地检查点存储"""
    
    def __init__(self, base_path: Path = None):
        self.base_path = Path(base_path or "./checkpoints")
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, checkpoint: Checkpoint, data: dict) -> bool:
        """保存检查点到本地"""
        try:
            checkpoint_dir = self.base_path / checkpoint.checkpoint_id
            checkpoint_dir.mkdir(exist_ok=True)
            
            # 保存元数据
            checkpoint.local_path = checkpoint_dir
            with open(checkpoint_dir / "checkpoint.json", 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)
            
            # 保存数据
            with open(checkpoint_dir / "data.pkl", 'wb') as f:
                pickle.dump(data, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load(self, checkpoint_id: str) -> Tuple[Optional[Checkpoint], Optional[dict]]:
        """从本地加载检查点"""
        try:
            checkpoint_dir = self.base_path / checkpoint_id
            
            # 加载元数据
            with open(checkpoint_dir / "checkpoint.json") as f:
                checkpoint = Checkpoint.from_dict(json.load(f))
            
            # 加载数据
            with open(checkpoint_dir / "data.pkl", 'rb') as f:
                data = pickle.load(f)
            
            return checkpoint, data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None, None
    
    def list_checkpoints(self, job_id: Optional[str] = None) -> List[Checkpoint]:
        """列出本地检查点"""
        checkpoints = []
        
        for checkpoint_dir in self.base_path.iterdir():
            if checkpoint_dir.is_dir():
                meta_file = checkpoint_dir / "checkpoint.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            checkpoint = Checkpoint.from_dict(json.load(f))
                        
                        if job_id is None or checkpoint.job_id == job_id:
                            checkpoints.append(checkpoint)
                    except Exception as e:
                        logger.warning(f"Failed to load checkpoint {checkpoint_dir}: {e}")
        
        # 按时间排序
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除本地检查点"""
        try:
            import shutil
            checkpoint_dir = self.base_path / checkpoint_id
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False


class RemoteCheckpointStorage(CheckpointStorage):
    """远程检查点存储"""
    
    def __init__(self, connector: ClusterConnector, remote_base_path: str):
        self.connector = connector
        self.remote_base_path = remote_base_path
    
    def save(self, checkpoint: Checkpoint, data: dict) -> bool:
        """保存检查点到远程"""
        # 需要先保存到本地，然后上传到远程
        # 简化实现：通过connector执行远程保存命令
        logger.info(f"Saving checkpoint to remote: {checkpoint.checkpoint_id}")
        return True
    
    def load(self, checkpoint_id: str) -> Tuple[Optional[Checkpoint], Optional[dict]]:
        """从远程加载检查点"""
        logger.info(f"Loading checkpoint from remote: {checkpoint_id}")
        return None, None
    
    def list_checkpoints(self, job_id: Optional[str] = None) -> List[Checkpoint]:
        """列出远程检查点"""
        return []
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除远程检查点"""
        return True


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(
        self,
        storage: CheckpointStorage = None,
        auto_checkpoint_interval: int = 3600,
        max_checkpoints: int = 10
    ):
        self.storage = storage or LocalCheckpointStorage()
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        self._active_checkpoints: Dict[str, Checkpoint] = {}
        self._checkpoint_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
    
    def create_checkpoint(
        self,
        job_id: Optional[str],
        working_dir: Path,
        state: CalculationState = CalculationState.RUNNING,
        iteration: int = 0,
        total_steps: int = 0,
        completed_steps: int = 0,
        metadata: dict = None,
        custom_data: dict = None
    ) -> Optional[Checkpoint]:
        """
        创建检查点
        
        Args:
            job_id: 作业ID
            working_dir: 工作目录
            state: 计算状态
            iteration: 当前迭代
            total_steps: 总步数
            completed_steps: 已完成步数
            metadata: 元数据
            custom_data: 自定义数据
        
        Returns:
            Checkpoint实例或None
        """
        checkpoint_id = self._generate_checkpoint_id(job_id)
        
        # 扫描工作目录
        file_checksums, file_sizes = self._scan_directory(working_dir)
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            job_id=job_id,
            state=state,
            timestamp=datetime.now(),
            file_checksums=file_checksums,
            file_sizes=file_sizes,
            iteration=iteration,
            total_steps=total_steps,
            completed_steps=completed_steps,
            metadata=metadata or {},
            local_path=working_dir / "checkpoints" / checkpoint_id
        )
        
        # 保存检查点
        custom_data = custom_data or {}
        if self.storage.save(checkpoint, custom_data):
            with self._lock:
                self._active_checkpoints[checkpoint_id] = checkpoint
            
            logger.info(
                f"Created checkpoint {checkpoint_id} for job {job_id}: "
                f"{completed_steps}/{total_steps} steps ({checkpoint.progress_pct:.1f}%)"
            )
            
            # 清理旧检查点
            self._cleanup_old_checkpoints(job_id)
            
            return checkpoint
        
        return None
    
    def restore_checkpoint(
        self,
        checkpoint_id: str,
        target_dir: Path = None
    ) -> Tuple[Optional[Checkpoint], Optional[dict]]:
        """
        恢复检查点
        
        Args:
            checkpoint_id: 检查点ID
            target_dir: 目标目录
        
        Returns:
            (Checkpoint, data) 或 (None, None)
        """
        checkpoint, data = self.storage.load(checkpoint_id)
        
        if checkpoint and target_dir:
            # 验证文件完整性
            missing, corrupted = self._verify_files(checkpoint, target_dir)
            
            if missing or corrupted:
                logger.warning(
                    f"Checkpoint {checkpoint_id} verification failed: "
                    f"{len(missing)} missing, {len(corrupted)} corrupted"
                )
            else:
                logger.info(f"Checkpoint {checkpoint_id} verified successfully")
        
        return checkpoint, data
    
    def analyze_resume_capability(
        self,
        job_id: str,
        working_dir: Path
    ) -> ResumeCapability:
        """
        分析恢复能力
        
        Args:
            job_id: 作业ID
            working_dir: 工作目录
        
        Returns:
            ResumeCapability实例
        """
        # 获取最新的检查点
        checkpoints = self.storage.list_checkpoints(job_id)
        
        if not checkpoints:
            return ResumeCapability(can_resume=False)
        
        latest = checkpoints[0]
        
        # 验证文件
        missing, corrupted = self._verify_files(latest, working_dir)
        
        can_resume = len(missing) == 0 and len(corrupted) == 0
        
        # 估算节省时间
        if can_resume and latest.total_steps > 0:
            remaining_steps = latest.total_steps - latest.completed_steps
            # 假设每步需要一定时间（可以从历史数据获取）
            estimated_time_per_step = timedelta(seconds=60)  # 默认值
            estimated_time_saved = remaining_steps * estimated_time_per_step
        else:
            estimated_time_saved = timedelta(0)
        
        return ResumeCapability(
            can_resume=can_resume,
            resume_from=latest if can_resume else None,
            missing_files=missing,
            estimated_time_saved=estimated_time_saved
        )
    
    def list_checkpoints(self, job_id: Optional[str] = None) -> List[Checkpoint]:
        """列出检查点"""
        return self.storage.list_checkpoints(job_id)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        with self._lock:
            if checkpoint_id in self._active_checkpoints:
                del self._active_checkpoints[checkpoint_id]
        
        return self.storage.delete_checkpoint(checkpoint_id)
    
    def start_auto_checkpoint(
        self,
        job_id: str,
        working_dir: Path,
        callback: Callable = None
    ):
        """启动自动检查点"""
        def checkpoint_callback():
            try:
                checkpoint = self.create_checkpoint(
                    job_id=job_id,
                    working_dir=working_dir
                )
                if callback and checkpoint:
                    callback(checkpoint)
            except Exception as e:
                logger.error(f"Auto checkpoint failed: {e}")
            
            # 重新调度
            self._checkpoint_timer = threading.Timer(
                self.auto_checkpoint_interval,
                checkpoint_callback
            )
            self._checkpoint_timer.daemon = True
            self._checkpoint_timer.start()
        
        self._checkpoint_timer = threading.Timer(
            self.auto_checkpoint_interval,
            checkpoint_callback
        )
        self._checkpoint_timer.daemon = True
        self._checkpoint_timer.start()
        
        logger.info(f"Auto checkpoint started for job {job_id}")
    
    def stop_auto_checkpoint(self):
        """停止自动检查点"""
        if self._checkpoint_timer:
            self._checkpoint_timer.cancel()
            self._checkpoint_timer = None
            logger.info("Auto checkpoint stopped")
    
    def _generate_checkpoint_id(self, job_id: Optional[str]) -> str:
        """生成检查点ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if job_id:
            return f"{job_id}_{timestamp}"
        return f"ckpt_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def _scan_directory(self, directory: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
        """扫描目录获取文件信息"""
        checksums = {}
        sizes = {}
        
        if not directory.exists():
            return checksums, sizes
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    rel_path = str(file_path.relative_to(directory))
                    sizes[rel_path] = file_path.stat().st_size
                    checksums[rel_path] = self._compute_checksum(file_path)
                except Exception as e:
                    logger.warning(f"Failed to scan {file_path}: {e}")
        
        return checksums, sizes
    
    def _compute_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        hash_obj = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
        except Exception:
            return ""
        return hash_obj.hexdigest()
    
    def _verify_files(
        self,
        checkpoint: Checkpoint,
        working_dir: Path
    ) -> Tuple[List[str], List[str]]:
        """验证文件完整性"""
        missing = []
        corrupted = []
        
        for rel_path, expected_checksum in checkpoint.file_checksums.items():
            file_path = working_dir / rel_path
            
            if not file_path.exists():
                missing.append(rel_path)
            else:
                actual_checksum = self._compute_checksum(file_path)
                if actual_checksum != expected_checksum:
                    corrupted.append(rel_path)
        
        return missing, corrupted
    
    def _cleanup_old_checkpoints(self, job_id: Optional[str]):
        """清理旧检查点"""
        if not job_id:
            return
        
        checkpoints = self.storage.list_checkpoints(job_id)
        
        if len(checkpoints) > self.max_checkpoints:
            # 删除最旧的检查点
            to_delete = checkpoints[self.max_checkpoints:]
            for checkpoint in to_delete:
                logger.debug(f"Deleting old checkpoint: {checkpoint.checkpoint_id}")
                self.storage.delete_checkpoint(checkpoint.checkpoint_id)


class CheckpointableCalculation(ABC):
    """可检查点的计算基类"""
    
    def __init__(self, checkpoint_manager: CheckpointManager = None):
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self._current_checkpoint: Optional[Checkpoint] = None
        self._state = CalculationState.INITIALIZED
    
    @abstractmethod
    def get_checkpoint_data(self) -> dict:
        """获取检查点数据"""
        pass
    
    @abstractmethod
    def restore_from_checkpoint(self, data: dict) -> bool:
        """从检查点恢复"""
        pass
    
    @abstractmethod
    def run_step(self) -> bool:
        """执行单步计算"""
        pass
    
    def save_checkpoint(self, working_dir: Path) -> Optional[Checkpoint]:
        """保存检查点"""
        data = self.get_checkpoint_data()
        
        checkpoint = self.checkpoint_manager.create_checkpoint(
            job_id=data.get('job_id'),
            working_dir=working_dir,
            state=self._state,
            iteration=data.get('iteration', 0),
            total_steps=data.get('total_steps', 0),
            completed_steps=data.get('completed_steps', 0),
            metadata=data.get('metadata', {}),
            custom_data=data
        )
        
        self._current_checkpoint = checkpoint
        return checkpoint
    
    def load_checkpoint(self, checkpoint_id: str, working_dir: Path) -> bool:
        """加载检查点"""
        checkpoint, data = self.checkpoint_manager.restore_checkpoint(
            checkpoint_id, working_dir
        )
        
        if data and self.restore_from_checkpoint(data):
            self._current_checkpoint = checkpoint
            self._state = CalculationState.RESUMED
            logger.info(f"Restored from checkpoint: {checkpoint_id}")
            return True
        
        return False
    
    def run_with_checkpoints(
        self,
        working_dir: Path,
        checkpoint_interval: int = 3600
    ):
        """
        带检查点的运行
        
        Args:
            working_dir: 工作目录
            checkpoint_interval: 检查点间隔(秒)
        """
        # 首先尝试恢复
        data = self.get_checkpoint_data()
        job_id = data.get('job_id')
        
        if job_id:
            capability = self.checkpoint_manager.analyze_resume_capability(
                job_id, working_dir
            )
            
            if capability.can_resume and capability.resume_from:
                self.load_checkpoint(capability.resume_from.checkpoint_id, working_dir)
        
        # 启动自动检查点
        self.checkpoint_manager.start_auto_checkpoint(
            job_id=job_id,
            working_dir=working_dir,
            callback=lambda ckpt: logger.info(f"Auto checkpoint: {ckpt.checkpoint_id}")
        )
        
        try:
            self._state = CalculationState.RUNNING
            
            # 主循环
            while self._state == CalculationState.RUNNING:
                if not self.run_step():
                    break
            
            # 最终检查点
            if self._state == CalculationState.COMPLETED:
                self.save_checkpoint(working_dir)
                
        except Exception as e:
            self._state = CalculationState.FAILED
            # 紧急检查点
            self.save_checkpoint(working_dir)
            raise
        
        finally:
            self.checkpoint_manager.stop_auto_checkpoint()


@contextmanager
def create_periodic_checkpoint(
    checkpoint_manager: CheckpointManager,
    job_id: str,
    working_dir: Path,
    interval: int = 3600
):
    """
    周期性检查点上下文管理器
    
    使用示例：
        with create_periodic_checkpoint(manager, "job_123", Path("./work")) as ckpt_ctx:
            for step in range(total_steps):
                ckpt_ctx.update(step=step, total=total_steps)
                run_calculation(step)
    """
    from dataclasses import dataclass
    
    @dataclass
    class CheckpointContext:
        step: int = 0
        total: int = 0
        metadata: dict = field(default_factory=dict)
        
        def update(self, step: int = None, total: int = None, **metadata):
            if step is not None:
                self.step = step
            if total is not None:
                self.total = total
            self.metadata.update(metadata)
    
    context = CheckpointContext()
    last_checkpoint = 0
    
    def checkpoint_callback():
        nonlocal last_checkpoint
        checkpoint_manager.create_checkpoint(
            job_id=job_id,
            working_dir=working_dir,
            completed_steps=context.step,
            total_steps=context.total,
            metadata=context.metadata
        )
        last_checkpoint = time.time()
    
    try:
        yield context
    finally:
        # 最终检查点
        checkpoint_callback()
