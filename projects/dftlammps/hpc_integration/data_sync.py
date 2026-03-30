#!/usr/bin/env python3
"""
data_sync.py
============
数据自动同步模块

支持协议：
- S3 (AWS S3兼容)
- MinIO (本地对象存储)
- SFTP (SSH文件传输)
- Rsync (增量同步)

功能：
- 双向同步
- 增量同步
- 自动重试
- 文件监控
- 压缩传输
"""

import os
import re
import io
import json
import time
import hashlib
import logging
import threading
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import fnmatch

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False


class SyncDirection(Enum):
    """同步方向"""
    UPLOAD = "upload"           # 本地 -> 远程
    DOWNLOAD = "download"       # 远程 -> 本地
    BIDIRECTIONAL = "bidirectional"  # 双向同步


@dataclass
class SyncTask:
    """同步任务"""
    name: str
    local_path: Path
    remote_path: str
    direction: SyncDirection
    
    # 过滤
    include_patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = field(default_factory=list)
    
    # 选项
    compress: bool = True
    delete_extraneous: bool = False
    verify_checksum: bool = True
    preserve_permissions: bool = True
    
    # 调度
    auto_sync: bool = False
    sync_interval: int = 300  # 秒
    
    # 状态
    last_sync: Optional[datetime] = None
    last_error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "local_path": str(self.local_path),
            "remote_path": self.remote_path,
            "direction": self.direction.value,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "compress": self.compress,
            "delete_extraneous": self.delete_extraneous,
            "verify_checksum": self.verify_checksum,
            "auto_sync": self.auto_sync,
            "sync_interval": self.sync_interval,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "last_error": self.last_error
        }


@dataclass
class SyncResult:
    """同步结果"""
    success: bool
    files_transferred: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    details: List[Dict] = field(default_factory=list)


@dataclass
class AutoSyncPolicy:
    """自动同步策略"""
    enabled: bool = True
    on_job_submit: bool = True      # 作业提交前同步输入
    on_job_complete: bool = True    # 作业完成后同步输出
    on_checkpoint: bool = True      # 检查点时同步
    sync_interval: int = 600        # 定期同步间隔(秒)
    max_sync_attempts: int = 3
    sync_timeout: int = 3600
    
    # 文件过滤
    input_extensions: List[str] = field(default_factory=lambda: [
        ".in", ".inp", ".json", ".xyz", ".POSCAR", ".potcar",
        ".incar", ".kpoints", ".lammps", ".data"
    ])
    output_extensions: List[str] = field(default_factory=lambda: [
        ".out", ".log", ".xml", ".h5", ".pb", ".ckpt",
        ".dump", ".xyz", ".csv", ".txt", ".err"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.tmp", "*.bak", "*~", ".*", "core.*"
    ])


class FileWatcher:
    """文件变更监控器"""
    
    def __init__(self, check_interval: int = 5):
        self.check_interval = check_interval
        self._watches: Dict[str, dict] = {}
        self._stop_event = threading.Event()
        self._watch_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()
    
    def add_watch(
        self,
        path: Path,
        patterns: List[str] = None,
        callback: Callable = None
    ):
        """添加文件监控"""
        path_str = str(path.resolve())
        
        with self._lock:
            self._watches[path_str] = {
                "path": path,
                "patterns": patterns or ["*"],
                "callback": callback,
                "snapshots": self._take_snapshot(path, patterns or ["*"]),
                "last_check": datetime.now()
            }
    
    def remove_watch(self, path: Path):
        """移除文件监控"""
        path_str = str(path.resolve())
        with self._lock:
            if path_str in self._watches:
                del self._watches[path_str]
    
    def start(self):
        """启动监控"""
        if self._watch_thread and self._watch_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        logger.info("File watcher started")
    
    def stop(self):
        """停止监控"""
        self._stop_event.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=5)
        logger.info("File watcher stopped")
    
    def _watch_loop(self):
        """监控循环"""
        while not self._stop_event.is_set():
            self._check_all_watches()
            self._stop_event.wait(self.check_interval)
    
    def _check_all_watches(self):
        """检查所有监控"""
        with self._lock:
            watches = list(self._watches.items())
        
        for path_str, watch in watches:
            try:
                new_snapshot = self._take_snapshot(watch["path"], watch["patterns"])
                changes = self._detect_changes(watch["snapshots"], new_snapshot)
                
                if changes:
                    watch["snapshots"] = new_snapshot
                    watch["last_check"] = datetime.now()
                    
                    # 调用回调
                    if watch["callback"]:
                        try:
                            watch["callback"](watch["path"], changes)
                        except Exception as e:
                            logger.error(f"Watch callback error: {e}")
                    
                    # 全局回调
                    for callback in self._callbacks:
                        try:
                            callback(watch["path"], changes)
                        except Exception as e:
                            logger.error(f"Global callback error: {e}")
                            
            except Exception as e:
                logger.error(f"Error checking watch {path_str}: {e}")
    
    def _take_snapshot(self, path: Path, patterns: List[str]) -> Dict[str, dict]:
        """拍摄文件快照"""
        snapshot = {}
        
        if not path.exists():
            return snapshot
        
        if path.is_file():
            files = [path]
        else:
            files = []
            for pattern in patterns:
                files.extend(path.rglob(pattern))
        
        for file_path in files:
            if file_path.is_file():
                stat = file_path.stat()
                snapshot[str(file_path)] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "checksum": self._compute_checksum(file_path)
                }
        
        return snapshot
    
    def _detect_changes(
        self,
        old_snapshot: Dict,
        new_snapshot: Dict
    ) -> List[Dict]:
        """检测文件变更"""
        changes = []
        
        # 新增或修改的文件
        for path, info in new_snapshot.items():
            if path not in old_snapshot:
                changes.append({"type": "created", "path": path, "info": info})
            elif old_snapshot[path]["checksum"] != info["checksum"]:
                changes.append({"type": "modified", "path": path, "info": info})
        
        # 删除的文件
        for path in old_snapshot:
            if path not in new_snapshot:
                changes.append({"type": "deleted", "path": path})
        
        return changes
    
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
    
    def register_callback(self, callback: Callable):
        """注册全局变更回调"""
        self._callbacks.append(callback)


class DataSyncManager:
    """数据同步管理器"""
    
    def __init__(
        self,
        storage_backend,
        policy: AutoSyncPolicy = None,
        max_workers: int = 4
    ):
        self.backend = storage_backend
        self.policy = policy or AutoSyncPolicy()
        self.max_workers = max_workers
        
        self._tasks: Dict[str, SyncTask] = {}
        self._watcher = FileWatcher()
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # 注册文件变更监听
        self._watcher.register_callback(self._on_file_change)
    
    def add_task(self, task: SyncTask):
        """添加同步任务"""
        with self._lock:
            self._tasks[task.name] = task
        
        # 如果需要自动同步，添加文件监控
        if task.auto_sync:
            self._watcher.add_watch(
                task.local_path,
                task.include_patterns,
                lambda path, changes: self._trigger_sync(task.name)
            )
    
    def remove_task(self, task_name: str):
        """移除同步任务"""
        with self._lock:
            if task_name in self._tasks:
                task = self._tasks.pop(task_name)
                self._watcher.remove_watch(task.local_path)
    
    def start_auto_sync(self):
        """启动自动同步"""
        self._watcher.start()
        
        self._stop_event.clear()
        self._sync_thread = threading.Thread(target=self._auto_sync_loop, daemon=True)
        self._sync_thread.start()
        
        logger.info("Auto sync started")
    
    def stop_auto_sync(self):
        """停止自动同步"""
        self._watcher.stop()
        self._stop_event.set()
        
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        
        logger.info("Auto sync stopped")
    
    def _auto_sync_loop(self):
        """自动同步循环"""
        while not self._stop_event.is_set():
            with self._lock:
                tasks = list(self._tasks.values())
            
            for task in tasks:
                if task.auto_sync and task.last_sync:
                    elapsed = (datetime.now() - task.last_sync).total_seconds()
                    if elapsed >= task.sync_interval:
                        self.sync_task(task.name)
            
            self._stop_event.wait(60)  # 每分钟检查一次
    
    def _on_file_change(self, path: Path, changes: List[Dict]):
        """文件变更回调"""
        # 查找相关的同步任务
        with self._lock:
            for task in self._tasks.values():
                if task.auto_sync and str(path).startswith(str(task.local_path)):
                    logger.debug(f"File change detected for task {task.name}")
                    self._trigger_sync(task.name)
    
    def _trigger_sync(self, task_name: str):
        """触发同步"""
        # 使用线程池执行同步
        thread = threading.Thread(
            target=self.sync_task,
            args=(task_name,),
            daemon=True
        )
        thread.start()
    
    def sync_task(self, task_name: str) -> SyncResult:
        """
        执行同步任务
        
        Args:
            task_name: 任务名称
        
        Returns:
            SyncResult实例
        """
        with self._lock:
            task = self._tasks.get(task_name)
        
        if not task:
            return SyncResult(success=False, errors=[f"Task {task_name} not found"])
        
        start_time = time.time()
        
        try:
            if task.direction == SyncDirection.UPLOAD:
                result = self._sync_upload(task)
            elif task.direction == SyncDirection.DOWNLOAD:
                result = self._sync_download(task)
            else:
                # 双向同步：先下载后上传
                dl_result = self._sync_download(task)
                ul_result = self._sync_upload(task)
                result = self._merge_results(dl_result, ul_result)
            
            task.last_sync = datetime.now()
            task.last_error = None
            
        except Exception as e:
            task.last_error = str(e)
            result = SyncResult(
                success=False,
                errors=[str(e)],
                duration_seconds=time.time() - start_time
            )
        
        result.duration_seconds = time.time() - start_time
        return result
    
    def _sync_upload(self, task: SyncTask) -> SyncResult:
        """上传同步"""
        return self.backend.upload_directory(
            local_path=task.local_path,
            remote_path=task.remote_path,
            include_patterns=task.include_patterns,
            exclude_patterns=task.exclude_patterns,
            delete_extraneous=task.delete_extraneous
        )
    
    def _sync_download(self, task: SyncTask) -> SyncResult:
        """下载同步"""
        return self.backend.download_directory(
            remote_path=task.remote_path,
            local_path=task.local_path,
            include_patterns=task.include_patterns,
            exclude_patterns=task.exclude_patterns,
            delete_extraneous=task.delete_extraneous
        )
    
    def _merge_results(self, r1: SyncResult, r2: SyncResult) -> SyncResult:
        """合并同步结果"""
        return SyncResult(
            success=r1.success and r2.success,
            files_transferred=r1.files_transferred + r2.files_transferred,
            bytes_transferred=r1.bytes_transferred + r2.bytes_transferred,
            duration_seconds=r1.duration_seconds + r2.duration_seconds,
            errors=r1.errors + r2.errors,
            details=r1.details + r2.details
        )
    
    def sync_before_job(
        self,
        local_work_dir: Path,
        remote_work_dir: str
    ) -> SyncResult:
        """
        作业提交前同步输入文件
        
        Args:
            local_work_dir: 本地工作目录
            remote_work_dir: 远程工作目录
        
        Returns:
            SyncResult实例
        """
        if not self.policy.on_job_submit:
            return SyncResult(success=True)
        
        task = SyncTask(
            name=f"pre_job_{local_work_dir.name}",
            local_path=local_work_dir,
            remote_path=remote_work_dir,
            direction=SyncDirection.UPLOAD,
            include_patterns=[f"*{ext}" for ext in self.policy.input_extensions],
            exclude_patterns=self.policy.exclude_patterns,
            compress=self.policy.enabled
        )
        
        return self._sync_upload(task)
    
    def sync_after_job(
        self,
        local_work_dir: Path,
        remote_work_dir: str
    ) -> SyncResult:
        """
        作业完成后同步输出文件
        
        Args:
            local_work_dir: 本地工作目录
            remote_work_dir: 远程工作目录
        
        Returns:
            SyncResult实例
        """
        if not self.policy.on_job_complete:
            return SyncResult(success=True)
        
        task = SyncTask(
            name=f"post_job_{local_work_dir.name}",
            local_path=local_work_dir,
            remote_path=remote_work_dir,
            direction=SyncDirection.DOWNLOAD,
            include_patterns=[f"*{ext}" for ext in self.policy.output_extensions],
            exclude_patterns=self.policy.exclude_patterns,
            compress=self.policy.enabled
        )
        
        return self._sync_download(task)
    
    def sync_checkpoint(self, checkpoint_path: Path, remote_path: str) -> SyncResult:
        """
        同步检查点
        
        Args:
            checkpoint_path: 检查点路径
            remote_path: 远程路径
        
        Returns:
            SyncResult实例
        """
        if not self.policy.on_checkpoint:
            return SyncResult(success=True)
        
        return self.backend.upload_file(checkpoint_path, remote_path)
    
    def list_tasks(self) -> List[SyncTask]:
        """列出所有任务"""
        with self._lock:
            return list(self._tasks.values())
    
    def get_task_status(self, task_name: str) -> Optional[Dict]:
        """获取任务状态"""
        with self._lock:
            task = self._tasks.get(task_name)
        
        if task:
            return task.to_dict()
        return None
    
    def wait_for_sync(self, task_name: str, timeout: int = 300) -> bool:
        """
        等待同步完成
        
        Args:
            task_name: 任务名称
            timeout: 超时时间(秒)
        
        Returns:
            是否成功完成
        """
        start = time.time()
        
        while time.time() - start < timeout:
            with self._lock:
                task = self._tasks.get(task_name)
            
            if task and task.last_sync:
                return True
            
            time.sleep(1)
        
        return False
