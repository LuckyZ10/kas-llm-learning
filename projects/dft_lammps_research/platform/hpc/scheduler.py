#!/usr/bin/env python3
"""
hpc_scheduler.py
================
HPC作业调度接口 - 支持Slurm/PBS/LSF

功能：
1. 统一的作业调度接口
2. 支持Slurm、PBS、LSF三大主流调度系统
3. 自动检测调度系统类型
4. 作业依赖管理和工作流编排
5. 资源配额查询和预估

作者: HPC & Performance Optimization Expert
日期: 2026-03-09
"""

import os
import re
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import threading
import queue

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    """支持的调度系统类型"""
    SLURM = "slurm"
    PBS = "pbs"
    LSF = "lsf"
    LOCAL = "local"  # 本地执行模式


class JobStatus(Enum):
    """作业状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ResourceRequest:
    """资源请求配置"""
    # 计算资源
    num_nodes: int = 1
    num_cores_per_node: int = 1
    num_gpus: int = 0
    gpu_type: Optional[str] = None
    memory_gb: Optional[float] = None
    memory_per_core_gb: Optional[float] = None
    
    # 时间限制
    walltime_hours: float = 24.0
    
    # 队列/分区
    queue: Optional[str] = None
    partition: Optional[str] = None  # Slurm特有
    
    # 特殊资源
    exclusive: bool = False
    constraint: Optional[str] = None  # 节点约束 (如 CPU 类型)
    reservation: Optional[str] = None
    
    # 软件环境
    modules: List[str] = field(default_factory=list)
    conda_env: Optional[str] = None
    singularity_image: Optional[str] = None
    
    # 其他选项
    priority: Optional[int] = None
    account: Optional[str] = None
    
    def to_slurm(self) -> str:
        """转换为Slurm格式"""
        lines = []
        
        # 节点和核心
        if self.num_gpus > 0:
            lines.append(f"#SBATCH --nodes={self.num_nodes}")
            lines.append(f"#SBATCH --ntasks-per-node={self.num_cores_per_node}")
            lines.append(f"#SBATCH --gpus={self.num_gpus}")
            if self.gpu_type:
                lines.append(f"#SBATCH --gres=gpu:{self.gpu_type}:{self.num_gpus}")
        else:
            lines.append(f"#SBATCH --nodes={self.num_nodes}")
            lines.append(f"#SBATCH --ntasks-per-node={self.num_cores_per_node}")
        
        # 内存
        if self.memory_gb:
            lines.append(f"#SBATCH --mem={int(self.memory_gb * 1024)}")
        elif self.memory_per_core_gb:
            lines.append(f"#SBATCH --mem-per-cpu={int(self.memory_per_core_gb * 1024)}")
        
        # 时间
        hours = int(self.walltime_hours)
        minutes = int((self.walltime_hours - hours) * 60)
        lines.append(f"#SBATCH --time={hours:02d}:{minutes:02d}:00")
        
        # 队列/分区
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        
        # 其他
        if self.exclusive:
            lines.append("#SBATCH --exclusive")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.reservation:
            lines.append(f"#SBATCH --reservation={self.reservation}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.priority:
            lines.append(f"#SBATCH --priority={self.priority}")
        
        return "\n".join(lines)
    
    def to_pbs(self) -> str:
        """转换为PBS格式"""
        lines = []
        
        # 节点和核心
        ncpus = self.num_nodes * self.num_cores_per_node
        lines.append(f"#PBS -l nodes={self.num_nodes}:ppn={self.num_cores_per_node}")
        
        # GPU
        if self.num_gpus > 0:
            lines.append(f"#PBS -l ngpus={self.num_gpus}")
        
        # 内存
        if self.memory_gb:
            lines.append(f"#PBS -l mem={int(self.memory_gb)}gb")
        
        # 时间
        hours = int(self.walltime_hours)
        minutes = int((self.walltime_hours - hours) * 60)
        lines.append(f"#PBS -l walltime={hours:02d}:{minutes:02d}:00")
        
        # 队列
        if self.queue:
            lines.append(f"#PBS -q {self.queue}")
        
        return "\n".join(lines)
    
    def to_lsf(self) -> str:
        """转换为LSF格式"""
        lines = []
        
        # 处理器
        nprocs = self.num_nodes * self.num_cores_per_node
        lines.append(f"#BSUB -n {nprocs}")
        lines.append(f"#BSUB -R 'span[ptile={self.num_cores_per_node}]'")
        
        # GPU
        if self.num_gpus > 0:
            lines.append(f"#BSUB -gpu 'num={self.num_gpus}:mode=exclusive_process'")
        
        # 内存
        if self.memory_gb:
            lines.append(f"#BSUB -M {int(self.memory_gb * 1024)}")
            lines.append(f"#BSUB -R 'rusage[mem={int(self.memory_gb * 1024)}]'")
        
        # 时间
        hours = int(self.walltime_hours)
        minutes = int((self.walltime_hours - hours) * 60)
        lines.append(f"#BSUB -W {hours}:{minutes:02d}")
        
        # 队列
        if self.queue:
            lines.append(f"#BSUB -q {self.queue}")
        
        return "\n".join(lines)


@dataclass
class JobSpec:
    """作业规格定义"""
    # 基本信息
    name: str
    working_dir: Path
    
    # 执行命令
    commands: List[str] = field(default_factory=list)
    executable: Optional[str] = None
    arguments: List[str] = field(default_factory=list)
    
    # 资源请求
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    
    # 环境设置
    env_vars: Dict[str, str] = field(default_factory=dict)
    pre_commands: List[str] = field(default_factory=list)
    post_commands: List[str] = field(default_factory=list)
    
    # 输入输出
    stdin_file: Optional[str] = None
    stdout_file: Optional[str] = "job.out"
    stderr_file: Optional[str] = "job.err"
    
    # 依赖
    dependencies: List[str] = field(default_factory=list)  # 依赖的作业ID列表
    dependency_type: str = "afterok"  # afterok, afterany, afternotok
    
    # 通知
    email: Optional[str] = None
    email_events: List[str] = field(default_factory=lambda: ["end", "fail"])
    
    # 数组作业
    array_range: Optional[Tuple[int, int, int]] = None  # (start, end, step)
    
    def __post_init__(self):
        self.working_dir = Path(self.working_dir)


@dataclass
class JobInfo:
    """作业信息"""
    job_id: str
    name: str
    status: JobStatus
    queue: Optional[str] = None
    submit_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    elapsed_time: Optional[timedelta] = None
    remaining_time: Optional[timedelta] = None
    nodes: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    reason: Optional[str] = None


class BaseScheduler(ABC):
    """调度器基类"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._check_availability()
    
    @abstractmethod
    def _check_availability(self) -> bool:
        """检查调度器是否可用"""
        pass
    
    @abstractmethod
    def submit(self, spec: JobSpec) -> str:
        """提交作业，返回作业ID"""
        pass
    
    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """取消作业"""
        pass
    
    @abstractmethod
    def query(self, job_id: Optional[str] = None) -> List[JobInfo]:
        """查询作业状态"""
        pass
    
    @abstractmethod
    def get_queues(self) -> List[Dict]:
        """获取可用队列信息"""
        pass
    
    @abstractmethod
    def estimate_wait_time(self, resources: ResourceRequest, queue: Optional[str] = None) -> Optional[timedelta]:
        """预估等待时间"""
        pass
    
    def wait_for_job(self, job_id: str, poll_interval: int = 30, timeout: Optional[float] = None) -> JobStatus:
        """等待作业完成"""
        start_time = time.time()
        
        while True:
            jobs = self.query(job_id)
            if not jobs:
                logger.warning(f"Job {job_id} not found")
                return JobStatus.UNKNOWN
            
            job = jobs[0]
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT]:
                logger.info(f"Job {job_id} finished with status: {job.status.value}")
                return job.status
            
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for job {job_id}")
                return JobStatus.UNKNOWN
            
            time.sleep(poll_interval)
    
    def generate_script(self, spec: JobSpec) -> str:
        """生成作业脚本"""
        raise NotImplementedError


class SlurmScheduler(BaseScheduler):
    """Slurm调度器"""
    
    def _check_availability(self) -> bool:
        try:
            result = subprocess.run(
                ["sbatch", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def generate_script(self, spec: JobSpec) -> str:
        """生成Slurm作业脚本"""
        lines = ["#!/bin/bash"]
        lines.append("#SBATCH --job-name=" + spec.name)
        
        # 资源请求
        lines.append(spec.resources.to_slurm())
        
        # 输入输出
        if spec.stdout_file:
            stdout = spec.working_dir / spec.stdout_file
            lines.append(f"#SBATCH --output={stdout}")
        if spec.stderr_file:
            stderr = spec.working_dir / spec.stderr_file
            lines.append(f"#SBATCH --error={stderr}")
        
        # 工作目录
        lines.append(f"#SBATCH --chdir={spec.working_dir}")
        
        # 依赖
        if spec.dependencies:
            dep_str = f"#SBATCH --dependency={spec.dependency_type}:{':'.join(spec.dependencies)}"
            lines.append(dep_str)
        
        # 数组作业
        if spec.array_range:
            start, end, step = spec.array_range
            lines.append(f"#SBATCH --array={start}-{end}:{step}")
        
        # 邮件通知
        if spec.email:
            lines.append(f"#SBATCH --mail-user={spec.email}")
            events = spec.email_events
            lines.append(f"#SBATCH --mail-type={','.join(events)}")
        
        lines.append("")
        
        # 环境变量
        for key, value in spec.env_vars.items():
            lines.append(f"export {key}='{value}'")
        
        if spec.env_vars:
            lines.append("")
        
        # 加载模块
        for module in spec.resources.modules:
            lines.append(f"module load {module}")
        
        if spec.resources.modules:
            lines.append("")
        
        # Conda环境
        if spec.resources.conda_env:
            lines.append(f"source activate {spec.resources.conda_env}")
            lines.append("")
        
        # 预执行命令
        for cmd in spec.pre_commands:
            lines.append(cmd)
        
        if spec.pre_commands:
            lines.append("")
        
        # 主命令
        if spec.executable:
            cmd = spec.executable + " " + " ".join(spec.arguments)
            lines.append(cmd)
        
        for cmd in spec.commands:
            lines.append(cmd)
        
        # 后执行命令
        if spec.post_commands:
            lines.append("")
            for cmd in spec.post_commands:
                lines.append(cmd)
        
        return "\n".join(lines)
    
    def submit(self, spec: JobSpec) -> str:
        """提交Slurm作业"""
        script = self.generate_script(spec)
        script_path = spec.working_dir / f"{spec.name}.slurm"
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(spec.working_dir)
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to submit job: {result.stderr}")
        
        # 解析作业ID
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from: {result.stdout}")
        
        job_id = match.group(1)
        logger.info(f"Submitted Slurm job {job_id}: {spec.name}")
        return job_id
    
    def cancel(self, job_id: str) -> bool:
        """取消Slurm作业"""
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    
    def query(self, job_id: Optional[str] = None) -> List[JobInfo]:
        """查询Slurm作业状态"""
        cmd = ["squeue", "--format=%i|%j|%T|%M|%L|%D|%R", "--noheader"]
        
        if job_id:
            cmd.extend(["-j", job_id])
        else:
            # 只查询当前用户的作业
            cmd.extend(["-u", os.environ.get("USER", "")])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # 检查作业是否已完成
            if job_id:
                return self._query_completed_job(job_id)
            return []
        
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) < 7:
                continue
            
            job_id_str, name, state, elapsed, remaining, nodes, reason = parts[:7]
            
            status = self._parse_slurm_status(state)
            
            job_info = JobInfo(
                job_id=job_id_str.strip(),
                name=name.strip(),
                status=status,
                elapsed_time=self._parse_time(elapsed),
                remaining_time=self._parse_time(remaining) if remaining != "NOT_SET" else None,
                reason=reason.strip() if reason not in ["None", ""] else None
            )
            jobs.append(job_info)
        
        return jobs
    
    def _query_completed_job(self, job_id: str) -> List[JobInfo]:
        """查询已完成的作业"""
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,JobName,State,ExitCode,Elapsed", "--noheader", "-P"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return []
        
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) < 5:
                continue
            
            job_id_str, name, state, exit_code, elapsed = parts[:5]
            
            # 只获取主作业，不是步骤
            if "." in job_id_str:
                continue
            
            status = self._parse_slurm_status(state)
            
            # 解析退出码
            exit_val = 0
            if ":" in exit_code:
                exit_val = int(exit_code.split(":")[0])
            
            if status == JobStatus.COMPLETED and exit_val != 0:
                status = JobStatus.FAILED
            
            job_info = JobInfo(
                job_id=job_id_str.strip(),
                name=name.strip(),
                status=status,
                exit_code=exit_val,
                elapsed_time=self._parse_time(elapsed)
            )
            jobs.append(job_info)
        
        return jobs
    
    def _parse_slurm_status(self, state: str) -> JobStatus:
        """解析Slurm状态"""
        state_lower = state.lower()
        
        if state_lower in ["running", "r"]:
            return JobStatus.RUNNING
        elif state_lower in ["pending", "pd"]:
            return JobStatus.PENDING
        elif state_lower in ["completed", "cd"]:
            return JobStatus.COMPLETED
        elif state_lower in ["failed", "f"]:
            return JobStatus.FAILED
        elif state_lower in ["cancelled", "ca"]:
            return JobStatus.CANCELLED
        elif state_lower in ["timeout", "to"]:
            return JobStatus.TIMEOUT
        else:
            return JobStatus.UNKNOWN
    
    def _parse_time(self, time_str: str) -> Optional[timedelta]:
        """解析时间字符串"""
        if not time_str or time_str in ["NOT_SET", "0:00"]:
            return None
        
        try:
            parts = time_str.split(":")
            if len(parts) == 2:
                # MM:SS
                minutes = int(parts[0])
                seconds = int(parts[1])
                return timedelta(minutes=minutes, seconds=seconds)
            elif len(parts) == 3:
                # HH:MM:SS 或 D-HH:MM:SS
                if "-" in parts[0]:
                    days, hours = parts[0].split("-")
                    return timedelta(days=int(days), hours=int(hours), 
                                   minutes=int(parts[1]), seconds=int(parts[2]))
                else:
                    return timedelta(hours=int(parts[0]), minutes=int(parts[1]), 
                                   seconds=int(parts[2]))
        except (ValueError, IndexError):
            pass
        
        return None
    
    def get_queues(self) -> List[Dict]:
        """获取Slurm分区信息"""
        result = subprocess.run(
            ["sinfo", "--format=%P|%a|%l|%D|%t|%N", "--noheader"],
            capture_output=True,
            text=True
        )
        
        queues = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split("|")
            if len(parts) < 6:
                continue
            
            partition, avail, timelimit, nodes, state, nodelist = parts[:6]
            
            queue_info = {
                "name": partition.rstrip("*"),  # 移除默认标记
                "available": avail == "up",
                "time_limit": timelimit,
                "total_nodes": nodes,
                "state": state,
                "is_default": partition.endswith("*")
            }
            queues.append(queue_info)
        
        return queues
    
    def estimate_wait_time(self, resources: ResourceRequest, queue: Optional[str] = None) -> Optional[timedelta]:
        """使用squeue预估等待时间"""
        # Slurm 没有内置的等待时间预估工具，这里提供一个基于队列状态的经验估计
        try:
            cmd = ["squeue", "--state=pending", "--noheader"]
            if queue:
                cmd.extend(["-p", queue])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                pending_count = len([l for l in result.stdout.strip().split("\n") if l])
                # 粗略估计：每个待处理作业增加5分钟等待
                estimated_minutes = min(pending_count * 5, 1440)  # 最多24小时
                return timedelta(minutes=estimated_minutes)
        except Exception:
            pass
        
        return None


class PBSScheduler(BaseScheduler):
    """PBS/Torque调度器"""
    
    def _check_availability(self) -> bool:
        try:
            result = subprocess.run(
                ["qsub", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 or "version" in result.stderr.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def generate_script(self, spec: JobSpec) -> str:
        """生成PBS作业脚本"""
        lines = ["#!/bin/bash"]
        lines.append("#PBS -N " + spec.name)
        
        # 资源请求
        lines.append(spec.resources.to_pbs())
        
        # 输入输出
        if spec.stdout_file:
            lines.append(f"#PBS -o {spec.working_dir / spec.stdout_file}")
        if spec.stderr_file:
            lines.append(f"#PBS -e {spec.working_dir / spec.stderr_file}")
        
        # 邮件通知
        if spec.email:
            lines.append(f"#PBS -M {spec.email}")
            events = []
            if "begin" in spec.email_events:
                events.append("b")
            if "end" in spec.email_events:
                events.append("e")
            if "fail" in spec.email_events:
                events.append("a")
            lines.append(f"#PBS -m {''.join(events)}")
        
        # 依赖
        if spec.dependencies:
            if spec.dependency_type == "afterok":
                dep_opt = "afterok"
            elif spec.dependency_type == "afterany":
                dep_opt = "afterany"
            else:
                dep_opt = "afternotok"
            dep_str = ":".join([f"{dep_opt}:{d}" for d in spec.dependencies])
            lines.append(f"#PBS -W depend={dep_str}")
        
        # 数组作业
        if spec.array_range:
            start, end, step = spec.array_range
            lines.append(f"#PBS -t {start}-{end}:{step}")
        
        lines.append("")
        lines.append(f"cd {spec.working_dir}")
        lines.append("")
        
        # 环境变量
        for key, value in spec.env_vars.items():
            lines.append(f"export {key}='{value}'")
        
        # 加载模块
        for module in spec.resources.modules:
            lines.append(f"module load {module}")
        
        # 预执行命令
        for cmd in spec.pre_commands:
            lines.append(cmd)
        
        # 主命令
        if spec.executable:
            lines.append(spec.executable + " " + " ".join(spec.arguments))
        
        for cmd in spec.commands:
            lines.append(cmd)
        
        # 后执行命令
        for cmd in spec.post_commands:
            lines.append(cmd)
        
        return "\n".join(lines)
    
    def submit(self, spec: JobSpec) -> str:
        """提交PBS作业"""
        script = self.generate_script(spec)
        script_path = spec.working_dir / f"{spec.name}.pbs"
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        result = subprocess.run(
            ["qsub", str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(spec.working_dir)
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to submit job: {result.stderr}")
        
        job_id = result.stdout.strip()
        logger.info(f"Submitted PBS job {job_id}: {spec.name}")
        return job_id
    
    def cancel(self, job_id: str) -> bool:
        """取消PBS作业"""
        result = subprocess.run(
            ["qdel", job_id],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    
    def query(self, job_id: Optional[str] = None) -> List[JobInfo]:
        """查询PBS作业状态"""
        cmd = ["qstat", "-f"]
        if job_id:
            cmd.append(job_id)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return []
        
        jobs = []
        # PBS -f 输出是多行格式，需要特殊解析
        current_job = {}
        for line in result.stdout.split("\n"):
            if line.startswith("Job Id:"):
                if current_job:
                    jobs.append(self._parse_pbs_job(current_job))
                current_job = {"id": line.split(":")[1].strip()}
            elif ":" in line and current_job is not None:
                key, value = line.split(":", 1)
                current_job[key.strip()] = value.strip()
        
        if current_job:
            jobs.append(self._parse_pbs_job(current_job))
        
        return jobs
    
    def _parse_pbs_job(self, job_dict: Dict) -> JobInfo:
        """解析PBS作业信息"""
        job_id = job_dict.get("id", "")
        name = job_dict.get("Job_Name", "")
        state = job_dict.get("job_state", "U")
        
        status_map = {
            "Q": JobStatus.PENDING,
            "R": JobStatus.RUNNING,
            "C": JobStatus.COMPLETED,
            "E": JobStatus.RUNNING,  # 正在退出
            "H": JobStatus.PENDING,  # 挂起
            "T": JobStatus.PENDING,  # 移动中
            "W": JobStatus.PENDING,  # 等待
            "S": JobStatus.PENDING,  # 暂停
            "U": JobStatus.UNKNOWN
        }
        
        return JobInfo(
            job_id=job_id,
            name=name,
            status=status_map.get(state, JobStatus.UNKNOWN)
        )
    
    def get_queues(self) -> List[Dict]:
        """获取PBS队列信息"""
        result = subprocess.run(
            ["qstat", "-q"],
            capture_output=True,
            text=True
        )
        
        queues = []
        # 解析qstat -q输出
        in_table = False
        for line in result.stdout.split("\n"):
            if "Queue" in line and "Memory" in line:
                in_table = True
                continue
            if in_table and line.strip() and not line.startswith("-"):
                parts = line.split()
                if len(parts) >= 5:
                    queues.append({
                        "name": parts[0],
                        "memory": parts[1],
                        "cpu_time": parts[2],
                        "walltime": parts[3],
                        "node": parts[4],
                        "run": parts[5] if len(parts) > 5 else "0",
                        "queued": parts[6] if len(parts) > 6 else "0"
                    })
        
        return queues
    
    def estimate_wait_time(self, resources: ResourceRequest, queue: Optional[str] = None) -> Optional[timedelta]:
        """预估PBS等待时间"""
        # PBS没有标准API预估等待时间
        return None


class LSFScheduler(BaseScheduler):
    """LSF调度器"""
    
    def _check_availability(self) -> bool:
        try:
            result = subprocess.run(
                ["bsub", "-V"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 or "Platform" in result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def generate_script(self, spec: JobSpec) -> str:
        """生成LSF作业脚本"""
        lines = ["#!/bin/bash"]
        lines.append(f"#BSUB -J {spec.name}")
        
        # 资源请求
        lines.append(spec.resources.to_lsf())
        
        # 输入输出
        if spec.stdout_file:
            lines.append(f"#BSUB -o {spec.working_dir / spec.stdout_file}")
        if spec.stderr_file:
            lines.append(f"#BSUB -e {spec.working_dir / spec.stderr_file}")
        
        # 邮件通知
        if spec.email:
            lines.append(f"#BSUB -u {spec.email}")
            events = []
            if "begin" in spec.email_events:
                events.append("1")
            if "end" in spec.email_events:
                events.append("2")
            if "fail" in spec.email_events:
                events.append("3")
            if events:
                lines.append(f"#BSUB -B -N")
        
        # 依赖
        if spec.dependencies:
            dep_str = " && ".join([f"done({d})" for d in spec.dependencies])
            lines.append(f"#BSUB -w '{dep_str}'")
        
        lines.append("")
        lines.append(f"cd {spec.working_dir}")
        lines.append("")
        
        # 环境变量
        for key, value in spec.env_vars.items():
            lines.append(f"export {key}='{value}'")
        
        # 加载模块
        for module in spec.resources.modules:
            lines.append(f"module load {module}")
        
        # 预执行命令
        for cmd in spec.pre_commands:
            lines.append(cmd)
        
        # 主命令
        if spec.executable:
            lines.append(spec.executable + " " + " ".join(spec.arguments))
        
        for cmd in spec.commands:
            lines.append(cmd)
        
        # 后执行命令
        for cmd in spec.post_commands:
            lines.append(cmd)
        
        return "\n".join(lines)
    
    def submit(self, spec: JobSpec) -> str:
        """提交LSF作业"""
        script = self.generate_script(spec)
        script_path = spec.working_dir / f"{spec.name}.lsf"
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        result = subprocess.run(
            ["bsub", "<", str(script_path)],
            capture_output=True,
            text=True,
            shell=True,
            cwd=str(spec.working_dir)
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to submit job: {result.stderr}")
        
        # 解析作业ID: "Job <123> is submitted to queue <normal>."
        match = re.search(r"Job <(\d+)>", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse job ID from: {result.stdout}")
        
        job_id = match.group(1)
        logger.info(f"Submitted LSF job {job_id}: {spec.name}")
        return job_id
    
    def cancel(self, job_id: str) -> bool:
        """取消LSF作业"""
        result = subprocess.run(
            ["bkill", job_id],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    
    def query(self, job_id: Optional[str] = None) -> List[JobInfo]:
        """查询LSF作业状态"""
        cmd = ["bjobs", "-a", "-o", "jobid:12 job_name:20 stat:10 queue:15 run_time:15", "-noheader"]
        
        if job_id:
            cmd.extend(["-J", job_id])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return []
        
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            job_id_str = parts[0]
            name = parts[1]
            state = parts[2]
            queue = parts[3] if len(parts) > 3 else None
            
            status_map = {
                "PEND": JobStatus.PENDING,
                "RUN": JobStatus.RUNNING,
                "DONE": JobStatus.COMPLETED,
                "EXIT": JobStatus.FAILED,
                "PSUSP": JobStatus.PENDING,
                "USUSP": JobStatus.PENDING,
                "SSUSP": JobStatus.PENDING
            }
            
            job_info = JobInfo(
                job_id=job_id_str,
                name=name,
                status=status_map.get(state, JobStatus.UNKNOWN),
                queue=queue
            )
            jobs.append(job_info)
        
        return jobs
    
    def get_queues(self) -> List[Dict]:
        """获取LSF队列信息"""
        result = subprocess.run(
            ["bqueues", "-o", "queue_name:15 priority:8 status:10 max:15 jl_u:10 njobs:10 pend:10 run:10 susp:10", "-noheader"],
            capture_output=True,
            text=True
        )
        
        queues = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                queues.append({
                    "name": parts[0],
                    "priority": parts[1],
                    "status": parts[2],
                    "max_jobs": parts[3] if len(parts) > 3 else "-",
                    "user_limit": parts[4] if len(parts) > 4 else "-",
                    "total_jobs": parts[5] if len(parts) > 5 else "0",
                    "pending": parts[6] if len(parts) > 6 else "0",
                    "running": parts[7] if len(parts) > 7 else "0",
                    "suspended": parts[8] if len(parts) > 8 else "0"
                })
        
        return queues
    
    def estimate_wait_time(self, resources: ResourceRequest, queue: Optional[str] = None) -> Optional[timedelta]:
        """预估LSF等待时间"""
        try:
            # LSF可以使用bwait或查看队列统计
            cmd = ["bqueues", "-o", "queue_name:15 pend:10", "-noheader"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 2:
                        if queue and parts[0] != queue:
                            continue
                        pending = int(parts[1]) if parts[1].isdigit() else 0
                        estimated_minutes = min(pending * 3, 1440)
                        return timedelta(minutes=estimated_minutes)
        except Exception:
            pass
        
        return None


class LocalScheduler(BaseScheduler):
    """本地执行调度器（用于测试和单机模式）"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._jobs: Dict[str, subprocess.Popen] = {}
        self._job_counter = 0
        self._lock = threading.Lock()
    
    def _check_availability(self) -> bool:
        return True
    
    def submit(self, spec: JobSpec) -> str:
        """本地执行作业"""
        with self._lock:
            self._job_counter += 1
            job_id = f"local_{self._job_counter}"
        
        # 创建执行脚本
        script_lines = ["#!/bin/bash"]
        script_lines.append(f"cd {spec.working_dir}")
        
        for key, value in spec.env_vars.items():
            script_lines.append(f"export {key}='{value}'")
        
        for cmd in spec.pre_commands:
            script_lines.append(cmd)
        
        if spec.executable:
            script_lines.append(spec.executable + " " + " ".join(spec.arguments))
        
        for cmd in spec.commands:
            script_lines.append(cmd)
        
        for cmd in spec.post_commands:
            script_lines.append(cmd)
        
        script_path = spec.working_dir / f"{spec.name}.sh"
        with open(script_path, 'w') as f:
            f.write("\n".join(script_lines))
        
        os.chmod(script_path, 0o755)
        
        # 启动进程
        stdout_path = spec.working_dir / spec.stdout_file if spec.stdout_file else "/dev/null"
        stderr_path = spec.working_dir / spec.stderr_file if spec.stderr_file else "/dev/null"
        
        with open(stdout_path, 'w') as out, open(stderr_path, 'w') as err:
            process = subprocess.Popen(
                ["bash", str(script_path)],
                stdout=out,
                stderr=err,
                cwd=str(spec.working_dir)
            )
        
        self._jobs[job_id] = process
        logger.info(f"Started local job {job_id}: {spec.name}")
        return job_id
    
    def cancel(self, job_id: str) -> bool:
        """取消本地作业"""
        if job_id in self._jobs:
            self._jobs[job_id].terminate()
            return True
        return False
    
    def query(self, job_id: Optional[str] = None) -> List[JobInfo]:
        """查询本地作业状态"""
        jobs = []
        
        for jid, process in list(self._jobs.items()):
            if job_id and jid != job_id:
                continue
            
            return_code = process.poll()
            
            if return_code is None:
                status = JobStatus.RUNNING
            elif return_code == 0:
                status = JobStatus.COMPLETED
            else:
                status = JobStatus.FAILED
            
            jobs.append(JobInfo(
                job_id=jid,
                name=f"local_job_{jid}",
                status=status,
                exit_code=return_code if return_code is not None else None
            ))
        
        return jobs
    
    def get_queues(self) -> List[Dict]:
        """获取本地队列信息"""
        return [{
            "name": "local",
            "available": True,
            "total_nodes": 1,
            "max_cores": os.cpu_count() or 1
        }]
    
    def estimate_wait_time(self, resources: ResourceRequest, queue: Optional[str] = None) -> Optional[timedelta]:
        """本地执行无需等待"""
        return timedelta(0)


def detect_scheduler() -> SchedulerType:
    """自动检测可用的调度系统"""
    # 按优先级检测
    schedulers = [
        (SchedulerType.SLURM, SlurmScheduler),
        (SchedulerType.PBS, PBSScheduler),
        (SchedulerType.LSF, LSFScheduler)
    ]
    
    for sched_type, sched_class in schedulers:
        try:
            scheduler = sched_class()
            if scheduler._check_availability():
                logger.info(f"Detected scheduler: {sched_type.value}")
                return sched_type
        except Exception:
            continue
    
    logger.info("No HPC scheduler detected, using local mode")
    return SchedulerType.LOCAL


class WorkflowManager:
    """工作流管理器 - 支持复杂的作业依赖关系"""
    
    def __init__(self, scheduler: Optional[BaseScheduler] = None):
        self.scheduler = scheduler or self._create_default_scheduler()
        self._jobs: Dict[str, str] = {}  # 任务名称 -> 作业ID
        self._dependencies: Dict[str, List[str]] = defaultdict(list)
    
    def _create_default_scheduler(self) -> BaseScheduler:
        """创建默认调度器"""
        sched_type = detect_scheduler()
        schedulers = {
            SchedulerType.SLURM: SlurmScheduler,
            SchedulerType.PBS: PBSScheduler,
            SchedulerType.LSF: LSFScheduler,
            SchedulerType.LOCAL: LocalScheduler
        }
        return schedulers[sched_type]()
    
    def submit_task(self, name: str, spec: JobSpec, depends_on: Optional[List[str]] = None) -> str:
        """提交任务"""
        # 处理依赖
        if depends_on:
            dep_job_ids = []
            for dep_name in depends_on:
                if dep_name in self._jobs:
                    dep_job_ids.append(self._jobs[dep_name])
                else:
                    logger.warning(f"Dependency {dep_name} not found")
            
            if dep_job_ids:
                spec.dependencies = dep_job_ids
        
        job_id = self.scheduler.submit(spec)
        self._jobs[name] = job_id
        return job_id
    
    def wait_for_all(self, timeout: Optional[float] = None) -> Dict[str, JobStatus]:
        """等待所有任务完成"""
        results = {}
        for name, job_id in self._jobs.items():
            status = self.scheduler.wait_for_job(job_id, timeout=timeout)
            results[name] = status
        return results
    
    def get_status(self) -> Dict[str, JobInfo]:
        """获取所有任务状态"""
        status = {}
        for name, job_id in self._jobs.items():
            jobs = self.scheduler.query(job_id)
            if jobs:
                status[name] = jobs[0]
        return status


# =============================================================================
# 便捷的工厂函数
# =============================================================================

def create_scheduler(scheduler_type: Optional[str] = None) -> BaseScheduler:
    """
    创建调度器实例
    
    Args:
        scheduler_type: 'slurm', 'pbs', 'lsf', 'local' 或 None（自动检测）
    
    Returns:
        BaseScheduler实例
    """
    if scheduler_type is None:
        sched_type = detect_scheduler()
    else:
        sched_type = SchedulerType(scheduler_type.lower())
    
    schedulers = {
        SchedulerType.SLURM: SlurmScheduler,
        SchedulerType.PBS: PBSScheduler,
        SchedulerType.LSF: LSFScheduler,
        SchedulerType.LOCAL: LocalScheduler
    }
    
    return schedulers[sched_type]()


def submit_dft_calculation(
    working_dir: str,
    calc_type: str = "vasp",
    structure_file: Optional[str] = None,
    resources: Optional[ResourceRequest] = None,
    scheduler: Optional[BaseScheduler] = None
) -> str:
    """
    便捷函数：提交DFT计算作业
    
    Args:
        working_dir: 工作目录
        calc_type: 计算类型 (vasp, quantum_espresso, abacus)
        structure_file: 结构文件路径
        resources: 资源请求
        scheduler: 调度器实例
    
    Returns:
        作业ID
    """
    sched = scheduler or create_scheduler()
    resources = resources or ResourceRequest(
        num_nodes=1,
        num_cores_per_node=32,
        memory_gb=128,
        walltime_hours=48
    )
    
    calc_commands = {
        "vasp": "mpirun -np $SLURM_NTASKS vasp_std",
        "quantum_espresso": "mpirun -np $SLURM_NTASKS pw.x -in pw.in",
        "abacus": "mpirun -np $SLURM_NTASKS abacus"
    }
    
    spec = JobSpec(
        name=f"{calc_type}_calc",
        working_dir=Path(working_dir),
        commands=[calc_commands.get(calc_type, calc_commands["vasp"])],
        resources=resources,
        pre_commands=["module load intel mpi vasp"] if calc_type == "vasp" else []
    )
    
    return sched.submit(spec)


def submit_gpu_job(
    working_dir: str,
    command: str,
    num_gpus: int = 1,
    gpu_type: Optional[str] = None,
    walltime_hours: float = 24.0,
    scheduler: Optional[BaseScheduler] = None
) -> str:
    """
    便捷函数：提交GPU作业（用于ML训练或MD）
    
    Args:
        working_dir: 工作目录
        command: 执行命令
        num_gpus: GPU数量
        gpu_type: GPU类型 (如 'a100', 'v100')
        walltime_hours: 运行时间
        scheduler: 调度器实例
    
    Returns:
        作业ID
    """
    sched = scheduler or create_scheduler()
    
    resources = ResourceRequest(
        num_nodes=1,
        num_cores_per_node=8,
        num_gpus=num_gpus,
        gpu_type=gpu_type,
        memory_gb=64 * num_gpus,
        walltime_hours=walltime_hours
    )
    
    spec = JobSpec(
        name="gpu_job",
        working_dir=Path(working_dir),
        commands=[command],
        resources=resources
    )
    
    return sched.submit(spec)


# =============================================================================
# 示例用法
# =============================================================================

if __name__ == "__main__":
    # 示例1: 自动检测并提交简单作业
    scheduler = create_scheduler()
    
    spec = JobSpec(
        name="test_job",
        working_dir=Path("./test_run"),
        commands=["echo 'Hello HPC'", "sleep 10", "echo 'Done'"],
        resources=ResourceRequest(
            num_nodes=1,
            num_cores_per_node=4,
            walltime_hours=1.0
        )
    )
    
    # job_id = scheduler.submit(spec)
    # print(f"Submitted job: {job_id}")
    
    # 示例2: 使用工作流管理器
    workflow = WorkflowManager(scheduler)
    
    # 阶段1: DFT计算
    dft_spec = JobSpec(
        name="dft_relaxation",
        working_dir=Path("./dft_run"),
        commands=["mpirun vasp_std"],
        resources=ResourceRequest(num_nodes=2, num_cores_per_node=32, walltime_hours=24)
    )
    # workflow.submit_task("dft", dft_spec)
    
    # 阶段2: 数据处理（依赖于DFT）
    process_spec = JobSpec(
        name="process_results",
        working_dir=Path("./process"),
        commands=["python analyze.py"],
        resources=ResourceRequest(num_nodes=1, num_cores_per_node=4, walltime_hours=2)
    )
    # workflow.submit_task("process", process_spec, depends_on=["dft"])
    
    # 示例3: 批量提交DFT计算
    structures = ["POSCAR_1", "POSCAR_2", "POSCAR_3"]
    job_ids = []
    for i, poscar in enumerate(structures):
        spec = JobSpec(
            name=f"vasp_calc_{i}",
            working_dir=Path(f"./calc_{i}"),
            commands=[f"cp {poscar} POSCAR", "mpirun vasp_std"],
            resources=ResourceRequest(num_nodes=1, num_cores_per_node=16, walltime_hours=12)
        )
        # job_id = scheduler.submit(spec)
        # job_ids.append(job_id)
    
    print("hpc_scheduler module loaded successfully")
