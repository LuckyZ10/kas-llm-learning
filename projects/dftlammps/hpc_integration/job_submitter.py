#!/usr/bin/env python3
"""
job_submitter.py
================
增强型作业提交器

支持功能：
- 多种计算类型模板 (VASP, Quantum ESPRESSO, LAMMPS, etc.)
- 智能作业数组构建
- 多种提交策略 (批量、依赖、工作流)
- 容器化执行支持
- 自动资源估算
"""

import os
import re
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
from enum import Enum, auto
from datetime import datetime, timedelta
import threading
import hashlib

from .cluster_connector import ClusterConnector, ClusterConfig, get_connector

logger = logging.getLogger(__name__)


class CalculationType(Enum):
    """支持的计算类型"""
    VASP = "vasp"
    QUANTUM_ESPRESSO = "quantum_espresso"
    ABACUS = "abacus"
    LAMMPS = "lammps"
    GROMACS = "gromacs"
    NEP = "nep"
    DEEPMD = "deepmd"
    ORCA = "orca"
    GAUSSIAN = "gaussian"
    CP2K = "cp2k"
    CUSTOM = "custom"


class SubmissionStrategy(Enum):
    """提交策略"""
    SINGLE = "single"           # 单作业
    ARRAY = "array"             # 数组作业
    DEPENDENT = "dependent"     # 依赖作业
    WORKFLOW = "workflow"       # 完整工作流
    HYBRID = "hybrid"           # 混合策略


@dataclass
class JobTemplate:
    """作业模板"""
    name: str
    calculation_type: CalculationType
    
    # 资源模板
    nodes: int = 1
    cores_per_node: int = 1
    gpus: int = 0
    gpu_type: Optional[str] = None
    memory_gb: Optional[float] = None
    walltime_hours: float = 24.0
    queue: Optional[str] = None
    
    # 软件环境
    modules: List[str] = field(default_factory=list)
    conda_env: Optional[str] = None
    container_image: Optional[str] = None
    
    # 执行命令模板
    executable: Optional[str] = None
    arguments: List[str] = field(default_factory=list)
    pre_commands: List[str] = field(default_factory=list)
    post_commands: List[str] = field(default_factory=list)
    
    # 环境变量
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    # 输入输出
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    
    # 特殊选项
    exclusive: bool = False
    mpi_implementation: str = "mpich"  # mpich, openmpi, intelmpi
    openmp_threads: int = 1
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['calculation_type'] = self.calculation_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "JobTemplate":
        data = data.copy()
        data['calculation_type'] = CalculationType(data.get('calculation_type', 'custom'))
        return cls(**data)
    
    @classmethod
    def for_vasp(cls, **kwargs) -> "JobTemplate":
        """创建VASP模板"""
        defaults = {
            'name': 'vasp_calc',
            'calculation_type': CalculationType.VASP,
            'nodes': 1,
            'cores_per_node': 32,
            'memory_gb': 128,
            'walltime_hours': 24.0,
            'modules': ['intel', 'mpi', 'vasp'],
            'executable': 'vasp_std',
            'input_files': ['POSCAR', 'POTCAR', 'INCAR', 'KPOINTS'],
            'output_files': ['OUTCAR', 'vasprun.xml', 'OSZICAR'],
            'mpi_implementation': 'intelmpi'
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_lammps(cls, **kwargs) -> "JobTemplate":
        """创建LAMMPS模板"""
        defaults = {
            'name': 'lammps_calc',
            'calculation_type': CalculationType.LAMMPS,
            'nodes': 1,
            'cores_per_node': 16,
            'memory_gb': 64,
            'walltime_hours': 48.0,
            'modules': ['mpi', 'lammps'],
            'executable': 'lmp',
            'arguments': ['-in', 'input.lammps'],
            'input_files': ['input.lammps', 'data.*'],
            'output_files': ['log.lammps', 'dump.*'],
            'mpi_implementation': 'mpich'
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_ml_training(cls, **kwargs) -> "JobTemplate":
        """创建ML训练模板"""
        defaults = {
            'name': 'ml_training',
            'calculation_type': CalculationType.DEEPMD,
            'nodes': 1,
            'cores_per_node': 8,
            'gpus': 2,
            'gpu_type': 'a100',
            'memory_gb': 256,
            'walltime_hours': 72.0,
            'modules': ['cuda', 'python', 'tensorflow'],
            'conda_env': 'mlip',
            'executable': 'dp',
            'arguments': ['train', 'input.json'],
            'input_files': ['input.json', 'training_data'],
            'output_files': ['model.ckpt*', 'lcurve.out']
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_nep(cls, **kwargs) -> "JobTemplate":
        """创建NEP训练模板"""
        defaults = {
            'name': 'nep_training',
            'calculation_type': CalculationType.NEP,
            'nodes': 1,
            'cores_per_node': 8,
            'gpus': 1,
            'gpu_type': 'a100',
            'memory_gb': 128,
            'walltime_hours': 168.0,
            'modules': ['cuda', 'nep'],
            'executable': 'nep',
            'input_files': ['nep.in', 'train.xyz'],
            'output_files': ['nep.txt', 'loss.out', 'model.com']
        }
        defaults.update(kwargs)
        return cls(**defaults)


@dataclass
class JobSpec:
    """作业规格"""
    template: JobTemplate
    working_dir: Path
    job_name: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    array_range: Optional[Tuple[int, int]] = None
    priority: Optional[int] = None
    
    def __post_init__(self):
        if self.job_name is None:
            self.job_name = self.template.name


@dataclass
class SubmittedJob:
    """已提交作业信息"""
    job_id: str
    job_name: str
    submit_time: datetime
    working_dir: Path
    template: JobTemplate
    status: str = "submitted"
    scheduler_type: str = "slurm"


class JobArrayBuilder:
    """作业数组构建器"""
    
    def __init__(self, template: JobTemplate, base_working_dir: Path):
        self.template = template
        self.base_working_dir = Path(base_working_dir)
        self.jobs: List[JobSpec] = []
        self._counter = 0
    
    def add_job(
        self,
        working_dir: Optional[Path] = None,
        job_name: Optional[str] = None,
        custom_inputs: Dict[str, Any] = None,
        dependencies: List[str] = None
    ) -> int:
        """
        添加作业到数组
        
        Returns:
            作业索引
        """
        self._counter += 1
        idx = self._counter
        
        if working_dir is None:
            working_dir = self.base_working_dir / f"job_{idx:04d}"
        
        if job_name is None:
            job_name = f"{self.template.name}_{idx:04d}"
        
        # 创建模板副本
        template_copy = JobTemplate.from_dict(self.template.to_dict())
        template_copy.name = job_name
        
        # 应用自定义输入
        if custom_inputs:
            for key, value in custom_inputs.items():
                if hasattr(template_copy, key):
                    setattr(template_copy, key, value)
        
        spec = JobSpec(
            template=template_copy,
            working_dir=Path(working_dir),
            job_name=job_name,
            dependencies=dependencies or []
        )
        
        self.jobs.append(spec)
        return idx
    
    def add_batch_from_files(
        self,
        file_pattern: str,
        file_to_workdir: Callable[[str], Path] = None
    ) -> List[int]:
        """
        从文件批量添加作业
        
        Args:
            file_pattern: 文件匹配模式 (如 "structures/POSCAR_*")
            file_to_workdir: 文件到工作目录的映射函数
        
        Returns:
            作业索引列表
        """
        import glob
        
        files = sorted(glob.glob(file_pattern))
        indices = []
        
        for file_path in files:
            if file_to_workdir:
                workdir = file_to_workdir(file_path)
            else:
                # 默认: 使用文件名创建工作目录
                stem = Path(file_path).stem
                workdir = self.base_working_dir / stem
            
            idx = self.add_job(
                working_dir=workdir,
                custom_inputs={'input_files': [file_path]}
            )
            indices.append(idx)
        
        logger.info(f"Added {len(indices)} jobs from pattern {file_pattern}")
        return indices
    
    def build_array_job(self) -> JobSpec:
        """
        构建数组作业
        
        Returns:
            数组作业规格
        """
        if not self.jobs:
            raise ValueError("No jobs added to array")
        
        # 使用第一个作业作为基础模板
        base_spec = self.jobs[0]
        
        return JobSpec(
            template=base_spec.template,
            working_dir=self.base_working_dir,
            job_name=f"{self.template.name}_array",
            array_range=(1, len(self.jobs))
        )
    
    def build_dependency_chain(self) -> List[JobSpec]:
        """
        构建依赖链
        
        Returns:
            按依赖顺序排列的作业规格列表
        """
        specs = []
        prev_job_id = None
        
        for i, spec in enumerate(self.jobs):
            if prev_job_id:
                spec.dependencies = [prev_job_id]
            specs.append(spec)
            # 注意: 实际的job_id需要在提交后才能确定
            prev_job_id = f"job_{i}"  # 占位符
        
        return specs
    
    def __len__(self) -> int:
        return len(self.jobs)


class EnhancedJobSubmitter:
    """增强型作业提交器"""
    
    def __init__(
        self,
        connector: ClusterConnector,
        scheduler_type: str = "slurm"
    ):
        self.connector = connector
        self.scheduler_type = scheduler_type.lower()
        self._submitted_jobs: Dict[str, SubmittedJob] = {}
        self._lock = threading.Lock()
    
    def submit(
        self,
        spec: JobSpec,
        dry_run: bool = False
    ) -> SubmittedJob:
        """
        提交单个作业
        
        Args:
            spec: 作业规格
            dry_run: 仅生成脚本不提交
        
        Returns:
            SubmittedJob实例
        """
        # 生成作业脚本
        script = self._generate_script(spec)
        
        if dry_run:
            logger.info(f"Dry run - script generated for {spec.job_name}")
            print(script)
            return SubmittedJob(
                job_id="dry_run",
                job_name=spec.job_name,
                submit_time=datetime.now(),
                working_dir=spec.working_dir,
                template=spec.template
            )
        
        # 确保工作目录存在
        self.connector.mkdir(str(spec.working_dir), parents=True)
        
        # 写入脚本
        script_path = spec.working_dir / f"{spec.job_name}.{self.scheduler_type}"
        self._write_script(script_path, script)
        
        # 提交作业
        job_id = self._execute_submit(str(script_path), spec)
        
        submitted = SubmittedJob(
            job_id=job_id,
            job_name=spec.job_name,
            submit_time=datetime.now(),
            working_dir=spec.working_dir,
            template=spec.template,
            scheduler_type=self.scheduler_type
        )
        
        with self._lock:
            self._submitted_jobs[job_id] = submitted
        
        logger.info(f"Submitted job {job_id}: {spec.job_name}")
        return submitted
    
    def submit_array(
        self,
        builder: JobArrayBuilder,
        max_parallel: int = None,
        dry_run: bool = False
    ) -> SubmittedJob:
        """
        提交数组作业
        
        Args:
            builder: 作业数组构建器
            max_parallel: 最大并行数
            dry_run: 仅生成脚本
        
        Returns:
            SubmittedJob实例
        """
        spec = builder.build_array_job()
        
        if max_parallel:
            spec.template.env_vars['SLURM_ARRAY_TASK_MAX'] = str(max_parallel)
        
        return self.submit(spec, dry_run)
    
    def submit_workflow(
        self,
        specs: List[JobSpec],
        dependencies: Dict[int, List[int]] = None,
        dry_run: bool = False
    ) -> List[SubmittedJob]:
        """
        提交工作流（带依赖关系）
        
        Args:
            specs: 作业规格列表
            dependencies: 依赖关系 {job_idx: [dep_idx1, dep_idx2]}
            dry_run: 仅生成脚本
        
        Returns:
            SubmittedJob列表
        """
        submitted = []
        job_id_map: Dict[int, str] = {}
        
        for i, spec in enumerate(specs):
            # 解析依赖
            if dependencies and i in dependencies:
                spec.dependencies = [
                    job_id_map[j] for j in dependencies[i]
                    if j in job_id_map
                ]
            
            job = self.submit(spec, dry_run)
            submitted.append(job)
            job_id_map[i] = job.job_id
        
        return submitted
    
    def _generate_script(self, spec: JobSpec) -> str:
        """生成作业脚本"""
        if self.scheduler_type == "slurm":
            return self._generate_slurm_script(spec)
        elif self.scheduler_type == "pbs":
            return self._generate_pbs_script(spec)
        elif self.scheduler_type == "lsf":
            return self._generate_lsf_script(spec)
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
    
    def _generate_slurm_script(self, spec: JobSpec) -> str:
        """生成Slurm脚本"""
        t = spec.template
        lines = ["#!/bin/bash"]
        lines.append(f"#SBATCH --job-name={spec.job_name}")
        lines.append(f"#SBATCH --nodes={t.nodes}")
        lines.append(f"#SBATCH --ntasks-per-node={t.cores_per_node}")
        
        # GPU
        if t.gpus > 0:
            lines.append(f"#SBATCH --gpus={t.gpus}")
            if t.gpu_type:
                lines.append(f"#SBATCH --gres=gpu:{t.gpu_type}:{t.gpus}")
        
        # 内存
        if t.memory_gb:
            lines.append(f"#SBATCH --mem={int(t.memory_gb * 1024)}")
        
        # 时间
        hours = int(t.walltime_hours)
        minutes = int((t.walltime_hours - hours) * 60)
        lines.append(f"#SBATCH --time={hours:02d}:{minutes:02d}:00")
        
        # 队列
        if t.queue:
            lines.append(f"#SBATCH --partition={t.queue}")
        
        # 输出
        lines.append(f"#SBATCH --output={spec.working_dir}/{spec.job_name}_%j.out")
        lines.append(f"#SBATCH --error={spec.working_dir}/{spec.job_name}_%j.err")
        
        # 工作目录
        lines.append(f"#SBATCH --chdir={spec.working_dir}")
        
        # 依赖
        if spec.dependencies:
            dep_str = ":".join(spec.dependencies)
            lines.append(f"#SBATCH --dependency=afterok:{dep_str}")
        
        # 数组
        if spec.array_range:
            start, end = spec.array_range
            lines.append(f"#SBATCH --array={start}-{end}")
        
        # 优先级
        if spec.priority:
            lines.append(f"#SBATCH --nice={spec.priority}")
        
        # 独占
        if t.exclusive:
            lines.append("#SBATCH --exclusive")
        
        lines.append("")
        
        # 环境变量
        for key, value in t.env_vars.items():
            lines.append(f"export {key}={value}")
        
        if t.env_vars:
            lines.append("")
        
        # OpenMP
        if t.openmp_threads > 1:
            lines.append(f"export OMP_NUM_THREADS={t.openmp_threads}")
            lines.append("")
        
        # 容器支持
        if t.container_image:
            lines.extend(self._generate_container_wrapper(t))
        else:
            # 模块
            for module in t.modules:
                lines.append(f"module load {module}")
            
            if t.modules:
                lines.append("")
            
            # Conda
            if t.conda_env:
                lines.append(f"source activate {t.conda_env}")
                lines.append("")
            
            # 预执行命令
            for cmd in t.pre_commands:
                lines.append(cmd)
            
            if t.pre_commands:
                lines.append("")
            
            # MPI执行
            if t.mpi_implementation == "intelmpi":
                mpi_cmd = "mpirun"
            else:
                mpi_cmd = "mpirun"
            
            total_cores = t.nodes * t.cores_per_node
            
            if t.executable:
                if total_cores > 1:
                    lines.append(f"{mpi_cmd} -np {total_cores} {t.executable} {' '.join(t.arguments)}")
                else:
                    lines.append(f"{t.executable} {' '.join(t.arguments)}")
            
            # 额外命令
            for cmd in t.commands:
                lines.append(cmd)
            
            # 后执行命令
            if t.post_commands:
                lines.append("")
                for cmd in t.post_commands:
                    lines.append(cmd)
        
        return "\n".join(lines)
    
    def _generate_pbs_script(self, spec: JobSpec) -> str:
        """生成PBS脚本"""
        t = spec.template
        lines = ["#!/bin/bash"]
        lines.append(f"#PBS -N {spec.job_name}")
        lines.append(f"#PBS -l nodes={t.nodes}:ppn={t.cores_per_node}")
        
        if t.gpus > 0:
            lines.append(f"#PBS -l ngpus={t.gpus}")
        
        if t.memory_gb:
            lines.append(f"#PBS -l mem={int(t.memory_gb)}gb")
        
        hours = int(t.walltime_hours)
        minutes = int((t.walltime_hours - hours) * 60)
        lines.append(f"#PBS -l walltime={hours:02d}:{minutes:02d}:00")
        
        if t.queue:
            lines.append(f"#PBS -q {t.queue}")
        
        lines.append(f"#PBS -o {spec.working_dir}/{spec.job_name}.out")
        lines.append(f"#PBS -e {spec.working_dir}/{spec.job_name}.err")
        
        if spec.dependencies:
            dep_str = ":".join([f"afterok:{d}" for d in spec.dependencies])
            lines.append(f"#PBS -W depend={dep_str}")
        
        if spec.array_range:
            start, end = spec.array_range
            lines.append(f"#PBS -t {start}-{end}")
        
        lines.append("")
        lines.append(f"cd {spec.working_dir}")
        lines.append("")
        
        # 环境变量
        for key, value in t.env_vars.items():
            lines.append(f"export {key}={value}")
        
        # 模块
        for module in t.modules:
            lines.append(f"module load {module}")
        
        if t.modules:
            lines.append("")
        
        # 预执行命令
        for cmd in t.pre_commands:
            lines.append(cmd)
        
        # 执行
        total_cores = t.nodes * t.cores_per_node
        
        if t.executable:
            if total_cores > 1:
                lines.append(f"mpirun -np {total_cores} {t.executable} {' '.join(t.arguments)}")
            else:
                lines.append(f"{t.executable} {' '.join(t.arguments)}")
        
        for cmd in t.commands:
            lines.append(cmd)
        
        if t.post_commands:
            lines.append("")
            for cmd in t.post_commands:
                lines.append(cmd)
        
        return "\n".join(lines)
    
    def _generate_lsf_script(self, spec: JobSpec) -> str:
        """生成LSF脚本"""
        t = spec.template
        lines = ["#!/bin/bash"]
        lines.append(f"#BSUB -J {spec.job_name}")
        
        total_cores = t.nodes * t.cores_per_node
        lines.append(f"#BSUB -n {total_cores}")
        lines.append(f"#BSUB -R 'span[ptile={t.cores_per_node}]'")
        
        if t.gpus > 0:
            lines.append(f"#BSUB -gpu 'num={t.gpus}:mode=exclusive_process'")
        
        if t.memory_gb:
            lines.append(f"#BSUB -M {int(t.memory_gb * 1024)}")
            lines.append(f"#BSUB -R 'rusage[mem={int(t.memory_gb * 1024)}]'")
        
        hours = int(t.walltime_hours)
        minutes = int((t.walltime_hours - hours) * 60)
        lines.append(f"#BSUB -W {hours}:{minutes:02d}")
        
        if t.queue:
            lines.append(f"#BSUB -q {t.queue}")
        
        lines.append(f"#BSUB -o {spec.working_dir}/{spec.job_name}.out")
        lines.append(f"#BSUB -e {spec.working_dir}/{spec.job_name}.err")
        
        if spec.dependencies:
            dep_str = " && ".join([f"done({d})" for d in spec.dependencies])
            lines.append(f"#BSUB -w '{dep_str}'")
        
        lines.append("")
        lines.append(f"cd {spec.working_dir}")
        lines.append("")
        
        # 环境变量
        for key, value in t.env_vars.items():
            lines.append(f"export {key}={value}")
        
        # 模块
        for module in t.modules:
            lines.append(f"module load {module}")
        
        if t.modules:
            lines.append("")
        
        # 执行
        if t.executable:
            lines.append(f"mpirun {t.executable} {' '.join(t.arguments)}")
        
        for cmd in t.commands:
            lines.append(cmd)
        
        return "\n".join(lines)
    
    def _generate_container_wrapper(self, template: JobTemplate) -> List[str]:
        """生成容器包装命令"""
        lines = []
        
        if template.container_image.endswith(".sif") or "singularity" in template.container_image:
            # Singularity
            bind_paths = ",".join([
                str(template.input_files[0]) if template.input_files else ".",
                "/tmp",
                "/scratch"
            ])
            
            lines.append(f"module load singularity")
            lines.append("")
            
            exec_cmd = f"singularity exec --bind {bind_paths} {template.container_image}"
            
            if template.executable:
                total_cores = template.nodes * template.cores_per_node
                if total_cores > 1:
                    lines.append(f"mpirun -np {total_cores} {exec_cmd} {template.executable} {' '.join(template.arguments)}")
                else:
                    lines.append(f"{exec_cmd} {template.executable} {' '.join(template.arguments)}")
        
        else:
            # Docker (通常不推荐在HPC上使用，除非有rootless配置)
            lines.append(f"docker run --rm -v $(pwd):/work -w /work {template.container_image} ")
        
        return lines
    
    def _write_script(self, path: Path, content: str):
        """写入脚本文件"""
        # 如果是SSH连接器，需要特殊处理
        if hasattr(self.connector, 'upload'):
            # 先写入本地临时文件
            local_path = Path(f"/tmp/{path.name}")
            local_path.write_text(content)
            self.connector.upload(str(local_path), str(path))
            local_path.unlink()
            
            # 设置执行权限
            self.connector.execute(f"chmod +x {path}")
        else:
            path.write_text(content)
            path.chmod(0o755)
    
    def _execute_submit(self, script_path: str, spec: JobSpec) -> str:
        """执行提交命令"""
        if self.scheduler_type == "slurm":
            cmd = f"sbatch {script_path}"
        elif self.scheduler_type == "pbs":
            cmd = f"qsub {script_path}"
        elif self.scheduler_type == "lsf":
            cmd = f"bsub < {script_path}"
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
        
        code, stdout, stderr = self.connector.execute(cmd)
        
        if code != 0:
            raise RuntimeError(f"Job submission failed: {stderr}")
        
        # 解析job_id
        return self._parse_job_id(stdout)
    
    def _parse_job_id(self, output: str) -> str:
        """解析作业ID"""
        if self.scheduler_type == "slurm":
            match = re.search(r"Submitted batch job (\d+)", output)
            if match:
                return match.group(1)
        elif self.scheduler_type == "pbs":
            # PBS返回格式: 12345.server
            return output.strip().split(".")[0]
        elif self.scheduler_type == "lsf":
            match = re.search(r"Job <(\d+)>", output)
            if match:
                return match.group(1)
        
        # 无法解析，返回原始输出
        return output.strip()
    
    def get_submitted_job(self, job_id: str) -> Optional[SubmittedJob]:
        """获取已提交作业信息"""
        return self._submitted_jobs.get(job_id)
    
    def list_submitted_jobs(self) -> List[SubmittedJob]:
        """列出所有已提交作业"""
        return list(self._submitted_jobs.values())
